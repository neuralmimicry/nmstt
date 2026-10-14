//! Text-to-speech for nmstt, using Piper (https://github.com/rhasspy/piper).
//!
//! Piper runs as a short-lived subprocess per request: text on stdin, WAV on
//! stdout. Requests are bounded by a semaphore (waiting is bounded too), each
//! synthesis has a hard timeout, and the child is killed if the request is
//! dropped, so a slow or stuck synthesis never blocks the STT paths.
//!
//! Configuration (environment):
//! - `NMSTT_TTS_ENABLED`        default on when a voice directory exists
//! - `NMSTT_TTS_PIPER_BIN`      default `/opt/piper/piper`
//! - `NMSTT_TTS_VOICE_DIR`      default `/app/voices` (`<voice>.onnx` + `<voice>.onnx.json`)
//! - `NMSTT_TTS_DEFAULT_VOICE`  default `en_GB-alan-medium`
//! - `NMSTT_TTS_WORKERS`        concurrent syntheses, default 2
//! - `NMSTT_TTS_TIMEOUT_MS`     per synthesis, default 30000
//! - `NMSTT_TTS_MAX_CHARS`      default 2000

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::sync::Semaphore;

#[derive(Debug, Clone)]
pub struct TtsConfig {
    pub piper_bin: PathBuf,
    pub voice_dir: PathBuf,
    pub default_voice: String,
    pub workers: usize,
    pub timeout: Duration,
    pub queue_timeout: Duration,
    pub max_chars: usize,
}

impl TtsConfig {
    pub fn from_env() -> Option<Self> {
        let get = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        let voice_dir =
            PathBuf::from(get("NMSTT_TTS_VOICE_DIR").unwrap_or_else(|| "/app/voices".into()));
        let enabled = match get("NMSTT_TTS_ENABLED").map(|v| v.to_lowercase()) {
            Some(v) => matches!(v.as_str(), "1" | "true" | "yes" | "on"),
            None => voice_dir.is_dir(),
        };
        if !enabled {
            return None;
        }
        let num = |k: &str, d: u64| get(k).and_then(|v| v.parse().ok()).unwrap_or(d);
        Some(Self {
            piper_bin: PathBuf::from(
                get("NMSTT_TTS_PIPER_BIN").unwrap_or_else(|| "/opt/piper/piper".into()),
            ),
            voice_dir,
            default_voice: get("NMSTT_TTS_DEFAULT_VOICE")
                .unwrap_or_else(|| "en_GB-alan-medium".into()),
            workers: num("NMSTT_TTS_WORKERS", 2).max(1) as usize,
            timeout: Duration::from_millis(num("NMSTT_TTS_TIMEOUT_MS", 30_000)),
            queue_timeout: Duration::from_millis(num("NMSTT_TTS_QUEUE_TIMEOUT_MS", 10_000)),
            max_chars: num("NMSTT_TTS_MAX_CHARS", 2000) as usize,
        })
    }
}

#[derive(Error, Debug, PartialEq)]
pub enum TtsError {
    #[error("text is empty")]
    EmptyText,
    #[error("text exceeds {0} characters")]
    TooLong(usize),
    #[error("invalid voice name")]
    InvalidVoice,
    #[error("voice not installed: {0}")]
    UnknownVoice(String),
    #[error("speed must be between 0.5 and 2.0")]
    InvalidSpeed,
    #[error("tts is busy")]
    Busy,
    #[error("synthesis timed out")]
    Timeout,
    #[error("synthesis failed: {0}")]
    Failed(String),
}

impl TtsError {
    pub fn status(&self) -> u16 {
        match self {
            Self::EmptyText | Self::TooLong(_) | Self::InvalidVoice | Self::InvalidSpeed => 400,
            Self::UnknownVoice(_) => 404,
            Self::Busy => 503,
            Self::Timeout => 504,
            Self::Failed(_) => 500,
        }
    }
}

pub struct Tts {
    cfg: TtsConfig,
    slots: Semaphore,
}

/// Remove control characters (keeping newlines as sentence breaks) and trim.
pub fn sanitize_text(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c == '\n' || c == '\r' || c == '\t' {
                ' '
            } else {
                c
            }
        })
        .filter(|c| !c.is_control())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Voice names map to files, so only allow a safe filename alphabet.
pub fn valid_voice_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
        && !name.contains("..")
}

impl Tts {
    pub fn new(cfg: TtsConfig) -> Self {
        let slots = Semaphore::new(cfg.workers);
        Self { cfg, slots }
    }

    pub fn config(&self) -> &TtsConfig {
        &self.cfg
    }

    /// Installed voices: `<name>.onnx` with a matching `<name>.onnx.json`.
    pub fn voices(&self) -> Vec<String> {
        let mut out: Vec<String> = std::fs::read_dir(&self.cfg.voice_dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                let stem = name.strip_suffix(".onnx")?.to_string();
                self.cfg
                    .voice_dir
                    .join(format!("{name}.json"))
                    .is_file()
                    .then_some(stem)
            })
            .collect();
        out.sort();
        out
    }

    fn voice_path(&self, voice: Option<&str>) -> Result<PathBuf, TtsError> {
        let name = voice
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .unwrap_or(&self.cfg.default_voice);
        if !valid_voice_name(name) {
            return Err(TtsError::InvalidVoice);
        }
        let path = self.cfg.voice_dir.join(format!("{name}.onnx"));
        if path.is_file() && Path::new(&format!("{}.json", path.display())).is_file() {
            Ok(path)
        } else {
            Err(TtsError::UnknownVoice(name.to_string()))
        }
    }

    /// Synthesize `text` to a WAV byte vector.
    pub async fn synthesize(
        &self,
        text: &str,
        voice: Option<&str>,
        speed: Option<f32>,
    ) -> Result<Vec<u8>, TtsError> {
        let text = sanitize_text(text);
        if text.is_empty() {
            return Err(TtsError::EmptyText);
        }
        if text.chars().count() > self.cfg.max_chars {
            return Err(TtsError::TooLong(self.cfg.max_chars));
        }
        let speed = speed.unwrap_or(1.0);
        if !(0.5..=2.0).contains(&speed) {
            return Err(TtsError::InvalidSpeed);
        }
        let model = self.voice_path(voice)?;

        let _slot = tokio::time::timeout(self.cfg.queue_timeout, self.slots.acquire())
            .await
            .map_err(|_| TtsError::Busy)?
            .map_err(|_| TtsError::Busy)?;

        let mut child = {
            let mut attempt = 0;
            loop {
                let spawned = Command::new(&self.cfg.piper_bin)
                    .arg("--model")
                    .arg(&model)
                    .arg("--output_file")
                    .arg("-")
                    // Piper's length_scale is the inverse of speaking speed.
                    .arg("--length_scale")
                    .arg(format!("{:.3}", 1.0 / speed))
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .kill_on_drop(true)
                    .spawn();
                match spawned {
                    Ok(c) => break c,
                    // ETXTBSY: the binary is momentarily open for writing elsewhere
                    // (e.g. being replaced, or a racing fork). Retry briefly.
                    Err(e) if e.raw_os_error() == Some(26) && attempt < 5 => {
                        attempt += 1;
                        tokio::time::sleep(Duration::from_millis(20 * attempt)).await;
                    }
                    Err(e) => return Err(TtsError::Failed(format!("cannot start piper: {e}"))),
                }
            }
        };

        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| TtsError::Failed("no stdin".into()))?;
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| TtsError::Failed("no stdout".into()))?;
        let mut stderr = child
            .stderr
            .take()
            .ok_or_else(|| TtsError::Failed("no stderr".into()))?;

        let work = async move {
            // Write and read concurrently so a large output can never deadlock the pipe.
            let writer = async move {
                stdin.write_all(text.as_bytes()).await?;
                stdin.write_all(b"\n").await?;
                stdin.shutdown().await?;
                // Close the pipe now: join! would otherwise keep stdin alive until
                // every branch finishes, and piper would wait for EOF forever.
                drop(stdin);
                Ok::<(), std::io::Error>(())
            };
            let mut audio = Vec::new();
            let mut err = Vec::new();
            let (w, r, e) = tokio::join!(
                writer,
                stdout.read_to_end(&mut audio),
                stderr.read_to_end(&mut err)
            );
            w.map_err(|e| TtsError::Failed(format!("write: {e}")))?;
            r.map_err(|e| TtsError::Failed(format!("read: {e}")))?;
            let _ = e;
            let status = child
                .wait()
                .await
                .map_err(|e| TtsError::Failed(format!("wait: {e}")))?;
            if !status.success() {
                let tail: String = String::from_utf8_lossy(&err)
                    .lines()
                    .last()
                    .unwrap_or("")
                    .chars()
                    .take(200)
                    .collect();
                return Err(TtsError::Failed(format!(
                    "piper exited with {status}: {tail}"
                )));
            }
            Ok(audio)
        };
        let audio = tokio::time::timeout(self.cfg.timeout, work)
            .await
            .map_err(|_| TtsError::Timeout)??;
        if audio.len() < 44 || &audio[0..4] != b"RIFF" || &audio[8..12] != b"WAVE" {
            return Err(TtsError::Failed("piper did not produce WAV audio".into()));
        }
        Ok(audio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::Arc;

    fn fixture(script: &str) -> (tempdir::Dir, Tts) {
        let dir = tempdir::Dir::new();
        let bin = dir.path().join("piper");
        std::fs::write(&bin, script).unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        let voices = dir.path().join("voices");
        std::fs::create_dir(&voices).unwrap();
        std::fs::write(voices.join("en_GB-test.onnx"), b"model").unwrap();
        std::fs::write(voices.join("en_GB-test.onnx.json"), b"{}").unwrap();
        std::fs::write(voices.join("orphan.onnx"), b"model").unwrap(); // no .json: not a voice
        let tts = Tts::new(TtsConfig {
            piper_bin: bin,
            voice_dir: voices,
            default_voice: "en_GB-test".into(),
            workers: 1,
            timeout: Duration::from_millis(1500),
            queue_timeout: Duration::from_millis(300),
            max_chars: 50,
        });
        (dir, tts)
    }

    // Fake piper: echo a minimal WAV header followed by the text it was given.
    const FAKE_OK: &str = "#!/bin/sh\nprintf 'RIFF\\000\\000\\000\\000WAVEfmt                                '\ncat\n";

    /// Minimal self-cleaning temp dir (avoids a dev-dependency).
    mod tempdir {
        pub struct Dir(std::path::PathBuf);
        impl Dir {
            pub fn new() -> Self {
                use std::sync::atomic::{AtomicUsize, Ordering};
                static N: AtomicUsize = AtomicUsize::new(0);
                let p = std::env::temp_dir().join(format!(
                    "nmstt-tts-test-{}-{}",
                    std::process::id(),
                    N.fetch_add(1, Ordering::Relaxed)
                ));
                std::fs::create_dir_all(&p).unwrap();
                Self(p)
            }
            pub fn path(&self) -> &std::path::Path {
                &self.0
            }
        }
        impl Drop for Dir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }

    #[test]
    fn sanitize_and_voice_names() {
        assert_eq!(sanitize_text("  Hello,\n\tworld\u{7}!  "), "Hello, world!");
        assert!(valid_voice_name("en_GB-alan-medium"));
        assert!(!valid_voice_name("../etc/passwd"));
        assert!(!valid_voice_name("a/b"));
        assert!(!valid_voice_name(""));
    }

    #[test]
    fn lists_only_complete_voices() {
        let (_d, tts) = fixture(FAKE_OK);
        assert_eq!(tts.voices(), vec!["en_GB-test".to_string()]);
    }

    #[tokio::test]
    async fn synthesizes_wav_and_passes_text_on_stdin() {
        let (_d, tts) = fixture(FAKE_OK);
        let wav = tts.synthesize("Hello there", None, None).await.unwrap();
        assert_eq!(&wav[0..4], b"RIFF");
        assert!(String::from_utf8_lossy(&wav).contains("Hello there"));
    }

    #[tokio::test]
    async fn rejects_bad_requests_before_spawning() {
        let (_d, tts) = fixture("#!/bin/sh\nexit 99\n");
        assert_eq!(
            tts.synthesize("   ", None, None).await,
            Err(TtsError::EmptyText)
        );
        assert_eq!(
            tts.synthesize(&"x".repeat(51), None, None).await,
            Err(TtsError::TooLong(50))
        );
        assert_eq!(
            tts.synthesize("hi", Some("../x"), None).await,
            Err(TtsError::InvalidVoice)
        );
        assert_eq!(
            tts.synthesize("hi", Some("orphan"), None).await,
            Err(TtsError::UnknownVoice("orphan".into()))
        );
        assert_eq!(
            tts.synthesize("hi", None, Some(5.0)).await,
            Err(TtsError::InvalidSpeed)
        );
    }

    #[tokio::test]
    async fn reports_piper_failure_and_non_wav_output() {
        let (_d, tts) = fixture("#!/bin/sh\necho 'model load failed' >&2\nexit 3\n");
        assert!(
            matches!(tts.synthesize("hi", None, None).await, Err(TtsError::Failed(m)) if m.contains("model load failed"))
        );
        let (_d2, tts2) = fixture("#!/bin/sh\ncat >/dev/null\necho notwav\n");
        assert!(matches!(
            tts2.synthesize("hi", None, None).await,
            Err(TtsError::Failed(_))
        ));
    }

    #[tokio::test]
    async fn times_out_and_bounds_concurrency() {
        let (_d, tts) = fixture("#!/bin/sh\nsleep 5\n");
        let tts = Arc::new(tts);
        let start = std::time::Instant::now();
        let (a, b) = tokio::join!(
            tts.synthesize("one", None, None),
            tts.synthesize("two", None, None)
        );
        // workers=1: the first times out (1.5 s), the second cannot get a slot within 0.3 s.
        let mut results = vec![a, b];
        results.sort_by_key(|r| format!("{r:?}"));
        assert!(results.contains(&Err(TtsError::Timeout)));
        assert!(results.contains(&Err(TtsError::Busy)));
        assert!(
            start.elapsed() < Duration::from_secs(4),
            "kill_on_drop must stop the stuck child"
        );
    }
}
