//! Optional, text-only use of Gail (the NeuralMimicry AI router) by nmstt.
//!
//! Audio never leaves nmstt: Whisper and Piper stay on-prem. Gail is only asked
//! to improve *text*, and only when a request opts in:
//! - `refine_transcript`: punctuation, casing and domain-term fixes after Whisper
//! - `speakable`: rewrite text into a speakable form before Piper
//!   (numbers, units, abbreviations, symbols)
//!
//! Every call has a hard timeout and the result is sanity-checked (length
//! ratio, no preamble). On any failure the caller keeps nmstt's own text, so
//! Gail can never fail or block a request.
//!
//! Configuration: `NMSTT_GAIL_URL` (e.g. http://gail.gail.svc.cluster.local:8080),
//! `NMSTT_GAIL_TOKEN` or `NMSTT_GAIL_TOKEN_FILE` (scope `llm`),
//! `NMSTT_GAIL_MODEL` (default `gail-auto`), `NMSTT_GAIL_TIMEOUT_MS` (default 8000).

use std::time::Duration;

use serde_json::{json, Value};
use tracing::warn;

const REFINE_PROMPT: &str = "You correct speech-to-text transcripts. Fix punctuation, capitalisation and obvious \
mis-hearings of technical or proper names. Do not add, remove or summarise content. Reply with the corrected \
transcript only.";
const SPEAKABLE_PROMPT: &str = "Rewrite the text so a text-to-speech engine reads it naturally in British English: \
spell out numbers, dates, times, currencies, units, symbols and abbreviations. Keep the meaning and wording \
otherwise unchanged. Reply with the rewritten text only.";

#[derive(Clone)]
pub struct Gail {
    url: String,
    token: String,
    model: String,
    timeout: Duration,
    http: reqwest::Client,
}

impl Gail {
    pub fn from_env() -> Option<Self> {
        let get = |k: &str| std::env::var(k).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        let url = get("NMSTT_GAIL_URL")?;
        let token = get("NMSTT_GAIL_TOKEN")
            .or_else(|| get("NMSTT_GAIL_TOKEN_FILE").and_then(|p| std::fs::read_to_string(p).ok()).map(|t| t.trim().to_string()))
            .filter(|t| !t.is_empty())?;
        let timeout = Duration::from_millis(get("NMSTT_GAIL_TIMEOUT_MS").and_then(|v| v.parse().ok()).unwrap_or(8000));
        Some(Self::new(url, token, get("NMSTT_GAIL_MODEL").unwrap_or_else(|| "gail-auto".into()), timeout))
    }

    pub fn new(url: String, token: String, model: String, timeout: Duration) -> Self {
        let http = reqwest::Client::builder().timeout(timeout).build().expect("http client");
        Self { url: url.trim_end_matches('/').to_string(), token, model, timeout, http }
    }

    async fn chat(&self, system: &str, user: &str) -> Option<String> {
        let body = json!({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.0,
            "max_tokens": (user.len() / 2 + 64).min(2048),
        });
        let send = self
            .http
            .post(format!("{}/v1/chat/completions", self.url))
            .bearer_auth(&self.token)
            .json(&body)
            .send();
        let resp = match tokio::time::timeout(self.timeout, send).await {
            Ok(Ok(r)) if r.status().is_success() => r,
            Ok(Ok(r)) => {
                warn!("gail returned {}", r.status());
                return None;
            }
            Ok(Err(e)) => {
                warn!("gail request failed: {e}");
                return None;
            }
            Err(_) => {
                warn!("gail timed out after {:?}", self.timeout);
                return None;
            }
        };
        let v: Value = resp.json().await.ok()?;
        v["choices"][0]["message"]["content"].as_str().map(|s| s.trim().to_string())
    }

    pub async fn refine_transcript(&self, text: &str) -> Option<String> {
        if text.trim().is_empty() {
            return None;
        }
        accept_rewrite(text, self.chat(REFINE_PROMPT, text).await?, 0.7, 1.4)
    }

    pub async fn speakable(&self, text: &str) -> Option<String> {
        if text.trim().is_empty() {
            return None;
        }
        // Spelling out numbers legitimately lengthens text.
        accept_rewrite(text, self.chat(SPEAKABLE_PROMPT, text).await?, 0.7, 3.0)
    }
}

impl Gail {
    /// Fire-and-forget: send an utterance's auditory spike frames with the text
    /// that goes with it to Gail, which mirrors the pair into AARNN's sensory
    /// input. Bounded (drops when 4 are in flight); never blocks or fails the caller.
    pub fn mirror_speech(&self, source: &'static str, text: String, frames: Vec<Vec<u16>>, lang: Option<String>) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        static SLOTS: std::sync::OnceLock<std::sync::Arc<tokio::sync::Semaphore>> = std::sync::OnceLock::new();
        if text.trim().is_empty() || frames.iter().all(Vec::is_empty) {
            return;
        }
        let slots = SLOTS.get_or_init(|| std::sync::Arc::new(tokio::sync::Semaphore::new(4))).clone();
        let Ok(permit) = slots.try_acquire_owned() else {
            warn!("speech mirror dropped: backlog full");
            return;
        };
        let pair_id = format!(
            "nmstt-{source}-{}-{}",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0),
            SEQ.fetch_add(1, Ordering::Relaxed)
        );
        let body = json!({
            "pair_id": pair_id, "source": source, "text": text, "lang": lang,
            "frame_ms": crate::cochlea::FRAME_MS, "bands": crate::cochlea::BANDS, "frames": frames,
        });
        let req = self.http.post(format!("{}/v1/mirror/speech", self.url)).bearer_auth(&self.token).json(&body);
        let timeout = self.timeout;
        tokio::spawn(async move {
            let _permit = permit;
            match tokio::time::timeout(timeout, req.send()).await {
                Ok(Ok(r)) if r.status().is_success() => {}
                Ok(Ok(r)) => warn!("speech mirror rejected: {}", r.status()),
                Ok(Err(e)) => warn!("speech mirror failed: {e}"),
                Err(_) => warn!("speech mirror timed out"),
            }
        });
    }
}

pub fn mirror_enabled() -> bool {
    !matches!(std::env::var("NMSTT_AARNN_MIRROR").ok().as_deref().map(str::trim), Some("0" | "false" | "off" | "no"))
}

/// Reject rewrites that are empty, chatty, or far from the original length.
pub fn accept_rewrite(original: &str, rewrite: String, min_ratio: f32, max_ratio: f32) -> Option<String> {
    let r = rewrite.trim().trim_matches('"').trim().to_string();
    if r.is_empty() {
        return None;
    }
    let lower = r.to_lowercase();
    if ["here is", "here's", "sure", "corrected transcript:", "rewritten text:"].iter().any(|p| lower.starts_with(p)) {
        return None;
    }
    let ratio = r.chars().count() as f32 / original.trim().chars().count().max(1) as f32;
    (min_ratio..=max_ratio).contains(&ratio).then_some(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{routing::post, Json, Router};

    #[test]
    fn rewrite_guards() {
        assert_eq!(accept_rewrite("hello world", "Hello, world.".into(), 0.7, 1.4), Some("Hello, world.".into()));
        assert_eq!(accept_rewrite("hello world", "Here is the corrected transcript: Hello".into(), 0.5, 5.0), None);
        assert_eq!(accept_rewrite("a long original transcript", "short".into(), 0.7, 1.4), None);
        assert_eq!(accept_rewrite("x", "   ".into(), 0.0, 9.0), None);
    }

    async fn serve(reply: &'static str, delay_ms: u64) -> String {
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move |Json(req): Json<Value>| async move {
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                assert_eq!(req["temperature"], 0.0);
                Json(json!({"choices": [{"message": {"content": reply}}]}))
            }),
        );
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn refines_and_falls_back_on_timeout_or_bad_output() {
        let ok = Gail::new(serve("Hello, Gail.", 0).await, "t".into(), "gail-auto".into(), Duration::from_secs(2));
        assert_eq!(ok.refine_transcript("hello gail").await, Some("Hello, Gail.".into()));
        let slow = Gail::new(serve("Hello, Gail.", 3000).await, "t".into(), "gail-auto".into(), Duration::from_millis(200));
        let t = std::time::Instant::now();
        assert_eq!(slow.refine_transcript("hello gail").await, None);
        assert!(t.elapsed() < Duration::from_secs(2));
        let chatty = Gail::new(serve("Sure! Hello, Gail.", 0).await, "t".into(), "gail-auto".into(), Duration::from_secs(2));
        assert_eq!(chatty.refine_transcript("hello gail").await, None);
        let down = Gail::new("http://127.0.0.1:9".into(), "t".into(), "gail-auto".into(), Duration::from_secs(1));
        assert_eq!(down.speakable("It costs £5").await, None);
    }
}
