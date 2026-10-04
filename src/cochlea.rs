//! Cochlea-like auditory front end for AARNN speech mirroring.
//!
//! Audio -> 32 mel bands (100 Hz-8 kHz, 25 ms window, 10 ms hop) -> log energy
//! -> delta (event) encoding: a band emits a spike when its log energy rises
//! more than `threshold` above the level at its last spike and is above the
//! noise floor. This is the sparse, timed, onset-driven representation a
//! silicon cochlea produces, which is what AARNN's AER sensory input expects.
//!
//! Output is sparse: one `Vec<u16>` of spiking band indices per 10 ms frame.

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

pub const BANDS: usize = 32;
pub const FRAME_MS: u32 = 10;
const WINDOW_MS: f32 = 25.0;
const F_MIN: f32 = 100.0;
const F_MAX: f32 = 8000.0;

pub struct Cochlea {
    rate: u32,
    win: usize,
    hop: usize,
    fft: Arc<dyn RealToComplex<f32>>,
    hann: Vec<f32>,
    filters: Vec<Vec<(usize, f32)>>, // per band: (fft bin, weight)
    pub threshold: f32,
    pub floor: f32,
}

fn hz_to_mel(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}
fn mel_to_hz(m: f32) -> f32 {
    700.0 * (10f32.powf(m / 2595.0) - 1.0)
}

impl Cochlea {
    pub fn new(rate: u32) -> Self {
        let win = ((rate as f32) * WINDOW_MS / 1000.0).round() as usize;
        let n_fft = win.next_power_of_two();
        let hop = (rate * FRAME_MS / 1000) as usize;
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(n_fft);
        let hann = (0..win)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (win as f32 - 1.0)).cos())
            .collect();
        // Triangular mel filters over the FFT bins.
        let f_max = F_MAX.min(rate as f32 / 2.0 - 1.0);
        let (m0, m1) = (hz_to_mel(F_MIN), hz_to_mel(f_max));
        let edges: Vec<f32> = (0..BANDS + 2)
            .map(|i| mel_to_hz(m0 + (m1 - m0) * i as f32 / (BANDS + 1) as f32))
            .collect();
        let bin_hz = rate as f32 / n_fft as f32;
        let filters = (0..BANDS)
            .map(|b| {
                let (lo, mid, hi) = (edges[b], edges[b + 1], edges[b + 2]);
                (0..=n_fft / 2)
                    .filter_map(|k| {
                        let f = k as f32 * bin_hz;
                        let w = if f > lo && f <= mid {
                            (f - lo) / (mid - lo)
                        } else if f > mid && f < hi {
                            (hi - f) / (hi - mid)
                        } else {
                            0.0
                        };
                        (w > 0.0).then_some((k, w))
                    })
                    .collect()
            })
            .collect();
        Self { rate, win, hop, fft, hann, filters, threshold: 0.35, floor: -9.0 }
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// Log band energies per frame (frames x BANDS).
    pub fn energies(&self, samples: &[f32]) -> Vec<[f32; BANDS]> {
        if samples.len() < self.win {
            return Vec::new();
        }
        let mut input = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        let mut out = Vec::with_capacity((samples.len() - self.win) / self.hop + 1);
        let mut start = 0;
        while start + self.win <= samples.len() {
            input.iter_mut().for_each(|x| *x = 0.0);
            for (i, (s, w)) in samples[start..start + self.win].iter().zip(&self.hann).enumerate() {
                input[i] = s * w;
            }
            if self.fft.process(&mut input, &mut spectrum).is_ok() {
                let mut frame = [0f32; BANDS];
                for (b, filt) in self.filters.iter().enumerate() {
                    let e: f32 = filt.iter().map(|(k, w)| w * spectrum[*k].norm_sqr()).sum();
                    frame[b] = (e + 1e-10).ln();
                }
                out.push(frame);
            }
            start += self.hop;
        }
        out
    }

    /// Delta-encoded sparse spike frames: band indices that fired in each frame.
    pub fn spikes(&self, samples: &[f32]) -> Vec<Vec<u16>> {
        let mut level = [f32::NEG_INFINITY; BANDS];
        self.energies(samples)
            .into_iter()
            .map(|frame| {
                let mut fired = Vec::new();
                for (b, e) in frame.iter().enumerate() {
                    if *e < self.floor {
                        level[b] = level[b].min(*e); // silence re-arms the band
                        continue;
                    }
                    if *e - level[b] > self.threshold {
                        fired.push(b as u16);
                        level[b] = *e;
                    } else {
                        // Slow decay so sustained sounds keep firing sparsely.
                        level[b] = level[b].max(*e - 1.0) - 0.05;
                    }
                }
                fired
            })
            .collect()
    }
}

/// Parse 16-bit PCM mono WAV (as Piper produces) into f32 samples and rate.
pub fn wav_pcm16_mono(wav: &[u8]) -> Option<(Vec<f32>, u32)> {
    if wav.len() < 44 || &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return None;
    }
    let mut i = 12;
    let (mut rate, mut channels, mut bits) = (0u32, 0u16, 0u16);
    while i + 8 <= wav.len() {
        let id = &wav[i..i + 4];
        let len = u32::from_le_bytes(wav[i + 4..i + 8].try_into().ok()?) as usize;
        let body = i + 8;
        if id == b"fmt " && body + 16 <= wav.len() {
            channels = u16::from_le_bytes([wav[body + 2], wav[body + 3]]);
            rate = u32::from_le_bytes(wav[body + 4..body + 8].try_into().ok()?);
            bits = u16::from_le_bytes([wav[body + 14], wav[body + 15]]);
        } else if id == b"data" {
            if channels != 1 || bits != 16 || rate == 0 {
                return None;
            }
            let end = (body + len).min(wav.len());
            let samples = wav[body..end]
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                .collect();
            return Some((samples, rate));
        }
        i = body + len + (len & 1);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, hz: f32, secs: f32, amp: f32) -> Vec<f32> {
        (0..(rate as f32 * secs) as usize)
            .map(|i| amp * (2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin())
            .collect()
    }

    #[test]
    fn silence_produces_no_spikes() {
        let c = Cochlea::new(16_000);
        assert!(c.spikes(&vec![0.0; 16_000]).iter().all(Vec::is_empty));
    }

    #[test]
    fn tone_onset_fires_the_matching_band_and_frames_are_10ms() {
        let c = Cochlea::new(16_000);
        let mut audio = vec![0.0; 3200]; // 200 ms silence
        audio.extend(tone(16_000, 1000.0, 0.5, 0.5));
        let frames = c.spikes(&audio);
        assert!((68..=72).contains(&frames.len()), "~70 frames of 10 ms, got {}", frames.len());
        let first = frames.iter().position(|f| !f.is_empty()).expect("onset spike");
        assert!((17..=21).contains(&first), "onset near 200 ms, frame {first}");
        // The band containing 1 kHz must be among the first spikes.
        let e = c.energies(&tone(16_000, 1000.0, 0.1, 0.5));
        let peak = (0..BANDS).max_by(|a, b| e[3][*a].total_cmp(&e[3][*b])).unwrap() as u16;
        assert!(frames[first].contains(&peak));
        // Sparse: far fewer events than frames x bands.
        let events: usize = frames.iter().map(Vec::len).sum();
        assert!(events < frames.len() * BANDS / 4);
    }

    #[test]
    fn different_tones_use_different_bands() {
        let c = Cochlea::new(22_050);
        let lo: std::collections::BTreeSet<u16> = c.spikes(&tone(22_050, 300.0, 0.3, 0.5)).concat().into_iter().collect();
        let hi: std::collections::BTreeSet<u16> = c.spikes(&tone(22_050, 4000.0, 0.3, 0.5)).concat().into_iter().collect();
        assert!(lo.iter().max() < hi.iter().min(), "low {lo:?} vs high {hi:?}");
    }

    #[test]
    fn parses_piper_style_wav() {
        let pcm: Vec<u8> = [0i16, 16384, -16384].iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut wav = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        wav.extend(16u32.to_le_bytes());
        wav.extend(1u16.to_le_bytes()); // PCM
        wav.extend(1u16.to_le_bytes()); // mono
        wav.extend(22_050u32.to_le_bytes());
        wav.extend((22_050u32 * 2).to_le_bytes());
        wav.extend(2u16.to_le_bytes());
        wav.extend(16u16.to_le_bytes());
        wav.extend(b"data");
        wav.extend((pcm.len() as u32).to_le_bytes());
        wav.extend(&pcm);
        let (s, r) = wav_pcm16_mono(&wav).unwrap();
        assert_eq!(r, 22_050);
        assert_eq!(s, vec![0.0, 0.5, -0.5]);
        assert!(wav_pcm16_mono(b"not a wav file at all, definitely not one").is_none());
    }
}
