use std::io::Cursor;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use serde::{Deserialize, Serialize};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::conv::IntoSample;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::sample::Sample;
use thiserror::Error;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tracing::{error, info};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[derive(Parser, Debug)]
#[command(name = "nmstt")]
#[command(about = "NeuralMimicry speech-to-text service")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "127.0.0.1:7079")]
    bind: String,
    #[arg(long, default_value = "en-GB")]
    lang: String,
    #[arg(long, default_value_t = 2)]
    threads: usize,
    #[arg(long, default_value_t = 0)]
    workers: usize,
    #[arg(long, default_value_t = 8_000_000)]
    max_audio_bytes: usize,
    #[arg(long, default_value_t = false)]
    translate: bool,
}

#[derive(Clone)]
struct AppState {
    contexts: Arc<Vec<Arc<Mutex<WhisperContext>>>>,
    concurrency: Arc<Semaphore>,
    rr: Arc<AtomicUsize>,
    default_lang: String,
    threads_per_request: usize,
    translate: bool,
    max_audio_bytes: usize,
    gesture_enabled: bool,
    bsl_enabled: bool,
    default_gesture_mode: String,
    default_avatar_mode: String,
    builtin_prompt: Option<String>,
    allow_client_prompt: bool,
    canonicalize_entities: bool,
    default_collaboration_mode: bool,
}

#[derive(Serialize)]
struct SttResponse {
    status: &'static str,
    text: String,
    lang: String,
    gesture_mode: String,
    avatar_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    bsl_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    avatar_motion: Option<AvatarMotion>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gesture_summary: Option<GestureSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gesture_timeline: Option<Vec<GestureTimelineEntry>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_analysis: Option<AudioAnalysis>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speaker_segments: Option<Vec<SpeakerSegment>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    collaboration_mode: Option<bool>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: &'static str,
}

#[derive(Deserialize)]
struct GesturePlanRequest {
    text: String,
    #[serde(default, alias = "gestureMode", alias = "motion_style", alias = "motionStyle")]
    gesture_mode: Option<String>,
    #[serde(default, alias = "avatarMode")]
    avatar_mode: Option<String>,
    #[serde(default, alias = "officeMode")]
    office_mode: Option<bool>,
}

#[derive(Serialize)]
struct GesturePlanResponse {
    status: &'static str,
    text: String,
    gesture_mode: String,
    avatar_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    bsl_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    avatar_motion: Option<AvatarMotion>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gesture_summary: Option<GestureSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gesture_timeline: Option<Vec<GestureTimelineEntry>>,
}

#[derive(Serialize, Clone)]
struct HandPose {
    thumb: f32,
    index: f32,
    middle: f32,
    ring: f32,
    pinky: f32,
}

#[derive(Serialize, Clone)]
struct Pose {
    #[serde(rename = "headYaw")]
    head_yaw: f32,
    #[serde(rename = "headPitch")]
    head_pitch: f32,
    #[serde(rename = "spineLean")]
    spine_lean: f32,
    #[serde(rename = "shoulderRise")]
    shoulder_rise: f32,
    #[serde(rename = "leftShoulderPitch")]
    left_shoulder_pitch: f32,
    #[serde(rename = "rightShoulderPitch")]
    right_shoulder_pitch: f32,
    #[serde(rename = "leftShoulderRoll")]
    left_shoulder_roll: f32,
    #[serde(rename = "rightShoulderRoll")]
    right_shoulder_roll: f32,
    #[serde(rename = "leftElbow")]
    left_elbow: f32,
    #[serde(rename = "rightElbow")]
    right_elbow: f32,
    #[serde(rename = "leftWristYaw")]
    left_wrist_yaw: f32,
    #[serde(rename = "rightWristYaw")]
    right_wrist_yaw: f32,
    #[serde(rename = "leftHip")]
    left_hip: f32,
    #[serde(rename = "rightHip")]
    right_hip: f32,
    #[serde(rename = "leftKnee")]
    left_knee: f32,
    #[serde(rename = "rightKnee")]
    right_knee: f32,
    #[serde(rename = "leftAnkle")]
    left_ankle: f32,
    #[serde(rename = "rightAnkle")]
    right_ankle: f32,
    #[serde(rename = "leftHand")]
    left_hand: HandPose,
    #[serde(rename = "rightHand")]
    right_hand: HandPose,
}

#[derive(Serialize, Clone)]
struct MotionKeyframe {
    t: u32,
    pose: Pose,
}

#[derive(Serialize, Clone)]
struct AvatarMotion {
    #[serde(rename = "duration_ms")]
    duration_ms: u32,
    keyframes: Vec<MotionKeyframe>,
}

#[derive(Serialize, Clone)]
struct GestureSummary {
    style: String,
    #[serde(rename = "token_count")]
    token_count: usize,
}

#[derive(Serialize, Clone)]
struct GestureTimelineEntry {
    word: String,
    intent: String,
    template: String,
    #[serde(rename = "start_ms")]
    start_ms: u32,
    #[serde(rename = "end_ms")]
    end_ms: u32,
}

#[derive(Clone)]
struct GestureToken {
    word: String,
    punctuation: String,
}

#[derive(Clone, Copy)]
struct MotionBlend {
    shoulder_pitch: f32,
    shoulder_roll: f32,
    elbow: f32,
    wrist: f32,
    hand_open: f32,
}

struct GesturePlan {
    motion: AvatarMotion,
    summary: GestureSummary,
    timeline: Vec<GestureTimelineEntry>,
    bsl_text: Option<String>,
}

#[derive(Serialize, Clone)]
struct AudioAnalysis {
    #[serde(rename = "speech_ratio")]
    speech_ratio: f32,
    #[serde(rename = "speech_confidence")]
    speech_confidence: f32,
    #[serde(rename = "noise_level")]
    noise_level: String,
    #[serde(rename = "noise_rms")]
    noise_rms: f32,
    #[serde(rename = "speech_rms")]
    speech_rms: f32,
    #[serde(rename = "snr_db")]
    snr_db: f32,
    #[serde(rename = "speaker_count_estimate")]
    speaker_count_estimate: u8,
    #[serde(rename = "speaker_turn_count")]
    speaker_turn_count: usize,
    #[serde(rename = "background_noise_detected")]
    background_noise_detected: bool,
    #[serde(rename = "collaboration_likely")]
    collaboration_likely: bool,
}

#[derive(Serialize, Clone)]
struct SpeakerSegment {
    speaker: String,
    text: String,
    #[serde(rename = "start_ms")]
    start_ms: u32,
    #[serde(rename = "end_ms")]
    end_ms: u32,
    confidence: f32,
}

struct InferenceResult {
    text: String,
    audio_analysis: AudioAnalysis,
    speaker_segments: Vec<SpeakerSegment>,
}

#[derive(Clone)]
struct SegmentDraft {
    text: String,
    start_ms: u32,
    end_ms: u32,
    speaker_turn_next: bool,
    rms: f32,
}

#[derive(Clone, Copy)]
struct AudioMetrics {
    speech_ratio: f32,
    speech_confidence: f32,
    noise_rms: f32,
    speech_rms: f32,
    snr_db: f32,
    background_noise_detected: bool,
    noise_level: &'static str,
}

#[derive(Error, Debug)]
enum SttError {
    #[error("invalid_audio")]
    InvalidAudio,
    #[error("unsupported_format")]
    UnsupportedFormat,
    #[error("stt_failed")]
    SttFailed,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let model_path = args.model.to_string_lossy().to_string();
    let threads = args.threads.max(1);
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let workers = if args.workers > 0 {
        args.workers
    } else {
        std::cmp::max(1, available / threads)
    };

    let mut contexts: Vec<Arc<Mutex<WhisperContext>>> = Vec::with_capacity(workers);
    for _ in 0..workers {
        let ctx =
            match WhisperContext::new_with_params(&model_path, WhisperContextParameters::default())
            {
                Ok(ctx) => ctx,
                Err(err) => {
                    error!("Failed to load model: {err}");
                    std::process::exit(1);
                }
            };
        contexts.push(Arc::new(Mutex::new(ctx)));
    }

    let state = AppState {
        contexts: Arc::new(contexts),
        concurrency: Arc::new(Semaphore::new(workers)),
        rr: Arc::new(AtomicUsize::new(0)),
        default_lang: args.lang.clone(),
        threads_per_request: threads,
        translate: args.translate,
        max_audio_bytes: args.max_audio_bytes,
        gesture_enabled: env_flag_any(&["NMSTT_GESTURE_ENABLED", "REFINER_STT_GESTURE_ENABLED"], true),
        bsl_enabled: env_flag_any(&["NMSTT_BSL_ENABLED", "REFINER_STT_BSL_ENABLED"], true),
        default_gesture_mode: env_string_any(&["NMSTT_GESTURE_DEFAULT_MODE", "REFINER_STT_GESTURE_DEFAULT_MODE"])
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "gesticulation".to_string()),
        default_avatar_mode: env_string_any(&[
            "NMSTT_GESTURE_DEFAULT_AVATAR_MODE",
            "REFINER_STT_GESTURE_DEFAULT_AVATAR_MODE",
        ])
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "chat".to_string()),
        builtin_prompt: default_stt_context_prompt(),
        allow_client_prompt: env_flag_any(&["NMSTT_PROMPT_ALLOW_CLIENT", "REFINER_STT_PROMPT_ALLOW_CLIENT"], false),
        canonicalize_entities: env_flag_any(
            &["NMSTT_CANONICALIZE_ENTITIES", "REFINER_STT_CANONICALIZE_ENTITIES"],
            true,
        ),
        default_collaboration_mode: env_flag_any(
            &["NMSTT_COLLABORATION_DEFAULT", "REFINER_STT_COLLABORATION_DEFAULT"],
            false,
        ),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/transcribe", post(transcribe))
        .route("/gesture-plan", post(gesture_plan))
        .with_state(Arc::new(state));

    let addr: SocketAddr = match args.bind.parse() {
        Ok(addr) => addr,
        Err(err) => {
            error!("Invalid bind address: {err}");
            std::process::exit(1);
        }
    };

    info!(
        "nmstt listening on {addr} | cpu={} workers={} threads_per_request={} max_audio_bytes={} translate={}",
        available,
        workers,
        threads,
        args.max_audio_bytes,
        args.translate
    );

    let listener = match TcpListener::bind(addr).await {
        Ok(listener) => listener,
        Err(err) => {
            error!("Failed to bind listener: {err}");
            std::process::exit(1);
        }
    };

    if let Err(err) = axum::serve(listener, app).await {
        error!("Server error: {err}");
    }
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

async fn transcribe(State(state): State<Arc<AppState>>, mut multipart: Multipart) -> Response {
    let mut audio: Option<Vec<u8>> = None;
    let mut lang: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut gesture_mode: Option<String> = None;
    let mut avatar_mode: Option<String> = None;
    let mut office_mode: Option<bool> = None;
    let mut collaboration_mode: Option<bool> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or_default().to_string();
        let name_key = normalize_token(&name);
        if matches!(name_key.as_str(), "audio" | "file") {
            match field.bytes().await {
                Ok(bytes) => {
                    if state.max_audio_bytes > 0 && bytes.len() > state.max_audio_bytes {
                        return (
                            StatusCode::PAYLOAD_TOO_LARGE,
                            Json(ErrorResponse {
                                error: "audio_too_large",
                            }),
                        )
                            .into_response();
                    }
                    audio = Some(bytes.to_vec())
                }
                Err(_) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: "invalid_audio",
                        }),
                    )
                        .into_response();
                }
            }
        } else if matches!(name_key.as_str(), "lang" | "language") {
            if let Ok(text) = field.text().await {
                if !text.trim().is_empty() {
                    lang = Some(text.trim().to_string());
                }
            }
        } else if name_key == "prompt" {
            if let Ok(text) = field.text().await {
                if !text.trim().is_empty() {
                    prompt = Some(text);
                }
            }
        } else if matches!(
            name_key.as_str(),
            "gesture_mode" | "gesturemode" | "motion_style" | "motionstyle"
        ) {
            if let Ok(text) = field.text().await {
                if !text.trim().is_empty() {
                    gesture_mode = Some(text.trim().to_string());
                }
            }
        } else if matches!(name_key.as_str(), "avatar_mode" | "avatarmode") {
            if let Ok(text) = field.text().await {
                if !text.trim().is_empty() {
                    avatar_mode = Some(text.trim().to_string());
                }
            }
        } else if matches!(name_key.as_str(), "office_mode" | "officemode") {
            if let Ok(text) = field.text().await {
                office_mode = parse_boolish(Some(text.as_str()));
            }
        } else if matches!(
            name_key.as_str(),
            "collaboration_mode"
                | "collaborationmode"
                | "collaboration"
                | "multi_speaker"
                | "multispeaker"
                | "multi_speaker_mode"
                | "multispeakermode"
        ) {
            if let Ok(text) = field.text().await {
                collaboration_mode = parse_boolish(Some(text.as_str()));
            }
        }
    }

    let audio = match audio {
        Some(data) if !data.is_empty() => data,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "audio_required",
                }),
            )
                .into_response();
        }
    };

    let lang = sanitize_lang(lang.as_deref(), &state.default_lang);
    let client_prompt = if state.allow_client_prompt {
        sanitize_prompt(prompt.as_deref())
    } else {
        None
    };
    let prompt = merge_stt_prompts(state.builtin_prompt.as_deref(), client_prompt.as_deref());
    let selected_gesture_mode = sanitize_gesture_mode(
        gesture_mode.as_deref(),
        &state.default_gesture_mode,
        state.bsl_enabled,
    );
    let selected_avatar_mode = sanitize_avatar_mode(
        avatar_mode.as_deref(),
        office_mode,
        &state.default_avatar_mode,
    );
    let selected_collaboration_mode =
        collaboration_mode.unwrap_or(state.default_collaboration_mode);
    let permit = match state.concurrency.clone().acquire_owned().await {
        Ok(permit) => permit,
        Err(_) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    error: "capacity_unavailable",
                }),
            )
                .into_response();
        }
    };

    let idx = state.rr.fetch_add(1, Ordering::Relaxed) % state.contexts.len();
    let ctx = state.contexts[idx].clone();
    let threads = state.threads_per_request;
    let translate = state.translate;
    let lang_for_inference = lang.clone();
    let prompt_for_inference = prompt.clone();
    let collaboration_for_inference = selected_collaboration_mode;

    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        run_inference(
            ctx,
            audio,
            &lang_for_inference,
            threads,
            translate,
            prompt_for_inference.as_deref(),
            collaboration_for_inference,
        )
    })
    .await;

    match result {
        Ok(Ok(inference)) => {
            let text = if state.canonicalize_entities {
                canonicalize_transcript_entities(&inference.text)
            } else {
                inference.text
            };
            let mut speaker_segments = inference.speaker_segments;
            if state.canonicalize_entities {
                for segment in &mut speaker_segments {
                    segment.text = canonicalize_transcript_entities(&segment.text);
                }
            }
            let plan = if state.gesture_enabled {
                plan_gesture_motion(&text, &selected_gesture_mode, &selected_avatar_mode)
            } else {
                None
            };
            (
                StatusCode::OK,
                Json(SttResponse {
                    status: "ok",
                    text,
                    lang,
                    gesture_mode: selected_gesture_mode.clone(),
                    avatar_mode: selected_avatar_mode.clone(),
                    bsl_text: plan.as_ref().and_then(|p| p.bsl_text.clone()),
                    avatar_motion: plan.as_ref().map(|p| p.motion.clone()),
                    gesture_summary: plan.as_ref().map(|p| p.summary.clone()),
                    gesture_timeline: plan.as_ref().map(|p| p.timeline.clone()),
                    audio_analysis: Some(inference.audio_analysis),
                    speaker_segments: if speaker_segments.is_empty() {
                        None
                    } else {
                        Some(speaker_segments)
                    },
                    collaboration_mode: Some(selected_collaboration_mode),
                }),
            )
                .into_response()
        }
        Ok(Err(SttError::InvalidAudio)) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "invalid_audio",
            }),
        )
            .into_response(),
        Ok(Err(SttError::UnsupportedFormat)) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "unsupported_format",
            }),
        )
            .into_response(),
        Ok(Err(SttError::SttFailed)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "stt_failed",
            }),
        )
            .into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "stt_task_failed",
            }),
        )
            .into_response(),
    }
}

async fn gesture_plan(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<GesturePlanRequest>,
) -> Response {
    let text = payload.text.trim().to_string();
    if text.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "text_required",
            }),
        )
            .into_response();
    }
    let selected_gesture_mode = sanitize_gesture_mode(
        payload.gesture_mode.as_deref(),
        &state.default_gesture_mode,
        state.bsl_enabled,
    );
    let selected_avatar_mode = sanitize_avatar_mode(
        payload.avatar_mode.as_deref(),
        payload.office_mode,
        &state.default_avatar_mode,
    );
    let plan = if state.gesture_enabled {
        plan_gesture_motion(&text, &selected_gesture_mode, &selected_avatar_mode)
    } else {
        None
    };
    (
        StatusCode::OK,
        Json(GesturePlanResponse {
            status: "ok",
            text,
            gesture_mode: selected_gesture_mode,
            avatar_mode: selected_avatar_mode,
            bsl_text: plan.as_ref().and_then(|p| p.bsl_text.clone()),
            avatar_motion: plan.as_ref().map(|p| p.motion.clone()),
            gesture_summary: plan.as_ref().map(|p| p.summary.clone()),
            gesture_timeline: plan.as_ref().map(|p| p.timeline.clone()),
        }),
    )
        .into_response()
}

fn env_string_any(names: &[&str]) -> Option<String> {
    for name in names {
        if let Ok(value) = std::env::var(name) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        },
        Err(_) => default,
    }
}

fn env_flag_any(names: &[&str], default: bool) -> bool {
    for name in names {
        if std::env::var(name).is_ok() {
            return env_flag(name, default);
        }
    }
    default
}

fn parse_boolish(input: Option<&str>) -> Option<bool> {
    let raw = input?.trim();
    if raw.is_empty() {
        return None;
    }
    match raw.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" | "office" => Some(true),
        "0" | "false" | "no" | "off" | "chat" => Some(false),
        _ => None,
    }
}

fn normalize_token(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut last_sep = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_sep = false;
        } else if !last_sep && !out.is_empty() {
            out.push('_');
            last_sep = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    out
}

fn sanitize_gesture_mode(input: Option<&str>, fallback: &str, bsl_enabled: bool) -> String {
    let fallback_norm = normalize_gesture_mode(fallback).unwrap_or("gesticulation");
    let selected = normalize_gesture_mode(input.unwrap_or(fallback)).unwrap_or(fallback_norm);
    if selected == "bsl" && !bsl_enabled {
        "gesticulation".to_string()
    } else {
        selected.to_string()
    }
}

fn normalize_gesture_mode(value: &str) -> Option<&'static str> {
    match normalize_token(value).as_str() {
        "bsl"
        | "sign"
        | "signing"
        | "sign_language"
        | "bsl_signing"
        | "british_sign_language"
        | "bsl_british_sign_language" => Some("bsl"),
        "gesticulation"
        | "gesture"
        | "gestures"
        | "default"
        | "semantic_gesticulation"
        | "conversational_gesticulation" => Some("gesticulation"),
        _ => None,
    }
}

fn sanitize_avatar_mode(input: Option<&str>, office_mode: Option<bool>, fallback: &str) -> String {
    if let Some(flag) = office_mode {
        return if flag {
            "office".to_string()
        } else {
            "chat".to_string()
        };
    }
    let selected = normalize_avatar_mode(input.unwrap_or(fallback)).unwrap_or("chat");
    selected.to_string()
}

fn normalize_avatar_mode(value: &str) -> Option<&'static str> {
    match normalize_token(value).as_str() {
        "office" | "office_mode" | "desk" | "work" => Some("office"),
        "chat" | "chat_mode" | "conversation" => Some("chat"),
        _ => None,
    }
}

fn clamp(value: f32, low: f32, high: f32) -> f32 {
    if value < low {
        low
    } else if value > high {
        high
    } else {
        value
    }
}

fn clamp_signed(value: f32) -> f32 {
    clamp(value, -1.0, 1.0)
}

fn clamp_unsigned(value: f32) -> f32 {
    clamp(value, 0.0, 1.0)
}

fn clamp_u32(value: u32, low: u32, high: u32) -> u32 {
    if value < low {
        low
    } else if value > high {
        high
    } else {
        value
    }
}

fn base_hand_pose() -> HandPose {
    HandPose {
        thumb: 0.18,
        index: 0.08,
        middle: 0.08,
        ring: 0.10,
        pinky: 0.14,
    }
}

fn base_pose() -> Pose {
    Pose {
        head_yaw: 0.0,
        head_pitch: 0.0,
        spine_lean: 0.0,
        shoulder_rise: 0.0,
        left_shoulder_pitch: 0.0,
        right_shoulder_pitch: 0.0,
        left_shoulder_roll: 0.0,
        right_shoulder_roll: 0.0,
        left_elbow: 0.32,
        right_elbow: 0.32,
        left_wrist_yaw: 0.0,
        right_wrist_yaw: 0.0,
        left_hip: 0.0,
        right_hip: 0.0,
        left_knee: 0.20,
        right_knee: 0.20,
        left_ankle: 0.0,
        right_ankle: 0.0,
        left_hand: base_hand_pose(),
        right_hand: base_hand_pose(),
    }
}

fn normalize_pose(mut pose: Pose) -> Pose {
    pose.head_yaw = clamp_signed(pose.head_yaw);
    pose.head_pitch = clamp_signed(pose.head_pitch);
    pose.spine_lean = clamp_signed(pose.spine_lean);
    pose.shoulder_rise = clamp_signed(pose.shoulder_rise);
    pose.left_shoulder_pitch = clamp_signed(pose.left_shoulder_pitch);
    pose.right_shoulder_pitch = clamp_signed(pose.right_shoulder_pitch);
    pose.left_shoulder_roll = clamp_signed(pose.left_shoulder_roll);
    pose.right_shoulder_roll = clamp_signed(pose.right_shoulder_roll);
    pose.left_elbow = clamp_unsigned(pose.left_elbow);
    pose.right_elbow = clamp_unsigned(pose.right_elbow);
    pose.left_wrist_yaw = clamp_signed(pose.left_wrist_yaw);
    pose.right_wrist_yaw = clamp_signed(pose.right_wrist_yaw);
    pose.left_hip = clamp_signed(pose.left_hip);
    pose.right_hip = clamp_signed(pose.right_hip);
    pose.left_knee = clamp_unsigned(pose.left_knee);
    pose.right_knee = clamp_unsigned(pose.right_knee);
    pose.left_ankle = clamp_signed(pose.left_ankle);
    pose.right_ankle = clamp_signed(pose.right_ankle);
    pose.left_hand.thumb = clamp_unsigned(pose.left_hand.thumb);
    pose.left_hand.index = clamp_unsigned(pose.left_hand.index);
    pose.left_hand.middle = clamp_unsigned(pose.left_hand.middle);
    pose.left_hand.ring = clamp_unsigned(pose.left_hand.ring);
    pose.left_hand.pinky = clamp_unsigned(pose.left_hand.pinky);
    pose.right_hand.thumb = clamp_unsigned(pose.right_hand.thumb);
    pose.right_hand.index = clamp_unsigned(pose.right_hand.index);
    pose.right_hand.middle = clamp_unsigned(pose.right_hand.middle);
    pose.right_hand.ring = clamp_unsigned(pose.right_hand.ring);
    pose.right_hand.pinky = clamp_unsigned(pose.right_hand.pinky);
    pose
}

fn blend_pose(a: &Pose, b: &Pose, alpha: f32) -> Pose {
    let t = clamp(alpha, 0.0, 1.0);
    let inv = 1.0 - t;
    normalize_pose(Pose {
        head_yaw: a.head_yaw * inv + b.head_yaw * t,
        head_pitch: a.head_pitch * inv + b.head_pitch * t,
        spine_lean: a.spine_lean * inv + b.spine_lean * t,
        shoulder_rise: a.shoulder_rise * inv + b.shoulder_rise * t,
        left_shoulder_pitch: a.left_shoulder_pitch * inv + b.left_shoulder_pitch * t,
        right_shoulder_pitch: a.right_shoulder_pitch * inv + b.right_shoulder_pitch * t,
        left_shoulder_roll: a.left_shoulder_roll * inv + b.left_shoulder_roll * t,
        right_shoulder_roll: a.right_shoulder_roll * inv + b.right_shoulder_roll * t,
        left_elbow: a.left_elbow * inv + b.left_elbow * t,
        right_elbow: a.right_elbow * inv + b.right_elbow * t,
        left_wrist_yaw: a.left_wrist_yaw * inv + b.left_wrist_yaw * t,
        right_wrist_yaw: a.right_wrist_yaw * inv + b.right_wrist_yaw * t,
        left_hip: a.left_hip * inv + b.left_hip * t,
        right_hip: a.right_hip * inv + b.right_hip * t,
        left_knee: a.left_knee * inv + b.left_knee * t,
        right_knee: a.right_knee * inv + b.right_knee * t,
        left_ankle: a.left_ankle * inv + b.left_ankle * t,
        right_ankle: a.right_ankle * inv + b.right_ankle * t,
        left_hand: HandPose {
            thumb: a.left_hand.thumb * inv + b.left_hand.thumb * t,
            index: a.left_hand.index * inv + b.left_hand.index * t,
            middle: a.left_hand.middle * inv + b.left_hand.middle * t,
            ring: a.left_hand.ring * inv + b.left_hand.ring * t,
            pinky: a.left_hand.pinky * inv + b.left_hand.pinky * t,
        },
        right_hand: HandPose {
            thumb: a.right_hand.thumb * inv + b.right_hand.thumb * t,
            index: a.right_hand.index * inv + b.right_hand.index * t,
            middle: a.right_hand.middle * inv + b.right_hand.middle * t,
            ring: a.right_hand.ring * inv + b.right_hand.ring * t,
            pinky: a.right_hand.pinky * inv + b.right_hand.pinky * t,
        },
    })
}

fn mode_amplitude(gesture_mode: &str, avatar_mode: &str) -> f32 {
    if gesture_mode == "bsl" {
        if avatar_mode == "office" {
            1.0
        } else {
            0.74
        }
    } else if avatar_mode == "office" {
        0.76
    } else {
        0.50
    }
}

fn rest_pose(gesture_mode: &str, amplitude: f32) -> Pose {
    let mut pose = base_pose();
    if gesture_mode == "bsl" {
        pose.head_pitch += 0.03 * amplitude;
        pose.shoulder_rise += 0.12 * amplitude;
        pose.left_shoulder_pitch += 0.22 * amplitude;
        pose.right_shoulder_pitch += 0.22 * amplitude;
        pose.left_shoulder_roll += 0.44 * amplitude;
        pose.right_shoulder_roll -= 0.44 * amplitude;
        pose.left_elbow += 0.10 * amplitude;
        pose.right_elbow += 0.10 * amplitude;
        pose.left_wrist_yaw -= 0.12 * amplitude;
        pose.right_wrist_yaw += 0.12 * amplitude;
        pose.left_hand.thumb += 0.04 * amplitude;
        pose.left_hand.index += 0.02 * amplitude;
        pose.left_hand.middle += 0.02 * amplitude;
        pose.left_hand.ring += 0.03 * amplitude;
        pose.left_hand.pinky += 0.03 * amplitude;
        pose.right_hand = pose.left_hand.clone();
    } else {
        pose.head_pitch += 0.03 * amplitude;
        pose.shoulder_rise += 0.08 * amplitude;
        pose.left_shoulder_pitch += 0.16 * amplitude;
        pose.right_shoulder_pitch += 0.16 * amplitude;
        pose.left_shoulder_roll += 0.26 * amplitude;
        pose.right_shoulder_roll -= 0.26 * amplitude;
        pose.left_wrist_yaw -= 0.08 * amplitude;
        pose.right_wrist_yaw += 0.08 * amplitude;
        pose.left_elbow -= 0.05 * amplitude;
        pose.right_elbow -= 0.05 * amplitude;
        pose.left_hand.thumb += 0.03 * amplitude;
        pose.left_hand.index += 0.06 * amplitude;
        pose.left_hand.middle += 0.08 * amplitude;
        pose.left_hand.ring += 0.10 * amplitude;
        pose.left_hand.pinky += 0.12 * amplitude;
        pose.right_hand = pose.left_hand.clone();
    }
    normalize_pose(pose)
}

fn split_word_and_punctuation(raw: &str) -> Option<GestureToken> {
    let mut word = String::new();
    let mut punctuation = String::new();
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '\'' {
            word.push(ch);
        } else {
            punctuation.push(ch);
        }
    }
    if word.is_empty() {
        None
    } else {
        Some(GestureToken { word, punctuation })
    }
}

fn tokenize_transcript(text: &str, max_tokens: usize) -> Vec<GestureToken> {
    let mut out = Vec::new();
    for part in text.split_whitespace() {
        if out.len() >= max_tokens {
            break;
        }
        if let Some(token) = split_word_and_punctuation(part) {
            out.push(token);
        }
    }
    out
}

fn normalized_word(word: &str) -> String {
    word.to_ascii_lowercase()
}

fn classify_intent(word: &str) -> &'static str {
    let lower = normalized_word(word);
    let plain = lower.replace('\'', "");
    if plain.chars().all(|ch| ch.is_ascii_digit()) {
        return "number";
    }
    match plain.as_str() {
        "hello" | "hi" | "hey" | "welcome" | "morning" | "afternoon" | "evening" => "greeting",
        "what" | "why" | "when" | "where" | "who" | "how" | "which" | "can" | "could" | "would"
        | "should" => "question",
        "yes" | "yeah" | "yep" | "sure" | "correct" | "agree" | "absolutely" | "definitely" => {
            "affirm"
        }
        "no" | "not" | "never" | "dont" | "cannot" | "cant" | "without" => "negate",
        "please" | "thanks" | "thank" | "appreciate" | "sorry" => "polite",
        "you" | "your" | "yours" | "we" | "our" | "us" | "they" | "their" | "them" | "i" | "me"
        | "my" => "directional",
        "build" | "create" | "deliver" | "deploy" | "launch" | "ship" | "run" | "send" | "plan"
        | "improve" | "analyze" | "assist" | "help" | "translate" | "sign" | "gesture"
        | "speak" => "action",
        "neuralmimicry" | "aarnn" | "aaron" | "refiner" | "continuum" | "tracey" | "trace"
        | "neuromorphic" | "avatar" | "office" | "chat" | "bsl" | "language" | "paul" | "isaac"
        | "isaacs" | "kirsten" | "scarlett" | "harriet" | "melissa" | "michael" | "christopher"
        | "benjamin" | "rebecca" => "topic",
        "the" | "a" | "an" | "to" | "of" | "in" | "on" | "at" | "for" | "and" | "or" | "if"
        | "is" | "are" | "was" | "were" | "be" | "it" | "as" | "with" | "by" | "from" => "filler",
        _ => "content",
    }
}

fn select_template(gesture_mode: &str, intent: &str, index: usize, word: &str) -> String {
    if gesture_mode == "bsl" {
        match intent {
            "greeting" | "question" | "affirm" | "negate" | "polite" | "topic" | "action"
            | "number" => intent.to_string(),
            "filler" => "rest".to_string(),
            _ => {
                if word.len() <= 2 {
                    "rest".to_string()
                } else if index % 2 == 0 {
                    "fingerspell_a".to_string()
                } else {
                    "fingerspell_b".to_string()
                }
            }
        }
    } else {
        match intent {
            "question" => "question".to_string(),
            "negate" => "negate".to_string(),
            "affirm" | "polite" | "greeting" => "acknowledge".to_string(),
            "directional" => "directional".to_string(),
            "filler" => "subtle".to_string(),
            "topic" | "action" => "emphasis".to_string(),
            _ => "explain".to_string(),
        }
    }
}

fn estimate_word_duration_ms(gesture_mode: &str, word: &str) -> u32 {
    let len = word.chars().count().max(1).min(12) as u32;
    if gesture_mode == "bsl" {
        clamp_u32(185 + len * 20, 165, 540)
    } else {
        clamp_u32(125 + len * 14, 110, 360)
    }
}

fn pause_after_word_ms(gesture_mode: &str, punctuation: &str) -> u32 {
    if punctuation.contains('.') || punctuation.contains('!') || punctuation.contains('?') {
        if gesture_mode == "bsl" {
            150
        } else {
            110
        }
    } else if punctuation.contains(',') || punctuation.contains(';') || punctuation.contains(':') {
        if gesture_mode == "bsl" {
            90
        } else {
            65
        }
    } else if gesture_mode == "bsl" {
        34
    } else {
        24
    }
}

fn word_intensity(gesture_mode: &str, intent: &str, word: &str) -> f32 {
    let len = word.chars().count() as f32;
    if gesture_mode == "bsl" {
        if intent == "filler" {
            return 0.58;
        }
        if matches!(
            intent,
            "question" | "negate" | "affirm" | "number" | "topic"
        ) {
            return 1.04;
        }
        return 0.90;
    }
    let mut score = 0.45 + len.min(10.0) * 0.03;
    if matches!(intent, "question" | "negate") {
        score += 0.20;
    }
    if matches!(intent, "topic" | "action") {
        score += 0.12;
    }
    if intent == "filler" {
        score *= 0.60;
    }
    clamp(score, 0.30, 1.10)
}

fn letter_hand_shape(letter: char, mirror: bool) -> HandPose {
    let lower = letter.to_ascii_lowercase();
    let idx = if lower.is_ascii_lowercase() {
        (lower as u8 - b'a') as usize
    } else {
        0
    };
    let row = idx / 5;
    let col = idx % 5;
    let curl_base = 0.10 + row as f32 * 0.16;
    let spread = (col as f32 - 2.0) * 0.08;
    HandPose {
        thumb: clamp(
            0.16 + (row % 3) as f32 * 0.08
                + if mirror {
                    spread * 0.35
                } else {
                    -spread * 0.35
                },
            0.0,
            1.0,
        ),
        index: clamp(
            curl_base + spread.abs() * if mirror { 0.55 } else { 0.35 },
            0.0,
            1.0,
        ),
        middle: clamp(
            curl_base + 0.06 + if col == 2 { 0.10 } else { 0.0 },
            0.0,
            1.0,
        ),
        ring: clamp(
            curl_base + 0.10 + if col >= 3 { 0.08 } else { 0.0 },
            0.0,
            1.0,
        ),
        pinky: clamp(
            curl_base + 0.14 + if col == 4 { 0.10 } else { 0.0 },
            0.0,
            1.0,
        ),
    }
}

fn apply_template(
    mut pose: Pose,
    gesture_mode: &str,
    template: &str,
    intensity: f32,
    index: usize,
    word: &str,
) -> Pose {
    let left_bias = if index % 2 == 0 { 1.0 } else { 0.86 };
    let right_bias = if index % 2 == 0 { 0.86 } else { 1.0 };
    let side_sway = if index % 2 == 0 { 0.06 } else { -0.06 };
    let blend = if gesture_mode == "bsl" {
        match template {
            "greeting" => MotionBlend {
                shoulder_pitch: 0.46,
                shoulder_roll: 0.68,
                elbow: 0.22,
                wrist: 0.44,
                hand_open: -0.08,
            },
            "question" => MotionBlend {
                shoulder_pitch: 0.38,
                shoulder_roll: 0.56,
                elbow: 0.20,
                wrist: 0.30,
                hand_open: 0.04,
            },
            "affirm" => MotionBlend {
                shoulder_pitch: 0.30,
                shoulder_roll: 0.40,
                elbow: 0.16,
                wrist: 0.28,
                hand_open: 0.30,
            },
            "negate" => MotionBlend {
                shoulder_pitch: 0.30,
                shoulder_roll: 0.50,
                elbow: 0.18,
                wrist: 0.40,
                hand_open: 0.34,
            },
            "polite" => MotionBlend {
                shoulder_pitch: 0.36,
                shoulder_roll: 0.46,
                elbow: 0.20,
                wrist: 0.08,
                hand_open: -0.03,
            },
            "topic" => MotionBlend {
                shoulder_pitch: 0.34,
                shoulder_roll: 0.50,
                elbow: 0.18,
                wrist: 0.34,
                hand_open: 0.18,
            },
            "action" => MotionBlend {
                shoulder_pitch: 0.42,
                shoulder_roll: 0.56,
                elbow: 0.24,
                wrist: 0.20,
                hand_open: 0.12,
            },
            "number" => MotionBlend {
                shoulder_pitch: 0.30,
                shoulder_roll: 0.44,
                elbow: 0.16,
                wrist: 0.38,
                hand_open: 0.16,
            },
            "fingerspell_a" | "fingerspell_b" => MotionBlend {
                shoulder_pitch: if template == "fingerspell_a" {
                    0.34
                } else {
                    0.32
                },
                shoulder_roll: if template == "fingerspell_a" {
                    0.50
                } else {
                    0.48
                },
                elbow: 0.20,
                wrist: if template == "fingerspell_a" {
                    0.22
                } else {
                    0.18
                },
                hand_open: 0.20,
            },
            _ => MotionBlend {
                shoulder_pitch: 0.20,
                shoulder_roll: 0.34,
                elbow: 0.14,
                wrist: 0.12,
                hand_open: 0.10,
            },
        }
    } else {
        match template {
            "question" => MotionBlend {
                shoulder_pitch: 0.24,
                shoulder_roll: 0.40,
                elbow: 0.14,
                wrist: 0.26,
                hand_open: 0.05,
            },
            "negate" => MotionBlend {
                shoulder_pitch: 0.26,
                shoulder_roll: 0.38,
                elbow: 0.18,
                wrist: 0.32,
                hand_open: 0.22,
            },
            "acknowledge" => MotionBlend {
                shoulder_pitch: 0.22,
                shoulder_roll: 0.30,
                elbow: 0.10,
                wrist: 0.14,
                hand_open: 0.14,
            },
            "directional" => MotionBlend {
                shoulder_pitch: 0.22,
                shoulder_roll: 0.34,
                elbow: 0.14,
                wrist: 0.28,
                hand_open: 0.10,
            },
            "emphasis" => MotionBlend {
                shoulder_pitch: 0.26,
                shoulder_roll: 0.40,
                elbow: 0.18,
                wrist: 0.24,
                hand_open: 0.16,
            },
            "subtle" => MotionBlend {
                shoulder_pitch: 0.12,
                shoulder_roll: 0.20,
                elbow: 0.08,
                wrist: 0.08,
                hand_open: 0.08,
            },
            _ => MotionBlend {
                shoulder_pitch: 0.20,
                shoulder_roll: 0.30,
                elbow: 0.12,
                wrist: 0.18,
                hand_open: 0.12,
            },
        }
    };

    pose.spine_lean += side_sway * intensity;
    pose.head_yaw += if gesture_mode == "gesticulation" {
        side_sway * 0.7 * intensity
    } else {
        side_sway * 0.25
    };
    pose.left_shoulder_pitch += blend.shoulder_pitch * intensity * left_bias;
    pose.right_shoulder_pitch += blend.shoulder_pitch * intensity * right_bias;
    pose.left_shoulder_roll += blend.shoulder_roll * intensity * left_bias;
    pose.right_shoulder_roll -= blend.shoulder_roll * intensity * right_bias;
    pose.left_elbow += blend.elbow * intensity * left_bias;
    pose.right_elbow += blend.elbow * intensity * right_bias;
    pose.left_wrist_yaw -= blend.wrist * intensity * left_bias;
    pose.right_wrist_yaw += blend.wrist * intensity * right_bias;

    let left_delta = blend.hand_open * intensity * (0.92 + 0.08 * left_bias);
    let right_delta = blend.hand_open * intensity * (0.92 + 0.08 * right_bias);
    pose.left_hand.thumb += left_delta + 0.02;
    pose.left_hand.index += left_delta;
    pose.left_hand.middle += left_delta + 0.02;
    pose.left_hand.ring += left_delta + 0.04;
    pose.left_hand.pinky += left_delta + 0.06;
    pose.right_hand.thumb += right_delta + 0.02;
    pose.right_hand.index += right_delta;
    pose.right_hand.middle += right_delta + 0.02;
    pose.right_hand.ring += right_delta + 0.04;
    pose.right_hand.pinky += right_delta + 0.06;

    if gesture_mode == "bsl" && (template == "fingerspell_a" || template == "fingerspell_b") {
        let letters: Vec<char> = word.chars().collect();
        let lead = letters.first().copied().unwrap_or('a');
        let trail = letters.last().copied().unwrap_or(lead);
        pose.left_hand = letter_hand_shape(lead, false);
        pose.right_hand = letter_hand_shape(trail, true);
        let twist = ((lead.to_ascii_lowercase() as i32 % 7) - 3) as f32 * 0.08;
        pose.left_wrist_yaw -= twist;
        pose.right_wrist_yaw += twist;
    }
    if template == "question" {
        pose.head_pitch -= 0.08 * intensity;
    } else if template == "affirm" {
        pose.head_pitch += 0.10 * intensity;
    } else if template == "negate" {
        pose.head_yaw -= 0.10 * intensity;
    }
    normalize_pose(pose)
}

fn push_frame(frames: &mut Vec<MotionKeyframe>, t: u32, pose: Pose) {
    let ts = if let Some(last) = frames.last() {
        if t <= last.t {
            last.t + 1
        } else {
            t
        }
    } else {
        t
    };
    frames.push(MotionKeyframe {
        t: ts,
        pose: normalize_pose(pose),
    });
}

/// Transform English text to BSL word order following BSL grammar rules.
///
/// BSL Grammar Differences from English:
/// 1. Topic-Comment structure (Topic first, then comment)
/// 2. Time indicators at sentence start
/// 3. Question words at end for questions
/// 4. No "to be" verbs (are, is, was, etc.)
/// 5. No articles (a, an, the)
/// 6. Different pronoun placement
/// 7. Adjectives after nouns
/// 8. No passive voice - convert to active
fn transform_to_bsl_order(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return String::new();
    }

    let mut result: Vec<String> = Vec::new();
    let mut i = 0;

    // Remove common articles and "to be" verbs
    let skip_words = ["a", "an", "the", "is", "are", "was", "were", "be", "been", "being"];

    // Check if question (ends with ?)
    let is_question = text.trim().ends_with('?');
    let mut question_word: Option<String> = None;

    // Time indicators - move to front
    let time_words = ["yesterday", "today", "tomorrow", "now", "later", "before", "after", "morning", "afternoon", "evening", "night"];
    let mut time_indicator: Option<String> = None;

    while i < words.len() {
        let word_lower = words[i].trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();

        // Capture time indicators
        if time_words.contains(&word_lower.as_str()) && time_indicator.is_none() {
            time_indicator = Some(words[i].to_string());
            i += 1;
            continue;
        }

        // Capture question words for end placement
        if is_question && ["what", "where", "when", "why", "who", "how", "which"].contains(&word_lower.as_str()) {
            question_word = Some(words[i].to_string());
            i += 1;
            continue;
        }

        // Skip articles and "to be" verbs
        if skip_words.contains(&word_lower.as_str()) {
            i += 1;
            continue;
        }

        // Skip "do/does/did" in questions
        if is_question && ["do", "does", "did"].contains(&word_lower.as_str()) {
            i += 1;
            continue;
        }

        result.push(words[i].to_string());
        i += 1;
    }

    // Build BSL sentence structure
    let mut bsl_words: Vec<String> = Vec::new();

    // 1. Time indicator first (if present)
    if let Some(time) = time_indicator {
        bsl_words.push(time);
    }

    // 2. Main content
    bsl_words.extend(result);

    // 3. Question word at end (if question)
    if let Some(qword) = question_word {
        bsl_words.push(qword);
    }

    bsl_words.join(" ")
}

fn plan_gesture_motion(text: &str, gesture_mode: &str, avatar_mode: &str) -> Option<GesturePlan> {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return None;
    }

    // Transform text for BSL if in BSL mode
    let (display_text, bsl_text) = if gesture_mode == "bsl" {
        let bsl_transformed = transform_to_bsl_order(&normalized);
        (bsl_transformed.clone(), Some(bsl_transformed))
    } else {
        (normalized.clone(), None)
    };

    let tokens = tokenize_transcript(&display_text, 28);
    if tokens.is_empty() {
        return None;
    }

    let amplitude = mode_amplitude(gesture_mode, avatar_mode);
    let rest = rest_pose(gesture_mode, amplitude);
    let mut frames: Vec<MotionKeyframe> = Vec::new();
    push_frame(&mut frames, 0, rest.clone());

    let mut timeline: Vec<GestureTimelineEntry> = Vec::new();
    let mut cursor: u32 = 80;
    let mut previous_pose = rest.clone();

    for (index, token) in tokens.iter().enumerate() {
        let intent = classify_intent(&token.word);
        let template = select_template(gesture_mode, intent, index, &token.word);
        let duration_ms = estimate_word_duration_ms(gesture_mode, &token.word);
        let pause_ms = pause_after_word_ms(gesture_mode, &token.punctuation);
        let intensity = amplitude * word_intensity(gesture_mode, intent, &token.word);
        let target = apply_template(
            rest.clone(),
            gesture_mode,
            &template,
            intensity,
            index,
            &token.word,
        );
        let attack = blend_pose(&previous_pose, &target, 0.58);
        let release = blend_pose(&target, &rest, 0.56);

        let attack_t = cursor + (duration_ms as f32 * 0.22) as u32;
        let peak_t = cursor + (duration_ms as f32 * 0.58) as u32;
        let release_t = cursor + (duration_ms as f32 * 0.90) as u32;

        push_frame(&mut frames, attack_t, attack);
        push_frame(&mut frames, peak_t, target.clone());
        push_frame(&mut frames, release_t, release.clone());

        timeline.push(GestureTimelineEntry {
            word: token.word.clone(),
            intent: intent.to_string(),
            template: template.clone(),
            start_ms: cursor,
            end_ms: cursor + duration_ms,
        });

        cursor = cursor.saturating_add(duration_ms).saturating_add(pause_ms);
        previous_pose = release;
    }

    let duration_ms = clamp_u32(cursor.saturating_add(180), 700, 18_000);
    push_frame(&mut frames, duration_ms, rest.clone());

    // Debug logging for first frame pose values
    if let Some(first_frame) = frames.first() {
        eprintln!("[DEBUG] First keyframe pose values for gesture_mode={}", gesture_mode);
        eprintln!("  left_shoulder_pitch: {:.3}", first_frame.pose.left_shoulder_pitch);
        eprintln!("  right_shoulder_pitch: {:.3}", first_frame.pose.right_shoulder_pitch);
        eprintln!("  left_shoulder_roll: {:.3}", first_frame.pose.left_shoulder_roll);
        eprintln!("  right_shoulder_roll: {:.3}", first_frame.pose.right_shoulder_roll);
        eprintln!("  left_elbow: {:.3}", first_frame.pose.left_elbow);
        eprintln!("  right_elbow: {:.3}", first_frame.pose.right_elbow);
    }

    Some(GesturePlan {
        motion: AvatarMotion {
            duration_ms,
            keyframes: frames,
        },
        summary: GestureSummary {
            style: if gesture_mode == "bsl" {
                "bsl_signing".to_string()
            } else {
                "semantic_gesticulation".to_string()
            },
            token_count: timeline.len(),
        },
        timeline,
        bsl_text,
    })
}

fn default_stt_context_prompt() -> Option<String> {
    if !env_flag_any(
        &["NMSTT_BUILTIN_CONTEXT_ENABLED", "REFINER_STT_BUILTIN_CONTEXT_ENABLED"],
        true,
    ) {
        return None;
    }
    let configured = env_string_any(&[
        "NMSTT_BUILTIN_CONTEXT_PROMPT",
        "REFINER_STT_BUILTIN_CONTEXT_PROMPT",
    ])
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    let fallback = "NeuralMimicry terminology: AARNN is pronounced Aaron; Tracey is spelled with an e not Tracy; founder Paul Isaac's; family names Kirsten, Scarlett, Harriet, Melissa, Michael, Christopher, Benjamin, Rebecca.";
    sanitize_prompt_with_limit(configured.as_deref().or(Some(fallback)), 640)
}

fn merge_stt_prompts(base: Option<&str>, extra: Option<&str>) -> Option<String> {
    let base_clean = sanitize_prompt_with_limit(base, 480);
    let extra_clean = sanitize_prompt_with_limit(extra, 320);
    match (base_clean, extra_clean) {
        (None, None) => None,
        (Some(base), None) => Some(base),
        (None, Some(extra)) => Some(extra),
        (Some(base), Some(extra)) => {
            let combined = format!("{base} {extra}");
            sanitize_prompt_with_limit(Some(combined.as_str()), 640)
        }
    }
}

fn normalize_word_key(word: &str) -> String {
    word.chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '\'')
        .collect::<String>()
        .to_ascii_lowercase()
}

fn split_token_edges(token: &str) -> Option<(&str, &str, &str)> {
    let mut start: Option<usize> = None;
    let mut end: usize = 0;
    for (idx, ch) in token.char_indices() {
        if ch.is_ascii_alphanumeric() || ch == '\'' {
            if start.is_none() {
                start = Some(idx);
            }
            end = idx + ch.len_utf8();
        }
    }
    let start = start?;
    let prefix = &token[..start];
    let core = &token[start..end];
    let suffix = &token[end..];
    Some((prefix, core, suffix))
}

fn canonicalize_core_word(
    core: &str,
    previous_plain: Option<&str>,
    has_aarnn_context: bool,
) -> Option<String> {
    let key = normalize_word_key(core);
    if key.is_empty() {
        return None;
    }
    let plain = key.replace('\'', "");
    let mapped = if plain == "aarnn" {
        Some("AARNN")
    } else if matches!(plain.as_str(), "aaron" | "arron" | "arun" | "aran") && has_aarnn_context {
        Some("AARNN")
    } else if plain == "neuralmimicry" {
        Some("NeuralMimicry")
    } else if plain == "refiner" {
        Some("Refiner")
    } else if plain == "continuum" {
        Some("Continuum")
    } else if matches!(plain.as_str(), "tracey" | "tracy") {
        Some("Tracey")
    } else if matches!(plain.as_str(), "paul") {
        Some("Paul")
    } else if matches!(plain.as_str(), "isaac" | "isaacs") {
        if previous_plain == Some("paul") || has_aarnn_context {
            Some("Isaac's")
        } else {
            None
        }
    } else if matches!(plain.as_str(), "kirsten" | "kirstin") {
        Some("Kirsten")
    } else if matches!(plain.as_str(), "scarlet" | "scarlett") {
        Some("Scarlett")
    } else if matches!(plain.as_str(), "harriet" | "harriett") {
        Some("Harriet")
    } else if matches!(plain.as_str(), "melissa" | "melisa") {
        Some("Melissa")
    } else if matches!(plain.as_str(), "michael" | "micheal") {
        Some("Michael")
    } else if matches!(plain.as_str(), "christopher" | "christoper" | "christofer") {
        Some("Christopher")
    } else if matches!(plain.as_str(), "benjamin" | "benjamen") {
        Some("Benjamin")
    } else if matches!(plain.as_str(), "rebecca" | "rebeca") {
        Some("Rebecca")
    } else {
        None
    };
    mapped.map(|value| value.to_string())
}

fn canonicalize_transcript_entities(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let raw_tokens: Vec<&str> = trimmed.split_whitespace().collect();
    if raw_tokens.is_empty() {
        return String::new();
    }
    let has_aarnn_context = raw_tokens.iter().any(|token| {
        if let Some((_, core, _)) = split_token_edges(token) {
            let plain = normalize_word_key(core).replace('\'', "");
            matches!(
                plain.as_str(),
                "aarnn"
                    | "aaron"
                    | "arron"
                    | "arun"
                    | "aran"
                    | "neuralmimicry"
                    | "neuromorphic"
                    | "refiner"
                    | "continuum"
                    | "tracey"
                    | "tracy"
                    | "security"
                    | "product"
                    | "platform"
                    | "swarm"
                    | "ai"
            )
        } else {
            false
        }
    });

    let mut previous_plain: Option<String> = None;
    let mut output: Vec<String> = Vec::with_capacity(raw_tokens.len());
    for token in raw_tokens {
        if let Some((prefix, core, suffix)) = split_token_edges(token) {
            let replacement =
                canonicalize_core_word(core, previous_plain.as_deref(), has_aarnn_context)
                    .unwrap_or_else(|| core.to_string());
            let plain = normalize_word_key(&replacement).replace('\'', "");
            if plain.is_empty() {
                previous_plain = None;
            } else {
                previous_plain = Some(plain);
            }
            output.push(format!("{prefix}{replacement}{suffix}"));
        } else {
            previous_plain = None;
            output.push(token.to_string());
        }
    }
    output.join(" ")
}

fn sanitize_lang(input: Option<&str>, fallback: &str) -> String {
    let candidate = input.unwrap_or(fallback).trim();
    if candidate.is_empty() {
        return fallback.to_string();
    }
    if candidate
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
    {
        candidate.to_string()
    } else {
        fallback.to_string()
    }
}

fn sanitize_prompt_with_limit(input: Option<&str>, max_len: usize) -> Option<String> {
    let raw = input?.trim();
    if raw.is_empty() {
        return None;
    }
    let max_len = max_len.max(64);
    let mut cleaned = String::new();
    for ch in raw.chars() {
        if ch == '\0' {
            continue;
        }
        if ch.is_control() && ch != '\n' && ch != '\t' && ch != ' ' {
            continue;
        }
        cleaned.push(ch);
        if cleaned.len() >= max_len {
            break;
        }
    }
    let collapsed = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        None
    } else {
        Some(collapsed)
    }
}

fn sanitize_prompt(input: Option<&str>) -> Option<String> {
    sanitize_prompt_with_limit(input, 640)
}

fn round_metric(value: f32, digits: i32) -> f32 {
    let factor = 10_f32.powi(digits.max(0));
    if !factor.is_finite() || factor <= 0.0 {
        return value;
    }
    (value * factor).round() / factor
}

fn whisper_timestamp_to_ms(timestamp: i64) -> u32 {
    let ticks = timestamp.max(0) as u64;
    let ms = ticks.saturating_mul(10);
    ms.min(u32::MAX as u64) as u32
}

fn ms_to_sample_index(ms: u32, sample_count: usize) -> usize {
    let idx = (ms as usize).saturating_mul(16);
    idx.min(sample_count)
}

fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum = samples
        .iter()
        .fold(0.0f32, |acc, sample| acc + sample * sample);
    (sum / samples.len() as f32).sqrt()
}

fn percentile_sorted(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let q = clamp(p, 0.0, 1.0);
    let idx = ((sorted.len() - 1) as f32 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn analyze_audio_metrics(samples: &[f32]) -> AudioMetrics {
    if samples.is_empty() {
        return AudioMetrics {
            speech_ratio: 0.0,
            speech_confidence: 0.0,
            noise_rms: 0.0,
            speech_rms: 0.0,
            snr_db: -12.0,
            background_noise_detected: false,
            noise_level: "low",
        };
    }

    let frame_size = 320;
    let mut frame_rms: Vec<f32> = samples.chunks(frame_size).map(compute_rms).collect();
    if frame_rms.is_empty() {
        frame_rms.push(compute_rms(samples));
    }

    let mut sorted = frame_rms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let noise_rms = percentile_sorted(&sorted, 0.20).max(0.0008);
    let upper_band = percentile_sorted(&sorted, 0.85).max(noise_rms + 0.0005);
    let speech_threshold = (noise_rms * 2.1).max(noise_rms + 0.003).max(0.006);

    let mut speech_sum = 0.0f32;
    let mut speech_count = 0usize;
    for rms in &frame_rms {
        if *rms >= speech_threshold {
            speech_sum += *rms;
            speech_count += 1;
        }
    }

    let speech_ratio = if frame_rms.is_empty() {
        0.0
    } else {
        speech_count as f32 / frame_rms.len() as f32
    };
    let speech_rms = if speech_count > 0 {
        speech_sum / speech_count as f32
    } else {
        upper_band
    };

    let snr_db = 20.0 * ((speech_rms + 1e-6) / (noise_rms + 1e-6)).log10();
    let snr_db = clamp(snr_db, -12.0, 42.0);
    let snr_score = clamp((snr_db + 12.0) / 54.0, 0.0, 1.0);
    let speech_confidence = clamp(0.18 + speech_ratio * 0.56 + snr_score * 0.36, 0.0, 1.0);

    let noise_level = if noise_rms > 0.040 || snr_db < 6.0 {
        "high"
    } else if noise_rms > 0.018 || snr_db < 14.0 {
        "moderate"
    } else {
        "low"
    };

    AudioMetrics {
        speech_ratio: round_metric(speech_ratio, 3),
        speech_confidence: round_metric(speech_confidence, 3),
        noise_rms: round_metric(noise_rms, 4),
        speech_rms: round_metric(speech_rms, 4),
        snr_db: round_metric(snr_db, 2),
        background_noise_detected: noise_level != "low",
        noise_level,
    }
}

fn segment_rms(samples: &[f32], start_ms: u32, end_ms: u32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let start_idx = ms_to_sample_index(start_ms, samples.len());
    let mut end_idx = ms_to_sample_index(end_ms, samples.len());
    if end_idx <= start_idx {
        end_idx = (start_idx + 320).min(samples.len());
    }
    if end_idx <= start_idx {
        return 0.0;
    }
    compute_rms(&samples[start_idx..end_idx])
}

fn estimate_segment_confidence(
    segment_rms_value: f32,
    noise_rms: f32,
    start_ms: u32,
    end_ms: u32,
    text: &str,
) -> f32 {
    let duration_ms = end_ms.saturating_sub(start_ms).max(80);
    let seg_snr_db = 20.0 * ((segment_rms_value + 1e-6) / (noise_rms.max(0.0008) + 1e-6)).log10();
    let snr_score = clamp((seg_snr_db + 8.0) / 28.0, 0.0, 1.0);
    let duration_score = clamp(duration_ms as f32 / 1800.0, 0.0, 1.0);
    let token_count = text.split_whitespace().count() as f32;
    let token_score = clamp(token_count / 8.0, 0.0, 1.0);
    round_metric(
        clamp(
            0.18 + snr_score * 0.54 + duration_score * 0.16 + token_score * 0.12,
            0.05,
            0.99,
        ),
        3,
    )
}

fn derive_speaker_segments(
    drafts: &[SegmentDraft],
    metrics: AudioMetrics,
    collaboration_mode: bool,
) -> (Vec<SpeakerSegment>, usize, u8) {
    if drafts.is_empty() {
        return (Vec::new(), 0, 1);
    }

    let explicit_turns = drafts.iter().filter(|seg| seg.speaker_turn_next).count();
    let mut inferred_turn_signals = 0usize;
    for idx in 1..drafts.len() {
        let prev = drafts[idx - 1].rms.max(0.0005);
        let curr = drafts[idx].rms.max(0.0005);
        let ratio = curr / prev;
        if ratio >= 1.9 || ratio <= 0.53 {
            inferred_turn_signals += 1;
        }
    }

    let mut speaker_count_estimate: u8 = 1;
    if collaboration_mode {
        if explicit_turns >= 1 {
            speaker_count_estimate = 2;
        } else if inferred_turn_signals >= 2 && drafts.len() >= 4 {
            speaker_count_estimate = 2;
        }
        if explicit_turns >= 4 && drafts.len() >= 6 {
            speaker_count_estimate = 3;
        }
    }

    if !collaboration_mode && speaker_count_estimate <= 1 {
        return (Vec::new(), explicit_turns, 1);
    }

    let mut speaker_index = 1usize;
    let mut inferred_turns_used = 0usize;
    let mut speaker_segments: Vec<SpeakerSegment> = Vec::with_capacity(drafts.len());

    for (idx, draft) in drafts.iter().enumerate() {
        let confidence = estimate_segment_confidence(
            draft.rms,
            metrics.noise_rms,
            draft.start_ms,
            draft.end_ms,
            &draft.text,
        );
        speaker_segments.push(SpeakerSegment {
            speaker: format!("S{}", speaker_index),
            text: draft.text.clone(),
            start_ms: draft.start_ms,
            end_ms: draft.end_ms,
            confidence,
        });

        if !collaboration_mode || speaker_count_estimate <= 1 {
            continue;
        }

        let mut should_turn = draft.speaker_turn_next;
        if !should_turn && idx + 1 < drafts.len() {
            let prev = draft.rms.max(0.0005);
            let next = drafts[idx + 1].rms.max(0.0005);
            let ratio = next / prev;
            if ratio >= 1.9 || ratio <= 0.53 {
                should_turn = true;
                inferred_turns_used += 1;
            }
        }

        if should_turn {
            if speaker_count_estimate <= 2 {
                speaker_index = if speaker_index == 1 { 2 } else { 1 };
            } else {
                speaker_index = (speaker_index % speaker_count_estimate as usize) + 1;
            }
        }
    }

    (
        speaker_segments,
        explicit_turns + inferred_turns_used,
        speaker_count_estimate,
    )
}

fn run_inference(
    ctx: Arc<Mutex<WhisperContext>>,
    audio: Vec<u8>,
    lang: &str,
    threads: usize,
    translate: bool,
    prompt: Option<&str>,
    collaboration_mode: bool,
) -> Result<InferenceResult, SttError> {
    let (samples, sample_rate) = decode_audio(&audio)?;
    let samples = resample_to_16k(samples, sample_rate)?;
    let audio_metrics = analyze_audio_metrics(&samples);

    let ctx = ctx.lock().map_err(|_| SttError::SttFailed)?;
    let mut whisper_state = ctx.create_state().map_err(|_| SttError::SttFailed)?;
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(threads as i32);
    params.set_translate(translate);
    params.set_language(Some(lang));
    params.set_tdrz_enable(collaboration_mode);
    if let Some(initial_prompt) = prompt {
        if !initial_prompt.trim().is_empty() {
            params.set_initial_prompt(initial_prompt);
        }
    }

    whisper_state
        .full(params, &samples)
        .map_err(|_| SttError::SttFailed)?;

    let num_segments = whisper_state
        .full_n_segments()
        .map_err(|_| SttError::SttFailed)?;
    let mut raw_segments: Vec<String> = Vec::new();
    let mut segment_drafts: Vec<SegmentDraft> = Vec::new();
    for i in 0..num_segments {
        let segment_text = whisper_state
            .full_get_segment_text(i)
            .map_err(|_| SttError::SttFailed)?;
        let clean = segment_text.trim();
        if clean.is_empty() {
            continue;
        }
        raw_segments.push(clean.to_string());

        let t0 = whisper_state
            .full_get_segment_t0(i)
            .map_err(|_| SttError::SttFailed)?;
        let t1 = whisper_state
            .full_get_segment_t1(i)
            .map_err(|_| SttError::SttFailed)?;
        let mut start_ms = whisper_timestamp_to_ms(t0);
        let mut end_ms = whisper_timestamp_to_ms(t1);
        if end_ms <= start_ms {
            end_ms = start_ms.saturating_add(120);
        }
        start_ms = start_ms.min(end_ms.saturating_sub(1));
        let rms = segment_rms(&samples, start_ms, end_ms);
        let speaker_turn_next = if collaboration_mode {
            whisper_state.full_get_segment_speaker_turn_next(i)
        } else {
            false
        };

        segment_drafts.push(SegmentDraft {
            text: clean.to_string(),
            start_ms,
            end_ms,
            speaker_turn_next,
            rms,
        });
    }

    let text = raw_segments.join(" ");
    let (speaker_segments, speaker_turn_count, speaker_count_estimate) =
        derive_speaker_segments(&segment_drafts, audio_metrics, collaboration_mode);
    let collaboration_likely = collaboration_mode
        && speaker_count_estimate > 1
        && audio_metrics.speech_ratio >= 0.12
        && audio_metrics.speech_confidence >= 0.28;

    Ok(InferenceResult {
        text: text.trim().to_string(),
        audio_analysis: AudioAnalysis {
            speech_ratio: audio_metrics.speech_ratio,
            speech_confidence: audio_metrics.speech_confidence,
            noise_level: audio_metrics.noise_level.to_string(),
            noise_rms: audio_metrics.noise_rms,
            speech_rms: audio_metrics.speech_rms,
            snr_db: audio_metrics.snr_db,
            speaker_count_estimate,
            speaker_turn_count,
            background_noise_detected: audio_metrics.background_noise_detected,
            collaboration_likely,
        },
        speaker_segments,
    })
}

fn decode_audio(data: &[u8]) -> Result<(Vec<f32>, u32), SttError> {
    let cursor = Cursor::new(data.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    let hint = Hint::new();
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|_| SttError::InvalidAudio)?;
    let mut format = probed.format;
    let track = format.default_track().ok_or(SttError::InvalidAudio)?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|_| SttError::InvalidAudio)?;

    let mut sample_rate = track.codec_params.sample_rate;
    let mut channels = track.codec_params.channels.map(|c| c.count());
    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(_)) => break,
            Err(_) => return Err(SttError::InvalidAudio),
        };
        let decoded = decoder
            .decode(&packet)
            .map_err(|_| SttError::InvalidAudio)?;
        let spec = decoded.spec();
        if sample_rate.is_none() {
            sample_rate = Some(spec.rate);
        }
        if channels.is_none() {
            channels = Some(spec.channels.count());
        }
        let channel_count = channels.unwrap_or(1);
        match decoded {
            AudioBufferRef::F32(buf) => write_mono(&buf, channel_count, &mut samples),
            AudioBufferRef::F64(buf) => write_mono(&buf, channel_count, &mut samples),
            AudioBufferRef::S8(buf) => write_mono(&buf, channel_count, &mut samples),
            AudioBufferRef::S16(buf) => write_mono(&buf, channel_count, &mut samples),
            AudioBufferRef::S24(buf) => write_mono(&buf, channel_count, &mut samples),
            AudioBufferRef::S32(buf) => write_mono(&buf, channel_count, &mut samples),
            AudioBufferRef::U8(buf) => write_mono(&buf, channel_count, &mut samples),
            AudioBufferRef::U16(buf) => write_mono(&buf, channel_count, &mut samples),
            AudioBufferRef::U24(buf) => write_mono(&buf, channel_count, &mut samples),
            AudioBufferRef::U32(buf) => write_mono(&buf, channel_count, &mut samples),
        }
    }

    let rate = sample_rate.ok_or(SttError::InvalidAudio)?;
    if samples.is_empty() {
        return Err(SttError::InvalidAudio);
    }
    Ok((samples, rate))
}

fn write_mono<T: Sample + IntoSample<f32>>(
    buf: &symphonia::core::audio::AudioBuffer<T>,
    channels: usize,
    out: &mut Vec<f32>,
) {
    let frames = buf.frames();
    let chan_count = if channels == 0 { 1 } else { channels };
    for frame in 0..frames {
        let mut sum = 0.0f32;
        for ch in 0..chan_count {
            let sample: f32 = buf.chan(ch)[frame].into_sample();
            sum += sample;
        }
        out.push(sum / chan_count as f32);
    }
}

fn resample_to_16k(samples: Vec<f32>, sample_rate: u32) -> Result<Vec<f32>, SttError> {
    if sample_rate == 16_000 {
        return Ok(samples);
    }
    if samples.is_empty() {
        return Err(SttError::UnsupportedFormat);
    }
    let ratio = 16_000_f64 / sample_rate as f64;
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 160,
        window: WindowFunction::BlackmanHarris2,
    };
    let chunk_size = 1024;
    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_size, 1)
        .map_err(|_| SttError::SttFailed)?;
    let mut output: Vec<f32> = Vec::new();
    let mut idx = 0usize;
    while idx < samples.len() {
        let end = std::cmp::min(idx + chunk_size, samples.len());
        let mut chunk = samples[idx..end].to_vec();
        if chunk.len() < chunk_size {
            chunk.resize(chunk_size, 0.0);
        }
        let out_chunks = resampler
            .process(&[chunk], None)
            .map_err(|_| SttError::SttFailed)?;
        if !out_chunks.is_empty() {
            output.extend_from_slice(&out_chunks[0]);
        }
        idx = end;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalizes_domain_terms_and_spelling() {
        let text = "Tell me about tracy and aaron in NeuralMimicry security.";
        let out = canonicalize_transcript_entities(text);
        assert!(out.contains("Tracey"));
        assert!(out.contains("AARNN"));
        assert!(out.contains("NeuralMimicry"));
    }

    #[test]
    fn canonicalizes_founder_and_family_names() {
        let text = "paul isaacs kirstin scarlet harriett melisa micheal christofer benjamen rebeca";
        let out = canonicalize_transcript_entities(text);
        assert_eq!(
            out,
            "Paul Isaac's Kirsten Scarlett Harriet Melissa Michael Christopher Benjamin Rebecca"
        );
    }

    #[test]
    fn merges_builtin_and_client_prompts_with_limit() {
        let out = merge_stt_prompts(
            Some("AARNN is pronounced Aaron"),
            Some("Tracey is spelled with an e"),
        )
        .unwrap_or_default();
        assert!(out.contains("AARNN"));
        assert!(out.contains("Tracey"));
        assert!(out.len() <= 640);
    }

    #[test]
    fn detects_noisy_audio_characteristics() {
        let mut samples = Vec::new();
        for i in 0..16_000 {
            let base = if i % 41 < 23 { 0.028 } else { -0.028 };
            let pulse = if i % 1600 < 180 { 0.040 } else { 0.0 };
            samples.push(base + pulse);
        }
        let metrics = analyze_audio_metrics(&samples);
        assert!(matches!(metrics.noise_level, "moderate" | "high"));
        assert!(metrics.speech_confidence > 0.0);
    }

    #[test]
    fn derives_multi_speaker_segments_from_turns() {
        let drafts = vec![
            SegmentDraft {
                text: "hello".to_string(),
                start_ms: 0,
                end_ms: 640,
                speaker_turn_next: true,
                rms: 0.032,
            },
            SegmentDraft {
                text: "hi there".to_string(),
                start_ms: 640,
                end_ms: 1260,
                speaker_turn_next: false,
                rms: 0.021,
            },
        ];
        let metrics = AudioMetrics {
            speech_ratio: 0.42,
            speech_confidence: 0.78,
            noise_rms: 0.009,
            speech_rms: 0.031,
            snr_db: 13.0,
            background_noise_detected: false,
            noise_level: "low",
        };
        let (segments, turns, speaker_count) = derive_speaker_segments(&drafts, metrics, true);
        assert_eq!(speaker_count, 2);
        assert!(turns >= 1);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker, "S1");
        assert_eq!(segments[1].speaker, "S2");
    }

    #[test]
    fn normalizes_human_readable_bsl_labels() {
        assert_eq!(
            sanitize_gesture_mode(Some("BSL (British Sign Language)"), "gesticulation", true),
            "bsl"
        );
        assert_eq!(
            sanitize_gesture_mode(Some("British Sign Language"), "gesticulation", true),
            "bsl"
        );
    }

    #[test]
    fn normalizes_avatar_mode_aliases() {
        assert_eq!(normalize_avatar_mode("Office Mode"), Some("office"));
        assert_eq!(normalize_avatar_mode("Chat Mode"), Some("chat"));
    }

    #[test]
    fn transforms_english_to_bsl_removes_articles_and_be_verbs() {
        let text = "The cat is on the mat.";
        let bsl = transform_to_bsl_order(text);
        // Should remove "the", "is"
        assert!(!bsl.contains("the"));
        assert!(!bsl.contains("The"));
        assert!(!bsl.contains("is"));
        assert!(bsl.contains("cat"));
        assert!(bsl.contains("on"));
        assert!(bsl.contains("mat"));
    }

    #[test]
    fn transforms_question_word_to_end_in_bsl() {
        let text = "What is your name?";
        let bsl = transform_to_bsl_order(text);
        // "What" should be at end, "is" removed
        assert!(bsl.ends_with("What") || bsl.ends_with("What?"));
        assert!(!bsl.starts_with("What"));
        assert!(!bsl.contains(" is "));
    }

    #[test]
    fn moves_time_indicators_to_front_in_bsl() {
        let text = "I went to the shop yesterday.";
        let bsl = transform_to_bsl_order(text);
        // "yesterday" should be at start
        assert!(bsl.starts_with("yesterday") || bsl.starts_with("Yesterday"));
        // Articles removed
        assert!(!bsl.contains("the"));
    }

    #[test]
    fn removes_do_does_did_in_questions() {
        let text = "Do you like coffee?";
        let bsl = transform_to_bsl_order(text);
        assert!(!bsl.contains("Do"));
        assert!(!bsl.contains("do"));
        assert!(bsl.contains("you"));
        assert!(bsl.contains("like"));
        assert!(bsl.contains("coffee"));
    }

    #[test]
    fn bsl_transformation_preserves_essential_words() {
        let text = "Hello, my name is Paul.";
        let bsl = transform_to_bsl_order(text);
        // Should keep Hello, my, name, Paul
        assert!(bsl.contains("Hello"));
        assert!(bsl.contains("my"));
        assert!(bsl.contains("name"));
        assert!(bsl.contains("Paul"));
        // Should remove "is"
        assert!(!bsl.contains("is"));
    }
}
