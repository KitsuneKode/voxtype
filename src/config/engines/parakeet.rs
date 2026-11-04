//! Parakeet engine configuration.

use serde::{Deserialize, Serialize};

use super::super::default_on_demand_loading;

/// Parakeet model architecture type
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ParakeetModelType {
    /// CTC (Connectionist Temporal Classification) - faster, character-level output
    Ctc,
    /// TDT (Token-Duration-Transducer) - recommended, proper punctuation and word boundaries
    #[default]
    Tdt,
}

/// Parakeet speech-to-text configuration (ONNX-based, alternative to Whisper)
/// Requires: cargo build --features parakeet
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ParakeetConfig {
    /// Path to model directory containing ONNX model files
    /// For TDT: encoder-model.onnx, decoder_joint-model.onnx, vocab.txt
    /// For CTC: model.onnx, tokenizer.json
    pub model: String,

    /// Model architecture type: "tdt" (default, recommended) or "ctc"
    /// Auto-detected from model directory structure if not specified
    #[serde(default)]
    pub model_type: Option<ParakeetModelType>,

    /// Load model on-demand when recording starts (true) or keep loaded (false)
    #[serde(default = "default_on_demand_loading")]
    pub on_demand_loading: bool,

    /// Use the cache-aware streaming pipeline (parakeet-rs `ParakeetUnified`)
    /// instead of the batch CTC/TDT models. When true, voxtype emits live
    /// partial transcripts during recording and types the final transcript
    /// on hotkey release. Requires a streaming-capable model directory
    /// (TDT v3 family with `tokenizer.model`).
    /// Default: false (batch pipeline, identical to pre-streaming behavior).
    #[serde(default)]
    pub streaming: bool,

    /// Streaming chunk length in seconds. Smaller = lower latency, more
    /// inference overhead. Maps to `UnifiedStreamingConfig::chunk_secs`.
    #[serde(default = "default_streaming_chunk_secs")]
    pub streaming_chunk_secs: f32,

    /// Streaming left context in seconds. Maps to
    /// `UnifiedStreamingConfig::left_context_secs`.
    #[serde(default = "default_streaming_left_context_secs")]
    pub streaming_left_context_secs: f32,

    /// Streaming right context in seconds. Maps to
    /// `UnifiedStreamingConfig::right_context_secs`.
    #[serde(default = "default_streaming_right_context_secs")]
    pub streaming_right_context_secs: f32,

    /// After a streaming session that types nothing until stop
    /// (`[output] streaming_commit = "on_stop"`), decode the whole recording
    /// again in one offline pass with the same model and deliver that
    /// instead of the streamed text. The offline pass sees the full
    /// recording on both sides of every word, so it is more accurate than
    /// the chunked stream, and it costs no extra VRAM. The live preview is
    /// unchanged. Falls back to the streamed text if the pass fails or the
    /// recording is longer than `streaming_final_pass_max_secs`.
    /// Default: false.
    #[serde(default)]
    pub streaming_final_pass: bool,

    /// Longest recording, in seconds, that gets the offline final pass.
    /// The encoder attends over the whole recording at once, so memory
    /// grows with length; longer sessions keep the streamed text.
    #[serde(default = "default_streaming_final_pass_max_secs")]
    pub streaming_final_pass_max_secs: f32,

    /// Longest piece, in seconds, the final pass decodes at once. Longer
    /// recordings are split at their pauses and each piece is decoded with
    /// its own full context. Past about a minute of audio the offline pass
    /// starts dropping punctuation and its cost grows faster than the audio,
    /// so splitting is both more accurate and faster. Pieces that finish
    /// while the user is still speaking are decoded in the background, so
    /// the wait at stop is one piece at most however long the recording.
    /// 0 decodes the whole recording at once. Default: 45.
    #[serde(default = "default_final_pass_segment_secs")]
    pub final_pass_segment_secs: f32,

    /// Settings that apply only when the model runs on the CPU (no GPU
    /// execution provider). The final transcript comes from the same model
    /// either way; these trade nothing in it for speed. See
    /// [`ParakeetCpuConfig`].
    #[serde(default)]
    pub cpu: ParakeetCpuConfig,
}

/// `[parakeet.cpu]`: overrides used only on the CPU execution provider.
///
/// A GPU makes the live preview stream nearly free; on the CPU it is the
/// most expensive part of a session (every chunk re-encodes its whole left
/// context), and when it falls behind real time the stop waits for it.
/// These keys give the preview a lighter streaming profile without touching
/// the final pass, which decodes the whole recording with full context.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct ParakeetCpuConfig {
    /// ONNX Runtime intra-op threads for the final pass. Unset uses one per
    /// performance core (efficiency cores and hyperthreads make it slower).
    #[serde(default)]
    pub threads: Option<usize>,


    /// Preview chunk length on CPU (overrides `streaming_chunk_secs`).
    #[serde(default)]
    pub streaming_chunk_secs: Option<f32>,

    /// Preview left context on CPU (overrides `streaming_left_context_secs`).
    #[serde(default)]
    pub streaming_left_context_secs: Option<f32>,

    /// Preview lookahead on CPU (overrides `streaming_right_context_secs`).
    #[serde(default)]
    pub streaming_right_context_secs: Option<f32>,
}

// parakeet-rs validates that each context length maps to a mel-frame count
// divisible by 8 (80ms steps); the previous 0.5/1.5/0.5 defaults failed that
// check, so streaming = true without explicit values could never start.
fn default_streaming_chunk_secs() -> f32 {
    0.32
}

fn default_streaming_left_context_secs() -> f32 {
    5.6
}

fn default_streaming_right_context_secs() -> f32 {
    0.32
}

fn default_streaming_final_pass_max_secs() -> f32 {
    180.0
}

fn default_final_pass_segment_secs() -> f32 {
    45.0
}

impl Default for ParakeetConfig {
    fn default() -> Self {
        Self {
            model: "parakeet-tdt-0.6b-v3".to_string(),
            model_type: None, // Auto-detect
            on_demand_loading: false,
            streaming: false,
            streaming_chunk_secs: default_streaming_chunk_secs(),
            streaming_left_context_secs: default_streaming_left_context_secs(),
            streaming_right_context_secs: default_streaming_right_context_secs(),
            streaming_final_pass: false,
            streaming_final_pass_max_secs: default_streaming_final_pass_max_secs(),
            final_pass_segment_secs: default_final_pass_segment_secs(),
            cpu: ParakeetCpuConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn test_parse_parakeet_model_type_tdt() {
        let toml_str = r#"
            engine = "parakeet"

            [hotkey]
            key = "SCROLLLOCK"

            [audio]
            device = "default"
            sample_rate = 16000
            max_duration_secs = 60

            [whisper]
            model = "base.en"
            language = "en"

            [output]
            mode = "type"

            [parakeet]
            model = "parakeet-tdt-0.6b-v3"
            model_type = "tdt"
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();
        let parakeet = config.parakeet.unwrap();
        assert_eq!(parakeet.model, "parakeet-tdt-0.6b-v3");
        assert_eq!(parakeet.model_type, Some(ParakeetModelType::Tdt));
    }

    #[test]
    fn test_parse_parakeet_model_type_ctc() {
        let toml_str = r#"
            engine = "parakeet"

            [hotkey]
            key = "SCROLLLOCK"

            [audio]
            device = "default"
            sample_rate = 16000
            max_duration_secs = 60

            [whisper]
            model = "base.en"
            language = "en"

            [output]
            mode = "type"

            [parakeet]
            model = "parakeet-ctc-0.6b"
            model_type = "ctc"
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();
        let parakeet = config.parakeet.unwrap();
        assert_eq!(parakeet.model, "parakeet-ctc-0.6b");
        assert_eq!(parakeet.model_type, Some(ParakeetModelType::Ctc));
    }

    #[test]
    fn test_parakeet_model_type_defaults_to_none_for_auto_detection() {
        let toml_str = r#"
            engine = "parakeet"

            [hotkey]
            key = "SCROLLLOCK"

            [audio]
            device = "default"
            sample_rate = 16000
            max_duration_secs = 60

            [whisper]
            model = "base.en"
            language = "en"

            [output]
            mode = "type"

            [parakeet]
            model = "parakeet-tdt-0.6b-v3"
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();
        let parakeet = config.parakeet.unwrap();
        // model_type should be None (will be auto-detected at runtime)
        assert!(parakeet.model_type.is_none());
    }

    #[test]
    fn test_parakeet_config_default() {
        let config = ParakeetConfig::default();
        assert_eq!(config.model, "parakeet-tdt-0.6b-v3");
        assert!(config.model_type.is_none());
        assert!(!config.on_demand_loading);
    }

    #[test]
    fn test_parakeet_model_type_enum_default() {
        // ParakeetModelType defaults to Tdt
        assert_eq!(ParakeetModelType::default(), ParakeetModelType::Tdt);
    }
}
