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

/// Parakeet speech-to-text configuration (ONNX or GGUF, alternative to Whisper)
/// Requires `parakeet` for ONNX or `parakeet-gguf` for GGUF.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ParakeetConfig {
    /// Model name, ONNX model directory, or path to a transcribe.cpp GGUF file.
    /// For TDT: encoder-model.onnx, decoder_joint-model.onnx, vocab.txt
    /// For CTC: model.onnx, tokenizer.json
    /// A name ending in `.gguf` runs through transcribe.cpp instead.
    pub model: String,

    /// transcribe.cpp compute backend for GGUF models (auto, cpu, vulkan, etc.).
    #[serde(default = "default_gguf_backend")]
    pub gguf_backend: String,

    /// Model architecture type: "tdt" (default, recommended) or "ctc"
    /// Auto-detected from model directory structure if not specified
    #[serde(default)]
    pub model_type: Option<ParakeetModelType>,

    /// Number of CPU threads for ONNX Runtime intra-op parallelism
    /// (default: 4, parakeet-rs's). Long dictations on the CPU are fastest
    /// at the physical core count. For GGUF models, transcribe.cpp's CPU
    /// threads (default: the library's).
    #[serde(default)]
    pub threads: Option<usize>,

    /// Load model on-demand when recording starts (true) or keep loaded (false)
    #[serde(default = "default_on_demand_loading")]
    pub on_demand_loading: bool,

    /// Unload the loaded model from memory/VRAM after this many seconds of
    /// inactivity (0 = keep loaded, default). When > 0, the model is not
    /// preloaded at daemon startup; it loads on first recording (while audio
    /// capture is already buffering, so no words are lost), stays warm across
    /// back-to-back recordings, and unloads once idle for this duration.
    #[serde(default)]
    pub idle_unload_secs: u64,

    /// Longest stretch of audio handed to the model in one pass, in seconds
    /// (default: 240). Longer recordings are split into overlapping windows
    /// and the transcripts are stitched back together.
    ///
    /// The TDT ONNX exports carry a fixed-size attention bias, and exceeding
    /// it does not degrade gracefully — ONNX Runtime aborts the run with
    /// "Attempting to broadcast an axis by a dimension other than 1" and the
    /// transcription is lost (#288). The reported failure was an 11-minute
    /// file whose 8691 encoder frames met a bias sized for 3691, which is
    /// about 295 seconds at the 80ms frame rate these models use. 240 leaves
    /// margin under that without cutting ordinary dictation into pieces.
    ///
    /// Set to 0 to disable windowing and send the whole buffer, which is the
    /// pre-#288 behavior.
    #[serde(default = "default_parakeet_max_window_secs")]
    pub max_window_secs: f32,

    /// Overlap between consecutive windows, in seconds (default: 5).
    /// The stitcher uses this shared audio to find where two windows agree,
    /// so it must be long enough to contain a few words.
    #[serde(default = "default_parakeet_window_overlap_secs")]
    pub window_overlap_secs: f32,

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

    /// Longest pause, in seconds, the live stream decodes; the rest of a
    /// longer pause is left out of it. After a pause longer than the
    /// stream's left context the model has no context left and drops the
    /// first word or two after it. Affects the live preview and live typing,
    /// not the final pass. 0 keeps every pause. Default: 2.5.
    #[serde(default = "default_streaming_max_pause_secs")]
    pub streaming_max_pause_secs: f32,

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

    /// Skip the final pass when the stream already got the take right, as
    /// far as can be told: its text ends a sentence (`.` `?` `!` `…`) and
    /// everything recorded after the audio behind that text is quiet. The
    /// streamed text is delivered at once instead of after a full decode.
    /// Worth it where the final pass is slow (CPU); on a GPU the pass costs
    /// little and is more accurate. Default: false.
    #[serde(default)]
    pub streaming_final_pass_skip_when_clean: bool,

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

    /// Seconds of silence appended to a final-pass piece (streaming models)
    /// that ends mid-speech, as when the key is released right on the last
    /// word: without audio after it the decoder drops or clips that word
    /// ("said Shaggy" -> "said Sh"). Pieces that already end in a pause are
    /// decoded unpadded. 0 disables. Default: 0.5. Measured on 73
    /// LibriSpeech clips with their trailing silence cut (CPU, fp16 unified
    /// model): WER 3.39% -> 2.26%; on the uncut clips nothing changes
    /// (2.17% both).
    #[serde(default = "default_pad_secs")]
    pub pad_secs: f32,

    /// Settings that apply only when the model runs on the CPU (no GPU
    /// execution provider). The final transcript comes from the same model
    /// either way; these trade nothing in it for speed. See
    /// [`ParakeetCpuConfig`].
    #[serde(default)]
    pub cpu: ParakeetCpuConfig,
}

fn default_gguf_backend() -> String {
    "auto".to_string()
}

/// Under the ~295s ceiling inferred from the #288 report, with margin.
fn default_parakeet_max_window_secs() -> f32 {
    240.0
}

/// Matches the overlap the Cohere checkpoint asks for; long enough to hold
/// several words for the stitcher to match on.
fn default_parakeet_window_overlap_secs() -> f32 {
    5.0
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

fn default_streaming_max_pause_secs() -> f32 {
    2.5
}

fn default_pad_secs() -> f32 {
    0.5
}

impl Default for ParakeetConfig {
    fn default() -> Self {
        Self {
            model: crate::model_catalog::default_model("parakeet").to_string(),
            gguf_backend: default_gguf_backend(),
            model_type: None, // Auto-detect
            threads: None,
            on_demand_loading: false,
            idle_unload_secs: 0,
            max_window_secs: default_parakeet_max_window_secs(),
            window_overlap_secs: default_parakeet_window_overlap_secs(),
            streaming: false,
            streaming_chunk_secs: default_streaming_chunk_secs(),
            streaming_left_context_secs: default_streaming_left_context_secs(),
            streaming_right_context_secs: default_streaming_right_context_secs(),
            streaming_final_pass: false,
            streaming_final_pass_max_secs: default_streaming_final_pass_max_secs(),
            streaming_final_pass_skip_when_clean: false,
            streaming_max_pause_secs: default_streaming_max_pause_secs(),
            final_pass_segment_secs: default_final_pass_segment_secs(),
            pad_secs: default_pad_secs(),
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
        assert_eq!(
            config.model,
            crate::model_catalog::default_model("parakeet")
        );
        assert!(config.model_type.is_none());
        assert!(!config.on_demand_loading);
    }

    #[test]
    fn test_parakeet_model_type_enum_default() {
        // ParakeetModelType defaults to Tdt
        assert_eq!(ParakeetModelType::default(), ParakeetModelType::Tdt);
    }
}
