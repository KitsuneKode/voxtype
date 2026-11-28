//! Parakeet streaming transcriber via `parakeet-rs::ParakeetUnified`
//!
//! Wraps the cache-aware streaming pipeline so voxtype can emit live partial
//! transcripts during recording and a final transcript on hotkey release.
//!
//! Selected when `[parakeet] streaming = true`. Otherwise the batch
//! `ParakeetTranscriber` is used.
//!
//! # Architecture
//!
//! - The transcriber holds a [`ParakeetUnifiedHandle`] (shared model weights)
//!   and a `Mutex<ParakeetUnified>` for batch-mode fallback. The same handle
//!   spawns fresh streaming instances per [`start_stream`] call.
//! - For batch [`Transcriber::transcribe`] the `Mutex<ParakeetUnified>` is
//!   reset between calls so no streaming cache leaks between recordings.
//! - For [`StreamingTranscriber::start_stream`] a fresh `ParakeetUnified` is
//!   constructed from the handle and moved into the spawned drive task.
//!
//! # Partial / Final policy
//!
//! `parakeet-rs` does not surface end-of-utterance signals from the unified
//! pipeline (that lives in the multitalker / EOU model). Until we add VAD-
//! based segmentation, this implementation:
//!
//! - emits one [`StreamingEvent::Partial`] per audio chunk that decoded new
//!   tokens, carrying *only those tokens* (a delta, not the cumulative
//!   transcript — `ParakeetUnified::transcribe_chunk` returns deltas);
//! - emits exactly one [`StreamingEvent::Final`] with the flushed tail on
//!   stream close (samples sender dropped or cancel signal), also a delta;
//! - emits exactly one [`StreamingEvent::Ended`] last.
//!
//! This gives us live OSD partials throughout the recording, with the final
//! commit happening at hotkey release. Mid-recording incremental typing
//! (commit-on-pause) is a follow-up once VAD-segmentation lands.

use super::parakeet::{build_execution_config, performance_cores, resolve_model_path};
use super::streaming::{StreamHandle, StreamProgress, StreamingEvent, StreamingTranscriber};
use super::vocab_rescore::{self, SpanScorer, TermSource};
use super::{TimedSegment, Transcriber};
use crate::config::ParakeetConfig;
use crate::error::TranscribeError;
use parakeet_rs::{
    ExecutionConfig, OfflineDecode, ParakeetUnified, ParakeetUnifiedHandle, UnifiedStreamingConfig,
};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};

/// Streaming-capable Parakeet transcriber backed by `ParakeetUnified`.
pub struct ParakeetStreamingTranscriber {
    /// Shared model weights. Cloned cheaply for each streaming session; the
    /// live preview and the final pass run the same model.
    handle: ParakeetUnifiedHandle,
    /// Longest piece the batch/final pass decodes at once
    /// (`final_pass_segment_secs`); 0 = whole recording.
    segment_secs: f32,
    /// `[parakeet] pad_secs`, in samples.
    pad: usize,
    /// Streaming-config snapshot used both for batch (so the handle's
    /// `from_shared_with_streaming_config` succeeds) and for spawning each
    /// streaming task.
    streaming_config: UnifiedStreamingConfig,
    /// Reusable batch instance for `Transcriber::transcribe`. Reset between
    /// calls so no inter-call state leaks.
    batch: Mutex<ParakeetUnified>,
    /// Progress of the current (or last) stream. Each stream gets its own,
    /// so a cancelled one still finishing a chunk cannot write into the next.
    progress: Mutex<Arc<Mutex<StreamProgress>>>,
    /// `[parakeet] streaming_max_pause_secs`, in samples (0 = off).
    max_pause: usize,
    /// `[parakeet] vocabulary_file` / `corrections_file` (None = off).
    vocabulary: Option<TermSource>,
    vocabulary_boost: f32,
    /// What the model runs on (see [`super::parakeet::execution_label`]).
    backend: &'static str,
}

impl ParakeetStreamingTranscriber {
    pub fn new(config: &ParakeetConfig) -> Result<Self, TranscribeError> {
        // Streaming-capability gate. The cache-aware streaming pipeline
        // needs both `tokenizer.model` at load time and a decoder graph
        // sized for the streaming inference loop. Models in the registry
        // that don't meet both (e.g. istupakov's TDT-v3) hit either a
        // tokenizer-not-found at load or an ONNX Gather shape error at
        // first chunk — neither is a useful diagnostic. Fail fast here
        // with a message that names the model and the known-good
        // alternative. Unknown model names (custom user directories)
        // fall through with a warning since we can't validate them.
        if crate::setup::model::is_known_parakeet_model(&config.model)
            && !crate::setup::model::is_streaming_compatible_parakeet(&config.model)
        {
            return Err(TranscribeError::InitFailed(format!(
                "Parakeet streaming is enabled but model `{}` does not support \
                 cache-aware streaming.\n\n\
                 Fix one of:\n\n  \
                 - Disable streaming in config.toml under [parakeet]:\n\n      \
                 streaming = false\n\n  \
                 - Or switch to the streaming-compatible model:\n\n      \
                 voxtype setup --download --model {streaming_model}\n      \
                 voxtype setup model --set {streaming_model}",
                config.model,
                streaming_model = crate::setup::model::DEFAULT_PARAKEET_STREAMING_MODEL,
            )));
        }
        if !crate::setup::model::is_known_parakeet_model(&config.model) {
            tracing::warn!(
                model = %config.model,
                "Parakeet streaming requested with a model not in the registry; \
                 cannot validate streaming compatibility ahead of time. If load \
                 fails below, switch [parakeet] model to {} or set streaming = false.",
                crate::setup::model::DEFAULT_PARAKEET_STREAMING_MODEL
            );
        }

        let model_path = resolve_model_path(&config.model)?;

        tracing::info!(
            "Loading Parakeet streaming (ParakeetUnified) model from {:?}",
            model_path
        );
        let start = std::time::Instant::now();

        let gpu = build_execution_config(None);
        let on_cpu = gpu.is_none();
        let final_exec = gpu.unwrap_or_else(|| {
            let threads = config
                .cpu
                .threads
                .or(config.threads)
                .unwrap_or_else(performance_cores);
            tracing::info!("Parakeet on CPU with {} threads", threads);
            super::parakeet::no_spin(ExecutionConfig::new().with_intra_threads(threads))
        });
        let backend = super::parakeet::execution_label(&final_exec);
        let handle = ParakeetUnifiedHandle::load(&model_path, Some(final_exec)).map_err(|e| {
            TranscribeError::InitFailed(format!(
                "Parakeet streaming (ParakeetUnified) init failed: {}\n\n\
                Streaming requires a TDT v3 model directory containing tokenizer.model.\n\
                If you're using a TDT v2 directory, switch to TDT v3 or set\n\
                [parakeet] streaming = false to use the batch pipeline.",
                e
            ))
        })?;

        // On the CPU the preview uses a lighter streaming profile
        // (`[parakeet.cpu] streaming_*`): less left context, so each chunk
        // re-encodes less. The final pass sees the whole recording either
        // way, so the delivered text does not depend on it.
        let cpu = &config.cpu;
        let pick = |cpu_value: Option<f32>, value: f32| {
            if on_cpu {
                cpu_value.unwrap_or(value)
            } else {
                value
            }
        };
        let streaming_config = UnifiedStreamingConfig {
            chunk_secs: pick(cpu.streaming_chunk_secs, config.streaming_chunk_secs),
            left_context_secs: pick(
                cpu.streaming_left_context_secs,
                config.streaming_left_context_secs,
            ),
            right_context_secs: pick(
                cpu.streaming_right_context_secs,
                config.streaming_right_context_secs,
            ),
        }
        .validate()
        .map_err(|e| {
            TranscribeError::InitFailed(format!(
                "Invalid Parakeet streaming config: {}. \
                Check streaming_chunk_secs, streaming_left_context_secs, \
                streaming_right_context_secs in [parakeet].",
                e
            ))
        })?;

        let batch = ParakeetUnified::from_shared_with_streaming_config(&handle, streaming_config)
            .map_err(|e| {
            TranscribeError::InitFailed(format!(
                "Failed to spawn batch ParakeetUnified instance: {}",
                e
            ))
        })?;

        tracing::info!(
            "Parakeet streaming model loaded in {:.2}s (chunk={:.2}s, \
            left={:.2}s, right={:.2}s)",
            start.elapsed().as_secs_f32(),
            streaming_config.chunk_secs,
            streaming_config.left_context_secs,
            streaming_config.right_context_secs,
        );

        Ok(Self {
            handle,
            segment_secs: config.final_pass_segment_secs,
            pad: (config.pad_secs.max(0.0) * 16000.0) as usize,
            streaming_config,
            batch: Mutex::new(batch),
            progress: Mutex::default(),
            // Below the left context, or the model still starts over cold.
            max_pause: (config
                .streaming_max_pause_secs
                .max(0.0)
                .min((streaming_config.left_context_secs - 0.5).max(1.0))
                * 16000.0) as usize,
            vocabulary: TermSource::new(
                config.vocabulary_file.as_deref(),
                config.corrections_file.as_deref(),
            ),
            vocabulary_boost: config.vocabulary_boost,
            backend,
        })
    }
}

impl Transcriber for ParakeetStreamingTranscriber {
    fn transcribe(&self, samples: &[f32]) -> Result<String, TranscribeError> {
        self.decode(samples, true)
    }

    fn transcribe_piece(&self, samples: &[f32]) -> Result<String, TranscribeError> {
        self.decode(samples, false)
    }

    fn transcribe_timed(&self, _samples: &[f32]) -> Result<Vec<TimedSegment>, TranscribeError> {
        // ParakeetUnified does not expose timed segments in the same shape as
        // the batch ParakeetTranscriber. Users who need timed segments should
        // run with [parakeet] streaming = false.
        Err(TranscribeError::InferenceFailed(
            "transcribe_timed is not supported in streaming mode. \
            Set [parakeet] streaming = false for timed segments."
                .to_string(),
        ))
    }

    fn as_streaming(&self) -> Option<&dyn StreamingTranscriber> {
        Some(self)
    }

    fn set_vocabulary(&self, terms: &[String]) {
        if let Some(vocabulary) = &self.vocabulary {
            vocabulary.set_profile_terms(terms);
        }
    }

    fn runtime_metadata(&self) -> super::RuntimeMetadata {
        super::RuntimeMetadata {
            backend: Some(self.backend.to_string()),
            ..Default::default()
        }
    }
}

impl ParakeetStreamingTranscriber {
    /// Offline decode, piece by piece for long recordings. Only the piece
    /// that ends the recording can be padded: the others end at a pause cut
    /// (see `segment::plan_segments`), where padding costs punctuation.
    fn decode(&self, samples: &[f32], ends_recording: bool) -> Result<String, TranscribeError> {
        if samples.is_empty() {
            return Err(TranscribeError::AudioFormat(
                "Empty audio buffer".to_string(),
            ));
        }

        let mut batch = self.batch.lock().map_err(|e| {
            TranscribeError::InferenceFailed(format!("Failed to lock ParakeetUnified mutex: {}", e))
        })?;

        // Long recordings are decoded piece by piece, cut at pauses: see
        // `segment::plan_segments`. A recording shorter than one piece is a
        // single full-context decode, exactly as before.
        let segments = if self.segment_secs > 0.0 {
            super::segment::plan_segments(samples, 16000, self.segment_secs)
        } else {
            vec![(0, samples.len())]
        };
        let mut parts = Vec::with_capacity(segments.len());
        for (start, end) in segments {
            // transcribe_audio resets the instance first, so no state from
            // a prior piece or call bleeds in.
            let piece = &samples[start..end];
            let pad = ends_recording && end == samples.len() && ends_in_speech(piece);
            batch.set_offline_trailing_silence(if pad { self.pad } else { 0 });
            let terms = self
                .vocabulary
                .as_ref()
                .map(|v| v.terms())
                .filter(|t| !t.is_empty());
            let piece = match terms {
                Some(terms) => self.decode_with_vocabulary(&mut batch, piece, &terms),
                None => batch.transcribe_audio(piece.to_vec(), 16000, 1),
            }
            .map_err(|e| {
                TranscribeError::InferenceFailed(format!("ParakeetUnified inference failed: {}", e))
            })?;
            let piece = piece.trim();
            if !piece.is_empty() {
                parts.push(piece.to_string());
            }
        }

        Ok(parts.join(" "))
    }

    /// Offline decode of one piece, rescored against `terms`.
    fn decode_with_vocabulary(
        &self,
        batch: &mut ParakeetUnified,
        piece: &[f32],
        terms: &[vocab_rescore::Term],
    ) -> parakeet_rs::Result<String> {
        let decode = batch.decode_offline(piece.to_vec())?;
        let started = std::time::Instant::now();
        let scorer = UnifiedScorer {
            model: batch,
            decode: &decode,
        };
        let ids = match vocab_rescore::rescore(
            &decode.tokens,
            decode.frame_count,
            terms,
            self.vocabulary_boost,
            &scorer,
        ) {
            Some((ids, replacements)) => {
                for r in &replacements {
                    tracing::info!(
                        "Parakeet vocabulary: {:?} -> {:?} ({:+.1} nats)",
                        r.heard,
                        r.term,
                        r.margin
                    );
                }
                ids
            }
            None => decode.tokens.iter().map(|t| t.0).collect(),
        };
        let ms = started.elapsed().as_millis();
        if ms >= 100 {
            tracing::info!("Parakeet vocabulary rescoring took {}ms", ms);
        } else {
            tracing::debug!("Parakeet vocabulary rescoring took {}ms", ms);
        }
        Ok(batch.vocab().decode(&ids))
    }
}

/// The unified model's decoder_joint, scoring spans of one decode.
struct UnifiedScorer<'a> {
    model: &'a ParakeetUnified,
    decode: &'a OfflineDecode,
}

impl SpanScorer for UnifiedScorer<'_> {
    fn piece(&self, id: usize) -> &str {
        self.model.vocab().piece(id).unwrap_or("")
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        self.model.vocab().encode(text)
    }

    fn score(
        &self,
        from: usize,
        frames: std::ops::Range<usize>,
        candidates: &[Vec<usize>],
    ) -> Result<Vec<f32>, String> {
        self.model
            .score_pieces(self.decode, from, frames, candidates)
            .map_err(|e| e.to_string())
    }
}

/// Whether `samples` stop mid-word: their last 200 ms are well above the
/// recording's noise floor (its quietest tenth of 20 ms windows). A key
/// released right after the last word leaves the decoder no audio past it,
/// and it drops or clips that word ("said Shaggy" -> "said Sh"); `pad_secs`
/// of silence brings it back. A recording that already ends in a pause is
/// decoded as it is: padding those costs sentence-final punctuation.
/// Measured on LibriSpeech clips: natural endings sit at 0.4-2.3x the
/// floor (three outliers up to 46x), endings cut mid-decay at 6-305x.
fn ends_in_speech(samples: &[f32]) -> bool {
    const WINDOW: usize = 320;
    const TAIL: usize = 3200;
    if samples.len() < TAIL * 2 {
        return true;
    }
    let rms = |w: &[f32]| (w.iter().map(|v| v * v).sum::<f32>() / w.len() as f32).sqrt();
    let mut windows: Vec<f32> = samples.chunks_exact(WINDOW).map(rms).collect();
    let tenth = windows.len() / 10;
    let floor = *windows.select_nth_unstable_by(tenth, f32::total_cmp).1;
    // Most of the tail, not its average: a key-release click is one loud window.
    let loud = samples[samples.len() - TAIL..]
        .chunks_exact(WINDOW)
        .filter(|w| rms(w) > 4.0 * floor.max(1e-4))
        .count();
    loud * 2 >= TAIL / WINDOW
}

/// `[parakeet] streaming_max_pause_secs`: keeps at most that much of any
/// pause in the audio the stream decodes. After a pause longer than the
/// left context (5.6 s), the model starts again with no context at all and
/// drops the first word or two of the next sentence ("in fact he is" ->
/// "he is"); a shorter pause leaves it some. Quiet is judged like
/// `segment`'s cuts, relative to the levels heard so far: closer to the
/// noise floor than to the speech level, in 1 dB bins.
struct PauseLimiter {
    max: usize,
    quiet_run: usize,
    histogram: [u32; 121],
    blocks: u32,
    /// The latest left-out audio, replayed when speech resumes: a soft word
    /// start ("h", "th", a breath) can sit below the quiet threshold.
    held: std::collections::VecDeque<f32>,
}

/// Left-out audio kept for the replay (250 ms).
const PAUSE_HOLD: usize = 4000;
/// A chunk this long is not a capture block but the stop's tail; never left out.
const PAUSE_MAX_CHUNK: usize = 8000;

impl PauseLimiter {
    fn new(max: usize) -> Self {
        Self {
            max,
            quiet_run: 0,
            histogram: [0; 121],
            blocks: 0,
            held: Default::default(),
        }
    }

    fn percentile(&self, fraction: f32) -> f32 {
        let target = (self.blocks as f32 * fraction) as u32;
        let mut seen = 0;
        for (bin, count) in self.histogram.iter().enumerate() {
            seen += count;
            if seen > target {
                return bin as f32 - 120.0;
            }
        }
        0.0
    }

    /// The part of `chunk` (plus any replayed audio) the stream decodes.
    fn feed(&mut self, chunk: &[f32]) -> Vec<f32> {
        if self.max == 0 || chunk.len() > PAUSE_MAX_CHUNK {
            self.quiet_run = 0;
            return self.resume(chunk);
        }
        let power = chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32;
        let db = (10.0 * (power + 1e-12).log10()).clamp(-120.0, 0.0);
        self.histogram[(db + 120.0) as usize] += 1;
        self.blocks += 1;
        // A second of audio before the levels mean anything.
        if self.blocks < 50 {
            return chunk.to_vec();
        }
        let floor = self.percentile(0.1);
        let speech = self.percentile(0.9);
        if speech - floor < 10.0 || db >= floor + 0.25 * (speech - floor) {
            self.quiet_run = 0;
            return self.resume(chunk);
        }
        let keep = self.max.saturating_sub(self.quiet_run).min(chunk.len());
        self.quiet_run += chunk.len();
        self.held.extend(&chunk[keep..]);
        let excess = self.held.len().saturating_sub(PAUSE_HOLD);
        self.held.drain(..excess);
        chunk[..keep].to_vec()
    }

    fn resume(&mut self, chunk: &[f32]) -> Vec<f32> {
        let mut out: Vec<f32> = self.held.drain(..).collect();
        out.extend_from_slice(chunk);
        out
    }
}

impl StreamingTranscriber for ParakeetStreamingTranscriber {
    fn stream_progress(&self) -> Option<StreamProgress> {
        let current = self.progress.lock().ok()?.clone();
        let progress = current.lock().ok()?.clone();
        Some(progress)
    }

    fn start_stream(
        &self,
        mut samples_rx: mpsc::Receiver<Vec<f32>>,
    ) -> Result<StreamHandle, TranscribeError> {
        // Spawn a fresh ParakeetUnified instance for this session.
        let mut unified =
            ParakeetUnified::from_shared_with_streaming_config(&self.handle, self.streaming_config)
                .map_err(|e| {
                    TranscribeError::InitFailed(format!(
                        "Failed to spawn streaming ParakeetUnified instance: {}",
                        e
                    ))
                })?;

        let progress: Arc<Mutex<StreamProgress>> = Arc::default();
        if let Ok(mut slot) = self.progress.lock() {
            *slot = progress.clone();
        }
        let right_context = self.streaming_config.right_context_samples();
        let mut pauses = PauseLimiter::new(self.max_pause);

        let (events_tx, events_rx) = mpsc::channel::<StreamingEvent>(64);
        let (cancel_tx, mut cancel_rx) = oneshot::channel::<()>();

        let task = tokio::task::spawn_blocking(move || -> Result<(), TranscribeError> {
            let segment_id: u64 = 0;
            let runtime = tokio::runtime::Handle::current();

            loop {
                // Check cancel without blocking. A cancelled stream ends at
                // once, without decoding its buffered tail: its text is
                // being discarded (cancel) or replaced by the final pass.
                match cancel_rx.try_recv() {
                    Ok(()) => {
                        tracing::debug!("Parakeet streaming session cancelled");
                        let _ = runtime.block_on(events_tx.send(StreamingEvent::Ended));
                        return Ok(());
                    }
                    Err(oneshot::error::TryRecvError::Closed) => {
                        // Sender dropped without sending; treat as cancel.
                        break;
                    }
                    Err(oneshot::error::TryRecvError::Empty) => {}
                }

                let chunk = match runtime.block_on(samples_rx.recv()) {
                    Some(c) => c,
                    None => break, // graceful EOF
                };

                if chunk.is_empty() {
                    continue;
                }
                let recorded = chunk.len();
                let chunk = pauses.feed(&chunk);
                if chunk.is_empty() {
                    if let Ok(mut p) = progress.lock() {
                        p.fed += recorded;
                    }
                    continue;
                }

                let text = match unified.transcribe_chunk(&chunk) {
                    Ok(t) => t,
                    Err(e) => {
                        let err = TranscribeError::InferenceFailed(format!(
                            "ParakeetUnified::transcribe_chunk failed: {}",
                            e
                        ));
                        let _ = runtime.block_on(events_tx.send(StreamingEvent::Error(err)));
                        let _ = runtime.block_on(events_tx.send(StreamingEvent::Ended));
                        return Ok(());
                    }
                };

                if let Ok(mut p) = progress.lock() {
                    p.fed += recorded;
                    if !text.is_empty() {
                        p.text.push_str(&text);
                        p.text_through = p.fed.saturating_sub(right_context);
                    }
                }

                // Every non-empty delta is new text. Comparing against the
                // previous delta (as a cumulative-transcript backend would)
                // drops genuine repeats like "no, no" spoken across chunks.
                if !text.is_empty() {
                    let _ = runtime
                        .block_on(events_tx.send(StreamingEvent::Partial { text, segment_id }));
                }
            }

            // Drain any buffered audio with flush() on close. The tail is a
            // delta too, so it is not trimmed: its leading space is the word
            // boundary with the text already emitted. On failure nothing new
            // was decoded, so there is no tail to send.
            let final_text = match unified.flush() {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!("ParakeetUnified::flush failed: {}", e);
                    String::new()
                }
            };

            if !final_text.is_empty() {
                let _ = runtime.block_on(events_tx.send(StreamingEvent::Final {
                    text: final_text,
                    segment_id,
                }));
            }
            let _ = runtime.block_on(events_tx.send(StreamingEvent::Ended));
            Ok(())
        });

        // Map the spawn_blocking JoinHandle to the trait's expected
        // JoinHandle<Result<(), TranscribeError>> shape.
        let task = tokio::spawn(async move {
            match task.await {
                Ok(r) => r,
                Err(join_err) => Err(TranscribeError::InferenceFailed(format!(
                    "Parakeet streaming task panicked: {}",
                    join_err
                ))),
            }
        });

        Ok(StreamHandle {
            events: events_rx,
            cancel: cancel_tx,
            task,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test that the type and trait signatures compile. Real model-
    /// driven tests require ONNX runtime + a downloaded model and live in
    /// the integration test suite gated behind `--features parakeet`.
    #[test]
    fn streaming_config_validation_rejects_zero_chunk() {
        let cfg = UnifiedStreamingConfig {
            chunk_secs: 0.0,
            left_context_secs: 1.0,
            right_context_secs: 0.5,
        };
        assert!(cfg.validate().is_err());
    }

    /// The `[parakeet] streaming_*_secs` defaults must pass parakeet-rs's
    /// own validation (each length a multiple of 80ms), or enabling
    /// streaming without hand-tuned values fails at model load.
    #[test]
    fn voxtype_streaming_defaults_validate() {
        let config = ParakeetConfig::default();
        let cfg = UnifiedStreamingConfig {
            chunk_secs: config.streaming_chunk_secs,
            left_context_secs: config.streaming_left_context_secs,
            right_context_secs: config.streaming_right_context_secs,
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn only_audio_cut_off_mid_speech_counts_as_ending_in_speech() {
        let speech = |i: usize| (i as f32 * 0.05).sin() * 0.2;
        let noise = |i: usize| ((i * 7919) % 97) as f32 * 2e-5;
        let mut paused: Vec<f32> = (0..32000).map(|i| speech(i) + noise(i)).collect();
        paused.extend((0..8000).map(noise));
        assert!(!ends_in_speech(&paused));
        let mut cut: Vec<f32> = (0..8000).map(noise).collect();
        cut.extend((0..32000).map(|i| speech(i) + noise(i)));
        assert!(ends_in_speech(&cut));
    }

    #[test]
    fn pauses_past_the_limit_are_left_out_of_the_stream() {
        let block = |loud: bool| -> Vec<f32> {
            (0..320)
                .map(|i| {
                    if loud {
                        (i as f32 * 0.3).sin() * 0.2
                    } else {
                        ((i * 31) % 7) as f32 * 1e-4
                    }
                })
                .collect()
        };
        let mut limiter = PauseLimiter::new(16000);
        // Half a second of room noise, then speech: the levels to judge by.
        let start: usize = (0..125).map(|i| limiter.feed(&block(i >= 25)).len()).sum();
        assert_eq!(start, 125 * 320);
        // 1 s of a 3 s pause is kept, cut mid-block where the limit falls.
        let pause: usize = (0..150).map(|_| limiter.feed(&block(false)).len()).sum();
        assert_eq!(pause, 16000);
        // Speech resumes with the last 250 ms of the pause replayed first.
        assert_eq!(limiter.feed(&block(true)).len(), PAUSE_HOLD + 320);
        assert_eq!(limiter.feed(&block(false)).len(), 320);
        // The stop's long tail chunk is never left out.
        let mut tail = PauseLimiter::new(16000);
        for i in 0..125 {
            let _ = tail.feed(&block(i >= 25));
        }
        for _ in 0..150 {
            let _ = tail.feed(&block(false));
        }
        assert_eq!(tail.feed(&vec![0.0; 9600]).len(), PAUSE_HOLD + 9600);
        assert_eq!(PauseLimiter::new(0).feed(&block(false)).len(), 320);
    }

    /// Runs only with `VOXTYPE_TEST_PARAKEET_MODEL` set to a
    /// parakeet-unified-en-0.6b directory (the model is not in the repo).
    #[test]
    fn vocabulary_rescoring_fixes_a_name_greedy_decoding_misspells() {
        let Some(model) = std::env::var_os("VOXTYPE_TEST_PARAKEET_MODEL") else {
            eprintln!("skipped: VOXTYPE_TEST_PARAKEET_MODEL is not set");
            return;
        };
        let wav = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/parakeet/kaliko_hesitated.wav"
        );
        let audio: Vec<f32> = hound::WavReader::open(wav)
            .unwrap()
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let vocabulary = dir.path().join("vocabulary.txt");
        std::fs::write(&vocabulary, "# names\nKaliko\n").unwrap();
        let config = ParakeetConfig {
            model: model.to_string_lossy().into_owned(),
            streaming: true,
            ..ParakeetConfig::default()
        };
        let decode = |config: &ParakeetConfig| match ParakeetStreamingTranscriber::new(config) {
            Ok(t) => t.transcribe(&audio).unwrap(),
            Err(e) => panic!("load: {e}"),
        };
        let plain = decode(&config);
        assert!(
            !plain.contains("Kaliko"),
            "greedy decode already right: {plain}"
        );
        let rescored = decode(&ParakeetConfig {
            vocabulary_file: Some(vocabulary),
            ..config
        });
        assert!(
            rescored.starts_with("Kaliko hesitated"),
            "{plain} -> {rescored}"
        );
    }

    /// Same model gate as above. Expected pieces come from Python
    /// `sentencepiece` on the model's tokenizer.model.
    #[test]
    fn vocabulary_terms_encode_like_sentencepiece() {
        let Some(model) = std::env::var_os("VOXTYPE_TEST_PARAKEET_MODEL") else {
            eprintln!("skipped: VOXTYPE_TEST_PARAKEET_MODEL is not set");
            return;
        };
        let config = ParakeetConfig {
            model: model.to_string_lossy().into_owned(),
            streaming: true,
            ..ParakeetConfig::default()
        };
        let Ok(t) = ParakeetStreamingTranscriber::new(&config) else {
            panic!("load failed")
        };
        let batch = t.batch.lock().unwrap();
        let vocab = batch.vocab();
        for (text, pieces) in [
            ("Kaliko", &["\u{2581}K", "al", "i", "k", "o"][..]),
            ("Shaggy", &["\u{2581}Sh", "ag", "gy"]),
            ("voxtype", &["\u{2581}v", "o", "xt", "y", "pe"]),
            ("Hyprland", &["\u{2581}H", "y", "pr", "land"]),
            ("Kubernetes", &["\u{2581}K", "u", "ber", "n", "et", "es"]),
            (
                "Hello, world.",
                &["\u{2581}He", "ll", "o", ",", "\u{2581}wor", "ld", "."],
            ),
        ] {
            let got: Vec<&str> = vocab
                .encode(text)
                .iter()
                .map(|&id| vocab.piece(id).unwrap())
                .collect();
            assert_eq!(got, pieces, "{text}");
        }
    }

    #[test]
    fn streaming_config_validation_accepts_defaults() {
        let cfg = UnifiedStreamingConfig::default();
        assert!(cfg.validate().is_ok());
    }

    /// Regression for #442: enabling streaming on a known non-streaming model
    /// (e.g. `parakeet-tdt-0.6b-v3`) must fail early in `new()` with a
    /// message that names the model and points the user at the
    /// streaming-compatible alternative, instead of falling through to an
    /// opaque ONNX Gather shape error at the first audio chunk.
    #[test]
    fn new_rejects_streaming_on_known_incompatible_model() {
        let cfg = ParakeetConfig {
            model: "parakeet-tdt-0.6b-v3".to_string(),
            streaming: true,
            ..ParakeetConfig::default()
        };
        // Can't use `.expect_err`: ParakeetStreamingTranscriber wraps
        // parakeet-rs types that don't derive Debug.
        let err = match ParakeetStreamingTranscriber::new(&cfg) {
            Ok(_) => panic!("streaming on a non-streaming model should error before model load"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("parakeet-tdt-0.6b-v3"),
            "error must name the configured model: {msg}"
        );
        assert!(
            msg.contains(crate::setup::model::DEFAULT_PARAKEET_STREAMING_MODEL),
            "error must name the streaming-compatible alternative: {msg}"
        );
        assert!(
            msg.contains("streaming = false") || msg.contains("voxtype setup model"),
            "error must offer at least one concrete fix: {msg}"
        );
    }
}
