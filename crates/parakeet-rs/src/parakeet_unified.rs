use crate::audio::{self, load_audio};
use crate::config::PreprocessorConfig;
use crate::decoder::{TimedToken, TranscriptionResult};
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_unified::{ParakeetUnifiedModel, UnifiedModelConfig};
use crate::nemotron::SentencePieceVocab;
use crate::timestamps::{process_timestamps, TimestampMode};
use crate::transcriber::Transcriber;
use ndarray::Array3;
use std::path::Path;
use std::sync::{Arc, Mutex};

const SAMPLE_RATE: usize = 16000;
const FEATURE_SIZE: usize = 128;
const HOP_LENGTH: usize = 160;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const PREEMPHASIS: f32 = 0.97;
const DECODER_LSTM_DIM: usize = 640;
const DECODER_LSTM_LAYERS: usize = 2;
const SUBSAMPLING_FACTOR: usize = 8;
const MAX_SYMBOLS_PER_STEP: usize = 10;

#[derive(Debug, Clone, Copy)]
pub struct UnifiedStreamingConfig {
    pub left_context_secs: f32,
    pub chunk_secs: f32,
    pub right_context_secs: f32,
}

impl Default for UnifiedStreamingConfig {
    fn default() -> Self {
        Self {
            left_context_secs: 5.6,
            chunk_secs: 0.56,
            right_context_secs: 0.56,
        }
    }
}

impl UnifiedStreamingConfig {
    fn frames_from_secs(secs: f32) -> usize {
        ((secs * SAMPLE_RATE as f32) / HOP_LENGTH as f32).round() as usize
    }

    pub fn validate(self) -> Result<Self> {
        let left_frames = self.left_context_frames();
        let chunk_frames = self.chunk_frames();
        let right_frames = self.right_context_frames();

        if chunk_frames == 0 {
            return Err(Error::Config(
                "Unified streaming chunk size must be greater than zero".to_string(),
            ));
        }

        for (name, frames) in [
            ("left_context_secs", left_frames),
            ("chunk_secs", chunk_frames),
            ("right_context_secs", right_frames),
        ] {
            if frames % SUBSAMPLING_FACTOR != 0 {
                return Err(Error::Config(format!(
                    "{name} must map to a mel-frame count divisible by {SUBSAMPLING_FACTOR}"
                )));
            }
        }

        Ok(self)
    }

    pub fn left_context_frames(self) -> usize {
        Self::frames_from_secs(self.left_context_secs)
    }

    pub fn chunk_frames(self) -> usize {
        Self::frames_from_secs(self.chunk_secs)
    }

    pub fn right_context_frames(self) -> usize {
        Self::frames_from_secs(self.right_context_secs)
    }

    pub fn total_window_frames(self) -> usize {
        self.left_context_frames() + self.chunk_frames() + self.right_context_frames()
    }

    pub fn left_context_samples(self) -> usize {
        self.left_context_frames() * HOP_LENGTH
    }

    pub fn chunk_samples(self) -> usize {
        self.chunk_frames() * HOP_LENGTH
    }

    pub fn right_context_samples(self) -> usize {
        self.right_context_frames() * HOP_LENGTH
    }

    pub fn total_window_samples(self) -> usize {
        self.total_window_frames() * HOP_LENGTH
    }

    pub fn chunk_encoder_frames(self) -> usize {
        self.chunk_frames() / SUBSAMPLING_FACTOR
    }

    pub fn left_context_encoder_frames(self) -> usize {
        self.left_context_frames() / SUBSAMPLING_FACTOR
    }
}

/// Shared handle to a loaded ParakeetUnified model.
/// The ONNX session is loaded once and reference-counted.
///
/// Use [`ParakeetUnifiedHandle::load`] to load from disk, then
/// [`ParakeetUnified::from_shared`] to spawn each stream with its own state.
#[derive(Clone)]
pub struct ParakeetUnifiedHandle {
    model: Arc<Mutex<ParakeetUnifiedModel>>,
    vocab: Arc<SentencePieceVocab>,
    preprocessor_config: Arc<PreprocessorConfig>,
    blank_id: usize,
}

pub struct ParakeetUnified {
    model: Arc<Mutex<ParakeetUnifiedModel>>,
    vocab: Arc<SentencePieceVocab>,
    preprocessor_config: Arc<PreprocessorConfig>,
    state_1: Array3<f32>,
    state_2: Array3<f32>,
    last_token: i32,
    blank_id: usize,
    streaming_config: UnifiedStreamingConfig,
    audio_buffer: Vec<f32>,
    buffer_start_sample: usize,
    next_chunk_start_sample: usize,
    accumulated_tokens: Vec<usize>,
    accumulated_timed_tokens: Vec<TimedToken>,
    offline_trailing_silence: usize,
}

impl ParakeetUnifiedHandle {
    /// Load the ParakeetUnified model, vocabulary, and preprocessor config
    /// from a directory.
    pub fn load<P: AsRef<Path>>(
        path: P,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let vocab = SentencePieceVocab::from_file(path.join("tokenizer.model"))?;
        let blank_id = vocab.size();

        let model_config = UnifiedModelConfig {
            vocab_size: vocab.size() + 1,
            blank_id,
            decoder_lstm_dim: DECODER_LSTM_DIM,
            decoder_lstm_layers: DECODER_LSTM_LAYERS,
            subsampling_factor: SUBSAMPLING_FACTOR,
        };

        let model = ParakeetUnifiedModel::from_pretrained(
            path,
            exec_config.unwrap_or_default(),
            model_config,
        )?;

        let preprocessor_config = PreprocessorConfig {
            feature_extractor_type: "ParakeetFeatureExtractor".to_string(),
            feature_size: FEATURE_SIZE,
            hop_length: HOP_LENGTH,
            n_fft: N_FFT,
            padding_side: "right".to_string(),
            padding_value: 0.0,
            preemphasis: PREEMPHASIS,
            processor_class: "ParakeetProcessor".to_string(),
            return_attention_mask: true,
            sampling_rate: SAMPLE_RATE,
            win_length: WIN_LENGTH,
        };

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            vocab: Arc::new(vocab),
            preprocessor_config: Arc::new(preprocessor_config),
            blank_id,
        })
    }
}

impl ParakeetUnified {
    pub fn from_pretrained<P: AsRef<Path>>(
        path: P,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        Self::from_pretrained_with_streaming_config(
            path,
            exec_config,
            UnifiedStreamingConfig::default(),
        )
    }

    pub fn from_pretrained_with_streaming_config<P: AsRef<Path>>(
        path: P,
        exec_config: Option<ExecutionConfig>,
        streaming_config: UnifiedStreamingConfig,
    ) -> Result<Self> {
        let handle = ParakeetUnifiedHandle::load(path, exec_config)?;
        Self::from_shared_with_streaming_config(&handle, streaming_config)
    }

    /// Spawn a new ParakeetUnified instance bound to a shared model, using the
    /// default streaming profile.
    pub fn from_shared(handle: &ParakeetUnifiedHandle) -> Self {
        // default config is pre-validated, so unwrap is safe
        Self::from_shared_with_streaming_config(handle, UnifiedStreamingConfig::default())
            .expect("default UnifiedStreamingConfig is always valid")
    }

    /// Spawn a new ParakeetUnified instance bound to a shared model with a
    /// custom streaming profile. Each instance owns independent decoder and
    /// audio-buffer state; the ONNX session is shared through the handle.
    pub fn from_shared_with_streaming_config(
        handle: &ParakeetUnifiedHandle,
        streaming_config: UnifiedStreamingConfig,
    ) -> Result<Self> {
        let streaming_config = streaming_config.validate()?;
        let blank_id = handle.blank_id;

        Ok(Self {
            model: Arc::clone(&handle.model),
            vocab: Arc::clone(&handle.vocab),
            preprocessor_config: Arc::clone(&handle.preprocessor_config),
            state_1: Array3::zeros((DECODER_LSTM_LAYERS, 1, DECODER_LSTM_DIM)),
            state_2: Array3::zeros((DECODER_LSTM_LAYERS, 1, DECODER_LSTM_DIM)),
            last_token: blank_id as i32,
            blank_id,
            streaming_config,
            audio_buffer: Vec::new(),
            buffer_start_sample: 0,
            next_chunk_start_sample: 0,
            accumulated_tokens: Vec::new(),
            accumulated_timed_tokens: Vec::new(),
            offline_trailing_silence: 0,
        })
    }

    pub fn streaming_config(&self) -> UnifiedStreamingConfig {
        self.streaming_config
    }

    pub fn preprocessor_config(&self) -> &PreprocessorConfig {
        &self.preprocessor_config
    }

    /// Samples of silence the next offline decodes append to their audio
    /// (normalized with the statistics of the audio alone). Streaming is
    /// unaffected.
    pub fn set_offline_trailing_silence(&mut self, samples: usize) {
        self.offline_trailing_silence = samples;
    }

    pub fn reset(&mut self) {
        self.state_1.fill(0.0);
        self.state_2.fill(0.0);
        self.last_token = self.blank_id as i32;
        self.audio_buffer.clear();
        self.buffer_start_sample = 0;
        self.next_chunk_start_sample = 0;
        self.accumulated_tokens.clear();
        self.accumulated_timed_tokens.clear();
    }

    pub fn get_timed_transcript(&self, mode: TimestampMode) -> TranscriptionResult {
        let text = self.get_transcript();
        let tokens = process_timestamps(&self.accumulated_timed_tokens, mode);
        TranscriptionResult { text, tokens }
    }

    pub fn get_transcript(&self) -> String {
        let valid: Vec<usize> = self
            .accumulated_tokens
            .iter()
            .copied()
            .filter(|&token| token < self.blank_id)
            .collect();
        self.vocab.decode(&valid)
    }

    pub fn transcribe_audio(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
    ) -> Result<String> {
        self.transcribe_offline(audio, sample_rate, channels, None)
            .map(|result| result.text)
    }

    pub fn transcribe_file<P: AsRef<Path>>(&mut self, audio_path: P) -> Result<String> {
        let (audio, spec) = load_audio(audio_path)?;
        self.transcribe_audio(audio, spec.sample_rate, spec.channels)
    }

    pub fn transcribe_chunk(&mut self, audio_chunk: &[f32]) -> Result<String> {
        self.audio_buffer.extend_from_slice(audio_chunk);
        self.process_ready_chunks(false)
    }

    pub fn flush(&mut self) -> Result<String> {
        self.process_ready_chunks(true)
    }

    fn process_ready_chunks(&mut self, flush: bool) -> Result<String> {
        let mut emitted = String::new();
        let chunk_samples = self.streaming_config.chunk_samples();
        let right_context_samples = self.streaming_config.right_context_samples();

        loop {
            let total_received = self.buffer_start_sample + self.audio_buffer.len();
            let ready = if flush {
                total_received > self.next_chunk_start_sample
            } else {
                total_received
                    >= self.next_chunk_start_sample + chunk_samples + right_context_samples
            };

            if !ready {
                break;
            }

            let (window_audio, left_encoder_frames, chunk_encoder_frames) =
                self.build_window_audio(self.next_chunk_start_sample, total_received, flush);
            if chunk_encoder_frames == 0 {
                break;
            }

            let features = audio::extract_features_raw(
                window_audio,
                SAMPLE_RATE as u32,
                1,
                &self.preprocessor_config,
            )?;
            let (encoded, encoded_len) = {
                let mut model = self.model.lock().map_err(|e| {
                    Error::Model(format!("Failed to acquire model lock: {e}"))
                })?;
                model.run_encoder(&features)?
            };

            let available_frames = (encoded_len as usize).min(encoded.shape()[2]);
            let start_frame = left_encoder_frames.min(available_frames);
            let end_frame = (start_frame + chunk_encoder_frames).min(available_frames);

            let absolute_frame_offset =
                self.next_chunk_start_sample / (HOP_LENGTH * SUBSAMPLING_FACTOR);
            let tokens =
                self.decode_encoder_frames(&encoded, start_frame, end_frame, absolute_frame_offset)?;
            self.accumulated_tokens
                .extend(tokens.iter().map(|(id, _)| *id));
            self.accumulated_timed_tokens
                .extend(self.tokens_to_timed(&tokens));
            emitted.push_str(&self.decode_incremental_tokens(&tokens));

            self.next_chunk_start_sample += chunk_samples;
            self.trim_audio_buffer();

            if flush && total_received <= self.next_chunk_start_sample {
                break;
            }
        }

        Ok(emitted)
    }

    fn build_window_audio(
        &self,
        chunk_start_sample: usize,
        total_received: usize,
        flush: bool,
    ) -> (Vec<f32>, usize, usize) {
        let left_context_samples = self.streaming_config.left_context_samples();
        let chunk_samples = self.streaming_config.chunk_samples();
        let right_context_samples = self.streaming_config.right_context_samples();

        let available_left = chunk_start_sample.saturating_sub(self.buffer_start_sample);
        let available_left = available_left.min(left_context_samples);
        let available_main = total_received.saturating_sub(chunk_start_sample).min(chunk_samples);
        let available_right = if flush {
            total_received
                .saturating_sub(chunk_start_sample + available_main)
                .min(right_context_samples)
        } else {
            right_context_samples
        };

        let window_start = chunk_start_sample.saturating_sub(available_left);
        let window_end = chunk_start_sample + available_main + available_right;
        let total_window_samples = window_end.saturating_sub(window_start);

        let left_encoder_frames = (available_left / HOP_LENGTH) / SUBSAMPLING_FACTOR;
        let chunk_encoder_frames = (available_main / HOP_LENGTH) / SUBSAMPLING_FACTOR;

        let mut window = vec![0.0f32; total_window_samples];
        let buffer_end = self.buffer_start_sample + self.audio_buffer.len();
        let copy_start = window_start.max(self.buffer_start_sample);
        let copy_end = window_end.min(buffer_end);

        if copy_end > copy_start {
            let src_start = copy_start - self.buffer_start_sample;
            let dst_start = copy_start - window_start;
            let len = copy_end - copy_start;
            window[dst_start..dst_start + len]
                .copy_from_slice(&self.audio_buffer[src_start..src_start + len]);
        }

        (window, left_encoder_frames, chunk_encoder_frames)
    }

    fn trim_audio_buffer(&mut self) {
        let keep_from = self
            .next_chunk_start_sample
            .saturating_sub(self.streaming_config.left_context_samples());
        if keep_from <= self.buffer_start_sample {
            return;
        }

        let drop = keep_from - self.buffer_start_sample;
        if drop == 0 {
            return;
        }

        if drop >= self.audio_buffer.len() {
            self.audio_buffer.clear();
            self.buffer_start_sample = keep_from;
            return;
        }

        self.audio_buffer.drain(0..drop);
        self.buffer_start_sample = keep_from;
    }

    fn decode_encoder_frames(
        &mut self,
        encoder_out: &Array3<f32>,
        start_frame: usize,
        end_frame: usize,
        absolute_frame_offset: usize,
    ) -> Result<Vec<(usize, usize)>> {
        self.decode_encoder_frames_recording(encoder_out, start_frame, end_frame, absolute_frame_offset, None)
    }

    /// Greedy decode of `start_frame..end_frame`. With `record`, also keeps
    /// the predictor input (previous token and LSTM state) that emitted
    /// each token, in token order.
    fn decode_encoder_frames_recording(
        &mut self,
        encoder_out: &Array3<f32>,
        start_frame: usize,
        end_frame: usize,
        absolute_frame_offset: usize,
        mut record: Option<&mut Vec<PredictorState>>,
    ) -> Result<Vec<(usize, usize)>> {
        let mut tokens = Vec::new();
        let hidden_dim = encoder_out.shape()[1];
        let end_frame = end_frame.min(encoder_out.shape()[2]);

        // Hold the lock once across the decoder loop to avoid per-step acquire/release.
        let mut model = self
            .model
            .lock()
            .map_err(|e| Error::Model(format!("Failed to acquire model lock: {e}")))?;

        for frame_idx in start_frame..end_frame {
            let frame = encoder_out
                .slice(ndarray::s![0, .., frame_idx])
                .to_owned()
                .to_shape((1, hidden_dim, 1))
                .map_err(|e| Error::Model(format!("Failed to reshape encoder frame: {e}")))?
                .to_owned();

            let absolute_frame = absolute_frame_offset + (frame_idx - start_frame);

            for _ in 0..MAX_SYMBOLS_PER_STEP {
                let (logits, new_state_1, new_state_2) = model.run_decoder(
                    &frame,
                    self.last_token,
                    &self.state_1,
                    &self.state_2,
                )?;

                let token_id = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(self.blank_id);

                if token_id == self.blank_id {
                    break;
                }

                tokens.push((token_id, absolute_frame));
                if let Some(record) = record.as_deref_mut() {
                    record.push(PredictorState {
                        last_token: self.last_token,
                        state_1: self.state_1.clone(),
                        state_2: self.state_2.clone(),
                    });
                }
                self.last_token = token_id as i32;
                self.state_1 = new_state_1;
                self.state_2 = new_state_2;
            }
        }

        Ok(tokens)
    }

    fn encoder_frame_to_seconds(frame: usize) -> f32 {
        (frame * SUBSAMPLING_FACTOR * HOP_LENGTH) as f32 / SAMPLE_RATE as f32
    }

    fn tokens_to_timed(&self, tokens: &[(usize, usize)]) -> Vec<TimedToken> {
        tokens
            .iter()
            .filter(|(id, _)| *id < self.blank_id)
            .map(|&(id, frame)| TimedToken {
                text: self.vocab.decode_single(id),
                start: Self::encoder_frame_to_seconds(frame),
                end: Self::encoder_frame_to_seconds(frame + 1),
            })
            .collect()
    }

    fn decode_incremental_tokens(&self, tokens: &[(usize, usize)]) -> String {
        let mut text = String::new();
        for &(token, _) in tokens {
            if token < self.blank_id {
                text.push_str(&self.vocab.decode_single(token));
            }
        }
        text
    }

    fn transcribe_offline(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult> {
        self.reset();

        let features = audio::extract_features_padded(
            audio,
            sample_rate,
            channels,
            &self.preprocessor_config,
            self.offline_trailing_silence,
        )?;
        let (encoded, encoded_len) = {
            let mut model = self
                .model
                .lock()
                .map_err(|e| Error::Model(format!("Failed to acquire model lock: {e}")))?;
            model.run_encoder(&features)?
        };
        let frame_count = (encoded_len as usize).min(encoded.shape()[2]);
        let tokens = self.decode_encoder_frames(&encoded, 0, frame_count, 0)?;
        self.accumulated_tokens = tokens.iter().map(|(id, _)| *id).collect();
        self.accumulated_timed_tokens = self.tokens_to_timed(&tokens);

        let text = self.get_transcript();
        let timed = match mode {
            Some(m) => process_timestamps(&self.accumulated_timed_tokens, m),
            None => self.accumulated_timed_tokens.clone(),
        };

        Ok(TranscriptionResult {
            text,
            tokens: timed,
        })
    }
}

/// Predictor (decoder LSTM) input at one point of a greedy decode.
#[derive(Clone)]
pub struct PredictorState {
    last_token: i32,
    state_1: Array3<f32>,
    state_2: Array3<f32>,
}

/// A full-context greedy decode that can be rescored afterwards: the
/// emitted tokens with their encoder frames, the predictor input that
/// emitted each one, and the encoder output itself.
pub struct OfflineDecode {
    /// `(token id, encoder frame)` in emission order, blanks excluded.
    pub tokens: Vec<(usize, usize)>,
    /// Predictor input that emitted `tokens[i]`; one extra entry at the end
    /// holds the state after the last token.
    states: Vec<PredictorState>,
    encoded: Array3<f32>,
    /// Encoder frames that carry audio (80 ms each).
    pub frame_count: usize,
}

impl ParakeetUnified {
    /// Like [`Self::transcribe_audio`] (16 kHz mono), but keeps what
    /// [`Self::score_pieces`] needs to rescore spans of the result.
    pub fn decode_offline(&mut self, audio: Vec<f32>) -> Result<OfflineDecode> {
        self.reset();
        let features = audio::extract_features_padded(
            audio,
            SAMPLE_RATE as u32,
            1,
            &self.preprocessor_config,
            self.offline_trailing_silence,
        )?;
        let (encoded, encoded_len) = {
            let mut model = self
                .model
                .lock()
                .map_err(|e| Error::Model(format!("Failed to acquire model lock: {e}")))?;
            model.run_encoder(&features)?
        };
        let frame_count = (encoded_len as usize).min(encoded.shape()[2]);
        let mut states = Vec::new();
        let tokens: Vec<(usize, usize)> = self
            .decode_encoder_frames_recording(&encoded, 0, frame_count, 0, Some(&mut states))?
            .into_iter()
            .filter(|(id, _)| *id < self.blank_id)
            .collect();
        states.push(PredictorState {
            last_token: self.last_token,
            state_1: self.state_1.clone(),
            state_2: self.state_2.clone(),
        });
        Ok(OfflineDecode { tokens, states, encoded, frame_count })
    }

    /// Log-probability (nats) of each candidate in `candidates` being
    /// exactly what the model emits over encoder `frames`, starting from
    /// the predictor state before `decode.tokens[from]` (or after the last
    /// token when `from == decode.tokens.len()`): the RNN-T forward
    /// algorithm over all alignments, ending in a blank on the last frame.
    /// All candidates share one decoder_joint run.
    pub fn score_pieces(
        &self,
        decode: &OfflineDecode,
        from: usize,
        frames: std::ops::Range<usize>,
        candidates: &[Vec<usize>],
    ) -> Result<Vec<f32>> {
        let state = decode
            .states
            .get(from)
            .ok_or_else(|| Error::Model(format!("score_pieces: no predictor state at token {from}")))?;
        let end = frames.end.min(decode.frame_count);
        if frames.start >= end || candidates.is_empty() {
            return Err(Error::Model("score_pieces: empty frame range or no candidates".into()));
        }
        let span = decode.encoded.slice(ndarray::s![.., .., frames.start..end]).to_owned();
        let targets: Vec<Vec<i32>> = candidates
            .iter()
            .map(|c| std::iter::once(state.last_token).chain(c.iter().map(|&t| t as i32)).collect())
            .collect();
        let (logits, [batch, time, max_u, vocab]) = {
            let mut model = self
                .model
                .lock()
                .map_err(|e| Error::Model(format!("Failed to acquire model lock: {e}")))?;
            model.run_joint_batch(&span, &targets, &state.state_1, &state.state_2)?
        };
        if batch != candidates.len() || vocab <= self.blank_id {
            return Err(Error::Model(format!(
                "score_pieces: unexpected joint output [{batch}, {time}, {max_u}, {vocab}]"
            )));
        }
        Ok(candidates
            .iter()
            .enumerate()
            .map(|(b, pieces)| {
                let at = |t: usize, u: usize| &logits[((b * time + t) * max_u + u) * vocab..][..vocab];
                rnnt_log_likelihood(time, pieces, self.blank_id, at)
            })
            .collect())
    }

    /// The tokenizer, to encode candidate text and read pieces back.
    pub fn vocab(&self) -> &SentencePieceVocab {
        &self.vocab
    }
}

/// RNN-T forward algorithm: log P(`pieces` | `time` frames), where `logits(t, u)`
/// is the joint output at frame `t` after the predictor has consumed the
/// first `u` pieces.
fn rnnt_log_likelihood<'a>(
    time: usize,
    pieces: &[usize],
    blank: usize,
    logits: impl Fn(usize, usize) -> &'a [f32],
) -> f32 {
    let log_softmax = |row: &[f32], id: usize| {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|v| (v - max).exp()).sum();
        row[id] - max - sum.ln()
    };
    let log_add = |a: f32, b: f32| {
        if a == f32::NEG_INFINITY {
            return b;
        }
        if b == f32::NEG_INFINITY {
            return a;
        }
        let m = a.max(b);
        m + ((a - m).exp() + (b - m).exp()).ln()
    };
    let u_len = pieces.len();
    // alpha[u]: log P(first u pieces emitted, now at frame t, not yet past it)
    let mut alpha = vec![f32::NEG_INFINITY; u_len + 1];
    alpha[0] = 0.0;
    for t in 0..time {
        // alpha holds arrivals at frame t (a blank on t-1); add emissions on t.
        for u in 1..=u_len {
            let emit = alpha[u - 1] + log_softmax(logits(t, u - 1), pieces[u - 1]);
            alpha[u] = log_add(alpha[u], emit);
        }
        if t + 1 < time {
            for u in 0..=u_len {
                alpha[u] += log_softmax(logits(t, u), blank);
            }
        }
    }
    alpha[u_len] + log_softmax(logits(time - 1, u_len), blank)
}

impl Transcriber for ParakeetUnified {
    fn transcribe_samples(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult> {
        self.transcribe_offline(audio, sample_rate, channels, mode)
    }
}

#[cfg(test)]
mod tests {
    use super::{rnnt_log_likelihood, UnifiedStreamingConfig};

    /// Sum of P over every RNN-T alignment, enumerated explicitly.
    fn brute_force(time: usize, pieces: &[usize], blank: usize, logits: &dyn Fn(usize, usize) -> Vec<f32>) -> f64 {
        let p = |t: usize, u: usize, id: usize| {
            let row = logits(t, u);
            let z: f64 = row.iter().map(|v| (*v as f64).exp()).sum();
            (row[id] as f64).exp() / z
        };
        fn walk(t: usize, u: usize, time: usize, pieces: &[usize], blank: usize, p: &dyn Fn(usize, usize, usize) -> f64) -> f64 {
            let mut total = 0.0;
            if u < pieces.len() {
                total += p(t, u, pieces[u]) * walk(t, u + 1, time, pieces, blank, p);
            }
            let b = p(t, u, blank);
            if t + 1 < time {
                total += b * walk(t + 1, u, time, pieces, blank, p);
            } else if u == pieces.len() {
                total += b;
            }
            total
        }
        walk(0, 0, time, pieces, blank, &p)
    }

    #[test]
    fn forward_algorithm_matches_enumerating_alignments() {
        let (vocab, blank) = (5usize, 4usize);
        let rows: Vec<f32> = (0..4 * 3 * vocab).map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.3).collect();
        let logits = |t: usize, u: usize| rows[(t * 3 + u) * vocab..][..vocab].to_vec();
        for (time, pieces) in [(1, vec![]), (1, vec![2]), (3, vec![1, 0]), (4, vec![3]), (4, vec![0, 2])] {
            let fast = rnnt_log_likelihood(time, &pieces, blank, |t, u| &rows[(t * 3 + u) * vocab..][..vocab]);
            let slow = brute_force(time, &pieces, blank, &logits).ln();
            assert!((fast as f64 - slow).abs() < 1e-4, "T={time} {pieces:?}: {fast} vs {slow}");
        }
    }

    #[test]
    fn default_streaming_profile_aligns_to_subsampling() {
        let config = UnifiedStreamingConfig::default().validate().unwrap();
        assert_eq!(config.left_context_frames(), 560);
        assert_eq!(config.chunk_frames(), 56);
        assert_eq!(config.right_context_frames(), 56);
        assert_eq!(config.left_context_encoder_frames(), 70);
        assert_eq!(config.chunk_encoder_frames(), 7);
    }
}
