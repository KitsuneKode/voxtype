//! cpal-based audio capture
//!
//! Uses the cpal crate for cross-platform audio input.
//! Works with PipeWire, PulseAudio, and ALSA backends.
//!
//! Note: cpal::Stream is not Send, so we run the audio capture in a
//! dedicated thread and communicate via channels.

use super::{capture_buffer::CaptureBuffer, AudioCapture};
use crate::config::AudioConfig;
use crate::error::AudioError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

/// How long `start()` waits for the capture thread to confirm the stream is
/// running before returning anyway.
const STREAM_READY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

/// Commands sent to the audio capture thread
enum CaptureCommand {
    Resume(mpsc::Sender<Vec<f32>>, oneshot::Sender<usize>),
    Stop(oneshot::Sender<Vec<f32>>),
    /// Get current samples and clear the buffer (for continuous recording)
    GetSamples(oneshot::Sender<Vec<f32>>),
    /// Copy the take from an index on without draining it (release tail)
    Peek(usize, usize, oneshot::Sender<Option<(usize, Vec<f32>)>>),
}

/// Parameters for building an audio input stream
struct StreamBuildParams {
    buffer: Arc<Mutex<CaptureBuffer>>,
    source_channels: usize,
}

/// The warm sender and the capture buffer stay usable after a panic while
/// they were locked; a poisoned lock must not leave dictation deaf.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Only dictation captures share a warm stream. Dropping its last owner also
/// wakes the capture thread, even when it is waiting on a long deadline.
#[derive(Clone, Default)]
pub(crate) struct DictationMic {
    parked: Arc<Mutex<Option<Parked>>>,
    #[cfg(test)]
    pub(crate) fixtures: Arc<Mutex<std::collections::VecDeque<Box<dyn AudioCapture>>>>,
}

/// A warm stream between takes: its capture thread and the way to reach it.
struct Parked {
    commands: std::sync::mpsc::Sender<CaptureCommand>,
    thread: Option<thread::JoinHandle<()>>,
}

impl DictationMic {
    /// Close a warm stream now rather than at the end of its window, and
    /// wait (at most a second) for its thread to release the device: the
    /// daemon exits without unwinding, which would cut that short.
    pub(crate) async fn close(&self) {
        let Some(Parked { commands, thread }) = lock(&self.parked).take() else {
            return;
        };
        drop(commands);
        if let Some(thread) = thread {
            let joined = tokio::task::spawn_blocking(move || thread.join());
            if tokio::time::timeout(Duration::from_secs(1), joined)
                .await
                .is_err()
            {
                tracing::warn!("The warm microphone did not close within a second");
            }
        }
    }

    pub(crate) fn capture(
        &self,
        config: &AudioConfig,
    ) -> Result<Box<dyn AudioCapture>, AudioError> {
        #[cfg(test)]
        if let Some(capture) = self.fixtures.lock().unwrap().pop_front() {
            return Ok(capture);
        }
        Ok(Box::new(self.cpal(config)?))
    }

    /// A capture that shares this warm stream only when `warm_mic_secs` is on.
    fn cpal(&self, config: &AudioConfig) -> Result<CpalCapture, AudioError> {
        let mut capture = CpalCapture::new(config)?;
        capture.warm = (config.warm_mic_secs > 0).then(|| self.clone());
        Ok(capture)
    }
}

/// cpal-based audio capture implementation
pub struct CpalCapture {
    warm: Option<DictationMic>,
    prefix: usize,
    /// Audio configuration
    config: AudioConfig,
    /// Command sender to the capture thread
    cmd_tx: Option<std::sync::mpsc::Sender<CaptureCommand>>,
    /// Handle to the capture thread
    thread_handle: Option<thread::JoinHandle<()>>,
}

impl CpalCapture {
    /// Create a new cpal audio capture instance
    pub fn new(config: &AudioConfig) -> Result<Self, AudioError> {
        Ok(Self {
            config: config.clone(),
            warm: None,
            prefix: 0,
            cmd_tx: None,
            thread_handle: None,
        })
    }
}

/// Find an audio input device by name with flexible matching.
///
/// Matching strategy (in order):
/// 1. Exact match (case-sensitive)
/// 2. Exact match (case-insensitive)
/// 3. Substring match: device name contains the search term (case-insensitive)
///
/// This allows users to specify either:
/// - Full cpal device names: "alsa_input.pci-0000_00_1f.3.analog-stereo"
/// - PipeWire/PulseAudio short names: "vox_buffer"
/// - Partial device names: "analog-stereo"
fn find_audio_device(host: &cpal::Host, device_name: &str) -> Result<cpal::Device, AudioError> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let search_lower = device_name.to_lowercase();
    let mut devices: Vec<(String, cpal::Device)> = Vec::new();

    // Iterate once and short-circuit immediately on exact match. On Linux ALSA,
    // `host.input_devices()` probes every PCM hint via `snd_pcm_open`, so
    // stopping on the first exact hit (and never re-enumerating a second time)
    // avoids probing unrelated dmix/dsnoop/OSS/HDMI devices.
    for device in host
        .input_devices()
        .map_err(|e| AudioError::Connection(e.to_string()))?
    {
        if let Ok(name) = device.name() {
            if name == device_name {
                tracing::debug!("Found audio device by exact match: {}", name);
                return Ok(device);
            }
            devices.push((name, device));
        }
    }

    // 2. Try exact match (case-insensitive)
    if let Some(pos) = devices
        .iter()
        .position(|(name, _)| name.to_lowercase() == search_lower)
    {
        let (name, device) = devices.swap_remove(pos);
        tracing::debug!(
            "Found audio device by case-insensitive match: {} (searched for: {})",
            name,
            device_name
        );
        return Ok(device);
    }

    // 3. Try substring match (case-insensitive)
    if let Some(pos) = devices
        .iter()
        .position(|(name, _)| name.to_lowercase().contains(&search_lower))
    {
        let (name, device) = devices.swap_remove(pos);
        tracing::debug!(
            "Found audio device by substring match: {} (searched for: {})",
            name,
            device_name
        );
        return Ok(device);
    }

    // No match found - provide helpful error with available devices
    let available = if devices.is_empty() {
        "No audio input devices found.".to_string()
    } else {
        format!(
            "Available devices:\n{}",
            devices
                .iter()
                .map(|(n, _)| format!("  - {}", n))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };

    Err(AudioError::DeviceNotFoundWithList {
        requested: device_name.to_string(),
        available,
    })
}

#[async_trait::async_trait]
impl AudioCapture for CpalCapture {
    async fn start(&mut self) -> Result<mpsc::Receiver<Vec<f32>>, AudioError> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let (chunk_tx, chunk_rx) = mpsc::channel(64);
        if let Some(warm) = &self.warm {
            let parked = lock(&warm.parked).take();
            if let Some(Parked { commands, thread }) = parked {
                let (ready, resumed) = oneshot::channel();
                if commands
                    .send(CaptureCommand::Resume(chunk_tx.clone(), ready))
                    .is_ok()
                {
                    if let Ok(prefix) = resumed.await {
                        self.prefix = prefix;
                        self.cmd_tx = Some(commands);
                        self.thread_handle = thread;
                        return Ok(chunk_rx);
                    }
                }
            }
        }
        self.prefix = 0;
        // Find the device and its config before spawning the capture
        // thread. ALSA blocks forever here when a sound server's socket
        // exists but nothing answers, and this runs on the daemon's event
        // loop, so it gets a deadline and a thread of its own.
        let wanted = self.config.device.clone();
        let lookup = move || -> Result<(cpal::Device, cpal::SupportedStreamConfig), AudioError> {
            let host = cpal::default_host();
            let device = if wanted == "default" {
                host.default_input_device()
                    .ok_or_else(|| AudioError::DeviceNotFound("default".to_string()))?
            } else {
                find_audio_device(&host, &wanted)?
            };
            let config = device
                .default_input_config()
                .map_err(|e| AudioError::Connection(e.to_string()))?;
            Ok((device, config))
        };
        let timeout = super::devices::SOUND_SERVER_TIMEOUT;
        let (device, supported_config) = super::devices::bounded_async(timeout, lookup)
            .await
            .ok_or(AudioError::NoSoundServer(timeout.as_secs()))??;

        let device_name = device.name().unwrap_or_else(|_| "unknown".to_string());
        tracing::info!("Using audio device: {}", device_name);

        let source_sample_rate = supported_config.sample_rate().0;
        let source_channels = supported_config.channels() as usize;
        let target_sample_rate = self.config.sample_rate;
        let sample_format = supported_config.sample_format();

        tracing::debug!(
            "Device config: {} Hz, {} channel(s), format: {:?}",
            source_sample_rate,
            source_channels,
            sample_format
        );

        // Create channels
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<CaptureCommand>();
        // The stream is built on the capture thread. Its outcome comes back
        // here so a device that refuses to open fails `start()` instead of
        // handing back a receiver that never yields a sample.
        let (ready_tx, ready_rx) = oneshot::channel::<Result<(), String>>();

        let warm_window = self.warm.as_ref().map(|_| self.config.warm_mic_window());
        let limit = (u64::from(source_sample_rate) * self.config.pre_roll().as_millis() as u64
            / 1000)
            .try_into()
            .unwrap_or(usize::MAX);
        // Spawn audio capture thread
        let thread_handle = thread::spawn(move || {
            // Build stream config
            let stream_config = cpal::StreamConfig {
                channels: supported_config.channels(),
                sample_rate: supported_config.sample_rate(),
                buffer_size: cpal::BufferSize::Default,
            };

            // A stream error is terminal for that stream: cpal will deliver no
            // further callbacks. Logging alone left capture silently dead
            // until the daemon restarted, which is what a USB or Bluetooth
            // microphone disconnect looks like in practice (#642).
            //
            // The flag is read by `is_stream_dead` so `start()` can rebuild
            // rather than hand back a stream that will never produce a sample.
            let stream_dead = Arc::new(AtomicBool::new(false));
            let device_label = device_name.clone();

            let mut initial = CaptureBuffer::Closed;
            initial.start(chunk_tx, source_sample_rate, target_sample_rate);
            let buffer = Arc::new(Mutex::new(initial));

            let open = || -> Result<cpal::Stream, String> {
                let dead = stream_dead.clone();
                let err_fn = move |err| {
                    tracing::error!("Audio stream error, capture is now dead: {}", err);
                    dead.store(true, Ordering::SeqCst);
                };
                let params = StreamBuildParams {
                    buffer: buffer.clone(),
                    source_channels,
                };
                let stream = match sample_format {
                    cpal::SampleFormat::F32 => {
                        build_stream::<f32>(&device, &stream_config, params, err_fn)
                    }
                    cpal::SampleFormat::I16 => {
                        build_stream::<i16>(&device, &stream_config, params, err_fn)
                    }
                    cpal::SampleFormat::U16 => {
                        build_stream::<u16>(&device, &stream_config, params, err_fn)
                    }
                    format => return Err(format!("unsupported sample format {format:?}")),
                }
                .map_err(|e| format!("could not open the input stream: {e}"))?;
                stream
                    .play()
                    .map_err(|e| format!("could not start the input stream: {e}"))?;
                Ok(stream)
            };

            let mut stream = match open() {
                Ok(stream) => stream,
                Err(message) => {
                    let _ = ready_tx.send(Err(message));
                    return;
                }
            };
            let _ = ready_tx.send(Ok(()));

            tracing::debug!("Audio capture thread started");

            let reopened = std::cell::Cell::new(false);
            let needed = digital_silence_limit(source_sample_rate);
            let reopen_if_silent = || {
                let silence = lock(&buffer).digital_silence();
                if !should_reopen(&reopened, silence, needed) {
                    return;
                }
                match open() {
                    Ok(fresh) => {
                        let old = std::mem::replace(&mut stream, fresh);
                        let _ = old.pause();
                        drop(old);
                        stream_dead.store(false, Ordering::SeqCst);
                        tell_microphone_reopened(&device_label);
                    }
                    Err(e) => tracing::error!(
                        "The microphone gave only digital silence and could not be reopened: {}",
                        e
                    ),
                }
            };

            serve_stream_ticking(cmd_rx, warm_window, SILENCE_POLL, reopen_if_silent, |cmd| {
                let mut buffer = lock(&buffer);
                match cmd {
                    CaptureCommand::Stop(reply) => {
                        let healthy = !stream_dead.load(Ordering::SeqCst);
                        if !healthy {
                            tracing::error!(
                                "Audio device '{}' failed during this recording",
                                device_label
                            );
                        }
                        let warm = warm_window.is_some() && healthy;
                        let _ = reply.send(buffer.stop(warm.then_some(limit)));
                        if warm {
                            StreamState::Warm
                        } else {
                            StreamState::Closed
                        }
                    }
                    CaptureCommand::Resume(tx, reply) => {
                        if stream_dead.load(Ordering::SeqCst) {
                            return StreamState::Closed;
                        }
                        reopened.set(false);
                        let prefix = buffer.start(tx, source_sample_rate, target_sample_rate);
                        if reply.send(prefix).is_ok() {
                            StreamState::Recording
                        } else {
                            StreamState::Closed
                        }
                    }
                    CaptureCommand::GetSamples(reply) => {
                        let _ = reply.send(buffer.drain());
                        StreamState::Recording
                    }
                    CaptureCommand::Peek(from, at_most, reply) => {
                        let _ = reply.send(buffer.captured_since(from, at_most));
                        StreamState::Recording
                    }
                }
            });
            // A macOS disconnect listener can hold its own stream clone.
            // Silence that callback and release the ring before pausing.
            *lock(&buffer) = CaptureBuffer::Closed;
            let _ = stream.pause();
            drop(stream);
            tracing::debug!("Audio capture thread stopped");
        });

        match tokio::time::timeout(STREAM_READY_TIMEOUT, ready_rx).await {
            Ok(Ok(Err(message))) => {
                let _ = thread_handle.join();
                return Err(AudioError::StreamError(message));
            }
            Ok(Err(_)) => {
                let _ = thread_handle.join();
                return Err(AudioError::StreamError(
                    "the capture thread exited before the stream started".to_string(),
                ));
            }
            // A slow backend keeps the old behaviour: carry on and let the
            // stream deliver samples when it comes up.
            Err(_) => tracing::warn!(
                "Audio stream not confirmed after {:?}; continuing",
                STREAM_READY_TIMEOUT
            ),
            Ok(Ok(Ok(()))) => {}
        }

        self.cmd_tx = Some(cmd_tx);
        self.thread_handle = Some(thread_handle);

        Ok(chunk_rx)
    }

    fn pre_roll_samples(&self) -> usize {
        self.prefix
    }

    async fn stop(&mut self) -> Result<Vec<f32>, AudioError> {
        // Send stop command and get samples back
        let samples = if let Some(cmd_tx) = self.cmd_tx.take() {
            let (response_tx, response_rx) = oneshot::channel();

            if cmd_tx.send(CaptureCommand::Stop(response_tx)).is_ok() {
                // Wait for response (with timeout)
                match tokio::time::timeout(std::time::Duration::from_secs(2), response_rx).await {
                    Ok(Ok(samples)) => {
                        if let Some(warm) = &self.warm {
                            *lock(&warm.parked) = Some(Parked {
                                commands: cmd_tx,
                                thread: self.thread_handle.take(),
                            });
                        }
                        samples
                    }
                    Ok(Err(_)) => {
                        return Err(AudioError::StreamError("Channel closed".to_string()))
                    }
                    Err(_) => return Err(AudioError::Timeout(2)),
                }
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Wait for thread to finish
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }

        let duration_secs = samples.len() as f32 / self.config.sample_rate as f32;
        tracing::debug!(
            "Audio capture stopped: {} samples ({:.2}s)",
            samples.len(),
            duration_secs
        );

        if samples.is_empty() {
            return Err(AudioError::EmptyRecording);
        }

        Ok(samples)
    }

    async fn captured_since(&mut self, from: usize, at_most: usize) -> Option<(usize, Vec<f32>)> {
        let cmd_tx = self.cmd_tx.as_ref()?;
        let (reply, response) = oneshot::channel();
        cmd_tx
            .send(CaptureCommand::Peek(from, at_most, reply))
            .ok()?;
        tokio::time::timeout(Duration::from_millis(500), response)
            .await
            .ok()?
            .ok()?
    }

    async fn get_samples(&mut self) -> Vec<f32> {
        // Get current samples without stopping
        if let Some(ref cmd_tx) = self.cmd_tx {
            let (response_tx, response_rx) = oneshot::channel();

            if cmd_tx.send(CaptureCommand::GetSamples(response_tx)).is_ok() {
                // Wait for response (with short timeout)
                match tokio::time::timeout(std::time::Duration::from_millis(500), response_rx).await
                {
                    Ok(Ok(samples)) => return samples,
                    Ok(Err(_)) => {
                        tracing::warn!("get_samples: channel closed");
                    }
                    Err(_) => {
                        tracing::warn!("get_samples: timeout");
                    }
                }
            }
        }
        Vec::new()
    }
}

enum StreamState {
    Recording,
    Warm,
    Closed,
}

/// How long a take may deliver nothing but exact zeros before its stream is
/// reopened. A sound server that stopped feeding a stream (seen after a warm
/// stream was resumed) delivers zeros for the whole take; a noise gate in
/// front of the microphone can too while nobody speaks, so the reopen
/// happens at most once a take and costs only silence.
const DIGITAL_SILENCE: Duration = Duration::from_millis(300);

/// How often a recording take's stream is checked for digital silence.
const SILENCE_POLL: Duration = Duration::from_millis(100);

/// [`DIGITAL_SILENCE`] in device samples.
fn digital_silence_limit(sample_rate: u32) -> usize {
    (u64::from(sample_rate) * DIGITAL_SILENCE.as_millis() as u64 / 1000) as usize
}

/// Whether to reopen the stream now: the take has delivered `silence`
/// samples of exact zeros and no sound, at least `needed`, and has not been
/// reopened yet.
fn should_reopen(reopened: &std::cell::Cell<bool>, silence: Option<usize>, needed: usize) -> bool {
    if reopened.get() || !silence.is_some_and(|zeros| zeros >= needed) {
        return false;
    }
    reopened.set(true);
    true
}

static TOLD_MICROPHONE_REOPENED: AtomicBool = AtomicBool::new(false);

/// Log every reopen; tell the user the first time in this run.
fn tell_microphone_reopened(device: &str) {
    tracing::warn!(
        "Audio device '{}' gave only digital silence for {}ms; reopened it",
        device,
        DIGITAL_SILENCE.as_millis()
    );
    if !TOLD_MICROPHONE_REOPENED.swap(true, Ordering::SeqCst) {
        crate::notification::send_sync(
            "Microphone gave no sound; reopened it",
            "If your first words are missing, say them again.",
        );
    }
}

#[cfg(test)]
fn serve_stream<T>(
    commands: std::sync::mpsc::Receiver<T>,
    window: Option<Duration>,
    handle: impl FnMut(T) -> StreamState,
) {
    serve_stream_ticking(commands, window, Duration::MAX, || {}, handle);
}

/// Serve the capture thread's commands until the stream closes. While a
/// take records, `tick` runs every `poll` without a command.
fn serve_stream_ticking<T>(
    commands: std::sync::mpsc::Receiver<T>,
    window: Option<Duration>,
    poll: Duration,
    mut tick: impl FnMut(),
    mut handle: impl FnMut(T) -> StreamState,
) {
    let mut deadline: Option<Instant> = None;
    loop {
        let command = match deadline {
            Some(end) => {
                let Some(left) = end.checked_duration_since(Instant::now()) else {
                    break;
                };
                commands
                    .recv_timeout(left)
                    .ok()
                    .filter(|_| Instant::now() < end)
            }
            None if poll == Duration::MAX => commands.recv().ok(),
            None => match commands.recv_timeout(poll) {
                Ok(command) => Some(command),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    tick();
                    continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => None,
            },
        };
        let Some(command) = command else { break };
        match handle(command) {
            StreamState::Recording => deadline = None,
            StreamState::Warm => deadline = window.map(|w| Instant::now() + w),
            StreamState::Closed => break,
        }
    }
}

/// Build an input stream for a specific sample type
fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    params: StreamBuildParams,
    err_fn: impl Fn(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream, AudioError>
where
    T: cpal::Sample + cpal::SizedSample + Send + 'static,
    f32: cpal::FromSample<T>,
{
    use cpal::traits::DeviceTrait;

    let StreamBuildParams {
        buffer,
        source_channels,
    } = params;
    let stream = device
        .build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                // Convert to f32 and mix to mono
                let mono_f32: Vec<f32> = data
                    .chunks(source_channels)
                    .map(|frame| {
                        let sum: f32 = frame
                            .iter()
                            .map(|&s| <f32 as cpal::FromSample<T>>::from_sample_(s))
                            .sum();
                        sum / source_channels as f32
                    })
                    .collect();

                if let Ok(mut buffer) = buffer.lock() {
                    buffer.push(mono_f32);
                }
            },
            err_fn,
            None,
        )
        .map_err(|e| AudioError::StreamError(e.to_string()))?;

    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::super::resampler::resample_buffer;

    #[test]
    fn a_poisoned_warm_sender_can_still_be_taken() {
        use super::*;
        let mic = DictationMic::default();
        let parked = mic.parked.clone();
        let _ = thread::spawn(move || {
            let _held = parked.lock().unwrap();
            panic!("a capture that panics while parking");
        })
        .join();
        assert!(mic.parked.is_poisoned());
        let (commands, _rx) = std::sync::mpsc::channel();
        *lock(&mic.parked) = Some(Parked {
            commands,
            thread: None,
        });
        assert!(lock(&mic.parked).take().is_some());
    }

    /// `warm_mic_secs = 0` builds a capture that never parks its stream, so
    /// every take opens the device afresh and carries no pre-roll.
    #[test]
    fn without_a_warm_window_a_capture_never_warms() {
        use super::*;
        let mic = DictationMic::default();
        let mut config = AudioConfig::default();
        assert_eq!(config.warm_mic_secs, 0);
        let cold = mic.cpal(&config).unwrap();
        assert!(cold.warm.is_none());
        assert_eq!(cold.pre_roll_samples(), 0);
        config.warm_mic_secs = 3;
        assert!(mic.cpal(&config).unwrap().warm.is_some());
    }

    /// The 22:19 take: a resumed warm stream delivered nothing but zeros.
    /// Fed through the real buffer, the take is reopened once after 300 ms
    /// of exact zeros, never before, and never again in that take; a take
    /// that delivers any sound first is left alone.
    #[test]
    fn a_take_of_digital_silence_is_reopened_once() {
        use super::super::capture_buffer::CaptureBuffer;
        use super::*;
        let needed = digital_silence_limit(48000);
        assert_eq!(needed, 14400);
        let reopened = std::cell::Cell::new(false);
        let mut buffer = CaptureBuffer::Closed;
        buffer.stop(Some(4800));
        buffer.push(vec![0.2; 4800]);
        let (tx, _rx) = tokio::sync::mpsc::channel(64);
        buffer.start(tx, 48000, 48000);
        for _ in 0..14 {
            buffer.push(vec![0.0; 1000]);
            assert!(!should_reopen(&reopened, buffer.digital_silence(), needed));
        }
        buffer.push(vec![0.0; 1000]);
        assert!(should_reopen(&reopened, buffer.digital_silence(), needed));
        buffer.push(vec![0.0; 48000]);
        assert!(
            !should_reopen(&reopened, buffer.digital_silence(), needed),
            "once a take"
        );

        reopened.set(false);
        let (tx, _rx) = tokio::sync::mpsc::channel(64);
        buffer.start(tx, 48000, 48000);
        buffer.push(vec![0.0; 1000]);
        buffer.push(vec![0.01; 10]);
        buffer.push(vec![0.0; 48000]);
        assert!(
            !should_reopen(&reopened, buffer.digital_silence(), needed),
            "a pause after speech"
        );
    }

    /// The silence check runs while a take records, never while the
    /// stream is parked warm.
    #[test]
    fn the_silence_check_runs_only_while_recording() {
        use super::*;
        let (commands, rx) = std::sync::mpsc::channel::<bool>();
        let ticks = Arc::new(Mutex::new(Vec::new()));
        let recording = Arc::new(AtomicBool::new(true));
        let (t, r) = (ticks.clone(), recording.clone());
        let thread = thread::spawn(move || {
            serve_stream_ticking(
                rx,
                Some(Duration::from_secs(3600)),
                Duration::from_millis(5),
                || t.lock().unwrap().push(r.load(Ordering::SeqCst)),
                |warm: bool| {
                    r.store(!warm, Ordering::SeqCst);
                    if warm {
                        StreamState::Warm
                    } else {
                        StreamState::Recording
                    }
                },
            )
        });
        thread::sleep(Duration::from_millis(40));
        commands.send(true).unwrap();
        thread::sleep(Duration::from_millis(40));
        let parked = ticks.lock().unwrap().len();
        thread::sleep(Duration::from_millis(40));
        assert_eq!(ticks.lock().unwrap().len(), parked, "no checks while warm");
        commands.send(false).unwrap();
        thread::sleep(Duration::from_millis(40));
        drop(commands);
        thread.join().unwrap();
        let ticks = ticks.lock().unwrap();
        assert!(ticks.len() > parked, "checks again after the resume");
        assert!(ticks.iter().all(|&recording| recording));
    }

    /// Shutdown closes a warm stream and waits for its thread, instead of
    /// leaving it to the end of its window.
    #[tokio::test]
    async fn closing_the_mic_ends_a_warm_stream_and_waits_for_it() {
        use super::*;
        let mic = DictationMic::default();
        mic.close().await;
        let (commands, rx) = std::sync::mpsc::channel::<CaptureCommand>();
        let released = Arc::new(AtomicBool::new(false));
        let flag = released.clone();
        let thread = thread::spawn(move || {
            serve_stream(rx, Some(Duration::from_secs(3600)), |_| StreamState::Warm);
            thread::sleep(Duration::from_millis(50));
            flag.store(true, Ordering::SeqCst);
        });
        *lock(&mic.parked) = Some(Parked {
            commands,
            thread: Some(thread),
        });
        mic.close().await;
        assert!(
            released.load(Ordering::SeqCst),
            "joined before close returned"
        );
        assert!(lock(&mic.parked).is_none());
    }

    #[test]
    fn warm_window_closes_without_another_command() {
        use super::*;
        let (tx, rx) = std::sync::mpsc::channel();
        let (closed, done) = std::sync::mpsc::channel();
        let window = Duration::from_millis(30);
        let thread = thread::spawn(move || {
            serve_stream(rx, Some(window), |()| StreamState::Warm);
            closed.send(()).unwrap();
        });
        let started = Instant::now();
        tx.send(()).unwrap();
        done.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(started.elapsed() >= window);
        assert!(tx.send(()).is_err());
        thread.join().unwrap();
    }

    #[test]
    fn warm_window_reuses_within_deadline_and_clears_buffer_on_expiry() {
        use super::*;
        let buffer = Arc::new(Mutex::new(CaptureBuffer::Closed));
        let buf_thread = buffer.clone();
        let (tx, rx) = std::sync::mpsc::channel::<CaptureCommand>();
        let (closed, done) = std::sync::mpsc::channel();
        let window = Duration::from_millis(60);
        let thread = thread::spawn(move || {
            serve_stream(rx, Some(window), |cmd| {
                let mut b = buf_thread.lock().unwrap();
                match cmd {
                    CaptureCommand::Stop(reply) => {
                        let _ = reply.send(b.stop(Some(3)));
                        StreamState::Warm
                    }
                    CaptureCommand::Resume(chunk_tx, reply) => {
                        let prefix = b.start(chunk_tx, 16000, 16000);
                        let _ = reply.send(prefix);
                        StreamState::Recording
                    }
                    CaptureCommand::GetSamples(reply) => {
                        let _ = reply.send(b.drain());
                        StreamState::Recording
                    }
                    CaptureCommand::Peek(from, at_most, reply) => {
                        let _ = reply.send(b.captured_since(from, at_most));
                        StreamState::Recording
                    }
                }
            });
            *buf_thread.lock().unwrap() = CaptureBuffer::Closed;
            closed.send(()).unwrap();
        });

        // Take 1 finishes and enters the warm window.
        let (stop_tx, stop_rx) = oneshot::channel();
        tx.send(CaptureCommand::Stop(stop_tx)).unwrap();
        let _ = stop_rx.blocking_recv().unwrap();

        // Ambient audio arrives while warm; only the last 3 samples survive.
        buffer.lock().unwrap().push(vec![1., 2., 3., 4., 5.]);

        // Take 2 resumes inside the warm window and receives the bounded pre-roll.
        let (chunk_tx, mut chunk_rx) = mpsc::channel(4);
        let (ready_tx, ready_rx) = oneshot::channel();
        tx.send(CaptureCommand::Resume(chunk_tx, ready_tx)).unwrap();
        assert_eq!(ready_rx.blocking_recv().unwrap(), 3);
        buffer.lock().unwrap().push(vec![6., 7.]);
        assert_eq!(chunk_rx.try_recv().unwrap(), vec![3., 4., 5.]);
        assert_eq!(chunk_rx.try_recv().unwrap(), vec![6., 7.]);

        // Take 2 stops and enters a fresh warm window; ambient audio arrives again.
        let (stop_tx, stop_rx) = oneshot::channel();
        tx.send(CaptureCommand::Stop(stop_tx)).unwrap();
        assert_eq!(stop_rx.blocking_recv().unwrap(), vec![3., 4., 5., 6., 7.]);
        buffer.lock().unwrap().push(vec![8., 9., 10.]);

        // Without another take, the warm window expires, closes the thread, and clears the ring.
        done.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(matches!(*buffer.lock().unwrap(), CaptureBuffer::Closed));
        let (probe_tx, _) = mpsc::channel(1);
        assert_eq!(buffer.lock().unwrap().start(probe_tx, 16000, 16000), 0);
        thread.join().unwrap();
    }
    // These moved off the removed linear-interpolation `resample()` and onto
    // the band-limited path (#641). They cover very short inputs, which is a
    // real case: a recording shorter than one FFT chunk still has to come out
    // the other side.

    #[test]
    fn test_resample_same_rate() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let result = resample_buffer(&samples, 16000, 16000);
        assert_eq!(result, samples);
    }

    #[test]
    fn test_resample_downsample() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = resample_buffer(&samples, 48000, 16000);
        // 48000 -> 16000 is 3:1, so 8 samples land around 3.
        assert!(
            result.len() >= 2 && result.len() <= 4,
            "expected about 3 samples, got {}",
            result.len()
        );
    }

    #[test]
    fn test_resample_upsample() {
        let samples = vec![1.0, 2.0];
        let result = resample_buffer(&samples, 8000, 16000);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_resample_empty() {
        let samples: Vec<f32> = vec![];
        assert!(resample_buffer(&samples, 48000, 16000).is_empty());
    }
}
