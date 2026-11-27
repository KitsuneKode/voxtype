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
        let mut capture = CpalCapture::new(config)?;
        capture.warm = (config.warm_mic_secs > 0).then(|| self.clone());
        Ok(Box::new(capture))
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

        let warm_window = self
            .warm
            .as_ref()
            .map(|_| Duration::from_secs(self.config.warm_mic_secs.into()));
        let limit = (u64::from(source_sample_rate) * u64::from(self.config.pre_roll_ms) / 1000)
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
            let stream_dead_for_cb = stream_dead.clone();
            let device_label = device_name.clone();
            let err_fn = move |err| {
                tracing::error!("Audio stream error, capture is now dead: {}", err);
                stream_dead_for_cb.store(true, Ordering::SeqCst);
            };

            let mut initial = CaptureBuffer::Closed;
            initial.start(chunk_tx, source_sample_rate, target_sample_rate);
            let buffer = Arc::new(Mutex::new(initial));
            let make_params = || StreamBuildParams {
                buffer: buffer.clone(),
                source_channels,
            };

            let stream_result = match sample_format {
                cpal::SampleFormat::F32 => {
                    build_stream::<f32>(&device, &stream_config, make_params(), err_fn)
                }
                cpal::SampleFormat::I16 => {
                    build_stream::<i16>(&device, &stream_config, make_params(), err_fn)
                }
                cpal::SampleFormat::U16 => {
                    build_stream::<u16>(&device, &stream_config, make_params(), err_fn)
                }
                format => {
                    let _ = ready_tx.send(Err(format!("unsupported sample format {format:?}")));
                    return;
                }
            };

            let stream = match stream_result {
                Ok(s) => s,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("could not open the input stream: {e}")));
                    return;
                }
            };

            if let Err(e) = stream.play() {
                let _ = ready_tx.send(Err(format!("could not start the input stream: {e}")));
                return;
            }
            let _ = ready_tx.send(Ok(()));

            tracing::debug!("Audio capture thread started");

            serve_stream(cmd_rx, warm_window, |cmd| {
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

fn serve_stream<T>(
    commands: std::sync::mpsc::Receiver<T>,
    window: Option<Duration>,
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
            None => commands.recv().ok(),
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
