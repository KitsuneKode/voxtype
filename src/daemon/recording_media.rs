use crate::{audio, config::AudioConfig};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, OwnedMutexGuard};

#[derive(Clone, Default)]
pub(super) struct Media(Arc<Mutex<MediaState>>);

#[derive(Default)]
struct MediaState {
    paused_media_players: Vec<String>,
    ducked_media_streams: Vec<audio::media::DuckedMediaStream>,
    media_fade_task: Option<tokio::task::JoinHandle<()>>,
    resume: Option<tokio::task::JoinHandle<()>>,
    config: AudioConfig,
}

pub(super) struct Lease(OwnedMutexGuard<MediaState>);
impl Drop for Lease {
    fn drop(&mut self) {
        self.0.restore_ducked_media_streams();
        self.0.resume_media_players();
    }
}

impl Media {
    pub async fn acquire(&self, config: &AudioConfig) -> Option<Lease> {
        if !config.pause_media && !config.duck_media {
            return None;
        }
        let mut state = self.0.clone().lock_owned().await;
        state.config = config.clone();
        if let Some(task) = state.resume.take() {
            let _ = task.await;
        }
        let mut lease = Lease(state);
        lease.0.pause_media_players().await;
        lease.0.duck_media_streams().await;
        Some(lease)
    }
    pub async fn settle(&self) {
        let _ = tokio::time::timeout(Duration::from_secs(2), async {
            let mut state = self.0.lock().await;
            if let Some(task) = state.resume.take() {
                let _ = task.await;
            }
            if let Some(task) = state.media_fade_task.take() {
                let _ = task.await;
            }
        })
        .await;
    }
}

impl MediaState {
    /// Pause MPRIS media players if configured, storing which ones were paused
    async fn pause_media_players(&mut self) {
        if self.config.pause_media {
            self.paused_media_players =
                audio::media::pause_playing_players(&self.config.pause_media_ignored_players).await;
        }
    }

    /// Duck active audio streams if configured, storing original volumes
    async fn duck_media_streams(&mut self) {
        if self.config.duck_media {
            // Wait out any restore still fading up. Its final write is what
            // puts the streams back at their true original volumes, and
            // enumerating before that lands would capture intermediate values
            // as the new originals — every fast toggle cycle would then store
            // a quieter baseline and media would drift down permanently.
            // Normally already finished, so this costs nothing.
            if let Some(task) = self.media_fade_task.take() {
                let _ = task.await;
            }
            let paused_apps = audio::media::player_apps(&self.paused_media_players).await;
            let (streams, fade) = audio::media::duck_playing_audio(
                self.config.duck_media_volume_percent,
                self.config.duck_media_fade_ms,
                &paused_apps,
            )
            .await;
            self.ducked_media_streams = streams;
            self.media_fade_task = fade;
        }
    }

    /// Restore any audio streams that were ducked at recording start
    fn restore_ducked_media_streams(&mut self) {
        if !self.ducked_media_streams.is_empty() {
            // Abort rather than await a fade still on its way down: this path
            // is synchronous, and the restore we are about to spawn ends by
            // writing the stored originals, so an interrupted duck ramp is
            // corrected either way.
            if let Some(task) = self.media_fade_task.take() {
                task.abort();
            }
            let streams = std::mem::take(&mut self.ducked_media_streams);
            self.media_fade_task = Some(tokio::spawn(audio::media::restore_ducked_audio(
                streams,
                self.config.duck_media_volume_percent,
                self.config.duck_media_fade_ms,
            )));
        }
    }

    /// Resume any MPRIS media players that were paused at recording start
    fn resume_media_players(&mut self) {
        if !self.paused_media_players.is_empty() {
            let players = std::mem::take(&mut self.paused_media_players);
            self.resume = Some(tokio::spawn(audio::media::resume_players(players)));
        }
    }
}
