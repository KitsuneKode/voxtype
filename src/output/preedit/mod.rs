//! Live dictation preedit in the focused text field (`[output] preedit`).
//!
//! While a streaming take records, voxtype binds as the seat's Wayland input
//! method (`zwp_input_method_v2`) and shows the words heard so far as
//! uncommitted, underlined preedit text in whatever text-input-v3 field has
//! focus. At the end the final text replaces the preedit in one
//! `commit_string`, with no synthesized keys, so the keyboard layout does not
//! matter. Fields without text-input-v3 (XWayland, terminals and browsers
//! without IME support) never activate the input method, and the take falls
//! back to the normal output chain.
//!
//! The binding lasts only for the take ([`PreeditHandle::begin`] /
//! [`PreeditHandle::end`]), and the keyboard is never grabbed. The Wayland
//! connection lives on its own thread ([`wayland`]); [`state::Ime`] decides
//! what to send.

pub mod state;
#[cfg(all(target_os = "linux", feature = "preedit"))]
mod wayland;

pub use state::CommitOutcome;

use state::{Event, Ime, Request};

use super::TextOutput;
use crate::error::OutputError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

/// `$XDG_RUNTIME_DIR/voxtype/field_text.json`: the focused field's text
/// around the cursor while voxtype holds the input method, for Shuush's
/// edit learning and spacing. Gone whenever no readable field is bound.
fn field_path() -> std::path::PathBuf {
    crate::config::Config::runtime_dir().join("field_text.json")
}

/// After a take, the binding stays on the same field this long so the field's
/// text can be read as the user edits it (see [`Driver::field_update`]);
/// focus leaving the field or the next take ends it sooner.
pub const LINGER: Duration = Duration::from_secs(120);

/// A command for the backend thread.
pub enum Cmd {
    /// Bind the input method for a take.
    Begin(Option<super::preview::TakePreview>),
    Set(String),
    /// Remove the preedit; the ack fires once the compositor has it.
    Clear(Option<oneshot::Sender<()>>),
    /// Commit text in place of the preedit; replies after the compositor has it.
    Commit(String, oneshot::Sender<CommitOutcome>),
    /// Clear the preedit; the binding stays on the field for [`LINGER`].
    End,
    /// Clear the preedit and release the input method now (a cancelled take).
    Release,
    /// Change nothing; the ack fires once everything sent before has reached
    /// the compositor.
    Flush(oneshot::Sender<()>),
}

/// One running backend: a thread that owns the input method binding.
pub struct Backend {
    tx: Sender<Cmd>,
    wake: Arc<dyn Fn() + Send + Sync>,
    shared: Arc<Shared>,
}

impl Backend {
    pub fn new(tx: Sender<Cmd>, wake: Arc<dyn Fn() + Send + Sync>, shared: Arc<Shared>) -> Self {
        Self { tx, wake, shared }
    }

    fn send(&self, cmd: Cmd) -> bool {
        if self.tx.send(cmd).is_err() {
            return false;
        }
        (self.wake)();
        true
    }
}

/// State the backend thread publishes for the daemon to read without a round trip.
#[derive(Default)]
pub struct Shared {
    /// A text-input-v3 field is focused and has activated the input method.
    pub active: AtomicBool,
    /// The backend thread is running.
    pub alive: AtomicBool,
}

/// Why a backend could not start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpawnError {
    /// The compositor lacks input-method-v2: never try again.
    Unsupported,
    /// Anything else (no Wayland connection yet): try again next take.
    Failed,
}

type Spawner = dyn Fn() -> Result<Backend, SpawnError> + Send + Sync;

/// The daemon's side of the preedit: cheap to clone, every call is
/// non-blocking except [`commit`](Self::commit) and [`clear_now`](Self::clear_now).
#[derive(Clone)]
pub struct PreeditHandle {
    inner: Arc<Inner>,
}

struct Inner {
    spawn: Box<Spawner>,
    backend: Mutex<Option<Backend>>,
    /// Set once the compositor turned out not to support the protocol.
    disabled: AtomicBool,
}

impl PreeditHandle {
    /// The Wayland input-method backend, or `None` when this build lacks the
    /// `preedit` feature.
    pub fn wayland() -> Option<Self> {
        #[cfg(all(target_os = "linux", feature = "preedit"))]
        {
            Some(Self::with_spawner(Box::new(wayland::spawn)))
        }
        #[cfg(not(all(target_os = "linux", feature = "preedit")))]
        {
            tracing::warn!(
                "[output] preedit is on, but this build lacks the `preedit` feature; ignoring it"
            );
            None
        }
    }

    pub fn with_spawner(spawn: Box<Spawner>) -> Self {
        Self {
            inner: Arc::new(Inner {
                spawn,
                backend: Mutex::new(None),
                disabled: AtomicBool::new(false),
            }),
        }
    }

    /// Bind the input method for a take, starting the backend thread if
    /// needed. A compositor without input-method-v2 disables preedit for the
    /// rest of the daemon's life (the backend logs why, once).
    pub fn begin(&self) {
        self.begin_with_preview(None);
    }

    pub fn begin_with_preview(&self, preview: Option<super::preview::TakePreview>) {
        if self.inner.disabled.load(Ordering::Relaxed) {
            return;
        }
        let mut backend = self.inner.backend.lock().unwrap_or_else(|e| e.into_inner());
        let running = backend
            .as_ref()
            .is_some_and(|b| b.shared.alive.load(Ordering::Acquire));
        if !running {
            match (self.inner.spawn)() {
                Ok(b) => *backend = Some(b),
                Err(e) => {
                    *backend = None;
                    if e == SpawnError::Unsupported {
                        self.inner.disabled.store(true, Ordering::Relaxed);
                    }
                    return;
                }
            }
        }
        if let Some(b) = backend.as_ref() {
            b.send(Cmd::Begin(preview));
        }
    }

    pub fn set(&self, text: &str) {
        self.send(Cmd::Set(text.to_string()));
    }

    pub fn clear(&self) {
        self.send(Cmd::Clear(None));
    }

    /// Clear the preedit and wait until the compositor has the request, so
    /// keys typed next cannot land inside it.
    pub async fn clear_now(&self) {
        let (tx, rx) = oneshot::channel();
        if self.send(Cmd::Clear(Some(tx))) {
            let _ = rx.await;
        }
    }

    /// Replace the preedit with `text`, inserted into the focused field.
    pub async fn commit(&self, text: &str) -> CommitOutcome {
        let (tx, rx) = oneshot::channel();
        if !self.send(Cmd::Commit(text.to_string(), tx)) {
            return CommitOutcome::Unavailable;
        }
        rx.await.unwrap_or(CommitOutcome::Unavailable)
    }

    /// Clear the preedit and release the input method.
    pub fn end(&self) {
        self.send(Cmd::End);
    }

    /// Clear the preedit and release the input method right away.
    pub fn release(&self) {
        self.send(Cmd::Release);
    }

    /// Release the input method and remove the field-text file before
    /// returning (shutdown: the process exits before the Wayland thread would).
    pub async fn release_now(&self) {
        self.release();
        self.flush().await;
        let _ = std::fs::remove_file(field_path());
    }

    /// Wait until every command sent so far has reached the compositor.
    pub async fn flush(&self) {
        let (tx, rx) = oneshot::channel();
        if self.send(Cmd::Flush(tx)) {
            let _ = rx.await;
        }
    }

    /// Whether the focused field currently listens to the input method.
    pub fn text_input_active(&self) -> bool {
        let backend = self.inner.backend.lock().unwrap_or_else(|e| e.into_inner());
        backend
            .as_ref()
            .is_some_and(|b| b.shared.active.load(Ordering::Acquire))
    }

    fn send(&self, cmd: Cmd) -> bool {
        let backend = self.inner.backend.lock().unwrap_or_else(|e| e.into_inner());
        backend.as_ref().is_some_and(|b| b.send(cmd))
    }
}

/// What the backend thread must do after [`Driver::command`] returns.
pub enum Ack {
    None,
    /// Send pending requests, wait for the compositor, then reply.
    Cleared(oneshot::Sender<()>),
    Committed(oneshot::Sender<CommitOutcome>, CommitOutcome),
}

/// What a backend does with the input method object; the real one talks
/// Wayland, the test one records.
pub trait Sink {
    /// Create the input method object. False when it cannot be created.
    fn bind(&mut self) -> bool;
    /// Destroy the input method object, if any.
    fn unbind(&mut self);
    fn send(&mut self, requests: Vec<Request>);
}

/// Applies commands and compositor events to the current binding. Shared by
/// the Wayland thread and the tests' fake backend.
pub struct Driver {
    preview: Option<super::preview::TakePreview>,
    ime: Option<Ime>,
    shared: Arc<Shared>,
    /// Set while bound after a take ended (no preedit, field reading only).
    linger_until: Option<Instant>,
    /// The field text last handed out by [`field_update`](Self::field_update).
    field: Option<(String, String)>,
    field_dirty: bool,
    /// When to next ask the field for its text (see [`Ime::ask_field`]).
    next_ask: Option<Instant>,
}

/// How often a bound field is asked for its text, so edits made after a
/// take are seen by a field that reports only when asked.
const FIELD_ASK: Duration = Duration::from_millis(500);

impl Driver {
    pub fn new(shared: Arc<Shared>) -> Self {
        Self {
            preview: None,
            ime: None,
            shared,
            linger_until: None,
            field: None,
            field_dirty: false,
            next_ask: None,
        }
    }

    /// When the Wayland thread must wake for [`tick`](Self::tick).
    pub fn wake_deadline(&self) -> Option<Instant> {
        [self.linger_until, self.next_ask]
            .into_iter()
            .flatten()
            .min()
    }

    /// When the binding kept after a take must be released.
    pub fn linger_deadline(&self) -> Option<Instant> {
        self.linger_until
    }

    /// Release a binding kept after a take once its time is up.
    pub fn tick(&mut self, now: Instant, sink: &mut dyn Sink) {
        if self.linger_until.is_some_and(|t| now >= t) {
            self.release(sink);
            self.publish();
        }
        match self.ime.as_ref().filter(|ime| ime.active()) {
            Some(ime) => {
                if self.next_ask.is_none_or(|t| now >= t) {
                    sink.send(ime.ask_field());
                    self.next_ask = Some(now + FIELD_ASK);
                }
            }
            None => self.next_ask = None,
        }
    }

    /// The focused field's (before, after) cursor text when it changed since
    /// the last call: `Some(None)` once it is gone (focus left, released,
    /// password field).
    pub fn field_update(&mut self) -> Option<Option<(String, String)>> {
        std::mem::take(&mut self.field_dirty).then(|| self.field.clone())
    }

    fn release(&mut self, sink: &mut dyn Sink) {
        self.linger_until = None;
        if let Some(mut ime) = self.ime.take() {
            sink.send(ime.clear());
            sink.unbind();
        }
    }

    pub fn command(&mut self, cmd: Cmd, sink: &mut dyn Sink) -> Ack {
        let ack = match cmd {
            Cmd::Begin(preview) => {
                self.preview = preview;
                let lingering = self.linger_until.take().is_some();
                match self.ime.as_mut() {
                    // Still on the field the last take went into: keep it.
                    Some(ime) if lingering && ime.active() => sink.send(ime.clear()),
                    // A take that never ended, or a binding with no field.
                    _ => {
                        self.release(sink);
                        if sink.bind() {
                            self.ime = Some(Ime::new());
                        }
                    }
                }
                Ack::None
            }
            Cmd::Set(text) => {
                if let Some(ime) = self.ime.as_mut() {
                    sink.send(ime.set(&text));
                }
                Ack::None
            }
            Cmd::Clear(ack) => {
                if let Some(ime) = self.ime.as_mut() {
                    sink.send(ime.clear());
                }
                ack.map_or(Ack::None, Ack::Cleared)
            }
            Cmd::Commit(text, reply) => {
                let outcome = match self.ime.as_mut() {
                    Some(ime) => {
                        let (outcome, requests) = ime.commit(&text);
                        sink.send(requests);
                        outcome
                    }
                    None => CommitOutcome::Unavailable,
                };
                Ack::Committed(reply, outcome)
            }
            Cmd::End => {
                match self.ime.as_mut() {
                    // Keep reading the field the text went into.
                    Some(ime) if ime.active() && self.linger_until.is_none() => {
                        sink.send(ime.clear());
                        self.linger_until = Some(Instant::now() + LINGER);
                    }
                    _ => self.release(sink),
                }
                Ack::None
            }
            Cmd::Release => {
                self.release(sink);
                Ack::None
            }
            Cmd::Flush(ack) => Ack::Cleared(ack),
        };
        self.publish();
        ack
    }

    pub fn event(&mut self, event: Event, sink: &mut dyn Sink) {
        let Some(ime) = self.ime.as_mut() else {
            return;
        };
        match &event {
            // The field's own text never reaches a log.
            Event::Surrounding { text, cursor } => {
                tracing::debug!(
                    len = text.len(),
                    cursor,
                    "Input method event: surrounding text"
                )
            }
            other => tracing::debug!(event = ?other, "Input method event"),
        }
        let unavailable = event == Event::Unavailable;
        sink.send(ime.on_event(event));
        if unavailable {
            tracing::warn!(
                "Another input method holds the seat; dictation preedit is off for this take"
            );
            sink.unbind();
            self.ime = None;
            self.linger_until = None;
        } else if self.linger_until.is_some() && !self.ime.as_ref().is_some_and(Ime::active) {
            // Focus left the field the text went into.
            self.release(sink);
        }
        self.publish();
    }

    /// Release everything (the daemon went away).
    pub fn shutdown(&mut self, sink: &mut dyn Sink) {
        self.release(sink);
        self.publish();
    }

    fn publish(&mut self) {
        let active = self.ime.as_ref().is_some_and(Ime::active);
        self.shared.active.store(active, Ordering::Release);
        if let Some(preview) = &self.preview {
            preview.field_changed(self.ime.as_ref().is_some_and(Ime::sensitive));
        }
        let field = self
            .ime
            .as_ref()
            .and_then(Ime::field)
            .map(|(b, a)| (b.to_string(), a.to_string()));
        if field != self.field {
            self.field = field;
            self.field_dirty = true;
        }
    }
}

/// First driver of the final-delivery chain while a preedit is showing:
/// commits the text through the input method, or clears the preedit and
/// hands the text to the next driver when the input method cannot express it.
pub struct PreeditOutput {
    handle: PreeditHandle,
    auto_submit: bool,
    append_text: Option<String>,
}

impl PreeditOutput {
    pub fn new(handle: PreeditHandle, config: &crate::config::OutputConfig) -> Self {
        Self {
            handle,
            auto_submit: config.auto_submit,
            append_text: config.append_text.clone(),
        }
    }
}

/// Text the input method commits as is. Newlines and other control
/// characters mean key presses to the typing drivers (Enter, Shift+Enter
/// with `shift_enter_newlines`, Tab), which a commit cannot reproduce.
fn committable(text: &str) -> bool {
    !text.chars().any(char::is_control)
}

#[async_trait::async_trait]
impl TextOutput for PreeditOutput {
    async fn output(&self, text: &str) -> Result<(), OutputError> {
        let mut full = text.to_string();
        if let Some(append) = &self.append_text {
            full.push_str(append);
        }
        if !committable(&full) {
            self.handle.clear_now().await;
            return Err(OutputError::InjectionFailed(
                "text has newlines or control keys; typing it instead".to_string(),
            ));
        }
        // Several commits for one text can each need a newer serial, and a
        // stale one is dropped silently: longer text goes the normal way.
        if full.len() > state::MAX_STRING_BYTES {
            self.handle.clear_now().await;
            return Err(OutputError::InjectionFailed(
                "text too long for one input method commit; typing it instead".to_string(),
            ));
        }
        match self.handle.commit(&full).await {
            CommitOutcome::Committed => {}
            outcome => {
                self.handle.clear_now().await;
                return Err(OutputError::InjectionFailed(format!(
                    "input method commit not possible ({outcome:?})"
                )));
            }
        }
        // The text is in the field now: never report failure past this
        // point, or the next driver would type it a second time.
        if self.auto_submit {
            if let Err(e) = super::wtype::send_enter_key().await {
                tracing::warn!(
                    "Auto-submit Enter after the input method commit failed: {}",
                    e
                );
            }
        }
        Ok(())
    }

    /// A text with a line break: the part before it is committed in place of
    /// the preedit, in the same commit that removes the preedit, and the next
    /// method types or pastes the rest. Clearing the preedit first and
    /// handing over the whole text left the field empty until the paste
    /// landed (about 200 ms in an Electron composer).
    async fn output_prefix(&self, text: &str) -> Result<usize, OutputError> {
        let mut full = text.to_string();
        if let Some(append) = &self.append_text {
            full.push_str(append);
        }
        let cut = text.find(char::is_control).unwrap_or(text.len());
        if committable(&full) || cut == 0 || cut == text.len() || cut > state::MAX_STRING_BYTES {
            return self.output(text).await.map(|()| text.len());
        }
        match self.handle.commit(&text[..cut]).await {
            CommitOutcome::Committed => Ok(cut),
            outcome => {
                self.handle.clear_now().await;
                Err(OutputError::InjectionFailed(format!(
                    "input method commit not possible ({outcome:?})"
                )))
            }
        }
    }

    async fn is_available(&self) -> bool {
        true
    }

    fn inserts_directly(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        // Auto-submit presses Enter, so it counts as a keystroke method and
        // is skipped like wtype when modifiers stay held.
        if self.auto_submit {
            "preedit+submit"
        } else {
            "preedit"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::{output_with_fallback, OutputOptions};

    /// The fake backend: a thread running the real [`Driver`] against a
    /// recording sink. Tests inject compositor events through `driver`.
    struct Fake {
        handle: PreeditHandle,
        driver: Arc<Mutex<(Driver, RecSink)>>,
        spawns: Arc<Mutex<usize>>,
    }

    #[derive(Default)]
    struct RecSink {
        bound: bool,
        bindable: bool,
        log: Vec<String>,
    }

    impl Sink for RecSink {
        fn bind(&mut self) -> bool {
            self.bound = self.bindable;
            self.log.push(format!("bind:{}", self.bound));
            self.bound
        }
        fn unbind(&mut self) {
            self.bound = false;
            self.log.push("unbind".to_string());
        }
        fn send(&mut self, requests: Vec<Request>) {
            for r in requests {
                self.log.push(match r {
                    Request::Preedit(t) => format!("preedit:{t}"),
                    Request::CommitString(t) => format!("commit_string:{t}"),
                    Request::DeleteBefore(n) => format!("delete_before:{n}"),
                    Request::Commit(s) => format!("commit:{s}"),
                });
            }
        }
    }

    impl Fake {
        fn new(protocol: bool, bindable: bool) -> Self {
            let shared = Arc::new(Shared::default());
            let driver = Arc::new(Mutex::new((
                Driver::new(shared.clone()),
                RecSink {
                    bindable,
                    ..RecSink::default()
                },
            )));
            let spawns = Arc::new(Mutex::new(0));
            let (d, s) = (driver.clone(), spawns.clone());
            let handle = PreeditHandle::with_spawner(Box::new(move || {
                *s.lock().unwrap() += 1;
                if !protocol {
                    return Err(SpawnError::Unsupported);
                }
                let (tx, rx) = std::sync::mpsc::channel::<Cmd>();
                let shared = shared.clone();
                shared.alive.store(true, Ordering::Release);
                let d = d.clone();
                std::thread::spawn(move || {
                    while let Ok(cmd) = rx.recv() {
                        let mut guard = d.lock().unwrap();
                        let (driver, sink) = &mut *guard;
                        match driver.command(cmd, sink) {
                            Ack::None => {}
                            Ack::Cleared(tx) => {
                                let _ = tx.send(());
                            }
                            Ack::Committed(tx, outcome) => {
                                let _ = tx.send(outcome);
                            }
                        }
                    }
                });
                Ok(Backend::new(tx, Arc::new(|| {}), shared))
            }));
            Self {
                handle,
                driver,
                spawns,
            }
        }

        fn event(&self, event: Event) {
            let mut guard = self.driver.lock().unwrap();
            let (driver, sink) = &mut *guard;
            driver.event(event, sink);
        }

        fn focus(&self) {
            self.event(Event::Activate);
            self.event(Event::Done);
        }

        fn unfocus(&self) {
            self.event(Event::Deactivate);
            self.event(Event::Done);
        }

        /// Wait for the backend to drain the commands sent so far.
        async fn settle(&self) {
            self.handle.flush().await;
        }

        fn log(&self) -> Vec<String> {
            std::mem::take(&mut self.driver.lock().unwrap().1.log)
        }
    }

    /// Stands in for wtype: records what it typed.
    struct Typist(Arc<Mutex<Vec<String>>>);

    #[async_trait::async_trait]
    impl TextOutput for Typist {
        async fn output(&self, text: &str) -> Result<(), OutputError> {
            self.0.lock().unwrap().push(text.to_string());
            Ok(())
        }
        async fn is_available(&self) -> bool {
            true
        }
        fn name(&self) -> &'static str {
            "wtype"
        }
    }

    fn options() -> OutputOptions<'static> {
        OutputOptions {
            history_context: None,
            pre_output_command: None,
            post_output_command: None,
            wait_for_modifier_release: false,
            modifier_release_timeout: std::time::Duration::ZERO,
        }
    }

    async fn deliver(fake: &Fake, config: &crate::config::OutputConfig, text: &str) -> Vec<String> {
        let typed = Arc::new(Mutex::new(Vec::new()));
        let chain: Vec<Box<dyn TextOutput>> = vec![
            Box::new(PreeditOutput::new(fake.handle.clone(), config)),
            Box::new(Typist(typed.clone())),
        ];
        output_with_fallback(&chain, text, options()).await.unwrap();
        let typed = typed.lock().unwrap().clone();
        typed
    }

    #[tokio::test]
    async fn a_password_field_hides_the_takes_preview_and_says_so() {
        let dir = tempfile::tempdir().unwrap();
        let live = dir.path().join("live_text");
        let marker = dir.path().join("sensitive_field");
        let preview = crate::output::preview::TakePreview::new(
            Some(live.clone()),
            dir.path(),
            Some(dir.path().join("pp_private")),
            false,
        );
        let fake = Fake::new(true, true);
        fake.handle.begin_with_preview(Some(preview.clone()));
        fake.settle().await;
        fake.focus();
        preview.update("my password is");
        assert!(!marker.exists(), "an ordinary field was called sensitive");
        assert_eq!(std::fs::read_to_string(&live).unwrap(), "my password is");

        fake.event(Event::ContentHidden(true));
        fake.event(Event::Done);
        assert!(marker.exists(), "the password field was not published");
        assert_eq!(std::fs::read_to_string(&live).unwrap(), "");
        assert!(preview.suppressed());

        fake.unfocus();
        assert!(!marker.exists(), "the marker outlived the password field");
        preview.clear();
    }

    #[tokio::test]
    async fn a_take_shows_partials_then_commits_the_final_text() {
        let fake = Fake::new(true, true);
        fake.handle.begin();
        fake.settle().await;
        fake.focus();
        fake.handle.set("hello");
        fake.handle.set("hello wold");
        fake.settle().await;
        assert!(fake.handle.text_input_active());

        let typed = deliver(&fake, &Default::default(), "Hello, world.").await;
        assert!(typed.is_empty(), "nothing typed when the commit lands");
        fake.handle.end();
        fake.settle().await;
        assert_eq!(
            fake.log(),
            [
                "bind:true",
                "preedit:hello",
                "commit:1",
                "preedit:hello wold",
                "commit:1",
                "preedit:",
                "commit_string:Hello, world.",
                "commit:1",
            ]
        );
        // The binding stays on the field until focus leaves it.
        fake.unfocus();
        assert_eq!(fake.log(), ["unbind"]);
    }

    #[tokio::test]
    async fn append_text_rides_along_in_the_commit() {
        let fake = Fake::new(true, true);
        fake.handle.begin();
        fake.settle().await;
        fake.focus();
        let config = crate::config::OutputConfig {
            append_text: Some(" ".to_string()),
            ..Default::default()
        };
        assert!(deliver(&fake, &config, "Hi.").await.is_empty());
        assert!(fake.log().contains(&"commit_string:Hi. ".to_string()));
    }

    #[tokio::test]
    async fn no_focused_text_input_falls_back_to_typing() {
        let fake = Fake::new(true, true);
        fake.handle.begin();
        fake.handle.set("hello");
        fake.settle().await;
        assert!(!fake.handle.text_input_active());
        assert_eq!(
            deliver(&fake, &Default::default(), "Hello.").await,
            ["Hello."]
        );
        assert!(!fake.log().iter().any(|l| l.starts_with("commit_string")));
    }

    #[tokio::test]
    async fn the_first_line_replaces_the_preedit_and_the_rest_is_typed() {
        let fake = Fake::new(true, true);
        fake.handle.begin();
        fake.settle().await;
        fake.focus();
        fake.handle.set("line one line two");
        fake.settle().await;
        fake.log();
        let typed = deliver(&fake, &Default::default(), "Line one.\nLine two.").await;
        assert_eq!(typed, ["\nLine two."]);
        assert_eq!(
            fake.log(),
            ["preedit:", "commit_string:Line one.", "commit:1"],
            "the preedit is replaced by the first line in one commit"
        );
    }

    /// Stands in for paste or wtype and writes into the sink's log, so the
    /// test sees one timeline of what the field went through.
    struct FieldTypist(Arc<Mutex<(Driver, RecSink)>>, bool);

    #[async_trait::async_trait]
    impl TextOutput for FieldTypist {
        async fn output(&self, text: &str) -> Result<(), OutputError> {
            self.0.lock().unwrap().1.log.push(format!("typed:{text}"));
            Ok(())
        }
        async fn is_available(&self) -> bool {
            true
        }
        fn inserts_directly(&self) -> bool {
            self.1
        }
        fn name(&self) -> &'static str {
            if self.1 {
                "wtype"
            } else {
                "paste"
            }
        }
    }

    /// What the field shows after each step: the committed text plus the
    /// preedit. Input method requests take effect at their `commit`, typed
    /// text at once. Returns every state from the preedit on.
    fn field_states(preedit: &str, log: &[String]) -> Vec<String> {
        let (mut committed, mut shown) = (String::new(), preedit.to_string());
        let (mut pending_preedit, mut pending_commit) = (None, String::new());
        let mut states = vec![shown.clone()];
        for line in log {
            if let Some(t) = line.strip_prefix("preedit:") {
                pending_preedit = Some(t.to_string());
            } else if let Some(t) = line.strip_prefix("commit_string:") {
                pending_commit.push_str(t);
            } else if line.starts_with("commit:") {
                committed.push_str(&std::mem::take(&mut pending_commit));
                if let Some(p) = pending_preedit.take() {
                    shown = p;
                }
                states.push(format!("{committed}{shown}"));
            } else if let Some(t) = line.strip_prefix("typed:") {
                committed.push_str(t);
                states.push(format!("{committed}{shown}"));
            }
        }
        states
    }

    #[tokio::test]
    async fn a_line_break_never_leaves_the_field_empty() {
        for (text, paste) in [
            ("First line.\nSecond line.", true),
            ("First line.\n\nSecond.\nThird.", true),
            ("One.\tTwo.", false),
        ] {
            let fake = Fake::new(true, true);
            fake.handle.begin();
            fake.settle().await;
            fake.focus();
            fake.handle.set("first line second line");
            fake.settle().await;
            fake.log();
            let chain: Vec<Box<dyn TextOutput>> = vec![
                Box::new(PreeditOutput::new(fake.handle.clone(), &Default::default())),
                Box::new(FieldTypist(fake.driver.clone(), !paste)),
            ];
            output_with_fallback(&chain, text, options()).await.unwrap();
            let states = field_states("first line second line", &fake.log());
            assert!(
                states.iter().all(|s| !s.is_empty()),
                "{text:?}: the field went empty: {states:?}"
            );
            assert_eq!(states.last().unwrap(), text, "{states:?}");
        }
    }

    #[tokio::test]
    async fn a_private_takes_other_lines_are_typed_never_pasted() {
        let fake = Fake::new(true, true);
        fake.handle.begin();
        fake.settle().await;
        fake.focus();
        fake.handle.set("user name password");
        fake.settle().await;
        fake.log();
        let mut chain: Vec<Box<dyn TextOutput>> = vec![
            Box::new(PreeditOutput::new(fake.handle.clone(), &Default::default())),
            Box::new(FieldTypist(fake.driver.clone(), false)),
            Box::new(FieldTypist(fake.driver.clone(), true)),
        ];
        // What the daemon does for a private take.
        chain.retain(|d| d.inserts_directly());
        output_with_fallback(&chain, "User name.\nPassword.", options())
            .await
            .unwrap();
        assert_eq!(
            fake.log(),
            [
                "preedit:",
                "commit_string:User name.",
                "commit:1",
                "typed:\nPassword."
            ]
        );
    }

    #[tokio::test]
    async fn cancel_clears_the_preedit_and_releases() {
        let fake = Fake::new(true, true);
        fake.handle.begin();
        fake.settle().await;
        fake.focus();
        fake.handle.set("never mind");
        fake.handle.end();
        fake.settle().await;
        assert_eq!(
            fake.log(),
            [
                "bind:true",
                "preedit:never mind",
                "commit:1",
                "preedit:",
                "commit:1",
            ]
        );
        // Kept on the field (no preedit) until focus leaves it.
        fake.unfocus();
        assert_eq!(fake.log(), ["unbind"]);
        assert!(!fake.handle.text_input_active());
        assert_eq!(fake.handle.commit("x").await, CommitOutcome::Unavailable);
    }

    #[tokio::test]
    async fn a_cancelled_take_releases_at_once() {
        let fake = Fake::new(true, true);
        fake.handle.begin();
        fake.settle().await;
        fake.focus();
        fake.handle.set("never mind");
        fake.handle.release();
        fake.settle().await;
        assert_eq!(
            fake.log(),
            [
                "bind:true",
                "preedit:never mind",
                "commit:1",
                "preedit:",
                "commit:1",
                "unbind"
            ]
        );
        assert!(!fake.handle.text_input_active());
    }

    #[tokio::test]
    async fn the_kept_binding_reports_field_edits_then_expires() {
        let fake = Fake::new(true, true);
        fake.handle.begin();
        fake.settle().await;
        fake.focus();
        fake.handle.end();
        fake.settle().await;
        fake.log();
        fake.event(Event::Surrounding {
            text: "Ask Groq now".into(),
            cursor: 12,
        });
        fake.event(Event::Done);
        let mut guard = fake.driver.lock().unwrap();
        let (driver, sink) = &mut *guard;
        assert_eq!(
            driver.field_update(),
            Some(Some(("Ask Groq now".to_string(), String::new())))
        );
        assert_eq!(
            driver.field_update(),
            None,
            "an unchanged field is reported once"
        );
        let deadline = driver.linger_deadline().expect("lingering");
        driver.tick(deadline, sink);
        assert_eq!(sink.log, ["unbind"]);
        assert_eq!(
            driver.field_update(),
            Some(None),
            "released: the field is gone"
        );
        assert_eq!(driver.linger_deadline(), None);
    }

    #[tokio::test]
    async fn another_input_method_makes_the_take_fall_back() {
        let fake = Fake::new(true, true);
        fake.handle.begin();
        fake.settle().await;
        fake.event(Event::Unavailable);
        assert_eq!(
            deliver(&fake, &Default::default(), "Hello.").await,
            ["Hello."]
        );
        assert_eq!(fake.log(), ["bind:true", "unbind"]);
        // The next take tries again.
        fake.handle.begin();
        fake.settle().await;
        assert_eq!(fake.log(), ["bind:true"]);
    }

    #[tokio::test]
    async fn focus_loss_mid_take_follows_the_focus() {
        let fake = Fake::new(true, true);
        fake.handle.begin();
        fake.settle().await;
        fake.focus();
        fake.handle.set("hello");
        fake.settle().await;
        fake.unfocus();
        assert!(!fake.handle.text_input_active());
        fake.handle.set("hello there");
        fake.settle().await;
        fake.focus();
        let typed = deliver(&fake, &Default::default(), "Hello there.").await;
        assert!(typed.is_empty());
        let log = fake.log();
        assert_eq!(
            &log[log.len() - 5..],
            [
                "preedit:hello there",
                "commit:3",
                "preedit:",
                "commit_string:Hello there.",
                "commit:3"
            ]
        );
    }

    #[tokio::test]
    async fn focus_lost_at_delivery_types_into_whatever_has_focus() {
        let fake = Fake::new(true, true);
        fake.handle.begin();
        fake.settle().await;
        fake.focus();
        fake.handle.set("hello");
        fake.settle().await;
        fake.unfocus();
        assert_eq!(
            deliver(&fake, &Default::default(), "Hello.").await,
            ["Hello."]
        );
    }

    #[tokio::test]
    async fn a_compositor_without_the_protocol_is_tried_once() {
        let fake = Fake::new(false, false);
        fake.handle.begin();
        fake.handle.set("hello");
        fake.handle.begin();
        assert_eq!(*fake.spawns.lock().unwrap(), 1);
        assert!(!fake.handle.text_input_active());
        assert_eq!(
            deliver(&fake, &Default::default(), "Hello.").await,
            ["Hello."]
        );
    }

    #[test]
    fn auto_submit_makes_it_a_keystroke_method() {
        let fake = Fake::new(false, false);
        let config = crate::config::OutputConfig {
            auto_submit: true,
            ..Default::default()
        };
        let out = PreeditOutput::new(fake.handle.clone(), &config);
        assert!(crate::output::is_keystroke_method(out.name()));
        let out = PreeditOutput::new(fake.handle.clone(), &Default::default());
        assert!(!crate::output::is_keystroke_method(out.name()));
    }
}
