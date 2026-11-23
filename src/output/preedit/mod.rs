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
use tokio::sync::oneshot;

/// A command for the backend thread.
pub enum Cmd {
    /// Bind the input method for a take.
    Begin,
    Set(String),
    /// Remove the preedit; the ack fires once the compositor has it.
    Clear(Option<oneshot::Sender<()>>),
    /// Commit text in place of the preedit; replies after the compositor has it.
    Commit(String, oneshot::Sender<CommitOutcome>),
    /// Clear the preedit and release the input method.
    End,
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
            b.send(Cmd::Begin);
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

    /// [`end`](Self::end), waiting until the compositor has it (shutdown).
    pub async fn end_now(&self) {
        self.end();
        self.flush().await;
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
    ime: Option<Ime>,
    shared: Arc<Shared>,
}

impl Driver {
    pub fn new(shared: Arc<Shared>) -> Self {
        Self { ime: None, shared }
    }

    pub fn command(&mut self, cmd: Cmd, sink: &mut dyn Sink) -> Ack {
        let ack = match cmd {
            Cmd::Begin => {
                if let Some(mut ime) = self.ime.take() {
                    // A take that never ended: drop its preedit first.
                    sink.send(ime.clear());
                    sink.unbind();
                }
                if sink.bind() {
                    self.ime = Some(Ime::new());
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
                if let Some(mut ime) = self.ime.take() {
                    sink.send(ime.clear());
                    sink.unbind();
                }
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
        tracing::debug!(?event, "Input method event");
        let unavailable = event == Event::Unavailable;
        sink.send(ime.on_event(event));
        if unavailable {
            tracing::warn!(
                "Another input method holds the seat; dictation preedit is off for this take"
            );
            sink.unbind();
            self.ime = None;
        }
        self.publish();
    }

    /// Release everything (the daemon went away).
    pub fn shutdown(&mut self, sink: &mut dyn Sink) {
        self.command(Cmd::End, sink);
    }

    fn publish(&self) {
        let active = self.ime.as_ref().is_some_and(Ime::active);
        self.shared.active.store(active, Ordering::Release);
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

    async fn is_available(&self) -> bool {
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
                "unbind",
            ]
        );
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
    async fn newlines_clear_the_preedit_before_typing() {
        let fake = Fake::new(true, true);
        fake.handle.begin();
        fake.settle().await;
        fake.focus();
        fake.handle.set("line one line two");
        fake.settle().await;
        fake.log();
        let typed = deliver(&fake, &Default::default(), "Line one.\nLine two.").await;
        assert_eq!(typed, ["Line one.\nLine two."]);
        assert_eq!(
            fake.log(),
            ["preedit:", "commit:1"],
            "cleared, nothing committed"
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
                "unbind"
            ]
        );
        assert!(!fake.handle.text_input_active());
        // Commands after the end go nowhere.
        assert_eq!(fake.handle.commit("x").await, CommitOutcome::Unavailable);
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
