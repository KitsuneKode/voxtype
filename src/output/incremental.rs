//! Deliver text that arrives in pieces as one output.
//!
//! A streamed post-process (`[output.post_process] streaming = true`) prints
//! the cleaned text a sentence at a time, and typing starts with the first
//! sentence. Everything a normal delivery does once still happens once:
//!
//! - The modifier-release wait and the pre-output hook run before the first
//!   piece ([`IncrementalOutput::begin`], which the daemon runs while the
//!   command is still working), the post-output hook after the last
//!   ([`IncrementalOutput::finish`]).
//! - `pre_type_delay_ms` applies to the first piece only; `append_text` and
//!   `auto_submit` to the last piece only. To know which piece is last, the
//!   newest piece is held until the next arrives or the stream ends.
//! - Leading whitespace of the whole text and its trailing whitespace are
//!   dropped (when the command's output is trimmed), and whitespace at the
//!   end of a piece is held until more text follows, so a final newline is
//!   never typed.
//!
//! Only keystroke drivers (wtype, eitype, dotool, ydotool) type piece by
//! piece. When the chain lands on anything else (clipboard; the modifier
//! wait timed out; no keystroke driver works), the pieces are collected and
//! delivered once at the end through the normal fallback order. If a driver
//! fails partway, what was typed stays typed and the rest goes to the
//! clipboard, with a notification saying so.

use super::{normalize_quotes, run_hook, OutputOptions, TextOutput};
use crate::error::OutputError;

/// Where a piece sits in the delivery; decides which per-delivery extras
/// (pre-type delay, appended text, auto-submit) its chain carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pub first: bool,
    pub last: bool,
}

/// Builds the output chain for a piece at a given position. Every chain it
/// returns must list the same methods in the same order (only per-delivery
/// options differ), so a method index stays valid across pieces.
pub type ChainFactory<'a> = Box<dyn Fn(Position) -> Vec<Box<dyn TextOutput>> + Send + Sync + 'a>;

/// The production factory: `config`'s chain, with the pre-type delay only
/// on the first piece and append/auto-submit only on the last.
pub fn chain_factory<'a>(config: &crate::config::OutputConfig) -> ChainFactory<'a> {
    let config = config.clone();
    Box::new(move |pos: Position| {
        let mut c = config.clone();
        if !pos.first {
            c.pre_type_delay_ms = 0;
            c.wtype_delay_ms = 0;
        }
        if !pos.last {
            c.auto_submit = false;
            c.append_text = None;
        }
        super::create_output_chain(&c)
    })
}

/// Drivers that can type a piece and then the next without side effects.
/// Paste drivers go through the clipboard and are left out on purpose.
fn types_incrementally(name: &str) -> bool {
    matches!(name, "wtype" | "eitype" | "dotool" | "ydotool")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    /// No piece typed yet.
    Undecided,
    /// Typing with the chain's method at this index.
    Typing(usize),
    /// Collecting for one delivery at the end.
    Collect,
}

pub struct IncrementalOutput<'a> {
    chain_for: ChainFactory<'a>,
    options: OutputOptions<'a>,
    trim: bool,
    skip_keystrokes: bool,
    begun: bool,
    method: Method,
    /// Nothing visible delivered or held yet (leading whitespace is dropped).
    at_start: bool,
    /// Trailing whitespace of the text so far, not typed yet.
    held_ws: String,
    /// The newest piece, typed when the next arrives or at the end.
    held: Option<String>,
    typed_any: bool,
    /// Text the keystroke driver typed.
    typed: String,
    /// Text left for the delivery at the end.
    collected: String,
}

impl<'a> IncrementalOutput<'a> {
    pub fn new(chain_for: ChainFactory<'a>, options: OutputOptions<'a>, trim: bool) -> Self {
        Self {
            chain_for,
            options,
            trim,
            skip_keystrokes: false,
            begun: false,
            method: Method::Undecided,
            at_start: true,
            held_ws: String::new(),
            held: None,
            typed_any: false,
            typed: String::new(),
            collected: String::new(),
        }
    }

    /// Modifier-release wait and pre-output hook. Safe to call once before
    /// the first piece; `push` calls it if it has not run.
    pub async fn begin(&mut self) {
        if self.begun {
            return;
        }
        self.begun = true;
        let chain = (self.chain_for)(Position { first: true, last: true });
        #[cfg(target_os = "linux")]
        if self.options.wait_for_modifier_release
            && super::chain_synthesizes_keystrokes(chain.iter().map(|o| o.name()))
        {
            let mut guard = super::modifier_guard::ModifierGuard::new();
            if guard
                .wait_for_release(self.options.modifier_release_timeout)
                .await
                .is_err()
            {
                tracing::warn!(
                    "Modifier keys still held after timeout; delivering through the clipboard"
                );
                crate::notification::send(
                    "Voxtype",
                    "Modifier key held too long, transcription copied to clipboard.",
                )
                .await;
                self.skip_keystrokes = true;
            }
        }
        drop(chain);
        if let Some(cmd) = self.options.pre_output_command {
            if let Err(e) = run_hook(cmd, "pre_output").await {
                tracing::warn!("{}", e);
            }
        }
    }

    /// Add the next piece of text.
    pub async fn push(&mut self, piece: &str) {
        self.begin().await;
        let mut text = std::mem::take(&mut self.held_ws);
        text.push_str(piece);
        if self.at_start && self.trim {
            text = text.trim_start().to_string();
        }
        let body_len = text.trim_end().len();
        if body_len == 0 {
            // Whitespace only: it belongs in front of the next piece.
            self.held_ws = text;
            return;
        }
        self.at_start = false;
        self.held_ws = text[body_len..].to_string();
        text.truncate(body_len);
        if let Some(prev) = self.held.replace(text) {
            self.deliver(prev, false).await;
        }
    }

    /// Whether any visible text has been pushed.
    pub fn has_text(&self) -> bool {
        !self.at_start
    }

    /// Deliver the last piece and anything collected, run the post-output
    /// hook, and return the whole delivered text.
    pub async fn finish(mut self) -> Result<String, OutputError> {
        self.begin().await;
        if let Some(mut last) = self.held.take() {
            if !self.trim {
                last.push_str(&self.held_ws);
            }
            self.deliver(last, true).await;
        }
        let result = self.deliver_collected().await;
        if let Some(cmd) = self.options.post_output_command {
            if let Err(e) = run_hook(cmd, "post_output").await {
                tracing::warn!("{}", e);
            }
        }
        result.map(|()| {
            let mut all = self.typed;
            all.push_str(&self.collected);
            all
        })
    }

    /// Stop without delivering what is still held (cancel). The post-output
    /// hook still runs, so a compositor submap entered by `begin` is reset.
    pub async fn abort(self) {
        if self.begun {
            if let Some(cmd) = self.options.post_output_command {
                if let Err(e) = run_hook(cmd, "post_output").await {
                    tracing::warn!("{}", e);
                }
            }
        }
    }

    async fn deliver(&mut self, text: String, last: bool) {
        let pos = Position {
            first: !self.typed_any,
            last,
        };
        let normalized = normalize_quotes(&text).into_owned();
        match self.method {
            Method::Collect => self.collected.push_str(&text),
            Method::Undecided => {
                let chain = (self.chain_for)(pos);
                for (i, out) in chain.iter().enumerate() {
                    if self.skip_keystrokes || !types_incrementally(out.name()) {
                        break;
                    }
                    if !out.is_available().await {
                        tracing::debug!("{} not available, trying next", out.name());
                        continue;
                    }
                    match out.output(&normalized).await {
                        Ok(()) => {
                            tracing::debug!("Typing streamed text via {}", out.name());
                            self.method = Method::Typing(i);
                            self.typed_any = true;
                            self.typed.push_str(&text);
                            return;
                        }
                        Err(e) => tracing::warn!("{} failed: {}, trying next", out.name(), e),
                    }
                }
                self.method = Method::Collect;
                self.collected.push_str(&text);
            }
            Method::Typing(i) => {
                let chain = (self.chain_for)(pos);
                let result = match chain.get(i) {
                    Some(out) => out.output(&normalized).await,
                    None => Err(OutputError::AllMethodsFailed),
                };
                match result {
                    Ok(()) => {
                        self.typed_any = true;
                        self.typed.push_str(&text);
                    }
                    Err(e) => {
                        tracing::warn!("Typing failed partway ({}); the rest goes to the clipboard", e);
                        self.method = Method::Collect;
                        self.collected.push_str(&text);
                    }
                }
            }
        }
    }

    async fn deliver_collected(&mut self) -> Result<(), OutputError> {
        if self.collected.is_empty() {
            return if self.typed_any || self.at_start {
                Ok(())
            } else {
                Err(OutputError::AllMethodsFailed)
            };
        }
        let normalized = normalize_quotes(&self.collected).into_owned();
        let partial = self.typed_any;
        let chain = (self.chain_for)(Position {
            first: !partial,
            last: true,
        });
        for out in &chain {
            // After a partial typing run, only a method that does not type
            // can take the rest without mixing it into the typed text.
            if (partial || self.skip_keystrokes) && super::is_keystroke_method(out.name()) {
                continue;
            }
            if !out.is_available().await {
                continue;
            }
            match out.output(&normalized).await {
                Ok(()) => {
                    if partial {
                        tracing::warn!("Rest of the text delivered via {}", out.name());
                        crate::notification::send(
                            "Voxtype",
                            "Typing stopped partway; the rest of the text is on the clipboard.",
                        )
                        .await;
                    }
                    return Ok(());
                }
                Err(e) => tracing::warn!("{} failed: {}, trying next", out.name(), e),
            }
        }
        Err(OutputError::AllMethodsFailed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct Mock {
        name: &'static str,
        log: Arc<Mutex<Vec<String>>>,
        fail_after: Option<usize>,
        pos: Position,
    }

    #[async_trait::async_trait]
    impl TextOutput for Mock {
        async fn output(&self, text: &str) -> Result<(), OutputError> {
            let mut log = self.log.lock().unwrap();
            let calls = log.iter().filter(|l| l.starts_with(self.name)).count();
            if self.fail_after.is_some_and(|n| calls >= n) {
                return Err(OutputError::AllMethodsFailed);
            }
            let tag = match (self.pos.first, self.pos.last) {
                (true, true) => "first+last",
                (true, false) => "first",
                (false, true) => "last",
                (false, false) => "mid",
            };
            log.push(format!("{}[{}]:{}", self.name, tag, text));
            Ok(())
        }
        async fn is_available(&self) -> bool {
            true
        }
        fn name(&self) -> &'static str {
            self.name
        }
    }

    fn factory(log: Arc<Mutex<Vec<String>>>, typer_fails_after: Option<usize>) -> ChainFactory<'static> {
        Box::new(move |pos| {
            let v: Vec<Box<dyn TextOutput>> = vec![
                Box::new(Mock { name: "wtype", log: log.clone(), fail_after: typer_fails_after, pos }),
                Box::new(Mock { name: "clipboard", log: log.clone(), fail_after: None, pos }),
            ];
            v
        })
    }

    fn opts() -> OutputOptions<'static> {
        OutputOptions {
            history_context: None,
            pre_output_command: None,
            post_output_command: None,
            wait_for_modifier_release: false,
            modifier_release_timeout: std::time::Duration::ZERO,
        }
    }

    async fn run(pieces: &[&str], trim: bool, fail_after: Option<usize>) -> (Vec<String>, String) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut out = IncrementalOutput::new(factory(log.clone(), fail_after), opts(), trim);
        out.begin().await;
        for p in pieces {
            out.push(p).await;
        }
        let text = out.finish().await.unwrap_or_default();
        let log = log.lock().unwrap().clone();
        (log, text)
    }

    #[tokio::test]
    async fn pieces_are_typed_in_order_with_extras_on_the_ends() {
        let (log, text) = run(&["One.", " Two.", " Three."], true, None).await;
        assert_eq!(
            log,
            vec!["wtype[first]:One.", "wtype[mid]: Two.", "wtype[last]: Three."]
        );
        assert_eq!(text, "One. Two. Three.");
    }

    #[tokio::test]
    async fn a_single_piece_is_first_and_last() {
        let (log, _) = run(&["Only one."], true, None).await;
        assert_eq!(log, vec!["wtype[first+last]:Only one."]);
    }

    #[tokio::test]
    async fn surrounding_whitespace_is_trimmed_and_inner_kept() {
        let (log, text) = run(&["\n  Lead-in:", "\n", "1. Item.\n", "2. Item.\n\n"], true, None).await;
        assert_eq!(text, "Lead-in:\n1. Item.\n2. Item.");
        assert_eq!(log.last().unwrap(), "wtype[last]:\n2. Item.");
    }

    #[tokio::test]
    async fn untrimmed_output_keeps_a_trailing_space() {
        let (_, text) = run(&["Hi.", " "], false, None).await;
        assert_eq!(text, "Hi. ");
    }

    #[tokio::test]
    async fn whitespace_only_output_delivers_nothing() {
        let (log, text) = run(&["  ", "\n"], true, None).await;
        assert!(log.is_empty());
        assert_eq!(text, "");
    }

    #[tokio::test]
    async fn a_typing_failure_sends_the_rest_to_the_clipboard() {
        let (log, text) = run(&["One.", " Two.", " Three."], true, Some(1)).await;
        assert_eq!(log, vec!["wtype[first]:One.", "clipboard[last]: Two. Three."]);
        assert_eq!(text, "One. Two. Three.");
    }

    #[tokio::test]
    async fn no_working_typer_collects_everything_for_one_delivery() {
        let (log, text) = run(&["One.", " Two."], true, Some(0)).await;
        assert_eq!(log, vec!["clipboard[first+last]:One. Two."]);
        assert_eq!(text, "One. Two.");
    }

    #[tokio::test]
    async fn curly_quotes_are_normalized_per_piece() {
        let (log, _) = run(&["It\u{2019}s fine."], true, None).await;
        assert_eq!(log, vec!["wtype[first+last]:It's fine."]);
    }
}
