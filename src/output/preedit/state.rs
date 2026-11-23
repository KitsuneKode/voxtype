//! Protocol-free model of one `zwp_input_method_v2` binding.
//!
//! The Wayland thread feeds it the compositor's events and the daemon's
//! commands and sends the requests it returns, so everything that decides
//! what the focused field shows is testable without a compositor.

/// Byte limit for one string in `set_preedit_string` / `commit_string`
/// (the protocol allows 4000; leave room for the message header).
pub const MAX_STRING_BYTES: usize = 3900;

/// A compositor event, already reduced to what the model cares about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Activate,
    Deactivate,
    /// `content_type` with a purpose whose text must not be shown (password, PIN).
    ContentHidden(bool),
    Done,
    /// Another input method holds the seat; this binding is dead.
    Unavailable,
}

/// A request for the Wayland thread to send, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request {
    Preedit(String),
    CommitString(String),
    Commit(u32),
}

/// How a [`Ime::commit`] ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitOutcome {
    /// The text went to the focused field through the input method.
    Committed,
    /// No text field with text-input-v3 is focused (XWayland, terminals and
    /// browsers without IME support): deliver some other way.
    NotActive,
    /// The input method is not bound (another IME owns the seat, the
    /// compositor lacks the protocol, or the binding was never made).
    Unavailable,
}

#[derive(Debug, Default)]
pub struct Ime {
    pending_active: Option<bool>,
    pending_hidden: Option<bool>,
    active: bool,
    hidden: bool,
    /// Number of `done` events seen: the serial `commit` must carry.
    serial: u32,
    /// What the daemon wants shown.
    text: String,
    /// What the field currently shows, as far as this binding knows.
    shown: String,
    unavailable: bool,
}

impl Ime {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a text-input-v3 field is focused and listening to us.
    pub fn active(&self) -> bool {
        self.active && !self.unavailable
    }

    pub fn unavailable(&self) -> bool {
        self.unavailable
    }

    pub fn on_event(&mut self, event: Event) -> Vec<Request> {
        match event {
            Event::Activate => {
                self.pending_active = Some(true);
                // Activation resets the content type to its default.
                self.pending_hidden.get_or_insert(false);
            }
            Event::Deactivate => self.pending_active = Some(false),
            Event::ContentHidden(hidden) => self.pending_hidden = Some(hidden),
            Event::Unavailable => {
                self.unavailable = true;
                self.active = false;
                self.shown.clear();
            }
            Event::Done => {
                self.serial = self.serial.wrapping_add(1);
                let was_active = self.active;
                if let Some(active) = self.pending_active.take() {
                    self.active = active;
                    // A (re)activation starts the field with no preedit: it
                    // is a fresh focus, maybe a different field.
                    self.shown.clear();
                }
                if let Some(hidden) = self.pending_hidden.take() {
                    self.hidden = hidden;
                }
                if !self.active && was_active {
                    self.shown.clear();
                }
                return self.sync();
            }
        }
        Vec::new()
    }

    /// Show `text` as the preedit (trimmed to its tail when too long).
    pub fn set(&mut self, text: &str) -> Vec<Request> {
        self.text = fit_tail(text);
        self.sync()
    }

    /// Remove the preedit.
    pub fn clear(&mut self) -> Vec<Request> {
        self.text.clear();
        self.sync()
    }

    /// Replace the preedit with `text`, committed into the field.
    pub fn commit(&mut self, text: &str) -> (CommitOutcome, Vec<Request>) {
        if self.unavailable {
            return (CommitOutcome::Unavailable, Vec::new());
        }
        if !self.active {
            self.text.clear();
            return (CommitOutcome::NotActive, Vec::new());
        }
        let mut requests = Vec::new();
        let mut first = true;
        for chunk in chunks(text) {
            if first {
                requests.push(Request::Preedit(String::new()));
                first = false;
            }
            requests.push(Request::CommitString(chunk.to_string()));
            requests.push(Request::Commit(self.serial));
        }
        if first && !self.shown.is_empty() {
            requests.push(Request::Preedit(String::new()));
            requests.push(Request::Commit(self.serial));
        }
        self.text.clear();
        self.shown.clear();
        (CommitOutcome::Committed, requests)
    }

    fn sync(&mut self) -> Vec<Request> {
        if !self.active() {
            return Vec::new();
        }
        let want = if self.hidden { "" } else { self.text.as_str() };
        if want == self.shown {
            return Vec::new();
        }
        self.shown = want.to_string();
        vec![
            Request::Preedit(self.shown.clone()),
            Request::Commit(self.serial),
        ]
    }
}

/// The last [`MAX_STRING_BYTES`] of `text`, cut at a char boundary and
/// marked with a leading ellipsis when anything was dropped.
fn fit_tail(text: &str) -> String {
    if text.len() <= MAX_STRING_BYTES {
        return text.to_string();
    }
    let budget = MAX_STRING_BYTES - '…'.len_utf8();
    let mut start = text.len() - budget;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    format!("…{}", &text[start..])
}

/// `text` in pieces of at most [`MAX_STRING_BYTES`], cut at char boundaries.
fn chunks(text: &str) -> impl Iterator<Item = &str> {
    let mut rest = text;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        let mut end = rest.len().min(MAX_STRING_BYTES);
        while !rest.is_char_boundary(end) {
            end -= 1;
        }
        let (head, tail) = rest.split_at(end);
        rest = tail;
        Some(head)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preedit(text: &str, serial: u32) -> Vec<Request> {
        vec![Request::Preedit(text.to_string()), Request::Commit(serial)]
    }

    fn activated() -> Ime {
        let mut ime = Ime::new();
        ime.on_event(Event::Activate);
        ime.on_event(Event::Done);
        ime
    }

    #[test]
    fn nothing_is_sent_before_the_compositor_activates_us() {
        let mut ime = Ime::new();
        assert!(ime.set("hello").is_empty());
        assert!(!ime.active());
        // Activation shows what was asked for meanwhile.
        assert!(
            ime.on_event(Event::Activate).is_empty(),
            "applies only at done"
        );
        assert_eq!(ime.on_event(Event::Done), preedit("hello", 1));
    }

    #[test]
    fn partials_update_the_preedit_and_skip_repeats() {
        let mut ime = activated();
        assert_eq!(ime.set("hel"), preedit("hel", 1));
        assert_eq!(ime.set("hello"), preedit("hello", 1));
        assert!(ime.set("hello").is_empty());
    }

    #[test]
    fn commit_replaces_the_preedit_with_the_final_text() {
        let mut ime = activated();
        ime.set("hello wold");
        let (outcome, requests) = ime.commit("Hello, world.");
        assert_eq!(outcome, CommitOutcome::Committed);
        assert_eq!(
            requests,
            vec![
                Request::Preedit(String::new()),
                Request::CommitString("Hello, world.".to_string()),
                Request::Commit(1),
            ]
        );
        // Nothing left to clear afterwards.
        assert!(ime.clear().is_empty());
    }

    #[test]
    fn empty_commit_only_clears_the_preedit() {
        let mut ime = activated();
        ime.set("hello");
        let (outcome, requests) = ime.commit("");
        assert_eq!(outcome, CommitOutcome::Committed);
        assert_eq!(requests, preedit("", 1));
    }

    #[test]
    fn cancel_clears_the_preedit() {
        let mut ime = activated();
        ime.set("hello");
        assert_eq!(ime.clear(), preedit("", 1));
        assert!(ime.clear().is_empty());
    }

    #[test]
    fn commit_without_a_focused_field_reports_not_active() {
        let mut ime = Ime::new();
        ime.set("hello");
        assert_eq!(ime.commit("hello"), (CommitOutcome::NotActive, Vec::new()));
        // The dropped text must not reappear if a field focuses later.
        ime.on_event(Event::Activate);
        assert!(ime.on_event(Event::Done).is_empty());
    }

    #[test]
    fn unavailable_disables_everything() {
        let mut ime = activated();
        ime.set("hello");
        assert!(ime.on_event(Event::Unavailable).is_empty());
        assert!(!ime.active());
        assert!(ime.set("hello world").is_empty());
        assert_eq!(ime.commit("x"), (CommitOutcome::Unavailable, Vec::new()));
    }

    #[test]
    fn focus_moving_to_another_field_carries_the_preedit_along() {
        let mut ime = activated();
        ime.set("hello");
        // Old field loses focus: it drops its own preedit, we send nothing.
        ime.on_event(Event::Deactivate);
        assert!(ime.on_event(Event::Done).is_empty());
        assert!(!ime.active());
        assert!(ime.set("hello there").is_empty());
        // New field focuses: it gets the current text.
        ime.on_event(Event::Activate);
        assert_eq!(ime.on_event(Event::Done), preedit("hello there", 3));
        // Commits carry the serial of the latest done.
        let (_, requests) = ime.commit("Hello there.");
        assert_eq!(requests.last(), Some(&Request::Commit(3)));
    }

    #[test]
    fn reactivation_of_the_same_field_resends_the_preedit() {
        let mut ime = activated();
        ime.set("hello");
        ime.on_event(Event::Deactivate);
        ime.on_event(Event::Activate);
        assert_eq!(ime.on_event(Event::Done), preedit("hello", 2));
    }

    #[test]
    fn password_fields_never_show_the_preedit_but_still_take_the_commit() {
        let mut ime = Ime::new();
        ime.set("secret");
        ime.on_event(Event::Activate);
        ime.on_event(Event::ContentHidden(true));
        assert!(ime.on_event(Event::Done).is_empty());
        assert!(ime.set("secret words").is_empty());
        let (outcome, requests) = ime.commit("secret words");
        assert_eq!(outcome, CommitOutcome::Committed);
        assert!(requests.contains(&Request::CommitString("secret words".to_string())));
        // A later field of normal type shows text again.
        ime.on_event(Event::Activate);
        ime.on_event(Event::Done);
        assert_eq!(ime.set("visible"), preedit("visible", 2));
    }

    #[test]
    fn long_text_is_shown_by_its_tail_and_committed_in_pieces() {
        let mut ime = activated();
        let long = "é".repeat(MAX_STRING_BYTES); // 2 bytes each
        let requests = ime.set(&long);
        let Request::Preedit(shown) = &requests[0] else {
            panic!("expected a preedit, got {requests:?}");
        };
        assert!(shown.len() <= MAX_STRING_BYTES);
        assert!(shown.starts_with('…'));
        assert!(long.ends_with(shown.trim_start_matches('…')));

        let (_, requests) = ime.commit(&long);
        let committed: String = requests
            .iter()
            .filter_map(|r| match r {
                Request::CommitString(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(committed, long);
        assert!(requests.iter().all(|r| match r {
            Request::CommitString(s) | Request::Preedit(s) => s.len() <= MAX_STRING_BYTES,
            Request::Commit(_) => true,
        }));
        assert_eq!(
            requests
                .iter()
                .filter(|r| matches!(r, Request::Commit(_)))
                .count(),
            2
        );
    }
}
