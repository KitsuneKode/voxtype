//! Protocol-free model of one `zwp_input_method_v2` binding.
//!
//! The Wayland thread feeds it the compositor's events and the daemon's
//! commands and sends the requests it returns, so everything that decides
//! what the focused field shows is testable without a compositor.

/// Byte limit for one string in `set_preedit_string` / `commit_string`
/// (the protocol allows 4000; leave room for the message header).
pub const MAX_STRING_BYTES: usize = 3900;

/// The preedit shows at most this much of the take's tail. A field redoes the
/// whole composition on every update (layout, and in ProseMirror editors a
/// transaction and a re-render), so the cost grows with the take: in T3 Code
/// an update took 10 ms to reach the screen early in a take, 30 ms at 1500
/// characters and 57 ms at 3000; showing only the tail keeps it near 10 ms.
/// The whole take is still committed at the end. When the tail outgrows this
/// it restarts at a word about half as far back, so the text in the field
/// moves every dozen words or so, not on every update.
pub const PREEDIT_TAIL_BYTES: usize = 240;

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
    /// `surrounding_text`: the field's text around the cursor (`cursor` is a
    /// byte offset into `text`).
    Surrounding {
        text: String,
        cursor: u32,
    },
}

/// A request for the Wayland thread to send, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request {
    Preedit(String),
    CommitString(String),
    /// `delete_surrounding_text`: this many bytes before the cursor.
    DeleteBefore(u32),
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
    pending_surrounding: Option<(String, usize)>,
    /// The focused field's text and cursor, as of the latest `done`.
    surrounding: Option<(String, usize)>,
    /// The field's text before the cursor when our preedit appeared in it.
    /// Surrounding text leaves the preedit out, so `base` + preedit reported
    /// as the text before the cursor means the field made it typed text.
    base: Option<String>,
    /// The preedit a field was showing when it lost us. Chromium keeps it as
    /// typed text (on blur, a click or a page shortcut).
    lost: String,
    /// Reactivated with `lost` pending and no field text yet: nothing is
    /// shown until the field reports what it kept.
    awaiting_field: bool,
    /// Byte offset in the daemon's text where the shown tail starts.
    tail_start: usize,
}

impl Ime {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a text-input-v3 field is focused and listening to us.
    pub fn active(&self) -> bool {
        self.active && !self.unavailable
    }

    pub fn sensitive(&self) -> bool {
        self.active() && self.hidden
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
            Event::Surrounding { text, cursor } => {
                self.pending_surrounding = Some((text, cursor as usize));
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
                let reactivated = self.pending_active == Some(true);
                self.awaiting_field = false;
                if let Some(active) = self.pending_active.take() {
                    self.active = active;
                    // A (re)activation starts the field with no preedit: it
                    // is a fresh focus, maybe a different field.
                    self.lose_shown();
                    if self.pending_surrounding.is_none() {
                        self.surrounding = None;
                    }
                }
                if let Some(hidden) = self.pending_hidden.take() {
                    self.hidden = hidden;
                }
                if let Some(surrounding) = self.pending_surrounding.take() {
                    self.surrounding = Some(surrounding);
                }
                if !self.active && was_active {
                    self.lose_shown();
                }
                if !self.active {
                    self.surrounding = None;
                }
                let kept = self.take_kept();
                // Chromium reports the field's text only after our next
                // commit: ask with an empty one before showing anything.
                if reactivated
                    && !self.lost.is_empty()
                    && self.base.is_some()
                    && self.field().is_none()
                {
                    self.awaiting_field = true;
                    return vec![Request::Commit(self.serial)];
                }
                let mut requests = self.sync();
                if let Some(len) = kept {
                    requests.retain(|r| !matches!(r, Request::Commit(_)));
                    requests.insert(0, Request::DeleteBefore(len));
                    requests.push(Request::Commit(self.serial));
                }
                return requests;
            }
        }
        Vec::new()
    }

    /// The focused field's text before and after the cursor, when a field
    /// whose text may be read is focused (never a password field).
    pub fn field(&self) -> Option<(&str, &str)> {
        if !self.active() || self.hidden {
            return None;
        }
        let (text, cursor) = self.surrounding.as_ref()?;
        let mut at = (*cursor).min(text.len());
        while !text.is_char_boundary(at) {
            at -= 1;
        }
        Some(text.split_at(at))
    }

    /// Show `text` as the preedit, or its tail when long (see
    /// [`PREEDIT_TAIL_BYTES`]).
    pub fn set(&mut self, text: &str) -> Vec<Request> {
        self.text = self.tail(text);
        self.sync()
    }

    fn tail(&mut self, text: &str) -> String {
        let mut start = self.tail_start.min(text.len());
        if text.len() - start > PREEDIT_TAIL_BYTES {
            start = text.len() - PREEDIT_TAIL_BYTES / 2;
        }
        while !text.is_char_boundary(start) {
            start += 1;
        }
        if start > 0 && !text[..start].ends_with(char::is_whitespace) {
            if let Some(space) = text[start..].find(char::is_whitespace) {
                start += space;
            }
        }
        let body = text[start..].trim_start();
        start = text.len() - body.len();
        self.tail_start = start;
        if start == 0 {
            body.to_string()
        } else {
            format!("\u{2026}{body}")
        }
    }

    /// An empty commit, for a field that reports its text only in answer to
    /// one (Chromium flushes surrounding text when it receives a `done`).
    /// Nothing while a preedit is shown: its own commits do the asking.
    pub fn ask_field(&self) -> Vec<Request> {
        if self.active() && !self.hidden && self.shown.is_empty() && !self.awaiting_field {
            vec![Request::Commit(self.serial)]
        } else {
            Vec::new()
        }
    }

    /// Remove the preedit.
    pub fn clear(&mut self) -> Vec<Request> {
        self.text.clear();
        self.tail_start = 0;
        self.warn_if_kept();
        self.lost.clear();
        self.sync()
    }

    fn lose_shown(&mut self) {
        if !self.shown.is_empty() {
            self.lost = std::mem::take(&mut self.shown);
        }
    }

    fn warn_if_kept(&self) {
        if !self.lost.is_empty() && !self.active() {
            tracing::warn!(
                "Dictation preedit: the field lost focus while showing live text and may have kept it"
            );
        }
    }

    /// The byte length to delete before the cursor when the field turned our
    /// preedit into typed text: its text before the cursor is exactly the
    /// text it had when the preedit appeared, followed by that preedit.
    fn take_kept(&mut self) -> Option<u32> {
        if !self.active() || self.hidden {
            return None;
        }
        let before = self.field()?.0.to_string();
        let lost = !self.lost.is_empty();
        let candidate = if lost { &self.lost } else { &self.shown };
        let kept = !candidate.is_empty()
            && self.base.as_deref().is_some_and(|base| {
                before.len() == base.len() + candidate.len()
                    && before.starts_with(base)
                    && before.ends_with(candidate.as_str())
            });
        if kept {
            tracing::info!(
                "Dictation preedit: the field kept the live text as typed text; removing it"
            );
            let len = candidate.len() as u32;
            self.lost.clear();
            // The field has no preedit of ours now: show it again.
            self.shown.clear();
            return Some(len);
        }
        if lost {
            // Dropped, or another field: it starts over from here.
            self.lost.clear();
            self.base = Some(before);
        }
        None
    }

    /// Replace the preedit with `text`, committed into the field.
    pub fn commit(&mut self, text: &str) -> (CommitOutcome, Vec<Request>) {
        if self.unavailable {
            return (CommitOutcome::Unavailable, Vec::new());
        }
        if !self.active {
            self.text.clear();
            self.warn_if_kept();
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
        self.tail_start = 0;
        self.shown.clear();
        self.lost.clear();
        self.base = None;
        (CommitOutcome::Committed, requests)
    }

    fn sync(&mut self) -> Vec<Request> {
        if !self.active() || self.awaiting_field {
            return Vec::new();
        }
        let want = if self.hidden { "" } else { self.text.as_str() };
        if want == self.shown {
            return Vec::new();
        }
        if want.is_empty() {
            self.base = None;
        } else if self.base.is_none() {
            self.base = self.field().map(|(before, _)| before.to_string());
        }
        self.shown = want.to_string();
        vec![
            Request::Preedit(self.shown.clone()),
            Request::Commit(self.serial),
        ]
    }
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

    fn surrounding(before: &str) -> Event {
        Event::Surrounding {
            text: before.to_string(),
            cursor: before.len() as u32,
        }
    }

    /// A take showing `live` in a field whose text before the cursor is `base`.
    fn showing(base: &str, live: &str) -> Ime {
        let mut ime = Ime::new();
        ime.on_event(Event::Activate);
        ime.on_event(surrounding(base));
        ime.on_event(Event::Done);
        ime.set(live);
        ime
    }

    const LIVE: &str = "Also it's very weird";

    fn removes_and_reshows(serial: u32) -> Vec<Request> {
        vec![
            Request::DeleteBefore(LIVE.len() as u32),
            Request::Preedit(LIVE.to_string()),
            Request::Commit(serial),
        ]
    }

    // Chromium keeps a preedit as typed text when its field loses focus; the
    // final commit then landed after that copy ("weirdAlso, it's very weird.").
    // Chromium 152 reports the field's text only after our next commit.
    #[test]
    fn a_preedit_kept_on_focus_loss_is_removed_once_the_field_reports_it() {
        let mut ime = showing("Done.\n\n", LIVE);
        ime.on_event(Event::Deactivate);
        assert!(ime.on_event(Event::Done).is_empty());
        ime.on_event(Event::Activate);
        assert_eq!(
            ime.on_event(Event::Done),
            vec![Request::Commit(3)],
            "asks for the field's text without showing the preedit again"
        );
        assert!(ime.set(LIVE).is_empty(), "still waiting for the field");
        ime.on_event(surrounding(&format!("Done.\n\n{LIVE}")));
        assert_eq!(ime.on_event(Event::Done), removes_and_reshows(4));
        let (_, requests) = ime.commit("Also, it's very weird.");
        assert!(!requests
            .iter()
            .any(|r| matches!(r, Request::DeleteBefore(_))));
    }

    // Chrome 158 sends the field's text with the activation.
    #[test]
    fn a_preedit_kept_on_focus_loss_is_removed_when_reported_at_activation() {
        let mut ime = showing("Done.\n\n", LIVE);
        ime.on_event(Event::Deactivate);
        ime.on_event(Event::Done);
        ime.on_event(Event::Activate);
        ime.on_event(surrounding(&format!("Done.\n\n{LIVE}")));
        assert_eq!(ime.on_event(Event::Done), removes_and_reshows(3));
    }

    // A click in the field or a page shortcut commits the composition with no
    // deactivation at all.
    #[test]
    fn a_preedit_the_field_committed_in_place_is_removed() {
        let mut ime = showing("Done. ", LIVE);
        ime.on_event(surrounding(&format!("Done. {LIVE}")));
        assert_eq!(ime.on_event(Event::Done), removes_and_reshows(2));
    }

    // Chromium 152 sends its text only in answer to a commit: with no preedit
    // (`read_field`), an empty one asks for it.
    #[test]
    fn a_field_is_asked_for_its_text_only_while_nothing_is_shown() {
        let mut ime = activated();
        assert_eq!(ime.ask_field(), vec![Request::Commit(1)]);
        ime.set("hello");
        assert!(ime.ask_field().is_empty(), "the preedit's own commits ask");
        ime.on_event(Event::Deactivate);
        ime.on_event(Event::Done);
        assert!(ime.ask_field().is_empty(), "no field");
    }

    #[test]
    fn a_field_still_showing_the_preedit_is_left_alone() {
        let mut ime = showing("Done. ", LIVE);
        ime.on_event(surrounding("Done. "));
        assert!(ime.on_event(Event::Done).is_empty());
    }

    // Text that merely ends with the same words, or another field, is the
    // user's own: nothing is deleted.
    #[test]
    fn text_that_only_ends_like_the_preedit_is_never_deleted() {
        for field in [LIVE.to_string(), format!("Other text. {LIVE}")] {
            let mut ime = showing("Done. ", LIVE);
            ime.on_event(Event::Deactivate);
            ime.on_event(Event::Done);
            ime.on_event(Event::Activate);
            ime.on_event(surrounding(&field));
            assert_eq!(ime.on_event(Event::Done), preedit(LIVE, 3), "{field:?}");
        }
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
    fn the_field_text_is_read_at_done_and_never_from_password_fields() {
        let mut ime = activated();
        ime.on_event(Event::Surrounding {
            text: "héllo world".into(),
            cursor: 6,
        });
        assert_eq!(ime.field(), None, "applies only at done");
        ime.on_event(Event::Done);
        assert_eq!(ime.field(), Some(("héllo", " world")));
        // A cursor inside a character is clamped back to its start.
        ime.on_event(Event::Surrounding {
            text: "é".into(),
            cursor: 1,
        });
        ime.on_event(Event::Done);
        assert_eq!(ime.field(), Some(("", "é")));
        ime.on_event(Event::ContentHidden(true));
        ime.on_event(Event::Done);
        assert_eq!(ime.field(), None);
        ime.on_event(Event::Deactivate);
        ime.on_event(Event::Done);
        ime.on_event(Event::Activate);
        ime.on_event(Event::Done);
        assert_eq!(
            ime.field(),
            None,
            "a new focus starts with no text until the field sends it"
        );
    }

    fn shown(requests: &[Request]) -> &str {
        match requests.first() {
            Some(Request::Preedit(text)) => text,
            other => panic!("expected a preedit, got {other:?}"),
        }
    }

    #[test]
    fn a_long_take_shows_its_tail_from_a_word_start() {
        let mut ime = activated();
        let words: Vec<String> = (0..300).map(|i| format!("word{i}")).collect();
        let take = words.join(" ");
        let requests = ime.set(&take);
        let tail = shown(&requests);
        assert!(
            tail.len() <= PREEDIT_TAIL_BYTES + '\u{2026}'.len_utf8(),
            "{tail}"
        );
        let body = tail.strip_prefix('\u{2026}').expect("marked as cut");
        assert!(take.ends_with(body));
        assert!(
            take.contains(&format!(" {body}")),
            "starts at a word: {body}"
        );

        // The whole take still goes into the field at the end.
        let (_, requests) = ime.commit(&take);
        assert!(requests.contains(&Request::CommitString(take.clone())));
    }

    #[test]
    fn the_tail_start_moves_every_few_words_not_on_every_update() {
        let mut ime = activated();
        let words: Vec<String> = (0..300).map(|i| format!("word{i}")).collect();
        let mut starts = Vec::new();
        for n in 1..=words.len() {
            let requests = ime.set(&words[..n].join(" "));
            let body = shown(&requests).trim_start_matches('\u{2026}').to_string();
            let first = body.split(' ').next().unwrap().to_string();
            if starts.last() != Some(&first) {
                starts.push(first);
            }
        }
        // About 120 bytes (~17 of these words) between moves.
        assert!(starts.len() < words.len() / 10, "{} moves", starts.len());
    }

    #[test]
    fn a_new_take_starts_showing_from_its_first_word() {
        let mut ime = activated();
        ime.set(&"long take ".repeat(60));
        ime.commit("Long take.");
        assert_eq!(ime.set("next"), preedit("next", 1));
        ime.set(&"another long take ".repeat(30));
        ime.clear();
        assert_eq!(ime.set("again"), preedit("again", 1));
    }

    #[test]
    fn a_revised_take_keeps_a_word_aligned_tail() {
        let mut ime = activated();
        let take = "alpha beta gamma delta ".repeat(20);
        ime.set(&take);
        // The final pass rewrites the take: the old cut lands mid-word.
        let revised = "Alpha, beta; gamma! delta? ".repeat(20);
        let requests = ime.set(&revised);
        let body = shown(&requests).trim_start_matches('\u{2026}');
        assert!(revised.ends_with(body));
        assert!(revised.contains(&format!(" {body}")), "{body}");
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
            Request::Commit(_) | Request::DeleteBefore(_) => true,
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
