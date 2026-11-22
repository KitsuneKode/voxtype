//! `[parakeet] vocabulary_file`: acoustic vocabulary rescoring of the
//! offline (final-pass) decode.
//!
//! Greedy RNN-T decoding picks the likeliest piece at every step, so a name
//! the model has rarely seen loses to a common word that sounds like it
//! ("Calico" for "Kaliko", "hyper land" for "Hyprland"). After the decode,
//! every span of one to four words that looks or sounds like a vocabulary
//! term is scored both ways by the model itself: log P(pieces) over the
//! span's encoder frames, from the predictor state at the span's start. The
//! term replaces the span when
//! `log P(term) + boost × pieces(term) >= log P(heard)`.
//! A term the audio does not support stays out however similar its spelling.
//!
//! The span logic here is model-independent (see [`SpanScorer`]); the
//! scoring itself lives in the vendored parakeet-rs (`score_pieces`).

use rphonetic::{DoubleMetaphone, Encoder};
use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

const MAX_SPAN_WORDS: usize = 4;
/// Terms past this are ignored (with a warning); each adds string
/// comparisons per span, and only the plausible ones reach the model.
const MAX_TERMS: usize = 128;
const WORD_MARK: char = '\u{2581}';
/// One-word spans never rescored unless a corrections row names them: with
/// "Theo" in the vocabulary, every "the" would otherwise be a candidate.
const FUNCTION_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for", "from", "go", "he", "her",
    "him", "his", "i", "if", "in", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or",
    "our", "she", "so", "than", "that", "the", "thee", "their", "them", "then", "there", "these",
    "they", "this", "those", "to", "too", "up", "us", "was", "we", "were", "what", "when", "which",
    "who", "will", "with", "you", "your",
];

/// One vocabulary term, prepared for matching.
#[derive(Debug, Clone, PartialEq)]
pub struct Term {
    pub text: String,
    /// Lowercase letters and digits only.
    key: String,
    words: usize,
    primary: String,
    alternate: String,
    /// Keys of known mishearings (`corrections_file`'s `heard` column).
    aliases: Vec<String>,
}

impl Term {
    fn new(text: &str) -> Option<Self> {
        let text = text.trim();
        let key = key_of(text);
        if key.is_empty() {
            return None;
        }
        let encoder = DoubleMetaphone::default();
        Some(Self {
            text: text.to_string(),
            primary: encoder.encode(&key),
            alternate: encoder.encode_alternate(&key),
            words: text.split_whitespace().count().max(1),
            key,
            aliases: Vec::new(),
        })
    }
}

fn key_of(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Build the term list from vocabulary lines (one term per line, `#`
/// comments), corrections rows (`heard<TAB>meant<TAB>...`) and extra terms.
pub fn parse_terms(vocabulary: &str, corrections: &str, extra: &[String]) -> Vec<Term> {
    let mut terms: Vec<Term> = Vec::new();
    let mut index_of: HashMap<String, usize> = HashMap::new();
    let mut skipped = 0usize;
    let mut add = |text: &str, alias: Option<&str>| {
        let text = text.trim();
        let index = match index_of.get(text) {
            Some(&i) => i,
            None if terms.len() >= MAX_TERMS => {
                skipped += 1;
                return;
            }
            None => {
                let Some(term) = Term::new(text) else { return };
                index_of.insert(term.text.clone(), terms.len());
                terms.push(term);
                terms.len() - 1
            }
        };
        if let Some(alias) = alias.map(key_of).filter(|a| !a.is_empty()) {
            if !terms[index].aliases.contains(&alias) {
                terms[index].aliases.push(alias);
            }
        }
    };
    let lines = |s: &str| {
        s.lines()
            .map(str::trim_end)
            .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
            .map(str::to_string)
            .collect::<Vec<_>>()
    };
    for line in lines(vocabulary) {
        add(&line, None);
    }
    for line in lines(corrections) {
        let mut cols = line.split('\t');
        if let (Some(heard), Some(meant)) = (cols.next(), cols.next()) {
            add(meant, Some(heard));
        }
    }
    for term in extra {
        add(term, None);
    }
    if skipped > 0 {
        tracing::warn!(
            "Parakeet vocabulary: {} more terms past the first {} ignored",
            skipped,
            MAX_TERMS
        );
    }
    terms
}

/// The configured term files, re-read only when one of them changes.
pub struct TermSource {
    vocabulary_file: Option<PathBuf>,
    corrections_file: Option<PathBuf>,
    cache: Mutex<Cache>,
}

#[derive(Default)]
struct Cache {
    stamps: Option<(Option<SystemTime>, Option<SystemTime>, Vec<String>)>,
    profile_terms: Vec<String>,
    terms: Arc<Vec<Term>>,
}

fn modified(path: &Option<PathBuf>) -> Option<SystemTime> {
    path.as_ref()
        .and_then(|p| std::fs::metadata(p).ok())
        .and_then(|m| m.modified().ok())
}

fn read(path: &Option<PathBuf>) -> String {
    match path {
        Some(p) => match std::fs::read(p) {
            Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            Err(e) => {
                tracing::warn!("Parakeet vocabulary: cannot read {}: {}", p.display(), e);
                String::new()
            }
        },
        None => String::new(),
    }
}

impl TermSource {
    /// `None` when neither file is configured (rescoring off).
    pub fn new(vocabulary_file: Option<&Path>, corrections_file: Option<&Path>) -> Option<Self> {
        if vocabulary_file.is_none() && corrections_file.is_none() {
            return None;
        }
        Some(Self {
            vocabulary_file: vocabulary_file.map(expand_home),
            corrections_file: corrections_file.map(expand_home),
            cache: Mutex::new(Cache::default()),
        })
    }

    /// Profile vocabulary for the next recordings, added to the files' terms.
    pub fn set_profile_terms(&self, terms: &[String]) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.profile_terms = terms.to_vec();
        }
    }

    /// Current terms. Costs two `stat`s unless a file (or the profile
    /// terms) changed since the last call.
    pub fn terms(&self) -> Arc<Vec<Term>> {
        let Ok(mut cache) = self.cache.lock() else {
            return Arc::default();
        };
        let stamps = (
            modified(&self.vocabulary_file),
            modified(&self.corrections_file),
            cache.profile_terms.clone(),
        );
        if cache.stamps.as_ref() != Some(&stamps) {
            let terms = parse_terms(
                &read(&self.vocabulary_file),
                &read(&self.corrections_file),
                &cache.profile_terms,
            );
            tracing::info!("Parakeet vocabulary: loaded {} terms", terms.len());
            cache.terms = Arc::new(terms);
            cache.stamps = Some(stamps);
        }
        cache.terms.clone()
    }
}

fn expand_home(path: &Path) -> PathBuf {
    match path.strip_prefix("~") {
        Ok(rest) => dirs::home_dir().map_or_else(|| path.to_path_buf(), |h| h.join(rest)),
        Err(_) => path.to_path_buf(),
    }
}

/// What rescoring needs from the model and its decode.
pub trait SpanScorer {
    /// SentencePiece string of a token (with `▁` word marks).
    fn piece(&self, id: usize) -> &str;
    fn encode(&self, text: &str) -> Vec<usize>;
    /// Log P of each candidate over encoder `frames`, starting from the
    /// predictor state before token `from`.
    fn score(
        &self,
        from: usize,
        frames: Range<usize>,
        candidates: &[Vec<usize>],
    ) -> Result<Vec<f32>, String>;
}

/// A word: its letter/digit tokens, and whether punctuation precedes it.
#[derive(Debug, PartialEq)]
struct Word {
    tokens: Range<usize>,
    after_punct: bool,
}

fn is_punct(piece: &str) -> bool {
    let rest = piece.trim_start_matches(WORD_MARK);
    !rest.is_empty() && rest.chars().all(|c| !c.is_alphanumeric())
}

fn words(ids: &[usize], scorer: &dyn SpanScorer) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut open = false;
    let mut punct = false;
    for (i, &id) in ids.iter().enumerate() {
        let piece = scorer.piece(id);
        if is_punct(piece) {
            open = false;
            punct = true;
        } else if piece.starts_with(WORD_MARK) || !open {
            words.push(Word {
                tokens: i..i + 1,
                after_punct: punct,
            });
            open = true;
            punct = false;
        } else if let Some(w) = words.last_mut() {
            w.tokens.end = i + 1;
        }
    }
    words
}

/// A span's words' keys, with what matching needs computed once.
struct Span {
    words: Vec<String>,
    key: String,
    primary: String,
    alternate: String,
}

impl Span {
    fn new(words: Vec<String>) -> Self {
        let key = words.concat();
        let encoder = DoubleMetaphone::default();
        Self {
            primary: encoder.encode(&key),
            alternate: encoder.encode_alternate(&key),
            words,
            key,
        }
    }
}

/// Whether the span could be a mishearing of `term`.
fn plausible(span: &Span, term: &Term) -> bool {
    let words = &span.words;
    let span_key = span.key.as_str();
    let span_words = words.len();
    if term.aliases.iter().any(|a| a == span_key) {
        return true;
    }
    if span_words + 1 < term.words || span_words > term.words + 1 {
        return false;
    }
    if span_words == 1 && FUNCTION_WORDS.contains(&span_key) {
        return false;
    }
    if span_key == term.key {
        return true;
    }
    // The term is already there as whole words and the span only adds
    // words around it: the boost would pay for deleting them ("Quilter is"
    // -> "Quilter"). "Mana shot" for "Manash" still counts.
    let has_term =
        (0..span_words).any(|i| (i + 1..=span_words).any(|j| words[i..j].concat() == term.key));
    if span_words > term.words && has_term {
        return false;
    }
    // An extra word must be (nearly) all spacing, never a real word the term
    // would delete: "hyper land" -> Hyprland, not "Mana shot" -> Manash.
    if span_words > term.words
        && span_key.chars().count() > term.key.chars().count() + (span_words - term.words)
    {
        return false;
    }
    // Double Metaphone codes stop at four letters, so length has to agree
    // too, or "hyper land today" would sound like "Hyprland".
    let (a, b) = (span_key.chars().count(), term.key.chars().count());
    if b < 3 || a * 2 < b || a * 2 > b * 3 {
        return false;
    }
    let similarity = strsim::normalized_levenshtein(span_key, &term.key);
    if similarity >= 0.75 {
        return true;
    }
    let sounds_alike = [&span.primary, &span.alternate]
        .iter()
        .any(|k| !k.is_empty() && (**k == term.primary || **k == term.alternate));
    similarity >= 0.5 && sounds_alike
}

/// "Rust", "Linear", "New York": capitals only at word starts.
fn is_titlecase(text: &str) -> bool {
    text.split_whitespace().all(|w| {
        let mut chars = w.chars();
        chars.next().is_some_and(char::is_uppercase) && chars.all(|c| !c.is_uppercase())
    })
}

/// One accepted replacement, for logging and tests.
#[derive(Debug, Clone, PartialEq)]
pub struct Replacement {
    pub tokens: Range<usize>,
    pub heard: String,
    pub term: String,
    /// `log P(term) + boost × pieces − log P(heard)`, in nats.
    pub margin: f32,
}

/// Rescore a decode (`tokens` = `(id, frame)` in order, `frame_count`
/// encoder frames) against `terms`. Returns the token ids with accepted
/// replacements applied, and the replacements; `None` when nothing changed.
pub fn rescore(
    tokens: &[(usize, usize)],
    frame_count: usize,
    terms: &[Term],
    boost: f32,
    scorer: &dyn SpanScorer,
) -> Option<(Vec<usize>, Vec<Replacement>)> {
    if terms.is_empty() || tokens.is_empty() {
        return None;
    }
    let ids: Vec<usize> = tokens.iter().map(|t| t.0).collect();
    let words = words(&ids, scorer);
    let text_of = |r: &Range<usize>| -> String {
        ids[r.clone()]
            .iter()
            .map(|&id| scorer.piece(id).replace(WORD_MARK, " "))
            .collect::<String>()
            .trim()
            .to_string()
    };
    let mut found: Vec<Replacement> = Vec::new();
    for start in 0..words.len() {
        for n in 1..=MAX_SPAN_WORDS.min(words.len() - start) {
            if words[start + 1..start + n].iter().any(|w| w.after_punct) {
                break;
            }
            let span = words[start].tokens.start..words[start + n - 1].tokens.end;
            // A word glued to "(" or "-" has no word mark; replacing it
            // would insert a space.
            if !scorer.piece(ids[span.start]).starts_with(WORD_MARK) {
                break;
            }
            let heard = text_of(&span);
            let candidate_span = Span::new(
                words[start..start + n]
                    .iter()
                    .map(|w| key_of(&text_of(&w.tokens)))
                    .collect(),
            );
            let picked: Vec<&Term> = terms
                .iter()
                .filter(|t| t.text != heard && plausible(&candidate_span, t))
                .collect();
            if picked.is_empty() {
                continue;
            }
            let heard_ids = ids[span.clone()].to_vec();
            let mut candidates = vec![heard_ids.clone()];
            let mut kept = Vec::new();
            for term in picked {
                let pieces = scorer.encode(&term.text);
                if pieces.is_empty() || pieces.contains(&0) || pieces == heard_ids {
                    continue;
                }
                candidates.push(pieces);
                kept.push(term);
            }
            if kept.is_empty() {
                continue;
            }
            let first_frame = tokens[span.start].1;
            let last_frame = tokens[span.end - 1].1;
            let end_frame = tokens
                .get(span.end)
                .map_or(frame_count, |t| t.1)
                .max(last_frame + 1);
            let scores = match scorer.score(span.start, first_frame..end_frame, &candidates) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("Parakeet vocabulary: scoring {:?} failed: {}", heard, e);
                    continue;
                }
            };
            let heard_score = scores[0];
            let best = kept
                .iter()
                .enumerate()
                .map(|(i, term)| {
                    // A plain capital is no acoustic evidence: "rust" stays
                    // "rust" unless the model itself prefers "Rust". Brand
                    // casing (PipeWire, voxtype) is what the file is for.
                    let term_boost = if heard.to_lowercase() == term.text.to_lowercase()
                        && is_titlecase(&term.text)
                    {
                        0.0
                    } else {
                        boost
                    };
                    let boosted = scores[i + 1] + term_boost * candidates[i + 1].len() as f32;
                    (term, boosted - heard_score, scores[i + 1])
                })
                .max_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((term, margin, score)) = best {
                tracing::debug!(
                    "Parakeet vocabulary: {:?} ({:.2}) vs {:?} ({:.2}): margin {:.2}",
                    heard,
                    heard_score,
                    term.text,
                    score,
                    margin
                );
                if margin >= 0.0 {
                    found.push(Replacement {
                        tokens: span,
                        heard,
                        term: term.text.clone(),
                        margin,
                    });
                }
            }
        }
    }
    if found.is_empty() {
        return None;
    }
    // Best margins first; a span that overlaps an accepted one is dropped.
    found.sort_by(|a, b| b.margin.total_cmp(&a.margin));
    let mut accepted: Vec<Replacement> = Vec::new();
    for r in found {
        if accepted
            .iter()
            .all(|a| r.tokens.end <= a.tokens.start || r.tokens.start >= a.tokens.end)
        {
            accepted.push(r);
        }
    }
    accepted.sort_by_key(|r| r.tokens.start);
    let mut out = Vec::with_capacity(ids.len());
    let mut at = 0;
    for r in &accepted {
        out.extend_from_slice(&ids[at..r.tokens.start]);
        out.extend(scorer.encode(&r.term));
        at = r.tokens.end;
    }
    out.extend_from_slice(&ids[at..]);
    Some((out, accepted))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Pieces are whole words (plus punctuation); scores come from a table.
    struct Fake {
        pieces: Vec<String>,
        scores: HashMap<String, f32>,
    }

    impl Fake {
        fn new(scores: &[(&str, f32)]) -> Self {
            Self {
                pieces: vec!["<unk>".into()],
                scores: scores.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            }
        }
        fn id(&mut self, piece: &str) -> usize {
            match self.pieces.iter().position(|p| p == piece) {
                Some(i) => i,
                None => {
                    self.pieces.push(piece.to_string());
                    self.pieces.len() - 1
                }
            }
        }
        /// Tokens for `text`, one frame each; known terms registered first.
        fn decode(&mut self, text: &str, terms: &[&str]) -> Vec<(usize, usize)> {
            for t in terms {
                for w in t.split_whitespace() {
                    self.id(&format!("{WORD_MARK}{w}"));
                }
            }
            let mut out = Vec::new();
            for w in text.split_whitespace() {
                let (core, punct) = match w.strip_suffix([',', '.']) {
                    Some(c) => (c, Some(&w[c.len()..])),
                    None => (w, None),
                };
                out.push(self.id(&format!("{WORD_MARK}{core}")));
                if let Some(p) = punct {
                    out.push(self.id(p));
                }
            }
            out.into_iter().enumerate().map(|(i, id)| (id, i)).collect()
        }
        fn text(&self, ids: &[usize]) -> String {
            ids.iter()
                .map(|&i| self.pieces[i].replace(WORD_MARK, " "))
                .collect::<String>()
                .trim()
                .to_string()
        }
    }

    impl SpanScorer for Fake {
        fn piece(&self, id: usize) -> &str {
            &self.pieces[id]
        }
        fn encode(&self, text: &str) -> Vec<usize> {
            text.split_whitespace()
                .map(|w| {
                    self.pieces
                        .iter()
                        .position(|p| *p == format!("{WORD_MARK}{w}"))
                        .unwrap_or(0)
                })
                .collect()
        }
        fn score(
            &self,
            _from: usize,
            _frames: Range<usize>,
            candidates: &[Vec<usize>],
        ) -> Result<Vec<f32>, String> {
            Ok(candidates
                .iter()
                .map(|c| *self.scores.get(&self.text(c)).unwrap_or(&-50.0))
                .collect())
        }
    }

    fn run(heard: &str, vocab: &str, scores: &[(&str, f32)]) -> String {
        let terms = parse_terms(vocab, "", &[]);
        let names: Vec<&str> = terms.iter().map(|t| t.text.as_str()).collect();
        let mut fake = Fake::new(scores);
        let tokens = fake.decode(heard, &names);
        match rescore(&tokens, tokens.len(), &terms, 2.5, &fake) {
            Some((ids, _)) => fake.text(&ids),
            None => fake.text(&tokens.iter().map(|t| t.0).collect::<Vec<_>>()),
        }
    }

    #[test]
    fn a_term_the_audio_supports_replaces_the_heard_span() {
        // One piece each: -6 + 2.5 >= -4.
        assert_eq!(
            run(
                "agreed Calico.",
                "Kaliko",
                &[("Calico", -4.0), ("Kaliko", -6.0)]
            ),
            "agreed Kaliko."
        );
    }

    #[test]
    fn a_term_the_audio_does_not_support_is_left_out() {
        assert_eq!(
            run(
                "agreed Calico.",
                "Kaliko",
                &[("Calico", -1.0), ("Kaliko", -9.0)]
            ),
            "agreed Calico."
        );
    }

    #[test]
    fn dissimilar_words_are_never_scored() {
        // Would win on score, but "banana" neither looks nor sounds like it.
        assert_eq!(
            run("a banana", "Kaliko", &[("banana", -9.0), ("Kaliko", 0.0)]),
            "a banana"
        );
    }

    #[test]
    fn multi_word_spans_can_become_one_term_but_not_across_punctuation() {
        let scores = [("hyper land", -3.0), ("Hyprland", -4.0)];
        assert_eq!(
            run("on hyper land today", "Hyprland", &scores),
            "on Hyprland today"
        );
        assert_eq!(
            run("on hyper, land today", "Hyprland", &scores),
            "on hyper, land today"
        );
    }

    #[test]
    fn a_term_already_heard_never_swallows_its_neighbours() {
        // "Quilter is" scored as "Quilter" would win on the boost alone.
        let scores = [("Quilter is", -3.0), ("Quilter", -4.0)];
        assert_eq!(
            run("Quilter is here", "Quilter", &scores),
            "Quilter is here"
        );
        // "shot" is a real word, not spacing: Manash may not eat it.
        let scores = [("Mana shot", -3.0), ("Manash", -4.0)];
        assert_eq!(
            run("by Mana shot today", "Manash", &scores),
            "by Mana shot today"
        );
        let scores = [("man ash", -3.0), ("Manash", -4.0)];
        assert_eq!(
            run("by man ash today", "Manash", &scores),
            "by Manash today"
        );
    }

    #[test]
    fn casing_alone_and_function_words_get_no_boost() {
        // Would win only through the boost.
        let scores = [("rust", -3.0), ("Rust", -4.0)];
        assert_eq!(run("written in rust", "Rust", &scores), "written in rust");
        let scores = [("Pipewire", -3.0), ("PipeWire", -4.0)];
        assert_eq!(
            run("so Pipewire works", "PipeWire", &scores),
            "so PipeWire works"
        );
        let scores = [("the", -3.0), ("Theo", -4.0)];
        assert_eq!(run("ask the about it", "Theo", &scores), "ask the about it");
    }

    #[test]
    fn term_list_is_capped() {
        let vocab: String = (0..20_000).map(|i| format!("Term{i}\n")).collect();
        let terms = parse_terms(&vocab, "", &[]);
        assert_eq!(terms.len(), MAX_TERMS);
    }

    #[test]
    fn corrections_heard_forms_make_a_span_a_candidate() {
        let terms = parse_terms("", "# c\nbox type\tvoxtype\tseed\t1\n", &[]);
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].text, "voxtype");
        assert_eq!(terms[0].aliases, vec!["boxtype".to_string()]);
        assert!(plausible(
            &Span::new(vec!["box".into(), "type".into()]),
            &terms[0]
        ));
    }

    #[test]
    fn terms_are_deduplicated_and_comments_skipped() {
        let terms = parse_terms(
            "# x\nGitHub\n\nGitHub\nPR\n",
            "gh\tGitHub\tu\t1\n",
            &["PR".into(), "Groq".into()],
        );
        let names: Vec<&str> = terms.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(names, ["GitHub", "PR", "Groq"]);
    }

    #[test]
    fn term_files_reload_only_when_they_change() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vocabulary.txt");
        std::fs::write(&path, "Kaliko\n").unwrap();
        let source = TermSource::new(Some(&path), None).unwrap();
        let first = source.terms();
        assert_eq!(first.len(), 1);
        assert!(
            Arc::ptr_eq(&first, &source.terms()),
            "unchanged file is not re-read"
        );
        std::fs::write(&path, "Kaliko\nShaggy\n").unwrap();
        let later = SystemTime::now() + std::time::Duration::from_secs(5);
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(later)
            .unwrap();
        assert_eq!(source.terms().len(), 2);
        source.set_profile_terms(&["Ruggedo".into()]);
        assert_eq!(source.terms().len(), 3);
    }
}
