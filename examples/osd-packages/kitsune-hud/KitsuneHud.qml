// kitsune-hud — voxtype Quickshell style package (compact island).
//
// One small pill:  [ ▂▅█▅▂ ]  Listening            0:14
//                  [ ~~~~~ ]  "…so the idea is we cache the tokens"
//
// Signals consumed (all from the parent OsdSurface):
//   - daemonState   recording / streaming / transcribing (StateReader)
//   - audio         peak/rms/vad frames (audio-bridge sidecar)
//   - liveText      $XDG_RUNTIME_DIR/voxtype/live_text, maintained by the
//                   voxtype-live-osd watcher (journal → file transport)
//
// With the voxtype fork's streaming preview, live words arrive while
// "streaming" (~3 updates/s); the cleanup script shows the raw text while
// "transcribing". The layout collapses to a compact pill when there is no
// text, grows to one line, then two, then scrolls like captions.
//
// The parent OsdSurface handles visibility (non-idle only) and
// click-through; this item fills the screen and positions its own card
// from the resolved config (position / margin_px / offset_x / offset_y).

import QtQuick
import Quickshell
import Quickshell.Io
import QtQuick.Effects

Item {
    id: root

    // Synced by OsdSurface._syncCustomItem()
    property string daemonState: "idle"
    property var audio: null
    property var theme: null
    property var recipe: null
    property string assetRoot: ""

    // vstate = daemonState with torn reads filtered out. The daemon rewrites
    // its one-word state file in place, so readers occasionally see "" for a
    // frame; deriving flags from that blinked the clock/label and reset the
    // session. Everything below reads `vstate`, never daemonState directly.
    property string vstate: "idle"
    function _isActive(s) { return s === "recording" || s === "streaming" || s === "transcribing"; }
    readonly property bool active: _isActive(vstate)
    readonly property bool listening: vstate === "recording" || vstate === "streaming"
    readonly property bool polishing: vstate === "transcribing"
    // Motion tokens (Emil Kowalski's curves). Dictation starts from a hotkey
    // tens of times a day, so nothing bounces and nothing exceeds ~220ms;
    // the morph is the one motion with a job (same object, new state).
    readonly property var easeOut:   [0.23, 1, 0.32, 1, 1, 1]      // enter / exit
    readonly property var easeInOut: [0.77, 0, 0.175, 1, 1, 1]     // on-screen morph
    // Size and scroll use a strong ease-out (200/220/240ms), not a spring. The
    // earlier spring (K=5, damping 0.45) measured fast in isolation, but in
    // the live pill the widening read as a slow wipe in screenshot bursts and
    // cut each new word in letter by letter; ease-out covers most of the
    // distance in the first frames and 200ms finishes before the next chunk
    // (~300ms), so retargets start from rest.
    // No system-wide reduced-motion signal on Hyprland; opt in per user.
    // Reduced = opacity/color only, no scale, rise or morph.
    readonly property bool reducedMotion: Quickshell.env("VOXTYPE_OSD_REDUCED_MOTION") === "1"
    // Liquid glass needs the compositor to blur behind the pill (Hyprland:
    // a `layerrule` with blur on and ignore_alpha 0.5 for namespace
    // voxtype-osd). Without it, or with VOXTYPE_OSD_REDUCED_TRANSPARENCY=1,
    // the tint goes back to near-solid so text never sits on raw content.
    readonly property bool reducedTransparency: Quickshell.env("VOXTYPE_OSD_REDUCED_TRANSPARENCY") === "1"

    // ------------------------------------------------------------------
    // Transcript transport
    property string liveText: ""
    // What the pill shows. Follows liveText while a session is active and
    // freezes through the exit, because the watcher empties live_text the
    // moment the state goes idle, which would collapse the pill mid-exit.
    property string shownText: ""
    // Change handlers test vstate directly: a sibling binding such as
    // `active` can still hold its old value while this handler runs.
    // While polishing, live_text is only ever replaced, never legitimately
    // emptied: an empty write there is the end-of-session clear, which can
    // land before the idle state does (separate file watches), and taking
    // it folded the pill back to "Polishing" under the delivered check.
    onLiveTextChanged: if (_isActive(vstate) && (liveText !== "" || vstate !== "transcribing")) shownText = liveText

    // ------------------------------------------------------------------
    // Exit and delivery confirmation
    // The surface used to vanish the instant the daemon went idle. Now the
    // pill holds it (`lingering`, read by OsdSurface) for a short exit, and
    // when the transcript was actually delivered, a check draws itself in
    // the meter's place first. "Delivered" comes from the daemon's
    // output_done marker (written only on success), never from the idle
    // edge, which also ends cancelled, empty and failed sessions.
    property bool lingering: false
    property bool confirmed: false
    property double outputDoneAt: 0
    readonly property int confirmHoldMs: 420
    // Delivered, but as heard: every cleanup provider failed and
    // voxtype-postprocess.sh fell through to the raw transcript. The check
    // turns amber and the pill says so, held long enough to read, instead of
    // passing for a cleaned insert ("never a silent fallback").
    property bool deliveredRaw: false
    property double ppResultAt: 0
    property string ppVia: ""
    readonly property int rawHoldMs: 1400
    // Cancelled (Escape / SUPER+SHIFT+D via voxtype-cancel.sh, which writes
    // `cancelled`): the transcript folds away, the bars lie flat and the
    // pill says so briefly, instead of vanishing like an empty session.
    property bool wasCancelled: false
    property double cancelledAt: 0
    readonly property int cancelHoldMs: 450
    // First Escape arms cancel for 600ms (voxtype-cancel.sh writes
    // `cancel_armed`); the pill says how to finish it.
    property bool cancelArmed: false
    FileView {
        path: root._runtimeDir.length > 0 ? root._runtimeDir + "/voxtype/cancel_armed" : ""
        watchChanges: true
        printErrors: false
        onFileChanged: reload()
        onLoaded: {
            const ms = Number((text() || "").trim());
            if (ms > 0 && root.active && ms >= root.localStart - 100 && Date.now() - ms < 600) {
                root.cancelArmed = true;
                armedTimer.interval = Math.max(1, 600 - (Date.now() - ms));
                armedTimer.restart();
            }
        }
    }
    Timer { id: armedTimer; onTriggered: root.cancelArmed = false }

    FileView {
        path: root._runtimeDir.length > 0 ? root._runtimeDir + "/voxtype/cancelled" : ""
        watchChanges: true
        printErrors: false
        onFileChanged: reload()
        onLoaded: {
            const ms = Number((text() || "").trim());
            if (ms > 0) root.cancelledAt = ms;
        }
    }

    FileView {
        path: root._runtimeDir.length > 0 ? root._runtimeDir + "/voxtype/pp_result" : ""
        watchChanges: true
        printErrors: false
        onFileChanged: reload()
        onLoaded: {
            const parts = (text() || "").trim().split(" ");
            const ms = Number(parts[0]);
            if (ms > 0) { root.ppResultAt = ms; root.ppVia = parts[1] || ""; }
        }
    }

    FileView {
        path: root._runtimeDir.length > 0 ? root._runtimeDir + "/voxtype/output_done" : ""
        watchChanges: true
        printErrors: false
        onFileChanged: reload()
        onLoaded: {
            const ms = Number((text() || "").trim().split(" ")[0]);
            if (ms > 0) root.outputDoneAt = ms;
        }
    }

    readonly property string _runtimeDir: {
        const xdg = Quickshell.env("XDG_RUNTIME_DIR");
        if (xdg && xdg.length > 0) return xdg;
        const uid = Quickshell.env("UID");
        return (uid && uid.length > 0) ? "/run/user/" + uid : "";
    }

    FileView {
        path: root._runtimeDir.length > 0 ? root._runtimeDir + "/voxtype/live_text" : ""
        watchChanges: true
        printErrors: false
        onLoaded: root.liveText = (text() || "").trim()
        onLoadFailed: root.liveText = ""
        onFileChanged: reload()
    }

    // Only the last two lines are visible (the transcript scrolls inside a
    // clipped viewport), so cap what gets laid out on long dictations. The
    // cut lands far above the visible lines; it is never on screen.
    function _cap(text) {
        const t = (text || "").trim();
        if (t.length <= 600) return t;
        const cut = t.indexOf(" ", t.length - 600);
        return t.substring(cut > 0 ? cut + 1 : t.length - 600);
    }
    readonly property string transcriptShown: _cap(shownText)
    readonly property bool hasText: transcriptShown.length > 0

    // ------------------------------------------------------------------
    // Streaming word reveal
    // Modelled on kobra.systems' streaming-word: every word that arrives
    // fades in over 180ms while its colour washes blue -> violet -> pink and
    // settles into the text colour, so the leading edge of the sentence
    // trails a short soft glow instead of words popping in. Nothing moves
    // (opacity and colour only), so it costs no layout. Words of one chunk
    // are staggered ~32ms apart (capped at 100ms for the whole chunk) to
    // read as a wave, never as added latency. A revised earlier word, a cap
    // slide, or reduced motion swaps instantly: fading rewritten words
    // reads as flicker.
    readonly property int revealMs: 180
    readonly property int revealStaggerMs: 32
    readonly property int revealStaggerCapMs: 100
    readonly property var washStops: [[115, 189, 255], [173, 141, 253], [234, 145, 188]]   // oklch 78/.13/250, 72/.16/295, 76/.12/350
    property var _words: []
    property var _births: []          // epoch ms each word starts revealing; 0 = settled
    property double _lastBirth: 0
    property double nowMs: Date.now()
    readonly property bool revealing: _lastBirth > 0 && nowMs < _lastBirth + revealMs
    function _esc(t) { return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
    function _trackWords(t) {
        const words = t.length > 0 ? t.split(" ") : [];
        const pw = _words;
        let append = !reducedMotion && words.length >= pw.length;
        for (let i = 0; append && i < pw.length; i++) {
            // the last previous word may still be growing ("hel" -> "hello")
            if (i < pw.length - 1 ? pw[i] !== words[i] : words[i].indexOf(pw[i]) !== 0) append = false;
        }
        let births = [];
        if (!append) {
            births = new Array(words.length).fill(0);
        } else {
            births = _births.slice(0, pw.length);
            const n = words.length - pw.length;
            if (n > 0) {
                const step = Math.min(revealStaggerMs, revealStaggerCapMs / n);
                const base = Date.now();
                for (let i = 0; i < n; i++) births.push(base + i * step);
                _lastBirth = births[births.length - 1];
            }
        }
        _words = words;
        _births = births;
        nowMs = Date.now();
    }
    onTranscriptShownChanged: _trackWords(transcriptShown)
    Timer { interval: 16; repeat: true; running: root.revealing; onTriggered: root.nowMs = Date.now() }
    function _wash(p) {     // 0..1 progress -> "#aarrggbb" for that moment of the reveal
        const a = 1 - Math.pow(1 - p, 2.4);                       // ease: quick start, soft landing
        const stops = washStops;
        const end = [fg.r * 255, fg.g * 255, fg.b * 255];
        let from, to, f;
        if (p < 0.3)      { from = stops[0]; to = stops[1]; f = p / 0.3; }
        else if (p < 0.6) { from = stops[1]; to = stops[2]; f = (p - 0.3) / 0.3; }
        else              { from = stops[2]; to = end;      f = (p - 0.6) / 0.4; }
        const c = (i) => (from[i] + (to[i] - from[i]) * f) / 255;
        return Qt.rgba(c(0), c(1), c(2), a).toString();
    }
    readonly property string transcriptMarkup: {
        const w = _words, b = _births, now = nowMs;
        if (!revealing || w.length === 0 || w.join(" ") !== transcriptShown) return _esc(transcriptShown);
        let out = "", plain = "";
        for (let i = 0; i < w.length; i++) {
            const sep = i > 0 ? " " : "";
            const p = b[i] > 0 ? Math.max(0, (now - b[i]) / revealMs) : 1;
            if (p >= 1) { plain += sep + _esc(w[i]); continue; }
            out += plain + sep + '<font color="' + _wash(p) + '">' + _esc(w[i]) + '</font>';
            plain = "";
        }
        return out + plain;
    }
    onHasTextChanged: if (!hasText) { transcript.resetOverflow(); card.peakTextW = 0; }

    // ------------------------------------------------------------------
    // Session clock
    // localStart: when this OSD saw the session begin (late by spawn time).
    // fileStart:  watcher's timestamp (≤150ms late). Prefer fileStart when it
    // belongs to this session, i.e. lands within 3s of localStart.
    property double localStart: 0
    property double fileStart: 0
    readonly property double startedAt:
        (fileStart > 0 && Math.abs(localStart - fileStart) < 3000) ? fileStart : localStart
    property int elapsed: 0

    function _clock(s) {
        const m = Math.floor(s / 60);
        const r = s % 60;
        return m + ":" + (r < 10 ? "0" : "") + r;
    }

    // Authoritative start time from the watcher (epoch ms, written on the
    // idle→active edge). The OSD process spawns ~0.2–0.6s after recording
    // starts, so a QML-side Date.now() would run late.
    FileView {
        path: root._runtimeDir.length > 0 ? root._runtimeDir + "/voxtype/session_start" : ""
        watchChanges: true
        printErrors: false
        onFileChanged: reload()
        onLoaded: {
            const ms = Number((text() || "").trim());
            if (ms > 0) root.fileStart = ms;
        }
    }

    Timer {
        interval: 250
        repeat: true
        triggeredOnStart: true
        running: root.listening
        onTriggered: {
            if (root.localStart <= 0) root.localStart = Date.now();
            root.elapsed = Math.floor((Date.now() - root.startedAt) / 1000);
        }
    }

    function _applyState(next) {
        if (next === "" || next === vstate) return;
        const wasActive = _isActive(vstate);
        vstate = next;
        if (!_isActive(next)) {
            if (wasActive) {
                lingering = true;
                exitGrace.restart();
            } else {
                _finishExit();
            }
        } else if (!wasActive) {
            shownAccent = _accentFor(next);
            shownLabel = _labelFor(next);
            // Fresh session (possibly cutting a previous exit short): clear
            // leftovers, restart clock, play entrance.
            exitGrace.stop();
            exitSeq.stop();
            confirmed = false;
            deliveredRaw = false;
            wasCancelled = false;
            cancelArmed = false;
            badge.gather = 0; badge.pop = 0; badge.draw = 0;
            lingering = false;
            liveText = "";
            shownText = "";
            localStart = Date.now();
            elapsed = 0;
            level = 0;
            ripple = [];
            speechAt = 0;
            signalAt = 0;
            speaking = false;
            micDead = false;
            tick = 0;
            entrance.restart();
        }
    }

    function _finishExit() {
        lingering = false;
        confirmed = false;
        deliveredRaw = false;
        wasCancelled = false;
        liveText = "";
        shownText = "";
        level = 0;
    }

    // The output marker and the idle state arrive through separate file
    // watches, in either order; give the marker a moment before deciding.
    Timer {
        id: exitGrace
        interval: 80
        onTriggered: {
            const fresh = Date.now() - root.outputDoneAt < 2500;
            const delivered = fresh && root.outputDoneAt >= root.localStart - 100;
            // Before `confirmed`: the check paints its colour as it starts drawing.
            root.deliveredRaw = delivered && root.ppVia === "passthrough"
                && root.ppResultAt >= root.localStart - 100;
            root.wasCancelled = !delivered && root.cancelledAt >= root.localStart - 100;
            root.cancelArmed = false;
            root.confirmed = delivered;
            if (root.wasCancelled) root.shownText = "";
            // The warning replaces the transcript, so the pill folds to the
            // label while it holds.
            if (root.deliveredRaw) root.shownText = "";
            exitSeq.start();
        }
    }

    // Hold briefly on a confirmation, then leave the way it entered: the
    // contents fade, the pill shrinks back into the dot it grew from, and the
    // dot fades in the last frames. Shorter than the entrance (240 vs 280ms).
    // Reduced motion: fade only.
    SequentialAnimation {
        id: exitSeq
        PauseAnimation { duration: root.deliveredRaw ? root.rawHoldMs : root.confirmed ? root.confirmHoldMs : root.wasCancelled ? root.cancelHoldMs : 0 }
        ScriptAction { script: card.settled = false }
        ParallelAnimation {
            NumberAnimation { target: card; property: "morph"; to: root.reducedMotion ? 1.0 : 0.0; duration: root.reducedMotion ? 0 : 240; easing.type: Easing.BezierSpline; easing.bezierCurve: root.easeInOut }
            SequentialAnimation {
                PauseAnimation { duration: root.reducedMotion ? 0 : 150 }
                NumberAnimation { target: card; property: "opacity"; to: 0.0; duration: root.reducedMotion ? 150 : 90; easing.type: Easing.BezierSpline; easing.bezierCurve: root.easeOut }
            }
        }
        ScriptAction { script: root._finishExit() }
    }
    onConfirmedChanged: if (confirmed) badge.play()
    // The OSD process is often spawned with daemonState already "recording",
    // so the change handler alone would miss the first session's edge.
    Component.onCompleted: _applyState(daemonState)
    onDaemonStateChanged: _applyState(daemonState)

    // ------------------------------------------------------------------
    // Level meter — modelled on ElevenLabs' BarVisualizer
    // (ui.elevenlabs.io/docs/components/bar-visualizer). Bar heights come
    // from a dB-scaled level; when nothing is being said the bars rest as
    // dim dots and a highlight carries the state instead:
    //   connecting  edge pairs step toward the centre, until audio arrives
    //   listening   the centre bar blinks (500ms): mic open, waiting
    //   speaking    every bar lit in the accent, heights ripple outward
    //   thinking    (Polishing) a highlight scans back and forth
    //
    // The bridge's `peak` is linear amplitude. Speech into a desk mic peaks
    // around -30..-12 dBFS (0.03..0.25 linear), so the old `peak * 1.8` kept
    // the bars under their 3px floor and all you saw while talking was the
    // drift wobble. In dB the room (measured -48..-42 dBFS) sits on the
    // floor and normal speech spans the full height.
    readonly property int barCount: 9
    readonly property int centerBar: Math.floor(barCount / 2)
    readonly property real floorDb: -48
    readonly property real ceilDb: -14
    readonly property real speechLevel: 0.3      // ≈ -38 dBFS on the scale above
    // Centre-weighted, indexed by distance from the centre bar.
    readonly property var profile: [1.0, 0.9, 0.78, 0.64, 0.5]
    property real level: 0
    property var ripple: []              // level history at ~30Hz, newest first
    property double speechAt: 0          // last frame with the level over speechLevel
    property double signalAt: 0          // last frame over digital silence
    property bool speaking: false        // VAD with a hold, so it doesn't flap between words
    property bool micDead: false         // nothing but digital silence: muted or wrong source
    property int tick: 0                 // step of the highlight sequence
    readonly property bool feeding: audio !== null && audio.running
    // `lingering`: the exit keeps the scan; without it the idle edge read
    // as "speaking" for a frame and the bars flared to full height.
    readonly property string meterMode:
        (polishing || lingering) ? "thinking"
      : !feeding  ? "connecting"
      : speaking  ? "speaking"
      :             "listening"

    function _dbLevel(peak) {
        if (peak <= 0) return 0;
        const db = 20 * Math.log10(peak);
        return Math.max(0, Math.min(1, (db - floorDb) / (ceilDb - floorDb)));
    }

    Connections {
        target: root.audio
        enabled: root.audio !== null
        function onFrameReceived(peak, rms, vad, tsMs) {
            const target = root._dbLevel(peak);
            root.level = target > root.level
                ? root.level * 0.3 + target * 0.7
                : root.level * 0.8 + target * 0.2;
            const now = Date.now();
            // Not the bridge's `vad`: it fires at -40 dBFS, right on this
            // room's noise peaks, so keyboard clicks lit the whole meter.
            // The smoothed level crossing ~-38 dBFS ignores single clicks.
            if (root.level > root.speechLevel) root.speechAt = now;
            if (peak > 0.0003) root.signalAt = now;     // -70 dBFS
        }
        function onDisconnected() { root.level = 0; }
    }

    // Shift the level into the ripple (bar d reads 2*d steps back, so a
    // syllable spreads from the centre to the edges in ~260ms) and settle
    // the speech / dead-mic holds.
    Timer {
        interval: 33
        repeat: true
        running: root.listening
        onTriggered: {
            const r = root.ripple.slice(0, 2 * root.centerBar);
            r.unshift(root.level);
            root.ripple = r;
            const now = Date.now();
            root.speaking = now - root.speechAt < 450;
            root.micDead = root.feeding && now - Math.max(root.signalAt, root.startedAt) > 2000;
        }
    }

    Timer {
        interval: root.meterMode === "connecting" ? Math.round(2000 / root.barCount)
                : root.meterMode === "thinking"   ? 90
                :                                   500
        repeat: true
        running: (root.active || root.lingering) && !root.reducedMotion && root.meterMode !== "speaking"
        onTriggered: root.tick++
    }

    function _highlighted(i) {
        if (wasCancelled) return false;
        const n = barCount;
        switch (meterMode) {
        case "speaking":
            return true;
        case "listening":
            return i === centerBar && (reducedMotion || tick % 2 === 0);
        case "connecting": {
            const k = tick % (centerBar + 1);
            return i === k || i === n - 1 - k;
        }
        case "thinking": {
            if (reducedMotion) return true;
            const p = tick % (2 * n - 2);
            return i === (p < n ? p : 2 * n - 2 - p);
        }
        }
        return false;
    }

    function _barHeight(i) {
        const maxH = 16, minH = 3;
        if (wasCancelled) return minH;
        if (meterMode === "thinking") return _highlighted(i) && !reducedMotion ? 8 : minH;
        if (meterMode === "connecting") return minH;
        const d = Math.abs(i - centerBar);
        const v = (ripple.length > 2 * d ? ripple[2 * d] : level) * profile[d];
        return Math.max(minH, Math.min(maxH, minH + v * (maxH - minH)));
    }

    // ------------------------------------------------------------------
    // Theme + config helpers
    function _color(role, fallback) {
        return (theme && theme.color) ? theme.color(role, fallback) : fallback;
    }
    function _cfg(key, fallback) {
        return (theme && theme.config && theme.config[key] !== undefined)
            ? theme.config[key] : fallback;
    }

    readonly property color fg: _color("foreground", "#EDEDF2")
    // Success green for the delivered check: soft enough to sit next to
    // white text on the dark pill without reading as an alert.
    readonly property color checkColor: "#86E0B4"
    readonly property color warnColor: _color("warning", "#F2CC4D")

    // SF Pro is the Dynamic Island face; it can't be redistributed, so fall
    // back to Inter, then IBM Plex Sans, then whatever fontconfig picks.
    readonly property string uiFont: {
        const have = Qt.fontFamilies();
        const chain = ["SF Pro Display", "Inter", "IBM Plex Sans"];
        for (let i = 0; i < chain.length; i++)
            if (have.indexOf(chain[i]) >= 0) return chain[i];
        return "";
    }
    // Listening accent for streaming sessions (the fork's daily mode). Empty =
    // the theme's "streaming" role.
    readonly property string listenAccent: "#FF7A98"
    // Bars keep the session's last color through the exit instead of
    // flipping to the idle fallback while they fade.
    property color shownAccent: accent
    onAccentChanged: if (_isActive(vstate)) shownAccent = accent
    function _accentFor(s) {
        return s === "transcribing" ? _color("transcribing", "#F2CC4D")
             : s === "streaming"    ? (listenAccent !== "" ? listenAccent : _color("streaming", "#66C7FF"))
             :                        _color("recording", "#F2594D");
    }
    readonly property color accent: _accentFor(vstate)

    // Silence is shown by dimming the bars, not by swapping the label — VAD
    // toggles on room noise and a flickering label reads as broken.
    function _labelFor(s) { return s === "transcribing" ? "Polishing" : "Listening"; }
    readonly property string label: _labelFor(vstate)
    property string shownLabel: label
    onLabelChanged: if (_isActive(vstate)) shownLabel = label

    // ------------------------------------------------------------------
    // Geometry
    // Placement from [osd], all in pixels: `position` picks the anchor
    // (top-/bottom- × left/center/right), `margin_px` is the gap to the
    // anchored screen edge(s) measured from the edge itself, so a top pill
    // must clear Waybar on its own, and offset_x/offset_y nudge the result
    // (+x right, +y down). The card is clamped so no offset pushes it
    // off-screen.
    readonly property string pos: String(_cfg("position", "bottom-center"))
    readonly property bool atTop: pos.indexOf("top") === 0
    function _margin() { return Math.max(0, Number(_cfg("margin_px", 10))); }
    readonly property int maxCardWidth: Math.min(440, root.width - 2 * _margin())
    readonly property int padX: 13
    readonly property int gap: 10

    // Shared optical baseline for the single-line state. Centering each Text
    // by its bounding box put the 12px clock visibly higher than the 13px
    // label (different ascent/descent). Instead both sit on one baseline,
    // placed so the CAP height is centered in the pill: baseline = center +
    // capHeight/2 (SF/Inter cap height ≈ 0.70em of the 13px label).
    readonly property real baselineBelowCenter: 13 * 0.35
    function _onBaseline(t) {
        // verticalCenterOffset that puts Text t's baseline on the shared line
        return baselineBelowCenter - (t.baselineOffset - t.implicitHeight / 2);
    }

    function _offset(key) { return Number(_cfg(key, 0)) || 0; }
    function _onScreen(v, room) { return Math.max(0, Math.min(room, v)); }
    function _cardX(w) {
        const x = pos.indexOf("left") >= 0 ? _margin()
                : pos.indexOf("right") >= 0 ? root.width - w - _margin()
                : (root.width - w) / 2;
        return _onScreen(x + _offset("offset_x"), root.width - w);
    }
    function _cardY(h) {
        const y = atTop ? _margin() : root.height - h - _margin();
        return _onScreen(y + _offset("offset_y"), root.height - h);
    }

    // ------------------------------------------------------------------
    // Card
    Rectangle {
        id: card

        readonly property real meterW: meter.width
        // Fixed slot for the clock, sized from the widest possible string.
        // Using clock.implicitWidth made the pill 1px wider/narrower as the
        // digits changed (tnum isn't honoured by this font/renderer), which
        // re-centred the pill and replayed the width morph every second.
        // Held for as long as a transcript is on screen, not just while the
        // clock shows: releasing it at stop widened the wrap area, so words
        // jumped back up a line the moment recording ended.
        readonly property real trailW: (clock.wanted || root.hasText) ? clock.width + root.gap + 4 : 0
        readonly property real textMaxW: root.maxCardWidth - 2 * root.padX - meterW - root.gap - trailW

        // Targets come from the text's final layout; the ease-out animates the
        // card toward them, and the clipped content viewport follows the
        // card's *animated* size, so new words are revealed by the growing
        // edge instead of spilling past it.
        readonly property real chrome: 2 * root.padX + meterW + root.gap + trailW
        // Natural (unwrapped) width from metrics, not transcript.implicitWidth:
        // the transcript narrows its own lines to make room for the leading
        // "…", which would feed back into its implicit width (binding loop).
        // (+slack: advanceWidth omits letter-spacing, and a pixel short wraps
        // the last word of a one-line caption.)
        readonly property real naturalW: Math.ceil(naturalText.advanceWidth + transcript.font.letterSpacing * root.transcriptShown.length) + 2
        readonly property real textW: Math.min(naturalW, textMaxW)
        TextMetrics { id: naturalText; font: transcript.font; text: root.transcriptShown }
        // Sized from the lines' real geometry (pitch, textH), not contentHeight:
        // once a lineLaidOut handler is connected, Qt reports contentHeight
        // without the proportional line spacing, so the pill came out ~3px short
        // and the last line sat low and was clipped.
        readonly property real lineH: transcript.pitch
        readonly property real viewH: Math.min(transcript.textH, 2 * lineH)
        // Width hugs the words but moves in a few tiers, not per word: it
        // steps up through 22/42/62/82/100% of the text area, so a short
        // "Hello." gets a small pill and a long dictation resizes about four
        // times in all (each one 250ms, edges anchored in between) instead
        // of twitching on every chunk. It only ever grows (peakTextW), so a
        // revised, shorter transcript can't make the pill breathe. Reset
        // with the session (hasText -> false). (Was a flat 60% minimum: no
        // twitching, but short phrases sat in a mostly empty pill.)
        property real peakTextW: 0
        onTextWChanged: if (root.hasText && textW > peakTextW) peakTextW = textW
        readonly property var widthTiers: [0.22, 0.42, 0.62, 0.82, 1.0]
        function _tier(w) {
            for (let i = 0; i < widthTiers.length; i++)
                if (w <= widthTiers[i] * textMaxW) return Math.ceil(widthTiers[i] * textMaxW);
            return textMaxW;
        }
        readonly property real holdTextW: _tier(Math.max(textW, peakTextW))
        readonly property real fullW: Math.min(root.maxCardWidth,
                        chrome + (root.hasText ? holdTextW : status.implicitWidth))
        readonly property real fullH: Math.max(34, (root.hasText ? viewH : status.implicitHeight) + 18)
        // Open/close morph, Dynamic Island style: 0 is a 12px dot at the
        // pill's centre, 1 the full pill. The contents only fade in over the
        // last 40% so they never ride on a half-grown pill (and the clip below
        // hides whatever does not fit yet).
        property real morph: 0
        readonly property real dot: 12
        readonly property real reveal: Math.max(0, Math.min(1, (morph - 0.6) / 0.4))
        width: dot + (fullW - dot) * morph
        height: dot + (fullH - dot) * morph
        radius: Math.min(height / 2, 18)
        clip: true
        // Whole pixels: fractional centring made edges shimmer between frames.
        x: Math.round(root._cardX(width))
        y: Math.round(root._cardY(height) + dropOffset)
        property real dropOffset: 0

        // Glass: a dark tint over the compositor's blur. 0.58 stays above
        // the blur rule's ignore_alpha (0.5), so the whole pill is blurred.
        color: Qt.alpha(root._color("background", "#111114"), root.reducedTransparency ? 0.95 : 0.58)
        border.width: 1
        // Briefly tinted by the delivered badge (flash 0..1), else a hairline.
        property real flash: 0
        border.color: Qt.tint(Qt.alpha("#ffffff", root.reducedTransparency ? 0.08 : 0.11), Qt.alpha(badge.tone, 0.4 * flash))

        // Soft contact shadow so the pill floats over content.
        layer.enabled: true
        layer.effect: MultiEffect {
            shadowEnabled: true
            shadowColor: "#000000"
            shadowOpacity: 0.42
            shadowBlur: 0.55
            shadowVerticalOffset: 3
            autoPaddingEnabled: true
        }

        // Morph only after the entrance: animating from the initial 0 width
        // showed bare text with no pill behind it for the first frames.
        property bool settled: false
        // Streaming morph: strong ease-out, not the entrance spring. The
        // spring took ~0.7s to widen and bounced, so each new word was wiped in
        // letter by letter by the growing edge. Ease-out covers most of the
        // distance in the first frames (the words are revealed fast, and the
        // tail fade carries the softness) and retargets cleanly when the next
        // chunk arrives mid-flight.
        Behavior on width  { enabled: card.settled && !root.reducedMotion; NumberAnimation { duration: 250; easing.type: Easing.BezierSpline; easing.bezierCurve: root.easeOut } }
        Behavior on height { enabled: card.settled && !root.reducedMotion; NumberAnimation { duration: 220; easing.type: Easing.BezierSpline; easing.bezierCurve: root.easeOut } }

        // Entrance: grows out of a dot (never from nothing: the dot is
        // visible from the first frame), 280ms strong ease-out, no overshoot,
        // since it's hotkey-triggered tens of times a day. No `from`, so a
        // session that starts mid-exit grows from wherever the shrink was.
        ParallelAnimation {
            id: entrance
            onStarted: card.settled = false
            onFinished: card.settled = true
            NumberAnimation { target: card; property: "morph";   to: 1.0; duration: root.reducedMotion ? 0 : 280; easing.type: Easing.BezierSpline; easing.bezierCurve: root.easeOut }
            NumberAnimation { target: card; property: "opacity"; to: 1.0; duration: 100; easing.type: Easing.BezierSpline; easing.bezierCurve: root.easeOut }
        }

        // Light on the glass: a sheen falling from the top edge, brightest in
        // its first pixel so the top of the rim reads lit and the bottom
        // stays dark, as if lit from above. Under the contents, inside the
        // rim (inset 1px).
        Rectangle {
            anchors.fill: parent
            anchors.margins: 1
            radius: Math.max(0, card.radius - 1)
            visible: !root.reducedTransparency
            gradient: Gradient {
                GradientStop { position: 0.0;  color: Qt.alpha("#ffffff", 0.16) }
                GradientStop { position: 0.06; color: Qt.alpha("#ffffff", 0.07) }
                GradientStop { position: 0.5;  color: Qt.alpha("#ffffff", 0.02) }
                GradientStop { position: 1.0;  color: Qt.alpha("#ffffff", 0.0) }
            }
        }

        // Everything inside the pill fades with the open/close morph. While
        // the pill opens or closes the contents keep their full-size layout,
        // centred, and the pill's clip reveals them (squeezing them with the
        // width ran the label into the clock). Once settled they follow the
        // pill's animated size, so streaming growth still reveals new words
        // at the moving edge.
        Item {
        id: inner
        anchors.centerIn: parent
        width: card.settled ? card.width : card.fullW
        height: card.settled ? card.height : card.fullH
        opacity: root.reducedMotion ? 1 : card.reveal

        // -- meter ------------------------------------------------------
        Row {
            id: meter
            anchors.left: parent.left
            anchors.leftMargin: root.padX
            anchors.verticalCenter: parent.verticalCenter
            spacing: 2.5
            // Delivered: the bars converge into the badge (see badge.gather),
            // fading only as they arrive at the centre.
            opacity: root.confirmed ? Math.max(0, Math.min(1, (1 - badge.gather) / 0.35)) : 1.0

            Repeater {
                model: root.barCount
                delegate: Rectangle {
                    required property int index
                    width: 3
                    height: root.confirmed ? root._barHeight(index) + (3 - root._barHeight(index)) * badge.gather
                                           : root._barHeight(index)
                    // Slide toward the centre bar (one bar pitch = 3 + 2.5).
                    transform: Translate { x: root.confirmed ? (root.centerBar - index) * 5.5 * badge.gather : 0 }
                    radius: 1.5
                    anchors.verticalCenter: parent.verticalCenter
                    // Unlit bars are neutral, so "speaking" (all lit) reads
                    // clearly against the single blinking bar of "listening".
                    // The 160ms colour fade gives the thinking scan a short tail.
                    color: root.confirmed ? badge.tone
                         : root._highlighted(index) ? root.shownAccent : Qt.alpha(root.fg, 0.22)
                    Behavior on height { enabled: !root.confirmed; NumberAnimation { duration: 90; easing.type: Easing.OutCubic } }
                    Behavior on color  { ColorAnimation  { duration: 160; easing.type: Easing.InOutQuad } }
                }
            }
        }

        // -- delivered badge ------------------------------------------
        // Plays once per dictation (tens a day) as feedback that the text
        // landed, so it stays fast and quiet: done in ~320ms, then held.
        //  - Gather (continuity): the meter becomes the badge. The bars
        //    slide into the centre, flatten to dots and take the badge's
        //    tone (130ms, strong ease-in-out: on-screen movement), fading
        //    only as they converge, so one object changes meaning instead
        //    of one fading out while another pops in.
        //  - Pop in: from that point a tinted disc scales 0.6 -> 1 with a
        //    small overshoot (OutBack 1.2, ~3% past) over 200ms.
        //  - Line drawing: the tick is one continuous stroke drawn by
        //    arc length, so the pen slows into the corner and through the
        //    short leg, then sweeps the long leg and eases onto the tip
        //    (strong ease-in-out over 220ms, starting 50ms in, once the
        //    disc is mostly there). Round caps and joins: a pen, not a wipe.
        //  - The pill's edge takes the same tone for a moment.
        // No halo ring: at this frequency it was decoration.
        // Amber instead of green when the text went in uncleaned.
        // Reduced motion: disc and tick appear in place, no scale or draw.
        Item {
            id: badge
            width: 18
            height: 18
            anchors.centerIn: meter
            visible: root.confirmed
            readonly property color tone: root.deliveredRaw ? root.warnColor : root.checkColor
            property real gather: 0
            property real pop: 0
            property real draw: 0
            function play() {
                badgeIn.stop();
                if (root.reducedMotion) { gather = 1; pop = 1; draw = 1; return; }
                gather = 0; pop = 0; draw = 0;
                badgeIn.start();
            }
            Rectangle {
                anchors.centerIn: parent
                width: parent.width; height: width; radius: width / 2
                color: Qt.alpha(badge.tone, 0.2)
                scale: 0.6 + 0.4 * badge.pop
                opacity: Math.min(1, badge.pop * 2.5)
            }
            Canvas {
                id: check
                anchors.fill: parent
                property real progress: badge.draw
                visible: progress > 0
                onProgressChanged: requestPaint()
                onPaint: {
                    const ctx = getContext("2d");
                    ctx.reset();
                    // short leg down-right, corner, long leg up-right
                    const pts = [[5.3, 9.3], [8.0, 12.0], [13.0, 6.3]];
                    const seg = [Math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]),
                                 Math.hypot(pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])];
                    let left = (seg[0] + seg[1]) * progress;
                    ctx.lineWidth = 2.0;
                    ctx.lineCap = "round";
                    ctx.lineJoin = "round";
                    ctx.strokeStyle = badge.tone;
                    ctx.beginPath();
                    ctx.moveTo(pts[0][0], pts[0][1]);
                    for (let i = 0; i < 2 && left > 0; i++) {
                        const t = Math.min(1, left / seg[i]);
                        ctx.lineTo(pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t,
                                   pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t);
                        left -= seg[i];
                    }
                    ctx.stroke();
                }
            }
            ParallelAnimation {
                id: badgeIn
                NumberAnimation { target: badge; property: "gather"; to: 1; duration: 130; easing.type: Easing.BezierSpline; easing.bezierCurve: root.easeInOut }
                SequentialAnimation {
                    PauseAnimation { duration: 90 }
                    NumberAnimation { target: badge; property: "pop"; to: 1; duration: 200; easing.type: Easing.OutBack; easing.overshoot: 1.2 }
                }
                SequentialAnimation {
                    PauseAnimation { duration: 130 }
                    NumberAnimation { target: badge; property: "draw"; to: 1; duration: 220; easing.type: Easing.BezierSpline; easing.bezierCurve: root.easeInOut }
                }
                SequentialAnimation {
                    NumberAnimation { target: card; property: "flash"; to: 1; duration: 80; easing.type: Easing.BezierSpline; easing.bezierCurve: root.easeOut }
                    NumberAnimation { target: card; property: "flash"; to: 0; duration: 320; easing.type: Easing.BezierSpline; easing.bezierCurve: root.easeOut }
                }
            }
        }

        // -- label or transcript ---------------------------------------
        Item {
            id: content
            anchors.left: meter.right
            anchors.leftMargin: root.gap
            anchors.verticalCenter: parent.verticalCenter
            anchors.verticalCenterOffset: root.hasText ? 0 : root._onBaseline(status)
            // A viewport onto the transcript, sized from the card's animated
            // geometry. Clipping here works because it happens inside the
            // card's layer, before the shadow effect composites it.
            width: Math.max(0, inner.width - card.chrome)
            height: root.hasText ? Math.max(0, inner.height - 18) : status.implicitHeight
            clip: root.hasText
            // Expose the top visible line's real baseline so the clock can
            // anchor to it (Qt's own metrics; the hand-computed offset left the
            // 12px clock 1.5px above the 13px label — measured at 4x zoom).
            // Scroll position is left out on purpose: the clock stays on the
            // top line's baseline while the lines glide past.
            baselineOffset: root.hasText ? transcript.baselineOffset - transcript.lead
                                         : status.y + status.baselineOffset

            Text {
                id: status
                // Crossfade with the transcript (was a hard swap). Centered in
                // content so it doesn't ride up when content grows to 2 lines.
                anchors.verticalCenter: parent.verticalCenter
                visible: opacity > 0
                Behavior on opacity { NumberAnimation { duration: 120; easing.type: Easing.BezierSpline; easing.bezierCurve: root.easeOut } }
                // A muted mic or wrong source sends pure digital silence; say
                // so instead of "Listening" to nothing for the whole take.
                readonly property bool warn: (root.micDead && root.listening) || root.deliveredRaw
                text: root.wasCancelled ? "Cancelled"
                    : root.cancelArmed ? "Esc again to cancel"
                    : root.deliveredRaw ? "Cleanup failed · typed as heard"
                    : root.micDead && root.listening ? "No mic input"
                    : root.shownLabel
                font.family: root.uiFont
                font.pixelSize: 13
                font.weight: Font.Medium
                font.letterSpacing: 0.2
                renderType: Text.NativeRendering
                color: warn ? root.warnColor : root.fg
                // The hint shows over a transcript too (it hides for 600ms).
                opacity: root.cancelArmed ? 0.78 : root.hasText ? 0 : (root.wasCancelled ? 0.55 : 0.78)
            }

            Text {
                id: transcript
                visible: opacity > 0
                // -- leading ellipsis, part of the line --------------------
                // When the dictation runs past the two visible lines, the top
                // line starts mid-sentence. That line is indented by the width
                // of "…" and the dots are drawn in the indent, so they read as
                // the first characters of the sentence, like "…component,
                // import this…". Indenting never re-wraps (only x moves), and
                // once the text wraps every line is laid out `ell` narrower, so
                // the indented line still fits the pill. When a new line begins
                // the indent hands over to the line sliding into the top slot:
                // the old dots ease out as it leaves, the new ones ease in as
                // it arrives, so nothing pops.
                readonly property real ell: Math.ceil(ellipsisSlot.advanceWidth) + 2
                // True while the text is wider than one line. A pure function of
                // the natural width, so it can't stick across sessions or flap.
                readonly property bool reserve: card.naturalW > card.textMaxW
                onReserveChanged: forceLayout()
                // Line geometry as laid out: every line is `pitch` tall, and
                // the first starts `lead` below the top (proportional spacing
                // adds its leading above the glyphs).
                property real pitch: 0
                property real lead: 0
                readonly property real textH: lineCount * pitch
                property int indentLine: -1          // line index at the top of the viewport
                property real indent: 0
                property real indentY: 0
                property int prevLine: -1            // line handing its dots back
                property real prevIndent: 0
                property real prevY: 0

                function resetOverflow() {
                    indentIn.stop(); indentOut.stop();
                    indentLine = -1; indent = 0; indentY = 0;
                    prevLine = -1; prevIndent = 0; prevY = 0;
                }
                function _retarget() {
                    const want = lineCount > 2 ? lineCount - 2 : -1;
                    if (want === indentLine) return;
                    indentOut.stop(); indentIn.stop();
                    prevLine = indentLine; prevIndent = indent; prevY = indentY;
                    indentLine = want;
                    indent = 0;
                    if (prevLine >= 0) { indentOut.from = prevIndent; indentOut.restart(); } else { prevIndent = 0; }
                    if (want >= 0) {
                        if (root.reducedMotion) indent = ell; else indentIn.restart();
                    }
                    forceLayout();
                }
                onLineCountChanged: {
                    _retarget();
                }
                onIndentChanged: forceLayout()
                onPrevIndentChanged: forceLayout()
                onLineLaidOut: line => {
                    if (line.number === 0) {
                        if (pitch !== line.height) pitch = line.height;
                        if (lead !== line.y) lead = line.y;
                    }
                    if (reserve) line.width = width - ell;
                    if (line.number === indentLine) { line.x = indent; indentY = line.y; }
                    else if (line.number === prevLine) { line.x = prevIndent; prevY = line.y; }
                }
                NumberAnimation { id: indentIn; target: transcript; property: "indent"; from: 0; to: transcript.ell; duration: 240; easing.type: Easing.BezierSpline; easing.bezierCurve: root.easeOut }
                NumberAnimation { id: indentOut; target: transcript; property: "prevIndent"; to: 0; duration: 240; easing.type: Easing.BezierSpline; easing.bezierCurve: root.easeOut }
                TextMetrics { id: ellipsisSlot; font: transcript.font; text: "\u2026" }
                Text {
                    x: 0
                    y: transcript.indentY
                    visible: opacity > 0
                    opacity: transcript.ell > 0 ? 0.62 * transcript.indent / transcript.ell : 0
                    text: "\u2026"; font: transcript.font; lineHeight: transcript.lineHeight
                    renderType: Text.NativeRendering; color: root.fg
                }
                Text {
                    x: 0
                    y: transcript.prevY
                    visible: opacity > 0
                    opacity: transcript.ell > 0 ? 0.62 * transcript.prevIndent / transcript.ell : 0
                    text: "\u2026"; font: transcript.font; lineHeight: transcript.lineHeight
                    renderType: Text.NativeRendering; color: root.fg
                }
                // The final width, not the animating viewport's: wrapping to a
                // width that changes every frame would reflow mid-glide.
                width: card.textW
                text: root.transcriptMarkup
                textFormat: Text.StyledText
                wrapMode: Text.WordWrap
                lineHeight: 1.16
                font.family: root.uiFont
                font.pixelSize: 14
                font.letterSpacing: 0.15
                renderType: Text.NativeRendering
                color: root.fg
                opacity: !root.hasText || root.cancelArmed ? 0 : (root.polishing && !shimmer.on ? 0.72 : 1.0)
                // The viewport clips the text to the opening pill, so it can
                // fade in with the morph instead of waiting for it.
                Behavior on opacity { NumberAnimation { duration: 140; easing.type: Easing.BezierSpline; easing.bezierCurve: root.easeOut } }

                // Caption scroll: lines keep their place and the block glides
                // up one line when a third begins, instead of words shifting
                // left on every update as a character-count tail did.
                y: root.hasText ? -lead + Math.min(0, card.viewH - textH) : 0
                Behavior on y { enabled: card.settled && !root.reducedMotion && root.hasText; NumberAnimation { duration: 240; easing.type: Easing.BezierSpline; easing.bezierCurve: root.easeOut } }
            }

            // Lines scrolled out of view leave through a short fade, not a
            // hard clip edge.
            Rectangle {
                anchors.top: parent.top
                anchors.left: parent.left
                anchors.right: parent.right
                height: 7
                visible: opacity > 0
                opacity: root.hasText && transcript.y < -transcript.lead - 1 ? 1 : 0
                Behavior on opacity { NumberAnimation { duration: 140; easing.type: Easing.BezierSpline; easing.bezierCurve: root.easeOut } }
                gradient: Gradient {
                    GradientStop { position: 0.0; color: card.color }
                    GradientStop { position: 1.0; color: Qt.alpha(card.color, 0) }
                }
            }
        }

        // -- polishing shimmer --------------------------------------------
        // ElevenLabs "Shimmering Text": while the cleanup runs, whatever the
        // pill says (transcript or "Polishing") rests dimmed and a soft band
        // of full-strength text sweeps across it, left to right, ~1.1s a
        // pass. Drawn from a live texture of `content`, so the transcript's
        // own layout (indents, scroll, ellipsis) is shimmered as-is.
        // Reduced motion: the plain 0.72 dim instead.
        ShaderEffectSource {
            id: contentTex
            sourceItem: content
            hideSource: shimmer.on
            visible: false
        }
        Item {
            id: shimmer
            readonly property bool on: root.polishing && !root.reducedMotion
            anchors.fill: content
            visible: on
            property real phase: 0
            NumberAnimation on phase { from: 0; to: 1; duration: 1100; loops: Animation.Infinite; running: shimmer.on }
            MultiEffect {
                anchors.fill: parent
                source: contentTex
                opacity: shimmer.on ? 0.45 : 1.0
                Behavior on opacity { NumberAnimation { duration: 200; easing.type: Easing.BezierSpline; easing.bezierCurve: root.easeOut } }
            }
            MultiEffect {
                anchors.fill: parent
                source: contentTex
                maskEnabled: true
                maskSource: band
                // Threshold 0 passes even fully transparent mask pixels, so
                // the whole text lit up; ramp around 0.5 instead.
                maskThresholdMin: 0.5
                maskSpreadAtMin: 1.0
            }
            Item {
                id: band
                anchors.fill: parent
                layer.enabled: true
                visible: false
                Rectangle {
                    width: Math.max(48, parent.width * 0.4)
                    height: parent.height
                    x: -width + (parent.width + width) * shimmer.phase
                    gradient: Gradient {
                        orientation: Gradient.Horizontal
                        GradientStop { position: 0.0; color: "#00ffffff" }
                        GradientStop { position: 0.5; color: "#ffffffff" }
                        GradientStop { position: 1.0; color: "#00ffffff" }
                    }
                }
            }
        }

        // -- clock ------------------------------------------------------
        Text {
            id: clock
            readonly property bool wanted: root.listening
            visible: opacity > 0
            opacity: wanted ? 0.5 : 0
            Behavior on opacity { NumberAnimation { duration: 160; easing.type: Easing.BezierSpline; easing.bezierCurve: root.easeOut } }
            anchors.right: parent.right
            anchors.rightMargin: root.padX
            anchors.baseline: content.baseline
            text: root._clock(root.elapsed)
            width: Math.ceil(clockSlot.advanceWidth)
            horizontalAlignment: Text.AlignRight
            font.family: root.uiFont
            // Same size as the label: at 1.6x scale a 12px clock rounded to a
            // baseline ~0.75px above the 13px label even when baseline-anchored.
            // Secondary emphasis comes from opacity, not size.
            font.pixelSize: 13
            font.weight: Font.Medium
            font.features: { "tnum": 1 }
            renderType: Text.NativeRendering
            color: root.fg

            TextMetrics {
                id: clockSlot
                font: clock.font
                // widest digits for the current magnitude; grows once at 10:00
                text: root.elapsed >= 600 ? "00:00" : "0:00"
            }
        }
        }
    }
}
