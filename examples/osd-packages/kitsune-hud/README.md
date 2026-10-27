# Kitsune HUD

`kitsune-hud` is a compact dictation pill for the Quickshell OSD: a level
meter, the state, a session clock and, with streaming, the live transcript as
rolling captions. It grows from a one-line pill to two lines of text and
scrolls like captions on long dictations.

```
[ ▂▅█▅▂ ]  Listening                 0:14
[ ▂▅█▅▂ ]  …so the idea is we cache the tokens
```

## Meter

The bar meter follows the states of ElevenLabs' BarVisualizer
(ui.elevenlabs.io/docs/components/bar-visualizer):

| State | Bars |
|---|---|
| connecting | edge pairs step toward the centre until the first audio frame |
| listening | dim dots; the centre bar blinks every 500ms |
| speaking | all bars lit, heights ripple out from the centre |
| polishing | a highlight scans back and forth while post-processing runs |

Heights come from the audio bridge's peak mapped in dBFS (-48 to -14), not
linear amplitude: speech into a desk mic peaks around -30 to -12 dBFS
(0.03–0.25 linear), which kept a linear meter at its floor. "Speaking" is the
smoothed level crossing about -38 dBFS, held 450ms between words. Tune
`floorDb`, `ceilDb` and `speechLevel` in `KitsuneHud.qml` for your mic.

## Motion

- **Open / close:** the pill grows out of a 12px dot at its centre (280ms
  strong ease-out, no overshoot: it opens from a hotkey many times a day) and
  shrinks back into it on exit (240ms), the contents fading over the last part
  of the open and the first part of the close.
- **Polishing shimmer:** while post-processing runs, the transcript (or the
  "Polishing" label) rests dimmed and a soft band of full-strength text sweeps
  across it, after ElevenLabs' Shimmering Text. Drawn from a live texture of
  the text, so the caption layout is shimmered as-is.
- `VOXTYPE_OSD_REDUCED_MOTION=1` keeps opacity and colour changes only.

## Feedback

- **Delivered:** when the daemon writes `output_done`, a tinted disc pops into
  the meter's place, the check draws itself over it, a halo ring spreads once
  and the pill's edge flashes the same tone, then the pill closes.
- **Delivered raw:** if a post-process command writes
  `$XDG_RUNTIME_DIR/voxtype/pp_result` as `<epoch ms> passthrough`, the check
  is amber and the pill reads "Cleanup failed · typed as heard". Without that
  file the pill shows the normal check.
- **No mic input:** two seconds of digital silence (a muted mic or the wrong
  source) replace "Listening" with a warning.

## Inputs

- `daemonState`, `audio`, `theme`: from the OSD host.
- `$XDG_RUNTIME_DIR/voxtype/live_text`: the transcript (`[output]
  streaming_preview_file = "auto"`).
- `session_start`, `output_done`, `pp_result`: optional; each is ignored
  when absent.

Placement comes from `[osd]` `position`, `margin_px`, `offset_x` and
`offset_y`. Set `VOXTYPE_OSD_REDUCED_MOTION=1` for opacity and colour changes
only.

## Run

```toml
[osd]
frontend = "quickshell"
style = "/path/to/examples/osd-packages/kitsune-hud"
```

Custom QML runs with the same trust level as local shell scripts. Review third
party packages before enabling them.
