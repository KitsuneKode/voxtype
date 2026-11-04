#!/usr/bin/env bash
# Build the wtype that voxtype's type mode uses and install it to ~/.local/bin
# (ahead of /usr/bin on the daemon's PATH).
#
# Why not the packaged wtype 0.4:
# - It drops apostrophes in Chromium/Electron apps ("It's" -> "Its"); fixed
#   upstream after 0.4 (printable keycodes for literal text, 82bca82).
# - It sleeps a hard-coded 2 ms after every key press and release, capping
#   typing at ~150-240 chars/s (a 3-minute dictation took 16 s to type).
#   scripts/wtype-key-gap.patch makes that gap WTYPE_KEY_GAP_US (default
#   2000, so other callers behave as before); the voxtype service sets it in
#   ~/.config/systemd/user/voxtype.service.d/wtype.conf.
# - Explicit keys (-k Return, -P Shift_L) got logical code N as their evdev
#   code, so voxtype's Shift+Enter reached Chromium as code=Escape. The patch
#   puts named keys on their real positions (Enter is code=Enter now). Note:
#   in a nested-Hyprland test Chromium still did not insert a line break for
#   Enter (or move focus on Tab) even with the right code; unverified on the
#   real session.
#
# Measured (2189 chars, nested Hyprland, Electron 43 textarea and a
# contenteditable doing work per keystroke, kitty): exact at every gap
# down to 0; 2000us = 240 c/s, 200us = ~2100 c/s, 0 = ~28000 c/s.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
rev=${WTYPE_REV:-4b9cc29}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

git clone -q https://github.com/atx/wtype "$work/wtype"
cd "$work/wtype"
git checkout -q "$rev"
git apply "$here/wtype-key-gap.patch"
mkdir -p b
proto=protocol/virtual-keyboard-unstable-v1.xml
wayland-scanner client-header "$proto" b/virtual-keyboard-unstable-v1-client-protocol.h
wayland-scanner private-code "$proto" b/virtual-keyboard-unstable-v1-protocol.c
gcc -O2 -Ib -o b/wtype main.c b/virtual-keyboard-unstable-v1-protocol.c \
    $(pkg-config --cflags --libs xkbcommon wayland-client)
install -Dm755 b/wtype "$HOME/.local/bin/wtype"
echo "installed $HOME/.local/bin/wtype (wtype $rev + key-gap patch)"
