#!/bin/bash
# VoiceStudio launcher — bypasses FUSE requirement by running from extracted directory
# Usage: ./run_voicestudio.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
# Extraction lives at ~/vs_extract, not inside voice_studio/
VS_EXTRACT="$HOME/vs_extract/squashfs-root"

if [ ! -d "$VS_EXTRACT" ]; then
    echo "ERROR: Extracted VoiceStudio not found at $VS_EXTRACT"
    echo "Download and extract the AppImage first:"
    echo "  cd $(dirname "$SCRIPT_DIR") && ./VoiceStudio-*.AppImage --appimage-extract"
    exit 1
fi

# Ensure sandbox permissions
if [ -f "$VS_EXTRACT/chrome-sandbox" ]; then
    sudo chown root:root "$VS_EXTRACT/chrome-sandbox" 2>/dev/null
    sudo chmod 4755 "$VS_EXTRACT/chrome-sandbox" 2>/dev/null
fi

cd "$VS_EXTRACT"
./voicestudio-electron --no-sandbox "$@"
