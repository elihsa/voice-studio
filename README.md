# Voice Studio — WhatsApp Voice Cloning Setup

## Location
`/home/eli/voice_studio/`

## Components

### 1. VoiceStudio App (v0.5.8)
- **AppImage**: `/home/eli/.local/bin/VoiceStudio` (already installed)
- **Extracted**: `/home/eli/vs_extract/squashfs-root/`
- **Launcher**: `./run_voicestudio.sh`

VoiceStudio is a desktop Electron app with 16 TTS engines including **GPT-SoVITS** — the best engine for WhatsApp audio (tolerant of compressed/noisy source).

### 2. WhatsApp Pre-Processor
`./whatsapp_preprocess.py`

Batch processes WhatsApp voice notes (OGG/OPUS) into clean WAV files optimized for voice cloning.

**Pipeline:**
1. Convert OGG/OPUS → WAV 24kHz mono (FFmpeg)
2. Noise reduction (spectral subtraction)
3. Trim silence
4. De-ess (reduce harsh sibilance from phone compression)
5. Normalize to -3dB peak

**Usage:**
```bash
cd /home/eli/voice_studio
python whatsapp_preprocess.py /path/to/whatsapp/audio ./clean_output
```

### 3. Quick Start Workflow

```bash
# 1. Launch VoiceStudio
./run_voicestudio.sh

# 2. Pre-process your WhatsApp notes
python whatsapp_preprocess.py ~/path/to/whatsapp/notes ./clean_wavs

# 3. In VoiceStudio:
#    - Select GPT-SoVITS engine
#    - Fine-tune on ./clean_wavs/*.wav (5-10 notes, ~1 min total)
#    - Generate speech
```

## Requirements
- FFmpeg (installed: `/usr/bin/ffmpeg`)
- Python 3 with scipy, numpy (installed in system Python)
- GTX 1060 6GB (CUDA available for GPT-SoVITS)

## Notes
- VoiceStudio is a **desktop app**, not a web server — it runs on your display (`:1`)
- The AppImage requires FUSE, so we run from the extracted directory
- GPT-SoVITS is the recommended engine for WhatsApp audio (best tolerance for compressed source)
- For accent preservation: use 5-10 notes from the same speaker, same style, same mic

## Troubleshooting

### "SUID sandbox" error
```bash
sudo chown root:root /home/eli/vs_extract/squashfs-root/chrome-sandbox
sudo chmod 4755 /home/eli/vs_extract/squashfs-root/chrome-sandbox
```

### No audio in output
- Check that input files are actual voice notes (not corrupted)
- Ensure notes are at least 0.5 seconds long
- Try with `--quiet` flag to see only errors

### VoiceStudio won't launch
- Ensure DISPLAY=:1 is set
- Kill existing instances: `pkill -f voicestudio-electron`
- Re-run the launcher script
