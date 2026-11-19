#!/usr/bin/env python3
"""
WhatsApp Voice Note Pre-Processor for Voice Cloning
=====================================================
Batch processes WhatsApp voice notes (OGG/OPUS) into clean WAV files
optimized for voice cloning (GPT-SoVITS, Chatterbox, etc).

Pipeline:
1. Convert OGG/OPUS → WAV 24kHz mono
2. Noise reduction (spectral subtraction via scipy)
3. Normalize peaks to -3dB
4. Trim silence
5. De-ess (reduce harsh sibilance)
6. Output clean WAV ready for cloning

Usage:
    python whatsapp_preprocess.py /path/to/whatsapp/audio /output/dir
"""

import os
import sys
import glob
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, wiener, sosfiltfilt, stft, istft
import subprocess
import argparse
import warnings
warnings.filterwarnings('ignore')


def convert_to_wav(input_path: str, output_path: str, sample_rate: int = 24000):
    """Convert any audio to WAV 24kHz mono using ffmpeg."""
    cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-ar', str(sample_rate),
        '-ac', '1',  # mono
        '-sample_fmt', 's16',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr[:200]}")
    return output_path


def load_audio(path: str):
    """Load WAV file, return sample_rate, audio_array (float32, normalized)."""
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0
    return sr, data


def save_audio(path: str, sr: int, data: np.ndarray):
    """Save audio as 16-bit WAV."""
    # Clip to prevent overflow
    data = np.clip(data, -1.0, 1.0)
    # Convert to int16
    data_int16 = (data * 32767).astype(np.int16)
    wavfile.write(path, sr, data_int16)


def noise_reduction(audio: np.ndarray, noise_floor_db: float = -40):
    """
    Spectral subtraction noise reduction via STFT.
    Estimates a noise profile from the quietest frames, then
    attenuates noise magnitude below the profile.
    """
    # Use a small analysis window/frame
    window = int(0.025 * 24000)  # 25ms at typical 24kHz
    if window < 128:
        window = 256
    hop = window // 4
    nfft = max(256, 1 << int(np.ceil(np.log2(window))))

    # Pad so we get whole frames
    if len(audio) < nfft:
        return audio  # too short to denoise meaningfully

    f, t, Zxx = signal.stft(audio, nperseg=nfft, noverlap=nfft - hop)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Estimate noise: take the 10% quietest frames by RMS
    frame_rms = np.sqrt(np.mean(mag ** 2, axis=0))
    n_noise = max(1, int(0.1 * frame_rms.shape[0]))
    quiet_idx = np.argsort(frame_rms)[:n_noise]
    noise_profile = np.mean(mag[:, quiet_idx], axis=1, keepdims=True)

    # Spectral subtraction with floor (avoid musical noise / over-aggression)
    alpha = 2.0        # subtraction factor
    floor_db = 12.0    # minimum attenuation floor in dB
    floor = 10 ** (-floor_db / 20.0)
    mask = np.maximum(1.0 - alpha * (noise_profile / (mag + 1e-12)), floor)
    mag_clean = mag * mask

    # Rebuild with original phase
    _, audio_clean = signal.istft(mag_clean * np.exp(1j * phase),
                                  nperseg=nfft, noverlap=nfft - hop)
    # Trim back to original length
    if len(audio_clean) > len(audio):
        audio_clean = audio_clean[:len(audio)]
    return audio_clean


def normalize_audio(audio: np.ndarray, target_peak_db: float = -3.0):
    """Normalize audio to target peak level in dB."""
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return audio  # Silent audio
    
    current_peak_db = 20 * np.log10(peak)
    gain_db = target_peak_db - current_peak_db
    gain_linear = 10 ** (gain_db / 20.0)
    
    normalized = audio * gain_linear
    # Clip to prevent overflow
    normalized = np.clip(normalized, -1.0, 1.0)
    return normalized


def trim_silence(audio: np.ndarray, sr: int, threshold_db: float = -40, 
                 min_silence_sec: float = 0.3):
    """Remove leading and trailing silence."""
    # Convert threshold to linear
    threshold = 10 ** (threshold_db / 20.0)
    
    # Find non-silent regions
    abs_audio = np.abs(audio)
    is_silent = abs_audio < threshold
    
    # Find first non-silent sample
    start_idx = 0
    for i in range(len(audio)):
        if not is_silent[i]:
            start_idx = i
            break
    
    # Find last non-silent sample
    end_idx = len(audio)
    for i in range(len(audio) - 1, -1, -1):
        if not is_silent[i]:
            end_idx = i + 1
            break
    
    # Ensure minimum length
    min_samples = int(min_silence_sec * sr)
    if end_idx - start_idx < min_samples:
        # Keep at least some audio
        center = (start_idx + end_idx) // 2
        start_idx = max(0, center - min_samples // 2)
        end_idx = min(len(audio), center + min_samples // 2)
    
    return audio[start_idx:end_idx]


def deess(audio: np.ndarray, sr: int, threshold_hz: float = 5000, 
          reduction_db: float = 3.0):
    """
    Simple de-esser: reduce harsh sibilance above threshold frequency.
    Applies a high-shelf low-pass blend only where high-freq energy is strong.
    """
    if len(audio) < 256:
        return audio
    nyq = sr / 2.0
    if threshold_hz >= nyq:
        return audio
    normalized_freq = threshold_hz / nyq

    # Low-pass (below threshold_hz) to isolate the sibilance-free signal
    sos = butter(4, normalized_freq, btype='low', output='sos')
    filtered = sosfiltfilt(sos, audio)

    # High-frequency content = original minus filtered
    high = audio - filtered
    # RMS of high-freq content (sibilance level)
    hi_rms = np.sqrt(np.mean(high ** 2) + 1e-12)
    if hi_rms < 1e-4:
        return audio  # not much sibilance
    # Reduce high-freq energy by reduction_db
    gain = 10 ** (-reduction_db / 20.0)
    # Blend: keep filtered fully, add reduced high back
    deessed = filtered + high * gain
    return deessed


def process_whatsapp_note(input_path: str, output_dir: str, verbose: bool = True):
    """
    Process a single WhatsApp voice note through the full pipeline.
    Returns the output WAV path or None on failure.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output filename
    output_wav = output_dir / f"{input_path.stem}_clean.wav"
    temp_wav = output_dir / f"{input_path.stem}_temp.wav"
    
    try:
        if verbose:
            print(f"Processing: {input_path.name}")
        
        # Step 1: Convert to WAV 24kHz mono
        convert_to_wav(str(input_path), str(temp_wav), sample_rate=24000)
        sr, audio = load_audio(str(temp_wav))
        
        if verbose:
            print(f"  → Converted: {sr}Hz, {len(audio)/sr:.2f}s")
        
        # Check if audio is too short or silent
        if len(audio) < sr * 0.5:  # Less than 0.5 seconds
            if verbose:
                print(f"  ✗ Too short, skipping")
            return None
        
        if np.max(np.abs(audio)) < 0.01:  # Nearly silent
            if verbose:
                print(f"  ✗ Too quiet, skipping")
            return None
        
        # Step 2: Noise reduction
        audio = noise_reduction(audio)
        if verbose:
            print(f"  → Noise reduction applied")
        
        # Step 3: Trim silence
        audio = trim_silence(audio, sr)
        if verbose:
            print(f"  → Trimmed: {len(audio)/sr:.2f}s")
        
        # Check again after trimming
        if len(audio) < sr * 0.3:  # Less than 0.3 seconds after trim
            if verbose:
                print(f"  ✗ Too short after trimming, skipping")
            return None
        
        # Step 4: De-ess
        audio = deess(audio, sr)
        if verbose:
            print(f"  → De-ess applied")
        
        # Step 5: Normalize to -3dB peak
        audio = normalize_audio(audio, target_peak_db=-3.0)
        if verbose:
            print(f"  → Normalized to -3dB")
        
        # Save final output
        save_audio(str(output_wav), sr, audio)
        
        # Clean up temp file
        if temp_wav.exists():
            temp_wav.unlink()
        
        if verbose:
            print(f"  ✓ Output: {output_wav.name}")
        
        return str(output_wav)
    
    except Exception as e:
        if verbose:
            print(f"  ✗ Error: {str(e)[:100]}")
        # Clean up temp file on error
        if temp_wav.exists():
            temp_wav.unlink()
        return None


def batch_process(input_dir: str, output_dir: str, verbose: bool = True):
    """
    Batch process all audio files in a directory.
    Supports: OGG, OPUS, MP3, M4A, WAV, FLAC
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Supported extensions
    extensions = ['*.ogg', '*.opus', '*.mp3', '*.m4a', '*.wav', '*.flac']
    
    audio_files = []
    for ext in extensions:
        audio_files.extend(input_path.rglob(ext))
    
    audio_files = list(set(str(f) for f in audio_files))
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return []
    
    print(f"\n{'='*60}")
    print(f"WhatsApp Voice Note Pre-Processor")
    print(f"{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Files:  {len(audio_files)}")
    print(f"{'='*60}\n")
    
    successful = []
    failed = []
    skipped = []
    
    for audio_file in sorted(audio_files):
        result = process_whatsapp_note(audio_file, output_path, verbose)
        if result:
            successful.append(result)
        elif result is None:
            skipped.append(str(audio_file))
        else:
            failed.append(str(audio_file))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Successful: {len(successful)}")
    print(f"✗ Failed:     {len(failed)}")
    print(f"- Skipped:    {len(skipped)}")
    print(f"{'='*60}\n")
    
    if failed and verbose:
        print("Failed files:")
        for f in failed[:10]:
            print(f"  - {f}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    return successful


def main():
    parser = argparse.ArgumentParser(
        description='Pre-process WhatsApp voice notes for voice cloning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python whatsapp_preprocess.py ~/WhatsApp/Media/audio ./clean_audio
  python whatsapp_preprocess.py ./voice_notes ./output --quiet
        """
    )
    parser.add_argument('input_dir', help='Directory containing WhatsApp audio files')
    parser.add_argument('output_dir', help='Output directory for cleaned WAV files')
    parser.add_argument('-q', '--quiet', action='store_true', 
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"ERROR: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    successful = batch_process(args.input_dir, args.output_dir, verbose=not args.quiet)
    
    if not successful:
        print("WARNING: No files were successfully processed.")
        sys.exit(1)
    
    print(f"Processed {len(successful)} files successfully.")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
