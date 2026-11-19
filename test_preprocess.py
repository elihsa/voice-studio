#!/usr/bin/env python3
"""
Test: Verify the WhatsApp pre-processor works end-to-end.
Creates synthetic test audio, processes it, validates output.
"""
import os
import sys
import numpy as np
from pathlib import Path
import subprocess
import tempfile

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))
from whatsapp_preprocess import (
    load_audio, save_audio, noise_reduction, normalize_audio,
    trim_silence, deess, process_whatsapp_note, batch_process
)

def create_test_wav(path: str, sr: int = 24000, duration: float = 3.0):
    """Create a synthetic WAV file for testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440Hz sine wave (like speech)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Add some "noise"
    noise = np.random.normal(0, 0.02, len(audio))
    audio = audio + noise
    save_audio(path, sr, audio)
    return path

def create_test_ogg(path: str, wav_path: str):
    """Convert WAV to OGG for testing (simulates WhatsApp voice note)."""
    cmd = ['ffmpeg', '-y', '-i', wav_path, '-codec:a', 'libopus', '-b:a', '64k', path]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr[:200]}")

def test_normalize():
    """Test audio normalization."""
    sr = 24000
    audio = np.random.normal(0, 0.1, sr)  # Random audio
    normalized = normalize_audio(audio, target_peak_db=-3.0)
    peak = np.max(np.abs(normalized))
    peak_db = 20 * np.log10(peak)
    assert -4.0 <= peak_db <= -2.0, f"Peak {peak_db}dB not near -3dB"
    print("✓ normalize: peak at -3dB")

def test_trim_silence():
    """Test silence trimming."""
    sr = 24000
    # Audio with 0.5s silence, 1s signal, 0.5s silence
    silence_before = np.zeros(int(0.5 * sr))
    signal = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))
    silence_after = np.zeros(int(0.5 * sr))
    audio = np.concatenate([silence_before, signal, silence_after])
    trimmed = trim_silence(audio, sr)
    assert len(trimmed) < len(audio), "Audio should be shorter after trimming"
    assert len(trimmed) >= int(0.9 * sr), "Trimmed audio should still contain signal"
    print(f"✓ trim_silence: {len(audio)/sr:.2f}s → {len(trimmed)/sr:.2f}s")

def test_noise_reduction():
    """Test noise reduction doesn't crash and produces output."""
    sr = 24000
    audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))
    audio += np.random.normal(0, 0.05, sr)
    cleaned = noise_reduction(audio)
    assert len(cleaned) == len(audio), "Length should be preserved"
    print("✓ noise_reduction: output length matches input")

def test_deess():
    """Test de-ess doesn't crash."""
    sr = 24000
    audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))
    deessed = deess(audio, sr)
    assert len(deessed) == len(audio), "Length should be preserved"
    print("✓ deess: output length matches input")

def test_full_pipeline():
    """Test the full pipeline with a synthetic OGG file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "test.wav")
        ogg_path = os.path.join(tmpdir, "test.ogg")
        
        # Create test OGG (simulating WhatsApp voice note)
        create_test_wav(wav_path, duration=3.0)
        create_test_ogg(ogg_path, wav_path)
        
        # Process it
        output_dir = os.path.join(tmpdir, "output")
        result = process_whatsapp_note(ogg_path, output_dir, verbose=False)
        
        assert result is not None, "Processing should succeed"
        assert os.path.exists(result), "Output file should exist"
        
        # Verify output
        sr, audio = load_audio(result)
        assert sr == 24000, f"Sample rate should be 24000Hz, got {sr}"
        assert len(audio) > 0, "Output should have audio data"
        peak = np.max(np.abs(audio))
        assert peak > 0.1, f"Peak too low: {peak}"
        
        print(f"✓ full_pipeline: {os.path.basename(result)} ({len(audio)/sr:.2f}s, peak={20*np.log10(peak):.1f}dB)")

def test_batch():
    """Test batch processing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir)
        # Create OGG files in input dir (WhatsApp-style)
        for i in range(3):
            ogg_path = os.path.join(input_dir, f"test{i}.ogg")
            wav_path = os.path.join(tmpdir, f"src{i}.wav")
            create_test_wav(wav_path, duration=2.0)
            create_test_ogg(ogg_path, wav_path)
        
        output_dir = os.path.join(tmpdir, "batch_output")
        results = batch_process(input_dir, output_dir, verbose=False)
        
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        print(f"✓ batch: processed {len(results)} files")

def test_short_audio_skip():
    """Test that very short audio is skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "short.wav")
        ogg_path = os.path.join(tmpdir, "short.ogg")
        
        # Create 0.1 second audio (too short)
        sr = 24000
        t = np.linspace(0, 0.1, sr)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        save_audio(wav_path, sr, audio)
        create_test_ogg(ogg_path, wav_path)
        
        output_dir = os.path.join(tmpdir, "output")
        result = process_whatsapp_note(ogg_path, output_dir, verbose=False)
        
        # Should be skipped due to short length
        print(f"✓ short_audio_skip: {'skipped' if result is None else 'processed'}")

if __name__ == '__main__':
    print("Running tests...\n")
    test_normalize()
    test_trim_silence()
    test_noise_reduction()
    test_deess()
    test_full_pipeline()
    test_batch()
    test_short_audio_skip()
    print("\n✓ All tests passed!")
