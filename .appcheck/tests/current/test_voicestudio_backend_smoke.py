#!/usr/bin/env python3
"""
appcheck: VoiceStudio runtime dependency smoke test.
Checks the bundled venv imports its core ML deps cleanly — the #1 cause of
"ready-made app broke". Uses a CLEAN environment (env -i) so the host
system PYTHONPATH can't leak the wrong numpy/torch into the venv.

PITFALL (regression): naive `subprocess.run([venv_py, ...])` inherits the
host PYTHONPATH, which swaps in the system numpy and makes torch/onnxruntime
fail with "No module named 'numpy._core._multiarray_umath'". Must strip env.

REGENERATE_ON_CHANGE: VoiceStudio runtime project deps
"""
import subprocess, sys, os

VENV = os.path.expanduser("~/.config/VoiceStudio/runtime/project/.venv")
PY = os.path.join(VENV, "bin", "python")

# Core modules VoiceStudio's backend loads at runtime
MODULES = ["torch", "demucs", "faster_whisper", "sherpa_onnx",
           "onnxruntime", "uvicorn", "omnivoice"]

# Strip everything except HOME/PATH so the venv uses ONLY its own site-packages
CLEAN_ENV = {
    "HOME": os.path.expanduser("~"),
    "PATH": "/usr/bin:/bin",
    "LD_LIBRARY_PATH": os.path.join(VENV, "lib", "python3.11", "site-packages",
                                    "torch", "lib"),
}


def check():
    if not os.path.exists(PY):
        return False, f"venv python not found at {PY}"
    failures = []
    for mod in MODULES:
        r = subprocess.run([PY, "-c", f"import {mod}"],
                           capture_output=True, text=True, timeout=90,
                           env=CLEAN_ENV)
        if r.returncode != 0:
            failures.append(f"{mod}: " + (r.stderr.strip().splitlines() or ["?"])[-1])
    if failures:
        return False, "import failures: " + " | ".join(failures)
    return True, f"all {len(MODULES)} core modules import cleanly in bundled venv"


def check_cuda():
    """GTX 1060 must be visible to torch."""
    r = subprocess.run([PY, "-c",
                        "import torch; print('cuda' if torch.cuda.is_available() else 'NO_CUDA', torch.version.cuda)"],
                       capture_output=True, text=True, timeout=90, env=CLEAN_ENV)
    out = (r.stdout + r.stderr).strip()
    if "NO_CUDA" in out or "cuda" not in out:
        return False, f"CUDA not available: {out[:120]}"
    return True, f"CUDA available: {out}"


if __name__ == "__main__":
    ok, msg = check()
    print(f"[{'PASS' if ok else 'FAIL'}] voicestudio_backend_smoke: {msg}")
    ok2, msg2 = check_cuda()
    print(f"[{'PASS' if ok2 else 'FAIL'}] voicestudio_cuda: {msg2}")
    sys.exit(0 if (ok and ok2) else 1)