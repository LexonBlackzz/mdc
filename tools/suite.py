import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""Run every clip in clips/ at several bitrates and report error.  python suite.py [kbps ...] [--save f.json] [--base f.json]
SNR = waveform match (higher better); LSD = spectrum shape error in dB (lower better), by band. --base prints the change vs a saved run."""
import sys, os, glob, json, time
import numpy as np
from scipy.signal import stft
import codec as c

HERE = os.path.dirname(os.path.abspath(__file__))
a = sys.argv[1:]
save = a[a.index("--save") + 1] if "--save" in a else None
base = json.load(open(a[a.index("--base") + 1])) if "--base" in a else {}
rates = [float(v) for v in a if v.replace(".", "").isdigit()] or [32, 64, 96]
BANDS = [(0, 4e3), (4e3, 12e3), (12e3, 20e3)]


def measure(x, y, r):
    X, Y = x.astype(float), y.astype(float)
    snr = 10 * np.log10((X ** 2).sum() / max(((X - Y) ** 2).sum(), 1e-9))
    f, _, A = stft(X.mean(1), r, nperseg=2048); _, _, B = stft(Y.mean(1), r, nperseg=2048)
    lsd = [np.sqrt(((20 * np.log10(abs(A[m]) + 16) - 20 * np.log10(abs(B[m]) + 16)) ** 2).mean()) for m in [(f >= lo) & (f < hi) for lo, hi in BANDS]]  # +16: ignore anything under ~1/2 LSB of rounding noise
    return [snr] + lsd


if __name__ == '__main__':
    res = {}
    for k in rates:
        print(f"\n=== target {k:g} kbps ===\n{'clip':10} {'kbps':>6} {'SNR':>6} {'LSD<4k':>7} {'4-12k':>6} {'12-20k':>7} {'enc s':>6}" + ("   vs base: SNR" if base else ""))
        rows = []
        for p in sorted(glob.glob(os.path.join(HERE, "clips", "*.wav"))):
            name = os.path.basename(p)[:-4]; x, r = c.read_wav(p)
            t = time.time(); b = c.encode_kbps(x, r, k); te = time.time() - t
            y, _ = c.decode(b); m = [len(b) * 8 / (len(x) / r) / 1000] + measure(x, y, r) + [te]; rows.append(m); res[f"{name}@{k:g}"] = m
            d = f"   {m[1] - base[f'{name}@{k:g}'][1]:+5.2f}" if f"{name}@{k:g}" in base else ""
            print(f"{name:10} {m[0]:6.1f} {m[1]:6.1f} {m[2]:7.2f} {m[3]:6.2f} {m[4]:7.2f} {m[5]:6.1f}{d}")
        mean = np.mean(rows, 0); print(f"{'MEAN':10} {mean[0]:6.1f} {mean[1]:6.1f} {mean[2]:7.2f} {mean[3]:6.2f} {mean[4]:7.2f} {mean[5]:6.1f}")
    if save: json.dump(res, open(os.path.join(HERE, save), "w")); print("\nsaved", save)
