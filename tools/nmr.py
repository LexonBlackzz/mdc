import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""Noise-to-mask ratio: how far the coding error sticks out above what the original masks. python nmr.py orig.wav a.wav b.wav ...  (lower = better)
Per 2048-STFT frame and Bark band: mask = original band energy spread over neighbours (Schroeder) minus an offset; NMR = error energy / mask in dB."""
import sys, numpy as np
from scipy.signal import stft
import codec as c

def bark(f): return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)

def nmr(x, y, r, t0=0, t1=None, off=12.0):
    n = len(x) if t1 is None else min(len(x), int(t1 * r)); x, y = x[int(t0 * r):n].astype(float), y[int(t0 * r):n].astype(float)
    y = y[:len(x)]; f, _, A = stft(x, r, nperseg=2048, axis=0); _, _, B = stft(y, r, nperseg=2048, axis=0)
    PA, PE = (abs(A) ** 2).sum(1) if False else abs(A) ** 2, abs(A - B) ** 2        # (bins, ch, frames)
    PA, PE = PA.sum(1), PE.sum(1); z = bark(f); nb = int(z[-1]) + 1; band = np.minimum(z.astype(int), nb - 1)
    E = np.stack([PA[band == b].sum(0) for b in range(nb)]); N = np.stack([PE[band == b].sum(0) for b in range(nb)])
    dz = np.arange(nb)[:, None] - np.arange(nb)[None, :]; sf = 10 ** ((15.81 + 7.5 * (dz + 0.474) - 17.5 * np.sqrt(1 + (dz + 0.474) ** 2)) / 10)
    M = sf @ E * 10 ** (-off / 10) + 1e3                                              # spread + tiny floor
    d = 10 * np.log10((N + 1e-9) / M); fc = np.array([f[band == b].mean() for b in range(nb)])
    live = E > 1e3 * 50
    out = {}
    for name, lo, hi in (("<500", 0, 500), ("0.5-2k", 500, 2000), ("2-6k", 2000, 6000), (">6k", 6000, 1e9)):
        m = ((fc >= lo) & (fc < hi))[:, None] & live; out[name] = float(np.maximum(d[m], 0).mean()) if m.any() else 0.0
    out["all"] = float(np.maximum(d[live], 0).mean()); return out

if __name__ == "__main__":
    x, r = c.read_wav(sys.argv[1])
    for p in sys.argv[2:]:
        y, _ = c.read_wav(p); o = nmr(x, y, r, 0, 33); print(f"{p.split('/')[-1]:28}", "  ".join(f"{k} {v:5.2f}" for k, v in o.items()))
