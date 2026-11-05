import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""Edge-case stress pass: does every odd input encode, decode to the same shape, stay finite, and land near its bitrate?  python edge.py"""
import traceback, numpy as np, codec as c
g = np.random.default_rng(0); r0 = 44100
def sig(n, ch=2, f=lambda t: 0 * t): t = np.arange(n) / r0; return np.stack([f(t)] * ch, 1)
tone = lambda t: 8000 * np.sin(2 * np.pi * 1000 * t)
CASES = {
 "silence 2s": (sig(r0 * 2), r0),
 "dc offset 5000": (sig(r0 * 2, f=lambda t: 5000 + 0 * t), r0),
 "1-LSB noise": (np.round(g.standard_normal((r0 * 2, 2))), r0),
 "full-scale white noise": (np.clip(g.standard_normal((r0 * 2, 2)) * 12000, -32768, 32767), r0),
 "clipped square 100Hz": (sig(r0 * 2, f=lambda t: 32767 * np.sign(np.sin(2 * np.pi * 100 * t))), r0),
 "left only": (np.stack([8000 * np.sin(2 * np.pi * 440 * np.arange(r0 * 2) / r0), np.zeros(r0 * 2)], 1), r0),
 "right only": (np.stack([np.zeros(r0 * 2), 8000 * np.sin(2 * np.pi * 440 * np.arange(r0 * 2) / r0)], 1), r0),
 "anti-phase": (np.stack([tone(np.arange(r0 * 2) / r0), -tone(np.arange(r0 * 2) / r0)], 1), r0),
 "mono (1 ch)": (sig(r0 * 2, 1, tone), r0),
 "3 channels": (np.concatenate([sig(r0, 1, tone), sig(r0, 1, lambda t: 5000 * np.sin(2 * np.pi * 300 * t)), g.standard_normal((r0, 1)) * 500], 1), r0),
 "single click": (np.pad(np.full((1, 2), 30000.0), ((r0, r0), (0, 0))), r0),
 "8 kHz": (sig(8000 * 3, f=tone), 8000), "22.05 kHz": (sig(22050 * 3, f=tone), 22050), "96 kHz": (sig(96000 * 3, f=tone), 96000),
}
for n in (1, 2, 10, 100, 1023, 1024, 1025, 3000): CASES[f"{n} samples"] = (sig(n, f=tone) + 0.0, r0)
bad = 0
for name, (x, r) in CASES.items():
    x = x.astype(np.float32); res = []
    for k in (2, 8, 32, 500):
        try:
            b = c.encode_kbps(x, r, k); y, rr = c.decode(b); dur = len(x) / r; kb = len(b) * 8 / dur / 1000
            ok = y.shape == x.shape and rr == r and np.isfinite(y).all()
            e = ((x - y).astype(float) ** 2).sum(); sn = 10 * np.log10(max((x.astype(float) ** 2).sum(), 1e-9) / max(e, 1e-9)) if (x ** 2).sum() > 0 else float("nan")
            res.append(f"{k}k->{kb:6.1f}kbps SNR {sn:5.1f}" + ("" if ok else " !!SHAPE/NAN"))
            bad += not ok
        except Exception as ex:
            bad += 1; res.append(f"{k}k->CRASH {type(ex).__name__}: {str(ex)[:60]}")
    print(f"{name:24} " + " | ".join(res))
print("\nproblems:", bad)
