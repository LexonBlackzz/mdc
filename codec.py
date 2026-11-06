# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LexonBlackzz
"""Toy lossy audio codec: MDCT (long frames, short blocks on attacks) -> per-band levels -> loudness-scaled quantizer
-> range coder (rc.py); the decoder noise-fills empty bins.
Usage: python codec.py enc in.flac out.mdc [<kbps>k | g] [--scope for oscilloscope music] [tilt] [cutoff Hz, 0 = none] [gamma] [noisefill] [trans]   |   dec in.mdc [out.wav]   |   (no args) self-test
"""
import os, sys, struct, wave, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import rc
from numba import njit

N = int(os.environ.get('NN', 1024))  # hop size of a long frame; its window is 2N
NS = 8; M = N // NS  # a short frame is NS blocks of M samples (NS * M = N, so every frame holds N coefficients)
OFF = (N - M) // 2  # where the short windows start inside a frame's 2N span (448)
CHUNK = 256  # frames per block: bounds memory, and is the progress-bar granularity
F32 = np.float32
KBPS_MIN, KBPS_MAX = 0.1, 10000.0
TILT = 0.0  # extra step growth with frequency on top of the loudness scaling: up to 1+TILT at the top bin. 0 sounded best.
GAMMA = 0.3  # step ~ (band level)^GAMMA: 0 = same noise everywhere, 1 = same SNR in every band. Tune by ear.
NF = None  # noise fill strength for bins that rounded to zero (0 = off). None = automatic from the bitrate: 0.8 at 1 kbps, 0.7 from 3 kbps up (spectrum error is lowest there at every rate tested)


def auto_nf(kbps):  # lots of holes at low bitrates, where the noise adds fidelity; at high bitrates it is mostly just extra noise
    return float(np.interp(kbps, [1, 3], [0.8, 0.7]))
TRANS = 8.0  # a 128-sample segment this many times louder (after a high-pass) than the average of the 8 before it makes its frame short; 0 = never
ENVC = False  # set by encode_kbps: noise envelopes in 6 dB steps instead of 3 dB
SCOPE = False  # set by encode_kbps(scope=True)
MS = True  # mid/side stereo: near-mono music leaves the side channel almost empty
RND = float(os.environ.get('RND', 0.35))  # rounding offset: 0.5 = plain rounding, lower = wider deadzone = more zeros
RECON = float(os.environ.get('RECON', -0.1))  # reconstruction offset toward zero, in steps
LOFF = int(os.environ.get('LOFF', 3))  # level 1 = RMS 2^((1-LOFF)/LRES): with LOFF > 0 bands quieter than one 16-bit step are still described
LRES = 2  # band levels are sent in steps of 1/LRES octave of amplitude (2 = 3 dB)
EDGES = (np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 1024]) * N // 1024).astype(np.int64)  # band edges in bins (21.5 Hz each)
NBF, WIDTH = len(EDGES) - 1, np.diff(EDGES)
BMAP = np.repeat(np.arange(NBF), WIDTH).astype(np.int16)  # bin -> band
GROUP = np.array([0] * 12 + [1] * 6 + [2] * 4)  # band -> frequency group (<1.4k, <5.5k, above) for the short-frame time envelope
GBIN, JBIN = GROUP[BMAP], np.arange(N) % NS  # bin -> its group / its short block (interleaved layout: index = bin * NS + block)
EDGES0, EDGE_CUSTOM = EDGES.copy(), 0
NB = int(os.environ.get('NB', 22))  # number of bands when the layout is adaptive
ADAPT = int(os.environ.get('ADAPT', 1))  # 1: band edges are chosen per file from its own spectrum (stored in the header)


def set_edges(e):
    """Switch the whole codec to another band layout (module tables are rebuilt)."""
    global EDGES, NBF, WIDTH, BMAP, GROUP, GBIN, CEN, I0, I1, WI, EDGE_CUSTOM, _SF, _ATH
    EDGES = np.asarray(e, dtype=np.int64); NBF, WIDTH = len(EDGES) - 1, np.diff(EDGES); EDGE_CUSTOM = int(not np.array_equal(EDGES, EDGES0))
    BMAP = np.repeat(np.arange(NBF), WIDTH).astype(np.int16)
    mid = (EDGES[:-1] + EDGES[1:]) / 2; GROUP = (mid > 64 * N // 1024).astype(int) + (mid > 256 * N // 1024); GBIN = GROUP[BMAP]
    CEN = mid; I0 = np.clip(np.searchsorted(CEN, XB, "right") - 1, 0, NBF - 1); I1 = np.clip(I0 + 1, 0, NBF - 1)
    WI = np.clip(np.where(I1 > I0, (XB - CEN[I0]) / np.maximum(CEN[I1] - CEN[I0], 1e-9), 0), 0, 1).astype(F32)
    _SF, _ATH = _mask_tables()


def adaptive_edges(coef, nb=22):
    """Band layout from the song itself: split the spectrum into nb bands so each band's long-term level is as flat as possible (weighted by how many Bark each bin spans, so the lows, where the ear resolves most, get the narrow bands)."""
    u = N // 256; U = 256
    S = 10 * np.log10(coef.astype(np.float64) ** 2 + 1).reshape(-1, N).mean(0).reshape(U, u).mean(1)  # typical level per 4-bin unit
    f = (np.arange(U + 1) * u) * 22050 / N; z = 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2); w = np.diff(z)
    c0, c1, c2 = (np.r_[0, np.cumsum(a)] for a in (w, w * S, w * S * S))
    i, j = np.arange(U + 1)[:, None], np.arange(U + 1)[None, :]
    with np.errstate(all="ignore"):
        sse = (c2[j] - c2[i]) - (c1[j] - c1[i]) ** 2 / (c0[j] - c0[i])
    sse = np.where((j > i) & (j - i <= 64) & np.isfinite(sse), sse, np.inf)
    D = np.full((nb + 1, U + 1), np.inf); D[0, 0] = 0; P = np.zeros((nb + 1, U + 1), np.int64)
    for m in range(1, nb + 1):
        t = D[m - 1][:, None] + sse; P[m] = t.argmin(0); D[m] = t.min(0)
    cuts = [U]
    for m in range(nb, 0, -1): cuts.append(P[m, cuts[-1]])
    return np.array(cuts[::-1]) * u


NFLO = 1.0  # noise fill strength below 500 Hz relative to NF (rises smoothly to 1.0 at 4 kHz): lows sound hissy long before highs do
SMOOTH = False  # noise fill follows a smooth line between band centres instead of a flat step per band
CEN, XB = (EDGES[:-1] + EDGES[1:]) / 2, np.arange(N) + 0.5
I0 = np.clip(np.searchsorted(CEN, XB, "right") - 1, 0, NBF - 1); I1 = np.clip(I0 + 1, 0, NBF - 1)  # the two band centres around each bin, and the weight of the upper one
WI = np.clip(np.where(I1 > I0, (XB - CEN[I0]) / np.maximum(CEN[I1] - CEN[I0], 1e-9), 0), 0, 1).astype(F32)
SEEDS, SB = 8, 3  # noise-fill codebook: the encoder picks 1 of SEEDS sign patterns (SB bits) for every long-frame band with at least ZMIN zeroed bins (ZMIN is in the header: seeds cost too much at low bitrates)
ZMIN = int(os.environ.get('ZMIN', 8))
CHOFF = int(os.environ.get('CHOFF', 113))  # each channel reads its noise pattern from a different row (identical noise in mid and side puts all of it into one speaker)
_rg = np.random.default_rng(7)
PAT = _rg.choice(np.array([-1, 1], F32), (251, N))  # sign patterns; frame t / candidate s uses row (t * SEEDS + s) % 251
HDR = "<IIHffBHffIfBIf"  # ..., smooth noise (unused), coefficient-stream bytes
HLEN = struct.calcsize(HDR)


def _basis(n):  # orthonormal MDCT basis, n x 2n
    return (np.sqrt(2 / n) * np.cos(np.pi / n * (np.arange(2 * n)[None, :] + 0.5 + n / 2) * (np.arange(n)[:, None] + 0.5))).astype(F32)


KBD = 4.0  # Kaiser-Bessel-derived slope: far less spectral leakage than a sine slope, so pure tones stay clean at low bitrates (music costs ~nothing)


def _rise(n):  # 0 -> 1 over n samples, w[i]^2 + w[n-1-i]^2 = 1 (Princen-Bradley)
    k = np.cumsum(np.kaiser(n + 1, np.pi * KBD)); return np.sqrt(k[:n] / k[n]).astype(F32)


def _longwin(lshort, rshort):
    """Window of a long frame. Next to a short frame the slope on that side shrinks to M samples (flat 1 then 0 around it), so overlaps still cancel."""
    w = np.empty(2 * N, F32)
    if lshort: w[:OFF], w[OFF:OFF + M], w[OFF + M:N] = 0, _rise(M), 1
    else: w[:N] = _rise(N)
    if rshort: w[N:N + OFF], w[N + OFF:N + OFF + M], w[N + OFF + M:] = 1, _rise(M)[::-1], 0
    else: w[N:] = _rise(N)[::-1]
    return w


BASIS, BASIS_S, WS = _basis(N), _basis(M), np.r_[_rise(M), _rise(M)[::-1]]  # a short window: slope up over M, slope down over M
LONGW = np.stack([_longwin(i & 1, i >> 1) for i in range(4)])  # indexed by left_short + 2 * right_short
SIDX = OFF + M * np.arange(NS)[:, None] + np.arange(2 * M)[None, :]  # NS short windows inside the 2N span


def varof(short):  # which long window each frame uses
    return (np.r_[False, short[:-1]] + 2 * np.r_[short[1:], False]).astype(int)


def midside(x):  # orthonormal and its own inverse
    return np.stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]], 1) * F32(0.70710678)


def weights(tilt):  # per-bin step multiplier, rises quadratically with frequency
    return (1 + tilt * (np.arange(N) / N) ** 2).astype(F32)


def nframes(n):
    return (n + (-n) % N + N) // N


def rhat(L):  # band level -> RMS amplitude; level 0 = empty band
    return np.where(L > 0, 2.0 ** ((L - LOFF) / LRES), 0.0)


def levels(coef):  # (ch, frames, N) -> int32 (ch, frames, bands): quantized log2 RMS of every band
    e = np.add.reduceat(coef * coef, EDGES[:-1], axis=2) / WIDTH
    return np.maximum(np.round(LRES * np.log2(np.sqrt(e) + 1e-9)) + LOFF, 0).astype(np.int32)


TGE = float(os.environ.get('TG', 0)); TG = TGE  # step follows frame loudness with this exponent (0 = off: plain gamma)
KAP = float(os.environ.get('KAP', 0))  # >0: no band's step may exceed KAP x its own rms (older fix for buried quiet bands)
ATHC = float(os.environ.get('ATHC', 15))  # the threshold curve is capped here (dB): it rises very steeply above 15 kHz and would delete audible highs
ATHF = float(os.environ.get('ATHF', 12))  # playback-level assumption for the hearing-threshold floor (dB; 0 = loud, 96 dB full scale)


def _mask_tables():
    global _BARK
    _BARK = 13 * np.arctan(0.00076 * EDGES * 22050 / N) + 3.5 * np.arctan((EDGES * 22050 / N / 7500) ** 2)
    fc = (EDGES[:-1] + EDGES[1:]) / 2 * (22050 / N); z = 13 * np.arctan(0.00076 * fc) + 3.5 * np.arctan((fc / 7500) ** 2)
    zl, zh = (13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2) for f in (EDGES[:-1] * 22050 / N, EDGES[1:] * 22050 / N))
    dz = z[:, None] - z[None, :]; sf = 10 ** ((15.81 + 7.5 * (dz + 0.474) - 17.5 * np.sqrt(1 + (dz + 0.474) ** 2)) / 10) * np.minimum(zh - zl, 1)[None, :]
    np.fill_diagonal(sf, 0); kf = fc / 1000
    ath = 3.64 * kf ** -0.8 - 6.5 * np.exp(-0.6 * (kf - 3.3) ** 2) + 1e-3 * kf ** 4  # threshold of hearing, dB SPL
    return sf.astype(F32), (0.37 * 10 ** (np.minimum(ath, 60) / 20)).astype(F32) ** 2
_SF, _ATH = _mask_tables()


TGW = int(os.environ.get('TGW', 6))  # frames either side that the loudness reference looks at


def frame_factor(L, gamma, segf=None):
    """Per-frame step multiplier (ch, frames, 1): the loudness-following part (TG) and the encoder's per-segment bit-allocation offsets (1.5 dB units, one per SEGL frames). None = 1."""
    fn = frame_norm(L); fac = None if fn is None or TG == gamma else fn ** (TG - gamma)
    if segf is not None:
        g = np.broadcast_to((2.0 ** (np.repeat(segf, SEGL)[:L.shape[1]] / 4.0))[None, :, None], (L.shape[0], L.shape[1], 1)); fac = g if fac is None else fac * g
    return fac


def frame_norm(L):
    """Background loudness per frame, (ch, frames, 1): the frame's rms, but a drum hit does not count, so only the quietest nearby frame (min over +-TGW frames, then smoothed) sets it."""
    if TG <= 0: return None
    p = np.sqrt(np.maximum((rhat(L) ** 2 * WIDTH).sum(-1) / N, 1.0)); F = p.shape[1]; k = 2 * TGW + 1
    pp = np.pad(p, ((0, 0), (TGW, TGW)), mode="edge"); m = np.lib.stride_tricks.sliding_window_view(pp, k, axis=1).min(-1)
    cs = np.cumsum(np.pad(m, ((0, 0), (TGW + 1, TGW)), mode="edge"), 1); cs = np.pad(cs, ((0, 0), (1, 0)))
    return ((cs[:, k + 1:k + 1 + F] - cs[:, 1:1 + F]) / k)[..., None] if F > 0 else m[..., None]


SEGL = int(os.environ.get('SEGL', 2))  # frames per bit-allocation segment
HIT = float(os.environ.get('HIT', 1.2))  # a segment costing more than HIT x the median of its neighbours counts as a hit
MOFF = float(os.environ.get('MOFF', 12))  # masking margin (dB below the spread band power) used when choosing allocation


def band_steps(L, step, gamma, fn=None):  # quantizer step per band
    r = rhat(L)
    s = step * r ** gamma
    if fn is not None: s = s * fn  # frame factor: loudness-following steps (TG) and the per-segment allocation offsets
    if KAP > 0: s = np.minimum(s, KAP * r)
    return np.where(L > 0, s, 1.0).astype(F32)


def expected(L, step, tilt, gamma, fn=None):  # what the decoder expects |q| to be, bucketed (-1 = empty band): context for the entropy coder
    m = rhat(L) / (band_steps(L, step, gamma, fn) * weights(tilt)[(EDGES[:-1] + EDGES[1:]) // 2])
    b = np.floor(2 * np.log2(np.maximum(m, 1e-6))).astype(np.int32) + 4
    return np.where(L > 0, np.clip(b, 0, rc.NB - 1), -1).astype(np.int8)


_T = {}


def bar(i, n, label):
    t0, i0 = _T.setdefault(label, (time.time(), i))
    el, done = time.time() - t0, i - i0
    eta = f"{int(el / done * (n - i) // 60)}:{int(el / done * (n - i) % 60):02d}" if done > 0 and i < n else "--:--"
    sys.stderr.write(f"\r{label:<9} [{'#' * (30 * i // n):<30}] {100 * i // n:3d}%  elapsed {int(el // 60)}:{int(el % 60):02d}  left {eta}   " + ("\n" if i >= n else ""))
    sys.stderr.flush()
    if i >= n: del _T[label]


def attacks(x):
    """Bool per 128-sample segment of the padded signal: is it an attack? (segment u covers original samples 128u - N ...)"""
    n = len(x); sig = np.zeros((nframes(n) + 1) * N, F32); sig[N:N + n] = np.diff(x.mean(1), prepend=0)  # first difference = crude high-pass
    e = (sig.reshape(-1, M) ** 2).sum(1).astype(np.float64)
    cs, u = np.r_[0, np.cumsum(e)], np.arange(len(e)); lo = np.maximum(u - 8, 0)
    prev = (cs[u] - cs[lo]) / np.maximum(u - lo, 1)  # mean energy of the previous 8 segments
    return (e > TRANS * prev + M * 30.0 ** 2) & (u >= 1) if TRANS > 0 else np.zeros(len(e), bool)


def detect(x):  # frames that switch to short blocks: those whose short windows (segments 8t+4 .. 8t+11) contain an attack
    F = nframes(len(x)); short = np.zeros(F, bool)
    t = (np.flatnonzero(attacks(x)) - 4) // 8
    short[t[(t >= 0) & (t < F)]] = True
    return short


def analyze(x, short, show=False):
    """MDCT of every frame: (channels, frames, N) float32. Short frames hold NS interleaved blocks (index = bin * NS + block)."""
    n, ch = x.shape
    pad = np.zeros((N + n + (-n) % N + N, ch), F32); pad[N:N + n] = x  # N lead-in, tail pad
    F = (len(pad) - N) // N; var = varof(short)
    out = np.empty((ch, F, N), F32)
    for c in range(ch):
        for s in range(0, F, CHUNK):
            e = min(F, s + CHUNK); sh = short[s:e]
            fr = pad[np.arange(2 * N)[None, :] + N * np.arange(s, e)[:, None], c]
            o = np.empty((e - s, N), F32)
            if (~sh).any(): o[~sh] = (fr[~sh] * LONGW[var[s:e][~sh]]) @ BASIS.T
            if sh.any(): o[sh] = ((fr[sh][:, SIDX] * WS) @ BASIS_S.T).transpose(0, 2, 1).reshape(-1, N)
            out[c, s:e] = o
            if show: bar(c * F + e, ch * F, "analyze")
    return out


def lowpass(coef, rate, hz):  # zero every bin above hz: spend the bits on what's left instead of flickering HF
    if hz > 0: coef[:, :, int(hz / (rate / 2) * N):] = 0
    return coef


def side_bytes(L):
    out = np.empty(L.size * 2 + 4096, np.uint8)
    while (m := rc.encode_side(L, out)) < 0: out = np.empty(len(out) * 2, np.uint8)
    return out[:m].tobytes()


def dequant(q, L, sb):
    return (q - np.sign(q) * F32(RECON)).astype(F32) * sb


FOLDW = float(os.environ.get('FOLDW', 0.75))  # 0 = noise fill is random; >0 = part of it (this share of the noise power) copies the shape of the highest lower band that has real coefficients ("copy-up" / folding)


@njit(cache=True)
def _fold(deq, q, sh, edges, sgn, out, has):
    """Texture for noise fill: in every band with few coefficients, |decoded| values of the nearest lower well-coded band, tiled over the band, random signs, unit rms."""
    F = deq.shape[0]; nb = len(edges) - 1
    for f in range(F):
        if sh[f]: continue
        last = -1
        for b in range(nb):
            lo, hi = edges[b], edges[b + 1]; w = hi - lo; nz = 0
            for k in range(lo, hi):
                if q[f, k] != 0: nz += 1
            if last >= 0 and nz * 2 < w:
                wl = edges[last + 1] - edges[last]; e = 0.0
                for k in range(lo, hi):
                    v = abs(deq[f, edges[last] + (k - lo) % wl]); out[f, k] = v * sgn[f, k]; e += v * v
                if e > 0:
                    r = 1.0 / np.sqrt(e / w)
                    for k in range(lo, hi): out[f, k] *= r
                    has[f, b] = True
                else:
                    for k in range(lo, hi): out[f, k] = 0.0
            if nz * 5 >= w: last = b


NFSM = float(os.environ.get('NFSM', 0.5))
NFIF = int(os.environ.get('NFIF', 1))
_IW = None


def _iw():  # (N, bands) interpolation weights between band centres
    c = (EDGES[:-1] + EDGES[1:]) / 2; k = np.arange(N) + 0.5; W = np.zeros((N, len(c)), F32)
    for i in range(N):
        j = np.searchsorted(c, k[i]); 
        if j == 0: W[i, 0] = 1
        elif j >= len(c): W[i, -1] = 1
        else: t = (k[i] - c[j - 1]) / (c[j] - c[j - 1]); W[i, j - 1] = 1 - t; W[i, j] = t
    return W


def fill_amp(deq, q, L, sbin):  # (frames, N) noise amplitude per bin before strength factors: the band's missing energy over its zero bins, capped at half a step
    en = np.add.reduceat(deq * deq, EDGES[:-1], axis=-1); z = np.add.reduceat((q == 0).astype(F32), EDGES[:-1], axis=-1)
    ab = np.sqrt(np.maximum(WIDTH * rhat(L) ** 2 - en, 0) / np.maximum(z, 1)).astype(F32)
    if NFSM > 0 and ab.shape[1] > 2:  # smooth the noise level over time (peaks only, so hits stay sharp) ...
        sm = ab.copy(); sm[:, 1:-1] = (1 - NFSM) * ab[:, 1:-1] + NFSM / 2 * (ab[:, :-2] + ab[:, 2:]); ab = np.minimum(sm, F32(1.4) * ab) if NFSM < 2 else ab
    if NFIF:
        global _IW
        if _IW is None or _IW[0] != EDGES.tobytes(): _IW = (EDGES.tobytes(), _iw())
        ab = ab @ _IW[1].T  # ... and across frequency (linear between band centres instead of a staircase)
    else: ab = ab[..., BMAP]
    return np.minimum(ab, F32(0.5) * sbin)


def fw(rate, nflo):  # noise weight: nflo below 500 Hz, smoothly up to 1 at 4 kHz
    s_ = np.clip(((np.arange(N) + 0.5) * rate / 2 / N - 500) / 3500, 0, 1)
    return (nflo + (1 - nflo) * s_ * s_ * (3 - 2 * s_)).astype(F32)


def seed_mask(q, L, short, zmin):  # bands that carry a noise seed: long frames, non-empty band, enough zeros (zmin 255 = none at all)
    if zmin >= 255: return np.zeros(L.shape, bool)
    return (np.add.reduceat((q == 0).astype(np.int32), EDGES[:-1], axis=-1) >= zmin) & (L > 0) & ~short[None, :, None]


def noise_plan(coef, q, L, short, sb, nf, rate, nflo, zmin, search=True, t0=0):
    """Per band, which of the SEEDS sign patterns best matches the original in the zeroed bins. Returns (bands that carry a seed, chosen index per band or None, noise amplitude per bin)."""
    F = coef.shape[1]; use = seed_mask(q, L, short, zmin)
    amp = fill_amp(dequant(q, L, sb), q, L, sb) * F32(nf) * fw(rate, nflo)
    if not search: return use, None, amp
    w = amp * coef * (q == 0)  # error of pattern p over the zeroed bins = const - 2 * sum(w * p), so the best pattern maximises sum(w * p)
    sc = np.stack([np.add.reduceat(w * PAT[((np.arange(F) + t0)[None, :] * SEEDS + j + CHOFF * np.arange(w.shape[0])[:, None]) % 251], EDGES[:-1], axis=-1) for j in range(SEEDS)])
    return use, np.argmax(sc, 0) * use, amp


RDA, RDB, RDN = float(os.environ.get('RDA', 0.1)), float(os.environ.get('RDB', 0.4)), float(os.environ.get('RDN', 4))  # deadzone rounding offset when the neighbours are all zero / busy
THREADS = min(8, os.cpu_count() or 1)
ZQ = float(os.environ.get('ZQ', 4.0))  # a +-1 becomes 0 when the decoder's own noise there is closer to the original than the reconstructed +-1 by this factor (0 = never)


RDL = float(os.environ.get('RDL', 0.25))  # rate-distortion trade: bigger = more magnitudes pulled down a step when that saves bits


def rdo(q, a, L):
    """Second pass: with a bit cost per (neighbourhood activity, magnitude) measured from this very quantisation, lower a 1 or 2 by one step when the bits saved are worth more than the added error."""
    m = np.abs(q); mc = np.minimum(m, 2).astype(np.float32); act = np.zeros_like(mc)
    act[:, :, 1:] += mc[:, :, :-1]; act[:, :, :-1] += mc[:, :, 1:]; act[:, 1:] += mc[:, :-1]; act[:, :-1] += mc[:, 1:]
    c = np.minimum(act, 6).astype(np.int32); live = (L > 0)[..., BMAP]; k = np.minimum(m, 3)
    cnt = np.ones((7, 4)); np.add.at(cnt, (c[live], k[live]), 1); bits = -np.log2(cnt / cnt.sum(1, keepdims=True))
    r_cur = bits[c, k]; r_low = bits[c, np.maximum(k - 1, 0)]; rec = lambda x: np.where(x > 0, x - RECON, 0.0)
    dd = (a - rec(m - 1)) ** 2 - (a - rec(m)) ** 2
    return np.where(live & (m >= 1) & (m <= 2) & (dd < RDL * (r_cur - r_low)), np.sign(q) * (m - 1), q).astype(np.int32)


def _quant(coef, L, short, t0, step, tilt, gamma, nf, nflo, rate, zmin, seeds, fn=None):
    sb = band_steps(L, step, gamma, fn)[..., BMAP] * weights(tilt); v = coef / sb; a = np.abs(v); q = np.copysign(np.floor(a + RND), v).astype(np.int32)
    if RDB != RDA:  # context-dependent deadzone: a lone nonzero among zeros is expensive, one among busy neighbours is cheap
        m = np.minimum(np.abs(q), 2).astype(np.float32); act = np.zeros_like(m)
        act[:, :, 1:] += m[:, :, :-1]; act[:, :, :-1] += m[:, :, 1:]; act[:, 1:] += m[:, :-1]; act[:, :-1] += m[:, 1:]
        q = np.copysign(np.floor(a + (RDA + (RDB - RDA) * np.minimum(act, RDN) / RDN)), v).astype(np.int32)
    q[(L <= 0)[..., BMAP]] = 0  # empty bands carry no coefficients
    if RDL > 0: q = rdo(q, a, L)
    F = coef.shape[1]
    if ZQ > 0 and nf > 0:  # the encoder knows exactly what noise the decoder will put in a zero: if that is closer than the +-1 would be, send a zero (also cheaper)
        use, best, amp = noise_plan(coef, q, L, short, sb, nf, rate, nflo, zmin, True, t0)
        noise = amp * PAT[((np.arange(F) + t0)[None, :, None] * SEEDS + best[..., BMAP] + CHOFF * np.arange(best.shape[0])[:, None, None]) % 251, np.arange(N)]
        one = (np.abs(q) == 1) & ~short[None, :, None]
        q[one & ((coef - noise) ** 2 < ZQ * (coef - np.sign(q) * (1 - F32(RECON)) * sb) ** 2)] = 0
    use, best, _ = noise_plan(coef, q, L, short, sb, nf, rate, nflo, zmin, seeds, t0)
    return q, use, best


MIXC = int(os.environ.get('MIXC', 1))  # 1: coefficients coded with the mixed-model coder (~2-3% smaller)


def pack(coef, L, side, tb, rate, n, step, tilt, nf, zmin=None, seeds=True, segf=None, mix=False):  # seeds=False: same size, seed bits left zero (skips the search)
    zmin = min(ZMIN if zmin is None else zmin, 255)
    step, tilt, gamma, nf, nflo, kap, tg = (float(F32(v)) for v in (step, tilt, GAMMA, nf, NFLO, KAP, TG))  # exactly what the header will hold, so encoder and decoder agree
    F = coef.shape[1]; short = np.unpackbits(np.frombuffer(tb, np.uint8, count=(F + 7) // 8))[:F].astype(bool) if tb else np.zeros(F, bool)
    cut = np.linspace(0, F, max(1, min(THREADS, F // 128)) + 1).astype(int)  # frames are independent here: quantise chunks in parallel (numpy drops the GIL)
    FN = frame_factor(L, gamma, segf); f = lambda a, b: _quant(coef[:, a:b], L[:, a:b], short[a:b], a, step, tilt, gamma, nf, nflo, rate, zmin, seeds, None if FN is None else FN[:, a:b])
    with ThreadPoolExecutor(len(cut) - 1) as ex: parts = list(ex.map(f, cut[:-1], cut[1:]))
    q, use = (np.concatenate([p[i] for p in parts], 1) for i in range(2)); best = np.concatenate([p[2] for p in parts], 1) if seeds else None
    tb2 = tb
    if segf is not None and tb:  # allocation offsets go after the short flags and band edges
        p = (F + 7) // 8 + ((1 + tb[(F + 7) // 8]) if EDGE_CUSTOM else 0)
        tb2 = tb[:p] + ((np.asarray(segf) + 32).astype(np.uint8).tobytes() if segf is not None else b"") + tb[p:]
    env = envelope(coef, env_sel(L, short)) if tb else b""
    nz = np.flatnonzero((q != 0).any(axis=(0, 1)))
    K = int(nz[-1]) + 1 if len(nz) else 1  # bins above the last nonzero one cost nothing
    out = np.empty(q.size // 2 + 4096, np.uint8); bc = expected(L, step, tilt, gamma, FN)
    enc = rc.encode_mix if mix else rc.encode
    while (m := enc(q, K, out, bc, BMAP)) < 0: out = np.empty(len(out) * 2, np.uint8)
    bits = np.unpackbits(best[use].astype(np.uint8)[:, None], axis=1)[:, 8 - SB:].reshape(-1) if seeds else np.zeros(SB * int(use.sum()), np.uint8)
    return struct.pack(HDR, rate, n, coef.shape[0], step, tilt, int(MS and coef.shape[0] == 2) | (int(ENVC) << 1) | (EDGE_CUSTOM << 2) | (int(segf is not None and bool(tb)) << 3) | (int(mix) << 4) | (int(TNS_USED and bool(tb)) << 5) | (int(DUP and bool(tb)) << 6), K, gamma, nf, len(side), kap, zmin, m, tg) + tb2 + side + env + out[:m].tobytes() + np.packbits(bits).tobytes()


def auto_cutoff(kbps):
    """Bandwidth the bits can actually pay for: below ~24 kbps the highs are cut (they only get noise at that price), more the lower the target. 0 = full band."""
    return 0 if kbps >= 24 else int(np.interp(np.log2(kbps), np.log2([1, 3, 6, 12, 24]), [5000, 7000, 10000, 14000, 20000]))


def mask_levels(L, coef, kbps, rate):
    """Material-dependent bandwidth: per frame, bands far below the loudest (A-weighted) band of that frame are dropped; the further, the lower the bitrate."""
    fc = (EDGES[:-1] + EDGES[1:]) / 2 * (rate / 2 / N); f2 = fc * fc; kf = fc / 1000
    ath = 0.37 * 10 ** ((np.minimum(3.64 * kf ** -0.8 - 6.5 * np.exp(-0.6 * (kf - 3.3) ** 2) + 1e-3 * kf ** 4, ATHC) + ATHF) / 20)  # threshold of hearing as band rms, assuming 0 dB SPL = rms 0.37 (full scale ~ 96 dB) shifted by ATHF dB for quieter playback
    quiet = ((rhat(L) < ath) | (L <= 0)).all(0, keepdims=True) & (L > 0); L, coef = np.where(quiet, 0, L), coef * ~quiet[..., BMAP]  # inaudible bands cost nothing
    if kbps >= 24: return L, coef
    aw = 20 * np.log10(12194 ** 2 * f2 ** 2 / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194 ** 2)))
    aud = np.where(L > 0, L + aw / 3.0, -99); T = float(np.interp(np.log2(kbps), np.log2([1, 3, 6, 12, 24]), [33, 39, 45, 51, 57]))
    keep = (L > 0) & (aud >= aud.max((0, 2), keepdims=True) - T / 3); keep = keep.any(0, keepdims=True) & (L > 0)  # a band stays in both channels if it is audible in either: dropping one side would leave the left or right louder than the other
    return np.where(keep, L, 0), coef * keep[..., BMAP]


def lowrate(kbps):
    """Returns (mono?, level smoothing strength). The band levels cost ~5 kbps unsmoothed, so the lower the target the more small wobbles in them are ignored."""
    return kbps < 5, float(np.interp(np.log2(kbps), np.log2([1, 2, 4, 8, 16, 32]), [96, 40, 20, 10, 6, 4]))


@njit(cache=True)
def _dp(tr, lam, a, up, S):
    """Cheapest level track per row of tr (P, F): pay (level - truth)^2 (x up when above the truth) for being off, lam for every change and a per step of change."""
    P, F = tr.shape; out = np.empty(tr.shape, np.int32)
    cost, new, f, g = np.empty(S), np.empty(S), np.empty(S), np.empty(S)
    fi, gi, src = np.empty(S, np.int64), np.empty(S, np.int64), np.empty((F, S), np.int64)
    for p in range(P):
        for j in range(S): cost[j] = (j - tr[p, 0]) ** 2 * (up if j > tr[p, 0] else 1.0)
        for t in range(1, F):
            f[0] = cost[0]; fi[0] = 0
            for j in range(1, S):
                if f[j - 1] + a < cost[j]: f[j] = f[j - 1] + a; fi[j] = fi[j - 1]
                else: f[j] = cost[j]; fi[j] = j
            g[S - 1] = cost[S - 1]; gi[S - 1] = S - 1
            for j in range(S - 2, -1, -1):
                if g[j + 1] + a < cost[j]: g[j] = g[j + 1] + a; gi[j] = gi[j + 1]
                else: g[j] = cost[j]; gi[j] = j
            for j in range(S):
                m, i = (f[j], fi[j]) if f[j] <= g[j] else (g[j], gi[j])
                if cost[j] <= lam + m: new[j] = cost[j]; src[t, j] = j
                else: new[j] = lam + m; src[t, j] = i
                new[j] += (j - tr[p, t]) ** 2 * (up if j > tr[p, t] else 1.0)
            for j in range(S): cost[j] = new[j]
        st = 0
        for j in range(S):
            if cost[j] < cost[st]: st = j
        for t in range(F - 1, -1, -1):
            out[p, t] = st
            if t > 0: st = src[t, st]
    return out


def smooth_levels(L, lam, a=0.3, up=1.5):
    """Like hold, but symmetric: small wobbles in either direction are ignored, real steps (a drum hit, a decay) are followed at once."""
    ch, F, nb = L.shape; S = int(L.max()) + 2
    tr = np.ascontiguousarray(L.transpose(0, 2, 1).reshape(-1, F)).astype(np.int32)
    return _dp(tr, float(lam), float(a), float(up), S).reshape(ch, nb, F).transpose(0, 2, 1).copy()


def envelope(coef, sel):
    """For each (channel, short frame) in sel: how loud each of the NS blocks is in each frequency group, in 3 dB steps below the loudest block (4 bits each).
    The decoder uses it to shape its noise fill in time, so a hit's noise doesn't spill before the hit. Channels with no levels in a frame get none."""
    c2 = coef[sel] ** 2  # (selected, N)
    E = np.stack([np.stack([c2[:, (GBIN == g) & (JBIN == j)].sum(-1) for j in range(NS)], -1) for g in range(3)], -2)  # (selected, 3, NS)
    q = np.clip(np.round(-np.log2(E / np.maximum(E.max(-1, keepdims=True), 1e-9) + 1e-12)), 0, 15).astype(np.int32)  # (selected, 3, NS)
    if ENVC: q = (q + 1) >> 1
    if not len(q): return b""
    a = np.ascontiguousarray(q.transpose(1, 0, 2).reshape(3, -1, 1)); out = np.empty(a.size * 2 + 4096, np.uint8)  # one track per frequency group, blocks in time order: coded like band levels
    while (m := rc.encode_side(a, out)) < 0: out = np.empty(len(out) * 2, np.uint8)
    return struct.pack("<I", m) + out[:m].tobytes()


def env_sel(L, short):  # (channels, frames): which short frames carry an envelope
    return short[None, :] & (L > 0).any(-1)


def rot(co, ang, inv=False):
    """Rotate the stereo pair of every band by its own angle: index 32 = left/right, 48 = mid/side, others follow the dominant direction. co: (2, frames, N)."""
    th = (ang.astype(np.float64) - 32) * np.pi / 64; cs, sn = np.cos(th)[BMAP].astype(F32), np.sin(th)[BMAP].astype(F32)
    if inv: sn = -sn
    return np.stack([cs * co[0] + sn * co[1], cs * co[1] - sn * co[0]])


def choose_angles(lr, g):
    """Per band, one rotation for the whole file: left/right, mid/side, or the principal axis, whichever looks cheapest to code at step g (mid/side unless clearly better)."""
    th = np.array([0.5 * np.arctan2(2 * (a * b).sum(), (a * a).sum() - (b * b).sum()) for a, b in
                   ((lr[0, :, lo:hi].astype(np.float64), lr[1, :, lo:hi].astype(np.float64)) for lo, hi in zip(EDGES[:-1], EDGES[1:]))])
    cands = [np.full(NBF, 48), np.full(NBF, 32), np.clip(np.round(th * 64 / np.pi) + 32, 0, 63).astype(int)]
    cost = []
    for ang in cands:
        co = rot(lr, ang); cost.append(np.add.reduceat(np.log2(1 + np.abs(co) / band_steps(levels(co), g, GAMMA)[..., BMAP]).sum((0, 1)), EDGES[:-1]))
    cost = np.array(cost); best = np.where(cost.min(0) < 0.99 * cost[0], cost.argmin(0), 0)
    return np.array(cands)[best, np.arange(NBF)].astype(np.uint8)


TNSON = int(os.environ.get('TNS', 1))  # temporal noise shaping on long frames: a short LPC filter across frequency flattens the frame's time envelope before quantising; the decoder undoes it, so quantisation noise and noise fill follow the envelope (sharp hits, no smear)
TNSG = float(os.environ.get('TNSG', 1.25))  # prediction gain a frame needs before the filter is used
TORD = 8
TNS_USED = 0


def tns_range(rate):  # bins the filter covers: ~1.5 kHz .. 16 kHz
    return int(round(1500 * 2 * N / rate)), min(N, int(16000 * 2 * N / rate))


@njit(cache=True)
def _tns_enc(coef, short, bs, be, thr, idx, on):
    ch, F, _ = coef.shape; n = be - bs
    r = np.zeros(TORD + 1); a = np.zeros(TORD + 1); tmp = np.zeros(TORD + 1); kq = np.zeros(TORD + 1); x = np.zeros(n)
    for c in range(ch):
        for f in range(F):
            if short[f]: continue
            for j in range(n): x[j] = coef[c, f, bs + j]
            for i in range(TORD + 1):
                acc = 0.0
                for j in range(n - i): acc += x[j] * x[j + i]
                r[i] = acc
            if r[0] < 1e-3: continue
            # Levinson-Durbin -> reflection coefficients
            e = r[0]; a[:] = 0.0; a[0] = 1.0; kk = np.zeros(TORD + 1)
            for i in range(1, TORD + 1):
                acc = r[i]
                for j in range(1, i): acc += a[j] * r[i - j]
                k = -acc / e; kk[i] = k
                for j in range(1, i): tmp[j] = a[j] + k * a[i - j]
                for j in range(1, i): a[j] = tmp[j]
                a[i] = k; e *= (1.0 - k * k)
                if e <= 1e-9 * r[0]: break
            if r[0] / max(e, 1e-12) < thr: continue
            # quantise the reflection coefficients (arcsine, 4 bits) and rebuild the filter from the quantised ones
            a[:] = 0.0; a[0] = 1.0
            for i in range(1, TORD + 1):
                q = int(np.round(np.arcsin(min(max(kk[i], -0.999), 0.999)) * 8.0 / (np.pi / 2.0)))
                q = min(max(q, -7), 7); idx[c, f, i - 1] = q; k = np.sin(q * (np.pi / 2.0) / 8.0)
                for j in range(1, i): tmp[j] = a[j] + k * a[i - j]
                for j in range(1, i): a[j] = tmp[j]
                a[i] = k
            on[c, f] = True
            for j in range(n):  # residual = A(z) x, filter state zero at the start of the range
                acc = x[j]
                for i in range(1, TORD + 1):
                    if j - i >= 0: acc += a[i] * x[j - i]
                coef[c, f, bs + j] = acc


@njit(cache=True)
def _tns_dec(deq, idx, on, short, bs, be):  # deq: (frames, N) of one channel, in place
    F = deq.shape[0]; n = be - bs; a = np.zeros(TORD + 1); tmp = np.zeros(TORD + 1); y = np.zeros(n)
    for f in range(F):
        if short[f] or not on[f]: continue
        a[:] = 0.0; a[0] = 1.0
        for i in range(1, TORD + 1):
            k = np.sin(idx[f, i - 1] * (np.pi / 2.0) / 8.0)
            for j in range(1, i): tmp[j] = a[j] + k * a[i - j]
            for j in range(1, i): a[j] = tmp[j]
            a[i] = k
        for j in range(n):  # x = e - sum a_i x[j-i]
            acc = deq[f, bs + j]
            for i in range(1, TORD + 1):
                if j - i >= 0: acc -= a[i] * y[j - i]
            y[j] = acc; deq[f, bs + j] = acc


def tns_pack(idx, on):
    """On/off bit per (channel, frame), then 8 nibbles per active frame."""
    fl = np.packbits(on.reshape(-1)).tobytes(); nib = (idx[on].reshape(-1) + 8).astype(np.uint8)
    if len(nib) % 2: nib = np.r_[nib, 0].astype(np.uint8)
    return struct.pack("<I", len(fl)) + fl + ((nib[0::2] << 4) | nib[1::2]).astype(np.uint8).tobytes()


def tns_unpack(b, o, ch, F):
    nfl = struct.unpack_from("<I", b, o)[0]; on = np.unpackbits(np.frombuffer(b, np.uint8, count=nfl, offset=o + 4))[:ch * F].astype(bool).reshape(ch, F)
    m = int(on.sum()) * TORD; nn = (m + 1) // 2; raw = np.frombuffer(b, np.uint8, count=nn, offset=o + 4 + nfl)
    nib = np.empty(nn * 2, np.int8); nib[0::2] = raw >> 4; nib[1::2] = raw & 15; idx = np.zeros((ch, F, TORD), np.int8); idx[on] = (nib[:m].reshape(-1, TORD) - 8)
    return idx, on, 4 + nfl + nn


def prepare(x, rate, cutoff, show=False, g=0.0):
    """g > 0: a guess of the final step, used to pick each band's stereo rotation (otherwise plain mid/side)."""
    short = detect(x); coef = analyze(x, short, show); tb = np.packbits(short).tobytes()
    set_edges(adaptive_edges(coef, NB) if ADAPT else EDGES0)
    if EDGE_CUSTOM: tb += bytes([NBF - 1]) + (EDGES[1:-1] // (N // 256)).astype(np.uint8).tobytes()
    if MS and x.shape[1] == 2:
        ang = choose_angles(coef, g) if g > 0 else np.full(NBF, 48, np.uint8); coef = rot(coef, ang); tb += ang.tobytes()
    coef = lowpass(coef, rate, cutoff)
    global TNS_USED
    TNS_USED = 0
    if TNSON and not SCOPE:
        ch, F, _ = coef.shape; idx = np.zeros((ch, F, TORD), np.int8); on = np.zeros((ch, F), bool); bs, be = tns_range(rate)
        _tns_enc(coef, short, bs, be, TNSG, idx, on)
        if on.any(): tb += tns_pack(idx, on); TNS_USED = 1
    return coef, levels(coef), tb


def encode(x, rate, step=1.0, show=False, tilt=TILT, cutoff=0):
    """x: float array (samples, channels). step = quantizer scale g (bigger = smaller file). Returns bytes."""
    coef, L, tb = prepare(x, rate, cutoff, show, step)
    return pack(coef, L, side_bytes(L), tb, rate, len(x), step, tilt, 1.0 if NF is None else NF)


DUP = False  # the file being coded was stereo, coded as one channel: the decoder gives both channels back
FIXED = None  # constant-quality mode: skip the bitrate search and code with exactly this step


def encode_kbps(x, rate, kbps, show=False, tilt=TILT, cutoff=0, scope=False, mono=False, step=None, automono=True):  # cutoff in Hz, 0 = full band; step = constant-quality (VBR) mode, kbps then only picks the settings; automono: two identical (or, up to 96 kbps, nearly identical) channels are coded as one
    """scope=True (oscilloscope music, anything where the left/right waveforms themselves matter): plain left/right coding, uniform noise in every band, no noise fill, no level hold."""
    global GAMMA, MS, NF, SCOPE, ENVC
    dup = False  # automatic mono: the decoder gives two channels back
    if automono and not mono and not scope and x.shape[1] == 2:
        d, m = x[:, 0].astype(np.float64) - x[:, 1], x.astype(np.float64).mean(1); dup = mono = bool(np.abs(d).max() <= 1 or (kbps <= 96 and (d * d).sum() / 2 <= 1e-5 * (m * m).sum()))
    if mono and x.shape[1] > 1: x = x.mean(1, keepdims=True).astype(F32)  # merged mono: all bits go to one channel
    saved = GAMMA, MS, NF, SCOPE, ENVC
    ENVC = kbps < 8 and not scope
    GAMMA = float(np.interp(kbps, [16, 32], [GLO, GAMMA]))  # low rates: spread the noise more evenly across bands so quiet highs (fast arps) survive
    if scope: GAMMA, MS, NF, SCOPE = 0.0, False, 0.0, True
    global KAP, TG, FIXED, DUP
    FIXED = step; DUP = dup
    TG = TGE if 'TG' in os.environ else (0.0 if scope else float(np.interp(kbps, [24, 32], [0.9, 0.0])))  # quiet passages keep their relative accuracy at low rates
    try:
        dur = len(x) / rate; kap0 = 0 if scope or kbps < 48 else float(np.interp(np.log2(kbps), np.log2([48, 64, 128]), [2.6, 2.2, 1.8]))
        for kap in ([kap0, kap0 * 1.35, kap0 * 1.8] if kap0 else []) + [0]:  # quiet bands next to loud ones keep a minimum SNR; if that does not fit the bitrate, relax it
            KAP = kap; b = _encode_kbps(x, rate, kbps, show, tilt, cutoff)
            if step or len(b) * 8 / dur / 1000 <= kbps * 1.02: break
        return b
    finally: GAMMA, MS, NF, SCOPE, ENVC = saved; KAP = 0


GLO = float(os.environ.get('GLO', 0.45))  # gamma at 16 kbps and below
SEGB = float(os.environ.get('SEGB', 0.3))  # 0 = constant step over the song (quiet/simple passages get few bits); 1 = every segment gets the same number of bits (like a constant-bitrate codec)


def choose_seg(coef, L, g, gamma, tilt, beta=None):
    """Spread bits between 'same step everywhere' and 'same bits everywhere', but never take bits from a hit: a segment that costs clearly more than its surroundings (a drum, an attack) keeps the constant step; the bits are moved between the steady segments (quiet intros get more, loud steady stretches give). Bits are estimated as sum log2(1 + |c| / step)."""
    beta = SEGB if beta is None else beta; F = coef.shape[1]; ns = -(-F // SEGL); grid = np.arange(-16, 17, 2)
    fac = frame_factor(L, gamma, None); sb = band_steps(L, g, gamma, fac)[..., BMAP] * weights(tilt); a = np.abs(coef) / sb
    B = np.zeros((ns, len(grid)))
    for j, o in enumerate(grid):
        f = np.log2(1 + a / 2.0 ** (o / 4.0)).sum((0, 2)); B[:, j] = np.pad(f, (0, ns * SEGL - F)).reshape(ns, SEGL).sum(1)
    B0 = np.maximum(B[:, list(grid).index(0)], 1.0); W = 8
    med = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(B0, W, mode="edge"), 2 * W + 1), axis=1)
    free = B0 <= HIT * med  # steady segments; the others (hits) keep the constant step
    want = B0.copy(); fb = B0[free]
    if free.any(): w = fb ** (1 - beta) * fb.mean() ** beta; want[free] = w * fb.sum() / w.sum()
    o = grid[np.abs(np.log(np.maximum(B, 1.0)) - np.log(want)[:, None]).argmin(1)].astype(np.int64)
    o[~free] = 0; return o


def _encode_kbps(x, rate, kbps, show=False, tilt=TILT, cutoff=0):
    """Find the step whose file just fits the target kbps: coarse search on a sample of frames, then a few exact passes."""
    assert KBPS_MIN <= kbps <= KBPS_MAX, f"kbps must be {KBPS_MIN}..{KBPS_MAX}"
    coef, L, tb = prepare(x, rate, cutoff, show); n = len(x); nf = auto_nf(kbps) if NF is None else NF
    zm = 255  # noise seeds cost more than they give at low bitrates, and are pointless without noise fill
    mono, lam = lowrate(kbps)
    if SCOPE: mono, lam = False, 0

    def fix(coef, L):  # very low targets: drop the second channel; cheaper level coding
        if mono and coef.shape[0] == 2 and MS: coef[1] = 0; L[1] = 0  # side channel dropped: decodes as mono
        if not SCOPE: L, coef = mask_levels(L, coef, kbps, rate)
        return coef, smooth_levels(L, lam) if lam else L
    coef, L = fix(coef, L)
    target, F = kbps * 125 * n / rate, coef.shape[1]  # bytes
    sel = np.concatenate([np.arange(b, min(F, b + 16)) for b in range(0, F, 320)]) if F > 640 else np.arange(F)  # 16-frame blocks spread over the song
    sub, Ls, scale = coef[:, sel], L[:, sel], F / len(sel)
    side_s = side_bytes(Ls)
    lo, hi = np.log(1e-3), np.log(1e6)  # lo = too big, hi = fits
    sp = []  # (log step, log size) of the sample's probes: the last ones sit next to the answer and give the local slope
    if FIXED: hi = np.log(FIXED)
    for i in range(0 if FIXED else 13):
        mid = (lo + hi) / 2; sz = len(pack(sub, Ls, side_s, b"", rate, n, np.exp(mid), tilt, nf, zm, False)) * scale; sp.append((mid, np.log(sz)))
        if sz > target: lo = mid
        else: hi = mid
        if show: bar(i + 1, 13, "estimate")
    near = np.array(sp[-5:]); slope0 = float(np.clip(np.polyfit(near[:, 0], near[:, 1], 1)[0], -3, -0.3)) if len(sp) and np.ptp(near[:, 0]) > 1e-9 else -0.8
    if MS and x.shape[1] == 2 and not mono:  # now that the step is roughly known: choose each band's stereo rotation and redo the transform
        coef, L, tb = prepare(x, rate, cutoff, False, float(np.exp(hi))); coef, L = fix(coef, L)
    segf = choose_seg(coef, L, float(np.exp(hi)), float(F32(GAMMA)), tilt) if SEGB > 0 and kbps >= 6 and not SCOPE else None
    if FIXED: return pack(coef, L, side_bytes(L), tb, rate, n, FIXED, tilt, nf, zm, True, segf, bool(MIXC))
    side, P, best, smallest, pts, ls = side_bytes(L), 7, None, None, [], hi  # the sample is only an estimate: refine on the whole song
    rm = 0.97 if MIXC else 1.0  # size of the mixed-coder file relative to the plain coder's: searched with the plain (fast) coder, then corrected from the real final size
    keep = None
    for rep in range(3 if MIXC else 1):
        tgt = target / rm; best, smallest, pts, ls = None, None, [], (hi if rep == 0 else ls)
        for i in range(P):  # secant search on log(size) vs log(step), starting from the sample's answer; keep the biggest file that fits
            sz = len(pack(coef, L, side, tb, rate, n, np.exp(ls), tilt, nf, zm, False, segf)); pts.append((ls, np.log(sz)))  # sizes only: the seed search runs once, at the end
            if sz <= tgt and (best is None or sz > best[0]): best = (sz, ls)
            if smallest is None or sz < smallest[0]: smallest = (sz, ls)
            if show: bar(i + 1, P, "fit")
            if best is not None and best[0] >= 0.99 * tgt: break  # within 1% under the target
            slope = min((pts[-1][1] - pts[-2][1]) / (pts[-1][0] - pts[-2][0]), -0.2) if len(pts) > 1 and abs(pts[-1][0] - pts[-2][0]) > 1e-6 else slope0
            ls = float(np.clip(ls + (np.log(tgt * 0.993) - pts[-1][1]) / slope, np.log(1e-3), np.log(1e6)))
        ls = (best or smallest)[1]; b = pack(coef, L, side, tb, rate, n, np.exp(ls), tilt, nf, zm, True, segf, bool(MIXC))
        if len(b) <= target and (keep is None or len(b) > len(keep)): keep = b
        if len(b) <= target and len(b) >= 0.985 * target: break
        rm = len(b) / max((best or smallest)[0], 1)
    if show: bar(P, P, "fit")
    return keep or b  # nothing fit (target below the format's floor): the last file we got


def decode(b, show=False):
    rate, n, ch, step, tilt, ms, K, gamma, nf, sl, kap, zmin, cl, tg = struct.unpack(HDR, b[:HLEN]); mixed = (ms >> 4) & 1; tnsflag = (ms >> 5) & 1; dupflag = (ms >> 6) & 1; envc, cust, segflag, ms = (ms >> 1) & 1, (ms >> 2) & 1, (ms >> 3) & 1, ms & 1
    global KAP, TG; KAP, TG = kap, tg; nflo = 1.0  # header slot formerly the unused NFLO: the step cap
    F, wts = nframes(n), weights(tilt); tl = (F + 7) // 8
    short = np.unpackbits(np.frombuffer(b, np.uint8, count=tl, offset=HLEN))[:F].astype(bool); var = varof(short)
    if cust: ni = b[HLEN + tl]; set_edges(np.r_[0, np.frombuffer(b, np.uint8, count=ni, offset=HLEN + tl + 1).astype(int) * (N // 256), N]); tl += 1 + ni
    else: set_edges(EDGES0)
    segf = None
    if segflag: ns_ = -(-F // SEGL); segf = np.frombuffer(b, np.uint8, count=ns_, offset=HLEN + tl).astype(np.int64) - 32; tl += ns_
    ang = np.frombuffer(b, np.uint8, count=NBF, offset=HLEN + tl) if ms and ch == 2 else None; tl += NBF if ang is not None else 0
    tns = None
    if tnsflag:
        tidx, ton, tn = tns_unpack(b, HLEN + tl, ch, F); tns = (tidx, ton); tl += tn
    L = rc.decode_side(np.frombuffer(b, np.uint8, count=sl, offset=HLEN + tl), ch, F, NBF)
    FN = frame_factor(L, gamma, segf); sel = env_sel(L, short); ns = int(sel.sum()); el, o = 0, HLEN + tl + sl
    if ns:
        ln = struct.unpack_from("<I", b, o)[0]; el = 4 + ln
        raw = rc.decode_side(np.frombuffer(b, np.uint8, count=ln, offset=o + 4), 3, ns * NS, 1)[..., 0].reshape(3, ns, NS).transpose(1, 0, 2)
    else: raw = np.zeros((0, 3, NS), np.int32)
    ee = 2.0 ** (-raw * (2 if envc else 1) / 2)  # relative amplitude of each block
    ENV = np.ones((ch, F, 3, NS), F32); ENV[sel] = ee / np.sqrt((ee ** 2).mean(-1, keepdims=True))  # mean-square-normalised
    c0 = HLEN + tl + sl + el; q = (rc.decode_mix if mixed else rc.decode)(np.frombuffer(b, np.uint8, count=cl, offset=c0), ch, F, K, N, expected(L, step, tilt, gamma, FN), BMAP)
    use = seed_mask(q, L, short, zmin); sd = np.zeros(use.shape, np.int64)  # noise pattern index per band
    sd[use] = np.unpackbits(np.frombuffer(b, np.uint8, offset=c0 + cl))[:SB * int(use.sum())].reshape(-1, SB) @ (1 << np.arange(SB - 1, -1, -1))
    y, FW, DEQ = np.zeros((ch, (F + 1) * N), F32), fw(rate, nflo), np.empty((ch, F, N), F32)
    for c in range(ch):
        for s in range(0, F, CHUNK):
            e = min(F, s + CHUNK); Lc, qc, sh = L[c, s:e], q[c, s:e], short[s:e]
            sbin = band_steps(Lc, step, gamma, None if FN is None else FN[c, s:e])[:, BMAP] * wts
            deq = dequant(qc, Lc, sbin)
            if nf > 0:  # noise-fill the zeros: the encoder's chosen sign pattern per band, at the band's missing energy
                amp = fill_amp(deq, qc, Lc, sbin) * F32(nf) * FW
                mult = np.ones(qc.shape, F32)
                if sh.any(): mult[sh] = ENV[c, s:e][sh][:, GBIN, JBIN]  # short frames: noise follows the block-by-block loudness
                pat = PAT[((np.arange(s, e) * SEEDS)[:, None] + sd[c, s:e][:, BMAP] + CHOFF * c) % 251, np.arange(N)[None, :]]
                if FOLDW > 0:
                    fo, hs = np.zeros(qc.shape, F32), np.zeros((e - s, NBF), bool); _fold(deq, qc, sh, EDGES, np.sign(pat[:, ::-1]).astype(F32), fo, hs)
                    pat = np.where(hs[:, BMAP], np.sqrt(F32(FOLDW)) * fo + np.sqrt(F32(1 - FOLDW)) * pat, pat)
                deq += np.where(qc == 0, pat * amp * mult, F32(0))
            if tns is not None: _tns_dec(deq, tns[0][c, s:e], tns[1][c, s:e], sh, *tns_range(rate))
            DEQ[c, s:e] = deq
    if ang is not None: DEQ = rot(DEQ, ang, True)  # back from the rotated stereo pair to left/right
    for c in range(ch):
        for s in range(0, F, CHUNK):
            e = min(F, s + CHUNK); deq, sh = DEQ[c, s:e], short[s:e]
            fr = np.empty((e - s, 2 * N), F32)
            if (~sh).any(): fr[~sh] = (deq[~sh] @ BASIS) * LONGW[var[s:e][~sh]]
            if sh.any():
                bl = (deq[sh].reshape(-1, M, NS).transpose(0, 2, 1) @ BASIS_S) * WS  # (k, NS, 2M)
                buf = np.zeros((len(bl), 2 * N), F32)
                for j in range(NS): buf[:, OFF + M * j:OFF + M * j + 2 * M] += bl[:, j]
                fr[sh] = buf
            y[c, s * N:e * N] += fr[:, :N].reshape(-1)  # TDAC overlap-add: 2nd half of frame i + 1st half of frame i+1
            y[c, (s + 1) * N:(e + 1) * N] += fr[:, N:].reshape(-1)
            if show: bar(c * F + e, ch * F, "decode")
    out = np.clip(y.T[N:N + n], -32768, 32767).astype(np.int16)
    return np.ascontiguousarray(np.repeat(out, 2, axis=1) if dupflag and out.shape[1] == 1 else out), rate  # dupflag: the input was stereo with (nearly) identical channels, coded as one


def read_wav(p):  # any libsndfile format: wav, flac, ogg, aiff...; 24-bit is reduced to 16
    import soundfile as sf
    x, r = sf.read(p, dtype="int16", always_2d=True)
    return x.astype(F32), r


def write_wav(p, y, rate):
    with wave.open(p, "wb") as w:
        w.setnchannels(y.shape[1]); w.setsampwidth(2); w.setframerate(rate); w.writeframes(y.tobytes())


if __name__ == "__main__":
    a = [v for v in sys.argv[1:] if v not in ("--scope", "--mono")]; scope = "--scope" in sys.argv; mono = "--mono" in sys.argv
    if a[:1] == ["enc"]:
        t0 = time.time(); x, r = read_wav(a[1]); v = a[3] if len(a) > 3 else "128k"; tl = float(a[4]) if len(a) > 4 else TILT; cut = float(a[5]) if len(a) > 5 else 0.0
        if len(a) > 6: GAMMA = float(a[6])
        if len(a) > 7: NF = float(a[7])
        if len(a) > 8: TRANS = float(a[8])
        if len(a) > 9: NFLO = float(a[9])
        if len(a) > 10: SMOOTH = bool(int(a[10]))
        b = encode_kbps(x, r, float(v[:-1]), True, tl, cut, scope, mono) if v.endswith("k") else encode(x, r, float(v), True, tl, cut or 0)
        open(a[2], "wb").write(b); print(f"{len(b) * 8 / (len(x) / r) / 1000:.2f} kbps, {time.time() - t0:.0f}s")
    elif a[:1] == ["dec"]:
        y, r = decode(open(a[1], "rb").read(), True); write_wav(a[2] if len(a) > 2 else a[1].rsplit(".", 1)[0] + ".wav", y, r)
    else:  # self-check: perfect-ish reconstruction through short blocks, shrinks, and hits a kbps target
        r = 44100; t = np.arange(r * 3) / r
        x = (8000 * np.sin(2 * np.pi * 440 * t) + 4000 * np.sin(2 * np.pi * 3000 * t) + np.random.randn(len(t)) * 100)[:, None].astype(F32)
        x[[30000, 60000, 61500, 100000], 0] += 30000  # clicks force short frames (and their start/stop neighbours)
        sh = detect(x); print(f"{sh.sum()} of {len(sh)} frames short")
        assert sh.sum() >= 3
        b = encode(x, r, 4.0); y, _ = decode(b)
        snr = 10 * np.log10((x ** 2).sum() / ((x - y) ** 2).sum())
        print(f"{x.size * 2} -> {len(b)} bytes ({x.size * 2 / len(b):.1f}x), SNR {snr:.1f} dB")
        assert y.shape == x.shape and snr > 40 and len(b) < x.size * 2
        got = len(encode_kbps(x, r, 64)) * 8 / 3 / 1000; print(f"target 64 kbps -> {got:.1f}"); assert got <= 64
