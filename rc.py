# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LexonBlackzz
"""Adaptive binary range coder (LZMA-style carry handling) + context models for the codec's two streams.

coefficients: unary magnitude bins (adaptive), Exp-Golomb escape (raw bits), raw sign.
  context = how big the decoder *expects* this band to be (from the band level) x already-coded neighbours
  (two bins below, three bins around it in the previous frame, and for channel 1 the same bin in channel 0).
band levels (side info): delta vs the previous frame, context = band id x the previous band's delta.
"""
import numpy as np
from numba import njit

SC = 4  # neighbourhood mean magnitude that starts shifting values down
UN, NB, NC, SU, TOP, PB = 14, 12, 14, 14, 1 << 24, 15  # unary bins, expected-size buckets, neighbour contexts, side slots, range floor, prob bits
LUT = np.searchsorted([1, 2, 3, 4, 6, 8, 11, 15, 21, 30, 44, 66, 100], np.arange(256), "right").astype(np.int8)  # neighbour sum -> 0..13


@njit(cache=True)
def _shift(st, out):  # st = [low, range, cache, cache_size, pos]
    low = st[0]
    if low < 0xFF000000 or low >= (1 << 32):
        carry, temp = low >> 32, st[2]
        while True:  # emit the cached byte and any pending 0xFF bytes, carry propagated
            out[st[4]] = (temp + carry) & 0xFF; st[4] += 1; temp = 0xFF; st[3] -= 1
            if st[3] == 0: break
        st[2] = (low >> 24) & 0xFF
    st[3] += 1
    st[0] = (low & 0x00FFFFFF) << 8


@njit(cache=True)
def _bit(st, out, pa, pb, i, bit):  # two adaptive estimates (fast + slow), averaged
    bound = (st[1] >> PB) * ((pa[i] + pb[i]) >> 1)
    if bit == 0:
        st[1] = bound; pa[i] += (32768 - pa[i]) >> 4; pb[i] += (32768 - pb[i]) >> 7
    else:
        st[0] += bound; st[1] -= bound; pa[i] -= pa[i] >> 4; pb[i] -= pb[i] >> 7
    while st[1] < TOP:
        st[1] <<= 8; _shift(st, out)


@njit(cache=True)
def _byp(st, out, bit):  # equiprobable bit
    st[1] >>= 1
    if bit: st[0] += st[1]
    while st[1] < TOP:
        st[1] <<= 8; _shift(st, out)


@njit(cache=True)
def _eg(st, out, e):  # Exp-Golomb, e >= 0: unary bit-length, then the low bits (all raw)
    x = e + 1; nb = 0
    while (x >> (nb + 1)) > 0: nb += 1
    for _ in range(nb): _byp(st, out, 1)
    _byp(st, out, 0)
    for j in range(nb - 1, -1, -1): _byp(st, out, (x >> j) & 1)


@njit(cache=True)
def _usum(q, c, t, k, K):  # uncapped weighted neighbour magnitude (weights sum to 9, 11 with the other channel)
    m = 0
    if k > 0: m += 2 * abs(np.int64(q[c, t, k - 1]))
    if k > 1: m += abs(np.int64(q[c, t, k - 2]))
    if t > 0:
        m += 2 * abs(np.int64(q[c, t - 1, k]))
        if k > 0: m += abs(np.int64(q[c, t - 1, k - 1]))
        if k + 1 < K: m += abs(np.int64(q[c, t - 1, k + 1]))
    if t > 1: m += abs(np.int64(q[c, t - 2, k]))
    if c > 0:
        m += 2 * abs(np.int64(q[0, t, k]))
        if k > 0: m += abs(np.int64(q[0, t, k - 1]))
        if k + 1 < K: m += abs(np.int64(q[0, t, k + 1]))
    return m


@njit(cache=True)
def _shof(s):  # loud neighbourhood: drop this many low bits from the value before modelling it (they are coded almost raw)
    mm, sh = s >> 3, 0
    while (mm >> sh) >= SC: sh += 1
    return sh


@njit(cache=True)
def _nbr(q, c, t, k, K, sh):  # sum of capped neighbour magnitudes (after dropping sh low bits), max 150
    m = 0
    if k > 0: m += 2 * min(abs(q[c, t, k - 1]) >> sh, 15)
    if k > 1: m += min(abs(q[c, t, k - 2]) >> sh, 15)
    if t > 0:
        m += 2 * min(abs(q[c, t - 1, k]) >> sh, 15)
        if k > 0: m += min(abs(q[c, t - 1, k - 1]) >> sh, 15)
        if k + 1 < K: m += min(abs(q[c, t - 1, k + 1]) >> sh, 15)
    if t > 1: m += min(abs(q[c, t - 2, k]) >> sh, 15)
    if c > 0:
        m += 2 * min(abs(q[0, t, k]) >> sh, 15)
        if k > 0: m += min(abs(q[0, t, k - 1]) >> sh, 15)
        if k + 1 < K: m += min(abs(q[0, t, k + 1]) >> sh, 15)
    return m


@njit(cache=True)
def encode(q, K, out, bctx, bmap):
    """q: int32 (ch, frames, N), only bins < K coded. bctx: int8 (ch, frames, bands) expected-size bucket, -1 = empty band (q must be 0 there).
    bmap: bin -> band. Returns bytes written, or -1 if `out` is too small."""
    ch, F, _ = q.shape
    pa = np.full(ch * NB * NC * UN, 16384, np.int32); pb = pa.copy(); la = np.full(64 * 4, 16384, np.int32); lb = la.copy()
    st = np.zeros(5, np.int64); st[1] = 0xFFFFFFFF; st[3] = 1
    lim = len(out) - (K * 24 + 64)  # worst case bytes for one frame
    for c in range(ch):
        for t in range(F):
            if st[4] > lim: return -1
            for k in range(K):
                bc = bctx[c, t, bmap[k]]
                if bc < 0: continue
                v = q[c, t, k]; a = abs(v); sh = _shof(_usum(q, c, t, k, K)); hi = a >> sh
                base = ((c * NB + max(bc - 2 * sh, 0)) * NC + LUT[_nbr(q, c, t, k, K, sh)]) * UN
                for i in range(UN):
                    if hi > i: _bit(st, out, pa, pb, base + i, 1)
                    else:
                        _bit(st, out, pa, pb, base + i, 0); break
                if hi >= UN: _eg(st, out, hi - UN)
                if sh > 0:  # low bits: the top one adaptive, the rest raw
                    _bit(st, out, la, lb, min(sh, 31) * 4 + min(hi, 3), (a >> (sh - 1)) & 1)
                    for j in range(sh - 2, -1, -1): _byp(st, out, (a >> j) & 1)
                if a: _byp(st, out, 1 if v < 0 else 0)
    for _ in range(5): _shift(st, out)
    return st[4]


# ---- mixed coefficient coder: several context models per unary decision, combined by a small online-trained logistic mixer ----
NM = 7; NIN = 2 * NM + 1  # models (each gives a fast and a slow estimate) + bias
_SQ = np.array([int(round(4096 / (1 + np.exp(-(d) / 256.0)))) for d in range(-2048, 2048)], np.int64).clip(1, 4095)  # squash: logistic domain -> 12-bit probability
_ST = np.zeros(4096, np.int64)
_pi = 0
for _d in range(-2047, 2048):  # stretch = inverse of squash
    _v = int(_SQ[_d + 2048])
    for _x in range(_pi, _v + 1): _ST[_x] = _d
    _pi = _v + 1
for _x in range(_pi, 4096): _ST[_x] = 2047
_RT = np.array([int(65536 / (n + 1.6)) for n in range(256)], np.int64)  # adaptation rate by hit count (slow estimate)
MSZ = max(2 * NB * 8 * UN, 2 * 16 * 6 * UN, 2 * 8 * 8 * 4 * UN, 2 * 256 * 3 * UN, 2 * NB * NC * UN, 2 * 64 * 4 * UN, 2 * 8 * 8 * 2 * UN)  # slots per model (upper bound over the five layouts)


@njit(cache=True)
def _mctx(q, c, t, k, K, sh, bc, bmap, base):
    """Context base offsets (without the unary index) of the NM models for coefficient (c, t, k)."""
    cc = min(c, 1)  # channels beyond the second share the second one's statistics
    s0 = min(abs(q[c, t - 1, k]) >> sh, 7) if t > 0 else 0
    s1 = min(abs(q[c, t, k - 1]) >> sh, 7) if k > 0 else 0
    s2 = min(abs(q[c, t - 2, k]) >> sh, 3) if t > 1 else 0
    nb = _nbr(q, c, t, k, K, sh)
    e = max(bc - 2 * sh, 0)
    base[0] = ((cc * NB + e) * NC + LUT[nb]) * UN
    base[1] = (((cc * 8 + s0) * 8 + s1) * 4 + s2) * UN
    base[2] = ((cc * 64 + bmap[k]) * 4 + min(LUT[nb] >> 2, 3)) * UN
    if c > 0:
        o0 = min(abs(q[0, t, k]) >> sh, 7)
        o1 = min((abs(q[0, t, k - 1]) + (abs(q[0, t, k + 1]) if k + 1 < K else 0)) >> sh, 7) if k > 0 else 0
    else:
        o0 = min(abs(q[c, t - 1, k + 1]) >> sh, 7) if (t > 0 and k + 1 < K) else 0
        o1 = min(abs(q[c, t - 1, k - 1]) >> sh, 7) if (t > 0 and k > 0) else 0
    base[3] = (((cc * 8 + o0) * 8 + o1) * 2 + (1 if e > 5 else 0)) * UN
    base[4] = ((cc * 256 + (k >> 2)) * 3 + min(e // 4, 2)) * UN
    base[5] = ((cc * NB + e) * 8 + s0) * UN
    w = 0
    if t > 0:
        for d in range(-3, 4):
            if 0 <= k + d < K: w += abs(q[c, t - 1, k + d])
    base[6] = ((cc * 16 + min(w >> sh, 15)) * 6 + e // 2) * UN


@njit(cache=True)
def _mpred(i, base, P1, P2, W, ws, stv, idx):
    """12-bit probability that the unary decision i is 'greater' (bit 1); fills stv/idx for the update."""
    dot = 0
    for j in range(NM):
        ix = j * MSZ + base[j] + i; idx[j] = ix
        a = _ST[P1[ix] >> 4]; b = _ST[P2[ix] >> 4]; stv[2 * j] = a; stv[2 * j + 1] = b
        dot += W[ws + 2 * j] * a + W[ws + 2 * j + 1] * b
    stv[2 * NM] = 256; dot += W[ws + 2 * NM] * 256
    d = min(max(dot >> 16, -2047), 2047)
    return _SQ[d + 2048]


@njit(cache=True)
def _mupd(bit, p12, P1, P2, N2, W, ws, stv, idx):
    err = ((bit << 12) - p12) * 8
    for j in range(NIN): W[ws + j] += (stv[j] * err + 0x8000) >> 16
    y = bit << 16
    for j in range(NM):
        ix = idx[j]
        P1[ix] += (y - P1[ix]) >> 4
        n = N2[ix]; P2[ix] += ((y - P2[ix]) * _RT[n]) >> 16
        if n < 120: N2[ix] = n + 1
        P1[ix] = min(max(P1[ix], 16), 65520); P2[ix] = min(max(P2[ix], 16), 65520)


@njit(cache=True)
def _bitp(st, out, p12, bit):  # p12 = 12-bit probability of a 1
    p0 = min(max((4096 - p12) << 3, 64), 32768 - 64)
    bound = (st[1] >> PB) * p0
    if bit == 0: st[1] = bound
    else: st[0] += bound; st[1] -= bound
    while st[1] < TOP:
        st[1] <<= 8; _shift(st, out)


@njit(cache=True)
def encode_mix(q, K, out, bctx, bmap):
    """Same stream layout as encode() but the unary decisions are coded with the mixed model."""
    ch, F, _ = q.shape
    P1 = np.full(NM * MSZ, 32768, np.int64); P2 = P1.copy(); N2 = np.zeros(NM * MSZ, np.int64)
    W = np.full(2 * NB * UN * NIN, 1 << 12, np.int64); stv = np.zeros(NIN, np.int64); idx = np.zeros(NM, np.int64); base = np.zeros(NM, np.int64)
    la = np.full(64 * 4, 16384, np.int32); lb = la.copy()
    st = np.zeros(5, np.int64); st[1] = 0xFFFFFFFF; st[3] = 1
    lim = len(out) - (K * 24 + 64)
    for c in range(ch):
        for t in range(F):
            if st[4] > lim: return -1
            for k in range(K):
                bc = bctx[c, t, bmap[k]]
                if bc < 0: continue
                v = q[c, t, k]; a = abs(v); sh = _shof(_usum(q, c, t, k, K)); hi = a >> sh
                _mctx(q, c, t, k, K, sh, bc, bmap, base)
                wsel = (min(c, 1) * NB + max(bc - 2 * sh, 0) // 4) * UN
                for i in range(UN):
                    ws = (wsel + i) * NIN; p12 = _mpred(i, base, P1, P2, W, ws, stv, idx)
                    bit = 1 if hi > i else 0
                    _bitp(st, out, p12, bit); _mupd(bit, p12, P1, P2, N2, W, ws, stv, idx)
                    if bit == 0: break
                if hi >= UN: _eg(st, out, hi - UN)
                if sh > 0:
                    _bit(st, out, la, lb, min(sh, 31) * 4 + min(hi, 3), (a >> (sh - 1)) & 1)
                    for j in range(sh - 2, -1, -1): _byp(st, out, (a >> j) & 1)
                if a: _byp(st, out, 1 if v < 0 else 0)
    for _ in range(5): _shift(st, out)
    return st[4]


@njit(cache=True)
def encode_side(L, out):
    """L: int32 (ch, frames, bands) levels, >= 0, bands <= 64. Returns bytes written, or -1 if `out` is too small."""
    ch, F, nb = L.shape
    pa = np.full(ch * 64 * 5 * SU, 16384, np.int32); pb = pa.copy()
    st = np.zeros(5, np.int64); st[1] = 0xFFFFFFFF; st[3] = 1
    for c in range(ch):
        for t in range(F):
            if st[4] > len(out) - (nb * 24 + 64): return -1
            pd = 0
            for b in range(nb):
                pred = L[c, t - 1, b] if t > 0 else (L[c, t, b - 1] if b > 0 else 0)
                d = L[c, t, b] - pred
                base = ((c * 64 + b) * 5 + min(max(pd, -2), 2) + 2) * SU
                _bit(st, out, pa, pb, base, 1 if d != 0 else 0)
                if d != 0:
                    _bit(st, out, pa, pb, base + 1, 1 if d < 0 else 0)
                    m = abs(d) - 1
                    for i in range(11):
                        if m > i: _bit(st, out, pa, pb, base + 2 + i, 1)
                        else:
                            _bit(st, out, pa, pb, base + 2 + i, 0); break
                    if m >= 11: _eg(st, out, m - 11)
                pd = d
    for _ in range(5): _shift(st, out)
    return st[4]


@njit(cache=True)
def _rd(st, buf):
    b = buf[st[2]] if st[2] < len(buf) else 0
    st[2] += 1
    return b


@njit(cache=True)
def _dbit(st, buf, pa, pb, i):  # st = [code, range, pos]
    bound = (st[1] >> PB) * ((pa[i] + pb[i]) >> 1)
    if st[0] < bound:
        st[1] = bound; pa[i] += (32768 - pa[i]) >> 4; pb[i] += (32768 - pb[i]) >> 7; bit = 0
    else:
        st[0] -= bound; st[1] -= bound; pa[i] -= pa[i] >> 4; pb[i] -= pb[i] >> 7; bit = 1
    while st[1] < TOP:
        st[1] <<= 8; st[0] = (st[0] << 8) | _rd(st, buf)
    return bit


@njit(cache=True)
def _dbyp(st, buf):
    st[1] >>= 1
    if st[0] >= st[1]: st[0] -= st[1]; bit = 1
    else: bit = 0
    while st[1] < TOP:
        st[1] <<= 8; st[0] = (st[0] << 8) | _rd(st, buf)
    return bit


@njit(cache=True)
def _deg(st, buf):
    nb = 0
    while nb < 40 and _dbyp(st, buf) == 1: nb += 1
    x = 1
    for _ in range(nb): x = (x << 1) | _dbyp(st, buf)
    return x - 1


@njit(cache=True)
def _start(buf):
    st = np.zeros(3, np.int64); st[1] = 0xFFFFFFFF
    for _ in range(5): st[0] = (st[0] << 8) | _rd(st, buf)  # first byte is always 0
    return st


@njit(cache=True)
def decode(buf, ch, F, K, N, bctx, bmap):
    q = np.zeros((ch, F, N), np.int32)
    pa = np.full(ch * NB * NC * UN, 16384, np.int32); pb = pa.copy(); la = np.full(64 * 4, 16384, np.int32); lb = la.copy()
    st = _start(buf)
    for c in range(ch):
        for t in range(F):
            for k in range(K):
                bc = bctx[c, t, bmap[k]]
                if bc < 0: continue
                sh = _shof(_usum(q, c, t, k, K))
                base = ((c * NB + max(bc - 2 * sh, 0)) * NC + LUT[_nbr(q, c, t, k, K, sh)]) * UN
                hi = 0
                while hi < UN and _dbit(st, buf, pa, pb, base + hi) == 1: hi += 1
                if hi == UN: hi += _deg(st, buf)
                a = hi << sh
                if sh > 0:
                    lo = _dbit(st, buf, la, lb, min(sh, 31) * 4 + min(hi, 3))
                    for j in range(sh - 2, -1, -1): lo = (lo << 1) | _dbyp(st, buf)
                    a |= lo
                if a: q[c, t, k] = -a if _dbyp(st, buf) else a
    return q


@njit(cache=True)
def _dbitp(st, buf, p12):
    p0 = min(max((4096 - p12) << 3, 64), 32768 - 64)
    bound = (st[1] >> PB) * p0
    if st[0] < bound:
        st[1] = bound; bit = 0
    else:
        st[0] -= bound; st[1] -= bound; bit = 1
    while st[1] < TOP:
        st[1] <<= 8; st[0] = (st[0] << 8) | _rd(st, buf)
    return bit


@njit(cache=True)
def decode_mix(buf, ch, F, K, N, bctx, bmap):
    q = np.zeros((ch, F, N), np.int32)
    P1 = np.full(NM * MSZ, 32768, np.int64); P2 = P1.copy(); N2 = np.zeros(NM * MSZ, np.int64)
    W = np.full(2 * NB * UN * NIN, 1 << 12, np.int64); stv = np.zeros(NIN, np.int64); idx = np.zeros(NM, np.int64); base = np.zeros(NM, np.int64)
    la = np.full(64 * 4, 16384, np.int32); lb = la.copy()
    st = _start(buf)
    for c in range(ch):
        for t in range(F):
            for k in range(K):
                bc = bctx[c, t, bmap[k]]
                if bc < 0: continue
                sh = _shof(_usum(q, c, t, k, K))
                _mctx(q, c, t, k, K, sh, bc, bmap, base)
                wsel = (min(c, 1) * NB + max(bc - 2 * sh, 0) // 4) * UN
                hi = 0
                while hi < UN:
                    ws = (wsel + hi) * NIN; p12 = _mpred(hi, base, P1, P2, W, ws, stv, idx)
                    bit = _dbitp(st, buf, p12); _mupd(bit, p12, P1, P2, N2, W, ws, stv, idx)
                    if bit == 0: break
                    hi += 1
                if hi == UN: hi += _deg(st, buf)
                a = hi << sh
                if sh > 0:
                    lo = _dbit(st, buf, la, lb, min(sh, 31) * 4 + min(hi, 3))
                    for j in range(sh - 2, -1, -1): lo = (lo << 1) | _dbyp(st, buf)
                    a |= lo
                if a: q[c, t, k] = -a if _dbyp(st, buf) else a
    return q


@njit(cache=True)
def decode_side(buf, ch, F, nb):
    L = np.zeros((ch, F, nb), np.int32)
    pa = np.full(ch * 64 * 5 * SU, 16384, np.int32); pb = pa.copy()
    st = _start(buf)
    for c in range(ch):
        for t in range(F):
            pd = 0
            for b in range(nb):
                pred = L[c, t - 1, b] if t > 0 else (L[c, t, b - 1] if b > 0 else 0)
                base = ((c * 64 + b) * 5 + min(max(pd, -2), 2) + 2) * SU
                d = 0
                if _dbit(st, buf, pa, pb, base) == 1:
                    neg = _dbit(st, buf, pa, pb, base + 1)
                    m = 0
                    while m < 11 and _dbit(st, buf, pa, pb, base + 2 + m) == 1: m += 1
                    if m == 11: m += _deg(st, buf)
                    d = -(m + 1) if neg else m + 1
                L[c, t, b] = pred + d
                pd = d
    return L


if __name__ == "__main__":  # exactness check: random data incl. huge values and empty bands must round-trip bit-exactly
    g = np.random.default_rng(0)
    bmap = np.repeat(np.arange(8), 8).astype(np.int16)
    bctx = g.integers(-1, NB, (2, 50, 8)).astype(np.int8)
    q = np.round(g.laplace(0, 3, (2, 50, 64)) * np.linspace(1, 0.05, 64)).astype(np.int32)
    q[0, 3, 5] = 10 ** 6; q[1, 7, 9] = -2 ** 30; bctx[0, 3, 0] = bctx[1, 7, 1] = 5
    q[bctx[:, :, bmap] < 0] = 0
    out = np.empty(1 << 20, np.uint8); m = encode(q, 64, out, bctx, bmap)
    assert (decode(out[:m], 2, 50, 64, 64, bctx, bmap) == q).all(); print(f"coef stream OK: {q.size} values -> {m} bytes")
    L = np.maximum(0, np.cumsum(g.integers(-3, 4, (2, 60, 22)), 1) + 20).astype(np.int32); L[1, 9, 4] = 900
    m = encode_side(L, out); assert (decode_side(out[:m], 2, 60, 22) == L).all(); print(f"side stream OK: {L.size} levels -> {m} bytes")
