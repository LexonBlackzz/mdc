import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""Streaming test: audio is fed in 20 ms blocks at real-time speed, an encoder process packs it into ~1 s packets (it sees C + 2E of audio per packet:
that is the lookahead), a decoder process turns the packets back into audio, and the main process plays the part of the sound card.
Packets are independent MDC files that overlap by 2E samples and are cross-faded, so this is the naive baseline: a stateful streaming bitstream would do better.

python tools/stream.py song.flac [out.wav] [--kbps 16 | 16,48@11,16@22] [--chunk 1.0] [--speed 1]
(--kbps takes a schedule: kbps@seconds, switching live; --speed 0 = as fast as possible)"""
import time, numpy as np, multiprocessing as mp

E, Z = 2048, 1024  # extra samples coded on each side of a packet; half-length of the cross-fade at a packet boundary


def target_at(sched, t):
    k = sched[0][1]
    for kb, s in sched:
        if s <= t: k = kb
    return k


def parse_sched(s):
    out = []
    for p in s.split(","):
        kb, _, at = p.partition("@"); out.append((float(kb), float(at or 0)))
    return sorted(out, key=lambda a: a[1])


def enc_worker(inq, outq, rate, ch, sched, chunk, ready):
    import codec as c
    c.encode_kbps(np.random.default_rng(0).standard_normal((rate, ch)) * 1000, rate, 16)  # spin-up (numba cache, thread pools) before the clock starts
    ready.set()
    Cn = int(chunk * rate); data = np.zeros((0, ch)); base = 0; total = 0; eof = False; i = 0; spent = 0.0; allowed = 0.0
    while True:
        while not eof and total < (i + 1) * Cn + E:  # wait for the lookahead
            blk = inq.get()
            if blk is None: eof = True
            else: data = np.concatenate([data, blk]); total += len(blk)
        if i * Cn >= total: break
        lo, hi = max(0, i * Cn - E), min(total, (i + 1) * Cn + E); last = eof and hi >= total
        t0 = time.time(); nominal = target_at(sched, i * chunk); dur_nom = min(Cn, total - i * Cn) / rate
        want = (nominal * 1000 * dur_nom + 0.5 * (allowed - spent)) / ((hi - lo) / rate) / 1000  # bit reservoir: pay back what earlier packets over/under-spent
        asked = float(np.clip(want, 0.7 * nominal * dur_nom * rate / (hi - lo), 1.3 * nominal))
        b = c.encode_kbps(np.ascontiguousarray(data[lo - base: hi - base]), rate, max(asked, 1.0), automono=False)  # packets must keep the channel count
        spent += len(b) * 8; allowed += nominal * 1000 * dur_nom
        outq.put(("pkt", i, lo, b, last, time.time(), time.time() - t0, nominal, asked, dur_nom))
        data = data[(i + 1) * Cn - E - base:] if (i + 1) * Cn - E > base else data; base = max(base, (i + 1) * Cn - E)
        i += 1
        if last: break
    outq.put(None)


def dec_worker(inq, outq, rate, ch, chunk, ready):
    import codec as c
    c.decode(c.encode_kbps(np.random.default_rng(0).standard_normal((rate // 2, ch)) * 1000, rate, 16))
    ready.set(); Cn = int(chunk * rate); prev = None; prev_lo = 0; emitted = 0
    while True:
        m = inq.get()
        if m is None: break
        _, i, lo, b, last, t_enc, enc_s, nominal, asked, dur_nom = m
        t0 = time.time(); y, _ = c.decode(b); y = np.asarray(y, np.float64); end = lo + len(y); bnd = i * Cn
        if i == 0: seg = y[:0]; start = 0
        else:  # blend the cross-fade zone of the boundary with the previous packet's copy of it
            za, zb = bnd - Z, min(bnd + Z, end); n = zb - za
            w = (np.sin(np.pi / 2 * (np.arange(n) + 0.5) / (2 * Z)) ** 2)[:, None]
            seg = prev[za - prev_lo: zb - prev_lo] * (1 - w) + y[za - lo: zb - lo] * w; start = za
        stop = end if last else (i + 1) * Cn - Z
        pcm = np.concatenate([seg, y[start + len(seg) - lo: stop - lo]]) if stop > start + len(seg) else seg
        outq.put(("pcm", start, pcm, time.time(), i, len(b), enc_s, time.time() - t0, nominal, asked, dur_nom))
        prev, prev_lo = y, lo
        if last: break
    outq.put(None)


def main():
    import argparse, threading, codec as c
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out", nargs="?", default="stream_out.wav")
    ap.add_argument("--kbps", default="16"); ap.add_argument("--chunk", type=float, default=1.0); ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args(); x, rate = c.read_wav(a.src); x = np.asarray(x, np.float64); n, ch = x.shape; sched = parse_sched(a.kbps)
    ctx = mp.get_context("spawn"); re_e, re_d = ctx.Event(), ctx.Event(); q_in, q_pkt, q_pcm = ctx.Queue(), ctx.Queue(), ctx.Queue()
    pe = ctx.Process(target=enc_worker, args=(q_in, q_pkt, rate, ch, sched, a.chunk, re_e), daemon=True)
    pd = ctx.Process(target=dec_worker, args=(q_pkt, q_pcm, rate, ch, a.chunk, re_d), daemon=True); pe.start(); pd.start()
    re_e.wait(); re_d.wait()  # both processes warmed up (the one-off spin-up happens here, before the stream starts)
    blk = int(0.02 * rate); fed = np.zeros(n); y = np.zeros((n, ch)); got = np.full(n, np.nan); stats = []
    print(f"{a.src}: {n / rate:.1f} s, {rate} Hz, {ch} ch   schedule {sched}   packet {a.chunk} s (+{2 * E} samples overlap)")
    t_start = time.time()

    def feeder():
        for k in range(0, n, blk):
            tgt = t_start + (min(k + blk, n)) / rate / (a.speed or 1e9)
            while a.speed and time.time() < tgt: time.sleep(0.002)
            fed[k: k + blk] = time.time(); q_in.put(x[k: k + blk])
        q_in.put(None)
    th = threading.Thread(target=feeder, daemon=True); th.start()
    while True:
        m = q_pcm.get()
        if m is None: break
        _, start, pcm, t_out, i, nbytes, enc_s, dec_s, nominal, asked, dur_nom = m
        y[start: start + len(pcm)] = pcm; got[start: start + len(pcm)] = t_out
        stats.append((i, nbytes, dur_nom, nominal)); lat = t_out - fed[min(start + len(pcm), n) - 1]
        print(f"  packet {i:3d} [{i * a.chunk:5.1f}s] target {nominal:5.1f}k asked {asked:5.1f}k -> {nbytes * 8 / dur_nom / 1000:5.1f}k   encode {enc_s:.2f}s decode {dec_s:.2f}s   audio out so far {start + len(pcm)} / {n}", flush=True)
    th.join(); pe.join(timeout=5); pd.join(timeout=5)
    lat = got - fed; lat = lat[~np.isnan(lat)]
    print(f"\nlatency input->output (s): min {lat.min():.2f}  mean {lat.mean():.2f}  max {lat.max():.2f}   (all {n / rate:.1f} s of audio came out: {np.isfinite(got).all()})")
    tot = sum(s[1] for s in stats) * 8 / (n / rate) / 1000; print(f"average bitrate {tot:.1f} kbps")
    c.write_wav(a.out, np.clip(np.round(y), -32768, 32767).astype(np.int16), rate)
    import suite, nmr
    snr = lambda ref, d, lo=0, hi=None: float(10 * np.log10((ref[lo:hi] ** 2).sum() / (((ref[lo:hi] - d[lo:hi]) ** 2).sum() + 1e-9)))
    print(f"streamed : SNR {snr(x, y):5.2f} dB   nmr {nmr.nmr(x, y, rate, 0, n / rate)['all']:.2f}")
    if len(sched) == 1:
        b = c.encode_kbps(x, rate, tot); yo, _ = c.decode(b); yo = np.asarray(yo, np.float64)
        print(f"offline  : SNR {snr(x, yo):5.2f} dB   nmr {nmr.nmr(x, yo, rate, 0, n / rate)['all']:.2f}   at the same {len(b) * 8 / (n / rate) / 1000:.1f} kbps (whole-file encoder)")
    else:
        for kb, s in sched:
            e = min([t for _, t in sched if t > s] + [n / rate]); lo, hi = int(s * rate), int(e * rate)
            print(f"  {s:5.1f}-{e:5.1f} s  schedule {kb:4.0f}k   SNR {snr(x, y, lo, hi):5.2f} dB")


if __name__ == "__main__":
    main()
