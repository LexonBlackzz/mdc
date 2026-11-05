import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""Player for .mdc (or wav/flac, handy for A/B) with MiniMeters-style views.
python player.py song.mdc        Space = pause, Left/Right = seek 5s
python player.py song.mdc --shot out.png   (headless: render a frame, no audio)
"""
import sys
import numpy as np, sounddevice as sd, pyqtgraph as pg
from scipy.signal import lfilter
from pyqtgraph.Qt import QtCore
import codec

NFFT, BINS, HIST, FPS = 4096, 256, 400, 30
NOTES = "C C# D D# E F F# G G# A A# B".split()


def load(path):
    if path.endswith(".mdc"): y, r = codec.decode(open(path, "rb").read(), True)
    else: x, r = codec.read_wav(path); y = x.astype(np.int16)
    return (np.repeat(y, 2, 1) if y.shape[1] == 1 else y[:, :2]), r  # always stereo


def kweight(fs):
    """ITU-R BS.1770 K-weighting biquads (shelf + highpass) for any sample rate."""
    K = np.tan(np.pi * 1681.9744509555319 / fs); Q = 0.7071752369554196
    Vh = 10 ** (3.999843853973347 / 20); Vb = Vh ** 0.4996667741545416; a0 = 1 + K / Q + K * K
    s1 = ([(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0], [1, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0])
    K = np.tan(np.pi * 38.13547087613982 / fs); Q = 0.5003270373253953; a0 = 1 + K / Q + K * K
    return s1, ([1, -2, 1], [1, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0])


class Player(pg.GraphicsLayoutWidget):
    def __init__(self, path):
        super().__init__()
        self.y, self.r = load(path); self.pos, self.paused = 0, False
        self.kw, self.f = kweight(self.r), np.fft.rfftfreq(NFFT, 1 / self.r)
        self.logf = np.geomspace(20, min(20000, self.r / 2), BINS)
        self.win, self.sm = np.hanning(NFFT).astype(np.float32), np.full(len(self.f), -120.0)
        self.hist = np.full((HIST, BINS), -120.0)
        self.setWindowTitle(path); self.resize(1100, 800)

        self.sp = self.addPlot(row=0, col=0, colspan=2)  # spectrum
        self.sp.setLogMode(x=True); self.sp.setXRange(np.log10(20), np.log10(20000)); self.sp.setYRange(-100, 0)
        self.curve = self.sp.plot(pen=(167, 139, 250), fillLevel=-100, brush=(167, 139, 250, 70))
        self.sg = self.addPlot(row=1, col=0, colspan=2)  # spectrogram: x = time, y = freq (low at bottom)
        self.sg.hideAxis("left"); self.sg.hideAxis("bottom"); self.sg.setMouseEnabled(False, False)
        self.img = pg.ImageItem(); self.sg.addItem(self.img)
        try: self.img.setLookupTable(pg.colormap.get("magma", source="matplotlib").getLookupTable(alpha=False))
        except Exception: pass
        self.scope = self.addPlot(row=2, col=0); self.scope.setYRange(-1, 1); self.scope.hideAxis("bottom")
        self.sc = self.scope.plot(pen=(167, 139, 250))
        self.vec = self.addPlot(row=2, col=1); self.vec.setAspectLocked(); self.vec.setRange(xRange=(-1, 1), yRange=(-1, 1))
        self.vs = self.vec.plot(pen=None, symbol="o", symbolSize=2, symbolPen=None, symbolBrush=(120, 220, 160, 140))

        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self.update_views); self.timer.start(1000 // FPS)

    def callback(self, out, frames, t, status):
        c = self.y[self.pos:self.pos + frames] if not self.paused else self.y[:0]
        out[:len(c)] = c; out[len(c):] = 0; self.pos += len(c)

    def lufs(self, p):  # momentary (400 ms) K-weighted loudness; 100 ms lead-in lets the filters settle
        w, tot = int(.1 * self.r), 0
        seg = self.y[max(0, p - int(.5 * self.r)):p].astype(np.float32) / 32768
        if len(seg) <= w: return -99.0
        for c in range(2):
            v = seg[:, c]
            for b, a in self.kw: v = lfilter(b, a, v)
            tot += np.mean(v[w:] ** 2)
        return -0.691 + 10 * np.log10(tot + 1e-12)

    def update_views(self):
        p = min(self.pos, len(self.y))
        seg = np.zeros((NFFT, 2), np.float32); h = self.y[max(0, p - NFFT):p]
        seg[NFFT - len(h):] = h / 32768
        mono = seg.mean(1)
        spec = 20 * np.log10(np.abs(np.fft.rfft(mono * self.win)) / (NFFT / 4) + 1e-6)
        self.sm = np.maximum(spec, self.sm - 1.5)  # fast attack, slow release
        self.curve.setData(self.f[1:], self.sm[1:])
        self.hist = np.roll(self.hist, -1, 0); self.hist[-1] = np.interp(self.logf, self.f, spec)
        self.img.setImage(self.hist, levels=(-100, -10))
        i = int(np.argmax(spec[1:])) + 1; fr = self.f[i]; n = 69 + 12 * np.log2(fr / 440); k = round(n)
        peak = f"{spec[i]:.1f}dB | {fr / 1000:.2f}kHz | {NOTES[k % 12]}{k // 12 - 1} {round((n - k) * 100):+d}c" if spec[i] > -70 else "-"
        cl = lambda s: f"{int(s // 60)}:{int(s % 60):02d}"
        self.sp.setTitle(f"{self.lufs(p):.1f} LUFS   |   {peak}   |   {cl(p / self.r)} / {cl(len(self.y) / self.r)}" + ("   [paused]" if self.paused else ""))
        z = np.nonzero((mono[:2048] < 0) & (mono[1:2049] >= 0))[0]; s = z[0] if len(z) else 0  # rising zero-cross trigger
        self.sc.setData(mono[s:s + 1024])
        L, R = seg[-1024:, 0], seg[-1024:, 1]
        self.vs.setData((L - R) * .7071, (L + R) * .7071)

    def keyPressEvent(self, e):
        k = e.key()
        if k == QtCore.Qt.Key_Space: self.paused = not self.paused
        elif k in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
            self.pos = int(np.clip(self.pos + (5 if k == QtCore.Qt.Key_Right else -5) * self.r, 0, len(self.y)))
        elif k == QtCore.Qt.Key_Escape: self.close()


if __name__ == "__main__":
    app = pg.mkQApp(); w = Player(sys.argv[1]); w.show()
    if "--shot" in sys.argv:  # headless check: fast-forward ~4 s through the views, save a screenshot
        w.timer.stop(); w.pos = len(w.y) // 3
        for _ in range(HIST): w.pos += w.r // FPS; w.update_views()
        app.processEvents(); w.grab().save(sys.argv[sys.argv.index("--shot") + 1])
    else:
        with sd.OutputStream(samplerate=w.r, channels=2, dtype="int16", callback=w.callback): app.exec_()
