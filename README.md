# MDC audio codec

MDC ("My Dearest Codec") is a lossy audio codec built for fun and judged by ear. It uses an MDCT with a KBD window and short blocks on attacks, followed by a context-modelled range coder (`rc.py`).
Its other tools are encoder-side band allocation, temporal noise shaping, copy-up noise fill and stereo rotation. The goal is to sound better than Opus at low bitrates.

## Use

```
pip install numpy numba scipy soundfile
python codec.py enc in.flac out.mdc 16k      # target bitrate in kbps
python codec.py dec out.mdc out.wav
```

Only `codec.py` and `rc.py` are needed to encode and decode. Files in `tools/` are measurement helpers: `nmr.py` and `suite.py` score quality, `edge.py` runs stress cases, and `player.py` is an A/B player (extra deps: sounddevice, pyqtgraph).
Run them from the repo root, for example `python tools/edge.py`.

## Status

Experimental. The bitstream changes between versions, so a `.mdc` file can only be decoded by the same version that wrote it.

## License

Apache-2.0, Copyright 2026 LexonBlackzz. See `LICENSE`.
