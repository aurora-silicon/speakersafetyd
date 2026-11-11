#!/usr/bin/env python3
"""Generate the J815 low-level speaker sweep: one exponential sine sweep per speaker, one speaker at a time.

usage: make_j815_sweep.py sweep.wav signal.json

Six S32 channels at 48 kHz in the speaker PCM's slot order. Woofers sweep 80 Hz-10 kHz at -12 dBFS, tweeters
400 Hz-16 kHz at -18 dBFS, 1.6 s each. Play it only with speakersafetyd running on the capped profile.
"""
import array, json, math, sys, wave
FS = 48000
order = ["Left Woofer 1", "Right Woofer 1", "Left Tweeter", "Right Tweeter", "Left Woofer 2", "Right Woofer 2"]
woofer = dict(f1=80.0, f2=10000.0, dbfs=-12.0)
tweeter = dict(f1=400.0, f2=16000.0, dbfs=-18.0)
LEAD, DUR, GAP, TAIL, FADE = 1.0, 1.6, 0.5, 0.5, 0.01
n_total = int(FS * (LEAD + len(order) * (DUR + GAP) + TAIL))
data = array.array("i", bytes(4 * 6 * n_total))
segs = []
for ch, name in enumerate(order):
    p = tweeter if "Tweeter" in name else woofer
    start = int(FS * (LEAD + ch * (DUR + GAP))); n = int(FS * DUR)
    amp = 10 ** (p["dbfs"] / 20) * 2147483647
    k = math.log(p["f2"] / p["f1"])
    for s in range(n):
        t = s / FS
        g = min(1.0, t / FADE, (DUR - t) / FADE)
        ph = 2 * math.pi * p["f1"] * DUR / k * (math.exp(t * k / DUR) - 1)
        data[(start + s) * 6 + ch] = int(amp * g * math.sin(ph))
    segs.append(dict(name=name, slot=ch, start_s=start / FS, dur_s=DUR, **p))
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(6); w.setsampwidth(4); w.setframerate(FS); w.writeframes(data.tobytes())
json.dump(dict(rate=FS, frames=n_total, segments=segs), open(sys.argv[2], "w"), indent=1)
