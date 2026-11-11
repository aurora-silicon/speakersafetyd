#!/usr/bin/env python3
"""Analyse a J815 speaker sweep recorded through speakersafetyd's sense capture.

usage: verify_j815_sweep.py <sweep dir> [j815.conf] [report.json]

The sweep dir holds signal.json (from make_j815_sweep.py), sense.raw or sense.raw.gz (from the daemon's
--sense-capture while the sweep played) and, optionally, daemon.log (the daemon's -v output for the power replay).

For each of the six speakers (played one at a time) it checks, from speakersafetyd's raw 12-channel S16 sense capture:
  1. map:   which sense pair answers, and that the other five stay quiet (crosstalk);
  2. order: voltage vs current inside the pair. With the right order the impedance has a resonance PEAK and rises
            at high frequency (voice-coil inductance); swapped, both invert (|Z_swapped| ~ 13.9 / |Z|);
  3. lag:   the current-vs-voltage sample offset. A one-sample error is ~75 deg of phase at 10 kHz, so only one lag
            leaves the high-frequency phase physical (between 0 and +80 deg, positive real part everywhere).
            Sign: lag +n = the current channel lags the voltage by n samples; speakersafetyd's
            current_delay_samples = n means the VOLTAGE lags (p = v[k] * i[k-n]), i.e. lag -n here;
  4. R:     the impedance minimum in the band (~ the voice-coil resistance) against the profile's z_nominal.
It also replays the daemon's per-period SENSE power lines (-v) against the capture, as testing/verify_j813_power.py
does for J813. Nothing here raises a limit; it only says whether the profile's channel layout is right.
"""
import gzip, json, math, re, sys
from pathlib import Path
import numpy as np

FS = 48000
CH = 12


def load_conf(path):
    spk, cur, glob = {}, None, {}
    for line in Path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(r'\[Speaker/(.+)\]', line)
        if m:
            cur = spk.setdefault(m[1], {}); continue
        if line.startswith('['):
            cur = glob if line == '[Globals]' else None; continue
        if cur is not None and '=' in line:
            k, v = (x.strip() for x in line.split('=', 1)); cur[k] = v
    return glob, spk


def smooth(x, n):
    if n < 2:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, mode='same')


def impedance(v, i, f1, f2, lag, nfft=4096, hop=512):
    """Z(f) = V/I (H1 estimate: averaged cross-spectrum over Hann blocks), current shifted by `lag` samples
    (positive: the current channel lags the voltage), inside [f1, f2]."""
    if lag > 0:
        v, i = v[:-lag], i[lag:]
    elif lag < 0:
        v, i = v[-lag:], i[:lag]
    win = np.hanning(nfft)
    svi = np.zeros(nfft // 2 + 1, complex); sii = np.zeros(nfft // 2 + 1); svv = np.zeros(nfft // 2 + 1)
    for a in range(0, len(v) - nfft + 1, hop):
        V = np.fft.rfft(v[a:a + nfft] * win); I = np.fft.rfft(i[a:a + nfft] * win)
        svi += V * np.conj(I); sii += np.abs(I) ** 2; svv += np.abs(V) ** 2
    f = np.fft.rfftfreq(nfft, 1 / FS)
    coh = np.abs(svi) ** 2 / np.maximum(svv * sii, 1e-30)
    band = (f >= f1) & (f <= f2) & (coh > 0.9)
    return f[band], svi[band] / sii[band]


def octave_smooth(f, z, frac=6):
    """1/frac-octave smoothing of complex Z on a log grid."""
    grid = np.geomspace(f[0] * 2 ** (0.5 / frac), f[-1] / 2 ** (0.5 / frac), 160)
    out = []
    for g in grid:
        sel = (f >= g * 2 ** (-0.5 / frac)) & (f <= g * 2 ** (0.5 / frac))
        out.append(np.median(z[sel].real) + 1j * np.median(z[sel].imag) if sel.any() else np.nan)
    return grid, np.array(out)


def judge(fz, z, znom):
    mag = np.abs(z); ph = np.degrees(np.angle(z))
    top = fz >= fz[-1] / 2 ** 0.75
    low = fz <= fz[0] * 4
    i_min = int(np.nanargmin(mag))
    i_pk = int(np.nanargmax(np.where(fz <= fz[-1] / 4, mag, -np.inf)))
    r_min = float(mag[i_min])
    return dict(
        r_min=r_min, f_rmin=float(fz[i_min]), r_peak=float(mag[i_pk]), f_peak=float(fz[i_pk]),
        peak_over_min=float(mag[i_pk] / r_min),
        hf_rise=float(np.nanmedian(mag[top]) / r_min),
        hf_phase=float(np.nanmedian(ph[top])),
        min_real=float(np.nanmin(z.real)),
        low_mag=float(np.nanmedian(mag[low])),
        r_ratio=float(r_min / znom) if znom else None,
    )


def physical(j):
    return (0.0 <= j['hf_phase'] <= 80.0) and j['min_real'] > 0.5 * j['r_min'] and j['hf_rise'] >= 1.0


def main():
    d = Path(sys.argv[1])
    conf = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else str(Path(__file__).resolve().parents[1] / 'conf/apple/j815.conf')
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else d / 'SPKSWEEP-REPORT.json'
    glob, spk = load_conf(conf)
    sig = json.loads((d / 'signal.json').read_text())
    rawp = d / 'sense.raw.gz'
    raw = np.frombuffer(gzip.open(rawp).read() if rawp.exists() else (d / 'sense.raw').read_bytes(), '<i2')
    raw = raw[: len(raw) // CH * CH].reshape(-1, CH).astype(float)
    print(f'capture: {len(raw)} frames = {len(raw) / FS:.2f} s, conf {conf}')

    # Locate the sweep in the capture: the energy envelope against the known on/off pattern of the six segments.
    hop = 480
    env = np.sqrt(np.mean(raw[: len(raw) // hop * hop].reshape(-1, hop, CH) ** 2, axis=(1, 2)))
    pat = np.zeros(int(sig['frames'] / hop) + 1)
    for s in sig['segments']:
        a = int(s['start_s'] * FS / hop); pat[a: a + int(s['dur_s'] * FS / hop)] = 1
    pat -= pat.mean()
    if len(env) < len(pat):
        sys.exit(f'capture ({len(env) * hop} frames) shorter than the signal ({sig["frames"]})')
    corr = np.correlate(env - env.mean(), pat, mode='valid')
    off = int(np.argmax(corr)) * hop
    print(f'signal starts at capture frame {off} ({off / FS:.2f} s)')

    results, ok_all = [], True
    for s in sig['segments']:
        name = s['name']; c = spk.get(name, {})
        a = off + int((s['start_s'] + 0.05) * FS); b = off + int((s['start_s'] + s['dur_s'] - 0.05) * FS)
        seg = raw[a:b]
        rms = np.sqrt(np.mean(seg ** 2, axis=0))
        pair_rms = np.array([math.hypot(rms[2 * p], rms[2 * p + 1]) for p in range(CH // 2)])
        act = int(np.argmax(pair_rms)); others = np.delete(pair_rms, act)
        xtalk = float(others.max() / pair_rms[act]) if pair_rms[act] else float('inf')
        vs = int(c.get('vs_chan', -1)); is_ = int(c.get('is_chan', -1))
        vsc = float(c.get('vs_scale', 14)); isc = float(c.get('is_scale', 3.75)); znom = float(c.get('z_nominal', 0))
        cfg_delay = int(c.get('current_delay_samples', 0))
        hyps = {}
        for label, (cv, ci) in (('even=V', (2 * act, 2 * act + 1)), ('odd=V', (2 * act + 1, 2 * act))):
            v = seg[:, cv] * vsc / 32768; i = seg[:, ci] * isc / 32768
            for lag in range(-3, 4):
                f, z = impedance(v, i, s['f1'] * 1.15, s['f2'] / 1.15, lag)
                fz, zs = octave_smooth(f, z)
                j = judge(fz, zs, znom); j['physical'] = physical(j)
                hyps[(label, lag, cv, ci)] = j
        # Crosstalk above the sense noise floor: the other pairs' RMS in this segment minus (in quadrature) their RMS
        # in the quiet lead-in before the first segment. Raw ratios of a quiet tweeter (-32 dB) segment are mostly noise.
        q = raw[off: off + int(max(0.2, sig['segments'][0]['start_s'] - 0.1) * FS)]
        qrms = np.sqrt(np.mean(q ** 2, axis=0))
        q_pair = np.array([math.hypot(qrms[2 * p], qrms[2 * p + 1]) for p in range(CH // 2)])
        excess = np.sqrt(np.maximum(pair_rms ** 2 - q_pair ** 2, 0))
        xtalk_raw = xtalk
        xtalk = float(np.delete(excess, act).max() / excess[act]) if excess[act] else float('inf')
        # Order: the amp drives a constant-amplitude sweep, so the VOLTAGE channel's envelope is flat while the
        # current follows 1/|Z| (dip at resonance, fall at high frequency). The flatter channel is the voltage.
        blk = 1024
        def flatness(x):
            e = np.sqrt(np.mean(x[: len(x) // blk * blk].reshape(-1, blk) ** 2, axis=1))
            e = e[3:-3]
            return float(np.std(e) / np.mean(e))
        cv_even, cv_odd = flatness(seg[:, 2 * act]), flatness(seg[:, 2 * act + 1])
        vch, ich = (2 * act, 2 * act + 1) if cv_even < cv_odd else (2 * act + 1, 2 * act)
        order_margin = max(cv_even, cv_odd) / max(min(cv_even, cv_odd), 1e-9)
        order_from = 'envelope'
        if order_margin < 1.5:
            # Flat |Z| (tweeters) or a shaped drive (the tweeter path high-passes) leaves both envelopes alike.
            # Then the order is the one that has a physical lag at all (swapping V and I negates the phase).
            phys = {(k[2], k[3]) for k, j in hyps.items() if j['physical']}
            if len(phys) == 1:
                vch, ich = phys.pop(); order_from = 'phase'
        cand = [k for k, j in hyps.items() if (k[2], k[3]) == (vch, ich) and j['physical']]
        best = max(cand, key=lambda k: hyps[k]['min_real'] / hyps[k]['r_min']) if cand else None
        verdict = []
        if act != vs // 2:
            verdict.append(f'answers on pair {act} (channels {2 * act}/{2 * act + 1}), profile expects pair {vs // 2}')
        if xtalk > 0.10:
            verdict.append(f'crosstalk {xtalk:.1%} on another pair')
        if order_margin < 1.5 and order_from != 'phase':
            verdict.append(f'voltage/current order unclear (envelope variation {cv_even:.3f} vs {cv_odd:.3f})')
        if (vch, ich) != (vs, is_):
            verdict.append(f'voltage is channel {vch}, current {ich}; profile has vs_chan {vs}, is_chan {is_}')
        if best is None:
            verdict.append('no current lag gives a physical impedance')
        else:
            lag = best[1]; jb = hyps[best]
            if -lag != cfg_delay:
                verdict.append(f'current lag {lag:+d} sample(s) (= current_delay_samples {-lag}); profile has {cfg_delay}')
            if len(cand) > 1:
                verdict.append(f'lag ambiguous: {sorted(k[1] for k in cand)} all physical')
            if not (0.8 <= jb['r_ratio'] <= 1.3):
                verdict.append(f'resistance {jb["r_min"]:.2f} ohm vs z_nominal {znom:.2f} ({jb["r_ratio"]:.2f}x)')
        print(f'\n   envelope variation: ch{2 * act} {cv_even:.3f}, ch{2 * act + 1} {cv_odd:.3f} -> voltage = ch{vch}'
              f' (order from {order_from}); crosstalk raw {xtalk_raw:.1%}, above noise {xtalk:.1%}')
        ok = not verdict
        ok_all &= ok
        print(f'\n== {name}: pair {act}, crosstalk {xtalk:.2%}, {"PASS" if ok else "CHECK: " + "; ".join(verdict)}')
        print('   order      lag  R_min(ohm)@Hz     peak(ohm)@Hz   pk/min  HFrise  HFphase  minRe  phys')
        for k, j in sorted(hyps.items(), key=lambda kv: (kv[0][2], kv[0][1])):
            mark = '<-' if k == best else '  '
            print(f'   {"V=%d I=%d" % (k[2], k[3]):9}  {k[1]:+d}  {j["r_min"]:6.2f} @{j["f_rmin"]:6.0f}  {j["r_peak"]:6.2f} @{j["f_peak"]:6.0f}'
                  f'  {j["peak_over_min"]:5.2f}  {j["hf_rise"]:5.2f}  {j["hf_phase"]:+6.1f}  {j["min_real"]:5.2f}  {"y" if j["physical"] else "-"} {mark}')
        results.append(dict(name=name, pair=act, crosstalk=xtalk, crosstalk_raw=xtalk_raw, order_from=order_from, passed=ok, problems=verdict, envelope_cv=[cv_even, cv_odd],
                            best=None if best is None else dict(v_chan=best[2], i_chan=best[3], lag=best[1], **hyps[best]),
                            profile=dict(vs_chan=vs, is_chan=is_, current_delay_samples=cfg_delay, z_nominal=znom)))

    # Replay the daemon's SENSE power lines (speakersafetyd -v) against the capture, if they are there.
    replay = {}
    log = (d / 'daemon.log').read_text(errors='replace') if (d / 'daemon.log').exists() else ''
    if 'panicked' in log:
        print('\nWARNING: speakersafetyd panicked during the sweep'); ok_all = False
    rows = {}
    for m in re.finditer(r'SENSE name=(.*?) frames=(\d+) rate=(\d+) delay=(\d+) priming=(\d+) power=(-?[\d.]+)', log):
        rows.setdefault(m[1], []).append((int(m[2]), int(m[4]), float(m[6])))
    for name, rr in rows.items():
        c = spk[name]; vs = int(c['vs_chan']); is_ = int(c['is_chan']); dl = int(c.get('current_delay_samples', 0))
        at, prev, errs = 0, None, []
        for frames, delay, power in rr:
            if at + frames > len(raw):
                break
            v = raw[at:at + frames, vs] * float(c['vs_scale']) / 32768
            i = raw[at:at + frames, is_] * float(c['is_scale']) / 32768
            if dl:
                al = np.empty_like(i); al[dl:] = i[:-dl]; al[:dl] = prev if prev is not None else float(c['is_scale']) * np.sign(v[0]); prev = i[-1]
            else:
                al = i
            errs.append(abs(float(np.mean(v * al)) - power)); at += frames
        replay[name] = dict(periods=len(errs), max_abs_error_W=max(errs) if errs else None)
    if replay:
        print('\ndaemon power replay (max |error| W per speaker):', {k: v['max_abs_error_W'] for k, v in replay.items()})
    report = dict(all_passed=bool(ok_all), signal_offset_frames=off, speakers=results, daemon_replay=replay,
                  note='Layout check only; raising the ceilings is a separate, stepwise decision.')
    out.write_text(json.dumps(report, indent=1) + '\n')
    print(f'\n{"ALL PASS" if ok_all else "NOT ALL PASS"} -> {out}')


if __name__ == '__main__':
    main()
