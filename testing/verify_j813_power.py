# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Acelogic
from pathlib import Path
import hashlib, json, re, sys
import numpy as np
r=Path.cwd()
c=Path(sys.argv[1])
raw=np.fromfile(c/'sense.raw','<i2').reshape(-1,8)
log=(c/'daemon.log').read_text()
assert 'panicked' not in log
names=['Left Woofer','Right Woofer','Left Tweeter','Right Tweeter']
rows={n:[] for n in names}
reset={n:True for n in names}
for line in log.splitlines():
    for n in names:
        if n+': Reset sense history' in line: reset[n]=True
    m=re.search(r'SENSE name=(.*?) frames=(\d+) rate=(\d+) delay=(\d+) priming=(\d+) power=(-?[\d.]+)',line)
    if not m: continue
    n,frames,rate,delay,prime,power=m.groups()
    row=dict(frames=int(frames),rate=int(rate),delay=int(delay),prime=int(prime),power=float(power),reset=reset[n])
    rows[n].append(row); reset[n]=False
results=[]
for ch,n in enumerate(names):
    at=0; previous=None; measured=[]; expected=[]; wrong=[]; resets=0
    for row in rows[n]:
        assert row['rate']==48000 and row['delay']==(1 if ch==0 else 0)
        end=at+row['frames']; assert end<=len(raw)
        if row['reset']: previous=None; resets+=1
        v=raw[at:end,ch*2].astype(float)*14/32768
        i=raw[at:end,ch*2+1].astype(float)*3.75/32768
        if ch==0:
            aligned=np.empty_like(i); aligned[1:]=i[:-1]
            aligned[0]=previous if previous is not None else 3.75*np.sign(v[0])
            assert row['prime']==int(previous is None)
            previous=i[-1]
        else:
            aligned=i; assert row['prime']==0
        expected.append(float(np.mean(v*aligned))); measured.append(row['power']); wrong.append(float(np.mean(v*i)))
        at=end
    expected=np.array(expected);measured=np.array(measured);wrong=np.array(wrong)
    assert len(expected)>20 and at==len(raw), (n,at,len(raw))
    error=abs(expected-measured)
    tolerance=1e-9+2e-5*abs(expected)
    assert np.all(error<=tolerance),(n,error.max())
    results.append({'name':n,'periods':len(expected),'frames':at,'history_resets':resets,'max_power_error_W':float(error.max()),'max_power_W':float(expected.max()),'difference_from_uncorrected_W':float(abs(expected-wrong).max()),'passed':True})
report={'capture_sha256':hashlib.sha256((c/'sense.raw').read_bytes()).hexdigest(),'log_sha256':hashlib.sha256((c/'daemon.log').read_bytes()).hexdigest(),'method':'Independent float64 replay of every actual-binary per-period power observation, including buffer continuity and conservative priming','all_passed':True,'drivers':results,'normal_volume_admitted':False}
(r/(sys.argv[2] if len(sys.argv)>2 else 'SENSE-DELAY-REPLAY.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
