import numpy as np
from la import LA

NEUT = 1498.0
CENTER = 1443.0

def clean(x, lo=900, hi=2100):
    x = x.astype(float).copy()
    bad = (x < lo) | (x > hi)
    if bad.any():
        idx = np.where(~bad, np.arange(len(x)), 0)
        np.maximum.accumulate(idx, out=idx)
        x = x[idx]
    return x, bad

def build(log, tof=True):
    d = LA(log, tof=tof)
    thr_h, b1 = clean(d.pwm[:, 1]); st_h, b2 = clean(d.pwm[:, 0])
    d.thr_h, d.st_h = thr_h, st_h
    d.nbad = int(b1.sum() + b2.sum())
    if d.at is not None:
        ao = LA.at_t(d.at, d.out, d.vt)
        before = d.vt < d.at[0]
        ao[before] = [CENTER, NEUT]
        d.thr = np.where(d.vsel, ao[:, 1], thr_h)
        d.st = np.where(d.vsel, ao[:, 0], st_h)
    else:
        d.thr, d.st = thr_h.copy(), st_h.copy()
    d.human = ~d.vsel
    return d

def states(thr, db=20.0, neut=NEUT):
    s = np.zeros(len(thr), int)          # 0 = N, -1 = F (forward), +1 = R (brake/reverse)
    s[thr < neut - db] = -1
    s[thr > neut + db] = 1
    return s

def runs(x):
    """return list of (start, end_exclusive, value)"""
    x = np.asarray(x)
    ch = np.nonzero(np.diff(x))[0] + 1
    st = np.r_[0, ch]; en = np.r_[ch, len(x)]
    return [(int(a), int(b), x[a]) for a, b in zip(st, en)]
