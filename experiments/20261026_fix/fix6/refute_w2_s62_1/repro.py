"""反証の確かめ(w2-safety-r62-safety-1): EscTrack の中立 0.08 秒で「下がれる」とする見積もりで 'brake' → 'reverse' に飛び、
CHECK_S のやり直しが止まるか。本物の drive._loop を tests/test_fix3_r2_1 の閉じたループ(元の偽物の ESC = R2.Car)で回す。
読むだけ・装置に触れない。一時の場所は fix6/tmp_refute_w2_s62_1(終わったら消す)。
    python -I repro.py [now|before] [drop_a drop_b veh_a veh_b] [neu]
"""
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))
import pytest  # noqa: E402
from jetformer import assist as A, drive  # noqa: E402
import test_fix3_r2_1 as R2  # noqa: E402
from test_launch_drive import C, N, PARAMS  # noqa: E402

TMP = ROOT / 'experiments' / '20261026_fix' / 'fix6' / 'tmp_refute_w2_s62_1'
_M = {}


def load(name):
    if name == 'now':
        return drive, A
    if name in _M:
        return _M[name]
    d = ROOT / 'tests' / 'fixtures' / 'fix6_safety_before'
    aspec = importlib.util.spec_from_loader('jetformer._w2_assist_bef', loader=None)
    amod = importlib.util.module_from_spec(aspec)
    exec(compile((d / 'assist.py').read_text(encoding='utf-8'), str(d / 'assist.py'), 'exec'), amod.__dict__)
    spec = importlib.util.spec_from_loader('jetformer._w2_drive_bef', loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile((d / 'drive.py').read_text(encoding='utf-8'), str(d / 'drive.py'), 'exec'), mod.__dict__)
    mod.assist_mod = amod
    _M[name] = (mod, amod)
    return mod, amod


def run(name, scene, neu=None, trace=False):
    mod, amod = load(name)
    PH, CAR = [], []
    orig = amod.Recovery.step

    def step(self, model_us, erpm, now, impact_g=0.0, *a, **kw):
        p0, b0 = self.phase, self.backed_s
        r = orig(self, model_us, erpm, now, impact_g, *a, **kw)
        PH.append((round(now - 100.0, 3), p0, self.phase, kw.get('hold_since') is not None,
                   round(float(kw.get('hold_back_s', 0.0)), 3), round(float(kw.get('hold_rev_s', 0.0)), 3),
                   round(b0, 3), round(self.backed_s, 3), int(erpm), self.retries,
                   None if r[0] is None else round(r[0][1])))
        return r

    class MyCar(R2.Car):
        def step(self, t, thr):
            if neu is not None:      # 中立で ready になる秒だけ変える(あとは R2.Car と同じ)
                was = self.braked
                res = super().step(t, thr)
                fwd, rev = thr < N - 20.0, thr > N + 40.0
                if not fwd and not rev and was and not self.rev_on:
                    self.ready, self.braked = (True, False) if self.neu_t >= neu - 1e-9 else (False, True)
            else:
                res = super().step(t, thr)
            CAR.append((round(t, 2), round(thr), self.ready, self.braked, self.rev_on, round(self.neu_t, 3), round(self.v, 2), round(self.x, 3)))
            return res
    TMP.mkdir(parents=True, exist_ok=True)
    amod.Recovery.step = step
    try:
        with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(TMP)) as td:
            mp.setattr(R2, 'Car', MyCar)
            p = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=1)
            sim, rec, pca = R2.run_cl(mod, mp, Path(td), p, scene)
    finally:
        amod.Recovery.step = orig
    return sim, rec, pca, PH, CAR


def metrics(sim, pca, xf=2.0, t_from=2.0):
    t_esc = next((t for t, s, th in pca.w if t > t_from and th < N - 125.0), None)
    end = (t_esc + 0.5) if t_esc else 7.0
    xs = [(t, x) for t, x, v, th in sim.log if t_from <= t <= end]
    back = round(xf - min(x for _, x in xs), 3)
    push = 0.0
    if t_esc:
        push = round(sum(R2.DT for t, x, v, th in sim.log if t_esc < t <= t_esc + 0.5 and x >= xf - 1e-9 and th < N - 125), 3)
    return {'t_esc': t_esc, 'back_m': back, 'esc_push_wall_s': push}


def main(name='now', da=2.11, db=2.26, va=2.26, vb=2.45, neu=None):
    sc = {'dur': 7.0, 'xf': 2.0, 'xr': None, 'model': (C, N - 120.0), 'drops': [(da, db)], 'veh_skip': [(va, vb)]}
    sim, rec, pca, PH, CAR = run(name, sc, neu)
    print('== %s drop=(%s,%s) veh_skip=(%s,%s) neu=%s' % (name, da, db, va, vb, neu))
    print('guards', [(e[1]['what'], e[1]['phase'], e[1]['backed_s']) for e in rec.events if e[0] == 'recovery_guard'])
    for x in PH:
        if 2.0 <= x[0] <= 4.3:
            print('rec t=%s %s->%s held=%s hb=%s hrev=%s backed %s->%s erpm=%s retr=%s out=%s' % x)
    for x in CAR:
        if 2.15 <= x[0] <= 2.85:
            print('esc', x)
    print('writes', [(round(t, 3), round(th)) for t, s, th in pca.w if 2.05 <= t <= 4.3])
    print('hits', sim.car.hits[:5])
    print('metrics', metrics(sim, pca))


if __name__ == '__main__':
    a = sys.argv[1:]
    try:
        main(a[0] if a else 'now', *[float(x) for x in a[1:5]], *([float(a[5])] if len(a) > 5 else []))
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
