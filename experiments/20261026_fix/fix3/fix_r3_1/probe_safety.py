"""直す 3 回目・1 組目 r3-r3-safety-1・safety-2 の確かめ(閉じたループ。本物の drive._loop と偽物の ESC・車 = refute3_safety/cl3.py)。
今の assist.py と、直す前の写し(tests/fixtures/fix3_r3_1_before/assist.py・drive.py)を同じ場面で回して比べる。
装置・GPU・API・jetformer-web には触れない。一時の場所は fix3/tmp_fixr31_probe(終わったら消す)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        experiments/20261026_fix/fix3/fix_r3_1/probe_safety.py

safety-1(反証 x2c_phase・verify closed_phase と同じ場面): 前の壁 2.00〜2.07 m(位相 8 通り)に約 1.8 m/s・6.5G で当たり、当たりと
  同時にカメラが L 秒(0.6・1.0・2.0)来ない。keep = 1・見張り 1・モデルは中立 − 120 µs。中立止めなしと keep = 0 も。
safety-2(反証 check_giveup_coast・x10b と同じ場面): 前の壁 2.0 m のリカバリーの 'reverse' に入って o 秒後から L 秒カメラが来ない。
  後ろの物は「中立止めなしで下がる所」の +0.15・+0.3・+0.5 m。中立の惰性の減速 1.5・3.0・4.5 m/s²。
出力: out/probe_safety.json
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIX3 = HERE.parent
sys.path.insert(0, str(FIX3 / 'refute3_safety'))
import cl3                                    # noqa: E402
import closed_loop as CL                      # noqa: E402
from cl3 import A, C, N, PARAMS, T0           # noqa: E402

cl3.TMP = FIX3 / 'tmp_fixr31_probe'
CL.TMP = cl3.TMP
BMOD, BAMOD = cl3.load_fixture('fix3_r3_1_before')
ORIG = CL.Car.step
DEC = [1.5]


def car_step(self, t, thr):
    """closed_loop.Car.step の中立の惰性 1.5 m/s² だけを DEC[0] にした写し(x10b と同じ)"""
    if not (thr < N - 20.0 or thr > N + 40.0):
        v_before = self.v
        r = ORIG(self, t, thr)
        dv = (DEC[0] - 1.5) * CL.DT
        if self.contact is None:
            if v_before > 0:
                self.v = max(0.0, self.v - dv)
            elif v_before < 0:
                self.v = min(0.0, self.v + dv)
        return r
    return ORIG(self, t, thr)


CL.Car.step = car_step


def run(p, sc_, before=False):
    if before:
        return cl3.run(p, sc_, mod=BMOD, amod=BAMOD, wrap=False)
    return cl3.run(p, sc_)


def s1_one(keep, L, xf, before=False):
    sc_ = {'dur': 8.0, 'xf': xf, 'model': (C, N - 120.0), 'drops': [], 'hit_drop': (0.0, L) if L else None}
    p = dict(PARAMS, recovery=1, start_kick=1, rec_back_guard=1, rec_hold_keep=keep)
    sim, rec, pca, ph = run(p, sc_, before)
    t_hit = sim.car.hits[0][0]
    back = t_hit + (L or 0.0)
    rev = [(t - T0) / 1e9 for t, r in rec.rows if not r['hold'] and int(r['assist']) == A.ASSIST_RECOVERY and r['out'][1] > N + 40]
    gu = [(t - T0) / 1e9 for t, r in rec.rows if not r['hold'] and int(r['assist']) == A.ASSIST_GIVEUP]
    xs = [r[1] for r in sim.log if r[0] >= back]
    push = sum(1 for r in sim.log if r[0] >= back and r[4] == 'front' and r[3] < N - 20) * CL.DT
    return {'result': 'giveup' if gu else ('reverse %.2f s after camera' % (rev[0] - back) if rev else 'none'),
            'giveup_after_camera_s': round(gu[0] - back, 3) if gu else None,
            'backed_m': round(xf - min(xs), 3), 'push_wall_s': round(push, 2),
            'ev': [(e[1]['what'], e[1].get('phase'), e[1].get('hold_s')) for e in rec.events if e[0] == 'recovery_guard'],
            'rows': [(round((t - T0) / 1e9, 3), round(r['out'][1], 1), int(r['assist']), bool(r['hold'])) for t, r in rec.rows],
            'pca': pca.w}


def s1():
    res = {}
    for keep in (1, 0):
        for L in (None, 0.6, 1.0, 2.0):
            for j in range(8):
                xf = round(2.00 + 0.01 * j, 2)
                now, bef = s1_one(keep, L, xf), s1_one(keep, L, xf, before=True)
                same = now['rows'] == bef['rows'] and now['pca'] == bef['pca'] and now['ev'] == bef['ev']
                res['keep=%d L=%s xf=%.2f' % (keep, L, xf)] = {
                    'now': {k: v for k, v in now.items() if k not in ('rows', 'pca')},
                    'before': {k: v for k, v in bef.items() if k not in ('rows', 'pca')}, 'same_as_before': same}
    return res


def s2_one(keep, drop=None, back=None, before=False):
    sc_ = {'dur': 8.0, 'xf': 2.0, 'model': (C, N - 120.0), 'drops': [drop] if drop else []}
    if back is not None:
        sc_['xr'] = 2.0 - back
    p = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=keep)
    sim, rec, pca, ph = run(p, sc_, before)
    out = {'rear_hits': [h for h in sim.car.hits if h[1] == 'rear'], 'min_x': round(min(r[1] for r in sim.log if r[0] > 1.6), 3),
           'ev': [(e[1]['what'], e[1].get('phase'), e[1].get('hold_s')) for e in rec.events if e[0] == 'recovery_guard'],
           'phases': cl3.phases(ph)[:12] if not before else None}
    if drop:
        end = drop[1]
        at = next(r for r in sim.log if r[0] >= end)
        xs = [r for r in sim.log if end <= r[0] <= end + 2.0]
        out.update(v_at_camera_back=round(at[2], 2), backed_after_camera_m=round(at[1] - min(r[1] for r in xs), 3),
                   thr_after=[(round(t - end, 3), round(th)) for t, s, th in pca.w if end <= t <= end + 0.6][:8],
                   halt_rows=sum(1 for t, r in rec.rows if not r['hold'] and (t - T0) / 1e9 >= end
                                 and int(r['assist']) == A.ASSIST_RECOVERY and r['out'][1] == N - 130.0
                                 and (t - T0) / 1e9 < end + 0.4))
    out['_rows'] = [(round((t - T0) / 1e9, 3), round(r['out'][1], 1), int(r['assist']), bool(r['hold'])) for t, r in rec.rows]
    out['_pca'] = pca.w
    return out


def s2():
    res = {}
    for dec in (1.5, 3.0, 4.5):
        DEC[0] = dec
        nh = s2_one(1, None, None)
        reach = round(2.0 - nh['min_x'], 3)
        tr = next(t for t, p0, p1 in nh['phases'] if p1 == 'reverse')
        res['dec=%.1f nohold' % dec] = {'reach': reach, 't_reverse': tr}
        for o, L in ((0.40, 0.55), (0.45, 0.60), (0.50, 0.55), (0.50, 0.70), (0.30, 1.0)):
            drop = (tr + o, tr + o + L)
            for add in (0.15, 0.3, 0.5, None):
                back = None if add is None else reach + add
                nohold = s2_one(1, None, back) if back is not None else None
                for keep in (0, 1):
                    now = s2_one(keep, drop, back)
                    bef = s2_one(keep, drop, back, before=True)
                    k = 'dec=%.1f back=%s rev+%.2f L=%.2f keep=%d' % (dec, 'none' if add is None else '+%.2f' % add, o, L, keep)
                    res[k] = {'now': {kk: v for kk, v in now.items() if not kk.startswith('_')},
                              'before': {kk: v for kk, v in bef.items() if not kk.startswith('_') and kk != 'phases'},
                              'nohold_rear_hits': None if nohold is None else nohold['rear_hits'],
                              'same_as_before': now['_rows'] == bef['_rows'] and now['_pca'] == bef['_pca']}
    DEC[0] = 1.5
    return res


def main():
    try:
        out = {'safety1': s1(), 'safety2': s2()}
    finally:
        shutil.rmtree(cl3.TMP, ignore_errors=True)
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'probe_safety.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for k, v in out['safety1'].items():
        print('S1', k, 'now:', v['now']['result'], v['now']['backed_m'], v['now']['push_wall_s'], '| before:', v['before']['result'],
              v['before']['backed_m'], v['before']['push_wall_s'], '| same' if v['same_as_before'] else '| CHANGED')
    for k, v in out['safety2'].items():
        if 'now' not in v:
            print('S2', k, v)
            continue
        n, b = v['now'], v['before']
        print('S2', k, 'now: rear', [h[2:] for h in n['rear_hits']], 'v@back', n.get('v_at_camera_back'), 'back',
              n.get('backed_after_camera_m'), 'halt', n.get('halt_rows'), '| before: rear', [h[2:] for h in b['rear_hits']],
              b.get('backed_after_camera_m'), '| nohold', None if v['nohold_rear_hits'] is None else [h[2:] for h in v['nohold_rear_hits']],
              '| same' if v['same_as_before'] else '| CHANGED')


if __name__ == '__main__':
    main()
