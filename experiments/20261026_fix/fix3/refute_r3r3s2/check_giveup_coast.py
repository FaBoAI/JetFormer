"""反証 r3-r3-safety-2 の打ち消し(直さない)。refute3_safety/cl3.py の閉じたループ(本物の drive._loop・assist)で、
'reverse' の途中の長い中立止め → 'hold_long' の中止(中立)のあと、後ろへ惰性で下がって後ろの物に当たるかを自分で確かめる。
装置・GPU・API に触れない。cl3 の一時ディレクトリはこの係の tmp に向け、終わったら消す"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'refute3_safety'))
import cl3                                    # noqa: E402
import closed_loop as CL                      # noqa: E402
from cl3 import A, C, N, PARAMS               # noqa: E402

cl3.TMP = HERE.parent / 'tmp_refute_r3r3s2'
CL.TMP = cl3.TMP
XF = 2.0
ORIG = CL.Car.step
DEC = [1.5]


def car_step(self, t, thr):
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


def one(keep, drop=None, back=None, model=(C, N - 120.0)):
    sc_ = {'dur': 8.0, 'xf': XF, 'model': model, 'drops': [drop] if drop else []}
    if back is not None:
        sc_['xr'] = XF - back
    p = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=keep)
    sim, rec, pca, ph = cl3.run(p, sc_)
    out = {'rear_hits': [h for h in sim.car.hits if h[1] == 'rear'],
           'front_hits': [h for h in sim.car.hits if h[1] == 'front'][:3],
           'min_x': round(min(r[1] for r in sim.log if r[0] > 1.6), 3),
           'ev': [(e[1]['what'], e[1].get('phase'), e[1].get('backed_s'), e[1].get('hold_s')) for e in rec.events
                  if e[0] == 'recovery_guard'],
           'phases': cl3.phases(ph)[:10],
           'giveup_rows': sum(1 for t, r in rec.rows if not r['hold'] and int(r['assist']) == A.ASSIST_GIVEUP)}
    if drop:
        end = drop[1]
        at = next(r for r in sim.log if r[0] >= end)
        xs = [r for r in sim.log if end <= r[0] <= end + 2.0]
        stop = next((r for r in xs if abs(r[2]) < 0.01), None)
        out.update(v_at_cam_back=round(at[2], 2), backed_after_m=round(at[1] - min(r[1] for r in xs), 3),
                   stop_t_after=None if stop is None else round(stop[0] - end, 2),
                   thr_after=[(round(t - end, 3), round(th)) for t, s, th in pca.w if end <= t <= end + 1.2][:6])
    return out


def main():
    res = {}
    for dec in (1.5, 3.0):
        DEC[0] = dec
        nh = one(1, None, None)
        reach = round(XF - nh['min_x'], 3)
        tr = next(t for t, p0, p1 in nh['phases'] if p1 == 'reverse')
        res['dec=%.1f nohold' % dec] = dict(nh, reach=reach, t_reverse=tr)
        for o, L in ((0.40, 0.55), (0.45, 0.60)):
            drop = (tr + o, tr + o + L)
            for back in (reach + 0.15, reach + 0.5):
                # 中立止めなし(同じ後ろの物)・keep=0・keep=1
                res['dec=%.1f back=+%.2f nohold' % (dec, back - reach)] = one(1, None, back)
                for keep in (0, 1):
                    res['dec=%.1f back=+%.2f rev+%.2f L=%.2f keep=%d' % (dec, back - reach, o, L, keep)] = one(keep, drop, back)
            # 後ろの物なし: 明けてから下がる距離
            for keep in (0, 1):
                res['dec=%.1f free rev+%.2f L=%.2f keep=%d' % (dec, o, L, keep)] = one(keep, drop, None)
    # モデルが前進でない日(中立・ブレーキ側)の keep=0 と keep=1(dec 1.5)
    DEC[0] = 1.5
    nh = res['dec=1.5 nohold']
    tr, reach = nh['t_reverse'], nh['reach']
    for name, model in (('model_neutral', (C, N)), ('model_brake60', (C, N + 60.0))):
        drop = (tr + 0.40, tr + 0.95)
        for keep in (0, 1):
            res['%s free rev+0.40 L=0.55 keep=%d' % (name, keep)] = one(keep, drop, None, model)
    for k, v in res.items():
        print(k, {kk: v.get(kk) for kk in ('reach', 't_reverse', 'rear_hits', 'min_x', 'ev', 'v_at_cam_back',
                                           'backed_after_m', 'stop_t_after', 'thr_after', 'giveup_rows') if kk in v})
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'check_giveup_coast.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    try:
        main()
    finally:
        shutil.rmtree(cl3.TMP, ignore_errors=True)
