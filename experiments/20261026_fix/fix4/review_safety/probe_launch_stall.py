"""反証 4 回目・安全 s2 の確かめ: 発進の前進(launch_boost = 1)・発進の補助(start_kick)の区間では Recovery.step が呼ばれない
(assist.Assist.step が先に返す)ので、Recovery.t_last は区間の始め(人側で reset)から None のまま。そのため、発進の途中で
カメラが止まり、止まっている間に前の壁に当たると、明けた最初のフレームの gap が None で _after_hold が呼ばれず、
r3-r3-safety-1 の「長い中立止めのあとの中止」が効かない(モデルの前進が壁を押し → stuck → 後退で 1 m 以上下がる)。
発進の補助も前進もない区間(Recovery が呼ばれている)なら同じ当たりで中止になる。

本物の drive._loop・偽物の ESC と車(fix3/refute2_safety/closed_loop.py)。装置・GPU・API には触れない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        experiments/20261026_fix/fix4/review_safety/probe_launch_stall.py
出力: out/probe_launch_stall.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rs_common as R          # noqa: E402
from rs_common import A, C, N, PARAMS, T0   # noqa: E402

PH = []


def wrap(orig):
    def step(self, model_us, erpm, now, impact_g=0.0, *a, **kw):
        p0 = self.phase
        t_last = self.t_last
        r = orig(self, model_us, erpm, now, impact_g, *a, **kw)
        PH.append((round(now - 100.0, 3), p0, self.phase, kw.get('hold_since') is not None, t_last is None,
                   round(float(kw.get('hold_g', 0.0)), 2)))
        return r
    return step


def model(t, car):
    """人側(1.0 秒まで)は止まっている(偽物の車は人側でも PCA の値で動くので、人側の間は中立)。AI 側は前進 120 µs"""
    return (C, N) if t < 1.0 else (C, N - 120.0)


def one(p, sc):
    del PH[:]
    sim, rec, pca, d = R.run_closed(p, sc, wrap=wrap)
    rows = [((t - T0) / 1e9, r) for t, r in rec.rows]
    end = sc['drops'][0][1] if sc['drops'] else None
    rev = [s for s, r in rows if not r['hold'] and int(r['assist']) == A.ASSIST_RECOVERY and float(r['out'][1]) > N + 40]
    gu = [s for s, r in rows if not r['hold'] and int(r['assist']) == A.ASSIST_GIVEUP]
    launch = [s for s, r in rows if not r['hold'] and int(r['assist']) == A.ASSIST_LAUNCH]
    kick = [s for s, r in rows if not r['hold'] and int(r['assist']) == A.ASSIST_START]
    xs = [x for t, x, *_ in sim.log]
    t_hit = sim.car.hits[0][0] if sim.car.hits else None
    after_hit = [x for t, x, *_ in sim.log if t_hit is not None and t >= t_hit]
    push = sum(R.CL.DT for t, x, v, thr, con, *_ in sim.log if end is not None and t >= end and con == 'front' and thr < N - 20)
    first_after = next(((s, round(float(r['out'][1])), int(r['assist'])) for s, r in rows if end is not None and s >= end
                        and not r['hold']), None)
    after_hold_call = [x for x in PH if x[3]]
    return {'hits': sim.car.hits, 'launch_rows': [round(launch[0], 3), round(launch[-1], 3), len(launch)] if launch else None,
            'kick_rows': [round(kick[0], 3), round(kick[-1], 3), len(kick)] if kick else None,
            'giveup_first': round(gu[0], 3) if gu else None, 'reverse_first': round(rev[0], 3) if rev else None,
            'backed_m_after_hit': round(max(after_hit) - min(after_hit), 3) if after_hit else None,
            'push_wall_after_camera_s': round(push, 2), 'first_row_after_camera': first_after,
            'first_recovery_call_after_hold': after_hold_call[:1],
            'events': [(e[1].get('what'), e[1].get('phase'), e[1].get('hold_s')) for e in rec.events if e[0] == 'recovery_guard']}


def main():
    out = {}
    base = dict(PARAMS, recovery=1, rec_back_guard=1, rec_hold_keep=1, tof_guard=1, launch_no_kick=0)
    try:
        for name, extra in (('launch_boost=1', {'launch_boost': 1, 'start_kick': 0}),
                            ('start_kick=1', {'launch_boost': 0, 'start_kick': 1}),
                            ('none', {'launch_boost': 0, 'start_kick': 0})):
            p = dict(base, **extra)
            for xf in (1.3, 1.6):
                for a in (1.1, 1.2, 1.3):
                    for L in (1.0, 2.0):
                        sc = {'dur': a + L + 5.0, 'xf': xf, 'model': model, 'drops': [(a, a + L)], 'veh_skip': []}
                        r = one(p, sc)
                        r['hit_during_stall'] = bool(r['hits']) and a <= r['hits'][0][0] < a + L
                        out['%s wall=%.1f stall=%.2f+%.1f' % (name, xf, a, L)] = r
                sc = {'dur': 7.0, 'xf': xf, 'model': model, 'drops': [], 'veh_skip': []}
                out['%s wall=%.1f no_stall' % (name, xf)] = one(p, sc)
    finally:
        R.cleanup()
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'probe_launch_stall.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    for k, v in out.items():
        print(k, v)


if __name__ == '__main__':
    main()
