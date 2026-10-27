"""反証 4 回目・安全 s3 の確かめ: rec_hold_keep = 1・rec_back_guard = 1 で、リカバリーの 'brake'・'wait'・'gap' の途中に車両の値
(VESC の eRPM)が 0.2〜0.5 秒来ない(中立止め「車両(PWM)の値が古い」)と、drive の veh_unseen_back が「0.06 秒より空いた間は
後ろへ動いていたとして数える」ため hold_back_s が rev_s 以上になり、assist の _after_hold が 'coast' に進め、
1 cm も下がっていない車が前の壁に向かって escape(前進 130 µs・ハンドル 0.85)を 0.5 秒出す。
同じ場面で、車両の値の止まりが無い・keep = 0・直す 3 回目の前(fix3_r3_1_before)と比べる。

本物の drive._loop・偽物の ESC と車(fix3/refute2_safety/closed_loop.py)。装置・GPU・API には触れない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        experiments/20261026_fix/fix4/review_safety/probe_veh_gap_brake.py
出力: out/probe_veh_gap_brake.json
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
        r = orig(self, model_us, erpm, now, impact_g, *a, **kw)
        PH.append((round(now - 100.0, 3), p0, self.phase, kw.get('hold_since') is not None,
                   round(float(kw.get('hold_back_s', 0.0)), 3), round(self.backed_s, 3)))
        return r
    return step


def first_brake_t(p):
    del PH[:]
    sc = {'dur': 6.0, 'xf': 2.0, 'model': (C, N - 120.0), 'drops': [], 'veh_skip': []}
    R.run_closed(p, sc, wrap=wrap)
    return next(t for t, p0, p1, *_ in PH if p1 == 'brake')


def one(p, skip, mod=None):
    del PH[:]
    sc = {'dur': 6.0, 'xf': 2.0, 'model': (C, N - 120.0), 'drops': [], 'veh_skip': [skip] if skip else []}
    if mod is None:
        sim, rec, pca, d = R.run_closed(p, sc, wrap=wrap)
    else:
        sim, rec, pca, d = R.run_closed(p, sc, mod=mod[0], amod=mod[1], wrap=wrap)
    t_hit = sim.car.hits[0][0]
    rows = [((t - T0) / 1e9, r) for t, r in rec.rows]
    # 最初のリカバリーの区間(最初の 'brake' から最初の escape の終わり + 0.1 秒)
    tb = next((t for t, p0, p1, *_ in PH if p1 == 'brake'), None)
    te = next((t for t, p0, p1, *_ in PH if p1 == 'escape' and t > (tb or 0)), None)
    win = [x for t, x, *_ in sim.log if tb is not None and tb <= t <= (te if te else tb + 2.0)]
    backed_before_escape = round(2.0 - min(win), 3) if win else None
    esc_rows_at_wall = 0
    for s, r in rows:
        if te is not None and te <= s < te + 0.6 and not r['hold'] and int(r['assist']) == A.ASSIST_RECOVERY \
                and float(r['out'][1]) == N - 130.0:
            lg = next(v for v in sim.log if v[0] >= s)
            if lg[4] == 'front':
                esc_rows_at_wall += 1
    rev_rows = sum(1 for s, r in rows if tb is not None and s >= tb and (te is None or s < te) and not r['hold']
                   and int(r['assist']) == A.ASSIST_RECOVERY and float(r['out'][1]) > N + 40)
    return {'t_brake': tb, 't_escape': te, 'backed_m_before_escape': backed_before_escape,
            'reverse_side_rows_before_escape': rev_rows, 'escape_rows_pushing_front_wall': esc_rows_at_wall,
            'hold_rows': sum(1 for s, r in rows if r['hold']),
            'after_hold_calls': [x for x in PH if x[3]][:3],
            'phases': [(t, p0, p1) for t, p0, p1, *_ in PH if p0 != p1][:8],
            'events': [(e[1].get('what'), e[1].get('phase'), e[1].get('backed_s')) for e in rec.events if e[0] == 'recovery_guard'],
            'front_hits': [h for h in sim.car.hits if h[1] == 'front'][:3]}


def main():
    BEF = R.load_snapshot('fix3_r3_1_before')
    out = {}
    try:
        for rev_s in (0.4, 0.3, 0.2):
            p1 = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=1, tof_guard=1, rev_s=rev_s)
            tb = first_brake_t(p1)
            out['rev_s=%.1f no_gap' % rev_s] = one(p1, None)
            for d0 in (0.05, 0.2, 0.3):          # 'brake' の始めからの遅れ(brake 0.25 秒・gap 0.15 秒)
                for L in (0.25, 0.35, 0.45):
                    skip = (tb + d0, tb + d0 + L)
                    k = 'rev_s=%.1f veh_gap=brake+%.2f..+%.2f' % (rev_s, d0, L)
                    out[k + ' now_keep1'] = one(p1, skip)
                    out[k + ' now_keep0'] = one(dict(p1, rec_hold_keep=0), skip)
                    out[k + ' before_keep1'] = one(p1, skip, BEF)
    finally:
        R.cleanup()
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'probe_veh_gap_brake.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    for k, v in out.items():
        print(k, {kk: v[kk] for kk in ('t_brake', 't_escape', 'backed_m_before_escape', 'reverse_side_rows_before_escape',
                                      'escape_rows_pushing_front_wall', 'hold_rows', 'events')})


if __name__ == '__main__':
    main()
