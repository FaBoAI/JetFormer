"""反証 1・2 回目の safety の指摘(r1-rs1-1・r1-rs1-2・r2-r2-rs-1・r2-r2-rs-2)が今のコードで直っているかを、閉じたループで流し直す"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cl3                                    # noqa: E402
from cl3 import A, C, N, PARAMS, drive        # noqa: E402

XF = 2.0
P = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=1)


def run(sc_, **kw):
    p = dict(P, **kw)
    return cl3.run(p, sc_)


def rear_push(sim):
    s = 0.0
    for (t0, x, v, thr, con, ro, sl), (t1, *_r) in zip(sim.log, sim.log[1:]):
        if con == 'rear' and thr > N + 40 and sl:
            s += t1 - t0
    return round(s, 3)


def main():
    res = {}
    # 基準: 中立止めなし。'reverse' の始まり
    sim, rec, pca, ph = run({'dur': 8.0, 'xf': XF, 'model': (C, N - 120.0)})
    tr = next(t for t, p0, p1 in cl3.phases(ph) if p1 == 'reverse')
    tb = next(t for t, p0, p1 in cl3.phases(ph) if p1 == 'brake')
    res['base'] = {'t_brake': tb, 't_reverse': tr, 'back_m': round(XF - min(r[1] for r in sim.log), 3)}
    # r1-rs1-1: 後ろの物 0.6 m(後退の途中で当たる)。当たりを中立止めが覆う
    out = []
    for back in (0.5, 0.6, 0.8):
        for o in (0.05, 0.1, 0.15, 0.2):
            for L in (0.2, 0.3, 0.45, 0.8, 1.5):
                sim, rec, pca, ph = run({'dur': 8.0, 'xf': XF, 'xr': XF - back, 'model': (C, N - 120.0),
                                         'drops': [(tr + o, tr + o + L)]})
                rp = rear_push(sim)
                rh = [h for h in sim.car.hits if h[1] == 'rear']
                out.append({'back': back, 'o': o, 'L': L, 'rear_push_s': rp, 'rear_hits': rh,
                            'ev': [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']})
    res['r1-rs1-1'] = {'max_rear_push_s': max(x['rear_push_s'] for x in out),
                       'worst': sorted(out, key=lambda x: -x['rear_push_s'])[:4]}
    # r2-r2-rs-1: 'reverse' の始めの中立止め → やり直し 1846 を書かない
    out = []
    for o in (0.03, 0.06, 0.1, 0.14):
        for L in (0.2, 0.3, 0.45):
            sim, rec, pca, ph = run({'dur': 8.0, 'xf': XF, 'model': (C, N - 120.0), 'drops': [(tr + o, tr + o + L)]})
            wmax = max(th for t, s, th in pca.w if tr - 0.1 <= t <= tr + 2.0)
            out.append({'o': o, 'L': L, 'max_written': wmax, 'back_m': round(XF - min(r[1] for r in sim.log), 3)})
    res['r2-r2-rs-1'] = {'max_written': max(x['max_written'] for x in out), 'max_back': max(x['back_m'] for x in out)}
    # r2-r2-rs-2: 長い中立止め(段の途中・止まっている時間の途中)→ 明けて後退側を出さない
    out = []
    t_stop = next(r[0] for r in sim.log if r[0] > 1.2 and abs(r[2]) < 0.01 and r[1] > 1.9)
    for name, t0 in (('stuck', t_stop + 0.05), ('brake', tb + 0.05), ('reverse', tr + 0.05)):
        for L in (0.6, 1.0, 2.0, 4.0):
            sim, rec, pca, ph = run({'dur': t0 + L + 3.0, 'xf': XF, 'model': (C, N - 120.0), 'drops': [(t0, t0 + L)]})
            after = [th for t, s, th in pca.w if t >= t0 + L]
            out.append({'where': name, 'L': L, 'rev_side_after': sum(1 for th in after if th > N + 40),
                        'ev': [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']})
    res['r2-r2-rs-2'] = out
    # r1-rs1-2: launch_no_kick 1 → 0 を区間の途中で(前が近い)/ start_kick 0 → 1 は断る
    def hook_off(t, d):
        if t >= 2.0 and d.params.get('launch_no_kick') == 1:
            d.params = dict(d.params, launch_no_kick=0)
    sc_ = {'dur': 6.0, 'xf': 1.6, 'model': (C, N), 'ai_from': 1.0}
    sim, rec, pca, ph = cl3.run(dict(P, start_kick=1, launch_boost=1, launch_no_kick=1, tof_guard=1), sc_, hook=hook_off)
    kick_rows = sum(1 for _, r in rec.rows if not r['hold'] and int(r['assist']) == A.ASSIST_START)
    d = drive.Driver.__new__(drive.Driver)
    d.params = dict(drive.DEFAULT_PARAMS, start_kick=0)
    d.thread = type('T', (), {'is_alive': lambda self: True})()
    d.lock = __import__('threading').Lock()
    d.st = {}
    try:
        d.set_params({'start_kick': 1})
        refused = False
    except ValueError:
        refused = True
    res['r1-rs1-2'] = {'kick_rows_after_no_kick_1to0': kick_rows, 'start_kick_0to1_refused': refused}
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    o = Path(__file__).resolve().parent / 'out'
    o.mkdir(exist_ok=True)
    (o / 'verify_prev.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    shutil.rmtree(cl3.TMP, ignore_errors=True)


if __name__ == '__main__':
    main()
