"""反証(r2-r2-rs-2): rec_hold_keep = 1 で 2〜5 秒の中立止め(カメラが来ない)のあと、明けてすぐ後退側を出すか。
本物の drive._loop を refute2_safety/closed_loop.py の偽物の車で回す(装置は開かない・直さない)。
比べるもの: 中立止めなしのリカバリー(普段の下がる距離・始まりから後退までの秒)、keep 0/1、中立止めの長さ 0.2〜5 秒。
壁に当たっていない場面は probe_open.py。"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'refute2_safety'))
import closed_loop as CL          # noqa: E402
from closed_loop import C, N, PARAMS   # noqa: E402

CL.TMP = HERE.parent / 'tmp_r2r2rs2'


def run(keep, drop=None, dur=9.0, model=(C, N - 120.0), xf=2.0):
    sc_ = {'dur': dur, 'xf': xf, 'model': model, 'drops': [drop] if drop else []}
    p = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=keep)
    sim, rec, pca = CL.run(p, sc_)
    return sim, rec, pca, list(CL.PH)


def summarize(sim, pca, ph, end):
    after_w = [(round(t - end, 3), th) for t, s, th in pca.w if t >= end]
    first_rev_cmd = next((t for t, th in after_w if th > N + 40), None)
    first_back = next((round(t - end, 3) for t, x, v, *_ in sim.log if t >= end and v < -0.05), None)
    xs = [x for t, x, *_ in sim.log if t >= end]
    x_end = next((x for t, x, *_ in sim.log if t >= end), None)
    back = round(x_end - min(xs), 3) if xs else None
    max_thr = max((th for t, th in after_w if t < 2.0), default=None)
    trans = [(t, p0, p1) for t, p0, p1, hg, ig in ph if p0 != p1 and t >= end - 0.01][:6]
    return {'first_rev_cmd_s': first_rev_cmd, 'first_backing_s': first_back, 'back_m': back, 'max_thr_2s': max_thr,
            'phases_after': trans}


def main():
    res = {}
    # 1. 中立止めなし(普段のリカバリー)
    sim, rec, pca, ph = run(1)
    t_start = next(t for t, p0, p1, hg, ig in ph if p0 is None and p1 is not None)
    times = {}
    for t, p0, p1, hg, ig in ph:
        if p0 != p1 and p1 not in times:
            times[p1] = t
    s = summarize(sim, pca, ph, t_start)
    s['phase_times'] = times
    s['hits'] = sim.car.hits
    res['nohold keep=1'] = s
    print('nohold', s)
    t_slow = None
    # slow_since が立った時刻(壁に当たって止まった時刻): sim.log で v が 0 になった最初
    t_slow = next(t for t, x, v, *_ in sim.log if t > 1.2 and abs(v) < 0.01 and x > 1.9)
    print('stopped at wall', t_slow, 'brake at', times.get('brake'))
    starts = {'stuck(no phase)+0.05': t_slow + 0.05,
              'brake+0.05': times['brake'] + 0.05,
              'gap+0.05': times['gap'] + 0.05,
              'reverse+0.05': times['reverse'] + 0.05}
    for name, t0 in starts.items():
        for L in (0.2, 0.5, 1.0, 2.0, 5.0):
            for keep in (0, 1):
                end = t0 + L
                sim, rec, pca, ph = run(keep, (t0, end), dur=end + 3.0)
                r = summarize(sim, pca, ph, end)
                r['holds'] = sum(1 for _, row in rec.rows if row['hold'])
                key = '%s L=%.1f keep=%d' % (name, L, keep)
                res[key] = r
                print(key, {k: r[k] for k in ('first_rev_cmd_s', 'first_backing_s', 'back_m', 'max_thr_2s')},
                      r['phases_after'][:3])

    # 2. 壁に当たっていない場面は probe_open.py
    (HERE / 'out.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    shutil.rmtree(CL.TMP, ignore_errors=True)


if __name__ == '__main__':
    main()
