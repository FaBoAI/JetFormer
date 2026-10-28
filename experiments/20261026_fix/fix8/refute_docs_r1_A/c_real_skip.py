"""打ち消しの票 A(f8r1-docs-residual2-reversed の (2)): 本物の後退の eRPM の並び(data/raw の VEHICLE.npz。読むだけ)を、
今の jetformer/assist.py の Recovery(見張り ON・esc_rev None)の「'gap' を飛ばした 'reverse'」(brake の中の 1 フレームの跳ねで
back_early)に置き、下がり始めた(フレームで eRPM > 300 を見た)あとに early_none('gap' からやり直し)が出るかを見る。
当たり・引っかかりの無い、ふつうの後退の並びでも起きるか(approvals 14 の 2 行目「起きるのは、下がっている途中で後ろに当たる・
引っかかるときなど」)を照らす。指摘の台本は使わない(Recovery を直接、自分で書いた手順で回す)。
並び: 引数の (記録, シーン, 秒) と、data/raw の 10/10 の AI 走行のリカバリーでない後退の並びを自分で拾ったもの。"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
RAW = JF / 'data' / 'raw'
spec = importlib.util.spec_from_file_location('assist_refA', JF / 'jetformer' / 'assist.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
C, N = 1441.0, 1496.0
FDT = 1.0 / 30


def seq_from(log, scene, t0, pre=0.05, dur=1.2):
    v = np.load(RAW / log / scene / 'VEHICLE.npz')
    ts = v['timestamps_us'].astype(np.int64)
    tt = (ts - ts[0]) / 1e6
    e = v['erpm'].astype(float)
    m = (tt >= t0 - pre) & (tt <= t0 + dur)
    return tt[m] - t0, e[m]          # t0 = 下がり始め(> 300 の最初のサンプル)が 0 付近


def run_one(st, se, d, phase_off=0.0):
    """st, se: 並び(相対秒・eRPM)。entry(back_early のフレーム)から d 秒あとに並びの 0 秒を置く。
    返り値: (early_none が出たか, 下がり始めを見たあとか, 出来事の列)"""
    r = A.Recovery((C, N))
    t = phase_off
    evs = []
    # 走る(armed)→ 止まる(0.8 秒で stuck → brake)
    while t < 1.0:
        r.step((C, N - 100.0), -2000.0, t, back_guard=True, esc_rev=None)
        t += FDT
    while r.phase is None and t < 3.0:
        r.step((C, N - 20.0), 0.0, t, back_guard=True, esc_rev=None)
        t += FDT
    assert r.phase == 'brake', r.phase
    t += FDT
    # brake の中の 1 フレームの跳ね(1322)= back_early
    r.step((C, N - 20.0), 1322.0, t, back_guard=True, esc_rev=None)
    if r.guard_event:
        evs.append((round(t, 3), r.guard_event))
        r.guard_event = None
    assert r.phase == 'reverse' and r.skip_gap, (r.phase, r.skip_gap)
    t_entry = t
    seen_back = False
    early_after = False
    end = t_entry + d + float(st[-1])
    while t < end:
        t += FDT
        rel = t - t_entry - d
        k = np.searchsorted(st, rel, side='right') - 1
        erpm = float(se[k]) if k >= 0 else 0.0
        ph0 = r.phase
        r.step((C, N - 20.0), erpm, t, back_guard=True, esc_rev=None)
        if ph0 == 'reverse' and erpm > A.BACKING_ERPM:
            seen_back = True
        if r.guard_event:
            evs.append((round(t - t_entry, 3), r.guard_event[0], ph0, erpm))
            if r.guard_event[0] == 'early_none' and seen_back:
                early_after = True
            r.guard_event = None
        if r.phase in ('coast', 'escape', None):
            break
    return early_after, seen_back, evs, r.phase


def find_backs(day='20261010', max_n=200):
    """10/10 の記録から、止まっていたあと(前 0.3 秒 |eRPM| < 150)に > 300 になり、そのあと 0.4 秒のうち 0.3 秒以上 > 300 の並び
    (本当に下がった)の始め。自分の決め方(指摘の決め方とは別)"""
    out = []
    for root in sorted(RAW.glob(day + '-*')):
        for s in sorted(d for d in root.iterdir() if d.is_dir()):
            p = s / 'VEHICLE.npz'
            if not p.exists():
                continue
            try:
                v = np.load(p)
                ts = v['timestamps_us'].astype(np.int64)
                e = v['erpm'].astype(float)
            except Exception:  # noqa: BLE001
                continue
            tt = (ts - ts[0]) / 1e6
            i = 1
            while i < len(e):
                if e[i] > 300 and e[i - 1] <= 300:
                    pre = (tt >= tt[i] - 0.3) & (tt < tt[i])
                    post = (tt >= tt[i]) & (tt < tt[i] + 0.4)
                    if pre.sum() >= 20 and np.all(np.abs(e[pre]) < 150) and post.sum() >= 30 \
                            and (e[post] > 300).mean() >= 0.75:
                        out.append((root.name, s.name, round(float(tt[i]), 3)))
                        i += 100
                        continue
                i += 1
    return out[:max_n]


def main():
    res = {'cited': {}, 'own': {}}
    cited = [('20261010-192141', '20261010-192141-0005', 15.18), ('20261010-181504', '20261010-181504-0005', 1.39)]
    ds = [round(0.05 + k / 120.0, 4) for k in range(int((0.40 - 0.05) * 120) + 1)]
    for log, sc, t0 in cited:
        st, se = seq_from(log, sc, t0)
        hits = []
        for d in ds:
            for ph in (0.0, 1 / 120, 2 / 120, 3 / 120):
                ea, sb, evs, endp = run_one(st, se, d, ph)
                if ea:
                    hits.append((d, round(ph, 4), evs))
        res['cited']['%s@%.2f' % (sc, t0)] = {'placements': len(ds) * 4, 'early_none_after_backing': len(hits),
                                              'example': hits[:2]}
        print('cited', sc, t0, 'placements', len(ds) * 4, 'early_none after backing seen:', len(hits))
        for h in hits[:2]:
            print('   d=%.3f ph=%.4f' % (h[0], h[1]), h[2])
    backs = find_backs()
    n_seq_hit, n_pl_hit, n_pl = 0, 0, 0
    ex = []
    for log, sc, t0 in backs:
        st, se = seq_from(log, sc, t0)
        hit_here = 0
        for d in ds[::3]:
            for ph in (0.0, 2 / 120):
                n_pl += 1
                ea, sb, evs, endp = run_one(st, se, d, ph)
                if ea:
                    hit_here += 1
        n_pl_hit += hit_here
        if hit_here:
            n_seq_hit += 1
            ex.append((log, sc, t0, hit_here))
    res['own'] = {'n_seq': len(backs), 'n_seq_with_early_none_after_backing': n_seq_hit, 'placements': n_pl,
                  'placements_hit': n_pl_hit, 'examples': ex[:20]}
    print('own 10/10 backing starts:', len(backs), 'seq with early_none after backing:', n_seq_hit,
          'placements', n_pl, 'hit', n_pl_hit)
    for x in ex[:20]:
        print('   ', x)
    # 比べ: なめらかな後退(0 → 1600 eRPM を 0.3 秒で)は起きない
    st = np.arange(0, 1.2, 0.01)
    se = np.clip(st / 0.3, 0, 1) * 1600 + 301 * (st >= 0)
    sm = sum(run_one(st, se, d, 0.0)[0] for d in ds)
    print('smooth ramp: early_none after backing', sm, '/', len(ds))
    res['smooth_hits'] = sm
    (Path(__file__).resolve().parent / 'out' / 'c_real_skip.json').write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                                                                         default=str))


if __name__ == '__main__':
    main()
