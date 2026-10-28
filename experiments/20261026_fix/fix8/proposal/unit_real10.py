"""fix8 係 safety-proposal: 本物の記録の eRPM(VEHICLE.npz。data/raw を読むだけ)を Recovery の 'reverse' に入れて、今(base)と
案 pA・pB・pC の「やり直しの決め」を比べる(Recovery だけの開いたループ。フレームは 30 fps、各フレームはそれより前の最新の
車両の値を見る = drive と同じ)。装置・GPU・API には触れない。

(a) spikes: 止まっている車の eRPM の跳ね(scan_bursts と同じ決め: eRPM > 300 の並びで hall_count の進みが 2 以下。
    置く窓 0.5 秒の hall_count の幅も 2 以下 = 本当に止まっている)を、
    ESC が後退に入らない(下がれない)ふつうの 'reverse' の始め −0.02〜+0.40 秒に置く(1/120 秒きざみ = フレームの位相 4 つ)。
    正しい動き = やり直し('gap' へ)。数える: やり直した(直った)/ 後退側を出し続けて escape(残り 1 の形)。
(b) backing: 本当に下がった並び(eRPM > 300 の並びを 0.15 秒以内の途切れでつなぎ、0.4 秒以上・hall_count が 20 以上進む・
    始めの前 0.3 秒は止まっている)を、ふつうの 'reverse' の始めから d 秒(0.05〜0.33)あとに下がり始めるように置く。
    数える: 下がり始めた(フレームで見えた)あとにやり直した = 下がった時間を捨てて多く下がる(悪い)。
(b2) backing_skip: 同じ並びを、'gap' を飛ばした 'reverse'(back_early)の始めから d 秒あとに置く(残り 2 の形)。
    数える: 下がり始めたあとの early_none(悪い)。
形: base / pA / pB / pC の見張り ON(esc_rev は書いた値を EscMayReverse に入れて渡す。(b2) は None = 分からない)、
    OFF(base)と、案を OFF にも入れた形(pXx。P8_OFF_TOO = True)。(b2) は ON だけ('gap' を飛ばすのは見張りだけ)。
出力 out/unit_real10.json
"""
import collections
import glob
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = '/home/jetson/jetformer/data/raw'
PSHA = {'base': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f',
        'pA': '82e2f8a584c66fe292d2122800dcacb928d93c24aba78ca53faa5331915528d3',
        'pB': '05d78b8aeaf09a64364e284191ce19259b3b0baee4d449674056f3fe6228e5b6',
        'pC': '7ac97688a93713704762dc29f3ff64d17dc2ec4b758332ec4200ffd2240585b8'}
C, N = 1441.0, 1496.0
FDT = 1.0 / 30
BACKING = 300


def load(name, offtoo=False):
    fp = HERE / name / 'assist.py'
    assert hashlib.sha256(fp.read_bytes()).hexdigest() == PSHA[name], name
    spec = importlib.util.spec_from_loader('jetformer._p8u_%s_%d' % (name, offtoo), loader=None)
    m = importlib.util.module_from_spec(spec)
    exec(compile(fp.read_text(encoding='utf-8'), str(fp), 'exec'), m.__dict__)
    m.P8_OFF_TOO = bool(offtoo)
    return m


def runs_of(pos):
    out, i0 = [], None
    for i, p in enumerate(pos):
        if p and i0 is None:
            i0 = i
        elif not p and i0 is not None:
            out.append((i0, i))
            i0 = None
    if i0 is not None:
        out.append((i0, len(pos)))
    return out


def scan():
    """戻り値 (spikes, backs, trunc): spikes = [(名前, 時刻の並び(始め 0), eRPM の並び)]・backs = 同じ(始め 0 = 下がり始めのサンプル)・
    trunc = 跳ねの窓で、車が動き始めた(hall_count が跳ねの前から 2 より動いた)サンプルから先を 0 にしたもの(跳ねの部分だけ)"""
    spikes, backs, trunc = [], [], []
    for fp in sorted(glob.glob(RAW + '/2026*/*/VEHICLE.npz')):
        if os.path.getsize(fp) == 0:
            continue
        try:
            d = np.load(fp)
            t = d['timestamps_us'].astype(np.int64) / 1e6
            e = d['erpm'].astype(np.float64)
            h = d['hall_count'].astype(np.int64)
        except Exception:      # noqa: BLE001
            continue
        if len(t) < 10:
            continue
        name = fp.split('/raw/')[1].rsplit('/', 1)[0]
        pos = e > BACKING
        rr = runs_of(pos)
        for i0, i1 in rr:                 # 跳ね(scan_bursts と同じ決め)
            tb = t[i1] if i1 < len(t) else t[i1 - 1] + 0.01
            j1 = int(np.searchsorted(t, tb + 0.05))
            dh = abs(int(h[min(j1, len(h) - 1)] - h[max(i0 - 1, 0)]))
            if dh <= 2 and t[i0] - t[max(i0 - 1, 0)] < 0.05:
                w = (t >= t[i0] - 0.05) & (t < t[i0] + 0.45)
                hw, ew = h[w], np.where(e[w] > BACKING, e[w], 0.0)
                h0 = int(h[max(i0 - 1, 0)])
                mv = np.nonzero(np.abs(hw - h0) > 2)[0]
                et = ew.copy()
                if len(mv):
                    et[mv[0]:] = 0.0
                trunc.append((name + '@%.2f' % (t[i0] - t[0]), t[w] - t[i0], et))
                if int(h[w].max() - h[w].min()) > 2:       # 窓の 0.5 秒のどこかで本当に動いた(跳ねのあと下がり始めた など)は除く
                    continue
                spikes.append((name + '@%.2f' % (t[i0] - t[0]), t[w] - t[i0], np.where(e[w] > BACKING, e[w], 0.0)))
        # 本当に下がった並び: 0.15 秒以内の途切れでつなぐ
        merged = []
        for i0, i1 in rr:
            if merged and t[i0] - t[merged[-1][1] - 1] <= 0.15:
                merged[-1] = (merged[-1][0], i1)
            else:
                merged.append((i0, i1))
        for i0, i1 in merged:
            dur = t[i1 - 1] - t[i0]
            dh = abs(int(h[min(i1, len(h) - 1)] - h[i0]))
            pre = (t >= t[i0] - 0.3) & (t < t[i0])
            if dur >= 0.4 and dh >= 20 and pre.sum() >= 10 and np.all(np.abs(e[pre]) <= BACKING):
                w = (t >= t[i0] - 0.3) & (t < t[i0] + 1.2)
                backs.append((name + '@%.2f' % (t[i0] - t[0]), t[w] - t[i0], e[w]))
    return spikes, backs, trunc


def trace_fn(tt, ee, t_at):
    """並び(始め 0)を時刻 t_at に置いた eRPM(フレームの時刻 tf より前の最新のサンプル。並びの外は 0)"""
    def f(tf):
        x = tf - t_at
        if x < tt[0] or x > tt[-1] + 0.02:
            return 0.0
        j = int(np.searchsorted(tt, x, side='right')) - 1
        return float(ee[j]) if j >= 0 else 0.0
    return f


class Car(object):
    """Recovery と EscMayReverse を drive と同じ順で回す(出す値を gate に入れ、次のフレームの esc_rev にする)"""

    def __init__(self, m, guard, esc_none=False):
        self.m, self.guard, self.esc_none = m, guard, esc_none
        self.r = m.Recovery((C, N))
        self.g = m.EscMayReverse(N)
        self.q = None
        self.log = []

    def step(self, t, erpm, model=(C, N + 20.0)):
        ns = int(round(t * 1e9))
        er = None
        if self.guard and not self.esc_none and self.q is not None:
            er = self.g.rev_s(self.q, ns) > 0.0
        self.q = ns
        p0 = self.r.phase
        o, k = self.r.step(model, erpm, t, 0.0, back_guard=self.guard, esc_rev=er)
        ev, self.r.guard_event = self.r.guard_event, None
        self.g.write(ns + 1, (o or model)[1])
        self.log.append((round(t, 4), p0, self.r.phase, erpm, ev and ev[0], (o or model)[1]))
        return o, k


def arm(car, t):
    for _ in range(40):
        car.step(t, -5000.0, model=(C, N - 150.0))
        t += FDT
    return t


def run_normal(m, guard, f, t_R_hint=None, t_end=4.5):
    """ふつうの 'reverse' まで進め、そのあと f(t) の eRPM で回す。戻り値 (入った時刻, 結果, log)"""
    car = Car(m, guard)
    t = arm(car, 0.0)
    t_R = None
    while t < t_end:
        e = f(t) if t_R_hint is not None else 0.0
        car.step(t, e)
        if t_R is None and car.r.phase == 'reverse':
            t_R = t
        if t_R is not None and car.r.phase != 'reverse':
            break
        t += FDT
    return t_R, car


def classify(car, t_R, t_vis=None):
    """'reverse' に入ったあと最初に出た段: gap(やり直し・early_none)/ coast・escape(やり直さない)"""
    after = [(t, p0, p1, ev) for t, p0, p1, e, ev, o in car.log if t >= t_R and p0 == 'reverse' and p1 != 'reverse']
    if not after:
        return 'stay', None
    t, p0, p1, ev = after[0]
    if p1 == 'gap':
        if t_vis is not None and t >= t_vis:
            return ('early_none_after_onset' if ev == 'early_none' else 'retry_after_onset'), round(t - t_R, 3)
        return ('early_none' if ev == 'early_none' else 'retry'), round(t - t_R, 3)
    return p1, round(t - t_R, 3)


def first_vis(f, t_from):
    """t_from 以降で初めて eRPM > 300 を見るフレームの時刻(フレームは t_from から FDT ごと)"""
    t = t_from
    for _ in range(200):
        if f(t) > BACKING:
            return t
        t += FDT
    return None


def main(parts=('a', 'b', 'b2', 'a2')):
    spikes, backs, trunc = scan()
    print('spikes', len(spikes), 'backs', len(backs), 'trunc', len(trunc), flush=True)
    V = {'base_on': (load('base'), True), 'pA_on': (load('pA'), True), 'pB_on': (load('pB'), True),
         'pC_on': (load('pC'), True), 'base_off': (load('base'), False),
         'pAx_off': (load('pA', True), False), 'pBx_off': (load('pB', True), False), 'pCx_off': (load('pC', True), False)}
    # 'reverse' に入る時刻(eRPM 0 のまま。形で同じ)
    t_R0 = run_normal(V['base_on'][0], True, lambda t: 0.0)[0]
    out = {'n_spikes': len(spikes), 'n_backs': len(backs), 't_R0': t_R0, 'a': {}, 'b': {}, 'b2': {}}
    if os.path.exists(HERE / 'out' / 'unit_real10.json'):
        old = json.loads((HERE / 'out' / 'unit_real10.json').read_text(encoding='utf-8'))
        out.update({k: v for k, v in old.items() if k not in ('n_spikes', 'n_backs', 't_R0')})
    out['n_trunc'] = len(trunc)
    for part, src in (('a', spikes), ('a2', trunc)):
        if part in parts:
            spike_part(out, part, src, V, t_R0)
    if 'b' in parts:
        part_b(out, backs, V, t_R0)
    if 'b2' in parts:
        part_b2(out, backs, V)
    (HERE / 'out' / 'unit_real10.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


def spike_part(out, part, spikes, V, t_R0):
    cnt = {k: collections.Counter() for k in V}
    ex = collections.defaultdict(list)
    for name, tt, ee in spikes:
        for q in range(-2, 49):            # 1/120 秒きざみ −0.017〜+0.40 秒
            off = q / 120.0
            f = trace_fn(tt, ee, t_R0 + off)
            seen_any = any(f(t_R0 + j * FDT) > BACKING for j in range(0, 14))
            if not seen_any:
                continue
            for k, (m, guard) in V.items():
                t_R, car = run_normal(m, guard, f, t_R_hint=True)
                assert abs(t_R - t_R0) < 1e-9 or k.endswith('_off'), (k, t_R, t_R0)
                res, dt_ = classify(car, t_R)
                cnt[k][res] += 1
                if res != 'retry':
                    ex[k].append((name, round(off, 3), res, dt_))
    out[part] = {k: dict(v) for k, v in cnt.items()}
    out[part + '_fail_spikes'] = {k: dict(collections.Counter(x[0] for x in v)) for k, v in ex.items()}
    # 跳ねの並びの形(フレームから見て続けて後ろへ動いて見える最長・1 フレームの途切れをつないだ最長)
    shape = {}
    for name, tt, ee in spikes:
        best = [0.0, 0.0]
        for q in range(4):
            f = trace_fn(tt, ee, q / 120.0)
            fr = [f(j * FDT + q / 120.0 * 0) > BACKING for j in range(-2, 16)]
            for mode in (0, 1):
                run, t0, miss = 0.0, None, 0
                for j, b in enumerate(fr):
                    if b:
                        miss = 0
                        t0 = j if t0 is None else t0
                        best[mode] = max(best[mode], (j - t0) * FDT)
                    else:
                        miss += 1
                        if mode == 0 or miss >= 2:
                            t0 = None
        shape[name] = [round(best[0], 3), round(best[1], 3)]
    out[part + '_spike_shape'] = shape
    print(part, out[part], flush=True)


def part_b(out, backs, V, t_R0):
    # (b) 本当に下がった並び・ふつうの 'reverse'
    cnt = {k: collections.Counter() for k in V}
    lost = {k: [] for k in V}
    for name, tt, ee in backs:
        for d in (0.05, 0.08, 0.12, 0.15, 0.2, 0.27, 0.33):
            for q in range(4):
                t_on = t_R0 + d + q / 120.0
                f = trace_fn(tt, ee, t_on)
                tv = first_vis(f, t_R0)
                for k, (m, guard) in V.items():
                    t_R, car = run_normal(m, guard, f, t_R_hint=True)
                    res, dt_ = classify(car, t_R, tv)
                    cnt[k][res] += 1
                    if res.endswith('after_onset'):
                        lost[k].append((name, d, q, dt_))
    out['b'] = {k: dict(v) for k, v in cnt.items()}
    out['b_lost_examples'] = {k: v[:8] for k, v in lost.items()}
    out['b_lost_files'] = {k: len(set(x[0] for x in v)) for k, v in lost.items()}
    print('b', out['b'], flush=True)


def part_b2(out, backs, V):
    # (b2) 'gap' を飛ばした 'reverse'(back_early)。ON だけ。esc_rev None(drive を通さない = 分からない)
    VB = {k: v for k, v in V.items() if k.endswith('_on')}
    cnt = {k: collections.Counter() for k in VB}
    lost = {k: [] for k in VB}
    for name, tt, ee in backs:
        for d in (0.1, 0.15, 0.2, 0.27, 0.33, 0.4):
            for q in range(4):
                for k, (m, guard) in VB.items():
                    car = Car(m, True, esc_none=True)
                    t = arm(car, 0.0)
                    while car.r.phase != 'brake':
                        car.step(t, 0.0)
                        t += FDT
                    car.step(t, 0.0)
                    t += FDT
                    car.step(t, 1322.0)             # 跳ね 1 つで back_early → 'gap' を飛ばした 'reverse'
                    assert car.r.phase == 'reverse' and car.r.skip_gap, (k, car.r.phase)
                    t_R = t
                    t_on = t_R + d + q / 120.0
                    f = trace_fn(tt, ee, t_on)
                    tv = first_vis(f, t_R + FDT)
                    t += FDT
                    while t < t_R + 2.0:
                        car.step(t, f(t))
                        if car.r.phase != 'reverse':
                            break
                        t += FDT
                    res, dt_ = classify(car, t_R, tv)
                    cnt[k][res] += 1
                    if res.endswith('after_onset'):
                        lost[k].append((name, d, q, dt_))
    out['b2'] = {k: dict(v) for k, v in cnt.items()}
    out['b2_lost_examples'] = {k: v[:8] for k, v in lost.items()}
    out['b2_lost_files'] = {k: len(set(x[0] for x in v)) for k, v in lost.items()}
    print('b2', out['b2'], flush=True)


if __name__ == '__main__':
    main(tuple(sys.argv[1:]) or ('a', 'b', 'b2', 'a2'))
