#!/usr/bin/env python3
"""反証 V-replay: 較正したシミュレーター(P_cal、jf-calib-v1。profile_real_v1 と同じ値)で、天井と自動合わせの閉ループ(2026-10-25)。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I sim_ceiling.py

- 運転手: 追従の運転手 PP(10/16 の pp_driver.mjs)。速さの表 k・上限 vmax を上げて、天井より速く走らせる。速さの指令は開ループ
  (速さを見ない。モデル B と同じく、電池が強いとそのまま速くなる)。電池 kBatt で同じ指令の速さを変える
- 天井: 実装の純粋な関数 safety_caps.ceil_step をそのまま。入力は シミュレーターの速さ + 測りの雑音(σ 0.05 m/s・
  3 サンプルの中央値)・0.1 秒前との差・ヨー(直近 2 フレームの平均)・舵の指令(µs に直す: 左 +557・右 −443 µs が端)
- アプリの遅れ 1 フレーム(cap_test と同じ): フレーム k の状態で決めた指令が、フレーム k+1 に効く
- 自動合わせ: 実装の speed_auto.lap_stats・fb_lap(周 = 参照ループの R1 の始まり s = 6.206 を越えるごと)。m は 1 秒 0.03 まで
- 見るもの: 天井の行き過ぎ(v − C の最大)・出力の向きの入れ替わり(1 秒あたり)・中立のフレーム・周の時間・衝突、
  自動合わせの m_fb の動き(揺れ・張り付き)
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_common as C  # noqa: E402
from rr_common import sc  # noqa: E402
from jetformer import speed_auto as sa  # noqa: E402

BRIDGE = os.path.join(C.HERE, 'sim_bridge.mjs')
NEUTRAL = 1496.0
US_PER_U = 498.0 / 7.24                # シミュレーターの速さの指令 1 → 68.8 µs(ブリッジの換算)
R1_START = 6.206


class Bridge(object):
    def __init__(self):
        self.p = subprocess.Popen(['node', BRIDGE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def send(self, obj):
        self.p.stdin.write(json.dumps(obj) + '\n')
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError('橋が止まった')
        return json.loads(line)

    def close(self):
        try:
            self.p.stdin.write('{"op":"quit"}\n')
            self.p.stdin.flush()
        except Exception:  # noqa: BLE001
            pass
        self.p.wait(timeout=10)


def episode(br, start, k, vmax, kb, ceil_on, auto_on=False, variant='impl', T=60.0, noise=0.05, seed=0, tg=1.0,
            cap_all=4.5, cap_left=4.0):
    rng = np.random.default_rng(seed * 1000 + start)
    p = C.P(variant, cap_all, cap_left)
    st = br.send({'op': 'reset', 'start': start, 'k': k, 'vmax': vmax, 'kBatt': kb, 'timeLimit': T})
    cs = sc.CeilState()
    vm_hist = []                        # (t, 測った速さ)
    raw3 = []                           # 3 サンプルの中央値のため
    yr = []
    queue = []
    fb = sa.FbState()
    m_applied = 1.0
    laps = []
    lap_frames = []
    last_cross = None
    s_prev = st['s']
    rec = {'t': [], 'v': [], 'cap': [], 'mode': [], 'f_in': [], 'f_out': [], 'sec': [], 'left': [], 'm': []}
    while not st['terminal']:
        t = st['t']
        # 測った速さ: 雑音 + 3 つの中央値(100 Hz の 3 サンプル ≈ 撮像のフレームの中の値)
        vraw = [st['v'] + rng.normal(0, noise) for _ in range(3)]
        vm = float(np.median(vraw))
        vm_hist.append((t, vm))
        while len(vm_hist) > 2 and vm_hist[1][0] <= t - 0.1 + 1e-9:
            vm_hist.pop(0)
        a = (vm - vm_hist[0][1]) / max(t - vm_hist[0][0], 1e-3) if vm_hist[0][0] <= t - 0.1 + 1e-9 else float('nan')
        yr.append(st['yaw_rate'])
        gz = float(np.mean(yr[-2:]))
        u_s, u_v = st['pp']['steering'], st['pp']['speed']
        n_s = max(-1.0, min(1.0, u_s / 0.52))
        steer_us = n_s * (557.0 if n_s >= 0 else 443.0)
        f_model = max(u_v, 0.0) * US_PER_U
        # 自動合わせ(m は 1 秒 0.03 まで)
        if auto_on:
            m_target = fb.m_fb
            m_applied += max(-0.03 / 30, min(0.03 / 30, m_target - m_applied))
            m_applied = min(m_applied, max(tg, 1.10) / tg)
        f_in = f_model * (m_applied if auto_on else 1.0)
        thr_in = NEUTRAL - f_in
        cs, o = sc.ceil_step(cs, t, vm, a, steer_us, gz, thr_in, NEUTRAL, True, p)
        thr_out = o.thr_us if ceil_on else thr_in
        f_out = NEUTRAL - thr_out
        u_out = f_out / US_PER_U if f_model > 0 else u_v
        rec['t'].append(t)
        rec['v'].append(st['v'])
        rec['cap'].append(o.cap)
        rec['mode'].append(o.mode if ceil_on else -1)
        rec['f_in'].append(f_in)
        rec['f_out'].append(f_out)
        rec['sec'].append(st['sec'])
        rec['left'].append(o.left)
        rec['m'].append(m_applied)
        lap_frames.append(sa.LapFrame(t, vm, True, 0, False, 0.0, 0.0, bool(ceil_on and o.thr_us != thr_in)))
        # 周: R1 の始まりを越えた
        s = st['s']
        if math.floor((s_prev - R1_START) / 26.1197) < math.floor((s - R1_START) / 26.1197) and s > s_prev:
            if last_cross is not None:
                ls = sa.lap_stats(lap_frames, last_cross, t)
                fb, ev, dl = sa.fb_lap(fb, ls, 1.0) if auto_on else (fb, 0, 0.0)
                laps.append({'t0': last_cross, 'lap_s': t - last_cross, 'v90': ls.v90, 'ceil_frac': ls.ceil_frac,
                             'event': sa.EVENT_NAMES.get(ev, ''), 'm_fb': fb.m_fb, 'warn': fb.warn, 'ok': ls.ok,
                             'reason': ls.reason})
                lap_frames = [f for f in lap_frames if f.t >= t]
            last_cross = t
        s_prev = s
        queue.append((u_s, u_out))
        act = queue.pop(0) if len(queue) > 1 else (0.0, 0.0)
        st = br.send({'op': 'step', 'steering': act[0], 'speed': act[1]})
    R = {k_: np.array(v) for k_, v in rec.items() if k_ != 'sec'}
    R['sec'] = np.array(rec['sec'])
    return st, laps, R


def ceil_metrics(R):
    """天井の揺れの目安。lim = 天井が前進を弱めた・中立にしたフレーム
    - steady_over: 天井が 0.5 秒以上変わっていない(下がっていない)ときの v − C の最大
    - entry_05: 天井が 0.3 m/s 以上下がった(左旋回の入口)とき、0.5 秒後の v − C(D-c4 の「4.05 以下」は +0.05)
    - reversals_per_s: 0.5 秒以上続けて弱めている間の、出力(前進の量)の増減の向きの入れ替わり(|Δ| > 2 µs)
    - df_med・df_p90: その間の 1 フレームの出力の変化 [µs]"""
    t, v, cap, mode, f = R['t'], R['v'], R['cap'], R['mode'], R['f_out']
    lim = mode >= 1
    n = len(t)
    out = {'frames': int(n), 'changed': float(np.mean(lim)) if n else 0.0, 'neutral': float(np.mean(mode == 2)) if n else 0.0}
    # 天井が最後に下がった時刻
    drop_t = np.full(n, -1e9)
    last = -1e9
    entries = []
    for i in range(1, n):
        if np.isfinite(cap[i]) and np.isfinite(cap[i - 1]) and cap[i] < cap[i - 1] - 1e-9:
            last = t[i]
            if cap[i - 1] - cap[i] >= 0.3 and v[i] > cap[i]:
                j = int(np.searchsorted(t, t[i] + 0.5))
                if j < n:
                    entries.append(float(v[j] - cap[j]))
        drop_t[i] = last
    steady = lim & (t - drop_t >= 0.5)
    out['steady_over'] = float(np.max(v[steady] - cap[steady])) if steady.any() else None
    out['entry_05_max'] = max(entries) if entries else None
    out['entries'] = len(entries)
    rev, dur, dfs = 0, 0.0, []
    i = 0
    while i < n:
        if lim[i]:
            j = i
            while j < n and lim[j]:
                j += 1
            if t[j - 1] - t[i] >= 0.5:
                df = np.diff(f[i:j])
                dfs += list(np.abs(df))
                sg = np.sign(df[np.abs(df) > 2.0])
                rev += int(np.sum(sg[1:] != sg[:-1]))
                dur += float(t[j - 1] - t[i])
            i = j
        else:
            i += 1
    out['reversals_per_s'] = rev / dur if dur > 0 else None
    out['limit_runs_s'] = dur
    out['df_med'] = float(np.median(dfs)) if dfs else None
    out['df_p90'] = float(np.percentile(dfs, 90)) if dfs else None
    return out


def main():
    t0 = time.time()
    br = Bridge()
    res = {'note': __doc__, 'ceiling': [], 'auto': []}
    starts = list(range(8))
    # 1. 天井だけ: PP の速さの表 × 電池 × OFF / impl / spec
    conds = [(k, 5.5, kb, mode, 0.05) for k in (1.2, 1.4) for kb in (0.85, 1.0, 1.12, 1.3) for mode in ('off', 'impl', 'spec')]
    conds += [(1.4, 5.5, kb, mode, nz) for kb in (1.0, 1.3) for mode in ('impl', 'spec') for nz in (0.0, 0.12)]
    for k, vmax, kb, mode, nz in conds:
            if True:
                eps = []
                for s0 in starts:
                    st, laps, R = episode(br, s0, k, vmax, kb, mode != 'off', variant='impl' if mode == 'off' else mode,
                                          noise=nz)
                    cm = ceil_metrics(R) if mode != 'off' else {}
                    left = R['left'].astype(bool)
                    eps.append({'start': s0, 'reason': st['reason'], 'collision': st['collision'], 't_end': st['t'],
                                'laps': [l['lap_s'] for l in laps], 'vmax': float(R['v'].max()),
                                'vmax_left': float(R['v'][left].max()) if left.any() else None,
                                'v_sw_max': float(R['v'][R['sec'] == 'SW'].max()) if (R['sec'] == 'SW').any() else None,
                                'ceil': cm})
                res['ceiling'].append({'k': k, 'vmax': vmax, 'kBatt': kb, 'mode': mode, 'noise': nz, 'episodes': eps})
                print('%.0fs k=%.1f kb=%.2f %s nz=%.2f: collide %d/%d laps %.2f' % (
                    time.time() - t0, k, kb, mode, nz, sum(e['collision'] is not None for e in eps), len(eps),
                    np.mean([x for e in eps for x in e['laps']]) if any(e['laps'] for e in eps) else float('nan')), flush=True)
    # 2. 天井 + 自動合わせ(90 秒): 電池 × 自動合わせ ON / OFF
    for k, vmax in ((1.0, 5.5), (1.2, 5.5)):
        for kb in (0.85, 1.0, 1.15):
            for auto in (False, True):
                eps = []
                for s0 in starts[:6]:
                    st, laps, R = episode(br, s0, k, vmax, kb, True, auto_on=auto, T=90.0)
                    eps.append({'start': s0, 'reason': st['reason'], 'collision': st['collision'], 't_end': st['t'],
                                'laps': laps, 'ceil': ceil_metrics(R), 'm_end': float(R['m'][-1])})
                res['auto'].append({'k': k, 'vmax': vmax, 'kBatt': kb, 'auto': auto, 'episodes': eps})
                print('%.0fs auto k=%.1f kb=%.2f auto=%s' % (time.time() - t0, k, kb, auto), flush=True)
    br.close()
    res['elapsed_s'] = round(time.time() - t0, 1)
    C.jdump(res, os.path.join(C.HERE, 'sim_ceiling.json'))
    print('done %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
