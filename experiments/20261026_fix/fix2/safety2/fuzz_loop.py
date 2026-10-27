"""反証 2 回目・safety(2026-10-26 fix2): 走行ループ(drive._loop)を、でたらめな場面で、発進の前進・発進の補助・速さの天井・
自動合わせ・区間の上限・ToF ガード・リカバリーを全部 ON(か組み合わせ)にして回し、モーターに出す値の決まりを確かめる。
装置は開かない(偽物は tests/test_launch_drive.py・test_drive_caps.py と同じ)。直さない。結果は out/fuzz_loop.json。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python experiments/20261026_fix/fix2/safety2/fuzz_loop.py [本数] [種]
"""
import collections
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))

from jetformer import assist as A          # noqa: E402
from jetformer import drive                # noqa: E402
from jetformer import safety_caps as sc    # noqa: E402
from test_launch_drive import C, N, PARAMS, build, run, tof_near   # noqa: E402

OUT = Path(__file__).resolve().parent / 'out' / 'fuzz_loop.json'
TMP = ROOT / 'experiments' / '20261026_fix' / 'fix2' / 'tmp_safety2'


def pw(rng, dur, choices, min_len=0.05, max_len=1.5):
    """区切りごとに値を選ぶ関数(始まりからの秒 → 値)"""
    cuts, t = [], 0.0
    while t < dur:
        L = rng.uniform(min_len, max_len)
        cuts.append((t, t + L, rng.choice(choices) if not callable(choices) else choices()))
        t += L
    def f(s):
        for a, b, v in cuts:
            if a <= s < b:
                return v
        return cuts[-1][2]
    return f, cuts


def scene(rng, dur):
    # CH3: 人側で始める(発進の前進が始まれる)。短い人側(1 フレーム)も混ぜる
    sel_cuts, t, cur = [], 0.0, False
    while t < dur:
        L = rng.choice([0.04, 0.07, 0.3, 0.8, 1.5, 2.5])
        sel_cuts.append((t, t + L, cur))
        t += L
        cur = not cur
    def sel(s):
        for a, b, v in sel_cuts:
            if a <= s < b:
                return v
        return sel_cuts[-1][2]
    # 速さ: 止まる・走る・後ろへ・衝突で急に止まる
    v_f, _ = pw(rng, dur, lambda: rng.choice([0.0, 0.0, 0.0, 0.1, 0.5, 1.2, 2.4, 3.0, 4.4, 4.8, 5.2, -0.3, -1.0]), 0.1, 1.2)
    gz_f, _ = pw(rng, dur, lambda: rng.choice([0.0, 0.0, 30.0, 70.0, -90.0, -200.0, 150.0]), 0.1, 1.0)
    ax_f, _ = pw(rng, dur, lambda: rng.choice([0.0] * 12 + [30.0, 60.0]), 0.01, 0.3)
    tof_f, _ = pw(rng, dur, lambda: rng.choice([None, None, None, 300, 450, 700, 1000, 1300, 1900]), 0.1, 1.5)
    pred_f, _ = pw(rng, dur, lambda: (rng.uniform(1000, 2000), rng.choice([N, N + 20, N + 100, N + 300, N - 30, N - 80,
                                                                            N - 150, N - 300, rng.uniform(1100, 1800)])),
                   0.05, 1.0)
    valid3_f, _ = pw(rng, dur, lambda: rng.random() > 0.05, 0.03, 0.5)
    skip_f, _ = pw(rng, dur, lambda: rng.random() < 0.06, 0.05, 0.3)
    return build(dur, sel=sel, v=v_f, gz=gz_f, ax=ax_f, tof=lambda s: None if tof_f(s) is None else tof_near(tof_f(s)),
                 pred=pred_f, valid3=valid3_f, veh_skip=skip_f)


def check(frames, rec, pca, p):
    """決まり: 前進は出どころ(モデル・発進の前進・補助・リカバリー)より強くしない、後ろ側は決まった場合だけ、
    人側・中立止めは何もしない、天井は v ≥ 天井 + 0.2 で前進を出さない、ゲインは上限まで"""
    bad = []
    erpm_of, last = {}, None
    for f in frames:                     # 走行ループが使う車両の値 = そのフレームまでで最後に届いた値(veh_skip のフレームは届かない)
        if not f['veh_skip']:
            last = f['veh']['erpm']
        erpm_of[f['cam_t']] = last
    s_of = {f['cam_t']: f['s'] for f in frames}
    tg = p['throttle_gain']
    gmax = max(tg, 1.10) + 1e-9
    us = p.get('launch_us', 250.0)
    rec_on = p.get('recovery', 1) == 1
    prev_sel = None
    for t, r in rec.rows:
        o = float(r['out'][1])
        s = s_of[t]
        e = erpm_of[t]
        if not np.isfinite(o) or not (1000 <= o <= 2000):
            bad.append(('not_finite_or_range', s, o))
            continue
        if r['hold']:
            if o != N:
                bad.append(('hold_not_neutral', s, o))
            continue
        ai = bool(r['select']) and not r['dry_run']
        k = int(r['assist'])
        m = float(r['model_out'][1])
        fwd, mfwd = N - o, N - m
        if not ai:
            if k != A.ASSIST_NONE or r['launch_on']:
                bad.append(('human_assist', s, k))
            continue
        g = float(r['gains'][1])
        if g > gmax:
            bad.append(('gain_over', s, g, tg))
        lim = {A.ASSIST_NONE: mfwd, A.ASSIST_ESC: max(mfwd, 0.0), A.ASSIST_START: max(mfwd, 80.0),
               A.ASSIST_LAUNCH: max(mfwd, us), A.ASSIST_RECOVERY: 130.0, A.ASSIST_GIVEUP: 0.0}[k]
        if fwd > lim + 1e-6:
            bad.append(('forward_over', s, k, round(o, 1), round(m, 1), lim))
        if o > N + 40.0:                         # 後ろ側(ブレーキ・後退)
            zone_brake = int(r.get('cap_mode', -1)) == 3 and r.get('cap_applied')
            if k in (A.ASSIST_LAUNCH, A.ASSIST_START, A.ASSIST_GIVEUP):
                bad.append(('reverse_in_' + str(k), s, round(o, 1)))
            elif k == A.ASSIST_NONE and rec_on and e >= A.STOPPED_ERPM and not zone_brake:
                bad.append(('reverse_while_stopped', s, round(o, 1), e))
        if r.get('ceil_on') and r.get('ceil_mode', -1) >= 0 and np.isfinite(r.get('ceil_cap', np.nan)):
            if r['ceil_v'] >= r['ceil_cap'] + 0.2 + 1e-9 and fwd > 1e-6:
                bad.append(('ceil_forward_over_cap', s, round(o, 1), r['ceil_v'], r['ceil_cap'], k))
    return bad


def main(n_runs=200, seed=1026):
    rng = random.Random(seed)
    TMP.mkdir(parents=True, exist_ok=True)
    out = {'runs': 0, 'rows': 0, 'bad': collections.Counter(), 'examples': [], 'assist_rows': collections.Counter(),
           'launch_runs': 0}
    for i in range(n_runs):
        p = dict(PARAMS, launch_boost=1, speed_ceiling=1, speed_auto=rng.choice([0, 1]), start_kick=rng.choice([0, 1]),
                 recovery=1, esc_reverse=1, tof_guard=rng.choice([1, 1, 2, 3]), zone_caps=rng.choice([0, 1]),
                 throttle_gain=rng.choice([0.5, 0.9, 1.0, 1.2]), launch_us=rng.choice([150.0, 200.0, 250.0, 300.0]))
        frames, imu = scene(rng, 8.0)
        with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(TMP)) as td:
            d, pca, rec = run(drive, mp, Path(td), p, frames, imu)
        bad = check(frames, rec, pca, p)
        out['runs'] += 1
        out['rows'] += len(rec.rows)
        out['launch_runs'] += int(d.st['stats']['launch_boosts'] > 0)
        for _, r in rec.rows:
            if not r['hold'] and r['select']:
                out['assist_rows'][int(r['assist'])] += 1
        for b in bad:
            out['bad'][b[0]] += 1
            if sum(1 for x in out['examples'] if x['kind'] == b[0]) < 5:
                out['examples'].append({'kind': b[0], 'run': i, 'detail': [x if not isinstance(x, float) else round(x, 3)
                                                                          for x in b[1:]],
                                        'params': {k: p[k] for k in ('speed_auto', 'start_kick', 'tof_guard', 'zone_caps',
                                                                     'throttle_gain', 'launch_us')}})
    out['bad'] = dict(out['bad'])
    out['assist_rows'] = {A.ASSIST_NAMES[k] or 'なし': v for k, v in out['assist_rows'].items()}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != 'examples'}, ensure_ascii=False))
    for x in out['examples'][:30]:
        print(x)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200, int(sys.argv[2]) if len(sys.argv) > 2 else 1026)
