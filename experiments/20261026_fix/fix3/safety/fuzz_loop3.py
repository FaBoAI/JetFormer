"""直す 3 回目 safety (4): 本物の走行ループ(drive._loop)を偽物のセンサーで、でたらめな場面に回す(反証 2 回目の
fix2/safety2/fuzz_loop.py の場面と決まりをそのまま使い、新しい項目の ON/OFF を混ぜる)。
混ぜるもの: rec_hold_keep・launch_no_kick・rec_back_guard・start_kick・recovery・launch_boost・速さの天井・自動合わせ・区間の上限・
ToF ガード・ゲイン・launch_us。中立止めを増やすため、車両の値が来ない区切り(0.17〜0.5 秒)を足す。
決まり(fuzz_loop.check の 7 つ + この台本の 5 つ):
  値は 1000〜2000 µs・数 / 中立止めは中立 / 人側では補助なし / 前進は出どころより強くならない / 止まっているときの後退側は
  リカバリーと区間のブレーキだけ / 天井 / ゲインの上限(ここまで fuzz_loop.check)
  + 人側の出力 = モデル(ゲインのあと)のまま / launch_no_kick = 1(と launch_boost = 1)で StartKick が出ない /
    rec_hold_keep = 1(と rec_back_guard = 1)で、リカバリー中止は同じ AI の区間の中で解けない(中立止めをはさんでも)/
    リカバリー ON で、後退の受け付けの種類の行が止まっているのに後退側を出さない / 中立止めの行の補助の種類は「なし」
装置は開かない。結果 out/fuzz_loop3_<種>.json

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python experiments/20261026_fix/fix3/safety/fuzz_loop3.py [本数] [種]
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
sys.path.insert(0, str(ROOT / 'experiments' / '20261026_fix' / 'fix2' / 'safety2'))

from jetformer import assist as A          # noqa: E402
from jetformer import drive                # noqa: E402
from test_launch_drive import C, N, PARAMS, build, run, tof_near   # noqa: E402
import fuzz_loop                           # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / 'out'
TMP = ROOT / 'experiments' / '20261026_fix' / 'fix3' / 'tmp_safety'


def stuck_scene(rng, dur=24.0):
    """リカバリー中止まで行く場面: 人側 → AI 側 → 前へ走る → 壁に当たって止まったまま(ときどき少し後ろへ・前へ)。
    モデルは弱い前進か中立の帯。人側に戻す区切り・前の物も混ぜる(中止が CH3 で解けるか・中立止めで解けないか)"""
    t_hit = rng.uniform(1.6, 3.0)
    wob, _ = fuzz_loop.pw(rng, dur, lambda: rng.choice([0.0] * 8 + [-0.4, 0.3]), 0.1, 0.6)
    hum = [(a, a + rng.uniform(0.1, 1.0)) for a in sorted(rng.uniform(4.0, dur - 1.0) for _ in range(rng.randint(0, 2)))]
    thr = rng.choice([N - 120.0, N - 60.0, N - 30.0, N + 10.0])
    tof_f, _ = fuzz_loop.pw(rng, dur, lambda: rng.choice([None, 300, 450, 900]), 0.3, 3.0)
    return build(dur, sel=lambda s: s >= 1.0 and not any(a <= s < b for a, b in hum),
                 v=lambda s: (2.0 if 1.0 <= s < t_hit else wob(s)) if s >= 1.0 else 0.0,
                 ax=lambda s: 45.0 if t_hit <= s < t_hit + 0.02 else 0.0,
                 tof=lambda s: None if tof_f(s) is None else tof_near(tof_f(s)),
                 pred=(C, thr))


def add_holds(rng, frames, dur):
    """車両の値が来ない区切りを足す(0.17〜0.5 秒 = 0.15 秒を超えると中立止め。1 本に 0〜6 か所)"""
    wins = []
    for _ in range(rng.randint(0, 6)):
        a = rng.uniform(0.5, dur - 0.6)
        wins.append((a, a + rng.uniform(0.17, 0.5)))
    for f in frames:
        if any(a <= f['s'] < b for a, b in wins):
            f['veh_skip'] = True
    return wins


def extra_check(frames, rec, p):
    bad = []
    s_of = {f['cam_t']: f['s'] for f in frames}
    erpm_of, last = {}, None
    for f in frames:
        if not f['veh_skip']:
            last = f['veh']['erpm']
        erpm_of[f['cam_t']] = last
    # drive は rec_hold_keep を後ろの見張りが ON のときだけ効かせる(サーバーは見張りなしの rec_hold_keep = 1 を断る)
    keep = int(p.get('rec_hold_keep', 0)) == 1 and int(p.get('rec_back_guard', 0)) == 1
    nokick = int(p.get('launch_no_kick', 0)) == 1 and int(p.get('launch_boost', 0)) == 1
    rec_on = int(p.get('recovery', 1)) == 1
    info = collections.Counter()
    in_giveup = False
    for t, r in rec.rows:
        s = s_of[t]
        o, m = float(r['out'][1]), float(r['model_out'][1])
        if r['hold']:
            if int(r['assist']) != A.ASSIST_NONE:
                bad.append(('hold_kind', s, int(r['assist'])))
            if not r['select']:
                in_giveup = False             # 人側の中立止め: 区間の終わり
            continue
        ai = bool(r['select']) and not r['dry_run']
        k = int(r['assist'])
        if not ai:
            in_giveup = False
            mo = (float(np.clip(r['model_out'][0], 1000, 2000)), float(np.clip(m, 1000, 2000)))
            if (float(r['out'][0]), o) != mo:
                bad.append(('human_out_changed', s, (round(float(r['out'][0]), 1), round(o, 1)), mo))
            continue
        if nokick and k == A.ASSIST_START:
            bad.append(('kick_with_launch_no_kick', s, round(o, 1)))
        if k == A.ASSIST_GIVEUP:
            in_giveup = True
        elif in_giveup:
            (bad.append if keep else lambda x: info.update([x[0]]))(('giveup_cleared_in_section', s, k, round(o, 1)))
            in_giveup = False
        e = erpm_of[t]
        if rec_on and k == A.ASSIST_ESC and o > N + 40.0 and e is not None and e >= A.STOPPED_ERPM:
            bad.append(('esc_reverse_while_stopped', s, round(o, 1), e))
    return bad, info


def main(n_runs=400, seed=1103):
    rng = random.Random(seed)
    TMP.mkdir(parents=True, exist_ok=True)
    out = {'runs': 0, 'rows': 0, 'hold_rows_ai': 0, 'bad': collections.Counter(), 'examples': [],
           'assist_rows': collections.Counter(), 'launch_runs': 0, 'info_off': collections.Counter(),
           'params_mix': collections.Counter(), 'giveup_runs': 0, 'kick_rows_by_nokick': collections.Counter(),
           'scene_kinds': collections.Counter(), 'giveup_then_hold_runs': collections.Counter()}
    for i in range(n_runs):
        p = dict(PARAMS, launch_boost=rng.choice([1, 1, 1, 0]), speed_ceiling=rng.choice([0, 1]), start_kick=rng.choice([0, 1, 1]),
                 recovery=rng.choice([1, 1, 1, 1, 0]), esc_reverse=1, tof_guard=rng.choice([1, 1, 2, 3]),
                 zone_caps=rng.choice([0, 1]), throttle_gain=rng.choice([0.5, 0.9, 1.0, 1.2]),
                 launch_us=rng.choice([150.0, 200.0, 250.0, 300.0]), rec_back_guard=rng.choice([0, 1]),
                 rec_hold_keep=rng.choice([0, 1]), launch_no_kick=rng.choice([0, 1]))
        p['speed_auto'] = rng.choice([0, 1]) if p['speed_ceiling'] else 0
        kind = 'stuck' if i % 4 == 3 else 'random'
        dur = 24.0 if kind == 'stuck' else 8.0
        frames, imu = stuck_scene(rng, dur) if kind == 'stuck' else fuzz_loop.scene(rng, dur)
        add_holds(rng, frames, dur)
        out['scene_kinds'][kind] += 1
        with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(TMP)) as td:
            d, pca, rec = run(drive, mp, Path(td), p, frames, imu)
        bad = fuzz_loop.check(frames, rec, pca, p)
        b2, info = extra_check(frames, rec, p)
        bad += b2
        out['runs'] += 1
        out['rows'] += len(rec.rows)
        out['launch_runs'] += int(d.st['stats']['launch_boosts'] > 0)
        out['giveup_runs'] += int(any(int(r['assist']) == A.ASSIST_GIVEUP for _, r in rec.rows))
        gi = next((j for j, (_, r) in enumerate(rec.rows) if int(r['assist']) == A.ASSIST_GIVEUP and not r['hold']), None)
        if gi is not None and any(r['hold'] and r['select'] for _, r in rec.rows[gi:]):
            out['giveup_then_hold_runs']['keep=%d,guard=%d' % (p['rec_hold_keep'], p['rec_back_guard'])] += 1     # 中止のあとに AI 側の中立止め
        out['info_off'].update(info)
        for k in ('rec_hold_keep', 'launch_no_kick', 'rec_back_guard', 'launch_boost', 'start_kick', 'recovery'):
            out['params_mix']['%s=%d' % (k, p[k])] += 1
        for _, r in rec.rows:
            if r['hold'] and r['select']:
                out['hold_rows_ai'] += 1
            if not r['hold'] and r['select']:
                out['assist_rows'][int(r['assist'])] += 1
                if int(r['assist']) == A.ASSIST_START and p['launch_boost'] == 1:
                    out['kick_rows_by_nokick']['launch_no_kick=%d' % p['launch_no_kick']] += 1
        for b in bad:
            out['bad'][b[0]] += 1
            if sum(1 for x in out['examples'] if x['kind'] == b[0]) < 5:
                out['examples'].append({'kind': b[0], 'run': i, 'detail': [x if not isinstance(x, float) else round(x, 3)
                                                                          for x in b[1:]],
                                        'params': {k: p[k] for k in ('rec_hold_keep', 'launch_no_kick', 'rec_back_guard',
                                                                     'launch_boost', 'start_kick', 'recovery', 'tof_guard',
                                                                     'zone_caps', 'speed_ceiling', 'throttle_gain')}})
    for k in ('bad', 'info_off', 'params_mix', 'kick_rows_by_nokick', 'scene_kinds', 'giveup_then_hold_runs'):
        out[k] = dict(out[k])
    out['assist_rows'] = {A.ASSIST_NAMES[k] or 'なし': v for k, v in out['assist_rows'].items()}
    out['seed'] = seed
    path = OUT_DIR / ('fuzz_loop3_%d.json' % seed)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != 'examples'}, ensure_ascii=False))
    for x in out['examples'][:30]:
        print(x)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400, int(sys.argv[2]) if len(sys.argv) > 2 else 1103)
