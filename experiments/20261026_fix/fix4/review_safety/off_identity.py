"""反証 4 回目・安全: スイッチが OFF のとき、今の drive.py・assist.py の出力が fix3 の前(tests/fixtures/fix3_before)と
ビット単位で同じか(PCA9685 に書いた値・記録の行・出来事)。直さない・装置に触れない。

場面:
  open   = fix2/safety2/fuzz_loop.scene・fix3/safety/fuzz_loop3.stuck_scene + 車両の値が来ない区切り(add_holds)の開いたループ
  closed = fix3/refute2_safety/closed_loop の偽物の ESC・車(前の壁・後ろの物・当たりに続くカメラの止まり・車両の値の止まり)
OFF の形:
  off       rec_hold_keep = 0・launch_no_kick = 0(rec_back_guard・launch_boost・start_kick などはでたらめ)
  keep_noguard  rec_hold_keep = 1・rec_back_guard = 0(drive は見張りが無いと残さない = OFF と同じはず)
  nokick_noboost launch_no_kick = 1・launch_boost = 0(発進の前進が OFF なら効かない = OFF と同じはず)

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        experiments/20261026_fix/fix4/review_safety/off_identity.py [open の本数] [closed の本数] [種]
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rs_common as R          # noqa: E402
from rs_common import A, C, N, PARAMS, drive   # noqa: E402
import fuzz_loop               # noqa: E402
import fuzz_loop3              # noqa: E402


def rand_params(rng, form):
    p = dict(PARAMS, launch_boost=rng.choice([1, 1, 0]), speed_ceiling=rng.choice([0, 1]), start_kick=rng.choice([0, 1, 1]),
             recovery=rng.choice([1, 1, 1, 0]), esc_reverse=rng.choice([1, 1, 0]), tof_guard=rng.choice([1, 2, 3]),
             zone_caps=rng.choice([0, 1]), throttle_gain=rng.choice([0.5, 0.9, 1.0, 1.2]),
             launch_us=rng.choice([150.0, 250.0, 300.0]), rec_back_guard=rng.choice([0, 1]),
             rev_s=rng.choice([0.2, 0.4, 0.8]), rev_us=rng.choice([150.0, 300.0, 500.0]),
             rec_hold_keep=0, launch_no_kick=0)
    p['speed_auto'] = rng.choice([0, 1]) if p['speed_ceiling'] else 0
    if form == 'keep_noguard':
        p.update(rec_hold_keep=1, rec_back_guard=0)
    elif form == 'nokick_noboost':
        p.update(launch_no_kick=1, launch_boost=0)
    return p


def closed_scene(rng):
    dur = rng.choice([8.0, 10.0])
    xf = rng.uniform(1.6, 2.8)
    sc = {'dur': dur, 'xf': xf, 'model': (C, N - rng.choice([90.0, 120.0, 120.0, 200.0])),
          'drops': [], 'veh_skip': []}
    if rng.random() < 0.6:
        sc['xr'] = xf - rng.uniform(0.6, 2.0)
    if rng.random() < 0.6:
        sc['hit_drop'] = (rng.choice([0.0, 0.02, 0.05]), rng.choice([0.12, 0.3, 0.6, 1.0, 2.0]))
    for _ in range(rng.randint(0, 4)):
        a = rng.uniform(1.0, dur - 0.5)
        sc['drops'].append((a, a + rng.uniform(0.1, 1.2)))
    for _ in range(rng.randint(0, 3)):
        a = rng.uniform(1.0, dur - 0.5)
        sc['veh_skip'].append((a, a + rng.uniform(0.05, 0.6)))
    if rng.random() < 0.3:
        a = rng.uniform(2.0, dur - 1.0)
        sc['human'] = [(a, a + rng.uniform(0.1, 0.8))]
    return sc


def main(n_open=150, n_closed=150, seed=4401):
    rng = random.Random(seed)
    B, BA = R.load_snapshot('fix3_before')
    res = {'seed': seed, 'open': {}, 'closed': {}, 'diffs': []}
    try:
        for form in ('off', 'keep_noguard', 'nokick_noboost'):
            cnt = {'runs': 0, 'same': 0, 'rows': 0, 'pca_writes': 0, 'giveup_rows': 0, 'rec_rows': 0, 'hold_rows': 0}
            for i in range(n_open):
                p = rand_params(rng, form)
                kind = 'stuck' if i % 3 == 2 else 'random'
                dur = 16.0 if kind == 'stuck' else 8.0
                frames, imu = fuzz_loop3.stuck_scene(rng, dur) if kind == 'stuck' else fuzz_loop.scene(rng, dur)
                fuzz_loop3.add_holds(rng, frames, dur)
                _, pa, ra = R.run_open(drive, p, frames, imu)
                _, pb, rb = R.run_open(B, p, frames, imu)
                ok1, d1 = R.same_rows(rb.rows, ra.rows)
                ok2 = pa.pairs == pb.pairs and pa.single == pb.single
                ok3, d3 = R.same_events(rb.events, ra.events)
                cnt['runs'] += 1
                cnt['rows'] += len(ra.rows)
                cnt['pca_writes'] += len(pa.pairs)
                cnt['giveup_rows'] += sum(1 for _, r in ra.rows if int(r['assist']) == A.ASSIST_GIVEUP)
                cnt['rec_rows'] += sum(1 for _, r in ra.rows if int(r['assist']) == A.ASSIST_RECOVERY)
                cnt['hold_rows'] += sum(1 for _, r in ra.rows if r['hold'])
                if ok1 and ok2 and ok3:
                    cnt['same'] += 1
                else:
                    res['diffs'].append({'form': form, 'loop': 'open', 'i': i, 'rows': d1, 'pca': ok2, 'events': str(d3)[:300]})
            res['open'][form] = cnt
            cnt = {'runs': 0, 'same': 0, 'rows': 0, 'pca_writes': 0, 'giveup_rows': 0, 'rec_rows': 0, 'hold_rows': 0,
                   'front_hits': 0, 'rear_hits': 0}
            for i in range(n_closed):
                p = rand_params(rng, form)
                sc = closed_scene(rng)
                sa, ra, pa, _ = R.run_closed(p, sc)
                sb, rb, pb, _ = R.run_closed(p, sc, mod=B, amod=BA)
                ok1, d1 = R.same_rows(rb.rows, ra.rows)
                ok2 = pa.w == pb.w
                ok3, d3 = R.same_events(rb.events, ra.events)
                cnt['runs'] += 1
                cnt['rows'] += len(ra.rows)
                cnt['pca_writes'] += len(pa.w)
                cnt['giveup_rows'] += sum(1 for _, r in ra.rows if int(r['assist']) == A.ASSIST_GIVEUP)
                cnt['rec_rows'] += sum(1 for _, r in ra.rows if int(r['assist']) == A.ASSIST_RECOVERY)
                cnt['hold_rows'] += sum(1 for _, r in ra.rows if r['hold'])
                cnt['front_hits'] += sum(1 for h in sa.car.hits if h[1] == 'front')
                cnt['rear_hits'] += sum(1 for h in sa.car.hits if h[1] == 'rear')
                if ok1 and ok2 and ok3 and sa.log == sb.log:
                    cnt['same'] += 1
                else:
                    res['diffs'].append({'form': form, 'loop': 'closed', 'i': i, 'rows': d1, 'pca': ok2,
                                         'events': str(d3)[:300], 'scene': sc})
            res['closed'][form] = cnt
            print(form, 'open', res['open'][form], 'closed', res['closed'][form], flush=True)
    finally:
        R.cleanup()
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / ('off_identity_%d.json' % seed)).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print('diffs', len(res['diffs']))
    for d in res['diffs'][:10]:
        print(d)


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:4]))
