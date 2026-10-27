"""反証 1 回目・safety の追加: (1) 場面 A の中立止めの長さを変えて、後ろの当たりを見落とす境目を出す。
(2) safety1-1 の直し(rec_hold_keep)がカメラが来ない中立止め・プロポの信号が来ない中立止めでも中止を残すか。
(3) 場面 B と同じ切り替えを、リカバリーの後退の途中にしたとき。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python experiments/20261026_fix/fix3/refute1_safety/probe_refute1_more.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_refute1 as P                  # noqa: E402
from probe_refute1 import A, C, N, PARAMS, build, mark, run, tl   # noqa: E402


def sweep_a():
    out = {}
    for drop_len in (0.067, 0.1, 0.134, 0.167, 0.2, 0.25):
        hit = 3.25
        for keep in (0, 1):
            frames, imu = P.scene_a(hit_at=hit, drop=(hit - 0.02, hit - 0.02 + drop_len))
            _, pca, rec = run(dict(PARAMS, recovery=1, rec_back_guard=1, rec_hold_keep=keep), frames, imu)
            r = P.summarize_rear(frames, rec, pca, hit)
            out['drop=%.3f keep=%d' % (drop_len, keep)] = {k: r[k] for k in ('reverse_s_after_rear_hit', 'guard_events',
                                                                               'first_last')}
    return out


def stuck(dur=24.0, cam_drop=None, rc_lost=None):
    frames, imu = build(dur, sel=lambda s: s >= 1.0, v=lambda s: 2.0 if 1.0 <= s < 2.0 else 0.0, pred=(C, N - 120.0),
                        valid3=(lambda s: not (rc_lost[0] <= s < rc_lost[1])) if rc_lost else True)
    if cam_drop:
        mark(frames, 'cam_drop', *cam_drop)
    return frames, imu


def giveup_paths():
    frames, imu = stuck()
    rows = tl(frames, run(dict(PARAMS, recovery=1, launch_boost=0), frames, imu)[2])
    g0 = next(s for s, k, h, o, sel in rows if k == A.ASSIST_GIVEUP)
    out = {'giveup_s': g0}
    for name, kw in (('cam_drop_0.3s', {'cam_drop': (g0 + 1.5, g0 + 1.8)}), ('rc_lost_0.3s', {'rc_lost': (g0 + 1.5, g0 + 1.8)})):
        for keep in (0, 1):
            frames, imu = stuck(**kw)
            rows = tl(frames, run(dict(PARAMS, recovery=1, launch_boost=0, rec_back_guard=keep, rec_hold_keep=keep), frames,
                                  imu)[2])
            after = [x for x in rows if x[0] >= g0 + 2.2 and not x[2]]
            out['%s keep=%d' % (name, keep)] = {
                'push_s_after': round(sum(1 for x in after if x[3] < N) / 30.0, 2),
                'giveup_rows_after': sum(1 for x in after if x[1] == A.ASSIST_GIVEUP), 'rows_after': len(after)}
    return out


def kick_mid_reverse():
    """場面 B の切り替え(launch_no_kick 1 → 0)を、最初のリカバリーの後退の途中にする"""
    frames, imu = P.scene_b()
    p = dict(PARAMS, recovery=1, launch_boost=1, launch_no_kick=1, start_kick=1, rec_back_guard=1)
    rows = tl(frames, run(p, frames, imu)[2])
    r0 = next(s for s, k, h, o, sel in rows if k == A.ASSIST_RECOVERY and o > N + 200.0 and s > 2.0)
    rev = [x for x in rows if x[1] == A.ASSIST_RECOVERY and x[3] > N + 200.0]
    at = r0 + 0.5
    idx = next(i for i, f in enumerate(frames) if f['s'] >= at)

    def hook(i, d):
        if i == idx:
            d.set_params({'launch_no_kick': 0})
    frames, imu = P.scene_b()
    rows2 = tl(frames, run(p, frames, imu, hook=hook)[2])
    seg = [x for x in rows2 if at <= x[0] < at + 2.5 and not x[2]]
    return {'first_recovery_brake_s': r0, 'changed_at_s': round(frames[idx]['s'], 3),
            'kinds_next_20': [x[1] for x in seg[:20]], 'outs_next_20': [x[3] for x in seg[:20]],
            'kick_s': round(sum(1 for x in seg if x[1] == A.ASSIST_START) / 30.0, 2),
            'reverse_rows_without_change': len(rev)}


def main():
    out = {'sweep_a': sweep_a(), 'giveup_paths': giveup_paths(), 'kick_mid_reverse': kick_mid_reverse()}
    (P.OUT / 'probe_refute1_more.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
