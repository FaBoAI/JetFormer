"""直す 3 回目 safety: ON の効きの数(報告 safety.md の表)。tests/test_rec_hold_keep.py の場面を本物の drive._loop(偽物のセンサー)で回す。
装置は開かない。結果 out/probe_fix3.json

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python experiments/20261026_fix/fix3/safety/probe_fix3.py
"""
import collections
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))

from jetformer import assist as A                    # noqa: E402
from jetformer import drive                          # noqa: E402
import test_rec_hold_keep as T                       # noqa: E402

OUT = Path(__file__).resolve().parent / 'out' / 'probe_fix3.json'
TMP = ROOT / 'experiments' / '20261026_fix' / 'fix3' / 'tmp_safety'
N = T.N


def loop(p, frames, imu, **kw):
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(TMP)) as td:
        return T.run(drive, mp, Path(td), p, frames, imu, **kw)


def seg_s(xs):
    return round(len(xs) / 30.0, 2)


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    res = {}
    # 1a: 中止のあとの中立止め
    frames, imu = T.stuck_scene(24.0)
    tl = T.timeline(frames, loop(T.P_STUCK, frames, imu)[2])
    g0 = next(s for s, k, h, o, sel in tl if k == A.ASSIST_GIVEUP)
    hold_at = round(g0 + 1.5, 3)
    frames, imu = T.stuck_scene(24.0, holds=[(hold_at, 0.2)])
    r = {'giveup_s': g0, 'hold_at_s': hold_at, 'hold_len_s': 0.2}
    for keep in (0, 1):
        t = T.timeline(frames, loop(dict(T.P_STUCK, rec_hold_keep=keep, rec_back_guard=keep), frames, imu)[2])
        aft = [x for x in t if x[0] >= hold_at and not x[2]]
        r['keep%d' % keep] = {'hold_frames': sum(1 for x in t if x[2] and x[4]),
                              'model_forward_s_after_hold': seg_s([x for x in aft if x[1] == A.ASSIST_NONE and x[3] < N]),
                              'giveup_s_after_hold': seg_s([x for x in aft if x[1] == A.ASSIST_GIVEUP]),
                              'out_after_hold': collections.Counter(x[3] for x in aft).most_common(2)}
    res['1a_giveup_then_hold'] = r
    # 1b: リカバリーの途中の中立止め
    frames, imu = T.stuck_scene(14.0)
    tl = T.timeline(frames, loop(T.P_STUCK, frames, imu)[2])
    r0 = next(s for s, k, h, o, sel in tl if k == A.ASSIST_RECOVERY)
    hold_at = round(r0 + 0.45, 3)
    frames, imu = T.stuck_scene(14.0, holds=[(hold_at, 0.2)])
    r = {'recovery_first_s': r0, 'hold_at_s': hold_at, 'scene_end_s': 14.0}
    for keep in (0, 1):
        _, _, rec = loop(dict(T.P_STUCK, rec_hold_keep=keep, rec_back_guard=keep), frames, imu)
        t = T.timeline(frames, rec)
        aft = [x for x in t if x[0] >= hold_at and not x[2]]
        starts = sum(1 for a, b in zip(t, t[1:]) if b[0] >= hold_at and b[1] == A.ASSIST_RECOVERY and a[1] != A.ASSIST_RECOVERY)
        r['keep%d' % keep] = {'recovery_s_after_hold': seg_s([x for x in aft if x[1] == A.ASSIST_RECOVERY]),
                              'recovery_starts_after_hold': starts,
                              'model_forward_s_after_hold': seg_s([x for x in aft if x[1] == A.ASSIST_NONE and x[3] < N]),
                              'giveup_first_s': next((x[0] for x in aft if x[1] == A.ASSIST_GIVEUP), None)}
    res['1b_hold_in_recovery'] = r
    # 10/10 の記録
    if (T.REC_1010 / 'AI_CMD.npz').exists():
        res['replay_1010'] = {'record_first_recovery_row': 177, 'off': T._replay_1010(False), 'on': T._replay_1010(True),
                              'off_no_noise': T._replay_1010(False, True), 'on_no_noise': T._replay_1010(True, True),
                              'human_takeover_row': 190, 'hold_rows': [141, 142], 'impact_row_note': '7.5G の 0.43 秒あとに中立止め'}
    # 発進の前進が出さない区間の StartKick
    r = {}
    for d_mm, floor in ((350, True), (500, True), (900, True), (350, False)):
        frames, imu = T.near_scene(d_mm)
        for nk in (0, 1):
            _, _, rec = loop(dict(T.PARAMS, launch_boost=1, start_kick=1, tof_guard=1, launch_no_kick=nk), frames, imu, floor=floor)
            rows = [x for _, x in rec.rows if x['select'] and not x['hold']]
            fwd = [x for x in rows if x['out'][1] < N]
            r['%s_nokick%d' % ('d%d' % d_mm if floor else 'nofloor', nk)] = {
                'skip': [e[1]['why'] for e in rec.events if e[0] == 'launch_boost'],
                'forward_s': seg_s(fwd), 'forward_us': sorted(set(round(N - x['out'][1], 1) for x in fwd))}
    res['launch_skip_kick'] = r
    # 後ろの見張りと組み合わせ(後退の途中に中立止め → 後ろに当たる)
    frames, imu = T.recovery_scene(back_from=2.95, hit_at=3.25, holds=[(3.05, 0.2)])
    s_of = {f['cam_t']: f['s'] for f in frames}
    r = {}
    for keep in (0, 1):
        for guard in (0, 1):
            _, _, rec = loop(dict(T.PARAMS, rec_back_guard=guard, rec_hold_keep=keep), frames, imu)
            r['guard%d_keep%d' % (guard, keep)] = {
                'reverse_side_s_after_hit': seg_s([1 for t, x in rec.rows if not x['hold'] and x['out'][1] > N + 40 and s_of[t] >= 3.3]),
                'reverse_side_s_total': seg_s([1 for t, x in rec.rows if not x['hold'] and x['out'][1] > N + 40]),
                'guard_events': [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard'],
                'recovery_starts': sum(1 for (_, a), (_, b) in zip(rec.rows, rec.rows[1:])
                                       if int(b['assist']) == A.ASSIST_RECOVERY and int(a['assist']) != A.ASSIST_RECOVERY)}
    res['back_guard_combo'] = r
    res['back_guard_combo_note'] = 'drive は rec_hold_keep を rec_back_guard = 1 のときだけ効かせる(guard0_keep1 = guard0_keep0)'
    # 見張りを条件にした理由(Assist の形。tests/test_rec_hold_keep.py の test_assist_keep_without_guard_pushes_rear_wall と同じ場面)
    def go(keep, guard):
        a = A.Assist((T.C, N)); t = 0.0; rev = 0
        for i in range(30):
            a.step((T.C, N - 150.0), -5000, True, t); t += T.DT
        for i in range(75):
            s = i * T.DT
            e = 0.0 if s < 0.85 else (2000.0 if s < 1.25 else 0.0)
            if 1.15 <= s < 1.25:
                a.launch_hold(); a.reset(keep_recovery=keep); t += T.DT; continue
            o, k = a.step((T.C, N + 20.0), e, True, t, impact_g=6.0 if 1.25 <= s < 1.33 else 0.0, rec_guard=guard)
            rev += int(s >= 1.25 and o[1] > N + 40)
            t += T.DT
        return seg_s([1] * rev)
    res['assist_rear_wall_after_hold'] = {'reverse_side_s_after_hit': {
        'keep0_guard0': go(False, False), 'keep0_guard1': go(False, True), 'keep1_guard0(drive では起きない)': go(True, False),
        'keep1_guard1': go(True, True)}}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
