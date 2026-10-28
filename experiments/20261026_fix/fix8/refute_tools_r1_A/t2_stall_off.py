"""指摘 2(w1-tools-r8-1-stall-inflates-off)を打ち消すつもりで確かめる。

回 3 の作り物(自分で決めた値): OFF 3-1・3-3・3-5・3-7(d_move 0.20・0.21・0.22・0.40、ramp 0.6)、ON 3-2・3-4・3-6・3-8
(d_move 0.15・0.20・0.28・0.31、ramp 0.3)。止まりが無ければ ON − OFF ≈ −0.125 秒「満たさない」になる形。
3-5(発進の前進 OFF)の発進の途中で、カメラが 0.9 秒止まる(AI_CMD・カメラのフレームが無い = drive の中立止めでフレームを書かない形)。
  none    止まり無し(対照)
  B_off   3-5 から先は試験の選択肢を OFF(手順書 3 章のわざと OFF)。発進 0.6 秒あと(1.27 m/s)から 0.9 秒止まり、中立の惰性で
          止まる → 明けてモデルが走り直して R1 に遅れて届く(OFF は中止しない = decided_safety 1.2 の「OFF は走り続ける」)
  A_on    両方 ON のまま。同じ止まりで止まる → 明けたフレームで中止(assist 3・出来事 hold_long)→ 2.5 秒後に人側(R1 に届かない)
  C_on    両方 ON のまま。止まりが発進の 1.0 秒あと(2.0 m/s)からで、0.65 m/s まで落ちて止まらない(中止しない)→ 遅れて届く
  E_off   3-6(発進の前進 ON)で B_off と同じ形(向きの確かめ)
⑤ --ignore-gain の ON − OFF の行・3-5 の行の起点まで・3-5 に印や注意が出るか。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path('/home/jetson/jetformer/experiments/20261026_fix/fix8/refute_tools_r1_A')))
import rcommon as rc  # noqa: E402

P = dict(rc.BASE)
ON = dict(P, launch_boost=1)
OPT_OFF = {'rec_back_guard': 0, 'rec_hold_keep': 0}


def stall_keys(d_move, ramp, t_s, t_e, v_s, v_e, t_go, until=30.0, stop_at=None):
    """発進(1.0 秒)→ 動き出し 1.0 + d_move → ramp で 2.0 m/s(途中で止まりが始まる)→ 止まりの間 v_s → v_e(stop_at で 0)
    → t_go から 0.6 秒で 2.0 m/s"""
    t_mv = 1.0 + d_move
    k = [(0.0, 0.0), (t_mv, 0.0)]
    if t_s < t_mv + ramp:
        k.append((t_s, v_s))
    else:
        k += [(t_mv + ramp, 2.0), (t_s, 2.0)]
    if stop_at is not None:
        k += [(stop_at, 0.0), (t_go, 0.0)]
    else:
        k += [(t_e, v_e)]
    k += [(t_go + 0.6, 2.0), (until, 2.0)]
    return k


def build(case):
    D = rc.MyDay('t2_%s' % case)
    logs = {}
    plan = ((12, P, 0.20, 0.6), (13, ON, 0.15, 0.3), (14, P, 0.21, 0.6), (15, ON, 0.20, 0.3),
            (16, P, 0.22, 0.6), (17, ON, 0.28, 0.3), (18, P, 0.40, 0.6), (19, ON, 0.31, 0.3))
    for blk, prm, dm, ramp in plan:
        prm = dict(prm)
        if case in ('B_off', 'E_off') and blk >= 16:
            prm.update(OPT_OFF)
        target = (blk == 16 and case in ('B_off', 'A_on', 'C_on')) or (blk == 17 and case == 'E_off')
        if not target:
            logs[blk] = D.add(blk, prm, dm, ramp)
            continue
        if case in ('B_off', 'E_off'):
            # 0.6 秒あとから 0.9 秒の止まり。その時の速さ(ramp の途中)から中立の惰性 1.5 m/s² で止まる
            t_mv = 1.0 + dm
            t_s = 1.6
            v_s = 2.0 * min(1.0, (t_s - t_mv) / ramp)
            stop_at = t_s + v_s / 1.5
            vk = stall_keys(dm, ramp, t_s, 2.5, v_s, 0.0, 2.55, stop_at=min(stop_at, 2.5))
            logs[blk] = D.add(blk, prm, vk=vk, stall=[(1.6, 2.5)], launch=[(1.0, 0.5)] if prm.get('launch_boost') else [])
        elif case == 'A_on':
            t_mv = 1.0 + dm
            v_s = 2.0 * min(1.0, (1.6 - t_mv) / ramp)
            stop_at = min(1.6 + v_s / 1.5, 2.5)
            vk = [(0.0, 0.0), (t_mv, 0.0), (1.6, v_s), (stop_at, 0.0), (30.0, 0.0)]
            ev = [{'kind': 'recovery_guard', 'what': 'hold_long', 'phase': None, 'backed_s': 0.0, 'hold_s': 0.9, 'run': 'runB',
                   'cam_epoch_us': int(rc.T0_US + 2.5e6), 't_us': int(rc.T0_US + 2.5e6)}]
            logs[blk] = D.add(blk, prm, vk=vk, ai=[(1.0, 5.0)], stall=[(1.6, 2.5)], giveup=[(2.5, 5.0)], events=ev)
        elif case == 'C_on':
            vk = stall_keys(dm, ramp, 2.0, 2.9, 2.0, 0.65, 2.9)
            logs[blk] = D.add(blk, prm, vk=vk, stall=[(2.0, 2.9)])
    return D, logs


def main():
    res, lines = [], []
    for case in ('none', 'B_off', 'A_on', 'C_on', 'E_off'):
        D, logs = build(case)
        j, out = rc.run_collect(D, ['--ignore-gain'], tag='t2_%s' % case)
        g = [x for x in rc.summary_of(j) if x.get('compare')]
        c = g[0]['compare'] if g else {}
        tgt = logs[17] if case == 'E_off' else logs[16]
        r = [x for x in j['launches'] if x['log'] == tgt]
        rec = {'case': case, 'd_to_r1': c.get('d_to_r1_s'), 'all_ok': c.get('all_ok'),
               'checks_false': {k: v for k, v in (c.get('checks') or {}).items() if v is not True},
               'off_median': ((g[0]['by_launch'].get('off') or {}).get('to_r1_median_s') if g else None),
               'on_median': ((g[0]['by_launch'].get('on') or {}).get('to_r1_median_s') if g else None),
               'off_n': ((g[0]['by_launch'].get('off') or {}).get('n') if g else None),
               'not_reached': sorted((g[0].get('not_reached') or {}).keys()) if g else None,
               'target_rows': [{k: x.get(k) for k in ('t_ai', 'to_r1_s', 'why_no_r1', 'after_abort', 'launch_state', 'move_0p2_s',
                                                     'first_lap_gap', 'midrun', 'trial')} for x in r],
               'mixed_options': (g[0].get('mixed_options') if g else None)}
        # 出力に「止まり」「途切れ」の言葉が出るか
        rec['stall_words_in_stdout'] = [x for x in out.splitlines() if ('途切れ' in x or '止まり' in x or 'カメラ' in x)]
        res.append(rec)
        lines.append('=== %s' % case)
        lines += ['  ' + x for x in out.splitlines() if 'ON − OFF' in x or x.startswith('  off') or x.startswith('  on ')
                  or 'R1 に届かなかった' in x or '注意' in x]
    (rc.HERE / 'out').mkdir(exist_ok=True)
    (rc.HERE / 'out' / 't2_stall_off.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    (rc.HERE / 'out' / 't2_stall_off.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for r in res:
        print('%-6s ON %s・OFF %s(n %s) d %s → %s  %s  届かない %s  対象の行 %s  止まりの言葉 %d' % (
            r['case'], r['on_median'], r['off_median'], r['off_n'], r['d_to_r1'],
            {True: '使ってよい', False: '満たさない', None: '—'}[r['all_ok']], r['checks_false'], r['not_reached'],
            [(x['t_ai'], x['to_r1_s'], x['why_no_r1']) for x in r['target_rows']], len(r['stall_words_in_stdout'])))


if __name__ == '__main__':
    main()
