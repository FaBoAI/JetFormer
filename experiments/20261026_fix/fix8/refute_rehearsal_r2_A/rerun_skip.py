"""打ち消しの票 A(反証 8-2 rehearsal-r2-unmarked-rerun-skip-drops-hit)。指摘の build_fake.py・gen_course_r8.py は使わない。
作り物の記録は試験の道具 tests/test_fix7_tools.write_log(本物の記録と同じ npz の形。距離で決めたコース)で自分で作り、本物の
standing_3lap.abort_resumes・analyze_log と本物の diag_launch/collect.py の main(RAW・LAB・BENCH・OUT だけ作り物へ)を流す。

回 3(block 12〜19 = 3-1〜3-8。OFF/ON の交互。test_fix7_tools.round3 の 'N' と同じ 8 本)の 3-2(ON)を次の形に替える:
  rear      1 周目の SW のあと 17.5 m でモデルのブレーキで止まり(10/23 の形。assist 0)、止まって 0.5 秒で人側(リカバリーの 0.8 秒の前)
            → 6 秒(HS に運ぶ。車輪の距離は進まない)→ AI 側(発進の前進がもう一度)→ 3 周
  rear_hit  rear の走り直しの最初の周の LH(距離 21.5 m)で水平 4.5G
  late_hit  rear_hit で人側が遅れ、止まって 0.8 秒からリカバリー(assist 2)が 0.15 秒出たあとに人側(印が出る形)
  base      3-2 もふつうの 1 本(比べの元)
それぞれ ⑤ を --ignore-gain(手順書 8.1)と、紙の「走り直し: 3-2」から --skip 記録名@AI側にした秒 を足した形で流す。
使い方: python -B rerun_skip.py <作業の場所>"""
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF))
import test_fix7_tools as t7  # noqa: E402  write_log・launch_keys・time_at_distance・PRM・ON・DESC・Day
import standing_3lap as s3  # noqa: E402

COLLECT = JF / 'experiments/20261026_fix/diag_launch/collect.py'


def load_collect(tag):
    loader = importlib.machinery.SourceFileLoader('collect_A_' + tag, str(COLLECT))
    spec = importlib.util.spec_from_file_location('collect_A_' + tag, str(COLLECT), loader=loader)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rear_book(D, block, prm, hit=False, late=False):
    base = t7.launch_keys(1.0, 0.15, 0.3)
    t_b = t7.time_at_distance(base + [(200.0, 2.0)], 17.1, 200.0)        # ここからモデルのブレーキ 0.4 秒(平均 1 m/s で 0.4 m)
    t_stop = t_b + 0.4
    t_h = t_stop + (0.95 if late else 0.5)                                # 人側
    t_res = round(t_h + 6.0, 2)                                           # HS に運んで AI 側
    dur = t_res + 32.0
    vk = base + [(t_b, 2.0), (t_stop, 0.0), (t_res + 0.15, 0.0), (t_res + 0.45, 2.0), (dur, 2.0)]
    s_stop = None
    hits = []
    if hit:
        hits = [(t7.time_at_distance(vk, 21.5, dur), 4.5)]
    log = D.add(block, prm, 0.15, 0.3, dur=dur, vk=vk, ai=[(1.0, t_h), (t_res, dur - 1.0)],
                launch=[(1.0, 0.5), (t_res, 0.5)], hits=hits)
    tg = np.arange(0.0, dur, 0.001)
    vg = np.interp(tg, [x[0] for x in vk], [x[1] for x in vk])
    sg = np.concatenate([[0.0], np.cumsum(0.5 * (vg[1:] + vg[:-1]) * np.diff(tg))])
    s_stop = float(np.interp(t_stop, tg, sg))
    rec = None
    if late:                       # 止まって 0.8 秒からリカバリーの 'brake'(assist 2)を 0.15 秒(人側にするまで)
        rec = (t_stop + 0.8, t_h)
        for sc in sorted((D.raw / log).iterdir()):
            p = sc / 'AI_CMD.npz'
            if not p.exists():
                continue
            z = dict(np.load(p))
            ts = (z['timestamps_us'].astype(np.int64) - t7.T0_US) / 1e6
            m = (ts >= rec[0] - 1e-9) & (ts < rec[1] - 1e-9) & z['select']
            if m.any():
                z['assist'][m] = 2
                np.savez(p, **z)
    return log, {'t_stop': round(t_stop, 3), 's_stop_m': round(s_stop, 2), 't_human': round(t_h, 3), 't_res': t_res,
                 'hit': [(round(a, 2), g) for a, g in hits], 'assist2': rec and [round(rec[0], 3), round(rec[1], 3)]}


def day(root, kind):
    D = t7.Day(root / kind)
    D.add(12, t7.PRM, 0.30, 0.6)          # OFF は起点まで約 3.1 秒(ON の約 2.8〜2.9 秒より 0.15 秒以上遅い = 当たりが無ければ「使ってよい」)
    info = None
    if kind == 'base':
        log = D.add(13, t7.ON, 0.15, 0.3)
    else:
        log, info = rear_book(D, 13, t7.ON, hit=kind in ('rear_hit', 'late_hit'), late=kind == 'late_hit')
    D.add(14, t7.PRM, 0.31, 0.6)
    D.add(15, t7.ON, 0.20, 0.3)
    D.add(16, t7.PRM, 0.32, 0.6)
    D.add(17, t7.ON, 0.28, 0.3)
    D.add(18, t7.PRM, 0.33, 0.6)
    D.add(19, t7.ON, 0.31, 0.3)
    return D, log, info


def run5(D, tag, args):
    m = load_collect(tag)
    m.RAW, m.LAB, m.BENCH, m.OUT = D.raw, D.tmp / 'nolab', D.bench, D.tmp / 'out'
    out = D.tmp / ('c5_%s.json' % tag)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.main(['--day', '20261103', '--out', str(out)] + list(args))
    return json.loads(out.read_text(encoding='utf-8')), buf.getvalue()


def verdicts(j):
    res = []
    for g in (j.get('summary') or []):
        c = g.get('compare') or {}
        if not c:
            continue
        bl = g.get('by_launch') or {}
        res.append({'session': g.get('session'), 'all_ok': c.get('all_ok'),
                    'false_checks': [k for k, v in (c.get('checks') or {}).items() if v is False],
                    'd_to_r1_s': c.get('d_to_r1_s'),
                    'on': {k: (bl.get('on') or {}).get(k) for k in ('n', 'restarts', 'collisions_to_r1', 'collisions_first_lap', 'logs')},
                    'off_n': (bl.get('off') or {}).get('n'),
                    'abort_resume_excluded': g.get('abort_resume_excluded')})
    return res


def main():
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    report = {}
    for kind in ('base', 'rear', 'rear_hit', 'late_hit'):
        D, log, info = day(root, kind)
        # ③ の印(本物の abort_resumes・analyze_log)
        r3 = s3.analyze_log(log, raw_dir=D.raw, labels_dir=D.tmp / 'nolab3')
        st = [{'t_ai': s.get('t_ai'), 'start_no': s.get('start_no'), 'after_abort': s.get('after_abort'),
               'counted': s3.counted(s), 'note': s.get('note') or s.get('kind') or s.get('why')}
              for s in (r3.get('standing') or [])]
        rep = {'log': log, 'info': info, 'standing_3lap': st}
        j, out = run5(D, kind + '_ig', ['--ignore-gain'])
        rep['c5_ignore_gain'] = verdicts(j)
        rep['c5_lines'] = [x for x in out.splitlines() if ('ON − OFF' in x or '使ってよい' in x or '満たさない' in x
                                                            or '走り直し' in x or '衝突' in x or 'skip' in x)][:30]
        rw = [r for r in j['launches'] if r['log'] == log]
        rep['c5_rows_of_book'] = [{'t_ai': r.get('t_ai'), 'start_no': r.get('start_no'), 'launch_state': r.get('launch_state'),
                                   'after_abort': r.get('after_abort'), 'to_r1_s': r.get('to_r1_s'),
                                   'collisions_first_lap': r.get('collisions_first_lap')} for r in rw]
        if info:
            # 紙の「走り直し: 3-2」の発進 = この本の 2 回目の発進(③⑤ の表に出る AI 側にした秒。30 fps のフレームの時刻)
            sk = '%s@%.2f' % (log, [r['t_ai'] for r in rw if (r.get('start_no') or 1) == 2][0])
            j2, out2 = run5(D, kind + '_skip', ['--ignore-gain', '--skip', sk])
            rep['skip_arg'] = sk
            rep['c5_ignore_gain_skip'] = verdicts(j2)
            rep['c5_skip_lines'] = [x for x in out2.splitlines() if ('ON − OFF' in x or '使ってよい' in x or '満たさない' in x
                                                                     or 'まとめから外した' in x or '衝突' in x)][:30]
        report[kind] = rep
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
