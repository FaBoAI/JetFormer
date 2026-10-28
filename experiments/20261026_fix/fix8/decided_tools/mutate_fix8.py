"""fix8 tools-decided: わざと壊す。本物のファイルは変えない。写し(fix8/tmp_tools_decided/mut/)に 1 か所ずつ壊しを入れ、環境変数
JF_FIX8_COLLECT・JF_FIX8_S3・JF_FIX8_QP・JF_FIX8_RUNS・JF_FIX8_DISK で tests/test_fix8_tools_decided.py を -x で流す(中の pytest は
1 つずつ jf-heavy)。結果は out/mutate_fix8_out.txt。写しと basetemp は最後に消す

  python3 mutate_fix8.py            # 全部
  python3 mutate_fix8.py 0 10 a     # 0〜9 番だけ(写しの場所と出力を分ける)
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
TMP = JF / 'experiments/20261026_fix/fix8/tmp_tools_decided/mut'
OUT = Path(__file__).resolve().parent / 'out' / 'mutate_fix8_out.txt'
SRC = {'S': JF / 'tools/standing_3lap.py', 'C': JF / 'experiments/20261026_fix/diag_launch/collect.py',
       'Q': JF / 'tools/quick_eval_by_params.py', 'R': JF / 'experiments/20261026_fix/plan/runs_1103.py',
       'D': JF / 'experiments/20261026_fix/fix3/disk/disk_check.py'}
ENV = {'S': 'JF_FIX8_S3', 'C': 'JF_FIX8_COLLECT', 'Q': 'JF_FIX8_QP', 'R': 'JF_FIX8_RUNS', 'D': 'JF_FIX8_DISK'}
M = [
    # ---- ① ③ standing_3lap.abort_resumes(⑤ も同じ関数を読む)
    ('S', '① ③ リカバリーの途中の形を見ない', "            elif has_rec and len(k) and int(a[k[-1]]) == ASSIST_RECOVERY:",
     "            elif False:"),
    ('S', '① ③ 区間の最後のフレームを中立止め・人側を除かずに見る', "            elif has_rec and len(k) and int(a[k[-1]]) == ASSIST_RECOVERY:",
     "            elif has_rec and int(a[i1]) == ASSIST_RECOVERY:"),
    ('S', '① ③ 区間の中にリカバリーがあれば印(最後でなくても)', "            elif has_rec and len(k) and int(a[k[-1]]) == ASSIST_RECOVERY:",
     "            elif has_rec and len(k) and bool((a[k] == ASSIST_RECOVERY).any()):"),
    ('S', '① ③ リカバリーを中立止め・途切れより先に見る(fix7 の印が変わる)',
     "            se = stop_end(i0, i1)        # fix7 r2: 中立止め・カメラの途切れのまま止まって終わった区間",
     "            se = None if (has_rec and len(k) and int(a[k[-1]]) == ASSIST_RECOVERY) else stop_end(i0, i1)"),
    ('S', '① ③ abort_s を区間の最初のリカバリーのフレームに', "                t_r = float(t[k[j]])",
     "                t_r = float(t[k[int(np.nonzero(a[k] == ASSIST_RECOVERY)[0][0])]])"),
    ('S', '① ③ rec_s を最後のリカバリーのフレームまで', "'rec_s': _r(float(t[i1]) - t_r, 2),", "'rec_s': _r(float(t[k[-1]]) - t_r, 2),"),
    ('S', '① ③ verpm の無い呼び方ではリカバリーを見ない(早い戻り)', "    if not has_abort and not has_rec and not can_stop:",
     "    if not has_abort and not can_stop:"),
    ('S', '① ③ 走り直しの区間が短くても続けない(carried)', "                pend = {'kind': 'recovery', 'abort_s': _r(t_r, 3), 'rec_s': _r(float(t[i1]) - t_r, 2),\n                        'ai_end_s': _r(float(t[i1]), 3), 'carried': 0}",
     "                pend = {'kind': 'recovery', 'abort_s': _r(t_r, 3), 'rec_s': _r(float(t[i1]) - t_r, 2),\n                        'ai_end_s': _r(float(t[i1]), 3), 'carried': -100}"),
    ('S', '① ③ リカバリーの印でも counted を False にしない', "                s['after_abort'] = v\n                s['counted'] = False",
     "                s['after_abort'] = v\n                s['counted'] = s['counted'] if v.get('kind') == 'recovery' else False"),
    ('S', '① ③ 注 RECOVERY_NOTE を出さない', "        out.append(RECOVERY_NOTE)", "        pass"),
    ('S', '① ③ 注 RECOVERY_NOTE をいつも出す', "    if any((s.get('after_abort') or {}).get('kind') == 'recovery' for r in results for s in r.get('standing') or []):\n        out.append(RECOVERY_NOTE)",
     "    if True:\n        out.append(RECOVERY_NOTE)"),
    ('S', '① ③ 表の形の文: リカバリーを中止の文で', "    if kind == 'recovery':\n        return '%s %s 秒から %s 秒' % (STOP_KIND_TEXT['recovery']",
     "    if kind == 'xrecovery':\n        return '%s %s 秒から %s 秒' % (STOP_KIND_TEXT['recovery']"),
    ('S', '① ③ 形の文の頭を変える', "STOP_KIND_TEXT['recovery'] = 'リカバリーの途中'", "STOP_KIND_TEXT['recovery'] = 'リカバリー'"),
    # ---- ① ⑤ collect
    ('C', '① ⑤ 組の行の説明にリカバリーの 1 文を足さない', "                    if any('・%s ' % RECOVERY_TEXT in x for x in g['abort_resume_excluded']) else '')",
     "                    if False else '')"),
    ('C', '① ⑤ 組の行の説明にリカバリーの 1 文をいつも足す', "                    if any('・%s ' % RECOVERY_TEXT in x for x in g['abort_resume_excluded']) else '')",
     "                    if True else '')"),
    ('C', '① ⑤ 形の文: リカバリーを中止の文で', "    if kind == 'recovery':\n        return '%s %s 秒から %s 秒' % (RECOVERY_TEXT",
     "    if kind == 'xrecovery':\n        return '%s %s 秒から %s 秒' % (RECOVERY_TEXT"),
    # ---- ② collect
    ('C', '② 途中で変えた・試しの本の空転を見ない', "        for key_, src_ in (('midrun', mids), ('trial', trials)):\n            if k not in src_:\n                continue\n            mt_on",
     "        for key_, src_ in ():\n            if k not in src_:\n                continue\n            mt_on"),
    ('C', '② 試しの本の空転を見ない(途中で変えた本だけ)', "        for key_, src_ in (('midrun', mids), ('trial', trials)):\n            if k not in src_:",
     "        for key_, src_ in (('midrun', mids),):\n            if k not in src_:"),
    ('C', '② 空転を checks の slip に入れない', "                            and not g.get('midrun_slip') and not g.get('trial_slip')),",
     "                            ),"),
    ('C', '② 比が出ないを slip_all_measured に入れない', "or g.get('midrun_slip_unmeasured')\n                    or g.get('trial_slip_unmeasured')):",
     "):"),
    ('C', '② いちばんひどい空転を数えない', "            mt_hi = [r for r in mt_on if slip_state(r.get('slip')) in ('high', 'extreme')]",
     "            mt_hi = [r for r in mt_on if slip_state(r.get('slip')) in ('high',)]"),
    ('C', '② OFF の本の空転も数える', "            mt_on = [r for r in src_[k] if r.get('launch_state') == 'on' and r.get('to_r1_s') is not None]",
     "            mt_on = [r for r in src_[k] if r.get('to_r1_s') is not None]"),
    ('C', '② R1 に届かなかった ON(空転の比を出さない)も数える', "            mt_on = [r for r in src_[k] if r.get('launch_state') == 'on' and r.get('to_r1_s') is not None]",
     "            mt_on = [r for r in src_[k] if r.get('launch_state') == 'on']"),
    ('C', '② launch_us が混ざっても全部の強さの空転を数える', "            if g.get('launch_us_used') is not None:\n                mt_on = [r for r in mt_on",
     "            if False:\n                mt_on = [r for r in mt_on"),
    ('C', '② 最初の周の欠けを数えない', "                if r.get('launch_state') == 'on' and r.get('first_lap_gap'):\n                    gap_on.append(r)",
     "                if False:\n                    gap_on.append(r)"),
    ('C', '② 欠けた発進の名前の後ろに「。試し」「。途中で変えた」を付けない', "'。' + gap_tag[id(r)] if id(r) in gap_tag",
     "'' if id(r) in gap_tag"),
    ('C', '② 空転の行を出さない', "            if g.get('%s_slip' % key_):\n                lines.append", "            if False:\n                lines.append"),
    ('C', '② 比が出ないの行を出さない', "            if g.get('%s_slip_unmeasured' % key_):\n                lines.append", "            if False:\n                lines.append"),
    # ---- ③ 文
    ('C', '③ ⑤ 混ざった注意の行の後ろの文を足さない', "                         '(3 章で止めて OFF にした本 X から先の OFF なら起こし直しではない = 数える。手順書 3・7 章)'\n", ""),
    ('R', '③ ② 印の後ろの文を前の文に戻す', "OPT_MARK_NOTE = '(再起動で OFF に戻った? 3 章で止めて OFF にした本 X から先なら数える。手順書 3・7 章)'",
     "OPT_MARK_NOTE = '(再起動で OFF に戻った? 手順書 7 章)'"),
    ('R', '③ ② 本「—」から始まった区切りに回の本の名前を足さない',
     "            dash[1].append('回の本は %s(No.%s)から' % (r['run_id'], r.get('ai_no') or '—'))", "            pass"),
    ('R', '③ ② 回の本から始まった区切りにも回の本の名前を足す', "            dash = segs[-1] if not r.get('run_id') else None",
     "            dash = segs[-1]"),
    ('R', '③ ② options_line で本「—」を区切りから外す', "    for r in rows:\n        o = opt_on(r)\n        if o != prev:",
     "    for r in [x for x in rows if x.get('run_id')]:\n        o = opt_on(r)\n        if o != prev:"),
    # ---- ④ 文
    ('Q', '① ④ 2 章の下の行にリカバリーの 1 文を足さない', "                   if any((s.get('after_abort') or {}).get('kind') == 'recovery'",
     "                   if False and any((s.get('after_abort') or {}).get('kind') == 'recovery'"),
    ('Q', '① ④ 2 章の下の行にリカバリーの 1 文をいつも足す', "                   if any((s.get('after_abort') or {}).get('kind') == 'recovery'\n                          for r in results for s in r.get('standing') or []) else '')]",
     "                   if True else '')]"),
    # ---- ④ disk_check
    ('D', '④ PYTEST_GB を 3.9 に戻す', "PYTEST_GB = 0.6\n", "PYTEST_GB = 3.9\n"),
    ('D', '④ PYTEST_GB_KEEP を 3.9 に', "PYTEST_GB_KEEP = 7.1", "PYTEST_GB_KEEP = 3.9"),
    ('D', '④ 行に付けないときの数を出さない', "'%.2f GB)。終わって消せば %.2f GB に戻る' % (after1 - PYTEST_GB, PYTEST_GB, PYTEST_GB_KEEP, after1 - PYTEST_GB_KEEP, after1))",
     "'%.2f GB)。終わって消せば %.2f GB に戻る' % (after1 - PYTEST_GB, PYTEST_GB, PYTEST_GB_KEEP, after1 - PYTEST_GB, after1))"),
    ('D', '④ 説明の文からオプションの名前を消す', "。-o tmp_path_retention_policy=failed(通った", "。(通った"),
    # ---- 対照
    ('C', '対照: CRIT の起点の差 0.15 → 0.10', "CRIT = {'to_r1_gain_s': 0.15,", "CRIT = {'to_r1_gain_s': 0.10,"),
]


def main(lo=0, hi=None, tag=''):
    global TMP, OUT
    if tag:
        TMP = TMP.parent / ('mut_' + tag)
        OUT = OUT.parent / ('mutate_fix8_out_%s.txt' % tag)
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)
    lines = []
    try:
        for n, (f, name, a, b) in enumerate(M):
            if n < lo or (hi is not None and n >= hi):
                continue
            src = SRC[f].read_text(encoding='utf-8')
            assert src.count(a) == 1, (n, name, src.count(a))
            p = TMP / ('%02d_%s' % (n, SRC[f].name))
            p.write_text(src.replace(a, b), encoding='utf-8')
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', **{ENV[f]: str(p)})
            bt = TMP / ('bt%02d' % n)
            cmd = ['/home/jetson/bin/jf-heavy', '--lock', 'none', '--mem', '1500M', '--',
                   '/home/jetson/miniconda/envs/jetracer/bin/python', '-m', 'pytest', '-x', '-q', '-p', 'no:cacheprovider',
                   '-o', 'tmp_path_retention_policy=failed', '-W', 'ignore::DeprecationWarning', '--basetemp=%s' % bt,
                   'tests/test_fix8_tools_decided.py']
            r = subprocess.run(cmd, cwd=str(JF), env=env, capture_output=True, text=True, timeout=1800)
            fl = [x.split(' - ')[0].replace('FAILED tests/test_fix8_tools_decided.py::', '') for x in r.stdout.splitlines()
                  if x.startswith('FAILED') or x.startswith('ERROR')]
            res = '落ちる' if r.returncode != 0 else '通る(見逃し)'
            lines.append('| %02d | %s | %s | %s | %s |' % (n, f, name, res, fl[0] if fl else '—'))
            print(lines[-1], flush=True)
            shutil.rmtree(bt, ignore_errors=True)
            p.unlink()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    OUT.write_text('\n'.join(['| # | ファイル | 壊し | 結果 | 落ちた最初の試験 |', '|---|---|---|---|---|'] + lines) + '\n',
                   encoding='utf-8')
    print('落ちる %d / %d' % (sum('落ちる' in x for x in lines), len(lines)))
    return 0 if all('落ちる' in x for x in lines) else 1


if __name__ == '__main__':
    av = sys.argv[1:]
    sys.exit(main(int(av[0]), int(av[1]), av[2]) if len(av) == 3 else main())
