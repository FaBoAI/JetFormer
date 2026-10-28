"""fix8 係 tools 直す 8-1 回目: わざと壊した ⑤ collect.py の写しで tests/test_fix8_tools_r1.py が落ちるかを見る(本物のファイルは
変えない。写しは ../tmp_tools_r1/mut/ に作り、最後に消す)。1 つずつ、環境変数 JF_FIX8R1_COLLECT に写しを渡して -x で流す。
00 は直す前の写し(tests/fixtures/fix8_tools_r1_before/collect.py.orig)そのもの。

  cd ~/jetformer && python3 experiments/20261026_fix/fix8/fix_tools_r1/mutate_r1.py   → out/mutate_r1_out.txt
  (子の pytest は jf-heavy --lock none --mem 1500M で流す)
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

JF = Path('/home/jetson/jetformer')
SRC = JF / 'experiments/20261026_fix/diag_launch/collect.py'
BEFORE = JF / 'tests/fixtures/fix8_tools_r1_before/collect.py.orig'
HERE = Path(__file__).resolve().parent
TMP = JF / 'experiments/20261026_fix/fix8/tmp_tools_r1/mut'
OUT = HERE / 'out' / 'mutate_r1_out.txt'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
TEST = 'tests/test_fix8_tools_r1.py'

# (名前, 壊し方の説明, [(元の文, 壊した文)])。元の文は今の collect.py にちょうど 1 回ある
M = [
    ('t1-none', '① 別の組の不具合を数えない', [("        if 'compare' not in g:\n            continue\n        us_used = g.get('launch_us_used')",
                                                    "        if True:\n            continue\n        us_used = g.get('launch_us_used')")]),
    ('t1-trial-only', '① 試しだけ数える(途中で変えた本・走り直しを数えない)',
     [("    for tag_, src_ in (('途中で変えた', mids), ('試し', trials), (RESUME_NAME, aborts)):\n        for k_, rs_ in src_.items():",
       "    for tag_, src_ in (('試し', trials),):\n        for k_, rs_ in src_.items():")]),
    ('t1-no-abort', '① 走り直しを数えない',
     [("    for tag_, src_ in (('途中で変えた', mids), ('試し', trials), (RESUME_NAME, aborts)):\n        for k_, rs_ in src_.items():",
       "    for tag_, src_ in (('途中で変えた', mids), ('試し', trials)):\n        for k_, rs_ in src_.items():")]),
    ('t1-same-form', '① 形が同じ組だけ(ゲインの違いだけ数える)',
     [("            if k2 == k or k2[:2] != k[:2]:", "            if k2 == k or k2[:2] != k[:2] or k2[3:] != k[3:]:")]),
    ('t1-same-gain', '① ゲインが同じ組だけ(形の違いだけ数える)',
     [("            if k2 == k or k2[:2] != k[:2]:", "            if k2 == k or k2[:2] != k[:2] or k2[2] != k[2]:")]),
    ('t1-any-model', '① モデルの違う組も数える', [("            if k2 == k or k2[:2] != k[:2]:", "            if k2 == k or k2[:1] != k[:1]:")]),
    ('t1-off-too', '① OFF の本の不具合も数える',
     [("            ex_on.setdefault(k_, []).extend((r, tag_) for r in rs_ if r.get('launch_state') == 'on')",
       "            ex_on.setdefault(k_, []).extend((r, tag_) for r in rs_)")]),
    ('t1-no-col', '① 衝突を数えない', [("                cl = r.get('collisions_first_lap') or []\n                if cl:",
                                        "                cl = r.get('collisions_first_lap') or []\n                if False:")]),
    ('t1-no-slip', '① 空転を数えない', [("                    if sk in ('high', 'extreme'):\n                        hi.append(name)",
                                         "                    if False:\n                        hi.append(name)")]),
    ('t1-no-unk', '① 比が出ないを数えない', [("                    elif sk == 'unknown':\n                        un.append(name)",
                                             "                    elif False:\n                        un.append(name)")]),
    ('t1-no-gap', '① 欠けを数えない', [("                if r.get('first_lap_gap'):\n                    gp.append(name)",
                                       "                if False:\n                    gp.append(name)")]),
    ('t1-no-us', '① 空転を強さで分けない', [("        us_used = g.get('launch_us_used')", "        us_used = None")]),
    ('t1-noreach-slip', '① R1 に届かない発進の空転も見る',
     [("                if r.get('to_r1_s') is not None and us_ok:", "                if us_ok:")]),
    ('t1-no-checks', '① 名前だけ出して条件に入れない', [("        for c in names:\n            chk[c] = False", "        for c in names:\n            pass")]),
    ('t1-no-line', '① 計画の組の注意の行を出さない', [("        if g.get('other_group_on'):    # 2026-10-28 fix8 tools r1", "        if False:    # 2026-10-28 fix8 tools r1")]),
    ('t1-no-src', '① 試しの組に「数えた」を付けない',
     [("            if g2 is not None and group_label(k) not in g2.get('on_faults_counted_in', []):",
       "            if False:")]),
    ('t2-thr-0.3', '② しきい 0.3 秒', [("LAUNCH_STALL_S = 0.22\n", "LAUNCH_STALL_S = 0.3\n")]),
    ('t2-thr-0.25', '② しきい 0.25 秒', [("LAUNCH_STALL_S = 0.22\n", "LAUNCH_STALL_S = 0.25\n")]),
    ('t2-thr-0.19', '② しきい 0.19 秒', [("LAUNCH_STALL_S = 0.22\n", "LAUNCH_STALL_S = 0.19\n")]),
    ('t2-hold-frames', '② 中立止めの行をフレームに数える',
     [("    tp = t[~h]\n    tp = tp[(tp > t0) & (tp < t_end)]", "    tp = t\n    tp = tp[(tp > t0) & (tp < t_end)]")]),
    ('t2-no-ends', '② 端(発進・起点)を入れない', [("    pts = np.concatenate([[t0], tp, [t_end]])", "    pts = np.concatenate([tp[:1], tp, tp[-1:]])")]),
    ('t2-past-r1', '② 起点のあとも見る(AI の区間の終わりまで)', [("            stl = launch_stall(z, i0, i1, t0, a1)", "            stl = launch_stall(z, i0, i1, t0, t1)")]),
    ('t2-no-mark', '② 行に印を付けない', [("            if stl:\n                row['launch_stall'] = stl", "            if False:\n                row['launch_stall'] = stl")]),
    ('t2-no-exclude', '② 比べから外さない',
     [("        if r.get('launch_stall') and r.get('to_r1_s') is not None:\n            # 2026-10-28 fix8",
       "        if False:\n            # 2026-10-28 fix8")]),
    ('t2-before-trial', '② 試しより先にカメラの止まりを見る',
     [("        if r.get('trial'):             # 2026-10-27 r3-r3-rec-3",
       "        if r.get('launch_stall') and r.get('to_r1_s') is not None:\n            stalls.setdefault(base(r), []).append(r)\n            continue\n"
       "        if r.get('trial'):             # 2026-10-27 r3-r3-rec-3")]),
    ('t2-on-pass', '② ON の止まりで満たさないにしない', [("            if g.get('launch_stall_on'):     # fix8 tools r1", "            if False:     # fix8 tools r1")]),
    ('t2-not-allr', '② 止まった発進を組の形・選択肢の数に入れない',
     [("for r in rs] + list(stalls.get(k, []))", "for r in rs]")]),
    ('t2-no-col', '② 止まった ON の衝突を数えない',
     [("                                              and not g.get('launch_stall_on_collisions')),\n                   # r3-r3-rec-4",
       "                                              ),\n                   # r3-r3-rec-4")]),
    ('t2-no-gap', '② 止まった ON の最初の周の欠けを数えない',
     [("        for r in stalls.get(k, []):\n            if r.get('launch_state') == 'on' and r.get('first_lap_gap'):",
       "        for r in []:\n            if r.get('launch_state') == 'on' and r.get('first_lap_gap'):")]),
    ('t2-no-slip', '② 止まった ON の空転を数えない',
     [("        for key_, src_ in (('midrun', mids), ('trial', trials), ('launch_stall', stalls)):",
       "        for key_, src_ in (('midrun', mids), ('trial', trials)):")]),
    ('t2-no-offbranch', '② OFF が全部外れたときに比べを出さない',
     [("        elif (on or g.get('launch_stall_on')) and not off and any(", "        elif False and any(")]),
    ('t2-no-line', '② 組の行を出さない', [("        if g.get('launch_stall_excluded'):     # 2026-10-28 fix8 tools r1", "        if False:     # 2026-10-28 fix8 tools r1")]),
    ('t2-no-online', '② ON の行を出さない', [("        if g.get('launch_stall_on'):\n            lines.append(", "        if False:\n            lines.append(")]),
    ('t2-no-rowtext', '② 発進の行の後ろの文を出さない',
     [("                  r['launch_stall']['gap_s'], r['launch_stall']['at_s']) if r.get('launch_stall') else ''))",
       "                  r['launch_stall']['gap_s'], r['launch_stall']['at_s']) if False else ''))")]),
    ('t3-note-none', '③ 短い止まりの印を付けない(注意の行が出ない)',
     [("                if sg:\n                    row['_launch_gap'] = sg", "                if False:\n                    row['_launch_gap'] = sg")]),
    ('t3-note-thr', '③ 注意のしきい 0.15 → 0.1 秒', [("LAUNCH_GAP_NOTE_S = 0.15\n", "LAUNCH_GAP_NOTE_S = 0.1\n")]),
    ('t3-note-line', '③ 注意の行を出さない', [("        if g.get('launch_gap_short'):\n            lines.append(", "        if False:\n            lines.append(")]),
    ('t3-note-all', '③ 比べに入らない本(試し)も注意に出す',
     [("        sh = [r for _, rs in sorted(groups.get(k, {}).items()) for r in rs if r.get('_launch_gap')]",
       "        sh = [r for _, rs in sorted(groups.get(k, {}).items()) for r in rs if r.get('_launch_gap')] + [r for r in trials.get(k, []) if r.get('_launch_gap')]")]),
    ('ctl-crit', '対照: CRIT の起点の差 0.15 → 0.10', [("CRIT = {'to_r1_gain_s': 0.15,", "CRIT = {'to_r1_gain_s': 0.10,")]),
]


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    src = SRC.read_text(encoding='utf-8')
    lines = ['# わざと壊した ⑤ collect.py の写しで %s を -x で流す(%s)' % (TEST, time.strftime('%Y-%m-%d %H:%M'))]
    cases = [('00-before', '直す前の写しそのもの', None)] + [('%02d-%s' % (i + 1, n), d, r) for i, (n, d, r) in enumerate(M)]
    n_fail = 0
    try:
        for name, desc, reps in cases:
            p = TMP / ('collect_%s.py' % name)
            if reps is None:
                shutil.copy(BEFORE, p)
            else:
                s = src
                for a, b in reps:
                    assert s.count(a) == 1, (name, a[:60], s.count(a))
                    s = s.replace(a, b)
                p.write_text(s, encoding='utf-8')
            env = dict(os.environ, JF_FIX8R1_COLLECT=str(p), PYTHONDONTWRITEBYTECODE='1')
            bt = TMP / ('bt_%s' % name)
            t0 = time.time()
            r = subprocess.run(['/home/jetson/bin/jf-heavy', '--lock', 'none', '--mem', '1500M', '--', PY, '-m', 'pytest', '-q', '-x',
                                '-p', 'no:cacheprovider', '-o', 'tmp_path_retention_policy=failed', '-W', 'ignore::DeprecationWarning',
                                '--basetemp=%s' % bt, TEST], cwd=str(JF), env=env, capture_output=True, text=True)
            out = r.stdout.splitlines()
            first = next((x for x in out if x.startswith('FAILED ') or x.startswith('ERROR ')), '')
            res = '落ちる' if r.returncode != 0 else '通る(見逃し)'
            n_fail += r.returncode != 0
            lines.append('%-22s %-40s %s %5.0f 秒  %s' % (name, desc, res, time.time() - t0, first[:200]))
            print(lines[-1], flush=True)
            shutil.rmtree(bt, ignore_errors=True)
        lines.append('落ちた %d / %d' % (n_fail, len(cases)))
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
        OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[-1:]))


if __name__ == '__main__':
    sys.exit(main())
