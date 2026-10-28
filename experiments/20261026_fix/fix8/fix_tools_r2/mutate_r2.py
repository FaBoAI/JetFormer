"""fix8 係 tools 直す 8-2 回目: わざと壊した ⑤ で新しい試験(tests/test_fix8_tools_r2.py)が落ちるかを見る。本物の collect.py は変えない
(fix8/tmp_tools_r2/mut/ に 1 か所ずつ壊した写しを作り、JF_FIX8R2_COLLECT で差し替えて pytest -x。写しと basetemp は最後に消す)。

    cd ~/jetformer && python3 experiments/20261026_fix/fix8/fix_tools_r2/mutate_r2.py
  → out/mutate_r2_out.txt(子の pytest は jf-heavy --lock none --mem 1500M)
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
SRC = JF / 'experiments/20261026_fix/diag_launch/collect.py'
BEFORE = JF / 'tests/fixtures/fix8_tools_r2_before/collect.py.orig'
TMP = JF / 'experiments/20261026_fix/fix8/tmp_tools_r2/mut'
OUT = Path(__file__).resolve().parent / 'out'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'

M = [
    # ---- w2-tools-r8-2-off-few-left
    ('01-f1-main-none', '① 比べのある組で off_all_compared を付けない',
     [("            if off_drop:               # fix8 tools r2: 比べから外れた OFF がある = 満たさない(残りの OFF の中央値は、どの本が外れたかで動く)\n"
       "                chk['off_all_compared'] = False\n",
       "            if False:\n                chk['off_all_compared'] = False\n")]),
    ('02-f1-no-stall', '① カメラの止まりの OFF を数えない(R1 に届かないだけ)',
     [("             for r in stalls.get(k, []) if r.get('launch_state', 'off') == 'off']",
       "             for r in stalls.get(k, []) if False]")]),
    ('03-f1-no-noreach', '① R1 に届かなかった OFF の本を数えない(カメラの止まりだけ)',
     [("               for r in nreach.get(k, {}).get('off', []) if unit_(r) not in off_units])]",
       "               for r in nreach.get(k, {}).get('off', []) if False])]")]),
    ('04-f1-noreach-per-launch', '① R1 に届かなかった OFF を発進ごとに数える(本が比べに残っても)',
     [("               for r in nreach.get(k, {}).get('off', []) if unit_(r) not in off_units])]",
       "               for r in nreach.get(k, {}).get('off', [])])]")]),
    ('05-f1-two-or-more', '① 外れた OFF が 2 本以上のときだけ(反証の係の最初の案「3 本未満」の形)',
     [("            if off_drop:               # fix8 tools r2: 比べから外れた OFF がある = 満たさない(残りの OFF の中央値は、どの本が外れたかで動く)\n",
       "            if len(off_drop) >= 2:\n")]),
    ('06-f1-branch3-old', '① OFF が全部 R1 に届かない日に比べを出さない(前の条件)',
     [("        elif (on or g.get('launch_stall_on')) and not off and off_drop:",
       "        elif (on or g.get('launch_stall_on')) and not off and any(r.get('launch_state', 'off') == 'off' for r in stalls.get(k, [])):")]),
    ('07-f1-branch3-offnostall', '① OFF が全部 R1 に届かない日にも off_no_stall を付ける',
     [("            if any(r.get('launch_state', 'off') == 'off' for r in stalls.get(k, [])):\n                ck['off_no_stall'] = False\n",
       "            ck['off_no_stall'] = False\n")]),
    ('08-f1-branch2-none', '① 比べられる ON が無い組で off_all_compared を付けない',
     [("            if off_drop:               # fix8 tools r2(あるときだけ)\n                ck['off_all_compared'] = False\n",
       "            if False:\n                ck['off_all_compared'] = False\n")]),
    ('09-f1-no-line', '① 「→ OFF の発進が比べから外れた N 本」の行を出さない',
     [("        if g.get('off_dropped'):               # 2026-10-28 fix8 tools r2(あるときだけ)\n",
       "        if False:\n")]),
    ('10-f1-trial-too', '① 試しの OFF の本も外れた OFF に数える',
     [("               for r in nreach.get(k, {}).get('off', []) if unit_(r) not in off_units])]",
       "               for r in nreach.get(k, {}).get('off', []) if unit_(r) not in off_units]\n"
       "            + [(r['log'], r['t_ai'], '試し') for r in trials.get(k, []) if r.get('launch_state', 'off') == 'off'])]")]),
    ('11-f1-first-only-blind', '① 本ごとの決めを全部の行(--first-only の前)で見る(比べに残った本の名前を発進の数で)',
     [("        off_units = {unit_(r) for r in groups.get(k, {}).get('off', [])}",
       "        off_units = {unit_(r) for r in rows if r.get('to_r1_s') is not None}")]),
    ('11b-f1-unit-log', '① 本を記録だけで見る(同じ block の別の記録のやり直しも外れた OFF に数える)',
     [("            return ('block', r['_block']) if r.get('_block') else ('log', r['log'])",
       "            return ('log', r['log'])")]),
    # ---- w2-tools-r8-2-short-gaps-add-up
    ('12-f2-thr-0.3', '② 中立の合計のしきい 0.3 秒',
     [('LAUNCH_STALL_SUM_S = 0.22\n', 'LAUNCH_STALL_SUM_S = 0.3\n')]),
    ('13-f2-thr-0.12', '② 中立の合計のしきい 0.12 秒(本物の 3 秒の窓の最大 0.200 より下)',
     [('LAUNCH_STALL_SUM_S = 0.22\n', 'LAUNCH_STALL_SUM_S = 0.12\n')]),
    ('14-f2-no-hold', '② 中立止めの行をフレームの無い間と同じに数える(行から中立にしない)',
     [("        s = np.where((kk < len(th)) & (first < pts[1:]), np.minimum(s, first), s)",
       "        s = s")]),
    ('15-f2-after-1frame', '② 中立の始めを前のフレームの 1/30 秒あとにする(0.1 秒でなく)',
     [('NEUTRAL_AFTER_S = 0.1 ', 'NEUTRAL_AFTER_S = 1 / 30.0 ')]),
    ('16-f2-no-exclude', '② 中立の合計で外さない',
     [("            elif nzl['neutral_s'] > LAUNCH_STALL_SUM_S:", "            elif False:")]),
    ('17-f2-past-r1', '② 中立の合計を AI の区間の終わりまで見る(R1 の起点のあとも)',
     [("            nzl = None if stl else launch_neutral(z, i0, i1, t0, a1)",
       "            nzl = None if stl else launch_neutral(z, i0, i1, t0, t1)")]),
    ('18-f2-no-ends', '② 端(発進・起点)を点に入れない',
     [("    pts = np.concatenate([[t0], tp, [t_end]])\n    th = t[h]",
       "    pts = tp if len(tp) >= 2 else np.concatenate([[t0], tp, [t_end]])\n    th = t[h]")]),
    ('19-f2-old-text', '② 短い止まりが続いた発進の名前の文を 1 回の止まりの文にする',
     [("    if 'neutral_s' in st:\n        return '短い止まりが続いた", "    if False:\n        return '短い止まりが続いた")]),
    ('20-f2-no-sum-line', '② 「→ うち短い止まりが続いた発進 N 本」の行を出さない',
     [("        if g.get('launch_stall_sum'):          # 2026-10-28 fix8 tools r2(あるときだけ)\n", "        if False:\n")]),
    ('21-f2-no-row-text', '② 発進の行の後ろの「カメラの止まりが続いた(…)」を出さない',
     [("                 if r.get('launch_stall') and 'neutral_s' in r['launch_stall'] else ''))",
       "                 if False else ''))")]),
    ('22-f2-note-n1', '② 注意を中立 1 回の発進にも出す',
     [("                if nzl['n'] >= 2 and nzl['neutral_s'] > LAUNCH_SUM_NOTE_S:", "                if nzl['n'] >= 1 and nzl['neutral_s'] > LAUNCH_SUM_NOTE_S:")]),
    ('23-f2-note-0.12', '② 注意のしきい 0.12 秒',
     [('LAUNCH_SUM_NOTE_S = 0.05\n', 'LAUNCH_SUM_NOTE_S = 0.12\n')]),
    ('24-f2-note-line', '② 注意の行を出さない',
     [("        if g.get('launch_sum_short'):          # 2026-10-28 fix8 tools r2(あるときだけ)\n", "        if False:\n")]),
    ('25-f2-note-trial', '② 比べに入らない本(試し)の短い止まりの続きも注意に出す',
     [("        su = [r for _, rs in sorted(groups.get(k, {}).items()) for r in rs if r.get('_launch_sum')]",
       "        su = [r for _, rs in sorted(groups.get(k, {}).items()) for r in rs if r.get('_launch_sum')] + [r for r in trials.get(k, []) if r.get('_launch_sum')]")]),
    ('26-f2-note-json', '② 注意の中だけの欄を JSON の発進の行に書く',
     [("                    row['_launch_sum'] = nzl", "                    row['launch_sum'] = nzl\n                    row['_launch_sum'] = nzl")]),
    ('27-ctl-crit', '対照: CRIT の起点の差 0.15 → 0.10',
     [("CRIT = {'to_r1_gain_s': 0.15,", "CRIT = {'to_r1_gain_s': 0.10,")]),
]


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    src = SRC.read_text(encoding='utf-8')
    lines = []
    jobs = [('00-before', '直す前の写しそのもの', None)] + M
    n_fail = 0
    for name, what, reps in jobs:
        p = TMP / ('collect_%s.py' % name)
        if reps is None:
            shutil.copy(BEFORE, p)
        else:
            s = src
            for a, b in reps:
                c = s.count(a)
                if c != 1:
                    lines.append('%-26s 壊しの元の文が %d か所(1 か所でない): %r' % (name, c, a[:80]))
                    s = None
                    break
                s = s.replace(a, b)
            if s is None:
                continue
            p.write_text(s, encoding='utf-8')
        bt = TMP / ('bt_%s' % name)
        env = dict(os.environ, JF_FIX8R2_COLLECT=str(p), PYTHONDONTWRITEBYTECODE='1')
        cmd = ['/home/jetson/bin/jf-heavy', '--lock', 'none', '--mem', '1500M', '--', PY, '-m', 'pytest', '-x', '-q', '-p',
               'no:cacheprovider', '-o', 'tmp_path_retention_policy=failed', '--basetemp=%s' % bt, '-W', 'ignore',
               'tests/test_fix8_tools_r2.py']
        r = subprocess.run(cmd, cwd=str(JF), env=env, capture_output=True, text=True)
        out = r.stdout + r.stderr
        failed = [x for x in out.splitlines() if x.startswith('FAILED ')]
        ok = r.returncode != 0 and bool(failed)
        n_fail += ok
        first = failed[0].split(' - ')[0].replace('FAILED tests/test_fix8_tools_r2.py::', '') if failed else '(落ちない)'
        lines.append('| %s | %s | %s | %s |' % (name, what, '落ちる' if ok else '**落ちない**', first))
        print(lines[-1], flush=True)
        shutil.rmtree(bt, ignore_errors=True)
    lines.append('落ちた %d / %d' % (n_fail, len(jobs)))
    shutil.rmtree(TMP, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'mutate_r2_out.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0 if n_fail == len(jobs) else 1


if __name__ == '__main__':
    sys.exit(main())
