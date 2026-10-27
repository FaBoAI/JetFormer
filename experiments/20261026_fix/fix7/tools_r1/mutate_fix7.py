"""fix7 係 tools 直す 7-1 回目: わざと壊した写しで tests/test_fix7_tools.py が落ちるかを見る。本物のファイルは変えない
(写しを fix7/tmp_tools/mut/ に作り、環境変数 JF_FIX7_COLLECT・JF_FIX7_S3・JF_FIX7_QP で差し替える)。終わったら写しと basetemp を消す。

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 python3 experiments/20261026_fix/fix7/tools_r1/mutate_fix7.py
      (中で 1 つずつ jf-heavy --lock none --mem 1500M -- python -m pytest -x を流す)。結果は out/mutate_fix7_out.txt
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
TMP = JF / 'experiments/20261026_fix/fix7/tmp_tools/mut'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
HEAVY = ['/home/jetson/bin/jf-heavy', '--lock', 'none', '--mem', '1500M', '--']
SRC = {'C': JF / 'experiments/20261026_fix/diag_launch/collect.py', 'S': JF / 'tools/standing_3lap.py',
       'Q': JF / 'tools/quick_eval_by_params.py'}
ENV = {'C': 'JF_FIX7_COLLECT', 'S': 'JF_FIX7_S3', 'Q': 'JF_FIX7_QP'}
M = [
    # ⑤ collect.py
    ('C', '⑤ 走り直しを見ない(ab をいつも空)',
     "        ab = s3.abort_resumes(t, ai, z.get('assist'), z.get('hold'), z.get('ai_cmd_select'), vt, d['Df'])",
     "        ab = {}"),
    ('C', '⑤ 発進の行に after_abort を付けない', "            if i0 in ab:", "            if False:"),
    ('C', '⑤ まとめで外さない', "        if r.get('after_abort') and r.get('to_r1_s') is not None:", "        if False:"),
    ('C', '⑤ R1 に届かなかった走り直しも外す(届かなかった発進に入れない)',
     "        if r.get('after_abort') and r.get('to_r1_s') is not None:", "        if r.get('after_abort'):"),
    ('C', '⑤ ON の走り直しの最初の周の衝突を数えない',
     "                                              and not g.get('abort_resume_on_collisions')),", "),"),
    ('C', '⑤ OFF の走り直しの衝突も ON として数える',
     "for r in aborts[k] if r.get('launch_state') == 'on')", "for r in aborts[k])"),
    ('C', '⑤ 名前の行を出さない', "        if g.get('abort_resume_excluded'):     # 2026-10-27 fix7(あるときだけ)", "        if False:"),
    ('C', '⑤ 届かなかった発進の after_abort を書かない', "        if ab_:", "        if False:"),
    ('C', '⑤ 届かなかった発進の行の後ろの文を出さない', "                if b.get('after_abort') else ''))", "                if False else ''))"),
    ('C', '⑤ 発進ごとの標準出力の行に印を出さない',
     "'R1 に届かなかった発進に数える') if r.get('after_abort') else ''))", "'R1 に届かなかった発進に数える') if False else ''))"),
    ('C', '⑤ 組の名前に起点までを出さない', "人側 %s 秒・起点まで %.3f 秒%s)' % (", "人側 %s 秒%.0s%s)' % ("),
    ('C', '対照: CRIT の起点の差 0.15 → 0.10', "CRIT = {'to_r1_gain_s': 0.15,", "CRIT = {'to_r1_gain_s': 0.10,"),
    # ③ standing_3lap.py(abort_resumes は ⑤ も使う)
    ('S', '③ abort_resumes がいつも空', "    if not len(a) or not bool((a == ASSIST_GIVEUP).any()):", "    if True:"),
    ('S', '③ 中立止めのフレームを除かない', "        k = k[~h[i0:i1 + 1] & s[i0:i1 + 1]]", "        k = k[s[i0:i1 + 1]]"),
    ('S', '③ AI_CMD が人側のフレームを除かない', "        k = k[~h[i0:i1 + 1] & s[i0:i1 + 1]]", "        k = k[~h[i0:i1 + 1]]"),
    ('S', '③ 続き(carried)を見ない', "        elif pend is not None and moved(i0, i1) < RESUME_MOVE_M:", "        elif False:"),
    ('S', '③ いつも続ける', "moved(i0, i1) < RESUME_MOVE_M:", "True:"),
    ('S', '③ 中止で終わっていなくても(区間の中に中止があれば)印',
     "        if len(k) and int(a[k[-1]]) == ASSIST_GIVEUP:", "        if len(k) and bool((a[k] == ASSIST_GIVEUP).any()):"),
    ('S', '③ 人側の秒を中止の区間の終わりから(続きで違う)',
     "human_s=_r(float(t[i0]) - prev_end, 2)", "human_s=_r(float(t[i0]) - pend['ai_end_s'], 2)"),
    ('S', '③ 中止の始めを戻さない(最後の中止のフレーム)', "                j -= 1", "                break"),
    ('S', '③ counted を False にしない', "                s['after_abort'] = v\n                s['counted'] = False", "                s['after_abort'] = v"),
    ('S', '③ 表の「数えない(リカバリー中止のあとの走り直し…)」を出さない',
     "            if s.get('after_abort'):     # 2026-10-27 fix7(あるときだけ)", "            if False:"),
    ('S', '③ 注の 1 行を出さない', "        out.append(ABORT_NOTE)", "        pass"),
    ('S', '③ 0.5 m → 0.3 m', "RESUME_MOVE_M = 0.5 ", "RESUME_MOVE_M = 0.3 "),
    # ④ quick_eval_by_params.py
    ('Q', '④ 走り直しを「前に衝突」に数える',
     "            and (s.get('coll_before') or not (s.get('midrun') or s.get('after_abort')))]",
     "            and (s.get('coll_before') or not s.get('midrun'))]"),
    ('Q', '④ 1 章の印「数えない発進 N(リカバリー中止のあとの走り直し)」を出さない', "    if exa:  ", "    if False:  "),
    ('Q', '④ 前に衝突のある走り直しも中止の印に数える',
     "    exa = [s for s in res.get('standing') or [] if s.get('valid') and s.get('after_abort') and not s.get('coll_before')",
     "    exa = [s for s in res.get('standing') or [] if s.get('valid') and s.get('after_abort') and True"),
    ('Q', '④ 動く → A1 に走り直しを入れる',
     "and s.get('move_after_s') is not None and not s.get('after_abort')]", "and s.get('move_after_s') is not None]"),
    ('Q', '④ 組に launch_to_r1_abort_excluded を書かない', "        if to_r1_ab:  ", "        if False:  "),
    ('Q', '④ 2 章の下の行を出さない', "    if any(g.get('launch_to_r1_abort_excluded') for g in groups):", "    if False:"),
]


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    out = []
    try:
        for i, (kind, what, a, b) in enumerate(M):
            src = SRC[kind].read_text(encoding='utf-8')
            n = src.count(a)
            if n != 1:
                out.append('%02d %s: 置き換えの元が %d か所(壊せない)' % (i, what, n))
                print(out[-1], flush=True)
                continue
            mp = TMP / ('m%02d_%s' % (i, SRC[kind].name))
            mp.write_text(src.replace(a, b), encoding='utf-8')
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', **{ENV[kind]: str(mp)})
            bt = TMP / ('bt%02d' % i)
            r = subprocess.run(HEAVY + [PY, '-m', 'pytest', '-x', '-q', '-p', 'no:cacheprovider', '-W', 'ignore::DeprecationWarning',
                                        '--basetemp=%s' % bt, 'tests/test_fix7_tools.py'],
                               cwd=str(JF), env=env, capture_output=True, text=True, timeout=1800)
            first = [x for x in r.stdout.splitlines() if x.startswith('FAILED ')]
            last = [x for x in r.stdout.splitlines() if 'passed' in x or 'failed' in x or 'error' in x][-1:] or ['?']
            out.append('%02d %s: %s(rc=%d)%s' % (i, what, '落ちる' if r.returncode != 0 else '**落ちない**', r.returncode,
                                                ' 最初: %s・%s' % (first[0][7:].split(' - ')[0] if first else '—', last[0])))
            print(out[-1], flush=True)
            shutil.rmtree(bt, ignore_errors=True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    bad = [x for x in out if '落ちない' in x or '壊せない' in x]
    out.append('全部 %d 通り・落ちる %d・落ちない/壊せない %d' % (len(M), len(M) - len(bad), len(bad)))
    (HERE / 'out' / 'mutate_fix7_out.txt').write_text('\n'.join(out) + '\n', encoding='utf-8')
    print(out[-1])
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
