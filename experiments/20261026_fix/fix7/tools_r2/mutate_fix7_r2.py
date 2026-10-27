"""fix7 tools r2: わざと壊す。本物のファイルは変えない。写し(fix7/tmp_tools_r2/mut/)に 1 か所ずつ壊しを入れ、環境変数
JF_FIX7_COLLECT・JF_FIX7_S3・JF_FIX7_QP で tests/test_fix7_tools_r2.py を -x で流す(中の pytest は 1 つずつ jf-heavy)。
結果は out/mutate_fix7_r2_out.txt。写しと basetemp は最後に消す"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
TMP = JF / 'experiments/20261026_fix/fix7/tmp_tools_r2/mut'
OUT = Path(__file__).resolve().parent / 'out' / 'mutate_fix7_r2_out.txt'
SRC = {'S': JF / 'tools/standing_3lap.py', 'C': JF / 'experiments/20261026_fix/diag_launch/collect.py',
       'Q': JF / 'tools/quick_eval_by_params.py'}
ENV = {'S': 'JF_FIX7_S3', 'C': 'JF_FIX7_COLLECT', 'Q': 'JF_FIX7_QP'}
M = [
    ('S', '③ 中立止めの形を見ない', "        if len(k) and h[k[-1]]:\n            j = len(k) - 1\n            while j > 0 and h[k[j - 1]]:",
     "        if False:\n            j = len(k) - 1\n            while j > 0 and h[k[j - 1]]:"),
    ('S', '③ カメラの途切れの形を見ない', "            if tb - ta > le.GAP_S:", "            if False:"),
    ('S', '③ 中立止めの長さ 0.5 → 0.6 秒', "if hs >= HOLD_KEEP_MAX_S - 1e-6 and", "if hs >= HOLD_KEEP_MAX_S + 0.1 and"),
    ('S', '③ 中立止めの終わりで止まっているかを見ない', "if hs >= HOLD_KEEP_MAX_S - 1e-6 and abs(float(np.interp(t[i1], vt_, ve))) < STOP_ERPM:",
     "if hs >= HOLD_KEEP_MAX_S - 1e-6:"),
    ('S', '③ 区間で進んだかを見ない', "        if not can_stop or moved(i0, i1) < RESUME_MOVE_M:\n            return None",
     "        if not can_stop:\n            return None"),
    ('S', '③ 途切れの間に止まったかを見ない', "                if float(e.min()) < STOP_ERPM:", "                if True:"),
    ('S', '③ 中立止めで AI_CMD が人側のフレームを除かない', "        k = k[s[i0:i1 + 1]]                      # AI_CMD が人側のフレームを除く\n", ""),
    ('S', '③ 中立止めの長さを最初の中立止めのフレームから', "            tp = float(t[k[j - 1]]) if j > 0 else float(t[k[j]])", "            tp = float(t[k[j]])"),
    ('S', '③ 途切れの決まり GAP_S → 0.5 秒', "            if tb - ta > le.GAP_S:", "            if tb - ta > 0.5:"),
    ('S', '③ analyze_log が eRPM を渡さない', "ex['vt'], Df, ex['verpm'])", "ex['vt'], Df)"),
    ('S', '③ 中止の印に kind を付けない', "            pend = {'kind': 'abort', 'abort_s'", "            pend = {'abort_s'"),
    ('S', '③ 表の文: 中立止めを中止の文で', "    if kind == 'hold':\n        return '%s %s 秒から %s 秒' % (STOP_KIND_TEXT['hold']",
     "    if kind == 'xhold':\n        return '%s %s 秒から %s 秒' % (STOP_KIND_TEXT['hold']"),
    ('S', '③ 途切れの止まりを始めの時刻だけで見る', "                e = np.abs(np.concatenate([ve[m], [float(np.interp(tb, vt_, ve))]]))",
     "                e = np.abs(np.array([float(np.interp(ta, vt_, ve))]))"),
    ('S', '③ 中止の無い本は {}(r1 の早い戻り)', "    if not has_abort and not can_stop:\n        return {}", "    if not has_abort:\n        return {}"),
    ('S', '③ 中立止めのあとの続き(carried)を見ない', "            elif pend is not None and moved(i0, i1) < RESUME_MOVE_M:\n                pend = dict(pend, carried=pend['carried'] + 1)",
     "            elif False:\n                pend = dict(pend, carried=pend['carried'] + 1)"),
    ('C', '⑤ launch_rows が eRPM を渡さない', "vt, d['Df'], d['erpm'])", "vt, d['Df'])"),
    ('C', '⑤ 走り直しの空転を checks の slip に入れない', "and not on_use.get('slip_extreme') and not g.get('abort_resume_slip')),",
     "and not on_use.get('slip_extreme')),"),
    ('C', '⑤ 走り直しの比が出ないを slip_all_measured に入れない', "if on_use.get('slip_unmeasured') or g.get('abort_resume_slip_unmeasured'):",
     "if on_use.get('slip_unmeasured'):"),
    ('C', '⑤ 走り直しの最初の周の欠けを数えない', "        gap_on += [r for r in aborts.get(k, []) if r.get('launch_state') == 'on' and r.get('first_lap_gap')]",
     "        gap_on += []"),
    ('C', '⑤ いちばんひどい空転を数えない', "if slip_state(r.get('slip')) in ('high', 'extreme')]", "if slip_state(r.get('slip')) in ('high',)]"),
    ('C', '⑤ OFF の走り直しの空転も数える', "            ab_on = [r for r in aborts[k] if r.get('launch_state') == 'on']", "            ab_on = list(aborts[k])"),
    ('C', '⑤ launch_us が混ざっても全部の走り直しの空転を数える', "            if g.get('launch_us_used') is not None:\n                ab_on = [r for r in ab_on",
     "            if False:\n                ab_on = [r for r in ab_on"),
    ('C', '⑤ 空転の行を出さない', "        if g.get('abort_resume_slip'):         # 2026-10-27 fix7 r2(あるときだけ)", "        if False:"),
    ('C', '⑤ 組の行に空転の比を出さない', "r['to_r1_s'], '・' + slip_text(r.get('slip')) if r.get('launch_state') == 'on' else '',", "r['to_r1_s'], '',"),
    ('C', '⑤ 止まった形の文: 中立止めを中止の文で', "    if kind == 'hold':\n        return '中立止め", "    if kind == 'xhold':\n        return '中立止め"),
    ('C', '⑤ 欠けた発進の行に走り直しの印を出さない', "'。' + RESUME_NAME if r.get('after_abort') else '') for r in gap_on]", "'') for r in gap_on]"),
    ('C', '⑤ 届いた走り直しを比べに戻す(r1 の直しを外す)', "        if r.get('after_abort') and r.get('to_r1_s') is not None:", "        if False:"),
    ('C', '対照: CRIT の起点の差 0.15 → 0.10', "CRIT = {'to_r1_gain_s': 0.15,", "CRIT = {'to_r1_gain_s': 0.10,"),
    ('Q', '④ 1 章の印の名前を変える', "RESUME_NAME = 'リカバリー中止のあとの走り直し'", "RESUME_NAME = '中止・中立止めのあとの走り直し'"),
    ('Q', '④ 2 章の下の行を出さない', "    if any(g.get('launch_to_r1_abort_excluded') for g in groups):", "    if False:"),
    ('Q', '④ 2 章の下の行に中立止めの 1 文を足さない', "                   if any((s.get('after_abort') or {}).get('kind') in ('hold', 'stall')",
     "                   if False and any((s.get('after_abort') or {}).get('kind') in ('hold', 'stall')"),
    ('S', '③ 中立止めの注 HOLD_NOTE を出さない', "        out.append(HOLD_NOTE)", "        pass"),
    ('S', '③ 中止の文を「リカバリー中止」に変える(手順書が引く言葉)', "STOP_KIND_TEXT = {'abort': '中止',", "STOP_KIND_TEXT = {'abort': 'リカバリー中止',"),
    ('C', '⑤ 発進の行の頭を変える(手順書が引く言葉)', "' 中止のあとの走り直し(%s・人側 %s 秒。%s)'", "' 走り直し(%s・人側 %s 秒。%s)'"),
    ('C', '⑤ 中止の文を「リカバリー中止」に変える', "    return '中止 %s 秒' % ab.get('abort_s')", "    return 'リカバリー中止 %s 秒' % ab.get('abort_s')"),
]


def main(lo=0, hi=None, tag=''):
    global TMP, OUT
    if tag:                      # 分けて流すとき(python3 mutate_fix7_r2.py 7 21 b): 写しの場所と出力を分ける
        TMP = TMP.parent / ('mut_' + tag)
        OUT = OUT.parent / ('mutate_fix7_r2_out_%s.txt' % tag)
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
                   '-W', 'ignore::DeprecationWarning', '--basetemp=%s' % bt, 'tests/test_fix7_tools_r2.py']
            r = subprocess.run(cmd, cwd=str(JF), env=env, capture_output=True, text=True, timeout=1800)
            fl = [x.split(' - ')[0].replace('FAILED tests/test_fix7_tools_r2.py::', '') for x in r.stdout.splitlines()
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
