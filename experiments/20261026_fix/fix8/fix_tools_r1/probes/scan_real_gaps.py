"""fix8 係 tools 直す 8-1 回目(読むだけ。data/raw・data/labels は読むだけ。⑤ の出力は ../../tmp_tools_r1/real_gaps に書き、終わって消す)。

本物の 10/10・10/23・10/24 の発進(⑤ collect.py --day D --ignore-gain の launches。直す前の写し
tests/fixtures/fix8_tools_r1_before/collect.py.orig)ごとに、AI 側にしてから R1 の起点まで(届かなかった発進は AI の区間の終わりか
8 秒まで)の、フレームの間のいちばん長いものを 3 つの数え方で出す(w1-tools-r8-1-stall-inflates-off の直しのしきいを決めるため):
  proc  AI の区間のフレームのうち中立止め(hold)でないもの(⑤ の直しで使う数え方。端の AI 側にした時刻・起点も入れる)
  raw   AI の区間のフレーム全部(中立止めの行もフレームとみる。③ standing_3lap の 3 周の途切れと同じ数え方)
  selB  打ち消しの票 B(refute_tools_r1_B/b2_launch_gaps_real.py)と同じ: AI_CMD の select・中立止めでない行どうし(間に人側の行が
        あれば数えない)
あわせて、比べ(起点まで)に入る発進(R1 に届いた・試し / 途中で変えた / 走り直しでない)を分けて、発進の窓の中のフレームの間の
全部の分布(0.05 秒ごと)を出す。出力 ../out/scan_real_gaps.txt・.json

  cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
      /home/jetson/miniconda/envs/jetracer/bin/python -B experiments/20261026_fix/fix8/fix_tools_r1/probes/scan_real_gaps.py
"""
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
for p in (JF, JF / 'tools', JF / 'tests'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from test_fix3_r1_2 import load_mod  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'out'
TMP = JF / 'experiments/20261026_fix/fix8/tmp_tools_r1/real_gaps'
COLLECT = JF / 'tests/fixtures/fix8_tools_r1_before/collect.py.orig'
DAYS = ('20261010', '20261023', '20261024')


def max_gap(ts, a, b):
    """a〜b の間の時刻の並び ts の最大の間(端の a・b も入れる。standing_3lap._max_gap と同じ)と、その始め"""
    ts = np.asarray(ts, float)
    m = (ts > a) & (ts < b)
    pts = np.concatenate([[a], ts[m], [b]])
    if len(pts) < 2:
        return 0.0, a
    d = np.diff(pts)
    k = int(np.argmax(d))
    return float(d[k]), float(pts[k])


def selb_gap(t, sel, hold, a, b):
    proc = np.nonzero(sel & ~hold & (t >= a - 1e-9) & (t <= b + 1e-9))[0]
    best = 0.0
    for i, j in zip(proc, proc[1:]):
        if not sel[i:j + 1].all():
            continue
        best = max(best, float(t[j] - t[i]))
    return best


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    res = {'launches': [], 'hist_compared': {}, 'hist_all': {}}
    lines = []
    try:
        for day in DAYS:
            m = load_mod('c_scan_%s' % day, COLLECT)
            out = TMP / ('c5_%s.json' % day)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                m.main(['--day', day, '--out', str(out), '--ignore-gain'])
            j = json.loads(out.read_text(encoding='utf-8'))
            blocks, names = m.bench_blocks()
            cache = {}
            for r in j['launches']:
                lg = r['log']
                if lg not in cache:
                    d = m.load(lg, True, blocks, names)
                    z = d['z']
                    cache[lg] = (np.asarray(z['t_s'], float), np.asarray(z['ai'], bool),
                                 np.asarray(z['hold'], bool) if 'hold' in z else np.zeros(len(z['t_s']), bool),
                                 np.asarray(z['ai_cmd_select'], bool) if 'ai_cmd_select' in z else np.ones(len(z['t_s']), bool),
                                 np.asarray(z['dry_run'], bool) if 'dry_run' in z else np.zeros(len(z['t_s']), bool))
                t, ai, hold, sel, dry = cache[lg]
                a = float(r['t_ai'])
                i0 = int(np.argmin(np.abs(t - a)))
                i1 = i0
                while i1 + 1 < len(t) and ai[i1 + 1]:
                    i1 += 1
                reached = r.get('to_r1_s') is not None
                b = a + float(r['to_r1_s']) if reached else min(float(t[i1]), a + 8.0)
                seg_t = t[i0:i1 + 1]
                seg_h = hold[i0:i1 + 1]
                g_proc, at_proc = max_gap(seg_t[~seg_h], a, b)
                g_raw, _ = max_gap(seg_t, a, b)
                g_selb = selb_gap(t, sel & ~dry, hold, a, b)
                compared = (reached and not r.get('after_abort') and not r.get('trial') and not r.get('midrun'))
                row = {'log': lg, 't_ai': a, 'to_r1_s': r.get('to_r1_s'), 'launch_state': r.get('launch_state'),
                       'session': r.get('session'), 'compared': compared, 'proc': round(g_proc, 3), 'proc_at': round(at_proc, 3),
                       'raw': round(g_raw, 3), 'selB': round(g_selb, 3), 'hold_rows': int(np.sum(seg_h & (seg_t >= a) & (seg_t <= b))),
                       'midrun': bool(r.get('midrun')), 'trial': bool(r.get('trial')), 'after_abort': bool(r.get('after_abort'))}
                res['launches'].append(row)
                # 窓の中のフレームの間の全部(proc)の分布
                pts = np.concatenate([[a], seg_t[~seg_h][(seg_t[~seg_h] > a) & (seg_t[~seg_h] < b)], [b]])
                for x in np.diff(pts):
                    key = '%.2f' % (np.floor(x / 0.05) * 0.05)
                    res['hist_all'][key] = res['hist_all'].get(key, 0) + 1
                    if compared:
                        res['hist_compared'][key] = res['hist_compared'].get(key, 0) + 1
        L = res['launches']
        lines.append('発進 %d(届いた %d・比べに入る %d)' % (len(L), sum(r['to_r1_s'] is not None for r in L), sum(r['compared'] for r in L)))
        for day in DAYS:
            Ld = [r for r in L if r['log'][:8] == day]
            for key in ('proc', 'raw', 'selB'):
                lines.append('%s %-4s: 発進 %d・いちばん長い %.3f 秒(届いた %.3f・比べに入る %.3f)・0.15 秒超 %d・0.2 秒超 %d・0.3 秒超 %d・0.5 秒超 %d' % (
                    day, key, len(Ld), max([r[key] for r in Ld], default=0),
                    max([r[key] for r in Ld if r['to_r1_s'] is not None], default=0),
                    max([r[key] for r in Ld if r['compared']], default=0),
                    sum(r[key] > 0.15 for r in Ld), sum(r[key] > 0.2 for r in Ld), sum(r[key] > 0.3 for r in Ld),
                    sum(r[key] > 0.5 for r in Ld)))
        lines.append('0.1 秒より長い(どれかの数え方で)発進:')
        for r in L:
            if max(r['proc'], r['raw'], r['selB']) > 0.1:
                lines.append('  ' + json.dumps(r, ensure_ascii=False))
        lines.append('発進の窓の中のフレームの間(proc)の分布(0.05 秒ごとの下の端: 数): 全部 %s' % json.dumps(
            dict(sorted(res['hist_all'].items())), ensure_ascii=False))
        lines.append('  比べに入る発進だけ %s' % json.dumps(dict(sorted(res['hist_compared'].items())), ensure_ascii=False))
        (OUT / 'scan_real_gaps.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        txt = '\n'.join(lines) + '\n'
        (OUT / 'scan_real_gaps.txt').write_text(txt, encoding='utf-8')
        print(txt)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == '__main__':
    main()
