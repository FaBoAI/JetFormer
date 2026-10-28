"""反証 8-2 回目・係 tools: 本物の 10/10・10/23・10/24(読むだけ。data/raw・data/labels は変えない)で、⑤ collect.py の 8-1 の直す前
(tests/fixtures/fix8_tools_r1_before/collect.py.orig)と今を、4 通りの引数(なし・--first-only・--ignore-gain・両方)+ 引数なし
(10/23・10/24 の道)で流して比べる。出力は out/real/ の下だけ。

あわせて、R1 に届いた発進ごとに、発進 → R1 の起点の AI 側の処理したフレーム(中立止めの行を除く)の間のいちばん長いもの・0.15 秒より
長い間の数・drive の中立の見積もりの合計(Σ max(0, 間 − 0.1))・中立止めの行の数を出す(今の launch_stall と同じ数え方)。

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -B experiments/20261026_fix/fix8/review_tools_r2/p5_real.py
  → out/p5_real.txt
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

import common as C
from common import t8, T

FIX = C.JF / 'tests/fixtures/fix8_tools_r1_before/collect.py.orig'
NEW = C.JF / 'experiments/20261026_fix/diag_launch/collect.py'
DAYS = ('20261010', '20261023', '20261024')
VARS = (('plain', []), ('first-only', ['--first-only']), ('ignore-gain', ['--ignore-gain']),
        ('ig-fo', ['--ignore-gain', '--first-only']))
O = C.OUT / 'real'


def nopath(t):
    return '\n'.join(x for x in t.splitlines() if not x.startswith('出力 '))


def run_one(ver, path, name, args):
    m = t8.load_mod('c8r2_real_%s_%s' % (ver, name.replace('-', '_')), path)
    out = O / ver / ('launches_%s.json' % name)
    out.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = m.main(args + ['--out', str(out)])
    (O / ver / ('launches_%s.stdout' % name)).write_text(buf.getvalue(), encoding='utf-8')
    return rc


def scan(lines):
    m = t8.load_mod('c8r2_real_scan', NEW)
    lines.append('# R1 に届いた発進の、発進 → R1 のフレームの間(中立止めの行を除く)')
    lines.append('  名前 状態 印 | いちばん長い間 | 0.15 秒より長い間の数 | Σ max(0, 間 − 0.1) | 窓の中の中立止めの行')
    tot = {'n': 0, 'cmp': 0}
    hist = {}
    for day in DAYS:
        blocks, names = m.bench_blocks()
        for log in m.logs((day,)):
            try:
                d = m.load(log, True, blocks, names)
            except Exception as e:      # noqa: BLE001
                lines.append('  %s 読めない(%s)' % (log, e))
                continue
            if not d['z']['ai'].any():
                continue
            rows = m.launch_rows(d, [], 'none')
            z = d['z']
            t = np.asarray(z['t_s'], float)
            h = np.asarray(z['hold'], bool) if 'hold' in z else np.zeros(len(t), bool)
            for r in rows:
                if r.get('to_r1_s') is None:
                    continue
                t0, t1 = r['t_ai'], r['t_ai'] + r['to_r1_s']
                m_ = (t >= t0 - 1e-6) & (t <= t1 + 1e-6)
                tp = t[m_ & ~h]
                tp = tp[(tp > t0 + 1e-9) & (tp < t1 - 1e-9)]
                pts = np.concatenate([[t0], tp, [t1]])
                g = np.diff(pts)
                mx = float(g.max())
                n15 = int(np.sum(g > 0.15))
                neu = float(np.sum(np.clip(g - 0.1, 0, None)))
                nh = int(np.sum(m_ & h))
                tag = [k for k in ('midrun', 'trial', 'after_abort', 'launch_stall', '_launch_gap') if r.get(k)]
                cmp_ = not any(r.get(k) for k in ('midrun', 'trial', 'after_abort', 'launch_stall'))
                tot['n'] += 1
                tot['cmp'] += int(cmp_)
                b = '%.2f' % (np.floor(mx / 0.05) * 0.05)
                hist[b] = hist.get(b, 0) + 1
                if mx > 0.1 or n15 or nh or neu > 0.02:
                    lines.append('  %s@%.2f %s %s | %.3f | %d | %.3f | %d' % (log, r['t_ai'], r.get('launch_state'), tag, mx, n15, neu, nh))
    lines.append('  R1 に届いた発進 %d(比べに入る形 %d)。いちばん長い間の分布(0.05 秒ごと): %s' % (
        tot['n'], tot['cmp'], json.dumps(dict(sorted(hist.items())), ensure_ascii=False)))
    lines.append('  (上の表は、いちばん長い間 > 0.10 秒・0.15 秒より長い間あり・中立止めの行あり・中立の見積もり > 0.02 秒の発進だけ)')
    lines.append('')


def main():
    lines = ['# ⑤ の出力: 直す前(8-1 の前の写し)と今(出力の場所の行は除く)']
    jobs = [('%s_%s' % (d, n), ['--day', d] + a) for d in DAYS for n, a in VARS] + [('default', [])]
    n_same = n_diff = 0
    for name, args in jobs:
        for ver, p in (('old', FIX), ('new', NEW)):
            run_one(ver, p, name, args)
        for suf in ('json', 'stdout'):
            a = (O / 'old' / ('launches_%s.%s' % (name, suf))).read_text(encoding='utf-8')
            b = (O / 'new' / ('launches_%s.%s' % (name, suf))).read_text(encoding='utf-8')
            same = nopath(a) == nopath(b) if suf == 'stdout' else a == b
            n_same += same
            n_diff += not same
            lines.append('  %-24s %-6s %s' % (name, suf, 'バイト単位で同じ' if same else '違う'))
            if not same and suf == 'json':
                ja, jb = json.loads(a), json.loads(b)
                lines.append('    launch_stall の付いた発進: %s' % [('%s@%.2f' % (r['log'], r['t_ai'])) for r in jb['launches']
                                                                  if r.get('launch_stall')])
                lines.append('    other_group_on: %s' % [g.get('other_group_on') for g in jb['summary'] if g.get('other_group_on')])
                lines.append('    判定の違い: %s' % [(g['session'], C.verdict(ga), C.verdict(g)) for ga, g in zip(ja['summary'], jb['summary'])
                                                if C.verdict(ga) != C.verdict(g)])
    lines.append('  同じ %d・違う %d' % (n_same, n_diff))
    lines.append('')
    # 今の版の、比べのある組の判定(読むだけ)
    lines.append('# 今の版の比べのある組(--ignore-gain)')
    for d in DAYS:
        j = json.loads((O / 'new' / ('launches_%s_ignore-gain.json' % d)).read_text(encoding='utf-8'))
        for g in j['summary']:
            if 'compare' in g or g.get('launch_gap_short') or g.get('launch_stall_excluded') or g.get('other_group_on'):
                lines.append('  %s %s' % (d, C.gtext(g)))
    lines.append('')
    scan(lines)
    C.write('p5_real.txt', lines)


if __name__ == '__main__':
    main()
