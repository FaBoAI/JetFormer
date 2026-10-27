#!/usr/bin/env python3
"""Jetson の時計が NTP の前(電源を切った前の晩の日付)のまま画面を開いたとき: /api/bench/candidates の today が 11/03 でない。
app.js の選ぶ欄はどの日のブロックを出し、手順書 4 章どおり「ベンチマーク 2」(1-2・止まって 3 周・モデル B)を選ぶと何を送るか"""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run3 import cands, run, server_params   # noqa: E402

c, di = cands()
c = dict(c, today='2026-11-02')                         # 時計が 11/2 の夜のまま(NTP の前)
models = sorted({cd['run'] for cd in c['candidates']})
steps = [['el', '#dr-bench-pick', 'html'], ['baseline'], ['pick', 0, 1], ['read', 'desc'], ['read', 'gain'], ['start']]
import run3
orig = run3.cands
run3.cands = lambda: (c, di)
res = run(steps, server_params(), extra={'models': models})
html = [x['value'] for x in res['steps'] if x['op'] == 'el'][0]
opts = re.findall(r'<option value="([^"]*)"[^>]*>([^<]*)', html)
print('選ぶ欄:', [o[1][:40] for o in opts[:4]], '…', len(opts) - 1, '本')
s = [x for x in res['steps'] if x['op'] == 'start'][0]
print('「ベンチマーク2」を選んで ▶: run=%s gain=%s desc=%s' % (s['run'], s['sent']['throttle_gain'], (s['desc'] or '')[:90]))
print('確認の文:', s['confirm'].splitlines()[1:3])
