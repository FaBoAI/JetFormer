#!/usr/bin/env python3
"""手順書 3 章の朝の順(画面を読み直す → 新しい画面の印を見るためにベンチマークを 1 つ選ぶ → 基準の設定 → 発進の前進の
車輪を浮かせた確かめ)で、確かめの走行がどんな説明欄・収集の印で送られ、サーバーの plan_check で何になるか"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run_sim import cands, blk, run, server_params, bench   # noqa: E402

c, di = cands()
for pick_no in (1, 13):
    steps = [['read', 'record'], ['read', 'desc'],
             ['pick', di, blk(c, di, pick_no)],            # 3 章: 「ベンチマークを 1 つ選んで … 新しい画面の印を見る」
             ['read', 'record'], ['read', 'desc'],
             ['baseline'],                                 # 3 章: 「基準の設定を画面に入れる」
             ['read', 'record'], ['read', 'desc'], ['read', 'gain'],
             ['check', '#dr-launch-boost', True],           # 3 章 発進の前進の確かめ 1.
             ['start']]                                    # 3 章 2.(CH3 は人側のまま ▶)
    res = run(steps, server_params())
    print('--- 朝に選んだベンチマーク', pick_no)
    for x in res['steps']:
        if x['op'] == 'read':
            print('  read', x['what'], repr(x['value'])[:160])
        if x['op'] == 'start':
            st = x
            print('  start: record=%s dry=%s launch_boost=%s throttle_gain=%s' % (x['record'], x['dry'], x['sent']['launch_boost'], x['sent']['throttle_gain']))
            print('  start: description=%r' % x['desc'])
    # サーバーの plan_check(本物の web/bench.py・本物の bench_candidates.json)
    desc2, diffs = bench.plan_check(st['desc'], c['baseline']['run'], 5, st['sent'])
    print('  plan_check の違い:', diffs)
    print('  サーバーが記録に書く説明欄(ai_no の前):', desc2)
