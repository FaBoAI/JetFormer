#!/usr/bin/env python3
"""手順書 3 章の「車輪を浮かせて dry run 30 秒」(モーターに出力しない ON)のあと、発進の前進の確かめをしない日(了承なし)は
「モーターに出力しない」を外す行が手順書に無い。4 章どおり 基準の設定 → ベンチマーク 1 → ゲイン → ▶ とすると?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run3 import cands, blk, run, server_params   # noqa: E402

c, di = cands()
steps = [
    ['baseline'],                                   # 3 章「基準の設定を画面に入れる」
    ['check', '#dr-dry', True], ['start'],          # 3 章「了承があれば dry run 30 秒」
    ['running', False],                             # ■ 停止
    # 4 章 1 本の手順(1-1)
    ['baseline'], ['pick', di, blk(c, di, 1)], ['click', '#btn-dr-tg-up'], ['click', '#btn-dr-tg-up'],
    ['set', '#dr-desc', None], ['start'],
]
# 説明欄は雛形のまま(battery など書き足した形にする)
res = run(steps[:-2] + [['read', 'desc'], ['start'], ['el', '#dr-dry', 'checked']], server_params())
for x in res['steps']:
    if x['op'] == 'start':
        print('start: dry=%s record=%s confirm_shown=%s gain=%s desc=%s' % (x['dry'], x['record'], bool(x['confirm']),
              x['sent'] and x['sent']['throttle_gain'], (x['desc'] or '')[:80]))
    elif x['op'] in ('read', 'el'):
        print(x)
print([t for t in res['toasts'] if '推論' in t or 'AI走行' in t])
