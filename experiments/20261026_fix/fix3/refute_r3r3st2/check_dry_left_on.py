#!/usr/bin/env python3
"""反証(r3-r3-state-2 の打ち消し): 手順書 3 章 86 行の dry run(「モーターに出力しない」ON)のあと、発進の前進の確かめ(88 行)を
しない日に、4 章どおり 基準の設定 → ベンチマーク 1 → ゲイン → ▶ とすると、dry のまま始まるか。本物の app.js を
refute3_state/sim3.js(node の vm・偽のサーバー)で流す。車・jetformer-web には触れない。"""
import sys
from pathlib import Path
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261026_fix/fix3/refute3_state')
from run3 import cands, blk, run, server_params   # noqa: E402

c, di = cands()
b1 = blk(c, di, 1)


def show(title, res):
    print('==', title)
    for x in res['steps']:
        if x['op'] == 'start':
            print('  start: dry=%s record=%s confirm_shown=%s refused=%s gain=%s desc=%r' % (
                x.get('dry'), x.get('record'), bool(x.get('confirm')), x.get('refused'),
                x.get('sent') and x['sent'].get('throttle_gain'), (x.get('desc') or '')[:90]))
        elif x['op'] in ('el', 'read'):
            print('  ', x)
    print('  toasts:', [t for t in res['toasts'] if '推論' in t or 'AI走行' in t or '走行を開始' in t])


# A: 発進の前進を使わない日。3 章 dry → 停止 → 4 章 1-1(基準の設定・ブロック 1・+0.01×2・説明欄を書き足す・▶)
A = [['baseline'], ['check', '#dr-dry', True], ['start'], ['running', False],
     ['el', '#dr-dry', 'checked'],
     ['baseline'], ['el', '#dr-dry', 'checked'],
     ['pick', di, b1], ['el', '#dr-dry', 'checked'],
     ['click', '#btn-dr-tg-up'], ['click', '#btn-dr-tg-up'],
     ['read', 'desc'], ['start'], ['el', '#dr-dry', 'checked']]
show('A: dry のあと外さずに 1-1', run(A, server_params()))

# B: 比べ。3 章 88 行の手順どおり dry を外した日(発進の前進の確かめをした日)
B = [['baseline'], ['check', '#dr-dry', True], ['start'], ['running', False],
     ['check', '#dr-dry', False],
     ['baseline'], ['pick', di, b1], ['click', '#btn-dr-tg-up'], ['click', '#btn-dr-tg-up'], ['start'], ['el', '#dr-dry', 'checked']]
show('B: dry を外した日の 1-1', run(B, server_params()))
