#!/usr/bin/env python3
"""段 Desk(2026-10-23・A10)の GPU の仕事。10/21 の desk_gpu.py をそのまま使い、保存先と候補だけを替える。
必ず jf-heavy --lock big を通す(1 つずつ)。車・装置には触れない(管理画面は GET の状態だけ読む)

    cd ~/jetformer
    J="/home/jetson/bin/jf-heavy --mem 3G --lock big --need-mb 3200 -- nice -n 10 /home/jetson/miniconda/envs/jetracer/bin/python -u"
    $J experiments/20261022_prep/desk/desk_gpu2.py crosscheck                                   # B の予測し直し(≤ 0.01 µs)
    $J experiments/20261022_prep/desk/desk_gpu2.py predict --ids SLOW-SEC0 SLOW1 SLOW-SEC1       # best・last(出来たものは飛ばす)
    $J experiments/20261022_prep/desk/desk_gpu2.py sanity --ids SLOW-SEC0 SLOW1 SLOW-SEC1        # 10/10 の AI 走行 21,759 フレーム

保存先: /home/jetson/jetformer_reports/prep_20261022/desk/pred/(予測・sanity_<id>.npz)、logs/ はこのフォルダの下。
10/21 の desk/pred・logs は読みも書きもしない(B・SLOW の値は desk_eval2.py が 10/21 のファイルを読むだけ)。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import desk_lib2 as D2  # noqa: E402

D2.register()
import desk_gpu as G  # noqa: E402  (10/21 の台本。D2 が sys.path に入れた)

G.OUT = Path('/home/jetson/jetformer_reports/prep_20261022/desk')
G.PRED = G.OUT / 'pred'
G.LOGS = HERE / 'logs'
G.D.NEW_TRAINED = D2.NEW_IDS

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('predict', 'sanity') and '--ids' not in sys.argv:
        raise SystemExit('--ids を付けてください(新しい 3 本: %s)' % ' '.join(D2.NEW_IDS))
    G.main()
