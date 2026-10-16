#!/usr/bin/env python3
"""透過的な大きなページ(THP)を、このプロセスとその子で使わないようにしてから、引数のコマンドに置き換わる

    python -I nothp.py <コマンド> [引数 ...]

この Jetson は THP が always で、学習のプロセスの匿名メモリが 2 MB 単位でふくらむ(メモリの上限 3G に近づく)。
prctl(PR_SET_THP_DISABLE, 1) は fork と execve をまたいで引き継がれる。計算の結果は変わらない(メモリの割り当て方だけ)。
"""
import ctypes
import os
import sys

PR_SET_THP_DISABLE = 41

if len(sys.argv) < 2:
    raise SystemExit(__doc__)
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0:
    print('[nothp] prctl に失敗しました(errno %d)。そのまま続けます' % ctypes.get_errno(), file=sys.stderr, flush=True)
os.execvp(sys.argv[1], sys.argv[1:])
