#!/usr/bin/env python3
"""V-drive: v_replay.py をガベージコレクションの計測つきで回す(遅いフレームが GC かどうか)"""
import gc
import runpy
import sys
import time

log = []
t = {}


def cb(phase, info):
    if phase == 'start':
        t['s'] = time.perf_counter()
    else:
        log.append((info['generation'], (time.perf_counter() - t['s']) * 1000.0))


gc.callbacks.append(cb)
sys.argv = ['v_replay.py'] + sys.argv[1:]
runpy.run_path('v_replay.py', run_name='__main__')
big = sorted(log, key=lambda x: -x[1])[:5]
print('gc collections', len(log), 'gen2', sum(1 for g, _ in log if g == 2), 'slowest', [(g, round(ms, 1)) for g, ms in big])
