"""全部の走行を読み、pickle に置く(再生の台本が読み直さないため)。読むだけ。
    jf-heavy --lock none --mem 1500M -- python -I rr_cache.py"""
import os
import pickle
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_common as C  # noqa: E402

CACHE = os.path.join(C.HERE, 'cache_runs.pkl')


def build():
    t0 = time.time()
    runs, skipped = {}, {}
    for log in C.all_logs():
        try:
            d = C.load_run(log)
        except Exception as e:  # noqa: BLE001
            skipped[log] = repr(e)[:200]
            continue
        if d is None:
            skipped[log] = 'フレームなし'
            continue
        mv = d['ai'] & (d['v'] > 0.5)
        if d['ai_s'] < 3.0 or mv.sum() < 60:
            skipped[log] = 'AI %.1f 秒・動いた AI のフレーム %d' % (d['ai_s'], mv.sum())
            continue
        runs[log] = d
    with open(CACHE, 'wb') as f:
        pickle.dump({'runs': runs, 'skipped': skipped, 'built_s': time.time() - t0}, f)
    return runs, skipped


def load():
    with open(CACHE, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    runs, skipped = build()
    for k, d in runs.items():
        print(k, d['session'], d['no'], 'ai_s=%.1f' % d['ai_s'], 'tg=%.2f' % d['tg'], 'laps=%d' % len(d['laps']),
              'coll=%d' % len(d['coll']), 'rej=%d' % len(d['rejected']))
    for k, v in skipped.items():
        print('skip', k, v)
