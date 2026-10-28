"""h10 の確かめ: 乱れなしの発進 1 つと、発進の途中のカメラの止まり 1 つで、ON(両方 1)と OFF(両方 0)を並べる(標準出力)"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h10  # noqa: E402

AI = 1.0
T_END = AI + 10.0


def scene(**kw):
    sc = {'dur': T_END + 0.2, 'xf': 1e9, 'ai_from': AI, 'corners': [3.5, 8.0, 13.0, 19.0, 24.0, 30.0],
          'turns': [1, 1, -1, 1, 1, -1], 'wall_gap': 0.7, 'vmax': 4.0, 'vcorner': 2.6, 'm0': 60.0, 'ramp_s': 0.8}
    sc.update(kw)
    sc['model'] = h10.make_model(sc)
    return sc


def main():
    try:
        for name, kw in [('plain', {}), ('plain_m0_10', {'m0': 10.0}), ('drop_0.3_at_1.6', {'drops': [(1.6, 1.9)]}),
                         ('drop_0.7_at_1.6', {'drops': [(1.6, 2.3)]}), ('glitch', {'glitch': [(1.5, 1322), (2.0, 1100)]})]:
            for lb in (0, 1):
                p = h10.base_params(launch_boost=lb)
                t0 = time.time()
                ron = h10.run(h10.ON(p), scene(**kw))
                roff = h10.run(h10.OFF(p), scene(**kw))
                cmp_ = h10.compare(ron, roff, T_END)
                print(name, 'lb', lb, 'same', cmp_['same'], 'n_pca', cmp_['n_pca'], 'rows', cmp_['n_rows'], 'ev', cmp_['n_ev'],
                      '%.2fs' % (time.time() - t0))
                print('  on ', json.dumps(h10.launch_cols(ron, AI, T_END), ensure_ascii=False, default=str))
                print('  off', json.dumps(h10.launch_cols(roff, AI, T_END), ensure_ascii=False, default=str))
                if not cmp_['same']:
                    print('  diff', json.dumps({k: cmp_[k] for k in ('pca_first', 'row_first', 'ev_first')},
                                               ensure_ascii=False, default=str)[:600])
    finally:
        h10.cleanup()


if __name__ == '__main__':
    main()
