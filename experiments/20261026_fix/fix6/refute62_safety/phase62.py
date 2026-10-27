"""反証 6-2・安全: tf62 で印のついた場面を、値の止まりなし(ng1)のフレームの並びの位相 10 通り(cam0 = 0.15 + k/300 秒)と
比べる。今(now1)・直す前(bef1)の値の止まりありも同じ位相 10 通りで回す。
    python -I phase62.py <tf62 の json> [esc] [runs,...]
出力: out/phase62_<種>_<esc>.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c62 as K          # noqa: E402
import tf62 as T         # noqa: E402
E = K.E


def main():
    f = sys.argv[1]
    esc = sys.argv[2] if len(sys.argv) > 2 else 'fake'
    if esc == 'slow':
        T.ESC.update(neu=0.12, rdelay=0.10)
    d = json.load(open(f))
    want = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else None
    out = {}
    try:
        for ex in d['examples']:
            if want and ex['run'] not in want:
                continue
            p = E.P1(**ex['params'])
            sc0 = T.scene(ex['kind'], ex['xf'], ex['mthr'], ex['bd'])
            if ex['rear'] is not None:
                sc0['xr'] = ex['xf'] - ex['rear']
            sc = dict(sc0, veh_skip=[tuple(x) for x in ex['veh_skip']], drops=[tuple(x) for x in ex['drops']])
            res = {'ng1': [], 'now1': [], 'bef1': []}
            for k in range(10):
                for nm, s, mod in (('ng1', sc0, 'now'), ('now1', sc, 'now'), ('bef1', sc, 'fix6_before')):
                    ss = dict(s, cam0=0.15 + k / 300.0)
                    sim, rec, pca = T.run(mod, p, ss)
                    m = T.win(sim, pca, p)
                    res[nm].append(None if m is None else (m['back_m'], m['esc_push_s'],
                                                           max([-h[2] for h in m['rear_win']], default=0.0)))
            summ = {}
            for nm, xs in res.items():
                xs = [x for x in xs if x is not None]
                summ[nm] = {'back_max': max(x[0] for x in xs), 'back_min': min(x[0] for x in xs),
                            'push_max': max(x[1] for x in xs), 'rear_v_max': max(x[2] for x in xs),
                            'rear_n': sum(1 for x in xs if x[2] > 0)}
            out[ex['run']] = {'flags': ex['flags'], 'summ': summ, 'res': res, 'veh_skip': ex['veh_skip'],
                              'drops': ex['drops'], 'params': ex['params'], 'kind': ex['kind'], 'xf': ex['xf'],
                              'rear': ex['rear']}
            print(ex['run'], ex['flags'], summ, flush=True)
    finally:
        K.cleanup()
    (HERE / 'out' / ('phase62_%d_%s.json' % (d['seed'], esc))).write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                                                                    default=str))


if __name__ == '__main__':
    main()
