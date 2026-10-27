"""反証 6-2・安全: tf62 の 1 場面を now1・now0(keep 0)・off(見張り 0・keep 0)・bef1・ng1・ng0 で回し、t_brake(ng1)から
4 秒の間に下がった最大の距離(前の壁から)・後ろの当たり・escape の始め・PCA の後退側の最後の時刻を並べる(読むだけ)。
    python -I cmp62.py <tf62 の json> <run> [esc] [rear_m を足す/変える]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c62 as K          # noqa: E402
import tf62 as T         # noqa: E402
E = K.E
from test_launch_drive import N   # noqa: E402


def main():
    f, run = sys.argv[1], int(sys.argv[2])
    esc = sys.argv[3] if len(sys.argv) > 3 else 'fake'
    rear_over = sys.argv[4] if len(sys.argv) > 4 else None
    if esc == 'slow':
        T.ESC.update(neu=0.12, rdelay=0.10)
    ex = next(x for x in json.load(open(f))['examples'] if x['run'] == run)
    p = E.P1(**ex['params'])
    sc0 = T.scene(ex['kind'], ex['xf'], ex['mthr'], ex['bd'])
    rear = ex['rear'] if rear_over is None else (None if rear_over == 'none' else float(rear_over))
    if rear is not None:
        sc0['xr'] = ex['xf'] - rear
    sc = dict(sc0, veh_skip=[tuple(x) for x in ex['veh_skip']], drops=[tuple(x) for x in ex['drops']])
    tb = ex['res']['ng1']['t_brake']
    out = {}
    try:
        for nm, mod, pp, s in (('now1', 'now', p, sc), ('now0', 'now', dict(p, rec_hold_keep=0), sc),
                               ('off', 'now', dict(p, rec_hold_keep=0, rec_back_guard=0), sc),
                               ('bef1', 'fix6_before', p, sc), ('ng1', 'now', p, sc0),
                               ('ng0', 'now', dict(p, rec_hold_keep=0), sc0)):
            sim, rec, pca = T.run(mod, pp, s)
            xs = [(t, x, v) for t, x, v, th in sim.log if tb - 0.3 <= t <= tb + 4.0]
            back = round(ex['xf'] - min(x for t, x, v in xs), 3)
            vmin = round(min(v for t, x, v in xs), 2)
            rears = [h for h in sim.car.hits if h[1] == 'rear']
            holds = sum(1 for t, r in rec.rows if r['hold'])
            out[nm] = {'back_m': back, 'v_back_max': vmin, 'rear': rears[:3], 'holds': holds,
                       'guards': E.guards(rec)[:4]}
            print(nm, out[nm], flush=True)
    finally:
        K.cleanup()


if __name__ == '__main__':
    main()
