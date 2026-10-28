"""後退の終わりのカメラの止まり(exp12 ov の coast+0)を、今の両方 ON・見張りだけ ON(rec_hold_keep 0)・両方 OFF と、前の版の両方 ON
(写し r1 = fix7 r2 の前・f7b = fix7 の前・f6b・f5b)で並べる(どの版からある形か・どのスイッチの形か)。ESC の模型 3 つ × 止まり 3・5 フレーム ×
場面 rev。下がった距離・後ろへの最高速・後ろ 1.0・1.2・1.4 m の物に当たる速さ・書いた値の並び(止まりの前後)。
    python -I -B ovcheck12.py → out/ovcheck12.json
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RS82_TAG', 'ovcheck')
sys.path.insert(0, str(HERE))
import h12  # noqa: E402
import exp12 as E  # noqa: E402

h11 = h12.h11


def main():
    h11.check_now()
    out = {'sha': h11.check_now(), 'rows': []}
    try:
        for esc in ('cl', 'fit', 'human'):
            bsd = {'base': 'rev', 'esc': esc, 'form': 'kai1'}
            p = E.params_of(bsd)
            tl = E.timeline(h12.run(h12.ON(p), E.build(bsd))[3])
            c = E.cam_of(tl['coast'])
            for n in (0, 3, 5):
                sd = dict(bsd, pert=[('drop', c, n)] if n else [])
                row = {'esc': esc, 'n': n, 'coast_cam': c}
                for name, ver, prm in (('on_now', 'now', h12.ON(p)), ('guard_only_now', 'now', dict(p, rec_back_guard=1, rec_hold_keep=0)),
                                       ('off_now', 'now', h12.OFF(p)), ('on_r1', 'r1', h12.ON(p)), ('on_f7b', 'f7b', h12.ON(p)),
                                       ('on_f6b', 'f6b', h12.ON(p)), ('on_f5b', 'f5b', h12.ON(p))):
                    res = h12.run(prm, E.build(sd), ver=ver)
                    m = E.metr(res, dict(sd, human_any=False))
                    w = [(round(t, 3), th) for t, s, th in res[2].w if c - 0.1 <= t <= c + 0.45]
                    row[name] = {'back': m['back'], 'vback': m['vback'], 'hit_1.0_1.2_1.4': [m['prof'][k] for k in ('1.0', '1.2', '1.4')],
                                 'writes': w}
                out['rows'].append(row)
                print(esc, n, {k: (v['back'], v['vback'], v['hit_1.0_1.2_1.4']) for k, v in row.items()
                               if isinstance(v, dict)}, flush=True)
    finally:
        h12.cleanup()
        (HERE / 'out' / 'ovcheck12.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
