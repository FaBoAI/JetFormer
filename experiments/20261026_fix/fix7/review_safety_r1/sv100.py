"""反証 7-1 回目・safety: 0.15 秒より短い車両の値の止まり(fix6 r1 9 章・r2 4 章の残り)を、本物と同じ 100 Hz の車両の値で、
本物の記録に出た長さ(サンプルの間が最大 0.035 秒 = 2 個まで抜ける。out/scan_veh_gaps.json)と、出ていない長さ(0.04〜0.12 秒)で
振る。場面 rev・brake × フレームの位相 6 通り × 止まりの始め(偽物の ESC が後退に入る頃の前後)。ON(両方 1)と OFF(両方 0)。
出力 out/sv100.json"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rv7  # noqa: E402
from grid_onoff import first_rec, scene  # noqa: E402


def main():
    rv7.check_now()
    p = rv7.base_params()
    out = {}
    try:
        for kind in ('brake', 'rev'):
            for cam0 in (0.15, 0.1567, 0.1633, 0.17, 0.1767, 0.1833):
                sc0 = scene(kind)
                sc0['veh_dt'] = 0.01
                sc0['cam0'] = cam0
                ng_on = first_rec(rv7.run('now', rv7.ON(p), sc0))
                ng_off = first_rec(rv7.run('now', rv7.OFF(p), sc0))
                tb = ng_on['t_brake']
                key0 = '%s cam0=%.4f' % (kind, cam0)
                out[key0] = {'ng_on': ng_on['back'], 'ng_off': ng_off['back'], 'cells': {}}
                t_on = tb + (0.08 if kind == 'brake' else 0.48)        # 偽物の ESC が後退に入る頃('brake' の始め / 'reverse' の始め + 0.08)
                for L in (0.01, 0.02, 0.04, 0.08, 0.12):
                    for k in range(-5, 26, 1):
                        a = round(t_on + 0.01 * k + 0.005, 3)
                        sc = dict(sc0, veh_skip=[(a, round(a + L, 3))])
                        on = first_rec(rv7.run('now', rv7.ON(p), sc))
                        off = first_rec(rv7.run('now', rv7.OFF(p), sc))
                        out[key0]['cells']['L=%.2f a=%.3f' % (L, a)] = (on['back'], off['back'], on['rear_v'], off['rear_v'])
                print(key0, 'ng', ng_on['back'], ng_off['back'],
                      {L: (max(v[0] or 0 for k, v in out[key0]['cells'].items() if k.startswith('L=%.2f' % L)),
                           max(v[1] or 0 for k, v in out[key0]['cells'].items() if k.startswith('L=%.2f' % L)))
                       for L in (0.01, 0.02, 0.04, 0.08, 0.12)}, flush=True)
    finally:
        rv7.cleanup()
    (HERE / 'out' / 'sv100.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
