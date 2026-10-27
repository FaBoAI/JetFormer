"""反証 7-1 回目・safety: exp_glitch の中立止めの側。リカバリーの 'brake' の途中にカメラが 0.2〜0.4 秒止まり(中立止め)、
その間の車両の値の 1 サンプルだけ eRPM が後ろへ 300 を超える(車は動かない)とき。ON は明けたフレームで hold_back_s > 0
(veh_unseen_back は隣り合うサンプルのどちらかが 300 を超えれば、その間を後ろへ動いていた秒に数える)→ _after_hold の
back_hold で 'gap' を飛ばして 'reverse' → 見た後退 > 0 なので CHECK_S のやり直しも効かない。OFF は中立止めで Recovery を戻す。
場面 rev(前の壁 2.0 m・モデルは前進 120 µs)。出力 out/exp_glitch2.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rv7  # noqa: E402
from grid_onoff import first_rec, scene  # noqa: E402
from exp_spring import summ  # noqa: E402


def main():
    rv7.check_now()
    out = {'cells': {}}
    p = rv7.base_params()
    try:
        ng = rv7.run('now', rv7.ON(p), scene('rev', dur=12.0))
        tb = first_rec(ng)['t_brake']
        out['t_brake'] = tb
        out['ng_on'] = summ(ng)
        for a0 in (0.0, 0.03, 0.06, 0.1):
            for L in (0.2, 0.3, 0.4):
                a = (round(tb + a0, 3), round(tb + a0 + L, 3))
                for k in range(1, 8):
                    tg = round(round((a[0] + 0.02 * k) / 0.02) * 0.02, 2)
                    if tg >= a[1]:
                        continue
                    sc = scene('rev', dur=12.0, drops=[a])
                    sc['glitch'] = [(tg, 1322)]
                    sc0 = scene('rev', dur=12.0, drops=[a])
                    row = {'on': summ(rv7.run('now', rv7.ON(p), sc)), 'off': summ(rv7.run('now', rv7.OFF(p), sc)),
                           'on_noglitch': summ(rv7.run('now', rv7.ON(p), sc0))}
                    key = 'A=%s tg=%.2f' % (a, tg)
                    out['cells'][key] = row
                    print(key, {k2: (m['back'], m['esc_push'], m['front_push'], m['guards'][:2]) for k2, m in row.items()},
                          flush=True)
    finally:
        rv7.cleanup()
    (HERE / 'out' / 'exp_glitch2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
