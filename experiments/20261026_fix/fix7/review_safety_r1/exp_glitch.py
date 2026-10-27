"""反証 7-1 回目・safety: リカバリーの 'brake'(前進で壁を押していたあとの後退側 = このESCはブレーキ。下がれない)の間に、
車両の値の 1 サンプルだけ eRPM が後ろへ 300 を超える(車は動かない)とき。本物の記録で、止まっている(hall_count が 0.1 秒
変わらない)のに eRPM が 1322・1687 などを 1 サンプル示した(out/scan_still_erpm.json。止まった直後に多い)。
rec_back_guard の back_early はそれだけで 'gap'(中立)を飛ばして 'reverse' へ進み、下がった時間 > 0 なので CHECK_S のやり直しも
効かない → このESCは中立を見ないので下がらない。OFF(両方 0)は 'brake' → 'wait' → 'gap' → 'reverse' でふつうに下がる。
場面 rev(前の壁 2.0 m・モデルは前進 120 µs で押し続ける)。回 1〜4 の形 3 通り。出力 out/exp_glitch.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rv7  # noqa: E402
from grid_onoff import first_rec, scene  # noqa: E402
from exp_spring import summ  # noqa: E402

N = rv7.N
FORMS = {'kai1': {}, 'kai2_ceil_auto': {'speed_ceiling': 1, 'speed_auto': 1, 'ceil_left_mps': 4.2},
         'kai3_launch': {'speed_ceiling': 1, 'launch_boost': 1}}


def main():
    rv7.check_now()
    out = {}
    try:
        for form, kw in FORMS.items():
            p = rv7.base_params(**kw)
            ng = rv7.run('now', rv7.ON(p), scene('rev', dur=12.0))
            tb = first_rec(ng)['t_brake']
            out[form] = {'t_brake': tb, 'ng_on': summ(ng), 'ng_off': summ(rv7.run('now', rv7.OFF(p), scene('rev', dur=12.0))),
                         'cells': {}}
            t0 = round(round(tb / 0.02) * 0.02, 2)
            for k in range(-1, 14):
                tg = round(t0 + 0.02 * k, 2)
                for e in (310, 1322):
                    sc = scene('rev', dur=12.0)
                    sc['glitch'] = [(tg, e)]
                    row = {'on': summ(rv7.run('now', rv7.ON(p), sc)), 'off': summ(rv7.run('now', rv7.OFF(p), sc))}
                    key = 'tg=%.2f e=%d' % (tg, e)
                    out[form]['cells'][key] = row
                    print(form, key, {k2: (m['back'], m['esc_push'], m['front_push'], len(m['starts']), m['giveup'], m['guards'][:2])
                                      for k2, m in row.items()}, flush=True)
    finally:
        rv7.cleanup()
    (HERE / 'out' / 'exp_glitch.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
