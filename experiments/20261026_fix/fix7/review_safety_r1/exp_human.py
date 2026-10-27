"""人が CH3 を人側にしてから AI 側へ戻したとき、残したリカバリー(ON)が人のあとまで続かないか(読むだけ)。
場面 rev。リカバリーの 'reverse'・'giveup' の途中に人側の区間(カメラの止まりの中・外、0.05〜0.4 秒)を入れる。
人側のあとの最初の AI のフレームで段が None(戻った)かを ON・OFF で。出力 out/exp_human.json"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rv7  # noqa: E402
from grid_onoff import first_rec, scene  # noqa: E402


def main():
    p = rv7.base_params()
    out = {}
    try:
        ng = rv7.run('now', rv7.ON(p), scene('rev', dur=8.0))
        tb = first_rec(ng)['t_brake']
        for t_h in (0.5, 0.6, 0.75):            # 'reverse' の頃(tb + 0.5〜0.75)
            for L in (0.05, 0.08, 0.12, 0.2, 0.4):
                for drop in (None, (-0.2, 0.45)):
                    sc = scene('rev', dur=8.0)
                    h0 = round(tb + t_h, 3)
                    sc['human'] = [(h0, round(h0 + L, 3))]
                    if drop is not None:
                        sc['drops'] = [(round(h0 + drop[0], 3), round(h0 + drop[1], 3))]
                    row = {}
                    for nm, pp in (('on', rv7.ON(p)), ('off', rv7.OFF(p))):
                        sim, rec, pca, ph = rv7.run('now', pp, sc)
                        after = [(t, p0, p1) for t, p0, p1, *_ in ph if t > h0 + L]
                        first = after[0] if after else None
                        rows_ai = [((t - rv7.T0) / 1e9, int(r['assist']), float(r['out'][1])) for t, r in rec.rows
                                   if (t - rv7.T0) / 1e9 > h0 + L and not r['hold'] and r['select']][:3]
                        row[nm] = {'first_after': first, 'rows_after': rows_ai}
                    key = 'h=(%.3f,+%.2f) drop=%s' % (h0, L, drop)
                    out[key] = row
                    print(key, row, flush=True)
    finally:
        rv7.cleanup()
    (HERE / 'out' / 'exp_human.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
