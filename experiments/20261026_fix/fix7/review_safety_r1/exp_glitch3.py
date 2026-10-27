"""exp_glitch と同じ(1 サンプルの eRPM の跳ね)を、本物と同じ 100 Hz の車両の値・人に寄せた偽物の ESC(中立 0.12 秒・後退側から
0.10 秒)で。跳ねの値は本物の記録の 1322・313(10/10 181504-0005 のリカバリーの 'brake' のあと)・301。出力 out/exp_glitch3.json"""
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
    p = rv7.base_params()
    out = {'cells': {}}
    hit = {'on': 0, 'off': 0, 'n': 0}
    try:
        base = scene('rev', dur=12.0, esc=(0.12, 0.10))
        base['veh_dt'] = 0.01
        ng = rv7.run('now', rv7.ON(p), base)
        tb = first_rec(ng)['t_brake']
        out['t_brake'] = tb
        out['ng_on'] = summ(ng)
        out['ng_off'] = summ(rv7.run('now', rv7.OFF(p), base))
        print('t_brake', tb, 'ng_on', out['ng_on']['back'], 'ng_off', out['ng_off']['back'], flush=True)
        for k in range(0, 30):
            tg = round(round(tb, 2) + 0.01 * k, 2)
            for e in (301, 313, 1322):
                sc = dict(base, glitch=[(tg, e)])
                row = {'on': summ(rv7.run('now', rv7.ON(p), sc)), 'off': summ(rv7.run('now', rv7.OFF(p), sc))}
                out['cells']['tg=%.2f e=%d' % (tg, e)] = row
                hit['n'] += 1
                hit['on'] += int((row['on']['back'] or 0) < 0.1)
                hit['off'] += int((row['off']['back'] or 0) < 0.1)
                if (row['on']['back'] or 0) < 0.1:
                    print('tg=%.2f e=%d' % (tg, e), {q: (m['back'], m['esc_push'], m['front_push'], m['guards'][:2]) for q, m in row.items()},
                          flush=True)
    finally:
        rv7.cleanup()
    out['count_back_lt_0.1'] = hit
    print(hit)
    (HERE / 'out' / 'exp_glitch3.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
