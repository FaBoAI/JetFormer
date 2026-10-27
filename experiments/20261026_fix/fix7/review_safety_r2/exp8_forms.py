"""反証 7-2 回目・safety(直さない・装置に触れない・GPU なし・API なし)。exp8_align の形を、回 1〜4 の実際の組み合わせで流す。
形: kai1(基本)・kai2(天井 + 自動合わせ・左 4.2)・kai3(天井 + 発進の前進)・kai4(発進の前進・天井 OFF・左 4.1)。
場面 rev・'brake' の始めから 0〜0.04 秒に 0.10・0.12 秒のカメラの止まり、止まりのあと 2 番目のフレームに跳ね 1322、その 3・6
フレームあとに 2 つ目。on(今・両方 1)・off(今・両方 0)・f7b_on(fix7 の直す前・両方 1)。あわせて、スイッチ全部 OFF の今と
fix7 の直す前の写しがビット単位で同じか(h8.same_run: PCA9685 に書いた値・記録の行・出来事)。
出力 out/exp8_forms.json
    python -I exp8_forms.py
"""
import collections
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RV8_TAG', 'forms')
sys.path.insert(0, str(HERE))
import h8  # noqa: E402
from g8 import first_rec, scene  # noqa: E402
from exp8_align import frames_after, sample_for, summ  # noqa: E402

FORMS = {'kai1': {}, 'kai2': {'speed_ceiling': 1, 'speed_auto': 1, 'ceil_left_mps': 4.2},
         'kai3': {'speed_ceiling': 1, 'launch_boost': 1}, 'kai4': {'launch_boost': 1}}


def main():
    out = {'cells': {}}
    cnt = collections.Counter()
    t_start = time.time()
    try:
        for form, kw in FORMS.items():
            p = h8.base_params(**kw)
            for vdt in (None, 0.01):
                base = scene('rev', dur=9.0)
                if vdt:
                    base['veh_dt'] = vdt
                st = 0.01 if vdt else 0.02
                tb = first_rec(h8.run('now', h8.OFF(p), base))['t_brake']
                for a0 in (0.0, 0.04):
                    for aL in (0.10, 0.12):
                        a = (round(tb + a0, 3), round(tb + a0 + aL, 3))
                        fr = frames_after(a[1], 12)
                        for j2 in (3, 6):
                            gl = [(sample_for(fr[2], st), 1322), (sample_for(fr[2 + j2], st), 1322)]
                            sc = dict(base, drops=[a], glitch=gl)
                            key = '%s vdt=%s tb=%.2f A=%s j2=%d' % (form, vdt, tb, a, j2)
                            r_on = h8.run('now', h8.ON(p), sc)
                            r_off = h8.run('now', h8.OFF(p), sc)
                            r_f7b = h8.run('f7b', h8.ON(p), sc)
                            r_off_f7b = h8.run('f7b', h8.OFF(p), sc)
                            same, why = h8.same_run(r_off, r_off_f7b)
                            row = {'on': summ(r_on), 'off': summ(r_off), 'f7b_on': summ(r_f7b), 'off_same_f7b': same,
                                   'off_diff': why, 'glitch': gl}
                            out['cells'][key] = row
                            cnt['n'] += 1
                            cnt['off_same' if same else 'off_DIFF'] += 1
                            for v in ('on', 'off', 'f7b_on'):
                                m = row[v]
                                if (m['back'] or 0.0) < 0.1 and (m['esc_push'] or 0) > 0.05:
                                    cnt[v + '_fail'] += 1
                            print(key, {v: (row[v]['back'], row[v]['esc_push'], row[v]['all_guards'][:3])
                                        for v in ('on', 'off', 'f7b_on')}, 'off_same', same, flush=True)
    finally:
        h8.cleanup()
    out['count'] = dict(cnt)
    out['elapsed_s'] = round(time.time() - t_start, 1)
    (HERE / 'out' / 'exp8_forms.json').write_text(json.dumps(out, ensure_ascii=False, default=str))
    print(json.dumps(out['count'], ensure_ascii=False), out['elapsed_s'], flush=True)


if __name__ == '__main__':
    main()
