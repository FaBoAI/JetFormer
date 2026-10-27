"""反証 7-2 回目・safety(直さない・装置に触れない・GPU なし・API なし)。
疑い: fix7 s4-a の EscMayReverse は READY_NEUTRAL_S = 0 なので、rec_hold_keep で残した中立止め(hold() が中立を書く)が
'brake' の間に 1 回あるだけで、ESC がまだ下がれない短い中立(閉じたループの偽物の ESC は 0.1 秒・人に寄せた ESC は 0.12 秒
要る)でも「後退で走らせていたかもしれない」(esc_rev True → esc_ran True)になり、その手順の 'brake'・'wait'・'gap' の
eRPM の跳ねを見張りがまた使う。2 つ目の跳ねが 'reverse' の CHECK_S の間に入ると s4-b の early_none も効かず、
w1-safety-rv7-safety-r1-1 と同じ「0 m・後退側を止まったまま 1.2 秒・escape が前の壁を 0.5 秒押す」に戻る。
場面 rev(前の壁 2.0 m・モデルは前進 120 µs)・回 1〜4 の形(h8.base_params)。
くみ:
  A: 'brake' の始め tb から a0 秒に aL 秒のカメラの止まり(フレームが来ない = hold('カメラの映像が来ていません'))
  S: 止まりの終わりのあと s0 秒に跳ね 1 つ目、その d 秒あとに 2 つ目(1322)・本物の跳ねの並び(exp7 の BURSTS)
比べ: on(今・両方 1)・off(今・両方 0)・f7b_on(fix7 の直す前・両方 1)・on_nohold(今・両方 1・同じ跳ねで止まりなし)
  ・on1(今・両方 1・同じ止まりで跳ね 1 つ目だけ = s4-b の early_none が効く対照)
出力 out/exp8_hold_<esc>_<vdt>.json
    python -I exp8_hold.py <orig|human> <50|100> [part/parts]
"""
import collections
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RV8_TAG', 'hold_%s' % '_'.join(sys.argv[1:4]).replace('/', '-'))
sys.path.insert(0, str(HERE))
import h8  # noqa: E402
from g8 import first_rec, scene  # noqa: E402

ESC = {'orig': None, 'human': (0.12, 0.10)}
BURSTS = {
    'b152707_0004': [1628, 0, 1525, 0, 0, 1572, 0, 0, 1289],
    'b211828_0004': [1687, 0, 0, 0, 0, 0, 1411, -1270, 1408, -2156, 1356, 0, 0, -394, 1528, 1358, 656, 385],
}


def summ(res, t_from=1.5):
    m = first_rec(res, t_from)
    sim, rec, pca, ph = res
    m['all_guards'] = [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']
    m['rev_on_t'] = next((t for t, r, c in sim.car.revlog if r), None)
    return m


def cells(esc, vdt):
    base = scene('rev', dur=9.0, esc=ESC[esc])
    if vdt == 100:
        base['veh_dt'] = 0.01
    st = 0.01 if vdt == 100 else 0.02
    tb = first_rec(h8.run('now', h8.OFF(h8.base_params()), base))['t_brake']
    print('t_brake', tb, flush=True)
    for a0 in (0.0, 0.03, 0.06):
        for aL in (0.10, 0.12, 0.14, 0.17):
            a = (round(tb + a0, 3), round(tb + a0 + aL, 3))
            for s0 in (0.02, 0.06):
                t1 = round(round((a[1] + s0) / st) * st, 2)
                for d in (0.04, 0.1, 0.2, 0.3):
                    t2 = round(round((t1 + d) / st) * st, 2)
                    yield 'A=%s s1=%.2f s2=%.2f' % (a, t1, t2), a, [(t1, 1322), (t2, 1322)]
                if vdt == 100:
                    for bn, seq in BURSTS.items():
                        gl = [(round(t1 + 0.01 * i, 2), e) for i, e in enumerate(seq) if e != 0]
                        yield 'A=%s %s t1=%.2f' % (a, bn, t1), a, gl
    print('base', base, flush=True)


def main(esc, vdt, part='0/1'):
    vdt = int(vdt)
    pi, pn = (int(x) for x in part.split('/'))
    out = {'esc': esc, 'vdt': vdt, 'cells': {}}
    cnt = collections.Counter()
    p = h8.base_params()
    t_start = time.time()
    try:
        base = scene('rev', dur=9.0, esc=ESC[esc])
        if vdt == 100:
            base['veh_dt'] = 0.01
        for i, (key, a, gl) in enumerate(cells(esc, vdt)):
            if i % pn != pi:
                continue
            sc = dict(base, drops=[a], glitch=gl)
            row = {'on': summ(h8.run('now', h8.ON(p), sc)),
                   'off': summ(h8.run('now', h8.OFF(p), sc)),
                   'f7b_on': summ(h8.run('f7b', h8.ON(p), sc)),
                   'on_nohold': summ(h8.run('now', h8.ON(p), dict(base, glitch=gl))),
                   'on1': summ(h8.run('now', h8.ON(p), dict(base, drops=[a], glitch=gl[:1])))}
            out['cells'][key] = row
            cnt['n'] += 1
            for v, m in row.items():
                b = m['back'] if m['back'] is not None else 0.0
                if b < 0.1 and (m['esc_push'] or 0) > 0.05:
                    cnt[v + '_fail'] += 1
                if m['rear']:
                    cnt[v + '_rear'] += 1
            if i % 10 == 0:
                print(i, key, {v: (m['back'], m['esc_push'], m['all_guards'][:3]) for v, m in row.items()}, flush=True)
    finally:
        h8.cleanup()
    out['count'] = dict(cnt)
    out['elapsed_s'] = round(time.time() - t_start, 1)
    suf = '' if pn == 1 else '_%d' % pi
    (HERE / 'out' / ('exp8_hold_%s_%d%s.json' % (esc, vdt, suf))).write_text(json.dumps(out, ensure_ascii=False, default=str))
    print(json.dumps(out['count'], ensure_ascii=False), out['elapsed_s'], flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:4])
