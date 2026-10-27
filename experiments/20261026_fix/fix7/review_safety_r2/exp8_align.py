"""反証 7-2 回目・safety(直さない・装置に触れない・GPU なし・API なし)。exp8_hold.py の続き。
跳ねをフレームが拾う時刻にそろえる(フレームは cam0 0.15 + k/30 秒。そのフレームが使う車両の値 = その時刻より前の最新のサンプル)。
くみ:
  A  'brake' の始め tb から a0 秒に aL 秒のカメラの止まり(hold('カメラの映像が来ていません') が中立を書く。aL が短いと中立は
     0.1 秒より短い = 閉じたループの偽物の ESC はまだ下がれない)
  S  止まりのあとの j1 番目のフレームに跳ね 1 つ目(1322)、その j2 フレームあとに 2 つ目(1322)。j2 = 0 は 1 つだけ(対照)
比べ: on(今・両方 1)・off(今・両方 0)・f7b_on(fix7 の直す前・両方 1)・on_nohold(今・両方 1・同じ跳ねで止まりなし)
出力 out/exp8_align_<esc>_<vdt>.json
    python -I exp8_align.py <orig|human> <50|100> [part/parts]
"""
import collections
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RV8_TAG', 'align_%s' % '_'.join(sys.argv[1:4]).replace('/', '-'))
sys.path.insert(0, str(HERE))
import h8  # noqa: E402
from g8 import first_rec, scene  # noqa: E402

ESC = {'orig': None, 'human': (0.12, 0.10)}
CAM0 = 0.15


def frames_after(t, n):
    """t より後のフレームの撮像時刻 n 個"""
    k = int(math.floor((t - CAM0) * 30)) + 1
    out = []
    while len(out) < n:
        tc = CAM0 + k / 30.0
        if tc > t + 1e-9:
            out.append(tc)
        k += 1
    return out


def sample_for(tc, st):
    """撮像時刻 tc のフレームが使う車両の値のサンプルの時刻(st 刻み。tc 以前の最新。drive.assemble_inputs は t <= cam_t)"""
    return round(math.floor((tc + 1e-6) / st) * st, 2)


def summ(res, t_from=1.5):
    m = first_rec(res, t_from)
    sim, rec, pca, ph = res
    m['all_guards'] = [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']
    m['rev_on_t'] = next((t for t, r, c in sim.car.revlog if r), None)
    return m


def cells(esc, vdt, tb):
    st = 0.01 if vdt == 100 else 0.02
    for a0 in (0.0, 0.04):
        for aL in (0.10, 0.12, 0.14, 0.16):
            a = (round(tb + a0, 3), round(tb + a0 + aL, 3))
            fr = frames_after(a[1], 16)
            for j1 in (1, 2, 3):
                for j2 in (0, 1, 2, 3, 5, 8):
                    t1 = sample_for(fr[j1], st)
                    gl = [(t1, 1322)]
                    if j2:
                        gl.append((sample_for(fr[j1 + j2], st), 1322))
                    yield 'A=%s j1=%d j2=%d' % (a, j1, j2), a, gl


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
        tb = first_rec(h8.run('now', h8.OFF(p), base))['t_brake']
        out['t_brake'] = tb
        for i, (key, a, gl) in enumerate(cells(esc, vdt, tb)):
            if i % pn != pi:
                continue
            sc = dict(base, drops=[a], glitch=gl)
            row = {'on': summ(h8.run('now', h8.ON(p), sc)),
                   'off': summ(h8.run('now', h8.OFF(p), sc)),
                   'f7b_on': summ(h8.run('f7b', h8.ON(p), sc)),
                   'on_nohold': summ(h8.run('now', h8.ON(p), dict(base, glitch=gl)))}
            row['glitch'] = gl
            out['cells'][key] = row
            cnt['n'] += 1
            j2 = int(key.split('j2=')[1])
            for v in ('on', 'off', 'f7b_on', 'on_nohold'):
                m = row[v]
                b = m['back'] if m['back'] is not None else 0.0
                if b < 0.1 and (m['esc_push'] or 0) > 0.05:
                    cnt['%s_fail_%s' % (v, 'one' if j2 == 0 else 'two')] += 1
            if i % 12 == 0:
                print(i, key, gl, {v: (row[v]['back'], row[v]['esc_push'], row[v]['all_guards'][:3])
                                   for v in ('on', 'off', 'f7b_on', 'on_nohold')}, flush=True)
    finally:
        h8.cleanup()
    out['count'] = dict(cnt)
    out['elapsed_s'] = round(time.time() - t_start, 1)
    suf = '' if pn == 1 else '_%d' % pi
    (HERE / 'out' / ('exp8_align_%s_%d%s.json' % (esc, vdt, suf))).write_text(json.dumps(out, ensure_ascii=False, default=str))
    print(json.dumps(out['count'], ensure_ascii=False), out['elapsed_s'], flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:4])
