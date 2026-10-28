"""fix8 係 safety-decided: 閉じたループ(本物の drive._loop。h10)で、発進から最初の周の終わり(AI 側にしてから 10 秒)までを
ON(rec_back_guard 1・rec_hold_keep 1)と OFF(両方 0)で同じ場面に並べ、PCA9685 に書いた値・記録の行(AI_CMD の全部の欄)・
出来事をフレームごとに比べる。回 3 の形(速さ 0.90・ToF 1・rev_s 0.4・rev_us 300・左 4.1・start_kick 1)× launch_boost 0 / 1。

    python -I exp10.py <grid> [part/parts]      → out/exp10_<grid>[_part].json
grid:
  plain    乱れなし。発進の形(最初の前進 m0・寄せる秒・向き・止まっている間のブレーキ)× 前の距離 × 電圧 × 天井(96 場面)
  drop     発進の途中のカメラの止まり: AI 側にしてから d0 = −0.1〜3.0 秒(0.1 刻み)に L 秒(0.12〜1.5 の 8 つ)× 最初の前進 m0
           10 / 60 × 電圧 1.0 / 0.85(1,024 場面)
  vskip    車両の値の止まり(中立止めの行が残る形): d0 × L(0.05〜0.6)(256 場面)
  glitch   eRPM の跳ね 1 つ(313・1322・1687・−2500)を AI 側の −0.3〜+3.0 秒(0.05 刻み)に(268 場面)
  burst    本物の跳ねの並び 4 つ(fix7 の burst と同じ並び)を −0.3〜+3.0 秒(0.1 刻み)に(136 場面)
  tof      ToF の欠け(0.03〜0.4 秒)を −0.3〜+3.0 秒に(170 場面)
  fuzz<seed>  でたらめな場面 150(発進の向き・前の距離・ToF の欠け・カメラの止まり・eRPM の跳ね・車両の値の止まり・段差の衝撃・
           電圧・モデルの形・天井・自動合わせ・カメラの位相)
どの場面も launch_boost 0 と 1 の両方で、ON と OFF を回す(1 場面 4 本)。
"""
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h10  # noqa: E402

N = h10.N
T_WIN = 10.0                     # 発進から最初の周の終わり(本物は約 9〜10 秒)
CORNERS = [8.0, 13.0, 19.0, 24.0, 30.0, 36.0]
TURNS = [1, 1, -1, 1, 1, -1]
# 本物の跳ねの並び(fix7 safety の burst。data/raw を読んで写した値)
BURSTS = {
    '160306-0002': [(0.0, 1322)],
    '181504-0005': [(0.0, 1115)],
    '152707-0004': [(0.0, 1628), (0.02, 1525), (0.05, 1572), (0.08, 1289)],
    '211828-0004': [(0.0, 1687), (0.02, 1411), (0.04, -1270), (0.06, 1408), (0.08, -2156), (0.1, 1356), (0.12, 1528),
                    (0.14, 1358)],
}


def scene(ai=1.0, front0=4.2, **kw):
    """front0 = 発進の位置から最初の壁までの距離 [m](車はその 0.7 m 手前で曲がり始める)"""
    corners = [round(front0 - 0.7, 3)] + [c for c in CORNERS if c > front0 + 1.0]
    sc = {'dur': ai + T_WIN + 0.3, 'xf': 1e9, 'ai_from': ai, 'corners': corners,
          'turns': [1] + TURNS[:len(corners) - 1], 'wall_gap': 0.7, 'vmax': 4.0, 'vcorner': 2.6, 'm0': 60.0, 'ramp_s': 0.8}
    sc.update(kw)
    sc['model'] = h10.make_model(sc)
    return sc


def params(lb, ceil=0, auto=0, **kw):
    p = h10.base_params(launch_boost=lb, speed_ceiling=ceil, ceil_all_mps=4.5, speed_auto=auto, **kw)
    if auto:
        p['auto_target_mps'] = 4.27
    return p


def one(sc, pkw):
    """1 場面 × launch_boost 0 / 1 × ON / OFF → 結果の dict"""
    ai = sc['ai_from']
    t_end = ai + T_WIN
    res = {}
    for lb in (0, 1):
        p = params(lb, **pkw)
        p = {k: v for k, v in p.items() if k in h10.drive.DEFAULT_PARAMS}
        ron = h10.run(h10.ON(p), sc)
        roff = h10.run(h10.OFF(p), sc)
        c = h10.compare(ron, roff, t_end)
        con, coff = h10.launch_cols(ron, ai, t_end), h10.launch_cols(roff, ai, t_end)
        res['lb%d' % lb] = {'same': c['same'], 'cmp': c, 'on': con, 'off': coff,
                            'n_glitch': ron[0].n_glitch}
    return res


def cells(grid):
    if grid == 'plain':
        for m0 in (0.0, 40.0, 80.0, 120.0):
            for ramp, steer0, brake0 in ((0.3, 0.0, 0.0), (1.2, 250.0, 0.0), (0.8, -300.0, 0.3)):
                for front0 in (2.2, 4.2):
                    for volt in (0.85, 1.1):
                        for ceil in (0, 1):
                            key = 'm0=%g ramp=%g steer0=%g brake0=%g front0=%g volt=%g ceil=%d' % (
                                m0, ramp, steer0, brake0, front0, volt, ceil)
                            yield key, scene(front0=front0, m0=m0, ramp_s=ramp, steer0=steer0, brake0=brake0, volt=volt), \
                                {'ceil': ceil}
    elif grid == 'drop':
        for m0 in (10.0, 60.0):
            for volt in (1.0, 0.85):
                for i in range(32):
                    d0 = round(-0.1 + 0.1 * i, 2)
                    for L in (0.12, 0.2, 0.3, 0.45, 0.55, 0.7, 1.0, 1.5):
                        a = round(1.0 + d0, 3)
                        yield 'm0=%g volt=%g d0=%+.2f L=%.2f' % (m0, volt, d0, L), \
                            scene(m0=m0, volt=volt, drops=[(a, round(a + L, 3))]), {}
    elif grid == 'vskip':
        for m0 in (10.0, 60.0):
            for i in range(16):
                d0 = round(-0.1 + 0.2 * i, 2)
                for L in (0.05, 0.12, 0.16, 0.2, 0.3, 0.4, 0.5, 0.6):
                    a = round(1.0 + d0, 3)
                    yield 'm0=%g d0=%+.2f L=%.2f' % (m0, d0, L), scene(m0=m0, veh_skip=[(a, round(a + L, 3))]), {}
    elif grid == 'glitch':
        for e in (313, 1322, 1687, -2500):
            for i in range(67):
                d0 = round(-0.3 + 0.05 * i, 2)
                yield 'e=%d d0=%+.2f' % (e, d0), scene(glitch=[(round(1.0 + d0, 2), e)]), {}
    elif grid == 'burst':
        for name, bs in BURSTS.items():
            for i in range(34):
                d0 = round(-0.3 + 0.1 * i, 2)
                yield '%s d0=%+.2f' % (name, d0), scene(glitch=[(round(1.0 + d0 + dt, 2), e) for dt, e in bs]), {}
    elif grid == 'tof':
        for L in (0.03, 0.08, 0.12, 0.2, 0.4):
            for i in range(34):
                d0 = round(-0.3 + 0.1 * i, 2)
                a = round(1.0 + d0, 3)
                yield 'L=%.2f d0=%+.2f' % (L, d0), scene(tof_skip=[(a, round(a + L, 3))]), {}
    elif grid.startswith('fuzz'):
        rng = random.Random(int(grid[4:]))
        for k in range(150):
            sc, pkw = fuzz_scene(rng)
            yield 'fz%d' % k, sc, pkw


def fuzz_scene(rng):
    ai = round(rng.uniform(0.6, 1.5), 3)
    front0 = round(rng.choice([rng.uniform(1.0, 2.0), rng.uniform(2.0, 4.5), rng.uniform(4.5, 8.0)]), 2)
    kw = dict(ai=ai, front0=front0, m0=round(rng.uniform(0.0, 130.0), 1), ramp_s=round(rng.uniform(0.2, 1.6), 2),
              steer0=round(rng.uniform(-380.0, 380.0), 0), dir_s=round(rng.uniform(0.2, 1.6), 2),
              brake0=rng.choice([0.0, 0.0, 0.0, round(rng.uniform(0.05, 0.4), 2)]),
              volt=round(rng.uniform(0.85, 1.12), 3), volt_model=round(rng.uniform(0.9, 1.1), 3),
              vmax=round(rng.uniform(3.2, 4.8), 2), vcorner=round(rng.uniform(1.8, 3.2), 2),
              wall_gap=round(rng.uniform(0.3, 1.2), 2), cam0=round(rng.uniform(0.13, 0.17), 4))
    span = (ai - 0.4, ai + T_WIN)

    def spans(n, lo, hi, early=0.5):
        out = []
        for _ in range(n):
            # 半分は発進の 3 秒の中に(発進の途中に入る形をよく見る)
            a = rng.uniform(ai - 0.2, ai + 3.0) if rng.random() < early else rng.uniform(*span)
            out.append((round(a, 3), round(a + rng.uniform(lo, hi), 3)))
        return sorted(out)
    kw['drops'] = spans(rng.choice([0, 0, 1, 1, 2, 3]), 0.04, 1.0)
    kw['tof_skip'] = spans(rng.choice([0, 0, 1, 2, 3]), 0.03, 0.4)
    kw['veh_skip'] = spans(rng.choice([0, 0, 0, 1, 2]), 0.02, 0.3)
    gl = []
    for _ in range(rng.choice([0, 1, 2, 4, 8])):
        t = rng.uniform(ai - 0.3, ai + 3.0) if rng.random() < 0.6 else rng.uniform(*span)
        if rng.random() < 0.3:
            name = rng.choice(sorted(BURSTS))
            gl += [(round(t + dt, 2), e) for dt, e in BURSTS[name]]
        else:
            gl.append((round(t, 2), rng.choice([301, 313, 1053, 1115, 1322, 1687, -1270, -2156, -3500])))
    kw['glitch'] = sorted(gl)
    kw['spikes'] = sorted((round(rng.uniform(ai + 0.3, ai + T_WIN), 2), round(rng.uniform(2.5, 4.5), 2))
                          for _ in range(rng.choice([0, 0, 0, 1, 2])))
    ceil = rng.choice([0, 1])
    auto = rng.choice([0, 1]) if ceil else 0
    return scene(**kw), {'ceil': ceil, 'auto': auto}


def main(grid, part='0/1'):
    pi, pn = (int(x) for x in part.split('/'))
    out = {'grid': grid, 'part': part, 'cells': {}, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    t0 = time.time()
    try:
        for i, (key, sc, pkw) in enumerate(cells(grid)):
            if i % pn != pi:
                continue
            info = {k: v for k, v in sc.items() if k != 'model'}
            r = one(sc, pkw)
            r['scene'] = info
            r['pkw'] = pkw
            out['cells'][key] = r
            if (i // pn) % 25 == 0:
                print(grid, part, i, key, {k: v['same'] for k, v in r.items() if k.startswith('lb')},
                      '%.0fs' % (time.time() - t0), flush=True)
    finally:
        h10.cleanup()
    out['elapsed_s'] = round(time.time() - t0, 1)
    suf = '' if pn == 1 else '_%d' % pi
    (HERE / 'out' / ('exp10_%s%s.json' % (grid, suf))).write_text(json.dumps(out, ensure_ascii=False, default=str))
    n = len(out['cells'])
    nd = sum(1 for r in out['cells'].values() for k in ('lb0', 'lb1') if not r[k]['same'])
    print('done', grid, part, n, 'diff', nd, out['elapsed_s'])


if __name__ == '__main__':
    main(*sys.argv[1:3])
