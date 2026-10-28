"""反証 8-1 回目・係 safety: 本物の drive._loop の閉じたループ(h11)で、両方 ON(rec_back_guard 1・rec_hold_keep 1)と
両方 OFF(0・0)を同じ場面で並べ、「ON のほうが OFF より悪い」場面を数える。全部 OFF は写し f7b(fix7 の前)の OFF とも
ビット単位で比べる。直さない・装置に触れない・GPU なし・API なし。

場面(前の係が見ていない組み合わせを主に):
  lw    発進 → 前の壁(HS から 1.1〜5 m)。回 1〜4 の形(発進の前進 0/1・start_kick・天井・自動合わせ・左 4.1/4.2)×
        モデル(ずっと前進 / 壁の手前でブレーキ側 = 10/23 の形 / 壁の手前で中立)× ESC の形(中立 0.02〜0.3 秒・人に寄せた・
        後退に入らない)× 後ろの物 × 50/100 Hz。乱れなし
  fz    でたらめ: lw の形に、eRPM の跳ね(1 つ・本物の並び)・カメラの止まり(当たりに続くものも)・ToF / IMU / 車両の値の欠け・
        跳ね返り・後ろ向きの坂・段差の衝撃・人側の区間(中立のまま / 人の後退 / 人の前進)を足す。長さ 10〜22 秒(リカバリーの続き)
出力 out/exp11_<kind>_<part>.json。一時の場所は fix8/tmp_review_safety_r1(終わったら消す)
    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -I -B experiments/20261026_fix/fix8/review_safety_r1/exp11.py <kind> <i/n> [seed] [count]
"""
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RS8_TAG', '_'.join(sys.argv[1:3]).replace('/', 'of') or 'main')
sys.path.insert(0, str(HERE))
import h11  # noqa: E402

C, N, A = h11.C, h11.N, h11.A
ESCS = {'orig': None, 'human': (0.12, 0.10), 'n002': (0.02, 0.08), 'n005': (0.05, 0.08), 'n020': (0.2, 0.08),
        'n030': (0.3, 0.08), 'norev': 'norev', 'n002r': (0.02, 0.02), 'humanr': (0.12, 0.02)}
BURSTS = {   # 本物の跳ねの並び(fix7 safety の burst と同じ。100 Hz のサンプルの並び)
    'b160306_0002': [1322],
    'b181504_0005': [1115],
    'b152707_0004': [1628, 0, 1525, 0, 0, 1572, 0, 0, 1289],
    'b211828_0004': [1687, 0, 0, 0, 0, 0, 1411, -1270, 1408, -2156, 1356, 0, 0, -394, 1528, 1358, 656, 385],
}


def model_from(spec):
    """spec: {'kind': const|brakewall|neutwall|react, 'm': 前進 µs, 'xf': 前の壁, 'db': 壁の手前でブレーキ・中立にする距離}"""
    kind, m = spec['kind'], float(spec['m'])
    xf, db = float(spec['xf']), float(spec.get('db', 0.45))
    if kind == 'const':
        return (C, N - m)
    if kind == 'brakewall':
        def f(t, car):
            return (C, N - m) if car.x < xf - db else (C, N + 200.0)
        return f
    if kind == 'neutwall':
        def f(t, car):
            return (C, N - m) if car.x < xf - db else (C, N)
        return f
    if kind == 'react':       # 当たった直後の 0.4 秒はブレーキ側(モデルが止まろうとする)、そのあと前進
        st = {'t_hit': None}

        def f(t, car):
            if car.contact == 'front' and st['t_hit'] is None:
                st['t_hit'] = t
            if st['t_hit'] is not None and st['t_hit'] <= t < st['t_hit'] + 0.4:
                return (C, N + 150.0)
            if car.contact != 'front' and st['t_hit'] is not None and t >= st['t_hit'] + 0.4:
                st['t_hit'] = None
            return (C, N - m)
        return f
    raise KeyError(kind)


def human_from(spec, a):
    """人側の区間の操作(区間の始め a から)。neutral / back(ブレーキ 0.27 → 中立 0.12 → 後退 +216 を 0.3 → 中立)/ fwd"""
    if spec in (None, 'neutral'):
        return N
    if spec == 'back':
        def f(t):
            el = t - a
            if el < 0.27:
                return N + 200.0
            if el < 0.39:
                return N
            if el < 0.69:
                return N + 216.0
            return N
        return f
    if spec == 'fwd':
        return N - 80.0
    raise KeyError(spec)


def build(sd):
    """場面の説明(JSON にできる dict)→ h11 の scene"""
    sc = {'dur': sd['dur'], 'xf': sd['xf'], 'ai_from': sd.get('ai_from', 1.0),
          'model': model_from(dict(sd['model'], xf=sd['xf'])), 'veh_dt': sd.get('veh_dt', 0.02)}
    e = ESCS[sd.get('esc', 'orig')]
    if e == 'norev':
        sc['no_rev'] = True
    elif e is not None:
        sc['esc'] = tuple(e)
    if sd.get('rear') is not None:
        sc['xr'] = sd['xf'] - sd['rear']
    for k in ('bounce', 'slope', 'slope_after_hit', 'coast', 'lag', 'glitch', 'drops', 'veh_skip', 'tof_skip', 'imu_skip', 'spikes', 'hit_drop',
              'cam0'):
        if sd.get(k) is not None:
            if k == 'hit_drop':
                sc[k] = tuple(sd[k])
            else:
                sc[k] = [tuple(x) for x in sd[k]] if isinstance(sd[k], list) else sd[k]
    if sd.get('human'):
        spans = [tuple(x[:2]) for x in sd['human']]
        sc['human'] = spans
        pats = [(x[0], x[1], human_from(x[2], x[0])) for x in sd['human']]

        def hthr(t):
            for a, b, f in pats:
                if a <= t < b:
                    return f(t) if callable(f) else f
            return N
        sc['human_thr'] = hthr
    return sc


def params_of(sd):
    return h11.base_params(**h11.FORMS[sd['form']])


def worse(mon, moff):
    """ON が OFF より悪い印(数で)"""
    w = []
    if mon['rear_n'] > moff['rear_n']:
        w.append('rear_more')
    if mon['rear_max_v'] > moff['rear_max_v'] + 0.1:
        w.append('rear_faster')
    if mon['front_push_s'] > moff['front_push_s'] + 0.3:
        w.append('front_push_more')
    if mon['zero_push'] > moff['zero_push']:
        w.append('zero_push')
    if mon['front_max_v'] > moff['front_max_v'] + 0.3:
        w.append('front_faster')
    if mon['rear_push_s'] > moff['rear_push_s'] + 0.1:
        w.append('rear_push_more')
    if mon['giveup_t'] is not None and moff['giveup_t'] is None:
        w.append('giveup_only_on:' + ','.join(sorted({g[0] for g in mon['guards'] if g[0] == 'hold_long'})))
    return w


def slim(m):
    keep = ('front_push_s', 'rear_push_s', 'rev_side_s', 'rev_dead_s', 'rear', 'rear_n', 'rear_max_v', 'front_n',
            'front_max_v', 'giveup_t', 'min_x', 'x_end', 'v_end', 'recoveries', 'holds', 'zero_push', 'launch_frames',
            'kick_frames')
    out = {k: m[k] for k in keep}
    out['guards'] = m['guards'][:12]
    out['segs'] = [{k: s.get(k) for k in ('t0', 't1', 'x0', 'back', 'vmin', 'esc_push', 'rev_dead_s', 'rev_side_s', 'at_wall')} for s in m['segs'][:8]]
    return out


def one(sd, idcheck=True):
    sc = build(sd)
    p = params_of(sd)
    ron = h11.run('now', h11.ON(p), sc)
    sc = build(sd)
    roff = h11.run('now', h11.OFF(p), sc)
    t_from = sd.get('ai_from', 1.0)
    mon, moff = h11.metrics(ron, t_from), h11.metrics(roff, t_from)
    row = {'on': slim(mon), 'off': slim(moff), 'worse': worse(mon, moff),
           'same_on_off': h11.same_run(ron, roff)[0]}
    if idcheck:
        sc = build(sd)
        rb = h11.run('f7b', h11.OFF(p), sc)
        ok, why = h11.same_run(roff, rb)
        row['off_eq_f7b'] = ok
        if not ok:
            row['off_f7b_diff'] = str(why)[:300]
    return row


def gen_lw():
    for form, xf, mk, m, esc, rear, vdt in itertools.product(
            list(h11.FORMS), (1.1, 2.0, 3.0), ('const', 'brakewall', 'react'), (40.0, 120.0, 200.0),
            list(ESCS), (None, 1.2), (0.01,)):
        if mk == 'brakewall' and m == 40.0:
            continue
        yield {'kind': 'lw', 'form': form, 'xf': xf, 'model': {'kind': mk, 'm': m}, 'esc': esc, 'rear': rear,
               'veh_dt': vdt, 'dur': 9.0}


def gen_rd():
    """10/23 の形(壁の手前でブレーキ側 → 止まって ESC が後退を待つ → リカバリー)× 後ろの物の距離(壁から)× ESC × 50/100 Hz"""
    for coast, rear, esc, m in itertools.product((1.5, 0.3, 3.0), [round(0.5 + 0.1 * k, 2) for k in range(20)] + [None],
                                                 ('orig', 'human', 'n002', 'n005', 'n020'), (120.0, 200.0)):
        yield {'kind': 'rd', 'form': 'kai1', 'xf': 2.4, 'model': {'kind': 'brakewall', 'm': m, 'db': 0.45}, 'esc': esc,
               'rear': rear, 'veh_dt': 0.01, 'dur': 5.5, 'coast': coast}


def gen_sl():
    """後ろ向きに転がる坂(前の壁に当たったあとから)× 長いカメラの止まり(残す ON の中止 hold_long を起こす)× 後ろの物"""
    for esc, slope, coast, rear, u, L in itertools.product(('orig', 'human'), (-0.5, -1.0, -2.0), (0.3, 1.5),
                                                          (None, 1.5, 2.5), (2.7, 3.0, 3.3, 3.6, 4.0, 4.5), (0.6, 1.0)):
        yield {'kind': 'sl', 'form': 'kai1', 'xf': 2.0, 'model': {'kind': 'const', 'm': 120.0}, 'esc': esc, 'rear': rear,
               'veh_dt': 0.01, 'dur': 12.0, 'slope': slope, 'coast': coast, 'slope_after_hit': True,
               'drops': [(u, round(u + L, 3))]}


def gen_rd2():
    """rd と同じ 10/23 の形で、ESC が指令に遅れて応える(lag。本物の 10/23 No.6 は 'gap' の中立の間も後退の速さが落ちなかった)
    × 下がり始めの遅れを短くした ESC(遅れの合計を本物の約 0.1 秒に)× 中立の惰性"""
    for lag, esc, coast, rear in itertools.product((0.05, 0.1, 0.15), ('n002r', 'humanr'), (0.5, 1.5),
                                                   [round(0.5 + 0.1 * k, 2) for k in range(14)] + [None]):
        yield {'kind': 'rd2', 'form': 'kai1', 'xf': 2.4, 'model': {'kind': 'brakewall', 'm': 120.0, 'db': 0.45}, 'esc': esc,
               'rear': rear, 'veh_dt': 0.01, 'dur': 5.5, 'coast': coast, 'lag': lag}


def gen_hm():
    """人が CH3 を取って戻す: 場面 rev(前の壁 2.0 m・ずっと前進 120 µs。最初のリカバリーは約 3.05 秒)× 回の形(発進の前進 0 / 1)×
    後ろの物 × 長いカメラの止まり(なし / リカバリーの途中 0.7 秒 = ON は中止)× 人側の区間(始め・長さ・人の操作)"""
    for form, rear, drop, h0, L, thr in itertools.product(('kai1', 'kai4_lb'), (None, 1.3), (None, (3.35, 4.05)),
                                                           (3.15, 3.45, 3.85, 4.6, 5.6), (0.3, 1.0), ('neutral', 'back', 'fwd')):
        sd = {'kind': 'hm', 'form': form, 'xf': 2.0, 'model': {'kind': 'const', 'm': 120.0}, 'esc': 'human', 'rear': rear,
              'veh_dt': 0.01, 'dur': 12.0, 'human': [(h0, round(h0 + L, 2), thr)]}
        if drop:
            sd['drops'] = [drop]
        yield sd


def t_hit_est(xf, m, form):
    """前の壁に当たるおよその時刻(AI 側 1.0 秒から)。偽物の車の前進: 目標 (m/80)×1.2 m/s に 0.3 秒で寄る"""
    lb = h11.FORMS[form].get('launch_boost', 0) == 1
    vt = max(min(max(m, 80.0 if not lb else 250.0) / 80.0 * 1.2, 6.0), 0.5)
    return 1.0 + 0.3 + xf / vt


def gen_fz(seed, count):
    rng = random.Random(seed)
    for i in range(count):
        form = rng.choice(list(h11.FORMS))
        xf = rng.choice((1.1, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0))
        mk = rng.choice(('const', 'const', 'brakewall', 'react', 'neutwall'))
        m = rng.choice((40.0, 80.0, 120.0, 160.0, 220.0))
        if mk == 'brakewall':
            m = max(m, 80.0)
        esc = rng.choice(list(ESCS))
        rear = rng.choice((None, None, 0.9, 1.1, 1.3, 1.6, 2.2))
        vdt = rng.choice((0.01, 0.02))
        dur = rng.choice((10.0, 14.0, 18.0, 22.0))
        th = t_hit_est(xf, m, form)
        sd = {'kind': 'fz', 'i': i, 'seed': seed, 'form': form, 'xf': xf, 'model': {'kind': mk, 'm': m,
              'db': rng.choice((0.3, 0.45, 0.7))}, 'esc': esc, 'rear': rear, 'veh_dt': vdt, 'dur': dur,
              'cam0': round(0.15 + rng.random() / 30.0, 4)}
        # 乱れ(時刻は当たりのおよその時刻 th から)
        if rng.random() < 0.6:
            gl = []
            for _ in range(rng.randint(1, 6)):
                t = round(th + rng.uniform(0.0, dur - th - 1.0), 2)
                if rng.random() < 0.35:
                    b = rng.choice(list(BURSTS.values()))
                    gl += [(round(t + 0.01 * k, 2), e) for k, e in enumerate(b) if e != 0]
                else:
                    gl.append((t, rng.choice((301, 313, 650, 1115, 1322, 1687, -1270, -2500))))
            sd['glitch'] = gl
        if rng.random() < 0.5:
            dr = []
            for _ in range(rng.randint(1, 3)):
                a = th + rng.uniform(-0.5, dur - th - 1.0)
                dr.append((round(a, 3), round(a + rng.choice((0.05, 0.08, 0.11, 0.13, 0.2, 0.3, 0.45, 0.6, 0.8, 1.2)), 3)))
            sd['drops'] = dr
        if rng.random() < 0.25:
            sd['hit_drop'] = (round(rng.uniform(0.0, 0.6), 3), rng.choice((0.12, 0.2, 0.35, 0.55, 0.8)))
        for key, p_ in (('tof_skip', 0.2), ('imu_skip', 0.2), ('veh_skip', 0.25)):
            if rng.random() < p_:
                sp = []
                for _ in range(rng.randint(1, 2)):
                    a = th + rng.uniform(0.0, dur - th - 1.0)
                    sp.append((round(a, 3), round(a + rng.choice((0.03, 0.08, 0.12, 0.16, 0.25, 0.4)), 3)))
                sd[key] = sp
        if rng.random() < 0.25:
            sd['bounce'] = rng.choice((0.1, 0.25, 0.4))
        if rng.random() < 0.2:
            sd['slope'] = rng.choice((-0.3, -0.6, -1.0))
            sd['coast'] = rng.choice((0.2, 0.4, 0.8))
        if rng.random() < 0.2:
            sd['spikes'] = [(round(th + rng.uniform(0.0, dur - th - 1.0), 2), rng.choice((2.6, 3.0, 4.0)))
                            for _ in range(rng.randint(1, 2))]
        if rng.random() < 0.3:
            hs = []
            for _ in range(rng.randint(1, 2)):
                a = round(th + rng.uniform(0.3, max(dur - th - 3.0, 0.4)), 2)
                hs.append((a, round(a + rng.choice((0.15, 0.3, 0.6, 1.0, 2.0)), 2),
                           rng.choice(('neutral', 'back', 'fwd'))))
            hs.sort()
            sd['human'] = hs
        yield sd


def main(kind, part='0/1', seed='0', count='0'):
    h11.check_now()
    pi, pn = (int(x) for x in part.split('/'))
    if kind == 'lw':
        gen = gen_lw()
    elif kind == 'rd':
        gen = gen_rd()
    elif kind == 'sl':
        gen = gen_sl()
    elif kind == 'rd2':
        gen = gen_rd2()
    elif kind == 'hm':
        gen = gen_hm()
    elif kind == 'fz':
        gen = gen_fz(int(seed), int(count))
    else:
        raise KeyError(kind)
    out = {'kind': kind, 'part': part, 'seed': seed, 'sha': h11.check_now(), 'rows': []}
    t0 = time.time()
    try:
        for i, sd in enumerate(gen):
            if i % pn != pi:
                continue
            row = one(sd)
            row['sd'] = sd
            out['rows'].append(row)
            if row['worse'] or not row.get('off_eq_f7b', True) or len(out['rows']) % 50 == 0:
                print(i, round(time.time() - t0), sd['form'], sd['xf'], sd['model'], sd['esc'], sd.get('rear'),
                      row['worse'], row.get('off_eq_f7b'), flush=True)
    finally:
        h11.cleanup()
        suf = '%s_%s' % (kind, part.replace('/', 'of')) + ('' if kind != 'fz' else '_s%s' % seed)
        (HERE / 'out' / ('exp11_%s.json' % suf)).write_text(json.dumps(out, ensure_ascii=False, default=str))
    n = len(out['rows'])
    print('done', kind, part, n, 'worse', sum(1 for r in out['rows'] if r['worse']),
          'off_neq_f7b', sum(1 for r in out['rows'] if not r.get('off_eq_f7b', True)), round(time.time() - t0), 's')


if __name__ == '__main__':
    main(*sys.argv[1:5])
