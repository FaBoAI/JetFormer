"""直す 7-1 回目・係 safety: 指摘 w1-safety-rv7-safety-r1-1(後ろの見張りの back_early が eRPM の 1 サンプルの跳ねで 'gap' を飛ばす)
の直しを、閉じたループ(本物の drive._loop・偽物の ESC と車 = fix3/refute2_safety/closed_loop.py。h7 = 反証の rv7 の写し)で確かめる。

1 つの場面を次の形で回す:
  f7b_on  = 直す前の写し(tests/fixtures/fix7_safety_before = fix6 r2 のあと)・rec_back_guard 1・rec_hold_keep 1
  on      = 今・両方 1
  on_ng   = 今・両方 1・ただし Recovery に esc_rev を渡さない(None = drive の EscTrack を使わない形。s4-b だけを見る)
  g_on    = 今・見張りだけ 1(rec_hold_keep 0)
  off     = 今・両方 0(f7b_off = 直す前の写し・両方 0 とビット単位で同じか same_off で見る)
場面のくみ(引数で選ぶ):
  glitch50  場面 rev(前の壁 2.0 m・モデルは前進 120 µs)・回 1〜4 の形 3 通り・最初の 'brake' の始め −0.02〜+0.26 秒に
            1 サンプルの跳ね 301・313・1322(反証の exp_glitch と同じ。値を 3 つに)
  glitch100 同じを 100 Hz の車両の値・人に寄せた偽物の ESC(中立 0.12 秒・後退側から 0.10 秒)で(反証の exp_glitch3)
  burst     本物の記録の跳ねの並び(10 ms ごと)を 'brake' の始めから 0〜0.30 秒に入れる。100 Hz・ESC 2 通り
  spring    壁を押していた車の 'brake' での戻り(反証の exp_spring の形。0.10〜0.25 m/s・0.02〜0.05 秒)
  brake     場面 brake(ESC が後退を待っている 10/23 の形)。止まりなし・後ろの物・跳ね。f7b_on と on がビット単位で同じか
  gapslow   'gap' の中立 0.15 秒では下がれない ESC(中立 0.2・0.25 秒)で、'gap' の間に 1 サンプルの跳ね
  revslow   同じ ESC(中立 0.2 秒)で、最初の 'reverse'(まだ下がらない)の間に 1 サンプルの跳ね(残ることの確かめ)
出力: out/exp7_<くみ>.json(場面ごとの数)・標準出力に数え。一時の場所は fix7/tmp_safety/h/<くみ>(終わったら消す)
    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261026_fix/fix7/safety/exp7.py <くみ>
"""
import collections
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RV7_TAG', sys.argv[1] if len(sys.argv) > 1 else 'main')
sys.path.insert(0, str(HERE))
import h7  # noqa: E402
from g7 import first_rec, scene  # noqa: E402

N = h7.N
FORMS = {'kai1': {}, 'kai2_ceil_auto': {'speed_ceiling': 1, 'speed_auto': 1, 'ceil_left_mps': 4.2},
         'kai3_launch': {'speed_ceiling': 1, 'launch_boost': 1}}
# 本物の記録の跳ね(10 ms ごとの eRPM。車は 0.1 秒動いていない = hall_count が ±1〜2 を行き来するだけ。data/raw を読むだけ)
BURSTS = {
    'b160306_0002': [1322],                                          # AI 側。止まった直後(-162, 0, 1322, 0, 0)
    'b181504_0005': [1115],                                          # AI 側。前の壁を押している間(0, 0, 1115, 0, 0)
    'b152707_0004': [1628, 0, 1525, 0, 0, 1572, 0, 0, 1289],        # 人側。0.08 秒に 4 回
    'b211828_0004': [1687, 0, 0, 0, 0, 0, 1411, -1270, 1408, -2156, 1356, 0, 0, -394, 1528, 1358, 656, 385],   # 人側
}


def ON(p):
    return dict(p, rec_back_guard=1, rec_hold_keep=1)


def G_ON(p):
    return dict(p, rec_back_guard=1, rec_hold_keep=0)


def OFF(p):
    return dict(p, rec_back_guard=0, rec_hold_keep=0)


VARIANTS = {
    'f7b_on': ('f7b', ON, None),
    'on': ('now', ON, None),
    'on_ng': ('now', ON, {'esc_rev': None}),
    'g_on': ('now', G_ON, None),
    'off': ('now', OFF, None),
}


def summ(res):
    sim, rec, pca, ph = res
    m = first_rec(res)
    m['starts'] = [t for t, p0, p1, *_ in ph if p0 is None and p1 == 'brake']
    m['phases'] = []
    last = None
    for t, p0, p1, *_r in ph:
        if p1 != last:
            m['phases'].append((t, p1))
            last = p1
    m['phases'] = m['phases'][:16]
    m['rev_on_t'] = next((t for t, r, c in sim.car.revlog if r), None)
    m['all_guards'] = [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']
    mm = h7.metrics(sim, rec, pca, t_from=1.5)
    m['rear_hits'] = mm['rear']
    return m


def t_brake(sc, kw):
    """止まりなし・跳ねなしで最初のリカバリーの 'brake'(後退側 中立 + 300)を書いた時刻(直す前の写し・両方 0)"""
    tb = first_rec(h7.run('f7b', OFF(h7.base_params(**kw)), sc))['t_brake']
    print('t_brake', tb, flush=True)
    return round(tb, 2)


def cells(kind):
    if kind in ('glitch50', 'glitch100'):
        forms = FORMS if kind == 'glitch50' else {'kai1': {}}
        for form, kw in forms.items():
            base = scene('rev', dur=12.0) if kind == 'glitch50' else scene('rev', dur=12.0, esc=(0.12, 0.10))
            if kind == 'glitch100':
                base['veh_dt'] = 0.01
            tb = t_brake(base, kw)
            step, n = (0.02, 15) if kind == 'glitch50' else (0.01, 30)
            for k in range(-1, n - 1) if kind == 'glitch50' else range(0, n):
                tg = round(tb + step * k, 2)
                for e in (301, 313, 1322):
                    yield '%s tg=%.2f e=%d' % (form, tg, e), dict(base, glitch=[(tg, e)]), kw
    elif kind == 'burst':
        for esc in (None, (0.12, 0.10)):
            base = scene('rev', dur=12.0, esc=esc)
            base['veh_dt'] = 0.01
            tb = t_brake(base, {})
            for bn, seq in BURSTS.items():
                for k in range(31):
                    t0 = round(tb + 0.01 * k, 2)
                    gl = [(round(t0 + 0.01 * i, 2), e) for i, e in enumerate(seq) if e != 0]
                    yield 'esc=%s %s t0=%.2f' % (esc, bn, t0), dict(base, glitch=gl), {}
    elif kind == 'spring':
        for spring in ((0.10, 0.03, 0.2), (0.14, 0.03, 0.2), (0.16, 0.02, 0.2), (0.16, 0.03, 0.2), (0.16, 0.05, 0.2),
                       (0.2, 0.03, 0.2), (0.25, 0.04, 0.2)):
            for cam0 in (0.15, 0.16, 0.17, 0.18):
                sc = scene('rev', dur=12.0)
                sc['cam0'] = cam0
                sc['spring'] = spring
                yield '%s cam0=%.2f' % (spring, cam0), sc, {}
    elif kind == 'brake':
        for esc in (None, (0.12, 0.10), (0.1, 0.08)):
            for vdt in (None, 0.01):
                for rear in (None, 1.35, 1.5):
                    sc = scene('brake', rear=rear, esc=esc, dur=9.0)
                    if vdt:
                        sc['veh_dt'] = vdt
                    yield 'esc=%s vdt=%s rear=%s clean' % (esc, vdt, rear), sc, {}
                    for tg in (2.62, 2.66, 2.70, 2.74, 2.78, 2.84, 2.90):
                        for e in (313, 1322):
                            yield ('esc=%s vdt=%s rear=%s tg=%.2f e=%d' % (esc, vdt, rear, tg, e),
                                   dict(sc, glitch=[(tg, e)]), {})
    elif kind == 'gapslow':
        for esc in ((0.2, 0.08), (0.25, 0.08)):
            for vdt in (None, 0.01):
                base = scene('rev', dur=12.0, esc=esc)
                if vdt:
                    base['veh_dt'] = vdt
                yield 'esc=%s vdt=%s clean' % (esc, vdt), base, {}
                for k in range(0, 18):
                    tg = round(2.77 + 0.01 * k, 2)
                    for e in (313, 1322):
                        yield 'esc=%s vdt=%s tg=%.2f e=%d' % (esc, vdt, tg, e), dict(base, glitch=[(tg, e)]), {}
    elif kind == 'revslow':
        # 残ることの確かめ: 'gap' 0.15 秒では下がれない ESC で、最初の 'reverse'(下がらない)の CHECK_S の間の跳ね。ON も OFF も
        # 'reverse' の中の跳ねは見た後退に数える(OFF は変えない)
        base = scene('rev', dur=12.0, esc=(0.2, 0.08))
        yield 'esc=(0.2, 0.08) clean', base, {}
        for k in range(0, 20):
            tg = round(2.96 + 0.02 * k, 2)
            for e in (313, 1322):
                yield 'esc=(0.2, 0.08) tg=%.2f e=%d' % (tg, e), dict(base, glitch=[(tg, e)]), {}
    else:
        raise SystemExit('くみ: glitch50 glitch100 burst spring brake gapslow revslow')


def main(kind, variants=None):
    print(h7.check_now(), flush=True)
    vs = variants.split(',') if variants else list(VARIANTS)
    out = {'kind': kind, 'variants': vs, 'cells': {}}
    cnt = collections.Counter()
    t_start = time.time()
    try:
        for i, (key, sc, kw) in enumerate(cells(kind)):
            p = h7.base_params(**kw)
            row = {}
            res = {}
            for v in vs:
                name, f, force = VARIANTS[v]
                res[v] = h7.run(name, f(p), sc, force=force)
                row[v] = summ(res[v])
            if 'off' in vs:
                ok, why = h7.same_run(res['off'], h7.run('f7b', OFF(p), sc))
                row['same_off'] = ok
                cnt['off_same' if ok else 'off_DIFF'] += 1
            if 'f7b_on' in vs and 'on' in vs:
                ok, why = h7.same_run(res['f7b_on'], res['on'])
                row['same_on_f7b'] = ok
                cnt['on_same_f7b' if ok else 'on_diff_f7b'] += 1
            out['cells'][key] = row
            cnt['n'] += 1
            for v in vs:
                m = row[v]
                b = m['back'] if m['back'] is not None else 0.0
                if b < 0.1:
                    cnt[v + '_back_lt0.1'] += 1
                if (m['esc_push'] or 0) > 0.05:
                    cnt[v + '_esc_push'] += 1
                if m['rear_hits']:
                    cnt[v + '_rear_hit'] += 1
            if i % 20 == 0:
                print(i, key, {v: (row[v]['back'], row[v]['esc_push'], row[v]['front_push'], row[v]['all_guards'][:3]) for v in vs},
                      flush=True)
    finally:
        h7.cleanup()
    out['count'] = dict(cnt)
    out['elapsed_s'] = round(time.time() - t_start, 1)
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / ('exp7_%s.json' % kind)).write_text(json.dumps(out, ensure_ascii=False, default=str))
    print(json.dumps(out['count'], ensure_ascii=False), out['elapsed_s'], flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:3])
