"""直す 7-2 回目・係 safety: 指摘 w2-safety-rv7-safety-r2-1 の直し(s4-c)を、閉じたループ(本物の drive._loop・偽物の ESC と車。
h9 = 反証の h8 の写し)のグリッドで確かめる。装置に触れない・GPU なし・API なし。

形(variants):
  on     = 今(s4-c)・rec_back_guard 1・rec_hold_keep 1(11/3 の形)
  r1_on  = この係が直す前(tests/fixtures/fix7_safety_r2_before = 7-1 回目のあと)・両方 1
  a_on   = 案 (a): 直す前の版の EscMayReverse.READY_NEUTRAL_S を 0.08 にした形(メモリの中だけ)・両方 1
  ab_on  = 今 + 案 (a)・両方 1
  g_on   = 今・見張りだけ 1(rec_hold_keep 0)
  on_ng  = 今・両方 1・Recovery に esc_rev を渡さない(None。s4-a を外した形)
  s00_on / s15_on / s30_on = 今の SURE_BACK_S を 0 / 0.15 / 0.3 にした形(メモリの中だけ。0 = 2 フレーム続いたら)
  off    = 今・両方 0(same_off: 直す前の写しの両方 0 とビット単位で同じか)
くみ(kind):
  align_<orig|human>_<50|100>  反証の exp8_align(止まり 0.10〜0.16 秒 × 位置 2 × 1 つ目の跳ね 3 位置 × 2 つ目 0〜8 フレームあと)
  forms                        反証の exp8_forms(回 1〜4 の形 4 通り × 50/100 Hz)
  hold_<orig|human>_<50|100>   反証の exp8_hold(跳ねをそろえない。本物の跳ねの並びを含む)
  glitch50・glitch100・burst・spring・brake・gapslow・revslow  7-1 回目の exp7 のくみ(同じ場面)
  rshort                       中立 0.02・0.03・0.05 秒で下がれる ESC(本物は中立 0.05 秒でも後退に入った)で 'brake' の中の
                               短い中立止め(+ 跳ね 2 つ)。後ろの物 なし・1.2・1.35・1.5 m(案 (a) の損の見積もり)
出力 out/exp9_<kind>.json。一時の場所は fix7/tmp_safety_r2/h/<kind>(終わったら消す)
    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261026_fix/fix7/safety_r2/exp9.py <kind> [variants,...]
"""
import collections
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RV9_TAG', '_'.join(sys.argv[1:2] + sys.argv[3:4]) or 'main')
sys.path.insert(0, str(HERE))
import h9  # noqa: E402
from g9 import first_rec, scene  # noqa: E402

N = h9.N
CAM0 = 0.15
ESC = {'orig': None, 'human': (0.12, 0.10)}
FORMS8 = {'kai1': {}, 'kai2': {'speed_ceiling': 1, 'speed_auto': 1, 'ceil_left_mps': 4.2},
          'kai3': {'speed_ceiling': 1, 'launch_boost': 1}, 'kai4': {'launch_boost': 1}}
FORMS7 = {'kai1': {}, 'kai2_ceil_auto': {'speed_ceiling': 1, 'speed_auto': 1, 'ceil_left_mps': 4.2},
          'kai3_launch': {'speed_ceiling': 1, 'launch_boost': 1}}
BURSTS8 = {
    'b152707_0004': [1628, 0, 1525, 0, 0, 1572, 0, 0, 1289],
    'b211828_0004': [1687, 0, 0, 0, 0, 0, 1411, -1270, 1408, -2156, 1356, 0, 0, -394, 1528, 1358, 656, 385],
}
BURSTS7 = dict({'b160306_0002': [1322], 'b181504_0005': [1115]}, **BURSTS8)


def ON(p):
    return dict(p, rec_back_guard=1, rec_hold_keep=1)


def G_ON(p):
    return dict(p, rec_back_guard=1, rec_hold_keep=0)


def OFF(p):
    return dict(p, rec_back_guard=0, rec_hold_keep=0)


VARIANTS = {
    'on': ('now', ON, None, None),
    'r1_on': ('r1', ON, None, None),
    'a_on': ('r1', ON, None, {'READY': 0.08}),
    'ab_on': ('now', ON, None, {'READY': 0.08}),
    'g_on': ('now', G_ON, None, None),
    'on_ng': ('now', ON, {'esc_rev': None}, None),
    's00_on': ('now', ON, None, {'SURE': 0.0}),        # 「2 フレーム続いたら」(反証の文の案 (b) の形)
    's15_on': ('now', ON, None, {'SURE': 0.15}),
    's30_on': ('now', ON, None, {'SURE': 0.3}),
    'off': ('now', OFF, None, None),
}
DEFAULT_VS = {'align': 'on,r1_on,a_on,off', 'forms': 'on,r1_on,a_on,off', 'hold': 'on,r1_on,a_on,off',
              'rshort': 'on,r1_on,a_on,off', 'brake': 'on,r1_on,g_on,off', 'exp7': 'on,r1_on,g_on,on_ng,off'}


def frames_after(t, n):
    k = int(math.floor((t - CAM0) * 30)) + 1
    out = []
    while len(out) < n:
        tc = CAM0 + k / 30.0
        if tc > t + 1e-9:
            out.append(tc)
        k += 1
    return out


def sample_for(tc, st):
    return round(math.floor((tc + 1e-6) / st) * st, 2)


def summ(res, t_from=1.5):
    sim, rec, pca, ph = res
    m = first_rec(res, t_from)
    m['all_guards'] = [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']
    m['rev_on_t'] = next((t for t, r, c in sim.car.revlog if r), None)
    mm = h9.metrics(sim, rec, pca, t_from=t_from)
    m['rear_hits'] = mm['rear']
    m['rear_max_v'] = mm['rear_max_v']
    m['front_hits'] = [h for h in mm['front'] if h[0] > (m['t_brake'] or 99.0)]
    return m


def tb_of(base, kw=None):
    return first_rec(h9.run('now', OFF(h9.base_params(**(kw or {}))), base))['t_brake']


def cells(kind):
    if kind.startswith('align_'):
        _, esc, vdt = kind.split('_')
        vdt = int(vdt)
        st = 0.01 if vdt == 100 else 0.02
        base = scene('rev', dur=9.0, esc=ESC[esc])
        if vdt == 100:
            base['veh_dt'] = 0.01
        tb = tb_of(base)
        for a0 in (0.0, 0.04):
            for aL in (0.10, 0.12, 0.14, 0.16):
                a = (round(tb + a0, 3), round(tb + a0 + aL, 3))
                fr = frames_after(a[1], 16)
                for j1 in (1, 2, 3):
                    for j2 in (0, 1, 2, 3, 5, 8):
                        gl = [(sample_for(fr[j1], st), 1322)]
                        if j2:
                            gl.append((sample_for(fr[j1 + j2], st), 1322))
                        yield 'A=%s j1=%d j2=%d' % (a, j1, j2), dict(base, drops=[a], glitch=gl), {}, {'two': j2 > 0}
    elif kind == 'forms':
        for form, kw in FORMS8.items():
            for vdt in (None, 0.01):
                base = scene('rev', dur=9.0)
                if vdt:
                    base['veh_dt'] = vdt
                st = 0.01 if vdt else 0.02
                tb = tb_of(base, kw)
                for a0 in (0.0, 0.04):
                    for aL in (0.10, 0.12):
                        a = (round(tb + a0, 3), round(tb + a0 + aL, 3))
                        fr = frames_after(a[1], 12)
                        for j2 in (3, 6):
                            gl = [(sample_for(fr[2], st), 1322), (sample_for(fr[2 + j2], st), 1322)]
                            yield ('%s vdt=%s A=%s j2=%d' % (form, vdt, a, j2), dict(base, drops=[a], glitch=gl), kw,
                                   {'two': True})
    elif kind.startswith('hold_'):
        _, esc, vdt = kind.split('_')
        vdt = int(vdt)
        st = 0.01 if vdt == 100 else 0.02
        base = scene('rev', dur=9.0, esc=ESC[esc])
        if vdt == 100:
            base['veh_dt'] = 0.01
        tb = tb_of(base)
        for a0 in (0.0, 0.03, 0.06):
            for aL in (0.10, 0.12, 0.14, 0.17):
                a = (round(tb + a0, 3), round(tb + a0 + aL, 3))
                for s0 in (0.02, 0.06):
                    t1 = round(round((a[1] + s0) / st) * st, 2)
                    for d in (0.04, 0.1, 0.2, 0.3):
                        t2 = round(round((t1 + d) / st) * st, 2)
                        yield ('A=%s s1=%.2f s2=%.2f' % (a, t1, t2), dict(base, drops=[a], glitch=[(t1, 1322), (t2, 1322)]),
                               {}, {'two': True})
                    if vdt == 100:
                        for bn, seq in BURSTS8.items():
                            gl = [(round(t1 + 0.01 * i, 2), e) for i, e in enumerate(seq) if e != 0]
                            yield 'A=%s %s t1=%.2f' % (a, bn, t1), dict(base, drops=[a], glitch=gl), {}, {'two': True}
    elif kind in ('glitch50', 'glitch100'):
        forms = FORMS7 if kind == 'glitch50' else {'kai1': {}}
        for form, kw in forms.items():
            base = scene('rev', dur=12.0) if kind == 'glitch50' else scene('rev', dur=12.0, esc=(0.12, 0.10))
            if kind == 'glitch100':
                base['veh_dt'] = 0.01
            tb = round(tb_of(base, kw), 2)
            step, n = (0.02, 15) if kind == 'glitch50' else (0.01, 30)
            for k in range(-1, n - 1) if kind == 'glitch50' else range(0, n):
                tg = round(tb + step * k, 2)
                for e in (301, 313, 1322):
                    yield '%s tg=%.2f e=%d' % (form, tg, e), dict(base, glitch=[(tg, e)]), kw, {}
    elif kind == 'burst':
        for esc in (None, (0.12, 0.10)):
            base = scene('rev', dur=12.0, esc=esc)
            base['veh_dt'] = 0.01
            tb = round(tb_of(base), 2)
            for bn, seq in BURSTS7.items():
                for k in range(31):
                    t0 = round(tb + 0.01 * k, 2)
                    gl = [(round(t0 + 0.01 * i, 2), e) for i, e in enumerate(seq) if e != 0]
                    yield 'esc=%s %s t0=%.2f' % (esc, bn, t0), dict(base, glitch=gl), {}, {}
    elif kind == 'spring':
        for spring in ((0.10, 0.03, 0.2), (0.14, 0.03, 0.2), (0.16, 0.02, 0.2), (0.16, 0.03, 0.2), (0.16, 0.05, 0.2),
                       (0.2, 0.03, 0.2), (0.25, 0.04, 0.2)):
            for cam0 in (0.15, 0.16, 0.17, 0.18):
                sc = scene('rev', dur=12.0)
                sc['cam0'] = cam0
                sc['spring'] = spring
                yield '%s cam0=%.2f' % (spring, cam0), sc, {}, {}
    elif kind == 'brake':
        for esc in (None, (0.12, 0.10), (0.1, 0.08)):
            for vdt in (None, 0.01):
                for rear in (None, 1.35, 1.5):
                    sc = scene('brake', rear=rear, esc=esc, dur=9.0)
                    if vdt:
                        sc['veh_dt'] = vdt
                    yield 'esc=%s vdt=%s rear=%s clean' % (esc, vdt, rear), sc, {}, {}
                    for tg in (2.62, 2.66, 2.70, 2.74, 2.78, 2.84, 2.90):
                        for e in (313, 1322):
                            yield ('esc=%s vdt=%s rear=%s tg=%.2f e=%d' % (esc, vdt, rear, tg, e),
                                   dict(sc, glitch=[(tg, e)]), {}, {})
    elif kind == 'gapslow':
        for esc in ((0.2, 0.08), (0.25, 0.08)):
            for vdt in (None, 0.01):
                base = scene('rev', dur=12.0, esc=esc)
                if vdt:
                    base['veh_dt'] = vdt
                yield 'esc=%s vdt=%s clean' % (esc, vdt), base, {}, {}
                for k in range(0, 18):
                    tg = round(2.77 + 0.01 * k, 2)
                    for e in (313, 1322):
                        yield 'esc=%s vdt=%s tg=%.2f e=%d' % (esc, vdt, tg, e), dict(base, glitch=[(tg, e)]), {}, {}
    elif kind == 'revslow':
        base = scene('rev', dur=12.0, esc=(0.2, 0.08))
        yield 'esc=(0.2, 0.08) clean', base, {}, {}
        for k in range(0, 20):
            tg = round(2.96 + 0.02 * k, 2)
            for e in (313, 1322):
                yield 'esc=(0.2, 0.08) tg=%.2f e=%d' % (tg, e), dict(base, glitch=[(tg, e)]), {}, {}
    elif kind == 'rshort':
        for neu in (0.02, 0.03, 0.05):
            for rear in (None, 1.2, 1.35, 1.5):
                base = scene('rev', dur=9.0, esc=(neu, 0.08), rear=rear)
                tb = tb_of(base)
                for a0 in (0.0, 0.04, 0.08, 0.12):
                    for aL in (0.10, 0.12, 0.14):
                        a = (round(tb + a0, 3), round(tb + a0 + aL, 3))
                        fr = frames_after(a[1], 12)
                        for two in (False, True):
                            gl = [(sample_for(fr[1], 0.02), 1322), (sample_for(fr[7], 0.02), 1322)] if two else []
                            yield ('neu=%.2f rear=%s A=%s two=%s' % (neu, rear, a, two),
                                   dict(base, drops=[a], glitch=gl), {}, {'two': two})
    else:
        raise SystemExit('くみ: align_* forms hold_* glitch50 glitch100 burst spring brake gapslow revslow rshort')


def main(kind, variants=None, suffix=''):
    print(h9.check_now(), flush=True)
    grp = kind.split('_')[0] if kind.split('_')[0] in DEFAULT_VS else ('brake' if kind == 'brake' else 'exp7')
    vs = (variants or DEFAULT_VS[grp]).split(',')
    out = {'kind': kind, 'variants': vs, 'cells': {}}
    cnt = collections.Counter()
    t_start = time.time()
    try:
        for i, (key, sc, kw, tag) in enumerate(cells(kind)):
            p = h9.base_params(**kw)
            row = {'tag': tag}
            res = {}
            for v in vs:
                name, f, force, patch = VARIANTS[v]
                res[v] = h9.run(name, f(p), sc, force=force, patch=patch)
                row[v] = summ(res[v])
            if 'off' in vs:
                ok, why = h9.same_run(res['off'], h9.run('r1', OFF(p), sc))
                row['same_off'] = ok
                cnt['off_same' if ok else 'off_DIFF'] += 1
                if not ok:
                    row['off_diff'] = str(why)
            if 'r1_on' in vs and 'on' in vs:
                ok, why = h9.same_run(res['r1_on'], res['on'])
                row['same_on_r1'] = ok
                cnt['on_same_r1' if ok else 'on_diff_r1'] += 1
            out['cells'][key] = row
            cnt['n'] += 1
            for v in vs:
                m = row[v]
                b = m['back'] if m['back'] is not None else 0.0
                fail = b < 0.1 and (m['esc_push'] or 0) > 0.05
                if fail:
                    cnt[v + '_fail'] += 1
                    if tag.get('two'):
                        cnt[v + '_fail_two'] += 1
                if m['rear_hits']:
                    cnt[v + '_rear_hit'] += 1
            if i % 20 == 0:
                print(i, key, {v: (row[v]['back'], row[v]['esc_push'], row[v]['all_guards'][:3]) for v in vs}, flush=True)
    finally:
        h9.cleanup()
    out['count'] = dict(cnt)
    out['elapsed_s'] = round(time.time() - t_start, 1)
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / ('exp9_%s%s.json' % (kind, ('_' + suffix) if suffix else ''))).write_text(json.dumps(out, ensure_ascii=False, default=str))
    print(json.dumps(out['count'], ensure_ascii=False), out['elapsed_s'], flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:4])
