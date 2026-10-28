"""fix8 係 safety-proposal: fix7/safety_r2/exp9.py の写し。11/3 のあとに向けた案 pA・pB・pC(fix8/proposal/<案>/assist.py)を、
今(base = jetformer/ の写し)と同じ閉じたループ(本物の drive._loop・偽物の ESC と車。h10 = h9 の写し)のグリッドで比べる。
足したもの: 形 pA/pB/pC(ON)・pX_off(案の写しの OFF。今の OFF とビット単位で同じかを見る)・pXx_off(案を OFF にも入れた形)、
くみ revslow_x・revslow_rear・dip_norm・dip_skip、summ に 最大の後ろへの速さ・escape までの秒・下がり終えるまでの秒・後ろの物を押した秒・
やり直しの数。出力は fix8/proposal/out/exp10_<kind>.json。一時の場所は fix8/tmp_proposal/h/<kind>(終わったら消す)。
元の説明: 直す 7-2 回目・係 safety: 指摘 w2-safety-rv7-safety-r2-1 の直し(s4-c)を、閉じたループ(本物の drive._loop・偽物の ESC と車。
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
import h10 as h9  # noqa: E402
from g10 import first_rec, scene  # noqa: E402

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
    'on': ('base', ON, None, None),
    'pA': ('pA', ON, None, None),
    'pB': ('pB', ON, None, None),
    'pC': ('pC', ON, None, None),
    'pA_g': ('pA', G_ON, None, None),
    'pB_g': ('pB', G_ON, None, None),
    'pC_g': ('pC', G_ON, None, None),
    'g_on_b': ('base', G_ON, None, None),
    'pA_off': ('pA', OFF, None, None),
    'pB_off': ('pB', OFF, None, None),
    'pC_off': ('pC', OFF, None, None),
    'pAx_off': ('pA', OFF, None, {'OFFTOO': True}),
    'pBx_off': ('pB', OFF, None, {'OFFTOO': True}),
    'pCx_off': ('pC', OFF, None, {'OFFTOO': True}),
    'r1_on': ('r1', ON, None, None),
    'a_on': ('r1', ON, None, {'READY': 0.08}),
    'ab_on': ('now', ON, None, {'READY': 0.08}),
    'g_on': ('now', G_ON, None, None),
    'on_ng': ('now', ON, {'esc_rev': None}, None),
    's00_on': ('now', ON, None, {'SURE': 0.0}),        # 「2 フレーム続いたら」(反証の文の案 (b) の形)
    's15_on': ('now', ON, None, {'SURE': 0.15}),
    's30_on': ('now', ON, None, {'SURE': 0.3}),
    'off': ('base', OFF, None, None),
}
PROPS = ('pA', 'pB', 'pC')
ALL_ON = 'on,pA,pB,pC'
DEFAULT_VS = {'align': ALL_ON, 'forms': ALL_ON, 'hold': ALL_ON, 'rshort': ALL_ON, 'brake': ALL_ON + ',g_on_b,pA_g,pB_g,pC_g',
              'exp7': ALL_ON, 'revslow': ALL_ON + ',off,pA_off,pB_off,pC_off,pAx_off,pBx_off,pCx_off',
              'dip': ALL_ON}


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
    tb, te = m['t_brake'], m['t_escape']
    if tb is not None:
        end = (te + 0.6) if te is not None else tb + 3.0
        seg = [(t, x, v) for t, x, v, *_ in sim.log if tb <= t <= end]
        m['max_back_v'] = round(max([-v for t, x, v in seg] + [0.0]), 3)       # 最初のリカバリーの最大の後ろへの速さ
        xm = min(x for t, x, v in seg)
        m['t_back_end'] = round(next(t for t, x, v in seg if x <= xm + 1e-9) - tb, 3)   # 下がり終えるまで(いちばん後ろ)の秒
        m['t_to_escape'] = round(te - tb, 3) if te is not None else None
        m['rear_push'] = round(sum(h9.DT for t, x, v, th, con, ro, sl in sim.log if tb <= t <= end and con == 'rear'
                                   and th > N + 40 and sl), 3)
        m['n_retry'] = sum(1 for e in ph if tb - 0.05 <= e[0] <= end and e[1] == 'reverse' and e[2] == 'gap')
    else:
        m.update(max_back_v=0.0, t_back_end=None, t_to_escape=None, rear_push=0.0, n_retry=0)
    return m


def tb_of(base, kw=None):
    return first_rec(h9.run('now', OFF(h9.base_params(**(kw or {}))), base))['t_brake']


def t_rev(base):
    """今の ON で、最初のリカバリーの 'gap'(中立 + 10)のあと初めて後退側(中立 + 300)を書いた時刻 = 最初の 'reverse' の始め"""
    sim, rec, pca, ph = h9.run('base', ON(h9.base_params()), base)
    tb = first_rec((sim, rec, pca, ph))['t_brake']
    tg = next(t for t, s_, th in pca.w if t > tb and abs(th - (N + 10.0)) < 1e-6)
    return next(t for t, s_, th in pca.w if t > tg and abs(th - (N + 300.0)) < 1e-6)


def t_onset(base):
    """今の ON で、最初のリカバリーで偽物の ESC が初めて後退に入った時刻(本当に下がり始め)"""
    sim, rec, pca, ph = h9.run('base', ON(h9.base_params()), base)
    tb = first_rec((sim, rec, pca, ph))['t_brake']
    return next(t for t, r, c in sim.car.revlog if r and t > tb)


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
    elif kind == 'revslow_x':
        # 残り 1 を広げた: 'gap' 0.15 秒では下がれない ESC 2 種 × 50/100 Hz × 最初の 'reverse' の始め −0.04〜+0.40 秒の跳ね 1 つ
        # (313・1322)+ 100 Hz は本物の跳ねの並び 4 つ
        for esc in ((0.2, 0.08), (0.25, 0.08)):
            for vdt in (None, 0.01):
                base = scene('rev', dur=12.0, esc=esc)
                st = 0.02
                if vdt:
                    base['veh_dt'] = vdt
                    st = 0.01
                tr = t_rev(base)
                yield 'esc=%s vdt=%s clean' % (esc, vdt), base, {}, {'clean': True}
                i0 = int(math.floor((tr - 0.04) / st + 1e-6))
                for k in range(int(round(0.44 / st)) + 1):
                    tg = round((i0 + k) * st, 2)
                    for e in (313, 1322):
                        yield 'esc=%s vdt=%s tg=%.2f e=%d' % (esc, vdt, tg, e), dict(base, glitch=[(tg, e)]), {}, {}
                if vdt:
                    for bn, seq in BURSTS7.items():
                        j0 = int(math.floor((tr - 0.04) / 0.01 + 1e-6))
                        for k in range(0, 40, 2):
                            t0 = round((j0 + k) * 0.01, 2)
                            gl = [(round(t0 + 0.01 * i, 2), e) for i, e in enumerate(seq) if e != 0]
                            yield 'esc=%s vdt=%s %s t0=%.2f' % (esc, vdt, bn, t0), dict(base, glitch=gl), {}, {}
    elif kind == 'revslow_rear':
        # 残り 1 の場面に後ろの物(1.2・1.35・1.5・2.0 m)。直したときに後ろへの当たりが増えないか(跳ねなしの同じ場面と比べる)
        for rear in (1.2, 1.35, 1.5, 2.0):
            base = scene('rev', dur=12.0, esc=(0.2, 0.08), rear=rear)
            tr = t_rev(base)
            yield 'rear=%s clean' % rear, base, {}, {'clean': True}
            i0 = int(math.floor((tr - 0.02) / 0.02 + 1e-6))
            for k in range(0, 21):
                tg = round((i0 + k) * 0.02, 2)
                yield 'rear=%s tg=%.2f e=1322' % (rear, tg), dict(base, glitch=[(tg, 1322)]), {}, {}
    elif kind.startswith('dip_norm'):
        # 残り 2 の形をふつうの 'reverse' で: 下がり始めが遅い ESC(後退側から rdelay 秒)で本当に下がっている途中、
        # 車両の値が 1 サンプル(または 0.04 秒)だけ 0 になる。後ろの物 なし・1.5 m。案 pA は s4-c の残りをふつうの 'reverse' に広げる
        vdt = 0.01 if kind.endswith('100') else None
        st = 0.01 if vdt else 0.02
        for rd in ((0.15, 0.27) if vdt else (0.08, 0.15, 0.2, 0.27)):      # 100 Hz は下がり始めの遅れ 2 つだけ(組を減らす)
            for rear in (None, 1.5):
                base = scene('rev', dur=9.0, esc=(0.1, rd), rear=rear)
                if vdt:
                    base['veh_dt'] = vdt
                ton = t_onset(base)
                yield 'rd=%.2f rear=%s clean' % (rd, rear), base, {}, {'clean': True}
                i0 = int(math.floor(ton / st + 1e-6))
                for k in range(int(round(0.40 / st)) + 1):
                    td = round((i0 + k) * st, 2)
                    for w in (1, 2):
                        n = 1 if w == 1 else int(round(0.04 / st))
                        gl = [(round(td + st * i, 2), 0) for i in range(n)]
                        yield ('rd=%.2f rear=%s td=%.2f w=%d' % (rd, rear, td, w), dict(base, glitch=gl), {},
                               {'w': w, 'dt_on': round(td - ton, 3)})
    elif kind.startswith('dip_skip'):
        # 残り 2 そのもの: 'gap' を飛ばした 'reverse'(短い中立止めのあとの跳ねで back_early)で、ESC は後退側から rdelay 秒で
        # 本当に下がり始め、その途中で車両の値が 1 サンプル(または 0.04 秒)だけ 0 になる。後ろの物 なし・1.5 m
        vdt = 0.01 if kind.endswith('100') else None
        st = 0.01 if vdt else 0.02
        for rd in ((0.2, 0.27) if vdt else (0.15, 0.2, 0.27, 0.35)):       # 100 Hz は下がり始めの遅れ 2 つだけ(組を減らす)
            for rear in (None, 1.5):
                base = scene('rev', dur=9.0, esc=(0.03, rd), rear=rear)
                if vdt:
                    base['veh_dt'] = vdt
                tb = tb_of(base)
                a = (round(tb, 3), round(tb + 0.12, 3))
                fr = frames_after(a[1], 12)
                base = dict(base, drops=[a], glitch=[(sample_for(fr[1], st), 1322)])
                ton = t_onset(base)
                yield 'rd=%.2f rear=%s clean' % (rd, rear), base, {}, {'clean': True}
                i0 = int(math.floor(ton / st + 1e-6))
                for k in range(int(round(0.40 / st)) + 1):
                    td = round((i0 + k) * st, 2)
                    for w in (1, 2):
                        n = 1 if w == 1 else int(round(0.04 / st))
                        gl = list(base['glitch']) + [(round(td + st * i, 2), 0) for i in range(n)]
                        yield ('rd=%.2f rear=%s td=%.2f w=%d' % (rd, rear, td, w), dict(base, glitch=gl), {},
                               {'w': w, 'dt_on': round(td - ton, 3)})
    else:
        raise SystemExit('くみ: align_* forms hold_* glitch50 glitch100 burst spring brake gapslow revslow rshort revslow_x '
                         'revslow_rear dip_norm[_100] dip_skip[_100]')


def main(kind, variants=None, suffix=''):
    print(h9.check_now(), flush=True)
    grp = kind.split('_')[0] if kind.split('_')[0] in DEFAULT_VS else ('brake' if kind == 'brake' else 'exp7')
    if variants == '-':
        variants = None
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
            for pv in PROPS:                       # 案の ON が今の ON とビット単位で同じか
                if pv in vs and 'on' in vs:
                    ok, why = h9.same_run(res['on'], res[pv])
                    row['same_' + pv] = ok
                    cnt[pv + ('_same_on' if ok else '_diff_on')] += 1
                    if not ok:
                        row['diff_' + pv] = str(why)[:200]
                if pv + '_g' in vs and 'g_on_b' in vs:
                    ok, why = h9.same_run(res['g_on_b'], res[pv + '_g'])
                    row['same_' + pv + '_g'] = ok
                    cnt[pv + ('_g_same' if ok else '_g_diff')] += 1
                if pv + '_off' in vs and 'off' in vs:   # 案の写しの OFF が今の OFF とビット単位で同じか(ON だけの案)
                    ok, why = h9.same_run(res['off'], res[pv + '_off'])
                    row['same_' + pv + '_off'] = ok
                    cnt[pv + ('_off_same' if ok else '_off_DIFF')] += 1
                if pv + 'x_off' in vs and 'off' in vs:  # 案を OFF にも入れた形が今の OFF と違うか
                    ok, why = h9.same_run(res['off'], res[pv + 'x_off'])
                    row['same_' + pv + 'x_off'] = ok
                    cnt[pv + ('x_off_same' if ok else 'x_off_diff')] += 1
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
            if tag.get('clean'):
                cnt['clean'] += 1
            if i % 20 == 0:
                print(i, key, {v: (row[v]['back'], row[v]['esc_push'], row[v]['all_guards'][:3]) for v in vs}, flush=True)
    finally:
        h9.cleanup()
    out['count'] = dict(cnt)
    out['elapsed_s'] = round(time.time() - t_start, 1)
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / ('exp10_%s%s.json' % (kind, ('_' + suffix) if suffix else ''))).write_text(json.dumps(out, ensure_ascii=False, default=str))
    print(json.dumps(out['count'], ensure_ascii=False), out['elapsed_s'], flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:4])
