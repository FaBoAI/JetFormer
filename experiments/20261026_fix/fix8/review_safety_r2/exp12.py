"""反証 8-2 回目・係 safety: 本物の drive._loop の閉じたループ(h12 = 8-1 の h11 + ESC の模型 18 通り)で、両方 ON と両方 OFF を
同じ場面で並べ、「ON のほうが OFF より悪い」場面を数える。8-1 と違う角度:
  tb   時間の境目: 乱れの無い ON の走りのリカバリーの段の移り(brake の始め・'gap'・'reverse'・最初に下がったフレーム・
       'reverse' + CHECK_S 0.35・最初に下がった + SURE_BACK_S 0.2・'coast'・'escape'・escape の終わり)の前後 1 フレームに、
       カメラの止まり(3・5・8・13・16 フレーム)・車両の値の止まり(0.1・0.16・0.25 秒)・eRPM の跳ね 1 つ・人側
       (0.15・0.5・1.5 秒の中立 / 0.5 秒の人の後退)・段差の衝撃 3G を入れる
  hm   人が CH3 を取る / 戻す: 段の移りの前 0.1 秒〜escape の終わり + 0.2 秒の各フレームで人側にして、0.1・0.3・1.2 秒で
       AI 側に戻す(人の操作は中立。0.6 秒は 10/28 08:40 の Jetson の再起動のあとに外した)。回の形 kai1(start_kick 1)と kai4_lb(発進の前進 1)
  hb   brake の中の中立止め: brake の始めの 1 フレーム前〜gap の始めの各フレームに、カメラの止まり 4・5・6・8・10・13 フレーム・
       車両の値の止まり 0.16・0.25 秒(ESC がブレーキのあとの中立で後退を受け付けるようになり、残りの brake で下がる形)
  ov   後退の終わりのカメラの止まり: 'reverse' → 'coast' の 4 フレーム前〜1 フレーム後から 3〜10 フレーム(tb で見つけた形の広さ)
  hmh  残した中立止めの中で人が取る: カメラの止まり 8・16 フレームを段の移りに入れ、その止まりの始め + 0.05・0.15・0.3 秒に
       人側 0.3・1.0 秒
出す値の比べ: 後ろの物の距離 DS(止まった壁から)ごとの最初に当たる速さ(物は置かずに走り、そこを越えたときの速さ。票 A の
hit_profile と同じ)・前の壁を押した秒・0 m の壁押し・前の当たりの速さ・ON だけの中止。記録の照らし(行の out と書いた値・
assist 5 と launch_on・中立止めの行)と、夜の道具(tools/standing_3lap.abort_resumes)の印と本当の形の照らし。
全部 OFF は、一部の組(ESC 2 つ fit・human と、どの組も乱れの無い場面)で写し f7b(fix7 の前)の OFF とビット単位で比べる。
出力 out/exp12_<kind>_<part>.jsonl(1 組 1 行。流し直すと済んだ組は飛ばす)。一時の場所は fix8/tmp_review_safety_r2(終わったら消す)
    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -I -B experiments/20261026_fix/fix8/review_safety_r2/exp12.py <kind> <i/n>
"""
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RS82_TAG', '_'.join(sys.argv[1:3]).replace('/', 'of') or 'main')
sys.path.insert(0, str(HERE))
import h12  # noqa: E402

h11 = h12.h11
sys.path.insert(0, str(h12.ROOT / 'tools'))
import label_events as le  # noqa: E402
import standing_3lap as s3  # noqa: E402

C, N, A, KV = h12.C, h12.N, h12.A, h12.KV
DS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6]
ESCS = sorted(h12.ESC18)
IDCHECK_ESC = ('fit', 'human')
REC_PH = ('brake', 'wait', 'gap', 'reverse', 'coast', 'escape', 'halt')


def model_from(kind, m, xf):
    m = float(m)
    if kind == 'const':
        return (C, N - m)
    if kind == 'react':       # 当たった直後の 0.4 秒はブレーキ側、そのあと前進(h11 exp11 の react と同じ)
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


BASES = {'rev': {'xf': 2.0, 'model': ('const', 120.0), 'dur': 9.0},
         'react': {'xf': 2.0, 'model': ('react', 120.0), 'dur': 9.0}}


def samp(c):
    """カメラの時刻 c 以前の最新の車両の値のサンプル時刻(100 Hz)"""
    return math.floor(c * 100 + 1e-6) / 100.0


def build(sd):
    b = BASES[sd['base']]
    sc = {'dur': sd.get('dur', b['dur']), 'xf': b['xf'], 'ai_from': 1.0, 'esc18': sd['esc'], 'veh_dt': 0.01,
          'model': model_from(b['model'][0], b['model'][1], b['xf'])}
    for p in sd.get('pert', []):
        k, c = p[0], float(p[1])
        if k == 'drop':
            sc.setdefault('drops', []).append((c - 0.002, c - 0.002 + p[2] / 30.0))
        elif k == 'veh':
            sc.setdefault('veh_skip', []).append((c - 0.0005, c - 0.0005 + p[2]))
        elif k == 'glitch':
            sc.setdefault('glitch', []).append((samp(c), int(p[2])))
        elif k == 'spike':
            sc.setdefault('spikes', []).append((round(samp(c) - 0.02, 2), float(p[2])))
        elif k == 'human':
            a = samp(c) - 0.0005
            sc.setdefault('human', []).append((a, a + float(p[2]), p[3] if len(p) > 3 else 'neutral'))
    if sc.get('human'):
        spans = [(a, b_) for a, b_, _ in sc['human']]
        pats = [(a, b_, h) for a, b_, h in sc['human']]

        def hthr(t):
            for a, b_, h in pats:
                if a <= t < b_:
                    if h == 'back':
                        el = t - a
                        return N + 200.0 if el < 0.27 else (N if el < 0.39 else (N + 216.0 if el < 0.69 else N))
                    return N
            return N
        sc['human'] = spans
        sc['human_thr'] = hthr
    return sc


def params_of(sd):
    return h12.base_params(**h12.FORMS[sd.get('form', 'kai1')])


def profile(sim, xf, t_from=1.0):
    """前の壁に最初に触れてから: 壁から d m 後ろに物があったら最初に当たる速さ(当たらない = 0)・下がった最大・後ろへの最高速"""
    log = sim.log
    i0 = next((i for i, r in enumerate(log) if r[0] >= t_from and r[4] == 'front'), None)
    if i0 is None:
        return {str(d): 0.0 for d in DS}, 0.0, 0.0
    out = {}
    part = log[i0:]
    for d in DS:
        v = 0.0
        for t, x, vv, *_ in part:
            if x <= xf - d:
                v = round(-vv, 3) if vv < -0.05 else 0.0
                break
        out[str(d)] = v
    return out, round(xf - min(r[1] for r in part), 3), round(-min(r[2] for r in part), 3)


def tool_marks(res):
    """記録の行から夜の道具の abort_resumes を作る(ラベルの ai = 車両の CH3。fix8 tools-decided の形)と、本当の形
    (人側にした直前の Recovery の段)。→ [{'i0': 走り直しの区間の始め [秒], 'mark': 印の kind か None, 'truth': 段}]"""
    sim, rec, pca, ph = res[:4]
    rows = [r for r in rec.rows]
    if not rows:
        return []
    t = np.array([(tn - h12.T0) / 1e9 for tn, _ in rows])
    ai = np.array([sim.sel(samp(x)) for x in t])
    assist = np.array([int(r['assist']) for _, r in rows], np.int8)
    hold = np.array([bool(r['hold']) for _, r in rows])
    sel = np.array([bool(r['select']) for _, r in rows])
    vt = np.array([x[0] for x in sim.log])
    verpm = np.array([-x[2] / KV for x in sim.log])
    Df = le.cumulative_distance(vt, verpm)[1]
    ab = s3.abort_resumes(t, ai, assist, hold, sel, vt, Df, verpm)
    out = []
    runs = list(le.runs(ai))
    for j, (i0, i1) in enumerate(runs):
        if j == 0:
            continue
        p_end = float(t[runs[j - 1][1]])
        ph_last = h12.phase_at(ph, p_end + 0.0085)      # 区間の最後のフレームのあとの段
        # 人側にしたあとの最初の Recovery.step(人側のフレーム)は段を戻すので、その前の段を見る
        out.append({'i0': round(float(t[i0]), 3), 'mark': (ab.get(i0) or {}).get('kind'), 'truth': ph_last,
                    'moved_m': round(float(np.interp(t[i0], vt, Df)), 2)})
    return out


def metr(res, sd):
    sim, rec, pca, ph, stats = res
    m = h11.metrics(res, 1.0)
    prof, back, vback = profile(sim, BASES[sd['base']]['xf'])
    o = {k: m[k] for k in ('front_push_s', 'rear_push_s', 'zero_push', 'giveup_t', 'giveup_rows', 'front_n', 'front_max_v',
                           'recoveries', 'holds', 'launch_frames', 'kick_frames', 'min_x')}
    o['guards'] = [g[0] for g in m['guards']][:12]
    o['prof'] = prof
    o['back'] = back
    o['vback'] = vback
    o['rows_bad'] = h12.check_rows(res)[:5]
    o['marks'] = tool_marks(res) if sd.get('human_any') else []
    return o


def worse(on, off):
    w = []
    for d in DS:
        a, b = on['prof'][str(d)], off['prof'][str(d)]
        if a > b + 0.1:
            w.append('rear_faster@%s' % d)
            break
    if on['front_push_s'] > off['front_push_s'] + 0.3:
        w.append('front_push_more')
    if on['zero_push'] > off['zero_push']:
        w.append('zero_push')
    if on['front_max_v'] > off['front_max_v'] + 0.3:
        w.append('front_faster')
    if on['giveup_t'] is not None and off['giveup_t'] is None:
        w.append('giveup_only_on')
    return w


def run_pair(sd, idcheck=False):
    p = params_of(sd)
    ron = h12.run(h12.ON(p), build(sd))
    roff = h12.run(h12.OFF(p), build(sd))
    mon, moff = metr(ron, sd), metr(roff, sd)
    row = {'on': mon, 'off': moff, 'worse': worse(mon, moff), 'same_on_off': h11.same_run(ron, roff)[0]}
    if idcheck:
        rb = h12.run(h12.OFF(p), build(sd), ver='f7b')
        ok, why = h11.same_run(roff, rb)
        row['off_eq_f7b'] = ok
        if not ok:
            row['off_f7b_diff'] = str(why)[:300]
    return row, ron


def timeline(ph):
    """乱れの無い ON の走りの最初のリカバリーの段の移り(Recovery.step の時刻 = カメラの時刻 + 0.008 秒)"""
    ev = {}
    tb = next((x[0] for x in ph if x[1] is None and x[2] == 'brake'), None)
    if tb is None:
        return ev
    ev['brake'] = tb
    for name, p1 in (('gap', 'gap'), ('reverse', 'reverse'), ('coast', 'coast'), ('escape', 'escape')):
        t = next((x[0] for x in ph if x[0] > tb and x[2] == p1 and x[1] != p1), None)
        if t is not None:
            ev[name] = t
    if 'reverse' in ev:
        tk = next((x[0] for x in ph if x[0] >= ev['reverse'] and x[2] == 'reverse' and x[6] > 300), None)
        if tk is not None:
            ev['backing'] = tk
            ev['sure'] = tk + 0.2
        ev['check'] = ev['reverse'] + 0.35
    tx = next((x[0] for x in ph if x[0] > tb and x[1] == 'escape' and x[2] is None), None)
    if tx is not None:
        ev['esc_end'] = tx
    steps = [x[0] for x in ph]
    snap = {}
    for k, v in ev.items():          # 次の Recovery.step の時刻へ
        s = next((x for x in steps if x >= v - 1e-6), None)
        if s is not None:
            snap[k] = round(s, 4)
    return snap


def cam_of(step_t):
    return round(step_t - 0.008, 4)


def gen_tb(bases=('rev', 'react')):
    for base, esc in itertools.product(bases, ESCS):
        yield {'kind': 'tb_base', 'base': base, 'esc': esc}


PERT_TB = ([('drop', n) for n in (3, 5, 8, 13, 16)] + [('veh', L) for L in (0.1, 0.16, 0.25)]
           + [('glitch', 1322)] + [('human', L, 'neutral') for L in (0.15, 0.5, 1.5)] + [('human', 0.5, 'back')]
           + [('spike', 3.0)])


def expand_tb(base_sd, tl):
    seen = set()
    for ev, st in sorted(tl.items(), key=lambda kv: kv[1]):
        for dk in (-1, 0, 1):
            c = round(cam_of(st) + dk / 30.0, 4)
            for pt in PERT_TB:
                key = (c, pt)
                if key in seen:
                    continue
                seen.add(key)
                sd = dict(base_sd, kind='tb', ev=ev, dk=dk, pert=[(pt[0], c) + tuple(pt[1:])])
                sd['dur'] = 9.0 if pt[0] != 'human' else 10.0
                sd['human_any'] = pt[0] == 'human'
                yield sd


def expand_hm(base_sd, tl, form):
    if 'brake' not in tl or 'esc_end' not in tl:
        return
    c0, c1 = cam_of(tl['brake']) - 0.1, cam_of(tl['esc_end']) + 0.2
    k0 = int(math.floor((c0 - 0.15) * 30))
    k1 = int(math.ceil((c1 - 0.15) * 30))
    for k in range(k0, k1 + 1):
        c = round(0.15 + k / 30.0, 4)
        for L in (0.1, 0.3, 1.2):
            yield dict(base_sd, kind='hm', form=form, pert=[('human', c, L, 'neutral')], dur=11.0, human_any=True,
                       rel=round(c - cam_of(tl['brake']), 3))


def expand_hmh(base_sd, tl):
    for ev, st in sorted(tl.items(), key=lambda kv: kv[1]):
        c = cam_of(st)
        for n in (8, 16):
            for dh in (0.05, 0.15, 0.3):
                if dh >= n / 30.0:
                    continue
                for L in (0.3, 1.0):
                    yield dict(base_sd, kind='hmh', ev=ev, pert=[('drop', c, n), ('human', round(c + dh, 4), L, 'neutral')],
                               dur=11.0, human_any=True)


def expand_hb(base_sd, tl):
    """'brake' の中の中立止め(ESC がブレーキのあとの中立で後退を受け付けるようになり、残りの 'brake' の後退側で下がる形):
    'brake' の始めの 1 フレーム前〜'gap' の始めの各フレームから、カメラの止まり 4・5・6・8・10・13 フレーム・車両の値の止まり 0.16・0.25 秒"""
    if 'brake' not in tl or 'gap' not in tl:
        return
    c0, c1 = cam_of(tl['brake']) - 1 / 30.0, cam_of(tl['gap'])
    k = 0
    while True:
        c = round(c0 + k / 30.0, 4)
        if c > c1 + 1e-6:
            break
        for pt in [('drop', n) for n in (4, 5, 6, 8, 10, 13)] + [('veh', 0.16), ('veh', 0.25)]:
            yield dict(base_sd, kind='hb', ev='brake+%d' % (k - 1), pert=[(pt[0], c) + tuple(pt[1:])], dur=9.0, human_any=False)
        k += 1


def expand_ov(base_sd, tl):
    """後退の終わりのカメラの止まり: 'reverse' → 'coast' のフレームの 4 フレーム前〜1 フレーム後から、カメラの止まり 3・4・5・6・8・10
    フレーム(止まりの間は前に書いた後退側が、中立止めが中立を書くまで最大 0.1 秒続く)"""
    if 'coast' not in tl:
        return
    cc = cam_of(tl['coast'])
    for k in range(-4, 2):
        c = round(cc + k / 30.0, 4)
        for n in (3, 4, 5, 6, 8, 10):
            yield dict(base_sd, kind='ov', ev='coast%+d' % k, pert=[('drop', c, n)], dur=9.0, human_any=False)


class JL(object):
    """1 組ごとに JSON の行を足す(Jetson が止まっても、それまでの組は残る)。流し直すと、済んだ組は飛ばす"""

    def __init__(self, path):
        self.path = path
        self.done, self.bases = set(), {}
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    x = json.loads(line)
                except ValueError:
                    continue            # 止まったときの書きかけの行
                if x.get('t') == 'base':
                    self.bases[skey(x['sd'])] = x
                elif x.get('t') == 'row':
                    self.done.add(skey(x['sd']))
        self.f = open(path, 'a')

    def add(self, rec):
        self.f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        self.f.flush()
        os.fsync(self.f.fileno())


def skey(sd):
    return json.dumps({k: v for k, v in sd.items() if k != 'human_any'}, sort_keys=True, default=str)


def load_jsonl(path):
    """→ {'kind', 'bases': [{'sd', 'tl', 'row'}], 'rows': [row(sd つき)]}(ana12 が読む)"""
    out = {'bases': [], 'rows': []}
    seen = set()
    for line in Path(path).read_text().splitlines():
        try:
            x = json.loads(line)
        except ValueError:
            continue
        k = (x.get('t'), skey(x.get('sd', {})))
        if k in seen:
            continue
        seen.add(k)
        if x.get('t') == 'base':
            out['bases'].append(x)
        elif x.get('t') == 'row':
            out['rows'].append(x['row'])
    return out


def main(kind, part='0/1'):
    h11.check_now()
    pi, pn = (int(x) for x in part.split('/'))
    jl = JL(HERE / 'out' / ('exp12_%s_%s.jsonl' % (kind, part.replace('/', 'of'))))
    n_new = n_worse = 0
    t_s = time.time()
    try:
        jobs = list(gen_tb(('rev', 'react') if kind in ('tb', 'hmh', 'hb', 'ov') else ('rev',)))
        only = os.environ.get('RS82_ESC')
        if only:
            jobs = [j for j in jobs if j['esc'] in only.split(',')]
        for ji, bsd in enumerate(jobs):
            if ji % pn != pi:
                continue
            forms = ('kai1',) if kind != 'hm' else (('kai1', 'kai4_lb') if bsd['esc'] in ('fit', 'cl', 'human', 'fit_lag0.1',
                                                                                          'fit_coast1.5', 'fit_const')
                                                    else ('kai1',))
            for form in forms:
                bsd_f = dict(bsd, form=form)
                b = jl.bases.get(skey(bsd_f))
                if b is None:
                    brow, bron = run_pair(dict(bsd_f, human_any=False), idcheck=True)
                    tl = timeline(bron[3])
                    jl.add({'t': 'base', 'kind': kind, 'sd': bsd_f, 'tl': tl, 'row': brow})
                    print('base', bsd_f, tl, brow['worse'], brow.get('off_eq_f7b'), round(time.time() - t_s), flush=True)
                else:
                    tl = b['tl']
                if kind == 'tb':
                    gen = expand_tb(bsd_f, tl)
                elif kind == 'hm':
                    gen = expand_hm(bsd_f, tl, form)
                elif kind == 'hmh':
                    gen = expand_hmh(bsd_f, tl)
                elif kind == 'hb':
                    gen = expand_hb(bsd_f, tl)
                elif kind == 'ov':
                    gen = expand_ov(bsd_f, tl)
                else:
                    raise KeyError(kind)
                lim = int(os.environ.get('RS82_LIMIT', '0') or 0)
                for si, sd in enumerate(gen):
                    if lim and si >= lim:
                        break
                    if skey(sd) in jl.done:
                        continue
                    row, _ = run_pair(sd, idcheck=bsd['esc'] in IDCHECK_ESC)
                    row['sd'] = {k: v for k, v in sd.items() if k != 'human_any'}
                    jl.add({'t': 'row', 'kind': kind, 'sd': row['sd'], 'row': row})
                    n_new += 1
                    if row['worse'] or not row.get('off_eq_f7b', True) or row['on']['rows_bad'] or row['off']['rows_bad']:
                        n_worse += 1
                        print(n_new, round(time.time() - t_s), sd['esc'], sd.get('ev'), sd.get('dk'), sd['pert'],
                              row['worse'], row.get('off_eq_f7b'), row['on']['rows_bad'][:1], row['off']['rows_bad'][:1],
                              flush=True)
    finally:
        h12.cleanup()
    print('done', kind, part, 'new', n_new, 'flagged', n_worse, 'skipped(done before)', len(jl.done), round(time.time() - t_s), 's')


if __name__ == '__main__':
    main(*sys.argv[1:3])
