"""fix8 係 safety-proposal: fix7/safety_r2/replay_all9.py の写し。本物の記録(data/raw の AI_CMD がある記録。読むだけ)を、
tests/test_recovery_back_guard の _replay と同じ形(開いたループ: 記録の車両の値・IMU・モデルの出力をそのまま入れる)で Assist に流し、
今(base = jetformer/ の写し)と案 pA・pB・pC の出す値を比べる。形:
  on    : 見張り ON(rec_guard True。書いた値を EscMayReverse に入れて esc_rev を渡す = drive と同じ)  base vs pA / pB / pC
  off   : 見張り OFF(rec_guard False・esc_rev None)                                                 base vs pA / pB / pC(ON だけの案 = 同じはず)
  offx  : 見張り OFF・案を OFF にも入れた形(写しの P8_OFF_TOO = True)                               base vs pA / pB / pC
どの形も、それぞれの Assist に自分の EscMayReverse を持たせる(自分の出した値を入れる)。開いたループなので、初めて違ったフレームの
あとは車の動きが記録と合わない。数えるもの: 本数・フレーム・リカバリーの始め・初めて違った記録の数とそのフレーム(段・eRPM)・
違ったフレームの数の合計(初めて違ったあとも、同じ入力で出す値が違ったフレームを数える)。
出力 out/replay10.json。装置・GPU・API には触れない
    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -I experiments/20261026_fix/fix8/proposal/replay10.py
"""
import collections
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')

RAW = Path('/home/jetson/jetformer/data/raw')
HERE = Path(__file__).resolve().parent
PSHA = {'base': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f',
        'pA': '82e2f8a584c66fe292d2122800dcacb928d93c24aba78ca53faa5331915528d3',
        'pB': '05d78b8aeaf09a64364e284191ce19259b3b0baee4d449674056f3fe6228e5b6',
        'pC': '7ac97688a93713704762dc29f3ff64d17dc2ec4b758332ec4200ffd2240585b8'}
PROPS = ('pA', 'pB', 'pC')


def load(name, tag, offtoo=False):
    fp = HERE / name / 'assist.py'
    assert hashlib.sha256(fp.read_bytes()).hexdigest() == PSHA[name], name
    spec = importlib.util.spec_from_loader('jetformer._p8_assist_%s_%s' % (name, tag), loader=None)
    m = importlib.util.module_from_spec(spec)
    exec(compile(fp.read_text(encoding='utf-8'), str(fp), 'exec'), m.__dict__)
    if offtoo:
        m.P8_OFF_TOO = True
    return m


def replay(d, mods):
    """mods = {形の名前: (モジュール, rec_guard)}。戻り値: 形ごとの (初めて違ったフレーム, 違ったフレームの数)"""
    z = np.load(d / 'AI_CMD.npz')
    v = np.load(d / 'VEHICLE.npz')
    im = np.load(d / 'IMU.npz')
    L = json.loads((d.parent / 'log.json').read_text(encoding='utf-8'))
    dr = L.get('drive') or {}
    p = dr.get('params') or {}
    ref = (float(dr.get('steer_center_us', 1441)), float(dr.get('throttle_neutral_us', 1496)))
    t = z['timestamps_us'].astype(np.int64)
    sel = z['select'].astype(bool)
    hold = z['hold'].astype(bool)
    dry = z['dry_run'].astype(bool) if 'dry_run' in z else np.zeros(len(t), bool)
    mo = z['model_out'].astype(float) if 'model_out' in z else z['out'].astype(float)
    lat = z['latency_ms'].astype(float) if 'latency_ms' in z else np.full(len(t), np.nan)
    vt = v['timestamps_us'].astype(np.int64)
    ve = v['erpm'].astype(float)
    it = im['timestamps_us'].astype(np.int64)
    acc = im['accel'].astype(float)
    hg = np.hypot(acc[:, 0], acc[:, 1]) / 9.80665
    st = {}
    for k, (m, guard) in mods.items():
        st[k] = {'a': m.Assist(ref), 'gate': m.EscMayReverse(ref[1]), 'q': None, 'guard': guard, 'm': m,
                 'first': None, 'ndiff': 0, 'events': collections.Counter(), 'out': None}
    kw = dict(esc_on=p.get('esc_reverse', 1) == 1, rec_on=p.get('recovery', 1) == 1, rev_us=float(p.get('rev_us', 300)),
              rev_s=float(p.get('rev_s', 0.4)), kick_on=int(p.get('start_kick', 0) or 0) == 1)
    info = {'frames': 0, 'rec_starts': 0, 'reverse_frames': 0}
    for i in range(len(t)):
        now_us = t[i] + (lat[i] * 1000 if np.isfinite(lat[i]) else 0)
        ns = int(now_us * 1000)
        if hold[i]:
            for s in st.values():
                s['a'].launch_hold(); s['a'].reset()
                if not sel[i] or dry[i]:
                    s['gate'].unknown(ns)
                else:
                    s['gate'].write(ns, ref[1])
            continue
        jv = np.searchsorted(vt, t[i], side='right') - 1
        e = ve[jv] if jv >= 0 else 0.0
        ji = np.searchsorted(it, now_us, side='right')
        idx = np.arange(max(0, ji - 10), ji)
        idx = idx[it[idx] > t[i] - 150_000]
        ig = float(hg[idx].max()) if len(idx) else 0.0
        s_ = bool(sel[i]) and not dry[i]
        b = st['base_on']['a'].rec.phase
        for k, s in st.items():
            er = None
            if s['guard']:
                if s['q'] is not None:
                    er = s['gate'].rev_s(s['q'], ns) > 0.0
                s['q'] = ns
            ph0 = s['a'].rec.phase
            o, kk = s['a'].step((mo[i, 0], mo[i, 1]), e, s_, now_us / 1e6, impact_g=ig, esc_rev=er, rec_guard=s['guard'], **kw)
            ev, s['a'].rec.guard_event = s['a'].rec.guard_event, None
            if ev:
                s['events'][ev[0]] += 1
            s['out'] = (o, kk, ph0, s['a'].rec.phase, ev and ev[0])
            if s_:
                s['gate'].write(ns + 1, o[1])
            else:
                s['gate'].unknown(ns + 1)
        if b is None and st['base_on']['a'].rec.phase == 'brake':
            info['rec_starts'] += 1
        if b == 'reverse':
            info['reverse_frames'] += 1
        info['frames'] += 1
        for k, s in st.items():
            if k.startswith('base'):
                continue
            ref_k = 'base_on' if k.endswith('_on') else 'base_off'
            r0, r1 = st[ref_k]['out'], s['out']
            if r0[0] != r1[0] or r0[1] != r1[1]:
                s['ndiff'] += 1
                if s['first'] is None:
                    s['first'] = {'i': i, 's': round((t[i] - t[0]) / 1e6, 3), 'phase0': r1[2], 'phase_base': r0[3],
                                  'phase_prop': r1[3], 'erpm': float(e), 'out_base': r0[0], 'out_prop': r1[0],
                                  'ev_prop': r1[4], 'ev_base': r0[4]}
    return info, {k: (s['first'], s['ndiff'], dict(s['events'])) for k, s in st.items()}


def main():
    base = load('base', 'b')
    mods = {'base_on': (base, True), 'base_off': (base, False)}
    for pn in PROPS:
        m = load(pn, 'n')
        mods[pn + '_on'] = (m, True)
        mods[pn + '_off'] = (m, False)
        mods[pn + 'x_off'] = (load(pn, 'x', offtoo=True), False)
    res = {'files': 0, 'skipped': 0, 'frames': 0, 'rec_starts': 0, 'reverse_frames_base_on': 0,
           'by_variant': {k: {'files_diverged': 0, 'frames_diff': 0, 'first': []} for k in mods if not k.startswith('base')},
           'events': {k: collections.Counter() for k in mods}}
    for d in sorted(RAW.glob('2026*/*')):
        if not (d / 'AI_CMD.npz').exists() or not (d / 'VEHICLE.npz').exists() or not (d / 'IMU.npz').exists():
            continue
        if not (d.parent / 'log.json').exists():
            res['skipped'] += 1
            continue
        try:
            info, r = replay(d, mods)
        except Exception as ex:      # noqa: BLE001  壊れた記録・古い形は数えて飛ばす(replay_all9 と同じ)
            res['skipped'] += 1
            print('skip', d.parent.name + '/' + d.name, repr(ex)[:120])
            continue
        res['files'] += 1
        res['frames'] += info['frames']
        res['rec_starts'] += info['rec_starts']
        res['reverse_frames_base_on'] += info['reverse_frames']
        for k, (first, nd, ev) in r.items():
            res['events'][k].update(ev)
            if k.startswith('base'):
                continue
            bv = res['by_variant'][k]
            bv['frames_diff'] += nd
            if first is not None:
                bv['files_diverged'] += 1
                bv['first'].append(dict(first, file=d.parent.name + '/' + d.name, ndiff=nd))
                print('diverge', k, d.parent.name + '/' + d.name, first, nd, flush=True)
    res['events'] = {k: dict(v) for k, v in res['events'].items()}
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'replay10.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != 'by_variant'}, ensure_ascii=False))
    print({k: (v['files_diverged'], v['frames_diff']) for k, v in res['by_variant'].items()})


if __name__ == '__main__':
    main()
