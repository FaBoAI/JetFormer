"""票 A の広さ。1 設定 1 行の jsonl(途中で止まっても読める)。
  part s1: 'reverse' が終わるフレーム(止まりなしの走りで)から k フレームずらした所から L フレームのカメラの止まり
           ESC 24(r1_A の 18 + この台本の 6)× モデル 2(前進 120・10/23 の形のブレーキ壁)× カメラの位相 2 × 処理の時間 2
  part s2: 止まりの始めを 'brake' の 0.1 秒前から escape の終わりまで 1 フレームずつ(L 3・5)。ESC 6・前進 120・位相 0・処理 0
  part s3: モデルの前進の強さ 40・60・200 µs と前の壁 3.0 m(ESC 6)で s1 の置き方
  part s4: 後ろへ動いている間はモデルが中立(lag0.5・lag0.8 = その秒のあと前進 200・neu = ずっと)(ESC 6)で s1 の置き方
    python -I -B sweep_rA.py PART SHARD NSHARD → out/PART_SHARDofNSHARD.jsonl
"""
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rA  # noqa: E402

DS = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
KS = (-2, -1, 0, 1)
LS = (2, 3, 4, 6, 8, 10, 15)
ESC6 = ('fit', 'cl', 'human', 'fit_lag0.1', 'fit_coast0.0', 'human_fwd015')


def model(name, xf):
    if name.startswith('fwd'):
        return rA.model_fwd(float(name[3:]))
    if name == 'bw':
        return rA.model_brakewall(120.0, xf, 0.45)
    if name.startswith('lag') or name == 'neu':
        # 後ろへ動いている間はモデルが中立(本物の 211828-0007 の 'coast' の始めの 4 フレームの形)。lagT: 後ろへ動き始めて
        # T 秒たったら前進 200 µs。neu: 後ろへ動いている間はずっと中立(止まったら前進 120)
        tn = None if name == 'neu' else float(name[3:])
        st = {'t_back': None}

        def f(t, car):
            if car.v < -0.05:
                if st['t_back'] is None:
                    st['t_back'] = t
                if tn is not None and t - st['t_back'] >= tn:
                    return (rA.C, rA.N - 200.0)
                return (rA.C, rA.N)
            st['t_back'] = None
            return (rA.C, rA.N - 120.0)
        return f
    raise ValueError(name)


def metrics(sim, rec, pca, ph, t_ref0=None):
    t0, t1 = rA.first_rec(ph)
    if t0 is None:
        return None
    x0, back, vpk, hits = rA.profile(sim, t0, DS)
    fp = 0.0
    for (ta, x, v, thr, con, *_), (tb, *_r) in zip(sim.log, sim.log[1:]):
        if t0 <= ta <= t0 + 3.5 and con == 'front' and thr < rA.N - 20:
            fp += tb - ta
    ev = [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']
    gu = any((not r['hold']) and int(r['assist']) == rA.A.ASSIST_GIVEUP for _, r in rec.rows)
    return {'t0': t0, 't1': t1, 'rev_end': rA.rev_end(ph, t0), 'back': back, 'vpk': vpk,
            'hits': [hits[d] for d in DS], 'fpush': round(fp, 3), 'ev': ev, 'giveup': gu,
            'fhits': [h for h in sim.car.hits if h[1] == 'front' and h[0] > t0]}


def sig(rec, pca):
    return hashlib.sha256(json.dumps([[round(x, 3) for x in w] for w in pca.w]).encode()).hexdigest()[:16]


def configs(part):
    if part == 's1':
        for esc, mdl, ph, proc in itertools.product(sorted(rA.ESC), ('fwd120', 'bw'), (0.0, 1.0 / 60), (0.0, 0.02)):
            yield {'esc': esc, 'model': mdl, 'xf': 2.0, 'phase': round(ph, 5), 'proc': proc}
    elif part == 's2':
        for esc in ESC6:
            yield {'esc': esc, 'model': 'fwd120', 'xf': 2.0, 'phase': 0.0, 'proc': 0.0}
    elif part == 's3':
        for esc, mdl, xf in itertools.product(ESC6, ('fwd40', 'fwd60', 'fwd200', 'fwd120'), (2.0, 3.0)):
            if mdl == 'fwd120' and xf == 2.0:
                continue
            yield {'esc': esc, 'model': mdl, 'xf': xf, 'phase': 0.0, 'proc': 0.0}
    elif part == 's4':
        for esc, mdl in itertools.product(ESC6, ('lag0.5', 'lag0.8', 'neu')):
            yield {'esc': esc, 'model': mdl, 'xf': 2.0, 'phase': 0.0, 'proc': 0.0}


def one_config(part, cf):
    kw = dict(xf=cf['xf'], phase=cf['phase'], proc=cf['proc'], dur=7.0)
    out = {'cf': cf}
    base = {}
    for sw in ('on', 'off'):
        r = rA.run(cf['esc'], sw, model(cf['model'], cf['xf']), **kw)
        base[sw] = r
    out['base_same'] = (base['on'][2].w == base['off'][2].w and
                        [(x[1]['out'], x[1]['assist']) for x in base['on'][1].rows] ==
                        [(x[1]['out'], x[1]['assist']) for x in base['off'][1].rows])
    sim, rec, pca, ph = base['on']
    out['base'] = metrics(*base['on'])
    out['base_off'] = metrics(*base['off'])
    if out['base'] is None or out['base']['rev_end'] is None:
        out['note'] = 'no recovery / no reverse end'
        return out
    fe = rA.frame_at(sim, out['base']['rev_end'])
    out['rev_end_frame'] = fe
    placements = []
    if part in ('s1', 's3', 's4'):
        for k, ln in itertools.product(KS, LS):
            placements.append((k, ln, fe + k))
    else:
        fb = rA.frame_at(sim, out['base']['t0'])
        fz = rA.frame_at(sim, out['base']['t1']) if out['base']['t1'] is not None else fe + 25
        for i0 in range(fb - 3, fz + 1):
            for ln in (3, 5):
                placements.append((i0 - fe, ln, i0))
    rows = []
    for k, ln, i0 in placements:
        row = {'k': k, 'L': ln}
        for sw in ('on', 'off'):
            r = rA.run(cf['esc'], sw, model(cf['model'], cf['xf']), drop_frames=[(i0, ln)], **kw)
            row[sw] = metrics(*r)
        rows.append(row)
    out['rows'] = rows
    return out


def main():
    for fn, h in rA.NOW.items():
        assert hashlib.sha256((rA.ROOT / 'jetformer' / fn).read_bytes()).hexdigest() == h, fn
    part, shard, nsh = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    path = HERE / 'out' / ('%s_%dof%d.jsonl' % (part, shard, nsh))
    done = set()
    if path.exists():
        for ln in path.read_text().splitlines():
            done.add(json.dumps(json.loads(ln)['cf'], sort_keys=True))
    t_s = time.time()
    n = 0
    try:
        with path.open('a') as f:
            for i, cf in enumerate(configs(part)):
                if i % nsh != shard:
                    continue
                key = json.dumps(cf, sort_keys=True)
                if key in done:
                    continue
                res = one_config(part, cf)
                f.write(json.dumps(res, ensure_ascii=False) + '\n')
                f.flush()
                n += 1
                print(part, shard, n, cf, '%.0fs' % (time.time() - t_s), flush=True)
    finally:
        rA.cleanup()


if __name__ == '__main__':
    main()
