"""fix8 係 safety 直す 8-1 回目: 案 qA・qB・qC を、打ち消しの票 A の 19 の ESC の模型 × 16 場面(sweep_A.py と同じ。10/23 の形 = 前進 m µs で
走り、壁 xf の db 手前からブレーキ側 → 止まって ESC が後退を待つ → リカバリー)で、今の両方 ON・両方 OFF と並べる。後ろの物は置かずに
走り、最初のリカバリーの始めの位置から d [m](0.1〜2.0 m・39 点)後ろに物があったら最初に当たる速さを出す(票 A の hit_profile。
当たる前の動きは物があってもなくても同じ。check_r1.py で物を置いた走りと照らす)。下がる距離・最高の後ろへの速さ・escape が前の
壁を押した秒も出す。案の写しの両方 OFF も流し、今の OFF と書いた値の列が同じかを見る。
    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -B experiments/20261026_fix/fix8/proposal_r1/sweep_r1.py [part/parts]
    → out/sweep_r1[_part].json
"""
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault('R1_TAG', 'sweep_%d' % os.getpid())
import cl_r1  # noqa: E402

DS = [round(0.1 + 0.05 * k, 3) for k in range(39)]        # 0.1〜2.0 m(票 A と同じ)
FORMS = (('now', 'on'), ('now', 'off'), ('qA', 'on'), ('qB', 'on'), ('qC', 'on'), ('qA', 'off'), ('qB', 'off'), ('qC', 'off'))


def scenes():
    for m, (xf, db), bu, vdt in itertools.product((120.0, 200.0), ((2.4, 0.45), (3.0, 0.6)), (200.0, 100.0), (0.01, 0.02)):
        yield {'m': m, 'xf': xf, 'db': db, 'brake_us': bu, 'veh_dt': vdt}


def escape_push(sim, pca, t0, t1):
    """最初のリカバリーの escape(前進 中立 − ESCAPE_US)を書いていた間に、前の壁に触れていた秒と、その間の前の当たり"""
    esc = cl_r1.N - cl_r1.A.Recovery.ESCAPE_US
    te = next((t for t, s, th in pca.w if t >= t0 - 1e-9 and abs(th - esc) < 1e-6), None)
    if te is None:
        return None, 0.0, []
    tend = (t1 if t1 is not None else te + 0.6) + 0.05
    push = sum(cl_r1.DT for r in sim.log if te <= r[0] < tend and r[4] == 'front')
    hits = [h for h in sim.car.hits if h[1] == 'front' and te <= h[0] <= tend]
    return round(te, 3), round(push, 2), hits


def one(esc, ver, sw, sc):
    sim, rec, pca, ph = cl_r1.run(cl_r1.ESC[esc], sw, m=sc['m'], xf=sc['xf'], db=sc['db'], veh_dt=sc['veh_dt'],
                                  brake_us=sc['brake_us'], dur=6.5, ver=ver)
    t0, t1 = cl_r1.first_rec(ph)
    w = [(round(t, 4), s, th) for t, s, th in pca.w]
    out = {'w_sha': hashlib.sha256(json.dumps(w).encode()).hexdigest()[:16]}
    if t0 is None:
        out['rec'] = None
        return out
    prof, x0, back, vpk = cl_r1.hit_profile(sim, t0, t1 if t1 is not None else 99.0, DS)
    pre = [(round(r[1], 6), round(r[2], 6), r[3]) for r in sim.log if r[0] < t0 - 1e-9]
    ev = [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']
    phs, pp = [], None
    for tt, p0, p1, *_ in ph:
        if t0 - 0.01 <= tt <= (t1 if t1 is not None else 99) and p1 != pp:
            phs.append((tt, p1))
            pp = p1
    te, push, fh = escape_push(sim, pca, t0, t1)
    # 後退側(中立 + 40 µs より上)を書いていた秒(最初のリカバリーの中)
    rev_s = 0.0
    seq = [(t, th) for t, s, th in pca.w if t0 - 1e-9 <= t <= (t1 if t1 is not None else 99)]
    for (ta, tha), (tb, _) in zip(seq, seq[1:]):
        if tha > cl_r1.N + 40:
            rev_s += tb - ta
    out.update({'rec': [t0, t1], 'x0': round(x0, 4), 'back': back, 'vpk': vpk, 'hit': {str(k): v for k, v in prof.items()},
                'pre_sha': hashlib.sha256(json.dumps(pre).encode()).hexdigest()[:16], 'events': ev, 'phases': phs[:10],
                't_escape': te, 'esc_push': push, 'esc_front_hits': fh, 'rev_cmd_s': round(rev_s, 3),
                # escape(0.6 秒)の間の前の壁までの最短 [m](0 = 前の壁に触れた)
                'front_gap_in_escape': round(sc['xf'] - max([r[1] for r in sim.log if te is not None and te <= r[0] <= te + 0.6]
                                                            or [x0]), 3)})
    return out


def main(part='0/1'):
    cl_r1.check_now()
    pi, pn = (int(x) for x in part.split('/'))
    res = []
    t_s = time.time()
    cells = [(esc, sc) for esc in sorted(cl_r1.ESC) for sc in scenes()]
    try:
        for k, (esc, sc) in enumerate(cells):
            if k % pn != pi:
                continue
            row = {'esc': esc, **sc}
            for ver, sw in FORMS:
                row['%s_%s' % (ver, sw)] = one(esc, ver, sw, sc)
            res.append(row)
            if len(res) % 16 == 0:
                print(esc, len(res), '%.0fs' % (time.time() - t_s), flush=True)
    finally:
        cl_r1.cleanup()
    name = 'sweep_r1.json' if pn == 1 else 'sweep_r1_%dof%d.json' % (pi, pn)
    (HERE / 'out' / name).write_text(json.dumps({'ds': DS, 'esc': cl_r1.ESC, 'forms': FORMS, 'rows': res,
                                                 'elapsed_s': round(time.time() - t_s, 1)}, ensure_ascii=False))
    print('done', len(res), '%.0fs' % (time.time() - t_s))


if __name__ == '__main__':
    main(*sys.argv[1:2])
