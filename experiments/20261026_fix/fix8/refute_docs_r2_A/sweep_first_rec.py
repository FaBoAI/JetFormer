"""打ち消しの票 A(反証 8-2・係 docs の指摘 f8r2-docs-1023-first-recovery-off を確かめる)。直さない・装置に触れない・API を呼ばない。

本物の drive._loop(今の jetformer/。sha256 を照らす)を、fix8/refute_safety_r1_A/cl_A.py の写し(cl_mine.py。TMP だけ自分の場所)の
閉じたループの偽物の ESC で回す。後ろの物は置かずに走り、最初のリカバリーの始めの位置から
  back      = リカバリーの間(始め〜段が None に戻るまで)にいちばん後ろへ行った距離 [m]
  t_1m/v_1m = 始めから 1.0 m 下がるまでの秒と、そのときの後ろへの速さ(下がらなければ None)
  hit[d]    = 始めの位置から d m 後ろに物があったら最初に当たる速さ(cl_A.hit_profile と同じ)
を、ON(見張り・残す)と OFF で出す。形は 2 つ:
  '1023'   : 10/23 の形(前進 m µs → 壁 xf の db 手前からブレーキ側 → 止まって ESC が後退を待つ → リカバリー)。sweep_A と同じ 16 場面
  'wall'   : ふつうの形(前進 m µs のまま前の壁に当たって止まる → リカバリー)。8 場面
ESC の模型は cl_A.ESC の 18 通り(fit = 10/23 No.6 に合わせた模型・cl・human と fit のまわり)。
    jf-heavy … python -I -B sweep_first_rec.py → out/sweep_first_rec.json
"""
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cl_mine as cl  # noqa: E402

NOW = {'drive.py': '3f2e31a22ecb9abdd9c8f84b3f787d075ef636b6d5679bad5c0fc28aee18388e',
       'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'}
DS = [round(0.1 + 0.05 * k, 3) for k in range(39)]        # 0.1〜2.0 m
C, N = cl.C, cl.N


def scenes_1023():
    for m, (xf, db), bu, vdt in itertools.product((120.0, 200.0), ((2.4, 0.45), (3.0, 0.6)), (200.0, 100.0), (0.01, 0.02)):
        yield {'form': '1023', 'm': m, 'xf': xf, 'db': db, 'brake_us': bu, 'veh_dt': vdt}


def scenes_wall():
    for m, xf, vdt in itertools.product((120.0, 200.0), (2.4, 3.0), (0.01, 0.02)):
        yield {'form': 'wall', 'm': m, 'xf': xf, 'veh_dt': vdt}


def fwd_model(m):
    def f(t, car):
        return (C, N - m)
    return f


def run_one(esc, sw, sc):
    if sc['form'] == '1023':
        sim, rec, pca, ph = cl.run(cl.ESC[esc], sw, m=sc['m'], xf=sc['xf'], db=sc['db'], veh_dt=sc['veh_dt'],
                                   brake_us=sc['brake_us'], dur=6.5)
    else:
        s = {'dur': 7.0, 'xf': sc['xf'], 'ai_from': 1.0, 'model': fwd_model(sc['m']), 'esc': cl.ESC[esc]}
        if sc['veh_dt'] != 0.02:
            s['veh_dt'] = sc['veh_dt']
        sim, rec, pca = cl._run(cl.base_params(**cl.SW[sw]), s)
        ph = list(cl.CL.PH)
    t0, t1 = cl.first_rec(ph)
    if t0 is None:
        return {'rec': None}
    te = t1 if t1 is not None else 99.0
    prof, x0, back, vpk = cl.hit_profile(sim, t0, te, DS)
    log = [r for r in sim.log if t0 - 1e-9 <= r[0] <= te + 1e-9]
    t_1m = v_1m = None
    for t, x, v, *_ in log:
        if x <= x0 - 1.0:
            t_1m, v_1m = round(t - t0, 3), round(-v, 3)
            break
    x_b = next((r[1] for r in log if r[0] >= t0 + 0.25 - 1e-9), x0)          # 'brake' の 0.25 秒で下がった分
    pre = [h for h in sim.car.hits if h[0] < t0 + 1e-9]
    ev = [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']
    phs, pp = [], None
    for tt, p0, p1, *_ in ph:
        if t0 - 0.01 <= tt <= te and p1 != pp:
            phs.append((tt, p1))
            pp = p1
    return {'rec': [t0, t1], 'x0': round(x0, 4), 'back': back, 'vpk': vpk, 't_1m': t_1m, 'v_1m': v_1m,
            'back_in_brake': round(x0 - x_b, 3), 'hit': {str(k): v for k, v in prof.items()}, 'events': ev,
            'phases': phs[:8], 'hits_before': pre, 'contact_at_t0': next((r[4] for r in sim.log if r[0] >= t0 - 1e-9), None)}


def main():
    for fn, h in NOW.items():
        assert hashlib.sha256((cl.ROOT / 'jetformer' / fn).read_bytes()).hexdigest() == h, fn
    only = sys.argv[1:]          # 試しに 1 つの ESC だけ
    res = []
    t_s = time.time()
    try:
        for esc in sorted(cl.ESC):
            if only and esc not in only:
                continue
            for sc in list(scenes_1023()) + list(scenes_wall()):
                row = {'esc': esc, **sc}
                for sw in ('on', 'off'):
                    row[sw] = run_one(esc, sw, sc)
                res.append(row)
            print(esc, len(res), '%.0fs' % (time.time() - t_s), flush=True)
    finally:
        cl.cleanup()
    name = 'sweep_first_rec%s.json' % ('_' + '_'.join(only) if only else '')
    (HERE / 'out' / name).write_text(json.dumps({'ds': DS, 'esc': cl.ESC, 'sha': NOW, 'rows': res}, ensure_ascii=False))
    print('出力', HERE / 'out' / name)


if __name__ == '__main__':
    main()
