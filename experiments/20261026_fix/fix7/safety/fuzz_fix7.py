"""直す 7-1 回目・係 safety: でたらめな閉じたループ(本物の drive._loop・偽物の ESC と車)に、車両の値の eRPM の跳ね(1 サンプル・
本物の記録の並び)を足して、直したあと(今)と直す前の写し(tests/fixtures/fix7_safety_before = fix6 r2 のあと)を比べる。
場面・決まりは fix4/review_safety/fuzz_closed.py(K1〜K7・K9)と fix5/safety/fuzz_fix5.py(R1・R2・R4・R5・R7)をそのまま
読み込む(台本は変えない)。1 つの場面を 6 通り:
  now1 = 今・rec_back_guard 1・rec_hold_keep 1(ほかのスイッチは でたらめ)   bef1 = 直す前の写しの同じ
  nowg = 今・見張りだけ(keep 0)                                               befg = 直す前の写しの同じ
  off4 = 4 つのスイッチ(rec_back_guard・rec_hold_keep・launch_boost・launch_no_kick)を全部 0 を今と直す前で → ビット単位で同じか
数えるもの: 決まりを破った数(今 / 直す前)・「下がらない(0.1 m 未満)まま escape が前の壁を 0.2 秒より押した」リカバリーの数
(指摘の形。zero_back_push)・後ろの当たり・前の壁を押した秒の合計。装置・GPU・API には触れない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261026_fix/fix7/safety/fuzz_fix7.py [本数] [種]
出力: fix7/safety/out/fuzz_fix7_<種>.json。一時の場所 fix7/tmp_safety/fz_<種>(終わったら消す)
"""
import collections
import json
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIX = HERE.parents[1]
sys.path.insert(0, str(FIX / 'fix5' / 'safety'))
import fuzz_fix5 as F          # noqa: E402

R = F.R
FC = F.FC
A, C, N, T0 = F.A, F.C, F.N, F.T0
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7401
R.TMP = FIX / 'fix7' / 'tmp_safety' / ('fz_%d' % SEED)
R.CL.TMP = R.TMP
R.SNAP_SHA['fix7_safety_before'] = {'drive.py': '696ec9e8e4b853ebd221428768a23e1a48ddf566d11d81df7c5279f82ea15fec',
                                    'assist.py': '951a4c122c954a12ecc5056d823729ab9934112977f30645f92b14b556bf50f4'}
OFF4 = F.OFF4
BURSTS = [[1322], [1115], [1628, 0, 1525, 0, 0, 1572, 0, 0, 1289],
          [1687, 0, 0, 0, 0, 0, 1411, -1270, 1408, -2156, 1356, 0, 0, -394, 1528, 1358, 656, 385]]


class GSim(FC.FSim):
    """FSim(止まらない衝撃)+ scene['glitch'] = [(t, eRPM)]: その時刻に来た車両の値のサンプルの eRPM だけを差し替える"""

    def _advance(self, t_end):
        gl = self.s.get('glitch') or []
        if not gl:
            return super()._advance(t_end)
        while self.t + R.CL.DT <= t_end + 1e-9:
            nt = round(self.t + R.CL.DT, 6)
            last = self.veh_buf[-1] if self.veh_buf else None
            super()._advance(nt)
            if self.veh_buf and self.veh_buf[-1] is not last:
                for ts, e in gl:
                    if abs(nt - ts) < R.CL.DT / 2:
                        self.veh_buf[-1][1]['erpm'] = int(e)
        super()._advance(t_end)


def add_glitch(rng, sc):
    gl = []
    for _ in range(rng.randint(3, 12)):
        t0 = round(round(rng.uniform(1.5, sc['dur'] - 0.3) / 0.02) * 0.02, 2)
        seq = rng.choice(BURSTS) if rng.random() < 0.3 else [rng.choice([301, 313, 1115, 1322, 1687])]
        step = 0.02
        for i, e in enumerate(seq[::2] if len(seq) > 1 else seq):     # 50 Hz の車両の値なので並びは 1 つおき
            if e:
                gl.append((round(t0 + step * i, 2), e))
    sc['glitch'] = gl
    return sc


def zero_back_push(sim, rec):
    """リカバリーの区切り(assist = リカバリーの行が続く間)ごとに、escape(中立 − 130)までに下がった距離と、escape の 0.5 秒で
    前の壁を押した秒。下がった距離 < 0.1 m で押した秒 > 0.2 秒の数"""
    rows = [((t - T0) / 1e9, r) for t, r in rec.rows if not r['hold']]
    segs, cur = [], []
    for s, r in rows:
        if int(r['assist']) == A.ASSIST_RECOVERY and r['select']:
            cur.append((s, float(r['out'][1])))
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    n = 0
    esc = N - A.Recovery.ESCAPE_US
    for sg in segs:
        t0 = sg[0][0]
        te = next((s for s, th in sg if abs(th - esc) < 1e-6), None)
        if te is None:
            continue
        xs = [x for t, x, *_ in sim.log if t0 <= t <= te]
        if not xs:
            continue
        back = xs[0] - min(xs)
        push = sum(R.CL.DT for t, x, v, thr, con, rv, sl in sim.log if te <= t < te + 0.5 and con == 'front'
                   and abs(thr - esc) < 1e-6)
        n += back < 0.1 and push > 0.2
    return n


def main(n=200, seed=SEED):
    rng = random.Random(seed)
    BEF = R.load_snapshot('fix7_safety_before')
    out = {'seed': seed, 'runs': 0, 'identity': collections.Counter(), 'bad_now1': collections.Counter(),
           'bad_bef1': collections.Counter(), 'bad_nowg': collections.Counter(), 'bad_befg': collections.Counter(),
           'cmp': collections.Counter(), 'examples': [], 'guards_now1': collections.Counter()}
    orig = R.CL.Sim
    R.CL.Sim = GSim
    try:
        for i in range(n):
            sc = add_glitch(rng, FC.rand_scene(rng))
            p1 = FC.rand_params(rng)
            pg = dict(p1, rec_hold_keep=0)
            del FC.PH[:]
            runs = {'now1': R.run_closed(p1, sc, wrap=FC.wrap)}
            ph1 = list(FC.PH)
            runs['bef1'] = R.run_closed(p1, sc, mod=BEF[0], amod=BEF[1])
            runs['nowg'] = R.run_closed(pg, sc)
            runs['befg'] = R.run_closed(pg, sc, mod=BEF[0], amod=BEF[1])
            poff = dict(p1, **OFF4)
            r_off_n = R.run_closed(poff, sc)
            r_off_b = R.run_closed(poff, sc, mod=BEF[0], amod=BEF[1])
            okf, whyf = F.same_run(r_off_n, r_off_b)
            out['identity']['off4_same' if okf else 'off4_DIFF'] += 1
            ok1, _ = F.same_run(runs['now1'], runs['bef1'])
            out['identity']['now1_same_bef1' if ok1 else 'now1_diff_bef1'] += 1
            okg, _ = F.same_run(runs['nowg'], runs['befg'])
            out['identity']['nowg_same_befg' if okg else 'nowg_diff_befg'] += 1
            m0 = FC.metrics(r_off_n[0], r_off_n[1])
            for nm, (s_, rc, pc, _) in runs.items():
                p = p1 if nm.endswith('1') else pg
                bad = FC.rules_one(s_, rc, p) + F.extra_rules(s_, rc, pc, p)
                m = FC.metrics(s_, rc)
                zb = zero_back_push(s_, rc)
                out['cmp']['zero_back_push_' + nm] += zb
                out['cmp']['front_push_s_' + nm] += m['front_push_s']
                out['cmp']['rear_hits_' + nm] += len(m['rear'])
                out['cmp']['rear_max_v_sum_' + nm] += m['rear_max_v']
                if m['rear_max_v'] > m0['rear_max_v'] + 0.05:
                    out['cmp']['rear_faster_than_off_' + nm] += 1
                for b in bad:
                    out['bad_' + nm][b[0]] += 1
                    if nm == 'now1' and sum(1 for x in out['examples'] if x['kind'] == b[0]) < 4:
                        out['examples'].append({'kind': b[0], 'run': i, 'detail': b[1:]})
            zb_off = zero_back_push(r_off_n[0], r_off_n[1])
            out['cmp']['zero_back_push_off'] += zb_off
            out['cmp']['front_push_s_off'] += m0['front_push_s']
            out['cmp']['rear_hits_off'] += len(m0['rear'])
            for e in runs['now1'][1].events:
                if e[0] == 'recovery_guard':
                    out['guards_now1'][e[1]['what']] += 1
            if zero_back_push(*runs['now1'][:2]) and sum(1 for x in out['examples'] if x['kind'] == 'zero_back_push') < 6:
                out['examples'].append({'kind': 'zero_back_push', 'run': i,
                                        'params': {k: p1[k] for k in ('rev_s', 'rev_us', 'launch_boost', 'start_kick')},
                                        'scene': {k: v for k, v in sc.items() if k != 'model'}})
            out['runs'] += 1
            if (i + 1) % 25 == 0:
                print(i + 1, dict(out['identity']), {k: round(v, 2) for k, v in out['cmp'].items() if k.startswith('zero')},
                      flush=True)
    finally:
        R.CL.Sim = orig
        R.cleanup()
        shutil.rmtree(R.TMP, ignore_errors=True)
    for k in ('identity', 'bad_now1', 'bad_bef1', 'bad_nowg', 'bad_befg', 'cmp', 'guards_now1'):
        out[k] = {kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in dict(out[k]).items()}
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / ('fuzz_fix7_%d.json' % seed)).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != 'examples'}, ensure_ascii=False))


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:3]))
