"""fix8 係 safety-proposal: fix7/safety_r2/fuzz_fix9.py の写し。1 つの場面を 今(base)・案 pA・pB・pC の ON(両方 1)と、
4 つのスイッチ全部 0(案の写しが今とビット単位で同じか)で回す。base・案はどれも fix8/proposal/<名前>/ の写しを読む(jetformer/ は読まない)。
出力 fix8/proposal/out/fuzz10_<種>.json。一時の場所 fix8/tmp_proposal/fz_<種>(終わったら消す)。
元の説明: 直す 7-2 回目・係 safety(7-1 回目の safety/fuzz_fix7.py の写し。直す前を tests/fixtures/fix7_safety_r2_before = 7-1 回目のあとにし、
場面に 0.10〜0.14 秒の短いカメラの止まりを 1〜3 か所足した。一時の場所・出力の名前も変えた):
でたらめな閉じたループ(本物の drive._loop・偽物の ESC と車)に、車両の値の eRPM の跳ね(1 サンプル・
本物の記録の並び)を足して、直したあと(今)と直す前の写しを比べる。
場面・決まりは fix4/review_safety/fuzz_closed.py(K1〜K7・K9)と fix5/safety/fuzz_fix5.py(R1・R2・R4・R5・R7)をそのまま
読み込む(台本は変えない)。1 つの場面を 6 通り:
  now1 = 今・rec_back_guard 1・rec_hold_keep 1(ほかのスイッチは でたらめ)   bef1 = 直す前の写しの同じ
  nowg = 今・見張りだけ(keep 0)                                               befg = 直す前の写しの同じ
  off4 = 4 つのスイッチ(rec_back_guard・rec_hold_keep・launch_boost・launch_no_kick)を全部 0 を今と直す前で → ビット単位で同じか
数えるもの: 決まりを破った数(今 / 直す前)・「下がらない(0.1 m 未満)まま escape が前の壁を 0.2 秒より押した」リカバリーの数
(指摘の形。zero_back_push)・後ろの当たり・前の壁を押した秒の合計。装置・GPU・API には触れない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261026_fix/fix7/safety_r2/fuzz_fix9.py [本数] [種]
出力: fix7/safety_r2/out/fuzz_fix9_<種>.json。一時の場所 fix7/tmp_safety_r2/fz_<種>(終わったら消す)
"""
import collections
import json
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIX = HERE.parents[1]
sys.path.insert(0, str(HERE))
import h10  # noqa: E402
sys.path.insert(0, str(FIX / 'fix5' / 'safety'))
import fuzz_fix5 as F          # noqa: E402

R = F.R
FC = F.FC
A, C, N, T0 = F.A, F.C, F.N, F.T0
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7901
R.TMP = FIX / 'fix8' / 'tmp_proposal' / ('fz_%d' % SEED)
R.CL.TMP = R.TMP
R.SNAP_SHA['fix7_safety_r2_before'] = {'drive.py': '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f',
                                       'assist.py': '6a8950d9963bb4a663dd540cb4b819f13f22f9da956a56a657c98a66ea717e3f'}
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
    for _ in range(rng.randint(1, 3)):          # 7-2 回目: 短いカメラの止まり(中立止めが 0.1 秒より短い)
        a = round(rng.uniform(1.5, sc['dur'] - 0.5), 2)
        sc['drops'].append((a, round(a + rng.choice([0.10, 0.11, 0.12, 0.14]), 3)))
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


VS = ('base1', 'pA1', 'pB1', 'pC1')


def main(n=200, seed=SEED):
    rng = random.Random(seed)
    MODS = {v: h10.load(v[:-1] if v != 'base1' else 'base') for v in VS}
    out = {'seed': seed, 'runs': 0, 'identity': collections.Counter(), 'cmp': collections.Counter(), 'examples': [],
           'guards_base1': collections.Counter(), 'diff_runs': collections.defaultdict(list)}
    for v in VS:
        out['bad_' + v] = collections.Counter()
    orig = R.CL.Sim
    R.CL.Sim = GSim
    try:
        for i in range(n):
            sc = add_glitch(rng, FC.rand_scene(rng))
            p1 = FC.rand_params(rng)
            runs = {}
            for v in VS:
                mod, amod = MODS[v]
                if v == 'base1':
                    del FC.PH[:]
                    runs[v] = R.run_closed(p1, sc, mod=mod, amod=amod, wrap=FC.wrap)
                else:
                    runs[v] = R.run_closed(p1, sc, mod=mod, amod=amod)
            poff = dict(p1, **OFF4)
            r_off_b = R.run_closed(poff, sc, mod=MODS['base1'][0], amod=MODS['base1'][1])
            for v in VS[1:]:
                r_off_p = R.run_closed(poff, sc, mod=MODS[v][0], amod=MODS[v][1])
                okf, whyf = F.same_run(r_off_b, r_off_p)
                out['identity'][v + '_off4_same' if okf else v + '_off4_DIFF'] += 1
                ok1, why1 = F.same_run(runs['base1'], runs[v])
                out['identity'][v + '_same_base1' if ok1 else v + '_diff_base1'] += 1
                if not ok1:
                    out['diff_runs'][v].append(i)
            m0 = FC.metrics(r_off_b[0], r_off_b[1])
            for nm, (s_, rc, pc, _) in runs.items():
                bad = FC.rules_one(s_, rc, p1) + F.extra_rules(s_, rc, pc, p1)
                m = FC.metrics(s_, rc)
                zb = zero_back_push(s_, rc)
                out['cmp']['zero_back_push_' + nm] += zb
                out['cmp']['front_push_s_' + nm] += m['front_push_s']
                out['cmp']['rear_hits_' + nm] += len(m['rear'])
                out['cmp']['rear_max_v_sum_' + nm] += m['rear_max_v']
                out['cmp']['min_x_sum_' + nm] += min(x for _, x, *_r in s_.log)
                if m['rear_max_v'] > m0['rear_max_v'] + 0.05:
                    out['cmp']['rear_faster_than_off_' + nm] += 1
                if nm != 'base1':
                    mb = FC.metrics(runs['base1'][0], runs['base1'][1])
                    if m['rear_max_v'] > mb['rear_max_v'] + 0.05:
                        out['cmp']['rear_faster_than_base1_' + nm] += 1
                    if len(m['rear']) > len(mb['rear']):
                        out['cmp']['more_rear_hits_than_base1_' + nm] += 1
                    xb = min(x for _, x, *_r in runs['base1'][0].log)
                    xm = min(x for _, x, *_r in s_.log)
                    if xm < xb - 0.05:
                        out['cmp']['backs_more_than_base1_' + nm] += 1
                for b in bad:
                    out['bad_' + nm][b[0]] += 1
                    if sum(1 for x in out['examples'] if x['kind'] == b[0] and x['v'] == nm) < 3:
                        out['examples'].append({'kind': b[0], 'v': nm, 'run': i, 'detail': b[1:]})
            out['cmp']['zero_back_push_off'] += zero_back_push(r_off_b[0], r_off_b[1])
            out['cmp']['rear_hits_off'] += len(m0['rear'])
            for e in runs['base1'][1].events:
                if e[0] == 'recovery_guard':
                    out['guards_base1'][e[1]['what']] += 1
            out['runs'] += 1
            if (i + 1) % 25 == 0:
                print(i + 1, dict(out['identity']), {k: round(v, 2) for k, v in out['cmp'].items() if k.startswith('zero')},
                      flush=True)
    finally:
        R.CL.Sim = orig
        R.cleanup()
        shutil.rmtree(R.TMP, ignore_errors=True)
    for k in ['identity', 'cmp', 'guards_base1', 'diff_runs'] + ['bad_' + v for v in VS]:
        out[k] = {kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in dict(out[k]).items()}
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / ('fuzz10_%d.json' % seed)).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != 'examples'}, ensure_ascii=False))


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:3]))
