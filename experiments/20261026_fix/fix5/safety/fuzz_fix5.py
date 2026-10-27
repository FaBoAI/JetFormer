"""fix5 係 safety: 直したあとの でたらめな閉じたループ(本物の drive._loop・偽物の ESC と車 = fix3/refute2_safety/closed_loop.py)。
反証 4 回目の fix4/review_safety/fuzz_closed.py の場面と決まり(K1〜K7・K9)をそのまま使い(台本は変えない・読み込むだけ)、
この係の決まりを足す。1 つの場面を 5 通りで回す:
  now1  = 今の drive.py・assist.py、rec_hold_keep = 1・rec_back_guard = 1(ほかのスイッチは でたらめ)
  now0  = 今、rec_hold_keep = 0(ほかは now1 と同じ)
  bef1  = 直す前の写し(tests/fixtures/fix5_safety_before)、keep 1・見張り 1(比べ: 指摘がどれだけ減ったか)
  bef0  = 直す前の写し、keep 0  → now0 とビット単位で同じか(PCA9685 に書いた値・記録の行・出来事)
  off   = 今と直す前の写しの両方で、4 つのスイッチ(rec_back_guard・rec_hold_keep・launch_boost・launch_no_kick)を全部 0
          → ビット単位で同じか
足した決まり(now1 で見る):
  R1 PCA9685 に書いた値と記録の出力が 1000〜2000 µs の数(nan・inf でない)
  R2 前進は出どころより強くならない(モデル・後退の受け付け = ゲインのあとのモデル、発進の補助 = 80 µs かモデル、
     発進の前進 = launch_us かモデル、リカバリー = 130 µs、中止 = 0)
  R4 発進の前進・補助・中止は後退側(> 中立 + 40)を出さない(K5 = 長く見ていない明けの後退側は fuzz_closed のまま)
  R5 人側の行は補助なし・発進の前進なし
  R7 中立止めの行は中立(スロットル = 中立・補助なし)
装置・GPU・API には触れない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261026_fix/fix5/safety/fuzz_fix5.py [本数] [種]
出力: fix5/safety/out/fuzz_fix5_<種>.json
"""
import collections
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RS = HERE.parents[1] / 'fix4' / 'review_safety'
sys.path.insert(0, str(RS))
import rs_common as R          # noqa: E402
import fuzz_closed as FC       # noqa: E402
from rs_common import A, C, N, T0   # noqa: E402

R.TMP = HERE.parents[1] / 'fix5' / 'tmp_safety' / 'fz'
R.CL.TMP = R.TMP
R.SNAP_SHA['fix5_safety_before'] = {'drive.py': 'c575af1262836c3e10bfe32b2ee34293117a2144b3e568acb326ada57d59a471',
                                    'assist.py': 'b2bc818c5d8c8d410269f3aaf852d458fcc0c31d8f514002d8e34aa2c163fae6'}
OFF4 = dict(rec_back_guard=0, rec_hold_keep=0, launch_boost=0, launch_no_kick=0)


def extra_rules(sim, rec, pca, p):
    bad = []
    for t, s_, th in pca.w:
        if not (math.isfinite(s_) and math.isfinite(th) and 1000.0 <= s_ <= 2000.0 and 1000.0 <= th <= 2000.0):
            bad.append(('R1_pca', t, s_, th))
    for t, r in rec.rows:
        s = (t - T0) / 1e9
        o_s, o_t = float(r['out'][0]), float(r['out'][1])
        if not (math.isfinite(o_s) and math.isfinite(o_t) and 1000.0 <= o_s <= 2000.0 and 1000.0 <= o_t <= 2000.0):
            bad.append(('R1_row', round(s, 3), o_s, o_t))
        k = int(r['assist'])
        if r['hold']:
            if o_t != N or k != A.ASSIST_NONE or r['launch_on']:
                bad.append(('R7_hold_not_neutral', round(s, 3), o_t, k))
            continue
        if not r['select']:
            if k != A.ASSIST_NONE or r['launch_on']:
                bad.append(('R5_assist_on_human', round(s, 3), k))
            continue
        fwd = max(N - o_t, 0.0)
        m_fwd = max(N - float(r['model_out'][1]), 0.0)
        lim = {A.ASSIST_NONE: m_fwd, A.ASSIST_ESC: m_fwd, A.ASSIST_START: max(m_fwd, A.StartKick.KICK_US),
               A.ASSIST_LAUNCH: max(m_fwd, float(p['launch_us'])), A.ASSIST_RECOVERY: A.Recovery.ESCAPE_US,
               A.ASSIST_GIVEUP: 0.0}[k]
        if fwd > lim + 1e-6:
            bad.append(('R2_forward_over_source', round(s, 3), k, round(fwd, 1), round(lim, 1)))
        if k in (A.ASSIST_START, A.ASSIST_LAUNCH, A.ASSIST_GIVEUP) and o_t > N + 40.0:
            bad.append(('R4_reverse_side', round(s, 3), k, o_t))
    return bad


def same_run(a, b):
    (sa, ra, pa, _), (sb, rb, pb, _) = a, b
    if pa.w != pb.w:
        return False, 'pca'
    ok, why = R.same_rows(ra.rows, rb.rows)
    if not ok:
        return False, ('rows', why)
    for (t0, r0), (t1, r1) in zip(ra.rows, rb.rows):
        if set(r0) != set(r1):
            return False, ('keys', t0)
    ok, why = R.same_events(ra.events, rb.events)
    return (True, None) if ok else (False, ('events', why))


def main(n=300, seed=4402):
    rng = random.Random(seed)
    BEF = R.load_snapshot('fix5_safety_before')
    out = {'seed': seed, 'runs': 0, 'bad': collections.Counter(), 'bad_bef1': collections.Counter(), 'examples': [],
           'identity': collections.Counter(), 'giveup_runs': collections.Counter(), 'rows': 0, 'kinds': collections.Counter(),
           'holds_ai': 0, 'cmp': collections.Counter()}
    orig = R.CL.Sim
    R.CL.Sim = FC.FSim
    try:
        for i in range(n):
            sc = FC.rand_scene(rng)
            p1 = FC.rand_params(rng)
            p0 = dict(p1, rec_hold_keep=0)
            del FC.PH[:]
            r_now1 = R.run_closed(p1, sc, wrap=FC.wrap)
            ph1 = list(FC.PH)
            r_now0 = R.run_closed(p0, sc)
            r_bef1 = R.run_closed(p1, sc, mod=BEF[0], amod=BEF[1])
            r_bef0 = R.run_closed(p0, sc, mod=BEF[0], amod=BEF[1])
            poff = dict(p1, **OFF4)
            r_off_n = R.run_closed(poff, sc)
            r_off_b = R.run_closed(poff, sc, mod=BEF[0], amod=BEF[1])
            ok0, why0 = same_run(r_now0, r_bef0)
            okf, whyf = same_run(r_off_n, r_off_b)
            out['identity']['keep0_same' if ok0 else 'keep0_DIFF'] += 1
            out['identity']['off4_same' if okf else 'off4_DIFF'] += 1
            s1, rc1, pc1, _ = r_now1
            s0, rc0, pc0, _ = r_now0
            sb, rcb, pcb, _ = r_bef1
            m1, m0, mb = FC.metrics(s1, rc1), FC.metrics(s0, rc0), FC.metrics(sb, rcb)
            bad = []
            if not ok0:
                bad.append(('ID_keep0_differs_from_before', why0))
            if not okf:
                bad.append(('ID_off4_differs_from_before', whyf))
            # fuzz_closed の決まり(K1・K6 は keep 0 と比べる。K2〜K5・K9 は 1 本の中)
            cont = [t for t, a0, a1, hs in ph1 if hs and a0 in A.Recovery.HOLD_HIT_PHASES]
            for h in m1['rear']:
                if not any(h[0] - 2.0 <= t <= h[0] for t in cont):
                    continue
                twin = [x for x in m0['rear'] if abs(x[0] - h[0]) <= 0.3]
                if not twin or -h[2] > max(-x[2] for x in twin) + 0.05:
                    bad.append(('K1_rear_after_kept_hold_worse_than_keep0', h, twin))
            if m1['front_push_s'] > m0['front_push_s'] + 0.5:
                bad.append(('K6_front_push_longer_than_keep0', m1['front_push_s'], m0['front_push_s']))
            bad += FC.rules_one(s1, rc1, p1)
            for t, a0, a1, hs in ph1:
                if a1 == 'halt' and FC.v_at(s1, t) > 0.3:
                    bad.append(('K7_halt_while_moving_forward', t, round(FC.v_at(s1, t), 2)))
            bad += extra_rules(s1, rc1, pc1, p1)
            # 直す前の写し(keep 1)の同じ決まり(比べ: 指摘の数)
            for b in FC.rules_one(sb, rcb, p1):
                out['bad_bef1'][b[0]] += 1
            if mb['front_push_s'] > m0['front_push_s'] + 0.5:
                out['bad_bef1']['K6_front_push_longer_than_keep0'] += 1
            if m1['rear_max_v'] > m0['rear_max_v'] + 0.05:
                out['cmp']['rear_faster_than_keep0_any'] += 1
            if mb['rear_max_v'] > m0['rear_max_v'] + 0.05:
                out['cmp']['bef1_rear_faster_than_keep0_any'] += 1
            out['cmp']['front_push_s_now1'] += m1['front_push_s']
            out['cmp']['front_push_s_bef1'] += mb['front_push_s']
            out['cmp']['front_push_s_now0'] += m0['front_push_s']
            for nm, rr in (('now1', rc1), ('now0', rc0), ('bef1', rcb)):
                if any(int(r['assist']) == A.ASSIST_GIVEUP for _, r in rr.rows):
                    out['giveup_runs'][nm] += 1
            out['rows'] += len(rc1.rows)
            out['holds_ai'] += sum(1 for _, r in rc1.rows if r['hold'] and r['select'])
            for _, r in rc1.rows:
                if not r['hold']:
                    out['kinds'][A.ASSIST_NAMES[int(r['assist'])] or 'none'] += 1
            out['runs'] += 1
            for b in bad:
                out['bad'][b[0]] += 1
                if sum(1 for x in out['examples'] if x['kind'] == b[0]) < 6:
                    out['examples'].append({'kind': b[0], 'run': i, 'detail': b[1:],
                                            'params': {k: p1[k] for k in ('launch_boost', 'launch_no_kick', 'start_kick',
                                                                          'speed_ceiling', 'zone_caps', 'tof_guard', 'rev_s',
                                                                          'rev_us', 'throttle_gain')},
                                            'scene': {k: v for k, v in sc.items() if k != 'model'}})
            if (i + 1) % 25 == 0:
                print(i + 1, dict(out['bad']), dict(out['identity']), flush=True)
    finally:
        R.CL.Sim = orig
        R.cleanup()
    for k in ('bad', 'bad_bef1', 'identity', 'giveup_runs', 'kinds', 'cmp'):
        out[k] = {kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in dict(out[k]).items()}
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / ('fuzz_fix5_%d.json' % seed)).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != 'examples'}, ensure_ascii=False))
    for x in out['examples']:
        print(x['kind'], x['run'], x['detail'], x['params'])


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:3]))
