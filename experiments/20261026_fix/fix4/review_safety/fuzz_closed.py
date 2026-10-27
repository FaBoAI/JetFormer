"""反証 4 回目・安全: でたらめな「閉じたループ」(本物の drive._loop・偽物の ESC と車 = fix3/refute2_safety/closed_loop.py)で、
3 回目で足したスイッチの組み合わせを、同じ場面の 3 通りで比べる:
  now1  = 今の drive.py・assist.py、rec_hold_keep = 1・rec_back_guard = 1(と launch_no_kick はでたらめ)
  now0  = 今、rec_hold_keep = 0(rec_back_guard・ほかは now1 と同じ)
  bef1  = 直す 3 回目の前(tests/fixtures/fix3_r3_1_before)、keep = 1・guard = 1
場面: 前の壁・後ろの物・当たりに続くカメラの止まり・でたらめなカメラの止まり・車両の値の止まり・人側の区切り・
      走っている間の「止まらない衝撃」(本物の記録で走行中に普通に出る 2.5G 以上。scan_g_moving.py)・モデルの前進とハンドル。
決まり(破ったら記録):
  K1 後ろの物に当たる速さ: now1 が now0 より 0.05 m/s 以上速い・now0 で当たらないのに now1 で当たる
  K2 中止(ASSIST_GIVEUP)の最初の行で、車が前へ 1.0 m/s 以上で走っている(中立・ハンドルまっすぐのまま惰性)
  K3 リカバリーの種類の行の前進が 130 µs より強い・中止の行が中立でない
  K4 中止のあと、同じ AI の区間(人側まで)で中立でない出力(rec_hold_keep = 1)
  K5 0.5 秒より長く見ていない明けから 0.3 秒の間に、後ろへ動いている車へ後退側(> 中立 + 40)を出す(リカバリー)
  K6 前の壁を押した秒(前に接して前進を出す): now1 が now0 より 0.5 秒以上長い
装置・GPU・API には触れない。出力 out/fuzz_closed_<種>.json

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        experiments/20261026_fix/fix4/review_safety/fuzz_closed.py [本数] [種]
"""
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rs_common as R          # noqa: E402
from rs_common import A, C, N, PARAMS, T0   # noqa: E402

CL = R.CL
PH = []      # 今の Recovery.step の (t, 段の前, 段のあと, hold_since が渡った)


def wrap(orig):
    def step(self, model_us, erpm, now, impact_g=0.0, *a, **kw):
        p0 = self.phase
        r = orig(self, model_us, erpm, now, impact_g, *a, **kw)
        PH.append((round(now - 100.0, 3), p0, self.phase, kw.get('hold_since') is not None))
        return r
    return step


class FSim(CL.Sim):
    """Sim + 走っている間の止まらない衝撃(scene['spikes'] = [(t, g)])"""

    def _advance(self, t_end):
        sp = self.s.get('spikes') or []
        while self.t + CL.DT <= t_end + 1e-9:
            nt = round(self.t + CL.DT, 6)
            for ts, g in sp:
                if abs(nt - ts) < CL.DT / 2 and not self.car.spike:
                    self.car.spike = [g, g]
            super()._advance(nt)
        super()._advance(t_end)


def rand_model(rng, dur):
    cuts, t = [], 0.0
    while t < dur:
        L = rng.uniform(0.3, 2.0)
        cuts.append((t, t + L, C + rng.choice([0.0, 0.0, 120.0, -150.0, 250.0]),
                     N - rng.choice([60.0, 100.0, 120.0, 150.0, 200.0, 250.0])))
        t += L

    def f(s, car):
        for a, b, st, th in cuts:
            if a <= s < b:
                return st, th
        return cuts[-1][2], cuts[-1][3]
    return f


def rand_scene(rng):
    dur = rng.choice([8.0, 10.0, 12.0])
    xf = rng.uniform(1.6, 3.2) if rng.random() < 0.85 else rng.uniform(6.0, 40.0)
    sc = {'dur': dur, 'xf': xf, 'model': rand_model(rng, dur), 'drops': [], 'veh_skip': [], 'spikes': []}
    if rng.random() < 0.7:
        sc['xr'] = xf - rng.uniform(0.5, 2.2)
    if rng.random() < 0.6:
        sc['hit_drop'] = (rng.choice([0.0, 0.0, 0.02, 0.05, 0.1]), rng.choice([0.12, 0.3, 0.45, 0.55, 0.6, 1.0, 2.0]))
    for _ in range(rng.randint(0, 5)):
        a = rng.uniform(1.0, dur - 0.5)
        sc['drops'].append((a, a + rng.choice([0.12, 0.2, 0.35, 0.45, 0.55, 0.7, 1.0, 1.5])))
    for _ in range(rng.randint(0, 3)):
        a = rng.uniform(1.0, dur - 0.5)
        sc['veh_skip'].append((a, a + rng.uniform(0.05, 0.5)))
    for _ in range(rng.randint(0, 6)):
        sc['spikes'].append((round(rng.uniform(1.2, dur - 0.2), 2), rng.choice([2.6, 3.0, 4.0, 6.0])))
    if rng.random() < 0.35:
        a = rng.uniform(3.0, dur - 1.0)
        sc['human'] = [(a, a + rng.uniform(0.1, 0.8))]
    if rng.random() < 0.3:
        sc['front_soft'] = rng.choice([0.3, 0.5])
    return sc


def rand_params(rng):
    p = dict(PARAMS, recovery=1, esc_reverse=1, start_kick=rng.choice([0, 1]), launch_boost=rng.choice([0, 1]),
             launch_no_kick=rng.choice([0, 1]), speed_ceiling=rng.choice([0, 1]), zone_caps=rng.choice([0, 1]),
             tof_guard=rng.choice([1, 1, 2, 3]), throttle_gain=rng.choice([0.8, 1.0, 1.2]),
             rev_s=rng.choice([0.2, 0.3, 0.4, 0.6]), rev_us=rng.choice([200.0, 300.0, 400.0]),
             rec_back_guard=1, rec_hold_keep=1)
    p['speed_auto'] = rng.choice([0, 1]) if p['speed_ceiling'] else 0
    return p


def metrics(sim, rec):
    log = sim.log
    rear = [h for h in sim.car.hits if h[1] == 'rear']
    front_push = 0.0
    for (t0, x, v, thr, con, rev_on, sl), (t1, *_r) in zip(log, log[1:]):
        if con == 'front' and thr < N - 20 and sl:
            front_push += t1 - t0
    return {'rear': rear, 'rear_max_v': max([-h[2] for h in rear], default=0.0), 'front_push_s': round(front_push, 2)}


def v_at(sim, s):
    for t, x, v, *_ in sim.log:
        if t >= s - 1e-9:
            return v
    return sim.log[-1][2]


def rules_one(sim, rec, p):
    """K2〜K5(1 本の中で見る)"""
    bad = []
    rows = [((t - T0) / 1e9, r) for t, r in rec.rows]
    in_gu = False
    last_proc = None
    for s, r in rows:
        if r['hold']:
            if not r['select']:
                in_gu = False
            continue
        if not r['select']:
            in_gu = False
            last_proc = None
            continue
        k = int(r['assist'])
        o_s, o_t = float(r['out'][0]), float(r['out'][1])
        if k == A.ASSIST_GIVEUP:
            if not in_gu:
                t_gu = s
                v = v_at(sim, s)
                if v >= 1.0:
                    bad.append(('K2_giveup_while_moving', round(s, 3), round(v, 2)))
            in_gu = True
            if o_t != N:
                bad.append(('K3_giveup_not_neutral', round(s, 3), o_t))
        elif in_gu and int(p['rec_hold_keep']) == 1:
            # 中止のあとに人側(CH3)を見たか(カメラの止まりの間に人側にしたときは記録の行が無い)
            if any(not sl for t, x, v, thr, con, rv, sl in sim.log if t_gu <= t <= s):
                in_gu = False
            elif not (o_t == N):
                bad.append(('K4_output_after_giveup', round(s, 3), k, round(o_t, 1)))
        if k == A.ASSIST_RECOVERY and N - o_t > 130.0 + 1e-6:
            bad.append(('K3_recovery_forward_over', round(s, 3), round(o_t, 1)))
        if k == A.ASSIST_RECOVERY and o_t > N + 40 and v_at(sim, s) > 1.0:
            bad.append(('K9_recovery_reverse_side_while_fast_forward', round(s, 3), round(v_at(sim, s), 2)))
        if last_proc is not None and s - last_proc > 0.5:
            unseen_end = s
        else:
            unseen_end = None
        if unseen_end is not None:
            # 明けた最初のフレームから 0.3 秒(下で見る)
            pass
        last_proc = s
    # K5: 長く見ていない明けから 0.3 秒
    procs = [s for s, r in rows if not r['hold'] and r['select']]
    ends = [b for a, b in zip(procs, procs[1:]) if b - a > 0.5]
    for e in ends:
        for s, r in rows:
            if e <= s < e + 0.3 and not r['hold'] and r['select'] and int(r['assist']) == A.ASSIST_RECOVERY \
                    and float(r['out'][1]) > N + 40 and v_at(sim, s) < -0.15:
                bad.append(('K5_reverse_after_long_unseen', round(e, 3), round(s, 3), round(v_at(sim, s), 2)))
                break
    return bad


def main(n=200, seed=4402):
    rng = random.Random(seed)
    BEF = R.load_snapshot('fix3_r3_1_before')
    out = {'seed': seed, 'runs': 0, 'bad': collections.Counter(), 'examples': [], 'cmp': collections.Counter(),
           'giveup_runs': collections.Counter(), 'halt_rows': 0}
    orig = CL.Sim
    CL.Sim = FSim
    try:
        for i in range(n):
            sc = rand_scene(rng)
            p1 = rand_params(rng)
            p0 = dict(p1, rec_hold_keep=0)
            del PH[:]
            s1, r1, pc1, _ = R.run_closed(p1, sc, wrap=wrap)
            ph1 = list(PH)
            s0, r0, pc0, _ = R.run_closed(p0, sc)
            sb, rb, pcb, _ = R.run_closed(p1, sc, mod=BEF[0], amod=BEF[1])
            m1, m0, mb = metrics(s1, r1), metrics(s0, r0), metrics(sb, rb)
            bad = []
            # K1: 中立止め(AI 側)のあと 1.5 秒以内の後ろの当たりで、keep 0 に同じ頃(±0.3 秒)の当たりが無い・0.05 m/s より速い
            # (keep 0 は中立止めで armed が消えて 2 回目のリカバリーをしないことがある。そのリカバリー自身の後退の当たりは
            #  rev_s・rev_us の話でこの直しとは別なので、中立止めのあとの当たりだけを見る)
            # 残した中立止めのあと、途中の段(後退側・下がる惰性)を続けた所(hold_since が渡り、段の前が後退側)から 2 秒以内の後ろの当たり
            cont = [t for t, p0, p1, hs in ph1 if hs and p0 in A.Recovery.HOLD_HIT_PHASES]
            for h in m1['rear']:
                if not any(h[0] - 2.0 <= t <= h[0] for t in cont):
                    continue
                twin = [x for x in m0['rear'] if abs(x[0] - h[0]) <= 0.3]
                if not twin or -h[2] > max(-x[2] for x in twin) + 0.05:
                    bad.append(('K1_rear_after_kept_hold_worse_than_keep0', h, twin, [t for t in cont if h[0] - 2.0 <= t <= h[0]]))
            if m1['rear_max_v'] > m0['rear_max_v'] + 0.05:
                out['cmp']['rear_faster_than_keep0_any'] += 1
            if m1['rear_max_v'] > mb['rear_max_v'] + 0.05:
                out['cmp']['rear_faster_than_before'] += 1
            if m1['front_push_s'] > m0['front_push_s'] + 0.5:
                bad.append(('K6_front_push_longer_than_keep0', m1['front_push_s'], m0['front_push_s']))
            bad += rules_one(s1, r1, p1)
            for t, p0, p1_, hs in ph1:          # K7: 'halt'(後ろへの惰性を止める前進)を、前へ動いている車に出した
                if p1_ == 'halt' and v_at(s1, t) > 0.3:
                    bad.append(('K7_halt_while_moving_forward', t, round(v_at(s1, t), 2)))
            out['halt_rows'] += sum(1 for _, r in r1.rows if not r['hold'] and int(r['assist']) == A.ASSIST_RECOVERY
                                    and float(r['out'][1]) == N - 130.0 and float(r['out'][0]) == C)
            for nm, rr in (('now1', r1), ('now0', r0), ('bef1', rb)):
                if any(int(r['assist']) == A.ASSIST_GIVEUP for _, r in rr.rows):
                    out['giveup_runs'][nm] += 1
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
                print(i + 1, dict(out['bad']), dict(out['giveup_runs']), flush=True)
    finally:
        CL.Sim = orig
        R.cleanup()
    for k in ('bad', 'cmp', 'giveup_runs'):
        out[k] = dict(out[k])
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / ('fuzz_closed_%d.json' % seed)).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != 'examples'}, ensure_ascii=False))
    for x in out['examples']:
        print(x['kind'], x['run'], x['detail'], x['params'])


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:3]))
