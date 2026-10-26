#!/usr/bin/env python3
"""最後の評価(D4・D5a・段 4 の門の欄)の数え直し(2026-10-26)。

eval_v2.py・desk_v2.py・desk_lib.py の関数は import しない。protocol_v2.json の文から定義を自分で書き、
保存した予測(eval/pred・desk/pred・desk/trt)と表(desk_inputs_20261022.npz・sets_v2.npz)だけから数える。
numpy だけ(torch・GPU を使わない)。ファイルは読むだけ。書くのは --out の JSON 1 つ。

定義(protocol_v2.json):
  H1 = bundle.primary(4,670)、介入 = bundle.interv(1,570)。L1 = |予測 − 正解| の 10 行 × 2 の平均 [µs]
  steer_bias・thr_bias = H1 の行 0 の 予測 − 人 の平均。steer_amp_ratio = H1 の予測ステアの std ÷ 人の std(10 行まとめて)
  車に出る = drive.apply_gains(ステア 1.0・trim 0・スロットルのゲインは前進側(raw < 中立)だけ・1000〜2000 に切る)
  brake: v = −eRPM × 0.5e-3 < 1.0 m/s のフレームで行 0 の生のスロットル > 1598 µs の割合。
         分母 = H1 の低速(FP32 best)+ 10/10 の AI 走行(TensorRT、part 0)の低速
  straight: sets.straight で、行 2 の車に出るスロットル(候補 ゲイン 0.90 − B ゲイン 0.90)の平均
  standstill: sets.stand で、行 0 の車に出るスロットル < 中立 − 30 µs の割合。門は ≥ B − 0.10(フレーム 300 以上)
  inference: TensorRT の 600 回の p99 ≤ 1.3 × 同じ日(20261026)の B の p99 の中央値
  sanity: TensorRT の出力が全部有限・500〜2500 µs・check_io(仕事の JSON)
  engine_ok: engine_check.json の accuracy_ok かつ actions_max_us ≤ 2.5、かつ仕事の JSON のエンジンが今のファイル
  amp 0.80〜1.25・l1_vs_b |l1_best − B| < 10・human_throttle thr_bias(best) ≤ 5 本の平均 + 4.082
  拒否権(n 本の腕): 介入 = 平均 − 5 本の平均 > thr_vs5(n).interv_best かつ全部の種 > 5 本の平均。
                     偏り = steer_bias の平均が 5 本の平均 ± thr_vs5(n).steer_bias_best の外
  D4: 2 本の平均 = (軽量型 種 k + 多段型 種 k)/ 2 の予測。d_k = L1(平均) − L1(軽量型)(best)。
      d の平均 ≤ −thr_pairs(k=2).l1_best かつ全部の d_k < 0、かつ門(推論を除く)を全部通る → 予備
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
REPO = Path('/home/jetson/jetformer')
RUNS = REPO / 'data' / 'runs'
BUNDLE = Path('/home/jetson/jetformer_reports/prep_20261022/desk/pred/desk_inputs_20261022.npz')
ZOO = REPO / 'data' / 'zoo' / 'policy_zoo_20261012' / 'pred'
RACC = REPO / 'data' / 'zoo' / 'real_accuracy_20261015' / 'pred'
B_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'
FIVE_FILES = {
    'A_s0': (ZOO / 'jetformer_v0_s0.npz', ZOO / 'jetformer_v0_s0_last.npz'),
    'A_s1': (ZOO / 'jetformer_v0_s1.npz', ZOO / 'jetformer_v0_s1_last.npz'),
    'v0-s2': (RACC / 'realacc_20261015-v0-s2.npz', RACC / 'realacc_20261015-v0-s2_last.npz'),
    'v0-s3': (RACC / 'realacc_20261015-v0-s3.npz', RACC / 'realacc_20261015-v0-s3_last.npz'),
    'B': (ZOO / 'B.npz', ZOO / 'B_last.npz'),
}
ARM = 'v2_20261025-v2h-aug2-e60'
MSF = 'v2_20261025-v2h-msf-aug2-e60'
DAY = '20261026'
GAIN = 0.90
LO, HI = 1000.0, 2000.0


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def car_thr(raw_thr, neutral, gain=GAIN):
    """drive.apply_gains のスロットルの式(配列)"""
    raw = np.asarray(raw_thr, np.float64)
    dt = raw - neutral
    out = neutral + np.where(dt < 0, gain * dt, dt)
    return np.clip(out, LO, HI)


def cfg_ref(run):
    c = json.load(open(RUNS / run / 'model_config.json'))
    return float(c['steer_center_us']), float(c['throttle_neutral_us'])


class Ctx(object):
    def __init__(self):
        self.proto = json.load(open(REP / 'protocol_v2.json'))
        self.bundle = npz(BUNDLE)
        self.key = str(self.bundle['key'])
        self.y = self.bundle['y'].astype(np.float64)
        self.prim = self.bundle['primary'].astype(bool)
        self.intv = self.bundle['interv'].astype(bool)
        self.v_h1 = -self.bundle['erpm'].astype(np.float64) * 0.5e-3
        self.sets = npz(REP / 'desk' / 'sets_v2.npz')
        self.sets_key = str(self.sets['key'])
        self.checks = []

    def check(self, what, ok, detail=''):
        self.checks.append({'what': what, 'ok': bool(ok), 'detail': detail})

    # -------------------------------------------------- 予測を読む
    def eval_pred(self, run, ck='best'):
        p = REP / 'eval' / 'pred' / ('%s%s.npz' % (run, '' if ck == 'best' else '_last'))
        z = npz(p)
        self.check('eval pred key・ckpt %s %s' % (run, ck), str(z['key']) == self.key and str(z['ckpt']) == ck
                   and z['pred'].shape == self.y.shape and np.isfinite(z['pred']).all(),
                   str(z['pred'].shape))
        return z['pred'].astype(np.float64)

    def five_pred(self, rid, ck='best'):
        p = FIVE_FILES[rid][0 if ck == 'best' else 1]
        z = npz(p)
        want = self.proto['reference_predictions_sha256'][rid][ck]
        self.check('5 本の予測の sha256 %s %s' % (rid, ck), sha(p) == want and str(z['key']) == self.key)
        return z['pred'].astype(np.float64)

    def tab_pred(self, run):
        z = npz(REP / 'desk' / 'pred' / ('%s_tab.npz' % run))
        self.check('tab pred key・best %s' % run, str(z['key']) == self.sets_key and str(z['ckpt']) == 'best'
                   and np.isfinite(z['pred']).all())
        return z['pred'].astype(np.float64)

    def trt_job_of(self, run):
        """run の今のエンジンを測った 20261026 の仕事(いちばん新しいもの)"""
        found = []
        for f in sorted((REP / 'desk' / 'trt').glob('*.json')):
            d = json.load(open(f))
            if run in d.get('engines', {}):
                found.append((d['job'], d))
        return found[-1] if found else (None, None)

    def trt_pred(self, run):
        job, d = self.trt_job_of(run)
        z = npz(REP / 'desk' / 'trt' / job / ('%s.npz' % run))
        e = d['engines'][run]
        eng = RUNS / run / 'jetformer_fp16.engine'
        st = os.stat(eng)
        cur = (st.st_size == e['engine']['size'] and st.st_mtime_ns == e['engine']['mtime_ns'])
        return job, d, z, cur


# ---------------------------------------------------------------- 指標

def h1_fields(C, pb, pl=None):
    m, y = C.prim, C.y
    d = np.abs(pb[m] - y[m])
    r = {'l1_best_us': float(d.mean()), 'l1_steer_best_us': float(d[:, :, 0].mean()),
         'l1_thr_best_us': float(d[:, :, 1].mean()),
         'l1_interv_us': float(np.abs(pb[C.intv] - y[C.intv]).mean()),
         'steer_bias_us': float((pb[m][:, 0, 0] - y[m][:, 0, 0]).mean()),
         'thr_bias_us': float((pb[m][:, 0, 1] - y[m][:, 0, 1]).mean()),
         'steer_amp_ratio': float(np.std(pb[m][:, :, 0]) / np.std(y[m][:, :, 0])),
         'n_h1': int(m.sum()), 'n_interv': int(C.intv.sum())}
    if pl is not None:
        r['l1_last_us'] = float(np.abs(pl[m] - y[m]).mean())
        r['l1_interv_last_us'] = float(np.abs(pl[C.intv] - y[C.intv]).mean())
    return r


def brake(C, pb_eval, trt_pred, trt_erpm, trt_part):
    """(heldout の数, 分母), (10/10 TensorRT の数, 分母), 割合"""
    vh = C.v_h1
    okh = C.prim & np.isfinite(vh) & (vh < 1.0)
    kh = int((okh & (pb_eval[:, 0, 1] > 1598.0)).sum())
    va = -np.asarray(trt_erpm, np.float64) * 0.5e-3
    oka = (np.asarray(trt_part) == 0) & np.isfinite(va) & (va < 1.0)
    ka = int((oka & (np.asarray(trt_pred, np.float64)[:, 0, 1] > 1598.0)).sum())
    n = int(okh.sum()) + int(oka.sum())
    return [kh, int(okh.sum())], [ka, int(oka.sum())], (kh + ka) / n


def straight(C, pt, ref_c, pt_b, ref_b):
    m = C.sets['straight'].astype(bool)
    a = car_thr(pt[m][:, 2, 1], ref_c)
    b = car_thr(pt_b[m][:, 2, 1], ref_b)
    return float(np.mean(a - b)), int(m.sum())


def standstill(C, pt, ref):
    m = C.sets['stand'].astype(bool)
    a = car_thr(pt[m][:, 0, 1], ref)
    return float(np.mean(a < ref - 30.0)), int(m.sum())


def sanity_trt(z, e):
    p = np.asarray(z['pred'], np.float64)
    fin = np.isfinite(p).all()
    rng = bool(((p >= 500.0) & (p <= 2500.0)).all())
    return bool(fin and rng and e.get('check_io_ok') is True), {'finite': bool(fin), 'in_500_2500': rng,
                                                                  'check_io_ok': e.get('check_io_ok')}


def b_p99_same_day(C):
    vals = []
    for f in sorted((REP / 'desk' / 'trt').glob('*.json')):
        d = json.load(open(f))
        if d.get('day') != DAY or B_RUN not in d.get('engines', {}):
            continue
        z = npz(REP / 'desk' / 'trt' / d['job'] / ('%s.npz' % B_RUN))
        t = np.asarray(z['ms_timed'], np.float64)
        vals.append((d['job'], float(np.percentile(t, 99)) if len(t) else None,
                     d['engines'][B_RUN].get('run_numpy_p99_ms')))
    mine = [v[1] for v in vals if v[1] is not None]
    return float(np.median(mine)), vals


def engine_ok(run, cur):
    e = json.load(open(RUNS / run / 'engine_check.json'))
    return bool(cur and e.get('accuracy_ok') is True and float(e.get('actions_max_us')) <= 2.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    C = Ctx()
    P = C.proto
    T = P['thresholds_us']['values']
    C.check('H1 4,670・介入 1,570', int(C.prim.sum()) == 4670 and int(C.intv.sum()) == 1570,
            '%d・%d' % (C.prim.sum(), C.intv.sum()))
    C.check('sets の直線 2,043・止まった場面 1,115', int(C.sets['straight'].sum()) == 2043 and
            int(C.sets['stand'].sum()) == 1115)

    # ---- 5 本
    five = {}
    for rid in FIVE_FILES:
        five[rid] = h1_fields(C, C.five_pred(rid, 'best'), C.five_pred(rid, 'last'))
    fm = {k: float(np.mean([five[r][k] for r in five])) for k in
          ('l1_best_us', 'l1_last_us', 'l1_interv_us', 'steer_bias_us', 'thr_bias_us')}
    B = five['B']

    # ---- B の C3 の欄
    refB = cfg_ref(B_RUN)
    ptB = C.tab_pred(B_RUN)
    b_stand, n_stand = standstill(C, ptB, refB[1])
    b_p99, b_jobs = b_p99_same_day(C)
    infer_lim = 1.3 * b_p99
    human_lim = fm['thr_bias_us'] + 4.082   # gates_v2.human_throttle の文の 4.082

    def gates_of(f, brk, strt, stand, san, eng, p99):
        g = {'engine_ok': eng, 'sanity_ok': san,
             'inference': None if p99 is None else bool(p99 <= infer_lim),
             'amp': bool(0.80 <= f['steer_amp_ratio'] <= 1.25),
             'l1_vs_b': bool(abs(f['l1_best_us'] - B['l1_best_us']) < 10.0),
             'brake': bool(brk <= 0.01),
             'standstill': bool(stand >= b_stand - 0.10) if n_stand >= 300 else None,
             'human_throttle': bool(f['thr_bias_us'] <= human_lim),
             'straight': bool(abs(strt) <= 10.0)}
        return g

    rows = {}
    trt_cache = {}
    for run in ['%s-s%d' % (ARM, k) for k in range(4)] + ['%s-s%d' % (MSF, k) for k in range(2)]:
        pb, pl = C.eval_pred(run, 'best'), C.eval_pred(run, 'last')
        f = h1_fields(C, pb, pl)
        ref = cfg_ref(run)
        pt = C.tab_pred(run)
        job, d, z, cur = C.trt_pred(run)
        trt_cache[run] = (pb, pt, z, ref, job, d, cur)
        bh, ba, bfrac = brake(C, pb, z['pred'], z['erpm'], z['part'])
        s, ns = straight(C, pt, ref[1], ptB, refB[1])
        st, _ = standstill(C, pt, ref[1])
        san, sd = sanity_trt(z, d['engines'][run])
        eng = engine_ok(run, cur)
        p99 = float(np.percentile(np.asarray(z['ms_timed'], np.float64), 99))
        g = gates_of(f, bfrac, s, st, san, eng, p99)
        rows[run] = dict(f, ref=ref, trt_job=job, engine_current=cur, brake_heldout=bh, brake_ai_1010=ba,
                         brake_lowspeed_frac=bfrac, straight_thr_vs_b_us=s, straight_frames=ns,
                         standstill_go_frac=st, sanity=sd, trt_p99_ms=p99, gates=g,
                         gates_pass=all(v is True for v in g.values()),
                         failed=[k for k, v in g.items() if v is False])

    # ---- D5a: 拒否権(n = 4)と種の順
    n = 4
    arm_rows = [rows['%s-s%d' % (ARM, k)] for k in range(n)]
    iv = [r['l1_interv_us'] for r in arm_rows]
    sb = [r['steer_bias_us'] for r in arm_rows]
    v_int_diff = float(np.mean(iv)) - fm['l1_interv_us']
    veto_int = bool(v_int_diff > T['vs5']['n4']['interv_best'] and all(x > fm['l1_interv_us'] for x in iv))
    sb_mean = float(np.mean(sb))
    veto_bias = bool(abs(sb_mean - fm['steer_bias_us']) > T['vs5']['n4']['steer_bias_best'])
    cands = [] if (veto_int or veto_bias) else [k for k in range(n) if arm_rows[k]['gates_pass']]
    d5a = {'final_arm': 'v2h-aug2-e60', 'veto': {'interv': veto_int, 'interv_diff': v_int_diff,
                                                  'interv_thr': T['vs5']['n4']['interv_best'],
                                                  'bias': veto_bias, 'steer_bias_mean': sb_mean,
                                                  'bias_value_minus_five': sb_mean - fm['steer_bias_us'],
                                                  'bias_thr': T['vs5']['n4']['steer_bias_best']},
           'C1': ('%s-s%d' % (ARM, cands[0])) if len(cands) > 0 else None,
           'C2': ('%s-s%d' % (ARM, cands[1])) if len(cands) > 1 else None,
           'seeds': {'s%d' % k: {'pass': arm_rows[k]['gates_pass'], 'failed': arm_rows[k]['failed']}
                     for k in range(n)}}

    # ---- D4: 2 本の平均
    d4 = {'pairs': {}, 'thr': T['pairs']['k2']['l1_best']}
    ds = []
    gate_ok = []
    for k in range(2):
        la, ms = '%s-s%d' % (ARM, k), '%s-s%d' % (MSF, k)
        pa, pta, za, refa, _, da, cura = trt_cache[la]
        pm, ptm, zm, refm, _, dm, curm = trt_cache[ms]
        C.check('D4 種 %d の 2 本の中心・中立が同じ' % k, refa == refm, str((refa, refm)))
        C.check('D4 種 %d の TensorRT のフレームの並びが同じ' % k,
                np.array_equal(za['erpm'], zm['erpm']) and np.array_equal(za['part'], zm['part']))
        pe = (pa + pm) / 2.0
        f = h1_fields(C, pe)
        d = f['l1_best_us'] - rows[la]['l1_best_us']
        ds.append(d)
        pt_e = (pta + ptm) / 2.0
        trt_e = (np.asarray(za['pred'], np.float64) + np.asarray(zm['pred'], np.float64)) / 2.0
        bh, ba, bfrac = brake(C, pe, trt_e, za['erpm'], za['part'])
        s, _ = straight(C, pt_e, refa[1], ptB, refB[1])
        st, _ = standstill(C, pt_e, refa[1])
        san = bool(rows[la]['gates']['sanity_ok'] and rows[ms]['gates']['sanity_ok']
                   and np.isfinite(trt_e).all() and ((trt_e >= 500) & (trt_e <= 2500)).all())
        eng = bool(rows[la]['gates']['engine_ok'] and rows[ms]['gates']['engine_ok'])
        g = gates_of(f, bfrac, s, st, san, eng, None)
        g.pop('inference')
        gp = all(v is True for v in g.values())
        gate_ok.append(gp)
        d4['pairs']['s%d' % k] = dict(f, light=la, msf=ms, light_l1_best_us=rows[la]['l1_best_us'],
                                      msf_l1_best_us=rows[ms]['l1_best_us'], d=d, brake_heldout=bh,
                                      brake_ai_1010=ba, brake_lowspeed_frac=bfrac, straight_thr_vs_b_us=s,
                                      standstill_go_frac=st, gates_except_inference=g,
                                      gates_pass_except_inference=gp,
                                      failed=[kk for kk, v in g.items() if v is False])
    d4['d_best'] = ds
    d4['mean_d_best'] = float(np.mean(ds))
    d4['all_d_negative'] = bool(all(x < 0 for x in ds))
    d4['l1_good'] = bool(d4['mean_d_best'] <= -d4['thr'] and d4['all_d_negative'])
    d4['reserve'] = bool(d4['l1_good'] and all(gate_ok))

    out = {'schema': 'jf-verify-final/v1', 'script': os.path.abspath(__file__), 'script_sha256': sha(__file__),
           'protocol_v2_sha256': sha(REP / 'protocol_v2.json'),
           'lock2_sha256': sha(REP / 'protocol_v2_lock2.json'),
           'inputs_sha256': {'bundle': sha(BUNDLE), 'sets_v2.npz': sha(REP / 'desk' / 'sets_v2.npz')},
           'five': five, 'five_mean': fm, 'B_ref': refB, 'B_standstill_go_frac': b_stand,
           'standstill_frames': n_stand, 'B_p99_same_day_median_ms': b_p99, 'B_p99_jobs': b_jobs,
           'inference_limit_ms': infer_lim, 'human_throttle_limit_us': human_lim,
           'rows': rows, 'D5a': d5a, 'D4': d4, 'checks': C.checks}
    with open(a.out, 'w') as fo:
        json.dump(out, fo, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    bad = [c for c in C.checks if not c['ok']]
    print('checks', len(C.checks), 'bad', len(bad), bad)
    print('D4', json.dumps({k: d4[k] for k in ('d_best', 'mean_d_best', 'l1_good', 'reserve')}))
    print('D5a', json.dumps({k: d5a[k] for k in ('C1', 'C2', 'veto')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
