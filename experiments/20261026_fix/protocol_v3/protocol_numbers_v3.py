#!/usr/bin/env python3
"""protocol_v3 の数(2026-10-26。numpy だけ。学習の前に、保存した予測だけから出す)。

出すもの(protocol_numbers_v3.json):
  1. 介入のステアの L1(best・last)の 5 本の平均・標準偏差と閾値(2 × 標準誤差。v2 と同じ式)
       腕 n 本 対 5 本: 2·sd·√(1/n + 1/5)。同じ種 k 組: 2·sd·√(2/k)
  2. 机上の新しい欄の 5 本の値(B と同じ作り方の 5 本。B 自身は 0)
       straight_thr_vs_b_us_1010・_1023(直線を回で分ける。行 2・車に出る値・ゲイン 0.90)
       standstill_back_frac(止まった場面で、行 0 の車に出るスロットル > そのモデルの中立 + 30 µs の割合)
     と、v2 の 22 本の同じ欄(新しい門が v2 の落ち方をとらえるかの自己試験)
  3. 学習中の物差し(probe)の 4 つの欄を、C3 の保存した予測から同じ行で出し直す(probe_check.json と照らす)
  4. v2 の 22 本の介入のステア・スロットルの L1(新しい拒否権の自己試験)
読むだけ: desk_inputs_20261022.npz・5 本の予測・desk/pred/*_tab.npz・desk/sets_v2.npz・diag_data/out/probe_set.npz・probe_check.json。
流し方: /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I protocol_numbers_v3.py
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path('/home/jetson/jetformer_reports/fix_20261026/protocol_numbers_v3.json')
REPO = Path('/home/jetson/jetformer')
DATA = REPO / 'data'
RUNS = DATA / 'runs'
REP2 = Path('/home/jetson/jetformer_reports/train_v2_20261025')
PROTO2 = REP2 / 'protocol_v2.json'
DESK = REP2 / 'desk'
BUNDLE = Path('/home/jetson/jetformer_reports/prep_20261022/desk/pred/desk_inputs_20261022.npz')
TABLE_KEY = '3d339512828b17195d568f1b93ee4a28208f2e85'
ZOO = DATA / 'zoo' / 'policy_zoo_20261012' / 'pred'
RACC = DATA / 'zoo' / 'real_accuracy_20261015' / 'pred'
PROBE_SET = REPO / 'experiments/20261026_fix/diag_data/out/probe_set.npz'
PROBE_CHECK = REPO / 'experiments/20261026_fix/diag_data/out/probe_check.json'
B_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'
FIVE = [
    ('A_s0', 'clean_driving_20261011-jetformer_v0-s0-nocmd-stride3-e30', ZOO / 'jetformer_v0_s0.npz', ZOO / 'jetformer_v0_s0_last.npz'),
    ('A_s1', 'clean_driving_20261011-jetformer_v0-s1-nocmd-stride3-e30', ZOO / 'jetformer_v0_s1.npz', ZOO / 'jetformer_v0_s1_last.npz'),
    ('v0-s2', 'realacc_20261015-v0-s2', RACC / 'realacc_20261015-v0-s2.npz', RACC / 'realacc_20261015-v0-s2_last.npz'),
    ('v0-s3', 'realacc_20261015-v0-s3', RACC / 'realacc_20261015-v0-s3.npz', RACC / 'realacc_20261015-v0-s3_last.npz'),
    ('B', B_RUN, ZOO / 'B.npz', ZOO / 'B_last.npz'),
]
GAIN = 0.90
ROW0, ROW2 = 0, 2
DRIVE_LO, DRIVE_HI = 1000.0, 2000.0
MARGIN = 30.0
PROBE_NEUTRAL = 1498.0


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def car_out(raw, ref, g):
    """drive.apply_gains と同じ(desk_v2.car_out の写し。スロットルだけ)"""
    raw = np.asarray(raw, np.float64)
    dt = raw[..., 1] - ref[1]
    t = ref[1] + np.where(dt < 0, g * dt, dt)
    return np.clip(t, DRIVE_LO, DRIVE_HI)


def ref_of(run):
    c = json.loads((RUNS / run / 'model_config.json').read_text())
    return (float(c['steer_center_us']), float(c['throttle_neutral_us']))


def thr(n, k):
    return {'vs5_n%d' % n: None}


def se_table(sd):
    out = {}
    for n in (1, 2, 3, 4):
        out['vs5_n%d' % n] = round(2.0 * sd * math.sqrt(1.0 / n + 1.0 / 5.0), 3)
    for k in (2, 3, 4):
        out['pairs_k%d' % k] = round(2.0 * sd * math.sqrt(2.0 / k), 3)
    return out


def part1(proto):
    ref = proto['reference_predictions_sha256']
    if sha256(BUNDLE) != ref['bundle']:
        raise SystemExit('bundle の sha256 が protocol_v2 と違う')
    with np.load(BUNDLE, allow_pickle=False) as z:
        if str(z['key']) != TABLE_KEY:
            raise SystemExit('bundle の key が違う')
        y = np.nan_to_num(np.asarray(z['y'], np.float64))
        prim, iv = z['primary'].astype(bool), z['interv'].astype(bool)
    vals = {}
    for rid, run, pb, pl in FIVE:
        row = {}
        for ck, p in (('best', pb), ('last', pl)):
            if sha256(p) != ref[rid][ck]:
                raise SystemExit('%s %s の sha256 が protocol_v2 と違う' % (rid, ck))
            with np.load(p, allow_pickle=False) as z:
                pr = np.asarray(z['pred'], np.float64)
            d = np.abs(pr - y)
            row['interv_steer_%s' % ck] = float(d[iv][:, :, 0].mean())
            row['interv_thr_%s' % ck] = float(d[iv][:, :, 1].mean())
            row['interv_%s' % ck] = float(d[iv].mean())
            row['h1_%s' % ck] = float(d[prim].mean())
        vals[rid] = row
    out = {'five': vals, 'stats': {}}
    for key in ('interv_steer_best', 'interv_steer_last', 'interv_thr_best', 'interv_best', 'h1_best', 'h1_last'):
        x = np.array([vals[r][key] for r, *_ in FIVE])
        sd = float(np.std(x, ddof=1))
        out['stats'][key] = {'mean': round(float(x.mean()), 4), 'sd': round(sd, 4), 'thresholds': se_table(sd)}
    return out


def desk_fields(pc, refc, pb, refb, S):
    st, sess, stand = S['straight'], S['sess'], S['stand']
    a = car_out(pc[:, ROW2], refc, GAIN)
    b = car_out(pb[:, ROW2], refb, GAIN)
    r = {}
    for name, m in (('straight_thr_vs_b_us', st), ('straight_thr_vs_b_us_1010', st & (sess == 0)),
                    ('straight_thr_vs_b_us_1023', st & (sess == 1))):
        r[name] = float(np.mean(a[m] - b[m])) if m.any() else None
    t0 = car_out(pc[stand][:, ROW0], refc, GAIN)
    r['standstill_back_frac'] = float(np.mean(t0 > refc[1] + MARGIN))
    r['standstill_go_frac'] = float(np.mean(t0 < refc[1] - MARGIN))
    r['n'] = {'straight': int(st.sum()), 'straight_1010': int((st & (sess == 0)).sum()),
              'straight_1023': int((st & (sess == 1)).sum()), 'stand': int(stand.sum())}
    return r


def load_tab(run):
    p = DESK / 'pred' / ('%s_tab.npz' % run)
    with np.load(p, allow_pickle=False) as z:
        if 'ckpt' in z.files and str(z['ckpt']) != 'best':
            raise SystemExit('%s は best でない' % p)
        return np.asarray(z['pred'], np.float64), sha256(p)


def part2(proto):
    with np.load(DESK / 'sets_v2.npz', allow_pickle=False) as z:
        S = {k: z[k] for k in ('straight', 'sess', 'stand', 'scene', 'frame')}
    S['straight'] = S['straight'].astype(bool)
    S['stand'] = S['stand'].astype(bool)
    pb, _ = load_tab(B_RUN)
    refb = ref_of(B_RUN)
    five = {}
    for rid, run, *_ in FIVE:
        pc, h = load_tab(run)
        five[rid] = dict(desk_fields(pc, ref_of(run), pb, refb, S), tab_sha256=h)
    stats = {}
    nonb = [r for r, *_ in FIVE if r != 'B']
    for key in ('straight_thr_vs_b_us', 'straight_thr_vs_b_us_1010', 'straight_thr_vs_b_us_1023',
                'standstill_back_frac', 'standstill_go_frac'):
        x5 = np.array([five[r][key] for r, *_ in FIVE])
        x4 = np.array([five[r][key] for r in nonb])
        stats[key] = {'mean5': round(float(x5.mean()), 4), 'sd5': round(float(np.std(x5, ddof=1)), 4),
                      'mean4_nonB': round(float(x4.mean()), 4), 'sd4_nonB': round(float(np.std(x4, ddof=1)), 4),
                      'min5': round(float(x5.min()), 4), 'max5': round(float(x5.max()), 4),
                      'n1_vs5_2se': round(2.0 * float(np.std(x5, ddof=1)) * math.sqrt(1.0 + 1.0 / 5.0), 4)}
    others = {}
    for p in sorted((DESK / 'pred').glob('*_tab.npz')):
        run = p.name[:-len('_tab.npz')]
        if run in [r for _, r, *_ in FIVE] or run.startswith('bench_'):
            continue
        pc, h = load_tab(run)
        others[run] = dict(desk_fields(pc, ref_of(run), pb, refb, S), tab_sha256=h)
    return {'five': five, 'stats': stats, 'v2_and_bfast': others, 'n_rows': int(len(S['sess']))}


def part3():
    """probe の st10・st23・stand を、C3 の保存した予測(FP32・ジャンル 5・best)から同じ行で出し直す"""
    with np.load(DESK / 'sets_v2.npz', allow_pickle=False) as z:
        sc, fr = z['scene'].astype(str), z['frame'].astype(np.int64)
    key = {(s, int(f)): i for i, (s, f) in enumerate(zip(sc, fr))}
    P = np.load(PROBE_SET)
    rows = {}
    for k in ('st10', 'st23', 'stand'):
        idx = [key.get((str(s), int(f)), -1) for s, f in zip(P[k + '_scene'], P[k + '_frame'])]
        if min(idx) < 0:
            raise SystemExit('probe の行が sets_v2 に無い: %s' % k)
        rows[k] = np.array(idx)
    pb, _ = load_tab(B_RUN)
    chk = json.loads(PROBE_CHECK.read_text())
    out = {}
    runs = [r for _, r, *_ in FIVE] + sorted(p.name[:-8] for p in (DESK / 'pred').glob('*_tab.npz')
                                             if not p.name.startswith('bench_'))
    for run in dict.fromkeys(runs):
        pc, _ = load_tab(run)
        r = {}
        for k in ('st10', 'st23'):
            r['%s_thr_vs_b_us' % k] = round(float(np.mean(pc[rows[k]][:, 2, 1] - pb[rows[k]][:, 2, 1])), 2)
        r['stand_brake_side_frac'] = round(float(np.mean(pc[rows['stand']][:, 0, 1] > PROBE_NEUTRAL + MARGIN)), 4)
        r['stand_go_frac'] = round(float(np.mean(pc[rows['stand']][:, 0, 1] < 1496.0 - MARGIN)), 4)
        if run in chk:
            r['probe_check_gpu'] = {k: chk[run][k] for k in ('st10_thr_vs_b_us', 'st23_thr_vs_b_us', 'stand_brake_side_frac', 'low10_brake_frac')}
        out[run] = r
    five_ids = [r for _, r, *_ in FIVE]
    stats = {}
    for k in ('st10_thr_vs_b_us', 'st23_thr_vs_b_us', 'stand_brake_side_frac'):
        x = np.array([out[r][k] for r in five_ids])
        stats[k] = {'mean5': round(float(x.mean()), 4), 'sd5': round(float(np.std(x, ddof=1)), 4),
                    'max5': round(float(x.max()), 4), 'min5': round(float(x.min()), 4)}
    return {'runs': out, 'stats_five': stats, 'n': {k: int(len(v)) for k, v in rows.items()}}


def part4(proto):
    """v2 の 22 本の介入のステアの L1(best・last)。新しい拒否権の自己試験(eval/pred の保存した予測)"""
    with np.load(BUNDLE, allow_pickle=False) as z:
        y = np.nan_to_num(np.asarray(z['y'], np.float64))
        iv = z['interv'].astype(bool)
    out = {}
    pred_dir = REP2 / 'eval' / 'pred'
    for p in sorted(pred_dir.glob('*.npz')):
        name = p.name[:-4]
        ck = 'last' if name.endswith('_last') else 'best'
        run = name[:-5] if ck == 'last' else name
        if run == B_RUN:
            continue
        with np.load(p, allow_pickle=False) as z:
            if str(z['key']) != TABLE_KEY:
                raise SystemExit('%s の key が違う' % p)
            pr = np.asarray(z['pred'], np.float64)
        d = np.abs(pr - y)
        out.setdefault(run, {})['interv_steer_%s' % ck] = round(float(d[iv][:, :, 0].mean()), 3)
        out[run]['interv_thr_%s' % ck] = round(float(d[iv][:, :, 1].mean()), 3)
    return out


def main():
    proto = json.loads(PROTO2.read_text())
    res = {'made_by': str(Path(__file__).resolve()), 'script_sha256': sha256(__file__),
           'protocol_v2_sha256': sha256(PROTO2),
           'part1_interv_steer': part1(proto), 'part2_desk_new_fields': part2(proto), 'part3_probe_from_tab': part3(), 'part4_v2_interv_steer': part4(proto)}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res['part1_interv_steer']['stats'], ensure_ascii=False, indent=1))
    print(json.dumps(res['part2_desk_new_fields']['stats'], ensure_ascii=False, indent=1))
    print(json.dumps(res['part3_probe_from_tab']['stats_five'], ensure_ascii=False, indent=1))
    for run, r in res['part2_desk_new_fields']['v2_and_bfast'].items():
        print(run[:40].ljust(40), {k: (round(v, 3) if isinstance(v, float) else None) for k, v in r.items()
                                   if k.startswith('straight') or k.startswith('standstill')})
    for run, r in res['part3_probe_from_tab']['runs'].items():
        if 'probe_check_gpu' in r:
            print('probe', run[:40], {k: r[k] for k in ('st10_thr_vs_b_us', 'st23_thr_vs_b_us', 'stand_brake_side_frac')},
                  r['probe_check_gpu'])
    for run, r in res['part4_v2_interv_steer'].items():
        print('interv', run[:40].ljust(40), r)
    print('書いた:', OUT)


if __name__ == '__main__':
    sys.exit(main())
