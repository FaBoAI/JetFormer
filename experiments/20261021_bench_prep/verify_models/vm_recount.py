#!/usr/bin/env python3
"""V-models(2026-10-21・反証する側): 机上のベンチの数を、保存した予測から「別のコード」で数え直す。numpy だけ・読むだけ。

desk_eval.py・desk_lib.py・recount.py・rl_evaluate・policy_zoo・label_events を import しない。
  - 予測の表(part・scene・frame)を splits.json と aligned から自分で作り直し、desk_inputs.npz の並びと比べる
  - 正解 y・衝突の前後・primary・interv を aligned から自分で作る
  - R1 の起点は data/labels/<log>.npz の loc_detail(10/11 に作られた別の計算)の「LH → R1」の切り替わりから推定する
    (label_events.course_anchors を呼ばない)
出力: <out>/recount.json と <out>/frames.npz(GPU の段が使う)
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

REPO = Path('/home/jetson/jetformer')
DS = REPO / 'data' / 'datasets'
LAB = REPO / 'data' / 'labels'
DESK = Path('/home/jetson/jetformer_reports/bench_prep_20261021/desk')
OUT = Path(sys.argv[1])
G = 9.80665

def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()

def splits(ds):
    return json.load(open(DS / ds / 'splits.json'))

def aligned(ds, s):
    with np.load(DS / ds / 'aligned' / (s + '.npz')) as z:
        return {k: z[k] for k in z.files}

def impact(ts_us, imu, g=2.5, before=1.0, after=2.0):
    t = ts_us.astype(np.float64) / 1e6
    hg = np.sqrt(imu[:, :, 0].astype(np.float64) ** 2 + imu[:, :, 1].astype(np.float64) ** 2).max(axis=1) / G
    ti = np.sort(t[hg >= g])
    if not len(ti):
        return np.zeros(len(t), bool)
    # フレーム t が、ある衝突 ti について ti-before <= t <= ti+after に入るか = [t-after, t+before] に ti があるか
    lo = np.searchsorted(ti, t - after, side='left')
    hi = np.searchsorted(ti, t + before, side='right')
    return hi > lo

# ---------------------------------------------------------------- 1. 予測の表を作り直す
rows = []   # (part, ds, scene, frame)
for part, ds in ((0, 'ai_mix_20261011'), (1, 'intervention_val_20261011')):
    for s in splits(ds)['val']:
        a = aligned(ds, s)
        ok = a['vehicle_valid'] if part == 0 else (a['vehicle_valid'] & a['future_valid'])
        for i in np.nonzero(ok)[0]:
            rows.append((part, ds, s, int(i)))
N = len(rows)
part = np.array([r[0] for r in rows], np.int8)
scene = np.array([r[2] for r in rows])
frame = np.array([r[3] for r in rows], np.int32)
DI = np.load(DESK / 'pred' / 'desk_inputs.npz')
table_same = bool(N == len(DI['part']) and np.array_equal(part, DI['part']) and np.array_equal(scene, DI['scene'].astype(str))
                  and np.array_equal(frame, DI['frame']))

y = np.full((N, 10, 2), np.nan)
fv = np.zeros(N, bool); vv = np.zeros(N, bool); imp = np.zeros(N, bool)
erpm = np.full(N, np.nan); t_us = np.zeros(N, np.int64); genre = np.zeros(N, np.int16); loc = np.full(N, -1, np.int16)
cam = np.empty(N, dtype=object)
cache = {}
for k, (p, ds, s, i) in enumerate(rows):
    key = (ds, s)
    if key not in cache:
        cache.clear()
        a = aligned(ds, s)
        a['_imp'] = impact(a['timestamps_us'], a['imu'])
        cache[key] = a
    a = cache[key]
    y[k] = a['future_actions'][i]
    fv[k] = a['future_valid'][i]; vv[k] = a['vehicle_valid'][i]; imp[k] = a['_imp'][i]
    erpm[k] = a['vehicle'][i, 3]; t_us[k] = a['timestamps_us'][i]; genre[k] = a['mode'][i]
    loc[k] = a['loc_code'][i] if 'loc_code' in a else -1
    cam[k] = str(DS / ds / str(a['cam_files'][i]))
held = json.load(open(REPO / 'data/rl/ai_mix_20261011/eval/heldout.json'))
clean_sc = set(held['human_clean_scenes'])
yfin = np.isfinite(y).all(axis=(1, 2))
primary = (part == 0) & np.array([s in clean_sc for s in scene]) & fv & vv & ~imp
interv = (part == 1) & fv & vv & yfin
yd = DI['y'].astype(np.float64)
y_same = bool(np.array_equal(np.isnan(y), np.isnan(yd)) and np.nanmax(np.abs(y - yd)) == 0.0)

# ---------------------------------------------------------------- 2. R1 の起点(labels の loc_detail の LH(7) → R1(1))
anchors = {}
for lg in sorted({s[:15] for s in scene[primary]}):
    z = np.load(LAB / (lg + '.npz'))
    det = z['loc_detail']; ts = z['t_us'].astype(np.int64)
    st = np.nonzero((det[1:] == 1) & (det[:-1] == 7))[0] + 1
    anchors[lg] = ts[st] / 1e6 + 0.05      # locate(): T + 0.05 >= ta で次の起点に入る → ta ≈ 最初の R1 のフレーム + 0.05 秒以内
ts_s = t_us / 1e6
r1pre = np.zeros(N, bool); r1id = np.full(N, -1)
aid = 0
anchor_list = []
for lg, A in anchors.items():
    m_log = primary & np.array([s[:15] == lg for s in scene])
    for ta in A:
        w = m_log & (ts_s >= ta - 1.0) & (ts_s < ta)
        if w.any():
            r1pre |= w; r1id[w] = aid; anchor_list.append((lg, float(ta), int(w.sum()))); aid += 1
desk_r1 = DI['r1_pre1s'].astype(bool)
r1_cmp = {'mine': int(r1pre.sum()), 'desk': int(desk_r1.sum()), 'both': int((r1pre & desk_r1).sum()),
          'anchors_mine': aid, 'anchors_desk': int(len(np.unique(DI['r1_anchor_id'][desk_r1]))),
          'anchors_by_log_mine': {lg: int(len(A)) for lg, A in anchors.items()}}

# ---------------------------------------------------------------- 3. 予測(保存したもの)と指標
DB = json.load(open(DESK / 'desk_bench.json'))
rowsdb = {c['id']: c for c in DB['candidates']}

def load_pred(files):
    ps = []
    for f in files:
        with np.load(f) as z:
            assert str(z['key']) == '3d339512828b17195d568f1b93ee4a28208f2e85', f
            ps.append(z['pred'].astype(np.float64))
    return np.mean(ps, axis=0)

def last_files(c):
    return c['pred_sources']['last']['files']

v = -erpm * 0.5e-3
low_ho = primary & (v < 1.0)
def metrics(P, PL, mask_r1):
    d = np.abs(P - np.nan_to_num(y))
    out = {
        'l1_best_us': float(d[primary].mean()),
        'l1_steer_best_us': float(d[primary][:, :, 0].mean()),
        'l1_thr_best_us': float(d[primary][:, :, 1].mean()),
        'l1_last_us': float(np.abs(PL - np.nan_to_num(y))[primary].mean()),
        'l1_interv_us': float(d[interv].mean()),
        'steer_amp_ratio': float(np.std(P[primary][:, :, 0]) / np.std(y[primary][:, :, 0])),
        'r1_app_thr_diff_us': float(np.mean(P[mask_r1][:, 2, 1] - y[mask_r1][:, 2, 1])),
        'brake_lowspeed_heldout': [int((P[low_ho][:, 0, 1] > 1598.0).sum()), int(low_ho.sum())],
    }
    return out

res = {}
for cid, c in rowsdb.items():
    files = c['pred_sources']['best']['files']
    shas_ok = all(sha(f) == h for f, h in zip(files, c['pred_sources']['best']['sha256']))
    P = load_pred(files); PL = load_pred(last_files(c))
    m_mine = metrics(P, PL, r1pre)
    m_desk_r1 = metrics(P, PL, desk_r1)['r1_app_thr_diff_us']
    # AI 10/10 の低速(desk の sanity の TensorRT の予測から)
    sp = np.load(DESK / 'pred' / ('sanity_%s.npz' % cid))
    va = -sp['erpm'] * 0.5e-3
    lo = va < 1.0
    ai_k, ai_n = int((sp['pred'][lo][:, 0, 1] > 1598.0).sum()), int(lo.sum())
    hk, hn = m_mine['brake_lowspeed_heldout']
    m_mine['brake_lowspeed_frac'] = (hk + ai_k) / (hn + ai_n)
    m_mine['brake_lowspeed_ai'] = [ai_k, ai_n]
    m_mine['r1_app_thr_diff_us_with_desk_anchors'] = m_desk_r1
    m_mine['pred_sha_ok'] = shas_ok
    cmpd = {}
    for k in ('l1_best_us', 'l1_steer_best_us', 'l1_thr_best_us', 'l1_last_us', 'l1_interv_us', 'steer_amp_ratio',
              'r1_app_thr_diff_us', 'brake_lowspeed_frac'):
        dv = c.get(k)
        mine = m_mine['r1_app_thr_diff_us_with_desk_anchors'] if k == 'r1_app_thr_diff_us' else m_mine[k]
        cmpd[k] = {'desk': dv, 'mine': round(mine, 5), 'abs_diff': None if dv is None else round(abs(mine - dv), 6)}
    cmpd['r1_app_thr_diff_us_my_anchors'] = {'desk': c.get('r1_app_thr_diff_us'), 'mine': round(m_mine['r1_app_thr_diff_us'], 4)}
    res[cid] = {'metrics': m_mine, 'compare': cmpd}

counts = {'rows': N, 'table_same_as_desk_inputs': table_same, 'y_same': y_same, 'primary': int(primary.sum()),
          'primary_same_as_desk': bool(np.array_equal(primary, DI['primary'].astype(bool))),
          'interv': int(interv.sum()), 'interv_same_as_desk': bool(np.array_equal(interv, DI['interv'].astype(bool))),
          'lowspeed_heldout': int(low_ho.sum()), 'r1': r1_cmp,
          'genre_primary': {int(g): int((genre[primary] == g).sum()) for g in np.unique(genre[primary])},
          'genre_r1pre': {int(g): int((genre[r1pre] == g).sum()) for g in np.unique(genre[r1pre])}}
OUT.mkdir(parents=True, exist_ok=True)
json.dump({'counts': counts, 'anchors': anchor_list, 'candidates': res}, open(OUT / 'recount.json', 'w'), ensure_ascii=False, indent=1)
sel = primary | interv
np.savez(OUT / 'frames.npz', idx=np.nonzero(sel)[0], cam=np.array([str(x) for x in cam[sel]]), part=part[sel], scene=scene[sel],
         frame=frame[sel], genre=genre[sel], primary=primary[sel], interv=interv[sel], r1pre=r1pre[sel], r1id=r1id[sel],
         erpm=erpm[sel], t_us=t_us[sel], loc=loc[sel], y=y[sel])
np.savez(OUT / 'frames_all.npz', primary=primary, interv=interv, r1pre=r1pre, r1id=r1id, loc=loc, genre=genre, erpm=erpm, y=y, t_us=t_us,
         scene=scene)
print(json.dumps(counts, ensure_ascii=False))
for cid, r in res.items():
    worst = max((v['abs_diff'] or 0) for k, v in r['compare'].items() if 'abs_diff' in v)
    print(cid, 'worst_abs_diff', worst, 'r1(my anchors)', r['compare']['r1_app_thr_diff_us_my_anchors'], 'sha', r['metrics']['pred_sha_ok'])
