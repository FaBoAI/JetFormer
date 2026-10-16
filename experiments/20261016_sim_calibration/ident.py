#!/usr/bin/env python3
"""シミュレーター較正(2026-10-16)S1: 実車のログから車の応答を同定し、プロファイル P_cal・P_cfg を固める

    jf-heavy --mem 1500M --lock none -- nice -n 10 python -I ident.py [--steps data,i1,...] [--force]

protocol.json(sha256 61fc91cb…)の identification 章(I1〜I9・select・freeze)のとおり。当てはめ(fit)のシーンだけを使う
(保留のシーン = 各ログのシーンの番号 k % 3 == 1 は、calib_data で読み込んだ直後に消す。I7 の固定の 16 回のブレーキだけ例外)。
各段の結果は ident/<段>.json に書き、既にあれば飛ばす(電源断に備える。--force で作り直す)。
読み取りだけ(CPU)。data/raw・data/labels・ユーザーのシミュレーターは変えない。実車・PCA9685・I2C には触れない。
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calib_data as cd  # noqa: E402
import calib_model as cm  # noqa: E402

REPORT = Path('/home/jetson/jetformer_reports/sim_calibration_20261016')
OUTD = REPORT / 'ident'
PROTO = REPORT / 'protocol.json'
PROTO_SHA = '61fc91cbf67768ecc325ba4ebb0f9470bfbbe7658c566cdbb02eb8ceb2fef52d'
SEED = 20261016
G = cm.G
L_WB = cm.WHEELBASE
SIM_MU = {'carpet': 0.8, 'shortcut': 0.7, 'turf': 0.6, 'ptfe': 0.22, 'blue-mat': 0.65}
MU_SCALE_GRID = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
SIM_DEFAULT_CFG = {'maxSpeed': 7.24, 'maxSteer': 0.52, 'steerRate': 1.2, 'driveTau': 0.4, 'driveAccelerationCap': 6.0,
                   'brakeDeceleration': 1.2, 'frictionMultiplier': 1.0, 'control_delay_frames': 0}
OPS = {0: 'human', 1: 'ai'}
CACHE = OUTD / 'cache_fit.npz'


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def jdump(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_np_default), encoding='utf-8')
    tmp.replace(path)


def _np_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def jload(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def r4(x, n=4):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), n)


def rolling_std(x, n):
    """後ろ向き n 点の標準偏差(NaN を含めば inf)"""
    x = np.asarray(x, float)
    good = np.isfinite(x)
    xx = np.where(good, x, 0.0)
    c1 = np.concatenate([[0.0], np.cumsum(xx)])
    c2 = np.concatenate([[0.0], np.cumsum(xx * xx)])
    cn = np.concatenate([[0], np.cumsum(~good)])
    idx = np.arange(len(x))
    a = np.maximum(0, idx - n + 1)
    cnt = idx + 1 - a
    m = (c1[idx + 1] - c1[a]) / cnt
    var = np.maximum(0.0, (c2[idx + 1] - c2[a]) / cnt - m * m)
    sd = np.sqrt(var)
    sd[(cn[idx + 1] - cn[a]) > 0] = np.inf
    sd[cnt < n] = np.inf
    return sd


# =================================================================== データ
def step_data(force=False):
    if CACHE.exists() and not force:
        return
    proto_sha = hashlib.sha256(PROTO.read_bytes()).hexdigest()
    assert proto_sha == PROTO_SHA, 'protocol.json が固めたものと違う'
    inv = jload(REPORT / 'inventory.json')
    brakes = [e for e in inv['brake_events'] if e['clean']]
    assert len(brakes) == 16
    S = {k: [] for k in ('log_i', 'scene', 'op', 'loc', 'clean', 'n_s', 'u_s', 'u_v', 'thr', 'off', 'v5', 'vs', 'acc', 'r', 'a_lat',
                         'steer_sd', 'g', 't', 't_us')}
    windows = []
    logs = cd.list_logs()
    recs_g = {}
    meta = {'logs': [], 'protocol_sha256': proto_sha}
    recs = {}
    for li, lg in enumerate(logs):
        spans = [(e['t_s'], e['dur_s']) for e in brakes if e['log'] == lg]
        rec = cd.load_record(lg, 'fit', brake_spans=spans)
        if rec is None:
            continue
        g = cd.battery_g(rec)
        recs[lg] = rec
        recs_g[lg] = g
        meta['logs'].append({'log': lg, 'ai_log': rec['ai_log'], 'n_scenes': len(rec['scene_names']),
                             'fit_scenes': [s for k, s in enumerate(rec['scene_names']) if k % 3 != 1 and s not in cd.EXCLUDED_SCENES],
                             'gz_bias_dps': rec['gz_bias_dps'], 'alat_bias': rec['alat_bias'], 'latency_ms_logwide': rec['latency_ms']})
    # 電池の基準(I9): 当てはめのシーンの人の g の中央値
    g_h = np.concatenate([recs_g[lg][recs[lg]['op'] == 0] for lg in recs])
    g_a = np.concatenate([recs_g[lg][recs[lg]['op'] == 1] for lg in recs])
    g_ref_h = float(np.nanmedian(g_h))
    g_ref_a = float(np.nanmedian(g_a)) if np.isfinite(g_a).any() else g_ref_h
    meta['g_ref'] = {'human': g_ref_h, 'ai': g_ref_a, 'n_human': int(np.isfinite(g_h).sum()), 'n_ai': int(np.isfinite(g_a).sum())}
    # 窓の k_batt の基準: 人の窓は protocol のとおり人の中央値。AI の窓は AI の中央値(deviations.md D2:
    # AI と人の対応の違いは I6 で別の対応として扱うので、電池の係数に混ぜない)
    g_ref = {0: g_ref_h, 1: g_ref_a}
    for li, lg in enumerate(logs):
        if lg not in recs:
            continue
        rec = recs[lg]
        g = recs_g[lg]
        ws = cd.make_windows(rec, g=g, g_ref=g_ref)
        for w in ws:
            w['log_i'] = li
        windows += ws
        m = rec['keep'] | rec['brake_only']
        sd = rolling_std(rec['steer'], 30)
        S['log_i'].append(np.full(m.sum(), li)); S['scene'].append(rec['scene'][m]); S['op'].append(rec['op'][m])
        S['loc'].append(rec['loc'][m]); S['clean'].append(rec['clean'][m])
        n_s = rec['u_s'] / cd.MAX_STEER
        S['n_s'].append(n_s[m]); S['u_s'].append(rec['u_s'][m]); S['u_v'].append(rec['u_v'][m]); S['thr'].append(rec['thr'][m])
        S['off'].append(rec['off'][m]); S['v5'].append(rec['v5'][m]); S['vs'].append(rec['vs'][m]); S['acc'].append(rec['acc'][m])
        S['r'].append(rec['r'][m]); S['a_lat'].append(rec['a_lat'][m]); S['steer_sd'].append(sd[m]); S['g'].append(g[m])
        S['t'].append(rec['t'][m]); S['t_us'].append(rec['t_us'][m])
        log(lg, 'windows', len(ws), 'samples', int(m.sum()))
    # ブレーキの出来事の区間だけの印(保留のシーンでも I7 のために残した部分)
    S = {k: np.concatenate(v) for k, v in S.items()}
    Wd = cd.stack_windows(windows)
    Wd['log_i'] = np.array([w['log_i'] for w in windows])
    np.savez_compressed(CACHE, **{'S_' + k: v for k, v in S.items()}, **{'W_' + k: v for k, v in Wd.items() if k != 'n'},
                        logs=np.array(logs))
    meta['n_windows'] = {'human': int((Wd['op'] == 0).sum()), 'ai': int((Wd['op'] == 1).sum())}
    meta['n_windows_by_sec'] = {OPS[o]: {s: int(((Wd['op'] == o) & (Wd['sec'] == s)).sum()) for s in np.unique(Wd['sec'])} for o in (0, 1)}
    meta['k_batt_windows'] = {OPS[o]: {'n_adjusted': int(((Wd['op'] == o) & (Wd['n_g'] >= 30)).sum()),
                                       'p10_p50_p90': np.percentile(Wd['k_batt'][Wd['op'] == o], [10, 50, 90]).round(3).tolist()} for o in (0, 1)}
    jdump(OUTD / 'data.json', meta)
    log('data', meta['n_windows'], meta['g_ref'])


def load_cache():
    z = np.load(CACHE, allow_pickle=False)
    S = {k[2:]: z[k] for k in z.files if k.startswith('S_')}
    W = {k[2:]: z[k] for k in z.files if k.startswith('W_')}
    W['n'] = len(W['op'])
    return S, W, list(z['logs'])


def sub_windows(W, m):
    out = {k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == (W['n'],) else v) for k, v in W.items()}
    out['n'] = int(np.sum(m))
    return out


# =================================================================== I1・I2 舵の対応とアンダーステア
STEER_FIT_KNOTS = np.array([-1.0, -0.9, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
BIN_EDGES = np.concatenate([[-1.0001], np.arange(-0.95, 0.951, 0.05), [1.0001]])
LOC_GM = cd.LOC.index('GM')
A_LIN_MAX = 5.0          # I2: 頭打ちと二重に数えないため、要求の横加速度 κ0 v² ≤ 5 m/s² の時刻だけ(deviations D3)
HUMAN_LEFT_MAX = 0.81    # 人の左はここまで。先は AI だけ(protocol I1)


def steer_samples(S, op, vlo, vhi):
    """舵が 0.3 秒ほぼ一定(標準偏差 ≤ 8 µs)・きれい・GM 以外(deviations D3)の時刻"""
    m = S['clean'] & (S['steer_sd'] <= 8) & np.isfinite(S['r']) & (S['v5'] >= vlo) & (S['v5'] <= vhi) & (S['loc'] != LOC_GM)
    if op == 0:
        m_h = m & (S['op'] == 0) & (S['n_s'] <= HUMAN_LEFT_MAX)
        m_a = m & (S['op'] == 1) & (S['n_s'] > HUMAN_LEFT_MAX)
        return m_h | m_a
    return m & (S['op'] == 1)


def bin_medians(n, delta, min_bin=15):
    """帯ごとの (中央値の n, 中央値の δ, 数)。帯の番号 → 値の dict"""
    b = np.digitize(n, BIN_EDGES) - 1
    out = {}
    for i in range(len(BIN_EDGES) - 1):
        k = b == i
        if k.sum() >= min_bin:
            out[i] = (float(np.median(n[k])), float(np.median(delta[k])), int(k.sum()))
    return out


def fit_steer_map(bins):
    """帯の中央値 → 区分線形・単調(節 STEER_FIT_KNOTS)を重みつき最小二乗(重み = min(数, 300))。返り値: 節の δ、帯の表"""
    keys = sorted(bins)
    xs = np.array([bins[i][0] for i in keys]); ys = np.array([bins[i][1] for i in keys])
    ws = np.array([min(bins[i][2], 300) for i in keys], float)
    A = cm.interp_matrix(xs, STEER_FIT_KNOTS)
    vals = cm.nnls_monotone(A, ys, ws)
    tab = [[round(bins[i][0], 4), bins[i][2], round(bins[i][1], 5)] for i in keys]
    return vals, tab


def merged_bins(bins_own, bins_fill):
    """自分の帯が無いところを、もう一方の帯で埋める(AI の舵の範囲の穴を人で埋める。deviations D3)"""
    out = dict(bins_fill)
    out.update(bins_own)
    filled = sorted(set(bins_fill) - set(bins_own))
    return out, filled


def kappa_of(n, vals, v=None, K=0.0):
    d = np.interp(n, STEER_FIT_KNOTS, vals)
    k0 = np.tan(d) / L_WB
    if v is None:
        return k0
    return k0 / (1.0 + K * v * v)


def step_i1_i2(force=False, n_boot=500):
    out_p = OUTD / 'i1_i2_steering.json'
    if out_p.exists() and not force:
        return jload(out_p)
    S, W, logs = load_cache()
    rng = np.random.default_rng(SEED)
    res = {'method': 'protocol I1・I2 + deviations D3(GM を除く・I2 は κ0 v² ≤ 5 m/s²・I1 と I2 を交互に当てはめ κ0 = κ(1+K v²)・'
                     'AI の帯が無い範囲は人の帯で埋める)'}
    Kgrid = np.arange(0, 0.2001, 0.002)
    data = {}
    for op in (0, 1):
        m1 = steer_samples(S, op, 0.8, 2.2)
        m2 = steer_samples(S, op, 2.2, 4.0)
        data[op] = dict(n1=S['n_s'][m1], v1=S['v5'][m1], r1=S['r'][m1], n2=S['n_s'][m2], v2=S['v5'][m2], r2=S['r'][m2],
                        sc1=S['log_i'][m1] * 1000 + S['scene'][m1], sc2=S['log_i'][m2] * 1000 + S['scene'][m2],
                        N1=int(m1.sum()), N2=int(m2.sum()))

    def fit_once(D, K, fill_bins=None, w1=None):
        n1, v1, r1 = D['n1'], D['v1'], D['r1']
        if w1 is not None:
            n1, v1, r1 = n1[w1], v1[w1], r1[w1]
        d1 = np.arctan(L_WB * (r1 / v1) * (1 + K * v1 * v1))
        bins = bin_medians(n1, d1)
        filled = []
        if fill_bins is not None:
            bins, filled = merged_bins(bins, fill_bins)
        vals, tab = fit_steer_map(bins)
        return vals, tab, bins, filled

    def fit_K(D, vals, w2=None):
        n2, v2, r2 = D['n2'], D['v2'], D['r2']
        if w2 is not None:
            n2, v2, r2 = n2[w2], v2[w2], r2[w2]
        k0 = kappa_of(n2, vals)
        lin = np.abs(k0) * v2 * v2 <= A_LIN_MAX
        if lin.sum() < 20:
            return 0.0, None
        kap = r2 / v2
        errs = np.array([np.sqrt(np.mean((kap[lin] - k0[lin] / (1 + Kc * v2[lin] ** 2)) ** 2)) for Kc in Kgrid])
        return float(Kgrid[int(np.argmin(errs))]), {'rmse_K0': float(errs[0]), 'rmse_best': float(errs.min()),
                                                   'improve': float(1 - errs.min() / errs[0]), 'n_lin': int(lin.sum())}

    fits = {}
    for op in (0, 1):
        D = data[op]
        K = 0.0
        hist = []
        for it in range(4):
            fill = None if op == 0 else fits[0]['bins']
            vals, tab, bins, filled = fit_once(D, K, fill)
            Kb, info = fit_K(D, vals)
            keep = info is not None and info['improve'] >= 0.03
            hist.append(dict(info or {}, iter=it, K_best=Kb, kept=keep))
            K_new = Kb if keep else 0.0
            if abs(K_new - K) < 1e-9:
                break
            K = K_new
        vals, tab, bins, filled = fit_once(D, K, None if op == 0 else fits[0]['bins'])
        fits[op] = {'vals': vals, 'tab': tab, 'bins': bins, 'filled': filled, 'K': K, 'hist': hist}
    # ブートストラップ(シーン単位、人と AI で同じシーンの引き方)
    uniq = np.unique(np.concatenate([data[o][k] for o in (0, 1) for k in ('sc1', 'sc2')]))
    boots = {0: [], 1: []}; bootK = {0: [], 1: []}
    for bi in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        cnt = dict(zip(*np.unique(pick, return_counts=True)))
        bb = {}
        for op in (0, 1):
            D = data[op]
            w1 = np.repeat(np.arange(D['N1']), np.array([cnt.get(s, 0) for s in D['sc1']], int))
            w2 = np.repeat(np.arange(D['N2']), np.array([cnt.get(s, 0) for s in D['sc2']], int))
            K = fits[op]['K']
            vb, _, bins_b, _ = fit_once(D, K, None if op == 0 else bb[0], w1=w1)
            bb[op] = bins_b
            boots[op].append(vb)
            Kb, info = fit_K(D, vb, w2=w2)
            bootK[op].append(Kb if (info is not None and info['improve'] >= 0.03) else 0.0)
    for op in (0, 1):
        f = fits[op]
        bt = np.array(boots[op])
        res[OPS[op]] = {
            'knots_n': STEER_FIT_KNOTS.tolist(), 'delta_rad': f['vals'].tolist(),
            'kappa0_per_m': (np.tan(f['vals']) / L_WB).tolist(),
            'delta_p5': np.percentile(bt, 5, axis=0).tolist(), 'delta_p95': np.percentile(bt, 95, axis=0).tolist(),
            'kappa0_p5': (np.tan(np.percentile(bt, 5, axis=0)) / L_WB).tolist(),
            'kappa0_p95': (np.tan(np.percentile(bt, 95, axis=0)) / L_WB).tolist(),
            'K_us_I2': f['K'], 'K_us_boot_p5_p50_p95': np.percentile(bootK[op], [5, 50, 95]).tolist(), 'K_iter': f['hist'],
            'bins_low_speed': f['tab'], 'bins_filled_from_human': [[round(float(BIN_EDGES[i]), 3), round(float(BIN_EDGES[i + 1]), 3)] for i in f['filled']],
            'n_samples_low': data[op]['N1'], 'n_samples_high': data[op]['N2'], 'n_boot': n_boot,
        }
        log('I1/I2', OPS[op], 'K', f['K'], 'kappa0', np.round(np.tan(f['vals']) / L_WB, 3), 'filled', len(f['filled']))
    res['n_scenes'] = int(len(uniq))
    res['sim_current'] = {'delta_rad_at_knots': (STEER_FIT_KNOTS * cm.MAX_STEER).tolist(),
                          'kappa_per_m_at_knots': (np.tan(STEER_FIT_KNOTS * cm.MAX_STEER) / L_WB).tolist(), 'K_us': 0.0}
    # 人と AI の違い(protocol I6 の 10% の決まり)。|u| = 0.5 rad(= 正規化 0.96)を主、正規化 0.5 も参考に。速さ 1.5 m/s の曲率
    cmp_ = {}
    for name, nn in (('u_s=+0.5rad', 0.5 / 0.52), ('u_s=-0.5rad', -0.5 / 0.52), ('n=+0.5', 0.5), ('n=-0.5', -0.5)):
        kh = float(kappa_of(nn, fits[0]['vals'], 1.5, fits[0]['K']))
        ka = float(kappa_of(nn, fits[1]['vals'], 1.5, fits[1]['K']))
        cmp_[name] = {'human_kappa_at_1.5mps': kh, 'ai_kappa_at_1.5mps': ka, 'rel_diff': (ka - kh) / kh}
    res['human_vs_ai'] = cmp_
    jdump(out_p, res)
    return res


# =================================================================== I3 路面ごとの横の限界
DEM_MIN = 4.0            # I3: 要求の横加速度の下限 [m/s²](直線の雑音を『滑り』に数えない。deviations D4)
TRACKS = Path('/home/jetson/jetformer_reports/overnight_20261011/course')
COURSE_MAP = TRACKS / 'course_map.json'
SEC_SURF = {'R1': 'shortcut', 'GM': 'turf', 'DE': 'carpet', 'LH': 'carpet', 'HS': 'carpet', 'SW': 'carpet'}


def demanded_alat(S, st):
    """要求の横加速度 |κ(u, v)| v²(操作者ごとの舵の対応と K)"""
    dem = np.full(len(S['op']), np.nan)
    for op in (0, 1):
        f = st[OPS[op]]
        m = S['op'] == op
        k = kappa_of(S['n_s'][m], np.array(f['delta_rad']), S['v5'][m], f['K_us_I2'])
        dem[m] = np.abs(k) * S['v5'][m] ** 2
    return dem


def ptfe_mask(S, logs):
    """AI の 5 本(overnight_20261011/course/tracks_*.npz)の位置で、PTFE・芝(PTFE 以外)・不明 を分ける。0 = 不明、1 = 芝、2 = PTFE"""
    cmj = jload(COURSE_MAP)
    tp = np.array(cmj['areas']['turf_low_mu']['polygon']); pp = np.array(cmj['areas']['ptfe_white_sheet']['polygon'])
    out = np.zeros(len(S['op']), int)
    for li, lg in enumerate(logs):
        p = TRACKS / ('tracks_%s.npz' % lg)
        if not p.exists():
            continue
        z = np.load(p)
        tu = z['t_us'].astype(np.int64); ok = z['valid'].astype(bool)
        tu, X, Z = tu[ok], z['x'][ok], z['z'][ok]
        m = S['log_i'] == li
        q = S['t_us'][m].astype(np.int64)
        j = np.clip(np.searchsorted(tu, q), 1, len(tu) - 1)
        j = np.where(np.abs(tu[j - 1] - q) < np.abs(tu[j] - q), j - 1, j)
        near = np.abs(tu[j] - q) <= 50_000
        x, zz = X[j], Z[j]
        in_t = (x >= tp[:, 0].min()) & (x <= tp[:, 0].max()) & (zz >= tp[:, 1].min()) & (zz <= tp[:, 1].max())
        in_p = (x >= pp[:, 0].min()) & (x <= pp[:, 0].max()) & (zz >= pp[:, 1].min()) & (zz <= pp[:, 1].max())
        lab = np.where(near & in_p, 2, np.where(near & in_t, 1, 0))
        out[np.nonzero(m)[0]] = lab
    return out


def step_i3(force=False, n_boot=1000):
    out_p = OUTD / 'i3_grip.json'
    if out_p.exists() and not force:
        return jload(out_p)
    S, W, logs = load_cache()
    st = jload(OUTD / 'i1_i2_steering.json')
    rng = np.random.default_rng(SEED + 3)
    dem = demanded_alat(S, st)
    al = np.abs(S['a_lat'])
    base = S['clean'] & np.isfinite(S['a_lat']) & (S['v5'] >= 0.5) & (S['steer_sd'] <= 8)
    slide = base & (dem >= DEM_MIN) & (dem >= 1.3 * al)
    literal = S['clean'] & np.isfinite(S['a_lat']) & (S['v5'] >= 0.5) & (dem >= 1.3 * al)
    sc = S['log_i'] * 1000 + S['scene']
    secname = np.array([cd.SEC_NAME[cd.LOC[c]] for c in S['loc']])

    def med_boot(m):
        x = al[m]; s = sc[m]
        if len(x) == 0:
            return None, None, None
        uniq = np.unique(s)
        bs = []
        for _ in range(n_boot):
            pick = rng.choice(uniq, len(uniq), replace=True)
            cnt = dict(zip(*np.unique(pick, return_counts=True)))
            w = np.array([cnt.get(v, 0) for v in s], int)
            if w.sum() == 0:
                continue
            bs.append(np.median(np.repeat(x, w)))
        return float(np.median(x)), float(np.percentile(bs, 5)), float(np.percentile(bs, 95))

    res = {'method': 'protocol I3 + deviations D4(滑りの時刻 = 要求 ≥ 1.3 × |a_lat| かつ 要求 ≥ %.1f m/s² かつ 舵が 0.3 秒一定。'
                     'カーペットは DE・LH・HS・SW をまとめる)' % DEM_MIN, 'by_section': {}}
    for sname in ('R1', 'GM', 'DE', 'LH', 'HS', 'SW'):
        m = slide & (secname == sname)
        med, lo, hi = med_boot(m)
        ml = literal & (secname == sname)
        res['by_section'][sname] = {'n': int(m.sum()), 'median_alat': med, 'p5': lo, 'p95': hi,
                                    'mu': None if med is None else med / G, 'enough': bool(m.sum() >= 200),
                                    'literal_rule_n': int(ml.sum()), 'literal_rule_median_alat': float(np.median(al[ml])) if ml.any() else None,
                                    'v_median': float(np.median(S['v5'][m])) if m.any() else None,
                                    'demanded_median': float(np.median(dem[m])) if m.any() else None}
    # 路面
    surf = {}
    m_car = slide & np.isin(secname, ['DE', 'LH', 'HS', 'SW'])
    med, lo, hi = med_boot(m_car)
    surf['carpet'] = {'mu': med / G, 'mu_p5': lo / G, 'mu_p95': hi / G, 'n': int(m_car.sum()), 'from': 'DE・LH・HS・SW の滑りの時刻'}
    r1 = res['by_section']['R1']
    if r1['enough']:
        surf['shortcut'] = {'mu': r1['mu'], 'mu_p5': r1['p5'] / G, 'mu_p95': r1['p95'] / G, 'n': r1['n'], 'from': 'R1 の滑りの時刻'}
    else:
        f = surf['carpet']['mu'] / SIM_MU['carpet']
        surf['shortcut'] = {'mu': SIM_MU['shortcut'] * f, 'mu_p5': None, 'mu_p95': None, 'n': r1['n'], 'from': '同定できず(カーペットの倍率)'}
    # GM: PTFE と芝
    pt = ptfe_mask(S, logs)
    m_gm = slide & (secname == 'GM')
    n_turf = int((m_gm & (pt == 1)).sum()); n_ptfe = int((m_gm & (pt == 2)).sum())
    res['gm_split'] = {'n_turf_ai_tracks': n_turf, 'n_ptfe_ai_tracks': n_ptfe, 'n_gm_unknown_position': int((m_gm & (pt == 0)).sum())}
    if n_turf >= 200 and n_ptfe >= 200:
        for nm, code in (('turf', 1), ('ptfe', 2)):
            med, lo, hi = med_boot(m_gm & (pt == code))
            surf[nm] = {'mu': med / G, 'mu_p5': lo / G, 'mu_p95': hi / G, 'n': int((m_gm & (pt == code)).sum()), 'from': 'GM・AI の位置で分けた'}
        res['gm_split']['used'] = 'separate'
    else:
        gm = res['by_section']['GM']
        if gm['enough']:
            mult = gm['mu'] / SIM_MU['turf']
            surf['turf'] = {'mu': SIM_MU['turf'] * mult, 'mu_p5': gm['p5'] / G, 'mu_p95': gm['p95'] / G, 'n': gm['n'],
                            'from': 'GM の滑りの時刻(PTFE と芝を分けられないので 1 つの倍率 %.3f)' % mult}
            surf['ptfe'] = {'mu': SIM_MU['ptfe'] * mult, 'mu_p5': SIM_MU['ptfe'] * gm['p5'] / G / SIM_MU['turf'],
                            'mu_p95': SIM_MU['ptfe'] * gm['p95'] / G / SIM_MU['turf'], 'n': 0,
                            'from': 'GM の倍率 %.3f × シミュレーターの 0.22(同定できず、比を保つ)' % mult}
            # 参考: AI の位置で分けた中央値(数が足りない)
            for nm, code in (('turf', 1), ('ptfe', 2)):
                mm = m_gm & (pt == code)
                res['gm_split']['median_alat_' + nm] = float(np.median(al[mm])) if mm.any() else None
        res['gm_split']['used'] = 'one multiplier (protocol I3 ptfe_vs_turf の足りないとき)'
    f = surf['carpet']['mu'] / SIM_MU['carpet']
    surf['blue-mat'] = {'mu': SIM_MU['blue-mat'] * f, 'mu_p5': None, 'mu_p95': None, 'n': 0, 'from': 'カーペットの倍率 × 0.65(シミュレーターの比のまま)'}
    for k in surf:
        surf[k]['sim_current'] = SIM_MU[k]
    res['surfaces'] = surf
    res['dem_min'] = DEM_MIN
    jdump(out_p, res)
    log('I3', {k: round(v['mu'], 3) for k, v in surf.items()}, res['gm_split'])
    return res


# =================================================================== I4 舵の遅れと速さ
DS_GRID = np.round(np.arange(0.0, 0.1501, 0.01), 3)
RS_GRID = np.array([1.2, 2, 3, 4, 6, 8, 12, np.inf])


def steer_prm(vals, K, d_s=0.0, r_s=np.inf):
    v = np.asarray(vals, float)
    return {'steer_L': v[5:].tolist(), 'steer_R': (-v[5::-1]).tolist(), 'K_us': float(K), 'd_s': float(d_s), 'r_s': float(r_s)}


def mu_windows(Wx, surf_mu):
    return np.array([surf_mu[s] for s in Wx['surface']])


def lat_grid(Wx, vals, K, mu, tru):
    """格子の各点で、窓ごとのヨーの二乗平均誤差(形 [d_s, r_s, 窓])"""
    out = np.zeros((len(DS_GRID), len(RS_GRID), Wx['n']))
    for i, d in enumerate(DS_GRID):
        for j, rs in enumerate(RS_GRID):
            rec = cm.replay(Wx, 'cal', steer_prm(vals, K, d, rs), mode='lat', mu=mu, record=('psi',))
            out[i, j] = cm.errors(rec, tru)['e_r'] ** 2
    return out


def step_i4(force=False, n_boot=1000):
    out_p = OUTD / 'i4_steer_dynamics.json'
    if out_p.exists() and not force:
        return jload(out_p)
    S, W, logs = load_cache()
    st = jload(OUTD / 'i1_i2_steering.json')
    gr = jload(OUTD / 'i3_grip.json')
    surf_mu = {k: v['mu'] for k, v in gr['surfaces'].items()}
    rng = np.random.default_rng(SEED + 4)
    res = {'method': 'protocol I4: 横だけの再生(実際の速さ)。格子 d_s × r_s、窓のヨーの RMSE(フレームの平均のヨー)を最小に。操作者ごと',
           'd_s_grid': DS_GRID.tolist(), 'r_s_grid': [None if not np.isfinite(x) else float(x) for x in RS_GRID]}
    grids = {}
    for op in (0, 1):
        Wx = sub_windows(W, W['op'] == op)
        tru = cm.truth(Wx)
        mu = mu_windows(Wx, surf_mu)
        f = st[OPS[op]]
        vals = np.array(f['delta_rad'])
        G1 = lat_grid(Wx, vals, f['K_us_I2'], mu, tru)
        grids[op] = G1
        tot = np.sqrt(G1.mean(axis=2))
        i, j = np.unravel_index(np.argmin(tot), tot.shape)
        # ブートストラップ(シーン単位)で最良の格子点の幅
        sc = np.array([hash((a, b)) for a, b in zip(Wx['log'], Wx['scene'])])
        uniq = np.unique(sc)
        bi_d, bi_r = [], []
        for _ in range(n_boot):
            pick = rng.choice(uniq, len(uniq), replace=True)
            cnt = dict(zip(*np.unique(pick, return_counts=True)))
            w = np.array([cnt.get(v, 0) for v in sc], float)
            tb = (G1 * w).sum(axis=2) / w.sum()
            a, b = np.unravel_index(np.argmin(tb), tb.shape)
            bi_d.append(DS_GRID[a]); bi_r.append(RS_GRID[b])
        # 部品の選別: K あり/なし(同じ d_s・r_s)
        rec0 = cm.replay(Wx, 'cal', steer_prm(vals, 0.0, DS_GRID[i], RS_GRID[j]), mode='lat', mu=mu, record=('psi',))
        e_noK = float(np.sqrt(np.mean(cm.errors(rec0, tru)['e_r'] ** 2)))
        # シミュレーターの今の値(遅れ 0・1.2 rad/s・直線の対応・シミュレーターの μ)
        mu0 = np.array([SIM_MU[s] for s in Wx['surface']])
        recd = cm.replay(Wx, 'sim', dict(SIM_DEFAULT_CFG), mode='lat', mu=mu0, record=('psi',))
        e_def = float(np.sqrt(np.mean(cm.errors(recd, tru)['e_r'] ** 2)))
        res[OPS[op]] = {
            'n_windows': Wx['n'], 'd_s': float(DS_GRID[i]), 'r_s': None if not np.isfinite(RS_GRID[j]) else float(RS_GRID[j]),
            'yaw_rmse_best': float(tot[i, j]), 'yaw_rmse_table': [[float(x) for x in row] for row in tot],
            'd_s_boot_p5_p50_p95': np.percentile(bi_d, [5, 50, 95]).tolist(),
            'r_s_boot_values': {str(k): int(v) for k, v in zip(*np.unique(np.array(bi_r).astype(str), return_counts=True))},
            'K_us_I2': f['K_us_I2'], 'yaw_rmse_K0_same_dyn': e_noK,
            'K_improve': 1 - float(tot[i, j]) / e_noK if f['K_us_I2'] > 0 else 0.0,
            'yaw_rmse_sim_default_lat_only': e_def,
        }
        log('I4', OPS[op], 'd_s', DS_GRID[i], 'r_s', RS_GRID[j], 'rmse', round(float(tot[i, j]), 4), 'default', round(e_def, 4), 'noK', round(e_noK, 4))
    # AI の窓で、人の舵の対応を使ったとき(AI 専用の対応を残すかの選別)
    Wa = sub_windows(W, W['op'] == 1)
    tru = cm.truth(Wa)
    mu = mu_windows(Wa, surf_mu)
    fh = st['human']
    Gh = lat_grid(Wa, np.array(fh['delta_rad']), fh['K_us_I2'], mu, tru)
    toth = np.sqrt(Gh.mean(axis=2))
    res['ai_windows_with_human_map'] = {'yaw_rmse_best': float(toth.min()), 'ai_map_rmse_best': res['ai']['yaw_rmse_best'],
                                        'improve_ai_map': 1 - res['ai']['yaw_rmse_best'] / float(toth.min())}
    d_ai, d_h = res['ai']['d_s'], res['human']['d_s']
    res['consistency'] = {'ai_minus_human_d_s': d_ai - d_h, 'consistent_0_to_0.025': bool(0 <= d_ai - d_h <= 0.025 + 1e-9)}
    res['use'] = 'AI の値(AI の窓 %d ≥ 100)' % res['ai']['n_windows'] if res['ai']['n_windows'] >= 100 else '人の値 + 0.008 秒'
    np.savez_compressed(OUTD / 'i4_grids.npz', human=grids[0], ai=grids[1], ai_humanmap=Gh)
    jdump(out_p, res)
    return res


# =================================================================== I5 惰性とこすれ
def coast_mask(S):
    """スロットル |thr − 1498| ≤ 12 µs が 0.3 秒以上続いた後の時刻・v > 1 m/s・きれい(inventory と同じ)"""
    neutral = np.abs(S['thr'] - cd.THR_N) <= 12
    m = np.zeros(len(neutral), bool)
    key = S['log_i'] * 1_000_000 + 0
    brk = np.concatenate([[True], (np.diff(S['log_i']) != 0) | (np.diff(S['t']) > 0.05)])
    run_start = None
    good = neutral & S['clean']
    for i in range(len(good)):
        if brk[i] or not good[i]:
            run_start = i if good[i] else None
        elif run_start is None:
            run_start = i
        if good[i] and run_start is not None and S['t'][i] - S['t'][run_start] >= 0.3 - 1e-9:
            m[i] = True
    return m & (S['vs'] > 1.0)


def step_i5(force=False, n_boot=2000):
    out_p = OUTD / 'i5_coast.json'
    if out_p.exists() and not force:
        return jload(out_p)
    S, W, logs = load_cache()
    rng = np.random.default_rng(SEED + 5)
    cm_ = coast_mask(S)
    dec = -S['acc']
    alat_vr = np.abs(S['v5'] * S['r'])
    alat_imu = np.abs(S['a_lat'])
    st = cm_ & (np.abs(S['off']) < 100)
    tu = cm_ & (np.abs(S['off']) > 250)
    sc = S['log_i'] * 1000 + S['scene']

    def fit(ws=None, wt=None, alat=alat_vr):
        d_s = dec[st] if ws is None else np.repeat(dec[st], ws)
        c0 = float(np.median(d_s))
        x = alat[tu] if wt is None else np.repeat(alat[tu], wt)
        y = (dec[tu] if wt is None else np.repeat(dec[tu], wt)) - c0
        k = float(np.sum(x * y) / np.sum(x * x)) if len(x) else float('nan')
        return c0, k

    c0, k_s = fit()
    c0_i, k_s_imu = fit(alat=alat_imu)
    uniq = np.unique(np.concatenate([sc[st], sc[tu]]))
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        cnt = dict(zip(*np.unique(pick, return_counts=True)))
        ws = np.array([cnt.get(v, 0) for v in sc[st]], int)
        wt = np.array([cnt.get(v, 0) for v in sc[tu]], int)
        if ws.sum() < 10 or wt.sum() < 10:
            continue
        bs.append(fit(ws, wt))
    bs = np.array(bs)
    # 区間ごとの残り(参考)
    resid_tab = []
    for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 9), (9, 20)):
        k = (alat_vr[tu] >= lo) & (alat_vr[tu] < hi)
        if k.sum() >= 10:
            resid_tab.append({'alat_band': [lo, hi], 'n': int(k.sum()), 'decel_median': float(np.median(dec[tu][k])),
                              'pred': c0 + k_s * float(np.median(alat_vr[tu][k]))})
    res = {'method': 'protocol I5: 直線(|ずれ| < 100 µs)の減速の中央値 → c0。曲がり(|ずれ| > 250 µs)の減速 − c0 を |a_lat| に原点を通る回帰 → k_s。'
                     '|a_lat| はシミュレーターと同じ定義 |v r|(実際の速さ × ヨー)。IMU の横加速度での値も参考に出す',
           'c0': c0, 'c0_p5_p95': np.percentile(bs[:, 0], [5, 95]).tolist(), 'k_s': k_s, 'k_s_p5_p95': np.percentile(bs[:, 1], [5, 95]).tolist(),
           'k_s_with_imu_alat': k_s_imu, 'n_straight': int(st.sum()), 'n_turning': int(tu.sum()),
           'straight_s': float(st.sum() * 0.01), 'turning_s': float(tu.sum() * 0.01),
           'straight_decel_p10_p90': np.percentile(dec[st], [10, 90]).tolist() if st.any() else None,
           'turning_decel_median': float(np.median(dec[tu])) if tu.any() else None,
           'turning_alat_median': float(np.median(alat_vr[tu])) if tu.any() else None,
           'by_operator_n': {'straight': {OPS[o]: int((st & (S['op'] == o)).sum()) for o in (0, 1)},
                             'turning': {OPS[o]: int((tu & (S['op'] == o)).sum()) for o in (0, 1)}},
           'residual_by_alat': resid_tab, 'prior': {'c0': 0.56, 'turning_decel': 2.03},
           'sim_current': {'coast_decel': 1.2, 'scrub': 0.0}}
    jdump(out_p, res)
    log('I5', 'c0', round(c0, 3), res['c0_p5_p95'], 'k_s', round(k_s, 4), res['k_s_p5_p95'], 'n', res['n_straight'], res['n_turning'])
    return res


# =================================================================== I7 ESC のブレーキ
def fit_brake(off, ab, w=None):
    """a_b(off) = s_b × max(0, off − db_b) の最小二乗(db_b は 0〜300 µs の格子、s_b ≥ 0 は閉じた形)"""
    w = np.ones(len(off)) if w is None else np.asarray(w, float)
    best = None
    for db in np.arange(0.0, 300.1, 1.0):
        x = np.maximum(0.0, off - db)
        sxx = np.sum(w * x * x)
        if sxx <= 0:
            continue
        s = max(0.0, np.sum(w * x * ab) / sxx)
        sse = np.sum(w * (ab - s * x) ** 2)
        if best is None or sse < best[2] - 1e-12:
            best = (float(s), float(db), float(sse))
    return best


def step_i7(force=False, n_boot=2000):
    out_p = OUTD / 'i7_brake.json'
    if out_p.exists() and not force:
        return jload(out_p)
    S, W, logs = load_cache()
    c5 = jload(OUTD / 'i5_coast.json')
    c0, k_s = c5['c0'], c5['k_s']
    inv = jload(REPORT / 'inventory.json')
    ev = [e for e in inv['brake_events'] if e['clean']]
    rows = []
    for e in ev:
        li = logs.index(e['log'])
        m = (S['log_i'] == li) & (S['t'] >= e['t_s'] - 1e-6) & (S['t'] <= e['t_s'] + e['dur_s'] + 1e-6)
        alat = float(np.nanmean(np.abs(S['v5'][m] * S['r'][m]))) if m.any() else float('nan')
        held = None
        if m.any():
            held = bool(int(np.median(S['scene'][m])) % 3 == 1)
        a_b = e['mean_decel'] - (c0 + k_s * alat)
        rows.append({'log': e['log'], 't_s': e['t_s'], 'op': e['op'], 'loc': e['loc'], 'dur_s': e['dur_s'], 'v0': e['v0'], 'v1': e['v1'],
                     'off_us': e['thr_off_med'], 'steer_off_us': e['steer_off_med'], 'mean_decel': e['mean_decel'],
                     'alat_mean': alat, 'a_b_literal': a_b, 'in_heldout_scene': held, 'n_samples': int(m.sum())})
    off = np.array([r['off_us'] for r in rows]); ab_lit = np.array([r['a_b_literal'] for r in rows])
    s_l, db_l, _ = fit_brake(off, ab_lit)
    # deviations D5: 惰性の差し引きを、I5 の直線(c0 + k_s|a_lat|)ではなく、実測の惰性の曲線(|a_lat| の帯ごとの中央値)にする。
    # 直線は |a_lat| 2〜6 m/s² でこすれを約 0.7 m/s² 多く見積もり(i5_coast.json の residual_by_alat)、曲がりながらの弱いブレーキの効きを負にするため
    cmask = coast_mask(S)
    al_c = np.abs(S['v5'][cmask] * S['r'][cmask]); dc = -S['acc'][cmask]
    edges = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 20.0])
    cx, cy, cn = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (al_c >= a) & (al_c < b)
        if k.sum() >= 20:
            cx.append(float(np.median(al_c[k]))); cy.append(float(np.median(dc[k]))); cn.append(int(k.sum()))
    coast_curve = {'alat': cx, 'decel': cy, 'n': cn}
    for r in rows:
        r['coast_decel_curve'] = float(np.interp(r['alat_mean'], cx, cy))
        r['a_b_d5'] = r['mean_decel'] - r['coast_decel_curve']
    ab = np.array([r['a_b_d5'] for r in rows])
    s_b, db_b, sse = fit_brake(off, ab)
    nom200 = s_b * max(0.0, 200 - db_b)
    rng = np.random.default_rng(SEED + 7)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(rows), len(rows))
        f = fit_brake(off[idx], ab[idx])     # D5 の a_b で
        if f is None:
            continue
        bs.append((f[0], f[1], f[0] * max(0.0, 200 - f[1])))
    bs = np.array(bs)
    p10, p50, p90 = np.percentile(bs[:, 2], [10, 50, 90])
    low200 = min(p10, 0.6 * nom200)
    high200 = max(p90, 1.4 * nom200)
    res = {'method': 'protocol I7 + deviations D5: a_b = 観測の平均の減速 − 実測の惰性の曲線(|a_lat| の帯の中央値)、'
                     'a_b(off) = s_b × max(0, off − db_b) の最小二乗(db_b は 1 µs 刻みの格子)。出来事のブートストラップ 2000 回',
           'events': rows, 'n_events': len(rows), 'n_strong_ge300': int((off >= 300).sum()),
           'n_in_heldout_scenes': int(sum(1 for r in rows if r['in_heldout_scene'])),
           'nominal': {'s_b': s_b, 'db_b': db_b, 'a_b_at_200us': nom200, 'a_b_at_350us': s_b * max(0.0, 350 - db_b)},
           'boot_a_b_200_p10_p50_p90': [float(p10), float(p50), float(p90)],
           'boot_s_b_p10_p90': np.percentile(bs[:, 0], [10, 90]).tolist(), 'boot_db_b_p10_p90': np.percentile(bs[:, 1], [10, 90]).tolist(),
           'scenarios': {
               'nominal': {'s_b': s_b, 'db_b': db_b, 'a_b_at_200us': nom200},
               'low': {'s_b': s_b * (low200 / nom200) if nom200 > 0 else 0.0, 'db_b': db_b, 'a_b_at_200us': float(low200),
                       'rule': 'p10 と 名目の 60% の小さい方'},
               'high': {'s_b': s_b * (high200 / nom200) if nom200 > 0 else 0.0, 'db_b': db_b, 'a_b_at_200us': float(high200),
                        'rule': 'p90 と 名目の 140% の大きい方'}},
           'total_decel_at_200us_straight': {'nominal': c0 + nom200, 'low': c0 + low200, 'high': c0 + high200},
           'protocol_literal': {'s_b': s_l, 'db_b': db_l, 'a_b_at_200us': s_l * max(0.0, 200 - db_l),
                                'a_b_at_350us': s_l * max(0.0, 350 - db_l),
                                'note': 'protocol I7 の字義どおり(I5 の直線を引く)。+200 µs は惰性と同じになる。上限の試験の追加の場合 literal として残す'},
           'coast_curve_used_d5': coast_curve,
           'c0_k_s_used': {'c0': c0, 'k_s': k_s},
           'sim_current': {'brakeDeceleration': 1.2, 'note': '惰性・ブレーキ・後退が同じ 1.2 m/s²'},
           'limit': '強いブレーキ(≥ 300 µs)は 3 回だけ。表は仮(計画項目 8 の後に v2)'}
    jdump(out_p, res)
    log('I7', res['nominal'], res['boot_a_b_200_p10_p50_p90'], 'low/high', low200, high200)
    return res


# =================================================================== I6 速さの対応と駆動
I6_NAMES = ['T0.5', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7.24', 'u_db', 'tau_up', 'a_cap', 'tau_down', 'd_v']
I6_LO = np.array([0] * 8 + [0.0, 0.15, 1.5, 0.05, 0.0])
I6_HI = np.array([8] * 8 + [0.45, 1.5, 8.0, 1.0, 0.15])
I6_X0 = np.array([0.5, 1.0, 2.0, 2.9, 3.6, 4.1, 4.3, 4.4, 0.1, 0.45, 4.0, 0.3, 0.04])
I6_STEP = np.array([0.15, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.05, 0.1, 0.8, 0.1, 0.02])


def i6_prm(x, fixed, variant):
    x = np.clip(x, I6_LO, I6_HI)
    T = np.maximum.accumulate(np.maximum(x[:8], 0.0))
    if variant.get('no_saturation'):
        # 頭打ちなし: 4 より上は 3〜4 の傾きのまま伸ばす
        sl = T[4] - T[3]
        T = np.concatenate([T[:5], T[4] + sl * (np.array([5.0, 6.0, 7.24]) - 4.0)])
    p = {'T': T.tolist(), 'u_db': float(x[8]), 'tau_up': float(x[9]), 'a_cap': float(x[10]),
         'tau_down': 0.0 if variant.get('no_tau_down') else float(x[11]),
         'd_v': 0.0 if variant.get('no_d_v') else float(x[12]),
         'c0': fixed['c0'], 'k_s': 0.0 if variant.get('no_k_s') else fixed['k_s'], 's_b': fixed['s_b'], 'db_b': fixed['db_b']}
    return p


def i6_objective(Wx, tru, mu, fixed, variant, use_kb=True, exact=False):
    kb = Wx['k_batt'] if use_kb else np.ones(Wx['n'])

    def f(x, return_parts=False):
        xc = np.clip(x, I6_LO, I6_HI)
        pen = 10.0 * float(np.sum((x - xc) ** 2))
        T = xc[:8]
        pen += 100.0 * float(np.sum(np.maximum(0.0, T[:-1] - T[1:]) ** 2))
        p = i6_prm(x, fixed, variant)
        if exact:
            vsim = cm.replay(Wx, 'cal', p, mode='lon', mu=mu, k_batt=kb, record=('v',))['v']
        else:
            vsim = cm.lon_fast(Wx, p, mu, kb, h=0.005)       # 当てはめは 5 ms の小区間(J の差 ≈ 0.2%)。最終の値は exact で出し直す
        e1 = vsim[:, -1] - tru['v'][:, -1]
        ew = np.mean((vsim[:, 1:] - tru['v'][:, 1:]) ** 2, axis=1)
        J = 0.5 * float(np.mean(e1 ** 2)) + 0.5 * float(np.mean(ew))
        if return_parts:
            return {'J': J, 'rmse_1s': float(np.sqrt(np.mean(e1 ** 2))), 'rmse_win': float(np.sqrt(np.mean(ew))),
                    'bias_1s_median': float(np.median(e1)), 'e1': e1, 'ew': ew}
        return J + pen
    return f


def i6_fixed():
    c5 = jload(OUTD / 'i5_coast.json'); b7 = jload(OUTD / 'i7_brake.json')
    return {'c0': c5['c0'], 'k_s': c5['k_s'], 's_b': b7['nominal']['s_b'], 'db_b': b7['nominal']['db_b']}


def run_i6_fit(op, variant_name, variant, x0=None, max_iter=2500, weights=None, tag=None):
    """1 つの当てはめ(結果は ident/i6/<名前>.json。あれば読むだけ)"""
    d = OUTD / 'i6'
    d.mkdir(exist_ok=True)
    name = tag or ('%s_%s' % (OPS[op], variant_name))
    p = d / (name + '.json')
    if p.exists():
        return jload(p)
    S, W, logs = load_cache()
    gr = jload(OUTD / 'i3_grip.json')
    surf_mu = {k: v['mu'] for k, v in gr['surfaces'].items()}
    Wx = sub_windows(W, W['op'] == op)
    if weights is not None:
        idx = np.repeat(np.arange(Wx['n']), weights)
        Wx = sub_windows(Wx, idx) if False else {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (Wx['n'],) else v) for k, v in Wx.items()}
        Wx['n'] = len(idx)
    tru = cm.truth(Wx)
    mu = mu_windows(Wx, surf_mu)
    fixed = i6_fixed()
    f00 = i6_objective(Wx, tru, mu, fixed, variant)
    if variant.get('prior') is not None:
        pu, pv, pw = (np.asarray(a_, float) for a_ in variant['prior'])

        def f0(x, return_parts=False):
            if return_parts:
                return f00(x, return_parts=True)
            p_ = i6_prm(x, fixed, variant)
            return f00(x) + STEADY_LAMBDA * float(np.sum(pw * (cm.speed_target(pu, p_) - pv) ** 2))
    else:
        f0 = f00
    to_full = variant.get('to_full')                 # 縮めたパラメーター → 13 個(AI の『人の対応 × 倍率』など)
    f = (lambda z, return_parts=False: f0(to_full(z), return_parts=return_parts)) if to_full else f0
    starts = [I6_X0] if x0 is None else ([np.asarray(x0, float)] if np.ndim(x0) == 1 else [np.asarray(z, float) for z in x0])
    step = variant.get('step', I6_STEP)
    t0 = time.time()
    best = None
    it = 0
    for z0 in starts:
        x, fx, n_it = cm.nm_restarts(f, z0, step, max_iter=max_iter, max_restarts=6 if weights is None else 1)
        it += n_it
        if best is None or fx < best[1]:
            best = (x, fx)
    x = best[0]
    it2 = 0
    zx = x
    x = to_full(zx) if to_full else x
    f = f0
    parts_fast = f(x, return_parts=True)
    parts = i6_objective(Wx, tru, mu, fixed, variant, exact=True)(x, return_parts=True) if weights is None else parts_fast
    out = {'op': OPS[op], 'variant': variant_name, 'x': np.clip(x, I6_LO, I6_HI).tolist(), 'names': I6_NAMES,
           'x_reduced': (np.asarray(zx).tolist() if to_full else None), 'n_starts': len(starts),
           'prm': i6_prm(x, fixed, variant), 'J': parts['J'], 'rmse_1s': parts['rmse_1s'], 'rmse_win': parts['rmse_win'],
           'bias_1s_median': parts['bias_1s_median'], 'J_fast_5ms': parts_fast['J'], 'exact': weights is None,
           'iters': it + it2, 'seconds': time.time() - t0, 'n_windows': Wx['n']}
    out['prm'] = {k: v for k, v in out['prm'].items()}
    jdump(p, out)
    log('I6', name, 'J', round(parts['J'], 5), 'rmse_1s', round(parts['rmse_1s'], 4), 'T', np.round(out['prm']['T'], 2),
        'u_db/tau_up/a_cap/tau_down/d_v', np.round(out['x'][8:], 3), '%.0fs' % out['seconds'])
    return out


def eval_i6(op, prm, use_kb=True):
    """与えたパラメーターで、その操作者の当てはめの窓の目的関数(速さだけの再生)"""
    S, W, logs = load_cache()
    gr = jload(OUTD / 'i3_grip.json')
    surf_mu = {k: v['mu'] for k, v in gr['surfaces'].items()}
    Wx = sub_windows(W, W['op'] == op)
    tru = cm.truth(Wx)
    mu = mu_windows(Wx, surf_mu)
    kb = Wx['k_batt'] if use_kb else np.ones(Wx['n'])
    rec = cm.replay(Wx, 'cal', prm, mode='lon', mu=mu, k_batt=kb, record=('v',))
    e1 = rec['v'][:, -1] - tru['v'][:, -1]
    ew = np.mean((rec['v'][:, 1:] - tru['v'][:, 1:]) ** 2, axis=1)
    return {'J': 0.5 * float(np.mean(e1 ** 2)) + 0.5 * float(np.mean(ew)), 'rmse_1s': float(np.sqrt(np.mean(e1 ** 2))),
            'rmse_win': float(np.sqrt(np.mean(ew))), 'bias_1s_median': float(np.median(e1))}


I6_VARIANTS = {'full': {}, 'no_d_v': {'no_d_v': True}, 'no_tau_down': {'no_tau_down': True},
               'no_saturation': {'no_saturation': True}, 'no_k_s': {'no_k_s': True}}
N_BOOT_I6 = 20


def step_i6_op(op):
    """1 人の操作者の I6: 複数の始点の当てはめ・部品の選別(入れ子)・AI は『人の対応 × 倍率』も・シーンのブートストラップ"""
    try1 = OUTD / 'i6_try1'
    starts = [I6_X0]
    for nm_ in ('%s_full' % OPS[op], '%s_no_d_v' % OPS[op], '%s_no_k_s' % OPS[op]):
        pth = try1 / (nm_ + '.json')
        if pth.exists():
            xx = np.array(jload(pth)['x'], float)
            xx[11] = max(xx[11], 0.06)
            starts.append(xx)
    full = run_i6_fit(op, 'full', {}, x0=starts)
    # 部品の選別: 1 つずつ落として当て直す。落とした方が良ければ、そこから full を当て直す(入れ子なので full ≤ 落とした版)
    for rnd in range(2):
        better = None
        for vn, vv in I6_VARIANTS.items():
            if vn == 'full':
                continue
            r = run_i6_fit(op, vn, vv, x0=full['x'], tag='%s_%s_r%d' % (OPS[op], vn, rnd))
            if vn != 'no_k_s' and r['J_fast_5ms'] < full['J_fast_5ms'] - 1e-9:
                xx = np.array(r['x'], float)
                if vn == 'no_tau_down':
                    xx[11] = 0.05
                if better is None or r['J_fast_5ms'] < better[1]:
                    better = (xx, r['J_fast_5ms'])
        if better is None:
            break
        log('I6', OPS[op], '落とした版の方が良かったので full を当て直す', round(better[1], 5))
        full = run_i6_fit(op, 'full', {}, x0=[np.array(full['x']), better[0]], tag='%s_full_r%d' % (OPS[op], rnd + 1))
    jdump(OUTD / 'i6' / ('%s_final_full.json' % OPS[op]), full)
    log('I6 done', OPS[op])


def scaled_variant(Th, flags=None):
    Th = np.asarray(Th, float)
    v = dict(flags or {})
    v['to_full'] = lambda z: np.concatenate([z[0] * Th, z[1:]])
    v['step'] = np.array([0.1, 0.05, 0.1, 0.8, 0.1, 0.02])
    return v


def step_i6_ai_scaled(Th, x_dyn0):
    """AI の対応 = 人の対応 × 倍率 s(deviations D7。形は人、高さは AI。動きの 5 つ u_db・τ_up・a_cap・τ_down・d_v は AI の窓で当てはめ)"""
    z0 = np.concatenate([[1.1], np.asarray(x_dyn0, float)])
    fits = {}
    for vn, vv in I6_VARIANTS.items():
        fits[vn] = run_i6_fit(1, 'scaled_' + vn, scaled_variant(Th, vv),
                              x0=[z0] if vn == 'full' else [np.array(fits['full']['x_reduced'])], tag='ai_scaled_%s' % vn)
    return fits


def step_i6_boot(op, variant_name, variant, x_full, tag_prefix):
    S, W, logs = load_cache()
    Wx = sub_windows(W, W['op'] == op)
    sc = np.array(['%s/%d' % (a, b) for a, b in zip(Wx['log'], Wx['scene'])])
    uniq = np.unique(sc)
    rng = np.random.default_rng(SEED + 60 + op)
    for bi in range(N_BOOT_I6):
        pick = rng.choice(uniq, len(uniq), replace=True)
        cnt = dict(zip(*np.unique(pick, return_counts=True)))
        w = np.array([cnt.get(v, 0) for v in sc], int)
        run_i6_fit(op, 'boot', variant, x0=[x_full], max_iter=500, weights=w, tag='%s_boot%02d' % (tag_prefix, bi))


def step_pcfg(force=False):
    out_p = OUTD / 'p_cfg_fit.json'
    if out_p.exists() and not force:
        return jload(out_p)
    S, W, logs = load_cache()
    tru = cm.truth(W)
    mu_sim = np.array([SIM_MU[s] for s in W['surface']])
    res = {'method': 'protocol model.P_cfg: sim-core.js の式のまま、設定欄の値だけを当てはめ。人と AI の窓をまとめる。'
                     '横(steerRate・maxSteer・frictionMultiplier)は横だけの再生のヨーの RMSE、前後(driveTau・driveAccelerationCap・'
                     'brakeDeceleration)は速さだけの再生の J(I6 と同じ)。遅れ 0/1/2 フレームごとに当てはめ、正規化した和で選ぶ',
           'by_delay': {}}

    def lat_f(n):
        def f(x):
            xc = np.clip(x, PCFG_LAT_LO, PCFG_LAT_HI)
            cfg = dict(SIM_DEFAULT_CFG, steerRate=float(xc[0]), maxSteer=float(xc[1]), frictionMultiplier=float(xc[2]), control_delay_frames=n)
            rec = cm.replay(W, 'sim', cfg, mode='lat', mu=mu_sim * xc[2], record=('psi',))
            return float(np.mean(cm.errors(rec, tru)['e_r'] ** 2)) + 10 * float(np.sum((x - xc) ** 2))
        return f

    def lon_f(n, fm):
        def f(x, parts=False):
            xc = np.clip(x, PCFG_LON_LO, PCFG_LON_HI)
            cfg = dict(SIM_DEFAULT_CFG, driveTau=float(xc[0]), driveAccelerationCap=float(xc[1]), brakeDeceleration=float(xc[2]),
                       frictionMultiplier=fm, control_delay_frames=n)
            rec = cm.replay(W, 'sim', cfg, mode='lon', mu=mu_sim * fm, record=('v',))
            e1 = rec['v'][:, -1] - tru['v'][:, -1]
            ew = np.mean((rec['v'][:, 1:] - tru['v'][:, 1:]) ** 2, axis=1)
            J = 0.5 * float(np.mean(e1 ** 2)) + 0.5 * float(np.mean(ew))
            if parts:
                return {'J': J, 'rmse_1s': float(np.sqrt(np.mean(e1 ** 2))), 'bias_1s_median': float(np.median(e1))}
            return J + 10 * float(np.sum((x - xc) ** 2))
        return f

    lat0 = lat_f(0)(np.array([1.2, 0.52, 1.0]))
    lon0 = lon_f(0, 1.0)(np.array([0.4, 6.0, 1.2]))
    res['default'] = {'yaw_mse_lat': lat0, 'yaw_rmse_lat': math.sqrt(lat0), 'J_lon': lon0}
    best = None
    for n in (0, 1, 2):
        xl, fl, _ = cm.nelder_mead(lat_f(n), np.array([4.0, 0.45, 0.9]), np.array([2.0, 0.05, 0.15]), max_iter=300, tol=1e-6)
        xl = np.clip(xl, PCFG_LAT_LO, PCFG_LAT_HI)
        fo = lon_f(n, float(xl[2]))
        xv, fv, _ = cm.nelder_mead(fo, np.array([0.6, 4.0, 0.8]), np.array([0.2, 1.5, 0.3]), max_iter=300, tol=1e-6)
        xv = np.clip(xv, PCFG_LON_LO, PCFG_LON_HI)
        pv = fo(xv, parts=True)
        score = fl / lat0 + pv['J'] / lon0
        res['by_delay'][str(n)] = {'steerRate': float(xl[0]), 'maxSteer': float(xl[1]), 'frictionMultiplier': float(xl[2]),
                                   'yaw_rmse_lat': math.sqrt(fl), 'driveTau': float(xv[0]), 'driveAccelerationCap': float(xv[1]),
                                   'brakeDeceleration': float(xv[2]), 'J_lon': pv['J'], 'rmse_1s': pv['rmse_1s'],
                                   'bias_1s_median': pv['bias_1s_median'], 'score': score}
        log('P_cfg', n, res['by_delay'][str(n)])
        if best is None or score < best[1]:
            best = (n, score)
    res['chosen_delay_frames'] = best[0]
    res['chosen'] = res['by_delay'][str(best[0])]
    jdump(out_p, res)
    return res



# =================================================================== I8・I9・選別・固める
def step_i8():
    """AI_CMD.latency_ms(撮像 → PCA9685 に書いた時刻)の分布。当てはめのシーンだけ"""
    import label_events as le
    rows = []
    allv = []
    for lg in cd.list_logs():
        L = le.load_log(lg)
        A = L['AI_CMD']
        if A is None or not len(A['timestamps_us']):
            continue
        sc = A['scene'].astype(int)
        names = L['scenes']
        keep = np.array([(k % 3 != 1) and names[k] not in cd.EXCLUDED_SCENES for k in sc])
        lat = A['latency_ms'].astype(float)
        ok = keep & np.isfinite(lat) & ~A['dry_run'].astype(bool)
        if ok.sum() < 30:
            continue
        x = lat[ok]
        allv.append(x)
        rows.append({'log': lg, 'n': int(ok.sum()), 'p50': float(np.percentile(x, 50)), 'p90': float(np.percentile(x, 90)),
                     'p99': float(np.percentile(x, 99)), 'max': float(x.max()),
                     'cmd_interval_ms_p50': float(np.median(np.diff(A['timestamps_us'][ok]) / 1e3))})
    allv = np.concatenate(allv)
    return {'method': 'protocol I8: AI_CMD.latency_ms(当てはめのシーン、dry_run を除く)', 'by_log': rows,
            'pooled': {'n': int(len(allv)), 'p50': float(np.percentile(allv, 50)), 'p90': float(np.percentile(allv, 90)),
                       'p99': float(np.percentile(allv, 99))},
            'p50_range_logs': [min(r['p50'] for r in rows), max(r['p50'] for r in rows)],
            'p99_range_logs': [min(r['p99'] for r in rows), max(r['p99'] for r in rows)],
            'decision': 'アプリの制御の遅れは 1 フレーム(33 ms)のまま。PWM → 車の応答の遅れは d_s・d_v に入れる。p99 の揺れは 1 フレームに含め、別に足さない'}


def step_i9(S):
    rows = []
    for li in np.unique(S['log_i']):
        for k in np.unique(S['scene'][S['log_i'] == li]):
            m = (S['log_i'] == li) & (S['scene'] == k) & np.isfinite(S['g'])
            if m.sum() < 30:
                continue
            ops = S['op'][m]
            rows.append({'log_i': int(li), 'scene': int(k), 'n': int(m.sum()), 'op': OPS[int(round(ops.mean()))],
                         'g_median': float(np.median(S['g'][m])), 'v_median': float(np.median(S['vs'][m]))})
    return rows


def i6_load(op, name):
    p = OUTD / 'i6' / ('%s_%s.json' % (OPS[op], name))
    return jload(p) if p.exists() else None


def i6_boot_summary(op, prefix=None):
    bs = [jload(p) for p in sorted((OUTD / 'i6').glob('%s_boot*.json' % (prefix or OPS[op])))]
    if not bs:
        return None
    X = np.array([b['x'] for b in bs])
    T = np.array([b['prm']['T'] for b in bs])
    return {'n_boot': len(bs), 'p5': np.percentile(X, 5, axis=0).tolist(), 'p95': np.percentile(X, 95, axis=0).tolist(),
            'T_p5': np.percentile(T, 5, axis=0).tolist(), 'T_p95': np.percentile(T, 95, axis=0).tolist(), 'names': I6_NAMES}


def sha256_text(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()



def steer6(vals):
    v = np.asarray(vals, float)
    return (v[5:] + 0.0).tolist(), (-v[5::-1] + 0.0).tolist()     # + 0.0 で −0.0 を 0.0 に


def step_final(force=False):
    """選別・人と AI の決め・プロファイル(P_cal・P_cfg・profile_real_v1)を書いて sha256 を記録"""
    prof_dir = REPORT / 'profiles'
    prof_dir.mkdir(exist_ok=True)
    if (prof_dir / 'SHA256SUMS').exists() and not force:
        log('プロファイルは固めてある(SHA256SUMS あり)。--force で作り直す')
        return
    st = jload(OUTD / 'i1_i2_steering.json'); gr = jload(OUTD / 'i3_grip.json'); d4 = jload(OUTD / 'i4_steer_dynamics.json')
    c5 = jload(OUTD / 'i5_coast.json'); b7 = jload(OUTD / 'i7_brake.json'); pc = jload(OUTD / 'p_cfg_fit.json')
    dmeta = jload(OUTD / 'data.json')
    S, W, logs = load_cache()
    i8 = step_i8()
    i9 = step_i9(S)
    i6d = OUTD / 'i6'

    def latest(prefix):
        c = sorted(i6d.glob(prefix + '_r*.json'))
        return jload(c[-1]) if c else (jload(i6d / (prefix + '.json')) if (i6d / (prefix + '.json')).exists() else None)

    def selection(base, getv):
        out = {}
        for vn in ('no_d_v', 'no_tau_down', 'no_saturation', 'no_k_s'):
            v = getv(vn)
            worsen = v['J'] / base['J'] - 1
            out[vn[3:]] = {'J_full': base['J'], 'J_without': v['J'], 'worse_without': worsen, 'keep': bool(worsen >= 0.03)}
        return out
    full = {0: jload(i6d / 'human_final_full.json'), 1: jload(i6d / 'ai_final_full.json')}
    sel = {'human': selection(full[0], lambda vn: latest('human_' + vn)), 'ai_free_reference': selection(full[1], lambda vn: latest('ai_' + vn))}
    # ---- 人の前後: 部品の選別 → (2 つ以上落とすなら当て直し)→ 定常の速さの確かめ(D10)
    drop_h = [k for k, v in sel['human'].items() if not v['keep']]
    if len(drop_h) == 0:
        lon_h = full[0]
    elif len(drop_h) == 1:
        lon_h = latest('human_no_' + drop_h[0])
    else:
        lon_h = run_i6_fit(0, 'combined', {('no_' + k): True for k in drop_h}, x0=[np.array(full[0]['x'])],
                           tag='human_final_no_%s' % '_no_'.join(drop_h))
    srows = steady_table(S)
    chk_h, worst_h = steady_check(lon_h['prm'], srows)
    steady = {'table_fit_scenes_human': srows, 'check_before': chk_h, 'worst_rel_before': worst_h, 'tol': STEADY_TOL, 'applied_prior': False}
    if worst_h > STEADY_TOL:
        J_before = lon_h['J']
        lon_h = run_i6_fit_prior(0, lon_h, {('no_' + k): True for k in drop_h}, srows, tag='human_final_prior')
        chk2, worst2 = steady_check(lon_h['prm'], srows)
        steady.update(applied_prior=True, check_after=chk2, worst_rel_after=worst2, J_before=J_before, J_after=lon_h['J'], T_before=chk_h)
    # ---- AI: 人の形 × 倍率(D7)。選別も同じ決まり
    Th_final = np.array(lon_h['prm']['T'], float)
    ai_sc = step_i6_ai_scaled(Th_final, np.array(lon_h['x'][8:], float))
    sel['ai_scaled'] = selection(ai_sc['full'], lambda vn: ai_sc[vn])
    drop_a = [k for k, v in sel['ai_scaled'].items() if not v['keep']]
    if len(drop_a) == 0:
        lon_a = ai_sc['full']
    elif len(drop_a) == 1:
        lon_a = ai_sc['no_' + drop_a[0]]
    else:
        lon_a = run_i6_fit(1, 'combined', scaled_variant(Th_final, {('no_' + k): True for k in drop_a}),
                           x0=[np.array(ai_sc['full']['x_reduced'])], tag='ai_scaled_final_no_%s' % '_no_'.join(drop_a))
    lon_a_scaled_ref = lon_a
    # ---- AI の対応 = 人の T(a u + b)(D7、PWM の経路の違い)。動き(τ_up・a_cap・こすれ・遅れ)は人の当てはめと同じ(同じ車)
    flags_h = {('no_' + k): True for k in drop_h}
    lon_a = run_i6_fit(1, 'ai_input_map', ai_input_variant(lon_h['x'], flags_h),
                       x0=[np.array([1.0, 0.0]), np.array([1.2, 0.2]), np.array([1.0, 0.5])], tag='ai_input_map')
    drop_a = list(drop_h)
    # ---- 人と AI(protocol I6 の 10% の決まり)。それぞれ部品の選別を済ませた形で比べる(deviations D11)
    Th = float(cm.speed_target(2.5, lon_h['prm']))
    Ta = float(cm.speed_target(2.5, lon_a['prm']))
    Ta_full = float(cm.speed_target(2.5, ai_sc['full']['prm']))
    Ta_free = float(cm.speed_target(2.5, full[1]['prm']))
    speed_diff = (Ta - Th) / Th
    steer_diff = max(abs(st['human_vs_ai'][k]['rel_diff']) for k in ('u_s=+0.5rad', 'u_s=-0.5rad'))
    switch = bool(abs(speed_diff) >= 0.10 or steer_diff >= 0.10)
    j_h_on_ai = eval_i6(1, lon_h['prm'])['J']
    imp_speed = 1 - lon_a['J'] / j_h_on_ai
    imp_steer = d4['ai_windows_with_human_map']['improve_ai_map']
    rule_speed_op = 1 if (switch and imp_speed >= 0.03) else 0
    rule_steer_op = 1 if (switch and imp_steer >= 0.03) else 0
    # deviations D11: 主のプロファイルは人の対応(速さ・舵)にする。AI だけの当てはめは、3 つの形のどれでも低いスロットルで物理に合わない
    # (中立に近い指令で 2.3 m/s など)うえ、10% の比べの結果が形で変わるため。AI の『熱い』スロットルは k_batt の相当値として渡す
    speed_op = 0
    steer_op = 0
    # AI の窓での k_batt の相当値(人のプロファイルのまま、AI の窓の全部に共通の係数を掛けて J が最小になる値)
    Wa_k = sub_windows(W, W['op'] == 1)
    tru_k = cm.truth(Wa_k)
    mu_k = mu_windows(Wa_k, {k_: v_['mu'] for k_, v_ in gr['surfaces'].items()})
    kgrid = np.round(np.arange(0.8, 1.61, 0.02), 2)
    kJ = []
    for kk_ in kgrid:
        vv_ = cm.lon_fast(Wa_k, lon_h['prm'], mu_k, Wa_k['k_batt'] * kk_, h=0.005)
        kJ.append(0.5 * float(np.mean((vv_[:, -1] - tru_k['v'][:, -1]) ** 2)) + 0.5 * float(np.mean((vv_[:, 1:] - tru_k['v'][:, 1:]) ** 2)))
    k_ai_eq = float(kgrid[int(np.argmin(kJ))])
    decision = {'speed_at_u2.5': {'human': Th, 'ai_input_map': Ta, 'ai_input_map_a_b': lon_a['x_reduced'],
                                  'ai_scaled_selected_reference': float(cm.speed_target(2.5, lon_a_scaled_ref['prm'])),
                                  'ai_scaled_before_selection_reference': Ta_full, 'ai_free_reference': Ta_free, 'rel_diff': speed_diff,
                                  'ai_scaled_s_reference': lon_a_scaled_ref['x_reduced'][0]},
                'curvature_at_u_s_0.5rad_1.5mps': {k: st['human_vs_ai'][k] for k in ('u_s=+0.5rad', 'u_s=-0.5rad')},
                'curvature_at_n_0.5_reference': {k: st['human_vs_ai'][k] for k in ('n=+0.5', 'n=-0.5')},
                'switch_10pct': switch, 'J_ai_windows': {'human_profile': j_h_on_ai, 'ai_input_map': lon_a['J'],
                                                         'ai_scaled_selected_reference': lon_a_scaled_ref['J'],
                                                         'ai_scaled_before_selection_reference': ai_sc['full']['J'], 'ai_free_reference': full[1]['J']},
                'ai_map_improve_speed_J_on_ai_windows': imp_speed,
                'ai_map_improve_steer_yaw_on_ai_windows': imp_steer,
                'protocol_rule_outcome': {'speed_map': 'ai_input_map' if rule_speed_op else 'human', 'steer_map': OPS[rule_steer_op]},
                'k_batt_ai_equivalent': k_ai_eq, 'k_batt_ai_equivalent_J': {'k=1': kJ[int(np.argmin(np.abs(kgrid - 1.0)))], 'best': min(kJ)},
                'speed_map_from': 'ai_input_map' if speed_op == 1 else 'human', 'steer_map_from': OPS[steer_op]}
    skey = 'ai_input_map' if speed_op == 1 else 'human'
    if speed_op == 0:
        lon, drop = lon_h, drop_h
    else:
        lon, drop = lon_a, drop_a
    lon_prm = lon['prm']
    d_v = lon_prm['d_v'] + (0.008 if (speed_op == 0 and lon_prm['d_v'] > 0) else 0.0)
    # 不確かさ: 決めた形(落とした部品も同じ)で、シーン単位のブートストラップ N_BOOT_I6 回
    bvar = {('no_' + k): True for k in drop_h}
    if steady['applied_prior']:
        bvar['prior'] = ([r_['u'] for r_ in srows], [r_['v_median'] for r_ in srows], [min(r_['n'], 500) / 500 for r_ in srows])
    step_i6_boot(0, 'boot', bvar, np.array(lon_h['x']), 'human')
    if speed_op == 1:
        Wa_ = sub_windows(W, W['op'] == 1)
        sca = np.array(['%s/%d' % (a_, b_) for a_, b_ in zip(Wa_['log'], Wa_['scene'])])
        uq = np.unique(sca)
        rng_b = np.random.default_rng(SEED + 71)
        for bi in range(N_BOOT_I6):
            hb = jload(OUTD / 'i6' / ('human_boot%02d.json' % bi))
            pick = rng_b.choice(uq, len(uq), replace=True)
            cnt = dict(zip(*np.unique(pick, return_counts=True)))
            w = np.array([cnt.get(v_, 0) for v_ in sca], int)
            run_i6_fit(1, 'boot', ai_input_variant(hb['x'], flags_h), x0=[np.array(lon['x_reduced'])], max_iter=300, weights=w,
                       tag='ai_input_map_boot%02d' % bi)
    boot = i6_boot_summary(speed_op, skey)
    # ---- 舵
    f_st = st[OPS[steer_op]]
    K_keep = steer_op == 1 and d4['ai']['K_improve'] >= 0.03 or steer_op == 0 and d4['human']['K_improve'] >= 0.03
    K_us = f_st['K_us_I2'] if K_keep else 0.0
    use_ai_dyn = d4['ai']['n_windows'] >= 100
    d_s = d4['ai']['d_s'] if use_ai_dyn else d4['human']['d_s'] + 0.008
    r_s = d4['ai']['r_s'] if use_ai_dyn else d4['human']['r_s']
    L6, R6 = steer6(f_st['delta_rad'])
    surf = {k: v['mu'] for k, v in gr['surfaces'].items()}
    name = 'jf-calib-v1-real-20261010'
    P_cal = {
        'schema': 'jf-calib-profile/v1', 'name': name, 'model': 'jf-calib-v1',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S+09:00'), 'protocol_sha256': PROTO_SHA,
        'source': '実車のログ 2026-10-10(当てはめのシーンだけ。各ログのシーンの番号 k % 3 != 1)。ident.py',
        'inputs_units': 'u_s [rad](±0.52 = PWM の端)・u_v [m/s](負 = ブレーキ/後退の要求)。ブリッジと RL の先生の単位のまま',
        'steering': {'map_from': OPS[steer_op], 'knots_n': cm.STEER_KNOTS.tolist(), 'delta_L_rad': L6, 'delta_R_rad': R6,
                     'note_knot0': 'n = 0 の値は左右で共通(delta_L_rad[0])。delta_R_rad[0] は使わない',
                     'd_s': d_s, 'r_s': r_s, 'K_us': K_us, 'wheelbase': L_WB},
        'surfaces_mu': surf, 'default_surface_mu': surf['carpet'],
        'longitudinal': {'map_from': skey, 'u_db': lon_prm['u_db'], 'T_knots_u': cm.SPEED_KNOTS.tolist(), 'T_values': lon_prm['T'],
                         'tau_up': lon_prm['tau_up'], 'a_cap': lon_prm['a_cap'], 'tau_down': lon_prm['tau_down'],
                         'c0': lon_prm['c0'], 'k_s': lon_prm['k_s'], 'd_v': d_v,
                         'brake': {'s_b': b7['nominal']['s_b'], 'db_b': b7['nominal']['db_b'], 'us_per_mps': cm.BRAKE_US_PER_MPS},
                         'brake_scenarios': {k: {'s_b': v['s_b'], 'db_b': v['db_b'], 'a_b_at_200us': v['a_b_at_200us']} for k, v in b7['scenarios'].items()},
                         'brake_literal_protocol': {k: b7['protocol_literal'][k] for k in ('s_b', 'db_b', 'a_b_at_200us')},
                         'esc_reverse': {'neutral_s': 0.12, 'target_mps': -0.8, 'accel': 1.5, 'stopped_mps': 0.05},
                         'k_batt': 1.0, 'k_batt_sensitivity': 0.8, 'k_batt_ai_equivalent': k_ai_eq},
        'longitudinal_alternatives_reference': {
            'note': '主のプロファイルではない。感度・参考用(deviations D7・D11)。J = 当てはめの AI の窓の速さの目的関数',
            'human_map': {'T_values': lon_h['prm']['T'], 'u_db': lon_h['prm']['u_db'], 'J_ai_windows': j_h_on_ai},
            'ai_input_map': {'T_values': lon_a['prm']['T'], 'u_db': lon_a['prm']['u_db'], 'a_b': lon_a['x_reduced'], 'J_ai_windows': lon_a['J']},
            'ai_only_fit_scaled': {k_: lon_a_scaled_ref['prm'][k_] for k_ in ('T', 'u_db', 'tau_up', 'a_cap', 'tau_down', 'c0', 'k_s', 'd_v')},
            'ai_only_fit_scaled_J_ai_windows': lon_a_scaled_ref['J']},
        'app_control_delay_frames': 1,
        'semantics': {
            'substep': 'sim-core.js と同じ h = min(substep, 0.012/maximumMotionRate, 残り)。config は元のまま',
            'delay': '小区間の始まりの時刻 τ(= sim_time + elapsed)に、時刻 τ − d を含むフレームの指令(floor((τ−d)/dt + 1e-9))。履歴が無ければ最初の指令',
            'steer': 'δ_t = 区分線形(n = u_s/0.52)。δ ← approach(δ, δ_t, r_s h)(r_s = null は無限 = そのまま)。'
                     'yawRate = clamp(v_next tan(δ)/L/(1 + K_us v_next²), ±μ g/max(|v_next|, .01))',
            'speed': 'a_lat = |v × 前の小区間の yawRate|。u_v ≥ 0: v_tgt = k_batt T(u_v)。'
                     'v_tgt > v ≥ 0: a = min(a_cap, μ g, (v_tgt − v)/τ_up) − k_s a_lat、a > 0 なら v_tgt で止める。'
                     '0 ≤ v_tgt ≤ v: a = −min(c0, (v − v_tgt)/τ_down) − k_s a_lat(τ_down = 0 なら v > v_tgt で c0)。'
                     'v < 0 で前進側: 1.5 m/s² で 0 へ。u_v < 0 かつ v > 0.05: a = −(c0 + k_s a_lat + s_b max(0, −u_v × 627.5 − db_b))。'
                     '止まって(v ≤ 0.05)ブレーキ側: 後退できる状態でなければ 0 へ減速して止まる。止まってブレーキ側を受けたあと、'
                     '前進側の指令(u_v ≥ 0)が 0.12 秒以上続くと後退できる状態になり、次のブレーキ側で −0.8 m/s へ 1.5 m/s²。v > 0.05 で状態を戻す。'
                     'v は 0 未満にしない(後退のときを除く)',
            'reference_impl': 'experiments/20261016_sim_calibration/calib_model.py の replay(physics="cal")'},
    }
    W_all = W
    tru_all = cm.truth(W_all)
    # ---- deviations D12: 舵の中心(n = 0)の値。I1 の値(低速の中心の曲率 +0.05 /m 前後)か 0(当て直し)か。結合した再生の向きの二乗誤差で選ぶ
    vals_i1 = np.array(f_st['delta_rad'], float)
    vals_0 = refit_steer_zero_center(f_st['bins_low_speed'])
    c_sel = {'rows': []}
    for name_, vv_ in (('I1', vals_i1), ('zero_center', vals_0)):
        P_try = json.loads(json.dumps(P_cal, default=_np_default))
        P_try['steering']['delta_L_rad'], P_try['steering']['delta_R_rad'] = steer6(vv_)
        mu_try = np.array([surf[x] for x in W_all['surface']])
        rec = cm.replay(W_all, 'cal', cal_prm_from_profile(P_try), 'coupled', mu=mu_try, k_batt=W_all['k_batt'])
        e = cm.errors(rec, tru_all)
        c_sel['rows'].append({'name': name_, 'delta_rad': vv_.tolist(), 'e_psi_rmse': float(np.sqrt(np.mean(e['e_psi'] ** 2))),
                              'e_psi_median': float(np.median(e['e_psi'])), 'e_r_rms': float(np.sqrt(np.mean(e['e_r'] ** 2))),
                              'e_p_rms': float(np.sqrt(np.mean(e['e_p'] ** 2))),
                              'e_psi_median_DE': float(np.median(e['e_psi'][W_all['sec'] == 'DE']))})
    best_c = min(c_sel['rows'], key=lambda r_: r_['e_psi_rmse'])
    c_sel['chosen'] = best_c['name']
    P_cal['steering']['delta_L_rad'], P_cal['steering']['delta_R_rad'] = steer6(best_c['delta_rad'])
    P_cal['steering']['center_choice'] = best_c['name']
    log('D12 舵の中心', [(r_['name'], round(r_['e_psi_rmse'], 2), round(r_['e_psi_median'], 2)) for r_ in c_sel['rows']], '→', best_c['name'])
    # ---- deviations D9: 路面の μ に共通の倍率 s を掛けるか(当てはめの窓の結合した再生のヨーの誤差。3% 以上良くなるときだけ)
    mu_sel = {'grid': MU_SCALE_GRID, 'rows': []}
    for sc_ in MU_SCALE_GRID:
        P_try = json.loads(json.dumps(P_cal, default=_np_default))
        mu_try = np.array([surf[x] * sc_ for x in W_all['surface']])
        rec = cm.replay(W_all, 'cal', cal_prm_from_profile(P_try), 'coupled', mu=mu_try, k_batt=W_all['k_batt'])
        e = cm.errors(rec, tru_all)
        mu_sel['rows'].append({'scale': sc_, 'e_r_rms': float(np.sqrt(np.mean(e['e_r'] ** 2))), 'e_psi_rmse': float(np.sqrt(np.mean(e['e_psi'] ** 2))),
                               'e_v_rmse': float(np.sqrt(np.mean(e['e_v'] ** 2))), 'e_p_rms': float(np.sqrt(np.mean(e['e_p'] ** 2))),
                               'e_r_rms_human': float(np.sqrt(np.mean(e['e_r'][W_all['op'] == 0] ** 2))),
                               'e_r_rms_ai': float(np.sqrt(np.mean(e['e_r'][W_all['op'] == 1] ** 2)))})
    base_r = mu_sel['rows'][0]['e_r_rms'] ** 2
    best_row = min(mu_sel['rows'], key=lambda r_: r_['e_r_rms'])
    mu_sel['improve_e_r_mse_best'] = 1 - best_row['e_r_rms'] ** 2 / base_r
    mu_scale = best_row['scale'] if mu_sel['improve_e_r_mse_best'] >= 0.03 else 1.0
    mu_sel['chosen'] = mu_scale
    P_cal['surfaces_mu_I3'] = dict(surf)
    P_cal['mu_scale'] = mu_scale
    P_cal['surfaces_mu'] = {k: v * mu_scale for k, v in surf.items()}
    P_cal['default_surface_mu'] = P_cal['surfaces_mu']['carpet']
    log('D9 μ の倍率', [(r_['scale'], round(r_['e_r_rms'], 4), round(r_['e_psi_rmse'], 2)) for r_ in mu_sel['rows']], '→', mu_scale)
    cfg_ch = pc['chosen']
    P_cfg = {
        'schema': 'jf-calib-profile-cfg/v1', 'name': 'jf-cfg-v1-real-20261010', 'created_at': P_cal['created_at'], 'protocol_sha256': PROTO_SHA,
        'sim_config': {'longitudinalModel': 'real-drive-response-estimate-v1', 'driveTau': cfg_ch['driveTau'],
                       'driveAccelerationCap': cfg_ch['driveAccelerationCap'], 'brakeDeceleration': cfg_ch['brakeDeceleration'],
                       'steerRate': cfg_ch['steerRate'], 'maxSteer': cfg_ch['maxSteer'], 'frictionMultiplier': cfg_ch['frictionMultiplier']},
        'actuator_delay_frames': pc['chosen_delay_frames'],
        'app_control_delay_frames_recommended': min(2, 1 + pc['chosen_delay_frames']),
        'note': 'sim-core.js の式のまま、設定欄の値だけ。actuator_delay_frames は PWM → 車の応答の遅れ(忠実度の再生で使う)。'
                '閉ループのアプリでは、撮像 → PWM の 1 フレームに足して設定する(最大 2)。brakeDeceleration は惰性とブレーキの両方に効く',
    }
    (prof_dir / 'P_cal.json').write_text(json.dumps(P_cal, ensure_ascii=False, indent=1, default=_np_default), encoding='utf-8')
    (prof_dir / 'P_cfg.json').write_text(json.dumps(P_cfg, ensure_ascii=False, indent=1, default=_np_default), encoding='utf-8')
    extra = {'selection': sel, 'decision': decision, 'mu_scale_selection': mu_sel, 'steady_check': steady, 'center_selection': c_sel, 'i8': i8, 'i9_scenes': i9, 'lon_fit_used': lon['variant'] if 'variant' in lon else None,
             'lon_fit_J': lon['J'], 'lon_fit_rmse_1s': lon['rmse_1s'], 'boot': boot, 'K_keep': K_keep, 'drop': drop}
    jdump(OUTD / 'final_decisions.json', extra)
    build_profile_real(P_cal, P_cfg, st, gr, d4, c5, b7, pc, dmeta, extra, full, steer_op, speed_op, skey)
    sums = []
    for fn in ('P_cal.json', 'P_cfg.json'):
        sums.append('%s  %s' % (sha256_text(prof_dir / fn), fn))
    sums.append('%s  %s' % (sha256_text(REPORT / 'profile_real_v1.json'), '../profile_real_v1.json'))
    (prof_dir / 'SHA256SUMS').write_text('\n'.join(sums) + '\n', encoding='utf-8')
    log('固めた', sums)



def build_profile_real(P_cal, P_cfg, st, gr, d4, c5, b7, pc, dmeta, extra, full, steer_op, speed_op, skey):
    """profile_real_v1.json: すべての値に、推定・不確かさ・使ったデータ・シミュレーターの今の値を付けた版(人が読む用)"""
    sto = OPS[steer_op]; spo = OPS[speed_op]
    f_st = st[sto]
    L6, R6 = P_cal['steering']['delta_L_rad'], P_cal['steering']['delta_R_rad']
    L6lo, R6hi = steer6(f_st['delta_p5']); L6hi, R6lo = steer6(f_st['delta_p95'])
    if P_cal['steering'].get('center_choice') == 'zero_center':
        L6lo[0] = L6hi[0] = 0.0
    sim_d = (cm.STEER_KNOTS * cm.MAX_STEER).tolist()
    lon = P_cal['longitudinal']
    boot = extra['boot']
    nm = I6_NAMES

    def bx(name):
        if boot is None:
            return None
        i = nm.index(name)
        return {'p5': boot['p5'][i], 'p95': boot['p95'][i], 'kind': 'シーン単位のブートストラップ %d 回(当て直し)' % boot['n_boot']}

    win = dmeta['n_windows']
    P = []

    def add(key, label, value, unit, unc, data, method, sim, note=None):
        P.append({'key': key, 'label': label, 'value': value, 'unit': unit, 'uncertainty': unc, 'data_used': data, 'method': method,
                  'sim_current': sim, 'note': note})

    n_lo = st[sto]['n_samples_low']
    add('steering.delta_L_rad', '舵の対応・左(正規化の指令 0・0.25・0.5・0.75・0.9・1.0 → 舵角 δ)', L6, 'rad',
        {'p5': L6lo, 'p95': L6hi, 'kind': 'シーン単位のブートストラップ 500 回'},
        '舵が 0.3 秒一定・0.8〜2.2 m/s・GM 以外の時刻 %d(%s の対応。人の左 0.81 より先は AI)' % (n_lo, sto), 'I1(帯の中央値 → 区分線形・単調)',
        sim_d, '曲率にすると κ0 = tan δ/0.26: ' + ', '.join('%.2f' % (math.tan(x) / L_WB) for x in L6) + ' /m(シミュレーター ' +
        ', '.join('%.2f' % (math.tan(x) / L_WB) for x in sim_d) + ')')
    add('steering.delta_R_rad', '舵の対応・右(大きさ)', R6, 'rad', {'p5': R6lo, 'p95': R6hi, 'kind': 'シーン単位のブートストラップ 500 回'},
        '同上', 'I1', sim_d, '曲率 ' + ', '.join('%.2f' % (math.tan(x) / L_WB) for x in R6) + ' /m。n = 0 の値は左(delta_L_rad[0])を使う')
    add('steering.K_us', 'アンダーステア K(κ = κ0/(1 + K v²))', P_cal['steering']['K_us'], 's²/m²',
        {'p5_p50_p95': st[sto]['K_us_boot_p5_p50_p95'], 'kind': 'シーン単位のブートストラップ 500 回'},
        '舵が一定・2.2〜4.0 m/s・要求の横加速度 ≤ 5 m/s² の時刻 %d' % st[sto]['n_samples_high'],
        'I2(1 次元の最小二乗)+ 選別(I4 のヨーの誤差が %.1f%% 減るので %s)' % (100 * d4[sto]['K_improve'], '残す' if extra['K_keep'] else '落とす'), 0.0)
    add('steering.d_s', '舵の遅れ(PWM を書いた時刻 → 舵が動き始める)', P_cal['steering']['d_s'], 's',
        {'p5_p50_p95': d4['ai']['d_s_boot_p5_p50_p95'], 'kind': 'シーン単位のブートストラップ 1000 回(格子の最良点)'},
        'AI の当てはめの窓 %d(人 %d 窓では %.2f 秒)' % (d4['ai']['n_windows'], d4['human']['n_windows'], d4['human']['d_s']),
        'I4(横だけの再生、格子 0〜0.15 秒 × 舵の速さ)', 0.0, 'AI − 人 = %.3f 秒(0〜0.025 なら整合: %s)' % (
            d4['consistency']['ai_minus_human_d_s'], d4['consistency']['consistent_0_to_0.025']))
    add('steering.r_s', '舵の速さ(δ が目標へ近づく速さ)', P_cal['steering']['r_s'], 'rad/s',
        {'boot_choice_counts': d4['ai']['r_s_boot_values'], 'kind': 'ブートストラップで最良になった回数'},
        '同上', 'I4', 1.2, '人の窓では %s rad/s' % d4['human']['r_s'])
    msc = P_cal.get('mu_scale', 1.0)
    for k, v in gr['surfaces'].items():
        add('surfaces_mu.' + k, '横の限界 μ(%s)' % k, v['mu'] * msc, '(μ g = 横加速度の上限)',
            None if v.get('mu_p5') is None else {'p5': v['mu_p5'] * msc, 'p95': v['mu_p95'] * msc,
                                                 'kind': 'シーン単位のブートストラップ 1000 回(I3 の値 × 倍率)'},
            '%s(滑りの時刻 %d)' % (v['from'], v['n']),
            'I3(滑りの時刻の |a_lat| の中央値 / g = %.3f)× 倍率 %.1f(deviations D9: 結合した再生のヨーで選んだ)' % (v['mu'], msc), v['sim_current'])
    T = lon['T_values']
    add('longitudinal.T_values', '速さの対応 T(u_v = 0.5・1・2・3・4・5・6・7.24 m/s の指令 → 目標の速さ)', T, 'm/s',
        None if boot is None else {'p5': boot['T_p5'], 'p95': boot['T_p95'], 'kind': 'シーン単位のブートストラップ %d 回' % boot['n_boot']},
        '人の当てはめの窓 %d(1 秒、0.5 秒ずらし)%s' % (win['human'], '。AI の対応 = 人の T(a u + b)、a = %.3f・b = %.3f を AI の窓 %d で当てはめ(deviations D7)' % (
            extra['decision']['speed_at_u2.5']['ai_input_map_a_b'][0], extra['decision']['speed_at_u2.5']['ai_input_map_a_b'][1], win['ai']) if skey == 'ai_input_map' else ''), 'I6(速さだけの再生、Nelder–Mead、目的 = 1 秒後と窓の中の速さの二乗誤差)',
        cm.SPEED_KNOTS.tolist(), 'スロットル PWM = 1498 − u_v × 68.78。シミュレーターは T(u) = u(頭打ちなし)')
    add('longitudinal.u_db', 'ESC の不感帯(この指令までは 0)', lon['u_db'], 'm/s(指令)', bx('u_db'), '同上', 'I6', 0.0,
        'PWM にすると 中立 − %.0f µs' % (lon['u_db'] * 498 / 7.24))
    add('longitudinal.tau_up', '加速の時定数 τ_up', lon['tau_up'], 's', bx('tau_up'), '同上', 'I6', 0.4, 'シミュレーターの driveTau')
    add('longitudinal.a_cap', '加速の上限 a_cap', lon['a_cap'], 'm/s²', bx('a_cap'), '同上', 'I6', 6.0, 'シミュレーターの driveAccelerationCap(どちらも μ g でも頭打ち)')
    add('longitudinal.tau_down', 'スロットルを戻したときの時定数 τ_down', lon['tau_down'], 's',
        bx('tau_down') if extra['selection']['human']['tau_down']['keep'] else None, '同上',
        'I6 + 選別(%s)' % ('残す' if extra['selection']['human']['tau_down']['keep'] else '落とす = 0'), None,
        'シミュレーターは戻しも一律 1.2 m/s²')
    add('longitudinal.c0', '惰性の減速(直線)', lon['c0'], 'm/s²', {'p5': c5['c0_p5_p95'][0], 'p95': c5['c0_p5_p95'][1], 'kind': 'シーン単位のブートストラップ 2000 回'},
        '中立 ±12 µs を 0.3 秒以上・1 m/s 超・舵 < 100 µs の時刻 %d(%.1f 秒、ほぼ人)' % (c5['n_straight'], c5['straight_s']), 'I5(減速の中央値)', 1.2,
        'シミュレーターの brakeDeceleration(惰性・ブレーキ・後退の共通)')
    add('longitudinal.k_s', '曲がりのこすれ k_s(減速 += k_s |a_lat|)', lon['k_s'], '(m/s²)/(m/s²)',
        {'p5': c5['k_s_p5_p95'][0], 'p95': c5['k_s_p5_p95'][1], 'kind': 'シーン単位のブートストラップ 2000 回'},
        '惰性で舵 > 250 µs の時刻 %d(%.1f 秒)' % (c5['n_turning'], c5['turning_s']),
        'I5(原点を通る回帰)+ 選別(%s)' % ('残す' if extra['selection']['human']['k_s']['keep'] else '落とす'), 0.0,
        '直線の回帰なので、|a_lat| 2〜6 では多め、9 以上では少なめ(i5_coast.json の residual_by_alat)')
    add('longitudinal.d_v', '速さの指令の遅れ', lon['d_v'], 's', bx('d_v') if extra['selection']['human']['d_v']['keep'] else None, '同上(I6)',
        'I6 + 選別(%s)%s' % ('残す' if extra['selection']['human']['d_v']['keep'] else '落とす = 0', '。人の対応なので + 0.008 秒(PCA9685 の周期の平均)' if speed_op == 0 else ''), 0.0)
    nb = b7['nominal']
    add('longitudinal.brake', 'ESC のブレーキ a_b(off) = s_b × max(0, off − db_b)(off = 中立からの µs)',
        {'s_b': nb['s_b'], 'db_b': nb['db_b'], 'a_b_at_200us': nb['a_b_at_200us'], 'a_b_at_350us': nb['a_b_at_350us']}, '(m/s²)/µs・µs',
        {'scenarios': b7['scenarios'], 'boot_a_b_200_p10_p50_p90': b7['boot_a_b_200_p10_p50_p90'], 'kind': '出来事のブートストラップ 2000 回'},
        'きれいなブレーキの出来事 16 回(固定。強い ≥ 300 µs は 3 回、保留のシーンのもの %d 回を含む)' % b7['n_in_heldout_scenes'],
        'I7 + deviations D5(実測の惰性の曲線を引く)', {'brakeDeceleration': 1.2},
        '字義どおりの I7(直線のこすれを引く)では db_b = %.0f µs で +200 µs は惰性と同じ(弱めのシナリオと同じ)。表は仮(計画項目 8 の後に v2)' % b7['protocol_literal']['db_b'])
    add('longitudinal.esc_reverse', 'ESC の後退の手順', lon['esc_reverse'], '-', None, 'explore_control.md 91 行(実車の観察)',
        'protocol の決まり(同定はしない)', '指令どおりすぐ後退')
    add('longitudinal.k_batt', '電池の係数(v_tgt に掛ける)', 1.0, '-', {'scene_g_table': 'ident/final_decisions.json の i9_scenes'},
        '当てはめのシーンの効き g(人 %.1f・AI %.1f µs/kERPM が 1.0)' % (dmeta['g_ref']['human'], dmeta['g_ref']['ai']), 'I9', None,
        '忠実度の再生では窓の前 5 秒の g から k_batt = clip(g_ref/g, 0.5, 1.2)。感度は 0.8')
    add('app_control_delay_frames', 'アプリの制御の遅れ(撮像 → PWM)', 1, 'フレーム(33 ms)',
        {'p50_range_ms': extra['i8']['p50_range_logs'], 'p99_range_ms': extra['i8']['p99_range_logs']},
        'AI_CMD.latency_ms(当てはめのシーン %d フレーム)' % extra['i8']['pooled']['n'], 'I8', 1, '変えない')
    # 目に見える量の比べ
    def vss(thr, prm=None):
        u = (1498 - thr) / 498 * 7.24
        return float(cm.speed_target(u, P_cal_lon_prm)) if prm is None else u
    P_cal_lon_prm = {'u_db': lon['u_db'], 'T': lon['T_values']}
    eq = {
        'full_lock_curvature_low_speed_per_m': {'left_real': math.tan(L6[-1]) / L_WB, 'right_real': math.tan(R6[-1]) / L_WB,
                                                'sim': math.tan(0.52) / L_WB},
        'target_speed_by_throttle_pwm': {str(t): {'real': vss(t), 'sim': (1498 - t) / 498 * 7.24} for t in (1440, 1400, 1350, 1300, 1250, 1200, 1150, 1100, 1000)},
        'coast_decel_straight': {'real': lon['c0'], 'sim': 1.2},
        'coast_decel_turning_alat5': {'real': lon['c0'] + lon['k_s'] * 5, 'sim': 1.2},
        'brake_total_decel_at_200us_straight': {'real_nominal': lon['c0'] + nb['a_b_at_200us'], 'sim': 1.2},
        'lateral_limit_mps2': {k: {'real': v['mu'] * msc * G, 'sim': v['sim_current'] * G} for k, v in gr['surfaces'].items()},
        'steer_delay_ms': {'real': 1000 * P_cal['steering']['d_s'], 'sim': 0},
    }
    out = {
        'schema': 'jf-sim-profile-real/v1', 'name': P_cal['name'], 'created_at': P_cal['created_at'], 'protocol_sha256': PROTO_SHA,
        'files': {'P_cal': 'profiles/P_cal.json', 'P_cfg': 'profiles/P_cfg.json', 'sha256': 'profiles/SHA256SUMS'},
        'data': {'logs': '2026-10-10 の実車のログ 23 本(壊れた 2 シーンを除く)', 'split': '各ログのシーンの番号 k % 3 != 1 だけ(保留のシーンは使っていない。I7 の固定の 16 回のブレーキだけ例外)',
                 'fit_windows': win, 'fit_windows_by_section': dmeta['n_windows_by_sec']},
        'decisions': extra['decision'], 'selection': extra['selection'], 'mu_scale_selection': extra['mu_scale_selection'],
        'parameters': P, 'equivalents_real_vs_sim': eq,
        'P_cfg': {'values': P_cfg['sim_config'], 'actuator_delay_frames': P_cfg['actuator_delay_frames'], 'fit': pc['by_delay'],
                  'default_fit': pc['default'], 'sim_current': {k: SIM_DEFAULT_CFG[k] for k in ('driveTau', 'driveAccelerationCap', 'brakeDeceleration', 'steerRate', 'maxSteer', 'frictionMultiplier')}},
        'not_modeled': ['摩擦円(縦と横の結合はこすれ k_s だけ)', '車輪の空転・滑りによる eRPM の過大', 'ToF の雑音・床の反射', 'カメラの画像', 'PCA9685 の周期(d_s・d_v に含める)'],
        'limits': ['実車のログは 10/10 の 1 日だけ。電圧の記録は無い(電池は g で代わりに見る)', 'ブレーキは 16 回(強いもの 3 回)で、表は仮',
                   '速さは eRPM から。滑ると大きく出る', 'PTFE と芝は分けられなかった(芝の倍率を PTFE にも掛けた)',
                   '忠実度(保留のシーン)はまだ。この値は当てはめのシーンだけで決めた'],
    }
    jdump(REPORT / 'profile_real_v1.json', out)



# =================================================================== 当てはめの窓での前後(参考。合格の判定ではない)
def cal_prm_from_profile(P_cal, scenario='nominal'):
    st = P_cal['steering']; lo = P_cal['longitudinal']
    b = lo['brake'] if scenario == 'nominal' else lo['brake_scenarios'][scenario]
    return {'steer_L': st['delta_L_rad'], 'steer_R': st['delta_R_rad'], 'K_us': st['K_us'], 'd_s': st['d_s'],
            'r_s': np.inf if st['r_s'] is None else st['r_s'], 'u_db': lo['u_db'], 'T': lo['T_values'], 'tau_up': lo['tau_up'],
            'a_cap': lo['a_cap'], 'tau_down': lo['tau_down'], 'c0': lo['c0'], 'k_s': lo['k_s'], 'd_v': lo['d_v'],
            's_b': b['s_b'], 'db_b': b['db_b']}


def step_insample(force=False):
    out_p = OUTD / 'insample_fit_windows.json'
    if out_p.exists() and not force:
        return jload(out_p)
    S, W, logs = load_cache()
    P_cal = jload(REPORT / 'profiles' / 'P_cal.json'); P_cfg = jload(REPORT / 'profiles' / 'P_cfg.json')
    tru = cm.truth(W)
    mu_def = np.array([SIM_MU[s] for s in W['surface']])
    mu_cal = np.array([P_cal['surfaces_mu'][s] for s in W['surface']])
    cfg = dict(SIM_DEFAULT_CFG); cfg.update({k: P_cfg['sim_config'][k] for k in ('driveTau', 'driveAccelerationCap', 'brakeDeceleration', 'steerRate', 'maxSteer', 'frictionMultiplier')})
    cfg['control_delay_frames'] = P_cfg['actuator_delay_frames']
    runs = {'P_default': cm.replay(W, 'sim', dict(SIM_DEFAULT_CFG), 'coupled', mu=mu_def),
            'P_cfg': cm.replay(W, 'sim', cfg, 'coupled', mu=mu_def * cfg['frictionMultiplier']),
            'P_cal': cm.replay(W, 'cal', cal_prm_from_profile(P_cal), 'coupled', mu=mu_cal, k_batt=W['k_batt']),
            'P_cal_k1': cm.replay(W, 'cal', cal_prm_from_profile(P_cal), 'coupled', mu=mu_cal)}
    errs = {k: cm.errors(v, tru) for k, v in runs.items()}
    groups = {'all': np.ones(W['n'], bool), 'human': W['op'] == 0, 'ai': W['op'] == 1}
    for sname in ('R1', 'GM', 'DE', 'LH', 'HS', 'SW'):
        groups[sname] = W['sec'] == sname
    res = {'note': '当てはめの窓(in-sample)での結合した再生。合格の判定は保留のシーン(S3)で行う。ここは参考', 'groups': {}}
    for gname, m in groups.items():
        row = {'n': int(m.sum())}
        for k, e in errs.items():
            row[k] = {'e_v_rmse': float(np.sqrt(np.mean(e['e_v'][m] ** 2))), 'e_v_median': float(np.median(e['e_v'][m])),
                      'e_r_rms': float(np.sqrt(np.mean(e['e_r'][m] ** 2))), 'e_psi_rmse': float(np.sqrt(np.mean(e['e_psi'][m] ** 2))),
                      'e_psi_median': float(np.median(e['e_psi'][m])), 'e_p_rms': float(np.sqrt(np.mean(e['e_p'][m] ** 2)))}
        for k in ('P_cfg', 'P_cal', 'P_cal_k1'):
            row[k]['improve'] = {mm: 1 - row[k][mm] / row['P_default'][mm] for mm in ('e_v_rmse', 'e_r_rms', 'e_psi_rmse', 'e_p_rms')}
        res['groups'][gname] = row
    jdump(out_p, res)
    a = res['groups']['all']
    log('in-sample', {k: {m: round(a[k][m], 3) for m in ('e_v_rmse', 'e_r_rms', 'e_psi_rmse', 'e_p_rms')} for k in errs})
    return res



# =================================================================== 定常の速さの確かめ(deviations D10)
STEADY_TOL = 0.10        # 直線の定常の速さ T(u) が、当てはめのシーンの人の『一定のスロットル』の表から 10% より外れたら、表を弱い事前として当て直す
STEADY_LAMBDA = 0.02


def steady_table(S):
    """当てはめのシーンの人の『一定のスロットル』(1 秒の標準偏差 < 8 µs・直線 |ずれ| < 100 µs・|加速| < 0.4・40 µs の帯)"""
    thr = S['thr']
    sd = rolling_std(thr, 100)
    m = S['clean'] & (S['op'] == 0) & (sd < 8) & (np.abs(S['off']) < 100) & (np.abs(S['acc']) < 0.4) & (S['vs'] > 0.3) & (thr < 1490)
    rows = []
    for lo in range(1100, 1480, 40):
        k = m & (thr >= lo) & (thr < lo + 40)
        if k.sum() >= 50:
            rows.append({'pwm_band': [lo, lo + 40], 'u': float(np.median((cd.THR_N - thr[k]) / 498 * 7.24)), 'n': int(k.sum()),
                         'v_median': float(np.median(S['vs'][k]))})
    return rows


def steady_check(prm, rows, scale=1.0):
    out = []
    for r_ in rows:
        if not (1.2 <= r_['u'] <= 4.6):
            continue
        t = float(cm.speed_target(r_['u'], prm)) / scale
        out.append(dict(r_, model=t, rel=(t - r_['v_median']) / r_['v_median']))
    return out, (max(abs(o['rel']) for o in out) if out else 0.0)


def run_i6_fit_prior(op, base, variant, rows, tag, to_full=None, x0=None):
    """事前つきの当てはめ: J + λ Σ w (T(u_i) − v_i)²(w = min(n, 500)/500)。人の窓"""
    d = OUTD / 'i6'
    p = d / (tag + '.json')
    if p.exists():
        return jload(p)
    S, W, logs = load_cache()
    gr = jload(OUTD / 'i3_grip.json')
    surf_mu = {k: v['mu'] for k, v in gr['surfaces'].items()}
    Wx = sub_windows(W, W['op'] == op)
    tru = cm.truth(Wx)
    mu = mu_windows(Wx, surf_mu)
    fixed = i6_fixed()
    f0 = i6_objective(Wx, tru, mu, fixed, variant)
    us = np.array([r_['u'] for r_ in rows]); vs = np.array([r_['v_median'] for r_ in rows])
    ws = np.array([min(r_['n'], 500) / 500 for r_ in rows])

    def f(x):
        p_ = i6_prm(x, fixed, variant)
        pen = STEADY_LAMBDA * float(np.sum(ws * (cm.speed_target(us, p_) - vs) ** 2))
        return f0(x) + pen
    t0 = time.time()
    x, fx, it = cm.nm_restarts(f, np.asarray(x0 if x0 is not None else base['x'], float), I6_STEP, max_iter=1500, max_restarts=6)
    pf = f0(x, return_parts=True)
    pe = i6_objective(Wx, tru, mu, fixed, variant, exact=True)(x, return_parts=True)
    out = {'op': OPS[op], 'variant': 'prior_' + base.get('variant', ''), 'x': np.clip(x, I6_LO, I6_HI).tolist(), 'names': I6_NAMES,
           'x_reduced': None, 'prm': i6_prm(x, fixed, variant), 'J': pe['J'], 'rmse_1s': pe['rmse_1s'], 'rmse_win': pe['rmse_win'],
           'bias_1s_median': pe['bias_1s_median'], 'J_fast_5ms': pf['J'], 'exact': True, 'objective_with_prior': fx,
           'prior': {'lambda': STEADY_LAMBDA, 'rows': rows}, 'iters': it, 'seconds': time.time() - t0, 'n_windows': Wx['n']}
    jdump(p, out)
    log('I6 prior', tag, 'J', round(pe['J'], 5), 'T', np.round(out['prm']['T'], 2), np.round(out['x'][8:], 3))
    return out



# =================================================================== D12 舵の中心(n = 0)の値 / D7 AI の速さの対応(指令の側の倍率とずれ)
def refit_steer_zero_center(tab):
    """帯の表 [n の中央値, 数, δ の中央値] から、n = 0 の値を 0 に固定して左右を別々に区分線形・単調で当てはめ直す"""
    tab = np.asarray(tab, float)
    vals = np.zeros(len(STEER_FIT_KNOTS))
    for side in (1, -1):
        m = (tab[:, 0] * side) > 0
        if not m.any():
            continue
        x = np.abs(tab[m, 0]); y = side * tab[m, 2]; w = np.minimum(tab[m, 1], 300)
        kn = np.array([0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
        A = cm.interp_matrix(x, kn)[:, 1:]                # 0 の節は 0 に固定
        C = np.tril(np.ones((5, 5)))
        M = A @ C
        Wm = M * np.sqrt(w)[:, None]; yw = y * np.sqrt(w)
        H = Wm.T @ Wm + 1e-9 * np.eye(5); gv = Wm.T @ yw
        pz = np.zeros(5)
        for _ in range(5000):
            old = pz.copy()
            for j_ in range(5):
                r_ = gv[j_] - H[j_] @ pz + H[j_, j_] * pz[j_]
                pz[j_] = max(0.0, r_ / H[j_, j_])
            if np.max(np.abs(pz - old)) < 1e-13:
                break
        mags = np.concatenate([[0.0], C @ pz])
        if side == 1:
            vals[5:] = mags
        else:
            vals[:6] = -mags[::-1]
    return vals


def ai_input_variant(xh, flags):
    """AI の T(u) = 人の T(a u + b)(PWM の経路の違い = 指令の側の倍率とずれ。deviations D7)。動きは人の当てはめのまま"""
    xh = np.asarray(xh, float)
    Th_knots = np.concatenate([[0.0, xh[8]], np.maximum.accumulate(np.clip(xh[:8], 0, None))])
    Th_u = np.concatenate([[0.0], [xh[8]], cm.SPEED_KNOTS])

    def to_full(z):
        a_, b_ = z
        x = xh.copy()
        uu = a_ * cm.SPEED_KNOTS + b_
        x[:8] = np.interp(uu, Th_u, Th_knots)
        x[8] = float(np.clip((xh[8] - b_) / max(a_, 1e-3), 0.0, 0.45))
        return x
    v = dict(flags)
    v['to_full'] = to_full
    v['step'] = np.array([0.1, 0.1])
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', default='data,i1,i3,i4,i5,i7')
    ap.add_argument('--op', type=int, default=None)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    steps = a.steps.split(',')
    if 'data' in steps:
        step_data(a.force)
    if 'i1' in steps:
        step_i1_i2(a.force)
    if 'i3' in steps:
        step_i3(a.force)
    if 'i4' in steps:
        step_i4(a.force)
    if 'i5' in steps:
        step_i5(a.force)
    if 'i7' in steps:
        step_i7(a.force)
    if 'i6' in steps:
        for op in ((0, 1) if a.op is None else (a.op,)):
            step_i6_op(op)
    if 'pcfg' in steps:
        step_pcfg(a.force)
    if 'final' in steps:
        step_final(a.force)
    if 'insample' in steps:
        step_insample(a.force)


if __name__ == '__main__':
    main()
