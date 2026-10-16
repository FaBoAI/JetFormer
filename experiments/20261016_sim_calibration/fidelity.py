#!/usr/bin/env python3
"""シミュレーター較正(2026-10-16)S3: 忠実度の確かめ(保留のシーン)と、試験 T3(Python の再生と JS のコアの一致)

    cd ~/jetformer/experiments/20261016_sim_calibration
    /home/jetson/bin/jf-heavy --mem 1500M --lock none -- nice -n 10 python fidelity.py --steps t3
    /home/jetson/bin/jf-heavy --mem 1500M --lock none -- nice -n 10 python fidelity.py --steps data,replay,metrics,report

protocol.json(sha256 61fc91cb…)の fidelity 章のとおり。
  - 窓: 保留のシーン(各ログのシーンの番号 k % 3 == 1)の 1 秒の窓(0.5 秒ずらし)。参考に 2 秒の窓(C5)
  - 再生: 較正版のコア(JS、replay_fidelity.mjs)を dt = 0.01 秒・障害物なしの広い世界で。最終の数字はこれ
  - 物理: P_default・P_cfg・P_cal(k_batt を窓の前 5 秒で合わせる)・P_cal_k1。参考にブレーキの弱め・強め
  - 指標: e_v(1 秒後の速さ)・e_r(窓の中のヨーレートの RMS)・e_psi(1 秒後の向き)・e_p(1 秒後の位置)
  - 90% 区間: シーン単位のブートストラップ 2000 回(種 20261016)。改善率 I = 1 − M(P)/M(P_default)
  - 合格の決まり C1〜C3(結果を見てから変えない)。参考: C4 安定性(簡単な当て直し)・C5 2 秒の窓・P_cfg の判定・P_cal_k1
  - 部品の診断(横だけ・縦だけ)は Python の参照の再生(calib_model.replay、T3 で JS と一致を確かめたもの)で出す
読み取りだけ。data/raw・data/labels は変えない。実車・PCA9685・I2C には触れない。
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calib_data as cd  # noqa: E402
import calib_model as cm  # noqa: E402

REPORT = Path('/home/jetson/jetformer_reports/sim_calibration_20261016')
OUTD = REPORT / 'fidelity'
PROTO_SHA = '61fc91cbf67768ecc325ba4ebb0f9470bfbbe7658c566cdbb02eb8ceb2fef52d'
PROFILE_SHA = {'P_cal.json': '4b0e9348e367e64d93c4dc491de26b5dd5c7e2375a60f6f618f0d24ad85ddc14',
               'P_cfg.json': 'b15a436b923c6e269e1cdc3b2e30368dcbaba126c387aac51b856c47d227d5c1'}
SEED = 20261016
N_BOOT = 2000
SIM_MU = {'carpet': 0.8, 'shortcut': 0.7, 'turf': 0.6, 'ptfe': 0.22, 'blue-mat': 0.65}
SIM_DEFAULT_CFG = {'maxSpeed': 7.24, 'maxSteer': 0.52, 'steerRate': 1.2, 'driveTau': 0.4, 'driveAccelerationCap': 6.0,
                   'brakeDeceleration': 1.2, 'frictionMultiplier': 1.0, 'control_delay_frames': 0}
PHYS_MAIN = ['P_default', 'P_cfg', 'P_cal', 'P_cal_k1', 'P_cal_brake_low', 'P_cal_brake_high']
METRICS = ['e_v_rmse', 'e_r_rms', 'e_psi_rmse', 'e_p_rms']
SECTIONS = ['R1', 'GM', 'DE', 'LH', 'HS', 'SW', 'unknown']
OPS = {0: 'human', 1: 'ai'}
NODE = 'node'


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def _clean(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def jdump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(_clean(json.loads(json.dumps(obj, default=_np_default))), f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def jload(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_frozen():
    assert hashlib.sha256((REPORT / 'protocol.json').read_bytes()).hexdigest() == PROTO_SHA, 'protocol.json が固めたものと違う'
    for f, s in PROFILE_SHA.items():
        assert hashlib.sha256((REPORT / 'profiles' / f).read_bytes()).hexdigest() == s, f + ' が固めたものと違う'


def load_profiles():
    return jload(REPORT / 'profiles' / 'P_cal.json'), jload(REPORT / 'profiles' / 'P_cfg.json')


def cal_prm(P_cal, scenario='nominal'):
    st = P_cal['steering']; lo = P_cal['longitudinal']
    b = lo['brake'] if scenario == 'nominal' else lo['brake_scenarios'][scenario]
    return {'steer_L': st['delta_L_rad'], 'steer_R': st['delta_R_rad'], 'K_us': st['K_us'], 'd_s': st['d_s'],
            'r_s': np.inf if st['r_s'] is None else st['r_s'], 'u_db': lo['u_db'], 'T': lo['T_values'], 'tau_up': lo['tau_up'],
            'a_cap': lo['a_cap'], 'tau_down': lo['tau_down'], 'c0': lo['c0'], 'k_s': lo['k_s'], 'd_v': lo['d_v'],
            's_b': b['s_b'], 'db_b': b['db_b']}


def cfg_prm(P_cfg):
    cfg = dict(SIM_DEFAULT_CFG)
    cfg.update({k: P_cfg['sim_config'][k] for k in ('driveTau', 'driveAccelerationCap', 'brakeDeceleration', 'steerRate', 'maxSteer', 'frictionMultiplier')})
    cfg['control_delay_frames'] = P_cfg['actuator_delay_frames']
    return cfg


# ------------------------------------------------------------------ JS の再生との受け渡し
def export_windows(W, out_dir, physics, n_win=cd.N_WIN):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(W['n'])
    n_all = cd.N_HIST + cd.N_WARM + n_win + 1
    files = {}
    for k in ('u_s', 'u_v', 'v'):
        a = np.ascontiguousarray(W[k], dtype='<f8')
        assert a.shape == (n, n_all), (k, a.shape)
        assert np.isfinite(a).all(), k + ' に NaN'
        (out_dir / (k + '.f64')).write_bytes(a.tobytes())
        files[k] = k + '.f64'
    header = {'n': n, 'n_hist': cd.N_HIST, 'n_warm': cd.N_WARM, 'n_win': n_win, 'files': files,
              'surface': [str(s) for s in W['surface']], 'k_batt': [float(x) for x in W['k_batt']], 'physics': list(physics)}
    (out_dir / 'header.json').write_text(json.dumps(header))
    return header


def run_node(out_dir):
    t = time.time()
    r = subprocess.run([NODE, str(HERE / 'replay_fidelity.mjs'), str(out_dir)], cwd=str(HERE), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('replay_fidelity.mjs が失敗: ' + r.stderr[-3000:])
    log('JS の再生', out_dir, round(time.time() - t, 1), 's', r.stdout.strip().replace('\n', ' / '))


def load_js(out_dir, name, n, n_win):
    a = np.frombuffer((Path(out_dir) / f'out_{name}.f64').read_bytes(), dtype='<f8')
    a = a.reshape(4, n, n_win + 1)
    return {'v': a[0], 'psi': a[1], 'x': a[2], 'y': a[3]}


# ------------------------------------------------------------------ T3
def step_t3():
    """当てはめの窓 50 個で、Python の参照の再生(calib_model.replay)と JS のコアが一致するか(速さ・ヨーレートの差 ≤ 1e-6)"""
    z = np.load(REPORT / 'ident' / 'cache_fit.npz', allow_pickle=False)
    W = {k[2:]: z[k] for k in z.files if k.startswith('W_')}
    W['n'] = len(W['op'])
    rng = np.random.default_rng(SEED)
    has_brake = (W['u_v'] < -60 / 627.5).any(axis=1)
    pick = []
    # 区間・操作者・ブレーキを含むように選ぶ(ブレーキ 6・R1 8・GM 8・DE 8・AI 10・残りは無作為)
    for m, k in ((has_brake, 6), (W['sec'] == 'R1', 8), (W['sec'] == 'GM', 8), (W['sec'] == 'DE', 8), (W['op'] == 1, 10)):
        idx = np.setdiff1d(np.nonzero(m)[0], pick)
        pick += list(rng.choice(idx, size=min(k, len(idx)), replace=False))
    rest = np.setdiff1d(np.arange(W['n']), pick)
    pick += list(rng.choice(rest, size=50 - len(pick), replace=False))
    pick = np.array(sorted(int(i) for i in pick))
    Wx = {k: (v[pick] if isinstance(v, np.ndarray) and v.shape[:1] == (W['n'],) else v) for k, v in W.items()}
    Wx['n'] = len(pick)
    P_cal, P_cfg = load_profiles()
    out_dir = OUTD / 't3'
    export_windows(Wx, out_dir, ['P_default', 'P_cfg', 'P_cal', 'P_cal_k1', 'P_cal_brake_high'])
    run_node(out_dir)
    mu_def = np.array([SIM_MU[s] for s in Wx['surface']])
    mu_cal = np.array([P_cal['surfaces_mu'][s] for s in Wx['surface']])
    cfg = cfg_prm(P_cfg)
    py = {'P_default': cm.replay(Wx, 'sim', dict(SIM_DEFAULT_CFG), 'coupled', mu=mu_def),
          'P_cfg': cm.replay(Wx, 'sim', cfg, 'coupled', mu=mu_def * cfg['frictionMultiplier']),
          'P_cal': cm.replay(Wx, 'cal', cal_prm(P_cal), 'coupled', mu=mu_cal, k_batt=Wx['k_batt']),
          'P_cal_k1': cm.replay(Wx, 'cal', cal_prm(P_cal), 'coupled', mu=mu_cal),
          'P_cal_brake_high': cm.replay(Wx, 'cal', cal_prm(P_cal, 'high'), 'coupled', mu=mu_cal, k_batt=Wx['k_batt'])}
    res = {'note': 'T3: 当てはめの窓 50 個(ブレーキ・R1・GM・DE・AI を含む)。Python = calib_model.replay、JS = 較正版のコア(replay_fidelity.mjs)。'
                   'ヨーレートは 10 ms ごとの Δψ/dt。P_cfg は JS が maxSteer 0.398 で小区間の長さ(0.012/最大の動きの速さ)を変えるため、'
                   'Python(小区間は既定の 0.52 で計算)との差は参考(合否は P_default・P_cal・P_cal_k1・P_cal_brake_high)',
           'windows': pick.tolist(), 'n_brake_windows': int(has_brake[pick].sum()), 'physics': {}}
    ok = True
    for name, p in py.items():
        j = load_js(out_dir, name, Wx['n'], cd.N_WIN)
        dv = np.max(np.abs(j['v'] - p['v']))
        r_js = np.diff(j['psi'], axis=1) / cm.DT
        r_py = np.diff(p['psi'], axis=1) / cm.DT
        dr = np.max(np.abs(r_js - r_py))
        dp = np.max(np.hypot(j['x'] - p['x'], j['y'] - p['y']))
        dpsi = np.max(np.abs(j['psi'] - p['psi']))
        strict = name != 'P_cfg'
        passed = bool(dv <= 1e-6 and dr <= 1e-6) if strict else None
        if strict and not passed:
            ok = False
        res['physics'][name] = {'max_abs_dv': float(dv), 'max_abs_dr': float(dr), 'max_abs_dpsi_rad': float(dpsi), 'max_pos_m': float(dp),
                                'strict': strict, 'pass': passed}
        log('T3', name, 'dv %.2e dr %.2e dpsi %.2e dp %.2e' % (dv, dr, dpsi, dp), passed)
    res['pass'] = ok
    jdump(REPORT / 'tests' / 't3_python_js.json', res)
    if not ok:
        raise SystemExit('T3 が通らない(JS を正とし、差を報告する)')
    return res


# ------------------------------------------------------------------ 保留のシーンの窓
def build_heldout(n_win=cd.N_WIN):
    meta_fit = jload(REPORT / 'ident' / 'data.json')
    g_ref = {0: meta_fit['g_ref']['human'], 1: meta_fit['g_ref']['ai']}       # 当てはめのシーンの基準(D2: 操作者ごと)
    old = cd.N_WIN
    cd.N_WIN = n_win
    try:
        windows, per_log = [], []
        for lg in cd.list_logs():
            rec = cd.load_record(lg, 'heldout')
            if rec is None:
                continue
            g = cd.battery_g(rec)
            ws = cd.make_windows(rec, g=g, g_ref=g_ref)
            for w in ws:
                # 窓の中のブレーキの要求(中立 + 60 µs 以上)
                w['brake_us_max'] = float(np.nanmax(w['thr'][cd.N_HIST + cd.N_WARM:] - cd.THR_N))
            windows += ws
            per_log.append({'log': lg, 'heldout_scenes': [s for k, s in enumerate(rec['scene_names']) if k % 3 == 1 and s not in cd.EXCLUDED_SCENES],
                            'windows': len(ws)})
            log(lg, 'held-out windows', len(ws))
    finally:
        cd.N_WIN = old
    W = cd.stack_windows(windows)
    W['brake_us_max'] = np.array([w['brake_us_max'] for w in windows])
    return W, {'g_ref': g_ref, 'logs': per_log}


def step_data(force=False):
    for n_win, tag in ((cd.N_WIN, 'w1'), (200, 'w2')):
        p = OUTD / f'heldout_{tag}.npz'
        if p.exists() and not force:
            continue
        W, meta = build_heldout(n_win)
        np.savez_compressed(p, **{k: v for k, v in W.items() if k != 'n'})
        jdump(OUTD / f'heldout_{tag}_meta.json', dict(meta, n=int(W['n']), n_win=n_win,
                                                     by_op={OPS[o]: int((W['op'] == o).sum()) for o in (0, 1)},
                                                     by_sec={s: int((W['sec'] == s).sum()) for s in np.unique(W['sec'])}))
        log('held-out', tag, W['n'])


def load_heldout(tag):
    z = np.load(OUTD / f'heldout_{tag}.npz', allow_pickle=False)
    W = {k: z[k] for k in z.files}
    W['n'] = len(W['op'])
    return W


def step_replay(force=False):
    for tag, n_win in (('w1', cd.N_WIN), ('w2', 200)):
        out_dir = OUTD / f'js_{tag}'
        if (out_dir / 'out.json').exists() and not force:
            continue
        W = load_heldout(tag)
        export_windows(W, out_dir, PHYS_MAIN if tag == 'w1' else ['P_default', 'P_cfg', 'P_cal', 'P_cal_k1'], n_win=n_win)
        run_node(out_dir)


# ------------------------------------------------------------------ 指標
def window_errors(sim, tru):
    return cm.errors(sim, tru)


def summarize(e, m):
    if m.sum() == 0:
        return None
    ev, er, eps, ep = e['e_v'][m], e['e_r'][m], e['e_psi'][m], e['e_p'][m]
    return {'n': int(m.sum()),
            'e_v_rmse': float(np.sqrt(np.mean(ev ** 2))), 'e_v_median': float(np.median(ev)), 'e_v_abs_p90': float(np.percentile(np.abs(ev), 90)),
            'e_r_rms': float(np.sqrt(np.mean(er ** 2))),
            'e_psi_rmse': float(np.sqrt(np.mean(eps ** 2))), 'e_psi_median': float(np.median(eps)),
            'e_p_rms': float(np.sqrt(np.mean(ep ** 2))),
            'e_vw_rms': float(np.sqrt(np.mean(e['e_vw'][m] ** 2)))}


def boot_improve(errs, base, names, m, units, n_boot=N_BOOT, seed=SEED):
    """シーン単位のブートストラップで、改善率 I = 1 − M(P)/M(P_default) の 90% 区間(同じ引き方を全部の物理に使う)"""
    idx = np.nonzero(m)[0]
    if len(idx) == 0:
        return {}
    u = units[idx]
    uniq, inv = np.unique(u, return_inverse=True)
    rng = np.random.default_rng(seed)
    groups = [idx[inv == k] for k in range(len(uniq))]

    def mets(e, sel):
        return {'e_v_rmse': np.sqrt(np.mean(e['e_v'][sel] ** 2)), 'e_r_rms': np.sqrt(np.mean(e['e_r'][sel] ** 2)),
                'e_psi_rmse': np.sqrt(np.mean(e['e_psi'][sel] ** 2)), 'e_p_rms': np.sqrt(np.mean(e['e_p'][sel] ** 2)),
                'e_v_median': np.median(e['e_v'][sel]), 'e_psi_median': np.median(e['e_psi'][sel])}
    samples = {nm: {k: [] for k in METRICS + ['e_v_median', 'e_psi_median']} for nm in names}
    base_s = {k: [] for k in METRICS}
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        sel = np.concatenate([groups[k] for k in pick])
        mb = mets(base, sel)
        for k in METRICS:
            base_s[k].append(mb[k])
        for nm in names:
            mp = mets(errs[nm], sel)
            for k in METRICS:
                samples[nm][k].append(1 - mp[k] / mb[k])
            samples[nm]['e_v_median'].append(mp['e_v_median'])
            samples[nm]['e_psi_median'].append(mp['e_psi_median'])
    out = {'n_units': len(uniq)}
    for nm in names:
        out[nm] = {('I_' + k if k in METRICS else k): [float(np.percentile(v, 5)), float(np.percentile(v, 95))] for k, v in samples[nm].items()}
    return out


def verdict(table, ci, name):
    """protocol fidelity.pass の C1〜C3 と判定"""
    a = table['all']
    I = {k: 1 - a[name][k] / a['P_default'][k] for k in METRICS}
    lo = {k: ci['all'][name]['I_' + k][0] for k in METRICS}
    c1_metric = {k: bool(I[k] >= 0.20 and lo[k] > 0) for k in METRICS}
    c1_ops = {}
    for op in ('human', 'ai'):
        r = table[op]
        c1_ops[op] = {k: bool(1 - r[name][k] / r['P_default'][k] >= 0.10) for k in ('e_v_rmse', 'e_r_rms', 'e_psi_rmse')}
    c1 = all(c1_metric.values()) and all(all(v.values()) for v in c1_ops.values())
    c2_noworse, c2_psi = {}, {}
    for s in ('R1', 'GM', 'DE', 'LH'):
        r = table.get(s)
        if not r or r['P_default']['n'] < 25:
            c2_noworse[s] = None
            continue
        c2_noworse[s] = {k: bool(r[name][k] <= 1.05 * r['P_default'][k]) for k in METRICS}
        if s in ('R1', 'GM'):
            c2_psi[s] = bool(1 - r[name]['e_psi_rmse'] / r['P_default']['e_psi_rmse'] >= 0.10)
    c2_nw = all(all(v.values()) for v in c2_noworse.values() if v is not None)
    c2 = c2_nw and all(c2_psi.values())
    c3 = {'all': {'e_v_median': bool(abs(a[name]['e_v_median']) <= 0.10), 'e_psi_median': bool(abs(a[name]['e_psi_median']) <= 2.0)}}
    for s in ('R1', 'GM', 'DE'):
        r = table.get(s)
        if r:
            c3[s] = {'e_v_median': bool(abs(r[name]['e_v_median']) <= 0.10), 'e_psi_median': bool(abs(r[name]['e_psi_median']) <= 3.0)}
    c3_all = all(all(v.values()) for v in c3.values())
    if c1 and c2 and c3_all:
        v = '合格'
    elif sum(c1_metric.values()) >= 3 and c2_nw and all(c3['all'].values()):
        v = '条件つき'
    else:
        v = '不合格'
    return {'improve_all': I, 'improve_ci90_low': lo, 'C1': {'pass': c1, 'metrics': c1_metric, 'operators': c1_ops},
            'C2': {'pass': c2, 'no_worse_1.05': c2_noworse, 'psi_improve_0.10': c2_psi, 'no_worse_pass': c2_nw},
            'C3': {'pass': c3_all, 'groups': c3}, 'verdict': v}


def tables_for(W, errs, names):
    groups = {'all': np.ones(W['n'], bool), 'human': W['op'] == 0, 'ai': W['op'] == 1}
    for s in SECTIONS:
        groups[s] = W['sec'] == s
    for op in (0, 1):
        for s in ('R1', 'GM', 'DE', 'LH'):
            groups[f'{OPS[op]}_{s}'] = (W['op'] == op) & (W['sec'] == s)
    v0 = W['v'][:, cd.N_HIST + cd.N_WARM]
    for lo_, hi_ in ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 9.0)):
        groups[f'v0_{lo_}-{hi_}'] = (v0 >= lo_) & (v0 < hi_)
    groups['brake_in_window'] = W['brake_us_max'] >= 60
    groups['k_batt_adjusted'] = np.abs(W['k_batt'] - 1) > 1e-9
    table = {}
    for g, m in groups.items():
        row = {'n': int(m.sum())}
        if m.sum() == 0:
            table[g] = row
            continue
        for nm in names:
            row[nm] = summarize(errs[nm], m)
        for nm in names:
            if nm != 'P_default':
                row[nm]['improve'] = {k: 1 - row[nm][k] / row['P_default'][k] for k in METRICS}
        table[g] = row
    return table, groups


def component_diagnostics(W, P_cal, P_cfg):
    """部品の診断(Python の参照の再生): 横だけ(実際の速さ)のヨーの誤差、縦だけ(実際のヨー)の速さの誤差"""
    tru = cm.truth(W)
    mu_def = np.array([SIM_MU[s] for s in W['surface']])
    mu_cal = np.array([P_cal['surfaces_mu'][s] for s in W['surface']])
    cfg = cfg_prm(P_cfg)
    out = {}
    for mode in ('lat', 'lon'):
        runs = {'P_default': cm.replay(W, 'sim', dict(SIM_DEFAULT_CFG), mode, mu=mu_def),
                'P_cfg': cm.replay(W, 'sim', cfg, mode, mu=mu_def * cfg['frictionMultiplier']),
                'P_cal': cm.replay(W, 'cal', cal_prm(P_cal), mode, mu=mu_cal, k_batt=W['k_batt'])}
        res = {}
        for nm, r in runs.items():
            e = cm.errors(r, tru)
            if mode == 'lat':
                res[nm] = {'e_r_rms': float(np.sqrt(np.mean(e['e_r'] ** 2))), 'e_psi_rmse': float(np.sqrt(np.mean(e['e_psi'] ** 2))),
                           'by_sec_e_r_rms': {s: float(np.sqrt(np.mean(e['e_r'][W['sec'] == s] ** 2))) for s in SECTIONS if (W['sec'] == s).any()}}
            else:
                res[nm] = {'e_v_rmse': float(np.sqrt(np.mean(e['e_v'] ** 2))), 'e_v_median': float(np.median(e['e_v'])),
                           'e_vw_rms': float(np.sqrt(np.mean(e['e_vw'] ** 2))),
                           'by_op_e_v_median': {OPS[o]: float(np.median(e['e_v'][W['op'] == o])) for o in (0, 1)}}
        out[mode] = res
    return out


def stability_c4(W, P_cal):
    """C4(参考): 保留のシーンで簡単に当て直したパラメーターと、当てはめの値の比(25% 超で『不安定』の印)。
    当て直すのは計算が軽いものだけ: 舵の遅れ d_s(横だけの格子)・μ の共通倍率(結合した再生の格子)・速さの対応の共通倍率(縦だけ)・
    いっぱいの曲率(左右、舵が 0.3 秒一定の時刻の r/v の中央値)・直線の惰性 c0(中立が 0.3 秒以上の時刻の減速の中央値)"""
    tru = cm.truth(W)
    prm = cal_prm(P_cal)
    mu_cal = np.array([P_cal['surfaces_mu'][s] for s in W['surface']])
    res = {}
    # d_s(r_s は固定)
    grid = np.round(np.arange(0.0, 0.1201, 0.01), 3)
    sc = []
    for d in grid:
        p = dict(prm, d_s=float(d))
        e = cm.errors(cm.replay(W, 'cal', p, 'lat', mu=mu_cal, k_batt=W['k_batt']), tru)
        sc.append(float(np.sqrt(np.mean(e['e_r'] ** 2))))
    d_best = float(grid[int(np.argmin(sc))])
    res['d_s'] = {'fit': prm['d_s'], 'heldout': d_best, 'grid': grid.tolist(), 'e_r_rms': sc}
    # μ の共通倍率(I3 の値に掛ける倍率。当てはめでは 1.2)
    mu_i3 = np.array([P_cal['surfaces_mu_I3'][s] for s in W['surface']])
    sgrid = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    sc = []
    for s in sgrid:
        e = cm.errors(cm.replay(W, 'cal', prm, 'coupled', mu=mu_i3 * s, k_batt=W['k_batt']), tru)
        sc.append(float(np.mean(e['e_r'] ** 2)))
    res['mu_scale'] = {'fit': P_cal['mu_scale'], 'heldout': sgrid[int(np.argmin(sc))], 'grid': sgrid, 'e_r_ms': sc}
    # 速さの対応の共通倍率(縦だけ、1 秒後の速さの二乗誤差)
    kgrid = np.round(np.arange(0.80, 1.2501, 0.025), 3)
    sc = []
    for k in kgrid:
        v = cm.lon_fast(W, prm, mu_cal, k_batt=W['k_batt'] * k)
        sc.append(float(np.mean((v[:, -1] - tru['v'][:, -1]) ** 2)))
    res['speed_map_scale'] = {'fit': 1.0, 'heldout': float(kgrid[int(np.argmin(sc))]), 'grid': kgrid.tolist(), 'e_v_ms': sc}
    # いっぱいの曲率・直線の惰性(窓の中のサンプルから)
    i0 = cd.N_HIST + cd.N_WARM
    us = W['u_s'][:, i0:]; v = W['v'][:, i0:]; r = W['r'][:, i0:]; uv = W['u_v'][:, i0:]
    steady = np.zeros_like(us, bool)
    for j in range(30, us.shape[1]):
        steady[:, j] = (np.ptp(us[:, j - 30:j + 1], axis=1) < 0.52 * 16 / 557)
    notgm = (W['sec'] != 'GM')[:, None]
    for side, m in (('left', us >= 0.52 * 0.97), ('right', us <= -0.52 * 0.97)):
        mm = m & steady & (v >= 0.8) & (v <= 2.2) & notgm
        kap = np.abs(r[mm] / v[mm])
        fit_val = math.tan(P_cal['steering']['delta_L_rad' if side == 'left' else 'delta_R_rad'][5]) / 0.26
        res[f'full_lock_curvature_{side}'] = {'fit': fit_val, 'heldout': float(np.median(kap)) if mm.sum() >= 30 else None, 'n': int(mm.sum())}
    neutral = np.abs(uv) < 12 / 498 * 7.24
    stn = np.zeros_like(neutral)
    for j in range(30, neutral.shape[1]):
        stn[:, j] = neutral[:, j - 30:j + 1].all(axis=1)
    straight = np.abs(us) < 0.52 * 100 / 557
    dv = -np.gradient(v, cm.DT, axis=1)
    mm = stn & straight & (v > 1.0)
    res['c0'] = {'fit': P_cal['longitudinal']['c0'], 'heldout': float(np.median(dv[mm])) if mm.sum() >= 30 else None, 'n': int(mm.sum())}
    for k, d in res.items():
        if d.get('heldout') is not None and d['fit']:
            d['rel_change'] = d['heldout'] / d['fit'] - 1
            d['unstable_gt25pct'] = bool(abs(d['rel_change']) > 0.25)
        else:
            d['rel_change'] = None
            d['unstable_gt25pct'] = None
    return res


def step_metrics():
    P_cal, P_cfg = load_profiles()
    out = {'schema': 'jf-calib-fidelity/v1', 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S+09:00'), 'protocol_sha256': PROTO_SHA,
           'profiles_sha256': PROFILE_SHA, 'engine': 'JS の較正版のコア(replay_fidelity.mjs、fast、dt 0.01、障害物なし)',
           'bootstrap': {'unit': 'シーン(ログ + シーン名)', 'n': N_BOOT, 'seed': SEED, 'ci': '90%(5〜95 パーセント点)'}}
    for tag, n_win, names in (('w1', cd.N_WIN, PHYS_MAIN), ('w2', 200, ['P_default', 'P_cfg', 'P_cal', 'P_cal_k1'])):
        W = load_heldout(tag)
        tru = cm.truth(W, n_frames=n_win)
        errs = {nm: window_errors(load_js(OUTD / f'js_{tag}', nm, W['n'], n_win), tru) for nm in names}
        table, groups = tables_for(W, errs, names)
        units = np.array([f'{a}|{b}' for a, b in zip(W['log'], W['scene_name'])])
        ci = {}
        boot_names = [nm for nm in names if nm != 'P_default']
        for g in (['all', 'human', 'ai', 'R1', 'GM', 'DE', 'LH', 'HS', 'SW'] if tag == 'w1' else ['all']):
            ci[g] = boot_improve(errs, errs['P_default'], boot_names, groups[g], units)
            log(tag, 'bootstrap', g)
        sec = {'n_windows': int(W['n']), 'n_scenes': int(len(np.unique(units))), 'table': table, 'ci90': ci}
        if tag == 'w1':
            sec['verdict'] = {nm: verdict(table, ci, nm) for nm in ('P_cal', 'P_cfg', 'P_cal_k1')}
            # ブレーキの窓(窓の中に中立 + 60 µs 以上のブレーキの要求)
            bm = groups['brake_in_window']
            sec['brake_windows'] = {'n': int(bm.sum()), 'n_strong_ge_300us': int((W['brake_us_max'] >= 300).sum()),
                                    'note': 'I7 の 16 回のうち 5 回は保留のシーンにある(protocol が固定)。ここは独立な確かめではない',
                                    'e_v': {nm: summarize(errs[nm], bm) for nm in names},
                                    'windows': [{'log': str(W['log'][i]), 'scene': str(W['scene_name'][i]), 't0': float(W['t0'][i]), 'op': OPS[int(W['op'][i])],
                                                 'brake_us_max': float(W['brake_us_max'][i]), 'v0': float(W['v'][i, cd.N_HIST + cd.N_WARM]),
                                                 'v_real_1s': float(tru['v'][i, -1]),
                                                 **{nm: float(errs[nm]['e_v'][i]) for nm in names}} for i in np.nonzero(bm)[0]]}
            sec['k_batt'] = {'p10_p50_p90': np.percentile(W['k_batt'], [10, 50, 90]).tolist(), 'n_adjusted': int(groups['k_batt_adjusted'].sum()),
                             'by_op_n_adjusted': {OPS[o]: int((groups['k_batt_adjusted'] & (W['op'] == o)).sum()) for o in (0, 1)}}
            log('部品の診断')
            sec['components'] = component_diagnostics(W, P_cal, P_cfg)
            log('C4')
            sec['C4_stability'] = stability_c4(W, P_cal)
            # 窓ごとの誤差(図と本体のセッション用)
            np.savez_compressed(OUTD / 'window_errors_w1.npz', **{f'{nm}__{k}': v for nm, e in errs.items() for k, v in e.items()},
                                sec=W['sec'], op=W['op'], log=W['log'], scene=W['scene_name'], t0=W['t0'], k_batt=W['k_batt'])
        out[tag] = sec
        log(tag, 'done', W['n'])
    jdump(REPORT / 'fidelity.json', out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', default='t3')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    check_frozen()
    steps = a.steps.split(',')
    if 't3' in steps:
        step_t3()
    if 'data' in steps:
        step_data(a.force)
    if 'replay' in steps:
        step_replay(a.force)
    if 'metrics' in steps:
        step_metrics()


if __name__ == '__main__':
    main()
