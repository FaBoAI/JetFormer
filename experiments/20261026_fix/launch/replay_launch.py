"""発進の前進(launch_boost)の記録の再生。2026-10-26(段 launch)

読むだけ。data/raw・data/labels は変えない。車・走行の API には触れない。GPU・torch は使わない(numpy・scipy)。

1. OFF の再生(試験 T13): 10/24 の B の発進 9 回の走行の記録(AI_CMD・VEHICLE・IMU)を、手を入れたあとの
   jetformer.assist.Assist に launch = None(OFF)で頭から流し、出した値が記録の out と同じかを見る。
   入力: model_out(ゲイン・ToF ガードの前)・その撮像時刻までの最新の車両の eRPM・select・hold・dry_run・ttc・
   直近 0.15 秒の IMU の衝撃・走行のパラメータ(log.json drive.params)。時刻 = 撮像時刻 + latency_ms(走行ループの
   time.monotonic() の代わり。差だけが効く)。
2. 始めたか: 同じ発進の最初の AI のフレームで、記録の値から発進の前進の始める条件(人側を見た・止めていない・|eRPM| < 400・
   TTC の直近 5 フレームの中央値 ≥ 3.3 秒・車両の値 0.1 秒以内)を数える。
3. 縮む時間の見積もり(閉ループ): diag_launch の車のモデル(sim.py の Car の写し・当てはめ sim.json)で、発進ごとに
   「LH の入口に着く時間」を
     基準 = 記録の out(時刻の順)
     OFF  = 手を入れたあとの Assist(launch = None)を、車のモデルの速さ・記録の「同じ場所」のモデルの出力・舵・ヨー・
            前の空き(x_free = 記録の TTC × max(v, 0.3))で回したもの
     ON   = 同じで launch = LaunchIn(250 µs・2.5 m/s・1.2 秒。走行ループと同じ速さ・加速・ヨーの作り方)
   で出し、基準 − ON・OFF − ON を「縮む時間」とする。加速の上限 a_cap は 7.13 m/s²(5・9 は感度)。

    cd /home/jetson/jetformer/experiments/20261026_fix/launch
    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I replay_launch.py     # → out/replay_launch.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from jetformer import assist as A  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
DIAG = ROOT / 'experiments' / '20261026_fix' / 'diag_launch'
LAB = ROOT / 'data' / 'labels'
KV = 0.5e-3
FPS = 30.0
DT = 0.01
B9 = [('20261024-162033', 9), ('20261024-162305', 10), ('20261024-183123', 11), ('20261024-183437', 12),
      ('20261024-183826', 14), ('20261024-184043', 15), ('20261024-184400', 16), ('20261024-184549', 17),
      ('20261024-185703', 18)]
WINDOW_S = 3.0                      # 発進の場面 = AI 側にしてから 3 秒
LAUNCH = dict(us=250.0, mps=2.5, max_s=1.2)


class _Sim(object):
    """diag_launch/sim.py の Car・sim_time_indexed・c_ratio・load と同じもの(sim.py は scipy を読むので写した。
    同じかは check_same_as_sim() で確かめる)"""

    class Car(object):
        def __init__(self, p, ratio=1.0, a_cap=None):
            self.alpha, self.f0, self.beta, self.gamma, self.delay = p[:5]
            self.alpha *= ratio
            self.a_cap = a_cap if a_cap is not None else np.inf

        def accel(self, f, v):
            drive = self.alpha * max(f - self.f0, 0.0)
            a = drive - self.beta * v - (self.gamma if v > 0.05 else 0.0)
            if v <= 0.0 and a < 0:
                a = 0.0
            return min(a, self.a_cap)

    @staticmethod
    def sim_time_indexed(car, t_cmd, f_cmd, v0, t_end):
        n = int(t_end / DT) + 1
        ts = np.arange(n) * DT
        v, x = np.zeros(n), np.zeros(n)
        v[0] = v0
        for k in range(n - 1):
            tq = ts[k] - car.delay
            j = int(np.searchsorted(t_cmd, tq, side='right') - 1)
            f = f_cmd[j] if j >= 0 else 0.0
            a = car.accel(f, v[k])
            v[k + 1] = max(v[k] + a * DT, 0.0)
            x[k + 1] = x[k] + 0.5 * (v[k] + v[k + 1]) * DT
        return ts, v, x

    @staticmethod
    def load():
        L = json.loads((DIAG / 'out' / 'launches.json').read_text())['launches']
        fit = json.loads((DIAG / 'out' / 'dyn_fit.json').read_text())
        return L, fit

    @staticmethod
    def c_ratio(fit, session):
        c = fit['best']['c_mps_per_us']
        ref = c['10/24 3回目']
        return c.get(session, ref) / ref


def _sim():
    return _Sim


def check_same_as_sim():
    """写しが diag_launch/sim.py と同じ数を出すか(scipy があるときだけ)。戻り値 True / None(確かめられない)"""
    try:
        import scipy  # noqa: F401
    except ImportError:
        return None
    spec = importlib.util.spec_from_file_location('diag_launch_sim', str(DIAG / 'sim.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    p5 = [0.04086, 5.2, 2.529, -0.873, 0.033]
    t = np.arange(0, 2.0, 1 / FPS)
    f = np.clip(300 * t, 0, 260)
    a = m.sim_time_indexed(m.Car(p5, 1.03, 7.13), t, f, 0.0, 3.0)
    b = _Sim.sim_time_indexed(_Sim.Car(p5, 1.03, 7.13), t, f, 0.0, 3.0)
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def load(log):
    import label_events as le
    L = le.load_log(log)
    drv = L['meta'].get('drive') or {}
    return {'A': L['AI_CMD'], 'V': L['VEHICLE'], 'I': L['IMU'], 'params': drv.get('params') or {},
            'ref': (float(drv.get('steer_center_us') or 1500.0), float(drv.get('throttle_neutral_us') or 1500.0))}


def sections(A_, erpm_at):
    """AI の区間(select・hold でない・dry run でない が続く)→ [(k0, k1, 止まって始まったか)]"""
    ai = A_['select'].astype(bool) & ~A_['hold'].astype(bool) & ~A_['dry_run'].astype(bool)
    out = []
    d = np.diff(np.concatenate([[0], ai.astype(np.int8), [0]]))
    for k0, k1 in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1):
        out.append((int(k0), int(k1), bool(abs(erpm_at[k0]) < A.LaunchBoost.START_ERPM)))
    return out


def replay_off(log, data=None):
    """記録を手を入れたあとの Assist(OFF)に流す → 1 フレームごとの (出した値, 記録の out, 比べたか, 種類)"""
    d = data or load(log)
    A_, V, I, p, ref = d['A'], d['V'], d['I'], d['params'], d['ref']
    at = A_['timestamps_us'].astype(np.int64)
    vt = V['timestamps_us'].astype(np.int64)
    it = I['timestamps_us'].astype(np.int64)
    vi = np.searchsorted(vt, at, side='right') - 1
    erpm = np.where(vi >= 0, V['erpm'][np.clip(vi, 0, None)], 0).astype(np.float64)
    hg = np.hypot(I['accel'][:, 0], I['accel'][:, 1]) / 9.80665
    lat = np.nan_to_num(A_['latency_ms'].astype(np.float64), nan=0.0)
    helper = A.Assist(ref)
    n = len(at)
    sim = np.full((n, 2), np.nan)
    kind = np.full(n, -1, np.int16)
    cmp_ = np.zeros(n, bool)
    lo, hi = 1000.0, 2000.0
    for k in range(n):
        if A_['hold'][k]:
            helper.launch_hold()
            helper.reset()
            continue
        t_proc = at[k] + int(lat[k] * 1000)
        j1 = np.searchsorted(it, t_proc, side='right')
        js = [j for j in range(max(j1 - 10, 0), j1) if it[j] > at[k] - 150_000]
        impact = float(max((hg[j] for j in js), default=0.0))
        ttc = float(A_['ttc'][k])
        mo = (float(A_['model_out'][k, 0]), float(A_['model_out'][k, 1]))
        o, kd = helper.step(mo, float(erpm[k]), bool(A_['select'][k]) and not bool(A_['dry_run'][k]), t_proc / 1e6,
                            esc_on=int(p.get('esc_reverse', 1)) == 1, rec_on=int(p.get('recovery', 1)) == 1,
                            impact_g=impact, rev_us=float(p.get('rev_us', A.DEFAULT_REV_US)),
                            rev_s=float(p.get('rev_s', A.DEFAULT_REV_S)), kick_on=int(p.get('start_kick', 0) or 0) == 1,
                            blocked=bool(np.isfinite(ttc) and ttc < 1.0), launch=None)
        sim[k] = (np.clip(o[0], lo, hi), np.clip(o[1], lo, hi))
        kind[k] = kd
        changed_later = bool(A_['guard_applied'][k]) or ('cap_applied' in A_ and bool(A_['cap_applied'][k])) \
            or ('ceil_applied' in A_ and bool(A_['ceil_applied'][k]))
        cmp_[k] = not changed_later and not bool(A_['dry_run'][k])
    rec = A_['out'].astype(np.float32)
    same = np.all(sim.astype(np.float32) == rec, axis=1) & (kind == A_['assist'].astype(np.int16))
    return {'t_us': at, 'sim': sim, 'rec': rec, 'cmp': cmp_, 'same': same, 'kind': kind, 'erpm': erpm,
            'sections': sections(A_, erpm)}


def start_check(d, k0, erpm):
    """記録の値で、発進の前進の始める条件を数える(最初の AI のフレーム k0)"""
    A_ = d['A']
    hold = A_['hold'].astype(bool)
    human = (~A_['select'].astype(bool) | A_['dry_run'].astype(bool)) & ~hold
    prev_h = np.nonzero(human[:k0])[0]
    armed = len(prev_h) > 0
    held = bool(armed and hold[prev_h[-1] + 1:k0].any()) if armed else False
    nh = [k for k in range(k0, -1, -1) if not hold[k]]
    last5 = []
    for k in nh:                                    # 中立で止めたフレームで空にする(走行ループと同じ)
        if k < k0 and hold[k + 1:k0 + 1].any():
            break
        last5.append(float(A_['ttc'][k]))
        if len(last5) == 5:
            break
    ttc_med = float(np.median(last5)) if len(last5) == 5 and np.all(np.isfinite(last5)) else None
    vt = d['V']['timestamps_us'].astype(np.int64)
    j = np.searchsorted(vt, A_['timestamps_us'][k0], side='right') - 1
    veh_age_ms = float((A_['timestamps_us'][k0] - vt[j]) / 1000.0) if j >= 0 else None
    why = ''
    if not armed:
        why = 'no_switch'
    elif held:
        why = 'hold'
    elif not abs(erpm[k0]) < A.LaunchBoost.START_ERPM:
        why = 'moving'
    elif veh_age_ms is None or veh_age_ms >= 100.0:
        why = 'veh'
    elif ttc_med is None:
        why = 'ttc_n'
    elif ttc_med < A.LaunchBoost.TTC_START_S:
        why = 'near'
    return {'armed': armed, 'held': held, 'erpm': float(erpm[k0]), 'ttc5': [round(x, 2) for x in last5],
            'ttc_med': None if ttc_med is None else round(ttc_med, 2), 'veh_age_ms': veh_age_ms, 'why_not': why,
            'would_start': why == ''}


# ---------------------------------------------------------------- 閉ループの見積もり
def profile(d, k0, t_len=8.0):
    """記録の発進(k0 から)の、車輪の距離 x に対するモデルの出力・舵・ヨー・前の空き"""
    A_, V, I = d['A'], d['V'], d['I']
    c, n = d['ref']
    at = A_['timestamps_us'].astype(np.int64)
    t0 = at[k0]
    vt = (V['timestamps_us'].astype(np.int64) - t0) / 1e6
    v = np.maximum(-V['erpm'].astype(np.float64) * KV, 0.0)
    m = (vt >= 0) & (vt <= t_len)
    tv, vv = vt[m], v[m]
    x_t = np.concatenate([[0.0], np.cumsum(0.5 * (vv[1:] + vv[:-1]) * np.diff(tv))])
    ka = np.nonzero((at >= t0) & (at <= t0 + int(t_len * 1e6)) & ~A_['hold'].astype(bool))[0]
    ta = (at[ka] - t0) / 1e6
    xa = np.interp(ta, tv, x_t)
    vi = np.searchsorted(V['timestamps_us'].astype(np.int64), at[ka], side='right') - 1
    v_g = np.maximum(-V['erpm'][np.clip(vi, 0, None)].astype(np.float64) * KV, 0.0)
    ttc = A_['ttc'][ka].astype(np.float64)
    x_free = np.where(np.isfinite(ttc), ttc * np.maximum(v_g, 0.3), 9.0)
    it = (I['timestamps_us'].astype(np.int64) - t0) / 1e6
    gz = I['gyro'][:, 2].astype(np.float64)
    gz5 = np.array([float(np.mean(gz[max(j - 5, 0):j])) if j > 0 else 0.0
                    for j in np.searchsorted(it, ta, side='right')])
    xi = np.maximum.accumulate(xa + np.arange(len(xa)) * 1e-9)          # 止まっている間も増える列にする
    return {'x': xi, 't': ta, 'f_model': n - A_['model_out'][ka, 1].astype(np.float64),
            'steer_rel': A_['out'][ka, 0].astype(np.float64) - c - float(d['params'].get('steer_trim_us', 0.0)),
            'x_free': x_free, 'gz5': gz5, 'f_out': n - A_['out'][ka, 1].astype(np.float64), 'v_rec': np.interp(ta, tv, vv)}


def _at(prof, key, x, t_stop):
    """距離 x の値。止まっている間(x がほぼ 0)は時刻 t_stop の値"""
    if x < 0.02:
        j = int(np.searchsorted(prof['t'], t_stop, side='right') - 1)
        return float(prof[key][max(min(j, len(prof['t']) - 1), 0)])
    return float(np.interp(x, prof['x'], prof[key]))


def run_closed(car, prof, d, x_end, launch_on, ttc5_start, armed, t_end=6.0):
    """手を入れたあとの Assist を 30 Hz で回し、車のモデルで速さ・距離を出す → (着いた時刻, そのときの速さ, 終わりの理由, 出した秒)"""
    c, n = d['ref']
    p = d['params']
    helper = A.Assist(d['ref'])
    if armed:
        helper.step((c, n), 0.0, False, -0.1)
    nsteps = int(t_end / DT) + 1
    v, x = 0.0, 0.0
    cmds = []
    vs = []                                      # 100 Hz の車両の値 (t, v)
    ttc_hist = list(ttc5_start)
    t_next = 0.0
    end_code, n_launch, t_launch_end = 0, 0, None
    for k in range(nsteps):
        t = k * DT
        vs.append((t, v))
        if t >= t_next - 1e-9:
            t_next += 1.0 / FPS
            last3 = sorted(q for _, q in vs[-3:])
            v3 = last3[len(last3) // 2]
            old = [q for tt, q in vs if tt <= t - 0.1][-3:]
            a = (v3 - sorted(old)[len(old) // 2]) / 0.1 if old else float('nan')
            xf = _at(prof, 'x_free', x, t)
            ttc = xf / max(v, 0.3)
            if k > 0:
                ttc_hist = (ttc_hist + [ttc])[-5:]
            thr_model = n - _at(prof, 'f_model', x, t)
            steer_rel = _at(prof, 'steer_rel', x, t)
            li = A.LaunchIn(v=v3, a=a, gz=_at(prof, 'gz5', x, t), steer_rel=steer_rel, ttc=ttc, ttc5=tuple(ttc_hist),
                            tof_ok=True, veh_ok=True, **LAUNCH) if launch_on else None
            out, kind = helper.step((c + steer_rel, thr_model), -v / KV, True, t,
                                    esc_on=int(p.get('esc_reverse', 1)) == 1, rec_on=int(p.get('recovery', 1)) == 1,
                                    kick_on=int(p.get('start_kick', 0) or 0) == 1, blocked=ttc < 1.0, launch=li)
            if kind == A.ASSIST_LAUNCH:
                n_launch += 1
            if helper.launch_end and not end_code:
                end_code, t_launch_end = helper.launch_end, t
            cmds.append((t, n - float(np.clip(out[1], 1000.0, 2000.0))))
        tq = t - car.delay
        f = 0.0
        for tc, fc in reversed(cmds):
            if tc <= tq + 1e-9:
                f = fc
                break
        acc = car.accel(f, v)
        v_new = max(v + acc * DT, 0.0)
        x_new = x + 0.5 * (v + v_new) * DT
        if x_new >= x_end:
            frac = (x_end - x) / max(x_new - x, 1e-9)
            return t + frac * DT, v + frac * (v_new - v), end_code, n_launch / FPS
        v, x = v_new, x_new
    return None, v, end_code, n_launch / FPS


def estimate(sel_logs=None):
    sim = _sim()
    L, fit = sim.load()
    sj = json.loads((DIAG / 'out' / 'sim.json').read_text())
    cm = sj['car_model']
    p5 = [cm['alpha_mps2_per_us'], cm['f0_us'], cm['beta_per_s'], cm['gamma_mps2'], cm['delay_s']]
    rows = []
    cache = {}
    for r in L:
        if r.get('x_lh_m') is None or not (1.0 <= r['x_lh_m'] <= 8.0) or not r.get('tgrid'):
            continue
        key = (r['log'], r['no'])
        main = r['model'] == 'B' and r['session'].startswith('10/24') and r.get('loss_s') is not None \
            and 0.8 <= r['gain'] <= 1.0
        wide = r.get('loss_s') is not None and 0.2 < r['loss_s'] < 1.2
        if not (main or wide):
            continue
        if sel_logs is not None and key not in sel_logs:
            continue
        if r['log'] not in cache:
            cache[r['log']] = load(r['log'])
        d = cache[r['log']]
        lab = json.loads((LAB / (r['log'] + '.json')).read_text())
        at = d['A']['timestamps_us'].astype(np.int64)
        k0 = int(np.argmin(np.abs(at - (int(lab['t0_us']) + int(round(r['t_ai'] * 1e6))))))
        vt = d['V']['timestamps_us'].astype(np.int64)
        vi = np.searchsorted(vt, at, side='right') - 1
        erpm = np.where(vi >= 0, d['V']['erpm'][np.clip(vi, 0, None)], 0).astype(np.float64)
        sc = start_check(d, k0, erpm)
        prof = profile(d, k0)
        g = r['tgrid']
        tg, fg, vg = np.array(g['t']), np.nan_to_num(np.array(g['f_out'], float)), np.array(g['v'], float)
        ratio = sim.c_ratio(fit, r['session'])
        row = {'log': r['log'], 'no': r['no'], 'model': r['model'], 'session': r['session'], 'gain': r['gain'],
               'main_B_1024': bool(main), 'x_lh_m': r['x_lh_m'], 'loss_s': r.get('loss_s'), 'start': sc, 'res': {}}
        ttc5 = [x if x is not None else float('nan') for x in sc['ttc5']][::-1]
        for cap_name, cap in (('a7', cm['a_cap_mps2']), ('a5', 5.0), ('a9', 9.0)):
            car = sim.Car(p5, ratio, cap)
            ts, vs_, xs = sim.sim_time_indexed(car, tg, fg, vg[0], 5.0)
            kk = np.searchsorted(xs, r['x_lh_m'])
            t_base = float(ts[min(kk, len(ts) - 1)])
            t_off, v_off, _, _ = run_closed(car, prof, d, r['x_lh_m'], False, ttc5, sc['armed'])
            t_on, v_on, code, s_on = run_closed(car, prof, d, r['x_lh_m'], True, ttc5, sc['armed'])
            row['res'][cap_name] = {'t_base': round(t_base, 3), 't_off': None if t_off is None else round(t_off, 3),
                                    't_on': None if t_on is None else round(t_on, 3),
                                    'save_vs_base': None if t_on is None else round(t_base - t_on, 3),
                                    'save_vs_off': None if (t_on is None or t_off is None) else round(t_off - t_on, 3),
                                    'v_lh_off': round(v_off, 2), 'v_lh_on': round(v_on, 2), 'end_code': code,
                                    'end': A.LAUNCH_END_NAMES.get(code, ''), 'launch_s': round(s_on, 3)}
        rows.append(row)
    return rows


def summarize(rows, sel, cap='a7'):
    xs = [r for r in rows if sel(r)]
    out = {'n': len(xs)}
    for k in ('save_vs_base', 'save_vs_off', 'launch_s'):
        v = [r['res'][cap][k] for r in xs if r['res'][cap][k] is not None]
        if v:
            out[k] = {'median': round(float(np.median(v)), 3), 'min': round(float(np.min(v)), 3),
                      'max': round(float(np.max(v)), 3)}
    dv = [r['res'][cap]['v_lh_on'] - r['res'][cap]['v_lh_off'] for r in xs]
    if dv:
        out['dv_lh_on_minus_off'] = {'median': round(float(np.median(dv)), 2), 'max': round(float(np.max(dv)), 2)}
    ends = {}
    for r in xs:
        ends[r['res'][cap]['end']] = ends.get(r['res'][cap]['end'], 0) + 1
    out['end_reasons'] = ends
    out['would_start_recorded'] = sum(1 for r in xs if r['start']['would_start'])
    out['why_not_recorded'] = {}
    for r in xs:
        w = r['start']['why_not']
        if w:
            out['why_not_recorded'][w] = out['why_not_recorded'].get(w, 0) + 1
    return out


def main():
    OUT.mkdir(exist_ok=True)
    res = {'created_by': 'experiments/20261026_fix/launch/replay_launch.py', 'launch': LAUNCH, 'off_replay': [],
           'same_as_diag_sim': check_same_as_sim()}
    # 1. OFF の再生(T13)
    for log, no in B9:
        d = load(log)
        r = replay_off(log, d)
        win = np.zeros(len(r['t_us']), bool)
        for k0, k1, stopped in r['sections']:
            if stopped:
                win |= (r['t_us'] >= r['t_us'][k0]) & (r['t_us'] <= r['t_us'][k0] + int(WINDOW_S * 1e6))
        c_all, c_win = r['cmp'], r['cmp'] & win
        bad = np.nonzero(c_all & ~r['same'])[0]
        res['off_replay'].append({'log': log, 'no': no, 'frames': int(len(r['t_us'])), 'compared': int(c_all.sum()),
                                  'same': int((c_all & r['same']).sum()), 'launch_window_compared': int(c_win.sum()),
                                  'launch_window_same': int((c_win & r['same']).sum()),
                                  'stopped_sections': int(sum(1 for s in r['sections'] if s[2])),
                                  'first_diff': [{'k': int(k), 'rec': r['rec'][k].tolist(), 'sim': r['sim'][k].round(2).tolist(),
                                                  'kind_sim': int(r['kind'][k])} for k in bad[:5]]})
        print('OFF', log, no, res['off_replay'][-1]['same'], '/', res['off_replay'][-1]['compared'],
              'window', res['off_replay'][-1]['launch_window_same'], '/', res['off_replay'][-1]['launch_window_compared'])
    # 2・3. 始めたか・縮む時間
    rows = estimate()
    res['per_launch'] = rows
    res['summary'] = {cap: {'B_1024': summarize(rows, lambda r: r['main_B_1024'], cap),
                            'wide_0p2_1p2': summarize(rows, lambda r: r['loss_s'] is not None and 0.2 < r['loss_s'] < 1.2, cap)}
                      for cap in ('a7', 'a5', 'a9')}
    (OUT / 'replay_launch.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for r in rows:
        if r['main_B_1024']:
            q = r['res']['a7']
            print('No.%s %s start=%s(%s ttc_med=%s) base=%.3f off=%s on=%s save_base=%s save_off=%s end=%s %.2fs vLH %.2f→%.2f'
                  % (r['no'], r['log'], r['start']['would_start'], r['start']['why_not'], r['start']['ttc_med'],
                     q['t_base'], q['t_off'], q['t_on'], q['save_vs_base'], q['save_vs_off'], q['end'], q['launch_s'],
                     q['v_lh_off'], q['v_lh_on']))
    print(json.dumps(res['summary'], ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
