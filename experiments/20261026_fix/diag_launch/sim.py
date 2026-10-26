"""原因 C の診断(3): 簡単な車のモデルと記録の再生で「発進の案で何秒縮むか」を見積もる。2026-10-26

読むだけ。車・走行の API には触れない。

1. 車のモデル(まっすぐ・前後だけ): v' = clip(α_s·(f(t − δ) − f0)⁺ − β·v − γ, −∞, a_cap)、v ≥ 0。
   f = 車に出した前進 [µs]、α_s = α × (回の電池の効き c_s ÷ 基準)(c_s は fit_dyn.py の回ごとの値)。
   α・f0・β・γ・δ を、記録の発進(AI 側にしてから LH の入口まで、最大 2.5 秒)の車輪の速さに合わせる。
   a_cap(加速の上限)は IMU の前の加速から(低い速さ・指令 250 µs 以上の 90% 点)。感度として 5・9 m/s² も出す。
2. 確かめ: 記録の指令(時刻の順のまま)を入れて、LH の入口に着く時刻を記録と比べる。
3. 案の再生: 発進ごとに、LH の入口(周の場所で LH が始まる距離)に着くまでの時間を、
     基準(記録の指令のまま)と案(下の規則)でシミュレーションし、差を「縮む時間」とする。
   LH〜R1 の起点は、記録でも発進と周でほぼ同じ時間(失う時間の内訳 lh_to_r1 ≈ 0)なので、案でも変わらないとみなす。
   案のあとのモデルの指令は、記録の発進の「同じ場所(発進からの距離)」の指令(model_out・ゲイン後)。
   別の見方として、周の同じ場所の指令(f_out_fly)でも計算する。
4. 案(どれも既定 OFF の新しい選択肢の候補):
     K0       今の発進の補助(記録のまま)
     KICK150  補助を 150 µs に。モデルのブレーキ(止まっている間)も補助で上書き。0.4 m/s で終わり
     L(U, vt) 発進の前進: AI 側にした最初のフレームから、前進を少なくとも U µs(モデルが強ければモデル)。
              速さ vt に届く・1.2 秒・モデルの舵が ±150 µs を超える、のどれかで終わり。止まっている間のモデルのブレーキは無視。
     IDEAL    加速の上限 a_cap で走り、周の速さに届いたら周の速さ(下限の目安)
5. 直線の長さ L ごとの見積もり(予選のスタートの線が分からないので): 止まってから L m を走る時間 − 周の速さで L m を走る時間。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I sim.py    # → out/sim.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
DT = 0.01
FPS = 30.0
T_MAX_FIT = 2.5
LAUNCH_T_MAX = 1.2
LAUNCH_STEER_END = 150.0
KICK_END_V = 0.4
U_LIST = (80, 150, 200, 250, 300)
VT_LIST = (2.0, 2.5, 3.0, 3.5)


def load():
    L = json.loads((OUT / 'launches.json').read_text())['launches']
    fit = json.loads((OUT / 'dyn_fit.json').read_text())
    return L, fit


def c_ratio(fit, session):
    c = fit['best']['c_mps_per_us']
    ref = c['10/24 3回目']
    return c.get(session, ref) / ref


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


def sim_time_indexed(car, t_cmd, f_cmd, v0, t_end):
    """記録の指令(フレームの時刻 t_cmd・値 f_cmd)を時刻の順に入れる → 時刻・速さ・距離"""
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


def sim_policy(car, policy, prof, v0, x_end, t_end=6.0):
    """案の規則で走らせる。prof = 発進の距離の格子(x・モデルの指令 f_model(x)・舵 s(x)・周の指令 f_fly(x)・周の速さ v_fly(x))。
    指令はフレームごと(30 Hz)に決め、δ 遅れて車に効く。戻り値: x_end に着いた時刻・その時の速さ・各時刻の (t, v, x, f)"""
    xs, fm, sm = prof['x'], prof['f_model'], prof['s']
    n = int(t_end / DT) + 1
    v, x = v0, 0.0
    cmds = []                    # (時刻, f)
    active, ended = True, False
    t_next_frame = 0.0
    hist = []
    f_now = 0.0
    for k in range(n):
        t = k * DT
        if t >= t_next_frame - 1e-9:
            t_next_frame += 1.0 / FPS
            fmod = float(np.interp(x, xs, fm))
            smod = float(np.interp(x, xs, sm))
            kind = policy['kind']
            if kind == 'kick':
                if active and v >= KICK_END_V:
                    active = False
                f_now = max(policy['U'], fmod) if active else fmod
            elif kind == 'launch':
                if active and (v >= policy['vt'] or t >= LAUNCH_T_MAX or abs(smod) > LAUNCH_STEER_END):
                    active = False
                f_now = max(policy['U'], fmod) if active else fmod
            else:
                f_now = fmod
            cmds.append((t, f_now))
        # 遅れ δ の指令
        tq = t - car.delay
        f = 0.0
        for tc, fc in reversed(cmds):
            if tc <= tq + 1e-9:
                f = fc
                break
        if policy['kind'] == 'ideal':
            vf = float(np.interp(x, xs, prof['v_fly']))
            if t < car.delay:
                v_new = v
            elif v < vf:
                v_new = min(v + car.a_cap * DT, vf)
            else:
                v_new = vf
        else:
            a = car.accel(f, v)
            v_new = max(v + a * DT, 0.0)
        x_new = x + 0.5 * (v + v_new) * DT
        hist.append((t, v, x, f))
        if x_new >= x_end:
            frac = (x_end - x) / max(x_new - x, 1e-9)
            return t + frac * DT, v + frac * (v_new - v), hist
        v, x = v_new, x_new
    return None, v, hist


def fit_launch_model(L, fit, delay):
    rows = []
    for r in L:
        if not r.get('tgrid') or r.get('x_lh_m') is None or not (1.0 <= r['x_lh_m'] <= 8.0):
            continue                # LH の手前の直線が 1〜8 m の発進だけ(R1 から 1 周走った発進などは外す)
        g = r['tgrid']
        t = np.array(g['t'])
        t_end = min(T_MAX_FIT, r['t_launch_to_lh'], t[-1])
        if t_end < 0.5:
            continue
        rows.append((r, t, np.array(g['f_out'], float), np.array(g['v'], float), t_end, c_ratio(fit, r['session'])))

    def resid(q):
        res = []
        for r, t, f, v, t_end, ratio in rows:
            car = Car([q[0], q[1], q[2], q[3], delay], ratio)
            ts, vs, xs = sim_time_indexed(car, t, np.nan_to_num(f), v[0], t_end)
            res.append(np.interp(t[t <= t_end], ts, vs) - v[t <= t_end])
        return np.concatenate(res)
    q0 = [0.03, 30.0, 1.0, 0.2]
    sol = least_squares(resid, q0, bounds=([0.001, -50, 0.0, -1.0], [0.2, 120, 5.0, 3.0]))
    rmse = float(np.sqrt(np.mean(sol.fun ** 2)))
    return sol.x, rmse, rows


def main():
    L, fit = load()
    best = None
    for delay in (0.033, 0.067, 0.1):
        q, rmse, rows = fit_launch_model(L, fit, delay)
        print('delay %.3f rmse %.3f m/s  alpha %.4f f0 %.1f beta %.3f gamma %.3f' % (delay, rmse, *q))
        if best is None or rmse < best[1]:
            best = (q, rmse, rows, delay)
    q, rmse, rows, delay = best
    p = [q[0], q[1], q[2], q[3], delay]
    a_cap = fit.get('a_imu_p90_low_speed_f250') or 7.0
    out = {'created_by': 'experiments/20261026_fix/diag_launch/sim.py',
           'car_model': {'alpha_mps2_per_us': round(q[0], 5), 'f0_us': round(q[1], 1), 'beta_per_s': round(q[2], 3),
                         'gamma_mps2': round(q[3], 3), 'delay_s': delay, 'rmse_v_mps': round(rmse, 3),
                         'a_cap_mps2': a_cap, 'ref_session': '10/24 3回目', 'n_launches_fit': len(rows)}}
    # 高い指令での確かめ(周の中の加速の表と比べる)
    car = Car(p, 1.0)
    chk = []
    for b in fit['bins']:
        if b['f'][0] >= 150 and b['n'] >= 30:
            fm, vm = (b['f'][0] + b['f'][1]) / 2, (b['v'][0] + b['v'][1]) / 2
            chk.append({'v': b['v'], 'f': b['f'], 'a_imu_med': b['a_imu_med'], 'a_imu_p90': b['a_imu_p90'],
                        'model_a': round(min(car.accel(fm, vm), a_cap), 2), 'n': b['n']})
    out['check_bins'] = chk
    # ---- 確かめ: 記録の指令で LH の入口に着く時刻
    val = []
    for r, t, f, v, t_end, ratio in rows:
        car = Car(p, ratio, a_cap)
        ts, vs, xs = sim_time_indexed(car, t, np.nan_to_num(f), v[0], 4.0)
        k = np.searchsorted(xs, r['x_lh_m'])
        t_sim = float(ts[min(k, len(ts) - 1)])
        val.append({'log': r['log'], 'no': r['no'], 't_rec': r['t_launch_to_lh'], 't_sim': round(t_sim, 3),
                    'err': round(t_sim - r['t_launch_to_lh'], 3)})
    out['validation'] = val
    errs = np.array([x['err'] for x in val])
    out['validation_summary'] = {'n': len(val), 'median_err_s': round(float(np.median(errs)), 3),
                                 'mae_s': round(float(np.mean(np.abs(errs))), 3), 'max_abs_s': round(float(np.max(np.abs(errs))), 3)}
    print('validation', out['validation_summary'])
    # ---- 案の再生
    policies = [{'name': 'K0', 'kind': 'none'}, {'name': 'KICK150', 'kind': 'kick', 'U': 150.0},
                {'name': 'KICK250', 'kind': 'kick', 'U': 250.0}]
    for U in U_LIST:
        for vt in VT_LIST:
            policies.append({'name': 'L%d_%.1f' % (U, vt), 'kind': 'launch', 'U': float(U), 'vt': vt})
    policies.append({'name': 'IDEAL', 'kind': 'ideal'})
    per = []
    for r in L:
        if not r.get('grid') or r.get('x_lh_m') is None or not (1.0 <= r['x_lh_m'] <= 8.0):
            continue
        gr = r['grid']
        x = np.array(gr['x'])
        prof_model = {'x': x, 'f_model': np.nan_to_num(np.array(gr['f_model_launch'], float)),
                      's': np.nan_to_num(np.array(gr['s_out_launch'], float)), 'v_fly': np.array(gr['v_fly'], float)}
        prof_fly = dict(prof_model, f_model=np.nan_to_num(np.array(gr['f_out_fly'], float)))
        ratio = c_ratio(fit, r['session'])
        g = r['tgrid']
        t, f, v = np.array(g['t']), np.nan_to_num(np.array(g['f_out'], float)), np.array(g['v'], float)
        row = {'log': r['log'], 'no': r['no'], 'model': r['model'], 'gain': r['gain'], 'session': r['session'],
               'x_lh_m': r['x_lh_m'], 'loss_s': r.get('loss_s'), 't_rec_to_lh': r['t_launch_to_lh'],
               't_fly_to_lh': r['t_fly_to_lh'], 'v_fly_at_lh': r['v_fly_at_lh'], 'res': {}}
        for cap_name, cap in (('a7', a_cap), ('a5', 5.0), ('a9', 9.0)):
            car = Car(p, ratio, cap)
            ts, vs, xs = sim_time_indexed(car, t, f, v[0], 5.0)
            k = np.searchsorted(xs, r['x_lh_m'])
            t_base = float(ts[min(k, len(ts) - 1)])
            res = {'t_base_sim': round(t_base, 3)}
            for pol in policies:
                if pol['kind'] == 'none':
                    continue
                for pname, prof in (('model', prof_model), ('fly', prof_fly)):
                    if pname == 'fly' and pol['kind'] == 'ideal':
                        continue
                    tt, v_lh, hist = sim_policy(car, pol, prof, v[0], r['x_lh_m'])
                    if tt is None:
                        continue
                    key = pol['name'] + ('' if pname == 'model' else '@fly')
                    res[key] = {'t': round(tt, 3), 'save': round(t_base - tt, 3), 'v_lh': round(v_lh, 2)}
            # 基準の再生を「同じ場所の指令」でもしておく(時刻の順の再生との差 = 再生のやり方の差)
            tt, v_lh, _ = sim_policy(car, {'kind': 'none'}, prof_model, v[0], r['x_lh_m'])
            res['K0_loc'] = {'t': round(tt, 3) if tt else None, 'v_lh': round(v_lh, 2)}
            row['res'][cap_name] = res
        per.append(row)
    out['per_launch'] = per

    # まとめ: 10/24 の B(ゲイン 0.85〜1.00)・HS から・周の比べがあるもの、と 全部
    def summarize(sel, cap='a7'):
        S = {}
        for pol in policies[1:]:
            for suf in ('', '@fly'):
                key = pol['name'] + suf
                vals = [x['res'][cap][key]['save'] for x in sel if key in x['res'][cap]]
                vlh = [x['res'][cap][key]['v_lh'] - x['v_fly_at_lh'] for x in sel if key in x['res'][cap]]
                dk = [x['res'][cap][key]['v_lh'] - x['res'][cap]['K0_loc']['v_lh'] for x in sel if key in x['res'][cap]]
                if vals:
                    S[key] = {'n': len(vals), 'save_median': round(float(np.median(vals)), 3),
                              'dv_lh_vs_K0_median': round(float(np.median(dk)), 2),
                              'dv_lh_vs_K0_max': round(float(np.max(dk)), 2),
                              'save_min': round(float(np.min(vals)), 3), 'save_max': round(float(np.max(vals)), 3),
                              'v_lh_minus_fly_median': round(float(np.median(vlh)), 2),
                              'v_lh_over_fly_n': int(np.sum(np.array(vlh) > 0.3))}
        return S
    sel_b24 = [x for x in per if x['model'] == 'B' and x['session'].startswith('10/24') and x['loss_s'] is not None
               and 0.8 <= x['gain'] <= 1.0]
    sel_all = [x for x in per if x['loss_s'] is not None and x['loss_s'] < 1.2 and x['loss_s'] > 0.2]
    out['summary'] = {'B_1024': {cap: summarize(sel_b24, cap) for cap in ('a7', 'a5', 'a9')},
                      'all_0p2_1p2': {cap: summarize(sel_all, cap) for cap in ('a7', 'a5', 'a9')},
                      'B_1024_logs': [(x['log'], x['no']) for x in sel_b24],
                      'all_logs': [(x['log'], x['no']) for x in sel_all]}

    # ---- 直線の長さ L ごと(10/24 3 回目の電池・ゲイン 0.9 のモデルの典型の指令)
    #   モデルの指令: 10/24 の B の発進の、時刻ごとの中央値(止まってから)。周の速さ v_fly = 3.6 m/s(No.9・3 回目の HS の中ごろ)
    Bt = [x for x in L if x.get('tgrid') and x['model'] == 'B' and x['session'] == '10/24 3回目' and len(x['tgrid']['t']) >= 60]
    T = np.arange(0, 2.0, 1 / FPS)
    fmed = np.nanmedian(np.stack([np.interp(T, x['tgrid']['t'], np.nan_to_num(np.array(x['tgrid']['f_model'], float))) for x in Bt]), axis=0)
    fout = np.nanmedian(np.stack([np.interp(T, x['tgrid']['t'], np.nan_to_num(np.array(x['tgrid']['f_out'], float))) for x in Bt]), axis=0)
    out['typical_model_cmd_1024_s3'] = {'t': T[::3].round(2).tolist(), 'f_model': fmed[::3].round(0).tolist(),
                                        'f_out': fout[::3].round(0).tolist(), 'n': len(Bt)}
    tab = []
    T6 = np.arange(0, 6.0, 1 / FPS)
    for v_fly in (3.2, 3.6, 4.0):
        # 長い直線では、1.0 秒からあとは「周の速さを保つ指令」f_hold 以上をモデルが出すとみなす(周の直線の指令に近い)
        f_hold = (p[2] * v_fly + p[3]) / p[0] + p[1]
        fm6 = np.interp(T6, T, fmed)
        fo6 = np.interp(T6, T, fout)
        fm6 = np.where(T6 >= 1.0, np.maximum(fm6, f_hold), fm6)
        fo6 = np.where(T6 >= 1.0, np.maximum(fo6, f_hold), fo6)
        for Ls in (1.5, 2.5, 4.0, 6.0, 8.0):
            xg = np.arange(0, Ls + 0.5, 0.1)
            car = Car(p, 1.0, a_cap)
            # 基準: 典型の指令を時刻の順に(記録の補助を含む f_out)
            ts, vs, xs = sim_time_indexed(car, T6, fo6, 0.0, 6.0)
            k = np.searchsorted(xs, Ls)
            t_base = float(ts[min(k, len(ts) - 1)])
            row = {'v_fly': v_fly, 'L': Ls, 'f_hold': round(f_hold, 1), 'fly_s': round(Ls / v_fly, 3),
                   'loss_K0': round(t_base - Ls / v_fly, 3)}
            # 案: モデルの指令は時刻の中央値を「同じ場所」に直したもの(基準の再生の x(t) で)
            fm_x = np.interp(xg, np.interp(T6, ts, xs), fm6)
            prof = {'x': xg, 'f_model': fm_x, 's': np.zeros_like(xg), 'v_fly': np.full_like(xg, v_fly)}
            for pol in policies[1:]:
                tt, vl, _ = sim_policy(car, pol, prof, 0.0, Ls)
                if tt is not None:
                    row['loss_' + pol['name']] = round(tt - Ls / v_fly, 3)
            tab.append(row)
    out['straight_table'] = tab
    (OUT / 'sim.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out['car_model'], ensure_ascii=False))
    for c in chk:
        print('check', c)
    for cap in ('a7', 'a5', 'a9'):
        print('== B_1024', cap)
        for k, s in out['summary']['B_1024'][cap].items():
            print('  %-14s %s' % (k, s))
    print('== all a7')
    for k, s in out['summary']['all_0p2_1p2']['a7'].items():
        print('  %-14s %s' % (k, s))
    for r in tab:
        print(r)


if __name__ == '__main__':
    main()
