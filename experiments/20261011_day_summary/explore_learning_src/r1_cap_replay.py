"""開ループの再生: 「モデル自身の未来ステア予測が右いっぱいに近づいたら速度に上限」を掛けていたら、
R1 の手前で間に合ったか。CPUのみ・読み取りのみ(AI走行 006/007、AI側だけ)。
- R1 の開始: ジャイロ z(+ = 左)を 50ms で平均し、-170 deg/s 以下になった最初の時刻。直前 3 秒に同じものが無いこと
  (collision_map.md §1.3 と同じ考え方。深さの条件は省略)。
- 進入速度: R1 開始の 0.4 秒前〜開始の最大 |eRPM|(collision_map.md と同じ)。
- 衝突: 水平 2.5G 以上が R1 開始の -0.2〜+1.2 秒にある(強い = 4G 以上)。
- 合図: AI_CMD.pred(ゲイン前、10フレーム先まで)のどれかのステアが TH_US 以下(右)になった、R1 開始前 1.5 秒以内の最初の時刻。
- 誤報: R1 開始の前後 1.5 秒の外で合図が出た回数(AI側・前進中)。
出力: r1_cap_replay.json(このフォルダ)
"""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.expanduser('~/jetformer_reports/summary_20261010/src'))
import runs_table as rt
G = 9.80665
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r1_cap_replay.json')
LOGS = {'20261010-172910': 'ai_run_006', '20261010-173445': 'ai_run_007'}
CAP_ERPM = 5500.0
KV = 0.5e-3
res = {'definitions': __doc__, 'cap_erpm': CAP_ERPM, 'runs': {}}
for TH in (1150.0, 1250.0):
    for log, name in LOGS.items():
        meta, ok, broken, missing, sj, data = rt.load_log(log)
        it = rt.cat(data['IMU'], 'timestamps_us'); acc = rt.cat(data['IMU'], 'accel'); gyr = rt.cat(data['IMU'], 'gyro')
        vt = rt.cat(data['VEHICLE'], 'timestamps_us'); erpm = rt.cat(data['VEHICLE'], 'erpm').astype(float)
        at = rt.cat(data['AI_CMD'], 'timestamps_us'); pred = rt.cat(data['AI_CMD'], 'pred'); sel = rt.cat(data['AI_CMD'], 'select')
        o = np.argsort(it); it, acc, gyr = it[o], acc[o], gyr[o]
        o = np.argsort(vt); vt, erpm = vt[o], erpm[o]
        o = np.argsort(at); at, pred, sel = at[o], pred[o], sel[o]
        gz = np.convolve(gyr[:, 2], np.ones(5) / 5, 'same')
        hg = np.hypot(acc[:, 0], acc[:, 1]) / G
        # AI側か(AI_CMD の select を IMU 時刻へ)
        k = np.clip(np.searchsorted(at, it) - 1, 0, len(at) - 1)
        ai_imu = sel[k] & (np.abs(it - at[k]) < 0.2e6)
        starts = []
        last = -1e18
        for i in np.nonzero((gz <= -170) & ai_imu)[0]:
            if it[i] - last > 3.0e6:
                starts.append(it[i])
            last = it[i]
        min_pred = pred[:, :, 0].min(axis=1)    # 10フレーム先までで最も右のステア(ゲイン前)
        sig = (min_pred <= TH) & sel
        passes = []
        for t0 in starts:
            w = (vt >= t0 - 0.4e6) & (vt <= t0)
            e_in = float(np.max(np.abs(erpm[w]))) if w.any() else None
            wi = (it >= t0 - 0.2e6) & (it <= t0 + 1.2e6)
            g = float(hg[wi].max()) if wi.any() else 0.0
            ws = np.nonzero(sig & (at >= t0 - 1.5e6) & (at <= t0))[0]
            lead = (t0 - at[ws[0]]) / 1e6 if len(ws) else None
            e_sig = None
            if len(ws):
                kv = np.clip(np.searchsorted(vt, at[ws[0]]) - 1, 0, len(vt) - 1)
                e_sig = float(abs(erpm[kv]))
            need = None
            if e_in is not None and lead:
                need = max(0.0, (e_in - CAP_ERPM) * KV) / lead    # m/s^2 で、開始までに上限へ落とすのに要る減速
            passes.append({'t_s': round((t0 - at[0]) / 1e6, 2), 'entry_erpm': e_in, 'max_g': round(g, 1),
                           'hit': g >= 2.5, 'hit_strong': g >= 4.0, 'signal_lead_s': None if lead is None else round(lead, 2),
                           'erpm_at_signal': e_sig, 'decel_needed_ms2': None if need is None else round(need, 2)})
        # 誤報: R1 窓の外での合図の立ち上がり(0.3秒以内はまとめる)
        kv = np.clip(np.searchsorted(vt, at) - 1, 0, len(vt) - 1)
        moving = np.abs(erpm[kv]) > 1000
        inwin = np.zeros(len(at), bool)
        for t0 in starts:
            inwin |= (at >= t0 - 1.5e6) & (at <= t0 + 1.5e6)
        on = sig & moving & ~inwin
        onsets, lastt = 0, -1e18
        for i in np.nonzero(on)[0]:
            if at[i] - lastt > 0.3e6:
                onsets += 1
            lastt = at[i]
        dt = np.clip(np.diff(at, append=at[-1]), 0, 0.1e6) / 1e6
        ai_min = float((dt * (sel & moving)).sum() / 60)
        P = [p for p in passes]
        fast = [p for p in P if p['entry_erpm'] and p['entry_erpm'] >= CAP_ERPM]
        summ = {'r1_passes': len(P), 'hits_2.5g': sum(p['hit'] for p in P), 'hits_4g': sum(p['hit_strong'] for p in P),
                'passes_entry_ge_cap': len(fast), 'hits_4g_among_ge_cap': sum(p['hit_strong'] for p in fast),
                'signal_found': sum(p['signal_lead_s'] is not None for p in P),
                'signal_lead_s_median': float(np.median([p['signal_lead_s'] for p in P if p['signal_lead_s'] is not None])) if any(p['signal_lead_s'] is not None for p in P) else None,
                'decel_needed_ms2_on_ge_cap': sorted([p['decel_needed_ms2'] for p in fast if p['decel_needed_ms2'] is not None]),
                'false_signal_onsets': onsets, 'ai_moving_min': round(ai_min, 2),
                'false_signal_per_min': round(onsets / ai_min, 1) if ai_min else None}
        res['runs'].setdefault(f'TH{int(TH)}', {})[name] = {'summary': summ, 'passes': P}
        print('TH', TH, name, json.dumps(summ, ensure_ascii=False))
json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
