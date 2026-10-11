"""r1_cap_replay.json の「R1 の衝突」を、collision_map.md 付録A の目視確認済みの R1 衝突と照合する。
- 007(173445)は時刻の基準が collision_map と同じ。R1 の板への正面衝突 8件: 35.12 58.19 92.99 143.17 157.78 210.43 244.30 271.11 s。
  100.17〜106.20 s は collision_map では「回復」として除外されている。
- 006(172910)は collision_map の時刻 = このログの時刻 + 20.022 s。collision_map の 006 の R1 は手前の軽い接触 4件だけ
  (55.44 83.19 102.87 139.02 s。強い衝突は 139.02 の 4.7G のみ)で、再生の窓(R1 開始の -0.2〜+1.2 s)には
  GM 入口の衝突(36.48 s)などが入ってしまうため、006 は「R1 の強い衝突」の集計から外す。
出力: r1_cap_check.json
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, 'r1_cap_replay.json')))
CM007 = [35.12, 58.19, 92.99, 143.17, 157.78, 210.43, 244.30, 271.11]
out = {}
for th, runs in d['runs'].items():
    P = runs['ai_run_007']['passes']
    matched = []
    for t in CM007:
        best = min(P, key=lambda p: abs(p['t_s'] - t))
        ok = -0.2 <= t - best['t_s'] <= 1.2
        matched.append({'cm_t': t, 'pass_t': best['t_s'], 'matched': ok, 'entry_erpm': best['entry_erpm'],
                        'signal_lead_s': best['signal_lead_s'], 'decel_needed_ms2': best['decel_needed_ms2']})
    m = [x for x in matched if x['matched']]
    leads = sorted(x['signal_lead_s'] for x in m if x['signal_lead_s'] is not None)
    fast = [x for x in m if x['entry_erpm'] >= 5500]
    out[th] = {'cm_r1_hits_007': len(CM007), 'matched': len(m), 'with_signal': len(leads),
               'lead_s_min_med_max': [leads[0], leads[len(leads) // 2] if len(leads) % 2 else (leads[len(leads)//2 - 1] + leads[len(leads)//2]) / 2, leads[-1]] if leads else None,
               'entry_ge_5500': len(fast),
               'brake_feasible_le_2.8': sum(1 for x in fast if x['decel_needed_ms2'] is not None and x['decel_needed_ms2'] <= 2.8),
               'coast_feasible_le_1.0': sum(1 for x in fast if x['decel_needed_ms2'] is not None and x['decel_needed_ms2'] <= 1.0),
               'false_signal_per_min': {k: v['summary']['false_signal_per_min'] for k, v in runs.items()},
               'events': matched}
    print(th, json.dumps({k: v for k, v in out[th].items() if k != 'events'}, ensure_ascii=False))
json.dump(out, open(os.path.join(HERE, 'r1_cap_check.json'), 'w'), ensure_ascii=False, indent=1)
