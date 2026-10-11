#!/usr/bin/env python3
"""区間ごとの速度の上限・目標を、記録された周の速度に当てはめたら、1周のタイムがどれだけ変わるか(運動学の目安、CPU)

    nice -n 15 python speed_plan_sim.py > ../speed_plan_sim.json

方法: sections.py と同じ周(R1 開始で区切り、ジャイロの向き ψ で区間)について、IMU の時刻で v = −eRPM×0.5e-3 を
積分して走行距離 s を作る。速度の上限 v_lim(s) を
  区間の中: 上限値、区間の前: sqrt(上限^2 + 2 A_BRAKE (s0 − s))(手前から減速)、区間の後: sqrt(上限^2 + 2 A_ACC (s − s1))
として、v' = min(v, v_lim) で走ったときの時間 Σ ds / v' と元の時間の差を出す。
速度の目標(下限)は、区間の中で v' = max(v, min(目標, 区間の入口からの加速 sqrt(v0^2 + 2 A_ACC x), 出口までの減速 ...))。
走る道すじ・ハンドル・空転は変わらないと仮定する(= 目安)。eRPM の速度は空転を含む(±6%)。
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import AI_LOGS, HUMAN_LOGS, KV, load, prev_index  # noqa: E402

A_BRAKE, A_ACC = 2.5, 1.5
SECS = ['R1', 'GM', 'DE', 'SW', 'HS', 'LH']
SEC = json.load(open('/home/jetson/jetformer_reports/summary_20261010/explore_control/sections.json'))


def lap_profiles(log):
    _, d = load(log, ('IMU', 'VEHICLE'))
    t = d['IMU']['timestamps_us']
    ts = (t - t[0]) / 1e6
    v = np.maximum(-d['VEHICLE']['erpm'][prev_index(d['VEHICLE']['timestamps_us'], t)].astype(float) * KV, 0.05)
    out = []
    laps = [l for x in SEC['logs'] if x['log'] == log for l in x['laps'] if all(l['sec'][k] for k in SECS)]
    for l in laps:
        i0 = int(np.searchsorted(ts, l['t0']))
        i1 = int(np.searchsorted(ts, l['t0'] + l['lap_s']))
        tt = ts[i0:i1]
        vv = v[i0:i1]
        dt = np.diff(tt, append=tt[-1] + 0.01)
        s = np.concatenate([[0.0], np.cumsum(vv[:-1] * dt[:-1])])
        # 区間の時刻 → 距離
        b = [l['t0']]
        for k in SECS:
            b.append(b[-1] + l['sec'][k]['t_s'])
        sb = np.interp(b, tt, s)
        # R1 の手前(前の周の LH の最後 0.6 s)は、この周の最初に含まれないので、上限は周の最後の LH 末尾 0.6 s に掛ける
        out.append({'lap': l, 'ts': tt, 'v': vv, 's': s, 'dt': dt, 'sb': dict(zip(SECS + ['END'], sb))})
    return out


def apply_caps(p, caps):
    """caps: [(s0, s1, vcap)]。v' = min(v, v_lim)"""
    s, v = p['s'], p['v']
    lim = np.full(len(s), 99.0)
    for s0, s1, c in caps:
        inside = (s >= s0) & (s <= s1)
        lim = np.minimum(lim, np.where(inside, c, 99.0))
        before = s < s0
        lim = np.minimum(lim, np.where(before, np.sqrt(c * c + 2 * A_BRAKE * np.maximum(s0 - s, 0)), 99.0))
        after = s > s1
        lim = np.minimum(lim, np.where(after, np.sqrt(c * c + 2 * A_ACC * np.maximum(s - s1, 0)), 99.0))
    vp = np.minimum(v, lim)
    ds = np.diff(s, append=s[-1])
    return float(np.sum(ds / np.maximum(vp, 0.05))) - float(np.sum(ds / np.maximum(v, 0.05)))


def apply_target(p, s0, s1, vt):
    """区間 [s0, s1] で速度の下限を vt にする(入口からは A_ACC で加速、出口 s1 までに元の速度へ A_BRAKE で戻せる範囲)"""
    s, v = p['s'], p['v']
    k0 = int(np.searchsorted(s, s0))
    k1 = min(int(np.searchsorted(s, s1)), len(s) - 1)
    v0, v1 = v[k0], v[k1]
    floor = np.zeros(len(s))
    inside = (s >= s0) & (s <= s1)
    acc_lim = np.sqrt(v0 * v0 + 2 * A_ACC * np.maximum(s - s0, 0))
    brk_lim = np.sqrt(v1 * v1 + 2 * A_BRAKE * np.maximum(s1 - s, 0))
    floor[inside] = np.minimum(vt, np.minimum(acc_lim, brk_lim))[inside]
    vp = np.maximum(v, floor)
    ds = np.diff(s, append=s[-1])
    return float(np.sum(ds / np.maximum(vp, 0.05))) - float(np.sum(ds / np.maximum(v, 0.05)))


def combined(p, caps, s0, s1, vt):
    """目標(下限)を掛けた後に上限を掛ける: v' = min(max(v, 下限), 上限)"""
    s, v = p['s'], p['v']
    lim = np.full(len(s), 99.0)
    for a, b, c in caps:
        lim = np.minimum(lim, np.where((s >= a) & (s <= b), c, 99.0))
        lim = np.minimum(lim, np.where(s < a, np.sqrt(c * c + 2 * A_BRAKE * np.maximum(a - s, 0)), 99.0))
        lim = np.minimum(lim, np.where(s > b, np.sqrt(c * c + 2 * A_ACC * np.maximum(s - b, 0)), 99.0))
    k0 = int(np.searchsorted(s, s0))
    k1 = min(int(np.searchsorted(s, s1)), len(s) - 1)
    v0, v1 = v[k0], v[k1]
    inside = (s >= s0) & (s <= s1)
    floor = np.zeros(len(s))
    floor[inside] = np.minimum(vt, np.minimum(np.sqrt(v0 * v0 + 2 * A_ACC * np.maximum(s - s0, 0)),
                                              np.sqrt(v1 * v1 + 2 * A_BRAKE * np.maximum(s1 - s, 0))))[inside]
    vp = np.minimum(np.maximum(v, floor), lim)
    ds = np.diff(s, append=s[-1])
    return float(np.sum(ds / np.maximum(vp, 0.05))) - float(np.sum(ds / np.maximum(v, 0.05)))


def main():
    res = {'definitions': __doc__, 'A_BRAKE': A_BRAKE, 'A_ACC': A_ACC, 'groups': {}}
    groups = {'ai': list(AI_LOGS), 'human': HUMAN_LOGS}
    for g, logs in groups.items():
        rows = []
        for log in logs:
            for p in lap_profiles(log):
                l, sb = p['lap'], p['sb']
                if l['driver'] != ('ai' if g == 'ai' else 'human'):
                    continue
                r = {'log': log, 'lap_s': l['lap_s'], 'n_imp': l['n_imp']}
                # R1 進入の上限: この周の最後の LH の末尾 0.6 s(= 次の R1 の手前)と、この周の R1 区間
                end = sb['END']
                s_lh_tail = float(np.interp(p['ts'][-1] - 0.6, p['ts'], p['s']))
                for cap in (2.75, 2.5):
                    r['R1cap_%.2f' % cap] = round(apply_caps(p, [(0.0, sb['GM'], cap), (s_lh_tail, end, cap)]), 3)
                for cap in (2.2, 2.0):
                    r['GMcap_%.1f' % cap] = round(apply_caps(p, [(sb['GM'], sb['DE'], cap)]), 3)
                # DE: GM 出口(ψ = −20)から 1.5 s の間
                s_de1 = float(np.interp(p['ts'][0] + sum(l['sec'][k]['t_s'] for k in ['R1', 'GM']) + 1.5, p['ts'], p['s']))
                for cap in (3.0, 2.75):
                    r['DEcap_%.2f' % cap] = round(apply_caps(p, [(sb['DE'], s_de1, cap)]), 3)
                # LH の進入: HS 区間(ψ +175〜+215)と LH の最初の 0.5 s
                s_lh_in = float(np.interp(p['ts'][0] + sum(l['sec'][k]['t_s'] for k in ['R1', 'GM', 'DE', 'SW', 'HS']) + 0.5, p['ts'], p['s']))
                for cap in (3.0, 2.75):
                    r['LHcap_%.2f' % cap] = round(apply_caps(p, [(sb['HS'], s_lh_in, cap)]), 3)
                # 全部の上限(R1 2.75、GM 2.2、DE 3.0、LH 3.0)
                r['ALLcaps'] = round(apply_caps(p, [(0.0, sb['GM'], 2.75), (s_lh_tail, end, 2.75), (sb['GM'], sb['DE'], 2.2),
                                                   (sb['DE'], s_de1, 3.0), (sb['HS'], s_lh_in, 3.0)]), 3)
                caps_all = [(0.0, sb['GM'], 2.75), (s_lh_tail, end, 2.75), (sb['GM'], sb['DE'], 2.2),
                            (sb['DE'], s_de1, 3.0), (sb['HS'], s_lh_in, 3.0)]
                for vt in (3.0, 3.3, 3.6):
                    # 目標は DE の角の後(GM 出口 +1.5 s)から HS の始まり(ψ +175)まで。HS〜LH の入口は上限 3.0 m/s
                    r['COMBINED_caps+target_%.1f' % vt] = round(combined(p, caps_all, max(s_de1, sb['DE']), sb['HS'], vt), 3)
                # ブレーキ点を LH の入口(ψ +215)の 1.0 m 手前に置いた場合(ToF の前方距離や GM 出口からの距離で決める想定)
                s_bp = sb['LH'] - 1.0
                caps_bp = [(0.0, sb['GM'], 2.75), (s_lh_tail, end, 2.75), (sb['GM'], sb['DE'], 2.2),
                           (sb['DE'], s_de1, 3.0), (s_bp, s_lh_in, 3.0)]
                r['ALLcaps_brakepoint1m'] = round(apply_caps(p, caps_bp), 3)
                for vt in (3.0, 3.3, 3.6):
                    r['COMBINED_brakepoint1m+target_%.1f' % vt] = round(combined(p, caps_bp, max(s_de1, sb['DE']), s_bp, vt), 3)
                # 後半の目標: SW 区間(ψ +40〜+175)と HS 区間(+175〜+215)。DE の角の後から LH の手前まで
                for vt in (3.0, 3.3, 3.6):
                    r['BACKtarget_%.1f' % vt] = round(apply_target(p, sb['SW'], sb['LH'], vt), 3)
                    r['BACKtarget_from_s_de1_%.1f' % vt] = round(apply_target(p, max(s_de1, sb['DE']), sb['LH'], vt), 3)
                rows.append(r)
        keys = [k for k in rows[0] if k not in ('log', 'lap_s', 'n_imp')]
        summ = {k: {'median_s': round(float(np.median([r[k] for r in rows])), 3),
                    'p90_s': round(float(np.percentile([r[k] for r in rows], 90)), 3),
                    'mean_s': round(float(np.mean([r[k] for r in rows])), 3)} for k in keys}
        res['groups'][g] = {'n_laps': len(rows), 'summary': summ, 'laps': rows}
    json.dump(res, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
