"""反証 V-drive: OFF(speed_ceiling = speed_auto = 0)の走行ループが、変更前の drive.py(写し)とビット単位で同じか。
実車の記録(10/24)の入力の列 + 床の基準あり(ToF ガードが動く)+ 電圧・途切れ・説明欄・既定でない値の組で回す。
    jf-heavy --lock none --mem 1500M -- python experiments/20261025_speed_control/verify/verify_off.py [log ...]
"""
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261025_speed_control/verify')
import vdrive_harness as H  # noqa: E402
from jetformer import drive  # noqa: E402

PB = {'steer_gain': 1.0, 'throttle_gain': 0.9, 'steer_trim_us': 0.0, 'lookahead': 3, 'osc_percent': 0.0, 'esc_reverse': 1,
      'recovery': 1, 'rev_us': 300.0, 'rev_s': 0.4, 'tof_guard': 1, 'speed_cap': 0.0, 'speed_scale': 1.0, 'tcp_mode': 0,
      'ensemble_j': 0, 'ensemble_m': 0.5, 'zone_caps': 0, 'start_kick': 1}
TIMING = {'fps', 'latency_ms', 'infer_ms', 'zone_ms', 'zone_ms_max'}


def vb_good(s, k):
    return 8.25 - 0.002 * s + 0.03 * math.sin(k * 0.7)


def vb_weird(s, k):
    if (k // 37) % 5 == 0:
        return None                       # 途切れ(bit5 = 0)
    if (k // 200) % 3 == 0:
        return 0.5                        # 分圧の外れ(0.5 V)
    if (k // 200) % 3 == 1:
        return 12.75                      # 張り付き
    return float('nan')                   # vbat_ok True なのに NaN(壊れた解読)


def compare(r0, r1, name):
    bad = []
    if r0['pca'].pairs != r1['pca'].pairs:
        n = next(i for i, (a, b) in enumerate(zip(r0['pca'].pairs, r1['pca'].pairs)) if a != b) \
            if len(r0['pca'].pairs) == len(r1['pca'].pairs) else 'len %d/%d' % (len(r0['pca'].pairs), len(r1['pca'].pairs))
        bad.append('PCA9685 の書き込みが違う(%s)' % n)
    if r0['pca'].single != r1['pca'].single:
        bad.append('PCA9685 の 1 ch の書き込みが違う')
    a, b = r0['rec'].rows, r1['rec'].rows
    if len(a) != len(b):
        bad.append('記録の行の数 %d/%d' % (len(a), len(b)))
    else:
        for (t0, x0), (t1, x1) in zip(a, b):
            if t0 != t1:
                bad.append('行の時刻')
                break
            for k, v in x0.items():
                if k == 'infer_ms':
                    continue
                if not H.same(v, x1.get(k, '<なし>')):
                    bad.append('欄 %s @%d: %r / %r' % (k, t0, v, x1.get(k)))
                    break
            if len(bad) > 3:
                break
    ev0 = [(k, json.dumps(v, sort_keys=True, default=str)) for k, v in r0['rec'].events]
    ev1 = [(k, json.dumps(v, sort_keys=True, default=str)) for k, v in r1['rec'].events]
    if ev0 != ev1:
        bad.append('出来事が違う: %s' % [e for e in ev1 if e not in ev0][:3])
    s0, s1 = r0['st'].get('stats', {}), r1['st'].get('stats', {})
    for k, v in s0.items():
        if k not in TIMING and s1.get(k) != v:
            bad.append('stats %s %r/%r' % (k, v, s1.get(k)))
    if r0['st'].get('laps') != r1['st'].get('laps'):
        bad.append('周回の数が違う')
    for k in ('message', 'recording_note', 'hold_reason'):
        if r0['st'].get(k) != r1['st'].get(k):
            bad.append('st[%s] %r / %r' % (k, r0['st'].get(k), r1['st'].get(k)))
    rows = [x for _, x in b]
    n_ai = sum(1 for x in rows if not x['hold'])
    n_hold = len(rows) - n_ai
    n_guard = sum(1 for x in rows if x.get('guard_applied'))
    n_assist = sum(1 for x in rows if x.get('assist'))
    sh = sum(1 for x in rows if x.get('ceil_mode') in (1, 2))
    err = r1['st'].get('speed_ctrl_error')
    print('%-46s %s  行 %d(中立止め %d・ガード %d・補助 %d・天井の影 %d)err=%s' % (
        name, 'OK  ' if not bad else 'NG  ', len(rows), n_hold, n_guard, n_assist, sh, err))
    for x in bad[:6]:
        print('     -', x)
    return not bad


def main(logs):
    before = H.load_before()
    ok = True
    for log in logs:
        variants = [
            ('B 既定(電圧なし)', PB, None, (), False, None, False),
            ('B + 電圧 8.2V + V_before', PB, H.vb_good if False else vb_good, (), False, 'V_before=8.30 V_after=__', False),
            ('ガード 3・区間 ON・trim+22・電圧おかしい', dict(PB, tof_guard=3, zone_caps=1, steer_trim_us=22.0, lookahead=1,
                                                   throttle_gain=1.1), vb_weird, (), False, 'V_before=5.0', False),
            ('途切れ(車両 0.12/0.2 秒・IMU・ToF)', PB, vb_good,
             (('VEHICLE', 20.0, 20.12), ('VEHICLE', 40.0, 40.2), ('IMU', 55.0, 55.15), ('TOF', 70.0, 70.3),
              (None, 90.0, 90.6)), False, None, False),
            ('dry run', PB, vb_good, (), True, None, False),
            ('OFF で値だけ既定でない', dict(PB, ceil_all_mps=4.0, ceil_left_mps=3.5, auto_target_mps=4.4, auto_vref_v=7.0),
             vb_weird, (), False, None, False),
            ('自動合わせ 1・天井 0(決まりを通さず)', dict(PB, speed_auto=1), vb_good, (), False, None, True),
        ]
        for name, params, vb, drop, dry, desc, raw in variants:
            t = time.time()
            src = H.LogSource(log, vbat=vb, drop=drop)
            r0 = H.run_loop(before, params, src, src.pred_of, dry_run=dry, description=desc)
            src = H.LogSource(log, vbat=vb, drop=drop)
            r1 = H.run_loop(drive, params, src, src.pred_of, dry_run=dry, description=desc)
            ok &= compare(r0, r1, '%s %s' % (log, name))
            print('     (%.0f 秒)' % (time.time() - t))
    print('全部 OK' if ok else 'NG あり')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:] or ['20261024-162033']))
