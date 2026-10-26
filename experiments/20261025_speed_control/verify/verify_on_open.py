"""反証 V-drive: ON の走行ループ(記録の入力の列・open loop・床の基準あり)。
  天井 ON: 舵は OFF と同じ・スロットルは OFF 以上(前進を強めない)・中立以下(ブレーキを出さない)か OFF と同じ。
           前進でない(ブレーキ・後退・中立)フレームは OFF と同じ。人のフレーム・中立止めは OFF と同じ。
  自動合わせ ON: 舵は同じ・後退側は同じ・前進の量の比は 0.85〜1.20・掛けたゲインは max(tg, 1.10) 以下。
  電圧がおかしいとき(0.5 V・12.75 V・途切れ)のゲイン。
"""
import collections
import math
import sys

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261025_speed_control/verify')
import vdrive_harness as H  # noqa: E402
from jetformer import drive  # noqa: E402
from verify_off import PB, vb_good  # noqa: E402

NEU = 1496.0


def rows(r):
    return [x for _, x in r['rec'].rows]


def check_ceiling(log, params, vb=None, desc=None, label=''):
    src = H.LogSource(log, vbat=vb)
    off = H.run_loop(drive, params, src, src.pred_of, description=desc)
    src = H.LogSource(log, vbat=vb)
    on = H.run_loop(drive, dict(params, speed_ceiling=1), src, src.pred_of, description=desc)
    a, b = rows(off), rows(on)
    assert len(a) == len(b)
    bad = collections.Counter()
    n_app = n_neu = n_human_diff = 0
    for x, y in zip(a, b):
        so, to = x['out']
        sn, tn = y['out']
        if sn != so:
            bad['舵が変わった'] += 1
        if tn < to:
            bad['前進を強めた'] += 1
        if tn > NEU and tn != to:
            bad['ブレーキ側に出した'] += 1
        if to >= NEU and tn != to:
            bad['前進でないフレームを変えた'] += 1
        if (not y['select'] or y['hold'] or y['dry_run']) and tn != to:
            bad['人・中立止めのフレームを変えた'] += 1
        if y['ceil_applied'] != (tn != to):
            bad['ceil_applied が出力の違いと合わない'] += 1
        n_app += tn != to
        n_neu += tn == NEU and to != NEU
    pairs_ok = all(p1[0] == p0[0] for p0, p1 in zip(off['pca'].pairs, on['pca'].pairs))
    print('%s %s 天井 ON: 変えた %d(%.1f%%)・中立にした %d・PCA の舵は同じ %s・違反 %s・stats ceil_applied %s' % (
        log, label, n_app, 100.0 * n_app / len(b), n_neu, pairs_ok, dict(bad) or 'なし',
        on['st']['stats'].get('ceil_applied')))
    return not bad and pairs_ok


def check_auto(log, params, vb, desc=None, label=''):
    src = H.LogSource(log, vbat=vb)
    off = H.run_loop(drive, dict(params, speed_ceiling=1), src, src.pred_of, description=desc)
    src = H.LogSource(log, vbat=vb)
    on = H.run_loop(drive, dict(params, speed_ceiling=1, speed_auto=1), src, src.pred_of, description=desc)
    a, b = rows(off), rows(on)
    tg = params['throttle_gain']
    bad = collections.Counter()
    gains = [y['gains'][1] for y in b if not y['hold']]
    ms = [y['sa_m'] for y in b if not y['hold'] and np.isfinite(y['sa_m'])]
    srcs = collections.Counter(int(y['sa_src']) for y in b if not y['hold'])
    for x, y in zip(a, b):
        if y['out'][0] != x['out'][0]:
            bad['舵が変わった'] += 1
        if y['gains'][1] > max(tg, 1.10) + 1e-9 or y['gains'][1] < tg * 0.85 - 1e-9:
            bad['ゲインが範囲の外'] += 1
        if (not y['select'] or y['dry_run']) and y['gains'][1] != tg:
            bad['人のフレームにゲインを掛けた'] += 1
        mo, mn = NEU - x['model_out'][1], NEU - y['model_out'][1]
        if mo <= 0 and mn != mo:
            bad['後退・ブレーキ側を変えた'] += 1
    dm = np.abs(np.diff(ms)) if len(ms) > 1 else np.array([0.0])
    ev = collections.Counter(int(y['sa_event']) for y in b)
    print('%s %s 自動合わせ ON: ゲイン %.3f〜%.3f(tg %.2f)・m %.3f〜%.3f・1 フレームの m の変化の最大 %.4f・源 %s・出来事 %s・違反 %s・end %s' % (
        log, label, min(gains), max(gains), tg, min(ms), max(ms), dm.max(), dict(srcs), dict(ev), dict(bad) or 'なし',
        {k: on['st']['auto_end'].get(k) for k in ('m_fb_end', 'laps_used', 'laps_skipped', 'warn')}))
    return not bad, (min(gains), max(gains))


if __name__ == '__main__':
    log = sys.argv[1] if len(sys.argv) > 1 else '20261024-162033'
    ok = True
    ok &= check_ceiling(log, PB, label='B')
    ok &= check_ceiling(log, dict(PB, tof_guard=3, zone_caps=1, steer_trim_us=22.0), vb_good, label='ガード 3・区間 ON')
    for name, vb, desc in (('電圧 8.2', vb_good, None), ('電圧なし・V_before 7.8', None, 'V_before=7.80'),
                           ('電圧 0.5 V(分圧の外れ)', lambda s, k: 0.5, None),
                           ('電圧 3.0 V', lambda s, k: 3.0, None),
                           ('電圧 12.75 V(張り付き)', lambda s, k: 12.75, None),
                           ('電圧 50 秒で途切れる', lambda s, k: 8.3 if s < 50 else None, None),
                           ('電圧 7.4→8.4 と跳ねる', lambda s, k: 7.4 if (k // 300) % 2 else 8.4, None)):
        o, g = check_auto(log, PB, vb, desc, label=name)
        ok &= o
    print('全部 OK' if ok else 'NG あり')
