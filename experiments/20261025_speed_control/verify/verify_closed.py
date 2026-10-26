"""反証 V-drive: 閉ループ(偽物の車・sim_source)で天井・自動合わせを走行ループごと回す。
  A 天井 ON/OFF × 電池 b: 区間の最高の速さ・周の時間・出し方の入れ替わり(1 秒あたり)・記録の決まり
  B 自動合わせ ON × b × 電圧の有無: 周ごとの m_fb・p90・揺れ・張り付き
  C 途切れ・人・ブレーキの予測・板の衝撃
"""
import collections
import json
import math
import sys

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261025_speed_control/verify')
import vdrive_harness as H  # noqa: E402
from sim_source import SimSource  # noqa: E402
from jetformer import drive  # noqa: E402
from jetformer import safety_caps as sc  # noqa: E402
from verify_off import PB  # noqa: E402

NEU = 1496.0
P0 = dict(PB, throttle_gain=0.9, start_kick=0)
OUT = {}


def rows_of(r):
    return [x for _, x in r['rec'].rows]


def invariants(rows):
    bad = collections.Counter()
    for x in rows:
        if x['hold']:
            continue
        tin, tout = x['ceil_in_us'], x['out'][1]
        if tout < tin:
            bad['前進を強めた'] += 1
        if tout != tin and tout > NEU:
            bad['ブレーキを出した'] += 1
        if tin >= NEU and tout != tin:
            bad['前進でない入力を変えた'] += 1
        if x['ceil_applied'] != (tout != tin):
            bad['applied の印が違う'] += 1
        if np.isfinite(x['ceil_steer']) and abs((x['out'][0] - 1441.0 - x['trim']) - x['ceil_steer']) > 1e-3:
            bad['舵が天井の見た舵と違う'] += 1
    return dict(bad)


def chatter(rows):
    """出し方(ceil_mode 0/1/2)が入れ替わった回数 / AI の秒"""
    ms = [x['ceil_mode'] for x in rows if not x['hold'] and x['select']]
    ch = sum(1 for a, b in zip(ms, ms[1:]) if a != b and a >= 0 and b >= 0)
    lim_neu = sum(1 for a, b in zip(ms, ms[1:]) if {a, b} == {1, 2})
    return ch / max(len(ms) / 30.0, 1e-9), lim_neu / max(len(ms) / 30.0, 1e-9)


def stand_3laps(src):
    """止まった状態から 3 周: 最初に 0.2 m/s を超えた時刻 → 3 つ目の周の区切り(位置で数える)"""
    t_go = next((t for t, s, v, n in src.log if v > 0.2), None)
    laps = [t for t in src.laps if t_go is not None and t > t_go]
    # 位置の区切り 0.6 m(R1 の手前)から始めているので、最初の区切りは 1 周目の終わり
    return (laps[2] - t_go) if len(laps) >= 3 else float('nan')


def exp_A():
    print('== A 天井 ON/OFF(tg 0.9、60 秒)')
    ok = True
    for b in (0.9, 1.0, 1.15, 1.3):
        res = {}
        for on in (0, 1):
            src = SimSource(t_end_s=60.0, b=b)
            r = H.run_loop(drive, dict(P0, speed_ceiling=on), src, src.pred_of, floor_ref=None)
            rows = rows_of(r)
            inv = invariants(rows)
            chg, ln = chatter(rows) if on else (0, 0)
            lt = src.lap_times()
            res[on] = dict(sw=src.seg_max('SW', 10), lh=src.seg_max('LH', 10), hs=src.seg_max('HS', 10),
                           de=src.seg_max('DE', 10), lap=float(np.median(lt)) if len(lt) else float('nan'),
                           stand3=stand_3laps(src), app=100.0 * r['st']['stats']['ceil_applied'] / max(1, len(rows)),
                           neu=r['st']['stats']['ceil_neutral'], chg=chg, ln=ln, inv=inv, err=r['st']['speed_ctrl_error'])
            ok &= not inv and r['st']['speed_ctrl_error'] is None
        a, c = res[0], res[1]
        print('b %.2f OFF: SW %.2f LH %.2f HS %.2f DE %.2f 周 %.2f 止まってから 3 周 %.2f | ON: SW %.2f LH %.2f HS %.2f DE %.2f 周 %.2f '
              '3 周 %.2f(+%.2f)変えた %.1f%% 中立 %d 入れ替わり %.1f/秒(弱め↔中立 %.1f/秒)違反 %s' % (
                  b, a['sw'], a['lh'], a['hs'], a['de'], a['lap'], a['stand3'], c['sw'], c['lh'], c['hs'], c['de'], c['lap'],
                  c['stand3'], c['stand3'] - a['stand3'], c['app'], c['neu'], c['chg'], c['ln'], c['inv'] or 'なし'))
        OUT.setdefault('A', {})[str(b)] = res
    return ok


def vb_consistent(b):
    v = min(8.45, max(7.3, 8.35 * b))
    return lambda t: v - 0.0006 * t + (0.05 * math.sin(t * 7.0))


def exp_B():
    print('== B 自動合わせ ON(天井 ON、tg 0.9、100 秒)')
    ok = True
    for b in (0.85, 0.9, 1.0, 1.1, 1.2, 1.3):
        for vlabel, vb in (('電圧なし', None), ('電圧 = 8.35·b', vb_consistent(b))):
            src = SimSource(t_end_s=100.0, b=b, vbat=vb)
            r = H.run_loop(drive, dict(P0, speed_ceiling=1, speed_auto=1), src, src.pred_of, floor_ref=None)
            rows = rows_of(r)
            laps = [(x['sa_lap_p90'], x['sa_lap_s'], x['sa_m_fb'], int(x['sa_event'])) for x in rows
                    if np.isfinite(x['sa_lap_s'])]
            mfb = [l[2] for l in laps]
            d = np.abs(np.diff(np.log([1.0] + mfb)))
            g = [x['gains'][1] for x in rows if not x['hold']]
            end = r['st']['auto_end']
            signs = [np.sign(b_ - a_) for a_, b_ in zip([1.0] + mfb, mfb) if abs(b_ - a_) > 1e-9]
            flips = sum(1 for a_, b_ in zip(signs, signs[1:]) if a_ != b_)
            last = [l[0] for l in laps[-3:]]
            print('b %.2f %-10s 周 %2d p90 %s | m_fb %s | Δ最大 %.3f 向きの入れ替わり %d ゲイン %.3f〜%.3f 終わり p90 %.2f 知らせ %s 違反 %s' % (
                b, vlabel, len(laps), ' '.join('%.2f' % l[0] for l in laps), ' '.join('%.3f' % m for m in mfb),
                d.max() if len(d) else 0, flips, min(g), max(g), np.mean(last) if last else float('nan'),
                end.get('warn_seen'), invariants(rows) or 'なし'))
            ok &= (len(d) == 0 or d.max() <= 0.03 + 1e-6) and max(g) <= 1.10 + 1e-9 and not invariants(rows)
            OUT.setdefault('B', {})['%s %s' % (b, vlabel)] = dict(p90=[l[0] for l in laps], m_fb=mfb, ev=[l[3] for l in laps],
                                                                 gmin=min(g), gmax=max(g), warn=end.get('warn_seen'))
    return ok


def exp_C():
    print('== C 途切れ・人・ブレーキ(天井 ON・自動合わせ ON、b 1.15、50 秒)')
    ok = True
    # SW は 1 周の中で ~ 位置 26〜30 m。速さから時刻はずれるので、範囲を広めに何回か入れる
    cases = [
        ('車両の値 0.12 秒途切れ(×6)', dict(drop=[('VEHICLE', t, t + 0.12) for t in (12.3, 19.6, 27.0, 33.4, 40.1, 46.0)])),
        ('IMU 0.15 秒途切れ(中立止め)', dict(drop=[('IMU', t, t + 0.15) for t in (14.0, 30.0)])),
        ('人が 0.5 秒(×4)', dict(select=lambda t: not any(a <= t < a + 0.5 for a in (13.0, 21.5, 29.0, 37.0)))),
        ('ブレーキの予測 0.4 秒(×3)', dict(brake_at=[(15.0, 15.4), (25.0, 25.4), (35.0, 35.4)])),
        ('R1 の板 7G', dict(bump_g=7.0)),
    ]
    for name, kw in cases:
        src = SimSource(t_end_s=50.0, b=1.15, vbat=vb_consistent(1.0), **kw)
        r = H.run_loop(drive, dict(P0, speed_ceiling=1, speed_auto=1), src, src.pred_of, floor_ref=None)
        rows = rows_of(r)
        inv = invariants(rows)
        ev = collections.Counter(int(x['sa_event']) for x in rows)
        human_gain = [x['gains'][1] for x in rows if not x['select'] and not x['hold']]
        print('%-28s SW 最高 %.2f LH %.2f DE %.2f 中立止め %d 出来事 %s 人のゲイン %s 違反 %s err %s end %s' % (
            name, src.seg_max('SW', 8), src.seg_max('LH', 8), src.seg_max('DE', 8), r['st']['stats']['holds'], dict(ev),
            sorted(set(human_gain)) or '-', inv or 'なし', r['st']['speed_ctrl_error'],
            {k: r['st']['auto_end'].get(k) for k in ('m_fb_end', 'laps_used', 'laps_skipped')}))
        ok &= not inv and r['st']['speed_ctrl_error'] is None and all(g == 0.9 for g in human_gain)
    return ok


if __name__ == '__main__':
    which = sys.argv[1:] or ['A', 'B', 'C']
    ok = True
    for w in which:
        ok &= {'A': exp_A, 'B': exp_B, 'C': exp_C}[w]()
    json.dump(OUT, open('/home/jetson/jetformer/experiments/20261025_speed_control/verify/closed_%s.json' % ''.join(which), 'w'),
              default=float, ensure_ascii=False, indent=1)
    print('全部 OK' if ok else 'NG あり')
