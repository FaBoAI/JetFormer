"""票 A: sweep_A.json(自分の閉じたループ)を数える。あわせて指摘の rd の出力(exp11_rd_*.json。指摘の台本は流さず、
出力の数だけ自分で数え直す)の偏り・差の大きさ・距離ごとの数。
    python -I -B ana_A.py → 標準出力・out/ana_A.json
"""
import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TH = 0.1


def binom_two_sided(k, n):
    """符号の検定(p = 0.5)の両側 p"""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    s = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * s)


def cls(von, voff):
    if von is None and voff is None:
        return 'neither'
    if voff is None:
        return 'on_only'
    if von is None:
        return 'off_only'
    if von > voff + TH:
        return 'on_faster'
    if voff > von + TH:
        return 'off_faster'
    return 'same'


def summarize(pairs):
    """pairs = [(von, voff, d)]"""
    c = collections.Counter(cls(a, b) for a, b, _ in pairs)
    dif_on = [a - b for a, b, _ in pairs if cls(a, b) == 'on_faster']
    dif_off = [b - a for a, b, _ in pairs if cls(a, b) == 'off_faster']
    on_only = [a for a, b, _ in pairs if cls(a, b) == 'on_only']
    off_only = [b for a, b, _ in pairs if cls(a, b) == 'off_only']
    k, n = c['on_faster'], c['on_faster'] + c['off_faster']

    def st(x):
        if not x:
            return None
        x = sorted(x)
        return {'n': len(x), 'mean': round(sum(x) / len(x), 3), 'median': round(x[len(x) // 2], 3), 'max': round(x[-1], 3)}
    # 当たったら 0 m/s より悪い: ON/OFF の当たりの速さの和(当たらない = 0)と、速さの 2 乗の和(当たりのエネルギーに比例)
    s_on = sum(a or 0.0 for a, _, _ in pairs)
    s_off = sum(b or 0.0 for _, b, _ in pairs)
    e_on = sum((a or 0.0) ** 2 for a, _, _ in pairs)
    e_off = sum((b or 0.0) ** 2 for _, b, _ in pairs)
    hits_on = sum(1 for a, _, _ in pairs if a is not None)
    hits_off = sum(1 for _, b, _ in pairs if b is not None)
    return {'n': len(pairs), 'counts': dict(c), 'sign_test_on_vs_off_faster': [k, n - k, round(binom_two_sided(k, n), 4)],
            'on_minus_off_when_on_faster': st(dif_on), 'off_minus_on_when_off_faster': st(dif_off),
            'on_only_speed': st(on_only), 'off_only_speed': st(off_only), 'hits_on': hits_on, 'hits_off': hits_off,
            'mean_hit_speed_on': round(s_on / max(len(pairs), 1), 3), 'mean_hit_speed_off': round(s_off / max(len(pairs), 1), 3),
            'mean_v2_on': round(e_on / max(len(pairs), 1), 3), 'mean_v2_off': round(e_off / max(len(pairs), 1), 3)}


def band(d):
    return '%.1f-%.1f' % (math.floor(d * 2) / 2, math.floor(d * 2) / 2 + 0.5)


def mine():
    J = json.loads((HERE / 'out' / 'sweep_A.json').read_text())
    ds = J['ds']
    rows = J['rows']
    out = {}
    # 0) リカバリーの前は ON と OFF が同じか(位置・速さ・書いた値の列の sha)
    bad = [(r['esc'], r['m'], r['xf'], r['brake_us'], r['veh_dt']) for r in rows
           if r['on']['rec'] is None or r['off']['rec'] is None or r['on']['pre_sha'] != r['off']['pre_sha']
           or r['on']['rec'][0] != r['off']['rec'][0]]
    out['pre_recovery_differs'] = bad
    out['front_hits_in_rec'] = sum(1 for r in rows for sw in ('on', 'off') if r[sw].get('front_hits'))
    allp, by_esc, by_band = [], collections.defaultdict(list), collections.defaultdict(list)
    near_by_esc = collections.defaultdict(list)
    backs = collections.defaultdict(list)
    for r in rows:
        backs[r['esc']].append((r['on']['back'], r['off']['back'], r['on']['vpk'], r['off']['vpk']))
        for d in ds:
            a, b = r['on']['hit'][str(d)], r['off']['hit'][str(d)]
            p = (a, b, d)
            allp.append(p)
            by_esc[r['esc']].append(p)
            by_band[band(d)].append(p)
            if d < 1.0:
                near_by_esc[r['esc']].append(p)
    out['all'] = summarize(allp)
    out['by_band'] = {k: summarize(v) for k, v in sorted(by_band.items())}
    out['by_esc'] = {k: summarize(v) for k, v in sorted(by_esc.items())}
    out['near_lt1m_by_esc'] = {k: summarize(v) for k, v in sorted(near_by_esc.items())}
    out['near_lt1m_all'] = summarize([p for p in allp if p[2] < 1.0])
    out['back_m'] = {k: {'on_mean': round(sum(x[0] for x in v) / len(v), 3), 'off_mean': round(sum(x[1] for x in v) / len(v), 3),
                         'on_gt_off': sum(1 for x in v if x[0] > x[1] + 0.01), 'off_gt_on': sum(1 for x in v if x[1] > x[0] + 0.01),
                         'n': len(v), 'vpk_on_mean': round(sum(x[2] for x in v) / len(v), 3),
                         'vpk_off_mean': round(sum(x[3] for x in v) / len(v), 3)} for k, v in sorted(backs.items())}
    # 見張りだけ(keep 0)は ON と同じか
    g = [(r['esc'], r['m'], r['xf'], r['brake_us'], r['veh_dt']) for r in rows if 'guard' in r
         and (r['guard']['hit'] != r['on']['hit'] or r['guard']['back'] != r['on']['back'])]
    out['guard_only_differs_from_on'] = g
    # 指摘の例(人に寄せた ESC・前進 120・壁 2.4・手前 0.45・ブレーキ側 200・100 Hz)を距離ごとに
    ex = next(r for r in rows if r['esc'] == 'human' and r['m'] == 120.0 and r['xf'] == 2.4 and r['brake_us'] == 200.0
              and r['veh_dt'] == 0.01)
    out['example_human'] = {'back_on': ex['on']['back'], 'back_off': ex['off']['back'], 'vpk_on': ex['on']['vpk'],
                            'vpk_off': ex['off']['vpk'], 'x0': ex['on']['x0'],
                            'by_d': {str(d): [ex['on']['hit'][str(d)], ex['off']['hit'][str(d)]] for d in ds},
                            'phases_on': ex['on']['phases'], 'phases_off': ex['off']['phases']}
    ex = next(r for r in rows if r['esc'] == 'fit' and r['m'] == 120.0 and r['xf'] == 2.4 and r['brake_us'] == 200.0
              and r['veh_dt'] == 0.01)
    out['example_fit'] = {'back_on': ex['on']['back'], 'back_off': ex['off']['back'], 'vpk_on': ex['on']['vpk'],
                          'vpk_off': ex['off']['vpk'], 'x0': ex['on']['x0'],
                          'by_d': {str(d): [ex['on']['hit'][str(d)], ex['off']['hit'][str(d)]] for d in ds},
                          'phases_on': ex['on']['phases'], 'phases_off': ex['off']['phases']}
    return out


def theirs():
    """指摘の rd の出力(読むだけ)。後ろの物の距離は壁から(sd.rear)。最初の後ろの当たりの速さ"""
    D = HERE.parent / 'review_safety_r1' / 'out'
    pairs, by_rear, by_esc_coast = [], collections.defaultdict(list), collections.defaultdict(list)
    for f in sorted(D.glob('exp11_rd_*of3.json')):
        J = json.loads(f.read_text())
        for r in J['rows']:
            sd = r['sd']
            if sd.get('rear') is None:
                continue

            def first(m):
                return abs(m['rear'][0][2]) if m['rear'] else None
            p = (first(r['on']), first(r['off']), sd['rear'])
            pairs.append(p)
            by_rear[sd['rear']].append(p)
            by_esc_coast['%s_coast%s' % (sd['esc'], sd.get('coast'))].append(p)
    return {'all': summarize(pairs), 'by_rear_from_wall': {str(k): summarize(v)['counts'] for k, v in sorted(by_rear.items())},
            'by_esc_coast': {k: summarize(v) for k, v in sorted(by_esc_coast.items())}}


def main():
    res = {'mine': mine()}
    try:
        res['theirs_rd'] = theirs()
    except Exception as ex:     # noqa: BLE001
        res['theirs_rd'] = repr(ex)
    (HERE / 'out' / 'ana_A.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    m = res['mine']
    print('pre-recovery differs:', m['pre_recovery_differs'], ' front hits in rec:', m['front_hits_in_rec'])
    print('guard-only differs from ON:', m['guard_only_differs_from_on'])
    print('== mine all:', json.dumps(m['all'], ensure_ascii=False))
    print('== mine near (<1.0 m from stop):', json.dumps(m['near_lt1m_all'], ensure_ascii=False))
    for k, v in m['by_band'].items():
        print('  band', k, v['counts'], 'sign', v['sign_test_on_vs_off_faster'], 'ON+', v['on_minus_off_when_on_faster'],
              'OFF+', v['off_minus_on_when_off_faster'], 'v2 on/off', v['mean_v2_on'], v['mean_v2_off'])
    print('== by esc (all d):')
    for k, v in m['by_esc'].items():
        nb = m['near_lt1m_by_esc'][k]
        b = m['back_m'][k]
        print('  %-16s all %s sign %s | v2 on %.2f off %.2f | near<1m %s | back on %.3f off %.3f (on>off %d off>on %d)' % (
            k, v['counts'], v['sign_test_on_vs_off_faster'], v['mean_v2_on'], v['mean_v2_off'], nb['counts'],
            b['on_mean'], b['off_mean'], b['on_gt_off'], b['off_gt_on']))
    for name in ('example_human', 'example_fit'):
        e = m[name]
        print('==', name, 'x0', e['x0'], 'back on/off', e['back_on'], e['back_off'], 'peak on/off', e['vpk_on'], e['vpk_off'])
        print('   phases on ', e['phases_on'])
        print('   phases off', e['phases_off'])
        print('   ', ' '.join('%s:%s/%s' % (d, *v) for d, v in e['by_d'].items() if float(d) <= 1.6))
    t = res['theirs_rd']
    if isinstance(t, dict):
        print('== theirs rd all:', json.dumps(t['all'], ensure_ascii=False))
        print('   by rear (from wall):')
        for k, v in t['by_rear_from_wall'].items():
            print('    ', k, v)
        for k, v in t['by_esc_coast'].items():
            print('   ', k, v['counts'], 'v2 on/off', v['mean_v2_on'], v['mean_v2_off'])
    else:
        print('theirs:', t)


if __name__ == '__main__':
    main()
