# 確認 line r3(再実行): 自前の走らせ方の結果を集計する。判定の式は protocol.json acceptance.line・variation_metrics と
# protocol_line_r3.json acceptance・definitions_added の文言から書いた(学習・評価・前の確認の集計コードは使わない)
import json, glob, sys, math, collections, statistics

OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3b'


def load(pat):
    rs = []
    for p in sorted(glob.glob(f'{OUT}/runs/{pat}')):
        for l in open(p):
            if l.strip():
                rs.append(json.loads(l))
    return rs


def q(a, p):
    if not a:
        return None
    b = sorted(a)
    i = p * (len(b) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return b[lo] + (b[hi] - b[lo]) * (i - lo)


def vm3(runs):
    # VM2 と同じ式で e_y: 5 cm のマスごとに、全部の回がそろうマスで平均からの |差| の平均
    if len(runs) < 2:
        return None
    nb = len(runs[0]['bins_ey'])
    vals = []
    for j in range(nb):
        col = [r['bins_ey'][j] for r in runs]
        if any(v is None for v in col):
            continue
        m = sum(col) / len(col)
        vals.append(sum(abs(v - m) for v in col) / len(col))
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def kind(r):
    ln = r['params']['line'] or {}
    f = ln.get('family')
    if f == 'const':
        c = ln.get('c0', 0)
        return 'const_left' if c >= .6 else 'const_right' if c <= -.6 else 'const_center' if abs(c) <= .2 else 'const_other'
    return f


def summ(rs):
    n = len(rs)
    vr = [r for r in rs if not r.get('invalid_spawn') and r['valid_r3']]
    trk = [x / 1e4 for r in vr for x in r['trk_all']]
    col = [r for r in rs if r.get('collisions', 0) > 0]
    stl = [r for r in rs if r.get('stall')]
    lt3 = [r for r in vr if r['min_c'] < .03]
    nolap = [r for r in rs if r.get('valid') and not r.get('stall') and r['valid_laps'] < 1]
    notvalid = [r for r in rs if not r.get('valid')]
    sh = [r for r in rs if r.get('sh_frames', 0) > 0]
    lab = collections.Counter()
    for r in rs:
        for k, v in r.get('labels', {}).items():
            if k != 'bytes_max':
                lab[k] += v
    out = {
        'n': n, 'valid_r3': len(vr), 'collisions': len(col), 'stalls': len(stl), 'below_3cm_valid': len(lt3), 'valid_but_no_lap': len(nolap),
        'not_valid_reasons': dict(collections.Counter(r.get('reason') for r in notvalid)),
        'min_clearance_all': min((r['min_c'] for r in rs if 'min_c' in r), default=None),
        'min_clearance_valid_r3': min((r['min_c'] for r in vr), default=None),
        'tracking_median': q(trk, .5), 'tracking_p95': q(trk, .95), 'tracking_frames': len(trk),
        'shield_runs': len(sh), 'shield_frames': sum(r.get('sh_frames', 0) for r in rs),
        'shield_fraction_all': sum(r.get('sh_frames', 0) for r in rs) / max(1, sum(r.get('frames', 0) for r in rs)),
        'shield_gt5pct': [r['id'] for r in rs if r.get('sh_frac', 0) > .05], 'shield_gt10pct': [r['id'] for r in rs if r.get('sh_frac', 0) > .10],
        'lap1_median': q([r['lap1'] for r in vr if r['lap1']], .5),
        'v_run_mean': statistics.mean([r['v_run'] for r in vr]) if vr else None,
        'integ_err_max': max((r.get('integ_err', 0) for r in rs if r.get('reason') == 'collection_duration_complete'), default=None),
        'labels': dict(lab), 'meta_shield_true': sum(1 for r in rs if r.get('meta_shield')),
        'large_heading_n': sum(1 for r in rs if abs(r['start']['dpsi']) >= 1),
        'large_heading_min_c': min((r['min_c'] for r in rs if abs(r['start']['dpsi']) >= 1 and 'min_c' in r), default=None),
    }
    if len(vr) >= 2:
        v, nb = vm3(vr)
        out['vm3'] = v
        out['vm3_bins'] = nb
    return out


def fails(rs):
    out = []
    for r in rs:
        bad = []
        if r.get('invalid_spawn'):
            bad.append('invalid_spawn')
        else:
            if r.get('collisions', 0) > 0:
                bad.append('collision')
            if r.get('stall'):
                bad.append('stall')
            if r.get('valid') and not r.get('stall') and r['valid_laps'] < 1:
                bad.append('no_valid_lap')
            if not r.get('valid') and r.get('collisions', 0) == 0:
                bad.append(r.get('reason'))
            if r.get('valid_r3') and r['min_c'] < .03:
                bad.append('below_3cm')
        if bad:
            out.append({'id': r['id'], 'what': bad, 'gate': r['gate'], 'off': round(r['off'], 4), 'yaw': round(r['yaw'], 3), 'dpsi': r['start']['dpsi'],
                        'c_start': r['start']['clearance'], 'line': r['params']['line'], 'c_v': r['params']['c_v'], 'min_c': r.get('min_c'), 'min_id': r.get('min_id'),
                        'min_s_mod': r.get('min_s_mod'), 'collision': r.get('collision'), 'stall_first': r.get('stall_first'), 'sh_frames': r.get('sh_frames'),
                        'fb': r.get('fb'), 'laps': [l['t'] for l in r.get('laps', [])], 'reason': r.get('reason')})
    return out


if __name__ == '__main__':
    pat = sys.argv[1] if len(sys.argv) > 1 else 'cand_*.jsonl'
    rs = load(pat)
    by = collections.defaultdict(list)
    for r in rs:
        by[r['set']].append(r)
    res = {'files': pat, 'n': len(rs), 'sets': {k: summ(v) for k, v in sorted(by.items())}, 'fails': {k: fails(v) for k, v in sorted(by.items())}}
    if 'U360' in by:
        u = sorted(by['U360'], key=lambda r: int(r['id'].split('-')[1]))
        res['U360_groups'] = [summ(u[i:i + 60]) for i in range(0, len(u), 60)]
        for g in res['U360_groups']:
            for k in ['labels', 'shield_gt5pct', 'shield_gt10pct']:
                g.pop(k, None)
        fam = collections.defaultdict(list)
        for r in by['U360'] + by.get('H180', []):
            if r.get('valid_r3'):
                fam[kind(r)].append(r)
        res['by_family'] = {k: {'n': len(v), 'trk_median': q([x / 1e4 for r in v for x in r['trk_all']], .5), 'trk_p95': q([x / 1e4 for r in v for x in r['trk_all']], .95),
                                'lap1_median': q([r['lap1'] for r in v if r['lap1']], .5), 'min_c': min(r['min_c'] for r in v)} for k, v in sorted(fam.items())}
    if 'CB264' in by:
        cc = collections.defaultdict(list)
        for r in by['CB264']:
            cc[r['corner']].append(r)
        res['CB_by_corner'] = {k: {kk: vv for kk, vv in summ(v).items() if kk in ('n', 'valid_r3', 'collisions', 'stalls', 'below_3cm_valid', 'min_clearance_all', 'shield_runs')} for k, v in sorted(cc.items())}
    if 'EDGE132' in by:
        cc = collections.defaultdict(list)
        for r in by['EDGE132']:
            cc[r['edge']].append(r)
        res['EDGE_by_edge'] = {k: {kk: vv for kk, vv in summ(v).items() if kk in ('n', 'valid_r3', 'collisions', 'stalls', 'below_3cm_valid', 'min_clearance_all', 'shield_runs')} for k, v in sorted(cc.items())}
    json.dump(res, open(sys.argv[2] if len(sys.argv) > 2 else f'{OUT}/summary_cand.json', 'w'), ensure_ascii=False, indent=1)
    for k, v in res['sets'].items():
        print(k, {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk not in ('labels', 'shield_gt5pct', 'shield_gt10pct')})
    for k, v in res['fails'].items():
        for f in v:
            print('  FAIL', k, json.dumps(f, ensure_ascii=False))
