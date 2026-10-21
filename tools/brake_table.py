#!/usr/bin/env python3
"""ブレーキの量の表(U6、2026-10-21): 記録の中のブレーキ(中立より後退側)の出来事から、µs → 減速 m/s² の表を床と芝で作る

    python tools/brake_table.py --day 20261023                 # その日のログ全部 → data/reports/brake_table_20261023/
    python tools/brake_table.py 20261023-163000 --surface 芝    # 路面を指定(説明欄に surface=芝 と書いてもよい)
    python tools/brake_table.py --day 20261010 --out DIR        # 10/10 の記録(ブレーキ 23 回・きれい 16 回、006 の AI 9 回)

読み取りだけ(numpy、GPU なし)。data/raw・data/labels は変えない。ラベル(tools/label_events.py)が要る。

出来事の決まり(experiments/20261016_sim_calibration/inventory.py と同じ):
  - 指令(100 Hz の VEHICLE の時刻): 人 = VEHICLE.pwm(スティック)、AI = AI_CMD.out を「撮像の時刻 + latency_ms」から
    次の指令まで保持(100 ms より古ければ欠け。hold・補助・ガード・dry_run のフレームは使わない)
  - ブレーキ = スロットル ≥ 中立 1498 + 60 µs が 0.1 秒以上続き、始まりの速さ ≥ 1.0 m/s
  - 速さ vs = −eRPM × 0.5e-3 の 9 点の移動平均、加速度 = vs の微分の 9 点の移動平均。減速 = その区間の加速度の平均の符号を逆にしたもの
  - きれい = 衝突の 1 秒前〜2 秒後・交代/戻し ±0.3 秒・基板の再起動 ±0.5 秒・hold/補助/リカバリー ±0.2 秒・指令の欠け・
    0.05 秒を超える途切れ を含まない
表:
  - 路面: 説明欄の surface=床|芝|floor|turf(または --surface)。無ければ場所のラベル(GM = 芝、ほか = 床)
  - 強さ: 中立からの µs の中央値で 弱 < 150・中 150〜300・強 ≥ 300
  - 減速(合計)と、ESC のブレーキの分 a_b = 減速 − 惰性(c0 + k_s·|横加速度|、P_cal の値)。
    a_b(off) = s_b × max(0, off − db_b) の最小二乗(db_b は 0〜300 µs の 1 µs 刻み)を路面ごとに当てはめる(3 回以上のとき)
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from jetformer import config  # noqa: E402
import label_events as le  # noqa: E402

SCHEMA = 'jf-brake-table/v1'
THR_N = 1498.0
BRAKE_MIN_US = 60.0
BRAKE_MIN_S = 0.1
V_START_MIN = 1.0
KV = 0.5e-3
LOC = ['unknown', 'R1', 'GM', 'DE', 'other', 'HS', 'LH']
STRENGTH = [(0, 150, '弱'), (150, 300, '中'), (300, 1e9, '強')]
PCAL = Path('/home/jetson/jetformer_reports/sim_calibration_20261016/profiles/P_cal.json')
SURFACE_WORDS = {'床': '床', 'floor': '床', 'carpet': '床', '芝': '芝', 'turf': '芝', 'grass': '芝'}


def movavg(x, n):
    if n <= 1 or len(x) < n:
        return x
    return np.convolve(x, np.ones(n) / n, mode='same')


def interval_mask(t_us, centers_us, before_s, after_s):
    m = np.zeros(len(t_us), bool)
    for c in centers_us:
        m |= (t_us >= c - before_s * 1e6) & (t_us <= c + after_s * 1e6)
    return m


def span_mask(t_s, spans, pad_s=0.0):
    m = np.zeros(len(t_s), bool)
    for a, b in spans:
        m |= (t_s >= a - pad_s) & (t_s <= b + pad_s)
    return m


def surface_of_description(desc):
    """説明欄の surface=… か、memo の中の 床/芝 の言葉"""
    if not desc:
        return None
    m = re.search(r'surface=(\S+)', desc)
    if m and m.group(1).lower() in SURFACE_WORDS:
        return SURFACE_WORDS[m.group(1).lower()]
    m = re.search(r'memo=(.*)$', desc)
    if m:
        hits = {v for k, v in SURFACE_WORDS.items() if k in m.group(1).lower()}
        if len(hits) == 1:
            return hits.pop()
    return None


def load_labels(log, labels_dir=None, raw_dir=None):
    """ラベル(data/labels/<ログ>.json・.npz)。無ければ tools/label_events.py の label_log でその場で作る(書き出さない)"""
    labels_dir = Path(labels_dir or config.DATA_DIR / 'labels')
    pj, pn = labels_dir / (log + '.json'), labels_dir / (log + '.npz')
    if pj.exists() and pn.exists():
        info = json.loads(pj.read_text(encoding='utf-8'))
        with np.load(pn) as z:
            lab = {k: z[k] for k in z.files}
        info['labels_source'] = 'file'
        return info, lab
    info, lab = le.label_log(log, raw_dir)
    if lab is None:
        raise ValueError('ラベルのファイルが無く、その場でも作れない(フレームが無い)')
    info['labels_source'] = 'computed'
    return info, lab


def record(log, labels_dir=None, raw_dir=None):
    """100 Hz の記録(inventory.analyse_log と同じ)"""
    info, lab = load_labels(log, labels_dir, raw_dir)
    L = le.load_log(log, raw_dir)
    info['problems_now'] = L.get('problems', [])
    V, I, A = L['VEHICLE'], L['IMU'], L['AI_CMD']
    t0 = info['t0_us']
    if V is None or I is None or len(V['timestamps_us']) < 200:
        return info, None
    vt_us = V['timestamps_us'].astype(np.int64)
    vt = (vt_us - t0) / 1e6
    sel = le.clean_select(vt, V['select'].astype(bool), V['valid'][:, 2].astype(bool))
    v = -V['erpm'].astype(float) * KV
    it_us = I['timestamps_us'].astype(np.int64)
    gz = np.interp(vt_us, it_us, I['gyro'][:, 2].astype(float)) * np.pi / 180.0
    dt = float(np.median(np.diff(vt)))
    ev = info['events']
    bad = interval_mask(vt_us, [c['t_us'] for c in ev.get('collisions', [])], 1.0, 2.0)
    bad |= interval_mask(vt_us, [c['t_us'] for c in ev.get('takeovers', [])], 0.3, 0.3)
    bad |= interval_mask(vt_us, [c['t_us'] for c in ev.get('handbacks', [])], 0.3, 0.3)
    bad |= span_mask(vt, [(r['t_last_s'], r['t_after_s']) for r in ev.get('board_reboots', [])], 0.5)
    for k in ('esc_assists', 'recoveries', 'giveups', 'holds'):
        bad |= span_mask(vt, [(r['t_s'], r['end_s']) for r in ev.get(k, [])], 0.2)
    bad |= np.concatenate([[False], np.diff(vt) > 0.05])
    steer = np.full(len(vt), np.nan)
    thr = np.full(len(vt), np.nan)
    hv = ~sel & V['valid'][:, 0].astype(bool) & V['valid'][:, 1].astype(bool)
    steer[hv] = V['pwm'][hv, 0]
    thr[hv] = V['pwm'][hv, 1]
    if A is not None and len(A['timestamps_us']):
        lat = A['latency_ms'].astype(float)
        lat_ok = np.isfinite(lat)
        med = float(np.median(lat[lat_ok])) if lat_ok.any() else 30.0
        t_eff = A['timestamps_us'].astype(np.int64) + (np.where(lat_ok, lat, med) * 1000).astype(np.int64)
        order = np.argsort(t_eff, kind='stable')
        t_eff = t_eff[order]
        out = A['out'][order].astype(float)
        lt = lab['t_us'].astype(np.int64)
        ki = np.clip(np.searchsorted(lt, A['timestamps_us'][order]), 0, len(lt) - 1)
        same = lt[ki] == A['timestamps_us'][order]
        okf = same & ~lab['hold'][ki] & (lab['assist'][ki] <= 0) & ~lab['guard_applied'][ki] & ~lab['dry_run'][ki]
        k = np.searchsorted(t_eff, vt_us, side='right') - 1
        kk = np.clip(k, 0, len(t_eff) - 1)
        fresh = (k >= 0) & ((vt_us - t_eff[kk]) <= 100_000) & okf[kk]
        av = sel & fresh
        steer[av] = out[kk[av], 0]
        thr[av] = out[kk[av], 1]
    op = np.where(sel, 1, 0)
    clean = np.isfinite(steer) & np.isfinite(thr) & ~bad
    lt = lab['t_us'].astype(np.int64)
    ki = np.clip(np.searchsorted(lt, vt_us), 1, len(lt) - 1)
    ki = np.where(np.abs(lt[ki - 1] - vt_us) < np.abs(lt[ki] - vt_us), ki - 1, ki)
    loc = np.where(np.abs(lt[ki] - vt_us) <= 100_000, lab['loc_code'][ki], 0).astype(int)
    vs = movavg(v, 9)
    acc = movavg(np.gradient(vs, vt), 9)
    center = np.where(op == 1, 1443.0, 1442.0)
    return info, {'t': vt, 'op': op, 'clean': clean, 'loc': loc, 'v': v, 'vs': vs, 'acc': acc, 'gz': gz, 'steer': steer,
                  'thr': thr, 'off': steer - center, 'dt': dt}


def brake_events(log, labels_dir=None, raw_dir=None, surface=None):
    info, r = record(log, labels_dir, raw_dir)
    if r is None:
        return []
    desc = info.get('description', '')
    surf_desc = surface or surface_of_description(desc)
    dt = r['dt']
    brake = (r['thr'] >= THR_N + BRAKE_MIN_US) & np.isfinite(r['thr'])
    out = []
    for a, b in le.runs(brake):
        if b - a + 1 < int(round(BRAKE_MIN_S / dt)) or r['vs'][a] < V_START_MIN:
            continue
        seg = slice(a, b + 1)
        loc = LOC[int(r['loc'][a])]
        off = float(np.median(r['thr'][seg] - THR_N))
        alat = float(np.nanmean(np.abs(r['vs'][seg] * r['gz'][seg])))
        out.append({'log': log, 't_s': round(float(r['t'][a]), 2), 'op': ['human', 'ai'][int(r['op'][a])],
                    'dur_s': round((b - a + 1) * dt, 2), 'v0': round(float(r['vs'][a]), 2), 'v1': round(float(r['vs'][b]), 2),
                    'mean_decel': round(float(-np.mean(r['acc'][seg])), 2), 'peak_decel': round(float(-np.min(r['acc'][seg])), 2),
                    'thr_off_med': round(off, 0), 'steer_off_med': round(float(np.median(np.abs(r['off'][seg]))), 0),
                    'alat_mean': round(alat, 2), 'clean': bool(r['clean'][seg].all()), 'loc': loc,
                    'surface': surf_desc or ('芝' if loc == 'GM' else '床'),
                    'surface_source': ('option' if surface else 'description') if surf_desc else 'loc',
                    'strength': next(n for lo, hi, n in STRENGTH if lo <= off < hi), 'description': desc})
    return out


FIT_MIN_GE150 = 5    # 当てはめを「仮」にしない、150 µs 以上の出来事の数(10/10 の床は 3 回で db_b 274 µs になった)


def fit_brake(off, ab):
    """a_b(off) = s_b × max(0, off − db_b) の最小二乗(ident.py の fit_brake と同じ)"""
    best = None
    for db in np.arange(0.0, 300.1, 1.0):
        x = np.maximum(0.0, off - db)
        sxx = np.sum(x * x)
        if sxx <= 0:
            continue
        s = max(0.0, np.sum(x * ab) / sxx)
        sse = np.sum((ab - s * x) ** 2)
        if best is None or sse < best[2] - 1e-12:
            best = (float(s), float(db), float(sse))
    return best


def load_coast(path=PCAL):
    try:
        lo = json.loads(Path(path).read_text())['longitudinal']
        return {'c0': lo['c0'], 'k_s': lo['k_s'], 'source': str(path)}
    except Exception:
        return {'c0': 0.58, 'k_s': 0.40, 'source': '既定(P_cal が読めない)'}


def table(events, coast=None, ops=('human', 'ai')):
    coast = coast or load_coast()
    ev = [dict(e) for e in events if e['clean'] and e['op'] in ops]
    for e in ev:
        e['a_b'] = round(e['mean_decel'] - (coast['c0'] + coast['k_s'] * e['alat_mean']), 2)
    rows = []
    for surf in ('床', '芝'):
        for lo, hi, name in STRENGTH:
            q = [e for e in ev if e['surface'] == surf and e['strength'] == name]
            if not q:
                rows.append({'surface': surf, 'strength': name, 'n': 0})
                continue
            md = [e['mean_decel'] for e in q]
            rows.append({'surface': surf, 'strength': name, 'n': len(q),
                         'off_us_median': float(np.median([e['thr_off_med'] for e in q])),
                         'v0_median': round(float(np.median([e['v0'] for e in q])), 2),
                         'decel_median': round(float(np.median(md)), 2), 'decel_min': min(md), 'decel_max': max(md),
                         'a_b_median': round(float(np.median([e['a_b'] for e in q])), 2)})
    fits = {}
    for surf in ('床', '芝'):
        q = [e for e in ev if e['surface'] == surf]
        n150 = sum(1 for e in q if e['thr_off_med'] >= 150)
        if len(q) >= 3:
            f = fit_brake(np.array([e['thr_off_med'] for e in q]), np.array([e['a_b'] for e in q]))
            if f:
                fits[surf] = {'n': len(q), 'n_ge150us': n150, 's_b': round(f[0], 5), 'db_b': f[1],
                              'a_b_at_200us': round(f[0] * max(0, 200 - f[1]), 2), 'a_b_at_350us': round(f[0] * max(0, 350 - f[1]), 2)}
                if n150 < FIT_MIN_GE150:   # 弱い出来事(a_b が負になりやすい)が中心になり、db_b が大きく出る(V-tools 4 章 9)
                    fits[surf].update(provisional=True, note='仮: 150 µs 以上が %d 回(%d 回未満)。弱いブレーキが中心の当てはめ' % (n150, FIT_MIN_GE150))
        else:
            fits[surf] = {'n': len(q), 'note': '3 回未満なので当てはめない'}
    return {'coast': coast, 'n_clean': len(ev), 'rows': rows, 'fits': fits, 'events': ev}


def to_markdown(res, title='ブレーキの量の表'):
    T = res['table']
    L = ['# %s' % title, '',
         '- 作成: %s(tools/brake_table.py、読み取りだけ)' % res['created_at'],
         '- ログ: %s' % ', '.join(res['logs']),
         '- ブレーキの出来事 %d 回(きれい %d 回。人 %d・AI %d)。表はきれいなものだけ' % (
             len(res['events']), sum(e['clean'] for e in res['events']), sum(e['op'] == 'human' and e['clean'] for e in res['events']),
             sum(e['op'] == 'ai' and e['clean'] for e in res['events'])),
         '- a_b = 減速 − 惰性(c0 %.2f + k_s %.2f × |横加速度|、%s)' % (T['coast']['c0'], T['coast']['k_s'], T['coast']['source']),
         '', '## µs → 減速(路面 × 強さ)', '',
         '| 路面 | 強さ | 回数 | 中立からの µs(中央値) | 始まりの速さ m/s | 減速 m/s²(中央値 [最小–最大]) | ESC の分 a_b m/s² |',
         '|---|---|---|---|---|---|---|']
    for r in T['rows']:
        if not r['n']:
            L.append('| %s | %s | 0 | — | — | — | — |' % (r['surface'], r['strength']))
            continue
        L.append('| %s | %s | %d | %.0f | %.2f | %.2f [%.2f–%.2f] | %.2f |' % (
            r['surface'], r['strength'], r['n'], r['off_us_median'], r['v0_median'], r['decel_median'], r['decel_min'],
            r['decel_max'], r['a_b_median']))
    L += ['', '## 当てはめ a_b(off) = s_b × max(0, off − db_b)', '', '| 路面 | 回数 | s_b | db_b µs | +200 µs | +350 µs |',
          '|---|---|---|---|---|---|']
    for s, f in T['fits'].items():
        if 's_b' in f:
            L.append('| %s | %d | %.4f | %.0f | %.2f m/s² | %.2f m/s²%s |' % (s, f['n'], f['s_b'], f['db_b'], f['a_b_at_200us'], f['a_b_at_350us'],
                                                                      '(**%s**)' % f['note'] if f.get('provisional') else ''))
        else:
            L.append('| %s | %d | — | — | — | %s |' % (s, f['n'], f['note']))
    by = res.get('table_by_op') or {}
    if len(by) > 1:
        L += ['', '## 人と AI を分けた表(人 = 送信機の測定。AI = 走行中のブレーキ・上限の層)', '',
              '| 操作 | 路面 | 強さ | 回数 | µs | 減速 m/s² | a_b m/s² |', '|---|---|---|---|---|---|---|']
        for op, T2 in by.items():
            for r in T2['rows']:
                if r['n']:
                    L.append('| %s | %s | %s | %d | %.0f | %.2f | %.2f |' % (
                        {'human': '人', 'ai': 'AI'}[op], r['surface'], r['strength'], r['n'], r['off_us_median'], r['decel_median'], r['a_b_median']))
    L += ['', '## 出来事の一覧', '', '| ログ | 時刻 s | 人/AI | 路面(決め方) | 強さ | µs | 秒 | 速さ m/s | 減速 m/s² | 場所 | きれい |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for e in res['events']:
        L.append('| %s | %.2f | %s | %s(%s) | %s | %.0f | %.2f | %.2f → %.2f | %.2f | %s | %s |' % (
            e['log'], e['t_s'], e['op'], e['surface'], e['surface_source'], e['strength'], e['thr_off_med'], e['dur_s'], e['v0'],
            e['v1'], e['mean_decel'], e['loc'], '○' if e['clean'] else '×'))
    if res.get('errors') or res.get('problems') or res.get('computed_labels'):
        L += ['', '## 読めなかったログ・飛ばしたもの', '']
        L += ['- %s: 読めなかった(%s)' % (lg, err) for lg, err in (res.get('errors') or {}).items()]
        L += ['- %s: 壊れた・無いファイルを飛ばした %d 件(%s)' % (lg, len(pr), '・'.join(pr[:5]) + (' ほか' if len(pr) > 5 else ''))
              for lg, pr in (res.get('problems') or {}).items() if pr]
        if res.get('computed_labels'):
            L.append('- ラベルのファイルが無く、その場で作った: %s' % '・'.join(res['computed_labels']))
    L += ['', '- 限界: 1 回の区間の平均の減速。電池の電圧・タイヤの温度は入っていない。強いブレーキが少ないと、当てはめは不確か。',
          '- 弱いブレーキ(< 150 µs)の a_b は、惰性の直線の式で引きすぎて負になることがある(10/16 の deviations D5)。合計の減速の列で読む。']
    return '\n'.join(L) + '\n'


def build(logs, labels_dir=None, raw_dir=None, surface=None, ops=('human', 'ai')):
    events, errors, problems, computed = [], {}, {}, []
    for lg in logs:
        try:
            events += brake_events(lg, labels_dir, raw_dir, surface)
        except Exception as e:
            errors[lg] = repr(e)
        pj = Path(labels_dir or config.DATA_DIR / 'labels') / (lg + '.json')
        if pj.exists():
            try:
                pr = json.loads(pj.read_text(encoding='utf-8')).get('problems') or []
            except Exception:
                pr = []
            if pr:
                problems[lg] = pr
        elif lg not in errors:
            computed.append(lg)
    return {'schema': SCHEMA, 'created_at': datetime.now().isoformat(timespec='seconds'), 'tool': 'tools/brake_table.py',
            'logs': list(logs), 'errors': errors, 'problems': problems, 'computed_labels': computed, 'events': events,
            'ops': list(ops), 'table': table(events, ops=ops),
            'table_by_op': {op: table(events, ops=(op,)) for op in ('human', 'ai') if any(e['op'] == op and e['clean'] for e in events)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('logs', nargs='*')
    ap.add_argument('--day')
    ap.add_argument('--labels', default=str(config.DATA_DIR / 'labels'))
    ap.add_argument('--raw', default=str(config.RAW_DIR))
    ap.add_argument('--surface', default=None, help='路面を全部のログでこれにする(床 / 芝)')
    ap.add_argument('--ops', default='human,ai', help='表に入れる操作者(human,ai)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)
    logs = list(args.logs)
    if args.day:
        logs += sorted(d.name for d in Path(args.raw).glob(args.day.replace('-', '') + '-*') if (d / 'log.json').exists())
    logs = sorted(set(logs))
    if not logs:
        ap.error('ログがありません')
    surf = SURFACE_WORDS.get(args.surface.lower(), args.surface) if args.surface else None
    res = build(logs, args.labels, args.raw, surf, tuple(args.ops.split(',')))
    out = Path(args.out) if args.out else config.DATA_DIR / 'reports' / ('brake_table_' + (args.day or logs[0][:8]).replace('-', ''))
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / 'brake_table.json.tmp'
    tmp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, out / 'brake_table.json')
    (out / 'brake_table.md').write_text(to_markdown(res), encoding='utf-8')
    print('ブレーキ %d 回(きれい %d)' % (len(res['events']), res['table']['n_clean']), '書き出し:', out)
    return res


if __name__ == '__main__':
    main()
