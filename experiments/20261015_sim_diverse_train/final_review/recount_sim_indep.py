#!/usr/bin/env python3
"""最終の確かめ(10/15 のレポートの段): Q1 の見出しの数を、走行の生の JSON だけから、評価の段とも recount_q1.py とも別のコードで数え直す。

    python3 -I recount_sim_indep.py

読むもの: ブラウザのアプリが保存した走行の JSON(outcome・metrics・trace・context・world)と、一式の ZIP の中の ONNX(sha256 を自分で計算)。
評価の段の道具・runs.csv・eval_sim.json・runs.json は「比べる相手」としてだけ読む(数えるのには使わない)。
  - 開始は、ファイル名ではなく context.start_variation を protocol の E0〜E11 の表と照らして決める(ファイル名と食い違えば記録)
  - 完走 c: outcome.success かつ reason == route_complete かつ 衝突 0・ゲート 33・必須横断 1 回
  - fs(ゲート 0 の 9 本): 最初の衝突のフレームより前の trace に x > 2.6 m がある
  - g4(ゲート 0 の 9 本): metrics.passed_gates ≥ 4(protocol の定義)。加えて「最初の周で点 4 の手前の線を越えるより先に数えたか」も出す
  - d: 完走は 1 周(26.1196 m)、ほかは world.route の閉じた折れ線への射影を「進む向きだけ」たどる(後ろ戻りは足さない)別の作り方。参考
  - 設定: モデルの sha256(ZIP の中の ONNX を自分で計算)・遅延 1・先読み 1・ゲイン 1・ジャンル 8・時間の上限 15 秒
  - 判定: protocol.json verdicts_Q1 の文を、ここで書き直したコードで当てはめる
出力: /home/jetson/jetformer_reports/sim_diverse_train_20261015/review/final/recount_sim_indep.json
"""
import hashlib
import json
import math
import os
import statistics
import zipfile

REP = '/home/jetson/jetformer_reports/sim_diverse_train_20261015'
OUT = os.path.join(REP, 'review/final/recount_sim_indep.json')
LAP = 26.1196
PROTO = json.load(open(os.path.join(REP, 'protocol.json'), encoding='utf-8'))
STARTS = {k: v for k, v in PROTO['eval_Q1']['starts'].items() if k.startswith('E')}
GATE0 = PROTO['eval_Q1']['starts']['gate0_starts']

# 腕 → (走行の JSON の場所の作り方, 一式の ZIP)。A0 の E0〜E5 は 10/13 の S-R 種 0 の評価(10/14 の A0 はこれを使った)
CL = os.path.join(REP, 'eval/closed_loop')
P14 = '/home/jetson/jetformer_reports/sim_dart_20261014/eval/closed_loop'
P13 = '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/eval/closed_loop'
ARMS = {
    'A0': (lambda e: (os.path.join(P13, 'SR-s0/SR-s0-%s-single.json' % e) if int(e[1:]) <= 5
                      else os.path.join(P14, 'A0-s0/A0-s0-%s-single.json' % e)),
           '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/bundles/simrl_bestlap_20261013-SR-s0.zip'),
    'A1': (lambda e: os.path.join(P14, 'A1-s0/A1-s0-%s-single.json' % e),
           '/home/jetson/jetformer_reports/sim_dart_20261014/bundles/simdart_20261014-A1-s0.zip'),
    'A2': (lambda e: os.path.join(P14, 'A2-s0/A2-s0-%s-single.json' % e),
           '/home/jetson/jetformer_reports/sim_dart_20261014/bundles/simdart_20261014-A2-s0.zip'),
    'S-DIV': (lambda e: os.path.join(CL, 'SDIV-s0/SDIV-s0-%s-single.json' % e), os.path.join(REP, 'bundles/simdiv_20261015-SDIV-s0.zip')),
    'S-MOD': (lambda e: os.path.join(CL, 'SMOD-s0/SMOD-s0-%s-single.json' % e), os.path.join(REP, 'bundles/simdiv_20261015-SMOD-s0.zip')),
    'A4': (lambda e: os.path.join(CL, 'A4-s0/A4-s0-%s-single.json' % e), os.path.join(REP, 'bundles/simdiv_20261015-A4-s0.zip')),
}


def onnx_sha(zp):
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.endswith('.onnx')]
        assert len(names) == 1, names
        return hashlib.sha256(z.read(names[0])).hexdigest()


def which_start(sv):
    for k, (g, off, yaw) in STARTS.items():
        if sv['gate'] == g and abs(sv['left_offset_m'] - off) < 1e-9 and abs(sv['left_yaw_deg'] - yaw) < 1e-9:
            return k
    return None


def forward_progress(r):
    """route の閉じた折れ線に沿って、前に進んだぶんだけを足す(近くの ±3 区間で射影)"""
    route = r['world']['route']
    n = len(route)
    seg = [math.dist(route[i], route[(i + 1) % n]) for i in range(n)]
    base = [sum(seg[:i]) for i in range(n)]
    total = sum(seg)

    def proj_on(i, x, z):
        (ax, az), (bx, bz) = route[i], route[(i + 1) % n]
        dx, dz = bx - ax, bz - az
        u = ((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz)
        u = min(1.0, max(0.0, u))
        return (x - ax - u * dx) ** 2 + (z - az - u * dz) ** 2, base[i] + u * seg[i]

    pts = [(t['x'], t['z']) for t in r['trace']]
    # 最初の点は全部の区間から一番近いところ
    best = min(range(n), key=lambda i: proj_on(i, *pts[0])[0])
    cur, s_prev = best, proj_on(best, *pts[0])[1]
    acc = 0.0
    for x, z in pts[1:]:
        cand = [(proj_on(i % n, x, z), i % n) for i in range(cur - 3, cur + 4)]
        (d2, s), cur = min(cand, key=lambda c: c[0][0])
        ds = s - s_prev
        if ds < -total / 2:
            ds += total
        elif ds > total / 2:
            ds -= total
        if ds > 0:
            acc += ds
        s_prev = s
    return acc / total * LAP


def one_run(path, arm, e, model_sha):
    r = json.load(open(path, encoding='utf-8'))
    ctx, pi, m, o = r['context'], r['context']['policy_info'], r['metrics'], r['outcome']
    st = which_start(ctx['start_variation'])
    coll = m.get('collision_events') or []
    first_coll_frame = min((c['frame'] for c in coll), default=None)
    complete = bool(o.get('success')) and o.get('reason') == 'route_complete' and m['collisions'] == 0 and m['passed_gates'] == 33 \
        and len(m.get('required_crossing_events') or []) == 1
    fs = any(t['x'] > 2.6 for t in r['trace'] if first_coll_frame is None or t['frame'] < first_coll_frame)
    settings = {
        'model_sha_ok': pi.get('model_sha256') == model_sha,
        'delay_ticks': ctx['execution']['control_delay']['ticks'],
        'lookahead': pi.get('lookahead'), 'steer_gain': pi.get('steer_gain'), 'throttle_gain': pi.get('throttle_gain'),
        'mode_id': pi.get('mode_id'), 'time_limit': r['config']['timeLimit'],
    }
    settings_ok = settings['model_sha_ok'] and settings['delay_ticks'] == 1 and settings['lookahead'] == 1 and \
        settings['steer_gain'] == 1 and settings['throttle_gain'] == 1 and settings['mode_id'] == 8 and settings['time_limit'] == 15
    d = LAP if complete else forward_progress(r)
    return {'file': path, 'start_from_context': st, 'start_matches_name': st == e, 'complete': complete, 'reason': o.get('reason'),
            'elapsed_s': round(m['elapsed'], 4), 'passed_gates': m['passed_gates'], 'collisions': m['collisions'],
            'fs': fs, 'g4': m['passed_gates'] >= 4, 'd_m': round(d, 3), 'settings': settings, 'settings_ok': settings_ok,
            'collision_xz': [round(coll[0]['x'], 2), round(coll[0]['z'], 2)] if coll else None,
            'collider_kind': coll[0]['kind'] if coll else None}


def main():
    res = {'schema': 'sim-diverse-train-final-recount-sim/v1', 'arms': {}}
    for arm, (fn, zp) in ARMS.items():
        sha = onnx_sha(zp)
        runs = {}
        for i in range(12):
            e = 'E%d' % i
            runs[e] = one_run(fn(e), arm, e, sha)
        g0 = [runs[e] for e in GATE0]
        comp = [e for e in runs if runs[e]['complete']]
        res['arms'][arm] = {
            'onnx_sha256_from_zip': sha, 'n_runs': 12,
            'c': len(comp), 'completed': comp,
            'fs': sum(x['fs'] for x in g0), 'g4': sum(x['g4'] for x in g0),
            'd_median_m': round(statistics.median(x['d_m'] for x in runs.values()), 2),
            'lap_median_s': round(statistics.median(runs[e]['elapsed_s'] for e in comp), 3) if comp else None,
            'all_settings_ok': all(x['settings_ok'] for x in runs.values()),
            'all_starts_match_names': all(x['start_matches_name'] for x in runs.values()),
            'runs': runs,
        }
    A = res['arms']
    c_ref = max(A[a]['c'] for a in ('A0', 'A1', 'A2'))
    s = A['S-DIV']
    if s['c'] >= 6 and s['c'] - c_ref >= 4:
        v1 = '効いた'
    elif s['c'] - c_ref >= 2 or s['g4'] >= 5:
        v1 = '一部効いた'
    elif s['c'] <= c_ref and s['fs'] <= 4:
        v1 = '悪くなった'
    else:
        v1 = '差なし'
    m, a1 = A['S-MOD'], A['A1']
    if m['c'] - a1['c'] >= 2 or (m['c'] == a1['c'] == 0 and m['g4'] - a1['g4'] >= 3) or \
            (m['g4'] == 0 and a1['g4'] == 0 and m['d_median_m'] >= 1.5 * a1['d_median_m']):
        v2 = '量よりばらつき'
    else:
        v2 = 'ばらつきの効果は見えない'

    def addon(x, y):
        dc, dg, df = x['c'] - y['c'], x['g4'] - y['g4'], x['fs'] - y['fs']
        both0 = x['c'] == 0 and y['c'] == 0
        if dc >= 2 or (both0 and (dg >= 3 or df >= 3)):
            return '上乗せあり'
        if dc <= -2 or (both0 and (dg <= -3 or df <= -3)):
            return '悪くなった'
        return '差なし'
    res['c_ref'] = c_ref
    res['verdicts'] = {'Q1_S-DIV_vs_yesterday': v1, 'Q1_variety_vs_volume': v2,
                       'Q1_DART_margin_addon': addon(A['S-DIV'], A['S-MOD']), 'Q1b_DAgger': addon(A['A4'], A['S-DIV'])}
    # 評価の段の数(eval_sim.json)と比べる
    ev = json.load(open(os.path.join(REP, 'eval_sim.json'), encoding='utf-8'))
    cmp = {}
    for arm in A:
        se = ev['summary'][arm]
        cmp[arm] = {'c': [A[arm]['c'], se['n_completed']], 'fs': [A[arm]['fs'], se['fs']], 'g4': [A[arm]['g4'], se['g4']],
                    'completed_same': sorted(A[arm]['completed']) == sorted(se['completed']),
                    'd_median_m': [A[arm]['d_median_m'], round(se['d_median_m'], 2)]}
        cmp[arm]['counts_same'] = cmp[arm]['c'][0] == cmp[arm]['c'][1] and cmp[arm]['fs'][0] == cmp[arm]['fs'][1] and \
            cmp[arm]['g4'][0] == cmp[arm]['g4'][1] and cmp[arm]['completed_same']
    ev_v = ev['verdicts']
    vcmp = {'Q1_S-DIV_vs_yesterday': [v1, ev_v['Q1_S-DIV_vs_yesterday'].get('verdict')],
            'Q1_variety_vs_volume': [v2, ev_v['Q1_variety_vs_volume'].get('verdict')],
            'Q1_DART_margin_addon': [res['verdicts']['Q1_DART_margin_addon'], ev_v['Q1_DART_margin_addon(S-DIV vs S-MOD)'].get('verdict')],
            'Q1b_DAgger': [res['verdicts']['Q1b_DAgger'], ev_v['Q1b_DAgger'].get('verdict')]}
    res['compare_eval_sim'] = {'arms': cmp, 'verdicts': vcmp,
                               'all_counts_same': all(x['counts_same'] for x in cmp.values()),
                               'all_verdicts_same': all(a == b for a, b in vcmp.values())}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    os.replace(tmp, OUT)
    for arm in A:
        x = A[arm]
        print('%-6s c=%d %s fs=%d g4=%d d=%.2f lap=%s settings_ok=%s starts_ok=%s' % (
            arm, x['c'], x['completed'], x['fs'], x['g4'], x['d_median_m'], x['lap_median_s'], x['all_settings_ok'], x['all_starts_match_names']))
    print('verdicts', res['verdicts'])
    print('compare', json.dumps(res['compare_eval_sim'], ensure_ascii=False))


if __name__ == '__main__':
    main()
