#!/usr/bin/env python3
"""閉ループ評価(段 E)の集計・判定・図(2026-10-14)

    node eval/verify_eval.mjs                     # 先に 1 走行ずつの確認と要約(closed_loop/runs_verified.json)
    python -I eval/analyze_eval.py                # この集計(ユーザーの確認器も、ここから別のプロセスで呼ぶ)

  1) E0 の単周(protocol とゲインの感度)を、ユーザーの qa/verify-learned-lap.mjs <結果> --model <ONNX> で確かめる
     (元のフォルダーのまま実行する。ユーザーのファイルは変更しない。結果は closed_loop/user_verifier/<走行>.json)
  2) 走行ごとの表(eval/runs.csv)・モデル × 開始の停止から 1 周 T(m, e)・フライングラップ
  3) protocol.json の acceptance_H2(結果を見る前に固定した基準)をそのまま当てる。基準は変えない
  4) 参考(合否に使わない): フライングラップ・オフライン L1(学習の段の train/offline_l1.json)・外部の参照との差・ゲインの感度・
     決定性の確認(S-R 種 0 の E0 を 2 回)・事後に足した参考の条件(extra)
  5) 図: fig/eval_laptimes.png・fig/eval_trajectories.png・fig/eval_speed_profiles.png・fig/eval_crash_map.png
  6) evaluation.json・evaluation.md(レポートのフォルダーの直下)
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_common import EVAL, MODEL_BY_LABEL, MODELS, REPORT, STARTS, USER_SIM  # noqa: E402

CL = EVAL / 'closed_loop'
FIG = REPORT / 'fig'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
COLORS = {'SR-s0': '#2a78d6', 'SR-s1': '#7fb0ea', 'ST-s0': '#eb6834', 'ST-s1': '#f3a77f', 'EXT': '#4a9a5b', 'RL': '#0b0b0b', 'teacher': '#8a8984'}
NAMES = {'SR-s0': 'S-R 種 0', 'SR-s1': 'S-R 種 1', 'ST-s0': 'S-T 種 0', 'ST-s1': 'S-T 種 1', 'EXT': '外部の参照(作者の ONNX)',
         'RL': 'RL の方策(遅延 0・Node)', 'teacher': 'RC 教師(Node)'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def run_meta(name):
    p = name.split('-')
    if p[0] in ('SR', 'ST'):
        label, rest = f'{p[0]}-{p[1]}', p[2:]
    else:
        label, rest = p[0], p[1:]
    return label, rest[0], rest[1], '-'.join(rest[2:]) or 'protocol'


def user_verifier(row, onnx):
    """ユーザーの確認器(CLI)。元のフォルダーで、そのまま実行する"""
    out = CL / 'user_verifier' / f"{row['name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(['node', 'qa/verify-learned-lap.mjs', row['file'], '--model', onnx, '--output', str(out)],
                       cwd=USER_SIM, capture_output=True, text=True, timeout=600)
    try:
        d = json.loads(out.read_text())
        res = d['results'][0]
        return {'pass': bool(d['pass']), 'exit_code': r.returncode, 'failure_count': res.get('failure_count'),
                'failures': res.get('failures', [])[:8], 'replay_elapsed': (res.get('replay') or {}).get('elapsed'), 'output': str(out)}
    except (OSError, ValueError, KeyError) as e:
        return {'pass': False, 'exit_code': r.returncode, 'error': f'{type(e).__name__}: {e}', 'stderr': r.stderr[-800:]}


def steer_stats(csv_path):
    """実際に使った舵の操作の荒さ(現実味の指標と同じ定義: 端 = |δ| ≥ 0.494 rad、変化の速さ = |Δδ|·30 の 95% 点)"""
    rows = list(csv.DictReader(open(csv_path)))
    st = [float(r['applied_steer_rad']) for r in rows if r['applied_steer_rad'] not in ('', 'undefined')]
    if len(st) < 3:
        return {}
    rates = sorted(abs(b - a) * 30 for a, b in zip(st, st[1:]))
    return {'steer_edge_frac': sum(abs(x) >= .494 for x in st) / len(st), 'steer_cmd_rate_p95': rates[int(.95 * (len(rates) - 1))]}


def load_reference():
    d = json.loads((REPORT / 'rl/final/final_eval_main.json').read_text())
    rl = {r['name']: r for r in d['rl']['single']}
    rlc = {r['name']: r for r in d['rl']['continuous']}
    te = {r['name']: r for r in d['teacher']['single']}
    tec = {r['name']: r for r in d['teacher']['continuous']}
    ref = {
        'T_RL': {e: rl[e]['single_lap_s'] for e in STARTS},
        'T_teacher': {e: te[e]['single_lap_s'] for e in ('E0', 'E1', 'E2')},
        'RL_flying': {e: {'lap1': rlc[e]['laps'][0]['elapsed'], 'lap2': rlc[e]['flying_lap_s']} for e in ('E0', 'E1')},
        'teacher_flying': {e: {'lap1': tec[e]['laps'][0]['elapsed'], 'lap2': tec[e]['flying_lap_s']} for e in ('E0', 'E1')},
        'RL_delay1_STD': d['rl']['STD_delay1_reference']['single_lap_s'],
        'teacher_delay1_STD': {'reason': d['teacher']['STD_delay1_reference']['reason'],
                               'passed_gates': d['teacher']['STD_delay1_reference']['passed_gates']},
        'learned_onnx_user_record_s': d['learned_onnx']['single_lap_s'],
        'records': {'RL': {e: rl[e]['record'] for e in STARTS}, 'teacher': {e: te[e]['record'] for e in ('E0', 'E1', 'E2')}},
        'source': str(REPORT / 'rl/final/final_eval_main.json'),
        'note': 'RL と教師は Node の元の sim-core.js(遅延 0・物理の上限どおり)。RL の段の最終の評価の値で、この段では走らせ直していない',
    }
    return ref


def acceptance(T, user_ok, seeds_spread_detail):
    """protocol.json acceptance_H2 をそのまま当てる"""
    starts = list(STARTS)
    sr = [(s, e) for s in (0, 1) for e in starts]
    comp = {(s, e): T.get((f'SR-s{s}', e)) is not None for s, e in sr}
    a_e0 = all(T.get((f'SR-s{s}', 'E0')) is not None and user_ok.get(f'SR-s{s}-E0-single') for s in (0, 1))
    n_comp = sum(comp.values())
    a = {'E0_both_seeds_complete_and_user_verifier_pass': a_e0, 'completed_of_12': n_comp, 'required': 9,
         'pass': bool(a_e0 and n_comp >= 9)}
    ratios = [T[(f'SR-s{s}', e)] / REF['T_RL'][e] for s, e in sr if comp[(s, e)]]
    b = {'ratios': ratios, 'median_ratio': med(ratios), 'threshold': 1.05,
         'pass': bool(ratios) and med(ratios) <= 1.05}
    pairs = [(s, e, T[(f'ST-s{s}', e)] - T[(f'SR-s{s}', e)]) for s, e in sr
             if T.get((f'SR-s{s}', e)) is not None and T.get((f'ST-s{s}', e)) is not None]
    spread = seeds_spread_detail['max']
    delta = max(.10, 2 * spread) if spread is not None else .10
    c = {'pairs': [{'seed': s, 'start': e, 'T_ST_minus_T_SR': d} for s, e, d in pairs], 'n_pairs': len(pairs),
         'seed_spread': seeds_spread_detail, 'delta': delta}
    if len(pairs) >= 3:
        c['rule'] = 'main'
        c['median_diff'] = med([d for *_, d in pairs])
        c['pass'] = c['median_diff'] > delta
    else:
        c['rule'] = 'fallback(組が 3 つ未満): S-R の E0 が両方の種とも外部の参照の E0 − 0.10 秒より速い'
        ext = T.get(('EXT', 'E0'))
        sr0 = [T.get((f'SR-s{s}', 'E0')) for s in (0, 1)]
        c['ext_E0'] = ext
        c['sr_E0'] = sr0
        c['pass'] = ext is not None and all(x is not None and x < ext - .10 for x in sr0)
    if a['pass'] and b['pass'] and c['pass']:
        verdict = '学習できた'
    elif a['pass'] and (b['pass'] or c['pass']):
        verdict = '一部できた'
    else:
        verdict = 'できなかった'
    return {'a_completion': a, 'b_close_to_rl': b, 'c_faster_than_teacher_trained': c, 'verdict': verdict}


def seed_spread(T):
    out = {}
    for arm in ('SR', 'ST'):
        m = [med([T.get((f'{arm}-s{s}', e)) for e in STARTS]) for s in (0, 1)]
        out[arm] = {'median_s0': m[0], 'median_s1': m[1], 'abs_diff': abs(m[0] - m[1]) if None not in m else None}
    vals = [v['abs_diff'] for v in out.values() if v['abs_diff'] is not None]
    out['max'] = max(vals) if vals else None
    out['note'] = '腕の種のどちらかが 1 回も完走しないと、その腕のばらつきは計算できない(計算できる腕だけで最大をとる。どちらもなければ Δ = 0.10 秒)'
    return out


def course(ax, world, plt):
    for s in world['surfaces']:
        if s['id'] == 'carpet':
            continue
        col = {'shortcut': '#3b3b3b', 'turf': '#7fbf7f', 'ptfe': '#e8e2c8', 'blue-mat': '#b9d3f0'}.get(s['id'], '#cccccc')
        ax.add_patch(plt.Rectangle((s['x'] - s['w'] / 2, s['z'] - s['d'] / 2), s['w'], s['d'], color=col, alpha=.3, lw=0))
    for seg in world['segments']:
        ax.plot([seg['a'][0], seg['b'][0]], [seg['a'][1], seg['b'][1]], color='#9b4a3c', lw=1.1, alpha=.75)
    for dsk in world['disks']:
        ax.add_patch(plt.Circle((dsk['x'], dsk['z']), max(dsk.get('radius', .02), .02), color='#6f6e69', alpha=.5, lw=0))
    for i, (gx, gz) in enumerate(world['route']):
        ax.plot(gx, gz, marker='o', ms=2.5, color='#c9a227', alpha=.9)
        if i % 4 == 0:
            ax.text(gx, gz - .12, str(i), fontsize=6, color=INK2, ha='center', clip_on=True)
    b = world['bounds']
    ax.set_xlim(b['minX'], b['maxX'])
    ax.set_ylim(b['maxZ'], b['minZ'])
    ax.set_aspect('equal')


def style(ax, title, ylabel='', xlabel=''):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc='left')
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def jp_font():
    import matplotlib
    from matplotlib import font_manager
    for name in ('Noto Sans CJK JP', 'Noto Sans JP', 'IPAexGothic', 'IPAGothic', 'TakaoGothic', 'VL Gothic', 'Droid Sans Fallback'):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams['font.family'] = name
            return name
    return None


def trace_of(path):
    r = json.loads(Path(path).read_text())
    return r['trace'], r['world']


def cumdist(tr):
    d, out = 0.0, [0.0]
    for a, b in zip(tr, tr[1:]):
        d += math.hypot(b['x'] - a['x'], b['z'] - a['z'])
        out.append(d)
    return out


def figures(rows, T, ref):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    jp_font()
    FIG.mkdir(exist_ok=True)
    by = {r['name']: r for r in rows}
    labels = [m['label'] for m in MODELS]
    starts = list(STARTS)
    # 1) 周回時間の棒(停止から 1 周)。完走しなかった回は × と通過ゲート数
    fig, ax = plt.subplots(figsize=(12.5, 4.8), facecolor=SURFACE)
    style(ax, '閉ループ評価: 停止から 1 周の時間(遅延 1 フレーム・ブラウザのアプリ)。× = 衝突、□ = 時間切れ(数字は通過したゲート / 33)。'
              '黒の線 = RL の方策(遅延 0)、灰の破線 = RC 教師', '時間 [秒]')
    w = .15
    for j, e in enumerate(starts):
        for k, lab in enumerate(labels):
            x = j + (k - 2) * w
            t = T.get((lab, e))
            r = by.get(f'{lab}-{e}-single')
            if t is not None:
                ax.bar(x, t, w * .92, color=COLORS[lab], label=NAMES[lab] if j == 0 else None)
                ax.text(x, t + .05, f'{t:.2f}', ha='center', va='bottom', fontsize=6, rotation=90, color=INK2)
            elif r is not None:
                ax.bar(x, 0.0, w * .92, color=COLORS[lab], label=NAMES[lab] if j == 0 else None)
                if r['collisions']:
                    ax.plot(x, 6.3, marker='x', color=COLORS[lab], ms=7, mew=2)
                else:
                    ax.plot(x, 6.3, marker='s', mfc='none', color=COLORS[lab], ms=7, mew=1.8)
                ax.text(x, 6.5, f"{r['passed_gates']}", ha='center', va='bottom', fontsize=7, color=COLORS[lab])
        ax.hlines(ref['T_RL'][e], j - 2.6 * w, j + 2.6 * w, color=COLORS['RL'], lw=1.6, label=NAMES['RL'] if j == 0 else None)
        if e in ref['T_teacher']:
            ax.hlines(ref['T_teacher'][e], j - 2.6 * w, j + 2.6 * w, color=COLORS['teacher'], lw=1.6, ls='--',
                      label=NAMES['teacher'] if j == 0 else None)
    ax.set_xticks(range(len(starts)))
    ax.set_xticklabels([f'{e}\n(ゲート {g}・{o:+.2f} m・{y:+.0f}°)' for e, (g, o, y) in STARTS.items()], fontsize=8)
    ax.set_ylim(6, 11.5)
    ax.legend(fontsize=7.5, ncol=4, loc='upper right', frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / 'eval_laptimes.png', dpi=130, facecolor=SURFACE)
    plt.close(fig)

    # 2) E0 の軌跡(上から): 左 = S-R と RL、右 = S-T・外部の参照と教師
    _, world = trace_of(ref['records']['RL']['E0'])
    rl_tr, _ = trace_of(ref['records']['RL']['E0'])
    te_tr, _ = trace_of(ref['records']['teacher']['E0'])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), facecolor=SURFACE)
    for ax, group, title in ((axes[0], ['SR-s0', 'SR-s1'], 'E0: S-R(RL のデータで学習)と RL の方策'),
                             (axes[1], ['ST-s0', 'ST-s1', 'EXT'], 'E0: S-T(教師のデータで学習)・外部の参照と RC 教師')):
        style(ax, title + '(× = 衝突。時間切れは 15 秒までの軌跡)', 'z [m]', 'x [m]')
        course(ax, world, plt)
        ref_tr, ref_lab = (rl_tr, 'RL') if group[0].startswith('SR') else (te_tr, 'teacher')
        ax.plot([t['x'] for t in ref_tr], [t['z'] for t in ref_tr], color=COLORS[ref_lab], lw=1.4, ls='--', label=NAMES[ref_lab])
        for lab in group:
            r = by.get(f'{lab}-E0-single')
            if not r:
                continue
            tr = json.loads(Path(r['file']).read_text())['trace']
            res = f"{r['single_lap_s']:.3f} 秒" if r['single_lap_s'] else f"{ {'collision': '衝突', 'time_limit': '時間切れ'}.get(r['reason'], r['reason'])} {r['passed_gates']}/33"
            ax.plot([t['x'] for t in tr], [t['z'] for t in tr], color=COLORS[lab], lw=1.5, label=f'{NAMES[lab]}: {res}')
            if r['collisions']:
                ax.plot(tr[-1]['x'], tr[-1]['z'], marker='x', color=COLORS[lab], ms=11, mew=2.5)
        ax.legend(fontsize=7.5, loc='lower left', frameon=True)
    fig.tight_layout()
    fig.savefig(FIG / 'eval_trajectories.png', dpi=120, facecolor=SURFACE)
    plt.close(fig)

    # 3) 速さの変化(E0、走った距離に対して)
    fig, ax = plt.subplots(figsize=(12.5, 4.4), facecolor=SURFACE)
    style(ax, 'E0 の速さ(走った距離に対して、1 周 = 約 26〜28 m)。× = 衝突。S-T は点 4 のゲートを外して 2 周目に入っている', '速さ [m/s]', '走った距離 [m]')
    # 最後のフレームは終わったあとの状態(速さ 0 に戻される)なので描かない。横軸は 1 周ぶん(30 m)まで
    for lab, tr in (('RL', rl_tr), ('teacher', te_tr)):
        tr = tr[:-1]
        ax.plot(cumdist(tr), [t['speed'] for t in tr], color=COLORS[lab], lw=1.3, ls='--', label=NAMES[lab])
    for lab in labels:
        r = by.get(f'{lab}-E0-single')
        if not r:
            continue
        tr = json.loads(Path(r['file']).read_text())['trace'][:-1]
        d = cumdist(tr)
        ax.plot(d, [t['speed'] for t in tr], color=COLORS[lab], lw=1.3, label=NAMES[lab])
        if r['collisions']:
            ax.plot(d[-1], tr[-1]['speed'], marker='x', color=COLORS[lab], ms=10, mew=2.5)
    ax.set_xlim(0, 30)
    ax.legend(fontsize=7.5, ncol=4, frameon=False, loc='lower center')
    fig.tight_layout()
    fig.savefig(FIG / 'eval_speed_profiles.png', dpi=130, facecolor=SURFACE)
    plt.close(fig)

    # 4) すべての走行の軌跡と、ぶつかった場所
    fig, ax = plt.subplots(figsize=(12, 7.4), facecolor=SURFACE)
    style(ax, 'すべての走行(単周 E0〜E5・連続 E0/E1・ゲインの感度・参考の条件)の軌跡。× = ぶつかった場所', 'z [m]', 'x [m]')
    course(ax, world, plt)
    seen = set()
    for r in rows:
        tr = json.loads(Path(r['file']).read_text())['trace']
        lab = r['label']
        ax.plot([t['x'] for t in tr], [t['z'] for t in tr], color=COLORS.get(lab, '#999'), lw=.7, alpha=.45,
                label=NAMES.get(lab) if lab not in seen else None)
        seen.add(lab)
        if r['collisions']:
            ax.plot(tr[-1]['x'], tr[-1]['z'], marker='x', color=COLORS.get(lab, '#999'), ms=9, mew=2.2)
    ax.legend(fontsize=7.5, loc='lower left')
    fig.tight_layout()
    fig.savefig(FIG / 'eval_crash_map.png', dpi=120, facecolor=SURFACE)
    plt.close(fig)

    # 5) S-R がぶつかった所の拡大(最初の直線): RL の線と S-R の E0〜E5(E5 は別の場所なので除く)
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2), facecolor=SURFACE, gridspec_kw={'height_ratios': [1.25, 1]})
    ax = axes[0]
    style(ax, '拡大: 最初の直線(E0〜E4 の開始)。黒の破線 = RL の方策の E0、青 = S-R の単周 E0〜E4(× = 衝突)', 'z [m]', 'x [m]')
    course(ax, world, plt)
    ax.plot([t['x'] for t in rl_tr[:70]], [t['z'] for t in rl_tr[:70]], color=COLORS['RL'], lw=1.6, ls='--', label=NAMES['RL'])
    for lab in ('SR-s0', 'SR-s1'):
        for e in ('E0', 'E1', 'E2', 'E3', 'E4'):
            r = by.get(f'{lab}-{e}-single')
            if not r:
                continue
            tr = json.loads(Path(r['file']).read_text())['trace']
            ax.plot([t['x'] for t in tr], [t['z'] for t in tr], color=COLORS[lab], lw=1.1, alpha=.85, label=NAMES[lab] if e == 'E0' else None)
            if r['collisions']:
                ax.plot(tr[-1]['x'], tr[-1]['z'], marker='x', color=COLORS[lab], ms=9, mew=2.2)
    ax.set_xlim(-1.2, 4.2)
    ax.set_ylim(2.95, 0.9)
    ax.legend(fontsize=8, loc='lower left')
    ax = axes[1]
    style(ax, 'S-R 種 0 の E0: JetFormer が実際に使った舵と、同じ状態で RL の方策が出す舵(付け直し)', '舵 [rad](正が左)', 'フレーム(30 Hz)')
    if TAKEOVER:
        row = next((r for r in TAKEOVER['rows'] if r['name'] == 'SR-s0-E0-single'), None)
        if row:
            pf = row['relabel']['per_frame']
            ax.plot([q[0] for q in pf], [q[1] for q in pf], color=COLORS['SR-s0'], lw=1.6, label='S-R 種 0 が使った舵(遅延 1)')
            ax.plot([q[0] for q in pf], [q[2] for q in pf], color=COLORS['RL'], lw=1.4, ls='--', label='同じ状態で RL の方策が出す舵(付け直し)')
            rl_a = json.loads(Path(ref['records']['RL']['E0']).read_text())['actions'][:len(pf)]
            ax.plot(range(len(rl_a)), [a['steering'] for a in rl_a], color=COLORS['teacher'], lw=1.1, ls=':', label='RL の方策が自分の走りで出した舵(記録の線)')
            k = row['takeover']['last_recoverable_frame']
            if k is not None:
                ax.axvline(k, color='#4a9a5b', lw=1, ls='-.')
                ax.text(k - .3, .42, f'このフレームまでなら RL が引き継いで完走(k={k})', fontsize=8, color='#4a9a5b', ha='right')
            ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(FIG / 'eval_sr_first_corner.png', dpi=125, facecolor=SURFACE)
    plt.close(fig)
    return ['fig/eval_laptimes.png', 'fig/eval_trajectories.png', 'fig/eval_speed_profiles.png', 'fig/eval_crash_map.png', 'fig/eval_sr_first_corner.png']


REF = None
TAKEOVER = None


def main():
    global REF, TAKEOVER
    REF = load_reference()
    tp = EVAL / 'takeover.json'
    TAKEOVER = json.loads(tp.read_text()) if tp.exists() else None
    data = json.loads((CL / 'runs_verified.json').read_text())
    rows = [r for r in data['rows'] if 'error' not in r]
    bundles = json.loads((EVAL / 'bundles' / 'bundles.json').read_text())
    onnx_sha = {m['label']: m['onnx_sha256'] for m in bundles['models']}
    problems = [r for r in data['rows'] if 'error' in r]
    plans = {}
    for lab in MODEL_BY_LABEL:
        for kind in ('protocol', 'extra'):
            p = CL / lab / f'plan_{kind}.json'
            if p.exists():
                for c in json.loads(p.read_text()):
                    plans[c['name']] = (kind, c)
    for r in rows:
        lab, start, kind, variant = run_meta(r['name'])
        r.update(start=start, kind=kind, variant=variant, plan_kind=plans.get(r['name'], ('?', None))[0])
        c = plans.get(r['name'], (None, None))[1]
        chk = []
        if r['model_sha256'] != onnx_sha[lab]:
            chk.append('モデルの sha256 が一式と違う')
        if c:
            if r['control_delay_ticks'] != c['control-delay-frames']:
                chk.append('遅延が計画と違う')
            if r['lookahead'] != c['model-lookahead'] or r['mode_id'] != c['model-mode']:
                chk.append('先読み・ジャンルが計画と違う')
            if abs((r['steer_gain'] or 0) - c['model-steer-gain']) > 1e-9 or abs((r['throttle_gain'] or 0) - c['model-throttle-gain']) > 1e-9:
                chk.append('ゲインが計画と違う')
            sv = r['start_variation']
            g, o, y = STARTS[start]
            if sv['gate'] != g or abs(sv['left_offset_m'] - o) > 1e-9 or abs(sv['left_yaw_deg'] - y) > 1e-6:
                chk.append('開始が計画と違う')
        r['setting_check'] = chk
        # 外したゲートを、元の経路の点の番号で(目標 i は回した経路の点 i+1 = 元の点 (開始ゲート + i + 1) mod 33)
        g0 = (r.get('start_variation') or {}).get('gate', 0)
        r['gate_misses_orig'] = [{'route_point': (g0 + m['gate'] + 1) % 33, 'order_in_lap': m['gate'] % 33 + 1, 'time': m['time'], 'lateral_m': m['lateral_m']}
                                 for m in r.get('gate_misses', [])]
        r.update(steer_stats(r['file'].replace('.json', '.trajectory.csv')))
        if start == 'E0' and kind == 'single' and r['control_delay_ticks'] == 1:
            r['user_verifier'] = user_verifier(r, MODEL_BY_LABEL[lab]['onnx'])
    proto = [r for r in rows if r['variant'] == 'protocol' and r['plan_kind'] == 'protocol']
    single = {(r['label'], r['start']): r for r in proto if r['kind'] == 'single'}
    T = {k: (r['single_lap_s'] if (r['completed'] and r['verification']['pass'] and not r['setting_check']) else None)
         for k, r in single.items()}
    user_ok = {r['name']: r.get('user_verifier', {}).get('pass', False) for r in rows if 'user_verifier' in r}
    spread = seed_spread(T)
    acc = acceptance(T, user_ok, spread)
    flying = {}
    for r in proto:
        if r['kind'] == 'flying':
            flying[f"{r['label']}/{r['start']}"] = {'reason': r['reason'], 'completed_20s': r['completed'],
                                                    'laps': [{'lap': l['lap'], 'elapsed': l['elapsed'], 'valid': l['valid']} for l in r['laps']],
                                                    'lap1_s': r['lap1_s'], 'flying_lap_s': r['flying_lap_s'], 'passed_gates': r['passed_gates'],
                                                    'collisions': r['collisions'], 'sim_time': r['sim_time'], 'gate_misses_orig': r['gate_misses_orig'],
                                                    'collision_event': r['collision_event']}
    det = {}
    a, b = [next((r for r in rows if r['name'] == n), None) for n in ('SR-s0-E0-single', 'SR-s0-E0-single-rep2')]
    if a and b:
        ta = list(csv.DictReader(open(a['file'].replace('.json', '.trajectory.csv'))))
        tb = list(csv.DictReader(open(b['file'].replace('.json', '.trajectory.csv'))))
        n = min(len(ta), len(tb))
        diff = max((abs(float(ta[i][k]) - float(tb[i][k])) for i in range(n) for k in ('x_m', 'z_m', 'speed_mps')), default=None)
        det = {'run1': {'reason': a['reason'], 'frames': a['frames'], 'lap_s': a['single_lap_s'], 'passed_gates': a['passed_gates']},
               'run2': {'reason': b['reason'], 'frames': b['frames'], 'lap_s': b['single_lap_s'], 'passed_gates': b['passed_gates']},
               'max_abs_diff_common_frames': diff, 'identical': a['frames'] == b['frames'] and diff == 0}
        det['rule'] = '違えば全モデルの E0 を 3 回ずつの中央値(deviations.md に記録)'
    gains = {r['variant']: {'reason': r['reason'], 'lap_s': r['single_lap_s'], 'passed_gates': r['passed_gates'],
                            'user_verifier': r.get('user_verifier', {}).get('pass')}
             for r in rows if r['label'] == 'SR-s0' and r['kind'] == 'single' and r['start'] == 'E0' and r['plan_kind'] == 'protocol' and r['variant'] != 'rep2'}
    extra = [{k: r.get(k) for k in ('name', 'label', 'start', 'kind', 'variant', 'reason', 'single_lap_s', 'lap1_s', 'flying_lap_s',
                                     'passed_gates', 'collisions', 'control_delay_ticks', 'lookahead', 'steer_gain', 'throttle_gain', 'min_clearance_m',
                                     'sim_time', 'collision_event', 'gate_misses_orig')} | {'user_verifier': r.get('user_verifier', {}).get('pass')}
             for r in rows if r['plan_kind'] == 'extra']
    offline = json.loads((REPORT / 'train/offline_l1.json').read_text())
    off = {}
    for sname, s in offline['sets'].items():
        for mk, v in s['models'].items():
            off.setdefault(mk, {})[sname] = {k: v[k] for k in ('steer_step1_us', 'throttle_step1_us', 'steer_mean10_us', 'throttle_mean10_us', 'both_mean10_us')}
    ext_diff = {e: (None if T.get(('EXT', e)) is None else {lab: (None if T.get((lab, e)) is None else T[(lab, e)] - T[('EXT', e)])
                                                            for lab in ('SR-s0', 'SR-s1', 'ST-s0', 'ST-s1')}) for e in STARTS}
    figs = figures(rows, T, REF)
    # 走行ごとの表
    cols = ['name', 'label', 'start', 'kind', 'variant', 'plan_kind', 'reason', 'completed', 'single_lap_s', 'lap1_s', 'flying_lap_s', 'passed_gates',
            'collisions', 'crossings', 'sim_time', 'frames', 'min_clearance_m', 'max_speed', 'mean_speed', 'steer_edge_frac', 'steer_cmd_rate_p95',
            'control_delay_ticks', 'lookahead', 'mode_id', 'steer_gain', 'throttle_gain', 'latency_ms_median', 'ort_ms_median', 'model_sha256']
    with open(EVAL / 'runs.csv', 'w', newline='') as f:
        wri = csv.writer(f)
        wri.writerow(cols + ['verification_pass', 'user_verifier_pass', 'setting_check'])
        for r in sorted(rows, key=lambda r: r['name']):
            wri.writerow([r.get(c) for c in cols] + [r['verification']['pass'], r.get('user_verifier', {}).get('pass', ''), ';'.join(r['setting_check'])])
    sessions = sorted(CL.glob('session_*.json'))
    sess = [json.loads(p.read_text()) for p in sessions]
    speed = [{'label': j['label'], 'fps': j.get('fps'), 'frames': j.get('frames'), 'timing': j.get('timing'), 'batch_wall_s': j.get('batch_wall_s')}
             for s in sess for j in s.get('jobs', []) if j.get('fps')]
    ev = {
        'schema': 'sim-rl-bestlap-evaluation/v1',
        'created_at': __import__('time').strftime('%Y-%m-%dT%H:%M:%S'),
        'protocol_sha256': sha(REPORT / 'protocol.json'),
        'protocol_sha256_recorded': (REPORT / 'protocol.sha256').read_text().split()[0],
        'where': 'ブラウザ(Firefox 157、DISPLAY :0、WebGL 2 = Mesa llvmpipe)のアプリ(コピーした dist/index.html、元とバイト単位で同じ)の連続試験。'
                 'ブリッジはユーザーの bridge.py(コピー、元と同じ)を専用の venv(onnxruntime 1.31.0、CPU)で 127.0.0.1:8792',
        'settings': {'control_delay_frames': 1, 'lookahead': 1, 'steer_gain': 1.0, 'throttle_gain': '1.0(外部の参照は 1.01)',
                     'mode': '8(外部の参照は 4)', 'single_lap_limit_s': 15, 'physics_and_sensors': 'REC60 と同じ(RC 応答の物理・measured-height-approx)'},
        'bundles': bundles['models'],
        'references': {k: v for k, v in REF.items() if k != 'records'},
        'T_single_lap_from_stop': {f'{lab}/{e}': T.get((lab, e)) for lab in MODEL_BY_LABEL for e in STARTS},
        'single_runs': {f'{lab}/{e}': {k: r.get(k) for k in ('reason', 'completed', 'single_lap_s', 'passed_gates', 'collisions', 'sim_time', 'frames',
                                                            'min_clearance_m', 'collision_event', 'gate_misses_orig', 'max_speed', 'steer_edge_frac')}
                        for (lab, e), r in single.items()},
        'flying': flying,
        'acceptance_H2': acc,
        'user_verifier': {r['name']: r['user_verifier'] for r in rows if 'user_verifier' in r},
        'determinism': det,
        'gain_sensitivity_SR_s0_E0': gains,
        'extra_runs_not_judged': extra,
        'offline_l1_us': off,
        'takeover_diagnostic_not_judged': ({'note': TAKEOVER['note'], 'rows': [{k: r[k] for k in ('name', 'browser_reason', 'browser_frames', 'replay_same')} |
            {'relabel_steer_abs_mean_rad': r['relabel']['steer_abs_mean_rad'], 'relabel_steer_abs_mean_first20_rad': r['relabel']['steer_abs_mean_first20_rad'],
             'relabel_speed_abs_mean_mps': r['relabel']['speed_abs_mean_mps'], 'last_recoverable_frame': r['takeover']['last_recoverable_frame'],
             'first_unrecoverable_frame': r['takeover']['first_unrecoverable_frame'], 'k0_lap_s': r['takeover']['k0_lap_s']} for r in TAKEOVER['rows']]}
            if TAKEOVER else None),
        'diff_vs_external_reference_s': ext_diff,
        'verification_all_runs': {'runs': len(rows), 'consistency_pass': sum(r['verification'].get('consistency_pass', False) for r in rows),
                                  'lap_rule_pass': sum(r['verification']['pass'] for r in rows),
                                  'setting_mismatches': [r['name'] for r in rows if r['setting_check']], 'unreadable': problems},
        'speed': speed,
        'figures': figs,
        'files': {'runs_csv': str(EVAL / 'runs.csv'), 'runs_verified': str(CL / 'runs_verified.json'), 'closed_loop': str(CL)},
    }
    (REPORT / 'evaluation.json').write_text(json.dumps(ev, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    (EVAL / 'verdict.json').write_text(json.dumps({'acceptance_H2': acc}, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'verdict': acc['verdict'], 'T': ev['T_single_lap_from_stop'], 'a': acc['a_completion'], 'b': acc['b_close_to_rl']['median_ratio'],
                      'c': {k: acc['c_faster_than_teacher_trained'].get(k) for k in ('rule', 'n_pairs', 'median_diff', 'delta', 'pass')},
                      'flying': flying, 'determinism': det, 'gains': gains, 'user_verifier': {k: v['pass'] for k, v in ev['user_verifier'].items()},
                      'consistency': ev['verification_all_runs']}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
