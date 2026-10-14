#!/usr/bin/env python3
"""閉ループ評価の集計・判定・図(sim_dart_20261014 の評価の段、2026-10-14)

    node eval/verify_eval.mjs                       # 1 走行ずつの確かめ(closed_loop/runs_verified.json)
    python -I eval/analyze_eval.py --prepare        # teacher_eval.mjs に渡す走行の一覧(eval/teacher_runs.json)
    node eval/teacher_eval.mjs --runs ... --out eval/teacher_eval.json
    python -I eval/analyze_eval.py                  # この集計

  1) E0 の単周を、ユーザーの qa/verify-learned-lap.mjs <結果> --model <ONNX> で確かめる(元のフォルダーでそのまま実行。書くのは
     この作業の closed_loop/user_verifier/ だけ)
  2) 完走 = route_complete・衝突 0・33 ゲート・黒シート 1 回で、再生の数え直しが一致し(verify_eval)、設定が計画どおり。E0 は確認器も PASS
  3) protocol.json verdicts(結果を見る前に固定)をそのまま当てる: Q1(A1 対 A0-s0)・Q2(A2 対 A0-s0)・先生に近い速さ・一番良い腕
  4) 比べる表: A0-s0(E0〜E5 はベストラップの記録、E6〜E11 は今日)・A0-s1・S-T 種 0/1・作者の ONNX(E0〜E5 はベストラップ、
     E0 は今日の陽性対照も)・A1・A2・T_fast と T_margin(Node・遅延 0)
  5) 図: fig/eval_completion.png・fig/eval_trajectories.png・fig/eval_deviation.png・fig/eval_laptimes.png・fig/eval_first_corner.png
  6) evaluation.json・evaluation.md(レポートのフォルダーの直下)
ベストラップの記録は読むだけ。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_common import (BESTLAP_REPORT, CL, CORE, EVAL, EXT_STARTS, MODEL_BY_LABEL, REPORT, STARTS, USER_SIM, run_meta)  # noqa: E402

FIG = REPORT / 'fig'
BL_CL = BESTLAP_REPORT / 'eval' / 'closed_loop'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
COLORS = {'A0-s0': '#2a78d6', 'A0-s1': '#7fb0ea', 'A1-s0': '#d6452a', 'A2-s0': '#4a9a5b', 'ST-s0': '#eb6834', 'ST-s1': '#f3a77f',
          'EXT': '#8a5bb5', 'T_fast': '#0b0b0b', 'T_margin': '#6f6e69'}
NAMES = {'A0-s0': 'A0(S-R 種 0。きれいな RL 20 分)', 'A0-s1': 'A0 種 1(S-R 種 1)', 'A1-s0': 'A1(A0 + DART 10 分)', 'A2-s0': 'A2(余裕の線 10 分)',
         'ST-s0': 'S-T 種 0(作者の先生のデータ)', 'ST-s1': 'S-T 種 1', 'EXT': '作者の ONNX', 'T_fast': 'T_fast(RL 最速の先生・Node・遅延 0)',
         'T_margin': 'T_margin(余裕の線の先生・Node・遅延 0)'}
SHORT = {'A0-s0': 'A0', 'A0-s1': 'A0 種1', 'A1-s0': 'A1', 'A2-s0': 'A2', 'ST-s0': 'S-T 種0', 'ST-s1': 'S-T 種1', 'EXT': '作者 ONNX',
         'T_fast': 'T_fast', 'T_margin': 'T_margin'}
BL_LABEL = {'SR-s0': 'A0-s0', 'SR-s1': 'A0-s1', 'ST-s0': 'ST-s0', 'ST-s1': 'ST-s1', 'EXT': 'EXT'}   # ベストラップの名前 → この報告の名前
ARM_TEACHER = {'A0-s0': 'fast', 'A0-s1': 'fast', 'A1-s0': 'fast', 'A2-s0': 'margin', 'ST-s0': 'fast', 'ST-s1': 'fast', 'EXT': 'fast'}
REASON_JA = {'collision': '衝突', 'time_limit': '時間切れ', 'route_complete': '完走', 'gate_miss': 'ゲート外し', 'aborted': '中断'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def fisher_two_sided(a, n1, b, n2):
    """2×2 の Fisher の正確検定(両側)。a/n1 と b/n2(参考)"""
    k = a + b
    n = n1 + n2

    def p(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)
    p0 = p(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-9)))


def user_verifier(name, file, onnx):
    out = CL / 'user_verifier' / f'{name}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(['node', 'qa/verify-learned-lap.mjs', file, '--model', onnx, '--output', str(out)],
                       cwd=USER_SIM, capture_output=True, text=True, timeout=600)
    try:
        d = json.loads(out.read_text())
        res = d['results'][0]
        return {'pass': bool(d['pass']), 'exit_code': r.returncode, 'failure_count': res.get('failure_count'),
                'failures': res.get('failures', [])[:8], 'replay_elapsed': (res.get('replay') or {}).get('elapsed'), 'output': str(out)}
    except (OSError, ValueError, KeyError) as e:
        return {'pass': False, 'exit_code': r.returncode, 'error': f'{type(e).__name__}: {e}', 'stderr': r.stderr[-800:]}


def load_rows():
    """この作業の走行(closed_loop/runs_verified.json)と、ベストラップの E0〜E5 の単周(読むだけ)"""
    mine = json.loads((CL / 'runs_verified.json').read_text())['rows']
    bl = json.loads((BL_CL / 'runs_verified.json').read_text())['rows']
    rows = []
    for r in mine:
        if 'error' in r:
            rows.append(r | {'source': 'today'})
            continue
        lab, start, kind, variant = run_meta(r['name'])
        rows.append(r | {'label': lab, 'start': start, 'kind': kind, 'variant': variant, 'source': 'today'})
    for r in bl:
        if 'error' in r:
            continue
        lab, start, kind, variant = run_meta(r['name'])
        if lab not in BL_LABEL or kind != 'single' or variant != 'protocol' or start not in CORE:
            continue
        rows.append(r | {'label': BL_LABEL[lab], 'bestlap_label': lab, 'start': start, 'kind': kind, 'variant': variant, 'source': 'bestlap_20261013'})
    return rows


def prepare():
    rows = load_rows()
    out = []
    for r in rows:
        if 'error' in r or r['kind'] != 'single':
            continue
        out.append({'name': r['name'], 'label': r['label'], 'file': r['file'], 'teacher': ARM_TEACHER[r['label']],
                    'takeover': r['source'] == 'today' and r['label'] != 'EXT' and r['variant'] == 'protocol'})
    p = EVAL / 'teacher_runs.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(p, len(out))


def setting_check(r, plan_c, onnx_sha):
    chk = []
    if onnx_sha and r['model_sha256'] != onnx_sha:
        chk.append('モデルの sha256 が一式と違う')
    if plan_c:
        if r['control_delay_ticks'] != plan_c['control-delay-frames']:
            chk.append('遅延が計画と違う')
        if r['lookahead'] != plan_c['model-lookahead'] or r['mode_id'] != plan_c['model-mode']:
            chk.append('先読み・ジャンルが計画と違う')
        if abs((r['steer_gain'] or 0) - plan_c['model-steer-gain']) > 1e-9 or abs((r['throttle_gain'] or 0) - plan_c['model-throttle-gain']) > 1e-9:
            chk.append('ゲインが計画と違う')
        g, o, y = STARTS[r['start']]
        sv = r['start_variation']
        if sv['gate'] != g or abs(sv['left_offset_m'] - o) > 1e-9 or abs(sv['left_yaw_deg'] - y) > 1e-6:
            chk.append('開始が計画と違う')
    return chk


# ---------------------------------------------------------------- 図
def jp_font():
    import matplotlib
    from matplotlib import font_manager
    for name in ('Noto Sans CJK JP', 'Noto Sans JP', 'IPAexGothic', 'IPAGothic', 'TakaoGothic', 'VL Gothic', 'Droid Sans Fallback'):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams['font.family'] = name
            return name
    return None


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


def course(ax, world, plt, labels_every=4):
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
    b = world['bounds']
    ax.set_xlim(b['minX'], b['maxX'])
    ax.set_ylim(b['maxZ'], b['minZ'])
    ax.set_aspect('equal')


def figures(rows_by, starts_cnt, T, teach, te_rows, world0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    jp_font()
    FIG.mkdir(exist_ok=True)
    figs = []
    # 1) 完走の数の棒
    arms = ['A0-s0', 'A1-s0', 'A2-s0', 'A0-s1', 'ST-s0', 'ST-s1', 'EXT']
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 4.9), facecolor=SURFACE, gridspec_kw={'width_ratios': [1.15, 1]})
    style(ax, '完走の数(単周・遅延 1・ブラウザのアプリ)。濃い = E0〜E5、薄い = E6〜E11。枠 = 走らせた本数', '完走した本数')
    for i, lab in enumerate(arms):
        c_core = sum(1 for e in CORE if T.get((lab, e)) is not None)
        c_ext = sum(1 for e in EXT_STARTS if T.get((lab, e)) is not None)
        n_core = sum(1 for e in CORE if (lab, e) in rows_by)
        n_ext = sum(1 for e in EXT_STARTS if (lab, e) in rows_by)
        col = COLORS[lab]
        ax.bar(i, n_core + n_ext, .62, color='none', edgecolor=GRID, lw=1.2)
        ax.bar(i, c_core, .62, color=col)
        ax.bar(i, c_ext, .62, bottom=c_core, color=col, alpha=.45)
        ax.text(i, c_core + c_ext + .2, f'{c_core + c_ext} / {n_core + n_ext}', ha='center', va='bottom', fontsize=10, color=INK)
    ax.set_xticks(range(len(arms)))
    TICK = {'A0-s0': 'A0\n(きれいな RL)', 'A1-s0': 'A1\n(A0 + DART)', 'A2-s0': 'A2\n(余裕の線)', 'A0-s1': 'A0 種 1', 'ST-s0': 'S-T 種 0\n(作者の先生)',
            'ST-s1': 'S-T 種 1', 'EXT': '作者の ONNX'}
    ax.set_xticklabels([TICK[a] for a in arms], fontsize=8)
    ax.set_ylim(0, 13.5)
    ax.text(len(arms) - .5, 12.8, 'T_fast・T_margin(Node・遅延 0)は 12 / 12', ha='right', fontsize=8, color=INK2)
    # 右: 止まるまで(完走は 1 周)に参照の周 R に沿って進んだ道のり。点 = 1 走行、横線 = 中央値
    style(ax2, '止まるまでに進んだ道のり(参照の周に沿った長さ。点 = 1 走行、横線 = 中央値)', '進んだ道のり [m]')
    L = None
    for i, lab in enumerate(arms):
        vals = []
        for e in STARTS:
            r = rows_by.get((lab, e))
            q = (r or {}).get('teacher_eval') or {}
            if r is None or q.get('progress_m') is None:
                continue
            L = L or (q['progress_m'] / q['lap_fraction'] if q.get('lap_fraction') else None)
            v = q['progress_m']
            vals.append(v)
            ok = T.get((lab, e)) is not None
            ax2.plot(i + (list(STARTS).index(e) % 7 - 3) * .04, v, marker='o' if ok else 'x', color=COLORS[lab], ms=5, mew=1.5, alpha=.85, ls='none')
        if vals:
            m = statistics.median(vals)
            ax2.hlines(m, i - .3, i + .3, color=COLORS[lab], lw=2.2)
            ax2.text(i + .32, m, f'{m:.1f}', va='center', fontsize=8, color=COLORS[lab])
    if L:
        ax2.axhline(L, color=INK2, lw=1, ls='--')
        ax2.text(len(arms) - .5, L + .6, f'1 周 = {L:.1f} m', ha='right', fontsize=8, color=INK2)
    ax2.set_xticks(range(len(arms)))
    ax2.set_xticklabels([TICK[a] for a in arms], fontsize=8)
    ax2.set_ylim(0, 48)
    fig.tight_layout()
    fig.savefig(FIG / 'eval_completion.png', dpi=130, facecolor=SURFACE)
    plt.close(fig)
    figs.append('fig/eval_completion.png')

    # 2) 軌跡: 腕ごとに全部の単周(× = 衝突、□ = 時間切れの終わり)と、その腕の先生の E0
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), facecolor=SURFACE)
    for ax, lab in zip(axes, ['A0-s0', 'A1-s0', 'A2-s0']):
        nc = sum(1 for e in STARTS if T.get((lab, e)) is not None)
        nr = sum(1 for e in STARTS if (lab, e) in rows_by)
        style(ax, f'{NAMES[lab]}: 完走 {nc} / {nr}(× 衝突・□ 時間切れ)', 'z [m]', 'x [m]')
        course(ax, world0, plt)
        tk = 'T_fast' if ARM_TEACHER[lab] == 'fast' else 'T_margin'
        xy = teach['E0'][ARM_TEACHER[lab]]['xy']
        ax.plot([p[0] for p in xy], [p[1] for p in xy], color=COLORS[tk], lw=1.3, ls='--', label=NAMES[tk] + ' の E0')
        for e in STARTS:
            r = rows_by.get((lab, e))
            if not r:
                continue
            tr = json.loads(Path(r['file']).read_text())['trace']
            if r['start_variation']['gate'] != 0:
                continue   # 開始の場所が違う E5・E10・E11 は、重なって読みにくいので描かない(表に書く)
            ok = T.get((lab, e)) is not None
            ax.plot([t['x'] for t in tr], [t['z'] for t in tr], color=COLORS[lab], lw=1.0 if ok else 1.3, alpha=.55 if ok else .95)
            if r['collisions']:
                ax.plot(tr[-1]['x'], tr[-1]['z'], marker='x', color=COLORS[lab], ms=9, mew=2.2)
            elif not ok:
                ax.plot(tr[-1]['x'], tr[-1]['z'], marker='s', mfc='none', color=COLORS[lab], ms=8, mew=1.8)
        ax.legend(fontsize=7.5, loc='lower left')
    fig.suptitle('ゲート 0 からの単周(E0〜E4・E6〜E9)の軌跡。E5・E10・E11 は別の場所から始まるので表に書いた', fontsize=10, color=INK2, x=.01, ha='left')
    fig.tight_layout()
    fig.savefig(FIG / 'eval_trajectories.png', dpi=115, facecolor=SURFACE)
    plt.close(fig)
    figs.append('fig/eval_trajectories.png')

    # 3) ずれの時間変化(その腕の先生の同じ開始の走りから)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), facecolor=SURFACE, sharey=True)
    for ax, lab in zip(axes, ['A0-s0', 'A1-s0', 'A2-s0']):
        tk = 'T_fast' if ARM_TEACHER[lab] == 'fast' else 'T_margin'
        style(ax, f'{SHORT[lab]}: {tk} の同じ開始の線からの距離(× = 衝突)', '線からの距離 [cm]', '時間 [秒]')
        for e in STARTS:
            q = te_rows.get(f'{lab}-{e}-single') or te_rows.get(f"{ {'A0-s0': 'SR-s0'}.get(lab, lab)}-{e}-single")
            r = rows_by.get((lab, e))
            if not q or not r:
                continue
            t = q['series']['t']
            d = [None if x is None else x * 100 for x in q['series']['path_teacher']]
            ok = T.get((lab, e)) is not None
            ax.plot(t, [x if x is not None else float('nan') for x in d], color=COLORS[lab], lw=1.0, alpha=.45 if ok else .9)
            if r['collisions'] and d and d[-1] is not None:
                ax.plot(t[-1], d[-1], marker='x', color=COLORS[lab], ms=8, mew=2)
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 60)
    fig.tight_layout()
    fig.savefig(FIG / 'eval_deviation.png', dpi=120, facecolor=SURFACE)
    plt.close(fig)
    figs.append('fig/eval_deviation.png')

    # 4) 止まるまでの時間(完走した走りは周回時間)と、先生の周回時間
    fig, ax = plt.subplots(figsize=(13.5, 4.8), facecolor=SURFACE)
    style(ax, '止まるまでの時間(完走した走りは停止から 1 周の時間)。棒の上の数字 = 数えたゲート / 33。線 = 先生の周回時間(Node・遅延 0)',
          '時間 [秒]')
    labs = ['A0-s0', 'A1-s0', 'A2-s0']
    w = .25
    for j, e in enumerate(STARTS):
        for k, lab in enumerate(labs):
            x = j + (k - 1) * w
            r = rows_by.get((lab, e))
            if r is None:
                continue
            t = T.get((lab, e))
            v = t if t is not None else r['sim_time']
            ax.bar(x, v, w * .9, color=COLORS[lab], alpha=1 if t is not None else .75, label=NAMES[lab] if j == 0 else None,
                   hatch=None if t is not None else None)
            ax.text(x, v + .1, str(r['passed_gates']), ha='center', va='bottom', fontsize=6.5, color=COLORS[lab])
        for kind, tk, ls in (('fast', 'T_fast', '-'), ('margin', 'T_margin', '--')):
            v = teach[e][kind]['lap_s']
            if v:
                ax.hlines(v, j - 1.6 * w, j + 1.6 * w, color=COLORS[tk], lw=1.3, ls=ls, label=NAMES[tk] if j == 0 else None)
    ax.set_xticks(range(len(STARTS)))
    ax.set_xticklabels([f'{e}\n(G{g}・{o:+.2f}・{y:+.0f}°)' for e, (g, o, y) in STARTS.items()], fontsize=7.5)
    ax.set_ylim(0, 12.5)
    ax.legend(fontsize=7.5, ncol=5, loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / 'eval_stop_times.png', dpi=130, facecolor=SURFACE)
    plt.close(fig)
    figs.append('fig/eval_stop_times.png')

    # 5) 止まった場所(全部の腕・全部の開始)
    fig, ax = plt.subplots(figsize=(12, 7.2), facecolor=SURFACE)
    style(ax, '止まった場所(× = 衝突、□ = 時間切れ)。色 = 腕。A0・S-T・作者の ONNX の E0〜E5 はベストラップの記録', 'z [m]', 'x [m]')
    course(ax, world0, plt)
    seen = set()
    for (lab, e), r in sorted(rows_by.items()):
        if T.get((lab, e)) is not None:
            continue
        ep = r.get('end_pose') or {}
        if 'x' not in ep:
            continue
        ax.plot(ep['x'], ep['z'], marker='x' if r['collisions'] else 's', mfc='none', color=COLORS[lab], ms=9, mew=2,
                label=NAMES[lab] if lab not in seen else None, ls='none')
        seen.add(lab)
    ax.legend(fontsize=7.5, loc='lower left')
    fig.tight_layout()
    fig.savefig(FIG / 'eval_failure_map.png', dpi=115, facecolor=SURFACE)
    plt.close(fig)
    figs.append('fig/eval_failure_map.png')
    return figs


# ---------------------------------------------------------------- 本体
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--no-user-verifier', action='store_true')
    a = ap.parse_args()
    if a.prepare:
        prepare()
        return
    protocol = json.loads((REPORT / 'protocol.json').read_text())
    rows = load_rows()
    problems = [r for r in rows if 'error' in r]
    rows = [r for r in rows if 'error' not in r]
    runs_json = json.loads((REPORT / 'train' / 'runs.json').read_text())['runs']
    bl_bundles = {m['label']: m for m in json.loads((BESTLAP_REPORT / 'eval/bundles/bundles.json').read_text())['models']}
    onnx_sha = {'A1-s0': runs_json['A1/s0']['onnx_sha256'], 'A2-s0': runs_json['A2/s0']['onnx_sha256'],
                'A0-s0': runs_json['A0/s0']['onnx_sha256'], 'A0-s1': bl_bundles['SR-s1']['onnx_sha256'],
                'ST-s0': bl_bundles['ST-s0']['onnx_sha256'], 'ST-s1': bl_bundles['ST-s1']['onnx_sha256'], 'EXT': bl_bundles['EXT']['onnx_sha256']}
    # 計画(この作業 + ベストラップの protocol の計画)
    plans = {}
    for d in [CL, BL_CL]:
        for p in d.glob('*/plan_*.json'):
            for c in json.loads(p.read_text()):
                plans[c['name']] = c
    bl_eval = json.loads((BESTLAP_REPORT / 'evaluation.json').read_text())
    for r in rows:
        r['setting_check'] = setting_check(r, plans.get(r['name']), onnx_sha.get(r['label']))
        g0 = (r.get('start_variation') or {}).get('gate', 0)
        r['gate_misses_orig'] = [{'route_point': (g0 + m['gate'] + 1) % 33, 'order_in_lap': m['gate'] % 33 + 1, 'time': m['time'], 'lateral_m': m['lateral_m']}
                                 for m in r.get('gate_misses', [])]
        if r['kind'] == 'single' and r['start'] == 'E0' and r['control_delay_ticks'] == 1:
            if r['source'] == 'today':
                if not a.no_user_verifier:
                    onnx = MODEL_BY_LABEL[r['label']]['onnx']
                    r['user_verifier'] = user_verifier(r['name'], r['file'], onnx)
            else:
                uv = bl_eval['user_verifier'].get(r['name'])
                if uv:
                    r['user_verifier'] = uv | {'source': 'bestlap_20261013'}
    # 走行の表(単周・protocol)
    single = {}
    for r in rows:
        if r['kind'] == 'single' and r['variant'] == 'protocol':
            single[(r['label'], r['start'])] = r

    def is_complete(r):
        ok = r['completed'] and r['verification']['pass'] and not r['setting_check'] and r['single_lap_s'] is not None
        if ok and r['start'] == 'E0' and 'user_verifier' in r:
            ok = ok and r['user_verifier'].get('pass', False)
        return ok
    T = {k: (r['single_lap_s'] if is_complete(r) else None) for k, r in single.items()}
    # 先生の目(teacher_eval.json)
    te = json.loads((EVAL / 'teacher_eval.json').read_text())
    te_rows = {r['name']: r for r in te['rows']}
    teach = {}
    for e, (g, o, y) in STARTS.items():
        key = f'{g}|{o:.4f}|{y:.4f}'
        if key in te['teacher_refs']:
            teach[e] = te['teacher_refs'][key]
    bl_to = {x['name']: x for x in json.loads((BESTLAP_REPORT / 'eval' / 'takeover.json').read_text())['rows']}
    for r in rows:
        q = te_rows.get(r['name'])
        if q and r['source'] != 'today' and r['name'] in bl_to and not q.get('takeover'):
            # ベストラップの引き継ぎ(eval/takeover.json、k を 3 おきに全部試したもの。読むだけ)
            b = bl_to[r['name']]
            k = b['takeover']['last_recoverable_frame']
            q['takeover'] = {'fast': {'every': b['takeover']['every'], 'last_recoverable_frame': k, 'k0_lap_s': b['takeover'].get('k0_lap_s'),
                                      'seconds_before_end': None if k is None else round((b['browser_frames'] - 1 - k) / 30, 4),
                                      'source': 'bestlap_20261013/eval/takeover.json'}}
        if q:
            # 参照の周 R に沿った進みが 1 周に届いた時刻(ゲートを数えたかとは関係ない。参考)
            ss, tt, L = q['series']['s'], q['series']['t'], q.get('loop_length_m')
            i = next((i for i, v in enumerate(ss) if L and v is not None and v - ss[0] >= L), None)
            q['progress_full_lap_t'] = tt[i] if i is not None else None
            r['teacher_eval'] = {k: q.get(k) for k in ('dev_fast_m', 'dev_teacher_m', 'path_dev_fast_m', 'path_dev_teacher_m', 'first_straight_passed',
                                                       'takeover', 'replay_same', 'teacher', 'first_gate_miss', 'gate_crossings', 'progress_m',
                                                       'lap_fraction', 'loop_length_m', 'progress_full_lap_t')}
    # 腕ごとのまとめ
    labels = ['A0-s0', 'A1-s0', 'A2-s0', 'A0-s1', 'ST-s0', 'ST-s1', 'EXT']
    summary = {}
    for lab in labels:
        rs = {e: single.get((lab, e)) for e in STARTS}
        ran = [e for e in STARTS if rs[e]]
        comp = [e for e in ran if T.get((lab, e)) is not None]
        stops = [rs[e]['sim_time'] for e in ran if T.get((lab, e)) is None]
        fs_starts = [e for e in ran if rs[e]['start_variation']['gate'] == 0]
        fs_pass = [e for e in fs_starts if (rs[e].get('teacher_eval') or {}).get('first_straight_passed')]
        tk = ARM_TEACHER[lab]
        ratios = [T[(lab, e)] / teach[e][tk]['lap_s'] for e in comp if e in teach and teach[e][tk]['lap_s']]
        summary[lab] = {
            'ran': ran, 'n_ran': len(ran), 'completed': comp, 'n_completed': len(comp),
            'n_completed_E0_E5': sum(1 for e in comp if e in CORE), 'n_ran_E0_E5': sum(1 for e in ran if e in CORE),
            'n_completed_E6_E11': sum(1 for e in comp if e in EXT_STARTS), 'n_ran_E6_E11': sum(1 for e in ran if e in EXT_STARTS),
            'lap_median_s': med([T[(lab, e)] for e in comp]), 'laps_s': {e: T[(lab, e)] for e in comp},
            'stop_time_median_s': med(stops), 'stop_times_s': {e: rs[e]['sim_time'] for e in ran if T.get((lab, e)) is None},
            'first_straight_passed': len(fs_pass), 'first_straight_starts': len(fs_starts), 'first_straight_pass_list': fs_pass,
            'teacher': tk, 'lap_ratio_to_teacher_median': med(ratios), 'lap_ratios': ratios,
            'progress_m_median_failed': med([(rs[e].get('teacher_eval') or {}).get('progress_m') for e in ran if T.get((lab, e)) is None]),
            'lap_fraction_by_start': {e: (rs[e].get('teacher_eval') or {}).get('lap_fraction') for e in ran},
        }
    # 先生(Node)
    for kind, lab in (('fast', 'T_fast'), ('margin', 'T_margin')):
        laps = {e: teach[e][kind]['lap_s'] for e in STARTS if e in teach}
        summary[lab] = {'n_ran': len(laps), 'n_completed': sum(1 for v in laps.values() if v), 'laps_s': laps, 'lap_median_s': med(list(laps.values())),
                        'min_clearance_m': {e: teach[e][kind]['min_clearance_m'] for e in teach}}
    # 判定(protocol.json verdicts をそのまま)
    c = {lab: summary[lab]['n_completed'] for lab in ('A0-s0', 'A1-s0', 'A2-s0')}
    n12 = {lab: summary[lab]['n_ran'] for lab in ('A0-s0', 'A1-s0', 'A2-s0')}

    def q_verdict(arm):
        d = c[arm] - c['A0-s0']
        fs = summary[arm]['first_straight_passed']
        if c[arm] >= 6 and d >= 4:
            v = '効いた'
        elif d >= 2 or fs >= 8:
            v = '一部効いた'
        else:
            v = '効かなかった'
        return {'c_arm': c[arm], 'c_A0_s0': c['A0-s0'], 'diff': d, 'n_runs_arm': n12[arm], 'n_runs_A0': n12['A0-s0'],
                'first_straight_passed_gate0_starts': fs, 'gate0_starts_run': summary[arm]['first_straight_starts'],
                'rule': protocol['verdicts']['Q1_DART' if arm == 'A1-s0' else 'Q2_MARGIN'],
                'rule_note': 'protocol は「ゲート 0 の開始 10 本(E0〜E4・E6〜E9)」と書くが、挙げた開始は 9 本。挙げた 9 本で数え、8 本以上を条件とした(deviations.md)',
                'verdict': v, 'fisher_p_two_sided_reference': fisher_two_sided(c[arm], n12[arm], c['A0-s0'], n12['A0-s0'])}
    verdicts = {'Q1_DART': q_verdict('A1-s0'), 'Q2_MARGIN': q_verdict('A2-s0')}
    a2_laps = summary['A2-s0']['laps_s']
    verdicts['Q2_MARGIN']['speed_cost'] = {
        'A2_lap_median_s': summary['A2-s0']['lap_median_s'],
        'T_fast_node_median_same_starts_s': med([teach[e]['fast']['lap_s'] for e in a2_laps]),
        'T_margin_node_median_same_starts_s': med([teach[e]['margin']['lap_s'] for e in a2_laps]),
        'A1_lap_median_same_starts_s': med([T.get(('A1-s0', e)) for e in a2_laps]),
        'per_start': {e: {'A2': a2_laps[e], 'T_fast': teach[e]['fast']['lap_s'], 'T_margin': teach[e]['margin']['lap_s'], 'A1': T.get(('A1-s0', e))} for e in a2_laps},
    }
    verdicts['Q3_BOTH'] = {'verdict': '今日は答えない(A3 は記録の段で削った。deviations.md 11)'}
    verdicts['Q4_DAGGER'] = {'verdict': '今日は答えない(A4 は電源断のあとの時間割で削った。deviations.md 17)'}
    verdicts['speed_vs_teacher'] = {lab: {'median_ratio': summary[lab]['lap_ratio_to_teacher_median'],
                                          'close_to_teacher': (summary[lab]['lap_ratio_to_teacher_median'] or 9) <= 1.05,
                                          'teacher': summary[lab]['teacher']} for lab in ('A1-s0', 'A2-s0')}
    cand = sorted(['A1-s0', 'A2-s0', 'A0-s0'], key=lambda lab: (-c[lab], summary[lab]['lap_median_s'] or 99))
    tie = c[cand[0]] == c[cand[1]] and summary[cand[0]]['lap_median_s'] is None and summary[cand[1]]['lap_median_s'] is None
    verdicts['best_arm'] = {'arm': None if tie else cand[0], 'rule': protocol['verdicts']['best_arm'], 'c': c,
                            'note': '完走の数が同じで、同点を分ける周回時間も無いときは決めない' if tie else None}
    # 決定性(A1 の E0 を 2 回)
    det = {}
    ra = next((r for r in rows if r['name'] == 'A1-s0-E0-single'), None)
    rb = next((r for r in rows if r['name'] == 'A1-s0-E0-single-rep2'), None)
    if ra and rb:
        ta = list(csv.DictReader(open(ra['file'].replace('.json', '.trajectory.csv'))))
        tb = list(csv.DictReader(open(rb['file'].replace('.json', '.trajectory.csv'))))
        n = min(len(ta), len(tb))
        diff = max((abs(float(ta[i][k]) - float(tb[i][k])) for i in range(n) for k in ('x_m', 'z_m', 'speed_mps')), default=None)
        det = {'run1': {'reason': ra['reason'], 'frames': ra['frames'], 'lap_s': ra['single_lap_s'], 'passed_gates': ra['passed_gates']},
               'run2': {'reason': rb['reason'], 'frames': rb['frames'], 'lap_s': rb['single_lap_s'], 'passed_gates': rb['passed_gates']},
               'max_abs_diff_common_frames': diff, 'identical': ra['frames'] == rb['frames'] and diff == 0,
               'rule': protocol['eval']['determinism_check']}
    # 陽性対照(今日の作者の ONNX の E0)
    ext_today = next((r for r in rows if r['name'] == 'EXT-E0-single' and r['source'] == 'today'), None)
    pos = None
    if ext_today:
        pos = {'reason': ext_today['reason'], 'lap_s': ext_today['single_lap_s'], 'passed_gates': ext_today['passed_gates'],
               'user_verifier': (ext_today.get('user_verifier') or {}).get('pass'), 'bestlap_E0_s': bl_eval['T_single_lap_from_stop'].get('EXT/E0'),
               'file': ext_today['file']}
    # 連続周回(参考)
    flying = {}
    for r in rows:
        if r['kind'] == 'flying' and r['source'] == 'today':
            flying[r['label']] = {k: r.get(k) for k in ('reason', 'completed', 'laps', 'lap1_s', 'flying_lap_s', 'passed_gates', 'collisions', 'sim_time',
                                                        'collision_event', 'gate_misses_orig')}
    # 点 4 のゲート(ゲート 0 の開始では目標 3)を通った横の位置。先生と腕
    def p4(crossings):
        c = next((x for x in (crossings or []) if x['point'] == 4), None)
        if not c:
            return None
        return c
    gp_rows, gp_num = {}, {}
    for e, (g, o, y) in STARTS.items():
        if g != 0 or e not in teach:
            continue
        row = {}
        gp_num[e] = {name: (p4(teach[e][kind].get('gate_crossings')) or {}).get('margin_m') for kind, name in (('fast', 'T_fast'), ('margin', 'T_margin'))}
        for kind, name in (('fast', 'T_fast'), ('margin', 'T_margin')):
            c = p4(teach[e][kind].get('gate_crossings'))
            row[name] = '—' if not c else f"{c['lateral_m']:+.3f} m({c['margin_m'] * 100:.1f} cm)"
        for lab in ('A1-s0', 'A2-s0', 'A0-s0'):
            r = single.get((lab, e))
            if not r:
                row[lab] = '·'
                continue
            c = p4((r.get('teacher_eval') or {}).get('gate_crossings'))
            if c is None:
                row[lab] = f"届く前に止まった({r['sim_time']:.2f} 秒)"
            elif c['counted']:
                row[lab] = f"{c['lateral_m']:+.3f} m({c['margin_m'] * 100:.1f} cm)"
            else:
                row[lab] = f"外し(線を {c['lateral_m']:+.2f} m で越えた。{c['t']:.2f} 秒)"
        gp_rows[e] = row
    gate_point4 = {'rows': gp_rows, 'teacher_margin_m': gp_num, 'text': (
        '横の位置は、点 4 のゲートの線(点 3 → 点 4 の向きに垂直)での、ゲートの中心からの横(正負は向きの左右。ゲートの半幅 0.65 m)。'
        '先生も A1・A2 も負の側(曲がりの内側)。括弧は 0.65 m − |横|(ゲートの端までの余裕)。数えたゲートは、sim が数えた最初のフレームの位置で測った(およその値)。'
        '「外し」は、ゲートの幅の中で線を越えず、曲がったあとに線の延長をずっと横で越えたもの(手前で内側へ曲がった)。')}
    world0 = json.loads(Path(single[('A1-s0', 'E0')]['file']).read_text())['world'] if ('A1-s0', 'E0') in single else None
    figs = figures(single, None, T, teach, te_rows, world0) if world0 else []
    cov = None
    if (EVAL / 'dart_coverage.json').exists():   # eval/dart_coverage.py(参考。結果を見てから足した)
        cv = json.loads((EVAL / 'dart_coverage.json').read_text())
        cov = {k: cv[k] for k in ('note', 'dataset', 'scenes', 'noisy_frames_total', 'event_starts_total', 'unperturbed_segments', 'run_ends_by_arm',
                                  'run_ends', 'A1_first_departure', 'segments_0_to_4')}
        cov['noisy_frac_per_segment'] = [round(a / b, 4) if b else None for a, b in zip(cv['noisy_frames_per_segment'], cv['frames_per_segment'])]
        cov['event_starts_per_segment'] = cv['event_starts_per_segment']
        if (FIG / 'eval_dart_coverage.png').exists():
            figs.append('fig/eval_dart_coverage.png')
    # 走行の表(CSV)
    cols = ['name', 'label', 'source', 'start', 'kind', 'variant', 'reason', 'completed', 'single_lap_s', 'lap1_s', 'flying_lap_s', 'passed_gates',
            'collisions', 'crossings', 'sim_time', 'frames', 'min_clearance_m', 'max_speed', 'mean_speed', 'control_delay_ticks', 'lookahead', 'mode_id',
            'steer_gain', 'throttle_gain', 'latency_ms_median', 'ort_ms_median', 'model_sha256']
    with open(EVAL / 'runs.csv', 'w', newline='') as f:
        wri = csv.writer(f)
        wri.writerow(cols + ['counted_complete', 'verification_pass', 'user_verifier_pass', 'setting_check', 'path_dev_teacher_0p5s_m', 'path_dev_teacher_end_m',
                             'path_dev_teacher_max_m', 'path_dev_fast_end_m', 'first_straight_passed', 'takeover_fast_last_k', 'takeover_teacher_last_k'])
        for r in sorted(rows, key=lambda r: (r['source'], r['name'])):
            tev = r.get('teacher_eval') or {}
            tko = tev.get('takeover') or {}
            wri.writerow([r.get(cc) for cc in cols] + [
                is_complete(r) if r['kind'] == 'single' else '', r['verification']['pass'], (r.get('user_verifier') or {}).get('pass', ''),
                ';'.join(r['setting_check']), (tev.get('path_dev_teacher_m') or {}).get('at_0p5s'), (tev.get('path_dev_teacher_m') or {}).get('at_end'),
                (tev.get('path_dev_teacher_m') or {}).get('max'), (tev.get('path_dev_fast_m') or {}).get('at_end'), tev.get('first_straight_passed'),
                (tko.get('fast') or {}).get('last_recoverable_frame'), (tko.get(tev.get('teacher') or 'fast') or {}).get('last_recoverable_frame')])
    sessions = sorted(CL.glob('session_*.json'))
    sess = [json.loads(p.read_text()) for p in sessions]
    speed = [{'label': j['label'], 'phase': j.get('phase'), 'fps': j.get('fps'), 'frames': j.get('frames'), 'timing': j.get('timing'),
              'batch_wall_s': j.get('batch_wall_s'), 'error': j.get('error')} for s in sess for j in s.get('jobs', []) if j.get('fps') is not None]

    def run_brief(r):
        tev = r.get('teacher_eval') or {}
        ce = r.get('collision_event') or {}
        return {'name': r['name'], 'source': r['source'], 'reason': r['reason'], 'counted_complete': is_complete(r), 'single_lap_s': r['single_lap_s'],
                'passed_gates': r['passed_gates'], 'collisions': r['collisions'], 'sim_time': r['sim_time'], 'frames': r['frames'],
                'min_clearance_m': r['min_clearance_m'], 'collision': ({'time': ce.get('time'), 'x': ce.get('x'), 'z': ce.get('z'), 'obstacle': ce.get('obstacle'),
                                                                      'collider': (ce.get('raw') or {}).get('collider'), 'kind': (ce.get('raw') or {}).get('kind'),
                                                                      'frame': (ce.get('raw') or {}).get('frame')}
                                                                     if ce else None),
                'end_pose': r.get('end_pose'), 'gate_misses_orig': r['gate_misses_orig'], 'verification_pass': r['verification']['pass'],
                'consistency_pass': r['verification'].get('consistency_pass'), 'setting_check': r['setting_check'],
                'user_verifier': (r.get('user_verifier') or {}).get('pass') if 'user_verifier' in r else None,
                'path_dev_teacher_m': tev.get('path_dev_teacher_m'), 'path_dev_fast_m': tev.get('path_dev_fast_m'),
                'dev_teacher_s_matched_m': tev.get('dev_teacher_m'), 'dev_fast_s_matched_m': tev.get('dev_fast_m'), 'first_straight_passed': tev.get('first_straight_passed'),
                'takeover': tev.get('takeover'), 'teacher_gate_miss': tev.get('first_gate_miss'),
                'progress_m': tev.get('progress_m'), 'lap_fraction': tev.get('lap_fraction'), 'progress_full_lap_t': tev.get('progress_full_lap_t'),
                'model_sha256': r['model_sha256'], 'ort_ms_median': r.get('ort_ms_median')}
    ev = {
        'schema': 'sim-dart-evaluation/v1',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'protocol_sha256': sha(REPORT / 'protocol.json'),
        'protocol_sha256_recorded': (REPORT / 'protocol.sha256').read_text().split()[0],
        'where': 'ブラウザ(snap の Firefox、DISPLAY :0)のアプリ(この作業のシミュレーターの写し dist/index.html、元とバイト単位で同じ)の連続試験。'
                 'ブリッジはユーザーの bridge.py(写し、元と同じ)を専用の venv(onnxruntime 1.31.0、CPU)で 127.0.0.1:8812',
        'settings': {'control_delay_frames': 1, 'lookahead': 1, 'steer_gain': 1.0, 'throttle_gain': '1.0(作者の ONNX は 1.01)',
                     'mode': '8(作者の ONNX は 4)', 'single_lap_limit_s': 15, 'physics_and_sensors': 'ベストラップの評価と同じ(REC60 の計画の物理・センサーの欄)',
                     'seed': 1, 'scenario': 'baseline'},
        'starts': {e: list(v) for e, v in STARTS.items()},
        'bundles': {lab: {'zip': MODEL_BY_LABEL[lab]['zip'], 'zip_sha256': sha(MODEL_BY_LABEL[lab]['zip']), 'onnx_sha256': onnx_sha.get(lab)}
                    for lab in MODEL_BY_LABEL},
        'T_single_lap_from_stop': {f'{lab}/{e}': T.get((lab, e)) for lab in labels for e in STARTS if (lab, e) in single},
        'single_runs': {f'{lab}/{e}': run_brief(r) for (lab, e), r in sorted(single.items())},
        'summary': summary,
        'teachers_node_delay0': {e: {kind: {k: teach[e][kind][k] for k in ('reason', 'lap_s', 'passed_gates', 'collisions', 'min_clearance_m')}
                                     for kind in ('fast', 'margin')} for e in teach},
        'verdicts': verdicts,
        'positive_control_today': pos,
        'determinism': det,
        'flying_reference': flying,
        'gate_point4': gate_point4,
        'dart_coverage_reference': cov,
        'user_verifier': {r['name']: r['user_verifier'] for r in rows if 'user_verifier' in r},
        'verification_all_runs': {'runs_today': sum(1 for r in rows if r['source'] == 'today'),
                                  'consistency_pass_today': sum(1 for r in rows if r['source'] == 'today' and r['verification'].get('consistency_pass')),
                                  'replay_same_teacher_eval': sum(1 for q in te['rows'] if q['replay_same']), 'teacher_eval_rows': len(te['rows']),
                                  'setting_mismatches': [r['name'] for r in rows if r['setting_check']], 'unreadable': problems},
        'speed': speed,
        'figures': figs,
        'baseline_sources': {'bestlap_runs_verified': str(BL_CL / 'runs_verified.json'), 'bestlap_evaluation': str(BESTLAP_REPORT / 'evaluation.json'),
                             'note': 'A0-s0 の E0〜E5・A0-s1・S-T・作者の ONNX の E0〜E5 は、ベストラップの評価の記録(同じアプリ・同じ計画・同じ物理)を読んだ。走らせ直していない'},
        'files': {'runs_csv': str(EVAL / 'runs.csv'), 'runs_verified': str(CL / 'runs_verified.json'), 'teacher_eval': str(EVAL / 'teacher_eval.json'),
                  'closed_loop': str(CL), 'screens': str(REPORT / 'screens')},
    }
    (REPORT / 'evaluation.json').write_text(json.dumps(ev, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'verdicts': verdicts, 'c': c, 'T': ev['T_single_lap_from_stop'], 'positive_control': pos, 'determinism': det,
                      'flying': {k: (v['reason'], v['lap1_s'], v['flying_lap_s']) for k, v in flying.items()},
                      'consistency': ev['verification_all_runs']}, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
