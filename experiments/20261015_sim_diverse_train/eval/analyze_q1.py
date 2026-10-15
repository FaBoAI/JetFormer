#!/usr/bin/env python3
"""Q1 の閉ループ評価の集計・判定・図(sim_diverse_train_20261015 の Q1 の評価の段、2026-10-15)

    node eval/verify_eval.mjs                     # 1 走行ずつの確かめ(closed_loop/runs_verified.json)
    python -I eval/analyze_q1.py --prepare        # teacher_eval.mjs に渡す走行の一覧(eval/teacher_runs.json)
    node eval/teacher_eval.mjs --runs ... --out eval/teacher_eval.json
    python -I eval/analyze_q1.py                  # この集計 → eval_sim.json(と render_q1_md.py で eval_sim.md)

10/14 の analyze_eval.py を元に作り直したもの(判定は今日の protocol.json verdicts_Q1 をそのままコードにした):
  1) E0 の単周を、ユーザーの qa/verify-learned-lap.mjs <結果> --model <ONNX> で確かめる(元のフォルダーでそのまま実行。
     書くのは今日の closed_loop/user_verifier/ だけ)
  2) 完走 = route_complete・衝突 0・33 ゲート・黒シート 1 回で、再生の数え直しが一致し(verify_eval)、設定が計画どおり。E0 は確認器も PASS
  3) 腕ごとの量: c(完走の数)・fs(ゲート 0 の開始 9 本で最初の直線を抜けた数)・g4(ゲート 0 の開始 9 本で点 4 のゲートを数えた数)・
     d(12 本の進んだ道のりの中央値。完走は 1 周)
  4) A0・A1・A2・S-T・作者の ONNX は 10/14 の evaluation.json と teacher_eval.json(同じアプリ・同じ計画・同じ物理)を読む。走らせ直さない
     (前日との同一性の確かめで違いが出て A2 を走らせ直したときは、今日の A2 を c_ref に入れる)
  5) 図: fig/eval_completion.png・fig/eval_distance_time.png・fig/eval_trajectories.png・fig/eval_first_corner.png・fig/eval_deviation.png
  6) eval_sim.json(レポートのフォルダーの直下)
前の作業の記録は読むだけ。実車・PCA9685・I2C には触れない。
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
from eval_common import (CL, CORE, EVAL, EXT_STARTS, GATE0_STARTS, MODEL_BY_LABEL, PREV_REPORT, REPORT, STARTS, USER_SIM, run_meta)  # noqa: E402

FIG = REPORT / 'fig'
PREV_EVAL = PREV_REPORT / 'evaluation.json'
PREV_TE = PREV_REPORT / 'eval' / 'teacher_eval.json'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
# 腕の色(腕ごとに固定。dataviz の検証済みの 6 色。A0 青・A1 橙・A2 水色・S-DIV 紫・S-MOD 桃・A4 黄)
COLORS = {'A0': '#2a78d6', 'A1': '#eb6834', 'A2': '#1baf7a', 'A2@1015': '#1baf7a', 'S-DIV': '#4a3aa7', 'S-MOD': '#e87ba4', 'A4': '#eda100',
          'S-T': '#8d8c86', 'EXT': '#52514e', 'T_fast': '#0b0b0b', 'T_margin': '#8d8c86'}
NAMES = {'A0': 'A0(きれいな RL 20 分)', 'A1': 'A1(A0 + DART 30 分)', 'A2': 'A2(余裕の線 10 分)', 'A2@1015': 'A2(今日走らせ直し)',
         'S-DIV': 'S-DIV(5 モード + DART + 余裕の線 45 分)', 'S-MOD': 'S-MOD(5 モードだけ 25 分)', 'A4': 'A4(S-DIV + DAgger)',
         'S-T': 'S-T 種 0(作者の先生のデータ)', 'S-T1': 'S-T 種 1', 'A0s1': 'A0 種 1', 'EXT': '作者の ONNX',
         'T_fast': 'T_fast(RL 最速の先生・Node・遅延 0)', 'T_margin': 'T_margin(余裕の線の先生・Node・遅延 0)'}
TICK = {'A0': 'A0\nきれいな RL', 'A1': 'A1\n+ DART', 'A2': 'A2\n余裕の線', 'A2@1015': 'A2\n今日', 'S-DIV': 'S-DIV\n今日の主', 'S-MOD': 'S-MOD\n5 モードだけ',
        'A4': 'A4\nDAgger', 'S-T': 'S-T 種 0\n(E0〜E5)', 'EXT': '作者 ONNX\n(E0〜E5)'}
# 腕 → (10/14 の single_runs のラベル、今日のラベル)
PREV_ARMS = {'A0': 'A0-s0', 'A1': 'A1-s0', 'A2': 'A2-s0', 'A0s1': 'A0-s1', 'S-T': 'ST-s0', 'S-T1': 'ST-s1', 'EXT': 'EXT'}
TODAY_ARMS = {'S-DIV': 'SDIV-s0', 'S-MOD': 'SMOD-s0', 'A4': 'A4-s0', 'A2@1015': 'A2-s0'}
REASON_JA = {'collision': '衝突', 'time_limit': '時間切れ', 'route_complete': '完走', 'gate_miss': 'ゲート外し', 'aborted': '中断'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def sign_test_p(wins, losses):
    """両側の符号検定(同点は除く。参考)"""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def fisher_two_sided(a, n1, b, n2):
    k, n = a + b, n1 + n2

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


def today_rows():
    p = CL / 'runs_verified.json'
    if not p.exists():
        return [], []
    rows, problems = [], []
    for r in json.loads(p.read_text())['rows']:
        if 'error' in r:
            problems.append(r)
            continue
        lab, start, kind, variant = run_meta(r['name'])
        rows.append(r | {'label': lab, 'start': start, 'kind': kind, 'variant': variant, 'source': 'today'})
    return rows, problems


def prepare():
    rows, _ = today_rows()
    out = []
    for r in rows:
        if r['kind'] != 'single':
            continue
        m = MODEL_BY_LABEL.get(r['label'], {})
        out.append({'name': r['name'], 'label': r['label'], 'file': r['file'], 'teacher': m.get('teacher', 'fast'),
                    'takeover': r['label'] not in ('EXT',) and r['variant'] == 'protocol'})
    p = EVAL / 'teacher_runs.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(p, len(out))


def setting_check(r, plan_c, onnx_sha):
    chk = []
    if onnx_sha and r['model_sha256'] != onnx_sha:
        chk.append('モデルの sha256 が一式と違う')
    if not plan_c:
        chk.append('計画が見つからない')
        return chk
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
    if r['config'].get('timeLimit') != plan_c['limit']:
        chk.append('時間の上限が計画と違う')
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


def course(ax, world, plt, gate_numbers=False, g0=0):
    for s in world['surfaces']:
        if s['id'] == 'carpet':
            continue
        col = {'shortcut': '#3b3b3b', 'turf': '#7fbf7f', 'ptfe': '#e8e2c8', 'blue-mat': '#b9d3f0'}.get(s['id'], '#cccccc')
        ax.add_patch(plt.Rectangle((s['x'] - s['w'] / 2, s['z'] - s['d'] / 2), s['w'], s['d'], color=col, alpha=.3, lw=0))
    for seg in world['segments']:
        ax.plot([seg['a'][0], seg['b'][0]], [seg['a'][1], seg['b'][1]], color='#9b4a3c', lw=1.1, alpha=.75)
    for dsk in world['disks']:
        ax.add_patch(plt.Circle((dsk['x'], dsk['z']), max(dsk.get('radius', .02), .02), color='#6f6e69', alpha=.6, lw=0))
    for i, (gx, gz) in enumerate(world['route']):
        ax.plot(gx, gz, marker='o', ms=2.5, color='#c9a227', alpha=.9)
        if gate_numbers and i in gate_numbers:
            ax.text(gx + .05, gz - .05, f'点 {i}', fontsize=8, color='#8a6d12', clip_on=True)
    b = world['bounds']
    ax.set_xlim(b['minX'], b['maxX'])
    ax.set_ylim(b['maxZ'], b['minZ'])
    ax.set_aspect('equal')


def gate_line(ax, route, k, color='#c9a227'):
    """点 k のゲートの線(点 k−1 → 点 k の向きに垂直、半幅 0.65 m)"""
    p0, p1 = route[k - 1], route[k]
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    n = math.hypot(dx, dz)
    lx, lz = -dz / n, dx / n
    ax.plot([p1[0] - .65 * lx, p1[0] + .65 * lx], [p1[1] - .65 * lz, p1[1] + .65 * lz], color=color, lw=2.2, alpha=.9, solid_capstyle='round')


def figures(arms_order, runs, summary, teach, world0, L):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    jp_font()
    FIG.mkdir(exist_ok=True)
    figs = []
    main_arms = [a for a in arms_order if a in ('A0', 'A1', 'A2', 'A2@1015', 'S-DIV', 'S-MOD', 'A4')]
    ref_arms = ['S-T', 'EXT']

    # 1) 完走の数(左)と進んだ道のり(右)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 5.2), facecolor=SURFACE, gridspec_kw={'width_ratios': [1, 1.2]})
    style(ax, '完走の数(単周・遅延 1・ブラウザのアプリ)。濃い = E0〜E5、薄い = E6〜E11、枠 = 走らせた本数', '完走した本数')
    allx = main_arms + ref_arms
    for i, arm in enumerate(allx):
        s = summary[arm]
        c_core = sum(1 for e in s['completed'] if e in CORE)
        c_ext = s['n_completed'] - c_core
        col = COLORS[arm]
        ax.bar(i, s['n_ran'], .62, color='none', edgecolor='#bdbcb6', lw=1.2)
        ax.bar(i, c_core, .62, color=col)
        ax.bar(i, c_ext, .62, bottom=c_core, color=col, alpha=.45)
        ax.text(i, max(c_core + c_ext, 0) + .25, f"{s['n_completed']} / {s['n_ran']}", ha='center', va='bottom', fontsize=10, color=INK)
        if arm in ('A0', 'A1', 'A2'):
            ax.text(i, -1.15, '10/14', ha='center', fontsize=7, color=INK2)
    ax.axvline(len(main_arms) - .5, color=GRID, lw=1)
    ax.set_xticks(range(len(allx)))
    ax.set_xticklabels([TICK[a] for a in allx], fontsize=8)
    ax.set_ylim(-1.5, 13.5)
    ax.text(len(allx) - .5, 12.8, 'T_fast・T_margin(Node・遅延 0)は 12 / 12', ha='right', fontsize=8, color=INK2)
    style(ax2, '止まるまでに進んだ道のり(参照の周に沿った長さ。● = 完走、× = 途中で止まった、横線 = 中央値 d)', '進んだ道のり [m]')
    for i, arm in enumerate(main_arms + ref_arms):
        vals = []
        for j, e in enumerate(STARTS):
            r = runs.get((arm, e))
            if r is None or r.get('d_value') is None:
                continue
            vals.append(r['d_value'])
            ok = r['counted_complete']
            ax2.plot(i + (j % 6 - 2.5) * .06, r['d_value'], marker='o' if ok else 'x', color=COLORS[arm], ms=6, mew=1.6, alpha=.9, ls='none')
        if vals:
            m = statistics.median(vals)
            ax2.hlines(m, i - .34, i + .34, color=COLORS[arm], lw=2.4)
            ax2.text(i + .36, m, f'{m:.1f}', va='center', fontsize=8, color=INK)
    if L:
        ax2.axhline(L, color=INK2, lw=1, ls='--')
        ax2.text(len(main_arms + ref_arms) - .5, L + .5, f'1 周 = {L:.1f} m', ha='right', fontsize=8, color=INK2)
    ax2.axvline(len(main_arms) - .5, color=GRID, lw=1)
    ax2.set_xticks(range(len(main_arms + ref_arms)))
    ax2.set_xticklabels([TICK[a] for a in main_arms + ref_arms], fontsize=8)
    allv = [r['d_value'] for (arm, e), r in runs.items() if arm in main_arms + ref_arms and r.get('d_value') is not None]
    ax2.set_ylim(0, max([30, (L or 0) + 3] + [v + 3 for v in allv]))
    fig.tight_layout()
    fig.savefig(FIG / 'eval_completion.png', dpi=130, facecolor=SURFACE)
    plt.close(fig)
    figs.append('fig/eval_completion.png')

    # 2) 開始ごとの、止まるまでの時間と道のり
    comp_arms = [a for a in main_arms if a != 'A0']
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 8.4), facecolor=SURFACE, sharex=True)
    style(ax, '開始ごとの止まるまでの時間(完走は停止から 1 周の時間)。棒の上 = 数えたゲート / 33。横線 = 先生の周回時間(Node・遅延 0)', '時間 [秒]')
    style(ax2, '開始ごとの止まるまでに進んだ道のり(完走は 1 周)。破線 = 1 周', '道のり [m]')
    w = .8 / max(1, len(comp_arms))
    for j, e in enumerate(STARTS):
        for k, arm in enumerate(comp_arms):
            x = j + (k - (len(comp_arms) - 1) / 2) * w
            r = runs.get((arm, e))
            if r is None:
                continue
            v = r['lap_s'] if r['counted_complete'] else r['sim_time']
            ax.bar(x, v, w * .88, color=COLORS[arm], label=NAMES[arm] if j == 0 else None)
            ax.text(x, v + .1, str(r['passed_gates']), ha='center', va='bottom', fontsize=6.5, color=INK2)
            ax2.bar(x, r['d_value'] or 0, w * .88, color=COLORS[arm], label=None)
        if e in teach:
            for kind, tk, ls in (('fast', 'T_fast', '-'), ('margin', 'T_margin', '--')):
                v = teach[e][kind]['lap_s']
                if v:
                    ax.hlines(v, j - .45, j + .45, color=COLORS[tk], lw=1.2, ls=ls, label=NAMES[tk] if j == 0 else None)
    if L:
        ax2.axhline(L, color=INK2, lw=1, ls='--')
    ax2.set_xticks(range(len(STARTS)))
    ax2.set_xticklabels([f'{e}\n(G{g}・{o:+.2f} m・{y:+.0f}°)' for e, (g, o, y) in STARTS.items()], fontsize=7.5)
    ax.set_ylim(0, 19.5)
    ax.legend(fontsize=7.5, ncol=3, loc='upper right', frameon=True, facecolor=SURFACE, edgecolor=GRID)
    fig.tight_layout()
    fig.savefig(FIG / 'eval_distance_time.png', dpi=125, facecolor=SURFACE)
    plt.close(fig)
    figs.append('fig/eval_distance_time.png')

    # 3) 軌跡と止まった場所(腕ごと。上 = ゲート 0 の 9 本、下 = E5・E10・E11)
    traj_arms = [a for a in main_arms if a != 'A0'] or main_arms
    fig, axes = plt.subplots(2, len(traj_arms), figsize=(5.6 * len(traj_arms), 9.6), facecolor=SURFACE, squeeze=False)
    for col, arm in enumerate(traj_arms):
        s = summary[arm]
        for row, starts in enumerate((GATE0_STARTS, ['E5', 'E10', 'E11'])):
            ax = axes[row][col]
            nc = sum(1 for e in starts if (runs.get((arm, e)) or {}).get('counted_complete'))
            nr = sum(1 for e in starts if (arm, e) in runs)
            style(ax, f"{NAMES[arm]}\n{'ゲート 0 の開始' if row == 0 else 'E5・E10・E11'}: 完走 {nc} / {nr}(× 衝突・□ 時間切れ)", 'z [m]', 'x [m]')
            course(ax, world0, plt)
            if row == 0 and 'E0' in teach:
                for kind, tk, ls in (('fast', 'T_fast', '--'), ('margin', 'T_margin', ':')):
                    xy = teach['E0'][kind]['xy']
                    ax.plot([p[0] for p in xy], [p[1] for p in xy], color=COLORS[tk], lw=1.1, ls=ls, label=NAMES[tk] + ' の E0')
            for e in starts:
                r = runs.get((arm, e))
                if not r or not r.get('file'):
                    continue
                tr = json.loads(Path(r['file']).read_text())['trace']
                ok = r['counted_complete']
                ax.plot([t['x'] for t in tr], [t['z'] for t in tr], color=COLORS[arm], lw=1.0 if ok else 1.4, alpha=.55 if ok else .95)
                if r['collisions']:
                    ax.plot(tr[-1]['x'], tr[-1]['z'], marker='x', color=INK, ms=9, mew=2.4)
                elif not ok:
                    ax.plot(tr[-1]['x'], tr[-1]['z'], marker='s', mfc='none', color=INK, ms=8, mew=1.8)
                if row == 1:
                    ax.plot(tr[0]['x'], tr[0]['z'], marker='o', color=COLORS[arm], ms=6)
                    ax.text(tr[0]['x'], tr[0]['z'], f' {e}', fontsize=7.5, color=INK)
            if row == 0:
                ax.legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    fig.savefig(FIG / 'eval_trajectories.png', dpi=105, facecolor=SURFACE)
    plt.close(fig)
    figs.append('fig/eval_trajectories.png')

    # 4) 最初の曲がり(点 3〜6)の拡大。10/14 はどの腕も点 4 のゲートを外した
    route = world0['route']
    fig, axes = plt.subplots(1, len(traj_arms), figsize=(5.4 * len(traj_arms), 6.4), facecolor=SURFACE, squeeze=False)
    for col, arm in enumerate(traj_arms):
        ax = axes[0][col]
        s = summary[arm]
        style(ax, f"{NAMES[arm]}\n点 4 のゲートを数えた: {s['g4']} / {s['gate0_ran']}(ゲート 0 の開始)", 'z [m]', 'x [m]')
        course(ax, world0, plt, gate_numbers=range(2, 10))
        for k in (3, 4, 5, 6, 7, 8):
            gate_line(ax, route, k, color='#d4537e' if k == 4 else '#c9a227')
        for kind, tk, ls in (('fast', 'T_fast', '--'), ('margin', 'T_margin', ':')):
            if 'E0' in teach:
                xy = teach['E0'][kind]['xy']
                ax.plot([p[0] for p in xy], [p[1] for p in xy], color=COLORS[tk], lw=1.2, ls=ls, label=NAMES[tk] + ' の E0')
        for e in GATE0_STARTS:
            r = runs.get((arm, e))
            if not r or not r.get('file'):
                continue
            tr = json.loads(Path(r['file']).read_text())['trace']
            ok = r['passed_gates'] >= 4
            ax.plot([t['x'] for t in tr], [t['z'] for t in tr], color=COLORS[arm], lw=1.3, alpha=.9 if ok else .6)
            if r['collisions']:
                ax.plot(tr[-1]['x'], tr[-1]['z'], marker='x', color=INK, ms=9, mew=2.4)
        ax.set_xlim(1.2, 5.0)
        ax.set_ylim(3.0, -0.4)
        ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(FIG / 'eval_first_corner.png', dpi=115, facecolor=SURFACE)
    plt.close(fig)
    figs.append('fig/eval_first_corner.png')

    # 5) 先生の線からの距離の時間変化(T_fast の同じ開始の線。前後 1.5 秒の窓)
    fig, axes = plt.subplots(1, len(traj_arms), figsize=(5.2 * len(traj_arms), 4.6), facecolor=SURFACE, sharey=True, squeeze=False)
    for col, arm in enumerate(traj_arms):
        ax = axes[0][col]
        style(ax, f"{NAMES[arm]}\nT_fast の同じ開始の線からの距離(× = 衝突、● = 完走)", '線からの距離 [cm]', '時間 [秒]')
        for e in STARTS:
            r = runs.get((arm, e))
            if not r or not r.get('series'):
                continue
            t = r['series']['t']
            d = [float('nan') if x is None else x * 100 for x in r['series']['path_fast']]
            ok = r['counted_complete']
            ax.plot(t, d, color=COLORS[arm], lw=1.0, alpha=.5 if ok else .9)
            if r['collisions'] and d:
                ax.plot(t[-1], d[-1], marker='x', color=INK, ms=8, mew=2)
            elif ok and d:
                ax.plot(t[-1], d[-1], marker='o', color=COLORS[arm], ms=5)
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 80)
    fig.tight_layout()
    fig.savefig(FIG / 'eval_deviation.png', dpi=120, facecolor=SURFACE)
    plt.close(fig)
    figs.append('fig/eval_deviation.png')
    return figs


# ---------------------------------------------------------------- 本体
def real_gate_miss(crossings, passed_gates, completed):
    """本当に外したゲート: 線を |横| > 0.65 m で越え(margin < 0)、しかもそのとき sim がそのゲートを数えなかったもの(counted False)。
    margin < 0 でも sim が数えたもの(すれすれ。測る位置の違い)は外しにしない。外したあと 1 周回って数え直した走り(S-DIV の E4)も外しに入る"""
    for c in crossings or []:
        if c.get('margin_m') is not None and c['margin_m'] < 0 and not c.get('counted'):
            return c
    return None


def tight_gates(crossings):
    """線の延長を |横| > 0.65 m で越えたが sim が数えたゲート(すれすれ。参考)"""
    return [c for c in crossings or [] if c.get('margin_m') is not None and c['margin_m'] < 0]


def unify_prev(prev_eval, prev_te_by_file, prev_te_by_name):
    """10/14 の single_runs(読むだけ)を今日の行と同じ形にする"""
    out = {}
    files = {}
    for p in (PREV_REPORT / 'eval' / 'closed_loop').glob('*/*.json'):
        if p.parent.name != 'user_verifier' and not p.name.startswith('plan_'):
            files[p.stem] = str(p)
    for p in Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/eval/closed_loop').glob('*/*-single.json'):
        files.setdefault(p.stem, str(p))
    for arm, lab in PREV_ARMS.items():
        for e in STARTS:
            b = prev_eval['single_runs'].get(f'{lab}/{e}')
            if not b:
                continue
            f = files.get(b['name'])
            if b['source'] != 'today' and b['name'] == 'EXT-E0-single':
                f = '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/eval/closed_loop/EXT/EXT-E0-single.json'
            q = prev_te_by_file.get(f) or prev_te_by_name.get(b['name']) or {}
            out[(arm, e)] = {
                'arm': arm, 'start': e, 'name': b['name'], 'source': '20261014' if b['source'] == 'today' else b['source'], 'file': f,
                'reason': b['reason'], 'counted_complete': b['counted_complete'], 'lap_s': b['single_lap_s'], 'passed_gates': b['passed_gates'],
                'collisions': b['collisions'], 'sim_time': b['sim_time'], 'frames': b['frames'], 'min_clearance_m': b['min_clearance_m'],
                'collision': b['collision'], 'end_pose': b['end_pose'], 'first_straight_passed': b['first_straight_passed'],
                'progress_m': b['progress_m'], 'gate_crossings': q.get('gate_crossings'),
                'first_gate_miss': real_gate_miss(q.get('gate_crossings'), b['passed_gates'], b['counted_complete']),
                'first_gate_line_over_0p65': (tight_gates(q.get('gate_crossings')) or [None])[0], 'path_dev_fast_m': b.get('path_dev_fast_m'),
                'path_dev_teacher_m': b.get('path_dev_teacher_m'), 'teacher': 'margin' if arm == 'A2' else 'fast', 'takeover': b.get('takeover'),
                'series': (q.get('series') or None), 'verification_pass': b.get('verification_pass'), 'setting_check': b.get('setting_check'),
                'model_sha256': b.get('model_sha256'), 'user_verifier': b.get('user_verifier')}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--no-user-verifier', action='store_true')
    a = ap.parse_args()
    if a.prepare:
        prepare()
        return
    protocol = json.loads((REPORT / 'protocol.json').read_text())
    V = protocol['verdicts_Q1']
    rows, problems = today_rows()
    runs_json = json.loads((REPORT / 'train' / 'runs.json').read_text())['runs']
    plans = {}
    for p in CL.glob('*/plan_*.json'):
        for c in json.loads(p.read_text()):
            plans[c['name']] = c
    onnx_sha = {}
    for lab, m in MODEL_BY_LABEL.items():
        if m.get('available'):
            onnx_sha[lab] = m.get('onnx_sha256_expected') or (sha(m['onnx']) if lab == 'EXT' else None)
    uv_cache = {}
    for r in rows:
        r['setting_check'] = setting_check(r, plans.get(r['name']), onnx_sha.get(r['label']))
        if r['kind'] == 'single' and r['start'] == 'E0' and not a.no_user_verifier:
            r['user_verifier'] = user_verifier(r['name'], r['file'], MODEL_BY_LABEL[r['label']]['onnx'])
            uv_cache[r['name']] = r['user_verifier']
    te_path = EVAL / 'teacher_eval.json'
    te = json.loads(te_path.read_text()) if te_path.exists() else {'rows': [], 'teacher_refs': {}}
    te_rows = {q['file']: q for q in te['rows']}
    teach = {}
    for e, (g, o, y) in STARTS.items():
        key = f'{g}|{o:.4f}|{y:.4f}'
        if key in te['teacher_refs']:
            teach[e] = te['teacher_refs'][key]
    L = next((q['loop_length_m'] for q in te['rows'] if q.get('loop_length_m')), 26.1)

    def is_complete(r):
        ok = r['completed'] and r['verification']['pass'] and not r['setting_check'] and r['single_lap_s'] is not None
        if ok and r['start'] == 'E0' and 'user_verifier' in r:
            ok = ok and r['user_verifier'].get('pass', False)
        return ok

    # 今日の走行(単周・protocol の名前)を腕 × 開始の表に
    runs = {}
    today_by_label = {v: k for k, v in TODAY_ARMS.items()}
    extra = {}   # 陽性対照・前日との同一性・決定性・連続周回
    for r in rows:
        q = te_rows.get(r['file']) or {}
        cc = is_complete(r) if r['kind'] == 'single' else None
        u = {'arm': today_by_label.get(r['label'], r['label']), 'start': r['start'], 'name': r['name'], 'source': 'today', 'file': r['file'],
             'reason': r['reason'], 'counted_complete': cc, 'lap_s': r['single_lap_s'] if cc else None, 'passed_gates': r['passed_gates'],
             'collisions': r['collisions'], 'sim_time': r['sim_time'], 'frames': r['frames'], 'min_clearance_m': r['min_clearance_m'],
             'collision': ({'time': (r['collision_event'] or {}).get('time'), 'x': (r['collision_event'] or {}).get('x'),
                            'z': (r['collision_event'] or {}).get('z'), 'obstacle': (r['collision_event'] or {}).get('obstacle'),
                            'kind': ((r['collision_event'] or {}).get('raw') or {}).get('kind'),
                            'frame': ((r['collision_event'] or {}).get('raw') or {}).get('frame')} if r['collision_event'] else None),
             'end_pose': r['end_pose'], 'first_straight_passed': q.get('first_straight_passed'), 'progress_m': q.get('progress_m'),
             'first_gate_miss': real_gate_miss(q.get('gate_crossings'), r['passed_gates'], bool(cc)),
             'first_gate_line_over_0p65': (tight_gates(q.get('gate_crossings')) or [None])[0],
             'path_dev_fast_m': q.get('path_dev_fast_m'), 'path_dev_teacher_m': q.get('path_dev_teacher_m'),
             'teacher': q.get('teacher'), 'takeover': q.get('takeover'), 'series': q.get('series'), 'replay_same': q.get('replay_same'),
             'verification_pass': r['verification']['pass'], 'consistency_pass': r['verification'].get('consistency_pass'),
             'setting_check': r['setting_check'], 'model_sha256': r['model_sha256'], 'user_verifier': (r.get('user_verifier') or {}).get('pass'),
             'ort_ms_median': r.get('ort_ms_median'), 'latency_ms_median': r.get('latency_ms_median'), 'max_speed': r.get('max_speed'),
             'mean_speed': r.get('mean_speed'), 'gate_crossings': q.get('gate_crossings'), 'laps': r.get('laps'), 'flying_lap_s': r.get('flying_lap_s'),
             'lap1_s': r.get('lap1_s'), 'kind': r['kind'], 'variant': r['variant'], 'label': r['label']}
        if r['kind'] == 'single' and r['variant'] == 'protocol' and r['label'] in today_by_label:
            runs[(u['arm'], r['start'])] = u
        else:
            extra[r['name']] = u
    # 10/14 の腕(読むだけ)
    prev_eval = json.loads(PREV_EVAL.read_text())
    prev_te = json.loads(PREV_TE.read_text())
    prev_rows = unify_prev(prev_eval, {q['file']: q for q in prev_te['rows']}, {q['name']: q for q in prev_te['rows']})
    runs.update(prev_rows)
    for r in runs.values():
        # d の 1 本ぶん: 完走は 1 周、途中で止まった走りは参照の周に沿って進んだ道のり
        r['d_value'] = L if r['counted_complete'] else r['progress_m']
        sv0 = STARTS[r['start']][0]
        r['g4'] = (r['passed_gates'] >= 4) if sv0 == 0 and r['start'] in GATE0_STARTS else None
    # 腕ごとのまとめ
    arms_order = ['A0', 'A1', 'A2'] + [a for a in ('A2@1015', 'S-DIV', 'S-MOD', 'A4') if any(k[0] == a for k in runs)] + ['A0s1', 'S-T', 'S-T1', 'EXT']
    summary = {}
    for arm in arms_order:
        rs = {e: runs.get((arm, e)) for e in STARTS}
        ran = [e for e in STARTS if rs[e]]
        if not ran:
            continue
        comp = [e for e in ran if rs[e]['counted_complete']]
        g0 = [e for e in ran if e in GATE0_STARTS]
        tk = 'fast'
        ratios = {e: rs[e]['lap_s'] / teach[e]['fast']['lap_s'] for e in comp if e in teach and teach[e]['fast']['lap_s']}
        ratios_m = {e: rs[e]['lap_s'] / teach[e]['margin']['lap_s'] for e in comp if e in teach and teach[e]['margin']['lap_s']}
        summary[arm] = {
            'source': rs[ran[0]]['source'], 'ran': ran, 'n_ran': len(ran), 'completed': comp, 'n_completed': len(comp),
            'n_completed_E0_E5': sum(1 for e in comp if e in CORE), 'n_ran_E0_E5': sum(1 for e in ran if e in CORE),
            'fs': sum(1 for e in g0 if rs[e]['first_straight_passed']), 'g4': sum(1 for e in g0 if rs[e]['g4']), 'gate0_ran': len(g0),
            'g4_starts': [e for e in g0 if rs[e]['g4']], 'fs_starts': [e for e in g0 if rs[e]['first_straight_passed']],
            'd_median_m': med([rs[e]['d_value'] for e in ran]), 'd_values': {e: rs[e]['d_value'] for e in ran},
            'stop_time_median_s': med([rs[e]['sim_time'] for e in ran if not rs[e]['counted_complete']]),
            'stop_times_s': {e: rs[e]['sim_time'] for e in ran if not rs[e]['counted_complete']},
            'laps_s': {e: rs[e]['lap_s'] for e in comp}, 'lap_median_s': med([rs[e]['lap_s'] for e in comp]),
            'lap_ratio_to_T_fast': ratios, 'lap_ratio_to_T_fast_median': med(list(ratios.values())),
            'lap_ratio_to_T_margin_median': med(list(ratios_m.values())),
            'passed_gates': {e: rs[e]['passed_gates'] for e in ran},
            'collision_kinds': {e: (rs[e]['collision'] or {}).get('kind') for e in ran if rs[e]['collisions']},
            'first_gate_miss_points': {e: (rs[e]['first_gate_miss'] or {}).get('point') for e in ran if rs[e].get('first_gate_miss')},
            'path_dev_fast_end_median_m': med([(rs[e]['path_dev_fast_m'] or {}).get('at_end') for e in ran]),
            'path_dev_fast_1s_median_m': med([(rs[e]['path_dev_fast_m'] or {}).get('at_1s') for e in ran]),
            'takeover_fast_seconds_before_end_median': med([((rs[e].get('takeover') or {}).get('fast') or {}).get('seconds_before_end') for e in ran]),
        }
    for kind, lab in (('fast', 'T_fast'), ('margin', 'T_margin')):
        laps = {e: teach[e][kind]['lap_s'] for e in STARTS if e in teach}
        summary[lab] = {'source': 'today(Node・遅延 0)', 'n_ran': len(laps), 'n_completed': sum(1 for v in laps.values() if v), 'laps_s': laps,
                        'lap_median_s': med(list(laps.values())),
                        'same_as_1014': all(abs((laps[e] or 0) - (prev_eval['teachers_node_delay0'].get(e, {}).get(kind, {}).get('lap_s') or 0)) < 1e-9
                                            for e in laps)}

    # ------------------------------------------------ 判定(protocol.json verdicts_Q1 をそのまま)
    sa_path = EVAL / 'same_as_yesterday.json'
    same = json.loads(sa_path.read_text()) if sa_path.exists() else None
    c = {arm: summary[arm]['n_completed'] for arm in summary if arm in ('A0', 'A1', 'A2', 'A2@1015', 'S-DIV', 'S-MOD', 'A4')}
    c_ref_arms = ['A0', 'A1', 'A2'] + (['A2@1015'] if 'A2@1015' in c else [])
    c_ref = max(c[x] for x in c_ref_arms)
    verdicts = {}
    if 'S-DIV' in summary:
        s = summary['S-DIV']
        cs, fs, g4 = s['n_completed'], s['fs'], s['g4']
        if cs >= 6 and cs - c_ref >= 4:
            v = '効いた'
        elif cs - c_ref >= 2 or g4 >= 5:
            v = '一部効いた'
        elif cs <= c_ref and fs <= 4:
            v = '悪くなった'
        else:
            v = '差なし'
        verdicts['Q1_S-DIV_vs_yesterday'] = {'verdict': v, 'c_S-DIV': cs, 'c_ref': c_ref, 'c_ref_arms': {x: c[x] for x in c_ref_arms},
                                             'fs_S-DIV': fs, 'g4_S-DIV': g4, 'gate0_ran': s['gate0_ran'], 'd_S-DIV_m': s['d_median_m'],
                                             'rule': V['Q1_S-DIV_vs_yesterday'],
                                             'fisher_p_vs_best_yesterday_reference': fisher_two_sided(cs, s['n_ran'], c_ref, 12)}
    d_A1 = summary['A1']['d_median_m']
    if 'S-MOD' in summary:
        s, a1 = summary['S-MOD'], summary['A1']
        cm, c1 = s['n_completed'], a1['n_completed']
        if cm - c1 >= 2:
            v, why = '量よりばらつき', 'c(S-MOD) − c(A1) ≥ 2'
        elif cm == c1 == 0 and s['g4'] - a1['g4'] >= 3:
            v, why = '量よりばらつき', 'c が両方 0 で g4(S-MOD) − g4(A1) ≥ 3'
        elif cm == c1 == 0 and s['g4'] == a1['g4'] == 0 and s['d_median_m'] >= 1.5 * d_A1:
            v, why = '量よりばらつき', f'どちらも点 4 を通らず d(S-MOD) ≥ 1.5 × d(A1) = {1.5 * d_A1:.2f} m'
        else:
            v, why = 'ばらつきの効果は見えない', 'どの条件にも当たらない(S-MOD は A1 より量が少ないので「量の効果のほうが大きい」とまでは言わない)'
        verdicts['Q1_variety_vs_volume'] = {'verdict': v, 'why': why, 'c_S-MOD': cm, 'c_A1': c1, 'g4_S-MOD': s['g4'], 'g4_A1': a1['g4'],
                                            'd_S-MOD_m': s['d_median_m'], 'd_A1_m': d_A1, 'threshold_1p5_d_A1_m': 1.5 * d_A1,
                                            'rule': V['Q1_variety_vs_volume(S-MOD を走らせたとき)']['量よりばらつき']}
        if 'S-DIV' in summary:
            sd = summary['S-DIV']
            dc, dg, df = sd['n_completed'] - cm, sd['g4'] - s['g4'], sd['fs'] - s['fs']
            both0 = sd['n_completed'] == 0 and cm == 0
            if dc >= 2 or (both0 and (dg >= 3 or df >= 3)):
                v = '上乗せあり'
            elif dc <= -2 or (both0 and (dg <= -3 or df <= -3)):
                v = '悪くなった'
            else:
                v = '差なし'
            verdicts['Q1_DART_margin_addon(S-DIV vs S-MOD)'] = {'verdict': v, 'c_diff': dc, 'g4_diff': dg, 'fs_diff': df, 'both_zero': both0,
                                                               'rule': V['Q1_variety_vs_volume(S-MOD を走らせたとき)']['DART・余裕の線の上乗せ(S-DIV 対 S-MOD)']}
    else:
        verdicts['Q1_variety_vs_volume'] = {'verdict': '今日は答えない(S-MOD を走らせていない)'}
    # 一番良い腕(完走 → g4 → d)
    cand = [x for x in ('S-DIV', 'S-MOD', 'A1', 'A2', 'A2@1015', 'A0') if x in summary]
    key = lambda x: (summary[x]['n_completed'], summary[x]['g4'], summary[x]['d_median_m'] or 0)  # noqa: E731
    ranked = sorted(cand, key=key, reverse=True)
    tie = len(ranked) > 1 and key(ranked[0]) == key(ranked[1])
    verdicts['best_arm'] = {'arm': None if tie else ranked[0], 'ranking': [[x, *key(x)] for x in ranked], 'rule': V['best_arm'],
                            'best_today_arm': next((x for x in ranked if x in ('S-DIV', 'S-MOD', 'A4')), None)}
    if 'A4' in summary:
        base = verdicts['best_arm']['best_today_arm'] or 'S-DIV'
        s4, sb = summary['A4'], summary[base]
        dc, dg, df = s4['n_completed'] - sb['n_completed'], s4['g4'] - sb['g4'], s4['fs'] - sb['fs']
        both0 = s4['n_completed'] == 0 and sb['n_completed'] == 0
        v = '上乗せあり' if (dc >= 2 or (both0 and dg >= 3)) else ('悪くなった' if (dc <= -2 or (both0 and (dg <= -3 or df <= -3))) else '差なし')
        verdicts['Q1b_DAgger'] = {'verdict': v, 'base_arm': base, 'c_diff': dc, 'g4_diff': dg, 'fs_diff': df, 'rule': V['Q1b_DAgger']}
    # 速さ(完走した走り)
    verdicts['speed'] = {arm: {'lap_median_s': summary[arm]['lap_median_s'], 'ratio_to_T_fast_median': summary[arm]['lap_ratio_to_T_fast_median'],
                               'ratio_to_T_margin_median': summary[arm]['lap_ratio_to_T_margin_median'],
                               'close_to_fastest_teacher': (summary[arm]['lap_ratio_to_T_fast_median'] or 9) <= 1.05,
                               'EXT_E0_s': 9.02198843516568}
                         for arm in ('S-DIV', 'S-MOD', 'A4') if arm in summary}
    verdicts['speed']['rule'] = V['speed']
    # 開始ごとに S-DIV と A2 の進んだ道のり(参考)
    paired = {}
    for arm in ('S-DIV', 'S-MOD'):
        if arm not in summary:
            continue
        wins = losses = ties = 0
        per = {}
        for e in STARTS:
            ra, rb = runs.get((arm, e)), runs.get(('A2', e))
            if not ra or not rb:
                continue
            da, db = ra['d_value'], rb['d_value']
            per[e] = [da, db]
            if abs(da - db) < 1e-6:
                ties += 1
            elif da > db:
                wins += 1
            else:
                losses += 1
        paired[f'{arm}_vs_A2'] = {'longer': wins, 'shorter': losses, 'ties': ties, 'sign_test_p_two_sided_reference': sign_test_p(wins, losses),
                                  'per_start_m': per}
    verdicts['paired_reference'] = paired
    # 作者の ONNX・S-T と E0〜E5 で並べる(判定には使わない)
    verdicts['external_reference_E0_E5'] = {arm: {'completed': summary[arm]['n_completed_E0_E5'], 'ran': summary[arm]['n_ran_E0_E5']}
                                            for arm in ('S-DIV', 'S-MOD', 'A4', 'A1', 'A2', 'A0', 'S-T', 'S-T1', 'EXT') if arm in summary}
    # 決定性(S-DIV の E0 を 2 回)
    det = None
    ra, rb = runs.get(('S-DIV', 'E0')), extra.get('SDIV-s0-E0-single-rep2')
    if ra and rb:
        ta = json.loads(Path(ra['file']).read_text())
        tb = json.loads(Path(rb['file']).read_text())
        n = min(len(ta['trace']), len(tb['trace']))
        diff = max((max(abs(ta['trace'][i]['x'] - tb['trace'][i]['x']), abs(ta['trace'][i]['z'] - tb['trace'][i]['z']),
                        abs(ta['trace'][i]['speed'] - tb['trace'][i]['speed'])) for i in range(n)), default=None)
        det = {'run1': {k: ra[k] for k in ('reason', 'frames', 'lap_s', 'passed_gates', 'sim_time')},
               'run2': {k: rb[k] for k in ('reason', 'frames', 'lap_s', 'passed_gates', 'sim_time')},
               'max_abs_diff_xz_speed': diff, 'identical': ra['frames'] == rb['frames'] and ra['reason'] == rb['reason'] and diff == 0}
    # 陽性対照・前日との同一性
    pos = extra.get('EXT-E0-single')
    a1 = extra.get('A1-s0-E0-single')
    # 連続周回(参考)
    flying = {k: {kk: v.get(kk) for kk in ('reason', 'laps', 'lap1_s', 'flying_lap_s', 'passed_gates', 'collisions', 'sim_time', 'collision')}
              for k, v in extra.items() if v['kind'] == 'flying'}
    world0 = None
    for key0 in (('S-DIV', 'E0'), ('A1', 'E0')):
        if key0 in runs and runs[key0].get('file'):
            world0 = json.loads(Path(runs[key0]['file']).read_text())['world']
            break
    figs = figures(arms_order, runs, summary, teach, world0, L) if world0 else []
    # 走行の表(CSV)
    cols = ['arm', 'start', 'name', 'source', 'reason', 'counted_complete', 'lap_s', 'passed_gates', 'collisions', 'sim_time', 'frames',
            'min_clearance_m', 'first_straight_passed', 'g4', 'progress_m', 'd_value', 'verification_pass', 'user_verifier', 'model_sha256']
    with open(EVAL / 'runs.csv', 'w', newline='') as f:
        wri = csv.writer(f)
        wri.writerow(cols + ['collision_kind', 'collision_x', 'collision_z', 'first_gate_miss_point', 'first_gate_miss_lateral_m',
                             'path_dev_fast_0p5s_m', 'path_dev_fast_1s_m', 'path_dev_fast_end_m', 'path_dev_fast_max_m',
                             'path_dev_teacher_end_m', 'takeover_fast_last_k', 'takeover_margin_last_k', 'setting_check'])
        for (arm, e), r in sorted(runs.items(), key=lambda kv: (arms_order.index(kv[0][0]) if kv[0][0] in arms_order else 99, list(STARTS).index(kv[0][1]))):
            pf, pt, tk = r.get('path_dev_fast_m') or {}, r.get('path_dev_teacher_m') or {}, r.get('takeover') or {}
            col = r.get('collision') or {}
            fg = r.get('first_gate_miss') or {}
            wri.writerow([r.get(k) for k in cols] + [col.get('kind'), col.get('x'), col.get('z'), fg.get('point'), fg.get('lateral_m'),
                         pf.get('at_0p5s'), pf.get('at_1s'), pf.get('at_end'), pf.get('max'), pt.get('at_end'),
                         (tk.get('fast') or {}).get('last_recoverable_frame'), (tk.get('margin') or {}).get('last_recoverable_frame'),
                         ';'.join(r.get('setting_check') or [])])
    sessions = sorted(CL.glob('session_*.json'))
    sess = [json.loads(p.read_text()) for p in sessions]
    speed = [{'label': j['label'], 'phase': j.get('phase'), 'attempt': j.get('attempt'), 'fps': j.get('fps'), 'frames': j.get('frames'),
              'timing': j.get('timing'), 'batch_wall_s': j.get('batch_wall_s'), 'error': j.get('error'), 'stopped': j.get('stopped')}
             for s in sess for j in s.get('jobs', []) if j.get('fps') is not None]

    def brief(r):
        return {k: r.get(k) for k in ('name', 'source', 'file', 'reason', 'counted_complete', 'lap_s', 'passed_gates', 'collisions', 'sim_time', 'frames',
                                      'min_clearance_m', 'collision', 'end_pose', 'first_straight_passed', 'g4', 'progress_m', 'd_value',
                                      'first_gate_miss', 'first_gate_line_over_0p65', 'path_dev_fast_m', 'path_dev_teacher_m', 'teacher', 'takeover', 'verification_pass',
                                      'consistency_pass', 'replay_same', 'setting_check', 'user_verifier', 'model_sha256', 'ort_ms_median',
                                      'max_speed', 'mean_speed')}
    today_runs = [r for r in rows]
    ev = {
        'schema': 'sim-diverse-train-eval-sim/v1',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'protocol_sha256': sha(REPORT / 'protocol.json'),
        'protocol_sha256_recorded': (REPORT / 'protocol.sha256').read_text().split()[0],
        'where': 'ブラウザ(snap の Firefox、DISPLAY :0)のアプリ(今日のシミュレーターの写し sim/dist/index.html、元とバイト単位で同じ)の連続試験。'
                 'ブリッジはユーザーの bridge.py(写し、元と同じ)を 10/14 の専用の venv(onnxruntime 1.31.0、CPU)で 127.0.0.1:8821',
        'settings': {'control_delay_frames': 1, 'lookahead': 1, 'steer_gain': 1.0, 'throttle_gain': '1.0(作者の ONNX は 1.01)',
                     'mode': '8(作者の ONNX は 4)', 'single_lap_limit_s': 15, 'physics_and_sensors': '10/14 と同じ(REC60 の計画の物理・センサーの欄・種 1・baseline)'},
        'starts': {e: list(v) for e, v in STARTS.items()},
        'loop_length_m': L,
        'bundles': {lab: {'zip': m['zip'], 'zip_sha256': sha(m['zip']), 'onnx_sha256_expected': onnx_sha.get(lab)}
                    for lab, m in MODEL_BY_LABEL.items() if m.get('available') and any(r['label'] == lab for r in rows)},
        'summary': summary,
        'runs': {f'{arm}/{e}': brief(r) for (arm, e), r in runs.items()},
        'extra_runs': {k: brief(v) | {'laps': v.get('laps'), 'flying_lap_s': v.get('flying_lap_s')} for k, v in extra.items()},
        'verdicts': verdicts,
        'positive_control_today': (brief(pos) if pos else None),
        'same_as_yesterday': same,
        'determinism_SDIV_E0': det,
        'flying_reference': flying,
        'teachers_node_delay0': {e: {kind: {k: teach[e][kind][k] for k in ('reason', 'lap_s', 'passed_gates', 'collisions', 'min_clearance_m')}
                                     for kind in ('fast', 'margin')} for e in teach},
        'user_verifier': uv_cache,
        'verification_all_runs': {'runs_today': len(today_runs),
                                  'consistency_pass_today': sum(1 for r in today_runs if r['verification'].get('consistency_pass')),
                                  'replay_same_teacher_eval': sum(1 for q in te['rows'] if q['replay_same']), 'teacher_eval_rows': len(te['rows']),
                                  'setting_mismatches': [r['name'] for r in today_runs if r['setting_check']], 'unreadable': problems},
        'speed_of_eval': speed,
        'figures': figs,
        'baseline_sources': {'prev_evaluation': str(PREV_EVAL), 'prev_teacher_eval': str(PREV_TE),
                             'note': 'A0・A1・A2・A0 種 1・S-T・作者の ONNX(E0〜E5)は 10/14 の評価の記録(同じアプリ・同じ計画・同じ物理)を読んだ。走らせ直していない'},
        'files': {'runs_csv': str(EVAL / 'runs.csv'), 'runs_verified': str(CL / 'runs_verified.json'), 'teacher_eval': str(te_path),
                  'closed_loop': str(CL), 'screens': str(REPORT / 'screens')},
    }
    (REPORT / 'eval_sim.json').write_text(json.dumps(ev, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'verdicts': {k: (v.get('verdict') if isinstance(v, dict) and 'verdict' in v else v) for k, v in verdicts.items()
                                   if k not in ('paired_reference', 'speed', 'external_reference_E0_E5')},
                      'c': c, 'g4': {x: summary[x]['g4'] for x in c}, 'fs': {x: summary[x]['fs'] for x in c},
                      'd': {x: summary[x]['d_median_m'] for x in c}, 'pos': (pos or {}).get('reason'), 'pos_lap': (pos or {}).get('lap_s'),
                      'determinism': det and det['identical'], 'flying': {k: (v['reason'], v['lap1_s'], v['flying_lap_s']) for k, v in flying.items()},
                      'consistency': ev['verification_all_runs']}, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
