"""10/15 の約 1 分の動画の図 5 枚(2160 × 1350)を、生の記録と評価の JSON から作る(最終の確かめの段、2026-10-15。CPU だけ)

    /home/jetson/miniconda/envs/jetracer/bin/python -I make_slides.py

形は 10/14 の動画の図(sim_rl_modes_20261013/video/fig/r5_v*.png、experiments/20261013_sim_rl_modes/r3/review/video/make_slides_r4.py)にそろえた
(関数はこのファイルに写した。元のファイルは読むだけ)。
  d1_problem   前回の課題: いろいろな走りは作れた / まねしたモデルは完走 0 / 前の混ぜ方は実機の精度が変わらない
  d2_plan      今日やったこと: 4 つのモデルと評価(シミュレーターの実際の画面)
  d3_sim_result 結果(シミュレーター): 完走・点 4 のゲートの数と、A2 と S-DIV の走った線(走行の生の JSON から描く)
  d4_sim_run   S-DIV の完走の走り(シミュレーターの実際の画面・車載カメラ)と、残った失敗の場所
  d5_real      結果(実機の保留データ): 種ごとの L1(数え直しの JSON)と判定
数は review/final/recount_sim_indep.json・recount_real_indep.json・eval_real.json・eval_sim.json から読む。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

REP = Path('/home/jetson/jetformer_reports/sim_diverse_train_20261015')
OUT = REP / 'video' / 'fig'
SCR = REP / 'screens'
LAP = SCR / 'lap_SDIV_E2'
MODES_TRAJ = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/fig/promote_record_trajectories.png')
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, GREEN, PURPLE, RED, GOLD, GRAY = '#2a78d6', '#eb6834', '#1bab7a', '#6a4fc0', '#c0392b', '#c98500', '#8a8985'
W_IN, H_IN, DPI = 14.4, 9.0, 150
ARM_COLOR = {'A0': GRAY, 'A1': GRAY, 'A2': GRAY, 'S-DIV': BLUE, 'S-MOD': PURPLE, 'A4': ORANGE}


def jp_font():
    for name in ('Noto Sans CJK JP', 'Noto Sans JP', 'IPAexGothic'):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams['font.family'] = name
            return name


def rj(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def slide(title, sub):
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor=SURFACE)
    fig.text(0.035, 0.935, title, fontsize=30, fontweight='bold', color=INK, va='center')
    fig.text(0.037, 0.875, sub, fontsize=15, color=INK2, va='center')
    return fig


def box(fig, x, y, w, h, edge, face, lines):
    ax = fig.add_axes([x, y, w, h])
    ax.axis('off')
    asp = (w * W_IN) / (h * H_IN)
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle='round,pad=0.0,rounding_size=0.03', transform=ax.transAxes,
                                facecolor=face, edgecolor=edge, linewidth=3, mutation_aspect=asp))
    n = len(lines)
    for i, (t, size, bold, col) in enumerate(lines):
        ax.text(0.04, 1 - (i + 0.75) / (n + 0.4), t, fontsize=size, fontweight='bold' if bold else 'normal', color=col,
                transform=ax.transAxes, va='center')
    return ax


def banner(fig, text, y=0.035):
    ax = fig.add_axes([0.03, y, 0.94, 0.075])
    ax.axis('off')
    asp = (0.94 * W_IN) / (0.075 * H_IN)
    ax.add_patch(FancyBboxPatch((0.005, 0.05), 0.99, 0.9, boxstyle='round,pad=0.0,rounding_size=0.012', transform=ax.transAxes,
                                facecolor='#f0f4fa', edgecolor=INK, linewidth=3.5, mutation_aspect=asp))
    ax.text(0.02, 0.5, text, fontsize=21, fontweight='bold', color=INK, transform=ax.transAxes, va='center')


def show(fig, im, rect, label=None, color=None, size=15):
    ax = fig.add_axes(rect)
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(color or GRID)
        sp.set_linewidth(3 if color else 2)
    if label:
        ax.set_title(label, fontsize=size, color=color or INK2, fontweight='bold', pad=6)
    return ax


def view3d(path):
    """追従の視点の窓(1400 × 778)から 3D の表示の部分だけを切り出す"""
    return Image.open(path).convert('RGB').crop((34, 116, 1038, 540))


def cam(path):
    return Image.open(path).convert('RGB')


# ---------------------------------------------------------------- コースと走った線(走行の生の JSON)

def runs_of(rec, arm):
    return {e: rj(r['file']) for e, r in rec['arms'][arm]['runs'].items()}


def draw_course(ax, world, gate4=True):
    for s in world['segments']:
        (ax_, az), (bx, bz) = s['a'], s['b']
        col = '#3a3a38' if s.get('kind') == 'curtain' else '#a9a7a0'
        ax.plot([ax_, bx], [az, bz], color=col, lw=1.6 if s.get('kind') == 'curtain' else 1.1, zorder=1, solid_capstyle='round')
    for d in world['disks']:
        if d.get('kind') == 'post-foot':
            ax.add_patch(plt.Circle((d['x'], d['z']), max(d['radius'], 0.05), color='#c9c7c0', zorder=1, lw=0))
    for rc in world.get('requiredCrossings', []):
        ax.fill_between([rc['minX'], rc['maxX']], -0.28, 0.28, color='#3d3d3a', alpha=0.25, zorder=0, lw=0)
    if gate4:
        r = world['route']
        p3, p4 = r[3], r[4]
        dx, dz = p4[0] - p3[0], p4[1] - p3[1]
        n = (dx * dx + dz * dz) ** 0.5
        lx, lz = -dz / n * 0.65, dx / n * 0.65
        ax.plot([p4[0] - lx, p4[0] + lx], [p4[1] - lz, p4[1] + lz], color=GOLD, lw=4, zorder=4, solid_capstyle='round')
        ax.text(p4[0] + 0.1, 3.05, '点 4 のゲート', color=GOLD, fontsize=13, fontweight='bold', zorder=5, va='top', ha='center')
    ax.set_aspect('equal')
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(3.6, -3.3)
    ax.axis('off')


def draw_runs(ax, runs, color, lw=1.4, alpha=0.85, mark=True):
    for e, r in runs.items():
        xs = [t['x'] for t in r['trace']]
        zs = [t['z'] for t in r['trace']]
        ok = r['outcome'].get('reason') == 'route_complete'
        ax.plot(xs, zs, color=color, lw=lw * (1.25 if ok else 1.0), alpha=alpha, zorder=3)
        if mark and r['outcome'].get('reason') == 'collision':
            ax.plot(xs[-1], zs[-1], marker='x', ms=12, mew=3.2, color=INK, zorder=6)
            ax.plot(xs[-1], zs[-1], marker='x', ms=9, mew=1.8, color=color, zorder=7)


# ---------------------------------------------------------------- 図

def d1_problem(rec, ereal):
    fig = slide('前回の課題: いろいろな走りは作れた。でも…', '10/14 に 5 つの走り方で見本を記録した。まねした JetFormer と、10/13 の混ぜ方の結果は次のとおり')
    show(fig, Image.open(MODES_TRAJ).convert('RGB'), [0.03, 0.40, 0.45, 0.40], '① 5 つの走り方の見本(10/14・上から見た図)', BLUE)
    fig.text(0.255, 0.365, '走る線のばらつきは前の先生の約 23 倍', fontsize=15, color=INK, ha='center', va='center', fontweight='bold')
    ax = fig.add_axes([0.52, 0.40, 0.45, 0.40])
    a2 = runs_of(rec, 'A2')
    draw_course(ax, next(iter(a2.values()))['world'])
    draw_runs(ax, a2, GRAY, lw=1.3)
    ax.set_title('② まねしたモデル(10/14 の A2)の 12 回の走り', fontsize=15, color=RED, fontweight='bold', pad=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    c = rec['arms']
    fig.text(0.745, 0.365, '完走 %d / 12。最初のカーブで内側を切って、柱に当たる(× = 衝突)' % c['A2']['c'], fontsize=14, color=INK,
             ha='center', va='center', fontweight='bold')
    A, C = ereal['A_mean_us']['best'], ereal['C_mean_us']['best']
    box(fig, 0.03, 0.14, 0.45, 0.19, BLUE, '#eef4fc', [
        ('① データは作れた', 17, True, INK),
        ('最速・速さいろいろ・絶対壁にぶつからない・離脱復活・経路いろいろ', 13.5, False, INK),
        ('75 シーン(25 分)+ 戻し方の DART・余裕の線', 13.5, False, INK)])
    box(fig, 0.52, 0.14, 0.45, 0.19, RED, '#fbefed', [
        ('② まねしたモデルは 1 周も完走できない(10/14: 0 / 12)', 15, True, INK),
        ('③ 前の先生のくり返しを実機に混ぜても(10/13)、', 14, False, INK),
        ('   実機の保留データの誤差は同じ: A %.2f µs・C %.2f µs(同等)' % (A, C), 14, False, INK)])
    banner(fig, '今日: このいろいろな走りのデータで学習して、走りと精度が上がるか確かめる')
    fig.savefig(OUT / 'd1_problem.png', facecolor=SURFACE)
    plt.close(fig)


def d2_plan():
    fig = slide('今日やったこと: いろいろな走りのデータで学習した', 'JetFormer(TransFuser軽量型)を、先に決めた手順で 5 本学習(シミュレーター用 3 本・実機用 2 本)。実車は走らせていない')
    show(fig, view3d(LAP / 'eval_SDIV-s0-E2-single_f123_window.png'), [0.03, 0.42, 0.45, 0.385], 'シミュレーター(学習した S-DIV が走行中・追従の視点)')
    for i, (f, t) in enumerate([('f030', '1.0 秒'), ('f072', '2.4 秒'), ('f175', '5.8 秒')]):
        show(fig, cam(LAP / ('eval_SDIV-s0-E2-single_%s_camera.png' % f)), [0.03 + i * 0.155, 0.13, 0.14, 0.23], '車載カメラ %s' % t, size=13)
    cards = [
        ('S-DIV(シミュレーター用・主)', '5 つの走り方 + DART + 余裕の線(45 分)', BLUE, '#eef4fc'),
        ('S-MOD(シミュレーター用)', '5 つの走り方だけ(25 分)→ ばらつきと量を分ける', PURPLE, '#f3f1fa'),
        ('A4 = DAgger(シミュレーター用)', 'S-DIV が走った状態に、先生の正しい操作を付けて追加', ORANGE, '#fdf1ea'),
        ('C-DIV(実機用・種 2 つ)', '実機 52 シーン + 5 つの走り方 60 シーン(前の C と同じ量)', GREEN, '#ecf7f2'),
    ]
    for i, (t, s, col, face) in enumerate(cards):
        box(fig, 0.51, 0.665 - i * 0.165, 0.46, 0.15, col, face, [(t, 16, True, INK), (s, 13.5, False, INK)])
    banner(fig, '評価: シミュレーターで 1 周 × 12 回 / 実機の保留データ 4,670 フレーム')
    fig.savefig(OUT / 'd2_plan.png', facecolor=SURFACE)
    plt.close(fig)


def bars(fig, rect, arms, vals, top, title, unit_note):
    ax = fig.add_axes(rect)
    xs = range(len(arms))
    ax.bar(xs, [max(v, 0.06) for v in vals], color=[ARM_COLOR[a] for a in arms], width=0.62, zorder=2)
    for i, v in enumerate(vals):
        ax.text(i, v + top * 0.03, '%d' % v, ha='center', va='bottom', fontsize=19, fontweight='bold', color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(arms, fontsize=13, color=INK2)
    ax.set_ylim(0, top * 1.22)
    ax.set_yticks([0, top // 2 if top % 2 == 0 else top // 2, top])
    ax.tick_params(axis='y', labelsize=11, colors=INK2)
    ax.grid(axis='y', color=GRID, lw=1, zorder=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#bdbbb5')
    ax.set_title(title, fontsize=16, color=INK, loc='left', pad=10, fontweight='bold')
    ax.text(1.0, -0.2, unit_note, transform=ax.transAxes, fontsize=11, color=INK2, ha='right', va='top')
    ax.axvspan(-0.5, 2.5, color='#f1f0ec', zorder=0)
    ax.text(1.0, top * 1.16, '前回 10/14', ha='center', va='center', fontsize=12, color=INK2)
    ax.text(4.0, top * 1.16, '今日', ha='center', va='center', fontsize=12, color=INK2)
    return ax


def d3_sim_result(rec):
    A = rec['arms']
    arms = ['A0', 'A1', 'A2', 'S-DIV', 'S-MOD', 'A4']
    fig = slide('結果(シミュレーター): S-DIV が 12 回中 %d 回完走' % A['S-DIV']['c'],
                '停止から 1 周の単周を 12 か所の開始から 1 本ずつ(制御の遅れ 1 フレーム)。数は走行の生の記録から数え直した')
    bars(fig, [0.06, 0.53, 0.33, 0.26], arms, [A[a]['c'] for a in arms], 12, '1 周を完走した数(12 回中)', '')
    bars(fig, [0.06, 0.16, 0.33, 0.26], arms, [A[a]['g4'] for a in arms], 9, '最初のカーブ(点 4)のゲートを通った数(9 回中)', '')
    ax = fig.add_axes([0.43, 0.15, 0.55, 0.68])
    a2, sd = runs_of(rec, 'A2'), runs_of(rec, 'S-DIV')
    draw_course(ax, next(iter(sd.values()))['world'])
    draw_runs(ax, a2, GRAY, lw=1.2, alpha=0.75)
    draw_runs(ax, sd, BLUE, lw=1.5, alpha=0.9)
    ax.set_title('走った線(上から見た図): 灰 = 10/14 の A2、青 = 今日の S-DIV、× = 衝突', fontsize=14.5, color=INK, pad=6)
    ax.annotate('S-DIV の衝突が集まる所\n(S 字の出口の柱)', xy=(2.35, 0.32), xytext=(-0.2, 0.95), fontsize=13, color=RED, fontweight='bold',
                ha='center', va='center', zorder=8, arrowprops=dict(arrowstyle='->', color=RED, lw=2))
    banner(fig, '判定「一部効いた」: 完走 0 → %d / 12、点 4 のゲート 0 → %d / 9(「効いた」は完走 6 回以上)' % (A['S-DIV']['c'], A['S-DIV']['g4']))
    fig.savefig(OUT / 'd3_sim_result.png', facecolor=SURFACE)
    plt.close(fig)


def d4_sim_run(rec):
    sd = rec['arms']['S-DIV']
    comp = sd['completed']
    laps = [sd['runs'][e]['elapsed_s'] for e in comp]
    crashes = [r for r in sd['runs'].values() if r['reason'] == 'collision']
    near = [r for r in crashes if r['collision_xz'] and 2.3 <= r['collision_xz'][0] <= 2.7 and 0.15 <= r['collision_xz'][1] <= 0.35]
    fig = slide('S-DIV の完走の走り(シミュレーターの実際の画面)',
                'E2(ゲート 0・右 3 cm・右 3°)からの 1 周 %.2f 秒。上 = 追従の視点、下 = JetFormer が見ている車載カメラ(224 × 224)' % sd['runs']['E2']['elapsed_s'])
    frames = [('f030', '1.0 秒'), ('f072', '2.4 秒'), ('f123', '4.1 秒'), ('f230', '7.7 秒')]
    for i, (f, t) in enumerate(frames):
        x = 0.03 + i * 0.237
        show(fig, view3d(LAP / ('eval_SDIV-s0-E2-single_%s_window.png' % f)), [x, 0.60, 0.225, 0.22], t, size=14)
        show(fig, cam(LAP / ('eval_SDIV-s0-E2-single_%s_camera.png' % f)), [x + 0.045, 0.375, 0.135, 0.20])
    box(fig, 0.03, 0.135, 0.44, 0.21, BLUE, '#eef4fc', [
        ('完走した %d 周は %.2f〜%.2f 秒' % (len(comp), min(laps), max(laps)), 17, True, INK),
        ('最速の先生(同じ開始)の 1.03 倍。作者の ONNX(9.02 秒)より速い', 13.5, False, INK),
        ('完走した開始: %s' % '・'.join(comp), 13.5, False, INK)])
    box(fig, 0.49, 0.135, 0.33, 0.21, RED, '#fbefed', [
        ('残った失敗: S 字の出口の柱', 17, True, INK),
        ('衝突 %d 本のうち %d 本が同じ場所' % (len(crashes), len(near)), 13.5, False, INK),
        ('(黒いシートの近道に入るところ)', 13.5, False, INK)])
    show(fig, cam(SCR / 'eval_SDIV-s0-E6-single_f060_camera.png'), [0.835, 0.135, 0.135, 0.21])
    fig.text(0.9025, 0.362, 'E6・衝突の約 1 秒前', fontsize=12, color=INK2, ha='center', va='center')
    banner(fig, '最初のカーブは越えた。次の難所は S 字の出口の柱')
    fig.savefig(OUT / 'd4_sim_run.png', facecolor=SURFACE)
    plt.close(fig)


def d5_real(rr, ev):
    L = rr['L_us']['best']
    fig = slide('結果(実機の保留データ): 精度は上がらなかった',
                '人のきれいな走行 11 シーン・4,670 フレームの誤差(L1、小さいほど人の運転に近い)。best の checkpoint、種 2 つ')
    ax = fig.add_axes([0.07, 0.17, 0.45, 0.63])
    groups = [('A\n実機だけ', ['A_s0', 'A_s1'], GRAY), ('C(10/13)\n前の先生のくり返し', ['C_s0', 'C_s1'], GRAY),
              ('C-DIV(今日)\nいろいろな走り', ['CDIV_s0', 'CDIV_s1'], BLUE)]
    a_mean = rr['A_mean_us']['best']
    ax.axhspan(a_mean - 0.626, a_mean + 0.626, color='#eceae4', zorder=0)
    ax.axhline(a_mean, color=INK2, lw=1.2, ls='--', zorder=1)
    ax.text(2.48, a_mean + 0.626 - 0.03, 'A の平均 ± 閾値 0.626 µs', fontsize=11.5, color=INK2, ha='right', va='top')
    for i, (lab, ks, col) in enumerate(groups):
        vals = [L[k] for k in ks]
        m = sum(vals) / 2
        ax.plot([i - 0.22, i + 0.22], [m, m], color=col, lw=4, solid_capstyle='round', zorder=3)
        for j, v in enumerate(vals):
            ax.plot(i + (-0.09 if j == 0 else 0.09), v, 'o', ms=11, color=col, mec=SURFACE, mew=2, zorder=4)
        ax.text(i + 0.27, m, '%.2f' % m, fontsize=17, fontweight='bold', color=INK, va='center', zorder=5,
                bbox=dict(boxstyle='round,pad=0.15', facecolor=SURFACE, edgecolor='none'))
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[0] for g in groups], fontsize=13, color=INK2)
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(48.6, 51.0)
    ax.set_ylabel('誤差 L1 [µs](丸 = 種、線 = 平均)', fontsize=12, color=INK2)
    ax.tick_params(axis='y', labelsize=11, colors=INK2)
    ax.grid(axis='y', color=GRID, lw=1, zorder=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    va = rr['verdicts']
    d_a = va['CDIV_vs_A']['best']['diff_us']
    box(fig, 0.56, 0.52, 0.41, 0.28, BLUE, '#eef4fc', [
        ('A と比べて: +%.2f µs(%.1f %%)→ 決まりでは「悪い」' % (d_a, 100 * d_a / a_mean), 15, True, INK),
        ('ただし際どい: 種の並びの余裕は %.3f µs だけ' % va['CDIV_vs_A']['best']['order_margin_us'], 13.5, False, INK),
        ('last では「同等」。実用上の目安 10 µs よりずっと小さい', 13.5, False, INK),
        ('C と比べて: +%.2f µs →「同等」' % va['CDIV_vs_C']['best']['diff_us'], 15, True, INK)])
    sim = ev['sim_modes_val15']
    ax2 = fig.add_axes([0.62, 0.17, 0.33, 0.24])
    vals = [sim['per_arm']['C']['h8']['l1_mean_us'], sim['per_arm']['C-DIV']['h8']['l1_mean_us']]
    ax2.barh([1, 0], vals, color=[GRAY, BLUE], height=0.55, zorder=2)
    for y, v in zip([1, 0], vals):
        ax2.text(v + 2, y, '%.1f µs' % v, va='center', fontsize=15, fontweight='bold', color=INK)
    ax2.set_yticks([1, 0])
    ax2.set_yticklabels(['C', 'C-DIV'], fontsize=13, color=INK2)
    ax2.set_xlim(0, 100)
    ax2.set_xticks([])
    for sp in ('top', 'right', 'bottom'):
        ax2.spines[sp].set_visible(False)
    ax2.set_title('参考: シミュレーターの保留 15 シーンの誤差', fontsize=14, color=INK, loc='left', pad=8, fontweight='bold')
    fig.text(0.62, 0.14, 'シミュレーターの中では効いた(実機には移らなかった)', fontsize=12.5, color=INK2)
    banner(fig, '実機の保留データの誤差はほぼ同じ(差 1.5%)。シミュレーターでの改善は実機に移らなかった')
    fig.savefig(OUT / 'd5_real.png', facecolor=SURFACE)
    plt.close(fig)


def main():
    jp_font()
    OUT.mkdir(parents=True, exist_ok=True)
    rec = rj(REP / 'review/final/recount_sim_indep.json')
    rr = rj(REP / 'review/final/recount_real_indep.json')
    ev = rj(REP / 'eval_real.json')
    d1_problem(rec, rr)
    d2_plan()
    d3_sim_result(rec)
    d4_sim_run(rec)
    d5_real(rr, ev)
    print('ok', OUT)


if __name__ == '__main__':
    main()
