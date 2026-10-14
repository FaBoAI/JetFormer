"""1 分動画の図(4 枚)を、この作業の生の記録から作る(確認の段、2026-10-14。CPU だけ)

    python3 -I make_slides.py

出力: /home/jetson/jetformer_reports/sim_rl_bestlap_20261013/video/fig/v{1..4}_*.png(2160 × 1350)
数字はすべて JSON から読む(evaluation.json・recording/summary.json・review/*.json)。画面は screens/ の実際の画面。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

REP = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013')
OUT = REP / 'video' / 'fig'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, GREEN, PURPLE, RED = '#2a78d6', '#eb6834', '#4a9a5b', '#6a4fc0', '#c0392b'
W_IN, H_IN, DPI = 14.4, 9.0, 150


def jp_font():
    for name in ('Noto Sans CJK JP', 'Noto Sans JP', 'IPAexGothic', 'Droid Sans Fallback'):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams['font.family'] = name
            return name


def load(p):
    return json.loads((REP / p).read_text(encoding='utf-8'))


def slide(title, sub):
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor=SURFACE)
    fig.text(0.035, 0.935, title, fontsize=30, fontweight='bold', color=INK, va='center')
    fig.text(0.037, 0.875, sub, fontsize=15, color=INK2, va='center')
    return fig


def box(fig, x, y, w, h, edge, face, lines):
    """枠つきの文字の箱。lines = [(文字, 大きさ, 太字, 色)]"""
    ax = fig.add_axes([x, y, w, h])
    ax.axis('off')
    asp = (w * W_IN) / (h * H_IN)                      # 角の丸みを縦横で同じに見せる
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle='round,pad=0.0,rounding_size=0.03', transform=ax.transAxes,
                                facecolor=face, edgecolor=edge, linewidth=3, mutation_aspect=asp))
    n = len(lines)
    for i, (t, size, bold, col) in enumerate(lines):
        ax.text(0.06, 1 - (i + 0.75) / (n + 0.4), t, fontsize=size, fontweight='bold' if bold else 'normal', color=col,
                transform=ax.transAxes, va='center')
    return ax


def banner(fig, text, y=0.035):
    ax = fig.add_axes([0.03, y, 0.94, 0.075])
    ax.axis('off')
    asp = (0.94 * W_IN) / (0.075 * H_IN)
    ax.add_patch(FancyBboxPatch((0.005, 0.05), 0.99, 0.9, boxstyle='round,pad=0.0,rounding_size=0.012', transform=ax.transAxes,
                                facecolor='#f0f4fa', edgecolor=INK, linewidth=3.5, mutation_aspect=asp))
    ax.text(0.02, 0.5, text, fontsize=21, fontweight='bold', color=INK, transform=ax.transAxes, va='center')


def course(ax, world, light=False):
    for s in world['segments']:
        kind = s.get('kind', 'rail')
        col = {'rail': '#b5554a', 'curtain': '#333333', 'gate-foot': '#888888'}.get(kind, '#999999')
        ax.plot([s['a'][0], s['b'][0]], [s['a'][1], s['b'][1]], color=col, lw=2.2 if kind != 'curtain' else 3.2, alpha=.55 if light else .9,
                solid_capstyle='round', zorder=2)
    for d in world['disks']:
        if d.get('kind') == 'post-foot':
            ax.add_patch(Rectangle((d['x'] - .08, d['z'] - .075), .16, .15, color='#9a9a96', alpha=.7, zorder=2))
    for s in world.get('surfaces') or []:
        if s.get('id') == 'shortcut':
            ax.add_patch(Rectangle((s['x'] - s['w'] / 2, s['z'] - s['d'] / 2), s['w'], s['d'], color='#5a5a5a', alpha=.55, zorder=1))
            ax.text(s['x'], s['z'], '黒シート', color='white', fontsize=11, ha='center', va='center', zorder=3)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def screen(fig, path, rect, label):
    ax = fig.add_axes(rect)
    ax.imshow(Image.open(path).convert('RGB'))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
        sp.set_linewidth(2)
    ax.set_title(label, fontsize=14, color=INK2, pad=6)


def v1_rl(ev, rv):
    rl = load('rl/final/records/rl_main_E0_single.json')
    te = load('rl/final/records/teacher_E0_single.json')
    ref = ev['references']
    fig = slide('強化学習: 作者の教師より 0.84 秒速い周', '停止から 1 周・同じ物理と判定(元の sim-core.js)・衝突 0・33 ゲート・黒シート 1 回')
    ax = fig.add_axes([0.02, 0.15, 0.58, 0.68])
    course(ax, rl['world'])
    ax.plot([t['x'] for t in te['trace']], [t['z'] for t in te['trace']], color=ORANGE, lw=3.2, label='作者の RC 教師 %.3f 秒' % ref['T_teacher']['E0'], zorder=4)
    ax.plot([t['x'] for t in rl['trace']], [t['z'] for t in rl['trace']], color=BLUE, lw=3.2, label='強化学習(PPO) %.3f 秒' % ref['T_RL']['E0'], zorder=5)
    ax.plot(rl['trace'][0]['x'], rl['trace'][0]['z'], marker='o', color=INK, ms=11, zorder=6)
    ax.text(rl['trace'][0]['x'] - .25, rl['trace'][0]['z'] - .12, 'スタート', fontsize=13, color=INK, ha='right', va='bottom')
    ax.legend(loc='upper left', fontsize=14, frameon=True)
    # 右: 周回時間の棒
    bx = fig.add_axes([0.66, 0.42, 0.31, 0.38])
    rows = [('停止から 1 周', ref['T_RL']['E0'], ref['T_teacher']['E0']), ('2 周目(フライングラップ)', ref['RL_flying']['E0']['lap2'], ref['teacher_flying']['E0']['lap2'])]
    for i, (lab, a, b) in enumerate(rows):
        y = 1.9 - i * 1.9
        bx.barh(y + .35, b, height=.6, color=ORANGE)
        bx.barh(y - .35, a, height=.6, color=BLUE)
        bx.text(b + .05, y + .35, '%.3f' % b, va='center', fontsize=15, color=INK2)
        bx.text(a + .05, y - .35, '%.3f' % a, va='center', fontsize=15, color=BLUE, fontweight='bold')
        bx.text(7.02, y + .95, lab.replace('\n', ''), va='center', ha='left', fontsize=15, color=INK, fontweight='bold')
    bx.set_xlim(7, 9.6)
    bx.set_yticks([])
    bx.set_xlabel('秒(短いほど速い)', fontsize=13, color=INK2)
    bx.tick_params(labelsize=12)
    for sp in ('top', 'right', 'left'):
        bx.spines[sp].set_visible(False)
    rs = rv['independent_recheck']['random_starts']
    p1, p2 = rs['results']['P1-single']['laps_s'][0], rs['results']['P2-single']['laps_s'][0]
    box(fig, 0.63, 0.15, 0.345, 0.2, PURPLE, '#f3f1fa', [('ランダムな開始 2 つでも有効', 17, True, INK),
                                                       ('P1 %.3f 秒・P2(ゲート %d から)%.3f 秒' % (p1, rs['draw']['P2']['gate'], p2), 14, False, INK2),
                                                       ('別に書いた積分と判定で数え直した', 14, False, INK2)])
    banner(fig, '強化学習のベストラップは本物(独立な確認 3 回 + 確認の段)')
    fig.savefig(OUT / 'v1_rl_vs_teacher.png', facecolor=SURFACE)
    plt.close(fig)


def v2_data(ev, rv):
    s = load('recording/summary.json')
    fig = slide('その走りを 20 分のシミュレーターデータに', 'アプリの収集器で記録(作者の 20 分データと同じ形)・JetFormer にジャンル 8 で取り込み')
    screen(fig, REP / 'screens/00_t00030_f227_window.png', [0.03, 0.39, 0.42, 0.40], '記録中のアプリ')
    screen(fig, REP / 'screens/00_t00030_f227_camera.png', [0.47, 0.39, 0.25, 0.40], '車載カメラ(224 × 224)')
    screen(fig, REP / 'screens/00_t00030_f227_tof.png', [0.73, 0.39, 0.25, 0.40], 'ToF(距離)')
    box(fig, 0.03, 0.14, 0.45, 0.22, BLUE, '#f1f6fd', [('%d シーン・%s フレーム = %.0f 分' % (s['scenes'], format(s['frames'], ','), s['seconds'] / 60), 19, True, INK),
                                                     ('カメラ・ToF・IMU・PWM、全シーンで衝突 0', 15, False, INK2),
                                                     ('検証 %d/%d 合格・取り込み直しても同じ' % (s['verification']['passed_scenes'], s['scenes']), 15, False, INK2)])
    l1, l2 = s['laps_recorded']['lap1']['mean'], s['laps_recorded']['lap2']['mean']
    t1, t2 = s['laps_teacher_same_starts']['lap1_mean'], s['laps_teacher_same_starts']['lap2_mean']
    box(fig, 0.51, 0.14, 0.46, 0.22, ORANGE, '#fdf4ef', [('周回時間の平均(60 シーン)', 19, True, INK),
                                                       ('1 周目 %.3f 秒(教師のデータ %.3f)' % (l1, t1), 15, False, INK2),
                                                       ('2 周目 %.3f 秒(教師のデータ %.3f)' % (l2, t2), 15, False, INK2)])
    banner(fig, 'データセット名: jetformer-simulator-rl-bestlap-20min(シーンタブにも 60 シーン)')
    fig.savefig(OUT / 'v2_dataset.png', facecolor=SURFACE)
    plt.close(fig)


def v3_closed(ev):
    sr = ev['single_runs']
    def done(prefix):
        ks = [k for k in sr if k.startswith(prefix)]
        return sum(sr[k]['completed'] for k in ks), len(ks), [sr[k]['single_lap_s'] for k in ks if sr[k]['completed']]
    srn, srN, _ = done('SR-')
    stn, stN, stt = done('ST-')
    exn, exN, ext = done('EXT/')
    crash = [sr[k]['sim_time'] for k in sr if k.startswith('SR-')]
    fig = slide('学んだモデルは、シミュレーターで完走できなかった', '閉ループ(ブラウザのアプリ・制御遅延 1 フレーム)・停止から 1 周・開始 6 条件 × 種 2(作者のモデルは 6 条件)')
    ax = fig.add_axes([0.2, 0.17, 0.52, 0.64])
    rows = [('作者のモデル\n(参照)', exn, exN, GREEN, '完走 ' + '・'.join('%.3f' % x for x in ext) + ' 秒'),
            ('S-T\n教師のデータ', stn, stN, ORANGE, '完走 ' + '・'.join('%.3f' % x for x in stt) + ' 秒'),
            ('S-R\n強化学習のデータ', srn, srN, BLUE, '12 回とも %.2f〜%.2f 秒で衝突' % (min(crash), max(crash)))]
    for i, (lab, n, N, col, note) in enumerate(rows):
        ax.barh(i, N, color='#ecebe7', height=.62)
        ax.barh(i, n, color=col, height=.62)
        ax.text(-0.4, i, lab, ha='right', va='center', fontsize=16, color=INK)
        ax.text(N + .3, i + .12, '%d / %d' % (n, N), va='center', fontsize=22, fontweight='bold', color=col)
        ax.text(N + .3, i - .22, note, va='center', fontsize=13, color=INK2)
    ax.set_xlim(0, 19)
    ax.set_ylim(-.6, 2.6)
    ax.axis('off')
    ax.set_title('完走した回', fontsize=16, color=INK2, loc='left')
    screen(fig, REP / 'screens/eval_ST-s0_E0-single_f93_window.png', [0.74, 0.50, 0.24, 0.30], '評価中のアプリ')
    screen(fig, REP / 'screens/eval_ST-s0_E0-single_f93_camera.png', [0.79, 0.16, 0.15, 0.27], 'モデルが見る画像')
    banner(fig, '判定 H2: できなかった(基準は結果を見る前に固定・変えていない)')
    fig.savefig(OUT / 'v3_closed_loop.png', facecolor=SURFACE)
    plt.close(fig)


def v4_why(ev, rv):
    dn = load('review/derived_numbers.json')
    fs = rv['failure_shape_numbers']
    rl = load('rl/final/records/rl_main_E0_single.json')
    te = load('rl/final/records/teacher_E0_single.json')
    fig = slide('なぜぶつかったか: 線の余裕と、戻し方', '最初の直線の拡大(強化学習のデータで学んだモデル S-R の E0〜E4、10 回)')
    ax = fig.add_axes([0.02, 0.46, 0.6, 0.39])
    course(ax, rl['world'], light=False)
    ax.plot([t['x'] for t in te['trace'][:70]], [t['z'] for t in te['trace'][:70]], color=ORANGE, lw=2.6, label='作者の教師')
    ax.plot([t['x'] for t in rl['trace'][:70]], [t['z'] for t in rl['trace'][:70]], color=INK, lw=2.6, ls='--', label='強化学習の線')
    first = True
    for lab in ('SR-s0', 'SR-s1'):
        for e in ('E0', 'E1', 'E2', 'E3', 'E4'):
            r = load(f'eval/closed_loop/{lab}/{lab}-{e}-single.json')
            tr = r['trace']
            ax.plot([t['x'] for t in tr], [t['z'] for t in tr], color=BLUE, lw=1.6, alpha=.8, label='学んだモデル S-R' if first else None)
            ax.plot(tr[-1]['x'], tr[-1]['z'], marker='x', color=RED, ms=12, mew=3)
            first = False
    ax.set_xlim(-1.2, 4.3)
    ax.set_ylim(3.0, 1.15)
    ax.legend(loc='upper left', fontsize=13, ncol=3)
    near = fs['first_straight_nearest_obstacle_m']
    ax.annotate('右の壁まで約 %.0f cm(教師は %.0f cm)' % (100 * near['RL_E0']['d'], 100 * near['teacher_E0']['d']), xy=(2.45, 2.66), xytext=(0.6, 1.62),
                fontsize=15, color=INK, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=INK, lw=1.8))
    # 下: 舵
    bx = fig.add_axes([0.07, 0.19, 0.53, 0.21])
    row = next(r for r in load('eval/takeover.json')['rows'] if r['name'] == 'SR-s0-E0-single')
    pf = row['relabel']['per_frame']
    bx.plot([q[0] / 30 for q in pf], [q[1] for q in pf], color=BLUE, lw=2.6, label='モデルが出した舵')
    bx.plot([q[0] / 30 for q in pf], [q[2] for q in pf], color=INK, lw=2.2, ls='--', label='同じ所で強化学習が出す舵')
    bx.axhline(0, color=GRID, lw=1)
    bx.set_xlabel('走り出してからの時間 [秒](S-R 種 0・E0)', fontsize=12, color=INK2)
    bx.set_ylabel('舵 [rad](+ が左)', fontsize=12, color=INK2)
    bx.legend(fontsize=12, loc='upper left')
    bx.tick_params(labelsize=11)
    for sp in ('top', 'right'):
        bx.spines[sp].set_visible(False)
    a, e = fs['SR_dev_at_crash_same_start_m'], fs['SR_dev_at_0p5s_same_start_m']
    tk = [r for r in load('eval/takeover.json')['rows'] if r['label'].startswith('SR') and r['name'].endswith('-single')]
    back = [r['browser_frames'] - r['takeover']['last_recoverable_frame'] for r in tk]
    box(fig, 0.65, 0.58, 0.33, 0.27, BLUE, '#f1f6fd', [('ずれを直す小さな舵を出さない', 16, True, INK),
                                                     ('強化学習の線からのずれ', 14, False, INK2),
                                                     ('0.5 秒で %.0f〜%.0f cm → 衝突時 %.0f〜%.0f cm' % (100 * e[0], 100 * e[1], 100 * a[0], 100 * a[1]), 14, False, INK2)])
    box(fig, 0.65, 0.36, 0.33, 0.2, GREEN, '#f0f7f2', [('強化学習なら立て直せた', 16, True, INK),
                                                     ('衝突の %.2f〜%.2f 秒前まで' % (min(back) / 30, max(back) / 30), 14, False, INK2)])
    g = fs['gate_point4_lateral_m']
    box(fig, 0.65, 0.14, 0.33, 0.2, ORANGE, '#fdf4ef', [('教師の線も余裕が小さい', 16, True, INK),
                                                      ('点 4 のゲートの端まで %.1f cm' % (100 * (0.65 - g['teacher_E0'])), 14, False, INK2)])
    banner(fig, '線の上しか習っていない + 余裕の小さい所がある → 小さなずれが衝突になる')
    fig.savefig(OUT / 'v4_why.png', facecolor=SURFACE)
    plt.close(fig)


def main():
    jp_font()
    OUT.mkdir(parents=True, exist_ok=True)
    ev = load('evaluation.json')
    rv = ev['review_20261014']
    v1_rl(ev, rv)
    v2_data(ev, rv)
    v3_closed(ev)
    v4_why(ev, rv)
    print(sorted(p.name for p in OUT.glob('v*.png')))


if __name__ == '__main__':
    main()
