"""1 分動画の図(6 枚)を、この作業とベストラップの生の記録から作る(確認の段、2026-10-14。CPU だけ)

    python3 -I make_slides.py

出力: /home/jetson/jetformer_reports/sim_dart_20261014/video/fig/v{1..6}_*.png(2160 × 1350)
数字・線はすべて記録の JSON から読む:
  - 走行: eval/closed_loop/*(今日)・ベストラップの eval/closed_loop/SR-s0・SR-s1(A0 の E0〜E5)
  - 先生の線: review/teacher_traces.json(ユーザーの sim-core.js で先生を走らせ直した軌跡。周回時間は評価の段の値と同じ)
  - 判定・中央値: review/recount.json(生の記録からの数え直し)、すき間: recording/summary.json・recording/A0_floor/deviation.json
  - 強化学習モード: /home/jetson/jetformer/data/sim_rl/modes/*/status.json(読むだけ)
v1 はベストラップの動画の図(その作業の記録から作った図)をそのまま写す。
"""
import json
import math
import shutil
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

REP = Path('/home/jetson/jetformer_reports/sim_dart_20261014')
BEST = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013')
OUT = REP / 'video' / 'fig'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, GREEN, PURPLE, RED, GRAY = '#2a78d6', '#eb6834', '#4a9a5b', '#6a4fc0', '#c0392b', '#8a8984'
ARM = {'A0': BLUE, 'A1': RED, 'A2': GREEN}
W_IN, H_IN, DPI = 14.4, 9.0, 150


def jp_font():
    for name in ('Noto Sans CJK JP', 'Noto Sans JP', 'IPAexGothic', 'Droid Sans Fallback'):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams['font.family'] = name
            return name


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def slide(title, sub):
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor=SURFACE)
    fig.text(0.035, 0.935, title, fontsize=30, fontweight='bold', color=INK, va='center')
    fig.text(0.037, 0.875, sub, fontsize=15, color=INK2, va='center')
    return fig


def banner(fig, text, y=0.035):
    ax = fig.add_axes([0.03, y, 0.94, 0.075])
    ax.axis('off')
    asp = (0.94 * W_IN) / (0.075 * H_IN)
    ax.add_patch(FancyBboxPatch((0.005, 0.05), 0.99, 0.9, boxstyle='round,pad=0.0,rounding_size=0.012', transform=ax.transAxes,
                                facecolor='#f0f4fa', edgecolor=INK, linewidth=3.5, mutation_aspect=asp))
    ax.text(0.02, 0.5, text, fontsize=21, fontweight='bold', color=INK, transform=ax.transAxes, va='center')


def course(ax, world, xlim=None, zlim=None):
    for s in world['segments']:
        kind = s.get('kind', 'rail')
        col = {'rail': '#b5554a', 'curtain': '#333333'}.get(kind, '#999999')
        ax.plot([s['a'][0], s['b'][0]], [s['a'][1], s['b'][1]], color=col, lw=3.0 if kind != 'curtain' else 4.0, alpha=.9,
                solid_capstyle='round', zorder=2)
    for d in world['disks']:
        if d.get('kind') == 'post-foot':
            ax.add_patch(Rectangle((d['x'] - .08, d['z'] - .075), .16, .15, color='#9a9a96', alpha=.8, zorder=2))
    for s in world.get('surfaces') or []:
        if s.get('id') == 'shortcut':
            ax.add_patch(Rectangle((s['x'] - s['w'] / 2, s['z'] - s['d'] / 2), s['w'], s['d'], color='#5a5a5a', alpha=.5, zorder=1))
    ax.set_aspect('equal')
    if xlim:
        ax.set_xlim(*xlim)
    if zlim:
        ax.set_ylim(*zlim)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)


def xz(rep):
    return [t['x'] for t in rep['trace']], [t['z'] for t in rep['trace']]


def crash_mark(ax, rep, color):
    ce = (rep['metrics'].get('collision_events') or [None])[0]
    if ce:
        ax.plot(ce['x'], ce['z'], marker='x', color=color, ms=16, mew=4, zorder=7)


def path_dist(run_xz, teach_xz, f, win=45):
    lo, hi = max(0, f - win), min(len(teach_xz) - 1, f + win)
    x, z = run_xz[f]
    best = 1e9
    for i in range(lo, hi):
        (ax_, az), (bx, bz) = teach_xz[i], teach_xz[i + 1]
        dx, dz = bx - ax_, bz - az
        L = dx * dx + dz * dz
        u = max(0.0, min(1.0, ((x - ax_) * dx + (z - az) * dz) / L)) if L > 0 else 0.0
        best = min(best, math.hypot(x - ax_ - u * dx, z - az - u * dz))
    return best


def gate0_runs():
    """ゲート 0 の開始の単周(A0 は E0〜E4 がベストラップの SR-s0、E6〜E9 が今日)"""
    runs = {'A0': [], 'A1': [], 'A2': []}
    for e in ['E0', 'E1', 'E2', 'E3', 'E4']:
        runs['A0'].append((e, load(BEST / f'eval/closed_loop/SR-s0/SR-s0-{e}-single.json')))
    for e in ['E6', 'E7', 'E8', 'E9']:
        runs['A0'].append((e, load(REP / f'eval/closed_loop/A0-s0/A0-s0-{e}-single.json')))
    for a in ['A1', 'A2']:
        for e in ['E0', 'E1', 'E2', 'E3', 'E4', 'E6', 'E7', 'E8', 'E9']:
            runs[a].append((e, load(REP / f'eval/closed_loop/{a}-s0/{a}-s0-{e}-single.json')))
    return runs


def v2_crash(teach):
    fig = slide('まねした JetFormer は、12 回とも約 1 秒で衝突', '10/13 の S-R(強化学習の走り 20 分で学習)・ブラウザのアプリ・制御遅延 1 フレーム・開始 6 通り × 種 2')
    reps = [(s, e, load(BEST / f'eval/closed_loop/SR-{s}/SR-{s}-{e}-single.json')) for s in ('s0', 's1') for e in ['E0', 'E1', 'E2', 'E3', 'E4', 'E5']]
    ax = fig.add_axes([0.02, 0.17, 0.6, 0.66])
    w = reps[0][2]['world']
    course(ax, w, xlim=(-1.3, 3.4), zlim=(1.25, 3.05))
    tx = teach['E0']['fast']['xz'][:75]
    ax.plot([p[0] for p in tx], [p[1] for p in tx], ls='--', color=INK, lw=2.4, label='強化学習の先生の線', zorder=4)
    for s, e, r in reps:
        if e == 'E5':
            continue
        x, z = xz(r)
        ax.plot(x, z, color=BLUE, lw=2.6, alpha=.75, zorder=5)
        crash_mark(ax, r, RED)
    ax.plot([], [], color=BLUE, lw=2.6, label='まねしたモデル(ゲート 0 からの 10 本)')
    ax.plot([], [], marker='x', color=RED, ms=14, mew=4, ls='', label='衝突')
    ax.text(-0.85, 2.02, 'スタート', fontsize=14, color=INK, ha='center', va='bottom')
    ax.text(-0.2, 2.97, '壁(レール)', fontsize=14, color='#b5554a', ha='center', va='bottom')
    ax.legend(loc='upper right', fontsize=13, frameon=True)
    bx = fig.add_axes([0.67, 0.2, 0.3, 0.6])
    t = sorted(r['metrics']['elapsed'] for _, _, r in reps)
    bx.barh(range(len(t)), t, color=BLUE, height=.7)
    bx.set_xlim(0, 2.0)
    bx.set_yticks([])
    bx.set_xlabel('衝突までの時間 [秒]', fontsize=14)
    bx.tick_params(labelsize=13)
    for sp in ('top', 'right', 'left'):
        bx.spines[sp].set_visible(False)
    bx.set_title('12 本の衝突の時刻 %.2f〜%.2f 秒' % (t[0], t[-1]), fontsize=15, color=INK)
    banner(fig, '停止から 1 周の完走: 0 / 12(最速の強化学習の先生は 12 / 12)')
    fig.savefig(OUT / 'v2_crash.png', facecolor=SURFACE)
    plt.close(fig)
    return {'crash_s_min': t[0], 'crash_s_max': t[-1], 'n': len(t)}


def v3_cause(teach, runs, floor):
    fig = slide('原因: 余裕のない線と、戻し方を知らないこと', '左: 最初の直線(上から見た図)  右: 先生の線からの距離(ゲート 0 から始めた 9 本、ぶつかるまで)')
    ax = fig.add_axes([0.02, 0.17, 0.47, 0.66])
    w = runs['A0'][0][1]['world']
    course(ax, w, xlim=(-1.3, 3.0), zlim=(1.6, 3.0))
    tx = teach['E0']['fast']['xz'][:75]
    ax.plot([p[0] for p in tx], [p[1] for p in tx], color=INK, lw=3, label='強化学習の先生の線', zorder=4)
    clear = floor['summary']['first_straight_min_dmin_p50_m']
    near = max(tx, key=lambda p: p[1] if p[0] < 2.8 else -9)
    ax.annotate('', xy=(near[0], 2.79), xytext=(near[0], near[1] + 0.02), arrowprops=dict(arrowstyle='<->', color=RED, lw=2.5))
    ax.text(near[0] - 0.12, 2.69, '壁まで 約 %.0f cm' % (clear * 100), fontsize=18, fontweight='bold', color=RED, va='center', ha='right')
    ax.text(-0.85, 2.04, 'スタート', fontsize=14, color=INK, ha='center', va='bottom')
    ax.legend(loc='upper right', fontsize=13)
    bx = fig.add_axes([0.6, 0.2, 0.37, 0.6])
    ends = []
    for e, r in runs['A0']:
        rx = [(t['x'], t['z']) for t in r['trace']]
        txe = [tuple(p) for p in teach[e]['fast']['xz']]
        d = [path_dist(rx, txe, f) * 100 for f in range(len(rx))]
        tt = [(f + 1) / 30 for f in range(len(rx))]
        bx.plot(tt, d, color=BLUE, lw=2.2, alpha=.8)
        bx.plot(tt[-1], d[-1], marker='x', color=RED, ms=12, mew=3)
        ends.append(d[-1])
    bx.axhline(10, color=RED, lw=1.5, ls='--')
    bx.text(0.03, 10.6, '10 cm', color=RED, fontsize=13)
    bx.set_xlabel('スタートからの時間 [秒]', fontsize=14)
    bx.set_ylabel('先生の線からの距離 [cm]', fontsize=14)
    bx.tick_params(labelsize=13)
    bx.set_xlim(0, 1.4)
    bx.grid(color=GRID)
    for sp in ('top', 'right'):
        bx.spines[sp].set_visible(False)
    bx.set_title('ずれは止まらずに広がり、%.0f〜%.0f cm で衝突' % (min(ends), max(ends)), fontsize=15, color=INK)
    banner(fig, 'きれいな走りだけを見て学んだので、ずれたときの戻し方を知らない')
    fig.savefig(OUT / 'v3_cause.png', facecolor=SURFACE)
    plt.close(fig)
    return {'A0_gate0_end_dist_cm': [round(min(ends), 1), round(max(ends), 1)], 'first_straight_clearance_cm': round(clear * 100, 1)}


def v4_fixes(teach, runs, rec, rec_a0_clear):
    fig = slide('今日の 2 つの対策', '1. DART: わざと線から外し、その状態で先生が出す「戻す操作」を正解にする   2. 壁から離れた、余裕のある線')
    # 左: 記録の段の図(D1 の 292 回の乱しの平均。記録の舵と付け替えた正解)の右半分
    im = Image.open(REP / 'fig/record_recovery_D1.png').convert('RGB')
    W, H = im.size
    crop = im.crop((int(W * 0.5), 0, W, H))
    ax = fig.add_axes([0.02, 0.17, 0.47, 0.62])
    ax.imshow(crop)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_title('1. DART(10 分): 乱しの間、正解は逆向きの舵', fontsize=17, fontweight='bold', color=INK, pad=10)
    bx = fig.add_axes([0.52, 0.17, 0.46, 0.62])
    w = runs['A0'][0][1]['world']
    course(bx, w, xlim=(-1.3, 3.0), zlim=(1.5, 3.0))
    f, m = teach['E0']['fast']['xz'][:75], teach['E0']['margin']['xz'][:75]
    bx.plot([p[0] for p in f], [p[1] for p in f], color=BLUE, lw=3, label='最速の線(昨日のデータ)', zorder=4)
    bx.plot([p[0] for p in m], [p[1] for p in m], color=GREEN, lw=3.4, label='余裕のある線(今日のデータ)', zorder=5)
    cf = rec['D1']['deviation']['first_straight_min_dmin_p50_m']
    cm = rec['D2']['deviation']['first_straight_min_dmin_p50_m']
    def z_at(line, x0):
        return min((p for p in line if p[1] > 1.8), key=lambda p: abs(p[0] - x0))[1]
    for line, x0, col in ((f, 2.2, BLUE), (m, 1.0, GREEN)):
        bx.annotate('', xy=(x0, 2.79), xytext=(x0, z_at(line, x0) + .015), arrowprops=dict(arrowstyle='<->', color=col, lw=2.5))
    bx.text(2.12, 2.69, '壁まで 約 %.0f cm' % (rec_a0_clear * 100), fontsize=16, color=BLUE, fontweight='bold', va='center', ha='right')
    bx.text(0.92, 2.63, '壁まで 約 %.0f cm' % (cm * 100), fontsize=16, color=GREEN, fontweight='bold', va='center', ha='right')
    bx.legend(loc='upper left', fontsize=13)
    bx.set_title('2. 余裕のある線(10 分): 最初の直線で壁から離れる', fontsize=17, fontweight='bold', color=INK, pad=10)
    banner(fig, 'どちらも 30 / 30 シーン合格(衝突 0)。A1 = 昨日 + DART、A2 = 余裕の線で学習')
    fig.savefig(OUT / 'v4_fixes.png', facecolor=SURFACE)
    plt.close(fig)
    return {'D1_first_straight_clear_cm': round(cf * 100, 1), 'D2_first_straight_clear_cm': round(cm * 100, 1)}


def v5_results(teach, runs, rc):
    fig = slide('結果: 最初の直線は抜けた。でも完走は 0', '停止から 1 周・12 本ずつ(種 0)・ブラウザのアプリ・制御遅延 1 フレーム。数は生の記録から数え直した')
    arms = [('A0', 'A0-s0', '昨日のデータ'), ('A1', 'A1-s0', '+ DART'), ('A2', 'A2-s0', '余裕のある線')]
    ax = fig.add_axes([0.15, 0.53, 0.27, 0.27])
    vals = [rc['arms'][k]['first_straight_passed'] for _, k, _ in arms]
    ax.barh([2, 1, 0], vals, color=[ARM[a] for a, _, _ in arms], height=.6)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([n for _, _, n in arms], fontsize=15)
    ax.set_xlim(0, 9.8)
    for y, v in zip([2, 1, 0], vals):
        ax.text(v + .15, y, '%d / 9' % v, va='center', fontsize=17, fontweight='bold')
    ax.set_title('最初の直線を抜けた本数(ゲート 0 の開始)', fontsize=15, loc='left')
    ax.set_xticks([])
    for sp in ('top', 'right', 'bottom'):
        ax.spines[sp].set_visible(False)
    bx = fig.add_axes([0.15, 0.19, 0.27, 0.25])
    st = [rc['arms'][k]['stop_time_median_s'] for _, k, _ in arms]
    bx.barh([2, 1, 0], st, color=[ARM[a] for a, _, _ in arms], height=.6)
    bx.set_yticks([2, 1, 0])
    bx.set_yticklabels([n for _, _, n in arms], fontsize=15)
    bx.set_xlim(0, 8.2)
    for y, v in zip([2, 1, 0], st):
        bx.text(v + .1, y, '%.1f 秒' % v, va='center', fontsize=17, fontweight='bold')
    bx.set_title('止まるまでの時間(12 本の中央値)', fontsize=15, loc='left')
    bx.set_xticks([])
    for sp in ('top', 'right', 'bottom'):
        bx.spines[sp].set_visible(False)
    cx = fig.add_axes([0.46, 0.14, 0.52, 0.64])
    w = runs['A1'][0][1]['world']
    course(cx, w, xlim=(1.0, 5.4), zlim=(-0.6, 3.0))
    f = teach['E0']['fast']['xz'][:150]
    cx.plot([p[0] for p in f], [p[1] for p in f], ls='--', color=INK, lw=2.4, zorder=4, label='最速の先生の線')
    for a in ('A1', 'A2'):
        for e, r in runs[a]:
            x, z = xz(r)
            n = min(len(x), 150)
            cx.plot(x[:n], z[:n], color=ARM[a], lw=2.0, alpha=.7, zorder=5)
        cx.plot([], [], color=ARM[a], lw=2.6, label={'A1': '+ DART', 'A2': '余裕のある線'}[a])
    # 点 4 のゲート(点 3 → 点 4 の向きに垂直・半幅 0.65 m)
    route = w['route']
    p3, p4 = route[3], route[4]
    n = math.hypot(p4[0] - p3[0], p4[1] - p3[1])
    ux, uz = (p4[0] - p3[0]) / n, (p4[1] - p3[1]) / n
    lx, lz = -uz, ux
    cx.plot([p4[0] - .65 * lx, p4[0] + .65 * lx], [p4[1] - .65 * lz, p4[1] + .65 * lz], color=ORANGE, lw=7, solid_capstyle='round', zorder=6)
    cx.plot([], [], color=ORANGE, lw=7, label='最初のカーブのゲート')
    cx.legend(loc='upper right', fontsize=13, framealpha=.95)
    cx.set_title('A1 の 6 本・A2 の 8 本が、このゲートを内側で外した(最初の 5 秒)', fontsize=15, color=INK)
    c = [rc['arms'][k]['completed'] for _, k, _ in arms]
    banner(fig, '完走: 昨日 %d / 12・DART %d / 12・余裕の線 %d / 12   判定: どちらも「一部効いた」' % tuple(c))
    fig.savefig(OUT / 'v5_results.png', facecolor=SURFACE)
    plt.close(fig)
    return {'first_straight': vals, 'stop_median_s': st, 'completed': c}


def v6_modes():
    fig = slide('強化学習モード(別の作業)の今', 'JetFormer の「シミュレーター」タブで選べるモード。合否は結果を見る前に決めた試験で判定')
    rows = [('fastest', '最速'), ('speed', '速さいろいろ'), ('no_wall', '絶対壁にぶつからない'), ('recover', '離脱復活'), ('line', '経路いろいろ'), ('obstacle', '障害物よけ')]
    st = {k: load(f'/home/jetson/jetformer/data/sim_rl/modes/{k}/status.json').get('label_ja') for k, _ in rows}
    ax = fig.add_axes([0.06, 0.15, 0.88, 0.66])
    ax.axis('off')
    for i, (k, name) in enumerate(rows):
        y = 1 - (i + 0.5) / len(rows)
        lab = st[k]
        col = GREEN if lab == '合格' else (ORANGE if '参考' in lab else GRAY)
        ax.add_patch(FancyBboxPatch((0.0, y - 0.065), 1.0, 0.13, boxstyle='round,pad=0.0,rounding_size=0.02', transform=ax.transAxes,
                                    facecolor='#ffffff', edgecolor=GRID, linewidth=2))
        ax.text(0.03, y, name, fontsize=24, fontweight='bold', color=INK, va='center', transform=ax.transAxes)
        ax.text(0.5, y, lab, fontsize=24, fontweight='bold', color=col, va='center', transform=ax.transAxes)
    fig.text(0.06, 0.1, '合格: 最速・速さいろいろ。壁にぶつからない・離脱復活は参考として選べる。今日の余裕のある線は、経路いろいろの方策を左へ寄せて使った',
             fontsize=14, color=INK2)
    fig.savefig(OUT / 'v6_modes.png', facecolor=SURFACE)
    plt.close(fig)
    return st


def main():
    jp_font()
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BEST / 'video/fig/v1_rl_vs_teacher.png', OUT / 'v1_bestlap.png')
    teach = load(REP / 'review/teacher_traces.json')['runs']
    runs = gate0_runs()
    rc = load(REP / 'review/recount.json')
    rec = load(REP / 'recording/summary.json')
    floor = load(REP / 'recording/A0_floor/deviation.json')
    nums = {'v2': v2_crash(teach), 'v3': v3_cause(teach, runs, floor), 'v4': v4_fixes(teach, runs, rec, floor['summary']['first_straight_min_dmin_p50_m']), 'v5': v5_results(teach, runs, rc), 'v6': v6_modes()}
    # 先生の線からの距離(1 秒の時点)を自前で数え直す(evaluation.md の 12.5 / 4.3 / 1.1 cm と比べる。A2 は余裕の線の先生から)
    d1 = {}
    for a, rr in runs.items():
        v = []
        for e, r in rr:
            rx = [(t['x'], t['z']) for t in r['trace']]
            txe = [tuple(p) for p in teach[e]['margin' if a == 'A2' else 'fast']['xz']]
            f = min(29, len(rx) - 1)
            v.append(path_dist(rx, txe, f) * 100)
        d1[a] = round(statistics.median(v), 2)
    nums['path_dist_1s_median_cm_gate0_starts'] = d1
    (OUT / 'slide_numbers.json').write_text(json.dumps(nums, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(nums, ensure_ascii=False))


if __name__ == '__main__':
    main()
