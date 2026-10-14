"""1 分動画の図(3 枚)を、この作業の生の記録から作る(審査の段、2026-10-14。CPU だけ)

    python3 -I make_slides.py

出力: /home/jetson/jetformer_reports/sim_rl_modes_20261013/video/fig/v{1..3}_*.png(2160 × 1350)
数字はすべて JSON から読む(登録簿 data/sim_rl/modes/<key>/{card,eval,status}.json・variation.json・review/fresh/<A|B>/<mode>.json)。
画面は screens/ の実際の画面(http://192.168.55.1:8080 の Firefox)。形は sim_rl_bestlap_20261013 の make_slides.py と同じ。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

REP = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013')
REG = Path('/home/jetson/jetformer/data/sim_rl/modes')
OUT = REP / 'video' / 'fig'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, GREEN, PURPLE, RED, GOLD, GRAY = '#2a78d6', '#eb6834', '#1bab7a', '#6a4fc0', '#c0392b', '#c98500', '#8a8985'
W_IN, H_IN, DPI = 14.4, 9.0, 150
NAMES = {'fastest': '最速', 'no_wall': '絶対壁にぶつからない', 'recover': '離脱復活', 'speed': '速さいろいろ', 'line': '経路いろいろ', 'obstacle': '障害物よけ'}
COLORS = {'fastest': BLUE, 'no_wall': ORANGE, 'recover': GREEN, 'speed': GOLD, 'line': GRAY, 'obstacle': GRAY}
BADGE = {'accepted': ('合格', '#e3f4e8', '#1d7a3e'), 'not_accepted': ('不合格(参考)', '#fdf0d5', '#9a6400'),
         'not_published': ('出していない', '#eeeeec', '#555555')}


def jp_font():
    for name in ('Noto Sans CJK JP', 'Noto Sans JP', 'IPAexGothic', 'Droid Sans Fallback'):
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
    """枠つきの文字の箱。lines = [(文字, 大きさ, 太字, 色)]"""
    ax = fig.add_axes([x, y, w, h])
    ax.axis('off')
    asp = (w * W_IN) / (h * H_IN)
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle='round,pad=0.0,rounding_size=0.03', transform=ax.transAxes,
                                facecolor=face, edgecolor=edge, linewidth=3, mutation_aspect=asp))
    n = len(lines)
    for i, (t, size, bold, col) in enumerate(lines):
        ax.text(0.05, 1 - (i + 0.75) / (n + 0.4), t, fontsize=size, fontweight='bold' if bold else 'normal', color=col,
                transform=ax.transAxes, va='center')
    return ax


def banner(fig, text, y=0.035):
    ax = fig.add_axes([0.03, y, 0.94, 0.075])
    ax.axis('off')
    asp = (0.94 * W_IN) / (0.075 * H_IN)
    ax.add_patch(FancyBboxPatch((0.005, 0.05), 0.99, 0.9, boxstyle='round,pad=0.0,rounding_size=0.012', transform=ax.transAxes,
                                facecolor='#f0f4fa', edgecolor=INK, linewidth=3.5, mutation_aspect=asp))
    ax.text(0.02, 0.5, text, fontsize=21, fontweight='bold', color=INK, transform=ax.transAxes, va='center')


def screen(fig, path, rect, label, crop=None):
    ax = fig.add_axes(rect)
    im = Image.open(path).convert('RGB')
    if crop:
        im = im.crop(crop)
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
        sp.set_linewidth(2)
    ax.set_title(label, fontsize=14, color=INK2, pad=6)


def badge(fig, x, y, state):
    t, face, col = BADGE[state]
    fig.text(x, y, ' %s ' % t, fontsize=13, color=col, fontweight='bold', va='center',
             bbox=dict(boxstyle='round,pad=0.25', facecolor=face, edgecolor=col, linewidth=1.2))


def v1_modes(cards):
    fig = slide('6 つの強化学習モードを用意', 'JetFormer の「シミュレーター」タブ → カード「強化学習モード」→「このモードで開く」/ 収集プラン(モード × 分)')
    keys = ['fastest', 'no_wall', 'recover', 'speed', 'line', 'obstacle']
    what = {'fastest': 'ベストラップの方策そのまま', 'no_wall': '壁・柱から離れて遅めに + 安全シールド',
            'recover': 'わざと乱して、コースへ戻る', 'speed': '速さ 50〜100% の 5 段',
            'line': '左寄り・右寄り・イン・アウトの線', 'obstacle': 'コースに足した障害物をよける'}
    for i, k in enumerate(keys):
        y = 0.775 - i * 0.112
        c = cards[k]
        fig.text(0.04, y, '強化学習モード(%s)' % NAMES[k], fontsize=16, fontweight='bold', color=COLORS[k] if k not in ('line', 'obstacle') else INK2, va='center')
        badge(fig, 0.385, y, c['state'])
        fig.text(0.04, y - 0.042, what[k], fontsize=13.5, color=INK2, va='center')
    screen(fig, REP / 'screens/record_tab_card_after.png', [0.52, 0.15, 0.46, 0.63], 'タブのカード(実際の画面)', crop=(680, 100, 1480, 830))
    banner(fig, '選べるのは 4 つ(合格 2・不合格(参考)2)。経路・障害物は不合格で出していない')
    fig.savefig(OUT / 'v1_modes.png', facecolor=SURFACE)
    plt.close(fig)


def v2_results(cards, evs, fresh):
    fig = slide('各モードの結果(シミュレーターの中)', '元の sim-core.js で試験。凍結した試験の組 + 審査で新しい種 2 つ(A・B)で走らせ直し')
    keys = ['fastest', 'no_wall', 'recover', 'speed']
    # 左: 停止から 1 周
    ax = fig.add_axes([0.13, 0.17, 0.21, 0.6])
    laps = [cards[k]['lap_from_stop_s'] for k in keys]
    for i, (k, v) in enumerate(zip(keys, laps)):
        ax.barh(3 - i, v, color=COLORS[k], height=.6)
        ax.text(v + .1, 3 - i, '%.2f 秒' % v, va='center', fontsize=15, color=INK, fontweight='bold')
    sp = evs['speed']['judged']['S3_lap_order']['value']
    ax.text(0.2, -0.75, '速さ: 100%% %.2f・75%% %.2f・50%% %.2f 秒' % (sp['t1'], sp['t75'], sp['t5']), fontsize=12, color=INK2)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels([{'no_wall': '壁にぶつからない'}.get(k, NAMES[k]) for k in keys], fontsize=14)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(-1.1, 3.6)
    ax.set_title('停止から 1 周', fontsize=16, color=INK, loc='left')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=11)
    # 右: 箱
    nw, rc = evs['no_wall']['judged'], evs['recover']['judged']
    fA, fB = fresh.get('A', {}), fresh.get('B', {})

    def nw_fresh(f):
        j = (f.get('no_wall') or {}).get('judged') or {}
        if not j:
            return '—'
        ok, n = (int(x) for x in j['N1_no_collision']['value'].split('/'))
        return '%d 回で衝突 %d・最小 %.1f cm' % (n, n - ok, 100 * j['N2_clearance_nominal']['value'])
    box(fig, 0.37, 0.56, 0.29, 0.25, ORANGE, '#fdf4ef', [
        ('絶対壁にぶつからない', 17, True, INK),
        ('試験 %s 回で衝突 0' % nw['N1_no_collision']['runs'], 14, False, INK2),
        ('最小すき間 %.1f cm(新しい開始では 7.0 cm)' % (100 * nw['N2_clearance_nominal']['value']), 14, False, INK2),
        ('物理の端(摩擦円+弱いブレーキ)で衝突', 14, False, RED)])
    box(fig, 0.68, 0.56, 0.29, 0.25, GREEN, '#f0f8f4', [
        ('離脱復活', 17, True, INK),
        ('立て直しの成功 %.0f%%(合格は 95%%)' % (100 * rc['C2_success']['value']), 14, False, INK2),
        ('立て直しの時間 中央値 %.2f 秒' % rc['C3_recovery_time']['value']['median'], 14, False, INK2),
        ('乱した 100 回で走り切るのは %s 回' % rc['C1_valid']['value'].split('/')[0], 14, False, RED)])
    sp_ = evs['speed']['judged']
    box(fig, 0.37, 0.17, 0.29, 0.34, GOLD, '#fdf7ea', [
        ('速さいろいろ・最速', 17, True, INK),
        ('速さいろいろ: %s 回走り切り・最小 %.1f cm' % (sp_['S1_valid']['value'].split('/')[0], 100 * sp_['S2_clearance']['value']), 14, False, INK2),
        ('最速: 停止から %.2f 秒・走りながら %.2f 秒' % (cards['fastest']['lap_from_stop_s'], cards['fastest']['flying_lap_s']), 14, False, INK2),
        ('どちらも合格', 14, True, '#1d7a3e')])

    def fresh_line(tag):
        f = fresh.get(tag, {})
        ok = [NAMES[k] for k in ('fastest', 'speed') if (f.get(k) or {}).get('all_judged_pass')]
        return '種 %s: %s は合格のまま' % (tag, '・'.join(ok) if ok else '—')
    box(fig, 0.68, 0.17, 0.29, 0.34, PURPLE, '#f3f1fa', [
        ('新しい種 2 つで走らせ直し', 17, True, INK),
        (fresh_line('A'), 14, False, INK2),
        (fresh_line('B'), 14, False, INK2),
        ('壁: 種 A %s' % nw_fresh(fA), 13, False, INK2),
        ('壁: 種 B %s' % nw_fresh(fB), 13, False, INK2),
        ('離脱復活: 走り切り 種 A %s 回・種 B %s 回' % tuple(((f.get('recover') or {}).get('judged') or {}).get('C1_valid', {}).get('value', '—/').split('/')[0]
                                                   for f in (fA, fB)), 13, False, INK2)])
    banner(fig, '合格は 最速・速さいろいろ。壁・離脱復活は「不合格(参考)」のまま出している')
    fig.savefig(OUT / 'v2_results.png', facecolor=SURFACE)
    plt.close(fig)


def v3_variation(var):
    t = var['teacher']['all60']
    fig = slide('データのばらつき: 前の 7.6 µs から', '記録した見本(各モード 15 シーン)を、前の教師のデータ(60 シーン)と同じ式で比べた')
    keys = ['fastest', 'no_wall', 'recover', 'speed']
    ax = fig.add_axes([0.16, 0.17, 0.27, 0.6])
    short = {'no_wall': '壁にぶつからない'}
    rows = [('前の教師', t['VM3_lateral_spread_m'], INK2)] + [(short.get(k, NAMES[k]), var['modes'][k]['VM3_lateral_spread_m'], COLORS[k]) for k in keys] + \
        [('4 モード合わせ', var['pooled']['VM3_lateral_spread_m'], PURPLE)]
    for i, (lab, v, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, 100 * v, color=col, height=.62)
        ax.text(100 * v + .1, y, '%.2f cm(%.1f倍)' % (100 * v, v / t['VM3_lateral_spread_m']), va='center', fontsize=13, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=13)
    ax.set_xlim(0, 8.6)
    ax.set_title('線のばらつき(走った線の横ずれの広がり)', fontsize=15, color=INK, loc='left')
    ax.set_xlabel('cm', fontsize=12, color=INK2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    screen(fig, REP / 'fig/record_trajectories.png', [0.47, 0.30, 0.51, 0.53], '記録した走り(上から見た図)')
    p = var['pooled']
    box(fig, 0.47, 0.13, 0.51, 0.15, PURPLE, '#f3f1fa', [
        ('舵の時刻のずれ: 前 %.1f µs → 4 モード %.0f µs(開始を散らした分も入る)' % (t['VM1_steer_time_aligned_us'], p['VM1_steer_time_aligned_us']), 13.5, False, INK),
        ('訪れた状態の数: %.1f 倍・開始ゲート 1 か所 → %d か所' % (p['ratio_vs_teacher']['VM7'], p['start_gates_distinct']), 13.5, False, INK)])
    banner(fig, '線のばらつきは 4 モード合わせで %.0f 倍。ただし壁にぶつからないは 1 本の線(%.1f 倍)' % (p['ratio_vs_teacher']['VM3'], var['modes']['no_wall']['ratio_vs_teacher']['VM3']))
    fig.savefig(OUT / 'v3_variation.png', facecolor=SURFACE)
    plt.close(fig)


def main():
    jp_font()
    OUT.mkdir(parents=True, exist_ok=True)
    keys = ['fastest', 'no_wall', 'recover', 'speed', 'line', 'obstacle']
    cards = {k: rj(REG / k / 'card.json') for k in keys}
    evs = {k: rj(REG / k / 'eval.json') for k in keys}
    fresh = {}
    for tag in ('A', 'B'):
        fresh[tag] = {}
        for k in keys:
            p = REP / 'review' / 'fresh' / tag / ('%s.json' % k)
            if p.exists():
                fresh[tag][k] = rj(p)
    var = rj(REP / 'variation.json')
    v1_modes(cards)
    v2_results(cards, evs, fresh)
    v3_variation(var)
    print(sorted(x.name for x in OUT.glob('v*.png')))


if __name__ == '__main__':
    main()
