"""1 分動画の図(3 枚)を、第 4 回の最終の状態の生の記録から作る(第 3〜4 回の審査の段、2026-10-15。CPU だけ)

    python3 -I make_slides_r4.py

出力: /home/jetson/jetformer_reports/sim_rl_modes_20261013/video/fig/r4_v{1..3}_*.png(2160 × 1350)
審査の段 1 回目の review/video/make_slides.py の写しを、第 4 回の数字に合わせて書き直したもの(前の図 v1〜v3 は残す)。
数字はすべて JSON から読む:
  登録簿 data/sim_rl/modes/<key>/{card,eval,status}.json(今)・昇格の前の写し r3/promote/before/registry/<key>/{card,eval}.json(前)、
  確認の段 check_no_wall_r2/summary.json(隅 K0)・r3/check_no_wall_r4/summary.json(隅の全部の組)・r3/check_{recover,line}_r4 の数、
  r3/variation.json(ばらつき)・r3/review/indep_variation_r4.json(審査の数え直し)。画面は r3/screens/ の実際の画面(http://192.168.55.1:8080 の Firefox)。
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
R3 = REP / 'r3'
REG = Path('/home/jetson/jetformer/data/sim_rl/modes')
BEFORE = Path('/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/promote/before/registry')
OUT = REP / 'video' / 'fig'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, GREEN, PURPLE, RED, GOLD, GRAY, VIOLET = '#2a78d6', '#eb6834', '#1bab7a', '#6a4fc0', '#c0392b', '#c98500', '#8a8985', '#7d4fc9'
W_IN, H_IN, DPI = 14.4, 9.0, 150
NAMES = {'fastest': '最速', 'no_wall': '絶対壁にぶつからない', 'recover': '離脱復活', 'speed': '速さいろいろ', 'line': '経路いろいろ', 'obstacle': '障害物よけ'}
COLORS = {'fastest': BLUE, 'no_wall': ORANGE, 'recover': GREEN, 'speed': GOLD, 'line': VIOLET, 'obstacle': GRAY}
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
    n_ok = sum(1 for c in cards.values() if c['state'] == 'accepted')
    fig = slide('強化学習モード 6 つ → %d つが合格' % n_ok, 'JetFormer の「シミュレーター」タブ → カード「強化学習モード」→「このモードで開く」/ 収集プラン(モード × 分)')
    keys = ['fastest', 'no_wall', 'recover', 'speed', 'line', 'obstacle']
    what = {'fastest': 'ベストラップの方策そのまま(いちばん速い線)', 'no_wall': '壁・柱から 8 cm 以上はなれて遅めに(安全シールド)',
            'recover': 'わざと乱して、コースへ戻る(乱した操作は学習に使わない)', 'speed': '速さ 50〜100% の 5 段',
            'line': '左寄り・右寄り・イン・アウトなど 8 つの線(4 cm シールド)', 'obstacle': '障害物をよける(試験で不合格)'}
    redo = {'no_wall', 'recover', 'line'}
    for i, k in enumerate(keys):
        y = 0.775 - i * 0.112
        c = cards[k]
        fig.text(0.04, y, '強化学習モード(%s)' % NAMES[k], fontsize=16, fontweight='bold', color=COLORS[k] if k != 'obstacle' else INK2, va='center')
        badge(fig, 0.385, y, c['state'])
        if k in redo:
            fig.text(0.455, y, '学び直し', fontsize=12, color=PURPLE, fontweight='bold', va='center')
        fig.text(0.04, y - 0.042, what[k], fontsize=13.5, color=INK2, va='center')
    screen(fig, R3 / 'screens/promote_tab_card_after.png', [0.535, 0.15, 0.445, 0.63], 'タブのカード(実際の画面、8080)', crop=(690, 110, 1470, 760))
    banner(fig, '選べるのは %d つ(全部「合格」)。障害物よけだけは不合格で出していない' % n_ok)
    fig.savefig(OUT / 'r4_v1_modes.png', facecolor=SURFACE)
    plt.close(fig)


def pair_axes(fig, rect, title, before, after, labels, unit, color, fmt, ref=None, ref_label=None, lower_better=False):
    ax = fig.add_axes(rect)
    vals = [before, after]
    ax.bar([0, 1], vals, color=[GRAY, color], width=0.62)
    top = max(vals + ([ref] if ref else [])) * 1.32 or 1
    for i, (v, lab) in enumerate(zip(vals, labels)):
        ax.text(i, v + top * 0.03, fmt % v, ha='center', va='bottom', fontsize=17, fontweight='bold', color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12.5, color=INK2)
    ax.set_ylim(0, top)
    if ref is not None:
        ax.axhline(ref, color=RED, lw=1.6, ls='--')
        ax.text(-0.42, ref + top * 0.01, ref_label, color=RED, fontsize=11.5, va='bottom', ha='left')
    ax.set_title(title, fontsize=15, color=INK, loc='left', pad=8)
    ax.set_ylabel(unit, fontsize=11, color=INK2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', labelsize=10)
    return ax


def v2_results(cards, evs, before, ck):
    fig = slide('学び直した 3 つ: 前の版 → 第 4 回', '結果を見る前に固めた試験(1 回だけ)+ 学習のコードを使わない確認。数字はシミュレーターの中')
    # 絶対壁にぶつからない: 物理の端の隅の衝突
    k0 = ck['nw_r2_K0']
    hc = ck['nw_r4_hard']
    pair_axes(fig, [0.06, 0.47, 0.25, 0.3], '壁: 物理の端の隅で衝突(確認の組)', 100 * k0['collisions'] / k0['n'], 100 * hc['coll'] / hc['n'],
              ['前 %d/%d 回' % (k0['collisions'], k0['n']), '今 %d/%d 回' % (hc['coll'], hc['n'])], '%', ORANGE, '%.0f%%')
    nwj = evs['no_wall']['judged']
    box(fig, 0.04, 0.15, 0.29, 0.25, ORANGE, '#fdf4ef', [
        ('絶対壁にぶつからない', 16, True, INK),
        ('試験 %s 回で衝突 0' % format(nwj['N1_r4']['runs'], ','), 13.5, False, INK2),
        ('最小すき間 %.1f cm = シールドが守る' % (100 * nwj['N2_r4']['value']), 13.5, False, INK2),
        ('方策だけなら %.1f cm まで寄る' % ck['nw_policy_only_cm'], 13.5, False, RED)])
    # 離脱復活
    rb, rn = before['recover']['judged'], evs['recover']['judged']
    pair_axes(fig, [0.39, 0.47, 0.25, 0.3], '離脱復活: 立て直しの成功', 100 * rb['C2_success']['value'], 100 * rn['C2']['value'],
              ['前(第 1 回)', '今(第 4 回)'], '%', GREEN, '%.0f%%', ref=95, ref_label='基準 95%')
    box(fig, 0.37, 0.15, 0.29, 0.25, GREEN, '#f0f8f4', [
        ('離脱復活', 16, True, INK),
        ('乱した 100 回で走り切り %s → %s 回' % (rb['C1_valid']['value'].split('/')[0], rn['C1_r3']['value'].split('/')[0]), 13.5, False, INK2),
        ('立て直せる乱しだけに作り直した', 13.5, False, INK2),
        ('停止から 1 周 %.2f 秒(基準 9.0)' % rn['C6_r4']['value']['lap_from_stop'], 13.5, False, RED)])
    # 経路いろいろ
    lb, ln = before['line']['judged'], evs['line']['judged']
    pair_axes(fig, [0.72, 0.47, 0.25, 0.3], '経路いろいろ: 最小すき間', 100 * lb['L2_clearance']['value'], 100 * ln['L2_r4']['value'],
              ['前(第 2 回)', '今(第 4 回)'], 'cm', VIOLET, '%.1f cm', ref=3, ref_label='基準 3 cm')
    box(fig, 0.70, 0.15, 0.29, 0.25, VIOLET, '#f4f0fb', [
        ('経路いろいろ', 16, True, INK),
        ('試験 %s 回で衝突 0・止まり 0' % ln['L8_no_collision']['value'].split('/')[1], 13.5, False, INK2),
        ('左右の線の差 %.0f cm・イン/アウト %.0f cm' % (100 * ln['L4_distinct']['value']['left_right_mean'], 100 * ln['L4_distinct']['value']['inside_outside_mean']), 13.5, False, INK2),
        ('シールドなしは角で崩れる(%s)' % ck['line_noshield_adv'], 13.5, False, RED)])
    banner(fig, '3 つとも、固めた試験と独立な確認(%s 回 + Firefox)で合格' % format(ck['check_runs_total'], ','))
    fig.savefig(OUT / 'r4_v2_results.png', facecolor=SURFACE)
    plt.close(fig)


def v3_variation(var, indep):
    t = var['teacher']['all60']
    fig = slide('データのばらつき(5 モードの見本)', '記録した見本(各モード 15 シーン)を、前の教師のデータ(60 シーン)と同じ式で比べた')
    keys = ['fastest', 'no_wall', 'recover', 'speed', 'line']
    ax = fig.add_axes([0.17, 0.2, 0.27, 0.6])
    short = {'no_wall': '壁にぶつからない'}
    rows = [('前の教師', t['VM3_lateral_spread_m'], INK2)] + [(short.get(k, NAMES[k]), var['modes'][k]['VM3_lateral_spread_m'], COLORS[k]) for k in keys] + \
        [('5 モード合わせ', var['pooled_5modes']['VM3_lateral_spread_m'], PURPLE)]
    for i, (lab, v, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, 100 * v, color=col, height=.62)
        ax.text(100 * v + .12, y, '%.2f cm(%.1f 倍)' % (100 * v, v / t['VM3_lateral_spread_m']), va='center', fontsize=13, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=13)
    ax.set_xlim(0, 15.5)
    ax.set_title('線のばらつき(走った線の横ずれの広がり)', fontsize=15, color=INK, loc='left')
    ax.set_xlabel('cm', fontsize=12, color=INK2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    screen(fig, R3 / 'fig/promote_record_trajectories.png', [0.47, 0.31, 0.51, 0.52], '記録した走り(上から見た図)')
    p = var['pooled_5modes']
    ip = indep['pooled_5modes']
    box(fig, 0.47, 0.13, 0.51, 0.16, PURPLE, '#f3f1fa', [
        ('訪れた状態の数 %.1f 倍・開始ゲート 1 か所 → %d か所' % (p['ratio_vs_teacher']['VM7'], p['start_gates_distinct']), 13.5, False, INK),
        ('審査の別の数え方でも同じ順(5 モード合わせ 線 %.1f cm・状態 %.1f 倍)' % (ip['lateral_spread_own_cm'], ip['state_cells_ratio_vs_teacher']), 13.5, False, INK)])
    banner(fig, '線のばらつきは 5 モード合わせで %.1f 倍。いちばん効くのは経路いろいろ(%.1f 倍)' % (
        p['ratio_vs_teacher']['VM3'], var['modes']['line']['ratio_vs_teacher']['VM3']))
    fig.savefig(OUT / 'r4_v3_variation.png', facecolor=SURFACE)
    plt.close(fig)


def main():
    jp_font()
    OUT.mkdir(parents=True, exist_ok=True)
    keys = ['fastest', 'no_wall', 'recover', 'speed', 'line', 'obstacle']
    cards = {k: rj(REG / k / 'card.json') for k in keys}
    evs = {k: rj(REG / k / 'eval.json') for k in keys}
    before = {k: rj(BEFORE / k / 'eval.json') for k in ('no_wall', 'recover', 'line')}
    ck2 = rj(REP / 'check_no_wall_r2/summary.json')
    ck4 = rj(R3 / 'check_no_wall_r4/summary.json')
    hard = {'n': 0, 'coll': 0}
    for v in ck4['hard_corner_by']['steerRate'].values():
        hard['n'] += v['n']
        hard['coll'] += v['coll']
    st_nw = rj(REG / 'no_wall/status.json')
    st_ln = rj(REG / 'line/status.json')
    st_rc = rj(REG / 'recover/status.json')
    ck = {'nw_r2_K0': ck2['ABX']['K0_corner'], 'nw_r4_hard': hard,
          'nw_policy_only_cm': st_nw['check']['numbers']['policy_only_min_cm'],
          'line_noshield_adv': '%s 回' % st_ln['check']['numbers']['noshield_adv_collisions'],
          'check_runs_total': st_nw['check']['runs_new'] + st_rc['check']['runs_new'] + st_ln['check']['runs_new']}
    var = rj(R3 / 'variation.json')
    indep = rj(R3 / 'review/indep_variation_r4.json')
    v1_modes(cards)
    v2_results(cards, evs, before, ck)
    v3_variation(var, indep)
    print(sorted(x.name for x in OUT.glob('r4_v*.png')), ck)


if __name__ == '__main__':
    main()
