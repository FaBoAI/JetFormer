#!/usr/bin/env python3
"""10月24日のレポート(実車ベンチマーク 2 日目・約 3 分)の動画の図 13 枚(2160 × 1350、CPU だけ・読むだけ)

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- nice -n 10 \\
        /usr/bin/python3 -I make_slides_1024.py [p1 k1 ...]

様式と部品(slide・card・banner・show・rbox・style_ax・tint・RawCam)は
day8_explainer_20261017/video/src/make_slides_day8.py をそのまま読み込んで使う(元のファイルは変えない)。
10/23 の real_bench_20261023/video/src/make_slides_bench.py と同じ形。
タイトル・今日のまとめ・課題の場面は図を使わない(描画側 make_digest.py が title と summary_items を描く)。

  p1_prev       前回(10/23)の課題と、今日の答え(10/23 の 3 分版の動画の最後の場面の本物の 1 コマ)
  k1_issue      今日の課題(10/23 の B の 1 周と、8 秒台・7 秒台の目標)
  k2_plan       対策: 1 回目は比べる走り、2・3 回目は速さのチューニング(走行タブの本物の画面の写真)
  k3_res1       結果: 1 回目(遅く入る型・2本の平均・判定は B のまま)
  k4_full       結果: 満充電で速すぎ、最終コーナー(SW の出口の出っ張り)に当たる(実車のカメラ cam_sw_exit)
  k5_best       結果: ベストは 9 番(No.9)。11 周の並びと実車のカメラ cam_no9_straight
  k6_r1         結果: R1 の入口の板の上下の衝撃(9 番・12 番の実車のカメラ)
  k7_de         結果: 3 回目は DE(cam_de_hit・cam_de_clean)
  k8_table      課題・対策・結果の表
  k9_cond       条件の違い(後輪の修理・電池・明るさ)
  k10_tt        3 周のタイムトライアルと予選の目標(25 秒を安定)
  k11_patterns  うまくいきそうな 3 つのパターン(課題と直し方の表)
  k12_todo      次の実車までにやること(25 秒の安定と上積み)

数の出典(読むだけ): bench_20261024_report/README.md・qualify_plan.md・fixes.md・tt_data.json・
cond_data/cond_summary.json・fix_data/r1_bump.json・experiments/20261024_report/col/col_summary.json・
bench_20261024_s1/session1_summary.md。
決まり: ToF は「STEVAL-VL53L9」と書く。時刻は書かない。予定の日付・曜日は書かない(「次の実車」と書く)。
方策の呼び名は TransFuser軽量型(モデル B と同じ形)など。カメラのコマは、人が写っていないと目で確かめたものだけ。
"""
import importlib.util
import json
import sys

sys.dont_write_bytecode = True                  # 読み込む day8 などのフォルダに __pycache__ を作らない
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402,F401
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

VIDEO = Path('/home/jetson/jetformer_reports/bench_20261024_report/video')   # src は experiments へのリンクなので絶対パス
OUT = VIDEO / 'fig'
D8SRC = Path('/home/jetson/jetformer_reports/day8_explainer_20261017/video/src/make_slides_day8.py')
_spec = importlib.util.spec_from_file_location('slides_day8', str(D8SRC))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
H.OUT = OUT                                     # save() は H.OUT に書く(day8 の fig には書かない)

slide, save, rbox, card, banner, show = H.slide, H.save, H.rbox, H.card, H.banner, H.show
tint, style_ax, RawCam = H.tint, H.style_ax, H.RawCam
SURFACE, INK, INK2, GRID = H.SURFACE, H.INK, H.INK2, H.GRID
BLUE, ORANGE, GREEN, RED, GRAY = H.BLUE, H.ORANGE, H.GREEN, H.RED, H.GRAY
PURPLE, GOLD = H.PURPLE, H.GOLD
W_IN, H_IN = H.W_IN, H.H_IN

REP = Path('/home/jetson/jetformer_reports')
BR = REP / 'bench_20261024_report'
FIG = BR / 'fig'
TT = json.load(open(BR / 'tt_data.json', encoding='utf-8'))
COND = json.load(open(BR / 'cond_data' / 'cond_summary.json', encoding='utf-8'))
BUMP = json.load(open(BR / 'fix_data' / 'r1_bump.json', encoding='utf-8'))
COL = json.load(open('/home/jetson/jetformer/experiments/20261024_report/col/col_summary.json', encoding='utf-8'))
PREV_VIDEO = REP / 'real_bench_20261023' / 'JetFormer_10月23日のレポート_実車ベンチマーク1日目_3分版.mp4'
PREV_FRAME = VIDEO / 'frames' / 'prev_issue_20261023.png'   # 上の動画の 174 秒(最後の「課題」の場面)の 1 コマ
UI_SHOT = REP / 'bench_20261024_s1' / 'ui_check' / 'ui_tune_block5.png'   # 走行タブの本物の画面の写真(10/24)
CAM = {k: FIG / ('cam_%s.png' % k) for k in ('sw_exit', 'no9_straight', 'de_hit', 'de_clean')}  # 人が写っていないコマ
LOG9, LOG12 = '20261024-162033', '20261024-183437'                   # 9 番(B 0.90・2 回目)・12 番(B 0.90・3 回目)
R1_FRAMES = ((LOG9, 39.37, '9 番(2 回目・速く入る)'), (LOG12, 30.11, '12 番(3 回目・遅めに入る)'))  # R1 の起点の 0.30 秒あと


def tbox(fig, x, y, w, h, edge, face, lw=2.4):
    """角の丸い枠(図の座標 0〜1)"""
    ax = fig.add_axes([x, y, w, h])
    ax.axis('off')
    asp = (w * W_IN) / (h * H_IN)
    ax.add_patch(FancyBboxPatch((0.004, 0.02), 0.992, 0.96, boxstyle='round,pad=0.0,rounding_size=0.025',
                                transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=lw, mutation_aspect=asp))
    return ax


def table(fig, x, y, w, h, cols, rows, head_face=INK, row_faces=None, size=12.5, hsize=13.5, first_bold=True,
          colors=None):
    """表: cols = [(見出し, 幅の割合)]、rows = [[文字…]](\\n で改行)。colors[r][c] で文字の色を変える"""
    ax = fig.add_axes([x, y, w, h])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    n = len(rows)
    hh = 0.11
    rh = (1 - hh) / n
    xs = np.cumsum([0] + [c[1] for c in cols])
    ax.add_patch(FancyBboxPatch((0, 1 - hh), 1, hh, boxstyle='square,pad=0', fc=head_face, ec='none'))
    for (t, _), x0, x1 in zip(cols, xs[:-1], xs[1:]):
        ax.text((x0 + x1) / 2, 1 - hh / 2, t, ha='center', va='center', fontsize=hsize, fontweight='bold', color='white')
    for r, row in enumerate(rows):
        y1 = 1 - hh - r * rh
        face = (row_faces[r] if row_faces else ('#ffffff' if r % 2 == 0 else '#f3f2ee'))
        ax.add_patch(FancyBboxPatch((0, y1 - rh), 1, rh, boxstyle='square,pad=0', fc=face, ec=GRID, lw=1.2))
        for c, (txt, x0) in enumerate(zip(row, xs[:-1])):
            col = (colors[r][c] if colors and colors[r][c] else INK)
            ax.text(x0 + 0.012, y1 - rh / 2, txt, ha='left', va='center', fontsize=size, color=col, linespacing=1.35,
                    fontweight='bold' if (c == 0 and first_bold) else 'normal')
    for x0 in xs[1:-1]:
        ax.plot([x0, x0], [0, 1 - hh], color=GRID, lw=1.2)
    return ax


def cam(name):
    return Image.open(CAM[name]).convert('RGB')


def laps(log):
    return [lp for lp in TT['laps'] if lp['log'] == log]


def lap_sum(setting, session):
    return next(s for s in TT['lap_summary'] if s['setting'] == setting and s['session'] == session)


# ---------------------------------------------------------------- p1 前回の課題と今日の答え
def p1_prev():
    fig = slide('前回の課題と、今日の答え',
                '前回(10/23)の動画の最後の「課題」の 5 つに、10/24 の実車でどう答えたか')
    fig.text(0.025, 0.80, '前回(10/23)の動画の最後の場面', fontsize=14, fontweight='bold', color=INK2, va='center')
    im = Image.open(PREV_FRAME).convert('RGB').crop((60, 140, 1420, 590))      # 上の帯(説明欄と時刻)は切り取る
    show(fig, im, [0.02, 0.50, 0.44, 0.44 * W_IN / H_IN * im.height / im.width], frame=True)
    card(fig, 0.02, 0.13, 0.44, 0.33, BLUE, '今日(10/24)走ったもの',
         ['AI 走行 20 本(3 回)・AI の時間 895 秒', '1 回目: 比べる走り(B・遅く入る型・2本の平均)',
          '2・3 回目: 速さのチューニング(速さ 0.85〜1.10)', 'モデル B = TransFuser軽量型(推奨のまま)'],
         face=tint(BLUE, 0.04), hsize=15, size=13.2, top=0.84, step=0.185)
    rows = [('○', GREEN, '発進の補助を実車で確かめる', '補助が出た発進 25 回(2 日で)。動き出しは中央 0.20 秒'),
            ('△', ORANGE, 'DE の揺れを抑える', '揺れの表を夜の集計に足した。3 回目に DE が 25 件'),
            ('×', RED, '1 ブロック 240 秒を走り切る', '1 本 3〜65 秒。比べる走りは 1 回目で切り上げ'),
            ('×', RED, '電池の番号と電圧を書く', 'メモ欄を足したが、20 本とも空のまま'),
            ('○', GREEN, '2 日目の並び', '1 回目は 10/23 と同じ並びで周を足した')]
    ax = tbox(fig, 0.50, 0.13, 0.48, 0.70, INK, '#ffffff')
    ax.text(0.04, 0.935, '前回の課題 → 今日の答え', fontsize=16, fontweight='bold', color=INK, transform=ax.transAxes,
            va='center')
    for i, (mk, col, head, ans) in enumerate(rows):
        yy = 0.80 - i * 0.165
        ax.text(0.05, yy, mk, fontsize=24, fontweight='bold', color=col, transform=ax.transAxes, va='center', ha='center')
        ax.text(0.11, yy + 0.035, head, fontsize=14.5, fontweight='bold', color=INK, transform=ax.transAxes, va='center')
        ax.text(0.11, yy - 0.04, ans, fontsize=13.2, color=INK2, transform=ax.transAxes, va='center')
    banner(fig, '発進の補助は効いた。DE・240 秒・電圧の記録は、まだ残っている', size=18.5)
    save(fig, 'p1_prev')


# ---------------------------------------------------------------- k1 今日の課題
def k1_issue():
    b23 = lap_sum('B 速さ1.00 ステア1.00', '10/23 公式')
    fig = slide('今日の課題: B より良いモデル探しと、B をもっと速く',
                '10/23 の公式の B は 1 周 約 9.4 秒。目標(ユーザーの依頼)は 1 周 8 秒台、あわよくば 7 秒台')
    ax = fig.add_axes([0.15, 0.20, 0.36, 0.56])
    ax.axvspan(7.0, 8.0, color=PURPLE, alpha=0.10, zorder=1)
    ax.axvspan(8.0, 9.0, color=GREEN, alpha=0.12, zorder=1)
    ax.text(7.5, 2.55, '7 秒台\n(あわよくば)', ha='center', va='center', fontsize=14, color=PURPLE, fontweight='bold')
    ax.text(8.5, 2.55, '8 秒台\n(目標)', ha='center', va='center', fontsize=14, color=GREEN, fontweight='bold')
    vals = [('10/23 公式 B\n中央値', b23['median_clean'], BLUE),
            ('10/23 公式 B\nベスト', b23['best_clean'], BLUE)]
    for y, (lab, v, col) in zip((1.4, 0.4), vals):
        ax.barh(y, v - 6.8, left=6.8, height=0.5, color=col, zorder=3)
        ax.text(v + 0.05, y, '%.2f 秒' % v, va='center', fontsize=15, fontweight='bold', color=INK)
    ax.set_yticks([1.4, 0.4])
    ax.set_yticklabels([v[0] for v in vals], fontsize=13)
    ax.set_xlim(6.8, 10.3)
    ax.set_ylim(-0.2, 3.1)
    ax.set_xlabel('1 周の時間(秒、きれいな周)', fontsize=13, color=INK2)
    style_ax(ax, 12.5)
    card(fig, 0.56, 0.58, 0.42, 0.21, BLUE, '課題 1: B より良いモデルはあるか',
         ['10/23 は候補が 1〜2 周しか走れず、上位 2 本は「なし」', '→ 同じ並びで走り、周を足す'],
         face=tint(BLUE, 0.04), hsize=15, size=13.2, top=0.78, step=0.27)
    card(fig, 0.56, 0.35, 0.42, 0.21, GREEN, '課題 2: B をもっと速く',
         ['ベストラップ 8 秒前半をねらう。あわよくば 7 秒台', '→ ぶつからない速さの条件を探す'],
         face=tint(GREEN, 0.05), hsize=15, size=13.2, top=0.78, step=0.27)
    card(fig, 0.56, 0.12, 0.42, 0.21, ORANGE, '気をつけること(10/23 の試し)',
         ['ゲイン 1.2〜1.4 では、衝突が 100 m あたり約 5 倍', 'DE(GM の出口)で当たりやすい'],
         face=tint(ORANGE, 0.04), hsize=15, size=13.2, top=0.78, step=0.27)
    banner(fig, '比べる走りで B を確かめ、そのあと「ぶつからずに速い」速さを探す', size=18.5)
    save(fig, 'k1_issue')


# ---------------------------------------------------------------- k2 対策
def k2_plan():
    fig = slide('対策: 1 回目は比べる走り、2・3 回目は速さのチューニング',
                '走行タブの「ベンチマーク」で、ブロックごとに速さ(スロットルのゲイン)を選べるようにした')
    shot = Image.open(UI_SHOT).convert('RGB')
    left = shot.crop((20, 70, 400, 560))
    fig.text(0.025, 0.80, '走行タブ(本物の画面)', fontsize=14, fontweight='bold', color=INK2, va='center')
    hgt = 0.56
    wid = hgt * H_IN / W_IN * left.width / left.height
    show(fig, left, [0.025, 0.20, wid, hgt], frame=True)
    toast = shot.crop((420, 618, 935, 666))
    tw = 0.30
    show(fig, toast, [0.025, 0.125, tw, tw * W_IN / H_IN * toast.height / toast.width], frame=False)
    x0 = 0.025 + wid + 0.03
    w = 0.975 - x0
    card(fig, x0, 0.59, w, 0.20, BLUE, '1 回目: 比べる走り(4 本)',
         ['B・遅く入る型・2本の平均・B を、同じ速さ 1.00 で', '10/23 と同じ並びで周を足し、B より良いかを見る'],
         face=tint(BLUE, 0.04), hsize=15, size=13.2, top=0.78, step=0.27)
    card(fig, x0, 0.36, w, 0.21, ORANGE, '2 回目: 速さのチューニング(満充電)',
         ['計画は速さ 1.10〜1.50 の段(遅く入る型と B)', '1.10 から強い衝突 → 満充電用の段 0.85〜0.95 を足した'],
         face=tint(ORANGE, 0.04), hsize=15, size=13.2, top=0.78, step=0.27)
    card(fig, x0, 0.13, w, 0.21, GREEN, '3 回目: 8 秒台の安定と 7 秒台の挑戦',
         ['B 0.90 のくり返し → 0.85・ステア 1.10 → 0.95・1.00', '区間の速度上限は OFF のまま(G1 が不合格)'],
         face=tint(GREEN, 0.05), hsize=15, size=13.2, top=0.78, step=0.27)
    banner(fig, '画面の「基準と違う: 速度 1.50」のように、変えた値はその場で見える', size=18.5)
    save(fig, 'k2_plan')


# ---------------------------------------------------------------- k3 結果: 1 回目
def k3_res1():
    b = lap_sum('B 速さ1.00 ステア1.00', '10/24 1回目')
    s = lap_sum('遅く入る型 速さ1.00 ステア1.00', '10/24 1回目')
    fig = slide('結果: 1 回目は「B のまま」',
                '1 回目(No.1〜4、速さ 1.00)。遅く入る型は R1 を緩めたが、ラップは B とほぼ同じ。2本の平均は走れない')
    ax = fig.add_axes([0.15, 0.47, 0.30, 0.28])
    rows = [('B(1 番)', 5016, BLUE), ('遅く入る型(2 番)', 4718, GREEN)]
    for y, (lab, v, col) in zip((1, 0), rows):
        ax.barh(y, v, color=col, height=0.55, zorder=3)
        ax.text(v + 60, y, format(v, ','), va='center', fontsize=15, fontweight='bold', color=INK)
    ax.set_yticks([1, 0])
    ax.set_yticklabels([r[0] for r in rows], fontsize=13.5)
    ax.set_xlim(0, 6300)
    ax.set_title('R1 の進入の速さ(eRPM の中央値)', fontsize=14.5, color=INK, fontweight='bold', loc='left')
    style_ax(ax, 12)
    ax2 = fig.add_axes([0.15, 0.15, 0.30, 0.24])
    rows2 = [('B(1 回目)', b['median_clean'], BLUE), ('遅く入る型', s['median_clean'], GREEN)]
    for y, (lab, v, col) in zip((1, 0), rows2):
        ax2.barh(y, v, color=col, height=0.55, zorder=3)
        ax2.text(v + 0.1, y, '%.2f 秒' % v, va='center', fontsize=15, fontweight='bold', color=INK)
    ax2.set_yticks([1, 0])
    ax2.set_yticklabels([r[0] for r in rows2], fontsize=13.5)
    ax2.set_xlim(0, 12.2)
    ax2.set_title('きれいな周の中央値(秒)', fontsize=14.5, color=INK, fontweight='bold', loc='left')
    style_ax(ax2, 12)
    card(fig, 0.50, 0.55, 0.48, 0.24, GREEN, '遅く入る型: 狙いどおりだが、速くならない',
         ['R1 の進入 約 6% 遅く(4,718 対 5,016)・衝突 0',
          'ラップは %.2f 秒(B は %.2f 秒)' % (s['median_clean'], b['median_clean']), 'ただし 1 本・5 通過だけ'],
         face=tint(GREEN, 0.05), hsize=15, size=13.2, top=0.80, step=0.21)
    card(fig, 0.50, 0.30, 0.48, 0.23, RED, '2本の平均: 走れない',
         ['発進の補助で一度動き出したが、', 'モデルの出力が中立〜わずかなブレーキになり止まった', '(10/23 の夜は走れた。昼の明るさの違いの疑い)'],
         face=tint(RED, 0.04), hsize=15, size=13.2, top=0.80, step=0.21)
    card(fig, 0.50, 0.12, 0.48, 0.16, INK, '事前登録(protocol)の判定',
         ['「B のまま」。1 回目は衝突/通過 0.14・ベスト 9.58 秒'],
         hcolor=INK, hsize=15, size=13.5, top=0.72, step=0.36)
    banner(fig, 'モデルは B のまま。ここから速さのチューニングへ切り替えた', size=18.5)
    save(fig, 'k3_res1')


# ---------------------------------------------------------------- k4 結果: 満充電で速すぎ
def k4_full():
    fig = slide('結果: 満充電で速すぎ、最終コーナーを曲がりきれない',
                '2 回目(満充電 8.4 V、ユーザーの話)。速さ 1.0〜1.1 では直線が 2〜3 割速くなり、SW の出口の出っ張りに当たる')
    show(fig, cam('sw_exit'), [0.025, 0.36, 0.27, 0.27 * W_IN / H_IN], '最終コーナーの出口(駐車場の出っ張り)',
         color=INK2, size=13)
    fig.text(0.025, 0.33, '9 番が当たらずに通った同じ場所(実車のカメラ)', fontsize=12, color=INK2, va='center')
    bands = [('3.5 未満', 0, 47), ('3.5〜4.0', 2, 34), ('4.0〜4.5', 2, 11), ('4.5 以上', 4, 6)]
    ax = fig.add_axes([0.37, 0.20, 0.27, 0.53])
    for i, (lab, k, n) in enumerate(bands):
        v = 100.0 * k / n
        ax.bar(i, v, color=RED if v > 50 else (ORANGE if v > 0 else GRAY), width=0.62, zorder=3)
        ax.text(i, v + 2, '%d / %d' % (k, n), ha='center', fontsize=14, fontweight='bold', color=INK)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b[0] for b in bands], fontsize=12.5)
    ax.set_ylim(0, 80)
    ax.set_xlabel('SW の中の最高の速さ(m/s)', fontsize=12.5, color=INK2)
    ax.set_ylabel('出っ張りに当たった通過の割合(%)', fontsize=12.5, color=INK2)
    style_ax(ax, 12)
    card(fig, 0.68, 0.48, 0.30, 0.31, ORANGE, '満充電で速すぎ',
         ['直線(eRPM 90% 点)', '1 回目 6,300〜6,950', '2 回目 1.0〜1.1 で 8,350〜9,390', '出っ張りに 9 件・全部 4G 以上'],
         face=tint(ORANGE, 0.04), hsize=15, size=13.2, top=0.84, step=0.185)
    card(fig, 0.68, 0.13, 0.30, 0.33, RED, '後輪が壊れた',
         ['9 件のうち 7 件は遅く入る型 1.0〜1.1', '10 番(B 1.00)で強い衝突 3 件、', '後輪が壊れて修理した',
          '(壊れた場所は記録では分からない)'],
         face=tint(RED, 0.04), hsize=15, size=13.2, top=0.84, step=0.185)
    banner(fig, '同じゲインでも電池で速さが変わる。満充電なら速さ 0.90 が合っていた', size=18.5)
    save(fig, 'k4_full')


# ---------------------------------------------------------------- k5 結果: ベストは 9 番
def k5_best():
    lp = laps(LOG9)
    s9 = lap_sum('B 速さ0.90 ステア1.00', '10/24 2回目')
    b23 = lap_sum('B 速さ1.00 ステア1.00', '10/23 公式')
    fig = slide('結果: ベストは 9 番。1 周 8.25 秒',
                '9 番 = B・速さ 0.90・ステア 1.00(2 回目・満充電・後輪の修理の前)。AI で続けて 11 周')
    ax = fig.add_axes([0.07, 0.17, 0.52, 0.60])
    xs = np.arange(1, len(lp) + 1)
    ys = [p['lap_s'] for p in lp]
    ax.plot(xs, ys, color=BLUE, lw=2.4, zorder=3)
    for x, p in zip(xs, lp):
        clean = p['clean']
        ax.scatter([x], [p['lap_s']], s=170 if clean else 210, marker='o' if clean else '^',
                   color=BLUE if clean else ORANGE, zorder=4, edgecolor='white', lw=1.5)
        ax.text(x, p['lap_s'] + 0.07, '%.2f' % p['lap_s'], ha='center', fontsize=12.5, fontweight='bold', color=INK)
    for v, lab, ls in ((9.0, '9.00 秒 = 3 周 27 秒', ':'), (8.333, '8.33 秒 = 3 周 25 秒', '-'),
                       (7.667, '7.67 秒 = 3 周 23 秒', '--')):
        ax.axhline(v, color=INK2, lw=1.4, ls=ls, zorder=2)
        ax.text(11.45, v, lab, va='center', ha='left', fontsize=12, color=INK2)
    ax.set_xlim(0.5, 11.4)
    ax.set_ylim(7.4, 9.35)
    ax.set_xticks(xs)
    ax.set_xlabel('周の順番', fontsize=12.5, color=INK2)
    ax.set_ylabel('1 周(秒)', fontsize=12.5, color=INK2)
    ax.scatter([], [], s=150, color=BLUE, label='きれいな周(%d 周)' % s9['clean'])
    ax.scatter([], [], s=150, marker='^', color=ORANGE, label='軽い接触の周(%d 周)' % (s9['laps'] - s9['clean']))
    ax.legend(fontsize=12.5, loc='lower left', frameon=False, ncol=2)
    style_ax(ax, 12)
    show(fig, cam('no9_straight'), [0.735, 0.47, 0.21, 0.21 * W_IN / H_IN], 'ベストの周の直線(4.2 m/s)',
         color=INK2, size=12.5)
    card(fig, 0.70, 0.13, 0.28, 0.31, BLUE, '9 番の数',
         ['ベスト %.2f 秒・中央値 %.2f 秒' % (s9['best_clean'], s9['median_clean']),
          '3 周: 最速 25.29 秒(軽い接触)', 'きれいな 3 周の最速 26.11 秒', '3 周 9 回とも 27 秒以下'],
         face=tint(BLUE, 0.04), hsize=15, size=13.2, top=0.84, step=0.185)
    banner(fig, '10/23 の B(中央値 %.2f 秒)より 1 周 約 %.1f 秒速い。8 秒台は出た'
           % (b23['median_clean'], b23['median_clean'] - s9['median_clean']), size=18.5)
    save(fig, 'k5_best')


# ---------------------------------------------------------------- k6 結果: R1 の入口の板
def k6_r1():
    no9 = [p for p in BUMP['passes'] if p['log'] == LOG9]
    n9 = sum(p['bump'] for p in no9)
    a9 = [p['a3_max'] for p in no9 if p['bump']]
    med9 = float(np.median([p['a3_max'] for p in no9]))      # README の「中央 3.9G」は 12 通過の中央
    fig = slide('結果: 9 番は、R1 の入口の板で上下の衝撃',
                'R1 の起点から 1.2 秒の 3 軸の加速度(大きさ − 1G)。水平 2.5G の決まりでは「きれい」に入るが、上下に強く揺れている')
    for k, (lg, t, lab) in enumerate(R1_FRAMES):
        im, _ = RawCam(lg).at(t)
        show(fig, im, [0.025 + k * 0.165, 0.47, 0.155, 0.155 * W_IN / H_IN], lab, color=INK2, size=12.5)
    fig.text(0.025, 0.415, 'どちらも R1 の入口の床の黒い板の縁を\n越えるところ(実車のカメラ、起点の 0.30 秒あと)', fontsize=12.5,
             color=INK2, va='center', linespacing=1.5)
    ax = fig.add_axes([0.44, 0.20, 0.25, 0.55])
    labs = ['5,000\n未満', '5,000〜\n5,500', '5,500〜\n6,000', '6,000\n以上']
    for i, (bd, lab) in enumerate(zip(BUMP['bands'], labs)):
        v = 100.0 * bd['bump'] / bd['passes']
        ax.bar(i, v, color=[GRAY, ORANGE, RED, RED][i], width=0.62, zorder=3)
        ax.text(i, v + 2, '%d%%' % round(v), ha='center', fontsize=15, fontweight='bold', color=INK)
        ax.text(i, 3, '%d/%d' % (bd['bump'], bd['passes']), ha='center', fontsize=11.5, color='white', fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(labs, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xlabel('R1 の進入の速さ(eRPM)', fontsize=12.5, color=INK2)
    ax.set_ylabel('3 軸 2.5G 以上の通過(%)', fontsize=12.5, color=INK2)
    ax.set_title('速く入るほど多い(121 通過)', fontsize=14, fontweight='bold', color=INK, loc='left')
    style_ax(ax, 12)
    card(fig, 0.72, 0.48, 0.26, 0.31, RED, '9 番',
         ['%d 通過中 %d で 2.5G 以上' % (len(no9), n9), '%.1f〜%.1fG・中央 %.1fG' % (min(a9), max(a9), med9),
          '主に上下の揺れ', '3 回目の B 0.90 は 6/19'],
         face=tint(RED, 0.04), hsize=15, size=13.2, top=0.84, step=0.185)
    card(fig, 0.72, 0.13, 0.26, 0.33, PURPLE, '止める条件の案',
         ['R1 の 3 軸 7G 以上が 1 回', 'または 1 本の中央 4.5G 以上', '→ 止めて後輪を見る', '→ 次はゲイン −0.05'],
         face=tint(PURPLE, 0.04), hsize=15, size=13.2, top=0.84, step=0.185)
    banner(fig, '速い周ほど板の衝撃も大きい。次の実車では 3 軸で数え、板の縁も目で見る', size=18.5)
    save(fig, 'k6_r1')


# ---------------------------------------------------------------- k7 結果: 3 回目は DE
def k7_de():
    s3 = lap_sum('B 速さ0.90 ステア1.00', '10/24 3回目')
    de = {k: COL['by_place'][k]['DE'] for k in ('1', '2', '3')}
    tot3 = sum(COL['by_place']['3'].values())
    fig = slide('結果: 3 回目は同じ設定なのに遅く、DE で当たる',
                '3 回目(後輪の修理のあと)。B 0.90 で 1 周 %.2f〜%.2f 秒(9 番は 8.25〜8.96)。衝突 %d 件中 %d 件が DE'
                % (s3['best_clean'], s3['median_clean'], tot3, de['3']))
    show(fig, cam('de_hit'), [0.025, 0.40, 0.20, 0.20 * W_IN / H_IN], '3 回目 12 番: 当たる 0.27 秒前', color=RED, size=12.5)
    show(fig, cam('de_clean'), [0.245, 0.40, 0.20, 0.20 * W_IN / H_IN], '9 番: 同じ場所', color=BLUE, size=12.5)
    fig.text(0.025, 0.365, '右の白い壁のすぐ横(12 番)と、道のまん中(9 番)。実車のカメラ', fontsize=12.5, color=INK2,
             va='center')
    ax = fig.add_axes([0.53, 0.47, 0.20, 0.29])
    for i, (lab, k, n, col) in enumerate((('9 番', 1, 12, BLUE), ('3 回目\nB 0.90', 11, 19, RED))):
        v = 100.0 * k / n
        ax.bar(i, v, color=col, width=0.6, zorder=3)
        ax.text(i, v + 2.5, '%d / %d' % (k, n), ha='center', fontsize=15, fontweight='bold', color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['9 番', '3 回目\nB 0.90'], fontsize=12.5)
    ax.set_ylim(0, 75)
    ax.set_title('DE で当たった通過(%)', fontsize=14, fontweight='bold', color=INK, loc='left')
    style_ax(ax, 12)
    ax2 = fig.add_axes([0.78, 0.47, 0.19, 0.29])
    for i, k in enumerate(('1', '2', '3')):
        ax2.bar(i, de[k], color=[BLUE, ORANGE, GREEN][i], width=0.6, zorder=3)
        ax2.text(i, de[k] + 0.6, '%d' % de[k], ha='center', fontsize=15, fontweight='bold', color=INK)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['1 回目', '2 回目', '3 回目'], fontsize=12.5)
    ax2.set_ylim(0, 30)
    ax2.set_title('DE の衝突の数', fontsize=14, fontweight='bold', color=INK, loc='left')
    style_ax(ax2, 12)
    card(fig, 0.50, 0.13, 0.48, 0.27, RED, '3 回目の試し',
         ['DE の衝突: 当たる前 約 2.9 m/s、右の白い壁に正面から', 'ステア 1.10: 切り返しが大きくなり悪化(3 周 29.4 秒台)',
          '速さ 0.95: ベスト 8.66 秒だが、DE で 8 通過中 4', '最後の 3 本(18〜20 番): きれいな周 0'],
         face=tint(RED, 0.04), hsize=15, size=13.2, top=0.84, step=0.185)
    banner(fig, '同じ設定が再現しない。DE の当たり方も、9 番とは違う走りになっていた', size=18.5)
    save(fig, 'k7_de')


# ---------------------------------------------------------------- k8 課題・対策・結果
def k8_table():
    fig = slide('課題・対策・結果', '10/24 の実車(AI 走行 20 本)')
    cols = [('課題', 0.22), ('対策', 0.30), ('結果', 0.40), ('', 0.08)]
    rows = [['B より良いモデル', '1 回目に比べる走り\n(B・遅く入る型・2本の平均)', '遅く入る型は衝突 0 だがラップは同じ。\n2本の平均は止まる → 判定は「B のまま」', '△'],
            ['B をもっと速く\n(8 秒台)', '速さの段 0.85〜1.10\n満充電用の段を足した', '9 番(速さ 0.90)でベスト 8.25 秒・\n中央値 8.53 秒。3 周 25.29 秒', '○'],
            ['ぶつからない\n速さの条件', '段ごとに走って確かめる', '満充電で 1.0 以上は最終コーナーに 9 件。\n9 番は R1 の板で毎周 上下の衝撃', '△'],
            ['同じ設定の再現', '3 回目に B 0.90 をくり返す', '1 周 9.3 秒台・DE に 25 件。\n後輪の修理のあとの左右差を疑う', '×'],
            ['7 秒台', '速さ 0.95・1.00 で挑戦', '出ない(0.95 の 8.66 秒が最速)', '×']]
    marks = {'○': GREEN, '△': ORANGE, '×': RED}
    colors = [[None, None, None, marks[r[3]]] for r in rows]
    ax = table(fig, 0.035, 0.15, 0.93, 0.68, cols, rows, size=14, hsize=15, colors=colors)
    for t in ax.texts:
        if t.get_text() in marks:
            t.set_fontsize(26)
            t.set_fontweight('bold')
    banner(fig, '8 秒台は出た。でも同じ設定が再現せず、7 秒台はまだ', size=18.5)
    save(fig, 'k8_table')


# ---------------------------------------------------------------- k9 条件の違い
def k9_cond():
    tg = COND['turn_gain_1p5_2p5']
    sess = {n: (1 if n <= 4 else (2 if n <= 10 else 3)) for n in range(1, 21)}
    fig = slide('条件の違い: 一番の疑いは、後輪の修理のあとの曲がり方',
                '同じ B 0.90・ステア 1.00 が 3 回目に再現しなかった理由を、9 番と 11・12・18 番でくらべた')
    ax = fig.add_axes([0.07, 0.17, 0.45, 0.58])
    for k, v in sorted(tg.items(), key=lambda kv: int(kv[0])):
        n = int(k)
        col = [BLUE, ORANGE, GREEN][sess[n] - 1]
        big = n in (9, 11, 12, 18)
        ax.scatter([n], [v['ratio']], s=260 if big else 110, color=col, zorder=3, edgecolor='white', lw=1.5)
        if big:
            dx, dy, ha = {9: (0, 0.022, 'center'), 11: (-0.3, -0.03, 'right'), 12: (0.3, -0.035, 'left'),
                          18: (0, -0.03, 'center')}[n]
            ax.text(n + dx, v['ratio'] + dy, '%d 番 %.2f' % (n, v['ratio']), ha=ha, fontsize=12.5,
                    fontweight='bold', color=INK)
    ax.axvline(10.5, color=RED, lw=2, ls='--', zorder=2)
    ax.text(10.65, 1.03, '後輪の修理', fontsize=13, color=RED, fontweight='bold', va='center')
    ax.axhline(0.90, color=INK2, lw=1.2, ls=':')
    ax.text(18.8, 0.905, '確かめの目安 0.90', fontsize=11.5, color=INK2, va='bottom', ha='right')
    ax.set_xlim(0.3, 19)
    ax.set_ylim(0.62, 1.06)
    ax.set_xticks(range(1, 19))
    ax.set_xlabel('AI 走行の番号(青 1 回目・橙 2 回目・緑 3 回目)', fontsize=12.5, color=INK2)
    ax.set_ylabel('左右の旋回の効きの比(左 ÷ 右)', fontsize=12.5, color=INK2)
    style_ax(ax, 12)
    card(fig, 0.56, 0.50, 0.42, 0.29, RED, '1. 後輪の修理(確からしさ 中〜高)',
         ['左右の比: 修理の前 0.84〜0.96 → 後 0.69〜0.81', 'まっすぐ進む舵が +22 µs(右へ流れる)',
          'DE で当たる通過 12% → 47%', 'GM で減速 → モデルが踏まない → 1 周 +0.8 秒'],
         face=tint(RED, 0.04), hsize=15, size=13.2, top=0.84, step=0.185)
    card(fig, 0.56, 0.30, 0.42, 0.18, ORANGE, '2. 電池(中)',
         ['1 回目(昼)は同じ指令で 7〜15% 弱い', '9 番と 3 回目の差は小さい'],
         face=tint(ORANGE, 0.04), hsize=15, size=13.2, top=0.76, step=0.27)
    card(fig, 0.56, 0.12, 0.42, 0.16, GREEN, '3. 明るさ(効いていない)',
         ['画像を ±10% 変えても、スロットルは 1 µs 以内'],
         face=tint(GREEN, 0.05), hsize=15, size=13.2, top=0.74, step=0.38)
    banner(fig, '次の実車の最初の 1 本で、左右の比 0.90 以上・まっすぐの舵 ±10 µs を確かめる', size=18.5)
    save(fig, 'k9_cond')


# ---------------------------------------------------------------- k10 3 周のタイムトライアルと目標
def k10_tt():
    groups = [('10/23 公式', '10/23 公式'), ('10/24 1回目', '1 回目'), ('10/24 2回目', '2 回目(9 番)'),
              ('10/24 3回目', '3 回目')]
    kinds = [('きれい', BLUE, 'o', 'きれい'), ('軽い接触だけ', ORANGE, '^', '軽い接触だけ'), ('強い衝突あり', GREEN, 'X', '強い衝突あり')]
    fig = slide('3 周のタイムトライアル: 目標は 25 秒を安定して',
                '予選(ユーザーの話): 40 チーム中 12 チームが決勝。3 周のタイムトライアル。トップ 約 23 秒・目安 27 秒')
    ax = fig.add_axes([0.13, 0.20, 0.50, 0.55])
    rng = np.random.RandomState(1)
    for gi, (key, lab) in enumerate(groups):
        y = len(groups) - 1 - gi
        ws = [w for w in TT['windows'] if w['session'] == key]
        for kname, col, mk, _ in kinds:
            xs = [w['total_s'] for w in ws if w['kind'] == kname]
            ax.scatter(xs, y + rng.uniform(-0.2, 0.2, len(xs)), s=120, color=col, marker=mk, zorder=3,
                       edgecolor='white', lw=1)
        le27 = sum(w['total_s'] <= 27.0 for w in ws)
        ax.text(31.2, y, '%d 回中 %d 回\n27 秒以下' % (len(ws), le27), va='center', fontsize=12.5, color=INK, fontweight='bold')
    for v, lab, ls in ((23, 'トップ\n約 23 秒', '--'), (25, '目標\n25 秒', '-'), (27, '目安\n27 秒', ':')):
        ax.axvline(v, color=INK if v == 25 else INK2, lw=2.2 if v == 25 else 1.5, ls=ls, zorder=2)
        ax.text(v, 3.62, lab, ha='center', va='bottom', fontsize=13, fontweight='bold', color=INK if v == 25 else INK2)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[1] for g in groups][::-1], fontsize=13.5)
    ax.set_xlim(22.5, 31)
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlabel('続けて走った 3 周の合計(秒、走りながら計測)', fontsize=12.5, color=INK2)
    for _, col, mk, lab in kinds:
        ax.scatter([], [], s=110, color=col, marker=mk, label=lab)
    ax.legend(fontsize=12, loc='lower left', frameon=True, ncol=1)
    style_ax(ax, 12)
    card(fig, 0.72, 0.53, 0.26, 0.26, BLUE, '目標: 3 周 25 秒',
         ['1 周 8.33 秒を安定して', '止まってからなら 1 周 8.12 秒', '(発進で 約 0.6 秒失う)'],
         face=tint(BLUE, 0.04), hsize=15, size=13.2, top=0.80, step=0.21)
    card(fig, 0.72, 0.12, 0.26, 0.39, INK, '今日の 3 周(38 回)',
         ['25 秒以下: 0', '26 秒以下: 6(すべて 9 番)', '27 秒以下: 10(9 番 9・17 番 1)', '最速 25.29 秒(9 番)',
          '24.33 秒は周の数え違い(半周)'],
         hcolor=INK, hsize=15, size=13.2, top=0.87, step=0.148)
    banner(fig, '25 秒 = 9 番の速い周を毎周。スタートの仕方と接触の扱いは、まだ聞いていない', size=18.5)
    save(fig, 'k10_tt')


# ---------------------------------------------------------------- k11 うまくいきそうな 3 つのパターン
def k11_patterns():
    fig = slide('うまくいきそうな 3 つのパターン', '見込みは 3 周・走りながら・粗い。パターン 2・3 は実車の根拠がまだ無い(仮説)')
    cols = [('パターン', 0.15), ('中身', 0.25), ('3 周の見込み', 0.15), ('課題', 0.23), ('直し方', 0.22)]
    rows = [['1  9 番の再現\n(土台・床の候補)', 'B・ステア 1.0。電圧と最初の\n1 本で速さを合わせ、車の\n左右をそろえる。コードは同じ',
             '直せたら\n25.3〜26.7 秒\n直せなければ\n27.4〜29.0 秒', '同じ設定でも再現しない。\n1 本の中で遅くなる。\nR1 の板で毎周 3〜6G',
             '後輪・舵のリンクの点検。\n最初の 1 本でゲインを直す。\n満充電で 3 周だけ'],
            ['2  速さの天井と\n自動合わせ\n(25 秒の候補)', '天井: 全体 4.5 m/s・\n左に曲がる最中 4.0 m/s。\n直線の速さを 4.1 m/s に\n自動で合わせる',
             '仮説\n24.7〜25.5 秒', '速い 4 周からの仮説。\n閉ループの根拠が無い。\n速くすると R1 の衝撃が増える',
             '既定は OFF。記録の再生で\n値を決め、実車で A/B/A。\n実装と ON に了承が要る'],
            ['3  B-FAST\n(上積みの候補)', '2 回目の B の速い 10 周を\n手本に、B から学び直す\n(人の走行と混ぜる)', '24.9〜27.5 秒\n(幅が広い)',
             '手本が少ない。\nAI の癖も覚える。\n机上で良くても走りは別', '重み 1.0〜1.2。机上の門を\n全部通ったものだけ走る。\n学習と走行に了承が要る']]
    faces = [tint(BLUE, 0.06), tint(ORANGE, 0.07), tint(GREEN, 0.07)]
    colors = [[BLUE, None, BLUE, None, None], [ORANGE, None, ORANGE, None, None], [GREEN, None, GREEN, None, None]]
    table(fig, 0.025, 0.15, 0.95, 0.70, cols, rows, row_faces=faces, size=13.2, hsize=15, colors=colors)
    banner(fig, '3 つに共通の土台: 車の左右をそろえる・電圧を毎回測る・周の数え方を直す', size=18.5)
    save(fig, 'k11_patterns')


# ---------------------------------------------------------------- k12 次の実車までにやること
def k12_todo():
    fig = slide('次の実車までにやること: 25 秒の安定と上積み',
                '了承が要るもの(走行の制御・学習・集計のコード)は、了承をもらってから進める')
    card(fig, 0.025, 0.47, 0.465, 0.32, BLUE, '土台(パターン 1)',
         ['1. 車の左右をそろえる(後輪の取り付け・トー・舵のリンク)', '2. 電圧を毎回測って、説明欄に書く',
          '3. 走ったあとの確認の道具(3 周・速さ・左右の比・R1 の衝撃)', '4. 周の数え方の直し(偽の周を外す)  了承'],
         face=tint(BLUE, 0.04), hsize=16, size=13.5, top=0.84, step=0.185)
    card(fig, 0.51, 0.47, 0.465, 0.32, ORANGE, '25 秒のために(了承が要る)',
         ['5. 速さの天井(パターン 2)', '6. 速さのゲインの自動合わせ(パターン 2)', '7. B-FAST の学習と机上の門(パターン 3)',
          '8. 落ちている試験 2 件の直し'],
         face=tint(ORANGE, 0.04), hsize=16, size=13.5, top=0.84, step=0.185)
    card(fig, 0.025, 0.13, 0.465, 0.31, GREEN, '上積み(25 秒より速く)',
         ['U1 天井を上げる段(4.8・4.2 m/s)', 'U2 発進の補助を強める(約 −0.1 秒)', 'U3 B-FAST + 天井',
          'U4 7 秒台は机上だけ(区間ごとの最速の和 7.86 秒/周)'],
         face=tint(GREEN, 0.05), hsize=16, size=13.5, top=0.84, step=0.185)
    card(fig, 0.51, 0.13, 0.465, 0.31, PURPLE, '次の実車の走り方(案)',
         ['1. パターン 1 で基準。2 本続けて 27 秒以下なら「床」', '2. パターン 2 を A/B/A で確かめる(25 秒の仮説)',
          '3. B-FAST(ゲイン 0.85 から)', '4. 上積み。止める条件: R1 の 3 軸 7G など'],
         face=tint(PURPLE, 0.04), hsize=16, size=13.5, top=0.84, step=0.185)
    banner(fig, 'まず土台で 9 番を再現し、了承をもらえたものから 25 秒と上積みへ', size=18.5)
    save(fig, 'k12_todo')


FIGS = {'p1': p1_prev, 'k1': k1_issue, 'k2': k2_plan, 'k3': k3_res1, 'k4': k4_full, 'k5': k5_best, 'k6': k6_r1,
        'k7': k7_de, 'k8': k8_table, 'k9': k9_cond, 'k10': k10_tt, 'k11': k11_patterns, 'k12': k12_todo}


def sources():
    """図に使う本物の画像(check_storyboard_3min.py もこの一覧で有無を確かめる)"""
    return [PREV_FRAME, PREV_VIDEO, UI_SHOT] + list(CAM.values()) + [Path('/home/jetson/jetformer/data/raw') / LOG9,
                                                                     Path('/home/jetson/jetformer/data/raw') / LOG12]


def main():
    H.jp_font()
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        FIGS[n]()


if __name__ == '__main__':
    main()
