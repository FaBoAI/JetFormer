"""方策ごとの構成図(act・tcp・msfusion・dinov2)。v0 の図は 2026-10-11 の日報の architecture.png(fig/arch_jetformer_v0.png)。
層の大きさは jetformer/policies/*.py の docstring・既定値、数字は eval/zoo_eval.json・train/summary.json から読む。"""
import sys

from zoolib import *  # noqa: F401,F403
from zoolib import Canvas, load, PCOL, INK, INK2, MUTED, GRID, BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, tint

D = load()
P = D['policy']
V0 = P['jetformer_v0']

# 共通の列(入力 → センサー別エンコーダ)
XI, WI = 1.0, 18.5
XE, WE = 22.0, 20.5
# act・tcp の列
XT, WT = 46.5, 13.0
XD, WD = 63.5, 19.5
XH, WH = 86.5, 21.5
XC, WC = 111.5, 22.5
TOP, BOT = 70.0, 32.0


def input_rows(c, rows, x_to, tok_x=None, stop=None):
    """rows = [(上端, 高さ, 色, 入力の行, エンコーダの行, トークン数, エンコーダが違うか)]。
    入力 → エンコーダ → x_to へ矢印(stop = {行の番号: 矢印の終わりの x})。→ 各行の中心の y のリスト"""
    ys = []
    stop = stop or {}
    for i, (ytop, h, col, inl, enl, ntok, diff) in enumerate(rows):
        yb = ytop - h
        yc = ytop - h / 2
        c.box(XI, yb, WI, h, tint(col, 0.16), col)
        c.lines(XI + WI / 2, yc, [(s, 17 if k == 0 and len(inl) > 1 else 16, 'bold' if k == 0 and len(inl) > 1 else 'normal', INK)
                                  for k, s in enumerate(inl)], lh=2.75)
        c.box(XE, yb, WE, h, tint(col, 0.30) if diff else 'white', col, lw=2.6 if diff else 2.0)
        c.lines(XE + WE / 2, yc, [(s, 17 if k == 0 and len(enl) > 1 else 16, 'bold' if k == 0 and len(enl) > 1 else 'normal', INK)
                                  for k, s in enumerate(enl)], lh=2.75)
        c.arrow(XI + WI, yc, XE, yc, color=col)
        c.arrow(XE + WE, yc, stop.get(i, x_to), yc, color=col)
        if ntok:
            c.txt((tok_x if tok_x is not None else XE + WE + 2.3), yc + 1.25, ntok, size=16, weight='bold')
        ys.append(yc)
    return ys


def std_rows():
    """v0 と同じトークン化(act・tcp)"""
    return [
        (70.0, 9.4, BLUE, ['カメラ', '224×224×3'], ['ResNet-18', 'ImageNet 事前学習', '→ 256×7×7'], '49', False),
        (59.4, 9.4, ORANGE, ['ToF VL53L9', '42×54×2', '距離+信頼度'], ['小さな CNN 4段', '+有効マスク', '→ 256×6×7'], '42', False),
        (48.8, 4.8, AQUA, ['IMU 10×6(0.1 秒)'], ['MLP'], '1', False),
        (42.8, 4.8, YELLOW, ['状態(eRPM)'], ['MLP'], '1', False),
        (36.8, 4.8, MAGENTA, ['走行ジャンル'], ['埋め込み'], '1', False),
    ]


def heads_row(c, items):
    for x, s in items:
        c.txt(x, 72.9, s, size=17, weight='bold', color=INK2)


def tiles(c, items, x0=1.0, y_top=27.0, tw=21.8, th=11.2, gap=1.2):
    """2×2 の数字のタイル。items = [(大, 1 行目, 2 行目, 色)]"""
    for i, (big, l1, l2, col) in enumerate(items):
        x = x0 + (i % 2) * (tw + gap)
        y = y_top - th - (i // 2) * (th + gap)
        c.tile(x, y, tw, th, big, l1, l2, color=col)


def table(c, x0, y_top, title, heads, rows, cols, note=None, rh=3.3, x_end=134.0):
    c.txt(x0, y_top, title, size=19, weight='bold', ha='left')
    y = y_top - 3.6
    if not any(heads):
        y += rh                 # 見出しの行なし
    nrow = len(rows) + (1 if any(heads) else 0)
    c.box(x0 - 0.6, y - rh * len(rows) - 1.9, x_end - x0 + 0.6, rh * nrow + 1.0 + (0 if any(heads) else 0.6), 'white', GRID,
          lw=1.4, z=1)
    for j, h in enumerate(heads):
        if h:
            c.txt(cols[j], y, h, size=17, weight='bold', color=INK2 if j < len(heads) - 1 else VIOLET, ha='left')
    for i, row in enumerate(rows):
        yy = y - rh * (i + 1)
        if i > 0 or any(heads):
            c.ax.plot([x0, x_end - 0.4], [yy + rh / 2, yy + rh / 2], color=GRID, lw=1.2, zorder=1)
        for j, s in enumerate(row):
            c.txt(cols[j], yy, s, size=16, color=INK2 if j == 0 else INK, weight='bold' if j == len(row) - 1 else 'normal',
                  ha='left')
    if note:
        c.txt(x0, y - rh * (len(rows) + 1) - 1.0, note, size=16, color=INK2, ha='left')


def safety_box(c, x, y, w, h):
    c.box(x, y, w, h, tint(INK2, 0.10), INK2)
    c.lines(x + w / 2, y + h / 2, [('同じ安全の仕組み', 16, 'bold', INK), ('先読み・ゲイン・', 16, 'normal', INK2),
                                   ('ToF ガード・', 16, 'normal', INK2), ('リカバリー・CH3', 16, 'normal', INK2)], lh=2.6)


def us(x, signed=False):
    return ('%+.1f µs' if signed else '%.1f µs') % x


# =========================================================================== ACT
def encoder_box(c, ntok):
    c.box(XT, BOT, WT, TOP - BOT, 'white', VIOLET, lw=2.4)
    c.lines(XT + WT / 2, (TOP + BOT) / 2, [('Encoder', 18, 'bold', INK), ('(v0 と同じ)', 16, 'normal', INK2), ('', 16, 'normal', INK),
                                          ('%d トークン' % ntok, 16, 'bold', INK), ('× 256 次元', 16, 'normal', INK),
                                          ('4 層', 16, 'normal', INK), ('8 ヘッド', 16, 'normal', INK), ('', 16, 'normal', INK),
                                          ('→ memory', 16, 'bold', VIOLET)], lh=2.75)


def fig_act():
    p = P['act']
    col = PCOL['act']
    c = Canvas()
    c.title('ACT型(Action Chunking Transformer を参考)',
            'ACT(Zhao ほか 2023、ロボットの腕の模倣学習)の形。10 個の行動クエリが Decoder で操作の列を読み出す')
    heads_row(c, [(XI + WI / 2, '入力'), (XE + WE / 2, 'エンコーダ'), (XT + WT / 2, '融合'), (XD + WD / 2, '読み出し'),
                  (XH + WH / 2, '出力'), (XC + WC / 2, '走行時')])
    input_rows(c, std_rows(), XT)
    encoder_box(c, 94)
    QB = 61.0
    c.box(XD, QB, WD, TOP - QB, tint(col, 0.18), col, lw=2.4)
    c.lines(XD + WD / 2, (TOP + QB) / 2, [('行動クエリ 10 個', 17, 'bold', INK), ('学習するベクトル', 16, 'normal', INK2)], lh=2.8)
    c.arrow(XD + WD / 2, QB, XD + WD / 2, QB - 3.4, color=col)
    c.box(XD, BOT, WD, QB - 3.4 - BOT, tint(col, 0.30), col, lw=2.6)
    c.lines(XD + WD / 2, (QB - 3.4 + BOT) / 2, [('Decoder', 18, 'bold', INK), ('2 層・8 ヘッド', 16, 'normal', INK), ('', 16, 'normal', INK),
                                               ('① 自己注意', 16, 'bold', INK), ('(クエリどうし)', 16, 'normal', INK2),
                                               ('② 交差注意', 16, 'bold', INK), ('(memory 参照)', 16, 'normal', INK2)], lh=2.8)
    c.arrow(XT + WT, 42.0, XD, 42.0, color=VIOLET)
    OB = 50.0
    c.box(XH, OB, WH, TOP - OB, tint(GREEN, 0.14), GREEN, lw=2.6)
    c.lines(XH + WH / 2, (TOP + OB) / 2, [('未来 10 フレーム', 17, 'bold', INK), ('クエリ k の出力 =', 16, 'normal', INK),
                                         ('k フレーム先の', 16, 'normal', INK), ('[ステア, スロットル]', 16, 'normal', INK),
                                         ('(33〜333 ms 先)', 16, 'normal', INK2)], lh=2.8)
    c.box(XH, BOT, WH, OB - 3.6 - BOT, 'white', col, lw=2.2)
    c.lines(XH + WH / 2, (OB - 3.6 + BOT) / 2, [('ジャンル別ヘッド', 17, 'bold', INK), ('Linear 256 → 2', 16, 'normal', INK),
                                               ('10 クエリで共有', 16, 'normal', INK2)], lh=2.85)
    c.arrow(XD + WD, 39.0, XH, 39.0, color=col)
    c.arrow(XH + WH / 2, OB - 3.6, XH + WH / 2, OB, color=GREEN)
    EB = 54.0
    c.box(XC, EB, WC, TOP - EB, 'white', col, lw=2.0, ls=(0, (4, 2)))
    c.lines(XC + WC / 2, (TOP + EB) / 2, [('予測の重ね合わせ', 16, 'bold', INK), ('(任意・既定オフ)', 16, 'normal', INK2),
                                         ('過去 0〜3 フレームの', 16, 'normal', INK), ('予測を混ぜる', 16, 'normal', INK)], lh=2.85)
    c.arrow(XH + WH, 62.0, XC, 62.0, color=GREEN)
    c.arrow(XC + WC / 2, EB, XC + WC / 2, EB - 3.6, color=MUTED)
    safety_box(c, XC, BOT, WC, EB - 3.6 - BOT)
    tiles(c, [('%.1fM' % p['params'], 'パラメータ', 'v0 %.1fM + Decoder' % V0['params'], INK),
              ('%.1f ms' % p['ms'], '推論 1 回(中央値)', 'v0 %.1f ms' % V0['ms'], INK),
              (us(p['L_mean']), '主指標(2 本の平均)', 'v0 より %s' % us(p['diff'], True), col),
              ('%.0f 分' % p['train_min'], '学習+エンジン 1 本', '30 エポック', INK)])
    table(c, 47.5, 27.0, '元の ACT との違い', ['', '元の ACT', 'この ACT型(既定)'],
          [('1 回の出力', '100 ステップ(2 秒)', '10 フレーム(0.33 秒)'),
           ('CVAE', 'あり(人のデータで効いた)', 'オフ(学習タブで選べる)'),
           ('重ね合わせ', '古い予測ほど重い', '走行タブで任意(新しいほど重い)'),
           ('入力', '腕のカメラ 4 台 + 関節', 'カメラ・ToF・IMU・状態'),
           ('損失', 'L1 + KL', 'L1(v0 と同じ)')],
          [47.5, 60.0, 89.0])
    c.save('arch_act.png')
    return c


# =========================================================================== TCP
def fig_tcp():
    p = P['tcp']
    col = PCOL['tcp']
    t = D['ev']['tcp']['per_run']
    c = Canvas()
    c.title('TCP型(TCP・TransFuser++ を参考)',
            'TCP(Wu ほか 2022)・TransFuser++(Jaeger ほか 2023)の形。通過点・目標速度・操作の 3 つを出す')
    heads_row(c, [(XI + WI / 2, '入力'), (XE + WE / 2, 'エンコーダ'), (XT + WT / 2, '融合'), (XD + WD / 2, '読み出し'),
                  (XH + WH / 2, '出力(3 つ)'), (XC + WC / 2, '走行時')])
    input_rows(c, std_rows(), XT)
    encoder_box(c, 94)
    QB = 59.8
    c.box(XD, QB, WD, TOP - QB, tint(col, 0.18), col, lw=2.4)
    c.lines(XD + WD / 2, (TOP + QB) / 2, [('クエリ 21 個', 17, 'bold', INK), ('通過点 10・速度 1', 16, 'normal', INK2),
                                         ('・操作 10', 16, 'normal', INK2)], lh=2.8)
    c.arrow(XD + WD / 2, QB, XD + WD / 2, QB - 3.4, color=col)
    c.box(XD, BOT, WD, QB - 3.4 - BOT, tint(col, 0.30), col, lw=2.6)
    c.lines(XD + WD / 2, (QB - 3.4 + BOT) / 2, [('Decoder', 18, 'bold', INK), ('2 層・8 ヘッド', 16, 'normal', INK), ('', 16, 'normal', INK),
                                               ('操作のクエリが', 16, 'normal', INK), ('通過点のクエリも', 16, 'normal', INK),
                                               ('参照できる', 16, 'normal', INK), ('(TCP の注意を', 16, 'normal', INK2),
                                               ('簡単にした形)', 16, 'normal', INK2)], lh=2.8)
    c.arrow(XT + WT, 42.0, XD, 42.0, color=VIOLET)
    outs = [(70.0, 8.8, '通過点', ['10 点・0.1 秒刻み', '1 秒先まで(m)'], False),
            (59.8, 8.8, '目標速度', ['0〜4.5 m/s の', '10 クラス'], False),
            (49.6, 8.8, '操作 actions', ['10 フレーム先の', '[ステア, スロットル]'], True)]
    for ytop, h, ttl, sub, main in outs:
        yb = ytop - h
        c.box(XH, yb, WH, h, tint(GREEN, 0.14) if main else tint(col, 0.14), GREEN if main else col, lw=2.6 if main else 2.0)
        c.lines(XH + WH / 2, yb + h / 2, [(ttl, 17, 'bold', INK)] + [(s, 16, 'normal', INK) for s in sub], lh=2.75)
        c.arrow(XD + WD, yb + h / 2, XH, yb + h / 2, color=GREEN if main else col)
        c.arrow(XH + WH, yb + h / 2, XC, yb + h / 2, color=GREEN if main else col)
    c.box(XH, BOT, WH, 39.4 - BOT, 'white', col, lw=2.0)
    c.lines(XH + WH / 2, (BOT + 39.4) / 2, [('ジャンル別の頭 × 3', 16, 'bold', INK), ('通過点は累積和で', 16, 'normal', INK2)],
            lh=2.8)
    CB = 45.6
    c.box(XC, CB, WC, TOP - CB, 'white', col, lw=2.4)
    c.lines(XC + WC / 2, (TOP + CB) / 2, [('制御器(ONNX の外)', 16, 'bold', INK), ('出し方 tcp_mode', 16, 'normal', INK2),
                                         ('0 操作だけ(既定)', 16, 'bold', GREEN), ('1 ハイブリッド', 16, 'normal', INK),
                                         ('2・3 試験的', 16, 'normal', INK2), ('速度の上限', 16, 'bold', INK),
                                         ('(前進を弱めるだけ)', 16, 'normal', INK2)], lh=2.8)
    c.arrow(XC + WC / 2, CB, XC + WC / 2, CB - 3.0, color=MUTED)
    safety_box(c, XC, BOT, WC, CB - 3.0 - BOT)
    w0, w1 = t['tcp_s0'], t['tcp_s1']
    ade = sorted([w0['waypoints']['ade_m'], w1['waypoints']['ade_m']])
    vm = (w0['target_speed']['mae_mps'] + w1['target_speed']['mae_mps']) / 2
    tiles(c, [('%.1fM' % p['params'], 'パラメータ', 'v0 %.1fM + Decoder' % V0['params'], INK),
              ('%.1f ms' % p['ms'], '推論 1 回(中央値)', '制御器 +約 0.2 ms', INK),
              (us(p['L_mean']), '主指標(操作の枝)', 'v0 より %s' % us(p['diff'], True), col),
              ('%.2f〜%.2f' % tuple(ade), '通過点 ADE [m]', '速度誤差 %.2f m/s' % vm, INK)])
    table(c, 47.5, 27.0, '学習のしかたと、元の研究との違い', ['', 'この TCP型'],
          [('正解', '記録の IMU のヨーと eRPM の速さを積分して作る'),
           ('損失', 'L1 + 0.5 × 通過点 L1 [m] + 0.1 × 速度の交差エントロピー'),
           ('制御器', 'pure pursuit(0.25 秒先)+ 速さごとの表。較正は学習のシーン'),
           ('元との違い', 'GRU なし(TensorRT でループを作らない)・価値の頭なし'),
           ('評価', '比べたのは操作の枝(tcp_mode 0)だけ。1〜3 は写像が粗い')],
          [47.5, 60.0], rh=3.3)
    c.save('arch_tcp.png')
    return c


# =========================================================================== 多段融合
def fig_msfusion():
    p = P['msfusion']
    col = PCOL['msfusion']
    c = Canvas()
    c.title('TransFuser多段型(本家と同じ多段融合)',
            'TransFuser(Prakash ほか 2021・Chitta ほか 2022)の形。画像と ToF の CNN を 4 つの段ごとに小さな Transformer でつなぐ')
    IY, IH = 59.6, 10.4        # 画像の行(下端・高さ)
    EY, EH = 46.8, 9.6         # 交換の行
    TY, TH = 33.0, 10.4        # ToF の行
    XS = [15.0, 29.0, 44.4, 59.8, 75.2]     # stem・段 1〜4 の左端
    SW = [12.0, 13.6, 13.6, 13.6, 13.6]
    img = [('stem', '64 ch', '56×56'), ('段 1', '64 ch', '56×56'), ('段 2', '128 ch', '28×28'), ('段 3', '256 ch', '14×14'),
           ('段 4', '512 ch', '7×7')]
    tof = [('stem', '32 ch', '48×56'), ('段 1', '64 ch', '24×28'), ('段 2', '128 ch', '12×14'), ('段 3', '256 ch', '6×7'),
           ('段 4', '512 ch', '6×7')]
    ex = [('交換 1', 'd64・2層'), ('交換 2', 'd128・2層'), ('交換 3', 'd256・2層'), ('交換 4', 'd512・1層')]
    c.txt(7.0, 72.9, '入力', size=17, weight='bold', color=INK2)
    c.txt((XS[0] + XS[4] + SW[4]) / 2, 72.9, 'ResNet-18 の 4 段 と ToF の 4 段 の CNN(段ごとに交換)', size=17, weight='bold',
          color=INK2)
    XF, WF = 91.6, 18.0
    XO, WO = 111.8, 22.2
    c.txt(XF + WF / 2, 72.9, '最後の融合', size=17, weight='bold', color=INK2)
    c.txt(XO + WO / 2, 72.9, '出力', size=17, weight='bold', color=INK2)
    # 入力
    c.box(1.0, IY, 12.0, IH, tint(BLUE, 0.16), BLUE)
    c.lines(7.0, IY + IH / 2, [('カメラ', 17, 'bold', INK), ('224×224', 16, 'normal', INK), ('RGB', 16, 'normal', INK2)], lh=2.8)
    c.box(1.0, TY, 12.0, TH, tint(ORANGE, 0.16), ORANGE)
    c.lines(7.0, TY + TH / 2, [('ToF', 17, 'bold', INK), ('42×54×2', 16, 'normal', INK), ('→ 48×56', 16, 'normal', INK2)], lh=2.8)
    c.arrow(13.0, IY + IH / 2, XS[0], IY + IH / 2, color=BLUE)
    c.arrow(13.0, TY + TH / 2, XS[0], TY + TH / 2, color=ORANGE)
    for k in range(5):
        x, w = XS[k], SW[k]
        c.box(x, IY, w, IH, 'white', BLUE, lw=2.0)
        c.lines(x + w / 2, IY + IH / 2, [(img[k][0], 17, 'bold', INK), (img[k][1], 16, 'normal', INK), (img[k][2], 16, 'normal', INK)],
                lh=2.8)
        c.box(x, TY, w, TH, 'white', ORANGE, lw=2.0)
        c.lines(x + w / 2, TY + TH / 2, [(tof[k][0], 17, 'bold', INK), (tof[k][1], 16, 'normal', INK), (tof[k][2], 16, 'normal', INK)],
                lh=2.8)
        if k < 4:
            c.arrow(x + w, IY + IH / 2, XS[k + 1], IY + IH / 2, color=BLUE)
            c.arrow(x + w, TY + TH / 2, XS[k + 1], TY + TH / 2, color=ORANGE)
        if k >= 1:
            c.box(x, EY, w, EH, tint(col, 0.30), col, lw=2.6)
            c.lines(x + w / 2, EY + EH / 2, [(ex[k - 1][0], 17, 'bold', INK), (ex[k - 1][1], 16, 'normal', INK),
                                            ('91 トークン', 16, 'normal', INK2)], lh=2.8)
            c.arrow(x + w / 2, EY + EH, x + w / 2, IY, color=col, lw=2.2, ms=13, style='<|-|>')
            c.arrow(x + w / 2, EY, x + w / 2, TY + TH, color=col, lw=2.2, ms=13, style='<|-|>')
    c.lines(XS[0] + SW[0] / 2, EY + EH / 2, [('段ごとに', 16, 'normal', INK2), ('平均して', 16, 'normal', INK2),
                                            ('注意して', 16, 'normal', INK2), ('戻して足す', 16, 'normal', INK2)], lh=2.4)
    # 最後の融合
    c.box(XF, TY, WF, IY + IH - TY, 'white', VIOLET, lw=2.4)
    c.lines(XF + WF / 2, (TY + IY + IH) / 2, [('Encoder', 18, 'bold', INK), ('(v0 型)', 16, 'normal', INK2), ('', 16, 'normal', INK),
                                              ('画像 49 + ToF 42', 16, 'normal', INK), ('+ IMU・状態', 16, 'normal', INK),
                                              ('+ ジャンル・出力', 16, 'normal', INK), ('= 95 トークン', 16, 'bold', INK),
                                              ('', 16, 'normal', INK), ('2 層・8 ヘッド', 16, 'normal', INK)], lh=2.8)
    c.arrow(XS[4] + SW[4], IY + IH / 2, XF, IY + IH / 2, color=BLUE)
    c.arrow(XS[4] + SW[4], TY + TH / 2, XF, TY + TH / 2, color=ORANGE)
    c.txt(XS[4] + SW[4] + 1.4, IY + IH / 2 + 1.3, '49', size=16, weight='bold')
    c.txt(XS[4] + SW[4] + 1.4, TY + TH / 2 + 1.3, '42', size=16, weight='bold')
    OB = 54.0
    c.box(XO, OB, WO, IY + IH - OB, tint(GREEN, 0.14), GREEN, lw=2.6)
    c.lines(XO + WO / 2, (OB + IY + IH) / 2, [('未来 10 フレーム', 17, 'bold', INK), ('ステア・スロットル', 16, 'normal', INK),
                                             ('(v0 と同じ)', 16, 'normal', INK2)], lh=2.85)
    c.box(XO, TY, WO, OB - 3.6 - TY, 'white', GREEN, lw=2.0)
    c.lines(XO + WO / 2, (TY + OB - 3.6) / 2, [('出力トークン', 16, 'normal', INK), ('→ 全結合', 16, 'normal', INK),
                                              ('ジャンル別ヘッド', 17, 'bold', INK), ('(v0 と同じ)', 16, 'normal', INK2)], lh=2.85)
    c.arrow(XF + WF, 40.0, XO, 40.0, color=VIOLET)
    c.arrow(XO + WO / 2, OB - 3.6, XO + WO / 2, OB, color=GREEN)
    c.txt(1.0, 30.0, 'ToF は上下 3 行・左右 1 列を 0(無効)で詰めて 48×56 にし、どの段も交換の格子(画像 7×7・ToF 6×7)で割り切れる大きさにした',
          size=16, color=INK2, ha='left')
    tiles(c, [('%.1fM' % p['params'], 'パラメータ', 'v0 %.1fM' % V0['params'], INK),
              ('%.1f ms' % p['ms'], '推論 1 回(中央値)', 'v0 %.1f ms' % V0['ms'], INK),
              (us(p['L_mean']), '主指標(2 本の平均)', 'v0 より %s' % us(p['diff'], True), YELLOW),
              ('%.0f 分' % p['train_min'], '学習+エンジン 1 本', 'いちばん長い', INK)], y_top=26.6, th=11.0)
    table(c, 47.5, 26.6, '本家 TransFuser との違い', ['', '本家', 'この多段型'],
          [('センサー', 'カメラ + LiDAR(鳥瞰図)', 'カメラ + ToF(前向きの距離画像)'),
           ('交換の格子', '8×8', '画像 7×7・ToF 6×7'),
           ('交換の深さ', '段ごとに 4 層以上', '2・2・2・1 層'),
           ('補助タスク', '深度・セグメンテーション', 'なし'),
           ('最後', '平均の特徴 → GRU で通過点', 'トークンの融合 → 操作')],
          [47.5, 60.5, 92.0], rh=3.25)
    c.save('arch_msfusion.png')
    return c


# =========================================================================== DINOv2
def fig_dinov2():
    p = P['dinov2']
    col = PCOL['dinov2']
    c = Canvas()
    c.title('DINOv2型(凍結した ViT-S/14 の画像特徴)',
            'DINOv2(Oquab ほか 2023、自己教師あり学習)の画像特徴を凍結して使い、ToF・IMU などと v0 型の Encoder で混ぜる')
    XP, WP = 45.4, 19.2          # パッチの平均・射影
    XT2, WT2 = 68.0, 15.0        # 融合
    XH2, WH2 = 86.6, 21.0
    XO, WO = 110.2, 23.8
    heads_row(c, [(XI + WI / 2, '入力'), (XE + WE / 2, 'エンコーダ'), (XP + WP / 2, '画像トークン'), (XT2 + WT2 / 2, '融合'),
                  (XH2 + WH2 / 2, '出力'), (XO + WO / 2, '走行で使う')])
    rows = [
        (70.0, 11.4, BLUE, ['カメラ', '224×224×3'], ['DINOv2 ViT-S/14', '凍結・12 ブロック', '384 次元・21.6M'], None, True),
        (57.6, 8.4, ORANGE, ['ToF VL53L9', '42×54×2'], ['小さな CNN 4段', '→ 256×6×7'], '42', False),
        (48.2, 4.0, AQUA, ['IMU 10×6(0.1 秒)'], ['MLP'], '1', False),
        (43.4, 4.0, YELLOW, ['状態(eRPM)'], ['MLP'], '1', False),
        (38.6, 4.0, MAGENTA, ['走行ジャンル'], ['埋め込み'], '1', False),
        (33.8, 4.0, VIOLET, ['出力用トークン'], ['学習するベクトル'], '1', False),
    ]
    ys = input_rows(c, rows, XT2, tok_x=XT2 - 2.0, stop={0: XP})
    yb, h = 70.0 - 11.4, 11.4
    c.box(XP, yb, WP, h, tint(col, 0.30), col, lw=2.6)
    c.lines(XP + WP / 2, yb + h / 2, [('パッチ 256 + CLS', 16, 'bold', INK), ('2×2 で平均 → 64', 16, 'normal', INK),
                                     ('→ 256 次元に', 16, 'normal', INK)], lh=2.8)
    c.arrow(XP + WP, ys[0], XT2, ys[0], color=BLUE)
    c.txt(XT2 - 2.0, ys[0] + 1.25, '65', size=16, weight='bold')
    BOT2 = 29.8
    c.box(XT2, BOT2, WT2, TOP - BOT2, 'white', VIOLET, lw=2.4)
    c.lines(XT2 + WT2 / 2, (TOP + BOT2) / 2, [('Encoder', 18, 'bold', INK), ('(v0 型)', 16, 'normal', INK2), ('', 16, 'normal', INK),
                                             ('111 トークン', 16, 'bold', INK), ('× 256 次元', 16, 'normal', INK),
                                             ('4層・8ヘッド', 16, 'normal', INK), ('', 16, 'normal', INK),
                                             ('学習するのは', 16, 'normal', INK2), ('この先だけ', 16, 'bold', col)], lh=2.75)
    c.box(XH2, 52.0, WH2, TOP - 52.0, tint(GREEN, 0.14), GREEN, lw=2.6)
    c.lines(XH2 + WH2 / 2, (TOP + 52.0) / 2, [('未来 10 フレーム', 17, 'bold', INK), ('ステア・スロットル', 16, 'normal', INK),
                                             ('(v0 と同じ)', 16, 'normal', INK2)], lh=2.85)
    c.box(XH2, BOT2, WH2, 48.4 - BOT2, 'white', GREEN, lw=2.0)
    c.lines(XH2 + WH2 / 2, (BOT2 + 48.4) / 2, [('出力トークン', 16, 'normal', INK), ('→ 全結合', 16, 'normal', INK),
                                              ('ジャンル別ヘッド', 17, 'bold', INK), ('(v0 と同じ)', 16, 'normal', INK2)], lh=2.85)
    c.arrow(XT2 + WT2, 39.0, XH2, 39.0, color=VIOLET)
    c.arrow(XH2 + WH2 / 2, 48.4, XH2 + WH2 / 2, 52.0, color=GREEN)
    c.box(XO, 52.0, WO, TOP - 52.0, 'white', col, lw=2.0)
    c.lines(XO + WO / 2, (TOP + 52.0) / 2, [('エンジン', 17, 'bold', INK), ('FP16 + LayerNorm・', 16, 'normal', INK),
                                           ('Softmax だけ FP32', 16, 'normal', INK), ('(fp16_norm_fp32)', 16, 'normal', INK2)], lh=2.85)
    c.box(XO, BOT2, WO, 48.4 - BOT2, 'white', col, lw=2.0, ls=(0, (4, 2)))
    c.lines(XO + WO / 2, (BOT2 + 48.4) / 2, [('重みのファイル', 17, 'bold', INK), ('dinov2_vits14_', 16, 'normal', INK),
                                            ('pretrain.pth(88 MB)', 16, 'normal', INK), ('ネットから自動では', 16, 'normal', INK2),
                                            ('取らない', 16, 'normal', INK2)], lh=2.85)
    c.arrow(XH2 + WH2, 61.0, XO, 61.0, color=GREEN)
    tiles(c, [('%.1fM' % p['params'], 'パラメータ', 'うち学習 %.2fM' % p['trainable'], INK),
              ('%.1f ms' % p['ms'], '推論 1 回(中央値)', '5 つで一番遅い', INK),
              (us(p['L_mean']), '主指標(2 本の平均)', 'v0 より %s' % us(p['diff'], True), col),
              ('%.0f MB' % p['peak_mb'], '学習のメモリ(最大)', 'v0 %.0f MB' % V0['peak_mb'], INK)], y_top=26.6, th=11.0)
    table(c, 47.5, 26.6, '得意・不得意と注意', ['', ''],
          [('得意', '画像のずれ・明るさの変化にとても強い(誤差の増え 0.3 µs 以内)'),
           ('不得意', 'ToF を 1 行ずらすと誤差が 12〜14 µs 増える(ほかは 0.2〜3 µs)'),
           ('精度', '普通の FP16 では種 1 が精度の確認に落ちた(2.85 µs > 2.5 µs)'),
           ('速さ', '走行ループの見積もり 8.3 ms(30 fps の 33 ms の内)'),
           ('学習', 'ViT は凍結。特徴はキャッシュしない(色の揺らぎと両立しない)')],
          [47.5, 58.0], rh=3.4)
    c.save('arch_dinov2.png')
    return c


if __name__ == '__main__':
    which = sys.argv[1:] or ['act', 'tcp', 'msfusion', 'dinov2']
    for w in which:
        globals()['fig_' + w]()
