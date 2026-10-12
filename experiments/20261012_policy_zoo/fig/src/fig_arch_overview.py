"""(動画 1 枚目)比べた 5 つの方策の構成の一覧。色の付いた箱 = v0 と違うところ。
数字(パラメータ・推論・学習の時間)は eval/zoo_eval.json・train/summary.json から読む。"""
from zoolib import *  # noqa: F401,F403
from zoolib import Canvas, load, PCOL, INK, INK2, MUTED, GRID, tint

D = load()
P = D['policy']
c = Canvas()
c.title('比べた 5 つの方策(Policy)',
        '入力と出力の形は全部同じ。違うのは「センサーの混ぜ方」と「出力の作り方」。色の付いた箱が v0 と違うところ')

# ---- 共通の入力(上の帯)
c.box(0.6, 70.6, 133.8, 4.6, tint(INK2, 0.07), GRID, lw=1.6)
c.txt(2.0, 72.9, '入力(全部同じ)', size=17, weight='bold', ha='left')
c.txt(22.5, 72.9, 'カメラ 224×224 ・ ToF 42×54×2 ・ IMU 10×6 ・ 状態(eRPM。操作値は 0)・ 走行ジャンル',
      size=17, ha='left', color=INK)

CW, GAP, X0 = 25.8, 1.2, 0.6
TOP, BOT = 68.8, 8.0
IN = 1.1                      # 箱の内側の余白
cards = [
    ('jetformer_v0', 'TransFuser軽量型', 'JetFormer v0・基準'),
    ('act', 'ACT型', '手本 ACT(2023)'),
    ('tcp', 'TCP型', '手本 TCP・TransFuser++'),
    ('msfusion', 'TransFuser多段型', '本家と同じ多段融合'),
    ('dinov2', 'DINOv2型', '凍結 ViT の画像特徴'),
]
# 箱の中身: (行の文字のリスト, 違うか)。行 = (文字, 太字か)
body = {
    'jetformer_v0': [
        ([('画像 ResNet-18 → 49', False)], False),
        ([('ToF CNN → 42', False)], False),
        ([('IMU・状態・ジャンル', False)], False),
        ([('Encoder 4層', True), ('95 トークンを自己注意', False)], False),
        ([('出力トークン 1 個', True), ('→ ジャンル別ヘッド', False)], False),
    ],
    'act': [
        ([('画像 ResNet-18 → 49', False)], False),
        ([('ToF CNN → 42', False)], False),
        ([('IMU・状態・ジャンル', False)], False),
        ([('Encoder 4層', True), ('94 トークン', False)], False),
        ([('Decoder 2層', True), ('行動クエリ 10 個', False)], True),
    ],
    'tcp': [
        ([('画像 ResNet-18 → 49', False)], False),
        ([('ToF CNN → 42', False)], False),
        ([('IMU・状態・ジャンル', False)], False),
        ([('Encoder 4層', True), ('94 トークン', False)], False),
        ([('Decoder 2層', True), ('クエリ 21 個', False)], True),
    ],
    'msfusion': [
        None,                       # 画像と ToF は下で特別に描く(交換の箱)
        None,
        ([('IMU・状態・ジャンル', False)], False),
        ([('Encoder 2層', True), ('最終段の 95 トークン', False)], False),
        ([('出力トークン 1 個', True), ('→ ジャンル別ヘッド', False)], False),
    ],
    'dinov2': [
        ([('DINOv2 凍結 → 65', True)], True),
        ([('ToF CNN → 42', False)], False),
        ([('IMU・状態・ジャンル', False)], False),
        ([('Encoder 4層', True), ('111 トークン', False)], False),
        ([('出力トークン 1 個', True), ('→ ジャンル別ヘッド', False)], False),
    ],
}
# 出力の箱(全方策)
outs = {
    'jetformer_v0': ([('10 フレーム先の操作', True)], False),
    'act': ([('10 フレーム先の操作', True), ('(クエリごとに 1 つ)', False)], True),
    'tcp': ([('通過点・目標速度', True), ('+ 操作(走行は操作)', False)], True),
    'msfusion': ([('10 フレーム先の操作', True)], False),
    'dinov2': ([('10 フレーム先の操作', True)], False),
}
inspire = {
    'jetformer_v0': ['手本 TransFuser', '(2021)を軽く'],
    'act': ['手本 ACT', '(Zhao ほか 2023)'],
    'tcp': ['手本 TCP(2022)・', 'TransFuser++(2023)'],
    'msfusion': ['手本 TransFuser', '(Chitta ほか 2022)'],
    'dinov2': ['手本 DINOv2', '(Oquab ほか 2023)'],
}

# 縦の配置(上 → 下)
ROWS = [(57.0, 4.4), (52.0, 4.4), (47.0, 4.4), (37.8, 6.6), (28.4, 6.6)]   # (下端, 高さ)
OUT = (20.4, 5.6)

for i, (key, name, sub) in enumerate(cards):
    col = PCOL[key]
    x = X0 + i * (CW + GAP)
    # カードの枠と見出し
    c.box(x, BOT, CW, TOP - BOT, 'white', GRID, lw=1.6, z=1)
    c.box(x, 62.6, CW, TOP - 62.6, tint(col, 0.22), col, lw=2.4, z=2)
    c.lines(x + CW / 2, 65.7, [(name, 19, 'bold', INK), (sub, 16, 'normal', INK2)], lh=2.75)
    hx = x + IN
    hw = CW - 2 * IN
    for j, item in enumerate(body[key]):
        yb, h = ROWS[j]
        if item is None:
            continue
        rows, diff = item
        fc, ec, lw = (tint(col, 0.18), col, 2.4) if diff else ('white', MUTED, 1.6)
        c.box(hx, yb, hw, h, fc, ec, lw=lw)
        c.lines(hx + hw / 2, yb + h / 2, [(s, 16, 'bold' if b else 'normal', INK) for s, b in rows], lh=2.75)
    if key == 'msfusion':
        # 画像と ToF の枝 + 4 段の交換
        yb2, h2 = ROWS[1]
        yb1, h1 = ROWS[0]
        bw = 13.4
        for (yb, h, s) in ((yb1, h1, '画像 4 段'), (yb2, h2, 'ToF 4 段')):
            c.box(hx, yb, bw, h, tint(col, 0.18), col, lw=2.4)
            c.txt(hx + bw / 2, yb + h / 2, s, size=16, weight='bold')
        ex = hx + bw + 1.0
        ew = hw - bw - 1.0
        c.box(ex, yb2, ew, yb1 + h1 - yb2, tint(col, 0.45), col, lw=2.6)
        c.lines(ex + ew / 2, yb2 + (yb1 + h1 - yb2) / 2, [('交換', 16, 'bold', INK), ('× 4 段', 16, 'normal', INK)],
                lh=2.75)
        c.arrow(hx + bw, yb1 + h1 / 2, ex, yb1 + h1 / 2, color=col, lw=2.0, ms=12, style='<|-|>')
        c.arrow(hx + bw, yb2 + h2 / 2, ex, yb2 + h2 / 2, color=col, lw=2.0, ms=12, style='<|-|>')
    # 矢印(入力の 3 段 → 融合 → 読み出し → 出力)
    cx = hx + hw / 2
    c.arrow(cx, ROWS[2][0], cx, ROWS[3][0] + ROWS[3][1], color=MUTED)
    c.arrow(cx, ROWS[3][0], cx, ROWS[4][0] + ROWS[4][1], color=MUTED)
    c.arrow(cx, ROWS[4][0], cx, OUT[0] + OUT[1], color=MUTED)
    rows, diff = outs[key]
    fc, ec, lw = (tint(col, 0.18), col, 2.4) if diff else (tint(INK2, 0.07), MUTED, 1.6)
    c.box(hx, OUT[0], hw, OUT[1], fc, ec, lw=lw)
    c.lines(cx, OUT[0] + OUT[1] / 2, [(s, 16, 'bold' if b else 'normal', INK) for s, b in rows], lh=2.75)
    # 数字
    p = P[key]
    ttxt = '全部' if p['trainable'] == p['params'] else '%.1fM' % p['trainable']
    stats = [('パラメータ', '%.1fM' % p['params']), ('うち学習する', ttxt), ('推論', '%.1f ms' % p['ms']),
             ('学習の時間', '%.0f 分/本' % p['train_min'])]
    for k, (lab, val) in enumerate(stats):
        yy = 17.6 - k * 2.6
        c.txt(hx + 0.2, yy, lab, size=16, color=INK2, ha='left')
        c.txt(hx + hw - 0.2, yy, val, size=16, weight='bold', ha='right')

# ---- 共通の出力(下の帯)
c.box(0.6, 1.2, 133.8, 5.2, tint(INK2, 0.07), GRID, lw=1.6)
c.txt(2.0, 3.8, '出力(全部同じ)', size=17, weight='bold', ha='left')
c.txt(22.5, 3.8, 'actions [10, 2](0.33 秒先までのステア・スロットル)→ 同じ安全の仕組み → 車',
      size=17, ha='left')
c.save('zoo_arch_overview.png')
