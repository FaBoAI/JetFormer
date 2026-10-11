"""course_map.json と course_map.png を作る"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import course_geom as cg
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrowPatch, Rectangle, Ellipse
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

OUT = '/home/jetson/jetformer_reports/overnight_20261011/course'

HOTSPOTS = [
    {'id': 'R1', 'name': 'R1 ショートカット右ヘアピン', 'center': [6.10, 2.58], 'radius_m': 0.40,
     'contact': {'object': '中央レール Z=2.80 の白い板 (X 5.46〜6.41) の南面。東端 X=6.41 に支柱', 'xz': [6.10, 2.80]},
     'how': '暗幕内の左ヘアピンを出た直後に右いっぱいで回る。曲がり切れないと白い板に正面から当たり、鼻先を押し付けて止まる(007 で強い衝突 8/20 通過)。'
            '006 の軽い接触はヘアピン内側のレール端 (7.51, 2.80) 付近',
     'evidence': ['collision_map.md §3.1 と cmap_seq_R1_007_143s.png: 右から白い板(右端に支柱)が正面に入り、当たった後は板と支柱を正面に見て止まる',
                  '床の黒い板(ショートカットの黒いスポンジ)が直前に写る',
                  'course.html では白い板は X 5.46〜6.41、隣の切れ目 X 6.41〜7.51 が黒いスポンジ']},
    {'id': 'DE', 'name': 'DE バックストレート終端(LED 矢印ゲートの北側支柱)', 'center': [5.05, 5.40], 'radius_m': 0.40,
     'contact': {'object': '外周の板の継ぎ目 (X 4.87〜4.97, Z=5.60) に立つ LED ゲートの北側支柱と、その西の赤い板の東端', 'xz': [4.92, 5.58]},
     'how': '芝の左ヘアピンを出て奥の直線を -X へ加速中、右前が外周の角に当たる(006 で 11/16 通過)。当たった後は外周を向いて止まるか、跳ね返って狭い道へ入る',
     'evidence': ['cmap_seq_DE_006_106s.png: 当たった直後、正面の外周は左が赤い板・右が白い板で、継ぎ目に支柱と縦のポールが立つ = course.html の X 4.87(赤)/4.97(白)の継ぎ目と一致',
                  '同じ周の 0.5〜1 s 前に LED 矢印の緑の表示が画面上端から抜ける(ゲートの約 1.1 m 手前)',
                  '写真 img_0522/0524 では、継ぎ目の西の赤い板がゲート支柱に向かって内側へ斜めに入っており、通路側に角が出ている(course.html は直線、未計測)']},
    {'id': 'GM_in', 'name': 'GM 入口(芝の左ヘアピン、内側の板の端)', 'center': [8.02, 3.92], 'radius_m': 0.30,
     'contact': {'object': 'Z=4.20 のレールの東端 (8.19, 4.20) の支柱。白い PTFE シートの角', 'xz': [8.19, 4.20]},
     'how': '右ヘアピンから左いっぱいへ早く切り替えすぎて、左前が内側の板の端に当たる(006 の最初の2周、steer_gain 1.8)',
     'evidence': ['collision_map.md §3.3 / cmap_seq_GMin_006_047s.png', '白い PTFE シート(2×2 枚)が正面に写った直後']},
    {'id': 'GM_out', 'name': 'GM 出口(芝の左ヘアピン、外側の板)', 'center': [8.95, 5.30], 'radius_m': 0.35,
     'contact': {'object': '奥の外周 Z=5.60(赤 X 6.83〜8.69 / 白 8.69〜9.55)と芝の斜めの板 (9.52〜9.65, 4.62〜5.56)', 'xz': [8.95, 5.60]},
     'how': '左いっぱいのまま膨らみ、右前が外側の赤白の板に触る(限界での曲がり不足、約 4900 eRPM)。そのまま約 1.4 s 後に DE で当たる流れになる',
     'evidence': ['collision_map.md §3.3 / cmap_seq_GMout_006_123s.png']},
]

def section_labels(C):
    out = {}
    for k, n in enumerate(C['section_ids']):
        m = np.nonzero(C['section'] == k)[0]
        out[n] = {'s_start_m': round(float(C['s'][m[0]]), 2), 's_end_m': round(float(C['s'][m[-1]]), 2),
                  'name': cg.CENTERLINE_SECTIONS[k][1], 'description': cg.CENTERLINE_SECTIONS[k][2]}
    return out

def write_json(geo, C):
    step = 5   # 0.1 m ごと
    j = dict(geo)
    j['centerline'] = {
        'note': '各レーンの中央を通る左回りの基準線(実際の走行ラインではない)。進行度 s は HS の始まり (2.45, 0.70) を 0 として進行方向に測る。'
                '狭い道は外側(Z≈5.25)を通る(カメラ画像で、独立した白い板が左に見える)。強化学習の報酬(コースに沿った進み)に使える',
        'length_m': round(C['length'], 3), 'sections': section_labels(C),
        'xz': np.round(C['xz'][::step], 3).tolist(), 's': np.round(C['s'][::step], 3).tolist(),
        'heading_rad': np.round(C['heading'][::step], 4).tolist(), 'section': [C['section_ids'][k] for k in C['section'][::step]]}
    j['hotspots'] = HOTSPOTS
    j['notes'] = [
        'LED 矢印ゲートの表示面は course.html では -X(西)向きだが、写真と車載カメラ(奥の直線を -X へ走るときに緑の矢印が正面に見える)から実物は +X(東)向き',
        '独立した白い板 (Z=4.90) は写真では黒テープ Z=4.90 より北(外周側)に見える。外側の狭い道は 0.7 m より狭い可能性がある',
        '外周 Z=5.60 の赤い板 (X 3.01〜4.87) は写真ではゲート支柱に向かって内側へ斜めに入っているように見える(DE の角)',
        'course.html 自身が estimated: true(公式図の寸法 + 写真からの推定)。数 10 cm 単位の誤差はありうる',
    ]
    json.dump(j, open(os.path.join(OUT, 'course_map.json'), 'w'), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- 描画
COL = dict(bg='#fcfcfb', floor='#cfd1cf', red='#c8102e', white='#ffffff', rail_edge='#5d5f5e', post='#2b2b2b',
           turf='#7cab4f', ptfe='#f3f5f2', tape_green='#12bea0', blue='#43b9e8', blue_tape='#264bbd', shortcut='#3b3d3b',
           curtain='#151515', tunnel_fill='#e9e9e7', text='#0b0b0b', text2='#52514e', line='#2a78d6', hot='#e34948',
           light='#7e65ec', led='#1baf7a')

def draw_base(ax, geo, C, labels=True, centerline=True, alpha_floor=1.0):
    ax.set_facecolor(COL['bg'])
    outer = [(0.20, 5.60), (9.55, 5.60), (10.30, 4.70), (10.30, 0.85), (9.85, 0.0), (5.60, 0.0), (5.60, -0.50), (1.88, -0.50),
             (1.88, 0.0), (0.0, 0.0), (0.0, 4.65)]
    ax.add_patch(Polygon(outer, closed=True, fc=COL['floor'], ec='none', alpha=alpha_floor, zorder=0))
    A = geo['areas']
    ax.add_patch(Polygon(A['turf_low_mu']['polygon'], fc=COL['turf'], ec=COL['tape_green'], lw=1.2, zorder=1))
    ax.add_patch(Polygon(A['ptfe_white_sheet']['polygon'], fc=COL['ptfe'], ec=COL['tape_green'], lw=1.2, zorder=2))
    for z in A['turf_low_mu']['tape_green_z']:
        ax.plot([cg.TURF_LEFT, cg.TURF_LEFT + 1.85], [z, z], color=COL['tape_green'], lw=1.2, zorder=2)
    for a, b in A['ptfe_white_sheet']['white_lines']:
        ax.plot([a[0], b[0]], [a[1], b[1]], color='#b9bdb7', lw=0.8, zorder=2)
    ax.add_patch(Polygon(A['blue_start_mat']['polygon'], fc=COL['blue'], ec=COL['blue_tape'], lw=1.5, zorder=1))
    bm = np.array(A['blue_start_mat']['polygon'])
    ax.plot([bm[:, 0].min(), bm[:, 0].max()], [2.95, 2.95], color=COL['blue_tape'], lw=1, zorder=2)
    ax.plot([1.275, 1.275], [2.15, 3.75], color=COL['blue_tape'], lw=1, zorder=2)
    ax.add_patch(Polygon(A['shortcut']['tape_outline'], fc='none', ec='#25282a', lw=1.0, zorder=1))
    ax.add_patch(Polygon(A['shortcut']['polygon'], fc=COL['shortcut'], ec='none', zorder=1))
    for p in A['parking']:
        ax.add_patch(Polygon(p['polygon'], fc='#e7e8e6', ec=p['color'], lw=2.2, zorder=1))
        if labels:
            ax.text(p['center'][0], p['center'][1], p['id'], ha='center', va='center', fontsize=10, color=p['color'], fontweight='bold', zorder=3)
    for l in A['start_lines']:
        ax.plot([l['a'][0], l['b'][0]], [l['a'][1], l['b'][1]], color='white', lw=3.0, solid_capstyle='butt', zorder=2)
    lt = A['lane_tape_Z4.90']
    ax.plot([lt['a'][0], lt['b'][0]], [lt['a'][1], lt['b'][1]], color='#3b3b3b', lw=0.8, ls=(0, (6, 4)), zorder=1)
    # 暗幕
    tw = geo['curtain_tunnel']['walls']
    poly = [w['a'] for w in tw if w['kind'] == 'wall'] + [tw[4]['b']]
    ax.add_patch(Polygon(poly, closed=True, fc='#bdbdbb', ec='none', alpha=0.55, zorder=1, hatch='////'))
    for w in tw:
        ax.plot([w['a'][0], w['b'][0]], [w['a'][1], w['b'][1]], color=COL['curtain'], lw=4.0 if w['kind'] == 'wall' else 3.0,
                solid_capstyle='round', zorder=4)
    dt = geo['curtain_tunnel']['divider_tape']
    ax.plot([dt['a'][0], dt['b'][0]], [dt['a'][1], dt['b'][1]], color='#3b3b3b', lw=0.8, ls=(0, (6, 4)), zorder=2)
    # レール
    for r in geo['rails']:
        a, b = r['a'], r['b']
        ax.plot([a[0], b[0]], [a[1], b[1]], color=COL['rail_edge'], lw=4.6, solid_capstyle='butt', zorder=5)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=COL['red'] if r['color'] == 'red' else COL['white'], lw=3.0, solid_capstyle='butt', zorder=6)
    P = np.array(geo['posts'])
    ax.scatter(P[:, 0], P[:, 1], s=7, color=COL['post'], zorder=7, linewidths=0)
    # ゲート
    lg = geo['gates']['light_gate']
    ax.plot([2.20, 2.20], [1.40, 4.20], color=COL['light'], lw=2.0, ls=(0, (4, 2)), zorder=8)
    ax.scatter([2.20], [2.80], s=60, marker='*', color=COL['light'], zorder=9)
    ax.add_patch(Circle(tuple(lg['floor_spots_center']), lg['floor_spots_radius_m'], fc=COL['light'], ec='none', alpha=0.08, zorder=1))
    ax.plot([4.90, 4.90], [4.20, 5.60], color=COL['led'], lw=3.0, zorder=8)
    ax.add_patch(Rectangle((4.90, 4.435), 0.07, 0.93, fc=COL['led'], ec='none', zorder=8))
    ax.add_patch(FancyArrowPatch((4.98, 4.90), (5.42, 4.90), arrowstyle='-|>', mutation_scale=11, color=COL['led'], lw=1.6, zorder=9))
    if centerline:
        xz = C['xz']
        ax.plot(xz[:, 0], xz[:, 1], color=COL['line'], lw=1.1, ls=(0, (5, 3)), zorder=3, alpha=0.9)
        L = C['length']
        for s0 in np.arange(1.2, L, 2.1):
            i = int(np.searchsorted(C['s'], s0)); j = min(i + 6, len(xz) - 1)
            ax.add_patch(FancyArrowPatch(tuple(xz[i]), tuple(xz[j]), arrowstyle='-|>', mutation_scale=13, color=COL['line'], lw=0, zorder=4))

def frame(ax):
    ax.set_xlim(-0.55, 10.85); ax.set_ylim(-0.95, 6.40); ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, 11, 1)); ax.set_yticks(np.arange(0, 6, 1))
    ax.tick_params(colors=COL['text2'], labelsize=8.5, length=3)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.grid(color='#9a9a96', alpha=0.18, lw=0.6, zorder=0)
    ax.set_xlabel('X [m](幅方向、左→右)', color=COL['text2'], fontsize=9)
    ax.set_ylabel('Z [m](奥行き、手前→奥)', color=COL['text2'], fontsize=9)

def label(ax, xy, txt, xytext, fs=8.6, color=None, ha='center', arrow=True, bold=False):
    kw = dict(fontsize=fs, color=color or COL['text'], ha=ha, va='center', zorder=20, fontweight='bold' if bold else 'normal',
              bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#c9c9c4', lw=0.6, alpha=0.93))
    if arrow:
        ax.annotate(txt, xy=xy, xytext=xytext, arrowprops=dict(arrowstyle='-', color='#6b6b67', lw=0.7, shrinkA=2, shrinkB=2), **kw)
    else:
        ax.text(xytext[0], xytext[1], txt, **kw)

def draw_hotspots(ax, show_text=True):
    for h in HOTSPOTS:
        c = h['center']
        ax.add_patch(Circle(tuple(c), h['radius_m'], fc=COL['hot'], ec=COL['hot'], alpha=0.22, lw=0, zorder=10))
        ax.add_patch(Circle(tuple(c), h['radius_m'], fc='none', ec=COL['hot'], lw=1.6, zorder=11))
        ax.scatter([h['contact']['xz'][0]], [h['contact']['xz'][1]], marker='x', s=40, color=COL['hot'], linewidths=2.0, zorder=12)

def main():
    geo = cg.build(); C = cg.centerline()
    write_json(geo, C)
    fig, ax = plt.subplots(figsize=(13.5, 8.6), dpi=150)
    fig.patch.set_facecolor(COL['bg'])
    draw_base(ax, geo, C)
    draw_hotspots(ax)
    frame(ax)
    # 区間ラベル
    secpos = {'HS': (4.9, 0.95), 'LH': (8.95, 1.05), 'R1': (6.95, 2.00), 'GM': (9.35, 3.55), 'DE': (6.45, 4.65), 'SW': (0.50, 3.05)}
    for k, (x, z) in secpos.items():
        ax.text(x, z, k, fontsize=10.5, color=COL['line'], fontweight='bold', ha='center', va='center', zorder=15,
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec=COL['line'], lw=0.8, alpha=0.95))
    # 部品のラベル
    label(ax, (9.1, 4.75), '芝(低μ)', (9.75, 3.05))
    label(ax, (8.69, 4.2), '白いPTFEシート\n(低μ)', (9.35, 2.38), fs=8.2)
    label(ax, (1.30, 3.45), '青いマット\n(スタート)', (0.82, 4.45), fs=8.2)
    label(ax, (6.96, 2.85), 'ショートカット\n(黒スポンジ 1 cm)', (5.55, 2.18), fs=8.2)
    label(ax, (8.9, 0.15), '黒い暗幕(高さ 1.33 m、屋根なし)', (8.80, -0.62), fs=8.2)
    label(ax, (7.78, 1.40), '中央フラップ', (7.15, 1.05), fs=7.8)
    label(ax, (2.20, 3.9), '照明ゲート', (2.95, 3.55), fs=8.2, color=COL['light'])
    label(ax, (4.90, 5.60), 'LED 矢印ゲート(表示は +X 向き)', (3.05, 6.12), fs=8.2, color='#0d6b48')
    label(ax, (3.70, 4.90), '狭い道の仕切り板', (3.75, 4.55), fs=7.8)
    label(ax, (3.20, 0.70), 'スタートライン', (3.40, 1.85), fs=8.2)
    label(ax, (0.45, 0.45), '斜めの板', (1.35, 0.38), fs=7.8)
    # ホットスポットの説明
    label(ax, (6.10, 2.58), 'R1  白い板に正面衝突', (4.65, 3.18), fs=8.8, color=COL['hot'], bold=True)
    label(ax, (5.05, 5.40), 'DE  ゲート支柱の外周の角', (6.55, 6.12), fs=8.8, color=COL['hot'], bold=True)
    label(ax, (8.02, 3.92), 'GM入口  内側の板の端', (7.0, 3.62), fs=8.8, color=COL['hot'], bold=True)
    label(ax, (8.95, 5.30), 'GM出口  外側の板', (9.70, 6.12), fs=8.8, color=COL['hot'], bold=True)
    # 走行方向
    ax.text(10.82, -0.86, '走行方向: 左回り(青い破線 = 各レーン中央の基準線、1周 %.1f m)' % C['length'], fontsize=8.6, color=COL['line'], ha='right', va='bottom')
    ax.plot([0.0, 1.0], [-0.80, -0.80], color=COL['text'], lw=2.2); ax.text(0.5, -0.72, '1 m', ha='center', va='bottom', fontsize=8, color=COL['text'])
    ax.set_title('JetFormer コース図(上から見た図、course.html の形状から作成)', loc='left', fontsize=13, color=COL['text'], pad=22)
    ax.text(0.0, 1.012, '赤い円 = 衝突が集中する場所(R1 / DE / GM、×= 当たる相手)。形状は course.html の推定値(公式図 + 写真)で、写真と食い違う点は course_map.json の notes に記載',
            transform=ax.transAxes, fontsize=8.6, color=COL['text2'], ha='left', va='bottom')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'course_map.png'), facecolor=COL['bg'])
    print('ok')

if __name__ == '__main__':
    main()
