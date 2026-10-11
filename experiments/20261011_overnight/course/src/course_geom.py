"""course.html (three.js viewer) の scene モジュールに書かれたコース形状を、平面図座標 [m] に写したもの。

座標: course.html の「公式図」座標そのまま。X = 幅方向 0〜10.30 m(左→右)、Z = 奥行き 0〜5.60 m(手前→奥)。
viewer 座標との関係は course.html と同じ x = X - 5.15, z = 2.8 - Z(three.js の y は高さ)。
上から見た図は X を右、Z を上に描く。走行は左回り(反時計回り)。

形状は course.html の buildCourse() の R()/B()/T()/L()/cloth() 呼び出しを手で写したもの
(数値は main.js のリテラルそのまま。frontStart / backStart / turfLeft など JS で計算している値は同じ式で計算)。
"""
import json, math
import numpy as np

COURSE = {'width': 10.30, 'depth': 5.60, 'railHeight': .089, 'railThickness': .019, 'lane': 1.40,
          'parkingDepth': .50, 'parkingLength': 1.24, 'lightHeight': .40, 'arrowHeight': .65,
          'tunnelWall': 1.33, 'tunnelPeak': 1.33 + .95 / math.sqrt(2)}

def jround(x):          # JS Math.round
    return int(math.floor(x + 0.5))

def v2p(x, z):          # viewer -> plan
    return [round(x + 5.15, 4), round(2.8 - z, 4)]

FRONT_START = 1.70 + math.sqrt(.95 ** 2 - (2.06 - 1.40) ** 2)   # 2.3833
BACK_START = 1.70 + math.sqrt(.95 ** 2 - (4.20 - 3.92) ** 2)    # 2.6078
TURF_LEFT = BACK_START + 3 * 1.86                                # 8.1878

def rails():
    """R(a, b, color, force) の全呼び出し。1枚 ≒ 1.86 m に分け、JS と同じ規則で赤/白を決める"""
    calls = []
    def R(a, b, c=0, f=False, group='', note=''):
        calls.append((list(map(float, a)), list(map(float, b)), c, f, group, note))
    # 外周
    R([.20, 5.60], [1.15, 5.60], group='outer'); R([1.15, 5.60], [3.01, 5.60], group='outer')
    R([3.01, 5.60], [4.87, 5.60], 1, group='outer', note='写真 img_0496/0522/0524 では、この赤い板が東端(LEDゲート支柱)へ向かって内側へ斜めに入っているように見える(未計測)')
    R([4.97, 5.60], [6.83, 5.60], group='outer'); R([6.83, 5.60], [8.69, 5.60], 1, group='outer'); R([8.69, 5.60], [9.55, 5.60], group='outer')
    R([9.55, 5.60], [10.30, 4.70], 1, group='outer'); R([10.30, 4.70], [10.30, 2.84], group='outer'); R([10.30, 2.84], [10.30, .85], 1, group='outer')
    R([10.30, .85], [9.85, 0], 1, group='outer'); R([9.85, 0], [8.0, 0], group='outer'); R([8.0, 0], [6.14, 0], group='outer')
    R([6.14, 0], [5.60, 0], 0, True, group='outer'); R([1.88, 0], [0, 0], 1, group='outer')
    R([0, 0], [0, 1.86], group='outer'); R([0, 1.86], [0, 3.72], 1, group='outer'); R([0, 3.72], [0, 4.65], group='outer'); R([0, 4.65], [.20, 5.60], 1, group='outer')
    R([.12, .78], [.78, .12], 1, group='outer_corner', note='左手前の外周の折り返しに見える斜めの板')
    # 中央の島
    R([1.70, 2.06], [1.70, 3.92], group='island', note='島の左端の板(1820 mm 板、支柱間 1860 mm)')
    R([1.70, 2.06], [FRONT_START, 1.40], 1, group='island'); R([1.70, 3.92], [BACK_START, 4.20], 1, group='island')
    for i in range(3):
        R([FRONT_START + i * 1.86, 1.40], [FRONT_START + (i + 1) * 1.86, 1.40], i % 2, group='rail_Z1.40')
        R([BACK_START + i * 1.86, 4.20], [BACK_START + (i + 1) * 1.86, 4.20], 1 if i % 2 else 0, group='rail_Z4.20')
    R([3.60, 2.80], [5.46, 2.80], 1, group='rail_Z2.80'); R([5.46, 2.80], [6.41, 2.80], group='rail_Z2.80', note='R1 で正面衝突する白い板(東端 X=6.41 に支柱)')
    R([7.51, 2.80], [9.37, 2.80], 1, group='rail_Z2.80')
    R([2.80, 4.90], [4.66, 4.90], group='narrow_lane_board', note='奥の通路を2本に分ける独立した白い板(写真では黒テープ Z=4.90 より外側=北に見える)')
    # 駐車区画(外周の外側 Z = -0.50〜0)
    for i in range(3):
        X, w = 2.50 + i * 1.24, 1.24
        R([X - w / 2, -.50], [X - w / 2, 0], group='parking')
        if i == 2: R([X + w / 2, -.50], [X + w / 2, 0], group='parking')
        R([X - w / 2, -.50], [X + w / 2, -.50], 0, True, group='parking')
    R([9.52, 5.56], [9.65, 4.62], 0, group='turf_angled_board', note='芝帯の斜めの板(course.html では1本だけ。写真 img_0529 では暗幕側にも赤い斜めの板が見える)')
    out = []
    for a, b, c, f, group, note in calls:
        dx, dz = b[0] - a[0], b[1] - a[1]; L = math.hypot(dx, dz); n = max(1, jround(L / 1.86))
        for i in range(n):
            f0, f1 = i / n, (i + 1) / n
            col = (c if f else (i + c) % 2)
            out.append({'a': [round(a[0] + dx * f0, 4), round(a[1] + dz * f0, 4)],
                        'b': [round(a[0] + dx * f1, 4), round(a[1] + dz * f1, 4)],
                        'color': 'red' if col else 'white', 'group': group, **({'note': note} if note else {})})
    return out

def posts(rs):
    seen, out = set(), []
    for r in rs:
        for p in (r['a'], r['b']):
            k = '%.4f,%.4f' % tuple(p)
            if k not in seen:
                seen.add(k); out.append(p)
    return out

def curtains():
    """黒い暗幕(高さ 1.33 m、屋根なし)。左(X=7.50)が入口で、中央のフラップで2レーンに分かれる"""
    start, sideEnd, end = 7.50, 9.45, 10.116
    segs = [((start, 2.80), (sideEnd, 2.80)), ((sideEnd, 2.80), (end, 1.925)), ((end, 1.925), (end, .875)),
            ((end, .875), (sideEnd, 0.0)), ((sideEnd, 0.0), (start, 0.0))]
    flap = [((start, 1.465), (start + .55, 1.465)), ((start + .55, 1.465), (start + .55, 1.335)), ((start + .55, 1.335), (start, 1.335))]
    return ([{'a': list(a), 'b': list(b), 'kind': 'wall'} for a, b in segs] +
            [{'a': list(a), 'b': list(b), 'kind': 'center_flap'} for a, b in flap])

def rect(x0, x1, z0, z1):
    return [[x0, z0], [x1, z0], [x1, z1], [x0, z1]]

def build():
    rs = rails()
    turf_x1 = TURF_LEFT + 1.85
    geo = {
        'meta': {
            'source': 'course.html (three.js viewer) の scene モジュール buildCourse() / COURSE / LANDMARKS を平面図座標に写したもの',
            'basis': "course.html 自身の記載: 'official-plan + measured mat + reference photos', estimated: true(公式図の寸法 + 青マット実測 + 写真から推定した位置)",
            'units': 'm',
            'frame': 'X = 幅方向 0〜10.30(左→右)、Z = 奥行き 0〜5.60(手前→奥)。原点は左手前の外周の支柱。'
                     'course.html の viewer 座標とは x = X - 5.15, z = 2.8 - Z。上面図は X 右、Z 上。',
            'driving_direction': 'counter-clockwise (上から見て左回り): 手前の直線を +X へ → 暗幕内で左Uターン → '
                                 'ショートカットで右Uターン → 芝で左Uターン → 奥の直線を -X へ → 左端で左Uターン',
            'created': '2026-10-11', 'generator': 'src/course_geom.py',
        },
        'course': COURSE,
        'rails': rs,
        'posts': posts(rs),
        'curtain_tunnel': {
            'walls': curtains(), 'height_m': 1.33, 'peak_m': round(COURSE['tunnelPeak'], 3), 'roof': False,
            'entrance': {'x': 7.50, 'z': [0.0, 2.80], 'note': '左側(X=7.50)が開いていて、中央のフラップ(X 7.50〜8.05, Z 1.335〜1.465)で手前レーン/2本目のレーンに分かれる'},
            'divider_tape': {'a': [FRONT_START + 3 * 1.86, 1.40], 'b': [10.116, 1.40], 'color': 'black'},
        },
        'areas': {
            'turf_low_mu': {'polygon': rect(TURF_LEFT, turf_x1, 2.80, 5.60), 'surface': '人工芝(外側)',
                            'lanes_note': 'Z 方向に 90 + 50 + 50 + 90 cm、X 方向に 100 + 85 cm',
                            'tape_green_z': [3.70, 4.70]},
            'ptfe_white_sheet': {'polygon': rect(TURF_LEFT, TURF_LEFT + 1.0, 3.70, 4.70), 'surface': '白い PTFE シート 1 × 1 m(内側、低μ)',
                                 'white_lines': [[[TURF_LEFT + .5, 3.70], [TURF_LEFT + .5, 4.70]], [[TURF_LEFT, 4.20], [TURF_LEFT + 1.0, 4.20]]]},
            'blue_start_mat': {'polygon': rect(1.70 - .85, 1.70, 2.95 - .80, 2.95 + .80), 'size_m': [.85, 1.60],
                               'note': '実測 85 × 160 cm、4枚のパネル。島の左端の板 (X=1.70) に接している'},
            'shortcut': {'polygon': rect(6.46, 7.46, 2.55, 3.05), 'tape_outline': rect(6.41, 7.51, 2.50, 3.10),
                         'raised_m': 0.01, 'ramp_m': 0.05,
                         'note': '中央のレール Z=2.80 の切れ目 (X 6.41〜7.51) に置いた 50 cm 角 2 枚の黒いスポンジ(1 cm 高)。この切れ目が R1 の右Uターン'},
            'parking': [
                {'id': 'P%d' % (i + 1), 'polygon': rect(2.50 + i * 1.24 - .62, 2.50 + i * 1.24 + .62, -.50, 0.0),
                 'color': c, 'center': [round(2.50 + i * 1.24, 3), -.25]}
                for i, c in enumerate(['#10b99a', '#f14b41', '#526bd8'])],
            'start_lines': [{'a': [x, .02], 'b': [x, 1.38], 'width_m': .05, 'color': 'white'} for x in (2.50, 4.30, 6.10)],
            'lane_tape_Z4.90': {'a': [.3, 4.90], 'b': [7.4, 4.90], 'color': 'black', 'width_m': .045},
            'carpet_grid_tape': {'z': [5.45, 3.63, 1.81, -0.01], 'x': [1.25, 5.05, 8.85],
                                 'note': '部屋のカーペットの黒テープ(コースの一部ではないがカメラに写る)'},
        },
        'gates': {
            'light_gate': {'x': 2.20, 'posts': [[2.20, 1.40], [2.20, 4.20]], 'crossbar_m': .58, 'light': [2.20, 2.80], 'light_height_m': .40,
                           'floor_spots_center': [2.50, 2.80], 'floor_spots_radius_m': .9,
                           'note': '島の左端の内側(2本目と3本目のレーンをつなぐ折り返し)の上。左回りの通常周回(ショートカット経由)では通らない'},
            'led_arrow_gate': {'x': 4.90, 'posts': [[4.90, 4.20], [4.90, 5.60]], 'panel_center': [4.90, 4.90], 'panel_m': [.93, .19],
                               'panel_height_m': [.42, .62], 'post_height_m': .65,
                               'facing_model': '-X (course.html では表示面が西向き)',
                               'facing_photos': '+X (写真 img_0515 で東側から緑の矢印が見え、img_0522「狭い道と矢印ゲート裏側」/img_0496 では西側から裏の基板が見える)',
                               'note': '奥の直線を -X へ走る車から正面に見える。北側の支柱が外周の板の継ぎ目(X 4.87〜4.97)に立っている = DE の衝突地点'},
        },
        'landmarks': [],
    }
    for lm in [
        {'id': 'parking', 'title': 'P1 / P2 / P3', 'subtitle': '50 × 124 cm・3色の駐車区画', 'position': [-1.41, .1, 3.05], 'photo': 'img_0537.jpg'},
        {'id': 'blue', 'title': '青いマット', 'subtitle': '実寸85 × 160 cm・4枚のパネル', 'position': [-3.875, .07, -.15], 'photo': 'img_0523.jpg'},
        {'id': 'light', 'title': '照明ゲート', 'subtitle': '光源高さ40 cm', 'position': [-2.95, .50, 0], 'photo': 'img_0530.jpg'},
        {'id': 'arrow', 'title': '矢印ゲート', 'subtitle': '支柱65 cm・表示93 × 19 cm', 'position': [-.25, .65, -2.1], 'photo': 'img_0515.jpg'},
        {'id': 'grass', 'title': '芝生・低μ路', 'subtitle': '外側人工芝・内側白いPTFEシート', 'position': [3.95, .08, -1.5], 'photo': 'img_0529.jpg'},
        {'id': 'tunnel', 'title': '黒い遮蔽エリア', 'subtitle': '壁高さ133 cm・屋根なし', 'position': [3.45, 1.40, 1.4], 'photo': 'img_0534.jpg'}]:
        x, y, z = lm['position']
        geo['landmarks'].append({**{k: v for k, v in lm.items() if k != 'position'}, 'plan_xz': v2p(x, z), 'height_m': y, 'viewer_xyz': lm['position']})
    return geo

# ---------------------------------------------------------------- 左回りの基準線(進行度 = 強化学習の報酬用)
def _arc(c, r, a0, a1, n):
    a = np.radians(np.linspace(a0, a1, n))
    return np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 1)

def _line(p, q, n):
    t = np.linspace(0, 1, n)[:, None]
    return np.asarray(p) * (1 - t) + np.asarray(q) * t

def _smooth(pts, it=3):
    """Chaikin(端は固定)"""
    P = np.asarray(pts, float)
    for _ in range(it):
        Q = [P[0]]
        for a, b in zip(P[:-1], P[1:]):
            Q += [.75 * a + .25 * b, .25 * a + .75 * b]
        Q.append(P[-1]); P = np.array(Q)
    return P

CENTERLINE_SECTIONS = [
    # id, 名前, 説明
    ('HS', 'ホームストレート', '手前のレーン(Z 0〜1.40)を +X へ。床の白いスタートライン 3 本、正面に黒い暗幕'),
    ('LH', '暗幕内の左ヘアピン', '暗幕の中で中央フラップの先を左に 180° 回る'),
    ('R1', 'ショートカット右ヘアピン', 'Z=2.80 のレールの切れ目(黒いスポンジ)を右に 180° 回る。外側は白い板(X 5.46〜6.41)'),
    ('GM', '芝(緑マット)左ヘアピン', 'Z=4.20 のレールの端 (X=8.19) を芝と白い PTFE の上で左に 180° 回る'),
    ('DE', 'バックストレート', '奥のレーン(Z 4.20〜5.60)を -X へ。右が外周、正面に LED 矢印ゲート。終端(X≈4.9)の北側支柱が DE 衝突点'),
    ('SW', '奥の狭い道〜左の折り返し', '独立した白い板(Z=4.90)の外側の狭い道を通り、青マットの横を左に 180° 回って手前のレーンへ'),
]

def centerline(ds=0.02, narrow='outer'):
    """各レーンの中央を通る左回りの閉じた基準線(実際の走行ラインではない)。
    戻り値: dict(xz [N,2], s [N] (m), heading [N] (rad, +X から反時計回り), section [N] (int), length)"""
    zn = 5.25 if narrow == 'outer' else 4.55
    segs = []
    segs.append(('HS', _line([2.45, 0.70], [8.05, 0.70], 200)))
    segs.append(('LH', _arc([8.05, 1.40], 0.70, -90, 90, 120)))
    segs.append(('R1', np.vstack([_line([8.05, 2.10], [7.51, 2.10], 20)[:-1], _arc([7.51, 2.80], 0.70, -90, -270, 120)])))
    segs.append(('GM', np.vstack([_line([7.51, 3.50], [8.19, 3.50], 20)[:-1], _arc([8.19, 4.20], 0.70, -90, 90, 120)])))
    de = _smooth([[8.19, 4.90], [6.20, 4.90], [5.30, 4.95], [4.90, zn - 0.05], [4.66, zn]], 3)
    segs.append(('DE', de))
    sw = _smooth([[4.66, zn], [2.80, zn], [2.05, 5.10], [1.15, 4.55], [0.85, 3.80], [0.85, 2.10],
                  [1.10, 1.25], [1.70, 0.80], [2.45, 0.70]], 4)
    segs.append(('SW', sw))
    pts, sec = [], []
    for k, (name, p) in enumerate(segs):
        p = np.asarray(p, float)
        if pts:
            p = p[1:]
        pts.append(p); sec += [k] * len(p)
    P = np.vstack(pts); sec = np.array(sec)
    # 等間隔に引き直す
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    L = seg[-1]
    s = np.arange(0, L, ds)
    x = np.interp(s, seg, P[:, 0]); z = np.interp(s, seg, P[:, 1])
    secs = sec[np.clip(np.searchsorted(seg, s, side='right') - 1, 0, len(sec) - 1)]
    xz = np.stack([x, z], 1)
    d = np.roll(xz, -1, 0) - np.roll(xz, 1, 0)
    hd = np.arctan2(d[:, 1], d[:, 0])
    return {'xz': xz, 's': s, 'heading': hd, 'section': secs, 'length': float(L),
            'section_ids': [c[0] for c in CENTERLINE_SECTIONS]}

def wall_segments(geo, include_parking=True):
    segs = [(r['a'], r['b']) for r in geo['rails'] if include_parking or r['group'] != 'parking']
    segs += [(w['a'], w['b']) for w in geo['curtain_tunnel']['walls']]
    return np.array(segs, float)    # [M, 2(a,b), 2(x,z)]

def dist_to_segments(P, S):
    """点 P [N,2] から線分 S [M,2,2] までの最短距離 [N] と最も近い線分の番号"""
    A = S[:, 0][None]; B = S[:, 1][None]; Q = P[:, None]
    AB = B - A; t = np.clip(((Q - A) * AB).sum(-1) / np.maximum((AB * AB).sum(-1), 1e-12), 0, 1)
    C = A + t[..., None] * AB
    dd = np.hypot(*(Q - C).transpose(2, 0, 1))
    k = dd.argmin(1)
    return dd[np.arange(len(P)), k], k

def crossings(P, S):
    """折れ線 P が線分 S を横切る回数(壁抜けの検出)。返り値: 横切った P の区間番号の配列"""
    p = P[:-1][:, None]; r = (P[1:] - P[:-1])[:, None]
    q = S[:, 0][None]; s = (S[:, 1] - S[:, 0])[None]
    cr = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    rxs = cr(r, s); qp = q - p
    with np.errstate(divide='ignore', invalid='ignore'):
        t = cr(qp, s) / rxs; u = cr(qp, r) / rxs
    hit = (np.abs(rxs) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return np.nonzero(hit.any(1))[0]
