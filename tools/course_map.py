#!/usr/bin/env python3
"""コース図(2026-10-10 のコース、上から見た図)と、当てはめに使う幾何の道具

形状は course.html(three.js の表示)の buildCourse() を平面図の座標に写したもの。夜間作業の
jetformer_reports/overnight_20261011/course/src/course_geom.py と同じ数値・同じ式で、走行ログの当てはめ
(tools/course_fit.py)から使えるように、リポジトリの中に置き直した。

座標: X = 幅方向 0〜10.30 m(左→右)、Z = 奥行き 0〜5.60 m(手前→奥)。原点は左手前の外周の支柱。
向きは +X から反時計回り(rad)。走行は左回り(反時計回り)。
形状は「公式図 + 青マットの実測 + 写真」からの推定で、数 cm〜10 cm 程度のずれはありうる。

    python tools/course_map.py      # 基準線の長さと区間を表示する
"""

import math

import numpy as np

COURSE = {'width': 10.30, 'depth': 5.60, 'lane': 1.40}

FRONT_START = 1.70 + math.sqrt(.95 ** 2 - (2.06 - 1.40) ** 2)   # 2.3833
BACK_START = 1.70 + math.sqrt(.95 ** 2 - (4.20 - 3.92) ** 2)    # 2.6078
TURF_LEFT = BACK_START + 3 * 1.86                                # 8.1878

# カメラに写る目印(床にあるもの)。axis の座標が value の線を、heading の向きに走っているときに横切る
# rows: 画面の下端に届く瞬間を使う。region: その線を横切るときに車がいるはずの範囲(X0, X1, Z0, Z1)
LANDMARKS = {
    'line': {'axis': 'x', 'values': [2.50, 4.30, 6.10], 'heading': 0.0, 'region': (1.6, 7.0, -0.2, 1.8),
             'note': '手前の直線の白いスタートライン 3 本(幅 5 cm、Z 0.02〜1.38)'},
    'turf_in': {'axis': 'x', 'values': [TURF_LEFT], 'heading': 0.0, 'region': (6.8, 8.6, 2.8, 4.2),
                'note': '芝の左端(X=8.19)。R1 を出て +X へ芝に入るとき'},
    'turf_out': {'axis': 'x', 'values': [TURF_LEFT], 'heading': math.pi, 'region': (7.6, 9.8, 4.2, 5.6),
                 'note': '芝の左端(X=8.19)。GM を出て -X へ芝から出るとき'},
    'mat_n': {'axis': 'z', 'values': [2.95 + .80], 'heading': -math.pi / 2, 'region': (0.0, 2.0, 3.0, 5.4),
              'note': '青マットの奥の端(Z=3.75)。左の折り返しを -Z へ走るとき(マットは車の左)'},
    'mat_s': {'axis': 'z', 'values': [2.95 - .80], 'heading': -math.pi / 2, 'region': (0.0, 2.0, 1.4, 3.8),
              'note': '青マットの手前の端(Z=2.15)'},
    'gate': {'axis': 'x', 'values': [4.90], 'heading': math.pi, 'region': (4.6, 8.8, 4.0, 5.6),
             'note': 'LED 矢印ゲート(X=4.90)の表示が画面の上端より上へ抜ける瞬間(床ではなく高さ 0.42〜0.62 m)'},
}
GATE_PANEL = {'x': 4.90, 'z': 4.90, 'half_width': .465}


def jround(x):
    """JS の Math.round"""
    return int(math.floor(x + 0.5))


def rails():
    """R(a, b, color, force) の全呼び出し(course_geom.py と同じ)。1枚 ≒ 1.86 m に分ける"""
    calls = []

    def R(a, b, c=0, f=False, group=''):
        calls.append((list(map(float, a)), list(map(float, b)), c, f, group))
    # 外周
    for a, b, c in [([.20, 5.60], [1.15, 5.60], 0), ([1.15, 5.60], [3.01, 5.60], 0), ([3.01, 5.60], [4.87, 5.60], 1),
                    ([4.97, 5.60], [6.83, 5.60], 0), ([6.83, 5.60], [8.69, 5.60], 1), ([8.69, 5.60], [9.55, 5.60], 0),
                    ([9.55, 5.60], [10.30, 4.70], 1), ([10.30, 4.70], [10.30, 2.84], 0), ([10.30, 2.84], [10.30, .85], 1),
                    ([10.30, .85], [9.85, 0], 1), ([9.85, 0], [8.0, 0], 0), ([8.0, 0], [6.14, 0], 0)]:
        R(a, b, c, group='outer')
    R([6.14, 0], [5.60, 0], 0, True, group='outer')
    R([1.88, 0], [0, 0], 1, group='outer')
    R([0, 0], [0, 1.86], group='outer'); R([0, 1.86], [0, 3.72], 1, group='outer')
    R([0, 3.72], [0, 4.65], group='outer'); R([0, 4.65], [.20, 5.60], 1, group='outer')
    R([.12, .78], [.78, .12], 1, group='outer_corner')
    # 中央の島
    R([1.70, 2.06], [1.70, 3.92], group='island')
    R([1.70, 2.06], [FRONT_START, 1.40], 1, group='island'); R([1.70, 3.92], [BACK_START, 4.20], 1, group='island')
    for i in range(3):
        R([FRONT_START + i * 1.86, 1.40], [FRONT_START + (i + 1) * 1.86, 1.40], i % 2, group='rail_Z1.40')
        R([BACK_START + i * 1.86, 4.20], [BACK_START + (i + 1) * 1.86, 4.20], 1 if i % 2 else 0, group='rail_Z4.20')
    R([3.60, 2.80], [5.46, 2.80], 1, group='rail_Z2.80'); R([5.46, 2.80], [6.41, 2.80], group='rail_Z2.80')
    R([7.51, 2.80], [9.37, 2.80], 1, group='rail_Z2.80')
    R([2.80, 4.90], [4.66, 4.90], group='narrow_lane_board')
    # 駐車区画(外周の外側 Z = -0.50〜0)
    for i in range(3):
        X, w = 2.50 + i * 1.24, 1.24
        R([X - w / 2, -.50], [X - w / 2, 0], group='parking')
        if i == 2:
            R([X + w / 2, -.50], [X + w / 2, 0], group='parking')
        R([X - w / 2, -.50], [X + w / 2, -.50], 0, True, group='parking')
    R([9.52, 5.56], [9.65, 4.62], 0, group='turf_angled_board')
    out = []
    for a, b, c, f, group in calls:
        dx, dz = b[0] - a[0], b[1] - a[1]
        n = max(1, jround(math.hypot(dx, dz) / 1.86))
        for i in range(n):
            f0, f1 = i / n, (i + 1) / n
            out.append({'a': [round(a[0] + dx * f0, 4), round(a[1] + dz * f0, 4)],
                        'b': [round(a[0] + dx * f1, 4), round(a[1] + dz * f1, 4)],
                        'color': 'red' if (c if f else (i + c) % 2) else 'white', 'group': group})
    return out


def curtains():
    """黒い暗幕(高さ 1.33 m)。左(X=7.50)が入口で、中央のフラップで2レーンに分かれる"""
    start, side_end, end = 7.50, 9.45, 10.116
    segs = [((start, 2.80), (side_end, 2.80)), ((side_end, 2.80), (end, 1.925)), ((end, 1.925), (end, .875)),
            ((end, .875), (side_end, 0.0)), ((side_end, 0.0), (start, 0.0))]
    flap = [((start, 1.465), (start + .55, 1.465)), ((start + .55, 1.465), (start + .55, 1.335)),
            ((start + .55, 1.335), (start, 1.335))]
    return ([{'a': list(a), 'b': list(b), 'kind': 'wall'} for a, b in segs] +
            [{'a': list(a), 'b': list(b), 'kind': 'center_flap'} for a, b in flap])


def build():
    """壁と区域(当てはめと図に使う分だけ)"""
    return {'rails': rails(), 'curtains': curtains(),
            'turf': [[TURF_LEFT, 2.80], [TURF_LEFT + 1.85, 2.80], [TURF_LEFT + 1.85, 5.60], [TURF_LEFT, 5.60]],
            'ptfe': [[TURF_LEFT, 3.70], [TURF_LEFT + 1.0, 3.70], [TURF_LEFT + 1.0, 4.70], [TURF_LEFT, 4.70]],
            'blue_mat': [[.85, 2.15], [1.70, 2.15], [1.70, 3.75], [.85, 3.75]],
            'shortcut': [[6.46, 2.55], [7.46, 2.55], [7.46, 3.05], [6.46, 3.05]],
            'start_lines': [[[x, .02], [x, 1.38]] for x in LANDMARKS['line']['values']],
            'gate': [[4.90, 4.20], [4.90, 5.60]]}


def wall_segments(geo=None, include_parking=True, two_sided=False):
    """壁(板と暗幕)の線分 [M, 2(a,b), 2(x,z)]。two_sided=True なら「両側を走れる壁」(奥の狭い道の仕切り板・
    暗幕の中央フラップ・駐車区画の仕切り)の印 [M] も返す"""
    geo = geo or build()
    rs = [r for r in geo['rails'] if include_parking or r['group'] != 'parking']
    segs = [(r['a'], r['b']) for r in rs] + [(w['a'], w['b']) for w in geo['curtains']]
    ts = [r['group'] in ('narrow_lane_board', 'parking') for r in rs] + [w['kind'] == 'center_flap' for w in geo['curtains']]
    if two_sided:
        return np.array(segs, float), np.array(ts, bool)
    return np.array(segs, float)


# ---------------------------------------------------------------- 左回りの基準線(各レーンの中央。進みの座標)
SECTIONS = [
    ('HS', 'ホームストレート', '手前のレーン(Z 0〜1.40)を +X へ。床の白いスタートライン 3 本、正面に黒い暗幕'),
    ('LH', '暗幕内の左ヘアピン', '暗幕の中で中央フラップの先を左に 180° 回る'),
    ('R1', 'ショートカット右ヘアピン', 'Z=2.80 のレールの切れ目(黒いスポンジ)を右に 180° 回る'),
    ('GM', '芝(緑マット)左ヘアピン', 'Z=4.20 のレールの端 (X=8.19) を芝と白い PTFE の上で左に 180° 回る'),
    ('DE', 'バックストレート', '奥のレーンを -X へ。右が外周、正面に LED 矢印ゲート。終端(X≈4.9)の北側支柱が DE 衝突点'),
    ('SW', '奥の狭い道〜左の折り返し', '奥の狭い道を通り、青マットの横を左に 180° 回って手前のレーンへ'),
]
SECTION_IDS = [s[0] for s in SECTIONS]


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
        Q.append(P[-1])
        P = np.array(Q)
    return P


def centerline(ds=0.02, narrow='outer'):
    """各レーンの中央を通る左回りの閉じた基準線(実際の走行ラインではない)。1周 約 25.48 m。
    戻り値: dict(xz [N,2], s [N] (m), heading [N] (rad), section [N] (int), length)。s=0 はホームストレートの入口"""
    zn = 5.25 if narrow == 'outer' else 4.55
    segs = [('HS', _line([2.45, 0.70], [8.05, 0.70], 200)),
            ('LH', _arc([8.05, 1.40], 0.70, -90, 90, 120)),
            ('R1', np.vstack([_line([8.05, 2.10], [7.51, 2.10], 20)[:-1], _arc([7.51, 2.80], 0.70, -90, -270, 120)])),
            ('GM', np.vstack([_line([7.51, 3.50], [8.19, 3.50], 20)[:-1], _arc([8.19, 4.20], 0.70, -90, 90, 120)])),
            ('DE', _smooth([[8.19, 4.90], [6.20, 4.90], [5.30, 4.95], [4.90, zn - 0.05], [4.66, zn]], 3)),
            ('SW', _smooth([[4.66, zn], [2.80, zn], [2.05, 5.10], [1.15, 4.55], [0.85, 3.80], [0.85, 2.10],
                            [1.10, 1.25], [1.70, 0.80], [2.45, 0.70]], 4))]
    pts, sec = [], []
    for k, (_, p) in enumerate(segs):
        p = np.asarray(p, float)
        if pts:
            p = p[1:]
        pts.append(p)
        sec += [k] * len(p)
    P = np.vstack(pts)
    sec = np.array(sec)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    L = seg[-1]
    s = np.arange(0, L, ds)
    xz = np.stack([np.interp(s, seg, P[:, 0]), np.interp(s, seg, P[:, 1])], 1)
    secs = sec[np.clip(np.searchsorted(seg, s, side='right') - 1, 0, len(sec) - 1)]
    d = np.roll(xz, -1, 0) - np.roll(xz, 1, 0)
    return {'xz': xz, 's': s, 'heading': np.arctan2(d[:, 1], d[:, 0]), 'section': secs, 'length': float(L),
            'section_ids': list(SECTION_IDS)}


def section_bounds(cl=None):
    """区間ごとの基準線上の範囲 {id: (s0, s1)} [m]"""
    cl = cl or centerline()
    out = {}
    for k, name in enumerate(SECTION_IDS):
        s = cl['s'][cl['section'] == k]
        out[name] = (float(s.min()), float(s.max()))
    return out


def wrap(a):
    """角度を -π〜π に"""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def dist_to_segments(P, S):
    """点 P [N,2] から線分 S [M,2,2] までの最短距離 [N] と、最も近い線分の番号"""
    P = np.asarray(P, float)
    A = S[:, 0][None]
    B = S[:, 1][None]
    Q = P[:, None]
    AB = B - A
    t = np.clip(((Q - A) * AB).sum(-1) / np.maximum((AB * AB).sum(-1), 1e-12), 0, 1)
    C = A + t[..., None] * AB
    dd = np.hypot(*(Q - C).transpose(2, 0, 1))
    k = dd.argmin(1)
    return dd[np.arange(len(P)), k], k


def crossings(P, S):
    """折れ線 P [N,2] が線分 S を横切った区間の番号(壁の突き抜けの検出)"""
    P = np.asarray(P, float)
    if len(P) < 2:
        return np.zeros(0, int)
    p = P[:-1][:, None]
    r = (P[1:] - P[:-1])[:, None]
    q = S[:, 0][None]
    s = (S[:, 1] - S[:, 0])[None]

    def cr(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    rxs = cr(r, s)
    qp = q - p
    with np.errstate(divide='ignore', invalid='ignore'):
        t = cr(qp, s) / rxs
        u = cr(qp, r) / rxs
    hit = (np.abs(rxs) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return np.nonzero(hit.any(1))[0]


def count_crossings(X, Z, S, valid=None, min_sep=1):
    """軌跡が壁を横切った回数。valid=False の点で軌跡を切る(区間のつなぎ目は数えない)"""
    X = np.asarray(X, float)
    Z = np.asarray(Z, float)
    ok = np.isfinite(X) & np.isfinite(Z)
    if valid is not None:
        ok &= np.asarray(valid, bool)
    n = 0
    idx = np.nonzero(ok)[0]
    if not len(idx):
        return 0
    cuts = np.nonzero(np.diff(idx) > min_sep)[0]
    for part in np.split(idx, cuts + 1):
        if len(part) >= 2:
            n += len(crossings(np.stack([X[part], Z[part]], 1), S))
    return n


class Centerline(object):
    """基準線への対応づけ(向きの合う最も近い点)と、進み(s)の計算"""

    def __init__(self, cl=None):
        cl = cl or centerline()
        self.cl = cl
        self.xz = cl['xz']
        self.h = cl['heading']
        self.s = cl['s']
        self.L = cl['length']
        self.sec = cl['section']
        d = np.roll(self.xz, -1, 0) - self.xz
        self.t = d / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
        self.n = np.stack([-self.t[:, 1], self.t[:, 0]], 1)    # 左向きの法線

    def match(self, P, H, max_dh=np.radians(60), max_d=2.0, chunk=400):
        """戻り値: idx(-1 = 対応なし)、符号付きの横ずれ(左 +)[m]"""
        P = np.asarray(P, float)
        H = np.asarray(H, float)
        idx = np.full(len(P), -1)
        off = np.full(len(P), np.nan)
        for a in range(0, len(P), chunk):
            p = P[a:a + chunk]
            h = H[a:a + chunk]
            good = np.isfinite(p).all(1) & np.isfinite(h)
            dd = np.hypot(p[:, None, 0] - self.xz[None, :, 0], p[:, None, 1] - self.xz[None, :, 1])
            dh = np.abs(wrap(h[:, None] - self.h[None]))
            dd[dh > max_dh] = np.inf
            dd[~good] = np.inf
            k = dd.argmin(1)
            dk = dd[np.arange(len(p)), k]
            ok = np.isfinite(dk) & (dk < max_d)
            k[~ok] = -1
            idx[a:a + chunk] = k
            kk = k[ok]
            off[a:a + chunk][ok] = ((p[ok] - self.xz[kk]) * self.n[kk]).sum(1)
        return idx, off

    def progress(self, X, Z, H, valid=None, max_dh=np.radians(75), max_d=2.5):
        """進み(1周の中の s、周をまたいで積算した s、周番号、区間、横ずれ)。
        valid=False のところと、区間が切れたところ(segment が変わる)では積算をつながない呼び方をすること"""
        P = np.stack([X, Z], 1)
        idx, off = self.match(P, H, max_dh=max_dh, max_d=max_d)
        sp = np.where(idx >= 0, self.s[np.clip(idx, 0, None)], np.nan)
        sec = np.where(idx >= 0, self.sec[np.clip(idx, 0, None)], -1)
        if valid is None:
            valid = np.isfinite(sp)
        cum = np.full(len(sp), np.nan)
        lap = np.zeros(len(sp), int)
        k = 0
        last = None
        for i in range(len(sp)):
            if not valid[i] or np.isnan(sp[i]):
                continue
            if last is not None:
                dsp = sp[i] - last
                if dsp < -self.L / 2:
                    k += 1
                elif dsp > self.L / 2:
                    k -= 1
            last = sp[i]
            lap[i] = k
            cum[i] = k * self.L + sp[i]
        return sp, cum, lap, sec, off

    def at(self, s):
        """基準線上の進み s [m](周回は折り返す)の点と向き"""
        s = np.mod(np.asarray(s, float), self.L)
        x = np.interp(s, self.s, self.xz[:, 0], period=self.L)
        z = np.interp(s, self.s, self.xz[:, 1], period=self.L)
        c = np.interp(s, self.s, np.cos(self.h), period=self.L)
        si = np.interp(s, self.s, np.sin(self.h), period=self.L)
        return np.stack([x, z], -1), np.arctan2(si, c)


if __name__ == '__main__':
    c = centerline()
    print('基準線 %.2f m' % c['length'])
    for k, (a, b) in section_bounds(c).items():
        print('  %s %.2f〜%.2f m' % (k, a, b))
