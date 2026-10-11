"""推測航法(ジャイロ z + ホールセンサーの距離)で車の軌跡を作り、コース図に合わせる。

1. 向き ψ = ∫(gz - バイアス)。バイアスは止まっている間(|eRPM|<1 かつ |gz|<5°/s)の中央値。
2. 距離 s = ホールセンサーの積算 × 5.0 mm(= 0.5e-3 m/s/eRPM。hall の速さは 0.100 × eRPM [回/s])。
3. 周回の区切り = R1(1周に1回だけの鋭い右ヘアピン)の開始。
4. 補正(周ごとのドリフト補正):
     ψ'(t) = g·ψ(t) + A(t) + Σ c_j·H(t - τ_j)    A(t): 周の区切りを節点とする折れ線、c_j: ジャイロが振り切れた衝突での段差
     p(t)  = σ ∫ ds·[cos ψ', sin ψ'] + B(t)        B(t): 同じ節点の折れ線(位置のずれ)
   g(ジャイロの倍率)・σ(距離の倍率)は区間で共通。
5. コース図の基準線(各レーンの中央)へ、進む向きが合う点(差 < 60°)だけを対応させ、
   点と線の距離(Huber)を最小化する(ICP + ガウス・ニュートン、ヤコビアンは数値微分)。
   回転・平行移動・縮尺は A, B, σ に含まれる。
"""
import numpy as np
from course_geom import dist_to_segments, crossings

KH = 5.0e-3

def still_bias(it, gz, vt, erpm):
    er = np.interp(it, vt, erpm)
    st = (np.abs(er) < 1) & (np.abs(gz) < 5)
    return (float(np.median(gz[st])) if st.sum() > 50 else 0.0), int(st.sum())

def raw_dr(d):
    gz = d.gyr[:, 2].astype(float)
    bias, nst = still_bias(d.it, gz, d.vt, d.erpm)
    dt = np.clip(np.diff(d.it, prepend=d.it[0]), 0, 0.05)
    psi = np.cumsum((gz - bias) * dt)                      # deg
    dh = -np.diff(d.hall).astype(float); vdt = np.diff(d.vt)
    bad = (np.abs(dh) > 60) | (vdt <= 0) | (vdt > 0.1)    # ボードの再起動などで hall が飛んだところは eRPM で埋める
    dh[bad] = (0.1 * (-d.erpm[1:]) * np.clip(vdt, 0, 0.1))[bad]
    sv = np.r_[0, np.cumsum(dh)] * KH
    s = np.interp(d.it, d.vt, sv)
    return psi, s, bias, nst, int(bad.sum())

def r1_starts(t, psi, yaw_thr=-150.0, depth=90.0, gap=3.0):
    gz = np.gradient(psi, t)
    gzs = np.convolve(gz, np.ones(5) / 5, 'same')
    cand = np.nonzero(gzs < yaw_thr)[0]
    out, last_right, i = [], -1e9, 0
    while i < len(cand):
        j = cand[i]; a = j
        while a > 0 and gzs[a - 1] < -30: a -= 1
        b = j
        while b < len(gzs) - 1 and gzs[b + 1] < -30: b += 1
        dep = psi[a] - psi[b]
        if dep >= depth and t[a] - last_right > gap:
            out.append(a)
        if dep >= 30: last_right = t[b]
        while i < len(cand) and cand[i] <= b: i += 1
    return np.array(out, int)

class Centerline:
    def __init__(self, cl):
        self.xz = cl['xz']; self.h = cl['heading']; self.s = cl['s']; self.L = cl['length']; self.sec = cl['section']
        d = np.roll(self.xz, -1, 0) - self.xz
        self.t = d / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
        self.n = np.stack([-self.t[:, 1], self.t[:, 0]], 1)    # 左向きの法線

    def match(self, P, H, max_dh=np.radians(60), max_d=2.0, chunk=400):
        """向きの合う最も近い点。戻り値: idx(-1 = 対応なし), 符号付き横ずれ(左 +)"""
        idx = np.full(len(P), -1); off = np.full(len(P), np.nan)
        for a in range(0, len(P), chunk):
            p = P[a:a + chunk]; h = H[a:a + chunk]
            dd = np.hypot(p[:, None, 0] - self.xz[None, :, 0], p[:, None, 1] - self.xz[None, :, 1])
            dh = np.abs(np.angle(np.exp(1j * (h[:, None] - self.h[None]))))
            dd[dh > max_dh] = np.inf
            k = dd.argmin(1); ok = np.isfinite(dd[np.arange(len(p)), k]) & (dd[np.arange(len(p)), k] < max_d)
            k[~ok] = -1; idx[a:a + chunk] = k
            kk = k[ok]
            off[a:a + chunk][ok] = ((p[ok] - self.xz[kk]) * self.n[kk]).sum(1)
        return idx, off

class Fitter:
    """パラメータ x = [g, sigma, A_0..A_{K-1}, Bx_0.., Bz_0.., c_0..c_{J-1}, D_0..D_{G-1}]
    g: ジャイロの倍率, sigma: 距離の倍率, A/B: 周の区切りを節点とする向き/位置の補正(折れ線),
    c: ジャイロが振り切れた衝突での向きの段差, D: カメラの目印ごとの「画面の端まで」の距離(グループごとに共通の未知数)"""
    def __init__(self, t, psi, s, knots, steps, anchor_idx, cl, use, g0=1.0, g_sd=0.007, n_groups=0, speed_k=False, k_sd=0.2):
        self.t = t; self.psi = np.radians(psi); self.ds = np.diff(s, prepend=s[0])
        self.speed_k = speed_k; self.k_sd = k_sd
        v = np.gradient(s, t); self.vraw = np.convolve(v, np.ones(15) / 15, 'same') - 2.5   # 距離の倍率を速さで変える場合の基準 2.5 m/s
        self.T = np.asarray(knots, float); self.tau = np.asarray(steps, float)
        self.K = len(self.T); self.J = len(self.tau); self.G = n_groups
        self.cl = cl; self.use = use; self.g0 = g0; self.g_sd = g_sd
        self.H = (t[None, :] >= self.tau[:, None]).astype(float) if self.J else np.zeros((0, len(t)))
        self.anchor_idx = anchor_idx
        self.ev = []          # (idx, target_x, group, weight)
        self.D0 = np.zeros(n_groups); self.D_sd = np.ones(n_groups)
        self.wk = np.zeros(0, int); self.wall_w = 0.0; self.cw = 1.0; self.dz = 0.45; self.dzw = 0.0

    def unpack(self, x):
        K, J = self.K, self.J
        o = 2 + 3 * K
        return x[0], x[1], x[2:2 + K], x[2 + K:2 + 2 * K], x[2 + 2 * K:o], x[o:o + J], x[o + J:o + J + self.G]

    def kval(self, x):
        return x[-1] if self.speed_k else 0.0

    def set_events(self, group, idx, target_x, D0, D_sd, w=1 / 0.05):
        """カメラの目印: 時刻 idx の車の X + D_group·cos(向き) = target_x"""
        self.ev = [e for e in self.ev if e[2] != group]
        if len(idx):
            self.ev.append((np.asarray(idx, int), np.asarray(target_x, float), group, w))
        self.D0[group] = D0; self.D_sd[group] = D_sd

    def pose(self, x):
        g, sg, A, Bx, Bz, c, D = self.unpack(x)
        a = np.interp(self.t, self.T, A) + (c @ self.H if self.J else 0)
        h = g * self.psi + a
        hm = np.r_[h[0], 0.5 * (h[1:] + h[:-1])]
        ds = self.ds * (1.0 + self.kval(x) * self.vraw) if self.speed_k else self.ds
        X = sg * np.cumsum(ds * np.cos(hm)) + np.interp(self.t, self.T, Bx)
        Z = sg * np.cumsum(ds * np.sin(hm)) + np.interp(self.t, self.T, Bz)
        return X, Z, h

    def init(self, anchors_xz, anchor_h):
        K = self.K
        x = np.zeros(2 + 3 * K + self.J + self.G + (1 if self.speed_k else 0)); x[0] = self.g0; x[1] = 1.0
        x[2 + 3 * K + self.J:2 + 3 * K + self.J + self.G] = self.D0
        ka = [int(np.argmin(np.abs(self.T - self.t[i]))) for i in self.anchor_idx]
        A = np.full(K, np.nan)
        for k, i in zip(ka, self.anchor_idx):
            A[k] = anchor_h - self.g0 * self.psi[i]
        good = np.isfinite(A)
        A = np.interp(np.arange(K), np.nonzero(good)[0], np.unwrap(A[good]))
        x[2:2 + K] = A
        X, Z, _ = self.pose(x)
        Bx = np.full(K, np.nan); Bz = np.full(K, np.nan)
        for k, i in zip(ka, self.anchor_idx):
            Bx[k] = anchors_xz[0] - X[i]; Bz[k] = anchors_xz[1] - Z[i]
        x[2 + K:2 + 2 * K] = np.interp(np.arange(K), np.nonzero(good)[0], Bx[good])
        x[2 + 2 * K:2 + 3 * K] = np.interp(np.arange(K), np.nonzero(good)[0], Bz[good])
        return x

    def wall_setup(self, P, idx, walls, margin=0.10, look=0.30):
        """壁の罰則の対応づけ: 各点の最寄りの壁と、その壁のどちら側が通路か(対応する基準線の点の側)"""
        self.walls = walls; self.margin = margin
        d, k = dist_to_segments(P, walls)
        sel = (d < look) & (idx >= 0)
        self.wk = np.where(sel)[0]; self.wseg = k[sel]
        A = walls[self.wseg, 0]; B = walls[self.wseg, 1]; C = self.cl.xz[idx[sel]]
        cr = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
        self.wside = np.sign(cr)

    def wall_res(self, P):
        if not len(self.wk):
            return np.zeros(0)
        p = P[self.wk]; A = self.walls[self.wseg, 0]; B = self.walls[self.wseg, 1]
        AB = B - A; L2 = np.maximum((AB * AB).sum(1), 1e-12)
        tt = ((p - A) * AB).sum(1) / L2
        inner = (tt > 0) & (tt < 1)
        perp = (AB[:, 0] * (p[:, 1] - A[:, 1]) - AB[:, 1] * (p[:, 0] - A[:, 0])) / np.sqrt(L2) * self.wside
        Cp = A + np.clip(tt, 0, 1)[:, None] * AB
        dist = np.where(inner, perp, np.hypot(*(p - Cp).T))
        return np.maximum(0.0, self.margin - dist)

    def residuals(self, x, idx, w, reg):
        X, Z, h = self.pose(x)
        u = self.use
        P = np.stack([X[u], Z[u]], 1)
        ok = idx >= 0
        r = np.zeros(len(idx))
        r[ok] = ((P[ok] - self.cl.xz[idx[ok]]) * self.cl.n[idx[ok]]).sum(1)
        rdz = np.sign(r) * np.maximum(0.0, np.abs(r) - self.dz)
        g, sg, A, Bx, Bz, c, D = self.unpack(x)
        rr = [r * w * self.cw, rdz * w * self.dzw,
              self.wall_w * self.wall_res(P) if self.wall_w else np.zeros(0),
              np.atleast_1d((g - self.g0) / self.g_sd), np.atleast_1d(reg['sigma'] * (sg - 1.0)),
              reg['A'] * np.diff(A), reg['B'] * np.diff(Bx), reg['B'] * np.diff(Bz), reg['c'] * c,
              (D - self.D0) / self.D_sd]
        if self.speed_k:
            rr.append(np.atleast_1d(self.kval(x) / self.k_sd))
        for i, tx, gr, ww in self.ev:
            rr.append(ww * (X[i] + D[gr] * np.cos(h[i]) - tx))
        return np.concatenate(rr)

    def event_residuals(self, x):
        X, Z, h = self.pose(x); D = self.unpack(x)[6]
        return [(gr, X[i] + D[gr] * np.cos(h[i]) - tx) for i, tx, gr, ww in self.ev]

    def fit(self, x, n_outer=12, n_gn=3, huber=0.25, reg=None, log=None, walls=None, wall_w=0.0,
            cw=1.0, dz=0.45, dzw=0.0, margin=0.10, look=0.30):
        """cw: 基準線への引き寄せの重み、dz/dzw: 基準線から dz 以上離れた分への罰則、wall_w/margin: 壁に margin より近づいた分への罰則"""
        self.wall_w = wall_w; self.wk = np.zeros(0, int); self.cw = cw; self.dz = dz; self.dzw = dzw
        reg = reg or {'sigma': 1 / 0.15, 'A': 1 / np.radians(8), 'B': 1 / 0.6, 'c': 1 / np.radians(30)}
        for it in range(n_outer):
            X, Z, h = self.pose(x)
            u = self.use
            idx, off = self.cl.match(np.stack([X[u], Z[u]], 1), h[u])
            ok = idx >= 0
            a = np.abs(off); w = np.ones(len(idx)); big = a > huber; w[big] = np.sqrt(huber / a[big]); w[~ok] = 0
            if wall_w and walls is not None:
                self.wall_setup(np.stack([X[u], Z[u]], 1), idx, walls, margin=margin, look=look)
            for _ in range(n_gn):
                r0 = self.residuals(x, idx, w, reg)
                Jm = np.empty((len(r0), len(x)))
                for j in range(len(x)):
                    e = 1e-7 if j < 2 else 1e-6
                    xx = x.copy(); xx[j] += e
                    Jm[:, j] = (self.residuals(xx, idx, w, reg) - r0) / e
                step = np.linalg.lstsq(Jm.T @ Jm + 1e-6 * np.eye(len(x)), -Jm.T @ r0, rcond=None)[0]
                x = x + step
            if log is not None:
                log.append({'iter': it, 'matched': float(ok.mean()), 'med_abs_off': float(np.nanmedian(a[ok])), 'g': float(x[0]), 'sigma': float(x[1])})
        return x

def evaluate(X, Z, use, cl, walls, h=None):
    P = np.stack([X, Z], 1)
    out = {}
    if h is not None:
        idx, off = cl.match(P[use], h[use])
        ok = idx >= 0
        out['matched_frac'] = float(ok.mean())
        out['cross_track_abs_median_m'] = float(np.median(np.abs(off[ok])))
        out['cross_track_abs_p90_m'] = float(np.percentile(np.abs(off[ok]), 90))
    dw, _ = dist_to_segments(P[use], walls)
    out['wall_dist_median_m'] = float(np.median(dw))
    out['frac_closer_than_8cm_to_wall'] = float((dw < 0.08).mean())
    cr = crossings(P[use], walls)
    out['wall_crossings'] = int(len(cr))
    return out, dw
