"""ToF(VL53L9, 42x54)による衝突予測と安全層のスケッチ(numpyのみ、1フレーム < 1 ms 目標)

使い方(drive.py の apply_gains の後、PCA9685 へ書く前を想定):
    guard = ToFGuard(np.load('tof_floor_ref.npy'))                # 自由走行で作った床の基準(mm, 42x54)
    steer_us, thr_us, info = guard.step(steer_us, thr_us, tof_raw, erpm, gyro_z_dps)

座標: x前方 / y左 [m]。ToFの列 0 = 車の左側(カメラと同じ向き。ヨー回転との相関で確認済み)。
IMUの軸: x = 右, y = 前, z = 上(加速度・旋回の相関から推定。gyro z 正 = 左旋回)。
しきい値・係数は 2026-10-10 のログ(ai_run_006/007 ほか)で決めたもので、要再確認。
"""
import numpy as np

STEER_CENTER, THR_NEUTRAL = 1443.0, 1498.0
KV = 0.50e-3                       # m/s per eRPM(前進 = eRPM 負)。ToF接近速度と求心加速度の2方法で 0.47-0.53
K_LEFT, K_RIGHT = 0.00275, 0.0039  # 曲率[1/m] / ステア us(中心から)。全切り 左 1.3 / 右 1.7 1/m で頭打ち
K_MAX_L, K_MAX_R = 1.30, 1.70
R0 = 20.5


def _rays(h=0.12, pitch_deg=9.6, v_deg=0.55, h_deg=1.2, c0=26.5):
    r, c = np.meshgrid(np.arange(42), np.arange(54), indexing='ij')
    th = np.deg2rad(pitch_deg)
    vf = np.ones(r.shape); vl = -(c - c0) * np.tan(np.deg2rad(h_deg)); vu = -(r - R0) * np.tan(np.deg2rad(v_deg))
    n = np.sqrt(vf ** 2 + vl ** 2 + vu ** 2)
    return (vf * np.cos(th) + vu * np.sin(th)) / n, vl / n


class ToFGuard(object):
    ROWS = slice(4, 28)            # 0-3行は遠景(スペックル多)、28行以降はバンパー
    REL, ABS_M = 0.20, 0.08        # 床より 20% かつ 8 cm 以上近い = 床から ~2.4 cm 以上の高さ(h=0.12 m のとき)
    DMAX = 2.0
    VERT_MIN = 4                   # 同じ列で距離のそろった 4 行以上 = 立った物(床のテープ・光沢の誤検出を除く)
    DD = 0.10
    K = 3                          # k 番目に近い点(単発ノイズ対策)
    HW, MARGIN = 0.10, 0.05        # 車の半幅(仮定)+ 余裕 [m]
    Y_OFF = -0.03                  # ToF は車体中心から 3 cm 右
    T_WARN, T_BRAKE = 0.5, 0.3     # TTC [s]
    NEAR = 0.8                     # 左右の非対称を数える距離 [m]
    STEER_BIAS_US = 150.0
    BRAKE_US = 150.0               # 前進中に中立より上 = ブレーキ(このESCは前進→後退を直接はブレーキとして扱う)
    VMIN = 0.5

    def __init__(self, floor_ref_mm, **geom):
        ref = np.asarray(floor_ref_mm, np.float32) / 1000.0
        f, l = _rays(**geom)
        self.ref = ref[self.ROWS]
        self.fwd = f[self.ROWS]; self.left = l[self.ROWS]
        self.n_rows = self.ref.shape[0]
        self.colL = np.zeros(54, bool); self.colL[:18] = True     # 車の左
        self.colR = np.zeros(54, bool); self.colR[36:] = True     # 車の右
        self.side_hist = []

    def mask(self, d, conf):
        m = (conf == 4) & (d > 0) & (d < self.ref * (1 - self.REL)) & (self.ref - d > self.ABS_M) & (d < self.DMAX)
        dm = np.where(m, d, -9.0)
        link = m[1:] & m[:-1] & (np.abs(dm[1:] - dm[:-1]) < self.DD)   # 縦に隣り合う同じ距離
        dn = np.zeros(m.shape, np.int8); up = np.zeros(m.shape, np.int8)
        for r in range(1, self.n_rows):
            dn[r] = np.where(link[r - 1], dn[r - 1] + 1, 0)
        for r in range(self.n_rows - 2, -1, -1):
            up[r] = np.where(link[r], up[r + 1] + 1, 0)
        return m & (dn + up + 1 >= self.VERT_MIN)

    def features(self, raw, erpm, gz_dps, steer_us):
        raw = np.asarray(raw)[self.ROWS]
        d = (raw & 0x1FFF).astype(np.float32); d[d == 0x1FFF] = 0
        d *= 0.001
        conf = raw >> 13
        m = self.mask(d, conf)
        v = max(-float(erpm) * KV, 0.0)
        x = (d * self.fwd)[m]; y = (d * self.left)[m] + self.Y_OFF
        k_gyro = np.clip(np.deg2rad(gz_dps) / max(v, 0.5), -2.0, 2.0)
        s = steer_us - STEER_CENTER
        k_cmd = float(np.clip(s * (K_LEFT if s > 0 else K_RIGHT), -K_MAX_R, K_MAX_L))
        w = self.HW + self.MARGIN
        inpath = (np.abs(y - 0.5 * k_gyro * x * x) < w) | (np.abs(y - 0.5 * k_cmd * x * x) < w)
        xs = x[inpath]
        x_free = float(np.partition(xs, self.K - 1)[self.K - 1]) if xs.size >= self.K else 9.0
        dd = d[:, :]
        nL = int((m & (dd < self.NEAR) & self.colL[None]).sum()); nR = int((m & (dd < self.NEAR) & self.colR[None]).sum())
        ttc = x_free / max(v, 0.3)
        return {'v': v, 'x_free': x_free, 'ttc': ttc, 'nL': nL, 'nR': nR, 'n_obs': int(m.sum())}

    def step(self, steer_us, thr_us, raw, erpm, gz_dps):
        f = self.features(raw, erpm, gz_dps, steer_us)
        self.side_hist = (self.side_hist + [(f['nL'], f['nR'])])[-12:]          # 約 0.4 s
        if f['v'] < self.VMIN or f['ttc'] >= self.T_WARN:
            f['action'] = None
            return steer_us, thr_us, f
        u = min(1.0, (self.T_WARN - f['ttc']) / (self.T_WARN - self.T_BRAKE))  # 0..1 の切迫度
        # 1) 速度: 警告域では前進を中立までに制限、TTC < T_BRAKE ではブレーキ
        if f['ttc'] < self.T_BRAKE:
            thr_us = max(thr_us, THR_NEUTRAL + self.BRAKE_US)
        else:
            thr_us = max(thr_us, THR_NEUTRAL - (1 - u) * (THR_NEUTRAL - thr_us))   # 切迫度に応じて中立へ寄せる
        # 2) ハンドル: 近い点の多い側から離す(左右がはっきりしているときだけ)
        nL = sum(a for a, b in self.side_hist); nR = sum(b for a, b in self.side_hist)
        asym = (nR - nL) / max(nR + nL, 1)
        if abs(asym) > 0.5:
            steer_us = steer_us + np.sign(asym) * u * self.STEER_BIAS_US        # 右が近い → 左へ(+us)
        f['action'] = 'brake' if f['ttc'] < self.T_BRAKE else 'slow'
        f['asym'] = asym
        return float(np.clip(steer_us, 1000, 2000)), float(thr_us), f
