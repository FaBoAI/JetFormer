"""TCP 型の制御器(numpy だけ。torch を読まない)。走行(drive._loop)と評価の道具が **同じ** このコードを post.Stage から呼ぶ

    from jetformer.policies.controller import TCPPost
    ctl = TCPPost(model_config['postprocess'], ref=(ステアの中心, スロットルの中立))
    pred_us, info = ctl.step({'actions': [10,2], 'waypoints': [10,2], 'speed_logits': [10]}, v_now, params, t_ns)

制御器は ONNX の外に置く(速度の上限・倍率・出し方を、エンジンを作り直さずに走行タブで変えるため。spec.md §3.2.4・§8)。
出力は [10,2] µs(33 ms 刻みの 1〜10 フレーム先の [ステア, スロットル])。ここから先(先読み・ゲイン・ToF ガード・補助/リカバリー・
CH3 の交代・センサーの途切れで中立)は今までと同じ。**tcp_mode = 0 かつ speed_cap = 0 なら actions の µs とビット単位で同じ**。

車の約束: ステア > 中心 = 左(曲率 κ > 0 = 左 = 自車の座標の y 正)。スロットルは中立より下 = 前進(前進の量 = 中立 − µs)。
速さ v_now = −eRPM × 0.5e-3 [m/s](後退は負)。ESC は前進中の後退側 = ブレーキ、後退は止まってから R → N → R だけ。

1 フレームの処理(ref = (中心 c, 中立 n)):
    1. 目標速度 v_hat = Σ softmax(speed_logits)·bins。上限 cap_eff: speed_cap > 0 ならその値、0 なら無し(∞)。上限が上がった・
       外れたときだけ前の値から 1.5 m/s² で上げ、5.0 m/s を超えたら無し(下げるときはすぐ。最初はすぐ目標の値。reset() は
       上げている途中の値を残す = 中立で止めても上げ方は途切れない)。
       v_cmd = min(speed_scale·v_hat, cap_eff)
    2. 前向きのスロットル thr_ff(v): ブレーキの規則(v_now − v > 0.3 かつ v_now > 1.0 → 中立 + 150 µs)→ v < 0.3 なら中立 →
       thr = c0 + c1·max(v_now, 0) + c2·(v − v_now)/0.33、前進の量 n − thr を [0, fwd_max_us] に切る(中立より上は出さない)
    3. 通過点を追うステア(k = 1..10、τ_k = k/30 秒): (0,0) + 通過点の折れ線の上で τ_k の位置 P_k・向き θ_k(頂点の向きを補間)、
       v_k = max(区間の長さ/0.1, 0.3)、
       L_k = max(0.3, 0.25·v_k) 先の点 G_k へ pure pursuit κ_k = 2y/(x² + y²)(P_k・θ_k の座標)→ 速度帯(v_now)の表で逆引き。
       道の全長が 0.2 m 未満なら操作の枝のステア
    4. tcp_mode: 0 control(既定。操作の枝。上限があるときだけスロットルを max(操作, 上限のスロットル))・1 hybrid(ステア = 操作、
       スロットル = thr_ff(v_cmd))・2 fusion(試験的。過去 1 秒の出力のステアで混ぜる)・3 trajectory(試験的。ステア = 通過点、
       スロットル = thr_ff(v_cmd))。1〜3 も最後に「上限のスロットルより前進を強くしない」
    5. 上限のスロットル(info['thr_cap']): thr_ff(cap_eff) を、走行の速度のゲイン(params の throttle_gain。前進側だけに掛かる。
       drive.apply_gains)で割り戻した値。ゲインを掛けた後の前進がちょうど thr_ff(cap_eff) になるので、ゲインが 1 でなくても
       上限は書いたとおりに効く(レビュー 2026-10-12: ゲイン 2.0 では上限より上で頭打ち、0.5 では上限 1.0 m/s で約 0.32 m/s に
       なっていた)。ゲイン 1.0(評価のゲート D の設定・params に無いとき)では thr_ff(cap_eff) そのもので、前と同じ。
       ブレーキ・中立の値はゲインが掛からないので割り戻さない。post.Stage は重ね合わせの後にもこの値で掛け直す
積分の項は使わない(開ループの評価で飽和し、走行と評価が食い違うため)。例外は出さない(入力が足りなければ branch='fallback')。
"""

import math
from collections import deque

import numpy as np

KIND = 'tcp_controller'
VERSION = 1
STEER_MIN_US, STEER_MAX_US = 1000.0, 2000.0
FPS = 30
SPEED_BINS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
# 先に固定した定数(spec.md §0.3・§3.2.4。検証のシーンで調整しない)
RULES = {'stop_v': 0.3, 'brake_us': 150.0, 'brake_dv': 0.3, 'brake_vmin': 1.0, 'cap_ramp_mps2': 1.5, 'cap_off_mps': 5.0}
PURSUIT = {'t_look_s': 0.25, 'l_min_m': 0.3, 'min_path_m': 0.2}
FUSION = {'turn_us': 100.0, 'turn_frac': 0.5, 'w_major': 0.7, 'window_s': 1.0}
FUSION_THR_W = 0.5                 # fusion のスロットル = 0.5·操作 + 0.5·thr_ff(v_cmd)
MIN_SEG_M = 1e-3                   # これより短い区間は向きを決めない(次の区間の向き)
V_SEG_MIN = 0.3                    # 通過点の区間の速さの下限 [m/s](見る距離の計算)
MODES = {0: 'control', 1: 'hybrid', 2: 'fusion', 3: 'trajectory'}
HIST_MAX = 256                     # fusion の履歴の上限(1 秒 = 30 個。時刻が飛んでも増え続けない)


def _f(x, default):
    """有限の float に(だめなら default)"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def softmax_expectation(logits, bins):
    """Σ softmax(logits)·bins(数値として安定な形)。だめなら nan"""
    z = np.asarray(logits, np.float64).reshape(-1)
    b = np.asarray(bins, np.float64).reshape(-1)
    if z.shape != b.shape or not np.all(np.isfinite(z)):
        return float('nan')
    p = np.exp(z - z.max())
    return float(np.dot(p, b) / p.sum())


def steer_from_curv(kappa, curv_row, offsets, center):
    """曲率 [1/m](配列可)→ ステア µs。表の逆引き(表の外は端 = いっぱいに切った値)、1000〜2000 に切る"""
    off = np.interp(kappa, curv_row, offsets)
    return np.minimum(np.maximum(center + off, STEER_MIN_US), STEER_MAX_US)     # np.clip より速い(小さな配列)


def _wrap(a):
    """角度を [−π, π) に"""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def pursuit_curvatures(wp, wp_dt, fps=FPS, steps=10, t_look=PURSUIT['t_look_s'], l_min=PURSUIT['l_min_m']):
    """自車の座標(x 前・y 左、m)の通過点 [W,2](0.1 秒刻み)→ ステップごとの追う曲率 κ [steps](左 正)と道の全長 [m]。
    道が決められない(全部の区間が 1 mm 未満)なら (None, 全長)"""
    wp = np.asarray(wp, np.float64).reshape(-1, 2)
    nw = len(wp)
    P = np.zeros((nw + 1, 2))
    P[1:] = wp                                                    # [W+1,2]((0,0) + 通過点)
    d = np.diff(P, axis=0)                                        # [W,2]
    seg = np.hypot(d[:, 0], d[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    good = seg >= MIN_SEG_M
    if not good.any():
        return None, total
    idx = np.arange(nw)
    nxt = np.minimum.accumulate(np.where(good, idx, nw)[::-1])[::-1]       # 次の(自分を含む)向きの決まる区間
    prv = np.maximum.accumulate(np.where(good, idx, -1))                     # 前の(自分を含む)向きの決まる区間
    use = np.where(nxt < nw, nxt, prv)
    theta = np.arctan2(d[use, 1], d[use, 0])                                 # 区間(弦)ごとの向き [W]
    # 頂点の向き: 原点 = 0(自車の座標の定義で x 軸 = 自車の向き)、途中 = 隣り合う弦の向きの平均(円なら接線と同じ)、
    # 最後 = 弦の向きから折り返し。P_k の向き θ_k は頂点の向きを時間で線形補間する(弦の向きのままだと、円の上でも
    # τ_k が区間のどこにあるかで追う曲率が 0.6〜1.4 倍に揺れる。0.1 秒の区間で 0.2 m 進むとき)
    h = np.empty(nw + 1)
    h[0] = 0.0
    if nw > 1:
        h[1:nw] = np.arctan2(np.sin(theta[:-1]) + np.sin(theta[1:]), np.cos(theta[:-1]) + np.cos(theta[1:]))
    h[nw] = theta[nw - 1] + _wrap(theta[nw - 1] - h[nw - 1])
    tau = np.arange(1, steps + 1) / float(fps)
    u = tau / float(wp_dt)
    j = np.minimum(np.floor(u).astype(np.int64), nw - 1)
    frac = np.minimum(u - j, 1.0)
    Pk = P[j] + frac[:, None] * d[j]
    sk = s[j] + frac * seg[j]
    th = h[j] + frac * _wrap(h[j + 1] - h[j])
    vk = np.maximum(seg[j] / float(wp_dt), V_SEG_MIN)
    Lk = np.maximum(l_min, t_look * vk)
    sg = sk + Lk
    m = np.minimum(np.maximum(np.searchsorted(s, sg, side='right') - 1, 0), nw - 1)
    sm = seg[m]
    f = np.where(sm > 0, (sg - s[m]) / np.where(sm > 0, sm, 1.0), 0.0)
    G = P[m] + np.minimum(np.maximum(f, 0.0), 1.0)[:, None] * d[m]
    beyond = sg > total                                                       # 端を超えたら最後の区間の向きに延ばす
    if beyond.any():
        th_last = theta[nw - 1]
        ext = (sg[beyond] - total)[:, None] * np.array([math.cos(th_last), math.sin(th_last)])
        G[beyond] = P[-1] + ext
    dx, dy = G[:, 0] - Pk[:, 0], G[:, 1] - Pk[:, 1]
    c, sn = np.cos(th), np.sin(th)
    x, y = c * dx + sn * dy, -sn * dx + c * dy
    kappa = 2.0 * y / np.maximum(x * x + y * y, 1e-6)
    return kappa, total


class TCPPost(object):
    """post.Stage が呼ぶ後処理(spec.md §2.8・§3.2.6)。postprocess = model_config.json の 'postprocess'、ref = (中心, 中立) µs"""

    kind = KIND

    def __init__(self, postprocess, ref):
        pp = dict(postprocess or {})
        if pp.get('kind', KIND) != KIND:
            raise ValueError('TCP の制御器ではない後処理です: %s' % pp.get('kind'))
        if int(pp.get('version', VERSION)) > VERSION:
            raise ValueError('新しい版の制御器の設定です(version %s)。プログラムを更新してください' % pp.get('version'))
        self.center, self.neutral = float(ref[0]), float(ref[1])
        self.pwm_center = float(pp.get('pwm_center', 1500.0))
        self.pwm_scale = float(pp.get('pwm_scale', 500.0))
        self.bins = np.asarray(pp.get('speed_bins', SPEED_BINS), np.float64)
        self.wp_dt = float(pp.get('wp_dt', 0.1))
        self.fps = float(pp.get('fps', FPS))
        st = pp.get('steer_table') or {}
        self.offsets = np.asarray(st.get('offset_us', []), np.float64)
        self.curv = np.asarray(st.get('curv', []), np.float64)
        self.bands = np.asarray(st.get('bands_mps', [2.0, 3.0]), np.float64)
        if self.offsets.ndim != 1 or len(self.offsets) < 2 or self.curv.ndim != 2 or \
                self.curv.shape != (len(self.bands) + 1, len(self.offsets)):
            raise ValueError('postprocess.steer_table の形が違います(offset_us %s・curv %s・bands_mps %s)' % (
                self.offsets.shape, self.curv.shape, self.bands.shape))
        if not (np.all(np.diff(self.offsets) > 0) and np.all(np.diff(self.curv, axis=1) > 0)):
            raise ValueError('postprocess.steer_table が単調に増えていません')
        ff = pp.get('throttle_ff') or {}
        try:
            self.c0, self.c1, self.c2 = float(ff['c0']), float(ff['c1']), float(ff['c2'])
        except (KeyError, TypeError, ValueError):
            raise ValueError('postprocess.throttle_ff に c0・c1・c2 がありません')
        self.tau = float(ff.get('tau_s', 0.33))
        self.fwd_max = max(float(ff.get('fwd_max_us', 0.0)), 0.0)
        self.rules = dict(RULES, **(pp.get('rules') or {}))
        self.pursuit = dict(PURSUIT, **(pp.get('pursuit') or {}))
        self.fusion = dict(FUSION, **(pp.get('fusion') or {}))
        # 速度帯ごとの「いっぱいに切った曲率」(v_curv の表示用)と、帯の上限
        self.full_left = self.curv[:, -1]
        self.full_right = -self.curv[:, 0]
        self.band_upper = np.concatenate([self.bands, [float(self.bins[-1]) if len(self.bins) else 4.5]])
        self.reset()

    # ---------------------------------------------------------------- 状態
    def reset(self):
        """中立で止めた後・シーンの切れ目。fusion の履歴は空にする。上限の今の値(cap_prev)は残す: 上限を上げた・外した直後に
        中立で止めても、次の推論から 1.5 m/s² の上げ方の続きになる(消すと上限が一気に新しい値へ跳ぶ。レビュー 2026-10-12)。
        時刻は忘れる(止めていた間の時間で一度に上げない。次の 1 フレームは 1/fps として数える)。
        上限が一定の評価(spec.md §5.5)では、最初のフレームから cap_prev = 目標の値なので、結果は前と同じ"""
        if not hasattr(self, 'cap_prev'):
            self.cap_prev = None
        self.t_prev = None
        self.hist = deque(maxlen=HIST_MAX)  # (時刻 秒, 出力したステア µs)

    # ---------------------------------------------------------------- 部品
    def thr_ff(self, v_target, v_now):
        """目標速度 → スロットル µs(前向きの写像。中立より上はブレーキの規則のときだけ)"""
        r, n = self.rules, self.neutral
        if v_now - v_target > r['brake_dv'] and v_now > r['brake_vmin']:
            return n + r['brake_us']
        if not v_target >= r['stop_v']:                 # nan も中立
            return n
        thr = self.c0 + self.c1 * max(v_now, 0.0) + self.c2 * (v_target - v_now) / self.tau
        fwd = min(max(n - thr, 0.0), self.fwd_max)
        return n - fwd

    def cap_raw(self, thr_cap, gain):
        """上限のスロットル thr_cap(ゲインの後に出したい値)→ ゲインの前の値。前進側だけ割り戻す(apply_gains と同じ:
        ゲインは中立より下 = 前進側だけに掛かる)。gain = 1 なら thr_cap そのもの(ビット単位で同じ)"""
        if thr_cap is None or gain == 1.0 or not thr_cap < self.neutral:
            return thr_cap
        g = max(gain, 1e-6)                               # ゲイン 0 = 前進なし: 上限は要らない(ほぼ −∞ になる)
        return self.neutral - (self.neutral - thr_cap) / g

    def band_of(self, v):
        return int(np.searchsorted(self.bands, v, side='right'))

    def v_curv_of(self, kappa):
        """予測の道の最大の |κ| に対し、いっぱいに切った曲率が 1.1·|κ| 以上になる最大の速度帯の上限(表示だけ)"""
        if kappa is None or not len(kappa):
            return float('nan')
        k = float(kappa[int(np.argmax(np.abs(kappa)))])
        full = self.full_left if k >= 0 else self.full_right
        ok = full >= 1.1 * abs(k)
        return float(self.band_upper[ok].max()) if ok.any() else 0.0

    def _cap(self, speed_cap, t_s):
        """→ cap_eff(上限なしは inf)"""
        r = self.rules
        target = speed_cap if speed_cap > 0 else math.inf
        prev = self.cap_prev
        if prev is None or target <= prev:
            cap = target                                  # 最初・下げる・同じ: すぐ
        else:
            dt = 1.0 / self.fps
            if self.t_prev is not None and t_s is not None and t_s > self.t_prev:
                dt = min(t_s - self.t_prev, 0.5)
            cap = min(target, prev + r['cap_ramp_mps2'] * dt)
            if cap > r['cap_off_mps']:                    # 5.0 m/s を超えたら「なし」
                cap = math.inf
        self.cap_prev = cap
        return cap

    # ---------------------------------------------------------------- 1 フレーム
    def step(self, outputs, v_now, params, t_ns):
        """→ (pred_us [10,2] float32, info)。例外は出さない"""
        params = params or {}
        mode = int(min(max(_f(params.get('tcp_mode', 0), 0.0), 0.0), 3.0))
        speed_cap = max(_f(params.get('speed_cap', 0.0), 0.0), 0.0)
        scale = max(_f(params.get('speed_scale', 1.0), 1.0), 0.0)
        v_ok = v_now is not None and math.isfinite(_f(v_now, float('nan')))
        vn = _f(v_now, 0.0)
        t_s = None if t_ns is None else _f(t_ns, float('nan')) / 1e9
        if t_s is not None and not math.isfinite(t_s):
            t_s = None
        gain = _f(params.get('throttle_gain', 1.0), 1.0)    # 走行の速度のゲイン(上限の割り戻しだけに使う)
        info = {'mode': mode, 'branch': MODES[mode], 'v_hat': float('nan'), 'v_cmd': float('nan'),
                'v_curv': float('nan'), 'cap': 0.0, 'thr_cap': float('nan')}
        acts = (outputs or {}).get('actions')
        try:
            act_us = np.asarray(acts, np.float32) * self.pwm_scale + self.pwm_center      # Stage.model_pred と同じ式
            if act_us.shape != (10, 2):
                raise ValueError
        except (TypeError, ValueError):
            info['branch'] = 'fallback'                   # 操作の枝も無い(普通は起きない)→ 中立
            out = np.empty((10, 2), np.float32)
            out[:, 0], out[:, 1] = self.center, self.neutral
            return out, info
        cap = self._cap(speed_cap, t_s)
        self.t_prev = t_s if t_s is not None else self.t_prev
        info['cap'] = cap if math.isfinite(cap) else 0.0
        v_hat = softmax_expectation(outputs.get('speed_logits'), self.bins) if 'speed_logits' in outputs else float('nan')
        info['v_hat'] = v_hat
        v_cmd = min(scale * v_hat, cap) if math.isfinite(v_hat) else float('nan')
        info['v_cmd'] = v_cmd
        kappa = None
        wps = outputs.get('waypoints')
        try:
            wps = np.asarray(wps, np.float64)
            if wps.ndim == 2 and wps.shape[1] == 2 and len(wps) >= 1 and np.all(np.isfinite(wps)):
                kappa, total = pursuit_curvatures(wps, self.wp_dt, self.fps, 10, self.pursuit['t_look_s'],
                                                  self.pursuit['l_min_m'])
                if total < self.pursuit['min_path_m']:
                    kappa = None                          # 止まっている: ステアの手がかりが無い
        except (TypeError, ValueError):
            kappa = None
        if kappa is not None and not np.all(np.isfinite(kappa)):
            kappa = None
        info['v_curv'] = self.v_curv_of(kappa)
        # 上限のスロットル(速さが分からなければ vn = 0 = 止まっているとして。上限は弱くしか効かない)。
        # 走行の速度のゲインで割り戻す(ゲインの後で thr_ff(cap) になるように。ゲイン 1.0 なら同じ値)
        thr_cap = self.cap_raw(self.thr_ff(cap, vn), gain) if math.isfinite(cap) else None
        if thr_cap is not None:
            info['thr_cap'] = float(thr_cap)

        mode_eff = mode
        if mode >= 1 and not (math.isfinite(v_cmd) and v_ok):
            info['branch'] = 'fallback'                   # 目標速度・今の速さが無い: 操作の枝(上限だけは掛ける)
            mode_eff = 0
        out = act_us.copy()
        if mode_eff == 0:
            if thr_cap is not None:                       # 上限なしなら actions とビット単位で同じ
                out[:, 1] = np.maximum(act_us[:, 1], np.float32(thr_cap))
        else:
            thr_cmd = self.thr_ff(v_cmd, vn)
            wp_steer = None
            if kappa is not None:
                band = min(max(self.band_of(vn), 0), len(self.curv) - 1)
                wp_steer = steer_from_curv(kappa, self.curv[band], self.offsets, self.center)
            if mode_eff == 1:
                out[:, 1] = thr_cmd
            elif mode_eff == 2:
                f = self.fusion
                ws = f['w_major']
                turning = self._turning(t_s)
                info['branch'] = 'fusion:ctrl' if turning else 'fusion:traj'
                if wp_steer is not None:
                    w_ctrl = ws if turning else 1.0 - ws
                    out[:, 0] = w_ctrl * act_us[:, 0] + (1.0 - w_ctrl) * wp_steer
                out[:, 1] = FUSION_THR_W * act_us[:, 1] + (1.0 - FUSION_THR_W) * thr_cmd
            else:
                if wp_steer is not None:
                    out[:, 0] = wp_steer
                else:
                    info['branch'] = 'fallback'           # 道が短い・無い: ステアは操作の枝(スロットルは制御器)
                out[:, 1] = thr_cmd
            if thr_cap is not None:
                out[:, 1] = np.maximum(out[:, 1], thr_cap)
        self._remember(t_s, out, params)
        return out.astype(np.float32, copy=False), info

    # ---------------------------------------------------------------- fusion の履歴
    def _turning(self, t_s):
        """過去 1 秒にこのオブジェクトが出したステアの半分以上が |ステア − 中心| > 100 µs か(空なら False)"""
        if t_s is not None:
            if self.hist and t_s < self.hist[-1][0] - 1e-9:
                self.hist.clear()                         # 時刻が戻った(別のシーン)
            while self.hist and self.hist[0][0] < t_s - self.fusion['window_s'] - 1e-9:
                self.hist.popleft()
        if not self.hist:
            return False
        turn = sum(1 for _, s in self.hist if abs(s - self.center) > self.fusion['turn_us'])
        return turn >= self.fusion['turn_frac'] * len(self.hist)

    def _remember(self, t_s, out, params):
        """出したステア(走行の先読みの行)を履歴へ。時刻が無ければ 1/fps ずつ進める"""
        k = int(min(max(_f(params.get('lookahead', 1), 1.0), 1.0), 10.0)) - 1
        if t_s is None:
            t_s = (self.hist[-1][0] + 1.0 / self.fps) if self.hist else 0.0
        elif self.hist and t_s < self.hist[-1][0] - 1e-9:
            self.hist.clear()
        self.hist.append((t_s, float(out[k, 0])))
        while self.hist and self.hist[0][0] < t_s - self.fusion['window_s'] - 1e-9:
            self.hist.popleft()
