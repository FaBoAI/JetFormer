"""安全層 v1: 区間の速度上限(R1 2.5・GM 2.2・DE 3.0 m/s)と、ジャイロの位相による区間判定(2026-10-16)

drive.py には走行のパラメータ zone_caps(既定 0 = OFF。起動のたびに OFF)でつないだ(2026-10-21、末尾の zone_frame)。
OFF でも区間・上限・「上限ならこう出した」スロットルを毎フレーム計算して記録に残す(影の判定)。出力は変えない。
**ON で走るのは、**10/23 の記録で机上のゲート G1(記録の再生)に合格し、ユーザーがもう一度了承したあと
(action_plan_20261016/plan.md 項目 9・13、bench_prep_20261021)。ここにあるのは純粋な関数だけで、
入出力・時刻の取得・ハードウェア(PCA9685・I2C)には触れない。同じ入力には同じ出力を返す。

仕様(summary_20261010/fast_and_safe_plan.md の N1・explore_control.md の C1/C2/C7、
sim_calibration_20261016/protocol.json の cap_test.governor_v1_sim):
  区間                     上限        ブレーキ
  R1_app(R1 の進入)      2.5 m/s     出す(前へ 1 m/s 以上、上限 + 0.3 m/s を超えたとき。中立 + 200 µs)
  R1_turn(R1 の旋回中)   2.5 m/s     出さない(中立まで)
  GM(芝・PTFE の U ターン) 2.2 m/s     出さない
  DE(GM 出口から 1.5 秒)   3.0 m/s     出さない
  unknown(位相が不確か)    3.0 m/s     出さない(最初の R1 の前・R1 から 16 秒超・向きが範囲外)
  none                     なし
  - v ≤ 上限: 指令を上限で切る。v > 上限: 中立(惰性)。指令を大きくはしない(出力 ≤ 入力)。舵は変えない
  - ブレーキは始めたら 0.2 秒は保つ(区間を出た・1 m/s 未満になったときはすぐやめる)。やめたら 0.5 秒は再び始めない
    (G1 の「0.5 秒以内のブレーキ ↔ 中立の往復 0 回」を作りで満たす)
  - 区間を出たら、上限を 1.5 m/s² で上げて戻す(モデルの再加速の突き上げを防ぐ)
  - R1 の合図は 2 つの OR: (a) ジャイロの位相(直前の R1 の起点から向き ψ ≥ 300 度 = LH の後半)、
    (b) モデル自身の予測ステアが右いっぱい(ゲイン前 1150 µs 以下)。(b) は位相が R1_turn・GM・DE のときは使わない
  - 右旋回(5 サンプル平均のヨー < −60 deg/s)が R1 の合図の中で始まったら、すぐ R1_turn(ブレーキなし)にする

単位: 速さ v [m/s](前進 +。v = −eRPM × 0.5e-3)、ヨー gz [deg/s](左 +)、時刻 t [秒]、スロットルの PWM [µs]
(中立 1498、前進は小さい値)。上限の層の中の指令は、ブリッジ・シミュレーターと同じ「速さの単位」u [m/s]
(thr ≤ 1498: u = (1498 − thr)/498 × 7.24、thr > 1498: u = −(thr − 1498)/502 × 0.8 = ブレーキの要求)。
シミュレーター(experiments/20261016_sim_calibration の governor)と同じ規則・同じ数字で動く。

版(2026-10-16 の G1 の再生のあと、fast_and_safe_plan.md 6 章の「落ちたら 1 回だけ直す」に当たる直し。既定は v1.1):
  v1   : 上の仕様のまま(VERSION_V1 = CapParams(brake_steer_release=None)・PHASE_V1・pred_min_speed 0)
  v1.1 : (A) 舵の指令が右へ 30% 以上なら、ブレーキを始めない・すぐやめる(舵の指令はヨーより約 0.1 秒先に出るので、
             R1 の旋回の始まりより先にブレーキが外れる。v1 では R1 の起点 +0.5 秒の中に 1〜2 フレーム残った)
         (B) 1 秒以上止まったら、位相を不確か(unknown)にする(止まっている間は向きを積分しないので、人が車を
             置き直すと位相が狂ったまま残る。v1 では 007 で 13 秒 GM のままだった)。R1_app の中で向きが 240 度を
             下回ったら(引き返した)、これも不確かにする
         (C) モデルの予測ステアの合図 (b) は、前へ 1 m/s 以上のときだけ使う(衝突のあとの低速の立て直しで出る合図を除く)
"""
import math
from typing import NamedTuple, Optional, Tuple

THR_NEUTRAL_US = 1498.0
FWD_SPAN_US = 498.0          # 中立 − 1000 µs
BRAKE_SPAN_US = 502.0        # 2000 − 中立 µs
MAX_SPEED_CMD = 7.24         # ブリッジの前進の端 [m/s]
BRAKE_CMD_SCALE = 0.8        # ブリッジの後退側の端 [m/s]
KV = 0.5e-3                  # m/s / eRPM(eRPM は負 = 前進)
INF = float('inf')

ZONES = ('none', 'unknown', 'R1_app', 'R1_turn', 'GM', 'DE')

# 速さの対応 T(u)(実車 10/10 の人の操作から同定した jf-calib-v1。profiles/P_cal.json、sha256 4b0e9348…)。
# 上限の速さ → 指令の PWM の換算(開ループの目安)にだけ使う。閉ループの「v > 上限なら中立」が主の仕組み。
CAL_T_KNOTS_U = (0.0, 0.4286189631087167, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.24)
CAL_T_VALUES = (0.0, 0.0, 0.18008916970987934, 1.0154669645131382, 2.191791568056699, 3.336828397976429,
                4.149389098689263, 5.0070424923922605, 5.007080520819097, 5.012401396968912)


# ------------------------------------------------------------------ 単位の換算
def thr_us_to_cmd(thr_us: float) -> float:
    """スロットルの PWM [µs] → 速さの単位の指令 u [m/s](ブリッジの pwm_to_action と同じ)"""
    if thr_us <= THR_NEUTRAL_US:
        return (THR_NEUTRAL_US - thr_us) / FWD_SPAN_US * MAX_SPEED_CMD
    return -(thr_us - THR_NEUTRAL_US) / BRAKE_SPAN_US * BRAKE_CMD_SCALE


def cmd_to_thr_us(u: float) -> float:
    """速さの単位の指令 u → スロットルの PWM [µs](thr_us_to_cmd の逆)"""
    if u >= 0:
        return THR_NEUTRAL_US - u / MAX_SPEED_CMD * FWD_SPAN_US
    return THR_NEUTRAL_US - u / BRAKE_CMD_SCALE * BRAKE_SPAN_US


def brake_cmd_of_us(brake_us: float) -> float:
    """中立 + brake_us [µs] のブレーキの要求を、速さの単位にしたもの(200 µs → −0.3187 m/s)"""
    return -brake_us / BRAKE_SPAN_US * BRAKE_CMD_SCALE


def speed_to_cmd_calibrated(v_cap: float) -> float:
    """定常の速さが v_cap になる指令 u = T⁻¹(v_cap)(較正の対応表の逆。表の頭打ち 5.0 m/s より上は端)"""
    if not math.isfinite(v_cap):
        return INF
    T, U = CAL_T_VALUES, CAL_T_KNOTS_U
    if v_cap <= T[2]:
        # 不感帯の上端から 0.5 の節まで
        return U[1] + (U[2] - U[1]) * max(v_cap, 0.0) / T[2]
    for j in range(2, len(T) - 1):
        if T[j] <= v_cap <= T[j + 1] and T[j + 1] > T[j]:
            return U[j] + (U[j + 1] - U[j]) * (v_cap - T[j]) / (T[j + 1] - T[j])
    return U[-1]


def speed_to_cmd_identity(v_cap: float) -> float:
    """シミュレーターの governor_v1_sim の決まり(指令の速さをそのまま上限で切る)"""
    return v_cap


# ------------------------------------------------------------------ 上限の層
class CapParams(NamedTuple):
    cap_R1_app: float = 2.5
    cap_R1_turn: float = 2.5
    cap_GM: float = 2.2
    cap_DE: float = 3.0
    cap_unknown: float = 3.0
    brake_zones: Tuple[str, ...] = ('R1_app',)
    brake_margin: float = 0.3          # v > 上限 + これ でブレーキを始める
    brake_min_speed: float = 1.0       # これ未満ではブレーキを始めない(ESC の後退を避ける)。ブレーキ中にこれを下回ればやめる
    brake_us: float = 200.0            # 中立 + 200 µs
    brake_hold_s: float = 0.2          # 始めたら保つ時間
    rebrake_lockout_s: float = 0.5     # やめたら再び始めない時間
    release_accel: float = 1.5         # 区間を出たときに上限を上げる速さ [m/s²]
    cap_free: float = 8.0              # 上げている上限がこれを超えたら「上限なし」に戻す
    cap_command: str = 'calibrated'    # 上限 → 指令: 'calibrated'(実車。T⁻¹)/ 'identity'(シミュレーター)
    brake_steer_release: Optional[float] = 0.3   # v1.1 (A): 舵の指令(正規化、左 +)が −これ 以下ならブレーキしない。None = v1

    def cap_of(self, zone: str) -> float:
        return {'R1_app': self.cap_R1_app, 'R1_turn': self.cap_R1_turn, 'GM': self.cap_GM, 'DE': self.cap_DE,
                'unknown': self.cap_unknown}.get(zone, INF)

    def command_of_cap(self, cap: float) -> float:
        if not math.isfinite(cap):
            return INF
        return speed_to_cmd_calibrated(cap) if self.cap_command == 'calibrated' else speed_to_cmd_identity(cap)


def params_real_car(**kw) -> CapParams:
    """実車の既定(上限 → 指令は較正の対応表の逆)"""
    return CapParams(**kw)


def params_sim(**kw) -> CapParams:
    """シミュレーターの governor_v1_sim(上限 → 指令はそのまま)"""
    kw.setdefault('cap_command', 'identity')
    return CapParams(**kw)


def params_v1(**kw) -> CapParams:
    """G1 の直しの前の版 v1(舵の条件なし)"""
    kw.setdefault('brake_steer_release', None)
    return CapParams(**kw)


def params_r1_275(**kw) -> CapParams:
    """R1 だけ 2.75 m/s の版(計画どおり再生で比べる値)"""
    kw.setdefault('cap_R1_app', 2.75)
    kw.setdefault('cap_R1_turn', 2.75)
    return CapParams(**kw)


class CapState(NamedTuple):
    t: Optional[float] = None          # 前の呼び出しの時刻
    cap_eff: float = INF               # 今の上限(区間を出たあとは 1.5 m/s² で上がる)
    braking: bool = False
    brake_t0: float = -INF             # ブレーキを始めた時刻
    brake_t1: float = -INF             # ブレーキをやめた時刻


class CapOut(NamedTuple):
    u: float                # 出す指令(速さの単位)。いつも ≤ 入力
    mode: str               # 'pass'(そのまま)/ 'cap'(上限で切った)/ 'neutral'(中立にした)/ 'brake'
    cap_eff: float          # 今の上限(なしは inf)
    zone: str
    brake_start: bool       # このフレームでブレーキを始めた


def cap_step(state: CapState, t: float, zone: str, v: float, u: float, p: CapParams = CapParams(),
             steer: Optional[float] = None) -> Tuple[CapState, CapOut]:
    """上限の層の 1 フレーム。zone は ZONES のどれか、v は今の速さ [m/s]、u はモデル(ゲインのあと)の指令(速さの単位)、
    steer は舵の指令(正規化 −1〜1、左 +。分からなければ None)"""
    if zone not in ZONES:
        raise ValueError('safety_caps: 知らない区間です: %r' % (zone,))
    dt = 0.0 if state.t is None else max(0.0, t - state.t)
    target = p.cap_of(zone)
    if target <= state.cap_eff:
        cap_eff = target
    else:
        cap_eff = min(target, state.cap_eff + p.release_accel * dt)
        if cap_eff >= p.cap_free:
            cap_eff = target
    braking, b0, b1 = state.braking, state.brake_t0, state.brake_t1
    in_brake_zone = zone in p.brake_zones
    if p.brake_steer_release is not None and steer is not None and steer <= -p.brake_steer_release:
        in_brake_zone = False          # v1.1 (A): 右へ切り始めた = R1 の旋回の入口
    if braking:
        held = t - b0
        if (not in_brake_zone) or v < p.brake_min_speed or (held >= p.brake_hold_s - 1e-9 and v <= cap_eff):
            braking, b1 = False, t
    start = False
    if (not braking and in_brake_zone and math.isfinite(cap_eff) and v > cap_eff + p.brake_margin and v >= p.brake_min_speed
            and t - b1 >= p.rebrake_lockout_s - 1e-9):
        braking, b0, start = True, t, True
    if braking:
        out_u, mode = min(u, brake_cmd_of_us(p.brake_us)), 'brake'
    elif v > cap_eff:
        out_u = min(u, 0.0)
        mode = 'neutral' if out_u < u else 'pass'
    else:
        c = p.command_of_cap(cap_eff)
        out_u = min(u, c)
        mode = 'cap' if out_u < u else 'pass'
    return CapState(t, cap_eff, braking, b0, b1), CapOut(out_u, mode, cap_eff, zone, start)


def steer_us_to_norm(steer_us: float, center_us: float = 1443.0) -> float:
    """舵の PWM → 正規化(左 +1 = 2000 µs、右 −1 = 1000 µs。ブリッジと同じ)"""
    span = (2000.0 - center_us) if steer_us >= center_us else (center_us - 1000.0)
    return max(-1.0, min(1.0, (steer_us - center_us) / span))


def cap_throttle_us(state: CapState, t: float, zone: str, v: float, thr_us: float,
                    p: CapParams = CapParams(), steer_us: Optional[float] = None,
                    steer_center_us: float = 1443.0) -> Tuple[CapState, float, CapOut]:
    """実車の PWM で使う形。thr_us・steer_us はモデルの出力(ゲインのあと)。返り値の PWM はいつも ≥ 入力(前進を強めない)"""
    u = thr_us_to_cmd(thr_us)
    steer = None if steer_us is None else steer_us_to_norm(steer_us, steer_center_us)
    state, out = cap_step(state, t, zone, v, u, p, steer)
    thr = thr_us if out.u >= u else max(thr_us, cmd_to_thr_us(out.u))
    return state, thr, out


# ------------------------------------------------------------------ ジャイロの位相(区間判定)
class PhaseParams(NamedTuple):
    still_speed: float = 0.025         # |v| がこれ未満でヨーが小さいとき、ずれを更新(|eRPM| < 50)
    moving_speed: float = 0.1          # これより速いときだけ向きを積分(|eRPM| > 200)
    max_dt: float = 0.2                # これより間が空いたサンプルは積分しない
    gz_clip: float = 990.0             # 衝撃で ±1000 dps に張り付くので切る
    turn_on: float = -60.0             # 右旋回の始まり(5 サンプル平均のヨー)
    r1_peak: float = -170.0            # R1 の旋回の最小のヨー
    r1_dpsi: float = -90.0             # R1 の旋回量
    r1_dpsi_short: float = -40.0       # 短い R1(速い周)は、直前の左旋回 > 90 度なら −40 度で認める
    r1_prev_left: float = 60.0         # 直前 2 秒の左旋回の量(LH の出口)
    r1_prev_left_short: float = 90.0
    r1_min_speed: float = 0.75         # 旋回の始まりの速さ(|eRPM| > 1500)
    r1_min_gap_s: float = 5.0          # 前の R1 の起点からの最短の間隔
    app_psi: float = 300.0             # R1 の進入(LH の後半)
    gm_exit_psi: float = -20.0         # GM の出口 = DE の始まり
    de_s: float = 1.5                  # DE の長さ [秒]
    lost_s: float = 16.0               # R1 からこれより経ったら位相は不確か
    psi_max: float = 420.0
    psi_min: float = -300.0
    hist_s: float = 2.0                # 「直前の左旋回」を見る長さ
    stop_speed: float = 0.3            # v1.1 (B): これより遅い状態が
    stop_reset_s: float = 1.0          #           これ以上続いたら位相を不確かにする(inf = v1)
    app_fallback_psi: float = 240.0    # v1.1 (B): R1_app の中で向きがこれを下回ったら不確か(−inf = v1)


class PhaseState(NamedTuple):
    t: Optional[float] = None
    heading: float = 0.0
    bias: float = 0.0
    bias_n: int = 0
    g_hist: Tuple[float, ...] = ()     # 直近 4 サンプルのヨー(ずれを引いたもの)
    h_hist: Tuple[Tuple[float, float], ...] = ()   # (時刻, 向き) を 0.05 秒ごと、直近 3 秒
    turn: Optional[Tuple[float, float, float, float, float]] = None   # 右旋回の候補 (t0, ψ0, 最小ヨー, 直前の左旋回, 始まりの速さ)
    anchor_t: Optional[float] = None   # 直前の R1 の起点(確定したもの)
    anchor_psi: float = 0.0
    stage: str = 'unknown'             # 起点からの段: 'R1_turn' → 'GM' → 'DE' → 'none' → 'R1_app'
    stage_t: float = 0.0               # 今の段に入った時刻
    stop_t0: Optional[float] = None    # 止まり始めた時刻(動いていれば None)


class PhaseOut(NamedTuple):
    zone: str                          # ZONES のどれか(ジャイロだけで決めたもの)
    rel_psi: Optional[float]           # 直前の R1 の起点からの向き [deg]
    turning_right: bool                # 右旋回の候補の中
    r1_confirmed: bool                 # このサンプルで R1 を確定した
    anchor_t: Optional[float]          # 確定した R1 の起点の時刻
    gz_smooth: float                   # 5 サンプル平均のヨー [deg/s]


def _heading_at(h_hist, t_query, default):
    """h_hist(時刻順)から t_query 以前で最後の向き"""
    out = default
    for tt, hh in h_hist:
        if tt <= t_query:
            out = hh
        else:
            break
    return out


def phase_step(state: PhaseState, t: float, gz_dps: float, v: float, p: PhaseParams = PhaseParams()) -> Tuple[PhaseState, PhaseOut]:
    """ジャイロの位相の 1 サンプル(IMU 100 Hz を想定)。v は同じ時刻の速さ(eRPM から)"""
    bias, bias_n = state.bias, state.bias_n
    if abs(v) < p.still_speed and abs(gz_dps) < 5.0:
        bias_n = min(bias_n + 1, 500)
        bias += (gz_dps - bias) / bias_n
    g = max(-p.gz_clip, min(p.gz_clip, gz_dps)) - bias
    heading = state.heading
    if state.t is not None:
        dt = t - state.t
        if 0 < dt <= p.max_dt and abs(v) > p.moving_speed:
            heading += g * dt
    gh = state.g_hist + (g,)
    gs = sum(gh) / len(gh)
    g_hist = gh[-4:]
    h_hist = state.h_hist
    if not h_hist or t - h_hist[-1][0] >= 0.05:
        h_hist = tuple(x for x in h_hist if x[0] >= t - (p.hist_s + 1.0)) + ((t, heading),)
    anchor_t, anchor_psi, stage, stage_t = state.anchor_t, state.anchor_psi, state.stage, state.stage_t
    stop_t0 = (state.stop_t0 if state.stop_t0 is not None else t) if abs(v) < p.stop_speed else None
    if stop_t0 is not None and t - stop_t0 >= p.stop_reset_s and anchor_t is not None:
        anchor_t, stage, stage_t = None, 'unknown', t      # v1.1 (B): 止まった(置き直されたかもしれない)
    turn = state.turn
    confirmed = False
    # ---- 右旋回の候補と R1 の確定
    if gs < p.turn_on:
        if turn is None:
            prev_left = heading - _heading_at(h_hist, t - p.hist_s, h_hist[0][1])
            turn = (t, heading, gs, prev_left, abs(v))
        else:
            turn = (turn[0], turn[1], min(turn[2], gs), turn[3], turn[4])
        t0, psi0, gmin, prev_left, v0 = turn
        dpsi = heading - psi0
        gap_ok = anchor_t is None or t0 - anchor_t > p.r1_min_gap_s
        if (gap_ok and gmin <= p.r1_peak and v0 >= p.r1_min_speed and prev_left > p.r1_prev_left
                and (dpsi <= p.r1_dpsi or (dpsi <= p.r1_dpsi_short and prev_left > p.r1_prev_left_short))):
            anchor_t, anchor_psi, stage, stage_t, confirmed = t0, psi0, 'R1_turn', t0, True
    else:
        turn = None
    turning_right = turn is not None
    # ---- 起点からの段
    rel = None if anchor_t is None else heading - anchor_psi
    if anchor_t is not None and not confirmed:
        if stage == 'R1_turn' and gs >= 0.0:
            stage, stage_t = 'GM', t
        if stage == 'GM' and rel >= p.gm_exit_psi:
            stage, stage_t = 'DE', t
        if stage == 'DE' and t - stage_t >= p.de_s:
            stage, stage_t = 'none', t
        if stage == 'none' and rel >= p.app_psi:
            stage, stage_t = 'R1_app', t
        if stage == 'R1_app' and rel < p.app_fallback_psi:
            anchor_t, stage, stage_t, rel = None, 'unknown', t, None    # v1.1 (B): 引き返した
    lost = anchor_t is None or (t - anchor_t > p.lost_s) or rel is None or rel > p.psi_max or rel < p.psi_min
    if lost:
        zone = 'unknown'
    elif stage == 'R1_app' and turning_right:
        zone = 'R1_turn'           # 確定前の右旋回(ブレーキを出さない)
    else:
        zone = stage
    new = PhaseState(t, heading, bias, bias_n, g_hist, h_hist, turn, anchor_t, anchor_psi, stage, stage_t, stop_t0)
    return new, PhaseOut(zone, rel, turning_right, confirmed, anchor_t, gs)


PHASE_V1 = PhaseParams(stop_reset_s=INF, app_fallback_psi=-INF)
PRED_MIN_SPEED = 1.0          # v1.1 (C)。v1 は 0


def combine_zone(zone_gyro: str, turning_right: bool, pred_signal: bool, v: Optional[float] = None,
                 pred_min_speed: float = PRED_MIN_SPEED) -> str:
    """ジャイロの位相と、モデルの予測ステアの合図 (b) を合わせた区間。
    R1_turn・GM・DE はジャイロを優先((b) は使わない)。(b) だけの合図の中で右旋回が始まったら R1_turn(ブレーキなし)。
    (b) は前へ pred_min_speed 以上のときだけ使う(v が None なら速さは見ない)"""
    if zone_gyro in ('R1_turn', 'GM', 'DE'):
        return zone_gyro
    if pred_signal and v is not None and v < pred_min_speed:
        pred_signal = False
    signal = zone_gyro == 'R1_app' or pred_signal
    if signal and turning_right:
        return 'R1_turn'
    if signal:
        return 'R1_app'
    return zone_gyro


def pred_signal_of(pred_steer_us, threshold_us: float = 1150.0) -> bool:
    """モデルの予測ステア(ゲイン前、先の数フレーム)のどれかが threshold_us 以下(右いっぱい)"""
    return any(x <= threshold_us for x in pred_steer_us)


# ------------------------------------------------------------------ 走行ループの包み(drive.py の zone_caps。2026-10-21)
# 区間の判定(ジャイロの位相)とブレーキの保持などの状態を、1 つの値(ZoneState)にまとめた純粋な包み。
# 入出力・時刻の取得・ハードウェアには触れない(時刻・速さ・ヨー・指令は呼ぶ側が渡す)。同じ入力の列には同じ出力。
# 規則は G1 の再生(experiments/20261016_sim_calibration/g1_replay.py の prepare・shadow、版 v1.1・R1 2.5 m/s)と同じ:
#   - IMU のサンプルごとに phase_step(速さは、そのサンプルの時刻までの最新の車両の eRPM から)
#   - AI のフレームの区間 = そのフレームの撮像時刻までの最後の IMU のサンプルの区間(ジャイロ)と、
#     モデルの予測ステアの合図 (b)(ゲイン前の予測のどれかが 1150 µs 以下・AI 側のフレームだけ)を combine_zone で合わせたもの
#   - 上限の層は AI 側のフレーム(AI 側・中立で止めていない・推論の確認でない・車両の値が 0.1 秒以内)だけで動かす。
#     それ以外のフレームで状態を戻す。AI 側のフレームの間が 0.5 秒を超えても戻す
#   - 層への指令は、車に出す直前のスロットル(ゲイン・ToF ガード・補助のあと)。舵は車に出す舵(中心 1443 µs で正規化)
class ZoneCapConfig(NamedTuple):
    cap: CapParams = CapParams()             # 実車の v1.1(R1 2.5・GM 2.2・DE 3.0 m/s、上限 → 指令は較正の T⁻¹)
    phase: PhaseParams = PhaseParams()       # v1.1 の位相
    pred_min_speed: float = PRED_MIN_SPEED   # v1.1 (C)
    pred_threshold_us: float = 1150.0        # 予測ステアの合図 (b)
    reset_gap_s: float = 0.5                 # AI 側のフレームの間がこれを超えたら層の状態を戻す
    fresh_s: float = 0.1                     # 車両の値がこれより古いフレームは層を動かさない
    steer_center_us: float = 1443.0          # 舵の正規化の中心(ブリッジ・G1 の再生と同じ)


ZONE_CAPS_VERSION = 'v1.1'
ZONE_CAPS_V1_1 = ZoneCapConfig()
CAP_MODES = ('pass', 'cap', 'neutral', 'brake')     # 記録の番号 0〜3(-1 = 層を動かしていない、-2 = 計算の失敗)


class ZoneState(NamedTuple):
    phase: PhaseState = PhaseState()
    cap: CapState = CapState()
    imu_t: Optional[float] = None      # 最後に入れた IMU のサンプルの時刻
    zone_gyro: str = 'unknown'         # そのサンプルの区間(ジャイロだけ)
    turning: bool = False              # そのサンプルで右旋回の候補の中
    rel_psi: Optional[float] = None    # そのサンプルの、直前の R1 の起点からの向き [deg]
    last_active_t: float = -INF        # 最後に層を動かしたフレームの時刻


class ZoneOut(NamedTuple):
    zone: str                  # このフレームの区間(ジャイロ + 予測ステアの合図)
    zone_gyro: str
    pred_signal: bool          # 予測ステアの合図 (b)(速さの条件の前)
    rel_psi: Optional[float]
    active: bool               # 層を動かしたか(AI 側のフレーム)
    mode: str                  # 'off'(動かしていない)/ 'pass' / 'cap' / 'neutral' / 'brake'
    cap_eff: float             # 今の上限 [m/s](なしは inf)
    thr_us: float              # 上限を入れたら出すスロットル [µs](いつも ≥ 入力 = 前進を強めない)。'off' なら入力のまま
    brake_start: bool


def zone_imu(state: ZoneState, t: float, gz_dps: float, v: float, cfg: ZoneCapConfig = ZONE_CAPS_V1_1) -> ZoneState:
    """IMU の 1 サンプル(時刻 t [秒]、ヨー gz [deg/s]、そのときの速さ v [m/s])。時刻が前のサンプルより前なら捨てる"""
    if state.imu_t is not None and t < state.imu_t:
        return state
    ph, o = phase_step(state.phase, t, gz_dps, v, cfg.phase)
    return state._replace(phase=ph, imu_t=t, zone_gyro=o.zone, turning=o.turning_right, rel_psi=o.rel_psi)


def zone_frame(state: ZoneState, t: float, thr_us: float, steer_us: float, v: float, pred_steer_us, active: bool,
               cfg: ZoneCapConfig = ZONE_CAPS_V1_1) -> Tuple[ZoneState, ZoneOut]:
    """AI のフレームの 1 回(撮像時刻 t [秒])。thr_us・steer_us は車に出す直前の値、v は撮像時刻までの最新の車両の速さ、
    pred_steer_us はモデルの予測ステア(ゲイン前、先の数フレーム)、active は層を動かすか(AI 側のフレーム)。
    返り値の thr_us を出すかどうかは呼ぶ側が決める(zone_caps が OFF なら出さない = 影の判定だけ)"""
    pred_sig = bool(active) and pred_signal_of(pred_steer_us, cfg.pred_threshold_us)
    zone = combine_zone(state.zone_gyro, state.turning, pred_sig, v=v, pred_min_speed=cfg.pred_min_speed)
    if not active:
        return (state._replace(cap=CapState()),
                ZoneOut(zone, state.zone_gyro, pred_sig, state.rel_psi, False, 'off', INF, thr_us, False))
    cap = state.cap if t - state.last_active_t <= cfg.reset_gap_s else CapState()
    cap, thr, o = cap_throttle_us(cap, t, zone, v, thr_us, cfg.cap, steer_us, cfg.steer_center_us)
    return (state._replace(cap=cap, last_active_t=t),
            ZoneOut(zone, state.zone_gyro, pred_sig, state.rel_psi, True, o.mode, o.cap_eff, thr, o.brake_start))


def zone_inactive(state: ZoneState) -> ZoneState:
    """中立で止めたフレーム(記録しない・層を動かさない)。層の状態だけ戻す(位相はそのまま)"""
    return state._replace(cap=CapState())


# ------------------------------------------------------------------ 速さの天井(drive.py の speed_ceiling。2026-10-25)
# 仕様: jetformer_reports/speed_control_20261025/spec.md 2 章。区間の判定(ジャイロの位相)は使わない。使うのは速さ・舵・ヨーだけ。
#   天井 C = 全体 cap_all(4.5 m/s)。左旋回 L(車に出す舵 ≥ +100 µs または 5 サンプル平均のヨー ≥ +45 deg/s、両方が
#   下回ってから 0.25 秒は L のまま)の間は cap_left(4.0 m/s)。L が終わったら 1.5 m/s² で cap_all へ戻す。
#   前進を PI で弱めて C に保つ(0.1 秒先の速さで早めに)。v ≥ C + 0.2 m/s の間は中立(惰性)。
#   いつも「入力以上(前進を強めない)・中立以下(ブレーキを出さない)」。舵は変えない。
# 仕様からの 2 つの直し(build_drive.md。偽物の車の閉ループで決めた。どちらも False にすると仕様の文のまま):
#   start_from_hold: 天井の手前(e < 0)で掛け始めるとき、許す量を入力のまま(仕様の手順 3)でなく
#       min(入力, f_hold(C) − Kp·e) から始める。入力(例 400 µs)から始めると、PI が保つ量(約 290 µs)まで下げるのに
#       大きな行き過ぎが要り(電池 b = 1.0 で C + 0.25 m/s・中立 6 フレーム)、試験 D-c4(≤ C + 0.15)に落ちる
#   release_bumpless: 外す(手順 7)のは、e < −0.2 が 0.2 秒続き、かつ PI の許す量が入力まで戻ったときだけ。
#       戻る前に外すと、入力の前進がいきなり戻り、掛け直しの行き過ぎをくり返す(左旋回 4.0 m/s の中で 4.32 m/s)
class CeilParams(NamedTuple):
    cap_all: float = 4.5               # 全体の天井 [m/s]
    cap_left: float = 4.0              # 左旋回中の天井 [m/s](cap_all より大きければ cap_all。走行の既定は drive.DEFAULT_PARAMS の ceil_left_mps)
    left_steer_us: float = 100.0       # 左旋回: 車に出す舵(中心 + トリムから、左 +)がこれ以上
    left_gz_dps: float = 45.0          # 左旋回: ヨー(5 サンプル平均、左 +)がこれ以上
    left_hold_s: float = 0.25          # 両方が下回ってからも左旋回とみる時間
    release_accel: float = 1.5         # 左旋回が終わったあと天井を戻す速さ [m/s²]
    pred_s: float = 0.10               # 先の速さ v̂ = v + clip(a, 0, accel_max) × pred_s
    accel_max: float = 3.0
    engage_e: float = -0.10            # e = v̂ − C がこれ以上で掛け始める [m/s]
    kp: float = 120.0                  # µs / (m/s)
    ki: float = 380.0                  # µs / (m/s·s)
    dt_max: float = 0.1                # PI の Δt の上限 [秒]
    neutral_margin: float = 0.20       # v ≥ C + これ の間は中立 [m/s]
    release_e: float = -0.20           # e がこれ未満で
    release_s: float = 0.20            # これだけ続いたら外す [秒]
    reset_gap_s: float = 0.5           # 動かすフレームの間がこれを超えたら状態を戻す [秒]
    hold_slope: float = 26.314         # f_hold: 落ち着いた直線の eRPM = slope × 指令 + offset(10/24 の 20 本の当てはめ)
    hold_offset: float = 1343.5
    start_from_hold: bool = True       # 仕様からの直し(上の説明)。False = 仕様の文のまま
    release_bumpless: bool = True      # 仕様からの直し(上の説明)。False = 仕様の文のまま

    def cap_left_eff(self) -> float:
        return min(self.cap_left, self.cap_all)


CEIL_OFF, CEIL_PASS, CEIL_LIMIT, CEIL_NEUTRAL = -1, 0, 1, 2     # 記録の ceil_mode(-2 = 計算の失敗は drive.py が入れる)
CEIL_MODE_NAMES = {-2: '計算の失敗', -1: '動かしていない', 0: 'そのまま', 1: '弱めた', 2: '中立にした'}


def ceil_params(cap_all: float = 4.5, cap_left: float = 4.0, **kw) -> CeilParams:
    """走行のパラメータ(ceil_all_mps・ceil_left_mps)から。左旋回中の天井は全体より大きくしない"""
    cap_all = float(cap_all)
    return CeilParams(cap_all=cap_all, cap_left=min(float(cap_left), cap_all), **kw)


def ceil_hold_us(cap: float, p: CeilParams = CeilParams()) -> float:
    """速さ cap [m/s] を保つ前進の量の目安 f_hold [µs](中立から)。4.0 m/s → 253、4.5 m/s → 291"""
    return max(0.0, (cap / KV - p.hold_offset) / p.hold_slope)


class CeilState(NamedTuple):
    t: Optional[float] = None          # 前に動かしたフレームの時刻
    cap: Optional[float] = None        # 今の天井 C
    left_t: float = -INF               # 左旋回の条件が最後に成り立った時刻
    engaged: bool = False              # 掛けている
    allow: float = INF                 # 許す前進の量 A [µs](中立から)
    e_prev: float = 0.0                # 前のフレームの e
    below_t0: Optional[float] = None   # e < release_e が始まった時刻


class CeilOut(NamedTuple):
    active: bool                       # 動かしたか(AI のフレーム)
    left: bool                         # 左旋回 L
    cap: float                         # 天井 C [m/s](動かしていなければ nan)
    v: float                           # 使った速さ
    v_pred: float                      # 先の速さ v̂(動かしていなければ nan)
    mode: int                          # CEIL_OFF / CEIL_PASS / CEIL_LIMIT / CEIL_NEUTRAL
    thr_us: float                      # 天井ならこう出すスロットル [µs](いつも 入力以上・中立以下。前進でない入力はそのまま)
    allow_us: float                    # 許す前進の量 A(掛けていなければ inf)
    engaged: bool


def ceil_step(state: CeilState, t: float, v: float, a: float, steer_rel_us: float, gz5_dps: float, thr_in_us: float,
              neutral_us: float, active: bool, p: CeilParams = CeilParams()) -> Tuple[CeilState, CeilOut]:
    """速さの天井の 1 フレーム。t 撮像時刻 [秒]、v 速さ [m/s](前へ +)、a 加速 [m/s²](分からなければ nan = 0 とみる)、
    steer_rel_us 車に出す舵 − (中心 + トリム) [µs](左 +)、gz5_dps ヨーの 5 サンプル平均 [deg/s](左 +)、
    thr_in_us 天井への入力のスロットル [µs]、neutral_us スロットルの中立、active 動かすか(AI のフレーム)。
    入出力・時刻の取得・ハードウェアには触れない。同じ入力の列には同じ出力"""
    nan = float('nan')
    if not active:
        return CeilState(), CeilOut(False, False, nan, v, nan, CEIL_OFF, thr_in_us, INF, False)
    if state.t is not None and t - state.t > p.reset_gap_s:
        state = CeilState()
    dt_raw = 0.0 if state.t is None else max(0.0, t - state.t)
    dt = min(dt_raw, p.dt_max)
    raw_left = steer_rel_us >= p.left_steer_us or gz5_dps >= p.left_gz_dps        # nan は成り立たない
    left_t = t if raw_left else state.left_t
    left = t - left_t <= p.left_hold_s + 1e-9
    target = p.cap_left_eff() if left else p.cap_all
    if state.cap is None or target <= state.cap:
        cap = target                                   # 下げるのはすぐ
    else:
        cap = min(target, state.cap + p.release_accel * dt_raw)
    dropped = state.cap is not None and cap < state.cap
    a_c = min(max(a, 0.0), p.accel_max) if math.isfinite(a) else 0.0
    v_pred = v + a_c * p.pred_s
    e = v_pred - cap
    f_in = neutral_us - thr_in_us
    if not f_in > 0.0:                 # 中立・ブレーキ・後退(と nan)はそのまま。掛けている状態を外す
        return (CeilState(t, cap, left_t, False, INF, e, None),
                CeilOut(True, left, cap, v, v_pred, CEIL_PASS, thr_in_us, INF, False))
    engaged, allow, below_t0 = state.engaged, state.allow, state.below_t0
    f_hold = ceil_hold_us(cap, p)
    if not engaged:
        if e >= p.engage_e:            # 掛け始め
            engaged, below_t0 = True, None
            if e >= 0.0:
                allow = min(f_in, f_hold)
            elif p.start_from_hold:
                allow = min(f_in, f_hold - p.kp * e)
            else:
                allow = f_in
    else:
        if dropped:
            allow = min(allow, f_hold)
        allow = min(max(allow - p.kp * (e - state.e_prev) - p.ki * e * dt, 0.0), f_in)
    if engaged:
        f_out = min(f_in, allow)
        if v >= cap + p.neutral_margin:
            f_out = 0.0
        if e < p.release_e:
            below_t0 = t if below_t0 is None else below_t0
            if t - below_t0 >= p.release_s - 1e-9 and (not p.release_bumpless or allow >= f_in):
                engaged, allow, below_t0 = False, INF, None
        else:
            below_t0 = None
    else:
        f_out = f_in
    if f_out >= f_in:
        mode, thr = CEIL_PASS, thr_in_us
    elif f_out <= 0.0:
        mode, thr = CEIL_NEUTRAL, neutral_us
    else:
        mode, thr = CEIL_LIMIT, min(max(neutral_us - f_out, thr_in_us), neutral_us)
    return (CeilState(t, cap, left_t, engaged, allow, e, below_t0),
            CeilOut(True, left, cap, v, v_pred, mode, thr, allow, engaged))
