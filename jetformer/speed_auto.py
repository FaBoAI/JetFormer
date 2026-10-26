"""速さのゲインの自動合わせ(drive.py の speed_auto。2026-10-25)の純粋な関数

仕様: jetformer_reports/speed_control_20261025/spec.md 3 章。直し: 同じフォルダの fix.md(2026-10-25)。
入出力・時刻の取得・ハードウェアには触れない。同じ入力には同じ出力を返す(走行ループ側のつなぎは drive._SpeedAuto)。

  掛ける速さのゲイン g = throttle_gain × m(上げるのは max(throttle_gain, 1.10) まで)
  m = clip(m_ff × m_fb, 0.85, 1.20)。掛ける m は 1 秒に 0.03 まで(走り出す前はそのまま)
  m_ff = 電圧の前送り clip((auto_vref_v ÷ V)^1, 0.90, 1.15)。1 秒に 0.01 まで(走り出す前はそのまま)
       電圧の源: 1 = 基板(直近 3 秒の 90% 点、100 サンプル以上。走り出す前は 20 以上)・2 = 説明欄の V_before=・0 = なし(1.00)
       どの源も 6.0〜9.0 V(2S)の外は「無し」と同じ(fix.md: 分圧の外れ・張り付きでゲインを上げない)
  m_fb = 周ごとの直し。R1 の起点ごとに、**周の時間** を目標 T* = 14.52 − 1.50 × auto_target_mps(4.27 → 8.12 秒)へ。
       不感帯 ±0.10 秒(8.12〜8.32 秒 = 了承の「1 周 約 8.1〜8.3 秒」)、Δ = clip(0.5 × ln(周 ÷ T*), ±0.03)、m_fb は 0.90〜1.10。
       使う周は 7〜16 秒(7 秒未満は偽の起点の周)。周の中の短い中立止め(4 フレームまで)は使う。
       上げない: 天井が 35% 以上のフレームで出力を変えた周・3 軸の衝撃 6.0 G 以上の周・衝突で使えなかった周の次の周。
       偽の起点(前に使った起点から向き < 270°)は使わない
       (仕様の物差しは周の速さ p90 だった。天井 ON では p90 が天井で頭打ちになり目標 4.20 に届かないので、周の時間に替えた。
        metric='p90' で仕様の文のまま = 不感帯 ±0.05 m/s・天井 10% で上げない)
"""
import math
from typing import NamedTuple, Optional, Tuple

NAN = float('nan')
METRIC_LAP, METRIC_P90 = 'lap_s', 'p90'


class AutoParams(NamedTuple):
    target_mps: float = 4.20           # 周の速さ p90 の目標(auto_target_mps。走行の既定は drive.DEFAULT_PARAMS の 4.27 = 1 周 8.12 秒)。周の時間の目標はここから(target_lap_s)
    vref_v: float = 8.35               # throttle_gain を決めた電圧(auto_vref_v)
    alpha: float = 1.0                 # 同じ指令の速さ ∝ 電圧^alpha
    m_lo: float = 0.85
    m_hi: float = 1.20
    gain_raise_max: float = 1.10       # 上げるのは max(throttle_gain, これ) まで
    m_rate: float = 0.03               # 掛ける m の変化 [1/秒]
    ff_lo: float = 0.90
    ff_hi: float = 1.15
    ff_rate: float = 0.01              # m_ff の変化 [1/秒]
    fb_lo: float = 0.90
    fb_hi: float = 1.10
    deadband: float = 0.05             # (metric p90)|p90 − 目標| がこれ以下なら直さない [m/s]
    lam: float = 0.5                   # 1 周で足りない分の何割を直すか
    step_max: float = 0.03             # 1 周の直しの上限(ln)
    lap_min_s: float = 5.0
    lap_max_s: float = 16.0
    gap_max_s: float = 0.2             # 周の中のフレームの間の上限
    impact_g: float = 2.5              # 水平の衝撃(重力を除かない hypot(ax, ay) ÷ g)がこれ以上 = 衝突
    ceil_frac_max: float = 0.10        # (metric p90)天井が出力を変えたフレームがこれ以上の周は上げない
    board_g: float = 6.0               # 3 軸の衝撃 |‖a‖/g − 1| がこれ以上の周は上げない(R1 の板)
    false_anchor_deg: float = 270.0    # 前に使った起点からの向きの変化がこれ未満の起点は使わない
    stop_v: float = 0.3                # これより遅い状態が
    stop_s: float = 1.0                # これ以上続いた周は使わない
    rail_laps: int = 3                 # 端に何周続けて張り付いたら知らせるか
    vbat_window_s: float = 3.0
    vbat_min_n: int = 100
    vbat_q: float = 90.0
    vbat_stale_s: float = 0.5
    vdesc_lo: float = 6.0              # 説明欄の電圧として受け付ける範囲 [V]
    vdesc_hi: float = 9.0
    moving_v: float = 0.3              # 走り出した = 一度でもこれ以上の速さ
    # ---- 直し(fix.md、2026-10-25)
    metric: str = METRIC_LAP           # 周の直しの物差し: 'lap_s' = 周の時間(既定)/ 'p90' = 周の速さ p90(仕様の文のまま)
    lap_a: float = 14.52               # 周の時間 ≈ lap_a − lap_b × p90(B の 60 周、相関 −0.91)。目標の周の時間を作る
    lap_b: float = 1.50
    deadband_s: float = 0.10           # (metric lap_s)|周の時間 − 目標| がこれ以下なら直さない [秒]
    ceil_frac_max_lap: float = 0.35    # (metric lap_s)天井が出力を変えたフレームがこれ以上の周は上げない
    hold_max_n: int = 4                # 周の中の短い中立止め(約 0.13 秒)はこの数まで使う(人・dry run は 1 つでも使わない)
    vboard_lo: float = 6.0             # 基板の電圧として受け付ける範囲 [V](2S。外れた値は分圧の外れ・張り付き = 無し)
    vboard_hi: float = 9.0
    vbat_min_n_still: int = 20         # 走り出す前は、基板の電圧をこの数(約 0.2 秒)で出してよい(止まっていれば下がりは無い)
    ff_jump_max: float = 0.02          # 走り出したあとの基板の最初の値は、今の m_ff との差がこれ以下のときだけそのまま入れる
    osc_min_step: float = 0.02         # 「揺れています」は、この大きさ以上の直し(ln)の向きの入れ替わりだけ数える
    lap_min_s_lap: float = 7.0         # (metric lap_s)これより短い周は使わない(天井 4.5 m/s で 29.7 m は 6.6 秒より速くならない。
                                       # 残った偽の起点の周 5.0〜5.3 秒(10/23 No.17・No.23)で目安を下げないため)

    def target_lap_s(self) -> float:
        """目標の周の時間 T* [秒](4.20 m/s → 8.22 秒、4.27 m/s → 8.12 秒)"""
        return self.lap_a - self.lap_b * self.target_mps


SRC_NONE, SRC_BOARD, SRC_DESC = 0, 1, 2
(EV_NONE, EV_FIXED, EV_UNUSED, EV_NORAISE_CEIL, EV_NORAISE_BOARD, EV_NORAISE_COLL, EV_DEADBAND, EV_FALSE_ANCHOR,
 EV_ABANDON_COLL) = range(9)
EVENT_NAMES = {EV_NONE: '', EV_FIXED: '直した', EV_UNUSED: '周を使わない', EV_NORAISE_CEIL: '上げない(天井)',
               EV_NORAISE_BOARD: '上げない(板の衝撃)', EV_NORAISE_COLL: '上げない(衝突の次)', EV_DEADBAND: '不感帯',
               EV_FALSE_ANCHOR: '偽の起点', EV_ABANDON_COLL: '閉じられない周(衝突。次の周は上げない)'}
SKIP_REASONS = ('short', 'long', 'not_ai', 'gap', 'assist', 'guard', 'collision', 'stopped')
SKIP_NAMES = {'short': '短すぎる周(7 秒未満。p90 の物差しは 5 秒未満)', 'long': '16 秒超', 'not_ai': 'AI でないフレーム(人・dry run・長い中立止め)',
              'gap': 'フレームの間 0.2 秒超', 'assist': '補助(リカバリーなど)', 'guard': 'ToF ガード',
              'collision': '衝突(水平 2.5G 以上)', 'stopped': '1 秒以上止まった'}
WARN_RAIL, WARN_OSC = 'rail', 'osc'
WARN_TEXT = {WARN_RAIL: '自動合わせが端に張り付いています(車・電池を確かめる)',
             WARN_OSC: '自動合わせが揺れています(上げ下げの向きが 3 周続けて入れ替わった)'}
OSC_SIGNS = 4                          # 向きが 3 回続けて入れ替わる = 続けて直した 4 周が +−+− か −+−+


def auto_params(target_mps: float = 4.20, vref_v: float = 8.35, **kw) -> AutoParams:
    return AutoParams(target_mps=float(target_mps), vref_v=float(vref_v), **kw)


def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def rate_limit(prev: Optional[float], target: float, max_step: float) -> float:
    """prev から target へ、1 回に max_step まで(prev が None ならそのまま)"""
    if prev is None:
        return target
    return clip(target, prev - max_step, prev + max_step)


def percentile(xs, q: float) -> float:
    """numpy.percentile(線形の内挿)と同じ値。空なら nan"""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return NAN
    pos = (n - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def board_v_ok(v, p: AutoParams = AutoParams()) -> bool:
    """電圧として使える値か(数で、6.0〜9.0 V)。外れた値は分圧の外れ・張り付き(0.5 V・12.75 V など)とみて使わない"""
    return isinstance(v, (int, float)) and math.isfinite(v) and p.vboard_lo <= v <= p.vboard_hi


def ff_of_voltage(v: float, p: AutoParams = AutoParams()) -> float:
    """電圧 V → 前送り m_ff。V が数でない・6.0〜9.0 V の外なら 1.00(前送りなし)"""
    if not board_v_ok(v, p):
        return 1.0
    return clip((p.vref_v / v) ** p.alpha, p.ff_lo, p.ff_hi)


def vbat_level(samples, p: AutoParams = AutoParams(), min_n: Optional[int] = None) -> float:
    """直近 3 秒の基板の電圧の列 → 上の包絡(90% 点)。6.0〜9.0 V の外は数えない。
    使える値が min_n(既定 vbat_min_n = 100)未満なら nan"""
    xs = [x for x in samples if board_v_ok(x, p)]
    if len(xs) < (p.vbat_min_n if min_n is None else min_n):
        return NAN
    return percentile(xs, p.vbat_q)


def v_before_of(description: str, p: AutoParams = AutoParams()) -> float:
    """説明欄の V_before=8.31(後ろの V は付いていてもよい)→ 電圧。無い・6.0〜9.0 V の外なら nan。
    人が書いた V_before が無く、基板の Vb_before= だけのときはそれを使う"""
    vals = {}
    for tok in (description or '').split():
        for key in ('V_before', 'Vb_before'):
            if tok.startswith(key + '=') and key not in vals:
                s = tok.split('=', 1)[1].strip().rstrip('Vv')
                try:
                    vals[key] = float(s)
                except ValueError:
                    pass
    for key in ('V_before', 'Vb_before'):
        x = vals.get(key)
        if x is not None and math.isfinite(x) and p.vdesc_lo <= x <= p.vdesc_hi:
            return x
    return NAN


def gain_of(throttle_gain: float, m: float, p: AutoParams = AutoParams()) -> float:
    """掛ける速さのゲイン g = throttle_gain × m。上げるのは max(throttle_gain, 1.10) まで"""
    tg = float(throttle_gain)
    return min(tg * m, max(tg, p.gain_raise_max))


def m_of(m_ff: float, m_fb: float, p: AutoParams = AutoParams()) -> float:
    return clip(m_ff * m_fb, p.m_lo, p.m_hi)


# ------------------------------------------------------------------ 周
class LapFrame(NamedTuple):
    t: float                           # 撮像時刻 [秒]
    v: float                           # 速さ [m/s](天井と同じ 3 サンプルの中央値)
    ai: bool                           # AI のフレーム(AI 側・dry run でない・中立で止めていない)
    assist: int                        # 補助の種類(0 = なし)
    guard: bool                        # ToF ガードが出力を変えた
    impact_g: float                    # このフレームまでに届いた IMU の水平の衝撃の最大 [G]
    a3_g: float                        # 同じく 3 軸の衝撃 |‖a‖/g − 1| の最大 [G]
    ceil_lim: bool                     # 天井が出力を変えた(OFF の日は「変えたはず」)
    hold: bool = False                 # 中立で止めたフレーム(周の中に hold_max_n までなら周を使う。fix.md)
    gain: float = NAN                  # 実際に掛けた速さのゲイン(AI のフレーム。走り終わりの目安に使う)


class LapStats(NamedTuple):
    t0: float
    t1: float
    dur: float
    n: int
    v90: float
    ok: bool                           # 直しに使える周
    reason: str                        # 使わない理由(SKIP_REASONS のどれか。使えれば '')
    collision: bool
    ceil_frac: float
    a3_max: float
    g_mean: float = NAN                # AI のフレームで掛けた速さのゲインの平均(無ければ nan)
    holds: int = 0                     # 周の中の中立止めのフレームの数


def lap_stats(frames, t0: float, t1: float, p: AutoParams = AutoParams()) -> LapStats:
    """[t0, t1) のフレームから周の数。frames は LapFrame の列(時刻の順)。
    1 回だけ見る(周を閉じるフレームの時間を短くする。fix.md)。使わない理由の順は short・long・not_ai・gap・assist・
    guard・collision・stopped"""
    dur = t1 - t0
    lap_min = p.lap_min_s if p.metric == METRIC_P90 else max(p.lap_min_s, p.lap_min_s_lap)
    n = n_ceil = n_hold = 0
    v_ai = []
    g_sum, g_n = 0.0, 0
    other = assist = guard = collision = stopped = gap = False
    a3_max = 0.0
    t_prev = slow_t0 = None
    for f in frames:
        if f.t < t0:
            continue
        if f.t >= t1:
            break
        n += 1
        if (f.t - t0 if t_prev is None else f.t - t_prev) > p.gap_max_s:
            gap = True
        t_prev = f.t
        if f.ai:
            v_ai.append(f.v)
            if math.isfinite(f.gain):
                g_sum += f.gain
                g_n += 1
        elif f.hold:
            n_hold += 1
        else:
            other = True
        if f.assist:
            assist = True
        if f.guard:
            guard = True
        if f.impact_g >= p.impact_g:
            collision = True
        if f.ceil_lim:
            n_ceil += 1
        if f.a3_g > a3_max:
            a3_max = f.a3_g
        if not stopped:
            if f.v < p.stop_v:                      # nan(中立止めのフレーム)は、止まった・走ったのどちらにも数えない
                slow_t0 = f.t if slow_t0 is None else slow_t0
                stopped = f.t - slow_t0 >= p.stop_s
            elif f.v >= p.stop_v:
                slow_t0 = None
    if t_prev is not None and t1 - t_prev > p.gap_max_s:
        gap = True
    v90 = percentile(v_ai, 90.0)
    ceil_frac = (n_ceil / n) if n else 0.0
    reason = ''
    if dur < lap_min:
        reason = 'short'
    elif dur > p.lap_max_s:
        reason = 'long'
    elif n == 0 or other or n_hold > p.hold_max_n or not v_ai:
        reason = 'not_ai'
    elif gap:
        reason = 'gap'
    elif assist:
        reason = 'assist'
    elif guard:
        reason = 'guard'
    elif collision:
        reason = 'collision'
    elif stopped:
        reason = 'stopped'
    if not reason and not math.isfinite(v90):
        reason = 'not_ai'
    return LapStats(t0, t1, dur, n, v90, not reason, reason, collision, ceil_frac, a3_max,
                    g_sum / g_n if g_n else NAN, n_hold)


class FbState(NamedTuple):
    m_fb: float = 1.0
    no_raise_next: bool = False        # 衝突で使えなかった周の次の 1 周は上げない
    rail_n: int = 0                    # 使った周のあと端にいた周の数(続けて)
    signs: Tuple[int, ...] = ()        # 続けて直した周の向き(+1/−1、直近 4 つ)。直さなかった周(不感帯・上げない)で切れる
    warn: str = ''                     # 今の知らせ WARN_RAIL / WARN_OSC / ''
    warn_seen: Tuple[str, ...] = ()    # この走行で出た知らせ
    laps_used: int = 0
    skipped: Tuple[Tuple[str, int], ...] = ()      # (理由, 数)
    ratios: Tuple[Tuple[float, ...], ...] = ()     # 使った周の (1 回で合わせる倍率 r, その周の m_ff, 掛けたゲインの平均,
                                                   #           上げてよい周か = 天井・板の衝撃で上げない周でない)
    last_v90: float = NAN
    last_lap_s: float = NAN


def _count(pairs, key):
    d = dict(pairs)
    d[key] = d.get(key, 0) + 1
    return tuple(sorted(d.items()))


def lap_ratio(lap: LapStats, p: AutoParams = AutoParams()) -> float:
    """1 回で目標に合わせるならゲインに掛ける倍率 r。周の時間: 周 ÷ T*(周の時間 ∝ 1 ÷ ゲインとみる。天井があると
    周の時間の効きは弱い(約 0.6〜0.9)ので、r は控えめ = 安全側)。p90: 目標 ÷ p90"""
    if p.metric == METRIC_P90:
        return p.target_mps / lap.v90
    return lap.dur / p.target_lap_s()


def fb_lap(state: FbState, lap: LapStats, m_ff: float, p: AutoParams = AutoParams()) -> Tuple[FbState, int, float]:
    """周を 1 つ閉じる → (状態, 出来事 EV_*, 掛けた Δ(ln))"""
    if not lap.ok:
        return (state._replace(no_raise_next=bool(lap.collision), skipped=_count(state.skipped, lap.reason),
                               last_v90=lap.v90, last_lap_s=lap.dur), EV_UNUSED, 0.0)
    r = lap_ratio(lap, p)
    used = state.laps_used + 1
    if p.metric == METRIC_P90:
        in_band, cf_max = abs(lap.v90 - p.target_mps) <= p.deadband, p.ceil_frac_max
    else:
        in_band, cf_max = abs(lap.dur - p.target_lap_s()) <= p.deadband_s + 1e-12, p.ceil_frac_max_lap
    raise_ok = lap.ceil_frac < cf_max and lap.a3_max < p.board_g
    ratios = state.ratios + ((r, float(m_ff), float(lap.g_mean), bool(raise_ok)),)
    if in_band:
        event, delta = EV_DEADBAND, 0.0
    else:
        delta = clip(p.lam * math.log(r), -p.step_max, p.step_max)
        event = EV_FIXED
        if delta > 0:
            if lap.ceil_frac >= cf_max:
                event, delta = EV_NORAISE_CEIL, 0.0
            elif lap.a3_max >= p.board_g:
                event, delta = EV_NORAISE_BOARD, 0.0
            elif state.no_raise_next:
                event, delta = EV_NORAISE_COLL, 0.0
    m_fb = clip(state.m_fb * math.exp(delta), p.fb_lo, p.fb_hi)
    at_rail = m_fb <= p.fb_lo + 1e-9 or m_fb >= p.fb_hi - 1e-9
    rail_n = state.rail_n + 1 if at_rail else 0
    if event == EV_FIXED and abs(delta) >= p.osc_min_step - 1e-12:
        signs = (state.signs + (1 if delta > 0 else -1,))[-OSC_SIGNS:]
    else:
        signs = ()                     # 不感帯・上げない・小さい直し(周のばらつきくらい)で切れる
    warn = ''
    if rail_n >= p.rail_laps:
        warn = WARN_RAIL
    elif len(signs) == OSC_SIGNS and all(signs[i] == -signs[i + 1] for i in range(OSC_SIGNS - 1)):
        warn = WARN_OSC
    seen = state.warn_seen + ((warn,) if warn and warn not in state.warn_seen else ())
    return (FbState(m_fb, False, rail_n, signs, warn, seen, used, state.skipped, ratios, lap.v90, lap.dur), event, delta)


def fb_abandon(state: FbState, collision: bool) -> Tuple[FbState, int]:
    """閉じられなかった周(起点を忘れた: 止まった・人に替わった・位相が不確かに戻った)。衝突があったら、その周は
    「使わない(衝突)」と数え、次の 1 周は上げない(fix.md。衝突 → 人が持って止める、のいちばん多い形)"""
    if not collision:
        return state, EV_NONE
    return state._replace(no_raise_next=True, skipped=_count(state.skipped, 'collision')), EV_ABANDON_COLL


def _median(xs):
    return percentile(xs, 50.0)


def end_summary(state: FbState, on: bool, throttle_gain: float, p: AutoParams = AutoParams()) -> dict:
    """走り終わりの目安(auto_end)。throttle_gain を変えるのは人(1 走行 = 1 設定)。
    使った周ごとに「その周で目標に合ったはずのゲイン」= 掛けたゲインの平均 × r(平均が無ければ throttle_gain × r)を作り、
    その中央値から: 次の走行で自動合わせを使うなら ÷ その周の m_ff(電圧の分は次の走行の前送りが入れる)を throttle_gain の
    0.90〜1.10 倍に、使わないならそのまま 0.85〜1.20 倍に丸める。ON・OFF とも同じ式(仕様の ON は throttle_gain × m_fb_end。
    3 周の予選では m_fb がほとんど動かないので、1 回で合わせる値に替えた。fix.md)。
    自動合わせと同じ歯止め: 上げない周(天井 35% 以上・板の衝撃 6G 以上)からは上げる目安を出さない(r を 1 まで)。
    上げる目安は max(throttle_gain, 1.10) まで(下げる目安はそのまま)"""
    tg = float(throttle_gain)
    out = {'on': bool(on), 'm_fb_end': round(state.m_fb, 4), 'laps_used': state.laps_used,
           'laps_skipped': dict(state.skipped), 'warn': state.warn, 'warn_text': WARN_TEXT.get(state.warn, ''),
           'warn_seen': list(state.warn_seen), 'metric': p.metric,
           'target_mps': p.target_mps, 'target_lap_s': round(p.target_lap_s(), 3),
           'suggest_throttle_gain': None, 'suggest_throttle_gain_auto': None, 'suggest_throttle_gain_fixed': None,
           'suggest_laps': 0}
    need = []
    for row in state.ratios:
        r, mff = float(row[0]), float(row[1])
        g = float(row[2]) if len(row) > 2 else NAN
        if not (math.isfinite(g) and g > 0):
            g = tg
        if len(row) > 3 and not row[3]:
            r = min(r, 1.0)
        if math.isfinite(r) and r > 0 and math.isfinite(mff) and mff > 0:
            need.append((g * r, mff))
    if need and tg > 0:
        out['suggest_laps'] = len(need)
        fixed = _median([x for x, _ in need])
        auto = _median([x / mff for x, mff in need])
        top = max(tg, p.gain_raise_max)
        out['suggest_throttle_gain_fixed'] = round(min(clip(fixed, tg * p.m_lo, tg * p.m_hi), top), 3)
        out['suggest_throttle_gain_auto'] = round(min(clip(auto, tg * p.fb_lo, tg * p.fb_hi), top), 3)
        out['suggest_throttle_gain'] = out['suggest_throttle_gain_auto']
    return out
