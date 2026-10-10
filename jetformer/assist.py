"""AI走行の補助(モデルの出力を受け取り、実際に車へ出す値を返す)

1. 後退の受け付け(EscReverse)
   このESCは「前進 → 後退」の指令をブレーキとして扱い、いったん中立に戻してからでないと後退しない。
   人は 後退側(ブレーキ)→ 中立(約0.15秒)→ 後退側 と操作して下がっている。2026-10-10 の ai_run_007 では、
   ぶつかって止まった後にAIが後退を1秒以上出し続けても(中立を挟まないので)車は動かなかった。
   止まっている(ほぼ止まっている)ときに後退が求められたら、ブレーキ → 中立 → 後退 の順に出す。
   後退は出しすぎないよう、強さ(中立から REV_CAP_US まで)と続ける長さ(REV_MAX_S)を制限する。

2. ぶつかって止まったときの自動リカバリー(Recovery)
   AI側なのに |eRPM| が小さいまま STUCK_S 秒(衝撃の直後は STUCK_AFTER_IMPACT_S 秒)続いたら、
   ブレーキ → 中立 → 後退(止まる前に切っていた向きと反対へハンドルを切る)→ 中立 → モデルへ戻す。
   戻してから COOLDOWN_S 秒は検出しない。GIVEUP_N 回が GIVEUP_WINDOW_S 秒以内に続いたら、あきらめて中立のまま
   人の交代を待つ(CH3を人側へ戻すと解除)。

時刻は秒(time.monotonic())で渡す。どちらも人が操縦中(select=False)なら何もしない。
"""

import collections

import numpy as np

ASSIST_NONE, ASSIST_ESC, ASSIST_RECOVERY, ASSIST_GIVEUP = 0, 1, 2, 3
ASSIST_NAMES = {ASSIST_NONE: '', ASSIST_ESC: '後退の受け付け', ASSIST_RECOVERY: 'リカバリー', ASSIST_GIVEUP: 'リカバリー中止'}


class EscReverse(object):
    DEADBAND_US = 40.0       # 中立から これ以上 大きい = 後退側(ブレーキ)
    MOVING_ERPM = 600        # これより速く前進中なら、後退側はブレーキとしてそのまま出す
    BRAKE_S = 0.12           # 止まっていても、まずブレーキ(後退側)を出す時間
    GAP_S = 0.2              # 中立を挟む時間
    REV_CAP_US = 250.0       # 後退の強さの上限(中立から)
    REV_MAX_S = 1.0          # 後退を続ける長さの上限(その後はモデルが後退をやめるまで中立)

    def __init__(self, neutral):
        self.n = float(neutral)
        self.reset()

    def reset(self):
        self.state = 'idle'          # idle / brake / gap / reverse / rest
        self.t_state = 0.0

    def _go(self, state, now):
        if self.state != state:
            self.state, self.t_state = state, now

    def step(self, thr, erpm, now):
        """戻り値 (出す値, 補助したか)"""
        if thr <= self.n + self.DEADBAND_US:            # 前進・中立: そのまま(後退の手順は解除)
            self.state = 'idle'
            return thr, False
        if self.state == 'rest':                        # 後退を長く出しすぎた: モデルが後退をやめるまで中立
            return self.n, True
        if self.state == 'reverse' or erpm > 300:       # 下がっている(後退が効いている)
            self._go('reverse', now)
            if now - self.t_state > self.REV_MAX_S:
                self._go('rest', now)
                return self.n, True
            capped = min(thr, self.n + self.REV_CAP_US)
            return capped, capped != thr
        if erpm < -self.MOVING_ERPM:                    # まだ前へ走っている: ブレーキとしてそのまま
            self._go('brake', now)
            return thr, False
        if self.state == 'idle':
            self._go('brake', now)
        if self.state == 'brake':
            if now - self.t_state < self.BRAKE_S:
                return thr, False
            self._go('gap', now)
        if now - self.t_state < self.GAP_S:             # gap: 中立を挟む
            return self.n, True
        self._go('reverse', now)
        return min(thr, self.n + self.REV_CAP_US), True


class Recovery(object):
    STOP_ERPM = 300
    ARM_ERPM = 1000          # 一度この速さで動いてから止まったときだけ検出する(発進前・スタート直後は対象外)
    STUCK_S = 0.8
    STUCK_AFTER_IMPACT_S = 0.3
    IMPACT_G = 2.5           # 水平方向の加速度(重力を除く)
    BRAKE_S = 0.12
    GAP_S = 0.2
    REV_US = 180.0           # 後退の強さ(中立から)
    REV_S = 0.6
    REV_STEER_FRAC = 0.6     # 後退中のハンドル: 止まる前と反対へ、いっぱいの何割か
    END_S = 0.15             # 後退の後の中立
    COOLDOWN_S = 1.5
    GIVEUP_N = 3
    GIVEUP_WINDOW_S = 10.0

    def __init__(self, ref, steer_range=500.0):
        self.center, self.n = float(ref[0]), float(ref[1])
        self.range = float(steer_range)
        self.count = 0
        self.history = collections.deque(maxlen=60)      # (t, モデルのステア)
        self.starts = collections.deque(maxlen=self.GIVEUP_N)
        self.reset()

    def reset(self):
        self.phase = None            # None / brake / gap / reverse / end / giveup
        self.t_phase = 0.0
        self.slow_since = None
        self.impact_t = None
        self.cooldown_until = 0.0
        self.rev_steer = self.center
        self.armed = False
        self.starts.clear()            # 人が交代して直したら、あきらめの数え直し

    def _start(self, now):
        self.count += 1
        self.starts.append(now)
        if len(self.starts) >= self.GIVEUP_N and now - self.starts[0] <= self.GIVEUP_WINDOW_S:
            self.phase, self.t_phase = 'giveup', now
            return
        before = [s for t, s in self.history if now - t <= 1.2 and t <= (self.slow_since or now)]
        pre = float(np.median(before)) if before else self.center
        dev = pre - self.center
        side = 0.0 if abs(dev) < 0.2 * self.range else -np.sign(dev)
        self.rev_steer = self.center + side * self.REV_STEER_FRAC * self.range
        self.phase, self.t_phase = 'brake', now

    def step(self, model_us, erpm, now, impact_g=0.0):
        """戻り値 (出す値 (steer, throttle) または None(モデルのまま), 補助の種類)"""
        if self.phase == 'giveup':
            return (self.center, self.n), ASSIST_GIVEUP
        if self.phase is None:
            self.history.append((now, model_us[0]))
            if impact_g >= self.IMPACT_G:
                self.impact_t = now
            if abs(erpm) >= self.ARM_ERPM:
                self.armed = True
            if not self.armed:
                return None, ASSIST_NONE
            if now < self.cooldown_until:
                self.slow_since = None
                return None, ASSIST_NONE
            if abs(erpm) < self.STOP_ERPM:
                if self.slow_since is None:
                    self.slow_since = now
                need = self.STUCK_S
                if self.impact_t is not None and now - self.impact_t < 1.0:
                    need = self.STUCK_AFTER_IMPACT_S
                if now - self.slow_since >= need:
                    self._start(now)
                    if self.phase == 'giveup':
                        return (self.center, self.n), ASSIST_GIVEUP
                else:
                    return None, ASSIST_NONE
            else:
                self.slow_since = None
                return None, ASSIST_NONE
        el = now - self.t_phase
        if self.phase == 'brake':
            if el < self.BRAKE_S:
                return (self.rev_steer, self.n + self.REV_US), ASSIST_RECOVERY
            self.phase, self.t_phase, el = 'gap', now, 0.0
        if self.phase == 'gap':
            if el < self.GAP_S:
                return (self.rev_steer, self.n), ASSIST_RECOVERY
            self.phase, self.t_phase, el = 'reverse', now, 0.0
        if self.phase == 'reverse':
            if el < self.REV_S:
                return (self.rev_steer, self.n + self.REV_US), ASSIST_RECOVERY
            self.phase, self.t_phase, el = 'end', now, 0.0
        if self.phase == 'end':
            if el < self.END_S:
                return (self.center, self.n), ASSIST_RECOVERY
        # 終わり: モデルへ戻す
        self.phase = None
        self.slow_since = None
        self.impact_t = None
        self.cooldown_until = now + self.COOLDOWN_S
        return None, ASSIST_NONE


class Assist(object):
    """走行ループから使う窓口。人が操縦中・出力しない確認中は何もしない"""

    def __init__(self, ref):
        self.ref = (float(ref[0]), float(ref[1]))
        self.esc = EscReverse(self.ref[1])
        self.rec = Recovery(self.ref)
        self.esc_count = 0
        self._esc_active = False

    def reset(self):
        self.esc.reset()
        self.rec.reset()
        self._esc_active = False

    def step(self, model_us, erpm, select, now, esc_on=True, rec_on=True, impact_g=0.0):
        """戻り値 (出す値 (steer, throttle), 補助の種類)"""
        if not select:
            self.reset()
            return model_us, ASSIST_NONE
        out, kind = model_us, ASSIST_NONE
        if rec_on:
            o, k = self.rec.step(model_us, erpm, now, impact_g)
            if o is not None:
                return (float(o[0]), float(o[1])), k
        elif self.rec.phase is not None:
            self.rec.reset()
        if esc_on:
            thr, changed = self.esc.step(model_us[1], erpm, now)
            if changed:
                out, kind = (model_us[0], float(thr)), ASSIST_ESC
                if not self._esc_active:
                    self.esc_count += 1
            self._esc_active = changed
        return out, kind

    @property
    def recoveries(self):
        return self.rec.count
