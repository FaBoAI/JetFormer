"""AI走行の補助(モデルの出力を受け取り、実際に車へ出す値を返す)

このESCの後退(2026-10-10 のログ116回分から確認。data/reports/ai_runs_20261010/human_recovery.md):
  ・前進の直後の後退側の指令は、必ずブレーキになる(何秒中立を挟んでも、前へ転がっている間は後退しない)
  ・止まってから(または後ろへ転がっている間に)後退側 → 中立(約0.12秒)→ 後退側 で下がる(89%)
  ・下がり始めるのは後退側を出してから約0.08秒後
人のリカバリー: ブレーキ約0.27秒 → 中立 → 後退(約1712µs)を約0.3秒 → 惰性で下がる → 止まる前の曲がる向きへ
ハンドルを切って前進(約1369µs・0.6秒)→ AIへ戻す。

1. 自動リカバリー(Recovery)
   検出: (a) 前へ走っていて衝撃(水平2.5G以上)を受け、0.6秒以内に止まり、そのまま STUCK_AFTER_IMPACT_S 進まない
         (b) 一度動いてから、前へも後ろへも進まない状態が STUCK_S 続いた
   動作: ブレーキ → 止まるのを待つ → 中立 → 後退(強さ rev_us・時間 rev_s。下がり始めなければ中立を長くしてやり直し)
         → 中立で惰性 → 止まる前の曲がる向きへハンドルを切って前進(ESCAPE)→ モデルへ戻す
   後退中に衝撃を受けたら(後ろにぶつかった)すぐ前進へ。GIVEUP_N 回が GIVEUP_WINDOW_S 秒以内に続いたら、
   あきらめて中立のまま人の交代を待つ(CH3を人側へ戻すと解除)。

2. 後退の受け付け(EscReverse) — リカバリーをOFFにしたときだけ使う
   モデル自身の後退の指令を、止まってから ブレーキ → 中立 → 後退 の順に出す(強さ・長さに上限)。
   リカバリーがONのときは、止まっている間のモデルの後退は中立にして、下がるのはリカバリーに任せる
   (2026-10-10、AIが自分で長く下がって後ろにぶつかった例があるため、下がる量を決まった範囲にする)。

時刻は秒(time.monotonic())で渡す。どちらも人が操縦中(select=False)なら何もしない。
"""

import collections

import numpy as np

ASSIST_NONE, ASSIST_ESC, ASSIST_RECOVERY, ASSIST_GIVEUP = 0, 1, 2, 3
ASSIST_NAMES = {ASSIST_NONE: '', ASSIST_ESC: '後退の受け付け', ASSIST_RECOVERY: 'リカバリー', ASSIST_GIVEUP: 'リカバリー中止'}

DEFAULT_REV_US = 300.0       # 後退の強さ(中立から)。人は約+210µs。2026-10-10 の +180µs は弱かった
DEFAULT_REV_S = 0.4          # 下がり始めてから後退を出し続ける時間(その後は惰性)
GAP_OFFSET_US = 10.0         # 中立を挟むときの値(中立 1478〜1538µs の帯の中で、わずかに後退側)
STOPPED_ERPM = -150          # これより遅く前へ転がっている(ほぼ止まった)ら、中立 → 後退 が効く
BACKING_ERPM = 300           # これより速く後ろへ動いていたら「下がっている」
CHECK_S = 0.35               # 後退を出してから、この時間で下がり始めなければやり直す(人は0.08秒で動く)


class Recovery(object):
    STOP_ERPM = 300
    ARM_ERPM = 1000          # 一度この速さで動いてから止まったときだけ検出する(発進前・スタート直後は対象外)
    STUCK_S = 0.8
    STUCK_AFTER_IMPACT_S = 0.5
    IMPACT_G = 2.5
    BRAKE_S = 0.25
    WAIT_STOP_S = 1.0        # ブレーキで止まるのを待つ上限(止まらなければもう一度ブレーキ)
    GAP_S = 0.15
    RETRY_GAP_S = 0.3
    RETRY_EXTRA_US = 50.0
    MAX_RETRIES = 2
    MAX_REV_TOTAL_S = 1.2    # 後退を出す長さの上限(下がっていなくても)
    REV_STEER_US = 150.0     # 後退中のハンドル: 止まる前と反対へ少しだけ(人はほぼ真ん中)
    COAST_S = 0.25
    ESCAPE_US = 130.0        # 前進(中立から)
    ESCAPE_S = 0.5
    ESCAPE_STEER_FRAC = 0.85 # 前進のハンドル: 止まる前に曲がっていた向きへ
    COOLDOWN_S = 1.0
    GIVEUP_N = 3
    GIVEUP_WINDOW_S = 20.0
    MAX_REV_US = 500.0

    def __init__(self, ref, steer_range=500.0):
        self.center, self.n = float(ref[0]), float(ref[1])
        self.range = float(steer_range)
        self.count = 0
        self.history = collections.deque(maxlen=60)      # (t, モデルのステア)
        self.starts = collections.deque(maxlen=self.GIVEUP_N)
        self.reset()

    def reset(self):
        self.phase = None            # None / brake / wait / gap / reverse / coast / escape / giveup
        self.t_phase = 0.0
        self.slow_since = None
        self.impact_t = None
        self.impact_moving = False
        self.cooldown_until = 0.0
        self.armed = False
        self.retries = 0
        self.backed_s = 0.0
        self.t_last = None
        self.turn = 0.0              # 止まる前に曲がっていた向き(+1 左 / -1 右 / 0 まっすぐ)
        self.starts.clear()          # 人が交代して直したら、あきらめの数え直し

    @property
    def active(self):
        return self.phase is not None

    def _go(self, phase, now):
        self.phase, self.t_phase = phase, now

    def _start(self, now):
        self.count += 1
        self.starts.append(now)
        if len(self.starts) >= self.GIVEUP_N and now - self.starts[0] <= self.GIVEUP_WINDOW_S:
            self._go('giveup', now)
            return
        before = [s for t, s in self.history if now - t <= 1.5 and t <= (self.slow_since or now)]
        dev = (float(np.median(before)) if before else self.center) - self.center
        self.turn = 0.0 if abs(dev) < 0.2 * self.range else float(np.sign(dev))
        self.retries = 0
        self.backed_s = 0.0
        self._go('brake', now)

    def _detect(self, model_us, erpm, now, impact_g):
        self.history.append((now, model_us[0]))
        if impact_g >= self.IMPACT_G:
            self.impact_t, self.impact_moving = now, erpm < -500 or self.impact_moving
        if self.impact_t is not None and now - self.impact_t > 1.2:
            self.impact_t, self.impact_moving = None, False
        if abs(erpm) >= self.ARM_ERPM:
            self.armed = True
        if not self.armed or now < self.cooldown_until:
            self.slow_since = None
            return False
        if -self.STOP_ERPM < erpm < BACKING_ERPM:          # 前へも後ろへも進んでいない
            if self.slow_since is None:
                self.slow_since = now
            need = self.STUCK_AFTER_IMPACT_S if (self.impact_t is not None and self.impact_moving) else self.STUCK_S
            return now - self.slow_since >= need
        self.slow_since = None
        return False

    def step(self, model_us, erpm, now, impact_g=0.0, rev_us=DEFAULT_REV_US, rev_s=DEFAULT_REV_S):
        """戻り値 (出す値 (steer, throttle) または None(モデルのまま), 補助の種類)"""
        dt = 0.0 if self.t_last is None else min(max(now - self.t_last, 0.0), 0.2)
        self.t_last = now
        if self.phase == 'giveup':
            return (self.center, self.n), ASSIST_GIVEUP
        if self.phase is None:
            if not self._detect(model_us, erpm, now, impact_g):
                return None, ASSIST_NONE
            self._start(now)
            if self.phase == 'giveup':
                return (self.center, self.n), ASSIST_GIVEUP
        rev = min(rev_us + self.RETRY_EXTRA_US * self.retries, self.MAX_REV_US)
        rev_steer = self.center - self.turn * self.REV_STEER_US
        el = now - self.t_phase
        if self.phase == 'brake':                       # 前進の直後の後退側は必ずブレーキ(捨てる1回目)
            if el < self.BRAKE_S:
                return (self.center, self.n + rev), ASSIST_RECOVERY
            self._go('wait', now)
            el = 0.0
        if self.phase == 'wait':                        # 前へ転がっている間は後退しないので、止まるのを待つ
            if erpm < STOPPED_ERPM:
                if el < self.WAIT_STOP_S:
                    return (self.center, self.n + rev), ASSIST_RECOVERY
                self._go('brake', now)
                return (self.center, self.n + rev), ASSIST_RECOVERY
            self._go('gap', now)
            el = 0.0
        if self.phase == 'gap':
            gap_s = self.RETRY_GAP_S if self.retries else self.GAP_S
            if el < gap_s:
                return (rev_steer, self.n + GAP_OFFSET_US), ASSIST_RECOVERY
            self._go('reverse', now)
            el = 0.0
        if self.phase == 'reverse':
            if impact_g >= self.IMPACT_G and el > 0.1:     # 後ろにぶつかった: すぐ前へ
                self._go('escape', now)
                return self._escape(now)
            if erpm > BACKING_ERPM:
                self.backed_s += dt
            if self.backed_s == 0.0 and el > CHECK_S:      # 下がり始めない: ESCが後退に入っていない
                if self.retries < self.MAX_RETRIES:
                    self.retries += 1
                    self._go('gap', now)
                    return (rev_steer, self.n + GAP_OFFSET_US), ASSIST_RECOVERY
                self._go('escape', now)                    # やり直しても下がらない: 前進で抜けてみる
                return self._escape(now)
            if self.backed_s < rev_s and el < self.MAX_REV_TOTAL_S:
                return (rev_steer, self.n + rev), ASSIST_RECOVERY
            self._go('coast', now)
            el = 0.0
        if self.phase == 'coast':                       # 中立で惰性(人は後退をすぐ離している)
            if el < self.COAST_S and erpm > 1000:
                return (rev_steer, self.n), ASSIST_RECOVERY
            self._go('escape', now)
        if self.phase == 'escape':
            return self._escape(now)
        return None, ASSIST_NONE

    def _escape(self, now):
        """止まる前に曲がっていた向きへハンドルを切って前進(後退から前進はすぐ切り替わる)→ 終わったらモデルへ"""
        if now - self.t_phase < self.ESCAPE_S:
            steer = self.center + self.turn * self.ESCAPE_STEER_FRAC * self.range
            return (steer, self.n - self.ESCAPE_US), ASSIST_RECOVERY
        self.phase = None
        self.slow_since = None
        self.impact_t, self.impact_moving = None, False
        self.cooldown_until = now + self.COOLDOWN_S
        return None, ASSIST_NONE


class EscReverse(object):
    """リカバリーをOFFにしたとき、モデル自身の後退の指令を ESC が受け付けるように出す"""
    DEADBAND_US = 40.0       # 中立から これ以上 大きい = 後退側(ブレーキ)
    BRAKE_S = 0.2
    GAP_S = 0.15
    RETRY_GAP_S = 0.3
    MAX_RETRIES = 2
    REV_MAX_S = 1.0          # 実際に下がり続けてよい長さ(その後 REST_S は中立)
    REST_S = 0.5

    def __init__(self, neutral):
        self.n = float(neutral)
        self.reset()

    def reset(self):
        self.state = 'idle'          # idle / brake / wait / gap / reverse / rest
        self.t_state = 0.0
        self.retries = 0
        self.backing_s = 0.0
        self.t_last = None

    def _go(self, state, now):
        self.state, self.t_state = state, now

    def step(self, thr, erpm, now, rev_us=DEFAULT_REV_US):
        """戻り値 (出す値, 補助したか)"""
        dt = 0.0 if self.t_last is None else min(max(now - self.t_last, 0.0), 0.2)
        self.t_last = now
        if self.state == 'rest' and now - self.t_state < self.REST_S:
            return (self.n, True) if thr > self.n + self.DEADBAND_US else (thr, False)
        if thr <= self.n + self.DEADBAND_US:            # 前進・中立: そのまま(手順は解除)
            self.reset()
            self.t_last = now
            return thr, False
        if self.state == 'rest':
            self.reset()
            self.t_last = now
        want = min(thr, self.n + rev_us)
        el = now - self.t_state
        if self.state == 'idle':
            self._go('brake', now)
            el = 0.0
        if self.state == 'brake':
            if el < self.BRAKE_S or erpm < STOPPED_ERPM:   # まだ前へ転がっている間はブレーキのまま
                return thr, False
            self._go('gap', now)
            el = 0.0
        if self.state == 'gap':
            if el < (self.RETRY_GAP_S if self.retries else self.GAP_S):
                return self.n + GAP_OFFSET_US, True
            self._go('reverse', now)
            el = 0.0
        if erpm > BACKING_ERPM:
            self.backing_s += dt
            if self.backing_s > self.REV_MAX_S:
                self._go('rest', now)
                return self.n, True
        elif self.backing_s == 0.0 and el > CHECK_S and self.retries < self.MAX_RETRIES:
            self.retries += 1
            self._go('gap', now)
            return self.n + GAP_OFFSET_US, True
        return want, want != thr


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

    def step(self, model_us, erpm, select, now, esc_on=True, rec_on=True, impact_g=0.0,
             rev_us=DEFAULT_REV_US, rev_s=DEFAULT_REV_S):
        """戻り値 (出す値 (steer, throttle), 補助の種類)"""
        if not select:
            self.reset()
            return model_us, ASSIST_NONE
        if rec_on:
            o, k = self.rec.step(model_us, erpm, now, impact_g, rev_us, rev_s)
            if o is not None:
                self.esc.reset()
                return (float(o[0]), float(o[1])), k
            # 止まっている間のモデルの後退は中立に(下がるのはリカバリーに任せる。前へ走っている間はブレーキとしてそのまま)
            if model_us[1] > self.ref[1] + EscReverse.DEADBAND_US and erpm >= STOPPED_ERPM:
                return (model_us[0], self.ref[1]), ASSIST_ESC
            return model_us, ASSIST_NONE
        if self.rec.active:
            self.rec.reset()
        if esc_on:
            thr, changed = self.esc.step(model_us[1], erpm, now, rev_us)
            if changed:
                if not self._esc_active:
                    self.esc_count += 1
                self._esc_active = True
                return (model_us[0], float(thr)), ASSIST_ESC
            self._esc_active = False
        return model_us, ASSIST_NONE

    @property
    def recoveries(self):
        return self.rec.count
