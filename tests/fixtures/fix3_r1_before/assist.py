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
   後ろの見張り(back_guard。drive の rec_back_guard、2026-10-26 fix2 safety2-1。既定 OFF。OFF は今までと同じ):
   直前のモデルの指令がブレーキ側だと、ESC は後退を待っている。そのとき最初の「ブレーキ」がそのまま後退になる
   (10/23 20261023-205458-0002: 'brake'・'gap' の間に下がり始め、'reverse' が 0 から数え直して 0.5 秒後退を出し、
   1.24 m 下がって後ろに 5.8G で当たった)。ON のときは、どの段でも
   ・後ろへ動いている(eRPM > BACKING_ERPM)なら、もう下がっているとして数える('brake'・'wait' → すぐ 'reverse'。
     'gap' の惰性も数える)。下がった時間が rev_s に届いたら惰性('coast')へ
   ・後ろへ動いている・直前 BACK_RECENT_S まで動いていたときの衝撃(IMPACT_G 以上)は、後ろにぶつかったとしてすぐ前進
     ('escape'。やり直しの +50 µs はしない)
   中立止めのあとも残す(Assist.reset の keep_recovery。drive の rec_hold_keep、2026-10-26 fix3 safety1-1。既定 OFF。
   OFF は今までと同じ = 中立止め 1 回で Recovery を全部戻す): ON のとき、AI 側のままの中立止め(カメラ 0.1 秒・処理の遅れ・
   センサーが古い・I2C の失敗など。1 フレームでもよい)では Recovery を戻さない。中止(giveup)は中止のまま、途中の段は続きから
   (段の時間は中立止めの間も進む)、armed・衝撃・あきらめの数も残す。戻すのは CH3 を人側にしたとき(step の select = False)と
   走行の始め(drive は走行ごとに Assist を作る)。後退の受け付け(EscReverse)は今まで通り戻す。
   drive は後ろの見張り(rec_back_guard)も ON のときだけ使う(残した後退の途中で後ろに当たったとき、見張りが無いと
   'reverse' のやり直しが後ろを押し続けるため。今までは中立止めの reset が手順を偶然切っていた)

2. 後退の受け付け(EscReverse) — リカバリーをOFFにしたときだけ使う
   モデル自身の後退の指令を、止まってから ブレーキ → 中立 → 後退 の順に出す(強さ・長さに上限)。
   リカバリーがONのときは、止まっている間のモデルの後退は中立にして、下がるのはリカバリーに任せる
   (2026-10-10、AIが自分で長く下がって後ろにぶつかった例があるため、下がる量を決まった範囲にする)。

3. 発進の補助(StartKick)・発進の前進(LaunchBoost、2026-10-26。既定 OFF)
   発進の前進: CH3 を AI 側にした最初のフレームで、止まっていて前が空いていれば、前進を少なくとも launch_us(中立から)
   出す。速さ・時間・ヨー・舵・衝撃・前が近い・CH3・中立で止めた・車両の値 のどれかで終わり、同じ AI の区間ではもう
   出さない(StartKick も出さない)。仕様は jetformer_reports/fix_20261026/protocol_v3.md 14 章。
   始めなかった区間(前が近い・ToF が無い など)では、仕様どおり今まで通り StartKick が出る(protocol_v3.json launch_boost.start)。
   launch_no_kick(drive の同じ名前、2026-10-26 fix3 safety1-2。既定 OFF。OFF は今までと同じ)が ON なら、発進の前進が
   「出さない」と決めた区間では StartKick も出さない(launch_boost = 0 のときは決めないので効かない)。

時刻は秒(time.monotonic())で渡す。どちらも人が操縦中(select=False)なら何もしない。
"""

import collections
import math
from typing import NamedTuple, Tuple

import numpy as np

ASSIST_NONE, ASSIST_ESC, ASSIST_RECOVERY, ASSIST_GIVEUP, ASSIST_START, ASSIST_LAUNCH = 0, 1, 2, 3, 4, 5
ASSIST_NAMES = {ASSIST_NONE: '', ASSIST_ESC: '後退の受け付け', ASSIST_RECOVERY: 'リカバリー', ASSIST_GIVEUP: 'リカバリー中止',
                ASSIST_START: '発進の補助', ASSIST_LAUNCH: '発進の前進'}

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
    BACK_RECENT_S = 0.3      # 後ろの見張り: 衝撃の直前これだけの間に後ろへ動いていたら「後ろにぶつかった」

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
        self.back_t = None           # 後ろの見張り: この手順の中で最後に後ろへ動いていた時刻
        self.guard_event = None      # 後ろの見張りが手順を変えた (何, 段, 下がった秒)。drive.py が記録に書いて None に戻す

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
        self.back_t = None
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

    def _guard(self, erpm, now, dt, impact_g, rev_s):
        """後ろの見張り(back_guard = True のときだけ呼ぶ)。手順を変えたら戻り値 (出す値, 種類)、変えなければ None
        (段を進めただけのときも None。あとの段の処理がそのまま続ける)"""
        backing = erpm > BACKING_ERPM
        if backing:
            self.back_t = now
        if impact_g >= self.IMPACT_G and self.back_t is not None and now - self.back_t <= self.BACK_RECENT_S:
            self.guard_event = ('rear_impact', self.phase, round(self.backed_s, 3))     # 後ろにぶつかった: すぐ前へ
            self._go('escape', now)
            return self._escape(now)
        if not backing:
            return None
        if self.phase in ('brake', 'wait'):            # 前のブレーキ側で ESC が後退を待っていた: もう下がっている
            self.guard_event = ('back_early', self.phase, round(self.backed_s, 3))
            self._go('reverse', now)                   # 下がった時間は 'reverse' が数える(このフレームから)
        elif self.phase == 'gap':                      # 中立の惰性で下がっている分も数える
            self.backed_s += dt
            if self.backed_s >= rev_s:
                self.guard_event = ('back_enough', self.phase, round(self.backed_s, 3))
                self._go('coast', now)
        return None

    def step(self, model_us, erpm, now, impact_g=0.0, rev_us=DEFAULT_REV_US, rev_s=DEFAULT_REV_S, back_guard=False):
        """戻り値 (出す値 (steer, throttle) または None(モデルのまま), 補助の種類)。
        back_guard = 後ろの見張り(既定 False = 今までと同じ。モジュールの説明 1 の終わり)"""
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
        if back_guard and self.phase in ('brake', 'wait', 'gap', 'reverse', 'coast'):
            g = self._guard(erpm, now, dt, impact_g, rev_s)
            if g is not None:
                return g
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


class StartKick(object):
    """発進の補助(2026-10-23 ユーザーの依頼「AIモードにしたら、すぐ走り出すように」)。
    CH3 を AI 側にした直後、止まっていてモデルが前へ進まない(中立付近)とき、弱い前進(中立 − KICK_US)を出して走り出させる。
    - 一度 MOVE_ERPM より速く前へ動いたら終わり(その AI の区間ではもう出さない。CH3 を人側へ戻すと次の AI 側で 1 回)
    - MAX_S 出しても動かなければやめる。前が近い(blocked)・モデルが後退かブレーキを出しているときは出さない
    - モデルがもっと強く前へ出しているときは、モデルのまま(補助は前進を弱めない)"""
    KICK_US = 80.0           # 中立から前へ(約 1.2 m/s の定常。人の発進は 1379〜1423 µs)
    MOVE_ERPM = -800         # これより速く前へ動いたら発進できた(eRPM は負 = 前進)
    MAX_S = 2.0

    def __init__(self, neutral_us):
        self.n = float(neutral_us)
        self.reset()

    def reset(self):
        self.done = False
        self.t0 = None

    def step(self, thr, erpm, now, blocked=False):
        """戻り値: 出すスロットル [µs]、補助しないなら None"""
        if self.done:
            return None
        if erpm <= self.MOVE_ERPM:
            self.done = True
            return None
        if self.t0 is None:
            self.t0 = now
        if now - self.t0 > self.MAX_S:
            self.done = True
            return None
        if blocked or thr > self.n + EscReverse.DEADBAND_US:
            return None
        want = self.n - self.KICK_US
        return want if thr > want else None


# ---------------------------------------------------------------- 発進の前進(2026-10-26。既定 OFF)
# 終わりの理由(AI_CMD の launch_end。0 = このフレームでは終わっていない)。1〜9 は protocol_v3 14 章の ①〜⑨。
# 10 は仕様に無い足し: 走行中に launch_boost を 1 → 0 にした(build_launch.md)
LAUNCH_END_NAMES = {0: '', 1: '速さに届いた', 2: '最大の時間', 3: 'ヨー 60 deg/s', 4: '舵 200 µs(1 m/s 以上)',
                    5: '衝撃 2.5G', 6: '前が近い・ToF ガード', 7: 'CH3 を人側', 8: '中立で止めた・走行の停止',
                    9: '車両の値が古い・後ろへ動いた', 10: 'OFF にした'}
# 始めなかった理由(記録の出来事と画面だけ。AI_CMD には書かない)
LAUNCH_SKIP_NAMES = {'off': 'OFF', 'no_switch': 'AI 側にしたのを見ていない(走行の始めから AI 側)',
                     'hold': '中立で止めたあと', 'model': 'モデルの出力が数でない', 'moving': '止まっていない(|eRPM| ≥ 400)',
                     'veh': '車両の値が古い', 'tof': 'ToF が無い・古い・ガード 0', 'ttc_n': 'TTC が 5 フレームそろわない',
                     'near': '前が近い(TTC の中央値 < 3.3 秒)', 'end': '始めのフレームで終わりの条件'}


class LaunchIn(NamedTuple):
    """発進の前進の 1 フレームの入力(drive.py が launch_boost = 1 のときだけ作る。OFF は None)"""
    us: float = 250.0                  # launch_us: 出す前進の下限 [µs](中立から。ゲインは掛けない)
    mps: float = 2.5                   # launch_mps: 0.1 秒先の速さの見込みがこれに届いたら終わり [m/s]
    max_s: float = 1.2                 # launch_max_s [秒]
    v: float = float('nan')            # 速さ [m/s](速さの天井と同じ: 車両の値 3 つの中央値)
    a: float = float('nan')            # 加速 [m/s²](速さの天井と同じ: 0.1 秒前の v との差)
    gz: float = float('nan')           # ヨー [deg/s](IMU 5 サンプルの平均、左 +)
    steer_rel: float = float('nan')    # 車に出す舵 − 中心 − trim [µs]
    ttc: float = float('nan')          # このフレームの ToF ガードの TTC [秒]
    ttc5: Tuple[float, ...] = ()       # 直近のフレームの TTC(このフレームを含む。中立で止めたら空にする)
    tof_ok: bool = False               # ToF ガードがある(tof_guard ≥ 1)・ToF の値が 0.1 秒以内
    veh_ok: bool = False               # 車両の値が 0.1 秒以内
    impact_g: float = 0.0              # 水平の衝撃 [G](直近 0.15 秒の最大)
    guard_changed: bool = False        # ToF ガードがこのフレームの出力を変えた


class LaunchBoost(object):
    """発進の前進(protocol_v3 14 章)。AI 側にした最初のフレームで、止まっていて前が空いていれば始め、
    スロットル = min(モデル, 中立 − us) を出す(前進だけ。止まっている間のモデルのブレーキは無視。舵はモデルのまま)。
    終わったら(どの理由でも)同じ AI の区間ではもう出さない。CH3 を人側へ戻すと、次の AI 側で 1 回"""
    US_RANGE = (150.0, 300.0)          # drive.PARAM_LIMITS と同じ(念のためここでも切る)
    MPS_RANGE = (2.0, 3.0)
    MAX_S_RANGE = (0.5, 2.0)
    START_ERPM = 400.0                 # |eRPM| < これ = 止まっている
    TTC_START_S = 3.3                  # 直近 5 フレームの TTC の中央値がこれ以上(止まっていれば前 1.0 m 以上空き)
    TTC_N = 5
    END_YAW_DPS = 60.0                 # ③
    END_STEER_US = 200.0               # ④(END_STEER_MPS 以上のとき)
    END_STEER_MPS = 1.0
    END_IMPACT_G = 2.5                 # ⑤
    END_TTC_S = 0.8                    # ⑥ v ≥ END_TTC_MPS で TTC < これが END_TTC_N フレーム続いた
    END_TTC_N = 2
    END_TTC_MPS = 0.3
    END_BACK_ERPM = 200.0              # ⑨ eRPM > これ = 後ろへ動いた(eRPM は負 = 前進)
    PRED_S, ACCEL_MAX = 0.10, 3.0      # ① 0.1 秒先の見込み v + clip(a, 0, 3) × 0.1(safety_caps.CeilParams と同じ)
    US_DEFAULT, MPS_DEFAULT, MAX_S_DEFAULT = 250.0, 2.5, 1.2   # 数でない値のとき(drive.DEFAULT_PARAMS・LaunchIn と同じ)

    def __init__(self, neutral_us):
        self.n = float(neutral_us)
        self.count = 0                 # 始めた回数(stats の launch_boosts)
        self.armed = False             # この走行で人側のフレームを見た(「AI 側にした」と分かる)
        self.last_end = 0              # 最後に終わった理由
        self.last_skip = ''            # 最後に始めなかった理由(LAUNCH_SKIP_NAMES の鍵)
        self.event = None              # このフレームの出来事 (種類, 情報)。drive.py が記録の出来事に書く
        self._section()

    def _section(self):
        self.decided = False           # この AI の区間で、始めるかを決めた
        self.active = False
        self.used = False              # この AI の区間で出した(終わったあとも StartKick を出さない)
        self.skipped = False           # この AI の区間で「出さない」と決めた(ON のときだけ。launch_no_kick で StartKick も出さない)
        self.held = False              # 人側のフレームのあとに中立で止めた
        self.t0 = None
        self.cfg = None                # 始めたときの (us, mps, max_s)
        self.near_n = 0
        self.frames = 0

    @staticmethod
    def _clip(x, rng, default):
        """範囲に切る。数でない値(NaN・±inf・None・文字)は既定値(2026-10-26 fix2: min(max(nan, lo), hi) は nan を通した。
        drive.py の入口(_clean_params)も ±inf を「数値ではありません」として断る)"""
        try:
            v = float(x)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(v):
            return float(default)
        return min(max(v, rng[0]), rng[1])

    def _end(self, code, now=None):
        self.active = False
        self.last_end = int(code)
        self.event = ('end', {'code': int(code), 'reason': LAUNCH_END_NAMES.get(int(code), ''), 'frames': self.frames,
                              's': None if now is None or self.t0 is None else round(now - self.t0, 3)})
        return int(code)

    def human(self):
        """CH3 が人側(または dry run)のフレーム。戻り値: このフレームで終わった理由(⑦。出していなければ 0)"""
        code = self._end(7) if self.active else 0
        self._section()
        self.armed = True
        return code

    def hold(self):
        """中立で止めた(⑧)。戻り値: 終わった理由(出していなければ 0)。止めたあとの最初の AI 側では始めない"""
        self.held = True
        return self._end(8) if self.active else 0

    def stop(self):
        """走行の停止(⑧)"""
        return self._end(8) if self.active else 0

    def _why_not(self, thr, erpm, inp):
        if inp is None:
            return 'off'
        if not self.armed:
            return 'no_switch'
        if self.held:
            return 'hold'
        if not math.isfinite(thr):
            return 'model'
        if not abs(float(erpm)) < self.START_ERPM:
            return 'moving'
        if not inp.veh_ok:
            return 'veh'
        if not inp.tof_ok:
            return 'tof'
        xs = [float(x) for x in inp.ttc5[-self.TTC_N:]]
        if len(xs) < self.TTC_N or not all(math.isfinite(x) for x in xs):
            return 'ttc_n'
        if not float(np.median(xs)) >= self.TTC_START_S:
            return 'near'
        return ''

    def _end_reason(self, erpm, now, inp):
        """終わりの条件(安全の順に見る)。0 = まだ続ける"""
        v = float(inp.v)
        if not inp.veh_ok or not math.isfinite(v) or float(erpm) > self.END_BACK_ERPM:
            return 9
        moving = v >= self.END_TTC_MPS
        self.near_n = self.near_n + 1 if (moving and math.isfinite(inp.ttc) and inp.ttc < self.END_TTC_S) else 0
        if inp.guard_changed or not inp.tof_ok or not math.isfinite(inp.ttc) or self.near_n >= self.END_TTC_N:
            return 6
        if float(inp.impact_g) >= self.END_IMPACT_G:
            return 5
        if math.isfinite(inp.gz) and abs(float(inp.gz)) >= self.END_YAW_DPS:
            return 3
        if v >= self.END_STEER_MPS and math.isfinite(inp.steer_rel) and abs(float(inp.steer_rel)) >= self.END_STEER_US:
            return 4
        a = float(inp.a)
        a_c = min(max(a, 0.0), self.ACCEL_MAX) if math.isfinite(a) else 0.0
        if v + a_c * self.PRED_S >= self.cfg[1]:
            return 1
        if now - self.t0 >= self.cfg[2]:
            return 2
        return 0

    def step(self, thr, erpm, now, inp):
        """AI 側のフレーム。thr = モデルのスロットル(ゲイン・ガードのあと)、inp = LaunchIn か None(OFF)。
        戻り値 (出すスロットル [µs] か None(モデルのまま), このフレームで終わった理由 0〜10)"""
        if not self.decided:
            self.decided = True
            why = self._why_not(thr, erpm, inp)
            if not why:
                self.cfg = (self._clip(inp.us, self.US_RANGE, self.US_DEFAULT),
                            self._clip(inp.mps, self.MPS_RANGE, self.MPS_DEFAULT),
                            self._clip(inp.max_s, self.MAX_S_RANGE, self.MAX_S_DEFAULT))
                self.t0, self.near_n, self.frames = now, 0, 0
                code = self._end_reason(erpm, now, inp)
                if code:
                    why = 'end'
                    self.skipped = True
                    self.last_skip = why
                    self.event = ('skip', {'why': why, 'reason': LAUNCH_SKIP_NAMES[why], 'code': code,
                                           'code_reason': LAUNCH_END_NAMES[code]})
                    return None, 0
                self.active = self.used = True
                self.count += 1
                self.event = ('start', {'us': self.cfg[0], 'mps': self.cfg[1], 'max_s': self.cfg[2],
                                        'erpm': float(erpm), 'ttc_med': round(float(np.median(inp.ttc5[-self.TTC_N:])), 3)})
            else:
                if inp is not None:
                    self.skipped = True
                    self.last_skip = why
                    self.event = ('skip', {'why': why, 'reason': LAUNCH_SKIP_NAMES[why]})
                return None, 0
        if not self.active:
            return None, 0
        if inp is None:                 # 走行中に OFF にした
            return None, self._end(10, now)
        code = self._end_reason(erpm, now, inp)
        if code:
            return None, self._end(code, now)
        want = self.n - self.cfg[0]
        self.frames += 1
        return (thr if (math.isfinite(thr) and thr < want) else want), 0     # 前進だけ(thr が nan・inf なら want)


class Assist(object):
    """走行ループから使う窓口。人が操縦中・出力しない確認中は何もしない"""

    def __init__(self, ref):
        self.ref = (float(ref[0]), float(ref[1]))
        self.esc = EscReverse(self.ref[1])
        self.rec = Recovery(self.ref)
        self.kick = StartKick(self.ref[1])
        self.boost = LaunchBoost(self.ref[1])
        self.esc_count = 0
        self.kick_count = 0
        self._esc_active = False
        self._kick_active = False
        self.launch_on = False          # このフレームで発進の前進を出した(AI_CMD の launch_on)
        self.launch_end = 0             # このフレームで発進の前進が終わった理由(AI_CMD の launch_end)

    def reset(self, keep_recovery=False):
        """中立で止めたあとなど。発進の補助は戻さない(CH3 を人側へ戻したときだけ、次の AI 側で 1 回)。
        keep_recovery = True(drive の rec_hold_keep = 1 で、AI 側のままの中立止め。2026-10-26 fix3 safety1-1)は、
        リカバリーを戻さない(中止・途中の段・armed・衝撃・あきらめの数をそのまま)。後退の受け付けは戻す。
        既定 False は今までと同じ(リカバリーも全部戻す)"""
        self.esc.reset()
        if not keep_recovery:
            self.rec.reset()
        self._esc_active = False

    def launch_hold(self):
        """中立で止めたフレーム(drive.py の hold が記録の前に呼ぶ)。戻り値: 発進の前進が終わった理由(⑧。無ければ 0)"""
        self.launch_on = False
        self.launch_end = self.boost.hold()
        return self.launch_end

    def step(self, model_us, erpm, select, now, esc_on=True, rec_on=True, impact_g=0.0,
             rev_us=DEFAULT_REV_US, rev_s=DEFAULT_REV_S, kick_on=False, blocked=False, launch=None, rec_guard=False,
             launch_no_kick=False):
        """戻り値 (出す値 (steer, throttle), 補助の種類)。blocked = 前が近い(発進の補助を出さない)。
        launch = 発進の前進の入力 LaunchIn(launch_boost = 1 のとき。None = OFF で今までと同じ)。
        rec_guard = リカバリーの後ろの見張り(drive の rec_back_guard。False = 今までと同じ)。
        launch_no_kick = 発進の前進が「出さない」と決めた区間では StartKick も出さない(drive の launch_no_kick。
        False = 今までと同じ。launch が None(OFF)なら決めないので効かない)"""
        self.boost.event = None
        self.launch_on = False
        if not select:
            self.launch_end = self.boost.human()
            self.reset()
            self.kick.reset()
            self._kick_active = False
            return model_us, ASSIST_NONE
        lb, self.launch_end = self.boost.step(model_us[1], erpm, now, launch)
        if launch_no_kick and self.boost.event is not None and self.boost.event[0] == 'skip':
            self.boost.event[1]['no_kick'] = True       # 記録の出来事: この区間は StartKick も出さない
        if lb is not None:              # 発進の前進(StartKick の場所。リカバリー・後退の受け付けより先)
            self.launch_on = True
            self._kick_active = False
            if abs(erpm) >= self.rec.ARM_ERPM:      # 出している間に動いたことはリカバリーに残す(終わったあとの衝突に備える)
                self.rec.armed = True
            return (model_us[0], float(lb)), ASSIST_LAUNCH
        # 発進の前進を出した区間では、終わったあとも StartKick を出さない。launch_no_kick なら「出さない」と決めた区間でも出さない
        if kick_on and not self.boost.used and not (launch_no_kick and self.boost.skipped):
            k = self.kick.step(model_us[1], erpm, now, blocked)
            if k is not None:
                if not self._kick_active:
                    self.kick_count += 1
                self._kick_active = True
                return (model_us[0], float(k)), ASSIST_START
        self._kick_active = False
        if rec_on:
            o, k = self.rec.step(model_us, erpm, now, impact_g, rev_us, rev_s, back_guard=rec_guard)
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
