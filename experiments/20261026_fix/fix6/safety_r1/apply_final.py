"""fix6 係 safety 直す 6-1 回目の直し(指摘 w1-safety-r61-safety-1 と、広げて見つけた同じ形)を、直す前の写し
tests/fixtures/fix6_safety_before/{assist.py, drive.py} に当てて final/ に書く(読み書きするのはこの 2 本の写しと final/ だけ)。
本物の jetformer/ へは、試験を流してから手で写す(cp)。

    python3 -I apply_final.py
"""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path('/home/jetson/jetformer/tests/fixtures/fix6_safety_before')
OUT = HERE / 'final'
SHA = {'assist.py': '80f20890804170bede9c7df6e7f05892ab4e5397f8c5f837fda49ddba64d1276',
       'drive.py': '041a66de7590d9915f3a3008b03aa08cec01b81fbc2a292f90f65875ee0be39c'}


def patch(name, reps):
    src = SRC / name
    assert hashlib.sha256(src.read_bytes()).hexdigest() == SHA[name], name
    s = src.read_text(encoding='utf-8')
    for old, new in reps:
        assert s.count(old) == 1, (name, s.count(old), old[:100])
        s = s.replace(old, new)
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(s, encoding='utf-8')


A_DOC_OLD = """     'brake'・'wait'・'gap' から 'reverse'・'coast' へ進めるのも、後ろへ動いていたサンプルがあったときだけ。前は 0.25〜0.45 秒
     値が来ないだけで 1 cm も下がっていない車が前の壁へ escape を約 0.5 秒出した
"""
A_DOC_NEW = """     'brake'・'wait'・'gap' から 'reverse'・'coast' へ進めるのも、後ろへ動いていたサンプルがあったときだけ。前は 0.25〜0.45 秒
     値が来ないだけで 1 cm も下がっていない車が前の壁へ escape を約 0.5 秒出した
   反証 6-1 回目の直し(2026-10-27 fix6 係 safety。どれも rec_hold_keep = 1・見張り ON の日だけ効く。OFF はビット単位で同じ):
   ・s3-a 車両の値が来なかった間に、値が古いまま(max_sensor_age 0.15 秒まで)出した後退側で車が下がり始めることがある
     (ESC の後退の入りぎわ・ESC が後退を待っていた 'brake')。drive はその間を、PCA9685 に書けた値から見積もった「ESC が後退で
     走らせていたかもしれない秒」(EscTrack)として hold_rev_s で渡す(hold_unseen_s は渡さない = 0)。_after_hold はそれを
     下がった時間に足す(どの段でも。ただし後退を HOLD_EST_MIN_LEFT_S = 0.2 秒(rev_s の下限)残すところまで)。足したら
     'brake'・'wait' → 'reverse'・'coast'、'gap' → 'coast' へ進める(後ろへ動いていたサンプルと同じ)。前は、明けたサンプルが
     300 eRPM 未満だと 1 秒も数えず、明けてから rev_s をまるごと下がり直した(値の止まりなしより最大 0.33 m 多く下がり、後ろ
     1.35〜1.6 m の物に当たった)。逆に、明けたサンプルが後ろへ動いていると値が来なかった間を全部(古い値で数えたフレームと
     二重に)数えて後退を早く切り、0.2〜0.4 m しか下がらずに前の壁へ escape を出した(fix3 r2 の数え方。どちらも同じ窓)
   ・s3-b 残した中立止めの間の出力は中立 = 惰性。'coast' の途中の中立止めは惰性の時間に数え(段の時間をずらさない)、
     中立止めのあとで後退を出し終えていたら('reverse' で下がった時間が rev_s 以上)、'coast' は中立止めの始めからとする。
     前は明けてから 'coast' の 0.25 秒をやり直し、keep 0 より最大 0.37 m 多く下がった
"""

A_CONST_OLD = """CHECK_S = 0.35               # 後退を出してから、この時間で下がり始めなければやり直す(人は0.08秒で動く)
"""
A_CONST_NEW = """CHECK_S = 0.35               # 後退を出してから、この時間で下がり始めなければやり直す(人は0.08秒で動く)
HOLD_EST_MIN_LEFT_S = 0.2    # 見ていない間の後退の見積もり(hold_rev_s)で後退を短くするのは、これだけ残すところまで(rev_s の下限。
                             # 2026-10-27 fix6 safety r1。見積もりの後退は中立止めで切られて遅く、そのまま差し引くと短い rev_s で
                             # 下がる距離が足りず escape が前の壁に当たる)
"""

A_TRACK_OLD = """        return want, want != thr


class StartKick(object):"""
A_TRACK_NEW = """        return want, want != thr


class EscTrack(object):
    \"\"\"PCA9685 に書けたスロットルから、ESC が後退で走らせていたかもしれない区間を見積もる(2026-10-27 fix6 safety r1。
    drive が rec_hold_keep = 1・rec_back_guard = 1 の日だけ使う。残した中立止めのあと、車両の値が来なかった間の数えにだけ)。
    決まりはこのESCの後退(モジュールの説明の始め。10/10 のログ 116 回分)のとおり:
    ・前進(中立 − FWD_US より前)を書いたら「下がれる」が消える(前進の直後の後退側はブレーキ)
    ・ブレーキ(下がれないときの後退側)のあと、中立の帯が READY_NEUTRAL_S 続いたら下がれる(人は約 0.12 秒・閉じたループの
      偽物の ESC は 0.1 秒を 10 ms 刻みで数える。多めに数える側で 0.08 秒)
    ・下がれるときに後退側(中立 + REV_US より後ろ)を書き始めたら、DELAY_S あとから後退側が終わるまで後退で走らせている
      (下がり始めは後退側から約 0.08 秒あと。10/10 のログは 100 Hz・偽物の ESC は 10 ms 刻みで 0.07 秒から走らせることが
      あるので、多めに数える側で 0.06 秒)。後退で走らせたあとの中立のあとも下がれる
    ・分からないとき(走行の始め・人側の間)は下がれるとする。前へ転がっている間は後退しない決まりは見ない
    どれも多めに数える側(見ていない間に下がったとして、そのあとの後退を短くする側)。本物の ESC は 89% がこの形\"\"\"
    FWD_US = 20.0            # 中立からこれより前 = 前進(中立の帯 約 1478〜1538 µs の前の端)
    REV_US = EscReverse.DEADBAND_US      # これより後ろ = 後退側(ブレーキか後退)
    READY_NEUTRAL_S = 0.08
    DELAY_S = 0.06
    MAX_RUNS = 8

    def __init__(self, neutral):
        self.n = float(neutral)
        self.runs = collections.deque(maxlen=self.MAX_RUNS)     # [後退で走らせ始めた時刻 ns, 終わった時刻 ns(None = 続いている)]
        self.side = None             # 最後に書けた値の側('f' 前進 / 'n' 中立の帯 / 'r' 後退側 / None 分からない)
        self.ready = True
        self.braked = False
        self.neu_t0 = None

    def unknown(self, t_ns):
        \"\"\"人側の間など、ESC に届いた値が分からない(下がれるとする)\"\"\"
        self._end(t_ns)
        self.side, self.ready, self.braked, self.neu_t0 = None, True, False, None

    def _end(self, t_ns):
        if self.runs and self.runs[-1][1] is None:
            self.runs[-1][1] = max(int(t_ns), self.runs[-1][0])

    def write(self, t_ns, thr):
        \"\"\"t_ns に thr [µs] を書けた(次に書くまで続く)\"\"\"
        t_ns, thr = int(t_ns), float(thr)
        side = 'f' if thr < self.n - self.FWD_US else ('r' if thr > self.n + self.REV_US else 'n')
        if (self.side == 'n' and self.braked and self.neu_t0 is not None
                and t_ns - self.neu_t0 >= self.READY_NEUTRAL_S * 1e9):
            self.ready, self.braked = True, False                # ブレーキのあとの中立が続いた: 下がれる
        if side == 'f':
            self._end(t_ns)
            self.ready = self.braked = False
            self.neu_t0 = None
        elif side == 'r':
            self.neu_t0 = None
            if self.side != 'r':                                 # 後退側の始め
                if self.ready:
                    self.runs.append([t_ns + int(self.DELAY_S * 1e9), None])
                else:
                    self.braked = True
        else:
            self._end(t_ns)
            if self.side != 'n':
                self.neu_t0 = t_ns
        self.side = side

    def rev_s(self, a_ns, b_ns):
        \"\"\"[a_ns, b_ns] のうち、後退で走らせていたかもしれない秒\"\"\"
        s = 0
        for t0, t1 in self.runs:
            lo, hi = max(t0, int(a_ns)), min(int(b_ns) if t1 is None else t1, int(b_ns))
            if hi > lo:
                s += hi - lo
        return s / 1e9


class StartKick(object):"""

A_AH_SIG_OLD = """    def _after_hold(self, gap, hold_s, hold_g, hold_back_s, rev_s, now, erpm=0.0, hold_unseen_s=0.0):"""
A_AH_SIG_NEW = """    def _after_hold(self, gap, hold_s, hold_g, hold_back_s, rev_s, now, erpm=0.0, hold_unseen_s=0.0, hold_rev_s=0.0):"""
A_AH_DOC_OLD = """        (2026-10-27 fix5 s3。既定 0 = 全部を後ろへ動いていたサンプルの秒として今までと同じ)。
"""
A_AH_DOC_NEW = """        (2026-10-27 fix5 s3。既定 0 = 全部を後ろへ動いていたサンプルの秒として今までと同じ)。hold_rev_s = 車両の値が来なかった間に
        ESC が後退で走らせていたかもしれない秒(drive が書いた値から見積もる。hold_back_s とは別。2026-10-27 fix6 s3-a。既定 0)。
"""
A_AH_COUNT_OLD = """        if p in self.HOLD_HIT_PHASES:
            count_unseen = p in ('reverse', 'coast') and (self.backed_s > 0.0 or seen > 0.0)     # 下がり始めたあと
            self.backed_s += hb if count_unseen else seen       # hold_unseen_s = 0 なら どちらも hold_back_s(今までと同じ)
"""
A_AH_COUNT_NEW = """        est = max(float(hold_rev_s), 0.0)
        if p in self.HOLD_HIT_PHASES:
            count_unseen = p in ('reverse', 'coast') and (self.backed_s > 0.0 or seen > 0.0)     # 下がり始めたあと
            self.backed_s += hb if count_unseen else seen       # hold_unseen_s = 0 なら どちらも hold_back_s(今までと同じ)
            if est > 0.0:       # fix6 s3-a: 見ていない間の後退の見積もり。後退を HOLD_EST_MIN_LEFT_S 残すところまで足す
                self.backed_s += min(est, max(0.0, rev_s - HOLD_EST_MIN_LEFT_S - self.backed_s))
"""
A_AH_TAIL_OLD = """        self.t_phase += hold_s                          # 段の時間は中立を出していた間は進めない
        if p == 'reverse':
            self.t_check = now                          # 下がり始めを待つ時間は明けたフレームから(中立で後退が切れた)
        if seen > 0.0:                                  # 実際に後ろへ動いていたサンプルがあった(fix5 s3。前は hold_back_s > 0)
            if p in ('brake', 'wait'):                  # 見張りと同じ: もう下がっている
                self.guard_event = ('back_hold', p, round(self.backed_s, 3))
                self._go('coast' if self.backed_s >= rev_s else 'reverse', now)
            elif p == 'gap' and self.backed_s >= rev_s:
                self.guard_event = ('back_hold', p, round(self.backed_s, 3))
                self._go('coast', now)
        return None
"""
A_AH_TAIL_NEW = """        if p != 'coast':                                # fix6 s3-b: 'coast' は中立止めの間も惰性(段の時間をずらさない)
            self.t_phase += hold_s                      # 段の時間は中立を出していた間は進めない
        if p == 'reverse':
            self.t_check = now                          # 下がり始めを待つ時間は明けたフレームから(中立で後退が切れた)
            if self.backed_s >= rev_s:                  # fix6 s3-b: 後退は中立止めの始めで出し終えていた。惰性はそこから
                self._go('coast', now)
                self.t_phase = now - hold_s
        if seen > 0.0 or est > 0.0:                     # 後ろへ動いていたサンプルか、見ていない間の後退の見積もりがあった
            if p in ('brake', 'wait'):                  # 見張りと同じ: もう下がっている
                self.guard_event = ('back_hold', p, round(self.backed_s, 3))
                self._go('coast' if self.backed_s >= rev_s else 'reverse', now)
                if self.phase == 'coast':
                    self.t_phase = now - hold_s         # fix6 s3-b: 惰性は中立止めの始めから
            elif p == 'gap' and self.backed_s >= rev_s:
                self.guard_event = ('back_hold', p, round(self.backed_s, 3))
                self._go('coast', now)
                self.t_phase = now - hold_s             # fix6 s3-b
        return None
"""
A_STEP_SIG_OLD = """             hold_g=0.0, hold_since=None, hold_back_s=0.0, hold_unseen_s=0.0):
        \"\"\"戻り値 (出す値 (steer, throttle) または None(モデルのまま), 補助の種類)。"""
A_STEP_SIG_NEW = """             hold_g=0.0, hold_since=None, hold_back_s=0.0, hold_unseen_s=0.0, hold_rev_s=0.0):
        \"\"\"戻り値 (出す値 (steer, throttle) または None(モデルのまま), 補助の種類)。"""
A_STEP_DOC_OLD = """        車両の値が来なかっただけの間の秒(2026-10-27 fix5 s3。既定 0 = 今までと同じ。_after_hold)\"\"\""""
A_STEP_DOC_NEW = """        車両の値が来なかっただけの間の秒(2026-10-27 fix5 s3。既定 0 = 今までと同じ。_after_hold)。hold_rev_s = 車両の値が来なかった
        間に ESC が後退で走らせていたかもしれない秒(drive の見積もり。2026-10-27 fix6 s3-a。既定 0 = 今までと同じ。_after_hold)\"\"\""""
A_STEP_CALL_OLD = """            r = self._after_hold(gap, hold_s, hold_g, hold_back_s, rev_s, now, erpm, hold_unseen_s)"""
A_STEP_CALL_NEW = """            r = self._after_hold(gap, hold_s, hold_g, hold_back_s, rev_s, now, erpm, hold_unseen_s, hold_rev_s)"""
A_AS_SIG_OLD = """             launch_no_kick=False, hold_g=0.0, hold_since=None, hold_back_s=0.0, hold_unseen_s=0.0, rec_keep=False):"""
A_AS_SIG_NEW = """             launch_no_kick=False, hold_g=0.0, hold_since=None, hold_back_s=0.0, hold_unseen_s=0.0, rec_keep=False,
             hold_rev_s=0.0):"""
A_AS_DOC_OLD = """        2026-10-27 fix3 r2-r2-rs-1・rs-2)。hold_unseen_s = Recovery.step の同じ名前(2026-10-27 fix5 s3。既定 0 = 今までと同じ)。
"""
A_AS_DOC_NEW = """        2026-10-27 fix3 r2-r2-rs-1・rs-2)。hold_unseen_s = Recovery.step の同じ名前(2026-10-27 fix5 s3。既定 0 = 今までと同じ)。
        hold_rev_s = Recovery.step の同じ名前(2026-10-27 fix6 s3-a。既定 0 = 今までと同じ)。
"""
A_AS_CALL_OLD = """                                 hold_since=hold_since, hold_back_s=hold_back_s, hold_unseen_s=hold_unseen_s)"""
A_AS_CALL_NEW = """                                 hold_since=hold_since, hold_back_s=hold_back_s, hold_unseen_s=hold_unseen_s,
                                 hold_rev_s=hold_rev_s)"""

D_DOC_OLD = """フレームの時刻を進める(s2)。衝撃だけの中止を走っている車にしないのは assist.py の中(s1)。
"""
D_DOC_NEW = """フレームの時刻を進める(s2)。衝撃だけの中止を走っている車にしないのは assist.py の中(s1)。
反証 6-1 回目の直し(2026-10-27 fix6 係 safety。同じく両方 1 の日だけ効く。どちらかが 0 ならビット単位で同じ): 両方 1 の日は
PCA9685 に書けたスロットル(人側は「分からない」)を assist.EscTrack に入れ、残した中立止めの間に車両の値が HOLD_VEH_GAP_S より
来なかった間は、前に見た値が HOLD_VEH_FAST_ERPM より速く後ろへ動いていたなら全部、そうでなければ EscTrack が見積もった
「ESC が後退で走らせていたかもしれない秒」を数える(前に見た値が後ろへ動いていたなら、その値で Recovery が数えたフレームの
分は除く)。見積もりは hold_back_s とは別の hold_rev_s で渡し(hold_unseen_s は渡さない)、assist.Recovery が後退を 0.2 秒残す
ところまで下がった時間に足す(s3-a)。前は、値が古いまま出した後退側で車が下がり始めても数えず、明けてから rev_s をまるごと
下がり直す(後ろへ多く)か、値が来なかった間を全部二重に数えて後退を早く切る(前の壁へ escape)かが、明けたサンプルの速さで
分かれた。'coast' の続きの直し(s3-b)は assist.py の中。
"""
D_CONST_OLD = """HOLD_VEH_GAP_S = 0.06 """
D_CONST_NEW = """HOLD_VEH_FAST_ERPM = 1000.0       # 残した中立止めの間に車両の値が来なかったとき、前に見た値がこれより速く後ろへ動いていたら、その間は全部
                                  # 下がっていたとして数える('coast' が続く速さ。中立の惰性ではすぐ止まらない。2026-10-27 fix6 safety r1)
HOLD_VEH_GAP_S = 0.06 """
D_STATE_OLD = """        # [3] = 数えた秒のうち、車両の値が来なかっただけの間(前後のサンプルはどちらも後ろへ動いていない)の秒(2026-10-27 fix5 s3。
        # Recovery が「下がり始めたあと」だけ下がった時間に数える。出力に効くのは残した中立止めのあとだけ)
        hold_veh = [None, 0.0, 0.0, 0.0]
        proc_cam = [None]
        hold_t0 = [None]            # 残した中立止めが中立を書き始めた時刻(time.monotonic()。helper.step の now と同じ時計)
"""
D_STATE_NEW = """        # [3] = その中立止めの前に処理したフレームの撮像時刻 ns・[4] = 車両の値が来なかった間に ESC が後退で走らせていたかもしれない
        # 秒(esc_track の見積もり。Recovery へ hold_rev_s で渡す)(2026-10-27 fix6 safety r1。fix5 の [3] = 値が来なかっただけの秒は
        # やめた)。出力に効くのは残した中立止めのあとだけ
        hold_veh = [None, 0.0, 0.0, None, 0.0]
        proc_cam = [None]
        hold_t0 = [None]            # 残した中立止めが中立を書き始めた時刻(time.monotonic()。helper.step の now と同じ時計)
        # PCA9685 に書けたスロットルから、ESC が後退で走らせていたかもしれない区間(2026-10-27 fix6 safety r1。assist.EscTrack)。
        # rec_hold_keep = 1・rec_back_guard = 1 の日だけ値を入れ、残した中立止めのあと車両の値が来なかった間の数えにだけ使う
        esc_track = assist_mod.EscTrack(ref[1])

        def keep_on():
            return int(p.get('rec_hold_keep', 0) or 0) == 1 and int(p.get('rec_back_guard', 0) or 0) == 1
"""
D_UNSEEN_OLD = """            BACKING_ERPM を超えていたら、その間を全部数える。間が HOLD_VEH_GAP_S より空いた(車両の値が来なかった)ら、
            動いていなくても数える(見ていない間に下がったかもしれない = 多めに数える側)。そのうち、どちらのサンプルも後ろへ
            動いていない間(値が来なかっただけ)の秒は hold_veh[3] にも足す(2026-10-27 fix5 s3。戻り値は前と同じ合計)\"\"\"
            s = 0.0
            for t_ns, x in list(veh_buf):
                if upto_ns is not None and t_ns > upto_ns:
                    break
                if hold_veh[0] is not None and t_ns <= hold_veh[0]:
                    continue
                e = float(x['erpm'])
                if hold_veh[0] is not None and (e > assist_mod.BACKING_ERPM or hold_veh[1] > assist_mod.BACKING_ERPM
                                                or t_ns - hold_veh[0] > HOLD_VEH_GAP_S * 1e9):
                    s += (t_ns - hold_veh[0]) / 1e9
                    if not (e > assist_mod.BACKING_ERPM or hold_veh[1] > assist_mod.BACKING_ERPM):
                        hold_veh[3] += (t_ns - hold_veh[0]) / 1e9
                hold_veh[0], hold_veh[1] = t_ns, e
            return s
"""
D_UNSEEN_NEW = """            BACKING_ERPM を超えていたら、その間を全部数える。間が HOLD_VEH_GAP_S より空いた(車両の値が来なかった)ら
            (2026-10-27 fix6 safety r1): 前に見た値が HOLD_VEH_FAST_ERPM より速く後ろへ動いていたなら全部数える(中立の惰性でも
            下がっていた)。そうでなければ、書いた値から見積もった「ESC が後退で走らせていたかもしれない秒」(esc_track)を
            hold_veh[4] に足す(戻り値には入れない)。どちらも、前に見た値が後ろへ動いていた(> BACKING_ERPM)なら、そのあと
            処理したフレームは Recovery がその値で dt を数えたので、前に処理したフレームの撮像時刻(hold_veh[3])から\"\"\"
            s = 0.0
            for t_ns, x in list(veh_buf):
                if upto_ns is not None and t_ns > upto_ns:
                    break
                if hold_veh[0] is not None and t_ns <= hold_veh[0]:
                    continue
                e = float(x['erpm'])
                if hold_veh[0] is not None:
                    if t_ns - hold_veh[0] > HOLD_VEH_GAP_S * 1e9:
                        a = hold_veh[0]
                        if hold_veh[1] > assist_mod.BACKING_ERPM and hold_veh[3] is not None:
                            a = max(a, hold_veh[3])
                        if hold_veh[1] > HOLD_VEH_FAST_ERPM:
                            s += max(t_ns - a, 0) / 1e9
                        else:
                            hold_veh[4] += esc_track.rev_s(a, t_ns)
                    elif e > assist_mod.BACKING_ERPM or hold_veh[1] > assist_mod.BACKING_ERPM:
                        s += (t_ns - hold_veh[0]) / 1e9
                hold_veh[0], hold_veh[1] = t_ns, e
            return s
"""
D_SEEN_OLD = """            hold_veh[0], hold_veh[1], hold_veh[2], hold_veh[3] = None, 0.0, 0.0, 0.0
            if cam_ns is None:
"""
D_SEEN_NEW = """            hold_veh[0], hold_veh[1], hold_veh[2], hold_veh[3], hold_veh[4] = None, 0.0, 0.0, cam_ns, 0.0
            if cam_ns is None:
"""
D_HOLDW_OLD = """            launch_note(cam.t_ns if cam is not None else None)
            out.neutral()
            stats['holds'] += 1
"""
D_HOLDW_NEW = """            launch_note(cam.t_ns if cam is not None else None)
            n_ok = out.neutral()
            if keep_on():                   # ESC に届いた値(人側・dry run は分からない。fix6 safety r1)
                if not veh_select or dry_run:
                    esc_track.unknown(clock.now_ns())
                elif n_ok:
                    esc_track.write(clock.now_ns(), out.neutral_values()[1])
            stats['holds'] += 1
"""
D_HOLDR_OLD = """                helper.reset()              # 中立で止めた後は、補助の手順を最初から
                hold_imu[1], hold_imu[2] = 0.0, False
                hold_veh[2] = hold_veh[3] = 0.0
"""
D_HOLDR_NEW = """                helper.reset()              # 中立で止めた後は、補助の手順を最初から
                hold_imu[1], hold_imu[2] = 0.0, False
                hold_veh[2] = hold_veh[4] = 0.0
"""
D_HELD_OLD = """            hold_back_s = (hold_veh[2] + veh_unseen_back(cam.t_ns)) if held else 0.0
            hold_unseen_s = hold_veh[3] if held else 0.0       # そのうち値が来なかっただけの秒(fix5 s3)
            hold_imu[1], hold_imu[2] = 0.0, False
            hold_veh[2] = hold_veh[3] = 0.0
"""
D_HELD_NEW = """            hold_back_s = (hold_veh[2] + veh_unseen_back(cam.t_ns)) if held else 0.0
            hold_rev_s = hold_veh[4] if held else 0.0          # 値が来なかった間の後退の見積もり(fix6 safety r1)
            hold_imu[1], hold_imu[2] = 0.0, False
            hold_veh[2] = hold_veh[4] = 0.0
"""
D_CALL_OLD = """                                       hold_since=hold_since, hold_back_s=hold_back_s, hold_unseen_s=hold_unseen_s,
"""
D_CALL_NEW = """                                       hold_since=hold_since, hold_back_s=hold_back_s, hold_rev_s=hold_rev_s,
"""
D_WRITE_OLD = """            written = out.neutral() if dry_run else out.write(*out_us)
            stats['i2c_errors'] = out.errors
"""
D_WRITE_NEW = """            written = out.neutral() if dry_run else out.write(*out_us)
            if written and keep_on():       # ESC に届いた値(人側・dry run は分からない。fix6 safety r1)
                if ai_frame:
                    esc_track.write(clock.now_ns(), out_us[1])
                else:
                    esc_track.unknown(clock.now_ns())
            stats['i2c_errors'] = out.errors
"""

patch('assist.py', [(A_DOC_OLD, A_DOC_NEW), (A_CONST_OLD, A_CONST_NEW), (A_TRACK_OLD, A_TRACK_NEW),
                    (A_AH_SIG_OLD, A_AH_SIG_NEW), (A_AH_DOC_OLD, A_AH_DOC_NEW), (A_AH_COUNT_OLD, A_AH_COUNT_NEW),
                    (A_AH_TAIL_OLD, A_AH_TAIL_NEW), (A_STEP_SIG_OLD, A_STEP_SIG_NEW), (A_STEP_DOC_OLD, A_STEP_DOC_NEW),
                    (A_STEP_CALL_OLD, A_STEP_CALL_NEW), (A_AS_SIG_OLD, A_AS_SIG_NEW), (A_AS_DOC_OLD, A_AS_DOC_NEW),
                    (A_AS_CALL_OLD, A_AS_CALL_NEW)])
patch('drive.py', [(D_DOC_OLD, D_DOC_NEW), (D_CONST_OLD, D_CONST_NEW), (D_STATE_OLD, D_STATE_NEW),
                   (D_UNSEEN_OLD, D_UNSEEN_NEW), (D_SEEN_OLD, D_SEEN_NEW), (D_HOLDW_OLD, D_HOLDW_NEW),
                   (D_HOLDR_OLD, D_HOLDR_NEW), (D_HELD_OLD, D_HELD_NEW), (D_CALL_OLD, D_CALL_NEW),
                   (D_WRITE_OLD, D_WRITE_NEW)])
for n in ('assist.py', 'drive.py'):
    print(n, hashlib.sha256((OUT / n).read_bytes()).hexdigest())
