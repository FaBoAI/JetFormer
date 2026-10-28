"""fix8 係 safety-proposal: 11/3 のあとに向けた直し方の案を、jetformer/ の写し(proposal/<案>/assist.py)に入れる。
本物の jetformer/ には書かない。base = 今(fix7 r2 のあと)の写しそのまま。drive.py はどの案も写しのまま(変えない)。
  pA = s4-c(0.2 秒続けて後ろへ動いたら「本当に下がっている」)を、'gap' を飛ばさないふつうの 'reverse' のやり直しの決めにも使う
  pB = pA + 1 フレームだけの途切れは続きとみる(続きの数えを切らない・決めるのは 2 フレーム続けて動いていないとき)
  pC = ふつうの 'reverse' で、1 フレームだけの後ろへの動き(次のフレームで止まった)は見た後退にしない(approvals 14 の文の案)
どれも見張り ON(back_guard)の日だけ。モジュールの定数 P8_OFF_TOO を True にすると OFF にも使う(台本が走りの間だけ変える)。
    python -I make_copies.py   → 写しを作り直し、sha256 と変えた行の数を出す
"""
import difflib
import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path('/home/jetson/jetformer/jetformer')
BASE_SHA = {'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f',
            'drive.py': '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rep(s, old, new, n=1):
    assert s.count(old) == n, (old[:80], s.count(old))
    return s.replace(old, new)


CONST_OLD = """                             # (2026-10-27 fix7 r2 s4-c。止まっている車の eRPM の跳ねは本物の記録で続けて最長 0.11 秒)
"""
CONST_NEW = CONST_OLD + """P8_OFF_TOO = False           # fix8 案: 案の決めを見張り OFF の日にも使う(台本が走りの間だけ True。既定 False = OFF は今と同じ)
"""

# --- 'reverse' の続けての後退の数え(今 = s4-c。'gap' を飛ばした 'reverse' だけ)
S4C_OLD = """            if back_guard and self.skip_gap:
                # fix7 r2 s4-c: 'gap' を飛ばして入った 'reverse' は、続けて SURE_BACK_S 後ろへ動いたら「本当に下がっている」
                # ('reverse' に入ったフレームは除く。途切れたら・中立止めのあとは数え直す)
                if phase0 == 'reverse' and erpm > BACKING_ERPM:
                    if self.run_t0 is None:
                        self.run_t0 = now
                    elif now - self.run_t0 >= SURE_BACK_S:
                        self.back_sure = True
                else:
                    self.run_t0 = None
                # まだ本当に下がっているを見ていなくて、このフレームで後ろへ動いていなければ下がっていない(跳ね・たわみの戻りだけ。
                # 前は 'reverse' の中の跳ね 1 つで early_none もやり直しも止まった。このフレームで後ろへ動いていれば次のフレームで決める)
                none_yet = not self.back_sure and erpm <= BACKING_ERPM
            else:"""

PA_S4C = """            if back_guard or P8_OFF_TOO:
                # fix8 案 pA: s4-c の「本当に下がっている」(続けて SURE_BACK_S 後ろへ動いた)を、'gap' を飛ばさないふつうの
                # 'reverse' のやり直しの決めにも使う(前はふつうの 'reverse' は 1 フレームの後ろへの動きでやり直しが止まった:
                # 'gap' 0.15 秒では下がれない ESC の、最初の 'reverse' の中の跳ね)
                if phase0 == 'reverse' and erpm > BACKING_ERPM:
                    if self.run_t0 is None:
                        self.run_t0 = now
                    elif now - self.run_t0 >= SURE_BACK_S:
                        self.back_sure = True
                else:
                    self.run_t0 = None
                none_yet = not self.back_sure and erpm <= BACKING_ERPM
            else:"""

PB_S4C = """            if back_guard or P8_OFF_TOO:
                # fix8 案 pB: pA(ふつうの 'reverse' にも s4-c)+ 1 フレームだけの途切れは続きとみる。続けての数えを切るのは
                # 2 フレーム続けて後ろへ動いていないとき、下がっていないと決めるのも 2 フレーム続けて動いていないとき
                # (下がり始めた車の eRPM が 1 フレームだけ 300 以下になっても、やり直して下がった時間を捨てない)
                if phase0 == 'reverse' and erpm > BACKING_ERPM:
                    self.p8_miss = 0
                    if self.run_t0 is None:
                        self.run_t0 = now
                    elif now - self.run_t0 >= SURE_BACK_S:
                        self.back_sure = True
                else:
                    self.p8_miss = 2 if phase0 != 'reverse' else self.p8_miss + 1
                    if self.p8_miss >= 2:
                        self.run_t0 = None
                none_yet = not self.back_sure and self.p8_miss >= 2
            else:"""

PC_S4C = S4C_OLD + """
                none_yet = None
            if none_yet is None and (back_guard or P8_OFF_TOO):
                # fix8 案 pC: ふつうの 'reverse' で、後ろへ動いたのが 1 フレームだけで次のフレームで止まったら、見た後退にしない
                # (approvals 14 の文の案)。このフレームで後ろへ動いていれば次のフレームで決める。2 フレーム以上続いたら今まで通り
                if phase0 == 'reverse':
                    if erpm > BACKING_ERPM:
                        self.p8_pend = 0.0 if self.p8_prev else dt
                        self.p8_prev = True
                    else:
                        if self.p8_prev and self.p8_pend > 0.0:
                            self.p8_iso += self.p8_pend
                        self.p8_pend, self.p8_prev = 0.0, False
                else:
                    self.p8_pend, self.p8_prev = 0.0, False
                none_yet = self.seen_rev_s - self.p8_iso <= 1e-9 and erpm <= BACKING_ERPM
            elif none_yet is None:"""

RETRY_OLD = """                if self.retries < self.MAX_RETRIES:
                    self.retries += 1
                    self.backed_s = self.est_s = 0.0       # 後退側を出しても下がらない: 見積もりの後退も無かった(fix6 r2)
                    self.seen_rev_s = 0.0
                    self.skip_gap = False
"""
RETRY_NEW = RETRY_OLD + """                    self.run_t0, self.back_sure = None, False      # fix8 案: やり直しの 'reverse' は数え直す
"""

RESET_OLD = """        self.back_sure = False       # その 'reverse' で SURE_BACK_S 続けて後ろへ動いた(本当に下がっている。fix7 r2 s4-c)
"""
RESET_B = RESET_OLD + """        self.p8_miss = 2             # fix8 案 pB: 続けて後ろへ動いていないフレームの数(2 以上 = 続きが切れた)
"""
RESET_C = RESET_OLD + """        self.p8_prev = False         # fix8 案 pC: 前のフレーム('reverse' の中)で後ろへ動いていた
        self.p8_pend = 0.0           # その 1 フレームだけの後ろへの動きの秒(次のフレームで止まったら p8_iso へ)
        self.p8_iso = 0.0            # 1 フレームだけで止まった後ろへの動きの秒(見た後退から除く)
"""
START_OLD = """        self.run_t0, self.back_sure = None, False
        self.back_t = None
        self._go('brake', now)"""
START_B = """        self.run_t0, self.back_sure = None, False
        self.p8_miss = 2
        self.back_t = None
        self._go('brake', now)"""
START_C = """        self.run_t0, self.back_sure = None, False
        self.p8_prev, self.p8_pend, self.p8_iso = False, 0.0, 0.0
        self.back_t = None
        self._go('brake', now)"""
EARLY_OLD = """                    self.run_t0, self.back_sure = None, False
                    self.backed_s = self.est_s = self.seen_rev_s = 0.0
                    self._go('gap', now)"""
EARLY_B = """                    self.run_t0, self.back_sure = None, False
                    self.p8_miss = 2
                    self.backed_s = self.est_s = self.seen_rev_s = 0.0
                    self._go('gap', now)"""
HOLD_OLD = """            self.run_t0 = None          # fix7 r2 s4-c: 続けて後ろへ動いていたかは、中立止めのあとのフレームから数え直す
"""
HOLD_B = HOLD_OLD + """            self.p8_miss = 2            # fix8 案 pB: 中立止めの前後はつなげない
"""
HOLD_C = HOLD_OLD + """            self.p8_pend, self.p8_prev = 0.0, False     # fix8 案 pC: 中立止めの前後はつなげない(前の 1 フレームは見たまま)
"""
RETRY_C = RETRY_NEW + """                    self.p8_prev, self.p8_pend, self.p8_iso = False, 0.0, 0.0
"""
RETRY_B = RETRY_NEW + """                    self.p8_miss = 2
"""


def build():
    src = (SRC / 'assist.py').read_text(encoding='utf-8')
    for fn, h in BASE_SHA.items():
        assert sha(SRC / fn) == h or fn == 'drive.py', ('jetformer/%s が fix7 r2 のあとと違う' % fn)
    out = {}
    s = rep(src, CONST_OLD, CONST_NEW)
    out['pA'] = rep(rep(s, S4C_OLD, PA_S4C), RETRY_OLD, RETRY_NEW)
    sb = rep(rep(s, S4C_OLD, PB_S4C), RETRY_OLD, RETRY_B)
    sb = rep(rep(rep(rep(sb, RESET_OLD, RESET_B), START_OLD, START_B), EARLY_OLD, EARLY_B), HOLD_OLD, HOLD_B)
    out['pB'] = sb
    sc = rep(rep(s, S4C_OLD, PC_S4C), RETRY_OLD, RETRY_C)
    sc = rep(rep(rep(sc, RESET_OLD, RESET_C), START_OLD, START_C), HOLD_OLD, HOLD_C)
    out['pC'] = sc
    for name, text in out.items():
        (HERE / name / 'assist.py').write_text(text, encoding='utf-8')
        d = list(difflib.unified_diff(src.splitlines(), text.splitlines(), lineterm='', n=0))
        print(name, sha(HERE / name / 'assist.py'), '+%d -%d' % (sum(1 for x in d if x.startswith('+') and not x.startswith('+++')),
                                                                  sum(1 for x in d if x.startswith('-') and not x.startswith('---'))))
    # drive.py はどの案も写しのまま
    for name in ('base', 'pA', 'pB', 'pC'):
        assert sha(HERE / name / 'drive.py') == BASE_SHA['drive.py'], name
    assert sha(HERE / 'base' / 'assist.py') == BASE_SHA['assist.py']


if __name__ == '__main__':
    build()
