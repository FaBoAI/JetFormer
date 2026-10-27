"""fix6 係 safety 直す 6-2 回目: 直す前(before/assist.py.orig = fix6 r1 のあと)から直しを入れた final/assist.py を作る。
drive.py は変えない(final/drive.py は写し)。
    python -I apply_final2.py
"""
import hashlib
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'before' / 'assist.py.orig'
DST = HERE / 'final' / 'assist.py'

DOC_OLD = """     前は明けてから 'coast' の 0.25 秒をやり直し、keep 0 より最大 0.37 m 多く下がった

2. 後退の受け付け(EscReverse)"""
DOC_NEW = """     前は明けてから 'coast' の 0.25 秒をやり直し、keep 0 より最大 0.37 m 多く下がった
   反証 6-2 回目の直し(2026-10-27 fix6 係 safety r2。どれも rec_hold_keep = 1・見張り ON の日だけ効く。OFF はビット単位で同じ):
   ・s3-c 見積もり(hold_rev_s)は書けた値から見た「下がれたかもしれない」で、下がったかは分からない(EscTrack は多めに数える側。
     このESCが下がれるまでの中立の秒も分からない)。そこで見積もりは下がった時間(backed_s)に今まで通り足すが(後退を 0.2 秒残す
     ところまで)、足した分を est_s に別に持ち、
     - 見積もりだけでは段を進めない('brake'・'wait' → 'reverse'・'coast'、'gap' → 'coast' は後ろへ動いていたサンプルがあったとき
       だけ。見積もりだけなら 'gap' の中立も残す)
     - 'reverse' の下がり始めを待つやり直し(CHECK_S)は、見た後退(backed_s − est_s。BACKING_ERPM を超えたサンプルで数えた秒)が
       0 かで決める。やり直すときは見積もりも捨てる(後退側を CHECK_S 出しても下がらない = 見ていない間も下がっていなかった)
     見積もりが外れていた(ESC はまだ下がれなかった)ときは、そのぶん後退が短くなる(後退を 0.2 秒残すところまで。閉じたループの
     偽物の ESC で確かめた場面では、それで escape が前の壁を押すことは無かった)。見積もりを捨てる決まり(明けたあと後ろへ動くのを
     見なければ捨てる)は入れない: 見ていない間に本当に下がって止まった車と見分けられず、捨てると fix6 r1 の指摘の形(値の止まり
     なしより 0.3〜0.45 m 多く下がる)に戻るため
     前は、ブレーキのあとの 0.10 秒の中立止め(閉じたループの偽物の ESC はまだ下がれない)を EscTrack が中立 0.08 秒で「下がれる」
     とし、続く車両の値の止まりの見積もり 0.07 秒だけで 'brake' から 'gap' の中立を飛ばして 'reverse' へ進み、backed_s > 0 なので
     やり直しも効かず、1796 µs を 1.2 秒出しても 0 m のまま escape が前の壁を約 0.5 秒押した(反証 w2-safety-r62-safety-1)。
     最初の 'gap' 0.15 秒では下がれない ESC(10/10 のログで 11%)で、後退の入りぎわに車両の値が 1 回止まるだけでも、見積もりで
     やり直しが止まり同じことが起きた(w2-safety-r62-safety-2)

2. 後退の受け付け(EscReverse)"""

REPL = [
    (DOC_OLD, DOC_NEW),
    ("""        self.backed_s = 0.0
        self.t_last = None
        self.turn = 0.0""",
     """        self.backed_s = 0.0
        self.est_s = 0.0             # backed_s のうち、見ていない間の後退の見積もり(hold_rev_s)で足した秒(2026-10-27 fix6 r2 s3-c)
        self.t_last = None
        self.turn = 0.0"""),
    ("""        self.retries = 0
        self.backed_s = 0.0
        self.back_t = None
        self._go('brake', now)""",
     """        self.retries = 0
        self.backed_s = 0.0
        self.est_s = 0.0
        self.back_t = None
        self._go('brake', now)"""),
    ("""        ESC が後退で走らせていたかもしれない秒(drive が書いた値から見積もる。hold_back_s とは別。2026-10-27 fix6 s3-a。既定 0)。
        手順を変えて出す値があれば""",
     """        ESC が後退で走らせていたかもしれない秒(drive が書いた値から見積もる。hold_back_s とは別。2026-10-27 fix6 s3-a。既定 0)。
        見積もりは下がった時間に足すが、足した分は est_s に別に持ち、見積もりだけでは段を進めない(fix6 r2 s3-c)。
        手順を変えて出す値があれば"""),
    ("""            if est > 0.0:       # fix6 s3-a: 見ていない間の後退の見積もり。後退を HOLD_EST_MIN_LEFT_S 残すところまで足す
                self.backed_s += min(est, max(0.0, rev_s - HOLD_EST_MIN_LEFT_S - self.backed_s))""",
     """            if est > 0.0:       # fix6 s3-a: 見ていない間の後退の見積もり。後退を HOLD_EST_MIN_LEFT_S 残すところまで足す
                add = min(est, max(0.0, rev_s - HOLD_EST_MIN_LEFT_S - self.backed_s))
                self.backed_s += add
                self.est_s += add       # fix6 r2 s3-c: 見積もりで足した分('reverse' のやり直しは見た後退だけで決める)"""),
    ("""        if seen > 0.0 or est > 0.0:                     # 後ろへ動いていたサンプルか、見ていない間の後退の見積もりがあった""",
     """        # 後ろへ動いていたサンプルがあった: 段を進める(fix6 r2 s3-c: 見積もりだけでは進めない。見積もりは「下がれたかもしれない」
        # だけで、前は ESC がまだ下がれない 'brake' から 'gap' の中立を飛ばして 'reverse' へ進み、0 m のまま escape が前の壁を押した)
        if seen > 0.0:"""),
    ("""            # 下がり始めない: ESCが後退に入っていない(t_check は段を変えたとき t_phase と同じ。残した中立止めのあとは明けたフレーム)
            if self.backed_s == 0.0 and now - self.t_check > CHECK_S:
                if self.retries < self.MAX_RETRIES:
                    self.retries += 1
                    self._go('gap', now)""",
     """            # 下がり始めない: ESCが後退に入っていない(t_check は段を変えたとき t_phase と同じ。残した中立止めのあとは明けたフレーム)。
            # 見るのは見た後退だけ(fix6 r2 s3-c: 見ていない間の後退の見積もり est_s は除く。est_s = 0 なら今までと同じ)
            if self.backed_s - self.est_s == 0.0 and now - self.t_check > CHECK_S:
                if self.retries < self.MAX_RETRIES:
                    self.retries += 1
                    self.backed_s = self.est_s = 0.0       # 後退側を出しても下がらない: 見積もりの後退も無かった(fix6 r2)
                    self._go('gap', now)"""),
]


def main():
    s = SRC.read_text(encoding='utf-8')
    for a, b in REPL:
        assert s.count(a) == 1, a[:80]
        s = s.replace(a, b)
    DST.parent.mkdir(exist_ok=True)
    DST.write_text(s, encoding='utf-8')
    shutil.copyfile(HERE / 'before' / 'drive.py.orig', HERE / 'final' / 'drive.py')
    for p in (DST, HERE / 'final' / 'drive.py'):
        print(hashlib.sha256(p.read_bytes()).hexdigest(), p.name)


if __name__ == '__main__':
    main()
