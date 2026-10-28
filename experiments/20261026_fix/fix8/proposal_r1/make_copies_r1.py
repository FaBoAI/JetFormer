"""fix8 係 safety 直す 8-1 回目(rs8-1-backearly-rear-faster): 11/3 のあとに向けた直し方の案の写しを作る(本物の jetformer/ には
書かない)。jetformer/assist.py(今の版 ad6db15b…)を読み、Recovery._guard の back_early の分かれだけを書き換えた写しを
proposal_r1/<案>/assist.py に書く。どちらも後ろの見張り(rec_back_guard = 1。_guard は見張り ON のときだけ呼ばれる)の日だけ効く
= 見張り OFF は今とビット単位で同じ。drive.py は変えない(台本は今の jetformer/drive.py を写しの assist で読み込む)。

  qA「数えるだけ(段は飛ばさない)」: 'brake'・'wait' の中で(ESC が後退で走らせていたかもしれない手順で)後ろへ動いたら、出来事
      back_early を手順ごとに 1 回記録し、下がった時間(backed_s)に数えるだけ。段は OFF と同じ 'brake'(0.25 秒)→ 'wait' → 'gap'
      (中立 0.15 秒)→ 'reverse' のまま進め、'reverse' は 'brake'・'gap' で数えた分も含めて rev_s に届いたら惰性へ(下がった時間が
      rev_s に届けば 'brake'・'gap' の中でも惰性へ。back_enough)。書く値の並びは、惰性に移るまで OFF と同じ(= 後退側を出す時間は
      OFF より短いか同じ)。
  qB「中立を 1 回はさむ」(反証の案 b): back_early のとき 'reverse' へ飛ばさず 'gap'(中立 0.15 秒)へ進める。下がった時間は
      back_early のフレームから数える('gap' の惰性も今と同じく数える)。'gap' のあとはふつうの 'reverse'(skip_gap を立てない =
      fix7 の s4-b・s4-c の 'gap' を飛ばした後退の決まりは通らない)。
    cd ~/jetformer && python3 -I experiments/20261026_fix/fix8/proposal_r1/make_copies_r1.py [check]
"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path('/home/jetson/jetformer/jetformer/assist.py')
NOW_SHA = 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'

OLD_EARLY = """        if self.phase in ('brake', 'wait'):            # 前のブレーキ側で ESC が後退を待っていた: もう下がっている
            self.guard_event = ('back_early', self.phase, round(self.backed_s, 3))
            self._go('reverse', now)                   # 下がった時間は 'reverse' が数える(このフレームから)
            self.skip_gap = True                       # fix7 s4-b: 'gap' を飛ばした(後退を見なければ 'gap' からやり直す)
        elif self.phase == 'gap':"""

NEW_EARLY = {
    'qA': """        if self.phase in ('brake', 'wait'):            # 前のブレーキ側で ESC が後退を待っていた: もう下がっている
            # 案 qA(fix8 r1): 段は飛ばさず(OFF と同じ 'brake' → 'wait' → 'gap' → 'reverse')、下がった時間に数えるだけ。
            # 前は 'reverse' へ飛ばして 'gap' の中立をはさまず後退側を出し続け、近い後ろの物に OFF より速く当たった
            if not self.q_early:
                self.q_early = True
                self.guard_event = ('back_early', self.phase, round(self.backed_s, 3))
            self.backed_s += dt
            if self.backed_s >= rev_s:
                self.guard_event = ('back_enough', self.phase, round(self.backed_s, 3))
                self._go('coast', now)
        elif self.phase == 'gap':""",
    'qC': """        if self.phase in ('brake', 'wait'):            # 前のブレーキ側で ESC が後退を待っていた: もう下がっている
            # 案 qC(fix8 r1): 段は飛ばさず(OFF と同じ 'brake' → 'wait' → 'gap' → 'reverse')、下がった時間に数えるだけ。'gap'(中立)の
            # 惰性はこの手順では数えない(後退側を書いていた間の下がった時間で rev_s を数える = 今の ON と同じ長さの後退側)
            if not self.q_early:
                self.q_early = True
                self.guard_event = ('back_early', self.phase, round(self.backed_s, 3))
            self.backed_s += dt
            if self.backed_s >= rev_s:
                self.guard_event = ('back_enough', self.phase, round(self.backed_s, 3))
                self._go('coast', now)
        elif self.phase == 'gap' and self.q_early:      # 案 qC: back_early のあとの 'gap' の惰性は数えない
            pass
        elif self.phase == 'gap':""",
    'qB': """        if self.phase in ('brake', 'wait'):            # 前のブレーキ側で ESC が後退を待っていた: もう下がっている
            # 案 qB(fix8 r1): 'reverse' へ飛ばさず 'gap'(中立)を 1 回はさむ。下がった時間はこのフレームから数える
            # (前は 'gap' の中立をはさまず後退側を出し続け、近い後ろの物に OFF より速く当たった)
            self.guard_event = ('back_early', self.phase, round(self.backed_s, 3))
            self.backed_s += dt
            self._go('gap', now)
        elif self.phase == 'gap':""",
}

# qA だけ: 手順ごとに 1 回の印(reset・_start で戻す)
OLD_RESET = """        self.noesc_noted = False     # この手順で back_no_esc を記録した(fix7 s4-a)
"""
NEW_RESET = """        self.noesc_noted = False     # この手順で back_no_esc を記録した(fix7 s4-a)
        self.q_early = False         # 案 qA・qC: この手順で back_early を記録した
"""
OLD_START = """        self.noesc_noted = False
        self.esc_ran = False
        self.run_t0, self.back_sure = None, False
"""
NEW_START = """        self.noesc_noted = False
        self.q_early = False
        self.esc_ran = False
        self.run_t0, self.back_sure = None, False
"""


def make(name, src):
    assert src.count(OLD_EARLY) == 1
    out = src.replace(OLD_EARLY, NEW_EARLY[name])
    if name in ('qA', 'qC'):
        for o, n in ((OLD_RESET, NEW_RESET), (OLD_START, NEW_START)):
            assert out.count(o) == 1, o
            out = out.replace(o, n)
    return out


def main(check=False):
    src = SRC.read_text(encoding='utf-8')
    assert hashlib.sha256(src.encode('utf-8')).hexdigest() == NOW_SHA
    res = {}
    for name in ('qA', 'qB', 'qC'):
        out = make(name, src)
        a, b = src.splitlines(), out.splitlines()
        import difflib
        dl = [x for x in difflib.unified_diff(a, b, lineterm='', n=0) if x[:1] in '+-' and x[:3] not in ('+++', '---')]
        h = hashlib.sha256(out.encode('utf-8')).hexdigest()
        res[name] = {'sha256': h, 'plus': sum(1 for x in dl if x.startswith('+')), 'minus': sum(1 for x in dl if x.startswith('-'))}
        compile(out, name, 'exec')
        if not check:
            d = HERE / name
            d.mkdir(exist_ok=True)
            (d / 'assist.py').write_text(out, encoding='utf-8')
        else:
            assert (HERE / name / 'assist.py').read_text(encoding='utf-8') == out, name
    for k, v in res.items():
        print(k, v)
    return res


if __name__ == '__main__':
    main(len(sys.argv) > 1 and sys.argv[1] == 'check')
