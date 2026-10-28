"""反証 8-1 回目・打ち消しの票 B・指摘 w1-tools-r8-1-trial-form(読むだけ。コード・文書は変えない)。

確かめること(見方 B = 11/3 の手順書どおりの使い方で届くか・どれだけ悪いか):
  1. 指摘の形(3-8 = 形の違う試しの ON で、最初の周に 衝突 6G / 空転 / IMU の欠け)で ⑤ --ignore-gain の答え(今の collect.py)。
  2. 同じ日で、まだ「試し」と書いていないとき(手順書 7 章「起こし直した」の行・8.1 ⑤ の「計画と違う形の本を「試し」か --skip で外すと
     1 つの形に戻る」の前)= 形の分かれ(form_split)で「満たさない」か。→ 手順書の直し方(試しと書く)で「使ってよい」に変わるか。
  3. その場の守り: 4 章 10 の car_check(1 本ごと。試しかどうかを見ない)が、その本で衝突・空転を出すか(7 章「発進の前進 ON の最初の周で
     LH・R1 に当たる → ON をやめる」「空転の比 > 1.3 → 200 / もう 200 µs 以下 → ⑤ は満たさない」の手がかり)。
  4. ⑤ の標準出力に、試しの組の「満たさない」「数える」の文が並ぶか(読む人が気づけるか)。
作り物は review_tools_r1/probe_t1_trial_form.py の day() と同じ作り方(写しを下に。試しの印を付けるかだけ足した)。作り物の置き場は
../tmp_refute_tools_B(終わって消す)。出力 out/b1_trial_form.txt

  PYTHONDONTWRITEBYTECODE=1 jf-heavy --lock none --mem 1500M -- python -B b1_trial_form.py
"""
import contextlib
import io
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REV = HERE.parent / 'review_tools_r1'
sys.path.insert(0, str(REV))
import common as C  # noqa: E402  (反証の係の共通部分。読むだけ)
from common import t7, t72, t8, PRM, ON200  # noqa: E402

C.TMP = HERE.parent / 'tmp_refute_tools_B'
OUT = HERE / 'out'


def day(tmp, shape, form, mark_trial=True):
    """probe_t1_trial_form.day の写し + mark_trial(False = まだ「試し」と書いていない)"""
    base_prm = dict(PRM)
    if form == 'left42':
        base_prm.update(speed_ceiling=1, ceil_left_mps=4.1)
    on_prm = dict(ON200, **{k: base_prm[k] for k in ('speed_ceiling', 'ceil_left_mps')})
    D = t7.Day(tmp)
    for blk, prm, dm, ramp in ((12, base_prm, 0.20, 0.6), (13, on_prm, 0.15, 0.3), (14, base_prm, 0.21, 0.6),
                               (15, on_prm, 0.20, 0.3), (16, base_prm, 0.22, 0.6), (17, on_prm, 0.28, 0.3), (18, base_prm, 0.23, 0.6)):
        D.add(blk, prm, dm, ramp)
    p38 = dict(on_prm)
    if form == 'ceil':
        p38.update(speed_ceiling=1)
    elif form == 'left42':
        p38.update(ceil_left_mps=4.2)
    kw = {}
    if shape == 'hit':
        kw['hits'] = [(3.0, 6.0)]
    if shape == 'gap':
        log = D.add(19, p38, 0.15, 0.3, dur=45.0, vk=t7.launch_keys(14.0, 0.15, 0.3, until=45.0), ai=[(14.0, 43.0)],
                    launch=[(14.0, 0.5)])
        (D.raw / log / ('%s-0002' % log) / 'IMU.npz').unlink()
    else:
        log = D.add(19, p38, 0.15, 0.3, **kw)
        if shape == 'spin':
            t72.scale_imu(D.raw, log, 1.0, 2.2, 0.55)
    if mark_trial:
        t8.make_kind(D.raw, log, 'trial')
    return D, log


def car_check_text(D, log):
    import car_check as cc
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            cc.main([log, '--raw', str(D.raw), '--labels', str(D.tmp / 'nolab'), '--no-json'])
        except SystemExit:
            pass
    return buf.getvalue()


def main():
    lines = []
    for shape in ('hit', 'spin', 'gap'):
        for form in ('same', 'ceil', 'left42'):
            for mark in (False, True):
                D, log = day(C.fresh('b1_%s_%s_%d' % (shape, form, mark)), shape, form, mark)
                j, out, _ = C.runc(D, 'new', C.IG, 'b1%s%s%d' % (shape, form, mark))
                lines.append('== 3-8 = ON・%s・形 %s・memo「試し」%s' % (shape, form, 'あり' if mark else 'なし(まだ書いていない)'))
                for g in j['summary']:
                    lines.append('  ' + C.group_text(g))
                    lines.append('    form_split %s' % g.get('form_split'))
                verdict_lines = [x for x in out.splitlines() if 'ON − OFF' in x or '→ 使ってよい' in x or '→ 満たさない' in x
                                 or ' → 試しの本' in x or 'うち ON の発進の最初の周の衝突' in x or '組に分かれた' in x
                                 or '最初の周の記録が欠けた発進' in x]
                lines += ['  | ' + x.strip()[:260] for x in verdict_lines]
                if mark and form == 'ceil' and shape in ('hit', 'spin'):
                    cc = car_check_text(D, log)
                    keep = [x for x in cc.splitlines() if '衝突' in x or '空転' in x or '判定' in x or '⑤' in x]
                    lines.append('  car_check(4 章 10。1 本ごと)の行:')
                    lines += ['  ¦ ' + x.strip()[:260] for x in keep]
    txt = '\n'.join(lines) + '\n'
    (OUT / 'b1_trial_form.txt').write_text(txt, encoding='utf-8')
    print(txt)
    if C.TMP.exists():
        shutil.rmtree(C.TMP)


if __name__ == '__main__':
    main()
