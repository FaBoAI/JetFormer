"""fix8 docs r1: 合わせた前の係の試験 2 本(tests/test_fix8_docs_decided.py・tests/test_fix7_docs_final.py の「済」と「わざと壊す」)が
弱まっていないかの確かめ。本物の文書は変えない(読んだ中身を写しの dict で壊し、試験の関数をそのまま呼ぶ)。
- 今の文書: 合わせた 2 本は通る。合わせる前の 2 本(before_tests/*.orig)は落ちる(合わせる要があった)
- この組の SUPERSEDE の書き換えを 1 つずつ戻す(fix8 docs r1 の文が消える)→ 合わせた 2 本の「済」の試験が落ちる
- 前の係の new のうち、この組が書き換えていない文を 1 つずつ壊す(まん中に 1 字)→ 前と同じく、その台本の「済」の試験が落ちる
- 合わせた「わざと壊す」の試験も今の文書で通る(68 個・15 個を同じ確かめで照らす)
出力: out/mutate_tests.txt"""
import importlib.util
import sys
import tempfile
from pathlib import Path

JF = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(JF)); sys.path.insert(0, str(JF / 'tools')); sys.path.insert(0, str(JF / 'tests'))


def load(n, p):
    from importlib.machinery import SourceFileLoader
    s = importlib.util.spec_from_loader(n, SourceFileLoader(n, str(p))); m = importlib.util.module_from_spec(s); sys.modules[n] = m
    s.loader.exec_module(m); return m


r81 = load('m_r81', HERE / 'docs_fix8_r1.py')
f8 = load('m_f8', JF / 'experiments/20261026_fix/fix8/docs_decided/docs_fix8_decided.py')
TD = load('m_t_dec', JF / 'tests/test_fix8_docs_decided.py')
TF = load('m_t_fin', JF / 'tests/test_fix7_docs_final.py')
OD = load('m_o_dec', HERE / 'before_tests/test_fix8_docs_decided.py.orig')
OF = load('m_o_fin', HERE / 'before_tests/test_fix7_docs_final.py.orig')
SD = load('m_s_dec', TD.SCRIPT)
SF = load('m_s_fin', TF.SCRIPT)
TMP = HERE.parent / 'tmp_docs_r1'
TMP.mkdir(exist_ok=True)


def run(fn, *a):
    try:
        fn(*a)
        return 'pass'
    except AssertionError:
        return 'FAIL'


def dec_done(live):
    return run(TD.test_script_edits_all_done, live, SD, r81)


def fin_done(live):
    with tempfile.TemporaryDirectory(dir=str(TMP)) as tmp:
        return run(TF.test_script_edits_all_done_and_earlier_intact, live, SF, f8, r81, Path(tmp))


bd = TD.read(TD.REPORTS)
bf = TF.read(TF.REPORTS)
lines = ['今の文書: 合わせた decided の済 %s・わざと壊す %s / 合わせた final の済 %s・わざと壊す %s' % (
    dec_done(bd), run(TD.test_reverting_each_edit_is_caught, bd, SD, r81), fin_done(bf),
    run(TF.test_reverting_each_edit_is_caught, bf, SF, f8, r81))]
with tempfile.TemporaryDirectory(dir=str(TMP)) as tmp:
    lines.append('合わせる前(.orig): decided の済 %s・わざと壊す %s / final の済 %s・わざと壊す %s' % (
        run(OD.test_script_edits_all_done, bd, SD), run(OD.test_reverting_each_edit_is_caught, bd, SD),
        run(OF.test_script_edits_all_done_and_earlier_intact, bf, SF, f8, Path(tmp)),
        run(OF.test_reverting_each_edit_is_caught, bf, SF, f8)))
n_ok = 0
for idx, (s, f, i, x, y) in enumerate(r81.SUPERSEDE):
    res = ['decided %s' % dec_done(dict(bd, **{f: bd[f].replace(y, x, 1)}))]
    if f in bf:              # final の試験の live は 4 本(fix7/README.md は試験が自分で読む)
        res.append('final %s' % fin_done(dict(bf, **{f: bf[f].replace(y, x, 1)})))
    ok = all(r.endswith('FAIL') for r in res)
    n_ok += bool(ok)
    lines.append('この組の SUPERSEDE %d(%s %s #%d)を戻す: %s' % (idx, s, f, i, ', '.join(res)))
# 書き換えていない前の係の new を 1 つずつ壊す
n2 = n2_ok = 0
for name, S, live, done, key in (('decided', SD, bd, dec_done, 'decided'), ('final', SF, bf, fin_done, 'final')):
    for f, edits in S.ALL:
        for i, (old, new) in enumerate(edits):
            if any((s, ff, ii) == (key, f, i) for s, ff, ii, _, _ in r81.SUPERSEDE):
                continue
            if key == 'final':
                new = f8.effective('final', f, i, new)
            if f not in live or live[f].count(new) != 1 or len(new) < 4:
                continue
            j = len(new) // 2
            broken = dict(live, **{f: live[f].replace(new, new[:j] + '〓' + new[j:], 1)})
            n2 += 1
            n2_ok += done(broken) == 'FAIL'
lines.append('この組の SUPERSEDE を戻すと合わせた「済」の試験が落ちる: %d / %d' % (n_ok, len(r81.SUPERSEDE)))
lines.append('書き換えていない前の係の new を壊すと合わせた「済」の試験が落ちる: %d / %d' % (n2_ok, n2))
out = HERE / 'out' / 'mutate_tests.txt'
out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
