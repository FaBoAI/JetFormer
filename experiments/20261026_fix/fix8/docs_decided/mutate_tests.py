"""fix8 docs-decided: 合わせた前の係の試験 3 本(test_fix7_docs_r1・r2・final の「済」と「わざと壊す」)が弱まっていないかの確かめ。
本物の文書は変えない(読んだ中身を写しの dict で壊し、試験の関数をそのまま呼ぶ)。
- SUPERSEDE の書き換えを 1 つずつ戻す(fix8 の文が消える)→ 3 本の「済」の試験が落ちる(superseded_present)
- 前の係の new のうち書き換えていない文を 1 つずつ壊す → 前と同じく「済」の試験が落ちる
出力: out/mutate_tests.txt"""
import importlib.util
import sys
import tempfile
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF)); sys.path.insert(0, str(JF / 'tools')); sys.path.insert(0, str(JF / 'tests'))


def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m


f8 = load('m_fix8', JF / 'experiments/20261026_fix/fix8/docs_decided/docs_fix8_decided.py')
T = {k: load('m_t_' + k, JF / ('tests/test_fix7_docs_%s.py' % k)) for k in ('r1', 'r2', 'final')}
S = {k: load('m_s_' + k, T[k].SCRIPT) for k in T}


def run(k, live):
    try:
        if k == 'final':
            with tempfile.TemporaryDirectory() as tmp:
                T[k].test_script_edits_all_done_and_earlier_intact(live, S[k], f8, Path(tmp))
        else:
            T[k].test_script_edits_all_done(live, S[k], f8)
        return 'pass'
    except AssertionError:
        return 'FAIL'


base = {k: T[k].read(T[k].REPORTS) for k in T}
lines = ['今の文書: ' + ', '.join('%s %s' % (k, run(k, base[k])) for k in T)]
n_ok = 0
for idx, (s, f, i, x, y) in enumerate(f8.SUPERSEDE):
    res = []
    for k in T:
        if f not in base[k]:
            continue
        live = dict(base[k], **{f: base[k][f].replace(y, x, 1)})
        res.append('%s %s' % (k, run(k, live)))
    ok = res and all(r.endswith('FAIL') for r in res)
    n_ok += bool(ok)
    lines.append('SUPERSEDE %2d (%s %s #%d) を戻す: %s' % (idx, s, f, i, ', '.join(res)))
# 書き換えていない前の係の new を 1 つずつ壊す(頭の 1 文字を変える)。その台本の試験が落ちるか
n2 = n2_ok = 0
for k in T:
    for f, edits in S[k].ALL:
        for i, (old, new) in enumerate(edits):
            if any((s, ff, ii) == (k, f, i) for s, ff, ii, _, _ in f8.SUPERSEDE):
                continue
            live = base[k]
            if f not in live or live[f].count(new) != 1 or len(new) < 4:
                continue
            j = len(new) // 2
            broken = dict(live, **{f: live[f].replace(new, new[:j] + '〓' + new[j:], 1)})
            n2 += 1
            n2_ok += run(k, broken) == 'FAIL'
lines.append('SUPERSEDE を戻すと落ちる: %d / %d' % (n_ok, len(f8.SUPERSEDE)))
lines.append('書き換えていない前の係の new を壊すと「済」の試験が落ちる: %d / %d' % (n2_ok, n2))
out = JF / 'experiments/20261026_fix/fix8/docs_decided/out/mutate_tests.txt'
out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
