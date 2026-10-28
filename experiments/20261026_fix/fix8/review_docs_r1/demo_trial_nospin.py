"""反証 8-1 docs: todo 3 章の「試しの本 … (ただし ON の発進の最初の周の衝突・空転・記録の欠けは ⑤ が数える = 回 3 は「満たさない」になる)」を、
今の ⑤ collect.py で照らす(作り物の回 3。tests/test_fix8_tools_decided.py の mtday・runc を借りる。読むだけ・一時の場所は消す)。
3-8 が「試し」の ON(空転なし・当たりなし・欠けなし)と、同じ本が空転(比 ≈ 1.8)の 2 通りで ⑤ の「ON − OFF」の行を出す。"""
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
import test_fix8_tools_decided as T  # noqa: E402

TMP = JF / 'experiments/20261026_fix/fix8/tmp_review_docs_r1/trial'
for shape in ('nospin', 'spin'):
    d = TMP / shape
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    D, log = T.mtday(d, 'trial', shape)
    j, out, _ = T.runc(D, 'new', T.IG, 'demo' + shape)
    g = T.g0(j)
    print('== 3-8 が試しの ON・%s(%s)' % ({'nospin': '空転なし・当たりなし・欠けなし', 'spin': '空転 比 ≈ 1.8'}[shape], log))
    for x in out.splitlines():
        if 'ON − OFF' in x or x.startswith('  試しの') or x.startswith('  → 試しの本'):
            print('  ', x[:260])
    print('   all_ok =', g['compare']['all_ok'], ' checks =', {k: v for k, v in g['compare']['checks'].items() if v is False})
shutil.rmtree(TMP, ignore_errors=True)
