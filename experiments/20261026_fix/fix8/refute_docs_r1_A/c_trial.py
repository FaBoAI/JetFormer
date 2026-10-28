"""打ち消しの票 A(f8r1-docs-todo-trial-wording): 今の ⑤ collect.py で、回 3 の 3-8 が「試し」の ON のとき
空転なし / 空転(比 ≈ 1.8)/ 最初の周の当たり の 3 通りで ⑤ の結論(all_ok・→ の行)を出す。
作り物の日は tests/test_fix8_tools_decided.py の mtday(試験の作り方)。試しの印が本当に付いたか(行の trial)も出す。
一時の場所は fix8/tmp_refute_docs_A(終わって消す)。"""
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
import test_fix8_tools_decided as T  # noqa: E402

TMP = JF / 'experiments/20261026_fix/fix8/tmp_refute_docs_A/trial'
cases = [('nospin', {}), ('spin', {}), ('nospin', {'hits': [(3.0, 6.0)]})]
try:
    for i, (shape, kw) in enumerate(cases):
        d = TMP / ('c%d' % i)
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        D, log = T.mtday(d, 'trial', shape, **kw)
        j, out, _ = T.runc(D, 'new', T.IG, 'rA%d' % i)
        g = T.g0(j)
        rows = T.rows_of(j, log)
        r = rows[0]
        print('== case %d: 3-8 = 試しの ON・shape=%s kw=%s (%s)' % (i, shape, kw, log))
        print('   row: trial=%r launch_state=%r slip_ratio=%r' % (r.get('trial'), r.get('launch_state'),
                                                                 (r.get('slip') or {}).get('ratio')))
        print('   n_on_in_compare=%r d_to_r1_s=%r' % (g['compare'].get('n_on'), g['compare'].get('d_to_r1_s')))
        print('   all_ok=%r  failed_checks=%r' % (g['compare']['all_ok'],
                                                 [k for k, v in g['compare']['checks'].items() if v is False]))
        for x in out.splitlines():
            if 'ON − OFF' in x or '試し' in x:
                print('   |', x[:240])
finally:
    shutil.rmtree(TMP.parent, ignore_errors=True)
