"""直す 6-1 回目・係 tools: probe_lost_ai の B(3-8 の AI 側のシーンごと書けずに落ちた本)・C(2 回目の発進のシーンが無い本)を、
car_check(--latest と名前)・③ standing_3lap・④ quick_eval_by_params の印で読むと何が出るか(この係のほかの道具は直さなくてよいかの
確かめ)。作り物は fix6/tmp_tools/r1_other の下で、最後に消す。出力 ../out/probe_other_tools.json"""
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_lost_ai as P  # noqa: E402
import car_check as cc  # noqa: E402
import standing_3lap as s3  # noqa: E402
import quick_eval_by_params as qp  # noqa: E402

TMP = P.JF / 'experiments/20261026_fix/fix6/tmp_tools/r1_other'


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    out = {}
    try:
        for case in ('B', 'C'):
            (TMP / case).mkdir()
            D, log = P.day(TMP / case, case)
            o = {}
            for name, argv in (('latest', ['--latest', '--day', '20261103']), ('by_name', [log])):
                so, se = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
                    rc = cc.main(argv + ['--raw', str(D.raw), '--labels', str(D.tmp / 'nolab'), '--no-json'])
                o['car_check_' + name] = {'rc': rc, 'stdout': so.getvalue().splitlines()[:4], 'stderr': se.getvalue()[:300]}
            r = s3.run([log], D.raw, D.tmp / 'nolab', None)
            md = s3.to_markdown(r)
            o['standing'] = {'standing': [(x.get('t_ai'), x.get('to_r1_s'), x.get('counted')) for x in r[0].get('standing') or []],
                             'missing_npz': r[0].get('missing_npz'),
                             'md_lines': [x for x in md.splitlines() if '無いので飛ばした' in x or '記録が欠けている' in x]}
            o['by_params_flags'] = qp.flags_of(r[0])
            out[case] = o
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    (HERE.parent / 'out' / 'probe_other_tools.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str),
                                                                 encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
