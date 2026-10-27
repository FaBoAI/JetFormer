"""p1 の B(3-8 の AI 側のシーンごと書けずに落ちた本)を car_check(--latest と名前)・③ standing_3lap で読むと何が出るか"""
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1_lost_ai_scene as P  # noqa: E402
import car_check as cc  # noqa: E402
import standing_3lap as s3  # noqa: E402

TMP = P.TMP


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    (TMP / 'B').mkdir()
    D, log = P.day(TMP / 'B', True)
    out = {}
    for name, argv in (('latest', ['--latest', '--day', '20261103']), ('by_name', [log])):
        so, se = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
            rc = cc.main(argv + ['--raw', str(D.raw), '--labels', str(D.tmp / 'nolab'), '--no-json'])
        out['car_check_' + name] = {'rc': rc, 'stdout': so.getvalue().splitlines()[:6], 'stderr': se.getvalue()[:400]}
    r = s3.analyze_log(log, D.raw, D.tmp / 'nolab', {'runB': 'B'})
    out['standing'] = {'standing': r.get('standing'), 'missing_npz': r.get('missing_npz'), 'error': r.get('error')}
    ev = (D.raw / log / 'events.jsonl').read_text()
    out['events_jsonl'] = ev.strip()
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    (P.OUT / 'p1b_other_tools.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    shutil.rmtree(TMP)


if __name__ == '__main__':
    main()
