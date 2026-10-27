"""係 tools の確かめ(2026-10-26 fix2): tools/quick_eval_by_params.py の本ごとの数を、quick_eval.py(--no-run。10/24 に作った
scorecard.json を読むだけ)の表と比べる。読むだけ。quick_eval.py は変えない(sha256 を前後で見る)。

    cd /home/jetson/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M --need-mb 800 -- \
        python3 -I experiments/20261026_fix/fix2/tools_check/check_quick_eval.py
    → 同じフォルダの check_quick_eval.json・check_quick_eval.txt
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, '/home/jetson/jetformer/tools')
import quick_eval_by_params as qp  # noqa: E402
import standing_3lap as s3  # noqa: E402

HERE = Path(__file__).resolve().parent
QE = Path('/home/jetson/jetformer/experiments/20261024_tune/quick_eval.py')
# 偽の起点を外したので通過が 1 つ減る記録(tests/test_lap_anchor.py の FALSE_LAPS。10/24 の No.7・8・10)
FALSE_ANCHOR_LOGS = {'20261024-160654', '20261024-160847', '20261024-162305'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse_qe(text):
    rows = []
    for line in text.splitlines():
        if not line.startswith('| ') or line.startswith('| No.') or line.startswith('| 段'):
            continue
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) != 15:
            continue
        rows.append({'no': c[0], 'cand': c[2], 'g': c[3], 'ai': c[5], 'n': c[6], 'clean': c[7], 'best': c[8], 'med': c[9],
                     'col': c[10], 'ent': c[11], 'to': c[12]})
    return rows


def main():
    h0 = sha(QE)
    out = {'quick_eval_sha256_before': h0}
    r = subprocess.run([sys.executable, '-I', str(QE), '1300', '--day=20261024', '--no-run'], cwd=str(QE.parent),
                       capture_output=True, text=True)
    qe = parse_qe(r.stdout)
    logs, _ = s3.day_logs('20261024', since='1300')
    res = s3.run(logs)
    runs = qp.run_rows(res)
    f2 = lambda x: '—' if x is None else '%.2f' % x      # noqa: E731
    diffs, same = [], 0
    for q, m in zip(qe, runs):
        mine = {'no': str(m['ai_no']), 'ai': '%.0f' % (m['ai_s'] or 0), 'n': str(m['passes']), 'clean': str(m['clean_laps']),
                'best': f2(m['best_clean_s']), 'med': f2(m['median_clean_s']), 'col': '%d(%d)' % (m['collisions'], m['strong']),
                'ent': qp._i(m['r1_entry_median_erpm']), 'to': str(m['takeovers_failure'])}
        d = {k: (q[k], v) for k, v in mine.items() if q[k] != v}
        if d:
            diffs.append({'log': m['log'], 'no': m['ai_no'], 'false_anchor_log': m['log'] in FALSE_ANCHOR_LOGS, 'diff': d})
        else:
            same += 1
    out.update(n_quick_eval=len(qe), n_mine=len(runs), same=same, diffs=diffs,
               unexpected=[x for x in diffs if not x['false_anchor_log'] or set(x['diff']) - {'n', 'ent'}],
               quick_eval_sha256_after=sha(QE))
    out['quick_eval_unchanged'] = out['quick_eval_sha256_before'] == out['quick_eval_sha256_after']
    (HERE / 'check_quick_eval.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    lines = ['quick_eval の本 %d・この台本の本 %d・同じ %d・違う %d(偽の起点のある記録だけで、通過と R1 の進入だけ: %s)' % (
        len(qe), len(runs), same, len(diffs), not out['unexpected'])]
    for x in diffs:
        lines.append('  %s No.%s 偽の起点 %s %s' % (x['log'], x['no'], x['false_anchor_log'], x['diff']))
    lines.append('quick_eval.py の sha256 %s… 前後で同じ: %s' % (h0[:12], out['quick_eval_unchanged']))
    (HERE / 'check_quick_eval.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
