"""反証 7-1・docs: ユーザーが読む文書の「…」の引用が、コード(画面・サーバー・道具)の文字列にあるかを照らす(読むだけ)。

引用を … ・HH:MM:SS・N・X などで切り、4 文字以上の切れ端がコードの文字列(連結をほどいた形)に無いものを出す。
文書の中の言葉(「試し」など)も出るので、出たものは人が読んで分ける。
"""
import re
import sys
from pathlib import Path

HOME = Path('/home/jetson')
JF = HOME / 'jetformer'
REP = HOME / 'jetformer_reports/fix_20261026'
DOCS = [REP / 'runbook_20261103.md', REP / 'todo_user_1026.md', REP / 'README.md', REP / 'fix3/approvals.md',
        REP / 'fix6/README.md']
CODE = [JF / 'jetformer/web/static/app.js', JF / 'jetformer/web/server.py', JF / 'jetformer/web/bench.py',
        JF / 'jetformer/drive.py', JF / 'jetformer/assist.py',
        JF / 'tools/car_check.py', JF / 'tools/standing_3lap.py', JF / 'tools/quick_eval_by_params.py',
        JF / 'tools/fw_vbat_check.py', JF / 'tools/bench_nightly.sh',
        JF / 'experiments/20261026_fix/diag_launch/collect.py', JF / 'experiments/20261026_fix/plan/runs_1103.py',
        JF / 'experiments/20240000' / 'x',
        JF / 'experiments/20261024_tune/quick_eval.py']


def norm_code(t):
    # Python の隣り合う文字列・JS の + 連結をほどく
    t = re.sub(r"'\s*\n\s*'", '', t)
    t = re.sub(r'"\s*\n\s*"', '', t)
    t = re.sub(r"'\s*\+\s*\n?\s*'", '', t)
    t = re.sub(r"`\s*\+\s*\n?\s*'", '', t)
    t = re.sub(r"'\s*\+\s*\n?\s*`", '', t)
    return t


corpus = ''
for p in CODE:
    if p.exists():
        corpus += norm_code(p.read_text(encoding='utf-8', errors='replace')) + '\n'

PH = re.compile(r'…|\.\.\.|HH:MM:SS|HH:MM|MM/DD|\bN\b|\bX\b|\bM\b|0\.xx|%s|\s+')
out = []
for d in DOCS:
    txt = d.read_text(encoding='utf-8')
    for ln, line in enumerate(txt.splitlines(), 1):
        for m in re.finditer(r'「([^「」]{4,400})」', line):
            q = m.group(1)
            pieces = [x for x in PH.split(q) if len(x) >= 4]
            miss = [x for x in pieces if x not in corpus]
            if miss:
                out.append('%s:%d 「%s」 無い切れ端: %s' % (d.name, ln, q[:120], ' | '.join(miss)[:200]))
print('\n'.join(out))
print('件数', len(out), file=sys.stderr)
