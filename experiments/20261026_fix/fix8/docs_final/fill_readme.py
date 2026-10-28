"""fix8/README.md の 10 章(今のファイルの sha256)を今のファイルから作り直す・9 章(試験)を out/results_readme.md の中身にする。
何度流しても同じ形(10 章・9 章の中身だけを置き換える)。使い方: python3 -I -B fill_readme.py"""
import hashlib
import sys
from pathlib import Path

REPORTS = Path('/home/jetson/jetformer_reports/fix_20261026')
JF = Path('/home/jetson/jetformer')
HERE = JF / 'experiments' / '20261026_fix' / 'fix8' / 'docs_final'
README = REPORTS / 'fix8' / 'README.md'
ROWS = [
    ('runbook_20261103.md', '手順書(この仕上げのあと)'), ('todo_user_1026.md', 'ユーザーが 11/3 までにすること(同)'),
    ('README.md', 'まとめ(同)'), ('fix3/approvals.md', '了承が要ること(同)'), ('fix7/README.md', '反証 7 回目のまとめ(8 回目の同期 1 のあと。仕上げは変えていない)'),
    ('fix8/proposal_safety_after1103.md', '11/3 のあとの安全の案(1〜9 章 + 仕上げの 10 章)'), ('fix8/docs_final.md', 'この仕上げの報告'),
    ('jetformer/assist.py', '安全(8 回目は変えていない)'), ('jetformer/drive.py', '走行ループ(10/28 02:33 の説明の文のあと)'),
    ('jetformer/web/server.py', 'サーバー(変えていない)'), ('jetformer/web/bench.py', 'ベンチマーク(変えていない)'),
    ('jetformer/speed_auto.py', '自動合わせ(変えていない)'), ('jetformer/web/static/app.js', '画面(fix8 ui r2・10/28 11:02)'),
    ('jetformer/web/static/style.css', '画面(fix8 ui r2・10/28 10:15)'), ('jetformer/web/static/index.html', '画面(変えていない)'),
    ('experiments/20261026_fix/diag_launch/collect.py', '⑤ 夜の道具(fix8 tools r2・10/28 10:53)'),
    ('experiments/20261026_fix/plan/runs_1103.py', '② 夜の道具(fix8 ui r1)'), ('tools/car_check.py', 'car_check(8 回目は変えていない)'),
    ('tools/standing_3lap.py', '③(fix8 tools-decided)'), ('tools/quick_eval_by_params.py', '④(fix8 tools-decided)'),
    ('data/drive_settings.json', '走行の設定(変えていない)'), ('data/bench_candidates.json', 'ベンチマークの候補(変えていない)'),
    ('experiments/20261026_fix/fix8/docs_final/docs_fix8_final.py', 'この仕上げの台本'),
    ('tests/test_fix8_docs_final.py', 'この仕上げの試験'),
]


def sha(rel):
    base = JF if rel.startswith(('jetformer/', 'tools/', 'experiments/', 'data/', 'tests/')) else REPORTS
    return hashlib.sha256((base / rel).read_bytes()).hexdigest()[:12]


def replace_section(t, head, nxt, body):
    i = t.index(head)
    j = t.index('\n', i) + 1
    k = t.index(nxt, j)
    return t[:j] + '\n' + body.rstrip('\n') + '\n\n' + t[k:]


def main():
    t = README.read_text(encoding='utf-8')
    table = ['| ファイル | sha256 | 中身 |', '|---|---|---|'] + ['| `%s` | `%s…` | %s |' % (rel, sha(rel), what) for rel, what in ROWS]
    t = replace_section(t, '## 10. 今のファイル', '## 11. 次に', '\n'.join(table))
    res = HERE / 'out' / 'results_readme.md'
    if res.exists():
        t = replace_section(t, '## 9. 試験', '## 10. 今のファイル', res.read_text(encoding='utf-8'))
    README.write_text(t, encoding='utf-8')
    print('書いた: %s(10 章 %d 行)' % (README, len(ROWS)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
