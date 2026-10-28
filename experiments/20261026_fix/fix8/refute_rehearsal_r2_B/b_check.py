"""打ち消しの票 B(反証 8-2 回目・係 rehearsal の 2 件)。読むだけ。手順書・コード・記録を変えない。
1) rehearsal-r2-unmarked-rerun-skip-drops-hit: 手順書で --skip の文の近くにある「当たりは数える / 当たった本は --skip しない」の文、
   collect.py の --skip の扱い(行ごと外す)と説明、standing_3lap.abort_resumes の印の形を並べる。
2) rehearsal-r2-vbefore-screen-vs-server: app.js の正規表現と speed_auto.v_before_of・car_check(section_scorecard._volt)の読み方を
   同じ入力で並べ、本物の記録(data/raw の log.json)で人が V_before に何を書いてきたかを数える。
"""
import glob
import json
import re
import sys
from collections import Counter

JF = '/home/jetson/jetformer'
RB = '/home/jetson/jetformer_reports/fix_20261026/runbook_20261103.md'
sys.path.insert(0, JF)
sys.path.insert(0, JF + '/tools')
from jetformer import speed_auto  # noqa: E402
import section_scorecard as sc  # noqa: E402

lines = open(RB, encoding='utf-8').read().split('\n')
print('== 1) 手順書の文(行番号: 抜き出し)')
pats = [r'ただし ON の走り直しの最初の周の衝突[^。]*。?[^)]*\)', r'印が無ければ、⑤ は main session が[^)]*\)',
        r'当たった・人が危なくて止めた → 回 4 は OFF[^。]*。[^。]*。', r'当たった本は `--skip` しない',
        r'`--skip 記録名`\(か `記録名@AI側の秒`\)はまとめから外す\([^)]*\)', r'印の無い走り直し\(紙に「走り直し」と書いた本\)は[^。]*。',
        r'\*\*発進の前進 ON の最初の周で LH・R1 に当たる\*\* \| [^|]*']
for p in pats:
    for i, l in enumerate(lines, 1):
        for m in re.finditer(p, l):
            print('  %d: %s' % (i, m.group(0)[:260]))
src = open(JF + '/experiments/20261026_fix/diag_launch/collect.py', encoding='utf-8').read().split('\n')
print('== collect.py の --skip')
for i, l in enumerate(src, 1):
    if 'rows_sum = [r for r in rows if not hit(r)]' in l or '当たった本は外さない' in l:
        print('  %d: %s' % (i, l.strip()[:200]))
st = open(JF + '/tools/standing_3lap.py', encoding='utf-8').read()
print('== standing_3lap.abort_resumes の形:', sorted(set(re.findall(r"'kind': '(\w+)'", st)) | set(re.findall(r"kind='(\w+)'", st))))

print('== 2) 同じ説明欄を 3 つの読み方で')
JSRE = re.compile(r'(^|\s)V_before=[0-9]')        # app.js: /(^|\s)V_before=\d/(JS の \d は ASCII の 0-9 だけ)
for v in ('8.31', '831', '8,31', '0.83', '4.15', '８.３１', '８．３１', '__', '8.31V'):
    d = 'x battery=#A V_before=%s V_after=__ check=合格' % v
    print('  V_before=%-6s 確認の文 %-6s サーバー %-6s car_check の電圧 %s' % (
        v, '説明欄' if JSRE.search(d) else 'なし', speed_auto.v_before_of(d), sc._volt(v)))
print('== 本物の記録で人が書いた V_before')
c = Counter()
n = 0
for f in sorted(glob.glob(JF + '/data/raw/*/log.json')):
    try:
        t = open(f, encoding='utf-8', errors='replace').read()
    except OSError:
        continue
    n += 1
    for m in re.findall(r'V_before=([^ "\\]*)', t):
        c[m] += 1
print('  log.json %d 本: %s' % (n, dict(c)))
