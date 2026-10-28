"""反証 8-1 docs: ユーザーが読む文書の「済んだのに まだ」「同じことが 2 か所で違う」「古い時刻」を、記録・git・ほかの文書と照らす(読むだけ)。

出力は標準出力(out/doc_checks.txt に保存して使う)。どの節も「文書の文」と「照らした元」を並べる。
"""
import datetime
import os
import re
import subprocess
from pathlib import Path

REP = Path('/home/jetson/jetformer_reports/fix_20261026')
JF = Path('/home/jetson/jetformer')


def line(name, n):
    return (REP / name).read_text(encoding='utf-8').splitlines()[n - 1]


def snip(text, pat, before=80, after=80):
    m = re.search(pat, text)
    return text[max(0, m.start() - before):m.end() + after] if m else '(無い: %s)' % pat


def mtime(p):
    return datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime('%m/%d %H:%M:%S')


print('== A. README 6.1 の 5: commit のファイル数')
t = line('README.md', 202)
print('文書(README.md 202 行):', snip(t, '9da6d0d', 60, 40))
for c in ('fce5d47', '9da6d0d'):
    st = subprocess.run(['git', '-C', str(JF), 'show', '--stat', '--format=%h %ci', c], capture_output=True, text=True).stdout
    print('git show --stat %s: %s' % (c, st.strip().splitlines()[-1].strip()))
print('文書(README.md 25 行 = 0 章 7):', snip(line('README.md', 25), 'fce5d47', 0, 60))

print()
print('== B. README 6 章の頭「案の係はまだ動いている」と 7 章の報告の一覧')
t = line('README.md', 192)
print('文書(README.md 192 行):', snip(t, 'まだ動いている', 60, 0))
prop = REP / 'fix8/proposal_safety_after1103.md'
print('fix8/proposal_safety_after1103.md の更新 %s・README.md の更新 %s' % (mtime(prop), mtime(REP / 'README.md')))
print('  案の報告の要点:', snip(prop.read_text(encoding='utf-8'), r'- \*\*おすすめ: pB', 0, 60))
ud = (REP / 'user_decisions.md').read_text(encoding='utf-8')
print('  user_decisions.md(更新 %s):' % mtime(REP / 'user_decisions.md'), snip(ud, 'safety-proposal', 0, 40))
rd = (REP / 'README.md').read_text(encoding='utf-8')
sec7 = rd[rd.index('## 7. 報告の一覧'):]
print('  README 7 章に proposal_safety_after1103 の行:', 'あり' if 'proposal_safety_after1103' in sec7 else '無い')

print()
print('== C. fix7/README.md 7 章 4「approvals 6・「ON の形」の注の行はまだ」')
print('文書(fix7/README.md 114 行):', snip(line('fix7/README.md', 114), r'⑤ は再起動の行だけ', 0, 40), '(更新 %s)' % mtime(REP / 'fix7/README.md'))
for pat in (r'\| 10/28 03:4x \| \(記録\)approvals 6', r'\| 10/28 03:4x \| \(注\)approvals 3 の \(c\) を決めたときの「ON の形」'):
    print('  user_decisions.md の行:', snip(ud, pat, 0, 60))
print('  fix8/decided_docs.md 6 章 3:', snip((REP / 'fix8/decided_docs.md').read_text(encoding='utf-8'), r'はまだ無い', 120, 10))

print()
print('== D. approvals 3 の表: 前の行と新しい決まりの行')
ap = (REP / 'fix3/approvals.md').read_text(encoding='utf-8').splitlines()
r71 = ap[70]
print('71 行(途中で OFF にした日):', snip(r71, r'読み方は main session が決める', 30, 3), '…', snip(r71, r'決まっていないのは', 2, 50), '…', snip(r71, r'決めたら main session がこの行に書く', 20, 3))
print('72 行(決まり):', ap[71][:90])
print('67 行(困ること(ON)):', snip(ap[66], r'45,844', 20, 30))
print('72 行(決まり):', snip(ap[71], r'45,918', 40, 30))
fr = (REP / 'fix3/fix_r3_1.md').read_text(encoding='utf-8')
print('  45,844 の元(fix3/fix_r3_1.md):', snip(fr, r'45,844', 40, 30))
sg = Path(__file__).resolve().parent / 'rerun/out/scan_gaps10.txt'
if sg.exists():
    print('  自分で流した scan_gaps10.py(AI 側の処理したフレームの間):')
    for x in sg.read_text(encoding='utf-8').splitlines()[:4]:
        print('    ', x[:140])

print()
print('== E. todo 3 章の「試し」の文と手順書 7 章')
td = (REP / 'todo_user_1026.md').read_text(encoding='utf-8').splitlines()
print('todo 72 行:', snip(td[71], r'ただし ON の発進', 2, 70))
rb = (REP / 'runbook_20261103.md').read_text(encoding='utf-8').splitlines()
print('手順書 268 行:', snip(rb[267], r'空転\(比 > 1\.3・比が出ない\)・最初の周の記録の欠けも数える', 0, 40))

print()
print('== F. 題・表の頭の時刻')
print('todo 1 行:', td[0])
print('  同じ文書の 10/28 の行(27 行):', snip(td[26], r'もう一度 10/28 00:53:23', 0, 10))
print('手順書 347 行(9 章の表の頭):', rb[346])
print('  同じ表の 362 行:', snip(rb[361], r'10/28 00:53:23', 30, 20))
print('approvals 9 行(最初の「今」):', snip(ap[8], r'今\(10/27 05:3x\)', 0, 110))
print('approvals 12 行(今の「今」):', ap[11][:60])
