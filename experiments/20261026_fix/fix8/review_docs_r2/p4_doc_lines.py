"""反証 8-2 docs: 指摘の文書の行を、今の文書(sha256 の頭つき)から機械で抜き出す(読むだけ)。

  A  approvals: 0 章の 2 の行・いちばん上の「今(10/27 17:3x)」の行・2 の表の「今(10/28 08:40)」の行(済んだ再起動の文と 08:40 の印)
  B  手順書・todo: 紙に「試し」→ main session の `--skip`(7 章「走った値が計画と違った」「起こし直した」・todo 3 章)と、
     8.1 ⑤ の「当たった本は `--skip` しない」・7 章の「試しの ON の … 数える」
  C  todo の 1 章「11/2 まで」の行・2 章 10 の行に、approvals 3 の 10/28 知らせ(10/23 の形)があるか(README 6.2 の 33・approvals 0 章の 3 と比べる)
  D  手順書 3 章「最初のリカバリー」の行の OFF の条件と、新しい「10/23 の形の後ろの近い物」の行・7 章の行の「OFF にしない」の文

    python3 -I -B p4_doc_lines.py → out/p4_doc_lines.txt
"""
import hashlib
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REP = Path('/home/jetson/jetformer_reports/fix_20261026')


def L(name):
    return (REP / name).read_text(encoding='utf-8').splitlines()


def snip(s, pat, b=60, a=160):
    m = re.search(pat, s)
    return ('…' + s[max(0, m.start() - b):m.end() + a] + '…') if m else '(無い: %s)' % pat


def main():
    out = []
    for n in ('runbook_20261103.md', 'todo_user_1026.md', 'README.md', 'fix3/approvals.md', 'fix7/README.md'):
        out.append('# %s sha256 %s' % (n, hashlib.sha256((REP / n).read_bytes()).hexdigest()[:12]))
    ap = L('fix3/approvals.md')
    out.append('\n== A approvals の済んだ再起動の文')
    for i, l in enumerate(ap, 1):
        if l.startswith('| 2 |') and 'jetformer-web の再起動の日' in l:
            out.append('approvals.md:%d(0 章の 2): %s' % (i, snip(l, r'10/28 02:33 の drive.py の説明の文', 0, 200)))
            out.append('   08:40 の印がこの行にある: %s' % ('08:40' in l))
        if l.startswith('- **今(10/27 17:3x'):
            out.append('approvals.md:%d(「今」の行): %s' % (i, snip(l, r'\*\*2 は済み\*\*', 0, 120)))
            out.append('   00:53・08:40 の印がこの行にある: %s・%s' % ('00:53' in l, '08:40' in l))
        if l.startswith('- 前の今'):
            out.append('approvals.md:%d: %s' % (i, l[:110]))
        if l.startswith('- **今(10/28 03:3x'):
            out.append('approvals.md:%d: %s' % (i, l[:60]))
        if l.startswith('| 今(10/28 08:40)'):
            out.append('approvals.md:%d(2 の表): %s' % (i, snip(l, r'上の行・0 章の 2', 0, 90)))
    st = subprocess.run(['systemctl', 'show', '-p', 'ActiveEnterTimestamp', '--value', 'jetformer-web'], capture_output=True, text=True).stdout.strip()
    out.append('systemctl show ActiveEnterTimestamp(読むだけ): %s' % st)
    rb = L('runbook_20261103.md')
    td = L('todo_user_1026.md')
    out.append('\n== B 紙の「試し」→ `--skip` と、試しの ON の不具合を数える決め')
    for i, l in enumerate(rb, 1):
        if l.startswith('| 走った値が計画と違った |'):
            out.append('runbook:%d(7 章「走った値が計画と違った」): %s' % (i, snip(l, r'ただし試しの ON の最初の周の衝突', 0, 160)))
            out.append('   同じ行: %s' % snip(l, r'⑤ の行に名前が無ければ', 0, 60))
        if l.startswith('| **jetformer-web か Jetson が起こし直した**'):
            out.append('runbook:%d(7 章「起こし直した」の終わり): %s' % (i, snip(l, r'それもできなければ紙に', 0, 120)))
        if l.startswith('| ⑤ | **回 3 は `--ignore-gain` で読む**'):
            out.append('runbook:%d(8.1 ⑤): %s' % (i, snip(l, r'当たった本は `--skip` しない', 120, 20)))
    for i, l in enumerate(td, 1):
        if 'できなければ紙' in l:
            out.append('todo:%d(3 章): %s' % (i, snip(l, r'できなければ紙', 60, 40)))
    c = (Path('/home/jetson/jetformer/experiments/20261026_fix/fix8/review_docs_r2/snap/collect.py')).read_text(encoding='utf-8')
    out.append('collect.py(写し)の --skip の説明: %s' % snip(c, r"ap\.add_argument\('--skip'", 0, 260).replace('\n', ' '))
    out.append('\n== C todo に approvals 3 の 10/28 知らせ(10/23 の形)があるか')
    for i, l in enumerate(td, 1):
        if l.startswith('| **11/2 まで**') or l.startswith('| 10 | ~~新しい安全のスイッチ'):
            out.append('todo:%d: 「10/23」%s・「知らせ」の中に 3 の知らせ %s・「困ること」%s | %s' % (
                i, '10/23' in l, bool(re.search(r'3[^0-9]{0,20}知らせ|知らせ[^|]{0,30}10/23', l)), '困ること' in l, l[:90]))
    rd = L('README.md')
    for i, l in enumerate(rd, 1):
        if l.startswith('| 33 |'):
            out.append('README:%d(6.2 の 33): %s' % (i, snip(l, r'\| main session → ユーザー', 40, 60)))
    for i, l in enumerate(ap, 1):
        if l.startswith('| 3 | 新しい安全のスイッチ'):
            out.append('approvals:%d(0 章の 3): %s' % (i, snip(l, r'10/28 知らせ', 0, 120)))
    out.append('\n== D 3 章「最初のリカバリー」の OFF の条件と、10/23 の形の行')
    for i, l in enumerate(rb, 1):
        if l.startswith('  - 見張り・残すが ON の日(11/3)は、**その日の最初のリカバリーを人が見る**'):
            out.append('runbook:%d(3 章 最初のリカバリー): %s' % (i, snip(l, r'1 m 以上下がる・後ろの物に近づく・おかしな動き', 0, 40)))
            out.append('   同じ行: %s' % snip(l, r'すぐ CH3 を人側 → 止めて OFF', 0, 40))
        if l.startswith('  - **見張り ON の日の、10/23 の形の後ろの近い物**'):
            out.append('runbook:%d(3 章 10/23 の形): %s' % (i, snip(l, r'約 1 m より遠い物には ON のほうが遅く当たるか当たらない', 0, 0)))
            out.append('   同じ行: %s' % snip(l, r'先に人側にしただけなら', 0, 80))
        if l.startswith('| リカバリーで車が 1 m 以上下がる・後ろの物に近づく |'):
            out.append('runbook:%d(7 章): %s' % (i, snip(l, r'先に人側にしただけなら', 0, 80)))
    for i, l in enumerate(td, 1):
        if '**その日の最初のリカバリーを見る**' in l:
            out.append('todo:%d(3 章): %s' % (i, snip(l, r'\*\*その日の最初のリカバリーを見る\*\*', 0, 70)))
    for i, l in enumerate(ap, 1):
        if l.startswith('| なぜ | ① 10/23 No.6'):
            out.append('approvals:%d(3 の なぜ): %s' % (i, snip(l, r'見張り ON の再生では', 0, 60)))
    txt = '\n'.join(out) + '\n'
    (HERE / 'out' / 'p4_doc_lines.txt').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
