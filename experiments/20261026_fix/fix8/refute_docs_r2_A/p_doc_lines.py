"""打ち消しの票 A: 文書の行を自分で読む(読むだけ)。
  A) 手順書 3 章の「最初のリカバリー」の行・「10/23 の形の後ろの近い物」の行・7 章の「1 m 以上下がる」の行の言葉
  B) 手順書 7 章「走った値が計画と違った」「起こし直した」・6 章・todo 3 章・8.1 ⑤ の --skip の言葉(試しの紙の道と「当たった本は --skip しない」)
  C) todo 1 章「11/2 まで」・2 章 10 の行に approvals 3 の 10/28 知らせがあるか(14 の知らせは同じ行にあるか)
  D) approvals の 0 章の 2・いちばん上の「今(10/27 17:3x)」の行・ほかの文書の「反証 8 回目のあと」の再起動の文に 08:40 の印があるか
    python -I -B p_doc_lines.py → 標準出力・out/p_doc_lines.txt
"""
import hashlib
import re
from pathlib import Path

R = Path('/home/jetson/jetformer_reports/fix_20261026')
DOCS = {'runbook': R / 'runbook_20261103.md', 'todo': R / 'todo_user_1026.md', 'readme': R / 'README.md',
        'approvals': R / 'fix3' / 'approvals.md', 'fix7readme': R / 'fix7' / 'README.md'}
TXT = {k: p.read_text(encoding='utf-8').splitlines() for k, p in DOCS.items()}


def line_with(doc, *keys):
    hits = [(i + 1, s) for i, s in enumerate(TXT[doc]) if all(k in s for k in keys)]
    return hits


def show(tag, doc, n, s, words):
    print('%s %s:%d  %s' % (tag, doc, n, '  '.join('%s=%s' % (w, w in s) for w in words)))


def main():
    for k, p in DOCS.items():
        print('sha', k, hashlib.sha256(p.read_bytes()).hexdigest()[:12])
    print('\n== A 最初のリカバリーの行')
    for n, s in line_with('runbook', 'その日の最初のリカバリーを人が見る'):
        m = re.search(r'1 m 以上下がる.{0,40}', s)
        print('runbook:%d  %s' % (n, m.group(0) if m else None))
        m = re.search(r'すぐ CH3 を人側 → [^(]{0,40}', s)
        print('   する: %s' % (m.group(0) if m else None))
        print('   「10/23 の形」の言葉がこの行にあるか', '10/23 の形' in s, '/ 見張り ON でふつうに下がる距離の数があるか',
              bool(re.search(r'ON[^。]{0,30}(1\.\d+|0\.\d+) m 下が', s)))
    for n, s in line_with('runbook', '10/23 の形の後ろの近い物', '先に人側にしただけなら'):
        m = re.search(r'先に人側にしただけなら[^。]*', s)
        print('runbook:%d  %s' % (n, m.group(0)))
    for n, s in line_with('todo', 'その日の最初のリカバリーを見る'):
        m = re.search(r'その日の最初のリカバリーを見る\*\*\(([^)]*)', s)
        print('todo:%d  %s' % (n, m.group(1)[:80] if m else None))
    for n, s in line_with('approvals', '見張り ON の再生では'):
        m = re.search(r'見張り ON の再生では[^)]*\)', s)
        print('approvals:%d  %s' % (n, m.group(0)))
    print('\n== B --skip と試し')
    for doc in ('runbook', 'todo'):
        for n, s in enumerate(TXT[doc], 1):
            for m in re.finditer(r'--skip', s):
                ctx = s[max(0, m.start() - 70):m.end() + 40]
                if '試し' in ctx or '当たった本' in ctx:
                    print('%s:%d  …%s…' % (doc, n, ctx))
    print('\n== C todo の答えの一覧')
    for n, s in line_with('todo', '**11/2 まで**', '4・7〜11・13'):
        show('C1', 'todo', n, s, ['10/23', '困ること', '3 の知らせ', '10/28 知らせ', '変えたい', '14(10/27 fix7 で残した安全の形の知らせ)'])
    for n, s in line_with('todo', '| 10 |', '両方 ON'):
        show('C2', 'todo', n, s, ['10/23', '困ること', '知らせ', '変えたい'])
    for n, s in line_with('todo', '| 16 |'):
        show('C3', 'todo', n, s, ['知らせ', '返事が無ければこのまま'])
    for n, s in line_with('todo', '10/23 の形'):
        show('C4', 'todo', n, s, ['人側にしてよい', '変えたい', '(c) のまま', 'approvals.md` 3'])
    for n, s in line_with('approvals', '| 3 | 新しい安全のスイッチ'):
        show('C5', 'approvals', n, s, ['10/28 知らせ', '返事が無ければ (c) のまま'])
    for n, s in line_with('readme', '| 33 |'):
        show('C6', 'readme', n, s[:400], ['変えたい', '11/2', '返事が無ければ (c) のまま'])
    print('\n== D 再起動の文')
    for n, s in line_with('approvals', '| 2 | jetformer-web の再起動の日'):
        show('D1', 'approvals', n, s, ['08:40', '反証 8 回目のあとにもう一度', '説明の文のぶんは反証 8 回目のあと'])
    for n, s in line_with('approvals', '今(10/27 17:3x'):
        show('D2', 'approvals', n, s, ['前の今', 'もう一度の再起動が要る', '00:53', '08:40'])
    for n, s in line_with('approvals', '今(10/28 03:3x'):
        show('D3', 'approvals', n, s, ['08:40', '要らなくなった'])
    for n, s in line_with('approvals', '| 今(10/28 08:40) |'):
        show('D4', 'approvals', n, s, ['0 章の 2', '要らなくなった'])
    for doc in DOCS:
        for n, s in enumerate(TXT[doc], 1):
            for m in re.finditer(r'反証 8 回目のあと(に|の)(main session が)?(もう一度|再起動)', s):
                tail = s[m.end():m.end() + 260]
                print('D5 %s:%d  08:40 の印が 260 字以内に=%s  …%s…' % (doc, n, '08:40' in tail or '要らなくなった' in tail,
                                                             s[max(0, m.start() - 30):m.end() + 30]))


if __name__ == '__main__':
    main()
