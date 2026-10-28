"""反証 8-1・docs: ユーザーが読む文書の「…」の引用が、コード(画面・サーバー・道具)の文字列にあるかを照らす(読むだけ)。

fix7/review_docs_r1/quotes.py の写しを広げた:
- 文書: 今の 7 本 + fix8 の報告。比べのため、fix8 docs-decided の前の控え(.bak-20261028-fix8-decided)も同じに照らす。
- コード: 今のファイルと、git の HEAD(9da6d0d = fix8 の前)の同じファイル。
出力:
  A) 今の文書 × 今のコードで無い切れ端のうち、控えの文書 × HEAD のコードでは無かったもの(= fix8 で増えた外れ)
  B) 今の文書の引用で、HEAD のコードにはあったが今のコードには無い切れ端(= fix8 tools の文の変えで古くなった引用)
引数: なし。読むだけ。
"""
import re
import subprocess
import sys
from pathlib import Path

HOME = Path('/home/jetson')
JF = HOME / 'jetformer'
REP = HOME / 'jetformer_reports/fix_20261026'
DOC_NAMES = ['runbook_20261103.md', 'todo_user_1026.md', 'README.md', 'fix3/approvals.md', 'fix3/disk.md',
             'fix6/README.md', 'fix7/README.md']
FIX8_REPORTS = ['fix8/decided_docs.md', 'fix8/decided_tools.md', 'fix8/decided_safety.md']
CODE_REL = ['jetformer/web/static/app.js', 'jetformer/web/server.py', 'jetformer/web/bench.py',
            'jetformer/drive.py', 'jetformer/assist.py',
            'tools/car_check.py', 'tools/standing_3lap.py', 'tools/quick_eval_by_params.py',
            'tools/fw_vbat_check.py', 'tools/bench_nightly.sh',
            'experiments/20261026_fix/diag_launch/collect.py', 'experiments/20261026_fix/plan/runs_1103.py',
            'experiments/20261026_fix/fix3/disk/disk_check.py',
            'experiments/20261024_tune/quick_eval.py']


def norm_code(t):
    t = re.sub(r"'\s*\n\s*'", '', t)
    t = re.sub(r'"\s*\n\s*"', '', t)
    t = re.sub(r"'\s*\+\s*\n?\s*'", '', t)
    t = re.sub(r"`\s*\+\s*\n?\s*'", '', t)
    t = re.sub(r"'\s*\+\s*\n?\s*`", '', t)
    return t


def corpus_now():
    c = ''
    for r in CODE_REL:
        p = JF / r
        if p.exists():
            c += norm_code(p.read_text(encoding='utf-8', errors='replace')) + '\n'
    return c


def corpus_head():
    c = ''
    for r in CODE_REL:
        try:
            t = subprocess.run(['git', '-C', str(JF), 'show', 'HEAD:' + r], capture_output=True, check=True).stdout
        except subprocess.CalledProcessError:
            continue
        c += norm_code(t.decode('utf-8', errors='replace')) + '\n'
    return c


PH = re.compile(r'…|\.\.\.|HH:MM:SS|HH:MM|MM/DD|\bN\b|\bX\b|\bY\b|\bZ\b|\bM\b|0\.xx|%s|\s+')


def misses(text, corpus):
    out = {}
    for ln, line in enumerate(text.splitlines(), 1):
        for m in re.finditer(r'「([^「」]{4,400})」', line):
            q = m.group(1)
            pieces = [x for x in PH.split(q) if len(x) >= 4]
            miss = tuple(x for x in pieces if x not in corpus)
            if miss:
                out.setdefault((q, miss), []).append(ln)
    return out


def main():
    now, head = corpus_now(), corpus_head()
    print('== A) fix8 で増えた外れ(今の文書 × 今のコード で無く、控えの文書 × HEAD のコード では無かった引用)')
    na = 0
    for name in DOC_NAMES:
        cur = (REP / name).read_text(encoding='utf-8')
        bak = REP / (name + '.bak-20261028-fix8-decided')
        old = bak.read_text(encoding='utf-8') if bak.exists() else ''
        m_now = misses(cur, now)
        m_old = misses(old, head)
        old_q = {q for (q, _m) in m_old}
        for (q, miss), lns in m_now.items():
            if q in old_q:
                continue
            na += 1
            print('%s:%s 「%s」 無い切れ端: %s' % (name, ','.join(map(str, lns)), q[:160], ' | '.join(miss)[:200]))
    print('件数 A', na)
    print()
    print('== B) HEAD のコードにはあったが今のコードには無い切れ端を引く引用(fix8 tools の文の変えで古くなった)')
    nb = 0
    for name in DOC_NAMES + FIX8_REPORTS:
        cur = (REP / name).read_text(encoding='utf-8')
        for ln, line in enumerate(cur.splitlines(), 1):
            for m in re.finditer(r'「([^「」]{4,400})」', line):
                q = m.group(1)
                pieces = [x for x in PH.split(q) if len(x) >= 4]
                gone = [x for x in pieces if x in head and x not in now]
                if gone:
                    nb += 1
                    print('%s:%d 「%s」 HEAD にあり今は無い: %s' % (name, ln, q[:160], ' | '.join(gone)[:200]))
    print('件数 B', nb)


if __name__ == '__main__':
    main()
