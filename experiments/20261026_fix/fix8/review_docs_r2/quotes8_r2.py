"""反証 8-2・docs: ユーザーが読む文書の「…」の引用が、コード(画面・サーバー・道具)の文字列にあるかを照らす(読むだけ)。

fix8/review_docs_r1/quotes8.py の写し。変えたこと:
- 比べの元の文書: 8-1 の直しの前の控え(.bak-20261028-fix8-docs-r1)。今の文書で外れ、控えの文書では外れていなかった引用だけを出す
  (= 8-1 の文書の直しと同期 1 で増えた外れ)。
- コード: 今のファイル(app.js 10/28 04:54・collect.py・runs_1103.py・style.css など)。B は git の HEAD(9da6d0d)にあり今は無い切れ端。
- コードに tools/section_scorecard.py・tools/disk 系を足した。
引数: なし。読むだけ。
"""
import re
import subprocess
from pathlib import Path

HOME = Path('/home/jetson')
JF = HOME / 'jetformer'
REP = HOME / 'jetformer_reports/fix_20261026'
DOC_NAMES = ['runbook_20261103.md', 'todo_user_1026.md', 'README.md', 'fix3/approvals.md', 'fix7/README.md']
BASE_SUF = '.bak-20261028-fix8-docs-r1'
CODE_REL = ['jetformer/web/static/app.js', 'jetformer/web/static/index.html', 'jetformer/web/server.py', 'jetformer/web/bench.py',
            'jetformer/drive.py', 'jetformer/assist.py',
            'tools/car_check.py', 'tools/standing_3lap.py', 'tools/quick_eval_by_params.py', 'tools/section_scorecard.py',
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


PH = re.compile(r'…|\.\.\.|HH:MM:SS|HH:MM|MM/DD|\bN\b|\bX\b|\bY\b|\bZ\b|\bM\b|\bT\b|0\.xx|%s|\s+')


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
    print('== A) 8-1 の直し・同期 1 で増えた外れ(今の文書 × 今のコードで無く、8-1 の前の控えの文書 × 今のコードでは無かった引用)')
    na = 0
    for name in DOC_NAMES:
        cur = (REP / name).read_text(encoding='utf-8')
        old = (REP / (name + BASE_SUF)).read_text(encoding='utf-8')
        m_now = misses(cur, now)
        old_q = {q for (q, _m) in misses(old, now)}
        old_all = set(re.findall(r'「([^「」]{4,400})」', old))
        for (q, miss), lns in m_now.items():
            if q in old_q or q in old_all:
                continue
            na += 1
            print('%s:%s 「%s」 無い切れ端: %s' % (name, ','.join(map(str, lns)), q[:200], ' | '.join(miss)[:240]))
    print('件数 A', na)
    print()
    print('== B) HEAD のコードにはあったが今のコードには無い切れ端を引く引用(今の文書の全部)')
    nb = 0
    for name in DOC_NAMES:
        cur = (REP / name).read_text(encoding='utf-8')
        for ln, line in enumerate(cur.splitlines(), 1):
            for m in re.finditer(r'「([^「」]{4,400})」', line):
                q = m.group(1)
                pieces = [x for x in PH.split(q) if len(x) >= 4]
                gone = [x for x in pieces if x in head and x not in now]
                if gone:
                    nb += 1
                    print('%s:%d 「%s」 HEAD にあり今は無い: %s' % (name, ln, q[:200], ' | '.join(gone)[:240]))
    print('件数 B', nb)


if __name__ == '__main__':
    main()
