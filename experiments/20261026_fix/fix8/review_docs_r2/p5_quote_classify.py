"""反証 8-2 docs: quotes8_r2.py の A の外れ(コードに無い切れ端)を分ける(読むだけ): 文書の中の行・章の名前(5 本のどれかに「」の外でもある)か、
コードが組み立てる文(画面の版・409 の時刻・python の can't open file・echo の文)か、どちらでもないか。

    python3 -I -B p5_quote_classify.py → out/p5_quote_classify.txt
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REP = Path('/home/jetson/jetformer_reports/fix_20261026')
DOCS = ['runbook_20261103.md', 'todo_user_1026.md', 'README.md', 'fix3/approvals.md', 'fix7/README.md']
CODE_BUILT = {
    '画面の版 9(app.js 10/28 04:54)': 'app.js 1494 `画面の版 ${DRIVE_CLIENT}(app.js MM/DD HH:MM)` + app.js の更新 10/28 04:54:46',
    '管理画面が古いままです(この画面の app.js は …、今のファイルは 10/28 04:54 に直したもの)': 'server.py 1054 の 409 + _hm(%m/%d %H:%M)',
    "python3: can't open file '/home/jetson/tools/fw_vbat_check.py'": 'python3 の文(fix8/fix_docs_r1 4 章で流した)',
    "can't open file": '同じ',
    'エラーの行なし': '手順書 3.1・todo 4 章のコマンドの echo',
    '10/28 08:40': '起動し直しの記録の時刻(文書の行の名前)',
}


def main():
    txt = (HERE / 'out' / 'quotes8_r2.txt').read_text(encoding='utf-8')
    alld = {n: (REP / n).read_text(encoding='utf-8') for n in DOCS}
    out = []
    n_doc = n_code = n_other = 0
    for line in txt.splitlines():
        m = re.match(r'(\S+?):([\d,]+) 「(.*)」 無い切れ端: (.*)$', line)
        if not m:
            continue
        name, lns, q, miss = m.groups()
        if q in CODE_BUILT:
            n_code += 1
            out.append('コードが組み立てる文: %s:%s 「%s」 = %s' % (name, lns, q, CODE_BUILT[q]))
            continue
        # 「」の外に同じ文があるか(行・章の名前・前に書いた文の引用の元)
        where = []
        for n, t in alld.items():
            s = re.sub(r'「[^「」]*」', '', t)
            if q.replace('**', '') in s.replace('**', ''):
                where.append(n)
        bak_hit = []
        if not where:
            for p in sorted(REP.glob('*.bak-*')) + sorted((REP / 'fix3').glob('approvals.md.bak-*')) + sorted((REP / 'fix7').glob('README.md.bak-*')):
                if q in p.read_text(encoding='utf-8', errors='replace'):
                    bak_hit.append(p.name)
        if where or bak_hit:
            n_doc += 1
            out.append('文書の中の名前・前の文: %s:%s 「%s」 → %s' % (name, lns, q[:80], (', '.join(where) or '控え ' + bak_hit[0])))
        else:
            n_other += 1
            out.append('どちらでもない: %s:%s 「%s」(無い切れ端 %s)' % (name, lns, q[:100], miss[:120]))
    out.append('数: 文書の中 %d・コードが組み立てる文 %d・どちらでもない %d' % (n_doc, n_code, n_other))
    t = '\n'.join(out) + '\n'
    (HERE / 'out' / 'p5_quote_classify.txt').write_text(t, encoding='utf-8')
    print(t)


if __name__ == '__main__':
    main()
