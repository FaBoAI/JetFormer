"""fix6 画面の係(反証 6 回目 w1-ui-ui61-1): ユーザーが見る文書の「画面の版 8」を、今の画面の版(app.js の DRIVE_CLIENT = 9)に
そろえる直し案。この係は文書を変えない(担当の外)。main session が中身を読んで入れる。

    python docs_ban9.py            # 直し案を今の文書に当てた差分を出すだけ(書かない)。どれか当たらなければ 1 で終わる
    python docs_ban9.py --check    # 今の文書だけを見る: 今の版と違う「画面の版 N」「版 N 以下」が残っていれば 1
    python docs_ban9.py --apply    # 書く(main session 用)。書く前に <名前>.bak-20261027-fix6-ui の控え(あれば作らない)

直し(old は文書の中でちょうど 1 回ある文。入れたあとに走らせると「済」で何もしない):
  手順書 3 章(86 行の 2 か所)・3.1 の表(125)・3.1 の下(133)・4 章 6(145)・7 章の古い画面の行(240)・付録の表(333 の 3 か所)、
  todo_user_1026.md(54・70・87)、README.md(22・203・230・245)、fix3/approvals.md(44)。
  手順書の頭の「直しの記録」の行(- **10/2… で始まる行)は前の直しの記録なので変えない(--check も見ない)。
"""
import argparse
import difflib
import re
import shutil
import sys
from pathlib import Path

REPORTS = Path('/home/jetson/jetformer_reports/fix_20261026')
JF = Path('/home/jetson/jetformer')
APP = JF / 'jetformer' / 'web' / 'static' / 'app.js'
SERVER = JF / 'jetformer' / 'web' / 'server.py'
BAK = '.bak-20261027-fix6-ui'

EDITS = [
    # 手順書 3 章: スマホの画面を読み直す(86 行)
    ('runbook_20261103.md', 'サーバーは、読み直していない画面(版 7 以下)と、',
     'サーバーは、読み直していない画面(版 8 以下)と、'),
    ('runbook_20261103.md', '「画面の版 8(app.js MM/DD HH:MM)」の文があるのを見る**',
     '「画面の版 9(app.js MM/DD HH:MM)」の文があるのを見る**'),
    # 3.1 の表(125 行)と表の下(133 行)
    ('runbook_20261103.md', '| 「画面の版 8(app.js MM/DD HH:MM)」 | 無い → もう一度読み直す |',
     '| 「画面の版 9(app.js MM/DD HH:MM)」 | 無い → もう一度読み直す |'),
    ('runbook_20261103.md', '**読み直していない画面(版 7 以下)と、',
     '**読み直していない画面(版 8 以下)と、'),
    # 4 章 6(145 行)
    ('runbook_20261103.md', '読み直した画面(画面の版 8)では', '読み直した画面(画面の版 9)では'),
    # 7 章の「管理画面が古いままです」の行(240 行)
    ('runbook_20261103.md', '▶ の横の「画面の版 8(app.js …)」を見る(3.1)', '▶ の横の「画面の版 9(app.js …)」を見る(3.1)'),
    # 付録の表の DRIVE_CLIENT_VERSION の行(333 行)
    ('runbook_20261103.md', '(`DRIVE_CLIENT_VERSION`。state-5 で 7、10/27 r2-r2-state-3 で 8 と app.js の更新時刻)',
     '(`DRIVE_CLIENT_VERSION`。state-5 で 7、10/27 r2-r2-state-3 で 8 と app.js の更新時刻、10/27 fix5 で 9 と起動の時刻 boot_at)'),
    ('runbook_20261103.md', '(版 7 以下と、画面を開いたあとに app.js を直した画面を断る)',
     '(版 8 以下と、画面を開いたあとに app.js を直した画面と、画面が読んだあとにサーバーが起き直した画面を断る)'),
    ('runbook_20261103.md', '「画面の版 8」の印で守る', '「画面の版 9」の印で守る'),
    # todo_user_1026.md(54・70・87 行)
    ('todo_user_1026.md', '「画面の版 8(app.js MM/DD HH:MM)」が出る(10/27 直す 2 回目 r2-r2-state-3。',
     '「画面の版 9(app.js MM/DD HH:MM)」が出る(10/27 直す 2 回目 r2-r2-state-3 で版 8、10/27 fix5 で版 9。'),
    ('todo_user_1026.md', '古い画面を断る版 8 と app.js の時刻', '古い画面を断る版 9 と app.js の時刻・起動の時刻'),
    ('todo_user_1026.md',
     '「画面の版 8(app.js …)」が出ることを見る。再起動のあとのサーバーは、版 7 以下の画面と、開いたあとに app.js を直した画面からの'
     '走行開始を断る(409。読み直せば走れる。10/27 r2-r2-state-3)。',
     '「画面の版 9(app.js …)」が出ることを見る。再起動のあとのサーバーは、版 8 以下の画面と、開いたあとに app.js を直した画面からの'
     '走行開始を断る(409「管理画面が古いままです…」。読み直せば走れる。10/27 r2-r2-state-3・fix5)。'),
    # README.md(22・203・230・245 行)
    ('README.md', '直した(画面の版 8)。', '直した(画面の版 8。10/27 fix5 で版 9)。'),
    ('README.md', '「画面の版 8」・起動し直しの文が読める', '「画面の版 9」・起動し直しの文が読める'),
    ('README.md', '(±0.01・版 8・日付の注意・起動し直しの知らせ)', '(±0.01・版 9・日付の注意・起動し直しの知らせ)'),
    ('README.md', '**直した**(版 8 と app.js の時刻。再起動のあとに効く)',
     '**直した**(版 8 と app.js の時刻。10/27 fix5 で版 9 と起動の時刻。再起動のあとに効く)'),
    # fix3/approvals.md(44 行。再起動の了承の「何を」)
    ('fix3/approvals.md', '古い画面を断る(版 8 と app.js の時刻)', '古い画面を断る(版 9 と app.js の時刻・起動の時刻)'),
]
FILES = sorted({f for f, _, _ in EDITS})
# 前の版の数を書いてよい所(記録・歴史の文。--check で見ない)
HISTORY = ('直した(画面の版 8。10/27 fix5 で版 9)', '版 8 と app.js の時刻。10/27 fix5 で版 9', 'r2-r2-state-3 で版 8、10/27 fix5 で版 9',
           'state-5 で 7、10/27 r2-r2-state-3 で 8')


def versions():
    """(app.js の DRIVE_CLIENT, app.js の走行開始の body の client, server.py の DRIVE_CLIENT_VERSION)"""
    js = APP.read_text(encoding='utf-8')
    py = SERVER.read_text(encoding='utf-8')
    a = int(re.search(r'^const DRIVE_CLIENT = (\d+);', js, re.M).group(1))
    b = int(re.search(r'const body = \{ client: (\d+),', js).group(1))
    c = int(re.search(r'^DRIVE_CLIENT_VERSION = (\d+)', py, re.M).group(1))
    return a, b, c


def apply_text(texts, edits=EDITS):
    """texts: {文書の名前: 中身}。戻り値 (新しい中身, [(文書, old の先頭, 'apply'|'done'|'missing'|'many')])"""
    out = dict(texts)
    rep = []
    for f, old, new in edits:
        t = out[f]
        n_old, n_new = t.count(old), t.count(new)
        if n_old == 1:
            out[f] = t.replace(old, new)
            rep.append((f, old, 'apply'))
        elif n_old == 0 and n_new >= 1:
            rep.append((f, old, 'done'))
        else:
            rep.append((f, old, 'missing' if n_old == 0 else 'many'))
    return out, rep


STALE = re.compile(r'画面の版 (\d+)|版 (\d+) 以下|版 (\d+) と app\.js|・版 (\d+)・')


def stale(texts, cur):
    """今の版 cur と違う版の数が、ユーザーが見る文に残っている所 [(文書, 行番号, 見つけた文)]。「版 N 以下」は cur - 1 が正しい"""
    bad = []
    for f, t in texts.items():
        for i, line in enumerate(t.splitlines(), 1):
            if f == 'runbook_20261103.md' and line.startswith('- **10/'):
                continue                                         # 手順書の頭の直しの記録
            for h in HISTORY:                                     # 記録の文だけを外す(同じ行のほかの所は見る)
                line = line.replace(h, '')
            for m in STALE.finditer(line):
                n = int(next(g for g in m.groups() if g is not None))
                want = cur - 1 if m.group(2) is not None else cur
                if n != want:
                    bad.append((f, i, m.group(0)))
    return bad


def read_all(root=REPORTS):
    return {f: (root / f).read_text(encoding='utf-8') for f in FILES}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args(argv)
    v = versions()
    print('画面の版: app.js DRIVE_CLIENT %d・走行開始の client %d・server.py DRIVE_CLIENT_VERSION %d' % v)
    if len(set(v)) != 1:
        print('版の数がそろっていません(先に app.js・server.py を見る)')
        return 1
    cur = v[0]
    texts = read_all()
    if a.check:
        bad = stale(texts, cur)
        for f, i, s in bad:
            print('残り: %s:%d「%s」' % (f, i, s))
        print('残り %d か所' % len(bad))
        return 1 if bad else 0
    new, rep = apply_text(texts)
    for f, old, st in rep:
        print('%-7s %s「%s…」' % ({'apply': '入れる', 'done': '済', 'missing': '無い', 'many': '2回以上'}[st], f, old[:40]))
    for f in FILES:
        if new[f] != texts[f]:
            sys.stdout.writelines(difflib.unified_diff(texts[f].splitlines(True), new[f].splitlines(True), f, f + '(直し案)', n=0))
    bad = stale(new, cur)
    for f, i, s in bad:
        print('直し案を入れても残る: %s:%d「%s」' % (f, i, s))
    ok = all(st in ('apply', 'done') for _, _, st in rep) and not bad
    if a.apply and ok:
        for f in FILES:
            if new[f] != texts[f]:
                p = REPORTS / f
                bak = p.with_name(p.name + BAK)
                if not bak.exists():
                    shutil.copy2(p, bak)
                p.write_text(new[f], encoding='utf-8')
                print('書いた: %s(控え %s)' % (p, bak.name))
    elif a.apply:
        print('当たらない直しか残りがあるので書きません')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
