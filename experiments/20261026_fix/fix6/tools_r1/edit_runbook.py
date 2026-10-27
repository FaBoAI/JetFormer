"""直す 6-1 回目・係 tools: 手順書 runbook_20261103.md の 7 章「いつでも」の行・8 章の頭・8.1 の ⑤ の行に、AI 側にしたシーンごと
書けなかった本(w1-tools-r6-1-tools-1)の読み方を足す。書く直前に、今の中身が直す前の控え(before/runbook_20261103.md.orig)と
同じか(ほかの係が同時に直していないか)を照らし、控え <手順書>.bak-fix6-tools を作ってから書く。

    python3 -I edit_runbook.py            # 差分を出すだけ
    python3 -I edit_runbook.py --apply    # 書く
"""
import difflib
import hashlib
import shutil
import sys
from pathlib import Path

RB = Path('/home/jetson/jetformer_reports/fix_20261026/runbook_20261103.md')
BEFORE = Path(__file__).resolve().parent / 'before' / 'runbook_20261103.md.orig'
BAK = RB.with_name(RB.name + '.bak-fix6-tools')

EDITS = [
    # 7 章「いつでも」の行
    ('⑤ は「記録が欠けている本 …」の 1 行(10/27 fix5 c1。前は「判定: 合格」「衝突 0 回」と出て、`--latest` も止まらなかった)。',
     '⑤ は「記録が欠けている本 …」の 1 行(10/27 fix5 c1。前は「判定: 合格」「衝突 0 回」と出て、`--latest` も止まらなかった)。'
     '**AI 側にしたシーンごと書けなかった本**(人側で置いて待った 1 つ目のシーンだけが残り、AI 側にしたあとのシーンの npz が無い。'
     'AI 側にしたのが 20 秒より後の本で起きる)は、⑤ が「skip … AI 側のフレームが無い(シーン …)」の行と「読めない記録」'
     '(events.jsonl の発進の前進の start があれば ON)に出し、ON なら「満たさない」。2 回目の発進のシーンごと落ちた本も、欠けた'
     'シーンの間の start で同じ(10/27 fix6。前は ⑤ だけが黙って比べから外し、最初の周に当たった ON の本が消えて「使ってよい」と出た)。'),
    # 8 章の頭
    ('シーンが欠けて最初の周の間の記録が途切れた ON の発進も「最初の周の記録が欠けた発進(on)」で同じ(10/27 r2-rec2-5)。',
     'シーンが欠けて最初の周の間の記録が途切れた ON の発進も「最初の周の記録が欠けた発進(on)」で同じ(10/27 r2-rec2-5)。'
     '**AI 側にしたシーンごと書けなかった本**(AI のフレームが 1 つも無い本。欠けたシーンにフレームが無く、その間は人側・AI 側が'
     '分からない)も「skip … AI 側のフレームが無い」と「読めない記録」で同じ。読めた本でも、**欠けたシーンの間に発進の前進の start '
     'がある**(2 回目の発進のシーンごと落ちた)なら、その発進を「読めない記録」(ON)に入れる(`--first-only` で、読めた発進のあとの '
     'start だけなら入れない)。ON/OFF は log.json・計画に、events.jsonl の start(出来事はすぐ書くので落ちても残る)も足す。'
     '欠けたシーンにもフレームがある本(VEHICLE だけ無いなど)は、フレームの select で人側・AI 側が分かるので前と同じ。'
     '発進の前進 OFF の本の、欠けたシーンの発進は記録に手がかりが無い → 動画・紙で(10/27 fix6)。'),
    # 8.1 の ⑤ の行の終わり
    ('「記録が欠けている本 …」= シーンの npz が無い・読めない本(欠けた区間の衝突は記録に無い → 7 章「いつでも」。最初の周に掛かれば'
     '「最初の周の記録が欠けた発進」。10/27 fix5 c1)',
     '「記録が欠けている本 …」= シーンの npz が無い・読めない本(欠けた区間の衝突は記録に無い → 7 章「いつでも」。読めた発進の最初の周に'
     '掛かれば「最初の周の記録が欠けた発進」、AI 側にしたシーンごと欠けた発進(AI のフレームが無い本・欠けたシーンの間の発進の前進の '
     'start)は「skip … AI 側のフレームが無い」・「読めない記録」で、どちらも ON の本なら満たさない。行は AI のフレームが無い本にも出る'
     '(「どこも AI 側にしていない = 発進なし」なら比べに入らない)。発進の前進 OFF の本の、欠けたシーンの発進は数えられない'
     '(動画・紙で)。10/27 fix5 c1・fix6)'),
]


def main():
    apply = '--apply' in sys.argv
    cur = RB.read_text(encoding='utf-8')
    if hashlib.sha256(cur.encode('utf-8')).hexdigest() != hashlib.sha256(BEFORE.read_bytes()).hexdigest():
        print('手順書が直す前の控えと違う(ほかの係が直した?)。書かない')
        return 1
    new = cur
    for old, rep in EDITS:
        n = new.count(old)
        if n != 1:
            print('当たらない(%d 回): %s' % (n, old[:60]))
            return 1
        new = new.replace(old, rep)
    sys.stdout.writelines(difflib.unified_diff(cur.splitlines(True), new.splitlines(True), 'before', 'after', n=0))
    if apply:
        if not BAK.exists():
            shutil.copy2(RB, BAK)
        if RB.read_text(encoding='utf-8') != cur:      # 書く直前にもう一度
            print('書く直前に手順書が変わった。書かない')
            return 1
        RB.write_text(new, encoding='utf-8')
        print('書いた', RB, hashlib.sha256(new.encode('utf-8')).hexdigest())
    return 0


if __name__ == '__main__':
    sys.exit(main())
