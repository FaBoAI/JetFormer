"""直す 6-2 回目・係 tools: 手順書 runbook_20261103.md の 7・8 章だけを直す(反証 6-2 回目 w2-tools-r62-1・r62-2)。

  python3 -I edit_runbook.py

書く前に、今の中身が控え(before/runbook_20261103.md.orig)と同じか(ほかの係が同時に直していないか)を照らす。違えば何もしない。
控えを ~/jetformer_reports/fix_20261026/runbook_20261103.md.bak-fix6-2-tools に置く。どの置き換えも 1 か所だけ(7・8 章の中)。
"""
import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RB = Path('/home/jetson/jetformer_reports/fix_20261026/runbook_20261103.md')
BAK = RB.with_name(RB.name + '.bak-fix6-2-tools')
ORIG = HERE / 'before' / 'runbook_20261103.md.orig'

REP = [
    # 7 章「左 4.2 で SW の出口・LH に 1 回当たる」
    ('| 左 4.2 で SW の出口・LH に 1 回当たる | 4.2 をやめて 4.1 に戻す |',
     '| 左 4.2 で SW の出口・LH に 1 回当たる(car_check の「衝突」の行の場所か、④ `by_params` 1 章の「最大 G・8G 以上・場所」。どちらも'
     '人側にした直後(AI の区間の終わりから 0.5 秒以内)の当たりも入る。④ は欄の後ろに「(うち人側にした直後 N: 場所 G)」。10/27 fix6-2。'
     '前は ④ が AI 側だけで、人が当たる直前に人側にした SW の出口の当たりが 0 に見えた) | 4.2 をやめて 4.1 に戻す |'),
    # 7 章「いつでも」: 電源断では events.jsonl の start も消える
    ('2 回目の発進のシーンごと落ちた本も、欠けたシーンの間の start で同じ(10/27 fix6。前は ⑤ だけが黙って比べから外し、最初の周に当たった'
     ' ON の本が消えて「使ってよい」と出た)。',
     '2 回目の発進のシーンごと落ちた本は、欠けたシーンの間に events.jsonl の start が残っていれば同じ(10/27 fix6。前は ⑤ だけが黙って'
     '比べから外し、最初の周に当たった ON の本が消えて「使ってよい」と出た)。**ただし電源断では events.jsonl の start も消える**(すぐには'
     'ディスクに書かれないので、落ちる前の約 30 秒に書いた行が残らない。本物の 10/10・10/23・10/24 の落ちた本 3 本はどれも電源断の形 = '
     '0 バイトの jpg・npz・scene.json。start が残るのは jetformer-web の落ちだけ)。そのため ⑤ は、**発進の前進 ON の本で最後の読める'
     'フレームのあとが欠けていれば**(落ちた形)、start が無くても「読めない記録」(ON。理由「シーン …(最後の読めるフレーム X 秒のあと)に'
     'フレームが無い…」)に入れて「満たさない」(10/27 fix6-2。前は start の行が消えると 2 回目の発進が黙って消え「使ってよい」と出た)。'
     '動画・紙で、欠けた区間に発進が無かった・当たっていないと確かめた本だけ main session が `--skip 記録名`(本ごと外れる。読めた発進も'
     '比べから外れる)。発進の前進 OFF の本の欠けた区間の発進は、記録に手がかりが無い → 動画・紙で。'),
    # 8 章の頭
    ('ON/OFF は log.json・計画に、events.jsonl の start(出来事はすぐ書くので落ちても残る)も足す。',
     'ON/OFF は log.json・計画に、events.jsonl の start も足す(jetformer-web の落ちでは残るが、**電源断では落ちる前の約 30 秒に書いた'
     '行が消える**。10/27 fix6-2)。そのため、**発進の前進 ON の本で最後の読めるフレームのあとが欠けた本**(電源断・落ちの形)は、start が'
     '無くても「読めない記録」(ON。理由「シーン …(最後の読めるフレーム X 秒のあと)にフレームが無い…」・「記録が欠けている本 …」の行の'
     '後ろにも 1 文)に入れ、「満たさない」(`--first-only` で、読めた発進のある本は入れない = 欠けた区間の発進は 2 回目以降。ON の手がかり'
     'は log.json・計画・events.jsonl の発進の前進の出来事(start・skip・end)・走行中に launch_boost を 1 にした、のどれか。AI のフレームが'
     '無い本の ON/OFF も同じ手がかりで決める。10/27 fix6-2)。真ん中だけ欠けた本(あとのシーンが読める)は前と同じ。'),
    # 8.1 ③
    ('空転がいちばんひどい発進・IMU が抜けた発進に「確かめる」の印(④ の最速・25 秒以下に入らない。10/27 r3-r3-rec-3・4) | `standing_3lap.md`',
     '空転がいちばんひどい発進・IMU が抜けた発進に「確かめる」の印(④ の最速・25 秒以下に入らない。10/27 r3-r3-rec-3・4)。2 章の本の'
     '衝突(強)と、1 章の「数えない(この本の前の衝突 N)」・発進の衝突(強)には、人側にした直後(AI の区間の終わりから 0.5 秒以内)の当たり'
     'も入る(car_check の「衝突」の行と同じ。印「(うち人側にした直後 N: 場所 G)」。10/27 fix6-2。走りながらの 3 周の窓は AI 側だけ) | '
     '`standing_3lap.md`'),
    # 8.1 ④
    ('で分けた周・衝突(最大 G・8G 以上・場所)・3 周の窓・止まってから 3 周・ぶつからずに 25 秒以下。',
     'で分けた周・衝突(最大 G・8G 以上・場所)・3 周の窓・止まってから 3 周・ぶつからずに 25 秒以下。**衝突(1 章の衝突(強)・最大 G・'
     '8G 以上・場所、2 章の衝突/通過・強い衝突)には人側にした直後(AI の区間の終わりから 0.5 秒以内)の当たりも入る**(car_check の「衝突」'
     'の行と同じ。1 章の欄の後ろに「(うち人側にした直後 N: 場所 G)」、2 章の下に入った組の行。① quick_eval(AI 側だけ)とはその分だけ'
     '違う。10/27 fix6-2。前は AI 側だけで、6 章の「左 4.2 … SW の出口・LH の当たり 0」が car_check と食い違った)。'),
    # 8.1 ⑤ の読み方
    ('発進の前進 OFF の本の、欠けたシーンの発進は数えられない(動画・紙で)。10/27 fix5 c1・fix6) |',
     '発進の前進 ON の本で最後の読めるフレームのあとが欠けた本(電源断で events.jsonl の start も消えた形)も「読めない記録」(ON)で満たさない'
     '(行の後ろに「最後の読めるフレーム X 秒のあとのシーン … にフレームが無い(落ちた形)…」。`--first-only` で読めた発進のある本は入れない。'
     '10/27 fix6-2)。発進の前進 OFF の本の、欠けたシーンの発進は数えられない(動画・紙で)。10/27 fix5 c1・fix6・fix6-2) |'),
]


def main():
    t = RB.read_text(encoding='utf-8')
    if hashlib.sha256(t.encode('utf-8')).hexdigest() != hashlib.sha256(ORIG.read_bytes()).hexdigest():
        print('手順書が控えと違う(ほかの係が直した?)。書かない', file=sys.stderr)
        return 1
    ch7 = t.index('## 7. 止める条件')
    ch9 = t.index('## 9. まだ無いもの')
    for a, b in REP:
        if t.count(a) != 1:
            print('1 か所でない: %s' % a[:60], file=sys.stderr)
            return 1
        i = t.index(a)
        if not (ch7 < i < ch9):
            print('7・8 章の外: %s' % a[:60], file=sys.stderr)
            return 1
    shutil.copy2(RB, BAK)
    for a, b in REP:
        t = t.replace(a, b)
    RB.write_text(t, encoding='utf-8')
    print('書いた', hashlib.sha256(t.encode('utf-8')).hexdigest())
    return 0


if __name__ == '__main__':
    sys.exit(main())
