"""fix8 係 docs 仕上げ(反証 8 回目の終わり。2026-10-28)の文書を読む試験(文書だけ。コードは変えていない)。台本
experiments/20261026_fix/fix8/docs_final/docs_fix8_final.py(35 個の直し。うち 37 か所は前の係の文の中の書き換え = SUPERSEDE)、
直す前の写し tests/fixtures/fix8_docs_final_before/(4 本と SHA256SUMS)、報告 ~/jetformer_reports/fix_20261026/fix8/docs_final.md・
まとめ fix8/README.md。

反証 8-2 回目の直しの係の「手順書に足す文」と、安全 8-2 回目の確かめて残った指摘・食い違い 1 つを、手順書・todo・README・
fix3/approvals.md に入れたかを見る:
- n0 頭の記録の 1 行(4 本とも。控えの名前・報告)と todo の題の時刻。
- n1 画面(fix8/fix_ui_r2.md 11 章): 3.1 の app.js 10/28 11:02・style.css 10:15、4 章 6 の電圧の源、4 章 8 の ■ とトースト、7 章「Failed to
  fetch」、todo 3 章の画面の行、README 6.2 の 11。「画面の版 9(app.js 10/28 04:54)」はもう「今」として出ない。
- n2 夜の道具(fix8/fix_tools_r2.md 10 章): 6 章の off_all_compared と、R1 に届かなかった OFF の扱いの変更(10/26 夜 fix3 の決めを変えた)、
  8.1 ⑤ の 3 つの行・3 章「わざと OFF にした日」、approvals 3 の決まり・8 の表(書き換えと新しい行)・0 章の 8。
- n3 安全(反証 8-2 回目 rs8-2-reverse-end-stall-overrun): approvals 3「困ること(ON)」・手順書 7 章「1 m 以上下がる」に同じ数の 1 文、
  approvals 14 の表の下の知らせ(表は fix7 の 4 行のまま)・0 章の 3・14、todo 1 章・2 章 10・16、README 6.2 の 37。
- n4 食い違い(main session の決め: 11/3 は手順書を正とする): 手順書 8.1 ⑤ に「⑤ の出力の --skip の文は、当たった・空転した・記録が
  欠けた ON の本には 11/3 は使わない」。
- n5 README 6 章の頭の反証 8 回目のまとめ(巡ごとの数・出切っていない・再起動は要らない)・6.3 の 38・7 章の fix8/README.md。
- n6 approvals 14「後ろ 1.2・1.35 m の物」の行の今の後退の幅(fix8/fix_docs_r2.md の「残したこと」5)。
- 案の報告(fix8/proposal_safety_after1103.md)は 10/28 07:46 の版(1〜9 章)のあとに 10 章を足しただけ。
- 文書の言葉・数とコード・記録: 安全の数は指摘・票の出力(review_safety_r2/out・refute_safety_r2_A/out・_B/verdicts.json)と同じ。
  文書が引く collect.py の言葉・app.js と style.css の時刻・assist.py の行は今のコードにある。再起動が要らないこと(assist.py・
  drive.py・web/server.py・web/bench.py は 10/28 08:40:33 の起動より前)。巡ごとの数は findings.json と verdicts.json から数え直す。
- わざと壊す: 直す前の写しでは n0〜n6 が全部落ちる。台本の 35 個の直しを 1 つずつ元に戻すと、その確かめが落ちる。ほかの壊しも落ちる。
- 前の係の直しと試験の確かめが崩れていないか(台本の earlier_problems: この組の書き換えを戻した文書で fix8 docs r2 → 同期 1 → fix8
  docs r1 → docs-decided → fix7 docs の仕上げ・r2・r1 の直しと試験の確かめ。今の文書で前の 7 組の確かめ)。

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider -o tmp_path_retention_policy=failed \\
        --basetemp=<消せる場所> tests/test_fix8_docs_final.py
"""
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

JF = Path('/home/jetson/jetformer')
REPORTS = Path('/home/jetson/jetformer_reports/fix_20261026')
BEFORE = JF / 'tests' / 'fixtures' / 'fix8_docs_final_before'
HERE8 = JF / 'experiments' / '20261026_fix' / 'fix8'
SCRIPT = HERE8 / 'docs_final' / 'docs_fix8_final.py'
RB, TODO, RD, AP, F7 = 'runbook_20261103.md', 'todo_user_1026.md', 'README.md', 'fix3/approvals.md', 'fix7/README.md'
FILES = (RB, TODO, RD, AP)
ALL7 = (RB, TODO, RD, AP, 'fix3/disk.md', F7, 'fix6/README.md')
BAK = '.bak-20261028-fix8-final'
STAMP = '10/28 12:4x'
R82_AFTER = {RB: '345add5a9d18', TODO: 'c3978ce6293e', RD: 'd22d8085378f', AP: 'de9921937573'}   # fix8 docs r2 の「後」(fix_docs_r2.md 7 章)
PROPOSAL = 'fix8/proposal_safety_after1103.md'
SAFE_ID = 'rs8-2-reverse-end-stall-overrun'
SAFE_MORE = 'ON は OFF より最大 +0.6 m(乱れの無いときより最大 +0.55 m)多く下がることがある'
SAFE_REAR = '後ろ 1.3〜1.7 m の物に ON だけ当たる場面がある'
SAFE_REAL = '後退の終わり 11 回でこの止まりは 0 回・1 回のリカバリーあたり約 0.4% の見込み'
SAFE_REV = 'モデルによっては逆に OFF のほうが多く下がる'
APP_NOW, CSS_NOW = (10, 28, 11, 2), (10, 28, 10, 15)       # 今の app.js・style.css の更新時刻(文書の「10/28 11:02」「10/28 10:15」)
APP_SHA, CSS_SHA = 'e61dfac905cb', 'b1e900656e24'           # fix8 ui r2 のあとの版(fix_ui_r2.md 9 章)
BOOT = datetime(2026, 10, 28, 8, 40, 33)                    # jetformer-web の起動(10/28 08:40 の Jetson の起動し直し)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(root, files=FILES):
    return {f: (Path(root) / f).read_text(encoding='utf-8') for f in files}


@pytest.fixture(scope='module')
def live():
    if not all((REPORTS / f).exists() for f in ALL7):
        pytest.skip('~/jetformer_reports の文書がありません')
    return read(REPORTS, ALL7)


@pytest.fixture(scope='module')
def before():
    sums = dict(reversed(x.split(None, 1)) for x in (BEFORE / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert set(sums) == set(FILES)
    for f, h in sums.items():
        assert _sha(BEFORE / f) == h, f
    return read(BEFORE)


@pytest.fixture(scope='module')
def script():
    return _load('docs_fix8_final_t8', SCRIPT)


def _sec(t, head, nxt):
    """見出し head から次の nxt まで。見出しが無ければ ''(直す前の写しでも確かめが例外にならないように)"""
    try:
        i = t.index(head)
        return t[i:t.index(nxt, i + len(head))]
    except ValueError:
        return ''


def rows(text, prefix, need=''):
    return [x for x in text.splitlines() if x.startswith(prefix) and need in x]


def one(text, prefix, need=''):
    r = rows(text, prefix, need)
    return r[0] if len(r) == 1 else ''


def _miss(probs, where, text, *words):
    for w in words:
        if w not in text:
            probs.append('%s: 「%s」が無い' % (where, w[:50]))


def _has(probs, where, text, *words):
    for w in words:
        if w in text:
            probs.append('%s: 前の文「%s」が残る' % (where, w[:50]))


def _ch(rb, n):
    heads = {3: ('## 3. 朝の準備', '### 3.1'), 31: ('### 3.1', '\n| 見ること |'), 4: ('## 4. 1 本の手順', '## 5. '),
             6: ('## 6. 本番の形の決め方', '## 7. '), 7: ('## 7. 止める条件', '## 8. '), 8: ('## 8. 休みと夜の集計', '## 9. ')}
    return _sec(rb, *heads[n])


def _td(td, n):
    heads = {1: ('## 1. 日ごと', '## 2. '), 2: ('## 2. 決めてほしいこと', '## 3. '), 3: ('## 3. 11/3 の当日', '## 4. ')}
    return _sec(td, *heads[n])


def _ap(ap, n):
    heads = {0: ('## 0. 一覧', '## 1. '), 3: ('## 3. 新しい安全のスイッチ', '\n## 4. '), 8: ('## 8. 回 3 の判定の読み方', '## 9. ')}
    if n == 14:
        return ap[ap.index('## 14. '):] if '## 14. ' in ap else ''
    return _sec(ap, *heads[n])


def _rd(rd, n):
    heads = {6: ('## 6. 残っていること', '### 6.1'), 62: ('### 6.2', '### 6.3'), 63: ('### 6.3', '### 6.4'), 7: ('## 7. 報告の一覧', '\n#  ')}
    if n == 7:
        return rd[rd.index('## 7. 報告の一覧'):] if '## 7. 報告の一覧' in rd else ''
    return _sec(rd, *heads[n])


# ================================================================== 確かめ(どれも問題の一覧。空なら OK)
def n0_problems(d):
    """頭の記録の 1 行(最初の見出し「## 」より前・控えの名前・報告)と todo の題の時刻"""
    p = []
    heads = {RB: '- **%s ' % STAMP, TODO: '- **%s ' % STAMP, RD: '- **%s 追記**' % STAMP, AP: '- **%s ' % STAMP}
    for f, h in heads.items():
        t = d[f]
        top = t[:t.index('\n## ')] if '\n## ' in t else t
        line = one(top, h, 'fix8 係 docs 仕上げ(反証 8 回目の終わり)')
        _miss(p, '%s の頭の記録' % f, line, '控え `%s%s`' % (Path(f).name, BAK), 'fix8/docs_final.md')
    title = d[TODO].splitlines()[0] if d[TODO] else ''
    if title != '# ユーザーが 11/3 までにすること(%s 現在)' % STAMP:
        p.append('todo の題: 仕上げの時刻でない(%s)' % title[:40])
    return p


def n1_problems(d):
    """画面: app.js 10/28 11:02・style.css 10:15・電圧の源・■ とトースト"""
    p = []
    rb, td, rd = d[RB], d[TODO], d[RD]
    r31 = one(_ch(rb, 31), '- **10/28 04:54 に `jetformer/web/static/app.js` を直した**')
    _miss(p, '手順書 3.1 app.js', r31, '**app.js は 10/28 11:02 にもう一度直した**(fix8 ui r2。', 'style.css も 10/28 10:15 に直した',
          '「画面の版 9(app.js 10/28 11:02)」(このあと誰かが app.js を直せばその時刻)', '**11:02 より前に開いた画面**',
          '10/28 04:54〜11:02 に開いた画面も', '今のファイルは 10/28 11:02 に直したもの)」→ 読み直す', '`fix8/fix_ui_r2.md`',
          '.py だけを見るので、app.js・style.css の名前は出ない')
    _has(p, '手順書 3.1 app.js', r31, '**04:54 より前に開いた画面**', '今のファイルは 10/28 04:54 に直したもの')
    if r31 and not (r31.find('`fix8/fix_ui_r1.md`)') < r31.find('**app.js は 10/28 11:02 にもう一度直した**')):
        p.append('手順書 3.1 app.js: 11:02 の文が 04:54 の直しの文のあとに無い')
    r46 = one(_ch(rb, 4), '   - **自動合わせ ON の本(2-3・2-4)は「電圧の源」も読む**')
    _miss(p, '手順書 4 章 6', r46, '「説明欄の V_before(走行のあいだ一定)」なら OK。(10/28 fix8 ui r2 の画面(app.js 11:02)では、「説明欄」と出るのはサーバーが'
          'その値を走行に使うときだけ', '6.0〜9.0 の数のとき', '831(点の打ち忘れ)・8,31・0.83・9.5・「8.31,」・セルの値 4.15 は「なし(1.00 倍。')
    c4 = _ch(rb, 4).splitlines()
    k = [i for i, x in enumerate(c4) if x == '8. 決めた周を走ったら、次の直線で CH3 を人側 → ■ 停止。']
    nxt = c4[k[0] + 1] if len(k) == 1 and k[0] + 1 < len(c4) else ''
    _miss(p, '手順書 4 章 8', nxt, '   - 読み直した画面(style.css 10/28 10:15)では、▶ のすぐあとに出るトースト', '■ の字は隠れない',
          '■ の字が隠れて見えたら画面を読み直す')
    _miss(p, '手順書 7 章 Failed to fetch', one(_ch(rb, 7), '| 走行タブを押したら「Failed to fetch」などのトーストが出た'),
          '(10/28 04:54 からの app.js(今は 10/28 11:02 の版)を読み直した画面。')
    r3s = one(_td(td, 3), '- 朝に **スマホの画面を読み直す**')
    _miss(p, 'todo 3 章 画面', r3s, '今の app.js は 10/28 11:02(▶ の横「画面の版 9(app.js 10/28 11:02)」', 'style.css は 10/28 10:15',
          '確認の文の「電圧の源: 説明欄の V_before」はサーバーが使う値(6.0〜9.0)のときだけ出る')
    _has(p, 'todo 3 章 画面', r3s, '今の app.js は 10/28 04:54')
    r11 = one(_rd(rd, 62), '| 11 |')
    _miss(p, 'README 6.2 の 11', r11, '画面の版 9(app.js 10/28 11:02)に読み直してから', '351×90 px の 2 行(前は 195×120 の 3 行)',
          '走行開始のトーストは ■ の 20 px 上')
    _has(p, 'README 6.2 の 11', r11, '画面の版 9(app.js 10/28 04:54)')
    for f in FILES:                                   # 04:54 の版を「今」の画面の印として出す文は無い
        if '「画面の版 9(app.js 10/28 04:54)」' in d[f]:
            p.append('%s: 「画面の版 9(app.js 10/28 04:54)」が残る' % f)
    return p


def n2_problems(d):
    """夜の道具: off_all_compared・短い止まりの続き・R1 に届かなかった OFF の扱いの変更"""
    p = []
    rb, ap = d[RB], d[AP]
    r6 = one(_ch(rb, 6), '| 発進の前進 ON(回 4) |')
    _miss(p, '手順書 6 章', r6, '**「→ OFF の発進が比べから外れた N 本」の行が出たら「満たさない」**(checks の off_all_compared。',
          '**R1 に届かなかった OFF の発進は「名前を出すだけで、組の答えは変えない」(10/26 夜 fix3 r1-rec3-1 の決め)だったのを、10/28 に変えた**',
          '同じ block で別の記録にやり直した本(同じベンチマーク N でもう 1 本)が比べに入っていれば数えない', '外すと残りの OFF で比べる')
    if r6 and not (0 <= r6.find('「R1 に届かなかった発進(on)」の行が出たら') < r6.find('**「→ OFF の発進が比べから外れた N 本」の行が出たら')):
        p.append('手順書 6 章: off_all_compared の文が R1 に届かなかった発進(on)の文のあとに無い')
    r5 = one(_ch(rb, 8), '| ⑤ | **回 3 は `--ignore-gain` で読む**')
    _miss(p, '手順書 8.1 ⑤', r5, '1 本でも外れたら「→ OFF の発進が比べから外れた N 本」(off_all_compared = 満たさない。下の文。10/28 fix8 tools r2)',
          '**「→ うち短い止まりが続いた発進 N 本」** = 1 回のフレームの間は 0.22 秒以下でも', '**「→ OFF の発進が比べから外れた N 本(…)」** = 6 章',
          '「(カメラの止まり・起点まで X 秒)」か「(R1 に届かなかった・理由)」', '**「注意: 発進から R1 の起点までに短い止まりが続いた発進 N 本」** = 中立が 2 回以上・'
          '合計 0.05〜0.22 秒', '(中立は 1 回・0.033 秒まで。`fix8/fix_tools_r2.md` 3 章。10/28 fix8 tools r2)')
    r3 = one(_ch(rb, 3), '  - **わざと OFF にした日の、本 X から先の読み方**')
    _miss(p, '手順書 3 章 わざと OFF', r3, '**発進の前進 OFF の本なら、その本の発進が比べに 1 本も残らなければ「→ OFF の発進が比べから外れた N 本」で満たさない**',
          '外した OFF の本も「→ OFF の発進が比べから外れた N 本」で満たさない(10/28 fix8 tools r2)。')
    _has(p, '手順書 3 章 わざと OFF', r3, 'その場で戻した発進を走り直しにする =「使ってよい」')
    r3d = one(_ap(ap, 3), '| 「途中で OFF にした日」の決まり(10/28) |')
    _miss(p, 'approvals 3 の決まり', r3d, '「→ OFF の発進が比べから外れた N 本」= 満たさない(checks の off_all_compared)**',
          'R1 に届かなかった OFF の扱いは 10/26 夜 fix3 r1-rec3-1 の決め(名前を出すだけで答えは変えない)を変えた = main session が了承',
          '「→ うち短い止まりが続いた発進 N 本」', '3 秒の窓 55,838 で合計の最大 0.200 秒・R1 に届いた発進 38 で 0.100 秒')
    a8 = _ap(ap, 8)
    r8 = one(a8, '| 発進から R1 の起点までにカメラが止まった発進')
    _miss(p, 'approvals 8 の表', r8, 'OFF の発進が 1 本でも外れたら満たさない(off_all_compared。全部なら off_no_stall も。10/28 fix8 tools r2)',
          '短い止まりが続いて drive の中立の合計が 0.22 秒より長い発進も外す')
    _has(p, 'approvals 8 の表', r8, 'OFF が全部外れたら満たさない(off_no_stall)')
    rn = one(a8, '| OFF の本の発進が R1 に届かず比べに 1 本も残らない(10/28 fix8 tools r2) |')
    _miss(p, 'approvals 8 の新しい行', rn, '満たさない(off_all_compared)', '**10/26 夜 fix3 r1-rec3-1 の決め(R1 に届かなかった OFF は名前を出すだけで答えは変えない)を変えた**')
    _miss(p, 'approvals 0 章の 8', one(_ap(ap, 0), '| 8 |'), 'R1 に届かなかった OFF の 10/26 夜 fix3 の決めを変えた')
    return p


NOTICE2_WORDS = ('**もう 1 つ(10/28 fix8 docs 仕上げ)**: 両方 ON の日、後退の終わりにカメラが 0.1〜0.3 秒止まると、ON は OFF より最大約 0.6 m 多く'
                 '下がることがある', SAFE_REAR, '11/3 のコードは変えない', '「1 m 以上下がる → すぐ人側」', '`README.md` 6.2 の 37')


def n3_problems(d):
    """安全(rs8-2-reverse-end-stall-overrun): 手順書 7 章・approvals 3・14・0 章・todo・README 6.2 の 37"""
    p = []
    rb, td, rd, ap = d[RB], d[TODO], d[RD], d[AP]
    r7 = one(_ch(rb, 7), '| リカバリーで車が 1 m 以上下がる')
    _miss(p, '手順書 7 章 1 m', r7, '**両方 ON の日、後退の終わりにカメラが 0.1〜0.3 秒止まると、' + SAFE_MORE + '**', SAFE_REAR, SAFE_REAL, SAFE_REV,
          '11/3 のコードは変えない(10/28 main session の決め)', '人の守りはこの行の「1 m 以上下がる → すぐ人側」', SAFE_ID)
    r3c = one(_ap(ap, 3), '| 困ること(ON) |')
    _miss(p, 'approvals 3 困ること(ON)', r3c, '**(10/28 知らせ。反証 8-2 回目 ' + SAFE_ID + ')両方 ON の日、後退の終わりにカメラが 0.1〜0.3 秒止まると、'
          + SAFE_MORE + '**', '96 設定とも ON が OFF と乱れの無い ON の両方より多く下がる', '**' + SAFE_REAR + '**',
          '**本物の 10/10〜10/24(59 本)の' + SAFE_REAL + '**', '**' + SAFE_REV + '**', '`fix8/proposal_safety_after1103.md` 10 章',
          '人の守りは手順書 7 章の「1 m 以上下がる → すぐ人側」')
    a14 = _ap(ap, 14)
    _miss(p, 'approvals 14', a14, '- **(10/28 知らせ。反証 8-2 回目 ' + SAFE_ID + '。fix8 docs 仕上げ。上の表は fix7 で残した 4 つのまま)', '**おすすめ: このまま**')
    body = [x for x in a14.splitlines() if x.startswith('| ') and not x.startswith('| こと |')]
    if len(body) != 4:
        p.append('approvals 14: 表の行が fix7 の 4 つでない(%d)' % len(body))
    a0 = _ap(ap, 0)
    _miss(p, 'approvals 0 章の 3', one(a0, '| 3 |'), '**もう 1 つ(10/28 fix8 docs 仕上げ)**: 両方 ON の日の後退の終わりのカメラの止まり')
    _miss(p, 'approvals 0 章の 14', one(a0, '| 14 |'), '10/28 に表の下へ知らせを 1 つ足した')
    _miss(p, 'todo 1 章 11/2 まで', one(_td(td, 1), '| **11/2 まで** |'), '**10/28 に 3 へ知らせ 1 つ**', *NOTICE2_WORDS)
    _miss(p, 'todo 2 章 10', one(_td(td, 2), '| 10 |'), '**10/28 に 3 へ知らせ 1 つ**', *NOTICE2_WORDS)
    _miss(p, 'todo 2 章 16', one(_td(td, 2), '| 16 |'), '14 には 10/28 に表の下へ知らせを 1 つ足した')
    c62 = _rd(rd, 62)
    r37 = one(c62, '| 37 |')
    _miss(p, 'README 6.2 の 37', r37, '| 37 | **ユーザーに伝えること**(10/28 fix8 docs 仕上げ。返事が無ければこのまま)', SAFE_MORE, SAFE_REAR, SAFE_REV,
          '② 回 3 の判定', '③ ⑤ の出力の「`--skip` で外して読み直せる」の文', '| main session → ユーザー | 11/2(返事が無ければこのまま) |')
    if r37 and not (0 <= c62.find('| 36 |') < c62.find('| 37 |')):
        p.append('README 6.2: 37 が 36 のあとに無い')
    return p


def n4_problems(d):
    """⑤ の出力の --skip の文は、当たった・空転した・記録が欠けた ON の本には 11/3 は使わない(手順書が正)"""
    p = []
    r5 = one(_ch(d[RB], 8), '| ⑤ | **回 3 は `--ignore-gain` で読む**')
    _miss(p, '手順書 8.1 ⑤', r5, '**⑤ の出力の `--skip` の文(「動画・紙で確かめ、当たっていない・車のせいでないと分かった本だけ、main session が --skip 記録名 で'
          '外して読み直せる」', '「--skip 記録名 で外せる」', '「決めは main session(--skip 記録名 で外すと、起点までの時間の比べからも外れる)」',
          'は、当たった・空転した・記録が欠けた ON の本には 11/3 は使わない = この手順書を正とする**', '⑤ の文は道具の形のまま')
    a = r5.find('`collect.py` の `--skip` の説明「当たった本は外さない」と同じ向きで、空転・欠けも(10/28 main session の決め・fix8 docs r2)。')
    if r5 and not (0 <= a < r5.find('**⑤ の出力の `--skip` の文(')):
        p.append('手順書 8.1 ⑤: 11/3 は使わないの文が --skip の文のあとに無い')
    return p


def n5_problems(d):
    """README 6 章の頭の反証 8 回目のまとめ・6.3 の 38・7 章の fix8/README.md と 8-2 回目の報告"""
    p = []
    rd = d[RD]
    head = one(_rd(rd, 6), '- (10/28 fix8 docs 仕上げ)**反証 8 回目はここで終える**')
    _miss(p, 'README 6 章の頭', head, 'まとめは **`fix8/README.md`**', '8-1 は新しい指摘 20 件 → 打ち消し 4 件・残す 16 件(16 件とも直した)',
          '8-2 は新しい指摘 10 件(どれも minor)→ 打ち消し 0 件・残す 10 件と、8-1 から渡された試験の古い期待 1 件 = 11 件とも直した',
          '**巡ごとに細かい場面になったが、2 巡目にも新しい指摘が出た = 出切っていない**', '8-2 の直しはどれも jetformer-web の再起動が要らない',
          'ユーザーに伝えることは 6.2 の 37')
    _miss(p, 'README 6.3 の 38', one(_rd(rd, 63), '| 38 |'), '反証 8-2 回目で残したこと', '`fix8/proposal_safety_after1103.md` 10 章',
          '① 3 秒・② 2.5 m', '`fix8/README.md` 3 章')
    c7 = _rd(rd, 7)
    _miss(p, 'README 7 章', one(c7, '| **`fix8/README.md`** |'), '**反証 8 回目(2 巡・8-1 20 件・8-2 10 件)の直しのまとめ**')
    _miss(p, 'README 7 章', one(c7, '| `fix8/fix_ui_r2.md`・`fix8/fix_tools_r2.md`・`fix8/fix_docs_r2.md`・`fix8/docs_final.md` |'), '反証 8-2 回目の直し',
          '10 章(後退の終わりのカメラの止まりの案。案の文だけ)')
    return p


def n6_problems(d):
    """approvals 14「後ろ 1.2・1.35 m の物」の行に今の後退の幅(fix8/fix_docs_r2.md の「残したこと」5)"""
    p = []
    r = one(_ap(d[AP], 14), '| 後ろ 1.2・1.35 m の物 |')
    _miss(p, 'approvals 14 の 1.2・1.35 m', r, '見張り ON の最初のリカバリーは 1.02〜1.97 m 下がる(同じ場面の OFF は 1.17〜2.91 m)',
          '「1.1〜1.4 m」は fix7 の閉じたループの幅', '| 10 のおすすめ (a)(11/3 は変えない。11/3 のリカバリーの記録で決め直す)のまま |')
    return p


CHECKS = {'n0': n0_problems, 'n1': n1_problems, 'n2': n2_problems, 'n3': n3_problems, 'n4': n4_problems, 'n5': n5_problems,
          'n6': n6_problems}
SOURCE_CHECK = {'ui r2 11 章(app.js 11:02・style.css 10:15・電圧の源・■ とトースト)': 'n1',
                'tools r2 10 章(off_all_compared・launch_neutral・R1 に届かない OFF)': 'n2', SAFE_ID: 'n3',
                'skip-contradiction(⑤ の --skip の文と手順書。11/3 は手順書が正)': 'n4',
                'docs r2 の残したこと・反証 8 回目のまとめ(README 6・7 章)': 'n5'}
EDIT_OVERRIDE = {(AP, 10): 'n6'}        # approvals 14 の直しは 1.2・1.35 m の行(n6)と知らせ(n3)の両方。戻すと n6 も落ちる


def edit_check(script):
    out = {x: SOURCE_CHECK[k] for k, v in script.SOURCES.items() for x in v}
    out.update(EDIT_OVERRIDE)
    out.update({x: 'n0' for x in script.LOGS})
    return out


# ================================================================== 文書
@pytest.mark.parametrize('name', sorted(CHECKS))
def test_live_documents(live, name):
    assert CHECKS[name](live) == []


@pytest.mark.parametrize('name', sorted(CHECKS))
def test_before_copies_fail(before, name):
    """直す前の写し(10/28 11:48 の形。fix8 docs r2 のあと)では、7 つとも落ちる"""
    assert CHECKS[name](before) != []


def test_before_copies_are_the_backups():
    """直す前の写しは文書の横の控え(.bak-20261028-fix8-final)と同じで、fix8 docs r2 の「後」の sha256"""
    for f in FILES:
        b = REPORTS / (f + BAK)
        assert _sha(b) == _sha(BEFORE / f), f
        assert _sha(b)[:12] == R82_AFTER[f], f


def test_sources_cover_every_edit(script):
    """台本の 35 個の直しは、どれも元(報告の文・指摘)の 1 つか頭の記録の行(LOGS)にちょうど 1 回入る。STAMP・BAK は試験と同じ"""
    used = sorted(x for v in script.SOURCES.values() for x in v) + sorted(script.LOGS)
    allk = sorted((f, i) for f, edits in script.ALL for i in range(len(edits)))
    assert sorted(used) == allk and len(allk) == 35
    assert sorted(script.SOURCES) == sorted(SOURCE_CHECK)
    assert script.STAMP == STAMP and script.BAK == BAK and script.FILES == FILES


def test_script_edits_all_done(live, script, tmp_path):
    """台本の 35 個の直しは今の文書でどれも「済」で new がちょうど 1 回。SUPERSEDE 37 か所は前の台本の new(今の形)の中にちょうど 1 回
    あった文の書き換えで、今の文書に入っている(消した文は 0 回)。前の係の直し(書き換えを戻した文書で)・前の 7 組の試験の確かめ(今の
    文書で)・この組の確かめも崩れていない(earlier_problems)"""
    _, rep = script.apply_text(live)
    assert len(rep) == 35 and [s for _, _, s in rep] == ['done'] * 35, [(f, o[:30], s) for f, o, s in rep if s != 'done']
    for f, edits in script.ALL:
        for _, new in edits:
            assert live[f].count(new) == 1, (f, new[:40])
    assert len(script.SUPERSEDE) == 37 and script.superseded_present(live) == []
    assert script.revert(live) != live
    assert script.earlier_problems(live, str(tmp_path)) == []


def test_reverting_each_edit_is_caught(live, script):
    """わざと壊す: 台本の直しを 1 つずつ元の文に戻すと、その確かめが落ちる(35 個)"""
    ec = edit_check(script)
    n = 0
    for f, edits in script.ALL:
        for k, (old, new) in enumerate(edits):
            broken = dict(live, **{f: live[f].replace(new, old, 1)})
            assert broken[f] != live[f], (f, k)
            assert CHECKS[ec[(f, k)]](broken) != [], (f, k, new[:40])
            n += 1
    assert n == 35


def test_more_breakages_are_caught(live):
    """台本の直しと別の形の壊し"""
    rb, td, ap, rd = live[RB], live[TODO], live[AP], live[RD]
    cases = [
        ('n1', RB, rb.replace('「画面の版 9(app.js 10/28 11:02)」(このあと誰かが', '「画面の版 9(app.js 10/28 04:54)」(このあと誰かが', 1)),
        ('n1', TODO, td.replace('今の app.js は 10/28 11:02', '今の app.js は 10/28 04:54', 1)),
        ('n1', RD, rd.replace('351×90 px の 2 行', '195×120 px の 3 行', 1)),
        ('n2', RB, rb.replace('(main session が了承。`fix8/fix_tools_r2.md` 8 章))', '(`fix8/fix_tools_r2.md` 8 章))', 1).replace(
            '**R1 に届かなかった OFF の発進は「名前を出すだけで、組の答えは変えない」(10/26 夜 fix3 r1-rec3-1 の決め)だったのを、10/28 に変えた**', '', 1)),
        ('n2', AP, ap.replace('合計の最大 0.200 秒', '合計の最大 0.120 秒', 1)),
        ('n2', AP, ap.replace('OFF の発進が 1 本でも外れたら満たさない(off_all_compared。全部なら off_no_stall も。10/28 fix8 tools r2)。',
                              'OFF が全部外れたら満たさない(off_no_stall)。', 1)),
        ('n3', RB, rb.replace('ON は OFF より最大 +0.6 m(乱れの無いときより最大 +0.55 m)', 'ON は OFF より最大 +0.36 m(乱れの無いときより最大 +0.36 m)', 1)),
        ('n3', AP, ap.replace('**' + SAFE_REAR + '**', '**後ろ 1.0〜1.6 m の物に ON だけ当たる場面がある**', 1)),
        ('n3', TODO, td.replace('`README.md` 6.2 の 37)', '`README.md` 6.2 の 36)', 1)),
        ('n3', AP, ap.replace('- **(10/28 知らせ。反証 8-2 回目 ' + SAFE_ID, '| (10/28 知らせ。反証 8-2 回目 ' + SAFE_ID, 1)),     # 表の行にすると fix7 の 4 行でない
        ('n4', RB, rb.replace('は、当たった・空転した・記録が欠けた ON の本には 11/3 は使わない = この手順書を正とする**',
                              'は、11/3 も使ってよい**', 1)),
        ('n5', RD, rd.replace('**巡ごとに細かい場面になったが、2 巡目にも新しい指摘が出た = 出切っていない**', '**出切った**', 1)),
        ('n5', RD, rd.replace('8-2 は新しい指摘 10 件(どれも minor)→ 打ち消し 0 件・残す 10 件', '8-2 は新しい指摘 14 件 → 残す 11 件', 1)),
        ('n6', AP, ap.replace('1.02〜1.97 m 下がる(同じ場面の OFF は 1.17〜2.91 m)', '1.1〜1.4 m 下がる', 1)),
        ('n0', TODO, td.replace(td.splitlines()[0], '# ユーザーが 11/3 までにすること(10/28 11:4x 現在)', 1)),
    ]
    for k, (name, f, t) in enumerate(cases):
        assert t != live[f], k
        assert CHECKS[name](dict(live, **{f: t})) != [], k


def test_proposal_chapter10_is_appended_only(script):
    """案の報告は 10/28 07:46 の版(1〜9 章。53,507 バイト・sha256 6773816649fb…)の後ろに 10 章を足しただけ。控えはその版"""
    raw = (REPORTS / PROPOSAL).read_bytes()
    assert script.proposal_state(raw) == 'done'
    assert hashlib.sha256(raw[:53507]).hexdigest() == script.PROPOSAL_BEFORE_SHA
    assert _sha(REPORTS / (PROPOSAL + BAK)) == script.PROPOSAL_BEFORE_SHA
    c10 = raw[53507:].decode('utf-8')
    assert c10.startswith('\n## 10. 追記(10/28 fix8 係 docs 仕上げ)')
    for w in ('**main session の決め(10/28 05:xx・11:3x): 11/3 の安全のコードは変えない**', '**写しは作っていない・閉じたループで比べていない**',
              '| **a1 超えた分だけ惰性を短くする**(比べる順の 1 番目) |', '`self.t_phase = now - hold_s - (self.backed_s - rev_s)`',
              '| a2 escape に進める |', '| b drive で中立をすぐ書く |', 'ON − OFF は最大 +0.605 m(k 0・3 フレーム)',
              '+0.127〜+0.547 m(中央 +0.238)', '98 / 45,861 = 0.21%', '121 / 67,620 = 0.18%', '比べるまでは、どれもすすめとは言えない'):
        assert w in c10, w
    for h in ('### 10.1 今の形', '### 10.2 案', '### 10.3 比べ方', '### 10.4 本物の ESC で確かめること', '### 10.5 おすすめ'):
        assert h in c10, h


# ================================================================== 文書の言葉・数とコード・記録
def test_findings_and_votes_recounted():
    """巡ごとの数は findings.json と verdicts.json から数え直す(規則: real かつ severity ≠ none が 1 票以上、かつ not-real が 2 票未満 →
    残す): 8-1 は新しい 20 件 → 残す 16・消す 4、8-2 は新しい 10 件(どれも minor)→ 残す 10・消す 0。安全 8-2 の 1 件は票 A・B とも real・minor"""
    def count(r):
        new, kept, sev = 0, [], {}
        for fj in sorted(HERE8.glob('review_*_%s/findings.json' % r)):
            team = fj.parent.name[len('review_'):-len('_' + r)]
            votes = {}
            for ab in 'AB':
                for v in json.loads((HERE8 / ('refute_%s_%s_%s' % (team, r, ab)) / 'verdicts.json').read_text(encoding='utf-8'))['verdicts']:
                    votes.setdefault(v['id'], []).append((bool(v['real']), v.get('severity')))
            for f in json.loads(fj.read_text(encoding='utf-8'))['findings']:
                new += 1
                sev[f['severity']] = sev.get(f['severity'], 0) + 1
                vs = votes[f['id']]
                assert len(vs) == 2, f['id']
                if sum(1 for x, s in vs if x and s not in (None, 'none')) >= 1 and sum(1 for x, _ in vs if not x) < 2:
                    kept.append((team, f['id']))
        return new, kept, sev
    n1, k1, _ = count('r1')
    assert (n1, len(k1)) == (20, 16)
    n2, k2, s2 = count('r2')
    assert (n2, len(k2), s2) == (10, 10, {'minor': 10})
    teams = sorted(t for t, _ in k2)
    assert teams == sorted(['docs'] * 4 + ['rehearsal'] * 2 + ['safety'] + ['tools'] * 2 + ['ui'])
    for ab in 'AB':
        v = json.loads((HERE8 / ('refute_safety_r2_%s' % ab) / 'verdicts.json').read_text(encoding='utf-8'))['verdicts']
        assert [(x['id'], x['real'], x['severity']) for x in v] == [(SAFE_ID, True, 'minor')]


def test_safety_numbers_are_in_the_outputs():
    """文書の安全の数 = 指摘・票の出力: ana2_s1.txt(前進 120 µs・96 設定)の k −1・0 × 3〜8 フレームは 96 / 96 で ON が悪く、ON − 乱れなし
    の最大 +0.547・ON − OFF の最大 +0.605、10 フレームは ON = OFF。ana2_s4.txt の lag0.8 は OFF のほうが悪い。票 B の 1.3〜1.7 m。
    本物の後退の終わり 11 回で止まり 0(0.1 秒より長い間は人の取り戻しだけ)。0.21%・0.18%・約 0.4%"""
    o = HERE8 / 'refute_safety_r2_A' / 'out'
    rx = re.compile(r'^(\S+)\s+k\s*([+-]\d+) L\s*(\d+) \|\s+(\d+) \|\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+) \| ([+-][\d.]+)\[([+-][\d.]+),([+-][\d.]+)\] \| '
                    r'([+-][\d.]+)\[([+-][\d.]+),([+-][\d.]+)\] \| ([+-][\d.]+)\[([+-][\d.]+),([+-][\d.]+)\]')
    rows1 = [rx.match(x) for x in (o / 'ana2_s1.txt').read_text(encoding='utf-8').splitlines()]
    f120 = [m for m in rows1 if m and m.group(1) == 'fwd120' and m.group(2) in ('-1', '+0')]
    win = [m for m in f120 if 3 <= int(m.group(3)) <= 8]
    assert len(win) == 8 and all(m.group(4) == '96' and m.group(5) == '96' for m in win)
    assert max(float(m.group(11)) for m in win) == 0.547 and max(float(m.group(17)) for m in win) == 0.605
    k0 = [m for m in win if m.group(2) == '+0']
    assert {(m.group(9), m.group(10), m.group(11)) for m in k0} == {('+0.238', '+0.127', '+0.547')}
    assert all(m.group(7) == '96' for m in f120 if int(m.group(3)) == 10)          # 10 フレームは ON = OFF
    rows4 = [rx.match(x) for x in (o / 'ana2_s4.txt').read_text(encoding='utf-8').splitlines()]
    lag = [m for m in rows4 if m and m.group(1) == 'lag0.8' and m.group(2) in ('-1', '+0') and 3 <= int(m.group(3)) <= 8]
    assert lag and all(m.group(6) == '6' and m.group(5) == '0' for m in lag)       # OFF のほうが悪い 6 / 6
    assert (min(float(m.group(13)) for m in lag), max(float(m.group(14)) for m in lag)) == (0.267, 0.465)
    assert (min(float(m.group(10)) for m in lag), max(float(m.group(11)) for m in lag)) == (0.124, 0.387)
    neu = [m for m in rows4 if m and m.group(1) == 'neu' and m.group(2) in ('-1', '+0') and 3 <= int(m.group(3)) <= 8]
    assert (min(float(m.group(13)) for m in neu), max(float(m.group(14)) for m in neu)) == (1.13, 5.255)
    va = json.loads((HERE8 / 'refute_safety_r2_A' / 'verdicts.json').read_text(encoding='utf-8'))['verdicts'][0]['evidence']
    assert '98/45,861 = 0.21%' in va and '1 回のリカバリーあたり約 0.4%' in va and 'ESC 24 × 位相 2 × 処理 2' in va
    vb = json.loads((HERE8 / 'refute_safety_r2_B' / 'verdicts.json').read_text(encoding='utf-8'))['verdicts'][0]['evidence']
    assert 'ON だけ当たるのは後ろ 1.3〜1.7 m の物だけ' in vb and '1 フレームあたり 0.18%(121/67,620)' in vb
    assert '1.0 m では ON 2.47・OFF 1.805・乱れなし 1.97 m/s で、ON が +0.5 m/s 強い' in vb
    rv = json.loads((HERE8 / 'review_safety_r2' / 'out' / 'realov12.json').read_text(encoding='utf-8'))
    assert rv['logs'] == 59 and rv['rev_last_rows'] == 11
    assert len(rv['rev_last_gap_gt_0p1']) == 4 and all(x['gap_s'] > 2.0 for x in rv['rev_last_gap_gt_0p1'])   # 人の取り戻しだけ
    fd = json.loads((HERE8 / 'review_safety_r2' / 'findings.json').read_text(encoding='utf-8'))['findings'][0]
    assert fd['id'] == SAFE_ID and '下がった距離 ON 1.678 m・最高 2.65 m/s / OFF 1.320 m' in fd['scenario']
    assert 't_phase = now − hold_s − (backed_s − rev_s)' in fd['fix'] and "'coast' を飛ばして escape" in fd['fix']   # 10 章の案 a1・a2


def test_words_match_the_code():
    """文書が引く言葉・時刻は今のコードにある: collect.py の出力の文・app.js と style.css の版と時刻・assist.py の 'reverse' の枝・
    speed_auto の 6.0〜9.0 V"""
    co = (JF / 'experiments' / '20261026_fix' / 'diag_launch' / 'collect.py').read_text(encoding='utf-8')
    for w in ("'  → OFF の発進が比べから外れた %d 本(", '= 満たさない(checks の off_all_compared。', "'  → うち短い止まりが続いた発進 ",
              '短い止まりが続いた: 中立の見積もりの合計', "' カメラの止まりが続いた(", "'  注意: 発進から R1 の起点までに短い止まりが続いた発進 %d 本(",
              "'カメラの止まり・起点まで %.3f 秒'", "'R1 に届かなかった・%s'", 'LAUNCH_STALL_SUM_S = 0.22', 'LAUNCH_SUM_NOTE_S = 0.05',
              '動画・紙で確かめ、当たっていない・車のせいでないと分かった本だけ、', "'main session が --skip 記録名 で外して読み直せる(外した名前が出る)')",
              'ON なので満たさない。--skip 記録名 で外せる', '決めは main session(--skip 記録名 で外すと、起点までの時間の比べからも外れる)',
              '当たった本は外さない'):
        assert w in co, w
    web = JF / 'jetformer' / 'web' / 'static'
    for name, sha, hm in (('app.js', APP_SHA, APP_NOW), ('style.css', CSS_SHA, CSS_NOW)):
        assert _sha(web / name).startswith(sha), name           # このあと誰かが直せば落ちて知らせる(文書の時刻を合わせる)
        t = datetime.fromtimestamp((web / name).stat().st_mtime)
        assert (t.month, t.day, t.hour, t.minute) == hm, (name, t)
    js = (web / 'app.js').read_text(encoding='utf-8')
    assert 'const DRIVE_CLIENT = 9' in js and "'説明欄の V_before(走行のあいだ一定)' : 'なし(1.00 倍。周ごとの直しだけ)';" in js
    assert 'const VDESC = [6.0, 9.0];' in js
    css = (web / 'style.css').read_text(encoding='utf-8')
    assert 'width: max-content; max-width: 90%' in css and 'body.driving .toast { bottom: 126px; }' in css
    asrc = (JF / 'jetformer' / 'assist.py').read_text(encoding='utf-8')
    assert ("        if p == 'reverse':\n            self.t_check = now" in asrc and
            "            if self.backed_s >= rev_s:                  # fix6 s3-b: 後退は中立止めの始めで出し終えていた。惰性はそこから\n"
            "                self._go('coast', now)\n                self.t_phase = now - hold_s" in asrc)
    from jetformer import assist as A
    assert A.Recovery.HOLD_KEEP_MAX_S == 0.5
    from jetformer import speed_auto
    ap_ = speed_auto.AutoParams()
    assert (ap_.vdesc_lo, ap_.vdesc_hi) == (6.0, 9.0)


def test_restart_not_needed():
    """再起動は要らない: assist.py・drive.py・web/server.py・web/bench.py は jetformer-web の起動(10/28 08:40:33)より前の更新のまま(読むだけ)。
    jetformer/ の .py にそれより新しいものが無い。systemd の起動の時刻が読めれば 08:40:33。このあと誰かが直すか再起動したら落ちて知らせる"""
    for rel in ('assist.py', 'drive.py', 'web/server.py', 'web/bench.py'):
        t = datetime.fromtimestamp((JF / 'jetformer' / rel).stat().st_mtime)
        assert t < BOOT, (rel, t)
    newer = [str(q) for q in sorted((JF / 'jetformer').rglob('*.py')) if q.stat().st_mtime > BOOT.timestamp()]
    assert newer == [], newer
    try:
        r = subprocess.run(['systemctl', 'show', '-p', 'ActiveEnterTimestamp', '--value', 'jetformer-web'], capture_output=True,
                           text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return
    if r.returncode == 0 and r.stdout.strip():
        assert r.stdout.strip() == 'Wed 2026-10-28 08:40:33 JST', r.stdout


def test_fix8_readme():
    """fix8/README.md(反証 8 回目のまとめ 1 枚): 章・巡ごとの数・出切っていない・再起動・おすすめ 3 行。sha256 の表の値は今のファイル"""
    t = (REPORTS / 'fix8' / 'README.md').read_text(encoding='utf-8')
    for h in ('## 0. ひと目で', '## 1. main session の決めの実装', '## 2. 8-1 の中身', '## 3. 8-2 の中身', '## 4. 残すこと',
              '## 5. 了承が要ること・知らせ', '## 6. ユーザーに伝えること', '## 7. 11/3 のあとに向けた安全の案', '## 8. 再起動が要るか',
              '## 9. 試験', '## 10. 今のファイル', '## 11. 次に'):
        assert h in t, h
    for w in ('**反証 8 回目は 8-2 で終える**', '**完全には出切っていない**', '| 8-1 | 20 | 4 | 16 | 16 |', '| 8-2 | 10 | 0 | 10 | 11 |',
              '→ **要らない**', SAFE_ID):
        assert w in t, w
    for i in ('f8r1-docs-residual2-reversed', 'w1-tools-r8-1-trial-form', 'rs8-1-backearly-rear-faster', 'toast-half-width-on-phone',
              'rehearsal-fw-cwd', 'f8r2-docs-paper-trial-skip-drops-faults', 'w2-tools-r8-2-off-few-left', 'float-stop-half-width-under-toast',
              'rehearsal-r2-vbefore-screen-vs-server', SAFE_ID, 'f8r1-docs-approvals3-old-row', 'rehearsal-v1-count'):
        assert t.count('**%s**' % i) == 1, i
    c10 = t[t.index('## 10. 今のファイル'):t.index('## 11. 次に')]
    rowsx = re.findall(r'^\| `([^`]+)` \| `([0-9a-f]{12})…` \|', c10, re.M)
    assert len(rowsx) >= 15, rowsx
    for rel, h in rowsx:
        base = REPORTS if not rel.startswith(('jetformer/', 'tools/', 'experiments/', 'data/', 'tests/')) else JF
        assert _sha(base / rel).startswith(h), rel
