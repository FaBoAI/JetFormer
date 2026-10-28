"""fix7 係 docs 直す 7-2 回目(2026-10-27 22:xx): 反証 7-2 回目で確かめて残った docs の指摘 5 件の文書の直しを読む試験(文書だけ。
コードは変えていない)。直しの台本は experiments/20261026_fix/fix7/docs_r2/docs_fix7_r2.py(45 個の置き換え)、直す前の文書の写しは
tests/fixtures/fix7_docs_r2_before/(7-1 回目の写し tests/fixtures/fix7_docs_before/ は 7-1 回目の試験が使うので別の場所)。

- w2-docs-w2-docs-r72-docs-1(major): 7 章「中立で停止中」の「CH3 を人側にしてから AI 側に戻す」が fix7 tools(中止のあとの走り直しは
  ⑤③④ が数えない)と合わず、「当たって止まったあと」の行とも食い違った → どの回でも HS に置き直してから・走り直しは数えない・
  3 周を数えるなら ■ でもう 1 本。8.1 の ③④⑤ の読み方。本物の ⑤③④ を作り物の回 3 で流し、手順書が引く言葉と同じ文を出すかも見る。
  本物の assist.LaunchBoost で「人側 → AI 側にするたびに 1 回・前が約 1 m 以上空いていれば」も見る。
- w2-docs-w2-docs-r72-docs-2: 3.1 の表が「19:36 の fix7 安全の直しが今のサーバーに無い」を見分けられなかった → 「起動のあとに直した
  コード」の行とコマンド。偽物の systemctl と作り物のパッケージで、文書のコマンドそのものを流す(新しいファイルだけ名前が出る)。
- w2-docs-w2-docs-r72-docs-3: 朝の確かめを収集したまま走った日の ② の「なし」の区切り → 起こし直しではないと書いた。本物の runs_1103 を
  作り物の記録で流し、出る行と手順書の引用が同じかを見る。
- w2-docs-w2-docs-r72-docs-4: 8.1 のコマンドの塊は ① が落ちると ②〜⑤ も落ちた → ① と ⑤ の cd を ( ) の中に。手順書から切り出した塊を
  本物の bash で流す(python3・jf-heavy だけ作り物。① や ⑤ が 1 や 130 で終わっても、次の行と次の休みの塊が回るか)。
- w2-docs-w2-docs-r72-docs-5: approvals 8 の表に形の分かれ・中止のあとの走り直しを足した。本物の ⑤ がその 2 つで言うことと照らす。
- 同じ根の足し(ほかの係が渡した「手順書に足す文」): fix7 safety 7 章 2・3、fix7 ui 10 章、fix7 ui r2 10 章 1・2(g6)。fix7 tools r2 8 章
  (中立で停止中のまま人側にしたあとの走り直し・ON の走り直しの空転と欠け)は g1・g5 に入れた。言葉が本物のコードにあるかも見る。
- わざと壊す: 直す前の写しでは 6 つの確かめが全部落ちる。台本の 45 個の直しのうち頭の記録の 5 個を除く 40 個を 1 つずつ元に戻すと、
  その指摘の確かめが落ちる。ほかの形の壊しも落ちる。7-1 回目の 44 個の直しと画面の版 9 の直しは、どれも「済」のまま。
- 2026-10-28 fix8 docs-decided(experiments/20261026_fix/fix8/docs_decided/docs_fix8_decided.py): 10/28 00:53:23 のもう一度の再起動の
  記録などを、この組の文の終わり(「 |」の前)に足した(SUPERSEDE。g1〜g6 が引く言葉は消していない)。台本の直しが「済」かは書き換えを
  戻した文書(fix8.revert)で、わざと壊すは書き換えたあとの new(fix8.effective)で照らす。書き換えた文がちょうど 1 回あることも見る。
- 2026-10-28 fix8 docs 同期 1(experiments/20261026_fix/fix8/docs_sync1/docs_fix8_sync1.py): 反証 8-1 回目の夜の道具の直し(OFF の本の
  カメラの止まり)を、この組の 3 章「わざと OFF にした日」の文の終わりに足した(SUPERSEDE 1 か所。g1〜g6 が引く言葉は消していない)。
  台本の直しが「済」かは 同期 1 → fix8 docs-decided の順に書き換えを戻した文書(fix8.revert(s1.revert(…)))で、わざと壊すは両方の
  書き換えのあとの形の new(fix8.effective → s1.effective)で照らす(弱めていない: 同じ 45・40 個と 7-1 回目の 44 個)。
- 2026-10-28 fix8 docs 直す 8-2 回目(experiments/20261026_fix/fix8/fix_docs_r2/docs_fix8_r2.py): 反証 8-2 回目の指摘で、この組の文のうち
  4 つ(手順書 3 章「最初のリカバリー」の「1 m 以上下がる・後ろの物に近づく・おかしな動き(…」・7 章 中立止めの「(印が無ければ … --skip」・
  8.1 ⑤ の「印の無い走り直し … で外す」・todo 3 章の「戻したら紙に「走り直し: 本・時刻」。」)と approvals 0 章の 2 を意図して書き換えた
  (SUPERSEDE)。「済」は fix8 docs r2 → 同期 1 → fix8 docs-decided の順に戻した文書(fix8.revert(s1.revert(r82.revert(…))))で、わざと
  壊すは 3 つの書き換えのあとの形(r82.effective(…, s1.effective(…, fix8.effective(…))))で照らす(弱めていない: 同じ 45・40 個と
  7-1 回目の 44 個)。g2 の「approvals 今の行」は、その行の頭を「前の今(10/27 17:3x。…」にした(f8r2-docs-approvals-restart-stale)ので、
  前の頭か今の頭の行を引く(中身の確かめは同じ)。
- 2026-10-28 fix8 docs 仕上げ(experiments/20261026_fix/fix8/docs_final/docs_fix8_final.py): 反証 8 回目の終わりに、この組の文のうち 1 つ
  (手順書 3 章「わざと OFF にした日」の終わり)を意図して書き換えた(SUPERSEDE。足しただけ)。「済」は仕上げ → fix8 docs r2 → 同期 1 →
  fix8 docs-decided の順に戻した文書(fix8.revert(s1.revert(r82.revert(fin.revert(…)))))で、わざと壊すは 4 つの書き換えのあとの形
  (fin.effective(…, r82.effective(…, s1.effective(…, fix8.effective(…)))))で照らす(弱めていない: 同じ 45・40 個と 7-1 回目の 44 個)。
"""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

JF = Path('/home/jetson/jetformer')
REPORTS = Path('/home/jetson/jetformer_reports/fix_20261026')
BEFORE = JF / 'tests' / 'fixtures' / 'fix7_docs_r2_before'
SCRIPT = JF / 'experiments' / '20261026_fix' / 'fix7' / 'docs_r2' / 'docs_fix7_r2.py'
GEN = JF / 'experiments' / '20261026_fix' / 'fix7' / 'tools_r1' / 'probes' / 'gen_course.py'
COLLECT = JF / 'experiments' / '20261026_fix' / 'diag_launch' / 'collect.py'
RUNS = JF / 'experiments' / '20261026_fix' / 'plan' / 'runs_1103.py'
FIX8 = JF / 'experiments' / '20261026_fix' / 'fix8' / 'docs_decided' / 'docs_fix8_decided.py'     # 2026-10-28 fix8 docs-decided
S1 = JF / 'experiments' / '20261026_fix' / 'fix8' / 'docs_sync1' / 'docs_fix8_sync1.py'          # 2026-10-28 fix8 docs 同期 1
R82 = JF / 'experiments' / '20261026_fix' / 'fix8' / 'fix_docs_r2' / 'docs_fix8_r2.py'           # 2026-10-28 fix8 docs 直す 8-2 回目
FIN = JF / 'experiments' / '20261026_fix' / 'fix8' / 'docs_final' / 'docs_fix8_final.py'         # 2026-10-28 fix8 docs 仕上げ
RB, TODO, RD, AP, F6 = 'runbook_20261103.md', 'todo_user_1026.md', 'README.md', 'fix3/approvals.md', 'fix6/README.md'
FILES = (RB, TODO, RD, AP, F6)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(root):
    return {f: (Path(root) / f).read_text(encoding='utf-8') for f in FILES}


@pytest.fixture(scope='module')
def live():
    if not all((REPORTS / f).exists() for f in FILES):
        pytest.skip('~/jetformer_reports の文書がありません')
    return read(REPORTS)


@pytest.fixture(scope='module')
def before():
    sums = dict(reversed(x.split(None, 1)) for x in (BEFORE / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert set(sums) == set(FILES)
    for f, h in sums.items():
        assert _sha(BEFORE / f) == h, f
    return read(BEFORE)


@pytest.fixture(scope='module')
def script():
    return _load('docs_fix7_r2_t', SCRIPT)


@pytest.fixture(scope='module')
def fix8():
    """fix8 docs-decided が意図して書き換えた所(SUPERSEDE)・revert・effective"""
    return _load('docs_fix8_decided_from_t2', FIX8)


@pytest.fixture(scope='module')
def s1():
    """fix8 docs 同期 1 が意図して書き換えた所(SUPERSEDE)・revert・effective(2026-10-28)"""
    return _load('docs_fix8_sync1_from_t2', S1)


@pytest.fixture(scope='module')
def r82():
    """fix8 docs 直す 8-2 回目が意図して書き換えた所(SUPERSEDE)・revert・effective(2026-10-28)"""
    return _load('docs_fix8_r2_from_t2', R82)


@pytest.fixture(scope='module')
def fin():
    """fix8 docs 仕上げが意図して書き換えた所(SUPERSEDE)・revert・effective(2026-10-28)"""
    return _load('docs_fix8_final_from_t2', FIN)


def chapter(t, head, nxt):
    i = t.index(head)
    return t[i:t.index(nxt, i + len(head))]


def rows(text, prefix, need=''):
    return [x for x in text.splitlines() if x.startswith(prefix) and need in x]


def one(text, prefix, need=''):
    r = rows(text, prefix, need)
    return r[0] if len(r) == 1 else ''


def _miss(probs, where, text, *words):
    for w in words:
        if w not in text:
            probs.append('%s: 「%s」が無い' % (where, w[:50]))


# ================================================================== 確かめ(どれも問題の一覧。空なら OK)
def g1_problems(d, tmp=None):
    """w2-docs-r72-docs-1: 中止・中立で停止中のあとに AI 側に戻すとき(HS に置き直す・走り直しは数えない・3 周はもう 1 本)"""
    p = []
    rb = d[RB]
    ch7 = chapter(rb, '## 7. 止める条件', '## 8. ')
    ch81 = chapter(rb, '### 8.1', '### 8.2')
    hold = one(ch7, '| 走行タブに「CH3: AI側 ・ 中立で停止中」')
    _miss(p, '7 章 中立止め', hold,
          '**中止・「中立で停止中」のあとに AI 側に戻すときは、どの回でも、人側のうちに車を HS のいつもの場所に置き直してから**',
          '人側 → AI 側にするたびに 1 回。止まっていて前が約 1 m 以上空いていれば 250 µs・最大 1.2 秒', '前へ 80 µs・最大 2 秒',
          '**同じ本の中で戻した発進(走り直し)は数えない**', '⑤ の「リカバリー中止のあとの走り直し N 本」',
          '③ の「数えない(リカバリー中止のあとの走り直し。中止 X 秒・人側 Y 秒)」', '④ の「数えない発進 N(リカバリー中止のあとの走り直し)」',
          '「中立で停止中」のまま人側にした(中止は記録に出ない)あとも同じ', '「中立止め X 秒から Y 秒」「カメラの途切れ X 秒から Y 秒」',
          'HS に置き直しても数えない',
          '**ただし ON の走り直しの最初の周の衝突・空転(比 > 1.3・比が出ない)・最初の周の記録の欠けは ⑤ が比べの条件に数える**',
          '中止・0.5 秒以上の中立止め・カメラの途切れで止まって終わった AI の区間のあとの発進に付く', '紙に「走り直し: 本・時刻」と書く',
          '`--skip 記録名@AI側にした秒`', '■ で止めて HS に置き、同じベンチマークでもう 1 本')
    for x in rb[rb.index('## 1. 時間割'):].splitlines():         # 「人側にしてから AI 側に戻す」を言う行には、置き直しの文が同じ行にある
        if 'CH3 を人側にしてから AI 側に戻す' in x and 'HS のいつもの場所に置き直してから' not in x:
            p.append('手順書: 「CH3 を人側にしてから AI 側に戻す」の行に HS に置き直す文が無い: %s' % x[:60])
    _miss(p, '7 章 当たって止まったあと', one(ch7, '| 当たって止まったあと、その場で AI 側に戻したくなった |'),
          '**リカバリー中止(「中立で停止中」)で止まったあとも同じ**', '止まってから 3 周を数えるなら ■ で止めてもう 1 本')
    _miss(p, '8.1 ⑤ の読み方', one(ch81, '| ⑤ |', '「記録が欠けている本 …」='),
          '「リカバリー中止のあとの走り直し N 本」= 同じ本の一つ前の AI の区間がリカバリー中止で終わり', 'HS に置き直しても入れない',
          '一つ前の区間が中立で停止中(中立止め 0.5 秒以上・カメラの途切れ)のまま人側にして止まって終わった発進も入る',
          '**ON の走り直しの空転・最初の周の記録の欠けも比べの条件に数える**',
          '「  → リカバリー中止のあとの走り直しの ON の発進が空転した N 本 …」= 空転の条件は満たさない',
          '「  → リカバリー中止のあとの走り直しの ON の発進で空転の比が出ない N 本」= slip_all_measured で満たさない',
          '「。リカバリー中止のあとの走り直し」= on_all_readable で満たさない',
          'うち ON の最初の周の衝突は数える)', 'R1 に届かなかった走り直しは「R1 に届かなかった発進」の行に入り(ON なら満たさない)',
          '「うちリカバリー中止のあとの走り直し N」', '「中止のあとの走り直し(中止 X 秒・人側 Y 秒。…)」', '`--skip 記録名@AI側にした秒`')
    _miss(p, '8.1 ③ の読み方', one(ch81, '| ③ | 「数えない(リカバリー中止のあとの走り直し。中止 X 秒・人側 Y 秒)」='),
          '最速・④ の 1 章・2 章(止まってから 3 周)に入れない', '「- リカバリー中止のあとの走り直し = …」の注',
          '「中立止め X 秒から Y 秒」「カメラの途切れ X 秒から Y 秒」= 中立で停止中のまま人側にしたあとの発進')
    _miss(p, '8.1 ④ の読み方', one(ch81, '| ④ | 1 章の印「数えない発進 N(リカバリー中止のあとの走り直し)」='),
          '「前に衝突」とは別', '2 章の「動く → A1 中央値」にも入れない', '「中立で停止中 … のあとの発進も同じ」')
    ap3 = chapter(d[AP], '## 3. 新しい安全のスイッチ', '## 4. ')
    _miss(p, 'approvals 3 困ること(ON)', one(ap3, '| 困ること(ON) |'), '戻した発進は ⑤ の比べ・③ の止まってから 3 周に数えない',
          'どの回でも、AI 側に戻す前に人側で HS に置き直す', '人側 → AI 側で発進の前進がもう一度出る',
          '中立で停止中のまま人側にしたあとも同じ', '戻した ON の発進の最初の周の衝突・空転・最初の周の記録の欠けは ⑤ が数える')
    _miss(p, 'todo 3 章', chapter(d[TODO], '## 3. 11/3 の当日', '## 4. '),
          '「中立で停止中」(リカバリー中止)で止まったあとも同じ(HS に置き直してから AI 側', '戻したら紙に「走り直し: 本・時刻」')
    return p


FIX7_WORDS = ('19:36',)


def code_cmds(d):
    """(手順書 3.1 の「起動のあとに直したコード」のコマンド, todo 4 章の 5 のコマンド)。無ければ ''"""
    t31 = chapter(d[RB], '### 3.1', '## 4. ')
    m = re.search(r'```bash\n# 起動のあとに直したコード[^\n]*\n([^\n]*)\n```', t31)
    td = d[TODO]
    t4 = td[td.index('## 4. jetformer-web の再起動の手順と確かめ'):]
    k = t4.find('\n# 5. 起動のあとに直したコードが無いか')
    c2 = ''
    if k >= 0:
        c2 = next((y for y in t4[k + 1:].splitlines() if y.strip() and not y.startswith('#')), '')
    return (m.group(1) if m else ''), c2


def g2_problems(d, tmp=None):
    """w2-docs-r72-docs-2: 19:36 からの fix7 安全の直しは今のサーバー(17:16 の起動)に無い。3.1 の表で見分けられる"""
    p = []
    rb, td, rd, ap, f6 = d[RB], d[TODO], d[RD], d[AP], d[F6]
    t31 = chapter(rb, '### 3.1', '## 4. ')
    _miss(p, '3.1 の表', one(t31, '| **起動のあとに直したコード**'), '「起動より新しいコードなし」', '今のサーバーに入っていない',
          'GET・journal・画面は正しい形のままなので、この行でしか分からない', '再起動して')
    c1, c2 = code_cmds(d)
    if not c1 or c1 != c2:
        p.append('3.1 と todo 4 章の 5 のコマンドが無いか違う')
    for w in ('ActiveEnterTimestamp', "pathlib.Path('/home/jetson/jetformer/jetformer').rglob('*.py')", 'st_mtime > s',
              '起動より新しいコードなし'):
        if w not in c1:
            p.append('3.1 のコマンドに「%s」が無い' % w)
    places = [
        ('手順書 1 章', chapter(rb, '## 1. 時間割', '## 2. ')),
        ('手順書 3.1 の 17:16 の記録', one(t31, '- **10/27 17:16:28 に再起動した**')),
        ('手順書 9 章 rec_hold_keep', one(chapter(rb, '## 9. まだ無いもの', '\n| 安全の理由で'), '| 中立止めでリカバリーの中止を解かない')),
        ('todo 1 章 再起動', one(td, '| 再起動は**済**')),
        ('todo 4 章', chapter(td, '## 4. jetformer-web の再起動の手順と確かめ', '```bash')),
        ('README 0 章 8', one(rd, '| 8 | リカバリーの穴')),
        ('README 6.2 の 8', one(chapter(rd, '### 6.2', '### 6.3'), '| 8 |')),
        ('README 6.2 の 26', one(chapter(rd, '### 6.2', '### 6.3'), '| 26 |')),
        ('README 6.4', one(rd, '| 再起動で 10/26〜10/27 のサーバーの直しが動く')),
        # 2026-10-28 fix8 docs r2(f8r2-docs-approvals-restart-stale): この行は「前の今(10/27 17:3x。…)」に名前を変えた(中身の文は同じ。
        # 下の「**2 は済み**(10/27 17:16:28 …」も同じ行で見る)
        ('approvals 今の行', one(ap, '- **今(10/27 17:3x。') or one(ap, '- 前の今(10/27 17:3x。')),
        ('approvals 0 章 2', one(chapter(ap, '## 0. 一覧', '## 1. '), '| 2 |')),
        ('approvals 2 今', one(chapter(ap, '## 2. jetformer-web の再起動の日', '## 3. '), '| 今 |')),
        ('fix6/README 7 章 1', one(chapter(f6, '## 7. 次に', '### 今のファイル'), '| 1 |')),
    ]
    for where, t in places:
        if '19:36' not in t or not any(w in t for w in ('入っていない', 'もう一度', '入る', '次の再起動のあと')):
            p.append('%s: 19:36 からの fix7 安全の直しが今のサーバーに無いことを書いていない' % where)
    for where, t in (('手順書 3.1 の 17:16 の記録', places[1][1]), ('todo 4 章', places[4][1]), ('approvals 2 今', places[11][1]),
                     ('README 6.2 の 8', places[6][1]), ('todo 1 章 再起動', places[3][1])):
        _miss(p, where, t, 'Jetson を入れ直さない日は main session が再起動する')
    _miss(p, 'todo 4 章の頭', places[4][1], "(下の 3'・4・5 でもよい)")
    for where, t in (('手順書 3.1 の 17:16 の記録', places[1][1]), ('todo 4 章', places[4][1]), ('approvals 2 今', places[11][1])):
        _miss(p, where, t, '7-1 回目 19:36・7-2 回目 22:02', '本当に下がっている(0.2 秒続けて後ろへ動く)のを見なければ')
    _miss(p, 'approvals 今の行', places[9][1], '**2 は済み**(10/27 17:16:28 に再起動。**10/27 19:36 からの fix7 安全の直しのため')
    for f, t in d.items():                                      # 「全部いっしょに … 入った」は 17:16 までの分だと書く
        for x in t.splitlines():
            if '全部いっしょに' in x and '入った' in x and '17:16 までに直した分' not in x:
                p.append('%s: 「全部いっしょに入った」に 17:16 までの分の断りが無い: %s' % (f, x[:60]))
    return p


def g3_problems(d, tmp=None):
    """w2-docs-r72-docs-3: 朝の確かめ(本「—」)の「なし」の区切りは起こし直しではない"""
    p = []
    rb = d[RB]
    ch3 = chapter(rb, '## 3. 朝の準備', '### 3.1')
    ch81 = chapter(rb, '### 8.1', '### 8.2')
    _miss(p, '3 章 5', one(ch3, '  5. CH3 を人側 → ■ 停止。'), '② `runs_1103` の表には本が「—」の行として出て',
          '「20261103-HHMMSS(No.1)から なし → 1-1(No.2)から 後ろの見張り・リカバリーを残す(前の本と違う)」の 2 区切りになる',
          'この「なし」の区切りは起こし直しではない')
    if '(説明欄に block が無いので、夜の道具はどの回の本にも入れない)。' in rb:
        p.append('3 章 5: 「夜の道具はどの回の本にも入れない」だけの文が残っている(② の区切りには出る)')
    _miss(p, '3 章 わざと OFF にした日', one(ch3, '  - **わざと OFF にした日の、本 X から先の読み方**'),
          '本が「—」の記録(朝の発進の前進の確かめを収集したまま走った試し。上の 5)は、「紙の本 X より前に OFF の本」に数えない')
    _miss(p, '8.1 ②', one(ch81, '| ② |'),
          '**本が「—」の記録(朝の発進の前進の確かめを収集したまま走った試し。3 章の 5)から始まる「…(No.1)から なし」の区切りは'
          '起こし直しではない**', '「→ 1-1(No.2)から 後ろの見張り・リカバリーを残す(前の本と違う)」からが朝の形')
    return p


def block81(rb):
    ch81 = chapter(rb, '### 8.1', '### 8.2')
    i = ch81.index('```bash\n') + len('```bash\n')
    return ch81[i:ch81.index('\n```', i) + 1]


FAKE_PY = r'''#!/bin/bash
f="$2"
if [ -f "$f" ]; then echo "RUN $PWD $f" >> "$LOG"; else echo "NOFILE $PWD $f" >> "$LOG"
  echo "python3: can't open file '$PWD/$f': [Errno 2] No such file or directory" >&2; exit 2; fi
case "$f" in *quick_eval.py) exit "${FAIL_QE:-0}";; *collect.py) exit "${FAIL_COLLECT:-0}";; esac
exit 0
'''
FAKE_JFH = '#!/bin/bash\nwhile [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done; shift; exec "$@"\n'
TREE = ('experiments/20261024_tune/quick_eval.py', 'experiments/20261026_fix/plan/runs_1103.py', 'tools/standing_3lap.py',
        'tools/quick_eval_by_params.py', 'experiments/20261026_fix/diag_launch/collect.py')


def run_block(block, tmp, fail_qe='1', fail_collect='130'):
    """手順書 8.1 の塊を 2 回(休み 2 回)本物の bash で流す。python3・jf-heavy だけ作り物(開くファイルがあるかを見て、
    ① と ⑤ は指定の終了コード)。HOME は作り物の場所(~/jetformer に空のファイル 5 本)。戻り値: 問題の一覧"""
    tmp = Path(tmp)
    home, b, log = tmp / 'home', tmp / 'bin', tmp / 'log.txt'
    for rel in TREE:
        (home / 'jetformer' / rel).parent.mkdir(parents=True, exist_ok=True)
        (home / 'jetformer' / rel).write_text('', encoding='utf-8')
    b.mkdir(exist_ok=True)
    for name, body in (('python3', FAKE_PY), ('jf-heavy', FAKE_JFH)):
        (b / name).write_text(body, encoding='utf-8')
        (b / name).chmod(0o755)
    if log.exists():
        log.unlink()
    if '/home/jetson' in block.replace('/home/jetson/bin/jf-heavy', ''):
        return ['塊に作り物にできない /home/jetson の道が残っている']
    text = block.replace('/home/jetson/bin/jf-heavy', str(b / 'jf-heavy'))
    cmd = text + 'echo "CWD $PWD" >> "$LOG"\n' + text + 'echo "CWD $PWD" >> "$LOG"\n'
    env = {'HOME': str(home), 'PATH': '%s:/usr/bin:/bin' % b, 'LOG': str(log), 'FAIL_QE': fail_qe, 'FAIL_COLLECT': fail_collect,
           'LANG': 'C.UTF-8'}
    r = subprocess.run(['bash', '-c', cmd], env=env, cwd=str(tmp), capture_output=True, text=True, timeout=60)
    got = log.read_text(encoding='utf-8').splitlines() if log.exists() else []
    jf = home / 'jetformer'
    want = (['RUN %s/experiments/20261024_tune quick_eval.py' % jf, 'RUN %s experiments/20261026_fix/plan/runs_1103.py' % jf,
             'RUN %s tools/standing_3lap.py' % jf, 'RUN %s tools/quick_eval_by_params.py' % jf,
             'RUN %s/experiments/20261026_fix/diag_launch collect.py' % jf, 'CWD %s' % jf] * 2)
    p = []
    if got != want:
        p.append('8.1 の塊(① %s・⑤ %s で終わる): 流れた順が違う: %s' % (fail_qe, fail_collect, [x.replace(str(tmp), '') for x in got]))
    if "can't open file" in r.stderr or 'No such file or directory' in r.stderr:
        p.append('8.1 の塊: %s' % r.stderr.strip().replace(str(tmp), '')[:200])
    return p


def g4_problems(d, tmp=None):
    """w2-docs-r72-docs-4: 8.1 の塊は ① や ⑤ が落ちても(1・130)、次の行と次の休みの塊が ~/jetformer で回る"""
    p = []
    rb = d[RB]
    blk = block81(rb)
    for x in blk.splitlines():
        if x.startswith('#') or 'cd ' not in x:
            continue
        if x.strip() != 'cd ~/jetformer' and not (x.startswith('( cd ~/jetformer/') and x.rstrip().endswith(' )')):
            p.append('8.1 の塊: cd が ( ) の外に残る行: %s' % x[:60])
    _miss(p, '8 章の頭', chapter(rb, '## 8. 休みと夜の集計', '### 8.1'), '8.1 の塊は ① と ⑤ の cd を ( ) の中でするので')
    if tmp is not None:
        p += run_block(blk, tmp)
    return p


def g5_problems(d, tmp=None):
    """w2-docs-r72-docs-5: approvals 8 の表に形の分かれ・中止のあとの走り直し"""
    p = []
    ap8 = chapter(d[AP], '## 8. 回 3 の判定の読み方', '## 9. ')
    _miss(p, 'approvals 8 形の分かれ', one(ap8, '| 同じ回の本が、天井・左・自動合わせ・目標の違いで組に分かれた'), 'その回のどの組も満たさない',
          'checks の one_form', '10/27 fix5 c3')
    _miss(p, 'approvals 8 走り直し', one(ap8, '| リカバリー中止のあとの走り直し(同じ本の中で'), '中立で停止中のまま人側にしたあと',
          '比べ(起点まで・失う・動く・LH)に入れない(HS に置き直しても', 'ON の最初の周の衝突は数える',
          'ON の走り直しの空転(比 > 1.3・比が出ない)・最初の周の記録の欠けも比べの条件に数える(満たさない)',
          'R1 に届かなかった走り直しは「R1 に届かなかった発進」で満たさない')
    _miss(p, 'approvals 8 の注', ap8, '「このまま」はこの 2 行も含む')
    _miss(p, 'todo 2 章 13', one(d[TODO], '| 13 |'), '形の分かれは「満たさない」', 'リカバリー中止のあとの走り直しは比べに入れない',
          'approvals 8 の表に 2 行')
    _miss(p, 'README 0 章 10', one(d[RD], '| 10 | 夜の道具の判定の穴'), 'リカバリー中止のあとの走り直しも直した')
    return p


def g6_problems(d, tmp=None):
    """同じ根の足し: fix7 safety 7 章 2・3、fix7 ui 10 章の「手順書に足す文」"""
    p = []
    rb = d[RB]
    ch3 = chapter(rb, '## 3. 朝の準備', '### 3.1')
    _miss(p, '3 章 最初のリカバリー', one(ch3, '  - 見張り・残すが ON の日(11/3)は、**その日の最初のリカバリーを人が見る**'),
          '前の壁に当たって止まったあと、ブレーキ(後退側)を 1 秒より長く出したまま下がらず、そのまま前進で壁を押す(約 0.5 秒)のも、'
          'おかしな動き')
    _miss(p, 'todo 3 章', chapter(d[TODO], '## 3. 11/3 の当日', '## 4. '), 'ブレーキ(後退側)を 1 秒より長く出したまま下がらず')
    _miss(p, '8 章の頭', chapter(rb, '## 8. 休みと夜の集計', '### 8.1'), '`what` = `back_no_esc`', '`early_none`',
          '`backed_s` が 0 より大きい `early_none` は、後ろへ動いたサンプル(跳ね)は見たが 0.2 秒続かず')
    ch7 = chapter(rb, '## 7. 止める条件', '## 8. ')
    _miss(p, '7 章 起こし直した', one(ch7, '| **jetformer-web か Jetson が起こし直した**'),
          '**起動のあとに入れ直した設定があります(今は …)**', '足りないものだけ入れ直す(ON のスイッチを押すと OFF になる',
          '「…は ON(起動のあとに入れ直し済み)」「今 X(既定 Y)」', '「…は既定に戻っています」なら全部既定',
          '「**走行は始まっていません(停止中)。Failed to fetch …。管理画面のサーバーが…**」', '朝の設定を入れ直してから ▶(10/27 fix7 ui r2)')
    _miss(p, '7 章 Failed to fetch', one(ch7, '| 走行タブを押したら「Failed to fetch」'),
          '「モデルの一覧を読めていません(…TensorRT エンジンが無いという意味ではありません)…」', '**エンジンは作り直さない**',
          '10/27 21:58 より前の古い画面')
    return p


CHECKS = {'g1': g1_problems, 'g2': g2_problems, 'g3': g3_problems, 'g4': g4_problems, 'g5': g5_problems, 'g6': g6_problems}


# ================================================================== 文書
@pytest.mark.parametrize('name', sorted(CHECKS))
def test_live_documents(live, name, tmp_path):
    assert CHECKS[name](live, tmp_path) == []


@pytest.mark.parametrize('name', sorted(CHECKS))
def test_before_copies_fail(before, name, tmp_path):
    """直す前の写し(10/27 18:54 の 7-1 回目のあと)では、6 つの確かめがどれも落ちる"""
    assert CHECKS[name](before, tmp_path) != []


def test_g4_block_survives_each_failure(live, before, tmp_path):
    """① か ⑤ が 1(落ちた)・130(Ctrl-C)で終わっても、どちらも 0 でも、塊は 2 回とも全部の行を ~/jetformer で回す。
    直す前の塊は、① が落ちると ②〜④ が「can't open file」、⑤ が cd で落ち、cwd が experiments/20261024_tune に残った(指摘の形)"""
    blk, old = block81(live[RB]), block81(before[RB])
    for i, (qe, co) in enumerate((('0', '0'), ('1', '0'), ('130', '0'), ('0', '1'), ('0', '130'), ('1', '130'))):
        assert run_block(blk, tmp_path / ('n%d' % i), qe, co) == [], (qe, co)
    bad = run_block(old, tmp_path / 'old', '1', '0')
    assert bad and any("can't open file" in x for x in bad)
    assert run_block(old, tmp_path / 'old0', '0', '0') == []          # 落ちなければ前の塊も回る(落ちたときだけの指摘)


def test_script_edits_all_done(live, script, fix8, s1, r82, fin):
    """直しの台本の 43 個の直しは、今の文書でどれも「済」(もう一度流しても何もしない)。7-1 回目の 44 個と画面の版 9 の直しも済のまま
    (7-1 回目の new の文は、どれも今の文書にちょうど 1 回ある)。
    2026-10-28 fix8 docs-decided: fix8 が意図して書き換えた所(SUPERSEDE)は書き換えを戻した文書で照らす。書き換えた文は今の文書に
    ちょうど 1 回(消した文は 0 回)。
    2026-10-28 fix8 docs 同期 1: 戻すのは 同期 1 → fix8 docs-decided の順(同期 1 の書き換えも今の文書にちょうど 1 回)。
    2026-10-28 fix8 docs r2: その前に fix8 docs r2 の書き換えを戻す(fix8 docs r2 の書き換えは今の文書にちょうど 1 回。fix8 docs-decided・
    同期 1 の書き換えは、fix8 docs r2 を戻した文書で照らす = fix8 docs r2 が approvals 0 章の 2 の中を書き換えたため)。
    2026-10-28 fix8 docs 仕上げ: さらにその前に仕上げの書き換えを戻す(仕上げの書き換えは今の文書にちょうど 1 回。前の 3 つの書き換えは
    仕上げを戻した文書で照らす = 仕上げが前の係の文の中も書き換えたため)"""
    assert fin.superseded_present(live) == []
    livef = fin.revert(live)
    assert r82.superseded_present(livef) == []
    assert fix8.superseded_present(r82.revert(livef)) == [] and s1.superseded_present(r82.revert(livef)) == []
    old = fix8.revert(s1.revert(r82.revert(livef)))
    _, rep = script.apply_text(old)
    assert len(rep) == 45 and [s for _, _, s in rep] == ['done'] * 45, [(f, o[:30], s) for f, o, s in rep if s != 'done']
    assert script.r1_intact(old) == []


# 台本の (文書, 何番目) → それを元に戻すと落ちる確かめ。書いていない 5 つは頭の記録の行だけ(「済」の試験が見る)
CATCH = {(RB, 1): 'g2', (RB, 2): 'g3', (RB, 3): 'g6', (RB, 4): 'g3', (RB, 5): 'g2', (RB, 6): 'g2', (RB, 7): 'g2', (RB, 8): 'g1',
         (RB, 9): 'g1', (RB, 10): 'g6', (RB, 11): 'g6', (RB, 12): 'g4', (RB, 13): 'g3', (RB, 14): 'g4', (RB, 15): 'g4',
         (RB, 16): 'g1', (RB, 17): 'g1', (RB, 18): 'g1', (RB, 19): 'g2', (RB, 20): 'g6', (RB, 21): 'g6',
         (TODO, 1): 'g2', (TODO, 2): 'g2', (TODO, 3): 'g2', (TODO, 4): 'g2', (TODO, 5): 'g1', (TODO, 6): 'g6', (TODO, 7): 'g5',
         (RD, 1): 'g2', (RD, 2): 'g5', (RD, 3): 'g2', (RD, 4): 'g2', (RD, 5): 'g2',
         (AP, 0): 'g2', (AP, 2): 'g2', (AP, 3): 'g2', (AP, 4): 'g1', (AP, 5): 'g5', (AP, 6): 'g5',
         (F6, 1): 'g2'}
LOG_ONLY = {(RB, 0), (TODO, 0), (RD, 0), (AP, 1), (F6, 0)}


def test_reverting_each_edit_is_caught(live, script, tmp_path, fix8, s1, r82, fin):
    """わざと壊す: 台本の直しを 1 つずつ元の文に戻すと、その指摘の確かめが落ちる(40 個)。戻さなければ通る。
    2026-10-28 fix8 docs-decided: fix8 が書き換えた直しは、書き換えたあとの形の new(fix8.effective)を元の文に戻す。
    2026-10-28 fix8 docs 同期 1: 同期 1 が書き換えた直し(3 章「わざと OFF にした日」)は、その書き換えのあとの形(s1.effective)。
    2026-10-28 fix8 docs r2: その上の書き換え(r82.effective)のあとの形"""
    n = 0
    for f, edits in script.ALL:
        for i, (old, new) in enumerate(edits):
            key = (f, i)
            if key not in CATCH:
                assert key in LOG_ONLY, key
                continue
            new = fin.effective('r2', f, i, r82.effective('r2', f, i, s1.effective('r2', f, i, fix8.effective('r2', f, i, new))))   # + 仕上げ
            assert live[f].count(new) == 1, key
            broken = dict(live, **{f: live[f].replace(new, old, 1)})
            assert CHECKS[CATCH[key]](broken, tmp_path / ('r%s_%d' % (f.replace('/', '_'), i))) != [], key
            n += 1
    assert n == len(CATCH) == 40


def test_more_breakages_are_caught(live, script, tmp_path):
    """わざと壊す(台本の直しと別の形)"""
    rb = live[RB]
    ch7_end = rb.index('| いつでも |')
    blk = block81(rb)
    cases = [
        ('g1', RB, rb[:ch7_end] + '| 新しい行 | 止まったら CH3 を人側にしてから AI 側に戻す |\n' + rb[ch7_end:]),   # 置き直しの無い戻し方
        ('g1', AP, live[AP].replace('どの回でも、AI 側に戻す前に人側で HS に置き直す', 'その場で AI 側に戻す')),
        ('g1', RB, rb.replace('HS に置き直しても数えない', 'HS に置き直せば数える', 1)),
        ('g2', TODO, live[TODO].replace(script.CMD_CODE, script.CMD_CODE.replace('st_mtime > s', 'st_mtime < s'))),
        ('g2', RB, rb.replace("rglob('*.py')", "glob('*.py')", 1)),                  # サブフォルダ(web・sensors)を見ない
        ('g2', AP, live[AP].replace('Jetson を入れ直さない日は main session が再起動する', 'main session が見る')),
        ('g2', RD, live[RD] + '\n- 10/26〜10/27 の直しは全部いっしょに再起動で入った。\n'),
        ('g3', RB, rb.replace('この「なし」の区切りは起こし直しではない', 'この「なし」の区切りは起こし直し')),
        ('g4', RB, rb.replace('( cd ~/jetformer/experiments/20261026_fix/diag_launch && ', 'cd ~/jetformer/experiments/20261026_fix/'
                              'diag_launch && ').replace('--out $R/launches_20261103_ig.json )', '--out $R/launches_20261103_ig.json')),
        ('g4', RB, rb.replace(blk, blk.replace('python3 -I tools/standing_3lap.py', 'cd tools && python3 -I standing_3lap.py'))),
        ('g5', AP, live[AP].replace('checks の one_form。10/27 fix5 c3) |\n', '10/27 fix5 c3) |\n')),
        ('g6', RB, rb.replace('ON のスイッチを押すと OFF になる', 'スイッチを押す')),
    ]
    for k, (name, f, t) in enumerate(cases):
        assert t != live[f], (k, name, f)
        assert CHECKS[name](dict(live, **{f: t}), tmp_path / ('m%d' % k)) != [], (k, name, f)


# ================================================================== 文書の言葉と道具・コードの本当の出力
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF / 'tests'))

PRM = {'throttle_gain': 0.9, 'steer_gain': 1.0, 'speed_ceiling': 0, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 0,
       'auto_target_mps': 4.27, 'launch_boost': 0, 'launch_us': 250.0, 'launch_mps': 2.5, 'launch_max_s': 1.2, 'zone_caps': 0,
       'start_kick': 1, 'rec_back_guard': 1, 'rec_hold_keep': 1, 'launch_no_kick': 0}
ON = dict(PRM, launch_boost=1)
DESC = 'ai_no=%d bench=real_bench_20261023 day=2026-11-03 block=%d cand=EXP model=runB memo=x'


def round3_day(root, case):
    """作り物の回 3(fix7 tools r1 の台本 gen_course と同じ作り方。1 周 18 m・2 m/s)。3-8 は 2 周目の R1 の 4.5 m 手前でカメラが 0.8 秒
    止まり、R1 の 3.5 m 手前で止まる → リカバリー中止 2 秒 → 人側 1.0 秒 → その場で AI 側。case: 'full' = 3 周走る / 'short' = 走り直しを
    1.0 秒で人側にしてそのまま止めた(R1 に届かない)/ 'stall' = 中止にならず、カメラが止まったまま人側にした(人側にした 0.5 秒あと
    まで止まる。手順書 7 章「中立で停止中 → 車に近づく前に CH3 を人側」の形)→ その場で AI 側・3 周 / 'split' = 走り直しなし・3-5 から
    左 4.1(形の分かれ。天井 ON・左 4.2 の日)"""
    gc = _load('gen_course_fix7_docs_r2', GEN)
    raw = root / 'raw'
    raw.mkdir(parents=True)
    bench = root / 'bench.json'
    bench.write_text(json.dumps({'candidates': [{'id': 'B', 'run': 'runB'}],
                                 'days': [{'date': '2026-11-03', 'blocks': [{'block': b, 'run_id': '3-%d' % (b - 11)}
                                                                            for b in range(12, 20)]}]}), encoding='utf-8')
    logs = []

    def add(block, prm, d_move, ramp, **kw):
        log = '20261103-13%02d00' % (len(logs) * 5)
        dur = kw.pop('dur', 30.0)
        vk = kw.pop('vk', None) or gc.launch_keys(1.0, d_move, ramp, until=dur)
        ai = kw.pop('ai', None) or [(1.0, dur - 2.0)]
        launch = kw.pop('launch', [(1.0, 0.5)] if prm.get('launch_boost') else [])
        gc.write_log(raw, log, dur, vk, ai, prm, DESC % (len(logs) + 1, block), launch=launch, **kw)
        logs.append(log)
        return log
    if case == 'split':
        for i, b in enumerate(range(12, 20)):
            on = b % 2 == 1
            prm = dict(PRM, speed_ceiling=1, ceil_left_mps=4.2 if b < 16 else 4.1, launch_boost=1 if on else 0)
            add(b, prm, *((0.15 + 0.03 * i, 0.3) if on else (0.20 + 0.01 * i, 0.6)))
        return raw, bench, None, None
    for b, prm, dm, rp in ((12, PRM, 0.20, 0.6), (13, ON, 0.15, 0.3), (14, PRM, 0.21, 0.6), (15, ON, 0.20, 0.3), (16, PRM, 0.22, 0.6),
                           (17, ON, 0.28, 0.3), (18, PRM, 0.23, 0.6)):
        add(b, prm, dm, rp)
    base = gc.launch_keys(1.0, 0.31, 0.3)
    t_st = gc.time_at_distance(base + [(200.0, 2.0)], 23.0 - 3.5 - 1.0, 200.0)
    t_stop, t_flip = t_st + 1.0, t_st + 3.0
    t_res = t_flip + 1.0
    if case in ('full', 'stall'):
        dur = t_res + 32.0
        vk = base + [(t_st, 2.0), (t_stop, 0.0), (t_res + 0.15, 0.0), (t_res + 0.45, 2.0), (dur, 2.0)]
        ai = [(1.0, t_flip), (t_res, dur - 1.0)]
    else:
        dur = t_res + 3.0
        vk = base + [(t_st, 2.0), (t_stop, 0.0), (t_res + 0.15, 0.0), (t_res + 0.45, 1.0), (t_res + 1.0, 1.0), (t_res + 2.0, 0.0),
                     (dur, 0.0)]
        ai = [(1.0, t_flip), (t_res, t_res + 1.0)]
    ev = [{'kind': 'recovery_guard', 'what': 'hold_long', 'phase': None, 'backed_s': 0.0, 'hold_s': 0.8, 'run': 'runB',
           'cam_epoch_us': int(gc.T0_US + round((t_st + 0.8) * 1e6)), 't_us': int(gc.T0_US + round((t_st + 0.8) * 1e6))}]
    if case == 'stall':
        log38 = add(19, ON, 0.31, 0.3, dur=dur, vk=vk, ai=ai, launch=[(1.0, 0.5), (t_res, 0.5)], stall=[(t_st, t_flip + 0.5)])
    else:
        log38 = add(19, ON, 0.31, 0.3, dur=dur, vk=vk, ai=ai, launch=[(1.0, 0.5), (t_res, 0.5)], stall=[(t_st, t_st + 0.8)],
                    giveup=[(t_st + 0.8, t_flip)], events=ev)
    return raw, bench, log38, t_res


def run5(root, raw, bench, tag):
    from test_fix3_r1_2 import load_mod
    m = load_mod('collect_fix7_docs_r2_%s' % tag, COLLECT)
    m.RAW, m.LAB, m.BENCH, m.OUT = raw, root / 'nolab', bench, root / 'out'
    buf = io.StringIO()
    out = root / ('%s.json' % tag)
    with contextlib.redirect_stdout(buf):
        m.main(['--day', '20261103', '--ignore-gain', '--out', str(out)])
    return json.loads(out.read_text(encoding='utf-8')), buf.getvalue()


@pytest.fixture(scope='module')
def tools_out(tmp_path_factory):
    """作り物の回 3 を本物の ⑤ collect.py(--ignore-gain)・③ standing_3lap・④ quick_eval_by_params に通した出力"""
    import standing_3lap as s3
    import quick_eval_by_params as qp
    res = {}
    for case in ('full', 'short', 'split', 'stall'):
        root = tmp_path_factory.mktemp('r3_' + case)
        raw, bench, log38, t_res = round3_day(root, case)
        j, out = run5(root, raw, bench, case)
        r = {'j': j, 'out': out, 'log38': log38, 't_res': t_res}
        if case in ('full', 'stall'):
            logs, _ = s3.day_logs('20261103', raw)
            R = s3.run(logs, raw, root / 'nolab', None)
            r['s3'] = [x for x in R if x['log'] == log38][0]['standing']
            r['s3_md'] = s3.to_markdown(R)
            r['qp_md'] = qp.to_markdown(R, '20261103', None, ('gain',))[0]
        res[case] = r
    return res


def test_g1_tools_print_what_the_runbook_quotes(live, tools_out):
    """手順書 7 章・8.1 が引く ⑤③④ の言葉(N・X・Y は数)を、本物の道具が作り物の回 3 で出す。道具の文の「手順書 7 章」は走り直しを説明する"""
    rb = live[RB]
    full, short = tools_out['full'], tools_out['short']
    out = full['out']
    g = full['j']['summary'][0]
    assert g.get('abort_resume_excluded') and len(g['abort_resume_excluded']) == 1         # 走り直しは比べに入らない
    line = [x for x in out.splitlines() if x.startswith('  リカバリー中止のあとの走り直し ')]
    assert len(line) == 1 and line[0].startswith('  リカバリー中止のあとの走り直し 1 本(') and '手順書 7 章' in line[0]
    assert '走り直し' in chapter(rb, '## 7. 止める条件', '## 8. ')
    assert re.search(r' 中止のあとの走り直し\(中止 [\d.]+ 秒・人側 [\d.]+ 秒。', out)                # 「中止のあとの走り直し(中止 X 秒・人側 Y 秒。…)」
    s38 = full['s3']
    assert [s['counted'] for s in s38 if s.get('after_abort')] == [False]                    # ③ は数えない
    assert re.search(r'数えない\(リカバリー中止のあとの走り直し。中止 [\d.]+ 秒・人側 [\d.]+ 秒\)', full['s3_md'])
    assert any(x.startswith('- リカバリー中止のあとの走り直し = ') for x in full['s3_md'].splitlines())   # ③ の表の下の注
    assert '数えない発進 1(リカバリー中止のあとの走り直し)' in full['qp_md']                              # ④ 1 章の印
    assert any(x.startswith('- リカバリー中止のあとの走り直し(') for x in full['qp_md'].splitlines())    # ④ 2 章の下の行
    # R1 に届かなかった走り直し: 「R1 に届かなかった発進」の行の後ろに「うちリカバリー中止のあとの走り直し N」・ON なら満たさない
    so = short['out']
    nr = [x for x in so.splitlines() if 'R1 に届かなかった発進' in x]
    assert nr and any('。うちリカバリー中止のあとの走り直し 1(' in x for x in nr)
    assert short['j']['summary'][0]['compare']['all_ok'] is False
    # 中立で停止中のまま人側にした形(カメラが止まったまま人側): 同じ名前で数えない。形の文が「カメラの途切れ X 秒から Y 秒」
    st = tools_out['stall']
    assert st['j']['summary'][0].get('abort_resume_excluded')
    assert any(x.startswith('  リカバリー中止のあとの走り直し 1 本(') for x in st['out'].splitlines())
    assert re.search(r' 中止のあとの走り直し\(カメラの途切れ [\d.]+ 秒から [\d.]+ 秒・人側 [\d.]+ 秒。', st['out'])
    assert [s['counted'] for s in st['s3'] if s.get('after_abort')] == [False]
    assert re.search(r'数えない\(リカバリー中止のあとの走り直し。カメラの途切れ [\d.]+ 秒から [\d.]+ 秒・人側 [\d.]+ 秒\)', st['s3_md'])
    assert any(x.startswith('- 「中立止め X 秒から Y 秒」「カメラの途切れ X 秒から Y 秒」のリカバリー中止のあとの走り直し = ')
               for x in st['s3_md'].splitlines())
    assert '中立で停止中(中立止め 0.5 秒以上・カメラの途切れ)のまま人側にして止まって終わった区間のあとの発進も同じ' in st['qp_md']
    st3 = (JF / 'tools' / 'standing_3lap.py').read_text(encoding='utf-8')
    assert "STOP_KIND_TEXT = {'abort': '中止', 'hold': '中立止め', 'stall': 'カメラの途切れ'}" in st3     # 「中立止め X 秒から Y 秒」も
    # ON の走り直しの空転・欠け(⑤ の行の言葉。手順書 8.1 ⑤ の引用)
    co = COLLECT.read_text(encoding='utf-8')
    for w in ("'  → %sの ON の発進が空転した %d 本(", "'  → %sの ON の発進で空転の比が出ない %d 本: %s → 空転を確かめられない = 満たさない(checks の slip_all_measured)",
              "'。' + RESUME_NAME if r.get('after_abort')", "RESUME_NAME = 'リカバリー中止のあとの走り直し'"):
        assert w in co, w
    for q in ('「リカバリー中止のあとの走り直し N 本」', '「うちリカバリー中止のあとの走り直し N」', '「中止のあとの走り直し(中止 X 秒・人側 Y 秒。…)」',
              '「数えない(リカバリー中止のあとの走り直し。中止 X 秒・人側 Y 秒)」', '「数えない発進 N(リカバリー中止のあとの走り直し)」'):
        assert q in rb, q


def test_g1_launch_boost_and_kick_as_the_runbook_says():
    """7 章の中立止めの行: 発進の前進 ON の本は人側 → AI 側にするたびに 1 回(止まっていて前が約 1 m 以上 = TTC の中央値 3.3 秒以上)、
    250 µs・最大 1.2 秒。発進の補助は前へ 80 µs・最大 2 秒(本物の assist)"""
    from jetformer import assist as A
    N = 1500.0
    assert (A.LaunchBoost.TTC_START_S, A.LaunchBoost.US_DEFAULT, A.LaunchBoost.MAX_S_DEFAULT) == (3.3, 250.0, 1.2)
    assert '止まっていれば前 1.0 m 以上空き' in Path(A.__file__).read_text(encoding='utf-8')
    assert (A.StartKick.KICK_US, A.StartKick.MAX_S) == (80.0, 2.0)

    def inp(ttc):
        return A.LaunchIn(us=250.0, mps=2.5, max_s=1.2, v=0.0, a=0.0, gz=0.0, steer_rel=0.0, ttc=ttc, ttc5=(ttc,) * 5,
                          tof_ok=True, veh_ok=True, impact_g=0.0, guard_changed=False)
    for ttc2, want in ((3.6, 2), (3.0, 1)):
        lb = A.LaunchBoost(N)
        t = 0.0
        for _ in range(10):
            lb.human()
            t += 1 / 30.0
        out, _ = lb.step(N - 40.0, 0.0, t, inp(10.0))
        assert lb.count == 1 and out == N - 250.0 and lb.event[0] == 'start'
        for _ in range(30):
            lb.human()
            t += 1 / 30.0
        out, _ = lb.step(N - 40.0, 0.0, t, inp(ttc2))
        assert lb.count == want, ttc2                       # 3.6 秒(約 1.2 m)ならもう一度出る・3.0 秒なら出ない
        assert (out == N - 250.0) == (want == 2)


def test_g2_code_check_command_names_only_newer_files(live, tmp_path):
    """3.1・todo 4 章の 5 のコマンドそのもの(パッケージの道だけ作り物に替える)を、偽物の systemctl で流す: 起動より新しい .py だけ名前が
    出る(サブフォルダも見る・app.js は見ない)。どれも古ければ「起動より新しいコードなし」"""
    cmd, _ = code_cmds(live)
    assert cmd
    pkg = tmp_path / 'pkg' / 'jetformer'
    real = "pathlib.Path('/home/jetson/jetformer/jetformer')"
    assert cmd.count(real) == 1
    cmd = cmd.replace(real, "pathlib.Path('%s')" % pkg)
    b = tmp_path / 'bin'
    b.mkdir()
    (b / 'systemctl').write_text('#!/bin/sh\necho "$FAKE_START"\n', encoding='utf-8')
    (b / 'systemctl').chmod(0o755)
    start = 'Tue 2026-10-27 17:16:28 JST'
    env = {'PATH': '%s:%s:/usr/bin:/bin' % (b, os.path.dirname(sys.executable)), 'FAKE_START': start, 'LANG': 'C.UTF-8'}
    s = int(subprocess.run(['date', '-d', start, '+%s'], capture_output=True, text=True, env=env, check=True).stdout)
    files = {'assist.py': s - 3600, 'drive.py': s + 8000, 'web/server.py': s - 100, 'sensors/vehicle.py': s + 10,
             'web/static/app.js': s + 9000}
    for rel, mt in files.items():
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text('x', encoding='utf-8')
        os.utime(pkg / rel, (mt, mt))

    def run():
        r = subprocess.run(['bash', '-c', cmd], env=env, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0 and r.stderr == '', r.stderr
        return r.stdout.splitlines()
    assert run() == ['起動より新しい: %s' % (pkg / 'drive.py'), '起動より新しい: %s' % (pkg / 'sensors' / 'vehicle.py')]
    for rel in ('drive.py', 'sensors/vehicle.py'):
        os.utime(pkg / rel, (s - 5, s - 5))
    assert run() == ['起動より新しいコードなし']


def test_g2_fix7_safety_described_is_on_disk():
    """文書が「今のサーバーに入っていない」と書く fix7 安全の直し(EscMayReverse・back_no_esc・early_none)は、ディスクのコードにあり、
    10/27 19:36 からの更新(文書の「19:36 から」)"""
    a = (JF / 'jetformer' / 'assist.py').read_text(encoding='utf-8')
    dr = (JF / 'jetformer' / 'drive.py').read_text(encoding='utf-8')
    assert 'class EscMayReverse' in a and "'back_no_esc'" in a and "'early_none'" in a and 'EscMayReverse' in dr
    assert re.search(r'^SURE_BACK_S = 0\.2 ', a, re.M) and re.search(r'^CHECK_S = 0\.35 ', a, re.M)      # 文書の「0.2 秒続けて」「0.35 秒」
    t0 = datetime(2026, 10, 27, 19, 36).timestamp()
    for f in ('assist.py', 'drive.py'):
        assert (JF / 'jetformer' / f).stat().st_mtime >= t0, f


def test_g3_runs_1103_prints_what_the_runbook_quotes(live, tmp_path, monkeypatch):
    """手順書 3 章の順(朝の発進の前進の確かめを収集したまま・memo に「試し」→ 試験の選択肢 ON → 1-1・1-2)の作り物の記録を本物の
    runs_1103 に通すと、手順書が引く「…(No.1)から なし → 1-1(No.2)から 後ろの見張り・リカバリーを残す(前の本と違う)」の行を出す"""
    m = _load('runs_1103_fix7_docs_r2', RUNS)
    raw = tmp_path / 'raw'
    m.RAW = raw
    base = {'throttle_gain': 0.9, 'speed_ceiling': 0, 'ceil_left_mps': 4.1, 'speed_auto': 0, 'launch_boost': 0, 'launch_us': 250.0,
            'steer_trim_us': 0.0, 'zone_caps': 0, 'rev_s': 0.4, 'rev_us': 300.0, 'rec_back_guard': 0, 'rec_hold_keep': 0,
            'launch_no_kick': 0}
    for name, desc, kw in (('20261103-081500', 'ai_no=1 memo=試し', {'launch_boost': 1}),
                           ('20261103-083000', 'day=2026-11-03 block=1 official=1 ai_no=2 battery=#A', {'rec_back_guard': 1, 'rec_hold_keep': 1}),
                           ('20261103-084000', 'day=2026-11-03 block=2 official=1 ai_no=3 battery=#A', {'rec_back_guard': 1, 'rec_hold_keep': 1})):
        d = raw / name
        d.mkdir(parents=True)
        (d / 'log.json').write_text(json.dumps({'description': desc, 'drive': {
            'run': 'clean_driving_20261011-nocmd-s3-20261011-024626', 'params': dict(base, **kw), 'dry_run': False}}), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['runs_1103.py', '--day', '20261103'])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.main()
    line = [x for x in buf.getvalue().splitlines() if x.startswith('試験の選択肢(')]
    assert len(line) == 1
    tail = '20261103-081500(No.1)から なし → 1-1(No.2)から 後ろの見張り・リカバリーを残す(前の本と違う)'
    assert line[0].endswith(tail)
    ch3 = chapter(live[RB], '## 3. 朝の準備', '### 3.1')
    assert '「%s」' % tail.replace('081500', 'HHMMSS') in ch3
    assert '「→ %s」' % tail.split(' → ')[1] in one(chapter(live[RB], '### 8.1', '### 8.2'), '| ② |')


def test_g5_collect_rules_the_approvals_table_lists(live, tools_out):
    """approvals 8 に足した 2 行のとおりに本物の ⑤ が言う: 形が分かれた回はどの組も満たさない(one_form)・中止のあとの走り直しは
    比べに入れない・R1 に届かなかった走り直しは満たさない"""
    sp = tools_out['split']
    assert '組に分かれた' in sp['out']
    groups = [g for g in sp['j']['summary'] if g.get('compare')]
    assert len(groups) == 2 and all(g['compare']['all_ok'] is False and g['compare']['checks'].get('one_form') is False for g in groups)
    assert any(g['compare']['checks'].get('to_r1') is True for g in groups)        # 起点までは満たすのに、形の分かれで満たさない
    assert tools_out['full']['j']['summary'][0]['abort_resume_excluded']
    assert tools_out['short']['j']['summary'][0]['compare']['all_ok'] is False
    ap8 = chapter(live[AP], '## 8. 回 3 の判定の読み方', '## 9. ')
    assert '組に分かれた' in ap8 and '走り直し' in ap8


def test_g6_words_are_in_the_code(live):
    """同じ根の足しで手順書が引く言葉が、本物の app.js・assist.py にある"""
    js = (JF / 'jetformer' / 'web' / 'static' / 'app.js').read_text(encoding='utf-8')
    for w in ('起動のあとに入れ直した設定があります(今は ', '朝の設定はスイッチと走行開始の確認の文で読む', 'は ON(起動のあとに入れ直し済み)',
              'は既定に戻っています', ' 今 ${p[k]} ${unit}(既定 ${BOOT_DEFAULTS[k]})',
              'モデルの一覧を読めていません(一覧の問い合わせが通らないか遅れています。TensorRT エンジンが無いという意味ではありません)。',
              "'走行は始まっていません(停止中)。'", 'が走行の一覧にありません(TensorRT エンジンが必要)'):
        assert w in js, w
    a = (JF / 'jetformer' / 'assist.py').read_text(encoding='utf-8')
    assert "self.guard_event = ('back_no_esc'" in a and "self.guard_event = ('early_none', 'reverse'" in a


# ================================================================== 既定値(この組はコードを変えていない)
def _assigns(src, names):
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    out[t.id] = ast.dump(node.value)
    return out


def test_defaults_unchanged():
    """drive.py の DEFAULT_PARAMS・PARAM_LIMITS・RESET_ON_BOOT が、fix7 安全の直しの前の写し(sha256 を照らす)と AST で同じ。
    文書の言う既定(試験の選択肢・発進の前進は既定 0・起動で 0)も同じ"""
    fx = JF / 'tests' / 'fixtures' / 'fix7_safety_before'
    sums = dict(reversed(x.split(None, 1)) for x in (fx / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert _sha(fx / 'drive.py') == sums['drive.py']
    names = ('DEFAULT_PARAMS', 'PARAM_LIMITS', 'RESET_ON_BOOT')
    now = _assigns((JF / 'jetformer' / 'drive.py').read_text(encoding='utf-8'), names)
    old = _assigns((fx / 'drive.py').read_text(encoding='utf-8'), names)
    assert set(now) == set(names) and now == old
    from jetformer import drive
    for k in ('rec_back_guard', 'rec_hold_keep', 'launch_no_kick', 'launch_boost'):
        assert drive.DEFAULT_PARAMS[k] == 0 and k in drive.RESET_ON_BOOT, k
