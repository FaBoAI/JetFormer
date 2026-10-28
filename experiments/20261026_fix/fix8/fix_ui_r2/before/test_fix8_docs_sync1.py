"""fix8 係 docs 同期 1(2026-10-28)の文書を読む試験(文書だけ。コードは変えていない)。台本
experiments/20261026_fix/fix8/docs_sync1/docs_fix8_sync1.py(42 個の直し。うち 20 か所は前の係の文の中の書き換え = SUPERSEDE)、
直す前の写し tests/fixtures/fix8_sync1_before/(5 本と SHA256SUMS)、報告 ~/jetformer_reports/fix_20261026/fix8/docs_sync_r1.md。

反証 8-1 回目の直しの報告の「手順書に足す文」(fix8/fix_ui_r1.md 10 章・fix_tools_r1.md 10 章・fix_safety_r1.md 3 章 A〜K)と
fix8/fix_docs_r1.md の「残したこと」を、手順書・todo・README・fix3/approvals.md・fix7/README.md に入れたかを見る:
- k0 頭の記録の 1 行(5 本とも。控えの名前)と todo の題の時刻。
- k1 安全(rs8-1-backearly-rear-faster): approvals 3「困ること(ON)・10/23 の形」(両向きの距離の帯の数・本物の 1 回・(c) を決めたときに
  無かった形・11/3 のコードは変えない・守りは人)・0 章の 3 の知らせ・手順書 3 章の行(「最初のリカバリー」の行より上)・7 章「1 m 以上
  下がる」の行の「リカバリーを待たずに人側にしてよい」・todo 3 章・README 6.2 の 33(ユーザーに伝えること)。「ON のほうが長く下がる」
  は書かない。
- k2 安全(rs8-1-real-record-backing-missed): 5 本の「235 本」は「」の中の引用だけ(235 は場面の数)・approvals 14 の 2・3 行目・
  fix7/README 1・2・3 章。
- k3 夜の道具(w1-tools-r8-1-trial-form・stall-inflates-off): approvals 3「途中で OFF にした日」の決まり・8 の表・手順書 3 章「わざと OFF
  にした日」・7 章「走った値が計画と違った」「起こし直した」・8.1 ⑤。
- k4 画面(runs-opt-mark-on-the-on-runs・stale-status-after-drive-tab-return・toast-half-width-on-phone): 手順書 3.1 の app.js の時刻・
  7 章「Failed to fetch」・8.1 の読み方の注意の ② の印・todo 3 章・README 6.2 の 11。
- k5 README 6 章の頭(反証 8-1 回目 20 → 16 件)・6.3 の 34(qC)・35(残したこと)・7 章。
- k6 10/28 08:40 の Jetson の起動し直し(main session の知らせ。jetformer-web 08:40:33): 「反証 8 回目のあとにもう一度再起動」の文
  (手順書 3.1・todo 1・4 章・README 6 章の頭・6.2 の 8・32・approvals の「今」の行と 2 の表・fix7/README 7 章の 2)に「入った」の印。
  読むだけの確かめの出力(experiments/20261026_fix/fix8/docs_sync1/out/restart_check_0840.txt)と、今のファイルの更新時刻で照らす。
- 文書の言葉・数がコードと記録にあるか(app.js・style.css・server.py・collect.py・runs_1103.py・assist.py の文字列と定数、本物の ⑤ と
  ② を作り物の日で流した出力、係の台本の出力 ana_r1.json・facts12.json・rec_concat12_*.json・scan_real_gaps.txt・heights.txt、
  指摘と票の数)。前の係の直しと試験の確かめが崩れていないか(台本の earlier_problems)。
- わざと壊す: 直す前の写しでは k0〜k6 が全部落ちる。台本の 42 個の直しを 1 つずつ元に戻すと、その確かめが落ちる。ほかの壊しも落ちる。
"""
import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pytest

JF = Path('/home/jetson/jetformer')
REPORTS = Path('/home/jetson/jetformer_reports/fix_20261026')
BEFORE = JF / 'tests' / 'fixtures' / 'fix8_sync1_before'
HERE8 = JF / 'experiments' / '20261026_fix' / 'fix8'
SCRIPT = HERE8 / 'docs_sync1' / 'docs_fix8_sync1.py'
RB, TODO, RD, AP, F7 = 'runbook_20261103.md', 'todo_user_1026.md', 'README.md', 'fix3/approvals.md', 'fix7/README.md'
FILES = (RB, TODO, RD, AP, F7)
ALL7 = (RB, TODO, RD, AP, 'fix3/disk.md', F7, 'fix6/README.md')
BAK = '.bak-20261028-fix8-sync1'
STAMP = '10/28 09:1x'
R81_AFTER = {RB: '68e710bc9681', TODO: '425d2c269b0e', RD: '0301ef5910b5', AP: 'cbd349ccaf8f', F7: '3bcfa41bb93c'}  # fix8 docs r1 の「後」

# 文書が引く言葉(コードの文字列と同じ。test_words_match_the_code が照らす)
REFUSE = 'サーバーの設定を読み込んでいます。少し待ってから押してください'
STALL = '発進から R1 の起点までにカメラが止まった発進'
HUMAN_EARLY = 'リカバリー(止まって約 0.8 秒で始まる)を待たずに CH3 を人側にしてよい'
NUMS = ('18 通り', '11,232 組', '5,184 組', 'ON が 0.1 m/s より速い 2,922 組・OFF が速い 622 組', '平均 +0.38 m/s・最大 +1.0 m/s',
        '0.30〜0.65 m だけ', '6,048 組では OFF が速い 2,020・OFF だけ当たる 3,226 に対し ON が速い 34・ON だけ当たる 8',
        '10/23 No.6 の 1 回だけ', '約 1.16〜1.22 m(ON のほうが良い距離)')


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
    return _load('docs_fix8_sync1_t6', SCRIPT)


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


def _ch3(rb):
    return _sec(rb, '## 3. 朝の準備', '### 3.1')


def _ch7(rb):
    return _sec(rb, '## 7. 止める条件', '## 8. ')


def _ap3(ap):
    return _sec(ap, '## 3. 新しい安全のスイッチ', '\n## 4. ')


# ================================================================== 確かめ(どれも問題の一覧。空なら OK)
def k0_problems(d):
    """頭の記録の 1 行(最初の見出し「## 」より前・控えの名前・報告)と todo の題の時刻"""
    p = []
    heads = {RB: '- **10/28 ', TODO: '- **10/28 ', RD: '- **10/28 ', AP: '- **10/28 ', F7: '- (10/28 '}
    for f, h in heads.items():
        t = d[f]
        top = t[:t.index('\n## ')] if '\n## ' in t else t
        line = one(top, h, 'fix8 係 docs 同期 1')
        _miss(p, '%s の頭の記録' % f, line, '控え `%s%s`' % (Path(f).name, BAK), 'docs_sync_r1.md', STAMP)
    title = d[TODO].splitlines()[0] if d[TODO] else ''
    if title != '# ユーザーが 11/3 までにすること(%s 現在)' % STAMP:
        p.append('todo の題: 同期 1 の時刻でない(%s)' % title[:40])
    return p


def k1_problems(d):
    """10/23 の形の後ろの近い物(両向きの距離の帯)・人がすること・ユーザーに伝えること"""
    p = []
    ap, rb, td, rd = d[AP], d[RB], d[TODO], d[RD]
    s3 = _ap3(ap).splitlines()
    i = next((k for k, x in enumerate(s3) if x.startswith('| 困ること(ON) |')), None)
    row = s3[i + 1] if i is not None and i + 1 < len(s3) else ''
    if not row.startswith('| 困ること(ON)・10/23 の形(10/28 知らせ。fix8 safety r1) |'):
        p.append('approvals 3: 「困ること(ON)」の行のすぐ下に「困ること(ON)・10/23 の形」の行が無い')
    _miss(p, 'approvals 3 困ること(ON)・10/23 の形', row, *NUMS)
    _miss(p, 'approvals 3 困ること(ON)・10/23 の形', row, "見張り ON は 'gap' の中立をはさまずに後退側を出し続ける",
          '**止まった所から約 1 m より近い後ろの物には、ON のほうが強く当たることがある**',
          '**約 1 m より遠い物には、ON のほうが遅く当たるか当たらない**',
          '**ユーザーが (c) を決めたとき(10/27 15:4x)の上の「困ること(ON)」には無かった形**',
          '11/3 のコードは変えない(main session の決め', '手順書 7 章の「リカバリーで車が 1 m 以上下がる・後ろの物に近づく」の行',
          '待たずに CH3 を人側にしてよい', '`fix8/proposal_safety_after1103.md` 9 章の qC', '(c) を変えたいときは一言(返事が無ければ (c) のまま)')
    _miss(p, 'approvals 0 章の 3', one(_sec(ap, '## 0. 一覧', '## 1. '), '| 3 |'), '**10/28 知らせ**', '「困ること(ON)・10/23 の形」の行',
          '返事が無ければ (c) のまま')
    # 手順書 3 章: 新しい行は「おかしな動きの今」「最初のリカバリー」の行より上
    c3 = _ch3(rb).splitlines()
    j = next((k for k, x in enumerate(c3) if x.startswith('  - **見張り ON の日の、10/23 の形の後ろの近い物**')), None)
    k = next((k for k, x in enumerate(c3) if x.startswith('  - 見張り・残すが ON の日(11/3)は、**その日の最初のリカバリーを人が見る**')), None)
    if j is None or k is None or not j < k:
        p.append('手順書 3 章: 10/23 の形の行が「最初のリカバリー」の行より上に無い')
    else:
        _miss(p, '手順書 3 章 10/23 の形', c3[j], *NUMS)
        _miss(p, '手順書 3 章 10/23 の形', c3[j], '**止まった所の後ろ約 1 m 以内に物・人・壁があるときは、' + HUMAN_EARLY + '**',
              '先に人側にしただけなら見張り・残すは OFF にしない', '7 章「リカバリーで車が 1 m 以上下がる・後ろの物に近づく」の行')
    r7 = one(_ch7(rb), '| リカバリーで車が 1 m 以上下がる・後ろの物に近づく | すぐ CH3 を人側(後ろの見張りが OFF だと')
    _miss(p, '手順書 7 章 1 m 以上下がる', r7, '止まった所の後ろ約 1 m 以内に物・人・壁があるときは、' + HUMAN_EARLY,
          'この形のリカバリーは始まるとすぐ(約 0.1 秒で)下がり始め', "見張り ON は 'gap' の中立をはさまないので、約 1 m より近い物には OFF"
          ' より強く当たることがある', '約 1 m より遠い物には ON のほうが遅く当たるか当たらない', 'その場で AI 側に戻さない',
          '紙に「走り直し: 本・時刻」', '先に人側にしただけなら見張り・残すは OFF にしない')
    t3 = _sec(td, '## 3. 11/3 の当日', '## 4. ')
    _miss(p, 'todo 3 章 10/23 の形', one(t3, '- (10/28 fix8 safety r1)**前の物の手前でブレーキのあと止まり(10/23 の形)'), HUMAN_EARLY,
          '11/3 のコードは変えない', '先に人側にしただけなら見張り・残すは OFF にしない', '紙に「走り直し: 本・時刻」')
    _miss(p, 'README 6.2 の 33', one(_sec(rd, '### 6.2', '### 6.3'), '| 33 |'), '**ユーザーに伝えること**',
          '(c) を決めたとき(10/27 15:4x)の「困ること(ON)」に無かった形', '11,232 組', '約 1.16〜1.22 m', '11/3 のコードは変えない',
          'リカバリーを待たずに人側にしてよい', '返事が無ければ (c) のまま')
    # 「ON のほうが長く下がる」は書かない(模型しだいで逆になる。fix8/fix_safety_r1.md 3 章 A のあと)
    for f in FILES:
        for w in ('ON のほうが長く下がる', 'ON の方が長く下がる', 'ON のほうが OFF より長く下がる'):
            if w in d[f]:
                p.append('%s: 「%s」と書いている' % (f, w))
    return p


REC235 = ('本物の記録(235 の場面を場面ごとに、と 1 本の記録の場面をつないだ 59 本。235 は記録の本数ではない)では起きていない'
          '(出す値が直す前と全部同じ。10/28 fix8 safety r1 がつないで照らした)')


def k2_problems(d):
    """235 は場面の数(1 本の記録の場面をつなぐと 59 本・160847 の 10 フレームだけ違い、そこで今の ON = OFF)"""
    p = []
    for f in FILES:
        if d[f].count('235 本') != d[f].count('235 本」'):
            p.append('%s: 「235 本」が引用のほかに残る' % f)
    c14 = d[AP][d[AP].index('## 14. '):] if '## 14. ' in d[AP] else ''
    _miss(p, 'approvals 14 の 2 行目', one(c14, "| 'gap' を飛ばした後退で、本当に下がり始めた車のやり直し"), REC235)
    _miss(p, 'approvals 14 の 3 行目', one(c14, '| ESC の見立て(`EscMayReverse`) |'), '**本物の ESC では確かめていない**',
          '見立てが外れる形(ESC の後退でない後ろへの動き)は本物の記録に 1 回ある: 10/24 20261024-160847 の 40.07 秒',
          'AI 側の間に約 0.13 m', '40.45 秒', '計 約 0.24 m', '(back_no_esc)', '**見張り OFF と同じ値を書いた**',
          '場面ごとの再生では場面のさかい目で見落としていた')
    f7 = d[F7]
    _miss(p, 'fix7/README 1 章の安全', one(_sec(f7, '## 1. 1 巡目の中身', '## 2. '), '- 安全(`fix_safety_r1.md`)'),
          '本物の記録の 235 の場面(238 のうち空の 3 つを除く。記録の本数ではない)', '59 本・同じ 122,767 フレーム',
          '10/24 20261024-160847 の 40.07〜40.37 秒の 10 フレームだけ違う', '**見張り OFF と同じ値**', 'hall_count −62909 → −62883',
          'ESC の後退ではない', '`tests/test_rec_fix8_safety_r1.py` が固定')
    _miss(p, 'fix7/README 2 章の安全', one(_sec(f7, '## 2. 2 巡目の中身', '## 3. '), '- 安全(`fix_safety_r2.md`)'),
          '本物の記録の 235 の場面(場面ごと)でも、場面をつないだ 59 本でも、出す値が 1 フレームも違わない')
    _miss(p, 'fix7/README 3 章の安全の 2 行目', one(_sec(f7, '## 3. 残すこと', '## 4. '), '| 安全 | s4-c の残り:'),
          '| 本物の記録(235 の場面を場面ごとに、と場面をつないだ 59 本)では起きていない(10/28 fix8 safety r1)')
    return p


def k3_problems(d):
    """夜の道具: 形・ゲインの違う試しの ON の不具合・発進から R1 の起点までにカメラが止まった発進(OFF の本も)"""
    p = []
    ap, rb = d[AP], d[RB]
    r3 = one(_ap3(ap), '| 「途中で OFF にした日」の決まり(10/28) |')
    _miss(p, 'approvals 3 途中で OFF にした日の決まり', r3, '両方 ON(明けて中止)なら「R1 に届かなかった発進」',
          '**OFF(明けて走り続ける)なら「' + STALL + '」(AI 側の処理したフレームの間が 0.22 秒より長い。比べから外す。ON の本なら満たさない)**',
          '=「使ってよい」の側には動かない(10/28 fix8 tools r1。', 'w1-tools-r8-1-stall-inflates-off', '両方 ON でも止まらない形も同じに外す',
          '「注意: 発進から R1 の起点までにフレームの間が 0.15〜0.22 秒の発進 N 本 …」', '0.200 秒、比べに入る 33 は 0.133 秒')
    if '⑤ はその発進を「R1 に届かなかった発進」(ON の組なら満たさない)、その場で' in r3:
        p.append('approvals 3 途中で OFF にした日の決まり: OFF の本の扱いの無い前の文が残る')
    a8 = _sec(ap, '## 8. 回 3 の判定の読み方', '## 9. ')
    _miss(p, 'approvals 8 試しの本', one(a8, '| 試しの本・途中で変えた本 |'),
          '形(天井・左・自動合わせ・目標)・ゲインが違って別の組になっても、同じ回・同じモデルの比べのある組の条件に数える(10/28 fix8 tools r1')
    _miss(p, 'approvals 8 カメラの止まり', one(a8, '| ' + STALL + '(発進から R1 の起点までの AI 側の処理したフレームの間が 0.22 秒より長い'),
          'ON・OFF とも比べに入れない', 'on_no_stall', 'off_no_stall', 'その ON の最初の周の衝突・空転・記録の欠けは数える')
    _miss(p, '手順書 3 章 わざと OFF にした日', one(_ch3(rb), '  - **わざと OFF にした日の、本 X から先の読み方**'),
          '本 X から先の OFF の本で、発進から R1 の起点までに AI 側の処理したフレームの間が 0.22 秒より長い',
          '⑤ が「' + STALL + '」として比べから外す(10/28 fix8 tools r1')
    c7 = _ch7(rb)
    _miss(p, '手順書 7 章 走った値', one(c7, '| 走った値が計画と違った |'),
          '試しの本が別の組になっても、同じ回・同じモデルの比べのある組(計画の組)の条件に数える',
          '「注意: 同じ回・同じモデルの別の組(…)の、比べから外した本(試し・途中で変えた・走り直し)の ON の発進に …」')
    _miss(p, '手順書 7 章 起こし直した', one(c7, '| **jetformer-web か Jetson が起こし直した**'),
          '(4 章 9。その本の ON の発進の最初の周の衝突・空転・欠けは、形が違っても回 3 の計画の組の条件に数える = 満たさない。10/28 fix8 tools r1)')
    r5 = one(_sec(rb, '### 8.1', '### 8.2'), '| ⑤ | **回 3 は `--ignore-gain` で読む**')
    _miss(p, '手順書 8.1 ⑤', r5, '**形(天井・左・自動合わせ・目標)・ゲインが違って試しの本・途中で変えた本・走り直しが別の組になっても',
          'checks の first_lap_collisions_0・slip・slip_all_measured・on_all_readable のどれか',
          '「→ この組の比べから外した本の ON の発進の不具合(衝突・空転・欠け)は、同じ回・同じモデルの比べのある組(…)の条件にも数えた」',
          '(「試し」にした本の ON の不具合は計画の組の条件に残る。`--skip` で外すと不具合も数えない = 当たった本は `--skip` しない',
          '「' + STALL + ' N 本(AI 側の処理したフレーム(中立止めの行を除く)の間が 0.22 秒より長い。…)」',
          '(3 章で途中から OFF にした日の OFF の本も同じ)', '「→ ON の発進で、発進から R1 までにカメラが止まった本がある = 満たさない」',
          '「→ OFF の発進が全部カメラの止まりで比べから外れた」', '「カメラの止まり(フレームの間 X 秒・Y 秒から。比べに入れない)」',
          '「注意: 発進から R1 の起点までにフレームの間が 0.15〜0.22 秒の発進 N 本」', '境目(−0.15 秒)から 0.1 秒以内')
    return p


def k4_problems(d):
    """画面: app.js 10/28 04:54・走行タブに戻った直後の「読み込んでいます」・② の印・トーストの幅"""
    p = []
    rb, td, rd = d[RB], d[TODO], d[RD]
    c31 = _sec(rb, '### 3.1', '\n| 見ること |')
    _miss(p, '手順書 3.1 app.js', one(c31, '- **10/28 04:54 に `jetformer/web/static/app.js` を直した**'), '「画面の版 9(app.js 10/28 04:54)」',
          '409「管理画面が古いままです(この画面の app.js は …、今のファイルは 10/28 04:54 に直したもの)」→ 読み直す', 'style.css も 04:59',
          '再起動は要らず', '.py だけを見るので、app.js・style.css の名前は出ない')
    _miss(p, '手順書 7 章 Failed to fetch', one(_ch7(rb), '| 走行タブを押したら「Failed to fetch」などのトーストが出た'),
          '**走行タブに戻った直後**(ライブのタブ・ほかのタブから)に ▶・基準の設定・ブロックを押すと「' + REFUSE + '」が出る',
          '1〜2 秒待って押し直す', '10/28 fix8 ui r1')
    c81 = _sec(rb, '### 8.1', '### 8.2')
    _miss(p, '手順書 8.1 ② の印', one(c81, '| ② の印 |'), '**その回で ON の選択肢がいちばん多い本(同じ数なら先の本)と比べて、ON の少ない本 = '
          'OFF で走った本に付く**', '表の下の「(… 回の本は X(No.Y)から)」と同じ本を指す', '回の形(天井・左など)・下がる量の印は今も回の最初の本と比べる')
    t3 = _sec(td, '## 3. 11/3 の当日', '## 4. ')
    _miss(p, 'todo 3 章 画面', one(t3, '- 朝に **スマホの画面を読み直す**'), '今の app.js は 10/28 04:54(▶ の横「画面の版 9(app.js 10/28 04:54)」',
          '409「管理画面が古いままです」→ 読み直す', '走行タブに戻った直後の ▶ は「' + REFUSE + '」')
    _miss(p, 'README 6.2 の 11', one(_sec(rd, '### 6.2', '### 6.3'), '| 11 |'), '幅 420 px の画面で ▶ の断りの出し直し(約 200 字)が幅 378・高さ 176 px'
          '(前は 210×313)', '見える高さが 760 px 以下の画面', '出ている 6 秒の間は「停止中」のバッジの行にまだ重なりうる')
    return p


def k5_problems(d):
    """README: 6 章の頭の反証 8-1 回目・6.3 の 34(11/3 のあとの安全の案 qC)・35(残したこと)・7 章の報告"""
    p = []
    rd = d[RD]
    head = _sec(rd, '## 6. 残っていること', '### 6.1')
    _miss(p, 'README 6 章の頭', one(head, '- (10/28 fix8 docs 同期 1)**反証 8-1 回目**'),
          '新しい指摘 20 件 → 打ち消しの 2 票で 16 件が残り、**16 件とも直した**', '画面 3', '夜の道具 2', '安全 2', '文書 9',
          '**11/3 の安全のコード(assist.py・drive.py)は変えない**', 'どれも jetformer-web の再起動が要らない', 'ユーザーに伝えることは 6.2 の 33')
    c63 = _sec(rd, '### 6.3', '### 6.4')
    _miss(p, 'README 6.3 の 34', one(c63, '| 34 |'), '安全の案 **qC**', '11,232 組のどれでも OFF より 0.1 m/s を超えて速くない',
          '下がる距離の中央 1.38 m', '`fix8/proposal_safety_after1103.md` 9 章', '11/3 は今のまま', 'jetformer-web の再起動')
    _miss(p, 'README 6.3 の 35', one(c63, '| 35 |'), '反証 8-1 回目で残したこと', '見える高さ 760 px 以下', '古い左の値',
          '`tests/test_vbat_ui.py`', '0.15〜0.22 秒の短いカメラの止まりは注意の行だけ', '`cat $R/standing_3lap.md` の行は無い',
          '`fix8/fix_ui_r1.md` 11 章', '`fix8/fix_tools_r1.md` 7 章', '`fix8/fix_safety_r1.md` 6 章', '`fix8/fix_docs_r1.md` 6 章')
    c7 = rd[rd.index('## 7. 報告の一覧'):] if '## 7. 報告の一覧' in rd else ''
    _miss(p, 'README 7 章', one(c7, '| `fix8/fix_ui_r1.md`・'), '`fix8/docs_sync_r1.md`', '画面 3・夜の道具 2・安全 2・文書 9 = 16 件',
          '9 章(10/23 の形の案 qC・qA', '「本物の記録 235 本」は 235 の場面 = 9 章 9.7')
    return p


BOOT_IN = '10/28 08:40 の Jetson の起動し直しで入った'


def k6_problems(d):
    """10/28 08:40 の Jetson の起動し直し(jetformer-web 08:40:33・「起動より新しいコードなし」): 「反証 8 回目のあとにもう一度再起動」に
    「入った」の印・3.1 と approvals 2 の表に記録の行(読むだけで見たもの)"""
    p = []
    rb, td, rd, ap, f7 = d[RB], d[TODO], d[RD], d[AP], d[F7]
    c31 = _sec(rb, '### 3.1', '\n| 見ること |')
    _miss(p, '手順書 3.1 00:53:23 の行', one(c31, '- **10/28 00:53:23 に main session がもう一度再起動した**'),
          'か、11/3 の朝に Jetson を入れたときに消える。(→ **10/28 08:40 の Jetson の起動し直しで消えた** = 下の「10/28 08:40」の行)')
    boot = ('jetformer-web は 08:40:33 に今のディスクのコードで起きた', 'active・NRestarts 0・「エラーの行なし」(`-b`)',
            '新しい項目が全部既定(0・4.1・250)・`boot_at` あり(08:40:54)', '「起動より新しいコードなし」')
    _miss(p, '手順書 3.1 08:40 の行', one(c31, '- **10/28 08:40 に Jetson が起動し直した**'), *boot)
    _miss(p, '手順書 3.1 08:40 の行', one(c31, '- **10/28 08:40 に Jetson が起動し直した**'),
          '**「反証 8 回目のあとに main session がもう一度再起動する」は要らなくなった**', 'を直したら、また要る')
    _miss(p, 'todo 1 章 再起動', one(_sec(td, '## 1. 日ごと', '## 2. '), '| 再起動は**済**'),
          '反証 8 回目のあとに main session がもう一度再起動する(4 章) → **' + BOOT_IN + '**')
    _miss(p, 'todo 4 章の頭', _sec(td, '## 4. jetformer-web の再起動', '```'), '→ **10/28 08:40 の Jetson の起動し直し(電源断かリセット)で消えた**',
          *boot)
    _miss(p, 'README 6 章の頭', one(_sec(rd, '## 6. 残っていること', '### 6.1'), '- (10/28 fix8 docs 同期 1)**反証 8-1 回目**'),
          '**drive.py の説明の文(02:33)は 10/28 08:40 の Jetson の起動し直しで入った**', '「反証 8 回目のあとにもう一度再起動」は要らなくなった')
    c62 = _sec(rd, '### 6.2', '### 6.3')
    _miss(p, 'README 6.2 の 8', one(c62, '| 8 |'), 'ユーザーが車のそばにいる日(drive.py の説明の文(02:33)のぶんは **' + BOOT_IN + '**')
    _miss(p, 'README 6.2 の 32', one(c62, '| 32 |'), '4 章の 1〜5 → **' + BOOT_IN + '** = この再起動は要らなくなった')
    top = ap[:ap.index('\n## ')] if '\n## ' in ap else ap
    _miss(p, 'approvals の今の行', one(top, '- **今(10/28 03:3x。'), '反証 8 回目のあとにもう一度 = 2 の「今」→ **' + BOOT_IN + '**')
    a2 = _sec(ap, '## 2. jetformer-web の再起動の日', '## 3. ').splitlines()
    i = next((k for k, x in enumerate(a2) if x.startswith('| 今 |')), None)
    row = a2[i + 1] if i is not None and i + 1 < len(a2) else ''
    if not row.startswith('| 今(10/28 08:40) | **10/28 08:40 に Jetson が起動し直した**'):
        p.append('approvals 2: 「今」の行のすぐ下に「今(10/28 08:40)」の行が無い')
    _miss(p, 'approvals 2 今(10/28 08:40)', row, *boot)
    _miss(p, 'approvals 2 今(10/28 08:40)', row, '**上の行・0 章の 2・下の「誰が」の「反証 8 回目のあとにもう一度」の再起動は要らなくなった**')
    _miss(p, 'fix7/README 7 章の 2', one(_sec(f7, '## 7. 次に', '### 今のファイル'), '| 2 |'),
          'drive.py の説明の文のぶんは反証 8 回目のあと → **' + BOOT_IN + '**')
    return p


CHECKS = {'k0': k0_problems, 'k1': k1_problems, 'k2': k2_problems, 'k3': k3_problems, 'k4': k4_problems, 'k5': k5_problems,
          'k6': k6_problems}
# 台本の直し (文書, ALL の何番目) → それを元に戻すと落ちる確かめ
EDIT_CHECK = {(RB, 0): 'k0', (RB, 1): 'k1', (RB, 2): 'k3', (RB, 3): 'k4', (RB, 4): 'k1', (RB, 5): 'k3', (RB, 6): 'k3', (RB, 7): 'k4',
              (RB, 8): 'k4', (RB, 9): 'k3', (RB, 10): 'k3', (RB, 11): 'k3',
              (TODO, 0): 'k0', (TODO, 1): 'k0', (TODO, 2): 'k4', (TODO, 3): 'k1',
              (RD, 0): 'k0', (RD, 1): 'k5', (RD, 2): 'k4', (RD, 3): 'k1', (RD, 4): 'k5', (RD, 5): 'k5',
              (AP, 0): 'k0', (AP, 1): 'k1', (AP, 2): 'k1', (AP, 3): 'k3', (AP, 4): 'k3', (AP, 5): 'k3', (AP, 6): 'k2', (AP, 7): 'k2',
              (F7, 0): 'k0', (F7, 1): 'k2', (F7, 2): 'k2', (F7, 3): 'k2',
              (RB, 12): 'k6', (TODO, 4): 'k6', (TODO, 5): 'k6', (RD, 6): 'k6', (RD, 7): 'k6', (AP, 8): 'k6', (AP, 9): 'k6', (F7, 4): 'k6'}


# ================================================================== 文書
@pytest.mark.parametrize('name', sorted(CHECKS))
def test_live_documents(live, name):
    assert CHECKS[name](live) == []


@pytest.mark.parametrize('name', sorted(CHECKS))
def test_before_copies_fail(before, name):
    """直す前の写し(10/28 05:21 の形。fix8 docs r1 のあと)では、7 つとも落ちる"""
    assert CHECKS[name](before) != []


def test_before_copies_are_the_backups():
    """直す前の写しは、文書の横の控え(.bak-20261028-fix8-sync1)と同じ中身。控えは fix8 docs r1 の「後」(その報告 7 章の sha256)"""
    for f in FILES:
        assert _sha(BEFORE / f) == _sha(REPORTS / (f + BAK)), f
        assert _sha(BEFORE / f).startswith(R81_AFTER[f]), f


def test_sources_cover_every_edit(script):
    """台本の 42 個の直しは、どれも報告の文 1 つ(SOURCES)か頭の記録の行(LOGS)にちょうど 1 回入る。確かめの対応も 42 個"""
    used = sorted(x for v in script.SOURCES.values() for x in v) + sorted(script.LOGS)
    allk = sorted((f, i) for f, edits in script.ALL for i in range(len(edits)))
    assert sorted(used) == allk and len(allk) == 42 and sorted(EDIT_CHECK) == allk
    assert script.STAMP == STAMP and script.BAK == BAK


def test_script_edits_all_done(live, script, tmp_path):
    """台本の 42 個の直しは今の文書でどれも「済」で new がちょうど 1 回。SUPERSEDE 20 か所は前の台本の new(今の形)の中にちょうど 1 回
    あった文の書き換えで、今の文書に入っている(消した文は 0 回)。前の係の直し(書き換えを戻した文書で)・前の係とこの組の試験の確かめ
    (今の文書で)も崩れていない"""
    _, rep = script.apply_text(live)
    assert len(rep) == 42 and [s for _, _, s in rep] == ['done'] * 42, [(f, o[:30], s) for f, o, s in rep if s != 'done']
    for f, edits in script.ALL:
        for _, new in edits:
            assert live[f].count(new) == 1, (f, new[:40])
    assert len(script.SUPERSEDE) == 20 and script.superseded_present(live) == []
    assert script.revert(live) != live
    assert script.earlier_problems(live, str(tmp_path)) == []


def test_reverting_each_edit_is_caught(live, script):
    """わざと壊す: 台本の直しを 1 つずつ元の文に戻すと、その確かめが落ちる(42 個)"""
    n = 0
    for f, edits in script.ALL:
        for k, (old, new) in enumerate(edits):
            broken = dict(live, **{f: live[f].replace(new, old, 1)})
            assert broken[f] != live[f], (f, k)
            assert CHECKS[EDIT_CHECK[(f, k)]](broken) != [], (f, k, new[:40])
            n += 1
    assert n == 42


def test_more_breakages_are_caught(live):
    """台本の直しと別の形の壊し"""
    ap, rb, rd, td, f7 = live[AP], live[RB], live[RD], live[TODO], live[F7]
    c3 = _ch3(rb).splitlines()
    near = next(x for x in c3 if x.startswith('  - **見張り ON の日の、10/23 の形の後ろの近い物**'))
    first = next(x for x in c3 if x.startswith('  - 見張り・残すが ON の日(11/3)は、**その日の最初のリカバリーを人が見る**'))
    cases = [
        ('k1', AP, ap.replace('ON が 0.1 m/s より速い 2,922 組', 'ON が 0.1 m/s より速い 2,923 組', 1)),
        ('k1', AP, ap.replace('(c) を変えたいときは一言(返事が無ければ (c) のまま) |', '|', 1)),
        ('k1', RB, rb.replace(near + '\n', '', 1).replace(first + '\n', first + '\n' + near + '\n', 1)),     # 最初のリカバリーの行より下へ
        ('k1', RB, rb.replace('。先に人側にしただけなら見張り・残すは OFF にしない(3 章の「最初のリカバリー」の行で決める) |', ' |', 1)),
        ('k1', TODO, td.replace('11/3 のコードは変えない。先に人側にしただけなら', '先に人側にしただけなら', 1)),
        ('k1', RD, rd.replace('のほうが遅く当たるか当たらない(閉じたループ', 'のほうが長く下がる(閉じたループ', 1)),
        ('k1', AP, ap.replace('**約 1 m より遠い物には、ON のほうが遅く当たるか当たらない**', '**ON のほうが長く下がる**', 1)),
        ('k2', AP, ap.replace(REC235, '本物の記録 235 本では起きていない(出す値が直す前と全部同じ)', 1)),
        ('k2', F7, f7.replace('59 本・同じ 122,767 フレーム', '235 本・同じ 122,767 フレーム', 1)),
        ('k2', RD, rd.replace('「本物の記録 235 本」は 235 の場面', '本物の記録 235 本は 235 の場面', 1)),
        ('k3', RB, rb.replace('(AI 側の処理したフレーム(中立止めの行を除く)の間が 0.22 秒より長い。…)」', '(…)」', 1)),
        ('k3', AP, ap.replace('**OFF(明けて走り続ける)なら', '**OFF(明けて走り続ける)も', 1)),
        ('k4', RB, rb.replace('「画面の版 9(app.js 10/28 04:54)」(このあと', '「画面の版 9(app.js 10/27 21:58)」(このあと', 1)),
        ('k4', RB, rb.replace('| ② の印 |', '| ② |', 1)),
        ('k4', RD, rd.replace('幅 378・高さ 176 px', '幅 210・高さ 313 px', 1)),
        ('k5', RD, rd.replace('**16 件とも直した**', '**15 件を直した**', 1)),
        ('k5', RD, rd.replace('11,232 組のどれでも OFF より 0.1 m/s を超えて速くない', '11,232 組で OFF より速い組もある', 1)),
        ('k0', TODO, td.replace('までにすること(%s 現在)' % STAMP, 'までにすること(10/28 05:2x 現在)', 1)),
        ('k6', AP, ap.replace('| 今(10/28 08:40) |', '| 今(10/28) |', 1)),
        ('k6', RB, rb.replace('・`boot_at` あり(08:40:54)', '', 1)),
        ('k6', TODO, td.replace('で消えた**: jetformer-web は 08:40:33 に', 'で消えた**: jetformer-web は 08:40 に', 1)),
    ]
    for k, (name, f, t) in enumerate(cases):
        assert t != live[f], k
        assert CHECKS[name](dict(live, **{f: t})) != [], k


# ================================================================== 文書の言葉・数とコード・記録
def test_findings_and_votes_are_twenty_and_sixteen():
    """README 6 章の頭の「新しい指摘 20 件 → 打ち消しの 2 票で 16 件」= 5 つの係の findings.json と票 A・B(どちらも real でない 4 件)。
    係ごとに 画面 3・夜の道具 2・安全 2・文書 9(docs 5 + rehearsal 4)"""
    teams = {'ui': '画面', 'tools': '夜の道具', 'safety': '安全', 'docs': '文書', 'rehearsal': '文書'}
    found, votes = {}, {}
    for t in teams:
        for x in json.loads((HERE8 / ('review_%s_r1' % t) / 'findings.json').read_text(encoding='utf-8'))['findings']:
            found[x['id']] = teams[t]
        for ab in ('A', 'B'):
            for v in json.loads((HERE8 / ('refute_%s_r1_%s' % (t, ab)) / 'verdicts.json').read_text(encoding='utf-8'))['verdicts']:
                votes.setdefault(v['id'], []).append(bool(v['real']))
    assert len(found) == 20 and set(votes) == set(found) and all(len(v) == 2 for v in votes.values())
    kept = [i for i in found if any(votes[i])]
    assert len(kept) == 16
    by = {}
    for i in kept:
        by[found[i]] = by.get(found[i], 0) + 1
    assert by == {'画面': 3, '夜の道具': 2, '安全': 2, '文書': 9}


def test_safety_numbers_are_in_the_outputs():
    """approvals 3・手順書 3 章・README 6.2 の 33 の数 = 係 safety の台本の出力(proposal_r1/out/ana_r1.json・fix_safety_r1/out/facts12.json)"""
    a = json.loads((HERE8 / 'proposal_r1' / 'out' / 'ana_r1.json').read_text(encoding='utf-8'))
    assert (a['esc_models'], a['cells'], a['pairs_per_form']) == (18, 288, 11232)
    lt1, ge1 = a['now_on_vs_now_off']['by_band']['lt1'], a['now_on_vs_now_off']['by_band']['ge1']
    assert lt1['n'] == 5184 and lt1['counts'] == {'a_faster': 2922, 'b_faster': 622, 'same': 1640}
    assert round(lt1['a_faster_by']['mean'], 2) == 0.38 and round(lt1['a_faster_by']['max'], 1) == 1.0
    assert ge1['n'] == 6048 and (ge1['counts']['a_faster'], ge1['counts']['a_only'], ge1['counts']['b_faster'],
                                 ge1['counts']['b_only']) == (34, 8, 2020, 3226)
    fit = a['fit_by_d']
    on = sorted(float(dd) for dd, v in fit.items() if v['on_faster'] == 16)
    assert (min(on), max(on)) == (0.3, 0.65) and fit['0.7']['on_faster'] == 4 and fit['0.75']['on_faster'] == 2
    assert all(v['on_faster'] == 0 for dd, v in fit.items() if float(dd) >= 0.8)
    f = json.loads((HERE8 / 'fix_safety_r1' / 'out' / 'facts12.json').read_text(encoding='utf-8'))
    no6 = f['no6']
    assert (no6['first_hit_ge2p5g'][0]['back_m'], no6['max_g']['back_m']) == (1.163, 1.223)        # 約 1.16〜1.22 m
    assert round(no6['first_backing_sample_s'] - no6['brake_write_s'], 2) == 0.08                    # 「約 0.1 秒で下がり始め」
    u = f['160847']
    assert u['ai_part']['hall'] == [-62909, -62883] and u['ai_part']['hall_down_steps'] == 0
    assert abs(u['ai_part']['back_m'] - 0.13) <= 0.0051 and abs(u['total']['back_m'] - 0.24) <= 0.0051     # 約 0.13 m・計 約 0.24 m
    assert u['ch3_human_from_s'] == 40.453 and u['ai_part']['erpm_max'] == 1383.0                       # 40.45 秒・eRPM 〜1400
    sys.path.insert(0, str(JF))
    from jetformer import assist as A
    assert A.Recovery.STUCK_S == 0.8                                                                  # 「止まって約 0.8 秒で始まる」


def test_real_record_numbers_are_in_the_outputs():
    """approvals 14・fix7/README の「235 の場面」「つないだ 59 本」「160847 の 10 フレーム」= rec_concat12_assist.json・rec_concat12_r1.json"""
    r = json.loads((HERE8 / 'fix_safety_r1' / 'out' / 'rec_concat12_assist.json').read_text(encoding='utf-8'))
    sc, cc = r['scene'], r['concat']
    assert (sc['units'], sc['frames'], len(sc['diff_on']), len(sc['diff_off'])) == (235, 122767, 0, 0)
    assert (cc['units'], cc['frames'], len(cc['diff_on']), len(cc['diff_off']), cc['rec_starts']) == (59, 122767, 10, 0, 16)
    assert {x['unit'] for x in cc['diff_on']} == {'20261024-160847'}
    ss = sorted(x['s'] for x in cc['diff_on'])
    assert (ss[0], ss[-1]) == (40.07, 40.369)
    for x in cc['diff_on']:
        assert x['now_on'] == x['now_off'] and x['f7b_on'][:2] == [1291.0, 1796.0], x
        assert x['now_on'][:2] in ([1441.0, 1796.0], [1291.0, 1506.0]), x
    assert sorted({x['unit'] for x in cc['backing_bwg']}) == ['20261023-205458', '20261024-160847'] and len(cc['backing_bwg']) == 11
    r1 = json.loads((HERE8 / 'fix_safety_r1' / 'out' / 'rec_concat12_r1.json').read_text(encoding='utf-8'))
    for k, n in (('scene', 235), ('concat', 59)):            # 7-1 回目のあと(7-2 回目 s4-c の前)と今: つないでも 1 フレームも違わない
        cnt = [len(r1[k][x]) if isinstance(r1[k][x], list) else r1[k][x] for x in ('diff_on', 'diff_off')]      # 一覧か数
        assert (r1[k]['units'], r1[k]['frames'], cnt) == (n, 122767, [0, 0]), k


def test_tools_numbers_are_in_the_outputs():
    """approvals 3 の「R1 に届いた発進 38 … 0.200 秒、比べに入る 33 は 0.133 秒」= fix_tools_r1/out/scan_real_gaps.txt"""
    t = (HERE8 / 'fix_tools_r1' / 'out' / 'scan_real_gaps.txt').read_text(encoding='utf-8')
    assert '発進 70(届いた 38・比べに入る 33)' in t
    assert '20261023 proc: 発進 29・いちばん長い 0.200 秒(届いた 0.200・比べに入る 0.067)' in t
    assert '20261024 proc: 発進 24・いちばん長い 2.331 秒(届いた 0.133・比べに入る 0.133)' in t
    assert '比べに入る発進だけ {"0.00": 3289, "0.05": 10, "0.10": 1}' in t


def test_ui_numbers_are_in_the_outputs():
    """README 6.2 の 11 の数 = fix_ui_r1/css/out の測り(幅 420 の断りの出し直し・見える高さ 760 px 以下の重なり)"""
    h = (HERE8 / 'fix_ui_r1' / 'css' / 'out' / 'heights.txt').read_text(encoding='utf-8')
    assert '今 | 高さ 760・幅 420: 箱 幅 378 高さ 176(上 564)・バッジ 上 551・重なる バッジ「停止中」' in h
    assert '今 | 高さ 664・幅 420: 箱 幅 378 高さ 176' in h and '今 | 高さ 860・幅 420: 箱 幅 378 高さ 176(上 664)・バッジ 上 601・重なる なし' in h
    assert '直す前 | 高さ 860・幅 420: 箱 幅 210 高さ 313' in h


def test_words_match_the_code():
    """文書が引く画面・道具の言葉と値は、今のコードの文字列・定数"""
    web = JF / 'jetformer' / 'web'
    js = (web / 'static' / 'app.js').read_text(encoding='utf-8')
    css = (web / 'static' / 'style.css').read_text(encoding='utf-8')
    srv = (web / 'server.py').read_text(encoding='utf-8')
    assert ": '%s', true);" % REFUSE in js
    assert 'if (drIdle()) { state.drive.paramsLoaded = false; state.drive.drTabGen = (state.drive.drTabGen || 0) + 1; }' in js
    assert "error ? 6000 : 3000" in js                                         # 断りのトーストは 6 秒
    assert 'el.textContent = `画面の版 ${DRIVE_CLIENT}`' in js and "const DRIVE_CLIENT = 9" in js
    t = datetime.fromtimestamp((web / 'static' / 'app.js').stat().st_mtime)
    assert (t.month, t.day, t.hour, t.minute) == (10, 28, 4, 54), t          # 「画面の版 9(app.js 10/28 04:54)」
    t = datetime.fromtimestamp((web / 'static' / 'style.css').stat().st_mtime)
    assert (t.month, t.day, t.hour, t.minute) == (10, 28, 4, 59), t
    assert 'width: max-content; max-width: 90%; max-width: min(90%, 960px);' in css
    assert "'管理画面が古いままです(この画面の app.js は %s、今のファイルは %s に直したもの)。'" in srv
    co = (JF / 'experiments' / '20261026_fix' / 'diag_launch' / 'collect.py').read_text(encoding='utf-8')
    for w in ('LAUNCH_STALL_S = 0.22', 'LAUNCH_GAP_NOTE_S = 0.15', "CRIT = {'to_r1_gain_s': 0.15,",
              "'  %s %%d 本(AI 側の処理したフレーム(中立止めの行を除く)の間が %%.2f 秒より'" % STALL,
              "'  → ON の発進で、発進から R1 までにカメラが止まった本がある = 満たさない(checks の on_no_stall。",
              "'  → OFF の発進が全部カメラの止まりで比べから外れた = 比べられない(満たさない。checks の off_no_stall)'",
              "'  注意: 発進から R1 の起点までにフレームの間が %.2f〜%.2f 秒の発進 %d 本(比べには入れた。",
              "' カメラの止まり(フレームの間 %.2f 秒・%.2f 秒から。比べに入れない)'",
              "'  注意: 同じ回・同じモデルの別の組(%s)の、比べから外した本(試し・途中で変えた・走り直し)の ON の発進に %s: %s → '",
              "'  → この組の比べから外した本の ON の発進の不具合(衝突・空転・欠け)は、同じ回・同じモデルの比べのある組(%s)の'",
              "names = [c for c, x in (('first_lap_collisions_0', col_n), ('slip', hi), ('slip_all_measured', un),",
              "('on_all_readable', gp)) if x]", 'if k2 == k or k2[:2] != k[:2]:', '当たった本は外さない'):
        assert w in co, w
    ru = (JF / 'experiments' / '20261026_fix' / 'plan' / 'runs_1103.py').read_text(encoding='utf-8')
    assert 'f = max(rs, key=lambda x: len(opt_on(x)))       # 同じ数なら先の本(max は最初のものを返す)' in ru
    assert "r['diffs'].append('回 %s の試験の選択肢が %s(No.%s)と違う: %s%s' % (" in ru


def test_tools_print_what_the_documents_quote(tmp_path):
    """本物の ⑤(collect.py)を、係 tools の作り物の日(tests/test_fix8_tools_r1.py の sday・tfday)で流すと、文書が引く行を出す:
    3-5 から OFF にした日(B)のカメラの止まりは「発進から R1 の起点までにカメラが止まった発進 1 本(…0.22 秒より長い。…)」で比べから
    外れ「使ってよい」と出ない・発進の行の後ろに「カメラの止まり(…。比べに入れない)」/ ON の発進なら「→ ON の発進で、…= 満たさない」/
    天井の違う試しの ON の 6G は計画の組の「注意: 同じ回・同じモデルの別の組(…)の、…」と試しの組の「→ この組の比べから外した本の …」"""
    sys.path.insert(0, str(JF))
    sys.path.insert(0, str(JF / 'tools'))
    sys.path.insert(0, str(JF / 'tests'))
    import test_fix8_tools_r1 as T
    D, log35 = T.sday(tmp_path / 'b', 'B')
    j, out, _ = T.runc(D, 'r1new', T.IG, 'sync1b')
    ln = [x for x in out.splitlines() if x.startswith('  ' + STALL + ' ')]
    assert len(ln) == 1 and ln[0].startswith('  %s 1 本(AI 側の処理したフレーム(中立止めの行を除く)の間が 0.22 秒より長い。' % STALL)
    assert '→ 使ってよい' not in out and T.g0(j)['compare']['all_ok'] is False
    assert any(x.startswith(log35) and ' カメラの止まり(フレームの間 ' in x and '秒から。比べに入れない)' in x for x in out.splitlines())
    D2, _ = T.sday(tmp_path / 'b2', 'B', boost35=True)
    _, out2, _ = T.runc(D2, 'r1new', T.IG, 'sync1b2')
    assert any(x.startswith('  → ON の発進で、発進から R1 までにカメラが止まった本がある = 満たさない(checks の on_no_stall。')
               for x in out2.splitlines())
    D3, _ = T.tfday(tmp_path / 't', kind='trial', shape='hit', form='ceil')
    j3, out3, _ = T.runc(D3, 'r1new', T.IG, 'sync1t')
    assert any(x.startswith('  注意: 同じ回・同じモデルの別の組(') and '比べから外した本(試し・途中で変えた・走り直し)の ON の発進に' in x
               for x in out3.splitlines())
    assert any(x.startswith('  → この組の比べから外した本の ON の発進の不具合(衝突・空転・欠け)は、同じ回・同じモデルの比べのある組(')
               for x in out3.splitlines())
    g = T.plan_group(j3)
    assert g['other_group_on']['checks'] == ['first_lap_collisions_0'] and T.verdict(g) == '満たさない'


def test_runs_1103_marks_the_off_run(tmp_path):
    """本物の ②(runs_1103.py)を、係 ui の作り物の日(tests/test_ui_fix8_r1.py の _write_day)で流すと、回の最初の本 2-1 が OFF で 2-2 から
    入れ直した日の印は OFF の 2-1 に付き(相手は ON がいちばん多い 2-2)、「計画と違う 1 本」(手順書 8.1 の「② の印」の行)"""
    sys.path.insert(0, str(JF))
    sys.path.insert(0, str(JF / 'tests'))
    import test_ui_fix8_r1 as U
    if [U.BLOCK_OF.get('%d-%d' % (r, i)) for r in U.ROUNDS for i in range(1, U.ROUNDS[r] + 1)] != list(range(1, 24)):
        pytest.skip('data/bench_candidates.json の 11/3 のブロックが前提と違う')
    raw = tmp_path / 'raw'
    U._write_day(raw, [('1-1', U.G), ('1-2', U.G), ('2-1', U.OFF), ('2-2', U.G), ('2-3', U.G)])
    table, rws = U._runs(U.RUNS, raw, 'runs_sync1')
    marks = {r['run_id']: [x for x in r['diffs'] if '試験の選択肢が' in x] for r in rws}
    assert [k for k, v in marks.items() if v] == ['2-1'] and marks['2-1'][0].startswith('回 2 の試験の選択肢が 2-2(No.4)と違う: ')
    assert [x for x in table if ' 本・計画と違う ' in x] == ['5 本・計画と違う 1 本']


def test_proposal_chapter9_is_what_the_documents_say():
    """README 6.3 の 34・approvals 3 の qC の文 = 案の報告の 9 章(すすめは qC・11/3 は今のまま・235 は場面の数)。9 章は 1〜8 章の後ろに
    足しただけ(頭の 33,170 バイトは係の報告 fix8/fix_safety_r1.md 5 章の sha256 c74b16cd…)"""
    pr = (REPORTS / 'fix8' / 'proposal_safety_after1103.md').read_text(encoding='utf-8')
    c9 = pr[pr.index('## 9. 追記(10/28 fix8 係 safety 直す 8-1 回目)'):]
    for w in ("**qC「段を飛ばさず数えるだけ・'gap' の惰性は数えない」(すすめ)**", '- **11/3 は今のまま**(main session の決め)',
              '入れるなら **qC を見張り ON の日だけ**', '| **qC** | **0 + 0** |', '**0.70 / 1.38 / 2.07 m**',
              '235 は記録の本数でなく場面の数(238 の場面のうち空の 3 つを除く)', '本物の ESC で確かめること(11/3 の記録で)'):
        assert w in c9, w
    raw = (REPORTS / 'fix8' / 'proposal_safety_after1103.md').read_bytes()
    assert hashlib.sha256(raw[:33170]).hexdigest().startswith('c74b16cd')
    s1 = (REPORTS / 'fix8' / 'fix_safety_r1.md').read_text(encoding='utf-8')
    assert '**main session の決め(10/28 08:xx)**: 11/3 の安全のコード(jetformer/assist.py・drive.py)は変えない' in s1
    assert '(ON が近い物で悪く・遠い物で良いの入れかわりで、本物の ESC で確かめていない動きを増やさない・再起動を増やさない)' in s1


def test_restart_record_matches_the_system():
    """k6 の記録(10/28 08:40 の起動し直し・jetformer-web 08:40:33・「起動より新しいコードなし」・GET の既定と boot_at 08:40:54)=
    読むだけの確かめの出力(out/restart_check_0840.txt。手順書 3.1・todo 4 章の 3〜5 と同じコマンド)。jetformer/ の .py は今も
    08:40:33 より古い(新しい .py があれば、その名前が「起動より新しい」に出る = 文書を直す)。systemd の起動の時刻が 08:40:33 のままなら
    それも照らす(このあと誰かが再起動したら照らさない)"""
    import subprocess
    out = (HERE8 / 'docs_sync1' / 'out' / 'restart_check_0840.txt').read_text(encoding='utf-8')
    for w in ('reboot   system boot  6.8.12-1021-tegr Wed Oct 28 08:40', '\nactive\n0\nWed 2026-10-28 08:40:33 JST\nエラーの行なし\n',
              'running False', 'boot_at 1793144454.003 10/28 08:40:54',
              'launch_boost=0 rec_back_guard=0 rec_hold_keep=0 launch_no_kick=0 speed_ceiling=0 speed_auto=0 zone_caps=0 '
              'ceil_left_mps=4.1 launch_us=250.0', '## todo 4 章の 5(起動のあとに直したコード)\n起動より新しいコードなし\n'):
        assert w in out, w
    t0 = datetime(2026, 10, 28, 8, 40, 33).timestamp()
    newer = [str(q) for q in sorted((JF / 'jetformer').rglob('*.py')) if q.stat().st_mtime > t0]
    assert newer == [], newer
    assert datetime.fromtimestamp((JF / 'jetformer' / 'drive.py').stat().st_mtime) < datetime(2026, 10, 28, 8, 40, 33)
    try:
        r = subprocess.run(['systemctl', 'show', '-p', 'ActiveEnterTimestamp', '--value', 'jetformer-web'], capture_output=True,
                           text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return
    if r.returncode == 0 and r.stdout.strip() == 'Wed 2026-10-28 08:40:33 JST':
        r2 = subprocess.run(['systemctl', 'show', '-p', 'NRestarts', '--value', 'jetformer-web'], capture_output=True, text=True, timeout=20)
        assert r2.stdout.strip() == '0'
