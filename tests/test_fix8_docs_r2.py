"""fix8 係 docs 直す 8-2 回目(2026-10-28)の文書を読む試験(文書だけ。コードは変えていない)。台本
experiments/20261026_fix/fix8/fix_docs_r2/docs_fix8_r2.py(40 個の直し。うち 23 か所は前の係の文の中の書き換え = SUPERSEDE)、
直す前の写し tests/fixtures/fix8_docs_r2_before/(4 本と SHA256SUMS)、報告 ~/jetformer_reports/fix_20261026/fix8/fix_docs_r2.md。

反証 8-2 回目で確かめて残った文書の指摘 5 件(係 docs 4・係 rehearsal 1)を、手順書・todo・README・fix3/approvals.md に入れたかを見る:
- m0 頭の記録の 1 行(4 本とも。控えの名前・報告)と todo の題の時刻。
- m1 f8r2-docs-paper-trial-skip-drops-faults・rehearsal-r2-unmarked-rerun-skip-drops-hit(同じ根): 当たった・空転した・記録が欠けた ON の
  本は --skip しない(試し・印の無い走り直し・落ちた本でも。10/28 main session の決め)。手順書 6・7・8 章と todo 3 章の --skip・「外す」・
  「紙に「走り直し」」の行はどれも、その行に「--skip しない」か「外さない」か「使わない」がある(todo は 3 章)。前の「当たっていないと分かった本だけ --skip」は消した。
- m2 f8r2-docs-1023-first-recovery-off: main session の決め(10/28 11:xx)。その場の守り(1 m 以上下がる・後ろの物に近づく → 人側)は
  残し、人側にしただけ・1 m 以上下がっただけでは OFF にしない。その日を OFF にするのはふつうのリカバリーでは起きない形(① 止まってから
  3 秒たっても下がらない ② 1 回で 2.5 m より長く下がる)だけ。手順書 3・7 章・approvals 3・todo 2・3 章が同じ数。前の「1 m 以上下がる →
  止めて OFF」は無い。数は台本の出力(閉じたループ・本物の記録)と同じ。
- m3 f8r2-docs-todo-no-approvals3-notice: todo 1 章「11/2 まで」・2 章 10 に approvals 3 の 10/28 の知らせ。
- m4 f8r2-docs-approvals-restart-stale: approvals 0 章の 2 と、いちばん上の 2 つ目の「今(10/27 17:3x)」(→「前の今」)に 08:40 の印。
- m5 README 6.2 の 36(ユーザーに伝えること)・6.3 の 35。
- 文書の言葉・数とコード・記録: 本物の collect.py と car_check を作り物の回 3 で流し、文書どおり(不具合のある ON の本は外さない)なら
  「満たさない」のまま・前の書き方(いつも --skip)なら「使ってよい」・不具合の無い本は外しても「使ってよい」。閉じたループを今の
  drive.py・assist.py で少し流し直して、ふつうの ON は ① ② に当たらず、知られている悪い形(0 m の壁押し・止まらない後退)は当たる。
  ana_first_rec_r2.json・real_rec_r2.json の数と文書の数。collect.py・car_check・app.js・assist.py の言葉と定数。
- わざと壊す: 直す前の写しでは m0〜m5 が全部落ちる。台本の 40 個の直しを 1 つずつ元に戻すと、その確かめが落ちる。ほかの壊しも落ちる。
- 前の係の直しと試験の確かめが崩れていないか(台本の earlier_problems: この組の書き換えを戻した文書で同期 1 → fix8 docs r1 →
  docs-decided → fix7 docs の仕上げ・r2・r1 の直しと試験の確かめ)。
- 2026-10-28 fix8 docs 仕上げ(experiments/20261026_fix/fix8/docs_final/docs_fix8_final.py): 反証 8 回目の終わりに、この組の文の中を意図して
  書き換えた(SUPERSEDE 4 か所: todo の題の時刻・todo 1 章「11/2 まで」と 2 章 10 の知らせのあとに approvals 3 のもう 1 つの知らせ・
  README 6.2 の 36 のあとに 37。m0〜m5 が引く言葉は消していない)。「済」・new の数・SUPERSEDE・earlier_problems は仕上げの書き換えを
  戻した文書(fin.revert)で、わざと壊すは書き換えたあとの形の new(fin.effective)で照らす(弱めていない: 同じ 40 個・23 か所・同じ確かめ)。
  m0 の todo の題は、この組の時刻か、そのあと仕上げが今に合わせた時刻(TITLE_STAMPS)。
- 2026-10-29 fix9(この文書のあと): main session の優先のモデル(走行タブの一覧の上から 5 つ)が drive.py を直し、Jetson の起動し直しの
  あと jetformer-web は 10/29 08:51:40 にそのコードで起きた。文書(approvals の 08:40 の印)は 10/28 の記録として正しいので変えず、
  test_numbers_are_in_the_outputs(閉じたループを回した drive.py の版)・test_restart_marks_match_the_system(起動の時刻と新しい .py)に
  「今の drive.py が fix9 の版(sha256)なら、文書のときの版は直す前の写し tests/fixtures/fix9_priority_before/ で照らし、今は fix9 の起動
  (10/29 08:51:40)と fix9 の drive.py 1 本だけを照らす」を足した(弱めていない。fix9 が走行のループを変えていないことは
  tests/test_fix9_priority_before.py の AST の照らしを同じ試験の中で呼んで見る。今の状態は fix9 の文書に書く)。

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider -o tmp_path_retention_policy=failed \\
        --basetemp=<消せる場所> tests/test_fix8_docs_r2.py
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
BEFORE = JF / 'tests' / 'fixtures' / 'fix8_docs_r2_before'
HERE8 = JF / 'experiments' / '20261026_fix' / 'fix8'
D2 = HERE8 / 'fix_docs_r2'
SCRIPT = D2 / 'docs_fix8_r2.py'
RB, TODO, RD, AP, F7 = 'runbook_20261103.md', 'todo_user_1026.md', 'README.md', 'fix3/approvals.md', 'fix7/README.md'
FILES = (RB, TODO, RD, AP)
ALL7 = (RB, TODO, RD, AP, 'fix3/disk.md', F7, 'fix6/README.md')
BAK = '.bak-20261028-fix8-docs-r2'
STAMP = '10/28 11:4x'
FIN = HERE8 / 'docs_final' / 'docs_fix8_final.py'          # 2026-10-28 fix8 docs 仕上げ
TITLE_STAMPS = (STAMP, '10/28 12:4x')    # todo の題の時刻: この組と、そのあとの仕上げ(題の時刻を今に。docs_fix8_final.STAMP)
S1_AFTER = {RB: '1325465f88ec', TODO: 'aacc5e8b5943', RD: '143b41f873da', AP: 'd35465402338'}     # 同期 1 の「後」(docs_sync_r1.md 7 章)
FIND = {'docs': ('f8r2-docs-paper-trial-skip-drops-faults', 'f8r2-docs-1023-first-recovery-off', 'f8r2-docs-todo-no-approvals3-notice',
                 'f8r2-docs-approvals-restart-stale'),
        'rehearsal': ('rehearsal-r2-unmarked-rerun-skip-drops-hit',)}
SCREEN = 'rehearsal-r2-vbefore-screen-vs-server'     # 同じ反証の rehearsal の 2 件目は画面の係(app.js)の直し = この組では扱わない

SKIP_NO = '当たった・空転した・記録が欠けた ON の本は `--skip` しない'
OFF_ONLY = '① 止まってから 3 秒たっても下がらない ② 1 回で 2.5 m より長く下がる'
RULE_WORDS = ('`--skip` しない', '外さない', 'には使わない')
# 2026-10-29 fix9(この文書のあと)の優先のモデルの drive.py の版の sha256 と、その直す前の写し(= この文書のときの drive.py 3f2e31a2…)。
# jetformer-web はそのあと 10/29 08:51:40 に起きた(Jetson の起動し直しのあと。fix9 のコード)。tests/test_fix3_r3_2_tools.py の
# LATER_FIX9_PRIORITY と同じ値
FIX9_PRIORITY_BEFORE = JF / 'tests' / 'fixtures' / 'fix9_priority_before'
LATER_FIX9_PRIORITY = {'jetformer/drive.py': 'ecfb6656cae624950d5de221dfa71a5ac31b84ba82dacfbea2136832eae92315'}
LATER_FIX9_BOOT = datetime(2026, 10, 29, 8, 51, 40)


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
    return _load('docs_fix8_r2_t7', SCRIPT)


@pytest.fixture(scope='module')
def fin():
    """fix8 docs 仕上げが意図して書き換えた所(SUPERSEDE)・revert・effective(2026-10-28)"""
    return _load('docs_fix8_final_from_t7', FIN)


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
    heads = {3: ('## 3. 朝の準備', '### 3.1'), 6: ('## 6. 本番の形の決め方', '## 7. '), 7: ('## 7. 止める条件', '## 8. '),
             8: ('## 8. 休みと夜の集計', '## 9. ')}
    return _sec(rb, *heads[n])


def _td3(td):
    return _sec(td, '## 3. 11/3 の当日', '## 4. ')


def _ap(ap, n):
    heads = {0: ('## 0. 一覧', '## 1. '), 3: ('## 3. 新しい安全のスイッチ', '\n## 4. '), 8: ('## 8. 回 3 の判定の読み方', '## 9. '),
             12: ('## 12. 知らせ', '## 13. ')}
    return _sec(ap, *heads[n])


# ================================================================== 確かめ(どれも問題の一覧。空なら OK)
def m0_problems(d):
    """頭の記録の 1 行(最初の見出し「## 」より前・控えの名前・報告)と todo の題の時刻"""
    p = []
    heads = {RB: '- **%s ' % STAMP, TODO: '- **%s ' % STAMP, RD: '- **%s 追記**' % STAMP, AP: '- **%s ' % STAMP}
    for f, h in heads.items():
        t = d[f]
        top = t[:t.index('\n## ')] if '\n## ' in t else t
        line = one(top, h, 'fix8 係 docs 直す 8-2 回目')
        _miss(p, '%s の頭の記録' % f, line, '控え `%s%s`' % (Path(f).name, BAK), 'fix8/fix_docs_r2.md')
    title = d[TODO].splitlines()[0] if d[TODO] else ''
    # 2026-10-28 fix8 docs 仕上げ: 題の時刻は、この組の時刻か、そのあと仕上げが今に合わせた時刻(同期 1 の TITLE_STAMPS と同じ考え)
    if title not in ['# ユーザーが 11/3 までにすること(%s 現在)' % s for s in TITLE_STAMPS]:
        p.append('todo の題: 8-2 回目の時刻でない(%s)' % title[:40])
    return p


def m1_problems(d):
    """当たった・空転した・記録が欠けた ON の本は --skip しない(試し・印の無い走り直し・落ちた本でも)"""
    p = []
    rb, td, ap, rd = d[RB], d[TODO], d[AP], d[RD]
    # どの --skip・「外す」の行にも、その行の中に決まりの言葉がある(手順書 6・7・8 章と todo 3 章)
    for where, sec in (('手順書 6 章', _ch(rb, 6)), ('手順書 7 章', _ch(rb, 7)), ('手順書 8 章', _ch(rb, 8))):
        for x in sec.splitlines():
            if '--skip' in x and not any(w in x for w in RULE_WORDS):
                p.append('%s: --skip の行に決まりの言葉が無い: %s' % (where, x[:50]))
    for x in _td3(td).splitlines():
        if any(w in x for w in ('--skip', '外して読み直す', '外すまで', '紙に「走り直し')) and not any(w in x for w in RULE_WORDS):
            p.append('todo 3 章: 外す行に決まりの言葉が無い: %s' % x[:50])
    r6 = one(_ch(rb, 6), '| 発進の前進 ON(回 4) |')
    _miss(p, '手順書 6 章', r6, '**ただし' + SKIP_NO + '**(車のせいでない理由・試し・走り直しでも。',
          '→ 動画・紙で確かめる(**記録が欠けた ON の本なので `--skip` しない**',
          '→ 紙でどの本か確かめ、ON でないと分かれば `--skip 記録名`(ON の本なら記録が欠けた ON の本 = `--skip` しない。')
    _has(p, '手順書 6 章', r6, '当たっていない・車のせいでないと分かった本だけ `--skip 記録名`', 'ON でない・当たっていないと分かれば `--skip')
    c7 = _ch(rb, 7)
    _miss(p, '手順書 7 章 1 m', one(c7, '| リカバリーで車が 1 m 以上下がる'),
          '**その本が当たった・空転した・記録が欠けたら main session は `--skip` しない**')
    _miss(p, '手順書 7 章 中立止め', one(c7, '| 走行タブに「CH3: AI側 ・ 中立で停止中」'),
          '(印が無ければ、⑤ は main session が `--skip 記録名@AI側にした秒` で外す。**ただしその本(ON)が当たった・空転した・記録が欠けた(',
          'car_check の「衝突」の行が「0 回」でない', '「! 記録が欠けている」')
    _miss(p, '手順書 7 章 走った値', one(c7, '| 走った値が計画と違った |'), '**下の `--skip` は、当たった・空転した・記録が欠けた ON の本には使わない**')
    _miss(p, '手順書 7 章 起こし直した', one(c7, '| **jetformer-web か Jetson が起こし直した**'),
          '(本ごと外す。7 章「走った値が計画と違った」の行と同じ)。**ただし' + SKIP_NO + '**(紙の「試し」でも。')
    r7a = one(c7, '| いつでも |')
    _miss(p, '手順書 7 章 いつでも', r7a, '**ON の本なら記録が欠けた ON の本 = `--skip` しない**',
          '**動画・紙で当たっていないと分かっても `--skip` しない**')
    _has(p, '手順書 7 章 いつでも', r7a, 'ON でない・当たっていないと分かれば main session', '確かめた本だけ main session が `--skip 記録名`')
    r5 = one(_ch(rb, 8), '| ⑤ | **回 3 は `--ignore-gain` で読む**')
    _miss(p, '手順書 8.1 ⑤', r5, '(main session が、車のせいでない理由で止めた本だけ。外した名前が出る)。**' + SKIP_NO + '**(試し・印の無い走り直し・'
          '落ちた本でも。例外なし。見分け方:', '「当たった本は外さない」と同じ向きで、空転・欠けも', 'fix_docs_r2/out/skip_check_r2.txt',
          'で外す(**' + SKIP_NO + '**。下の「`--skip`」の文。fix8 docs r2)', '当たった本は `--skip` しない(空転した・記録が欠けた ON の本も',
          '当たった・空転した・記録が欠けた ON の本は外さない = 上の「`--skip`」の文')
    t3 = _td3(td)
    _miss(p, 'todo 3 章 10/23 の形', one(t3, '- (10/28 fix8 safety r1)**前の物の手前でブレーキのあと止まり'),
          '紙に「走り直し: 本・時刻」(その本が当たった・空転した・記録が欠けたら main session は `--skip` しない')
    _miss(p, 'todo 3 章 押し続けたら', one(t3, '- 当たって止まった車が AI 側のまま前へ押し続けたら'),
          '戻したら紙に「走り直し: 本・時刻」(その本が当たった・空転した・記録が欠けたら main session は `--skip` しない')
    _miss(p, 'todo 3 章 試し', one(t3, '- (10/27 直す 3 回目・2 組目)計画と違って走った本'),
          '当たった・空転した・記録が欠けた ON の本は main session も `--skip` しない')
    r77 = one(t3, '- (10/27 直す 2 回目・2 組目)Jetson か jetformer-web が走っている最中に落ちたら')
    _miss(p, 'todo 3 章 落ちた', r77, '記録が欠けた ON の本なので main session も外さない', 'ON でないと分かれば main session が外して読み直す(ON の本なら外さない')
    _has(p, 'todo 3 章 落ちた', r77, '紙・動画で確かめて main session が外すまで', '当たっていないと分かれば main session が外して')
    s8 = _ap(ap, 8)
    _miss(p, 'approvals 8', s8, '紙・動画で確かめてから main session が `--skip 記録名` で外して読み直す。**' + SKIP_NO + '**',
          '③ の紙の「走り直し」→ `--skip 記録名@AI側にした秒` は、**その本(ON)が当たった・空転した・記録が欠けたときは使わない**')
    _miss(p, 'approvals 12', one(_ap(ap, 12), '| 落ちた発進の前進 ON の本'),
          '**→ 10/28 main session の決め: 落ちた ON の本は記録が欠けた ON の本なので `--skip` しない = その日の回 4 は発進の前進 OFF**')
    _miss(p, 'README 6.3 の 35', one(_sec(rd, '### 6.3', '### 6.4'), '| 35 |'), '`--skip` は不具合も外す(→ 10/28 fix8 docs r2: 手順書・todo・approvals は')
    return p


def m2_problems(d):
    """その日を OFF にする「おかしな動き」= ふつうのリカバリーでは起きない形だけ(① 3 秒・② 2.5 m)。人側にしただけでは OFF にしない"""
    p = []
    rb, td, ap = d[RB], d[TODO], d[AP]
    c3 = _ch(rb, 3)
    first = one(c3, '  - 見張り・残すが ON の日(11/3)は、**その日の最初のリカバリーを人が見る**')
    _miss(p, '手順書 3 章 最初のリカバリー', first, '**その場の守り**(7 章。人側にするだけ): 1 m 以上下がる・後ろの物に近づく',
          '**人側にしただけ・1 m 以上下がっただけでは見張り・残すを OFF にしない**', '576 場面とも 1.02〜1.97 m 下がった',
          '本物の 10/23 の AI 側のまま終わったリカバリー(`rev_s` 0.4・`rev_us` 300)は 0.94 m',
          '**その日を OFF にするのは、ふつうのリカバリーでは起きない「おかしな動き」を見たときだけ**', '0.6 秒の内に 10/23 の形かを見分けなくてよい',
          '**① 下がらない**(0 m の壁押し): 前の壁・物で止まってから **3 秒** たっても 1 度も下がらない', '0.64〜2.22 秒で下がり始める(本物 0.83〜1.48 秒)',
          '96 場面は 96 とも当たる', '**② 長く下がる**: 1 回のリカバリーで、止まった所から **2.5 m** より長く下がる', '72 場面とも 3.63〜6.18 m',
          '人側にして形を見届けなかったときは OFF にせず', 'fix_docs_r2/out/ana_first_rec_r2.txt')
    if first and not (0 <= first.find('① ② のおかしな動き(① の例:') < first.find('→ すぐ CH3 を人側 → 止めて OFF')):
        p.append('手順書 3 章 最初のリカバリー: 「止めて OFF」が ① ② のおかしな動きのあとに無い')
    _miss(p, '手順書 3 章 10/23 の形', one(c3, '  - **見張り ON の日の、10/23 の形の後ろの近い物**'),
          '先に人側にしただけなら見張り・残すは OFF にしない(**OFF にするのは、下の「最初のリカバリー」の行の「おかしな動き」' + OFF_ONLY + ' を見たときだけ**')
    _miss(p, '手順書 7 章 1 m', one(_ch(rb, 7), '| リカバリーで車が 1 m 以上下がる'),
          '**1 m 以上下がった・後ろの物に近づいて人側にしただけでは OFF にしない**', '「おかしな動き」' + OFF_ONLY + ' のときだけ')
    s3 = _ap(ap, 3)
    _miss(p, 'approvals 3 決まり', one(s3, '| 決まり |'), '**「おかしければ」は、ふつうのリカバリーでは起きない形だけ** = 手順書 3 章の「おかしな動き」' + OFF_ONLY)
    _miss(p, 'approvals 3 なぜ', one(s3, '| なぜ |'), '見張り ON でもふつうに約 1〜2 m 下がる', 'ON の最初のリカバリーは 1.02〜1.97 m(No.6 に合わせた模型 1.28〜1.40 m',
          '10/23 の形で 1.17〜2.91 m・No.6 に合わせた模型 2.30〜2.42 m', 'ふつうのリカバリー(`rev_s` 0.4・`rev_us` 300)は 0.94 m')
    _miss(p, 'approvals 3 10/23 の形', one(s3, '| 困ること(ON)・10/23 の形'), '人側にしただけ・1 m 以上下がっただけでは見張り・残すを OFF にしない(OFF に'
          'するのは手順書 3 章の「最初のリカバリー」の行の「おかしな動き」' + OFF_ONLY + ' のときだけ')
    t3 = _td3(td)
    t67 = one(t3, '- 試験の選択肢は、**11/3 は')
    _miss(p, 'todo 3 章 最初のリカバリー', t67, '(1 m 以上下がる・後ろの物に近づく → すぐ人側。**人側にしただけ・1 m 以上下がっただけでは OFF にしない**',
          '① 前の壁・物で止まってから 3 秒たっても 1 度も下がらない', '② 1 回のリカバリーで止まった所から 2.5 m より長く下がる → すぐ人側 → 止めて OFF')
    _miss(p, 'todo 3 章 10/23 の形', one(t3, '- (10/28 fix8 safety r1)**前の物の手前でブレーキのあと止まり'),
          'OFF にするのは上の「その日の最初のリカバリーを見る」の行の ① ② のときだけ')
    _miss(p, 'todo 2 章 10', one(_sec(td, '## 2. 決めてほしいこと', '## 3. '), '| 10 |'),
          'おかしければ止めて OFF(「おかしければ」は、ふつうのリカバリーでは起きない形 = 手順書 3 章の ' + OFF_ONLY + ' のときだけ')
    # 前の「1 m 以上下がる → 止めて OFF」は無い(4 本のどこにも)
    for f in FILES:
        _has(p, f, d[f], '1 m 以上下がる・後ろの物に近づく → すぐ人側 → 止めて OFF', '1 m 以上下がる・後ろの物に近づく・おかしな動き(')
    # 数がそろう(3 秒・2.5 m)。ほかの秒・m の ① ② は無い
    for f in FILES:
        for m in re.finditer(r'止まってから \**(\d+(?:\.\d+)?) 秒\** ?たっても', d[f]):
            if m.group(1) != '3':
                p.append('%s: ① の秒が 3 でない(%s)' % (f, m.group(1)))
        for m in re.finditer(r'(?:止まった所から|1 回で) \**(\d+(?:\.\d+)?) m\** ?より長く下がる', d[f]):
            if m.group(1) != '2.5':
                p.append('%s: ② の m が 2.5 でない(%s)' % (f, m.group(1)))
    return p


NOTICE_WORDS = ('**10/28 に 3 へ知らせ 1 つ**', '10/23 の形(前の物の手前でブレーキのあと止まり、すぐ下がる形)では、見張り ON は止まった所から約 1 m より'
                '近い後ろの物に OFF より強く当たることがある', '11/3 のコードは変えない', '「リカバリーを待たずに人側にしてよい」',
                '(c) を変えたいときは一言・返事が無ければ (c) のまま', '「困ること(ON)・10/23 の形」', '`README.md` 6.2 の 33')


def m3_problems(d):
    """todo 1 章「11/2 まで」と 2 章 10 に approvals 3 の 10/28 の知らせ"""
    p = []
    td = d[TODO]
    _miss(p, 'todo 1 章 11/2 まで', one(_sec(td, '## 1. 日ごと', '## 2. '), '| **11/2 まで** |'), '3 は決まり((c) 両方 ON。', *NOTICE_WORDS)
    _miss(p, 'todo 2 章 10', one(_sec(td, '## 2. 決めてほしいこと', '## 3. '), '| 10 |'), *NOTICE_WORDS)
    return p


def m4_problems(d):
    """approvals 0 章の 2 と、いちばん上の 2 つ目の「今(10/27 17:3x)」(→「前の今」)に 08:40 の起動し直しの印"""
    p = []
    ap = d[AP]
    r2 = one(_ap(ap, 0), '| 2 |')
    _miss(p, 'approvals 0 章の 2', r2, '反証 8 回目のあとにもう一度 → **10/28 08:40 の Jetson の起動し直しで入った**(jetformer-web 08:40:33。'
          '「起動より新しいコードなし」= 2 の「今(10/28 08:40)」の行。この再起動は要らなくなった',
          '説明の文のぶんは反証 8 回目のあと → **10/28 08:40 の起動し直しで入った** = 要らなくなった) |')
    top = ap[:ap.index('\n## ')] if '\n## ' in ap else ap
    if rows(top, '- **今(10/27 17:3x。'):
        p.append('approvals の頭: 「今(10/27 17:3x)」の行が「今」のまま')
    _miss(p, 'approvals の頭', one(top, '- 前の今(10/27 17:3x。'), '**今は下の「今(10/28 03:3x)」の行**', 'この行は 10/27 17:3x の記録のまま',
          '**10/28 08:40 の Jetson の起動し直しで入った**', '= 再起動は要らない', '**2 は済み**(10/27 17:16:28 に再起動。')
    if not one(top, '- **今(10/28 03:3x。'):
        p.append('approvals の頭: 指す先の「今(10/28 03:3x)」の行が無い')
    return p


def m5_problems(d):
    """README 6.2 の 36(ユーザーに伝えること: 10/28 main session の決め 2 つ)"""
    p = []
    r36 = one(_sec(d[RD], '### 6.2', '### 6.3'), '| 36 |')
    _miss(p, 'README 6.2 の 36', r36, '**ユーザーに伝えること**(10/28 main session の決め・fix8 docs r2。返事が無ければこのまま)',
          '「おかしければ」は、ふつうのリカバリーでは起きない形(' + OFF_ONLY + ')だけ', '② ' + SKIP_NO, '`fix3/approvals.md` 12',
          '| main session → ユーザー | 11/2(返事が無ければこのまま) |')
    return p


CHECKS = {'m0': m0_problems, 'm1': m1_problems, 'm2': m2_problems, 'm3': m3_problems, 'm4': m4_problems, 'm5': m5_problems}
FINDING_CHECK = {'f8r2-docs-paper-trial-skip-drops-faults': 'm1', 'rehearsal-r2-unmarked-rerun-skip-drops-hit': 'm1',
                 'f8r2-docs-1023-first-recovery-off': 'm2', 'f8r2-docs-todo-no-approvals3-notice': 'm3',
                 'f8r2-docs-approvals-restart-stale': 'm4'}
README_EDIT = {(RD, 1): 'm5', (RD, 2): 'm1'}


def edit_check(script):
    out = {x: FINDING_CHECK[k] for k, v in script.FINDINGS.items() for x in v}
    out.update(README_EDIT)
    out.update({x: 'm0' for x in script.LOGS})
    return out


# ================================================================== 文書
@pytest.mark.parametrize('name', sorted(CHECKS))
def test_live_documents(live, name):
    assert CHECKS[name](live) == []


@pytest.mark.parametrize('name', sorted(CHECKS))
def test_before_copies_fail(before, name):
    """直す前の写し(10/28 09:10 の形。同期 1 のあと)では、6 つとも落ちる"""
    assert CHECKS[name](before) != []


def test_before_copies_are_the_backups():
    """直す前の写しは文書の横の控え(.bak-20261028-fix8-docs-r2)と同じで、同期 1 の「後」の sha256"""
    for f in FILES:
        b = REPORTS / (f + BAK)
        assert _sha(b) == _sha(BEFORE / f), f
        assert _sha(b)[:12] == S1_AFTER[f], f


def test_findings_are_five_and_every_edit_has_a_source(script):
    """確かめて残った指摘は 5 件(係 docs 4・係 rehearsal 1。どれもどれかの票が real)。台本の 40 個の直しは、どれか 1 つの指摘か頭の
    記録の行(LOGS)にちょうど 1 回入る。rehearsal の 2 件目(画面の電圧の源)は画面の係の直し"""
    found = {}
    for team, ids in FIND.items():
        got = {x['id'] for x in json.loads((HERE8 / ('review_%s_r2' % team) / 'findings.json').read_text(encoding='utf-8'))['findings']}
        assert set(ids) <= got, team
        votes = {}
        for ab in ('A', 'B'):
            for v in json.loads((HERE8 / ('refute_%s_r2_%s' % (team, ab)) / 'verdicts.json').read_text(encoding='utf-8'))['verdicts']:
                votes.setdefault(v['id'], []).append(bool(v['real']))
        for i in ids:
            assert len(votes[i]) == 2 and any(votes[i]), i
            found[i] = team
        if team == 'rehearsal':
            assert got - set(ids) == {SCREEN}
    assert sorted(found) == sorted(script.FINDINGS) and len(found) == 5
    used = sorted(x for v in script.FINDINGS.values() for x in v) + sorted(script.LOGS)
    allk = sorted((f, i) for f, edits in script.ALL for i in range(len(edits)))
    assert sorted(used) == allk and len(allk) == 40
    assert script.STAMP == STAMP and script.BAK == BAK and script.FILES == FILES


def test_script_edits_all_done(live, script, fin, tmp_path):
    """台本の 40 個の直しは今の文書でどれも「済」で new がちょうど 1 回。SUPERSEDE 23 か所は前の台本の new(今の形)の中にちょうど 1 回
    あった文の書き換えで、今の文書に入っている(消した文は 0 回)。前の係の直し(書き換えを戻した文書で)・前の係とこの組の試験の確かめ
    も崩れていない(earlier_problems)。
    2026-10-28 fix8 docs 仕上げ: どれも仕上げの書き換え(SUPERSEDE)を戻した文書 live2 で照らす(仕上げの書き換えは今の文書にちょうど 1 回)"""
    assert fin.superseded_present(live) == []
    live2 = fin.revert(live)
    assert live2 != live
    _, rep = script.apply_text(live2)
    assert len(rep) == 40 and [s for _, _, s in rep] == ['done'] * 40, [(f, o[:30], s) for f, o, s in rep if s != 'done']
    for f, edits in script.ALL:
        for _, new in edits:
            assert live2[f].count(new) == 1, (f, new[:40])
    assert len(script.SUPERSEDE) == 23 and script.superseded_present(live2) == []
    assert script.revert(live2) != live2
    assert script.earlier_problems(live2, str(tmp_path)) == []


def test_reverting_each_edit_is_caught(live, script, fin):
    """わざと壊す: 台本の直しを 1 つずつ元の文に戻すと、その確かめが落ちる(40 個)。
    2026-10-28 fix8 docs 仕上げ: 仕上げが書き換えた直し(todo の題・知らせ・README 6.2 の 36)は、書き換えたあとの形の new(fin.effective)を
    元の文に戻す"""
    ec = edit_check(script)
    n = 0
    for f, edits in script.ALL:
        for k, (old, new) in enumerate(edits):
            new = fin.effective('r82', f, k, new)
            assert live[f].count(new) == 1, (f, k)
            broken = dict(live, **{f: live[f].replace(new, old, 1)})
            assert broken[f] != live[f], (f, k)
            assert CHECKS[ec[(f, k)]](broken) != [], (f, k, new[:40])
            n += 1
    assert n == 40


def test_more_breakages_are_caught(live):
    """台本の直しと別の形の壊し"""
    rb, td, ap, rd = live[RB], live[TODO], live[AP], live[RD]
    cases = [
        ('m2', RB, rb.replace('止まってから **3 秒** たっても', '止まってから **2 秒** たっても', 1)),
        ('m2', TODO, td.replace('止まった所から 2.5 m より長く下がる', '止まった所から 1.0 m より長く下がる', 1)),
        ('m2', TODO, td.replace('(1 m 以上下がる・後ろの物に近づく → すぐ人側。', '(1 m 以上下がる・後ろの物に近づく → すぐ人側 → 止めて OFF。', 1)),
        ('m2', RB, rb.replace('① ② のおかしな動き(① の例:', 'おかしな動き(① の例:', 1)),
        ('m2', AP, ap.replace('1.17〜2.91 m', '1.17〜2.90 m', 1)),
        ('m2', RB, rb.replace('0.64〜2.22 秒で下がり始める', '0.64〜1.56 秒で下がり始める', 1)),
        ('m1', RB, rb.replace('**下の `--skip` は、当たった・空転した・記録が欠けた ON の本には使わない**', '', 1)),
        ('m1', RB, rb.replace('**ただしその本(ON)が当たった・空転した・記録が欠けた(', '**その本(ON)が当たった・空転した・記録が欠けた(', 1)),
        ('m1', TODO, td.replace('ON でないと分かれば main session が外して読み直す(ON の本なら外さない',
                                '当たっていないと分かれば main session が外して読み直す(', 1)),
        ('m1', AP, ap.replace('**その本(ON)が当たった・空転した・記録が欠けたときは使わない**', '**いつも使う**', 1)),
        ('m3', TODO, td.replace('3 は決まり((c) 両方 ON。**10/28 に 3 へ知らせ 1 つ**', '3 は決まり((c) 両方 ON。**10/28 の知らせ**', 1)),
        ('m3', TODO, td.replace('(c) を変えたいときは一言・返事が無ければ (c) のまま(`fix3', '(`fix3', 1)),
        ('m4', AP, ap.replace('- 前の今(10/27 17:3x。', '- **今(10/27 17:3x。', 1)),
        ('m4', AP, ap.replace('説明の文のぶんは反証 8 回目のあと → **10/28 08:40 の起動し直しで入った** = 要らなくなった) |',
                              '説明の文のぶんは反証 8 回目のあと) |', 1)),
        ('m5', RD, rd.replace('| 36 | **ユーザーに伝えること**', '| 36 | 伝えること', 1)),
        ('m0', TODO, td.replace(td.splitlines()[0], '# ユーザーが 11/3 までにすること(10/28 09:1x 現在)', 1)),
    ]
    for k, (name, f, t) in enumerate(cases):
        assert t != live[f], k
        assert CHECKS[name](dict(live, **{f: t})) != [], k


# ================================================================== 文書の言葉・数とコード・記録
def test_numbers_are_in_the_outputs():
    """① ② の数と文書の数は、台本の出力(out/ana_first_rec_r2.json = 閉じたループ・本物の記録)と同じ。閉じたループは今の drive.py・
    assist.py で回した(sha256)。10/29 fix9(優先のモデル)が drive.py を直したあとは、回したときの drive.py は fix9 の直す前の写し
    (sha256 を照らす)で、fix9 が変えたのは走行タブの一覧の list_models と足した定数・関数だけ(走行のループ Driver は同じ。
    tests/test_fix9_priority_before.py の AST の照らしをここで呼ぶ)"""
    A = json.loads((D2 / 'out' / 'ana_first_rec_r2.json').read_text(encoding='utf-8'))
    assert A['NOBACK_S'] == 3.0 and A['BACK_M'] == 2.5
    S = A['sets']
    dpy = JF / 'jetformer' / 'drive.py'
    if _sha(dpy) == LATER_FIX9_PRIORITY['jetformer/drive.py']:
        import test_fix9_priority_before as P9
        P9.assert_drive_changed_only_priority(dpy)
        dpy = FIX9_PRIORITY_BEFORE / 'drive.py'
        assert _sha(dpy) == P9.sums()['drive.py']
    for name, n_esc in (('normal18', 18), ('gapslow', 6), ('k1_noback', 6), ('k2_runaway', 3), ('k3_limits', 3)):
        assert S[name]['n_esc'] == n_esc, name
        assert S[name]['sha'] == {'drive.py': _sha(dpy), 'assist.py': _sha(JF / 'jetformer' / 'assist.py')}, name
    nor = [S['normal18']['all_on'], S['gapslow']['all_on']]
    assert sum(x['runs'] for x in nor) == 576 and all(x['no_rec'] == 0 for x in nor)
    assert round(min(x['back_m']['min'] for x in nor), 2) == 1.02 and round(max(x['back_m']['max'] for x in nor), 2) == 1.97
    assert sum(x['ge1m'] for x in nor) == 576
    assert round(min(x['noback_s']['min'] for x in nor), 2) == 0.64 and round(max(x['noback_s']['max'] for x in nor), 2) == 2.22
    assert all(x['shape1'] == 0 and x['shape2'] == 0 and x['noback_none'] == 0 and x['esc_push_gt_0p05'] == 0 for x in nor)
    off23 = [S['normal18']['1023_off'], S['gapslow']['1023_off']]
    assert round(min(x['back_m']['min'] for x in off23), 2) == 1.17 and round(max(x['back_m']['max'] for x in off23), 2) == 2.91
    fit = S['normal18']['fit_only']
    assert (round(fit['1023_on']['min'], 2), round(fit['1023_on']['max'], 2)) == (1.28, 1.40)
    assert (round(fit['1023_off']['min'], 2), round(fit['1023_off']['max'], 2)) == (2.30, 2.42)
    k1 = S['k1_noback']['all_on']
    assert k1['runs'] == 144 and k1['noback_none'] == 96 and k1['shape1'] == 96 and k1['esc_push_gt_0p05'] == 96 and k1['shape2'] == 0
    k2 = S['k2_runaway']['all_on']
    assert k2['runs'] == 72 and k2['shape2'] == 72 and (round(k2['back_m']['min'], 2), round(k2['back_m']['max'], 2)) == (3.63, 6.18)
    assert S['k3_limits']['all_on']['shape2'] == 72
    R = A['real']
    assert R['n'] == 6 and R['shape1'] == 0 and R['shape2'] == 0
    assert (round(R['noback_s']['min'], 2), round(R['noback_s']['max'], 2)) == (0.83, 1.48)
    assert [round(x[2], 2) for x in R['ai_through']] == [0.94]
    # 10/10 の前の弱い後退(rev_us が log.json に無い本)は、0 m のまま 3.5 秒の間に 2 回(= ① の形の本物)
    assert [(x[0], round(x[1], 2), x[2], x[3]) for x in R['old_weak'][:2]] == [('20261010-181504', 26.34, 0.0, None),
                                                                                ('20261010-181504', 29.84, 0.0, None)]


def test_closed_loop_still_separates(tmp_path):
    """閉じたループ(本物の drive._loop・偽物の ESC)を今のコードで少し流し直す: ふつうの ON(fit の 10/23 の形とふつうの形・'gap' 0.15 秒では
    下がれない ESC)は ① ② に当たらず、0 m の壁押し(後退を受け付けない ESC)は ①、止まらない後退(車輪の値が後ろへの動きを出さない)は ②"""
    sys.path.insert(0, str(D2))
    F = _load('first_rec_r2_t7', D2 / 'first_rec_r2.py')
    F.cl.TMP = F.cl.CL.TMP = tmp_path / 'cl'
    S = F.esc_sets()
    sc = list(F.scenes())
    s1023, swall = sc[0], sc[16]
    assert s1023['form'] == '1023' and swall['form'] == 'wall'
    try:
        for name, ek, s in (('normal18', 'fit', s1023), ('normal18', 'fit', swall), ('gapslow', 'fit_neu0.25', swall)):
            r = F.run_one(S[name][ek], 'on', s, F.SET_OPTS[name])
            assert r['rec'] and r['back_m'] <= 2.5 and r['noback_s'] is not None and r['noback_s'] <= 3.0, (name, ek, r)
            assert r['back_m'] >= 1.0, (name, ek, r)        # 1 m 以上下がるのはふつう
        r = F.run_one(S['k1_noback']['fit_neu99.0'], 'on', swall, F.SET_OPTS['k1_noback'])
        assert r['noback_s'] is None and r['back_m'] == 0.0 and r['esc_push_noback_s'] > 0.05 and r['giveup'], r
        r = F.run_one(S['k2_runaway']['fit_erpm0back'], 'on', swall, F.SET_OPTS['k2_runaway'])
        assert r['back_m'] > 2.5 and r['noback_s'] is not None and r['noback_s'] <= 3.0, r
    finally:
        F.cl.cleanup()


def test_skip_rule_with_the_real_collect(tmp_path):
    """本物の collect.py(⑤ --ignore-gain)と car_check を作り物の回 3 で流す: 紙の「試し」の ON の本・印の無い走り直しの最初の周の当たり・
    空転・欠けは car_check に出て(衝突・発進の空転の比の「!」・記録が欠けている)、文書どおり外さなければ「満たさない」・前の書き方(いつも
    --skip)なら「使ってよい」。不具合の無い本は外しても「使ってよい」。out/skip_check_r2.json と同じ答え"""
    sys.path.insert(0, str(D2))
    K = _load('skip_check_r2_t7', D2 / 'skip_check_r2.py')
    saved = json.loads((D2 / 'out' / 'skip_check_r2.json').read_text(encoding='utf-8'))
    want = {(k, r['shape']): (r['doc']['word'], r['old']['word'], r['plain']['word']) for k in ('trial', 'rerun') for r in saved[k]}
    k = 0
    for kind, shapes in (('trial', ('hit', 'spin', 'gap', 'clean')), ('rerun', ('hit', 'spin', 'gap', 'clean'))):
        for shape in shapes:
            k += 1
            if kind == 'trial':
                sh, kw = K.TRIAL[shape]
                D, log = K.T.mtday(tmp_path / ('t%d' % k), 'none', sh, **kw)
                sk = log
            else:
                D, log, info = K.rerun_day(tmp_path / ('r%d' % k), shape)
            rc, txt = K.car_check_text(D, log)
            fl = K.cc_flags(txt)
            faulty = fl['collision'] or fl['slip'] or fl['missing']
            assert faulty == (shape != 'clean'), (kind, shape, fl)
            assert fl[{'hit': 'collision', 'spin': 'slip', 'gap': 'missing', 'clean': 'collision'}[shape]] == (shape != 'clean')
            j0, _, _ = K.T.runc(D, 'new', list(K.T.IG), 't7%d0' % k)
            if kind == 'rerun':
                rows = [r for r in j0['launches'] if r['log'] == log]
                r2 = [r for r in rows if (r.get('start_no') or 1) == 2][0]
                assert r2.get('after_abort') is None          # 印の無い走り直し(リカバリーの前に人側)
                sk = '%s@%.2f' % (log, r2['t_ai'])
            j1, _, _ = K.T.runc(D, 'new', list(K.T.IG) + ['--skip', sk], 't7%d1' % k)
            plain, skipped = K.word(K.verdict(j0)), K.word(K.verdict(j1))
            doc = plain if faulty else skipped
            assert (doc, skipped, plain) == (('満たさない', '使ってよい', '満たさない') if faulty else ('使ってよい',) * 3), (kind, shape)
            assert (doc, skipped, plain) == want[(kind, shape)], (kind, shape)


def test_words_match_the_code():
    """文書が引く言葉・数は今のコードにある"""
    co = (JF / 'experiments' / '20261026_fix' / 'diag_launch' / 'collect.py').read_text(encoding='utf-8')
    assert '当たった本は外さない' in co and "ap.add_argument('--skip'" in co
    cc = (JF / 'tools' / 'car_check.py').read_text(encoding='utf-8')
    for w in ("'%s 衝突(AI・立て直し中とリカバリー中%s): %s'", '発進の空転の比(発進の前進 ON', "'!' if ls['warn'] else ' '",
              '→ 比が出ない発進は動画で空転を見る', "return ('記録が欠けている(%s。"):
        assert w in cc, w
    app = (JF / 'jetformer' / 'web' / 'static' / 'app.js').read_text(encoding='utf-8')
    assert "CH3: AI側 ・ リカバリーをやめて中立" in app
    from jetformer import assist as A
    from jetformer import drive
    assert A.Recovery.STUCK_S == 0.8 and A.DEFAULT_REV_S == 0.4 and A.DEFAULT_REV_US == 300.0
    assert A.Recovery.GAP_S == 0.15 and A.Recovery.RETRY_GAP_S == 0.3 and A.Recovery.MAX_RETRIES == 2
    assert A.Recovery.STOP_ERPM == 300 and A.Recovery.GIVEUP_N == 3 and A.Recovery.GIVEUP_WINDOW_S == 20.0
    assert A.ASSIST_NAMES[A.ASSIST_GIVEUP] == 'リカバリー中止'
    assert drive.PARAM_LIMITS['rev_s'] == (0.2, 1.2) and drive.PARAM_LIMITS['rev_us'] == (150.0, 500.0)
    assert '2026-10-10 の +180µs は弱かった' in (JF / 'jetformer' / 'assist.py').read_text(encoding='utf-8')


def test_restart_marks_match_the_system():
    """approvals の 08:40 の印: jetformer-web の起動は 10/28 08:40:33 のままで、jetformer/ の .py にそれより新しいものが無い(読むだけ)。
    このあと assist.py・drive.py・server.py・bench.py が直されて再起動したら落ちて知らせる → 印の文を今に合わせる。
    10/29 fix9(この印のあと): 今の drive.py が fix9 の版(sha256)なら、起動は 10/29 08:51:40・08:40:33 より新しい .py は drive.py だけ
    (その直す前の写しが印のときの版 3f2e31a2…)・10/29 08:51:40 より新しい .py は無い(印は 10/28 の記録として正しい)"""
    out = subprocess.run(['systemctl', 'show', 'jetformer-web', '-p', 'ActiveEnterTimestamp'], capture_output=True, text=True).stdout
    if not out.strip():
        pytest.skip('systemctl show が読めない')
    fix9 = _sha(JF / 'jetformer' / 'drive.py') == LATER_FIX9_PRIORITY['jetformer/drive.py']
    assert ('2026-10-29 08:51:40' if fix9 else '2026-10-28 08:40:33') in out, out
    boot = datetime(2026, 10, 28, 8, 40, 33).timestamp()
    newer = [p.name for p in (JF / 'jetformer').rglob('*.py') if p.stat().st_mtime > boot]
    assert newer == (['drive.py'] if fix9 else []), newer
    if fix9:
        assert _sha(FIX9_PRIORITY_BEFORE / 'drive.py').startswith('3f2e31a22ecb')
        newer9 = [p.name for p in (JF / 'jetformer').rglob('*.py') if p.stat().st_mtime > LATER_FIX9_BOOT.timestamp()]
        assert newer9 == [], newer9
