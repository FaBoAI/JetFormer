"""読むだけ: 直したい文(候補)が文書にちょうど 1 回あるか・前の new のどれに掛かるか・足す所(‸)がどの new の中か"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spans_final import spans, inside, REPORTS, FILES
RB, TODO, RD, AP = 'runbook_20261103.md', 'todo_user_1026.md', 'README.md', 'fix3/approvals.md'
texts = {f: (REPORTS / f).read_text(encoding='utf-8') for f in FILES}
sp, bad = spans(texts)
print('new が 1 回でない:', bad)
C = [
 (RB, 'R0 log', '‸\n- **10/28 11:4x fix8 係 docs 直す 8-2 回目**'),
 (RB, 'R1 3.1', '読み直した画面の ▶ の横は「画面の版 9(app.js 10/28 04:54)」(このあと誰かが app.js を直せばその時刻)。**04:54 より前に開いた画面**(スマホで開いたまま・前の日の画面)は ▶ で 409「管理画面が古いままです(この画面の app.js は …、今のファイルは 10/28 04:54 に直したもの)」→ 読み直す'),
 (RB, 'R2 4-6', '「説明欄の V_before(走行のあいだ一定)」なら OK。‸「なし(1.00 倍。周ごとの直しだけ)」ならキャンセル'),
 (RB, 'R3 4-8', '8. 決めた周を走ったら、次の直線で CH3 を人側 → ■ 停止。‸\n   - **止まって 3 周の本は'),
 (RB, 'R3b ff', '(10/28 04:54 の app.js を読み直した画面。'),
 (RB, 'R4 6', '前はその本を黙って外して「使ってよい」と出た)。‸当たった・人が危なくて止めた → 回 4 は OFF。'),
 (RB, 'R5 3-off', '⑤ はそれを「R1 に届かなかった発進」・その場で戻した発進を走り直しにする =「使ってよい」の側には動かない。'),
 (RB, 'R5b 3-off-end', '= この形でも「使ってよい」の側には動かない(8.1 ⑤・`fix3/approvals.md` 3)。‸\n\n### 3.1'),
 (RB, 'R6 7-1m', '3 章の「10/23 の形の後ろの近い物」・`fix3/approvals.md` 3。10/28 fix8 safety r1)。‸人側にしたあとは'),
 (RB, 'R7 8.1-5 tools', '(どちらの行も出ない。`fix8/fix_tools_r1.md` 3 章)‸ |'),
 (RB, 'R8 8.1-5 skip', '`collect.py` の `--skip` の説明「当たった本は外さない」と同じ向きで、空転・欠けも(10/28 main session の決め・fix8 docs r2)。‸「途中で変えた N 本」'),
 (TODO, 'T0 title', '# ユーザーが 11/3 までにすること(10/28 11:4x 現在)'),
 (TODO, 'T1 log', '‸- **10/28 11:4x fix8 係 docs 直す 8-2 回目**'),
 (TODO, 'T2 app', '今の app.js は 10/28 04:54(▶ の横「画面の版 9(app.js 10/28 04:54)」。このあと app.js を直せばその時刻)。'),
 (TODO, 'T3a 1章', '`README.md` 6.2 の 33。10/28 fix8 docs r2)‸)。6(E5'),
 (TODO, 'T3b 2章10', '`README.md` 6.2 の 33。10/28 fix8 docs r2)。‸11/3 は朝の準備で'),
 (TODO, 'T4 2章16', '8 の 3 つのうち ①② は 10/28 に main session が変えた = 2 章 13)‸ |'),
 (RD, 'D0 log', '‸\n- **10/28 11:4x 追記**'),
 (RD, 'D1 6.2-11', '画面の版 9(app.js 10/28 04:54)に読み直してから'),
 (RD, 'D2 6.2-37', '‸\n\n### 6.3 11/3 のあとでもよい(小さいもの)'),
 (RD, 'D3 6-head', 'ユーザーに伝えることは 6.2 の 33。‸\n\n### 6.1'),
 (RD, 'D5 6.3-38', '‸\n\n### 6.4 確かめていないこと(主張)'),
 (AP, 'P0 log', '‸- **10/28 11:4x fix8 係 docs 直す 8-2 回目**'),
 (AP, 'P1 3-ON', '動く・空転は同じ。`fix8/decided_safety.md` 1.2・1.6 節)‸ |'),
 (AP, 'P1b 3-OFFday-a', '両方 ON でも止まらない形も同じに外す)‸。0.15〜0.22 秒の短い止まり'),
 (AP, 'P1c 3-OFFday-b', '「使ってよい」が境目から 0.1 秒以内なら main session が動画で確かめて決める)‸。0.15 秒以下は前からの'),
 (AP, 'P2 8-row', 'OFF が全部外れたら満たさない(off_no_stall)。'),
 (AP, 'P2b 8-row-end', '0.15〜0.22 秒の短い止まりは比べに入れて注意の行に名前‸ |'),
 (AP, 'P2c 8-newrow', '‸\n\n- 車のせいでない理由(人・物がコースに入った)で止めた本は'),
 (AP, 'P4 14-row', '(最大 1.83 m/s。直す前の同じ場面の当たりの最大と同じ)‸ |'),
 (AP, 'P4b 14-new', '11/3 のリカバリーの記録で決め直す)のまま |‸'),
 (AP, 'P5a 0-3', '11/3 のコードは変えない。返事が無ければ (c) のまま)‸ | 決まり |'),
 (AP, 'P5b 0-8', '③ はこのまま)‸ | 11/2 |'),
 (AP, 'P5c 0-14', '| 14 | 10/27 fix7(反証 7 回目)で残した安全の形(知らせ) | **このまま**(11/3 は今のコード。返事が無ければこのまま)‸ | 11/2 |'),
 (AP, 'P6 3-why', '1.17〜2.91 m'),
]
for f, name, x in C:
    t = texts[f]
    if '‸' in x:
        y = x.replace('‸', '')
        n = t.count(y)
        if n != 1:
            print('%-16s %s: 文書に %d 回' % (name, f, n)); continue
        pos = t.index(y) + x.index('‸')
        print('%-16s 足す所 → %s' % (name, inside(sp, f, pos) or 'どの new の中でもない'))
    else:
        n = t.count(x)
        if n != 1:
            print('%-16s %s: 文書に %d 回' % (name, f, n)); continue
        a = t.index(x); b = a + len(x)
        own = sorted({(k, i) for s, e, k, i in sp.get(f, []) if s < b and a < e})
        inner = sorted({(k, i) for s, e, k, i in sp.get(f, []) if s <= a and b <= e})
        print('%-16s 掛かる %s / 中に含む %s' % (name, own or 'なし', inner or 'なし'))
