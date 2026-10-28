===== README.md
[+11 新しい行] - **10/28 03:3x 追記**(fix8 係 docs-decided。下の 23:5x・22:xx のあと。控え `README.md.bak-20261028-fix8-decided`。報告 `fix8/decided_docs.md`): 0 章 7・8・3.2 の 4・6 章の頭・6.1 の 3・5・6.2 の 8・26・29・32・6.3 の 31・6.4・7 章を今に(jetformer-web は 10/28 00:53:23 にもう一度再起動・commit 9da6d0d・反証 8 回目の途中・drive.py の説明の文のため反証 8 回目のあとにもう一度再起動・pytest 一式は `-o tmp_path_retention_policy=failed` で約 0.6 GB)。
[24→25 insert] …/27 の直しと反証 2〜6 回目。1635 ファイル)を push **済**【- 】【+ ・**9da6d0d**(10/28 00:56。反証 7 回目の直し)も push **済**】
[25→26 insert] …fix7 安全の直しは次の再起動か 11/3 の朝の Jetson の起動で入る【- 】【+  → **10/28 00:53:23 のもう一度の再起動で入った**】
[131→132 insert] …fy.md`)。空き 8 GB を切らずに流すには、始めに約 15 GB 要る。【- 】【+ **10/28 の一式は `-o tmp_path_retention_policy=failed`(通った試験の一時ファイルをすぐ消す)を付けて約 0.6 GB**(10/28 00:08〜00:52: 空き 10493M → いちばん少なくて 9913M(00:44)。終わりの作業場所 55M。`fix7/pytest_full_20261028.txt`)。空き 8 GB を切らずに流すには、始めに約 8.6 GB(上の約 15 GB と 7.1 GB は付けない形)。**一式はこのオプションを付けて流す**。】
[131→132 replace] …ix3/disk/disk_check.py` の `PYTEST_GB` は 【- 3】【+ 0】
[131→132 replace] …3/disk/disk_check.py` の `PYTEST_GB` は 3.【- 9】【+ 6(10/28】
[131→132 insert] …disk/disk_check.py` の `PYTEST_GB` は 3.9 【- 】【+ fix8 tools。付けないとき】
[131→132 delete] …isk/disk_check.py` の `PYTEST_GB` は 3.9 の【- まま】【+ 】
[131→132 replace] …/disk_check.py` の `PYTEST_GB` は 3.9 のまま 【- = 「pytest 一式の最中」の空きを約 3】【+ 7】
[131→132 replace] …T_GB` は 3.9 のまま = 「pytest 一式の最中」の空きを約 3.【- 2】【+ 1】
[131→132 replace] … は 3.9 のまま = 「pytest 一式の最中」の空きを約 3.2 GB 【- 多く出す(作業場所の差】【+ は `PYTEST_GB_KEEP`】
[131→132 insert] …約 3.2 GB 多く出す(作業場所の差)。10/27 fix7 docs r1【- 】【+ ・10/28 fix8 docs】
[+192 新しい行] - (10/28 fix8 係 docs-decided)fix7 の走りの直しは **10/28 00:53:23 のもう一度の再起動で入った**(6.2 の 8)。commit **9da6d0d**(10/28 00:56)を push 済み。試験一式は 10/28 にも 3415 passed(6.1 の 3)。**反証 8 回目の途中**: 最初の係が決めを入れた(`fix8/decided_tools.md` = approvals 8 の「fix7 で残した読み方」①② を夜の道具に・`fix8/decided_safety.md` = approvals 3 の (a) の前提の確かめと drive.py の説明の文・`fix8/decided_docs.md` = 文書)。11/3 のあとに向けた安全の案の係はまだ動いている。drive.py の説明の文(10/28 02:33。コメントだけ)のため、**反証 8 回目のあとにもう一度再起動**(6.2 の 8・32)。
[198→200 insert] …4 passed。`fix6/pytest_full_20261027.txt`【- 】【+ 。10/28 00:08〜00:52 にもう一度 **3415 passed**・2 failed・4 skipped。落ちた 2 件は試験を 81a4d34 に固定して 15 passed。`fix7/pytest_full_20261028.txt`。作業場所は 3.2 の 4】
[200→202 insert] …| ~~commit~~ **済**(fce5d47 を 10/27 17:19【- 】【+  に push。反証 7 回目の直しは **9da6d0d** を 10/28 00:56】
[208→210 insert] …あとに直したコード」の行・`todo_user_1026.md` 4 章の 5)【- 】【+ → **10/28 00:53:23 に main session がもう一度再起動して入った**(3.1 の「起動より新しいコードなし」)。10/28 02:33 の drive.py の説明の文(コメントだけ。動きは同じ)のため、**反証 8 回目のあとにもう一度再起動**(それまで 3.1 の「起動のあとに直したコード」は drive.py の名前を出す)】
[214→216 insert] …**: 10/27 19:36 からの fix7 安全の直し → 6.2 の 8【- 】【+ 。10/28 00:53:23 に入った】
[216→218 insert] …| なるべく早く(11/3 の前のもう一度の再起動は今も要る = 6.2 の 8【- 】【+  → 10/28 00:53:23 に済。**反証 8 回目を始めた** = 6 章の頭・6.2 の 32】
[+220 新しい行] | 32 | 反証 8 回目の残り(11/3 のあとに向けた安全の案の係。`fix8/proposal_safety_after1103.md`)と、そのあとの jetformer-web の再起動(10/28 02:33 の drive.py の説明の文を入れる。コメントだけ。手順は `todo_user_1026.md` 4 章の 1〜5)・試験一式・commit | main session | 11/3 の前 |
[236→239 insert] …章。安全の形と読み方の知らせは `fix3/approvals.md` 14・8【- 】【+ 。10/28: `drive.py` の説明の文・`runs_1103` の印の文と本「—」の区切り・`PYTEST_GB` は fix8 の係が直した(`fix8/decided_safety.md`・`fix8/decided_tools.md`)。夜の道具の印の付かない走り直しのうち、リカバリーの途中で人側にしたあとは印が付くようになった(approvals 8 の ①)】
[242→245 insert] …7 19:36 からの fix7 安全の直しは次の再起動のあと(6.2 の 8)【- 】【+ → 10/28 00:53:23 のもう一度の再起動のあとも、GET・journal・「起動より新しいコードなし」・本物の画面を読むだけで見た(走行はまだ)】
[+299 新しい行] | `fix8/decided_tools.md`・`fix8/decided_safety.md`・`fix8/decided_docs.md` | 反証 8 回目の最初の係(10/28): main session の決め(approvals 8 の ①②)を夜の道具に・approvals 3 の (a) の前提の確かめと drive.py の説明の文・文書を今に |
===== fix6/README.md
[+7 新しい行] - 10/28 03:3x fix8 係 docs-decided(控え `README.md.bak-20261028-fix8-decided`): 7 章 1 に「10/28 00:53:23 のもう一度の再起動で fix7 の直しが入った・一式は `-o tmp_path_retention_policy=failed` で約 0.6 GB」。報告 `../fix8/decided_docs.md`。
[82→83 insert] …き 11 GB → 3.4 GB。`../README.md` 3.2 の 4)【- 】【+ 。→ 10/28: fix7 の直しは 10/28 00:53:23 のもう一度の再起動で入った・一式は `-o tmp_path_retention_policy=failed` を付けて約 0.6 GB(`../fix7/README.md` 7 章・`../README.md` 3.2 の 4)】
===== fix7/README.md
[+9 新しい行] - (10/28 03:3x fix8 係 docs-decided。控え `README.md.bak-20261028-fix8-decided`)7 章の「次に」の済んだ行に印だけ付けた(再起動は 10/28 00:53:23・試験一式 3415 passed と commit 9da6d0d・反証 8 回目を始めた・approvals 3 は (a))。ほかは 10/27 23:5x の記録のまま。
[110→111 insert] …| 1 | 【- 】【+ **済**(10/28: 反証 8 回目を始めた)。】
[111→112 insert] …| 2 | 【- 】【+ **済**(10/28 00:53:23。drive.py の説明の文のぶんは反証 8 回目のあと)。】
[112→113 insert] …| 3 | 【- 】【+ **済**(10/28: 3415 passed・commit 9da6d0d を push)。】
[113→114 insert] …| 4 | 【- 】【+ **済**(10/28: approvals 3 は (a)・①〜④ は fix8 の係が直した・⑤ は再起動の行だけ(approvals 6・「ON の形」の注の行はまだ))。】
