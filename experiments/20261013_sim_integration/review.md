# シミュレーター統合の厳しめのレビューと修正(sim_integration_20261013 / review)

- 実施日: 2026-10-13 18:20〜(17:57 の再起動のあと。前の段階の成果物はすべて残っていた。このレビューの途中のファイルはなかった)
- 対象: `jetformer/sim/` 一式、`jetformer/web/server.py` の 7 行、管理画面の「シミュレーター」タブ、`tests/test_sim_*.py`
- やり方: コードを読む(全ファイル)、本番の 8080 に読み取りだけの試し(越境のパス・キャッシュの見出し)、試験インスタンス(127.0.0.1:18080、
  `e2e/e2e_server.py`)で本物のブリッジを使った確認、CPU だけの pytest。
- jetformer-web は再起動していない。git commit もしていない。実車・PCA9685・GPU には触れていない。

## 0. 結論

- 前の段階までの作りは、セキュリティの土台(アップロードした JS を実行しない・中継は決まった API だけ・ブリッジは 127.0.0.1 だけ・
  ZIP とパスの検査・大きさの上限・出力のエスケープ)がしっかりしていて、越境や中継の抜け道は見つからなかった。
- 直したのは「電源断・強制終了・同時操作」のときの弱さが中心(9 件)。今日 Jetson が 2 回不意に再起動したので、ここを優先した。
- テスト: シミュレーターのテストは 159 → 169 件(新しい 10 件。孤児のブリッジのテストは、直す前のコードで落ちることを確かめた)。テスト一式の結果は §4。
- **直したコード(Python)は、次に jetformer-web を再起動したときに有効になる。** 起動スクリプト `bridge_runner.py` はブリッジを起動するたびに
  読み直されるので、次のブリッジの起動から有効(今の jetformer-web が渡す引数のままでも動く。見張りだけが働かない)。

## 1. 見つけて直したこと

| # | 重さ | 内容 | 直し方 | テスト |
|---|---|---|---|---|
| 1 | 中 | **JetFormer が強制終了するとブリッジが孤児として残る。** systemd の再起動では cgroup ごと止まる(KillMode=control-group)ので本番は大丈夫。手で動かした JetFormer や試験インスタンスを kill -9 すると、ブリッジ(モデル込みで数百 MB)とポートが残り続ける。8 GB・スワップなしではメモリ不足の原因になる | `bridge_runner.py` に `--parent-pid`(親がいなくなったら自分で止まる見張り)を追加し、`bridge_proc.py` が JetFormer の pid を渡す。**直している途中で、見張りが最初の作りでは動かないことを試験インスタンスで見つけた**(親と一緒に閉じたパイプへ書いて BrokenPipeError になり、止める前にスレッドが終わっていた)。書き込みの失敗を無視するようにして、本物のブリッジで kill -9 から 0.63 秒で止まることを確かめた | `test_runner_stops_when_parent_dies`(直す前のコードで落ちることを確認済み) |
| 2 | 中 | **メモリが尽きたときに、どれが止められるか決まっていない** | ブリッジの `oom_score_adj` を 500 にする(JetFormer やデスクトップより先にブリッジが止められる。ブリッジは止まっても自動で起動し直す) | `test_child_gets_parent_pid_and_oom_score` |
| 3 | 中 | **電源断に弱い書き方。** 導入の展開・準備の記録・`active.json`・コースの登録・既定の設定を fsync せずに書いていた。導入の直後(約 30 秒以内)に電源が切れると、「導入済み」の記録だけがあって中身が 0 バイトの版が残りうる。その版は「導入済み」なので導入し直せず、使っている版なので削除もできず、画面からは直せない | 展開が終わったら `os.sync()` してから置き、記録はそのあとに書く(1 ファイルずつの fsync は SD カードでは遅い: 50 件で 0.94 秒)。記録は `paths.write_durable()`(tmp に書いて fsync → 置き換え → フォルダーも fsync)。venv の準備も記録の前に `os.sync()`。コースは中身を fsync してから置く | `test_js_pairs_and_durable_records`・`test_delete_and_default_are_durable` |
| 4 | 中 | **作業途中のファイルが残り続ける。** 電源断や強制終了で `.upload-*.zip`(最大 64 MiB)・`.tmp-*`・`.bundle-*.zip`(モデル、約 60 MB)・`selftest-*` が残り、だれも消さない。さらにコースの置き換えが 2 つの `os.replace` の間で止まると、コースが一覧から消える(前のコースは `.old-*` に残る) | `courses.sweep_stale()` と `web.sweep_stale()`: 起動時(`web.init`)とコースの登録の前に、10 分より古い作業ファイルを消す。`.old-*` は、元の場所が空いていれば戻す。削除は名前を変えてから消す(消している途中で切れても半分のコースが出ない) | `test_sweep_stale_after_power_loss`・`test_init_sweeps_stale_temp_files` |
| 5 | 中 | **同じコースを同時に送ると 500。** 2 つが同時に「まだない」と判断し、2 つ目の `os.replace` が ENOTEMPTY で落ちる。テクスチャの検査のメモリ(最悪 250 MB 程度)も同時に積み上がる | コースの登録・置き換え・削除・既定の変更を 1 つのロック(`courses._LOCK`)で順番にする | `test_concurrent_uploads_of_same_course_do_not_500`(4 つ同時 → 登録 1・登録済み 3)。試験インスタンスでも 4 つ同時に送って 200 × 4 |
| 6 | 低〜中 | **「信頼する JS」の範囲が広すぎた。** trusted-js は scene.js と world.js の sha256 が「導入済みの版のどれかのファイル」と別々に一致すればよかった。そのため、版のほかの JS(`dist/app.js`・テストなど)を scene.js として動かしたり、v3 の scene.js と v2 の world.js を組み合わせたりできた。版のコードは信頼しているので被害は小さいが、意図(コースの JS だけを動かす)と違う | 版の中で **同じフォルダーにそろっている (scene.js, world.js) の組**(今の版では `dist/` と v2 のスナップショットの 2 組)と一致するときだけ trusted-js にする(`install.js_pairs`・`Release.trusted_js`)。登録済みの trusted-js も、JS を書き換えられたら簡易描画に落ちる | `test_trusted_js_needs_the_release_pair`。試験インスタンスで本物の版を使い、v2 の組 → trusted-js、`app.js` を scene.js にしたもの・v3 の組を入れ替えたもの → data |
| 7 | 低 | **「JetFormer に取り込む」を 2 回押すと、同じ ZIP の取り込みが 2 本動く**(2 本目は「Directory not empty」で失敗する)。取り込み中に作業フォルダーを消すこともできた | 取り込み中のセッション ID を覚え、2 回目と削除は 409(日本語の説明つき) | `test_import_twice_and_delete_while_importing_are_409` |
| 8 | 低 | **中継のパスの照合が `re.match` + `$`** だった。`$` は末尾の改行の前でも一致するので、`status\n` が通りうる(httpx の InvalidURL で 500 になる)。今は Starlette の経路が末尾の `%0a` を落としてから渡すので届かない | `fullmatch` に変更(多層防御) | `test_trailing_newline_is_not_proxied` |
| 9 | 低 | 数字でない `mode_id`(`/api/sim/bridge/load_model`)や `Content-Length`(コースのアップロード)が 500 になる。ブリッジが受け取ったモデルの写し(56 MB)が、ブリッジを止めても次の起動まで残る(verify.md §5 の 5) | 400 で日本語の説明を返す。ブリッジを止めるときにも `runtime/uploads/` を空にする | `test_runner_listens_on_loopback_and_stops` に追加 |

## 2. 確かめて問題がなかったこと

### 2.1 セキュリティ
- **パスの越境**(本番の 8080 に読み取りだけで試した。すべて 404): `/sim/c/default/..%2f..%2fbridge.py`・`%2e%2e/bridge.py`・`..%5cbridge.py`・
  `assets/..%2f..%2fbridge.py`・`_jf/../../bridge.py`・`/sim/static/..%2fweb.py`・`/sim/c/..%2f..%2f/index.html`・`%00`・`/sim/c/DEFAULT/`・
  `/sim/bridge/api/..%2f..%2fbridge.py`。パスは要素ごとに `..`・空・`\`・`:`・NUL を断り、`resolve()` 後にベースの中か・途中にシンボリックリンクが
  ないかを確かめている。`/sim/static/` は 4 つの名前だけ。コースの ZIP・本体の ZIP の中のパス・種類(リンク・特殊ファイル・暗号化)も検査している。
- **中継**: 通すのはメソッドとパスの 12 組だけ(それ以外は 404・405)。クエリを渡すのはモデルの送信 2 つだけ。Host は `127.0.0.1:<ポート>` に書き換え、
  渡すヘッダーは Content-Type・Content-Length・Accept だけ(Origin・Referer・Cookie は捨てる)。POST は Content-Length が必須で chunked は断る。
  本文はためずに流し、組ごとの上限(推論 2 MiB〜モデル 300 MiB)を超えたら切る。`/api/jetformer/*` の送り先はこの Jetson の JetFormer に固定。
- **ブリッジは 127.0.0.1 だけ**: bind はユーザーの `bridge.py` の中で `("127.0.0.1", port)` に固定。起動スクリプトはポートしか渡さない。
  ブリッジ自身も Host と Origin を確かめる(LAN の Host は 403。verify 段階で LAN の IP からつながらないことを確認済み)。
- **大きさ**: コースの ZIP は受け取りながら 64 MiB で切る・展開後 128 MiB・256 件・圧縮率 200 倍まで・physics-world.json 4 MiB・JSON の
  NaN/Infinity と深さ 8 段超えを拒否・テクスチャ 1 枚 8 MiB・4096 px・16 枚(ヘッダーで大きさを見てから展開し、再エンコードする)。
  本体の ZIP は 4 GiB・展開後 2 GiB・5000 件・全ファイルを sha256 で照合。
- **アップロードしたコースの JS**: 実行しない(版の組と完全に一致するときだけ版の JS を使う。#6 で範囲を狭めた)。data コースは JetFormer の汎用描画が
  サーバーの作る JSON(`js_json`: `<` `>` `&` もエスケープ)だけを読む。コース名などは画面で `textContent` か `esc()` で出す。
  テクスチャは再エンコードし、拡張子で Content-Type を決め、`nosniff` を付ける(SVG は octet-stream)。**このリスクは十分小さく、許容できる。**
- **残るリスク(前からあるもの。今回の統合で広げていない)**:
  - JetFormer には認証も CSRF 対策もない。LAN の誰でも、また LAN のブラウザで開いた悪意のあるページからも、本文なしの POST
    (たとえば `/api/sim/bridge/start`)は送れる。PUT・DELETE・JSON の POST は CORS の事前確認で止まる(JetFormer は CORS を許可していない)。
    これは既存の `/api/*` 全体と同じ。対策(トークンや Origin の確認)は別の作業として提案する。
  - シミュレーターのページは JetFormer と同じオリジンで動くので、版の JS と shim は JetFormer の API を呼べる(どちらも信頼するコードとして扱う設計)。
    CSP(design.md §7 の 6、P2)は入れていない。

### 2.2 コースの切り替えの正しさ(本番の 8080 で確認)
- 配った scene.js・world.js の sha256 が、Default は `dist/`(`878fc934…`・`40998601…`)、v2 はスナップショット(`6e4a7265…`・`ec3a813d…`)と一致。
- `_jf/config.js` はコースごとに違い(world hash `0a464384` と `547d7e80`)、ETag も違う。どのファイルも `Cache-Control: no-cache` + ETag。
  同じ URL の再確認は 304、別のコースの URL に同じ ETag を送っても 200(キャッシュはコースの URL ごとなので混ざらない)。
- `/sim/` は 307(キャッシュされない)で既定のコースへ。既定のコースを消すと Default に戻る。

### 2.3 丈夫さ
- **ブリッジの異常終了**: 10 分間に 3 回まで 2・5・15 秒おいて起動し直し、超えたら失敗(ログの末尾つき)。モデルは読み込み直しが必要で、ページの帯が知らせる。
- **jetformer-web の再起動**: systemd(KillMode=control-group・KillSignal=SIGINT)がブリッジにも SIGINT を送り、終了処理も止める。ブリッジは自動では起動しない。
  ジョブの履歴はメモリにだけあるので消える(取り込みの結果のデータセットは残る)。
- **電源断**: §1 の #3・#4。17:57 の不意の再起動のあとに確かめたところ、導入した版の 849 ファイルはすべて sha256 が一致、venv に 0 バイトの .so・.py なし。
- **大きな本文**: 中継もコースのアップロードもためずに流して上限で切る。
- **同時に使う人**: ブリッジは全体で 1 つ・モデルも 1 つ。収集は同時に 1 つだけ(ブリッジが断る)。**2 つのページで同時に AI 評価をすると結果が正しくない**:
  モデルの前回の出力の記憶(`JetFormerController` の episode)が呼ばれるたびに入れ替わって消えるため。別のページがモデルを読み込み直すと、
  もう一方は「モデルが変わった」(409)になる。シミュレーター本体の作りなので JetFormer 側では直さず、画面(「同時に AI で走らせられるのは 1 つのタブだけ」)と
  README に書いた。

### 2.4 ユーザーのシミュレーターのファイル
- `~/sim/course-simulator-complete.zip`: sha256 `62f6e566…`(導入のときと同じ)。
- `~/sim/extracted/course-simulator/`: COMPLETE-MANIFEST.json の 851 ファイルがすべて大きさと sha256(50 MB 未満のもの)で一致。14:52 より後に変わったファイルなし。`__pycache__` なし。
- 導入した写し `data/sim/releases/20261013-144013-6f610dd5/`: 849 ファイル(除外した dataset-downloads・runtime 以外)がすべて一致。
- 今回もユーザーのファイルは 1 つも変えていない(直したのは JetFormer のファイルだけ)。

## 3. 直さなかったこと・あとでやるとよいこと

1. 認証・CSRF(§2.1。既存の JetFormer 全体の課題)と、`/sim/c/*` の CSP。
2. ブリッジのメモリの上限(今は jetformer-web の cgroup の中で上限なし)。`oom_score_adj` で先に止められるようにはした。上限を付けるなら
   `systemd-run --user --scope -p MemoryMax=` で包む方法がある(jetformer-web の再起動のときの止め方と合わせて考える必要がある)。
3. JS ごとアップロードしたコース(trusted-js)のレポートの world_version は元のまま(verify.md §5 の 3。形は同じなので害はない)。
4. あとから入れた版に、アップロード済みのコースと同じ id の同梱コースがあると、アップロードしたほうが一覧に出なくなる(今の版では起きない)。
5. ユーザーの Mac のブラウザでの確認(verify.md §6。LAN のアドレスで開いたときのリンクの書き換えなど)。
6. jetformer-web の再起動(このレビューの Python の修正を有効にするため)。条件は design.md §8.2 と同じ(学習・収集・走行・実行中のジョブがない)。
   再起動したら `curl http://127.0.0.1:8080/api/sim/status` が 200 を返すことと、タブからブリッジを起動して `/proc/<pid>/oom_score_adj` が 500 になることを確かめる。

## 4. テスト

- シミュレーターのテスト(12 ファイル): 直す前 159 passed・1 skipped(16 秒) → 直したあと **169 passed・1 skipped**(約 45 秒)。
  fsync が増えたぶん遅くなった(SD カードなので fsync 1 回が約 20 ms)。
- 試験インスタンス(127.0.0.1:18080。`sim/` は本物へのリンク、コースは試験用のフォルダー)で本物のブリッジを使った確認: 下の表がすべて期待どおり。
  結果は `review/smoke_result.json`、スクリプトは `review/smoke.py`。

  | 確認 | 結果 |
  |---|---|
  | ブリッジの起動・中継(LAN の Host と Origin を付けて) | running・200 |
  | ブリッジの `oom_score_adj`・`--parent-pid` | 500・試験インスタンスの pid |
  | `/sim/bridge/api/status%0a`・`%0d%0a`・`%0aX` | 200(Starlette が末尾の改行を落として status として扱う)・404・404。500 なし |
  | v2 の組のコース / `app.js` を scene.js にしたもの / v3 の組を入れ替えたもの | trusted-js(`547d7e80`)/ data / data |
  | 同じコースを 4 つ同時に送る | 200 × 4(登録 1・登録済み 3) |
  | 試験インスタンスを kill -9 | ブリッジが 0.63 秒で自分で止まった(直す前は 15 秒たっても残っていた) |
  | 後片付け | 試験のコース 4 つを削除・作業フォルダーは空・ブリッジとサーバーのプロセスなし |

- テスト一式(`jf-heavy --mem 2500M --lock big`、`CUDA_VISIBLE_DEVICES=`、`--deselect tests/test_session.py::test_run_lasts_exactly_from_start_beep_to_end_beep`):
  結果は §4.1。

### 4.1 テスト一式の結果

- 途中の版(`logs/pytest_full_5_review.log`): 666 passed・3 skipped・1 deselected(4 分 27 秒)。
- **最後の版(すべての修正のあと。`logs/pytest_full_6_review_final.log`): 666 passed・3 skipped・1 deselected・失敗 0(4 分 4 秒)。**
  verify 段階の 656 件に、このレビューで足した 10 件が加わった。ほかの機能(収集・データセット・学習・走行など)のテストはすべてそのまま通っている。
- テスト一式は data/ を集めない(`pytest.ini`)ので、導入した写しに `__pycache__` はできていない(確認済み)。

## 5. 変えたファイル(すべて `/home/jetson/jetformer` の中。コミットはメインセッションが行う)

- `jetformer/sim/paths.py`: `fsync_dir`・`fsync_file`・`write_durable` を追加
- `jetformer/sim/install.py`: `js_pairs`・`Release.trusted_js`、展開のあと `os.sync()`、記録と `active.json` を `write_durable` で
- `jetformer/sim/env.py`: 記録の前に `os.sync()`、記録を `write_durable` で
- `jetformer/sim/courses.py`: `_LOCK`・`sweep_stale()`・組での信頼(`_trusted_js`)・登録の中身の fsync・削除は名前を変えてから・既定の設定を `write_durable` で
- `jetformer/sim/web.py`: 起動時の `sweep_stale()`・取り込み中の 409・`mode_id` と `Content-Length` の確認
- `jetformer/sim/proxy.py`: `fullmatch`
- `jetformer/sim/bridge_runner.py`: `--parent-pid`(見張り)・`--oom-score-adj`・止めるときの `uploads/` の片付け・出力先が閉じていても止まらないように
- `jetformer/sim/bridge_proc.py`: `--parent-pid <JetFormer の pid>` と `--oom-score-adj 500` を渡す
- `README.md`: シミュレーターの節に「制限・注意」を追加
- テスト: `tests/test_sim_{proxy,courses,install,datasets,bridge_proc,runner}.py` に 10 件追加(と `test_runner_listens_on_loopback_and_stops` に確認を追加)
- 静的ファイル(`sim-shim.js`・`app.js`・`index.html`)は変えていない
