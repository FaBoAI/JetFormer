# シミュレーター統合の本番確認(sim_integration_20261013 / verify)

- 実施: 2026-10-13 17:00〜17:55(Jetson の再起動 16:08 の後)。対象は本番の jetformer-web(:8080)。
- 結論: **本番の画面で 20 項目すべて OK**(最終の実行 17:49:49〜17:51:46、`verify/verify_result.json`)。
  確認の途中で **shim の不具合を 1 つ見つけて直した**(127.0.0.1 で開いたページが、収集の終了後に固まる)。画面の小さな不具合も 1 つ直した。
  テスト一式は 656 passed・3 skipped・1 deselected(`logs/pytest_full_4_verify.log`)。
- 実車・PCA9685・GPU は使っていない。ブリッジの推論は CPU の onnxruntime で動かした。git commit はしていない。

## 1. jetformer-web の再起動

- 再起動の前に条件を確かめた(17:00)。学習・走行・収集・セッション・実行中のジョブはどれも動いていなかった。
  - `train running=false`
  - `drive phase=idle`
  - `recorder.recording=false`
  - `session.active=false`
  - running のジョブ 0 件
- `systemctl restart jetformer-web` を実行し、2 秒後に `/api/sim/status` が 200 を返した。
  - 版 `20261013-144013-6f610dd5`、Python 環境は準備済み(3.13.14 / onnxruntime 1.31.0)、ブリッジは停止中。
  - コースは `default`(v3、`0a464384`)と `indoor-course-eval-2-shortcut`(v2、`547d7e80`)の 2 つ。
- 再起動したのはこの 1 回だけ。その後に直した 2 つのファイル(sim-shim.js と app.js)は静的ファイルで、毎回ディスクから配信されるため、再起動はいらない。
  sim-shim.js はページ側で `?v=<mtime>` が付き、app.js は no-cache で配信される。

## 2. 確認の方法

- `verify/verify_live.py` を jf-heavy で実行した(`--mem 2500M --lock big`)。
  - geckodriver(ポート 4444)と Firefox 157 を WebDriver で操作した。
  - 画面での操作は、利用者と同じように要素をクリックし、ファイルを選び、確認ダイアログに答えた。
  - API は、状態を確かめるときと後片付けにだけ使った。
- **WebGL のため、ヘッドレスではなく DISPLAY=:0 の Firefox を使った。**
  - この Jetson のヘッドレス Firefox では WebGL 2 が使えない。別の作業 `sim_rl_bestlap_20261013/webgl_probe/default` で確認済み。
  - :0 では NVIDIA の OpenGL で描画できた(各ページで `webgl2: true`)。
- snap の Firefox は自分の systemd スコープに移るため、jf-heavy の上限から外れる。そのため、そのスコープにも `MemoryMax=2500M` を付けた。
  - この上限のおかげで、§4 の不具合で Firefox のメモリが増え続けたときも、OOM で止まったのは Firefox だけだった。Jetson は固まらなかった。
- 終わったときに、Firefox・geckodriver・ブリッジがすべて止まっていることを確かめた。`pgrep` は空だった。

## 3. 結果(最終の実行。写真は `screens/`)

| # | 確かめたこと | 結果 | 写真 |
|---|---|---|---|
| 0 | 学習・走行・収集・ジョブが動いていない | OK | |
| 1 | 「シミュレーター」タブを開く。版・Python 環境・ブリッジ(停止中)・コース Default と旧 v2(hash・線分などの数・使える教師)が出る。モデル B が選択肢にある。選べるモデルは 24、対応しないモデルは 8 で、対応しないものは理由つきで選べない | OK | `01_sim_tab.png` |
| 2 | タブの「起動」でブリッジが 1.8 秒で動作中になる。127.0.0.1:<空きポート> だけで待ち受け、LAN の IP からはつながらない。中継 `/sim/bridge/api/status` は、LAN の Host・Origin を付けても 200 | OK | `02_bridge_running.png` |
| 3a | 配信される scene.js・world.js の sha256 が、リリース(Default)とスナップショット(v2)の値と一致する。index.html に shim が入っている | OK | |
| 3b | `/sim/` を開くと `/sim/c/default/` に移る。WebGL で描画される。world hash `0a464384`、教師 5 つ、上の帯にコース選択。読み込みエラー 0 件 | OK | `03_default.png`・`03_default_camera.png` |
| 4a | 旧 v2 が WebGL で描画される。hash `547d7e80`、教師は slow と speed だけ、20 分の収集条件は隠れる | OK | `04_v2.png`・`04b_turf_default_vs_v2.png`(芝生の白い板の違い) |
| 4b | 帯のコース選択で v2 から Default に切り替わる | OK | |
| 5a | タブでコース Default とモデル B(`clean_driving_20261011-nocmd-s3-20261011-024626`)を選び、「新しいタブで開く」を押す。`?model=` の受け渡しで ZIP → bundle → activate が自動で進み、4.3 秒で「読み込み完了」 | OK | `05a_model_b_loaded.png` |
| 5b | 「学習済みAI / ONNX」で約 8 秒評価する。推論 12 回が中継経由でブリッジに届き、予測が返って車が 0.07 m 進む。画面の推論待ちは 241 ms(4 回目の実行では 182 ms) | OK | `05b_model_b_ai_running.png`・`05b_model_b_camera.png` |
| 6a | 経路追従デモで約 5 秒収集する(「自動走行の教師値」を使う)。「収集終了・ZIP作成」で 16 フレーム。保存リンクは `/sim/bridge/api/datasets/download/<id>` で、取得すると 200(ZIP) | OK | |
| 6b | シミュレーター内の「JetFormerへ送る」を使う。送り先は shim が `http://127.0.0.1:8080` に固定する。「接続・取り込み状況を確認」→「選択したZIPを送信」で、9 秒で「取り込み完了」 | OK | `06_dataset_a_sent_to_jetformer.png` |
| 7a・7b | 同じ条件でもう 1 つ収集し(21 フレーム)、タブの「JetFormer に取り込む」を押す。ジョブが完了し、一覧が「取り込み済み」になる | OK | `07_sim_tab_datasets_jobs.png` |
| 8a | データセット タブに 2 つとも出る(🎮 シミュレーターの印、simulator のタグ) | OK | `08_datasets_tab.png` |
| 8b | シーン タブに展開されたシーン `ds-<データセット名>-0001` が 2 つ出る | OK | `09_scenes_tab.png` |
| 9 | 確認用のデータセット 2 つ、シーン 2 つ、ブリッジの作業フォルダー 2 つを消す(すべて 200)。消した後にシーンは 198 件(確認前と同じ)で、`data/raw/ds-verify-*` もない | OK | |
| 10a | 旧 v2 のスナップショットの写し(world_version だけを `v2-copy-verify-174949` に変えた ZIP)を、タブの「ZIP を選ぶ」でアップロードする。種類は trusted-js(本物の v2 の JS)、hash は `547d7e80`。「同じ形のコースがあります」と警告が出て、一覧と選択肢に加わる | OK | `10a_course_uploaded.png` |
| 10b | そのコースを選んで開く。WebGL で描画され、hash は `547d7e80`、帯にコース名が出る。v2 の写真と画素単位で同じ(コース部分の差は 0 画素) | OK | `10b_uploaded_course_open.png` |
| 10c | 「削除」→ 確認ダイアログ →「OK」で一覧から消える。API とページはどちらも 404 になり、「登録しました」の表示も消える | OK | `10c_course_deleted.png` |
| 11 | タブの「停止」でブリッジが止まる。子プロセスは残らない | OK | |

- 中継の遅れは `verify/proxy_latency.json` で測った。比べたのは、ブリッジへ直接送る場合と JetFormer の中継を通す場合で、それぞれ 40 回。
  - status: 2.1 ms → 4.8 ms(中央値)
  - 収集の frame(163 KB の JSON): 15.0 ms → 22.7 ms
  - 中継で増える時間は 1 回あたり 3〜8 ms。
- Jetson の Firefox は、実行するたびに速さが変わった。
  - AI 評価の推論: 1 秒あたり 1.5〜4 回。
  - デモでの収集: 1 秒あたり 3〜6 フレーム。
  - 最終の実行では、別の作業の強化学習(CPU 200% と node 3 つ)が同時に動いていた。
  - 遅いのはブラウザ側の描画と画像の作成で、中継ではない。Mac のブラウザでは描画を Mac が受け持つので、速くなるはず(Mac での確認は §6)。

## 4. 見つけて直した不具合

### 4.1 sim-shim.js: 127.0.0.1 で開いたページが、収集を終えると固まる(重大・修正済み)

- **症状**
  - 1 回目と 2 回目の実行で、「収集終了・ZIP作成」の数十秒後に Firefox が固まった。
  - メモリが 2.5 GB の上限に達し、OOM で止まった。
    - 1 回目は Firefox の親プロセス(anon 1.48 GB)が止まった。
    - 2 回目はページのプロセス(1.29 GB)が止まった。
  - `journalctl -k`: `Memory cgroup out of memory: Killed process … (firefox / Isolated Web Co)`
- **原因**
  - shim の `fixLinks()` は、`#jetformer-open` などのリンクの href を MutationObserver で見張って書き換える。
  - このとき、書き換え後の値が今の値と同じでも `setAttribute` を呼んでいた。DOM は、同じ値の setAttribute でも変化の記録を作る。そのため、自分の書き込みで自分がまた呼ばれる無限ループになった。
  - これが起きるのは、ページを 127.0.0.1 や localhost で開いたときだけ。そのとき送り先(`http://127.0.0.1:8080`)は `location.origin + '/'` と同じになる。
  - アプリが href を入れ直すときに始まる。具体的には、収集を終えたとき(`setArchive` → `refreshLink`)と、「接続・取り込み状況を確認」を押したとき。
  - Mac から LAN のアドレスで開いた場合は、2 回目の書き換えでループ側の条件から外れるため、ループにならない。node のテストでも確かめた。
  - 前の段階のブラウザ確認では、ブラウザで収集をしていなかったので見つからなかった。
- **修正**
  - `jetformer/sim/static/sim-shim.js` の `fix()` を、値が変わるときだけ `setAttribute` するように変えた。
- **確認**
  - `verify/shim_loop_repro.py` で、修正の前と後を比べた。href を 1 回変えるだけの手順。
    - 修正前は、その 1 行の execute が 8 秒で返らなかった(`shim_loop_repro_before.json`)。
    - 修正後は 25 ms で返り、その後 10 秒間も応答した。メモリは約 720 MB のままだった(`shim_loop_repro_after.json`)。
  - 回帰テスト `tests/test_sim_js.py::test_shim_link_fix_does_not_loop` を足した。node と簡易的な DOM で、127.0.0.1 と LAN の 2 通りを試す。
    - 修正前は、127.0.0.1 の場合に呼び出しが 201 回(打ち切り)になり、テストは失敗した。修正後は通る。
  - 修正後の本番の画面で、最終の実行を含め 2 回とも 20/20 OK だった。

### 4.2 app.js: アップロードしたコースを削除しても「登録しました」の表示が残る(軽微・修正済み)

- 削除に成功したら、`#sim-course-upload` を空にし、「コース … を削除しました」とトーストを出すようにした。
- 最終の実行の 10c で確かめた(`upload_msg == ''`)。

## 5. 気づいたこと(不具合ではない・利用者への説明が要るもの)

1. **経路追従デモ(教師)だけで集めたデータは、そのままでは取り込めない。**
   - シミュレーターの既定では、手動走行だけが教師値になる。
   - 「自動走行の教師値」にチェックを入れずに集めると、JetFormer は「学習に使えるフレームがありません(future_valid・vehicle_valid・mode を確かめてください)」で取り込みを断る。
   - 3 回目の実行で起きた。エラーは処理状況に日本語で出ていて、仕組みどおりの動き。
   - 短すぎる収集も同じ扱いになる。最後の 10 フレームは future_valid が false になるため。
2. **「収集終了・ZIP作成」は、走行中は押せない時間がある。**
   - シミュレーター本体(app.js)は、毎フレームの送信が終わるまでこのボタンを disabled にする。
   - 一時停止してから押すと確実(確認スクリプトもそうしている)。
3. **v2 の写しのように JS ごとアップロードしたコース(trusted-js)では、レポートの world_version が元のまま(`indoor-course-eval-2-shortcut`)になる。**
   - 本物の world.js の定数が使われるため。world hash は同じ(`547d7e80`)。
   - JetFormer の一覧での名前と world_version は、アップロードしたもの(`v2-copy-verify-…`)になる。
   - 形を変えたコースは data 形式(簡易描画)になるので、world_version はアップロードしたものになる。
4. **収集のジャンルが 5(低速走行)になった。**
   - モデル B を読み込むと、シミュレーターが収集ジャンルをモデルの走行モード(5)に合わせるため。
   - 既定の 8(シミュレーター)のまま集めたいときは、収集の前に選び直す。
5. **ブリッジが受け取ったモデルの写しは、ブリッジを止めても残る。**
   - `data/sim/runtime/uploads/` に、ONNX(56 MB)と JSON が残る。設計どおり、次の起動のときに消える。
   - 今回のものは手で消した。止めるときにも消すと、より丁寧になる(未対応)。
6. **ジョブの履歴(処理状況)には、3 回目の実行で失敗した取り込みが 1 件残っている。**
   - ジョブはメモリにだけあるので、次に jetformer-web を再起動すると消える。
   - そのデータセット自体は作られていない。

## 6. 残っていること

- **ユーザーの Mac での確認(design.md §8.3 の 12)。**
  - 今回は Jetson 上の Firefox で、127.0.0.1:8080 として開いた。
  - LAN のアドレスで開いたときのリンクの書き換えと中継は、node のテストと、中継に LAN の Host を付けた確認(200)だけで確かめている。
  - Mac で確かめること:
    - Default での手動走行とデモ
    - タブからモデル B を選んで AI 評価
    - 1 シーンの収集と取り込み(タブから、シミュレーター内の「JetFormerへ送る」から)
    - v2 とアップロードしたコースへの切り替え
- コミットに含めるもの: 前の段階の一覧に加えて、今回直した `jetformer/sim/static/sim-shim.js`・`jetformer/web/static/app.js` と、テストの追加分 `tests/test_sim_js.py`。

## 7. ファイル

- `verify/verify_live.py`(本番の画面の E2E)・`verify/verify_result.json`(最終の結果)・`verify/run.log`
- `verify/proxy_latency.py`・`verify/proxy_latency.json`(中継の遅れ)
- `verify/shim_loop_repro.py`・`verify/shim_loop_repro_before.json`・`verify/shim_loop_repro_after.json`(不具合の再現と修正の確認)
- `verify/v2-copy-verify-174949.zip`(アップロードに使ったコースの ZIP。v2 のスナップショットの写しで、world_version だけを変えたもの)
- `screens/*.png`(最終の実行の写真)
- `screens/run1_failed/`・`run2_failed/`(OOM で止まった実行)と `screens/run3_failed/`(確認手順の誤り)の記録
- `logs/pytest_full_4_verify.log`(修正後のテスト一式: 656 passed・3 skipped・1 deselected)
