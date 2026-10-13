# シミュレーター統合の実装と立ち上げの記録(sim_integration_20261013 / implementation)

- 実施日: 2026-10-13(16:10 の再起動の後に再開。前回の途中の成果物 `jetformer/sim/{__init__,paths,ziputil}.py`・`tests/fixtures/sim/`・`data/sim/wheels/` を確かめて続きから作った。壊れたファイルはなかった)
- 設計: `design.md`(この版の実装はそれに従う。違うところは §3)
- git commit はしていない(メインセッションが行う)。jetformer-web は再起動していない(§5)。

## 1. できたもの

| 区分 | ファイル(すべて `/home/jetson/jetformer` の中) |
|---|---|
| 導入・環境・プロセス | `jetformer/sim/install.py`・`env.py`・`requirements-bridge.txt`・`bridge_runner.py`・`bridge_proc.py`・`datasets.py`・`paths.py`・`ziputil.py` |
| 配信・中継 | `jetformer/sim/web.py`(Router・`init`・`shutdown`)・`proxy.py`・`static/sim-shim.js`・`static/sim-shim.css` |
| コース | `jetformer/sim/worldhash.py`・`courses.py`・`coursedata/known_courses.json`・`coursedata/default-v3-physics-world.json`・`static/course-scene.js`・`static/course-world.js` |
| 画面・組み込み | `jetformer/web/server.py`(+7 行)・`jetformer/web/static/{index.html,app.js,style.css}`(「シミュレーター」タブ)・`README.md`(「シミュレーター」の節) |
| テスト | `tests/test_sim_{install,env,runner,bridge_proc,worldhash,courses,web,proxy,datasets,models,js,real}.py`・`tests/_sim_testlib.py`・`tests/fixtures/sim/*`・`pytest.ini` |
| 確認用(リポジトリの外) | `e2e/e2e_server.py`(試験インスタンス)・`e2e/e2e_check.py`・`e2e/e2e_browser.py`・結果 `e2e/e2e_result.json`・`e2e/browser/result.json` と画面写真 |

## 2. 実物の導入(済み)

- `~/sim/course-simulator-complete.zip` → `data/sim/releases/20261013-144013-6f610dd5/`(851 ファイル・254 MB、2.7 秒)。全ファイルを COMPLETE-MANIFEST.json の sha256 と照合。
  `dataset-downloads/`(709 MB)と `runtime/` は入れていない。`最初にお読みください.txt` は入れて警告に記録。`data/sim/active.json` = この版。
- `~/sim` の ZIP は変わっていない(大きさ・mtime・sha256 `62f6e566…` を前後で確認)。導入した写しにも `__pycache__` はない。
- Python 環境 `data/sim/venv`(221 MB): jetracer の Python 3.13.14 から作り、`data/sim/wheels/` の wheel からオフラインで入った(12 秒)。
  numpy 2.2.6・opencv-python-headless 4.13.0.92・onnxruntime 1.31.0・protobuf 7.36.2・flatbuffers 25.12.19・packaging 26.3。ブリッジの self-test 13/13。
- コース: Default = 既知の v3(`0a464384`)、同梱 = 旧 v2(`547d7e80`)。アップロードしたコースは今はない(確認で作ったものは消した)。

## 3. 設計から変えたところ

1. **同梱データの置き場所**: `jetformer/sim/data/` は `.gitignore` の `data/` に当たって git に入らないので、`jetformer/sim/coursedata/` にした。
2. **`pytest.ini` を追加**(`norecursedirs` に `data` を足しただけ。ほかは pytest の既定と同じ)。導入した写し `data/sim/releases/<版>/tests/` に
   ユーザーのテストがあり、テスト一式がそれを集めて import してしまう(= ユーザーのコードが動く。onnxruntime がなくて収集エラーにもなる)ため。
   1 回目の全体テストでこれが起きた(import の途中で止まり、`PYTHONDONTWRITEBYTECODE=1` なので写しには何も書かれていない)。
3. **汎用描画は照明を足さない**: app.js が Hemisphere と Directional を入れており、本物の scene.js も照明を足さないため(足すと明るさが変わる)。
4. **`/api/sim/datasets`** は `{datasets, incomplete, free_gb, in_flight}`(保存されていない作業フォルダーも消せるように)。
   **コースのアップロードのエラー**は `{detail, errors:[…]}`(既存の `api()` は detail を出し、タブは一覧で出す)。
5. **env.status()** は 2 秒ごとに呼ばれるので venv の Python を実行しない(pyvenv.cfg の home と python のリンク先があるかだけを見る)。版の完全一致は準備の最後に確かめる。
6. ブリッジの子プロセスの環境に `CUDA_VISIBLE_DEVICES=''`・`OMP_NUM_THREADS=4`・`OPENBLAS_NUM_THREADS=2` を入れた(GPU を使わないことの念押し)。
7. テクスチャの名前は設計の正規表現どおり `.jpg`・`.png` だけ(`.jpeg` はエラーで知らせる)。
8. P2 のうち入れたもの: `POST /api/sim/bridge/load_model`(ブラウザなしの確認用)、node での JetFormer の JS のテスト、本物のブリッジのテスト。
   入れていないもの: 前の版の Default を残す、同梱データセットの取り込みボタン、idle 停止、preview.png、同梱モデルの選択肢、CSP。

## 4. テスト

- 全体(`CUDA_VISIBLE_DEVICES= nice python -m pytest -q --deselect tests/test_session.py::test_run_lasts_exactly_from_start_beep_to_end_beep`、jf-heavy 経由):
  最後の版で 642 passed・3 skipped・1 deselected(`logs/pytest_full_3.log`、3 分 12 秒)。その後の変更は使っていない import を 3 つ消しただけ(該当テストは再実行して通過)。
- シミュレーターのテストは CPU だけ(GPU・実車・ネットワークなし)。pip・venv は差し替え。ブリッジは偽物(標準ライブラリだけ)と、導入済みの本物(`test_sim_real.py`。
  未導入なら skip)。node のテストは JetFormer の JS(汎用描画・shim)だけを動かし、ユーザーの JS は動かさない。

## 5. 立ち上げの確認(jetformer-web は再起動していない)

jetformer-web の起動処理は PCA9685 の出力を止める(`_release_outputs`)ので、この段階では再起動せず、新しい経路だけを持つ試験インスタンス
(`e2e/e2e_server.py`、127.0.0.1:18080、server.py は import しない)で確かめた。データの置き場所は試験用(`sim/` と `runs/` だけ本物へのリンク)。

**design.md §8.3 の 1〜11(`e2e/e2e_result.json`)**: すべて OK。
- ブリッジは 1.3 秒で running、`127.0.0.1:<空きポート>` だけで待ち受け(LAN の IP からはつながらない)。中継は LAN の Host・Origin を付けても 200。
- コース: Default を書き出して world_version と板 1 枚を変えたものが data で登録、再送は「登録済み」、形だけ違うと 409、`../` は 400。
- ページ: 3 コースとも shim が入り、scene.js・world.js の sha256 が期待どおり、carpet.jpg 200。
- モデル: `clean_driving_20261011-jetformer_v0-s1-ft-real-lr1e-4-e12` を bundle → activate(8 秒)→ predict 200(1 回 70〜120 ms、ORT 4 スレッド)。
- データ: 中継経由で start → frame 15 → finish → 一覧 → 取り込み(シーンタブへの展開も)→ runtime の片付け。停止後に子プロセスは残らない。

**Firefox(DISPLAY=:0、WebGL あり、jf-heavy 経由。`e2e/browser/`)**: 11 項目中 10 OK。NG の 1 つは `/favicon.ico` の 404(ブラウザが勝手に取りに行くもの)を
数えていただけで、スクリプトは直した(Firefox は再実行していない)。確かめたこと:
- 管理画面の「シミュレーター」タブが表示される(版・Python 環境・2 コース・33 モデル)。
- アップロードしたコース(簡易描画)が動き、全ファイル 200、テクスチャ読み込み後に「評価を開始」が押せる(assetsReady)、ページの world hash が登録時と同じ(`b0d462cd`)。
- 帯・コース名・教師が slow/speed だけ・20 分の収集条件を隠す・接続先は読み取り専用・送り先は `http://127.0.0.1:8080`。
- `?model=` で JetFormer のモデルが自動で読み込まれ(7 秒)、AI 評価で走る(推論 38 回が中継経由でブリッジに届いた)。
- Default は `0a464384`(教師 5 つ)、旧 v2 は `547d7e80`(教師 2 つ)。
- 終了後、Firefox・geckodriver・ブリッジ・試験インスタンスはすべて止めた。

## 6. メインセッションに残すこと

1. **jetformer-web の再起動**(新しいコードを読み込むため)。design.md §8.2 の条件(学習・収集・走行・実行中のジョブがない)を満たすときだけ。
   16:5x 時点では満たしていた(train running=false、drive idle、recorder/session なし、running のジョブなし)。
   再起動後に `curl http://127.0.0.1:8080/api/sim/status` と `http://<jetson>:8080/sim/` を確かめる。ブリッジは自動では起動しない(タブの「起動」)。
2. **ユーザーの Mac での確認**(design.md §8.3 の 12): Default で手動・低速デモ、`?model=` で AI 評価、1 シーン収集 →「JetFormer に取り込む」、
   v2 とアップロードしたコースへの切り替え、ダウンロードのリンク、シミュレーター内の「JetFormerへ送る」(LAN のオリジンでの shim の書き換え)。
   Jetson の Firefox では 127.0.0.1 で開いたので、Mac(LAN のアドレス)からの書き換えは node のテストでだけ確かめている。
3. コミットに入れるもの: §1 の表(`jetformer/sim/` 一式・`coursedata/` を含む)、`tests/test_sim_*.py`・`tests/_sim_testlib.py`・`tests/fixtures/sim/`、`pytest.ini`、
   README・server.py・画面の 3 ファイル。別の実験のファイル(`experiments/20261013_sim_rl_bestlap/`・`tools/course_track.py` など)は混ぜない。
