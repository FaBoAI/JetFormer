# シミュレーター(CourseLab)を JetFormer Web UI に統合する設計書(sim_integration_20261013 / design)

- 作成日: 2026-10-13
- 依頼(ユーザー): 「/home/jetson/sim/course-simulator-complete.zip にシミュレータをアップしました。WebUIに統合してください。現在のコースはDefaultコースですが、データをアップしたら、ほかのコースも選べるようにしたいですね」
- 元にした調査: `understand_bridge.md`(ブリッジ)、`understand_app.md`(ブラウザアプリ)、understand_course(コース。報告本文のみ。要点は本書の §4 に取り込んだ)
- 対象: `/home/jetson/jetformer`(git main。FastAPI `jetformer/web/server.py`、systemd `jetformer-web`、:8080)と、ユーザーのシミュレーター `/home/jetson/sim/course-simulator-complete.zip`(展開済みの写しは `/home/jetson/sim/extracted/course-simulator/`)
- この段階では設計だけを行った。コード・jetformer-web・ユーザーのファイルは変えていない。

---

## 0. 要約と決定事項

### 0.1 できあがりの姿
1. JetFormer の画面に「シミュレーター」タブが増える。そこで本体の導入、Python 環境の準備、ブリッジの起動と停止、コースの選択とアップロード、モデルを選んで開く、集めたデータの取り込みができる。
2. シミュレーターは `http://<jetson>:8080/sim/c/<コースID>/` で開く。描画(WebGL)は Mac のブラウザ、ONNX の CPU 推論とデータ保存は Jetson 上のブリッジが受け持つ。
3. コースは URL で選ぶ。`default`(今の Default = v3)、同梱の旧 v2、アップロードしたコースが選べる。ページを読み込み直すたびに、その URL のコースになる。
4. JetFormer で学習したモデルは `?model=<run>` を付けて開くと、自動でシミュレーターに読み込まれる。
5. シミュレーターで集めたデータセットは、タブのボタンで JetFormer に直接取り込める。ファイルの送り直しはしない。シミュレーター内の「JetFormerへ送る」ボタンもそのまま使える。

### 0.2 主な決定(実装する人はこれに従う)
| 項目 | 決定 |
|---|---|
| ユーザーのファイル | 一切変えない。ZIP から JetFormer が持つ場所に写して使い、その写しも変えない(`-B` で `__pycache__` も作らない) |
| 本体の置き場所 | `data/sim/releases/<release_id>/`(ZIP の `course-simulator/` の中身。`dataset-downloads/` と `runtime/` は除く)。使う版は `data/sim/active.json` で決める。`data/` は git 管理外 |
| 本体の導入元 | Jetson 上のフォルダー `~/sim/`(`JETFORMER_SIM_INBOX` で変更可)にある ZIP だけ。ブラウザからのアップロードでは本体(=Python コード)を入れない |
| Python 環境 | JetFormer の Python(conda `jetracer` の 3.13)から作る専用 venv `data/sim/venv`。JetFormer 側で固定した版とハッシュ(`jetformer/sim/requirements-bridge.txt`)で入れる。jetracer env には何も入れない |
| 準備のきっかけ | タブのボタン → JetFormer の処理状況(jobs)で実行。サーバー起動時には何もしない |
| ブリッジ | JetFormer が持つ起動スクリプト `jetformer/sim/bridge_runner.py` を venv の Python で子プロセスとして動かす。`bridge.BridgeServer` を直接作り、127.0.0.1 の空きポート(OS が選ぶ)で待ち受ける。`--policy` は使わない |
| 起動の方針 | ボタンを押したときだけ起動する(サーバー起動時は起動しない)。異常終了したら 10 分間で 3 回まで自動で再起動し、それを超えたら「失敗」で止める |
| 配信 | シミュレーターの静的ファイルは JetFormer が自分で配信する(ブリッジの静的配信は使わない)。`/sim/c/<cid>/` で、`scene.js`・`world.js`・`assets/` だけをコースごとに差し替え、ほかは `dist/` から返す |
| API の中継 | ブリッジの API は `/sim/bridge/api/*` で中継する。JetFormer 自身の管理 API は `/api/sim/*`。どちらも JetFormer の既存の `/api/datasets/*`・`/api/models/*` と衝突しない |
| ブリッジ URL の問題 | app.js は変えず、JetFormer が持つ小さなスクリプト(shim)を index.html に差し込み、`http://127.0.0.1:8766/api/...` への通信を `/sim/bridge/api/...` に書き換える |
| コースの JS | アップロードされた JS は実行しない。JS で描くのは、導入したリリースの COMPLETE-MANIFEST.json と SHA256 が一致するもの(Default、同梱 v2)だけ。それ以外のコースは `physics-world.json`(データ)を JetFormer の汎用描画 JS で描く |
| コースの形式 | シミュレーター作者のスナップショット形式(`manifest.json` + `physics-world.json` [+ JS + assets])と、JetFormer 形式(`course.json` + `physics-world.json` + assets)の ZIP |
| 既定のコース | `default`(使っているリリースの dist のコース = 今は v3)。タブで別のコースを既定にもできる |
| モデルの受け渡し | ブラウザ側(shim)が `/api/models/<run>/bundle.zip` を取り、アプリの「ZIPからモデルを読み込む」を自動で押す。アプリ本来の流れ(トークン、モード、ゲイン)がそのまま使える |
| データの受け渡し | タブから `dataset_import.import_archive()` をブリッジの ZIP に直接かける(既存のジョブ)。アプリ内の送信は、送り先を `http://127.0.0.1:8080` に固定して中継する |

### 0.3 全体の流れ
```
Mac のブラウザ ──http──> JetFormer :8080 (uvicorn, jetracer env)
   /             JetFormer の画面(「シミュレーター」タブを追加)
   /api/sim/*    管理 API(導入・準備・起動停止・コース・データ取り込み)  [jetformer/sim/web.py]
   /sim/c/<cid>/ シミュレーターのページ(dist + コースの差し替え + shim)     [jetformer/sim/web.py]
   /sim/static/* shim など JetFormer のファイル                           [jetformer/sim/static/]
   /sim/bridge/api/* ──httpx(非同期・ストリーム)──> ブリッジ 127.0.0.1:<空きポート>
                                                     [data/sim/venv の python -I -B jetformer/sim/bridge_runner.py]
                                                     └ bridge.BridgeServer(… runtime_root=data/sim/runtime)
                                                         ├ onnxruntime CPU(4 スレッド・nice 10)
                                                         └ /api/jetformer/* → http://127.0.0.1:8080/api/datasets/import(従来の送信)
```

### 0.4 優先度
- **P1**: 今回の実装で必ず入れる。ユーザーの依頼(統合 + コース選択 + アップロード)に必要なもの。
- **P2**: 余裕があれば入れる。入れなくても依頼は満たせる。

---

## 1. 導入(install)

### 1.1 データの置き場所(すべて `config.DATA_DIR` = `/home/jetson/jetformer/data` の下。git 管理外)
```
data/sim/
  releases/<release_id>/        ZIP の course-simulator/ の中身(dataset-downloads/・runtime/ を除く)。導入後は読み取り専用として扱う
  releases/<release_id>.json    導入記録 {release_id, source_file, source_bytes, manifest_sha256, created_at(manifest), installed_at,
                                          files, bytes, skipped:[...], warnings:[...], default_course{world_version, scene_sha256, world_sha256}}
  active.json                   {"release": "<release_id>"}(書くときは .tmp → os.replace)
  venv/                         ブリッジ用の Python(§1.3)
  venv.json                     準備の記録 {python, python_version, packages{名前: 版}, requirements_sha256, created_at, self_test:"ok"|...}
  wheels/                       (任意)オフライン用の wheel。あれば PyPI より先に使う
  runtime/                      BridgeServer の runtime_root(uploads/、datasets/、jetformer-transfers.json)
  logs/bridge.log(.1, .2)       ブリッジのログ(§2.5)
data/sim_courses/
  settings.json                 {"default": "<course_id>"}
  <course_id>/                  アップロードしたコース(§4)
```
- パスは import 時に固定せず、関数で毎回 `config.DATA_DIR` から作る(`jetformer/sim/paths.py`: `sim_dir()`, `releases_dir()`, `runtime_dir()`, `courses_dir()`, `inbox_dir()`)。テストは既存のテストと同じく `monkeypatch.setattr(config, 'DATA_DIR', tmp)` だけで切り替えられるようにする。
- `inbox_dir()` は `os.environ.get('JETFORMER_SIM_INBOX', ~/sim)`。
- `release_id` は `<manifest の created_at を YYYYMMDD-HHMMSS>-<COMPLETE-MANIFEST.json の sha256 先頭8桁>`。今回の ZIP は `20261013-144013-6f610dd5` になる。

### 1.2 ZIP からの取り込み(何度でもやり直せる。新しい版にも使う)
**導入元**: `inbox_dir()` 直下の `*.zip`(通常ファイルのみ。シンボリックリンク・サブフォルダーは見ない。4 GiB まで)。
タブは、各 ZIP の中央ディレクトリと `COMPLETE-MANIFEST.json` だけを読んで(数ミリ秒)、「シミュレーター本体 / 導入済み / 対象外」を表示する。結果は (パス, 大きさ, mtime) でキャッシュする。

**判定**: ZIP 内の全エントリーが 1 つの最上位フォルダー(今回は `course-simulator/`)の下にあり、`<top>/COMPLETE-MANIFEST.json` の `schema == "course-simulator-complete-v1"` で、`files` が `[{path, bytes, sha256}]` の配列であること。

**手順**(`jetformer/sim/install.py: install_release(zip_path, activate=True, progress=None) -> dict`。ジョブで実行):
1. 中央ディレクトリの検査。次のどれかがあれば拒否する。
   - 絶対パス、`..`、`\`、`:`、NUL、空の要素
   - シンボリックリンクや特殊ファイル(`external_attr` の種類が通常ファイル・ディレクトリ以外)
   - 暗号化されたエントリー、stored/deflate 以外の圧縮方式
   - 大文字小文字を無視した重複
   - エントリー数 5000 超、展開後の合計(除外分を除く)2 GiB 超、圧縮率 200 倍超のエントリー
2. `COMPLETE-MANIFEST.json`(4 MiB まで)を読む。`parse_constant` で NaN/Infinity を拒否し、`files` の各 `path` にも 1 と同じ検査をする。
3. 除外: `dataset-downloads/`(709 MB。取り込み済みの 20 分データ)、`runtime/`、`__MACOSX/`、`.DS_Store`。除外したものは導入記録の `skipped` に残す。
4. 必須ファイルの確認: `bridge.py`、`dataset_recorder.py`、`jetformer_policy.py`、`jetformer_upload.py`、`calibration.example.json`、`requirements-model.txt`、`dist/index.html`、`dist/app.js`、`dist/scene.js`、`dist/world.js`、`dist/sim-core.js`、`qa/combine_scene_datasets.py`、`training/speed_physics.py`。
5. 空き容量の確認: 展開する量 + 1 GiB 以上の空きがなければ 507 相当のエラーにする(今の空きは約 25 GB、展開量は約 230 MB)。
6. `releases/.tmp-<uuid>/` に 1 MiB ずつ展開する。書きながら sha256 を計算し、manifest と一致しなければ全体を中止して削除する。ファイルは 0644、フォルダーは 0755 にする(ZIP の権限は使わない)。
7. manifest にないファイルの扱い:
   - `COMPLETE-MANIFEST.json` 自身は入れる。
   - 最上位の `*.txt`・`*.md`(今回の `最初にお読みください.txt`)は入れて、`warnings` に記録する。
   - それ以外(特に `.py`・`.js`)は入れずに `skipped` に記録する。
8. manifest にあるのに ZIP にないファイル(除外分を除く)があれば中止する。
9. `os.replace(.tmp-<uuid>, releases/<release_id>)` → 導入記録 `<release_id>.json` を書く。
10. `activate=True` で、ブリッジが止まっているか、まだ有効な版がなければ `active.json` を書き換える。ブリッジが動いているときは `activated:false` と「ブリッジを止めてから『この版を使う』を押してください」を返す。
- **やり直し**: 同じ `release_id` のフォルダーと記録がそろっていれば、展開せずに「導入済み」で成功にする(ジョブの result に `already_installed:true`)。片方しかなければ、作り直す。
- **新しい版**: 新しい ZIP を `~/sim/` に置いて「導入」を押す。古い版はそのまま残り、タブの「この版を使う」で切り替えたり、「削除」したりできる(削除できるのは有効でない版だけ)。
- **P2**: 版を切り替えるとき、前の版の Default コースを「前の版の Default」としてコース一覧に残す(§4.1 の trusted-js。JS は導入済みのリリースから来たものなので信頼できる)。
- **P2**: 同梱データセット(`dataset-downloads/*.zip`)を ZIP から一時ファイルに取り出し、`dataset_import.import_archive()` で取り込むボタン(今回の 20 分データはもう取り込み済み)。

### 1.3 ブリッジ用の Python 環境
- **作り方**: `sys.executable -m venv data/sim/venv`。JetFormer は jetracer env の Python 3.13.14 で動いているので、`sys.executable` がそのまま使える(パスを決め打ちしない)。system の python3.12 には venv・ensurepip がないので使わない。
- **入れるもの**(JetFormer 側で固定する。`jetformer/sim/requirements-bridge.txt`。ハッシュは scratchpad の wheel から計算した値):
```
# シミュレーターのブリッジ用(Python 3.13 / aarch64)。requirements-model.txt の版に合わせる
numpy==2.2.6 --hash=sha256:f92729c95468a2f4f15e9bb94c432a9229d0d50de67304399627a943201baa2f
opencv-python-headless==4.13.0.92 --hash=sha256:eb60e36b237b1ebd40a912da5384b348df8ed534f6f644d8e0b4f103e272ba7d
onnxruntime==1.31.0 --hash=sha256:ad663106f6eeff3d454f24a786450459d07f30e74863851104fc1b8b3f368127
protobuf==7.36.2 --hash=sha256:e11e1f0180583a2af89db6a2ecd9e8dc40aa6d2988ca175bfd0e6d12ea72d74e
flatbuffers==25.12.19 --hash=sha256:7634f50c427838bb021c2d66a3d1168e9d199b0607e6329399f04846d42e20b4
packaging==26.3 --hash=sha256:d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c
```
  - 入れ方: `venv/bin/python -m pip install --require-hashes --only-binary=:all: --no-deps -r requirements-bridge.txt`。`data/sim/wheels/` に wheel があれば `--no-index --find-links data/sim/wheels` を先に試し、だめなら `--index-url https://pypi.org/simple` を使う。
  - wheel(68 MB)は今 scratchpad(`/tmp/claude-1000/-home-jetson-jetracer/f44d13ba-776f-4620-a46a-2574918483a9/scratchpad/wheels/`)にある。消える前に、立ち上げの段階で `data/sim/wheels/` に写しておく。
  - **版の突き合わせ**: リリースの `requirements-model.txt` の 3 つの版(numpy・opencv-python-headless・onnxruntime)が上の固定版と違えば、準備は始めずに「シミュレーターが求める版が変わりました。JetFormer 側の固定版の更新が必要です」で止める(勝手に別の版を入れない)。
- **確認**(準備の最後と、タブの状態表示のとき):
  - `venv/bin/python -I -c "import numpy, cv2, onnxruntime; print(...)"` で版が完全一致すること。opencv はヘッドレス版の `4.13.0.92`。jetracer env の CUDA 版とは別物。
  - 準備の最後に、ブリッジの self-test(13 件)を `bridge_runner.py --self-test` で 1 回実行する。
- **壊れたとき**: jetracer env の Python を上げると venv が壊れる。状態表示で `venv/bin/python` が動かなければ「準備し直す」を出す。「準備し直す」は venv を消してから作り直す。
- **大きさ**: venv 約 221 MB、wheel 68 MB、リリース約 230 MB。

### 1.4 準備の起動(ボタン + ジョブ)
- `POST /api/sim/setup {"force": false}` → `jobs.submit('sim_setup', 'シミュレーターの準備', fn)`。進み具合は次のとおり出す。
  - 0.05「venv を作っています」
  - 0.1〜0.8「部品を入れています(pip)」。pip の出力の行数で進める
  - 0.85「版を確かめています」
  - 0.9「ブリッジの自己診断」
  - 1.0「準備できました」
- `POST /api/sim/install {"file": "<inbox の ZIP 名>", "activate": true}` → `jobs.submit('sim_install', 'シミュレーターの導入: <名前>', fn)`。`file` は inbox 直下の名前と完全一致するものだけ受け付ける(パスは受け付けない)。
- ブリッジが動いている間は、準備も版の切り替えも 409 で断る(「ブリッジを止めてから」)。
- 準備と導入は同時に 1 本だけ(モジュール内のロック)。
- JetFormer の起動時には導入・準備・ブリッジ起動のどれも行わない。

---

## 2. ブリッジのプロセス管理(`jetformer/sim/bridge_proc.py`、`jetformer/sim/bridge_runner.py`)

### 2.1 起動スクリプト `bridge_runner.py`(JetFormer が持つ。venv の Python で動き、JetFormer のコードは import しない)
```
data/sim/venv/bin/python -I -B jetformer/sim/bridge_runner.py \
    --sim-dir data/sim/releases/<id> --runtime-root data/sim/runtime \
    [--port 0] [--ort-threads 4] [--nice 10] [--self-test]
```
処理の流れ:
1. `os.nice(nice)` を呼ぶ。
2. `sys.path.insert(0, sim_dir)` を行い、`import bridge` する。bridge・dataset_recorder・jetformer_policy・qa・training は namespace package として読まれる。`-I` なので、cwd とスクリプトのフォルダーは sys.path に入らない。
3. `--self-test` のときは `bridge.main(['--self-test'])` の終了コードで終わる。
4. onnxruntime のスレッド数を絞る。`import onnxruntime as ort` して、`ort.InferenceSession` をサブクラスに差し替える。`sess_options` が渡されていなければ、`intra_op_num_threads=N`、`inter_op_num_threads=1` を入れる。jetformer_policy は関数の中で `import onnxruntime as ort` してから `ort.InferenceSession(...)` を呼ぶので、ファイルを変えずに効く。
5. `runtime/uploads/` の中身を消す。トークンはメモリにしかなく、再起動後は使えないため。2 GiB の上限で詰まらないようにする。
6. `server = bridge.BridgeServer(port, Path(sim_dir)/'dist', bridge.PolicyRunner(), runtime_root=runtime_root)` を作る。ブリッジは 127.0.0.1 に固定で bind し、Host の検査には実際のポートを使う。
7. stdout に `{"event":"listening","port":<実ポート>,"pid":<pid>}` を 1 行出して flush する。
8. SIGTERM を受けたら、別スレッドで `server.shutdown()` を呼ぶ。SIGINT は KeyboardInterrupt として扱う。
9. `server.serve_forever(poll_interval=0.2)` を回し、最後に `server_close()` する。
- `webbrowser.open` は呼ばない(main() を使わないため)。環境変数 `JETFORMER_MODEL/CONFIG/CALIBRATION` は親が渡さない。

### 2.2 管理クラス `BridgeProcess`
```python
class BridgeProcess:
    def __init__(self, *, python, runner, sim_dir, runtime_root, log_path, ort_threads=4, nice=10, port=0,
                 restart_limit=3, restart_window_s=600, backoff_s=(2, 5, 15), health_timeout_s=15,
                 popen=subprocess.Popen, clock=time.monotonic): ...
    def start(self) -> dict          # すぐ返す(state='starting')。起動できなければ RuntimeError(日本語)
    def stop(self, timeout=5.0) -> dict
    def status(self) -> dict         # 通信しない。{state, pid, port, started_at, restarts, last_exit, last_error, rss_mb}
    @property
    def port(self) -> int | None     # running のときだけ値を持つ
    def logs(self, n=200) -> dict    # {lines:[...], suppressed:{predict:n, frame:n}}
    def note_activity(self): ...     # 中継が呼ぶ(P2 の idle 停止に使う)
```
- 状態: `stopped → starting → running → stopping → stopped`。異常終了で再起動を使い切ったら `failed`。
- 外から使うのは `jetformer/sim/web.py` のモジュール変数 1 つだけにする(ブリッジは全体で 1 つ)。
- **start の流れ**:
  1. 次の前提を確かめる。満たさなければ 409 にする。
     - リリースが導入済みで、active.json があること
     - venv.json が ready であること
     - `MemAvailable` が 1 GiB 以上あること(8 GB・スワップなしで OOM 再起動の前歴があるため)
  2. `Popen([python, '-I', '-B', runner, ...], stdout=PIPE, stderr=STDOUT, stdin=DEVNULL, cwd=runtime_root, env=最小限)` で起動する。env は PATH・LANG・HOME と `PYTHONDONTWRITEBYTECODE=1`。
  3. 読み取りスレッドが stdout を 1 行ずつ読み、最初の `listening` 行からポートを得る(20 秒で時間切れ)。
  4. 監視スレッドが `GET http://127.0.0.1:<port>/api/status` を 0.5 秒ごとに送り(urllib、1 秒で時間切れ)、200 が返ったら `running` にする。15 秒で時間切れなら止めて `failed` にする。
- **stop**: SIGINT を送って 5 秒待つ → SIGTERM で 2 秒 → SIGKILL。`stopping` の間は再起動しない。
- **自動再起動**: 自分で止めていないのに終了した(`returncode` を問わない)とき。
  - 直近 10 分の再起動が 3 回未満なら、backoff(2, 5, 15 秒)を待ってから同じ引数で起動し直す。
  - 超えたら `failed` にして、`last_error` に最後のログ 20 行の要約を入れる。
  - 再起動すると、有効なモデルと収集中のセッションは消える。状態に `restarted_at` を入れ、shim の帯で「ブリッジが再起動しました。モデルを読み込み直してください」と出す。
- **JetFormer の終了**: lifespan の終了処理で `stop(timeout=3)` を呼ぶ。systemd の unit は `KillSignal=SIGINT` で、KillMode が既定の control-group なので、子にも SIGINT が届く。
  - jetformer-web を再起動するとブリッジも止まり、収集中のデータは失われる。これはタブに注意書きとして書く。
  - 再起動の後に自動では起動しない。
- **固定ポートが欲しいとき**: `JETFORMER_SIM_BRIDGE_PORT` で指定できる。使用中なら起動は失敗とし、黙って別のポートに切り替えない。
- **他の作業との関係**:
  - 学習中・AI 走行中でも起動はできる。CPU だけを使い、nice 10・4 スレッドで動く。タブに「学習中は推論が遅くなります」と表示する。
  - GPU は使わない(onnxruntime の CPU 版だけを入れる)。
- **P2**: 中継の通信が 60 分なく、収集中でもなければ止める(idle 停止)。

### 2.3 ブリッジの状態の取得
- `GET /api/sim/status` は、`BridgeProcess.status()` に加えて、ブリッジの `/api/status` の結果を入れる。取得は中継と同じ httpx.AsyncClient で、1 秒で時間切れ、2 秒キャッシュする。
- 入れるもの: `policy_loaded`、`name`、`dataset_collection.recording`、`model_upload.available`。
- RSS は `/proc/<pid>/status` の VmRSS から読む。

### 2.4 サーバーを止めないこと
- 起動・停止・導入・準備は、いずれも同期の `def` エンドポイントにしてスレッドプールで動かすか、ジョブに入れる。待ち時間の長い処理はイベントループに置かない。
- 監視と読み取りはデーモンスレッドで行う。
- ブリッジへの通信(中継・状態)は `httpx.AsyncClient` だけを使う。jetracer env に httpx 0.28.1 がある。
  - 理由: ブリッジの `/api/jetformer/check` と `upload` は、ブリッジから JetFormer を同期的に呼ぶ。JetFormer 側で同期の HTTP をイベントループの中で待つと、デッドロックになる。

### 2.5 ログ
- 読み取りスレッドが stdout と stderr(まとめたもの)を受け取り、2 か所に振り分ける。
  - (a) 直近 300 行のリングバッファ(タブの「ログ」に出す)
  - (b) `logging.handlers.RotatingFileHandler(data/sim/logs/bridge.log, 5 MB × 3)`
- ブリッジは 1 リクエストごとに `[bridge] POST /api/datasets/frame -> 200` を出す(収集中は約 30 行/秒)。次の 2 種類の 200 応答の行は書かずに数だけ数え、`suppressed` に出す。
  - `POST /api/predict -> 200`
  - `POST /api/datasets/frame -> 200`
- それ以外の行(エラーを含む)はすべて残す。
- journald には流さない。

---

## 3. HTTP の統合(`jetformer/sim/web.py` の APIRouter、`jetformer/sim/proxy.py`)

### 3.1 経路の一覧
| メソッド | パス | 中身 |
|---|---|---|
| GET | `/sim` | 307 → `/sim/` |
| GET | `/sim/` | 307 → `/sim/c/<既定のコース>/`(クエリは引き継ぐ) |
| GET | `/sim/c/{cid}` | 308 → `/sim/c/{cid}/`(**末尾の `/` は必須**。index.html は相対パスで読み込むため) |
| GET/HEAD | `/sim/c/{cid}/` と `/sim/c/{cid}/index.html` | 変換した `dist/index.html`(§3.3) |
| GET/HEAD | `/sim/c/{cid}/_jf/config.js` | 生成する設定スクリプト(§3.3) |
| GET/HEAD | `/sim/c/{cid}/_jf/course-data.js` | 生成するコースのデータ(data コースだけ。§4.6) |
| GET/HEAD | `/sim/c/{cid}/{path:path}` | ファイルの解決(§3.2) |
| GET/HEAD | `/sim/static/{name}` | JetFormer のファイル(`sim-shim.js`、`sim-shim.css`、`course-scene.js`、`course-world.js`) |
| GET/HEAD/POST | `/sim/bridge/api/{path:path}` | ブリッジの中継(§3.4) |
| | `/api/sim/*` | 管理 API(§3.5) |

- `cid` は `^[a-z0-9][a-z0-9-]{0,47}$` とする。合わなければ 404。
- `/sim/c/...` へのリクエストのうち、ページとそのファイルの取得には JetFormer の既存の `_NoCacheStatic` と同じく `Cache-Control: no-cache` を付ける。さらに ETag(`release_id`・パス・大きさ・mtime から作る)を付け、`If-None-Match` に 304 を返す。
  - `vendor/three.core.js` などを毎回送り直さないため。
  - three.js はコースの URL ごとに 1 回ずつ取得される(LAN なので許容する)。

### 3.2 `/sim/c/{cid}/{path}` のファイルの解決
```
course = courses.get_course(cid)                         # なければ 404
rel    = safe_rel(path)                                   # 下記の検査。だめなら 404
if rel in ('', 'index.html'):         変換した index.html
elif rel == '_jf/config.js':          生成(classic script)
elif rel == '_jf/course-data.js':     生成(data コースのみ。ほかは 404)
elif rel.startswith('_jf/'):          404(JetFormer の予約名前空間)
elif rel in ('scene.js', 'world.js'):
    course.kind == 'release'     → <release>/dist/<rel>
    course.kind == 'snapshot'    → <release>/qa/course-snapshots/<id>/<rel>   (SHA256 は登録時に確認済み)
    course.kind == 'trusted-js'  → data/sim_courses/<cid>/<rel>               (同上)
    course.kind == 'data'        → jetformer/sim/static/course-scene.js / course-world.js
elif rel.startswith('assets/'):  course.assets_dir/<name> があればそれ、なければ <release>/dist/assets/<name>
else:                            <release>/dist/<rel>
```
- `safe_rel` の検査:
  - Starlette が URL をデコードした後の文字列に対して行う。
  - 要素が空でないこと。`.` と `..` を含まないこと。`\`・NUL・`:` を含まないこと。全体が 256 文字以下であること。
  - 最後に `resolve()` した結果がベースのフォルダーの中にあり、通常ファイルであることを確かめる。途中にシンボリックリンクがあれば拒否する(`os.lstat` で確認)。
- 返す種類(拡張子で固定する):
  - `.js`・`.mjs` → `text/javascript; charset=utf-8`。モジュールは MIME を厳しく見るため。
  - `.css` → `text/css`、`.html` → `text/html`、`.json` → `application/json`、`.jpg` → `image/jpeg`、`.png` → `image/png`。
  - それ以外は `application/octet-stream`。
  - すべてに `X-Content-Type-Options: nosniff` を付ける。
- `diagnosis/` はユーザーのリリースにある診断ページで、信頼できるので、そのまま配信してよい。

### 3.3 index.html の変換と shim
**変換**(`dist/index.html` を mtime でキャッシュする):
- 次の 3 つがそろっているかを確かめる。どれかが欠けたら 500「シミュレーターの index.html の形が想定と違います(版が変わった可能性があります)」にする。
  - `<meta charset="utf-8">` がちょうど 1 つ
  - `<script type="importmap">` がちょうど 1 つ
  - `src="./app.js"` が 1 つ
- `<meta charset="utf-8">` の直後に次の 3 つを入れる。classic script はモジュールより先に動くので、app.js が通信を始める前に書き換えが効く。
```html
<script src="./_jf/config.js"></script>
<script src="/sim/static/sim-shim.js?v=<mtime>"></script>
<link rel="stylesheet" href="/sim/static/sim-shim.css?v=<mtime>">
```
- **HTML にユーザーの文字列(コース名など)を埋め込まない。** 表示の差し替えは shim が DOM の `textContent` で行う。

**`_jf/config.js`**(生成。`json.dumps(..., ensure_ascii=True)` の後で `<`・`>`・`&` を `<` などに置き換え、`window.JETFORMER_SIM = Object.freeze(<json>);` とする):
```json
{"course": {"id": "default", "name": "Default", "world_version": "indoor-course-eval-3-turf-board-removed",
            "world_hash": "0a464384", "renderer": "simulator", "has_shortcut": true,
            "size_m": [10.85, 6.8],
            "features": {"teachers": ["slow","speed","speed-fast","speed-rc","speed-optimized"], "collection_plan": true}},
 "bridge": "/sim/bridge", "jetformer_port": 8080, "courses_api": "/api/sim/courses", "status_api": "/api/sim/status",
 "home": "/", "release": "20261013-144013-6f610dd5"}
```

**`jetformer/sim/static/sim-shim.js`**(JetFormer のファイル。classic script):
1. **ブリッジへの通信の書き換え**: understand_app.md §3.3 のコードを使う。
   - `window.fetch` と `XMLHttpRequest.prototype.open` をラップする。
   - 書き換えるのは、`http://` の絶対 URL のうち、origin が「今の `#endpoint` の値の origin」(読めなければ `http://127.0.0.1:8766`)と一致し、パスが `/api/` で始まるものだけ。行き先は `location.origin + '/sim/bridge' + pathname + search` にする。
   - 相対 URL は書き換えない(JetFormer の `/api/...` を呼ぶため)。
   - `bridgeURL()` はループバックの偽の origin を返し続けるので、app.js の origin の確認(351 行・428 行)も通る。
   - app.js 57 行の処理(ページが localhost で開かれたときに `#endpoint` を `location.origin` にする)も、「今の endpoint と比べる」ので混線しない。
2. **リンクの書き換え**: `#dataset-download`・`#collection-download`・`#jetformer-open` の `href` を MutationObserver で見張る。前の 2 つは `/sim/bridge/api/datasets/download/<id>` にし、`#jetformer-open` は `location.origin + '/'` にする。
3. **入力欄**: `#endpoint.readOnly = true` にし、近くに「JetFormer 経由で接続します」と表示する。
4. **localStorage**:
   - モジュールより前に `course-jetformer-url` を `http://127.0.0.1:<jetformer_port>` にする。
   - `course-20s-scene-ids`・`course-20s-collection` の 2 つのキーに限り、`Storage.prototype.getItem/setItem/removeItem` をラップして、末尾に `:<world_version>` を付ける。コースが違う 20 秒シーンの ID が混ざらないようにするため。
   - 失敗しても無視する(try/catch)。
5. **上の帯**(JetFormer の DOM。`body` の先頭に入れ、文字は textContent だけで書く):
   - 「JetFormer に戻る」のリンク
   - コースの選択(`/api/sim/courses` から作る。選ぶと `location.href = '/sim/c/<id>/' + 今の ?model`)
   - ブリッジの状態(`/api/sim/status` を 5 秒ごとに確認)。止まっていれば「起動」ボタン(`POST /api/sim/bridge/start`)を出す。
   - モデルの読み込み状況
   - data コースなら「簡易描画のコース(カメラ画像は Default と同じにはなりません)」と出す。
   - ページの遷移は app.js の `beforeunload` が収集中に確認を出すので、安全に切り替えられる。
6. **コース向けの表示の調整**:
   - `.view-label` をコース名にし、`.view-note span` を `size_m` にする。`document.title` に「 — <コース名>」を足す。
   - `features.teachers` に入っていない `#teacher-profile` の option を消す。選ばれていた値が消えたら `slow` にする。
   - `features.collection_plan` が false なら `#collection-plan` を隠し、「Default コースだけで使えます」と出す。
   - `has_shortcut` が false なら `#shortcut-status` を隠す。
7. **自動接続**: `window.courseSimulator` が現れたら(app.js の最後で作られる。10 秒待つ)、`#connect` を 1 回押す。中身は GET /api/status だけ。
8. **モデルの受け渡し**: §5.1。
- この shim は JetFormer が持つファイルで、ユーザーのファイルには入れない。
- シミュレーターの版が変わって要素の id が見つからないときは、その機能だけを飛ばし、`console.warn` と帯への注意を出す(ページは壊さない)。

### 3.4 ブリッジ API の中継 `/sim/bridge/api/{path}`(`jetformer/sim/proxy.py`)
**中継する組み合わせ**(これ以外は 404 にし、本文は `{"error":{"code":"not_proxied","message":"..."}}`):
| メソッド | パス(`api/` の後) | 本文の上限 | 読み取りの時間切れ | 備考 |
|---|---|---|---|---|
| GET/HEAD | `status` | なし | 10 秒 | |
| POST | `predict` | 2 MiB | 15 秒 | AI 走行中は毎フレーム |
| POST | `models/upload` | 256 MiB | 300 秒 | クエリ `kind`・`name` を引き継ぐ |
| POST | `models/bundle` | 300 MiB | 300 秒 | クエリ `name` |
| POST | `models/activate` | 2 MiB | 180 秒 | |
| POST | `datasets/start`, `datasets/frame` | 2 MiB | 30 秒 | |
| POST | `datasets/finish` | 16 MiB | 300 秒 | |
| POST | `datasets/collection` | 2 MiB | 900 秒 | ブリッジは同期的に ZIP を作る |
| GET/HEAD | `datasets/download/<32hex>` | なし | 1 回の読み取りごとに 60 秒 | 700 MB 級をストリームで返す |
| POST | `jetformer/check`, `jetformer/upload` | 64 KiB | 120 秒 / 60 秒 | 本文の `url` を書き換える(下記) |
| GET | `jetformer/status/<id>` | なし | 15 秒 | |

**要求の扱い**:
- `Host: 127.0.0.1:<実ポート>` にする。ブリッジは Host を完全一致で確かめる。
- ブリッジに渡すヘッダーは `Content-Type`・`Content-Length`・`Accept` だけにする。`Origin`・`Referer`・`Cookie` などは渡さない(ブリッジは LAN の Origin を 403 にする)。
- POST には `Content-Length` が必須(なければ 411)。`Transfer-Encoding: chunked` は 400 にする(ブリッジも chunked を拒否する)。上限を超えたら、読む前に 413 にする。
- 本文はためずに流す。
  - `request.stream()` を async generator で httpx の `content=` に渡し、`Content-Length` ヘッダーを明示する(明示すれば httpx は `Transfer-Encoding: chunked` を付けない。jetracer env の httpx 0.28.1 で、async generator を content にした Request のヘッダーを見て確認済み。テストでも確かめる)。
  - 実際の長さを数え、宣言と違えば打ち切る。
  - ブリッジは本文をすべてメモリに読むが、JetFormer 側では持たない(8 GB・スワップなし)。
- `jetformer/check|upload` だけは本文(64 KiB まで)を読む。`url` を必ず `http://127.0.0.1:<config.WEB_PORT>` に置き換えてから送る(`Content-Length` も計算し直す)。
  - 送り先はこの Jetson の JetFormer だけなので、それ以外の LAN のアドレスにブリッジから PUT させない。
- ブリッジが動いていないとき: 503 `{"error":{"code":"bridge_not_running","message":"シミュレーターのブリッジが起動していません。JetFormer の「シミュレーター」タブで起動してください。"}}`。app.js はブリッジのエラーの形(`error.message`)を画面に出すので、同じ形にする。
- 時間切れは 504 `bridge_timeout`、接続できないときは 502 `bridge_unreachable` にする。

**応答の扱い**:
- `client.send(req, stream=True)` → `StreamingResponse(upstream.aiter_raw(), status_code=…, headers=…, background=BackgroundTask(upstream.aclose))` で返す。
- 引き継ぐヘッダー: `Content-Type`・`Content-Length`・`Content-Disposition`・`Cache-Control`・`X-Content-Type-Options`。
- 引き継がないヘッダー: `Access-Control-*`(同じオリジンなので不要)・`Connection`・`Keep-Alive`・`Transfer-Encoding`。

**クライアント**:
- モジュールで 1 つの `httpx.AsyncClient` を使う。`limits=Limits(max_connections=16, max_keepalive_connections=0)` とする(ブリッジは毎回 `Connection: close` を返すため)。
- タイムアウトは connect 3 秒・write 300 秒・pool 5 秒。read は上の表のとおり。
- 終了時に `aclose()` する。

**数える**:
- 中継中のリクエストを種類ごとに数える(`finish`・`collection`・`download` の件数)。データの削除(§5.2)が、作成中の ZIP を消さないようにするため。
- 最後に通信した時刻を `BridgeProcess.note_activity()` に知らせる。

### 3.5 JetFormer の管理 API `/api/sim/*`(JSON。エラーは既存と同じく `HTTPException(status, '日本語の文')`。画面の `api()` は `detail` を出す)
| メソッド | パス | 内容 |
|---|---|---|
| GET | `/api/sim/status` | `{release:{installed,id,created_at,installed_at,default_world_version,warnings} , inbox:[{file,bytes,mtime,kind:"simulator"|"other",release_id,installed}], releases:[{id,active,installed_at,bytes}], env:{ready,python_version,packages,problem}, bridge:{state,pid,port,started_at,restarts,restarted_at,last_error,rss_mb,policy_loaded,policy_name,recording}, busy:{training,driving,recording}, default_course}` |
| POST | `/api/sim/install` | `{file, activate}` → job |
| POST | `/api/sim/releases/{id}/activate` | 有効な版を切り替える(ブリッジ停止中のみ) |
| DELETE | `/api/sim/releases/{id}` | 有効でない版を消す |
| POST | `/api/sim/setup` | `{force}` → job |
| POST | `/api/sim/bridge/start` | → `{state:"starting"}`。前提が足りなければ 409 |
| POST | `/api/sim/bridge/stop` | `{force}`。ブリッジが収集中なら、`force` がない限り 409 |
| GET | `/api/sim/bridge/log?lines=200` | `{lines, suppressed}` |
| GET | `/api/sim/courses` | `{default, courses:[…]}`(§4.5) |
| PUT | `/api/sim/courses/import?filename=…[&replace=1]` | コースの ZIP を本文で受け取り、検証して登録する(同期。数秒以内) |
| GET | `/api/sim/courses/{cid}` | 1 件分の詳細(警告と統計を含む) |
| DELETE | `/api/sim/courses/{cid}` | アップロードしたコースだけ削除できる |
| POST | `/api/sim/courses/{cid}/default` | 既定のコースにする |
| GET | `/api/sim/courses/{cid}/package.zip` | JetFormer 形式で書き出す(雛形として使う。§4.3) |
| GET | `/api/sim/models` | JetFormer の学習済みモデルに `sim_compatible` と `reason` を付けたもの(§5.1) |
| GET | `/api/sim/datasets` | ブリッジが作ったデータセットの一覧(§5.2) |
| POST | `/api/sim/datasets/{sid}/import` | → job(`kind='import'`) |
| DELETE | `/api/sim/datasets/{sid}` | ブリッジの作業フォルダーを消す(§5.2) |
| (P2) POST | `/api/sim/bridge/load_model` | `{run, mode_id?}`。サーバー側で bundle と activate を行う(ブラウザなしの確認用) |
| (P2) GET | `/api/sim/courses/{cid}/preview.png` | 衝突形状の上面図(cv2) |

- `/api/sim/*` は JetFormer の既存の経路と重ならない(`/api/sim` という経路はない)。ブリッジの API は `/sim/bridge/api/*` だけで、JetFormer の `/api` 直下には何もつながない。

### 3.6 server.py への組み込み(変更は最小限)
```python
from ..sim import web as sim_web
...
sim_web.init(jobs=jobs, busy=lambda: {'training': trainer.running, 'driving': driver.running,
                                       'recording': bool(recorder.recording or collect.active)})
app.include_router(sim_web.router)
# lifespan の終了処理(driver.shutdown() の後)に:
sim_web.shutdown()          # ブリッジを止め(3 秒)、httpx クライアントを閉じる
```
- `jetformer/sim/web.py` は server.py を import しない(循環を避け、テストではサーバー全体を立てずに Router だけを試せるようにする)。
- `init()` では `bridge=`・`courses=` も差し替えられるようにする(テスト用)。

---

## 4. コースの登録簿(`jetformer/sim/courses.py`、`jetformer/sim/worldhash.py`)

### 4.1 コースの種類
| kind | 例 | 描画 | JS の出どころ | 削除 |
|---|---|---|---|---|
| `release` | `default`(今は v3 = `indoor-course-eval-3-turf-board-removed`、hash `0a464384`、線分 51・柱 94・路面 5・ゲート 33) | シミュレーターの scene.js | 有効なリリースの `dist/` | 不可 |
| `snapshot` | `indoor-course-eval-2-shortcut`(旧 v2、hash `547d7e80`、52・98・5・33) | 同上(スナップショットの scene.js) | 有効なリリースの `qa/course-snapshots/<id>/`。スナップショットの manifest と COMPLETE-MANIFEST の両方と SHA256 が一致するものだけ | 不可 |
| `trusted-js` | (アップロードした)JS が既知のリリースのファイルと一致するもの / (P2)版を切り替えたときの「前の版の Default」 | アップロードの scene.js | `data/sim_courses/<cid>/`。SHA256 が導入済みリリースのどれかの COMPLETE-MANIFEST に載っているものだけ | 可 |
| `data` | アップロードした普通のコース | JetFormer の汎用描画(`course-scene.js`) | なし(アップロードの JS は捨てる) | 可 |

- `release` と `snapshot` のコースは保存しない。有効なリリースを見て毎回作る(mtime でキャッシュする)。
- 表示名と説明は JetFormer が持つ `jetformer/sim/data/known_courses.json` から取る。scene.js・world.js の sha256 から `{label, world_version, world_hash, stats, size_m}` を引く。
  - Default(v3)は「Default(標準コース)」。
  - v2 は「旧コース v2(芝生の白い板あり)」。
- 知らないリリースの Default(将来の版)も扱えるようにする。
  - `WORLD_VERSION` は world.js の本文から正規表現 `export const WORLD_VERSION\s*=\s*'([^']{1,80})'` で読む(実行しない)。
  - hash は「未計算」とする。(P2)node が使えるなら、リリースの `qa/load-course-world.mjs` を使って計算する。これは導入済みリリースのコードなので、本体と同じく信頼してよい。

### 4.2 コースの JS をどう扱うか(決定)
- **アップロードされたコースの JS は実行しない。**
  - シミュレーターのページは JetFormer と同じオリジン(:8080)で動く。そのため、コースの JS は認証なしで JetFormer の全 API を呼べる。`/api/drive/start` と token による生存確認で、**実車を動かせてしまう**。
  - JS の中身を検査しても安全は保証できない。
- 衝突形状・路面・範囲・ゲート列・必須横断・開始位置は、`physics-world.json`(makeWorld の出力。シミュレーター自身のスナップショット形式)でデータとして完全に表せる。そのため、新しいコースはデータだけで受け付けられる。
- JS で描くのは、`scene.js` と `world.js` の **両方の** sha256 が、導入済みリリースの COMPLETE-MANIFEST.json(+ `known_courses.json`)に載っている場合だけ(`trusted-js`)。
  - 本体(Python コード)と同じく、リリースは Jetson 上のファイルからしか入らない(§1.2)。そのため、JS の信頼の根拠は本体と同じになる。
- アップロードに JS が入っていて一致しなければ、JS は捨てる。そのうえで `data` として登録し、警告「JS は実行しません。簡易描画で表示します」を付ける。
- 将来の選択肢(今回は入れない):
  - Jetson のフォルダーから JS 付きで登録する方法(シェルで置く = コードの導入と同じ扱い)
  - CSP や別オリジンで隔離して実行する方法

### 4.3 アップロードの形式(ZIP。64 MiB まで)
どちらの形式も、中身が 1 つの最上位フォルダーの下にあってよい。

**(a) シミュレーターのスナップショット形式**(`qa/course-snapshots/<id>/` と同じ形)
```
manifest.json        {world_version, files:{"physics-world.json": sha256, ["scene.js": sha256, "world.js": sha256]}, world_hash, [segments], [disks], [reason]}
physics-world.json   makeWorld(buildCourse().physics) の JSON {segments, disks, surfaces, bounds, route, requiredCrossings, spawn}
[scene.js, world.js] §4.2 のとおり(既知の SHA256 のときだけ使う)
[assets/*.jpg|png]   テクスチャ(任意)
```
**(b) JetFormer 形式**(`format: "jetformer-sim-course/v1"`)
```
course.json          {"format":"jetformer-sim-course/v1", "name":"自宅コース 1", "world_version":"home-course-1-v1",
                      ["id":"home-course-1"], ["description":"..."], ["world_hash":"xxxxxxxx"],
                      ["visual": {"floor_texture":"carpet.jpg", "surface_textures":{"turf":"turf.jpg", ...},
                                  "colors": {"rail":"#f2f2ee", "curtain":"#1b1b1d"}}]}
physics-world.json   (a) と同じ
[assets/*.jpg|png]
```
- `course.json` と `manifest.json` が両方あるときは、名前などは `course.json` から取る。`manifest.json` の sha256 と world_hash の確認も行う。
- シミュレーター本体の ZIP(`COMPLETE-MANIFEST.json` や `bridge.py` がある)が送られたときは、「シミュレーター本体の ZIP です。~/sim に置いて『導入』してください」で断る。
- **雛形**: `GET /api/sim/courses/{cid}/package.zip` は (b) の形で書き出す。
  - Default は、JetFormer が持つ `jetformer/sim/data/default-v3-physics-world.json` を使う。understand_course で `optimized-reference.js` の EXPECTED_WORLD から取り出し、spawn を足したもの。hash `0a464384` を確認済みで、今は `/tmp/sim_integration_20261013_course/v3-physics-world.json` にある。dist の sha256 が既知の v3 と一致するときだけ出す。
  - v2 はスナップショットの physics-world.json を、data コースは保存してあるものを使う。
  - これを編集して(板を足す・動かす、μ を変える、ゲート列を変えるなど)、world_version を変えてアップロードすれば、新しいコースになる。
  - 形式の説明は README の「シミュレーター」の節に書く。

### 4.4 検証(`courses.validate_package(path) -> (meta, world, assets, warnings)`。エラーは `CourseError(errors: list[str])`)
- **ZIP**:
  - 64 MiB 以下、展開後 128 MiB 以下、256 エントリー以下。
  - 絶対パス・`..`・`\`・`:`・NUL・シンボリックリンク・特殊ファイル・暗号化・大文字小文字を無視した重複・stored/deflate 以外の圧縮は拒否する。
  - 受け付ける名前は `manifest.json`・`course.json`・`physics-world.json`・`scene.js`・`world.js`・`assets/<名前>.(jpg|jpeg|png)`・`README*.txt|md` だけ。それ以外は無視して警告を出す。
  - `__MACOSX/` と `.DS_Store` は黙って無視する。
- **JSON**:
  - `physics-world.json` は 4 MiB 以下、ほかは 64 KiB 以下。
  - `parse_constant` で NaN/Infinity を拒否する。入れ子は 8 段まで。
  - 型と範囲は**黙って丸めずにエラーにする**(sim-core の normalizedWorld は黙って既定値にするため)。
- **値**(sim-core の normalizedWorld の範囲に合わせ、それより狭くする):
  | 項目 | 条件 |
  |---|---|
  | segments | 0〜2000 件。`a`,`b` = [x,z]、`radius` 0<r≤0.5、`height` 0<h≤5、`kind` ∈ {rail, curtain, gate-foot} |
  | disks | 0〜4000 件。`x`,`z`、`radius` 0<r≤0.5、`height` 0<h≤5、`kind` ∈ {post-foot, post, gate-post} |
  | surfaces | 1〜64 件。`id` は `^[a-z0-9][a-z0-9-]{0,31}$` で重複なし、`w`,`d` 0<…≤50、`mu` 0.01〜3 |
  | 座標 | すべて \|x\|,\|z\| ≤ 50 m |
  | bounds | `minX<maxX`、`minZ<maxZ`、幅と奥行きは 60 m 以下 |
  | route | 3〜400 点。隣り合う点は 0.05 m より離れていること。すべて bounds の内側(0.2 m の余裕) |
  | requiredCrossings | 0〜8 件。`surface_id` が surfaces にあること。`minX<maxX` で、開口がその路面の中にあること。`direction` は ±1 |
  | spawn | 必須(`{x,z,yaw}`)。bounds の内側。線分と柱の外にあること: 中心からの距離が「半径 + 0.10 m」未満ならエラー、「半径 + 0.23 m(車体 0.42×0.19 の対角の半分)」未満なら警告 |
- **名前など**:
  - `world_version` は `^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$`。`name` は 1〜40 文字で、制御文字を含まないこと。
  - `id` は `course.json.id`、なければ world_version を小文字にして `[a-z0-9-]` に寄せたもの。`^[a-z0-9][a-z0-9-]{1,47}$` を満たすこと。
  - 予約語(`default`・`static`・`bridge`・`api`・`c`・`_jf`)は使えない。
- **一意であること**:
  - world_version はコースごとに一意でなければならない。データセットの contract、speed-manifest、model_config の `simulator_world_version` が、world_version でコースを見分けているため。
  - 同じ world_version で正準 hash も同じなら「登録済み」として成功にする。hash が違えば 409「同じ world_version で形が違うコースがあります。world_version を変えてください」にする。
  - 同じ id があれば、`replace=1` のときだけ置き換える(release と snapshot は置き換えられない)。
- **world hash**(`worldhash.py`。understand_course の試作 `/tmp/sim_integration_20261013_course/worldhash.py` を移す):
  - sim-core の normalizedWorld を Python に移し、JS と同じ数値の文字列化をしたうえで、`JSON.stringify` 相当の文字列に FNV-1a 32bit をかける。
  - 開始ゲート 0・ずれ 0 の正準 hash を保存する。`manifest.world_hash`・`course.json.world_hash` があれば一致を確かめ、違えば拒否する(シミュレーター本体の版の違いが疑われる)。
  - FNV 32bit は衝突し得るので、正規化した JSON の sha256(`canonical_sha256`)も保存し、同じかどうかの判定にはこちらを使う。
  - 計算に使ったリリースの `dist/sim-core.js` の sha256 も記録する(sim-core の版で hash の定義が変わった前例がある: v1 の 326800ab)。
- **テクスチャ**:
  - jpg/png のマジックナンバーを見る。1 枚 8 MiB 以下、4096 px 以下、16 枚以下。
  - `cv2.imdecode` で読めることを確かめ、**再エンコードして保存する**(jpg は q90、png はそのまま PNG。メタデータを落とすため)。SVG・HTML などは受け付けない。
  - 名前は `^[a-z0-9][a-z0-9_-]{0,47}\.(jpg|png)$` に限る。

### 4.5 保存・一覧・選択・削除・既定
- 保存: `data/sim_courses/.tmp-<uuid>/` に書いてから `os.replace` で `data/sim_courses/<cid>/` にする。
  ```
  course.json          JetFormer の登録情報 {format:"jetformer-sim-course-meta/v1", id, name, kind, world_version, world_hash,
                        canonical_sha256, sim_core_sha256, stats{segments,disks,surfaces,gates,crossings}, size_m, has_shortcut,
                        source{filename, bytes, sha256, format:"snapshot"|"jetformer"}, visual, warnings, created_at}
  physics-world.json   検証済みの world(元の値のまま。正規化は sim-core に任せる)
  assets/              再エンコードしたテクスチャ
  [scene.js, world.js] trusted-js のときだけ
  ```
- 一覧 `GET /api/sim/courses` の各要素: `{id, name, kind, renderer:"simulator"|"generic", world_version, world_hash, stats, size_m, features, deletable, is_default, warnings, created_at, open_url:"/sim/c/<id>/"}`。並び順は default、snapshot、アップロード(新しい順)。
- 既定: `data/sim_courses/settings.json` の `default` に入れる。最初は `default`。`/sim/` はこのコースに転送し、タブでは印を付ける。
  - 既定のコースを削除したら `default` に戻す。
  - settings.json が壊れていたら `default` として扱う。
- 削除: data と trusted-js だけ。`shutil.rmtree` の前に、`resolve()` が `courses_dir()` の中にあることを確かめる。
  - 開いているページは読み込み済みのモジュールで動き続け、次に読み込むときに 404 になる。

### 4.6 選んだコースをどう反映するか
- **URL で選ぶ**: `/sim/c/<cid>/`。ページを読み込むたびにコースが決まる。URL がコースごとに違うので、モジュールのキャッシュも取り違えない。別々のタブで違うコースを同時に開くこともできる。ただしブリッジとモデルは 1 つを共有する(§10)。
- **release / snapshot / trusted-js**: `scene.js`・`world.js` をそのコースのファイルにする(§3.2)。テクスチャは `./assets/`(ページの URL が基準)で読まれるので、コースの assets があればそれを、なければ dist のものを返す。
- **data(汎用描画)**: JetFormer のファイルを返す。
  - `/sim/c/<cid>/scene.js` → `jetformer/sim/static/course-scene.js`
    - `import * as THREE from 'three'` と `import DATA from './_jf/course-data.js'` を使う。importmap が同じなので、THREE は 1 つに保たれる。
    - `export function buildCourse(scene)` は次を作る。
      - 床: bounds + 0.5 m、カーペットのテクスチャ
      - 路面: id ごとのテクスチャ(carpet→`carpet.jpg`+normal、turf→`turf.jpg`+normal、blue-mat→`blue-pad.jpg`、shortcut→`black-mat.jpg`、ptfe→白、ほか→灰色)。路面ごとに y を 0.5 mm ずつずらす
      - rail: 幅 2r・高さ h の白い板(Box)
      - curtain: 暗い色の布(Box)
      - gate-foot: 小さな Box
      - post: 円柱
      - post-foot: 0.16×0.15×h の Box
      - gate-post: 円柱
      - 照明: Hemisphere と Directional
    - 戻り値は `{physics: structuredClone(DATA.physics), dimensions, landmarks:[], animated:[], course, room:null, curtains:[]}`。
    - **TextureLoader で 1 枚以上(床の carpet.jpg)を必ず読む。** app.js の `assetsReady` は `THREE.DefaultLoadingManager.onLoad` でしか true にならず、1 枚も読まないと「評価を開始」が押せないままになるため。
    - ToF は描画したメッシュから計算する(`scene.overrideMaterial = depthMaterial`)ので、衝突物はすべて、その高さのメッシュとして描く。
  - `/sim/c/<cid>/world.js` → `jetformer/sim/static/course-world.js`
    - `export const WORLD_VERSION = DATA.world_version;`
    - `export function makeWorld(physics) { return {...physics, route: DATA.world.route.map(p => [...p]), requiredCrossings: DATA.world.requiredCrossings.map(c => ({...c})), spawn: {...DATA.world.spawn}}; }`
  - `/sim/c/<cid>/_jf/course-data.js` → `export default <json>;`(サーバーが生成する)
    - 中身は `{format:"jetformer-sim-course-data/v1", world_version, name, physics:{segments,disks,surfaces,bounds}, world:{route,requiredCrossings,spawn}, visual:{…}}`。
    - `json.dumps(ensure_ascii=True)` の後で `<`・`>`・`&` もエスケープする。文字列を JS のコードとして組み立てることはしない。
  - ブラウザで計算した world hash(レポートの `world_hash`。開始ゲート 0・ずれ 0 のとき)は、登録時の正準 hash と一致するはず。これは E2E で確かめる(§8.3)。
  - 見た目は scene.js と同じではない(部屋、LED 矢印、テープ、ラベルなどがない)。そのため、カメラ画像を使う AI の評価は Default と直接比べられない。タブと帯にそう書く。

### 4.7 コースによって使えない機能
- 教師 `speed-fast`(v3 専用の誘導経路を、確認なしで追う)、`speed-rc`・`speed-optimized`(v3 以外では日本語のエラーで止まる)、「20分・60シーンの収集条件」(RC 教師を使う)は、**Default でだけ使える**。
- `features` の決め方: `cid == 'default'` なら教師 5 つすべてと `collection_plan: true`。それ以外は `teachers: ["slow","speed"]`、`collection_plan: false`。
- shim は使えない option とボタンを隠す(§3.3-6)。手動、低速デモ、speed 教師、学習済み AI の評価はどのコースでも使える。
- 開始ゲートの上限は app.js が route の長さで直すので、対応はいらない。
- 俯瞰カメラの位置は約 11×7 m のコース向けの固定値。大きなコースではドラッグで直してもらう(対象外)。

---

## 5. JetFormer との連携

### 5.1 学習済みモデルをシミュレーターに読み込む
- **一覧** `GET /api/sim/models`: `model_export.list_models()` に次の 2 つを足す。
  - `sim_compatible`: `policy == 'jetformer_v0'` のとき true。ブリッジの jetformer_policy は io_spec `jetformer_v0` だけに対応する。
  - `reason`: true でないときの理由(「シミュレーターは TransFuser軽量型(jetformer_v0)だけに対応しています」)。
  - 並び順は既存のまま(新しい順)。
- **受け渡し(P1、ブラウザ側)**: タブで「モデル」を選んで開くと、URL は `/sim/c/<cid>/?model=<run>` になる。app.js はクエリを見ないので、そのまま動く。shim は次の順に処理する。
  1. app の準備(`window.courseSimulator` が現れること)と、ブリッジの running を待つ。
  2. 同じオリジンの `GET /api/models/<run>/bundle.zip`(既存の経路。`model_export.iter_bundle_zip`)を fetch して、帯に進み具合を出す。
  3. `new File([blob], '<run>.zip', {type:'application/zip'})` を作り、`DataTransfer` で `#model-bundle-file.files` に入れて `change` を送る。
  4. `#load-bundle.click()` を押す。この先はアプリ本来の流れになる。すなわち `POST /api/models/bundle`(→ 中継)→ モード選択 → `POST /api/models/activate`。
  - ページがトークンを持つので、「設定を反映してリセット」も使える。
  - 56 MB の ZIP が Mac と Jetson の間を 2 回通る(LAN なので数秒)。
  - DataTransfer のコンストラクターは Chrome・Firefox・Safari 14.1 以降で使える。使えなければ、帯に「ZIP をダウンロードして手で選んでください」とリンクを出す。
  - 読み込みが終わったら、`history.replaceState` で `?model` を消す。再読み込みしても、もう一度は読み込まない。
- **(P2)サーバー側で有効にする**: `POST /api/sim/bridge/load_model {run, mode_id?}`。
  - JetFormer が `iter_bundle_zip(run)` を `Content-Length` 付きで `/api/models/bundle` に送り、続けて `/api/models/activate` を呼ぶ。実測では 0.29 秒 + 1.02 秒だった。
  - 送る前に `bundle_bytes_estimate` を見る。正確な長さが要るので、一時ファイルに書いてから送ってよい(56 MB)。
  - この方法では、ページはトークンを持たないので「設定を反映」は使えない。主にブラウザなしの E2E 確認に使う。
- **(P2)同梱モデル**: リリースの `models/shortcut-rc-finetuned` などは、io_spec・model_config・ONNX がそろっている。これらを `/api/sim/release-models/<name>/bundle.zip` で ZIP にして(ONNX は ZIP_STORED)、選択肢に「同梱: …」として出す。`modelB` は io_spec がないので出さない。

### 5.2 シミュレーターで集めたデータセットを JetFormer に取り込む
- **一覧** `GET /api/sim/datasets`: `data/sim/runtime/datasets/<sid32>/archive-info.json` を読む。このファイルは、ブリッジが ZIP を `.part` から置き換えた後に書くので、あれば完成している。
  - 確認すること: `sid` が `^[0-9a-f]{32}$` であること。シンボリックリンクでないこと。`schema` と `session_id` が一致すること。`filename` が basename の `.zip` であること。ファイルの大きさが `bytes` と同じであること。ブリッジの `archive_path()` と同じ確認をする。
  - 返すもの: `{session_id, filename, bytes, sha256, kind: collection-summary.json があれば "collection" なければ "scene", dataset_name, created_at(mtime), scenes, frames, duration_s(collection-summary から), imported: bool, import_name}`。
  - `dataset_name` は ZIP の中で `dataset_card.json` を持つ最上位フォルダーの名前(中央ディレクトリだけを見る)。
  - `imported` は、`DATASET_DIR/<dataset_name>/import_source.json` の `file` と `bytes` が一致するかで決める。
- **取り込み(P1)** `POST /api/sim/datasets/{sid}/import`:
  1. 上の確認をもう一度行う。
  2. 空き容量を確かめる。ZIP の 2.2 倍以上がなければ 507 にする(展開とシーンへの展開の分)。
  3. 既存の `jobs.submit('import', 'シミュレーターから取り込み: <filename>(<MB>)', lambda progress: dataset_import.import_archive(zip_path, simulator=True, source_name=filename, progress=progress))` を使う。
  - ファイルの写しは作らない。`dataset_import` がそのまま、名前の確認、学習に使える形かの確認、simulator タグ、シーンタブへの展開を行う。同じファイルなら「取り込み済み」として成功する。
  - **どちらのジョブも「処理状況」に出る**(データセットタブとシミュレータータブの両方)。
- **アプリ内の「JetFormerへ送る」(従来の流れ)**: そのまま使える。
  - shim が送り先を `http://127.0.0.1:8080` にし、中継も本文の `url` をそこに固定する。
  - ブリッジは `PUT /api/datasets/import?simulator=1` に 1 MiB ずつ送る。
  - JetFormer の受け口は async で、中継も async なので詰まらない。
  - ただし `data/uploads` に写しを作るので、大きなデータではタブからの取り込みの方が速く、容量も少なくて済む。タブにそう書く。
- **削除** `DELETE /api/sim/datasets/{sid}`: `runtime/datasets/<sid>/` ごと消す(ZIP、展開形、`_frames/*.npz`)。ブリッジには削除の API がないので、JetFormer が持つ runtime を直接消す。
  - 次のときは 409 にする。
    - ブリッジが収集中で、その `session_id` が使われている(`dataset_collection.session_id`)
    - 中継で `datasets/finish`・`datasets/collection`・`download` が動いている(§3.4 の数)
  - 消したセッションを、あとで「まとめ ZIP」に入れようとするとブリッジがエラーにする。画面に「まとめ ZIP を作り終えてから消してください」と書く。
  - 空き容量の目安を画面に出す。ブリッジの「まとめ ZIP」は「20 GiB + 元データの 2 倍」の空きを求める。今の空きは約 25 GB なので余裕は少ない。

---

## 6. 画面(「シミュレーター」タブ)
- `index.html` の nav の最後に `<button class="tab" data-tab="sim">シミュレーター</button>` を、本文に `<section id="tab-sim" class="tab-page">` を足す。
- `app.js` の `showTab` に `if (name === 'sim') loadSim();` を足す。タブを開いている間だけ、2 秒ごとに `/api/sim/status` を呼ぶ。コースとデータセットの一覧は、タブを開いたときと操作の後に読む。
- 既存の書き方(`api()`、`toast()`、`card`・`card-title`・`pill`・`muted small`、XHR の進み具合)に合わせる。文字は日本語にし、利用者が入れた文字列(コース名など)は必ず `textContent` かエスケープで出す。

**カード 1「シミュレーター本体」**
- 状態: 「導入済み: 20261013-144013-6f610dd5(2026-10-13 14:40 作成)/ Default コース indoor-course-eval-3-turf-board-removed」、または「未導入」。
- 「~/sim にある ZIP」の一覧: 名前、大きさ、種類(シミュレーター本体 / 対象外)、状態(導入済み / 未導入)と「導入」ボタン(ジョブ)。
- 導入済みの版の一覧(2 つ以上あるとき): 「この版を使う」「削除」。

**カード 2「Python 環境(ブリッジ用)」**
- 状態: 「準備済み(Python 3.13.14 / onnxruntime 1.31.0 / numpy 2.2.6 / opencv 4.13.0.92)」、「未準備」、または問題の内容。
- 「準備する」・「準備し直す」(ジョブ)。所要時間の目安(数分、約 290 MB)。

**カード 3「ブリッジ」**
- 状態の pill: 停止中 / 起動中 / 動作中 / 停止中… / 失敗。動作中なら、ポート、PID、動作時間、メモリ(MB)、読み込み中のモデル、収集中かどうかを出す。
- 「起動」「停止」(収集中なら確認の後に `force`)。「ログ」(details の中に pre を置き、開いたときだけ読む)。
- 注意書き: 「推論は Jetson の CPU で動きます(GPU は使いません)。学習中は遅くなります。」「管理画面を再起動すると、ブリッジも止まり、収集中のデータは失われます。」

**カード 4「コース」**
- 一覧(カードまたは表): 既定の印、名前、world_version、hash、種類のバッジ(「標準」「同梱」「アップロード」「簡易描画」)、線分・柱・ゲートの数、大きさ、使える教師、警告。操作は「開く」「既定にする」「書き出し」「削除」(アップロードしたものだけ)。
- アップロード: ファイルを選ぶ(.zip)→ `PUT /api/sim/courses/import?filename=`(XHR で進み具合を出す)。エラーは一覧の形で出す。
- 説明(details): 受け付ける 2 つの形式と、「Default を書き出して編集し、world_version を変えてアップロード」の手順。「アップロードした JS は実行しません」。

**カード 5「シミュレーターを開く」**
- 選択: コース(既定が選ばれた状態)、モデル(「なし」+ JetFormer の学習済みモデル。対応しないものは選べず理由を出す。新しい順)。
- 「新しいタブで開く」: ブリッジが止まっていれば、先に `POST /api/sim/bridge/start` を呼び、`window.open('/sim/c/<cid>/' + (run ? '?model=' + encodeURIComponent(run) : ''))` で開く。
  - iframe ではなく新しいタブにする。app.js はフォーカスが外れると走行を止め、キー操作もフォーカスに依存するため。
  - (P2)「この画面の中で開く」(iframe)。
- 注意: 「同時に AI で走らせられるのは 1 つのタブだけです(ブリッジのモデルは 1 つ)。」

**カード 6「シミュレーターで集めたデータセット」**
- 一覧: ファイル名、種類(シーン / まとめ)、大きさ、作成日時、シーン数・フレーム数、取り込みの状態。操作は「JetFormer に取り込む」(ジョブ)と「削除」。
- 処理状況: `/api/jobs` の `kind in ('sim_install','sim_setup','import')` を、既存の `loadJobs` と同じ書き方で `#sim-jobs` に出す。

---

## 7. セキュリティ
1. **アップロードからコードを実行しない**
   - コースはデータだけを受け付ける。JS は、導入済みリリースと SHA256 が一致するときだけ使う(§4.2)。
   - シミュレーター本体(Python と JS)は、Jetson 上の `~/sim/` からしか導入できない。ブラウザからの本体アップロードは作らない。
   - ブリッジは `--policy` なしで起動し、任意の `.py` を import しない。
   - モデル ZIP の中のコードは、ブリッジが実行しない(`executed_bundle_code:false`)。
   - 本体 ZIP は、COMPLETE-MANIFEST.json の sha256 と全ファイルを照合する。
2. **パスの越境**
   - `cid` とファイル名は正規表現で制限する。パスは要素ごとに検査し、`resolve()` 後にベースのフォルダーの中にあることを確かめ、シンボリックリンクを拒否する。`_jf/` は予約にする。
   - ZIP を展開するときは、§1.2・§4.4 の検査を行い、作業フォルダーで作ってから `os.replace` する。
   - inbox の ZIP は名前で選ばせ、パスは受け付けない。
   - データの取り込みと削除は `sid` の正規表現と `archive-info.json` で確かめる。
3. **大きさ**
   - コースの ZIP: 64 MiB、展開後 128 MiB、256 件。physics-world.json 4 MiB。テクスチャは 1 枚 8 MiB・4096 px・16 枚まで。
   - 本体 ZIP: 4 GiB、展開後 2 GiB、5000 件。
   - 中継: §3.4 の上限。ブリッジ自身の上限と同じか、それより小さくする。
   - 受け取った本文はストリームで扱い、上限を超えた時点で切る。
   - JSON は NaN と Infinity を拒否し、入れ子の深さを制限する。
4. **ブリッジ**
   - 127.0.0.1 にだけ bind し(ブリッジのコードで固定)、ポートを外に出さない。
   - 中継は `/sim/bridge/api/*` の決まった組み合わせだけを通す。ヘッダーは許可したものだけを渡し、Host を書き換え、Origin・Referer・Cookie を捨てる。
   - `/api/jetformer/*` の `url` は、この Jetson の JetFormer に固定する。これにより、ブリッジを使って LAN の別のアドレスへ GET や PUT をさせない。
5. **出力**
   - 利用者の文字列は HTML にも JS にも埋め込まない。設定とコースのデータは `json.dumps(ensure_ascii=True)` + `<`・`>`・`&` のエスケープで出す。
   - 画面では textContent で出す。
   - Content-Type は拡張子で固定し、nosniff を付ける。テクスチャは再エンコードして出す。
6. **同じオリジンで動くもの**
   - シミュレーターのページ(リリースのコードと JetFormer の shim)は、JetFormer の API を呼べる。どちらも信頼できるコードとして扱う。
   - JetFormer には今も認証がない(LAN の誰でも操作できる)。この統合でその範囲は変わらない。
   - (P2)多層防御として、`/sim/c/*` に CSP を付ける。`script-src 'self'`+importmap の hash、`connect-src 'self'`、`object-src 'none'`、`base-uri 'self'`、`form-action 'none'`、`frame-ancestors 'self'`。入れる前に Mac の Chrome と Safari で確かめる。
7. **資源**
   - ブリッジは 1 つだけ。起動前にメモリを確かめ(MemAvailable ≥ 1 GiB)、nice 10・ORT 4 スレッドで動かす。GPU は使わない。
   - 導入・取り込みの前に空き容量を確かめる。
   - ログは大きさで回し、本文やクエリは書かない(ブリッジ自身も書かない)。

---

## 8. テストと E2E の確認計画

### 8.1 CPU だけで動く pytest(`tests/test_sim_*.py`。GPU・実車・ネットワークは使わない)
材料は `tests/fixtures/sim/` に写す(小さく、ユーザーのシミュレーターから取ったもの)。
- `v2-physics-world.json`(25 KB。スナップショットのもの)
- `v3-physics-world.json`(20 KB。抽出したもの)
- `v2-manifest.json`
- 小さな jpg と png
- 偽のリリースを作るヘルパー(conftest か各テスト内): `course-simulator/` に COMPLETE-MANIFEST・dist/index.html(本物と同じ head の形)・app.js・scene.js・world.js・sim-core.js と、スタブの bridge.py などを置いた ZIP を tmp に作る。

| ファイル | 主な確認 |
|---|---|
| `test_sim_install.py` | 正しい ZIP が導入でき、`release_id` が `<YYYYMMDD-HHMMSS>-<sha8>` になる。`dataset-downloads/`・`runtime/` が除外される。もう一度導入すると already_installed になる。sha256 の不一致・manifest にないファイル(.py はスキップ、最上位 .txt は入れる)・manifest にあるのに ZIP にないファイル・`..`・絶対パス・`\`・シンボリックリンク・重複・暗号化・大きすぎ・schema 違いを扱える。active.json は、ブリッジが動いているときは切り替わらない。inbox の一覧と判定、名前以外(パス)を拒否すること |
| `test_sim_env.py` | requirements-bridge.txt の書式とハッシュ。リリースの requirements-model.txt との突き合わせ(違えば止まる)。pip のコマンドの組み立て(wheels があれば `--no-index`)。subprocess は差し替えて、実際には入れない |
| `test_sim_bridge_proc.py` | 偽の runner(`sys.executable -c` で listening 行を出し、小さな HTTP サーバーを立てる)を使う。start → running、port の取得、stop(SIGINT)、異常終了 → backoff で再起動 → 上限で failed、自分で止めたときは再起動しない、ログのリングと predict/frame 行の数え上げ、メモリ不足で 409、時間切れ |
| `test_sim_runner.py` | `bridge_runner.py` を、偽のシミュレーターフォルダー(BridgeServer と PolicyRunner の最小スタブ)で動かす。`-I -B` で sys.path に sim_dir が入る。uploads が消える。listening 行が出る。ORT の差し替え(onnxruntime がなければ、その部分だけ skip) |
| `test_sim_worldhash.py` | v2 → `547d7e80`、v3 → `0a464384`。JS の数値の文字列化(整数、負の 0、1e-7、1e21)。post-foot の矩形化。route の重複と閉じる点の除去 |
| `test_sim_courses.py` | 2 つの形式の登録。JS の扱い(既知の sha なら trusted-js、そうでなければ JS を捨てて data と警告)。§4.4 の各エラー(NaN、範囲外、未知の kind、route が短い、crossing の路面がない、spawn が衝突、id の予約語、world_version の重複と hash 違い = 409、同じものの再登録 = 成功)。テクスチャの再エンコードと偽の拡張子。本体 ZIP の拒否。削除は release と snapshot ではできない。既定の設定と、既定を消したときに戻ること。書き出し → 再アップロードで同じ hash になること |
| `test_sim_web.py` | `FastAPI()` に Router だけを付けて TestClient で試す。`/sim` → `/sim/` → `/sim/c/<既定>/`、`/sim/c/x` → `/sim/c/x/` の転送。index に 3 行が入り、形が違えば 500。config.js のエスケープ(`</script>` を含む名前)。ファイルの解決(release / snapshot / data、assets の差し替え、dist への戻り)。`..`・`%2e%2e`・`\`・NUL・`_jf/` の拒否。MIME と nosniff。ETag と 304。未知のコースは 404 |
| `test_sim_proxy.py` | 偽ブリッジ(127.0.0.1:0 の ThreadingHTTPServer。本物と同じく Host・Origin・Content-Length・chunked を確かめる)に対して、TestClient から中継する。Host の書き換え、Origin を捨てること、Content-Length がそのまま届き chunked にならないこと、クエリの引き継ぎ、50 MB の本文を流せること、上限超えで 413、Content-Length なしで 411、決まっていない組み合わせで 404、`jetformer/*` の url の書き換え、ダウンロードのストリームとヘッダー、止まっているとき 503、時間切れで 504、エラー本文の形 |
| `test_sim_datasets.py` | runtime の一覧(正しいもの、sid の形が違う、シンボリックリンク、大きさの不一致、`.part` だけ)。collection と scene の判定。取り込みのジョブ(`dataset_import.import_archive` を差し替えて引数を確認)。容量不足で 507。削除の 409(収集中、中継中) |
| `test_sim_models.py` | `/api/sim/models` が jetformer_v0 だけを使えるようにすること(`model_export.list_models` を差し替え) |

- 実物を使うテスト(`@pytest.mark.skipif`: `data/sim/venv` とリリースがないときは skip): runner で本物のブリッジを起動し、中継経由で `status` が 200 になること、Host の検査、`--self-test` の 13 件。
- node を使うテスト(`/usr/bin/node` とリリースがないときは skip): リリースの `sim-core.js` を node で読み、data コースの `makeWorld(buildCourse().physics)` 相当の hash と Python の hash が一致すること。scene.js は実行せず、physics-world.json だけを渡す。
- JS の単体テスト(P2): `sim-shim.js` の書き換え関数を node で試す(DOM は使わない部分だけ)。

### 8.2 立ち上げ(実装の後。jetformer-web の再起動が必要)
- 再起動してよいのは次のすべてを満たすときだけ。
  - `GET /api/train/status` が running でない
  - `GET /api/status` の recorder が収集中でない
  - `GET /api/drive/status` が走行中でない
  - `GET /api/jobs` に running のジョブがない
- 再起動はメインセッションが行う。
- 再起動の前に、wheel を scratchpad から `data/sim/wheels/` に写しておく。

### 8.3 E2E の確認(Jetson の上で curl とヘッドレス Firefox を使う。実車と GPU は使わない)
1. `GET /api/sim/status` → 未導入で、inbox に `course-simulator-complete.zip`(kind simulator)がある。
2. `POST /api/sim/install {"file":"course-simulator-complete.zip"}` → ジョブが done になる。`data/sim/releases/20261013-144013-6f610dd5/` ができ、`dataset-downloads/` はない。`/home/jetson/sim` は変わっていない(sha256 と mtime で確認)。
3. `POST /api/sim/setup` → done。版が一致し、self-test 13/13 が通る。
4. `POST /api/sim/bridge/start` → 数秒で running。`ss -ltnp` で 127.0.0.1 にだけ bind している。
5. `GET /sim/bridge/api/status` → 200(policy_loaded:false)。LAN の Host を付けても、中継が 127.0.0.1 に書き換えるので 200 になる。ブリッジのポートに LAN から直接はつながらない。
6. `GET /sim/c/default/`、`/sim/c/indoor-course-eval-2-shortcut/`、`/sim/c/<アップロード>/`: shim が入っている。scene.js と world.js の sha256 が期待どおり。`assets/carpet.jpg` が 200。
7. コースのアップロード: Default の `package.zip` を書き出し、world_version と板 1 枚を変えて `PUT /api/sim/courses/import` → 200、data。もう一度送ると「登録済み」。world_version を変えずに形だけ変えると 409。`../` を含む ZIP は 400。
8. モデル: (P2 の API か、同じ手順をスクリプトで)JetFormer の jetformer_v0 のモデル 1 つを `/sim/bridge/api/models/bundle` → `activate` → status の policy_loaded:true。テスト用の観測(ブリッジの self-test と同じ形)で predict が 200 になる。
9. データ: 中継経由で `datasets/start` → `frame` を 15 回(224 PNG・ToF・IMU の合成)→ `finish`。`/api/sim/datasets` に出る。「取り込む」で、名前 `e2e_sim_*` のデータセットが JetFormer にできる。確認したら、そのデータセットとシーン(`ds-e2e_sim_*`)と runtime のセッションを消す。
10. ヘッドレス Firefox(WebGL なし): `/sim/c/<data コース>/` で、全モジュールが 200 で読まれること、`fetch('http://127.0.0.1:8766/api/status')` が `/sim/bridge/api/status` に届くことを確かめる。
11. `POST /api/sim/bridge/stop` → stopped。子プロセスが残っていない。
12. **ユーザーの Mac での確認**(WebGL が要るので、最後に依頼する)。
    - Default で手動走行と低速デモ。
    - JetFormer のモデルを `?model=` で読み込んで AI 評価。
    - 1 シーン収集 → 「JetFormer に取り込む」。
    - v2 とアップロードしたコースへの切り替え。data コースで「評価を開始」が押せること(assetsReady)。
    - レポートの world_hash が登録の hash と一致すること。
    - ダウンロードのリンク。
    - シミュレーター内の「JetFormerへ送る」。

---

## 9. 実装の分担とファイル
すべて `/home/jetson/jetformer` の中に置く。ユーザーの `/home/jetson/sim` は読むだけ。git commit はメインセッションが行う。

| 段階 | 担当するファイル(新規は *) | 他の段階に約束するもの |
|---|---|---|
| A: 導入・環境・プロセス | `jetformer/sim/__init__.py`*、`paths.py`*、`install.py`*、`env.py`*、`requirements-bridge.txt`*、`bridge_runner.py`*、`bridge_proc.py`*、`datasets.py`*、`tests/test_sim_install.py`*・`test_sim_env.py`*・`test_sim_bridge_proc.py`*・`test_sim_runner.py`*・`test_sim_datasets.py`* | `install.active_release() -> Release|None`(`.id`、`.root`、`.manifest`、`.trusted_sha256: set[str]`)、`install.list_inbox()`、`install.list_releases()`、`install.install_release(name, activate, progress)`、`install.activate(id)`、`install.delete_release(id)`、`env.status()`、`env.setup(force, progress)`、`BridgeProcess`(§2.2)、`datasets.list_sim_datasets()`、`datasets.archive_path(sid)`、`datasets.delete(sid, in_flight)` |
| B: 配信と中継 | `jetformer/sim/web.py`*(Router、`init`、`shutdown`、ページとファイル、`/api/sim/*` の受け口)、`proxy.py`*、`static/sim-shim.js`*、`static/sim-shim.css`*、`tests/test_sim_web.py`*・`test_sim_proxy.py`*・`test_sim_models.py`* | `/api/sim/*` の JSON の形(§3.5)。A と C の関数を呼ぶだけで、自分では検証しない |
| C: コース | `jetformer/sim/worldhash.py`*、`courses.py`*、`data/known_courses.json`*、`data/default-v3-physics-world.json`*、`static/course-scene.js`*、`static/course-world.js`*、`tests/fixtures/sim/*`*、`tests/test_sim_worldhash.py`*・`test_sim_courses.py`* | `courses.list_courses()`、`get_course(cid) -> Course`(`id, kind, name, world_version, world_hash, renderer, file_for(rel) -> Path|None, assets_dir, features, has_shortcut, size_m, deletable`)、`import_package(path, filename, replace)`、`delete_course(cid)`、`get_default()`、`set_default(cid)`、`course_data_module(cid) -> str`、`export_package(cid) -> bytes|iter`、`CourseError` |
| D: 画面と組み込み | `jetformer/web/server.py`(§3.6 の数行)、`jetformer/web/static/index.html`・`app.js`・`style.css`(タブ)、`README.md`(「シミュレーター」の節: 使い方、コースの形式、JS を実行しない理由、注意) | なし |
| E: 立ち上げ | wheel を `data/sim/wheels/` に写す、§8.2 の条件で再起動、§8.3 の 1〜11、報告(この段階の Markdown は `/home/jetson/jetformer_reports/sim_integration_20261013/` に置く) | |

- A・B・C は、上の関数の約束を守れば同時に進められる。D は B の API の形だけに依存する。
- B と C は、`release` の情報を A の `install.active_release()` から取る。テストでは偽物に差し替える。
- 試作物は `/tmp`(再起動で消える)にある。C は早めに写しておく。
  - `/tmp/sim_integration_20261013_course/worldhash.py`
  - `/tmp/sim_integration_20261013_course/regen.py`
  - `/tmp/sim_integration_20261013_course/v3-physics-world.json`
  - shim の試作 `/tmp/claude-1000/-home-jetson-jetracer/f44d13ba-776f-4620-a46a-2574918483a9/scratchpad/simtest/shim.js`
- docstring と画面の文字は日本語で書き、既存のコードの書き方(短い docstring、`HTTPException(…, '日本語')`、jobs)に合わせる。
- `.gitignore`: `data/` はすでに無視されているので、`data/sim/` と `data/sim_courses/` は追加しなくてよい。`tests/fixtures/sim/` と `jetformer/sim/data/` はコミットする(小さい)。
- 新しい依存はない。jetracer env の fastapi 0.143・starlette 1.7・httpx 0.28.1・cv2・numpy をそのまま使う。
- 作業ツリーには、別の実験の未追跡ファイルがある(`tools/sim_cotrain*.py`、`tools/course_track.py`、`tests/test_sim_cotrain*.py`、`experiments/20261013_sim_rl_bestlap/`、変更済みの `tools/rl_policies.py`)。これらには触らず、コミットにも混ぜない。新しいテストの名前(`test_sim_install.py` など)は、これらと重ならない。

---

## 10. 対象外(今回はやらない)
- ユーザーのシミュレーターのファイル(dist/app.js など)の書き換えと、作者への修正の依頼。
- アップロードされたコースの JS の実行、隔離した実行(別オリジンや sandbox)。
- コースの編集画面(GUI のエディター)。course.html や course_map.json からの自動変換(understand_course §4 の案。P3)。写真や Gaussian Splatting からのコースの生成。
- 汎用描画の見た目を scene.js と同じにすること(部屋、LED 矢印、テープ、ラベル)。
- Default 以外のコースで RC 教師・軌跡最適化教師・speed-fast・20 分 60 シーン計画を使えるようにすること。
- コースごとのカメラ位置・車体寸法・ゲート幅の変更。z 一定以外の必須横断。
- 複数の利用者やタブで同時に AI 走行すること(ブリッジもモデルも 1 つ)。同時に収集できるのも 1 本(ブリッジの仕様)。
- ブリッジの GPU 推論(TensorRT)。GPU は別の実験が使っている。
- 収集の途中からの再開(ブリッジが対応していない)。jetformer-web の再起動をまたいだ、ブリッジの存続。
- JetFormer の認証の追加。
- シミュレーター本体をブラウザからアップロードして導入すること。
- Mac で単体で動かす使い方(start.command)は、これまでどおりユーザーの ZIP で使える(変えない)。
- 学習や走行の側でコースを見分けること(world_version のタグ付けなど)。
- Windows 用の .bat。

---

## 付録 A: 確かめてある数値と参照
- ZIP: `/home/jetson/sim/course-simulator-complete.zip`(926,949,582 B、853 エントリー、すべて `course-simulator/` の下、シンボリックリンクなし)。COMPLETE-MANIFEST.json は schema `course-simulator-complete-v1`、created_at `2026-10-13T14:40:13+09:00`、files 851 件、sha256 `6f610dd5c0da…`。manifest にないのは `COMPLETE-MANIFEST.json` と `最初にお読みください.txt` の 2 つ。
- Default: `dist/scene.js` sha256 `878fc934af28…`、`dist/world.js` `4099860131f4…`、`dist/sim-core.js` `aaeb23798ee4…`、`dist/app.js` `4a43205fbca1…`、`bridge.py` `e04aefe19940…`。
- v2 スナップショット: scene.js `6e4a72655f5a…`、world.js `ec3a813d7d4e…`、physics-world.json `a2596189f0ef…`、world_hash `547d7e80`。スナップショットの manifest と COMPLETE-MANIFEST の両方と一致している。
- ブリッジ: 推論の中央値は約 147 ms(ORT 4 スレッドなら単体で約 59 ms)、RSS は約 290 MiB。bundle は 0.29 秒、activate は 1.02 秒(understand_bridge §10)。
- app.js が使うのは、scene.js の `buildCourse(scene).physics` と、world.js の `makeWorld`・`WORLD_VERSION` だけ(`app.js:34` `const model = buildCourse(scene), world=makeWorld(model.physics);`)。
- 深度は `scene.overrideMaterial = depthMaterial` で、シーン全体を描いて作る(汎用描画のメッシュも ToF に写る)。
- JetFormer: fastapi 0.143.0、starlette 1.7.0、httpx 0.28.1、uvicorn 0.54.0、Python 3.13.14。node は v24.19.0。空きは約 25 GB、メモリは 7.3 GiB(スワップなし)。

## 付録 B: ユーザーに確かめたいこと(実装は既定の案で進めてよい)
1. 旧 v2 を最初からコースの選択肢に出してよいか。既定の案では「旧コース v2」として出す。
2. 今後の「コースのデータ」が、シミュレーター作者のスナップショット(JS 付き)で届く場合の扱い。JS がリリースに入っていなければ簡易描画になる。見た目まで同じにしたいときは、作者に新しい版の本体 ZIP として出してもらい、`~/sim` に置いて導入する。
3. Default を「導入した版の dist のコース」とする(新しい版でコースが変われば Default も変わる)ことでよいか。v3 に固定したいなら、P2 の「前の版の Default を残す」を入れる。
