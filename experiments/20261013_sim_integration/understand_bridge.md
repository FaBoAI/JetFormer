# CourseLab ブリッジ調査(understand_bridge)

対象: `/home/jetson/sim/extracted/course-simulator/`
読んだファイル(全文): `bridge.py`, `dataset_recorder.py`, `jetformer_policy.py`, `jetformer_upload.py`, `requirements-model.txt`, `start.command`(と `start-modelB/rc/shortcut/speed.command`)。補助として `qa/combine_scene_datasets.py`、`dist/app.js` と `dist/jetformer-link.js` のブリッジ呼び出し部分、README の該当節も読んだ。
シミュレーターのファイルは一切変更していない(`__pycache__` もできていない。実行はすべて `-B`、作業は scratchpad の venv)。

---

## 0. 要点(結合する人向け)

1. **ブリッジは 127.0.0.1 に固定で bind し、`Host` と `Origin` を厳しく確認する。** Mac のブラウザ → JetFormer(:8080) → ブリッジという逆プロキシにするなら、プロキシ側で次の 3 つが必須。
   - `Host: 127.0.0.1:<bridge_port>` に書き換える(LAN の Host だと `403 host_not_allowed`。実測済み)
   - `Origin` を消す(`http://192.168.5.14:8080` だと `403 origin_not_allowed`。実測済み。同一オリジンでも POST には Origin が付く)
   - リクエストの target を `//` で始めない(`400 invalid_path`)
2. **フロントの `dist/app.js` はループバック以外のブリッジ URL を受け付けない。** `bridgeURL()`(app.js 208行)は `http:` かつホスト名が `127.0.0.1`/`localhost`/`[::1]` の場合しか通さず、しかも `u.origin` を返すのでパス(例 `/sim`)が消える。ダウンロードリンク(351行・428行)も `url.origin===bridgeURL()` と `pathname.startsWith('/api/datasets/download/')` を確認する。そのため **`/sim/` の下で配信するには、app.js の上書きか、fetch・XHR・`<a href>` を書き換えるシムのどちらかが必要**。詳しくは §8。
3. **相対パスでの配信には対応している。** `index.html` は `./styles.css`・`./app.js` と importmap(`./vendor/three.module.js`)だけを使う。`/sim/`(末尾スラッシュあり)で配信すれば静的ファイルはそのまま動く。`/sim` → `/sim/` へのリダイレクトだけは必要。
4. **パスを指定してサーバー側でモデルを有効にする HTTP エンドポイントはない。** 有効化にはこのプロセスで発行したアップロードトークンが必要。ただし JetFormer の `GET /api/models/<run>/bundle.zip` をそのまま `POST /api/models/bundle` に送り、続けて `POST /api/models/activate` を呼べば有効化できる(バックエンド→ブリッジのループバックで実測。bundle 0.29 秒、activate 1.0 秒、`clean_driving_20261011-jetformer_v0-s1-ft-real-lr1e-4-e12`)。
5. **Jetson で動かせる。** conda `jetracer` の Python 3.13.14 から `python -m venv` で venv を作れ、pip も入る。ピン留めされた wheel は 3 つとも aarch64/cp313 で PyPI にある(`--only-binary=:all:` で確認済み)。venv は約 221 MB。ブリッジの `--self-test`(13件)、同梱の unittest(model_upload 14件、dataset_recorder 13件、dataset_collection 7件、jetformer_upload 16件)はこの venv で全部通った。
6. **ONNX の CPU 推論は 1 ステップ 50〜150 ms(約 7〜20 Hz)。** シミュレーターは推論を待って 1 ステップずつ進む(app.js の `pending`)ので、結果は変わらず、実時間より遅く進むだけ。RSS は約 290 MiB。
7. **運用上の注意:** `runtime/uploads/` は再起動しても消えない。合計 2 GiB を超えるとアップロードを拒否する(56 MB のモデルで約 36 回分)。データセット収集の作業ファイル `_frames/*.npz` も残る。削除する API はないので、ラッパー側での片付けが必要。

---

## 1. CLI と起動

`python bridge.py [--policy FILE.py] [--port N] [--no-browser] [--self-test]`

| 引数 | 内容 |
|---|---|
| `--policy` | 明示した `.py` だけを import する(`predict(observation)` が必須。`initialize_policy()`、`POLICY_NAME`、`CAPABILITIES`、`POLICY_INFO` は任意)。読み込みに失敗しても起動は続き、`policy_error` に理由が入る |
| `--port` | 既定は 8766。範囲は 0..65535。**使用中のときは自動で空きポート(port 0)に切り替わり**、stdout に `[bridge] http://127.0.0.1:<port>` を出す(flush あり)。0 を指定するとランダムなポート |
| `--no-browser` | 付けないと 0.3 秒後に `webbrowser.open(url)` を呼ぶ。**Jetson やサービスとして動かすときは必須**(デスクトップでブラウザが開く恐れがある) |
| `--self-test` | 標準ライブラリだけで動く HTTP テストを実行する |

- **`--host`、`--static-root`、`--runtime-root` はない。** bind 先は `("127.0.0.1", port)` に固定。`main()` は `BridgeServer(port, ROOT/"dist", runner)` を作る。runtime の既定は `static_root.parent/"runtime"` で、結果は `ROOT/runtime`(ユーザーのフォルダー)になる。
- ただし `BridgeServer(port, static_root, runner, runtime_root=None)` はコンストラクタで静的ルートと runtime を受け取れる。**ラッパーで import して直接作れば、ユーザーのフォルダーに書かずに動かせる**(§9 で実測)。
- 終了は SIGINT(KeyboardInterrupt → `server_close()`)。jetformer-web の unit は `KillSignal=SIGINT`、`KillMode` は既定の control-group なので、子プロセスにも SIGINT が届いて正しく終わる。
- ログ: リクエストごとに stderr へ `[bridge] METHOD /path -> code` を出す(本文とクエリは出さない)。収集中は 30 行/秒程度になるので、journald に流さず、ファイルに回すか捨てる方がよい。
- `start.command` の流れ: python3.13→3.12→3.11→python3 の順に 3.11 以上 3.14 未満を探す → `.venv` を作る → `importlib.metadata.version` で 3 パッケージの版を**完全一致で**確認 → 合わなければ `pip install --index-url https://pypi.org/simple --only-binary=:all: -r requirements-model.txt` → `bridge.py "$@"`。`start-modelB.command` は `JETFORMER_MODEL/CONFIG/CALIBRATION` を export して `--policy jetformer_policy.py` を付ける。`start-rc/shortcut/speed.command` は env を設定して `start.command --policy jetformer_policy.py` を exec する。
  - なお `jetracer` env の opencv は `4.13.0.92+jp72cuda132` なので、start.command の版の完全一致確認は失敗する(start.command は使わないので問題にはならない)。

## 2. エンドポイント一覧

共通の仕様:
- エラーは `{"error":{"code":..., "message":...}}`。
- 応答にはすべて `Cache-Control: no-store`、`X-Content-Type-Options: nosniff`、`Connection: close` が付く。
- 本文は `Content-Length` が必須(411)。chunked は不可(400)。`Content-Type` は完全一致(415)。サイズ上限を超えると 413。
- 本文を読むときのソケットのタイムアウト: 上限が 2 MiB を超えるエンドポイントは 120 秒、それ以外は 10 秒。

| Method | Path | 本文 / 上限 | 応答 |
|---|---|---|---|
| GET/HEAD | `/api/status` | – | `{policy_loaded, name, policy_name, schema:"course-sim-observation-v1", action_schema:"course-sim-action-v1", capabilities{camera,ranges,depth,imu}, protocol, model_upload{available,max_onnx_bytes:256MiB,max_json_bytes:1MiB,max_zip_bytes:300MiB,local_only}, [policy_capabilities], [policy_info], [policy_error], jetformer_upload{available,local_bridge_only,simulator}, dataset_collection{available,local_only, + 収集中なら session_id/scene/frames/frame_count/recording/training_label_mode/max_frames/download_url。それ以外は recording:false,max_frames,schema}}`。dataset_collection の詳細は DatasetManager の初期化後にだけ入る |
| POST | `/api/predict` | `application/json`、2 MiB | 観測(§3)→ `{steering, speed, [info]}`。エラー: 400 invalid_observation/invalid_policy_identity、503 policy_not_loaded/**policy_busy(ロックを待たない)**、409 policy_identity_changed、502 invalid_action、500 policy_failed |
| POST | `/api/models/upload?kind=onnx\|config\|calibration&name=<file>` | onnx は `application/octet-stream` で 256 MiB、config/calibration は `application/json` で 1 MiB(JSON オブジェクト必須)。さらに `runtime/uploads` の残り容量(合計 2 GiB)でも制限される | `{token(32hex), kind, name, size, sha256}`。409 upload_busy(upload_lock を待たない) |
| POST | `/api/models/bundle?name=<zip名>` | `application/zip`、300 MiB | `{onnx_token, config_token, calibration_token, io_spec_token, io_spec, model_config, model_sha256, defaults{mode_id,lookahead,steer_gain,throttle_gain}, ignored_files[], executed_bundle_code:false}`。400 invalid_bundle、500 bundle_storage_failed、409 upload_busy |
| POST | `/api/models/activate` | JSON、2 MiB: `{onnx_token, config_token, [calibration_token], [mode_id], [lookahead], [steer_gain], [throttle_gain]}` | 成功時は `/api/status` と同じ形(`policy_loaded:true`, `name:"JetFormer · <model>"`, `policy_info{...model_sha256, model_config_sha256, effective_calibration_sha256, mode_id, trained_modes, max_speed_mps, max_steering_rad, simulator_physics, simulator_sensors, [simulator_execution], io_spec…}`)。409 policy_busy、400 model_load_failed(**失敗しても前のモデルは残る**) |
| POST | `/api/datasets/start` | JSON、2 MiB: `{controller: manual\|demo\|external, mode_id: 0..5\|8 (既定5), split: train\|val\|test, include_automated_labels: bool, name: 1..100文字・パス不可, context{…最大2MiB…}, config{…}, [calibration], [model_config]}` | `{session_id, scene, frames, frame_count, recording:true, training_label_mode, max_frames:18000, download_url:null}`。**同時に収集できるのは全体で 1 本だけ** |
| POST | `/api/datasets/frame` | JSON、2 MiB: `{session_id, observation, action{steering,speed,[info]}, [pose{x,z,yaw}], [state]}` | 上と同じ形(frames が増える)。条件: 224x224 PNG、dt=1/30、フレームが連続、`depth.packed_data` が 2268 個の uint16、imu が 10x6 で timestamps 付き。上限は 18000 フレーム、2 GB |
| POST | `/api/datasets/finish` | JSON、**16 MiB**: `{session_id, [report]}`(report は最大 8 MB) | `{…, download_url:"/api/datasets/download/<sid>", filename, training_frames, bytes, sha256, split}`。2 回呼んでも同じ結果になる |
| POST | `/api/datasets/collection` | JSON、2 MiB: `{name, session_ids[1..1000], expected_frames(600), expected_duration_s(20), require_collision_free}` | `{schema, session_id, filename, sha256, bytes, collection_schema, scenes, frames, duration_s, training_frames, splits{train,val,test}, download_url}`。**リクエストを処理するスレッドの中で同期的に実行する**(app は 600 秒のタイムアウトで待つ)。条件: 出力 5 GiB 以下、**空き容量が 20 GiB + 元データの 2 倍以上**(現在の空きは 26.7 GB) |
| GET/HEAD | `/api/datasets/download/<sid32>` | – | `application/zip` を 1 MiB ずつストリーム、`Content-Disposition: attachment`。**ブリッジを再起動しても `archive-info.json` から取り出せる** |
| POST | `/api/jetformer/check` | JSON: `{url, [session_id]}` | `{target_url, connected, compatible, dataset, existing, import_status: not_selected\|not_imported\|already_imported\|conflict, content_verified, modes, dataset_count, disk, warnings, message}`。400/502 |
| POST | `/api/jetformer/upload` | JSON: `{url, session_id}` | **202** で job `{id, job_id, status_url, state:"queued", target_url, session_id, dataset{session_id,name,filename,bytes,sha256,scenes,frames,duration_s,train_scenes,val_scenes,mode_frames,synthetic}, bytes_sent,total_bytes,progress, remote_job_id, remote_job, result, error, content_verified, can_retry, message, created_at, updated_at}`。409 jetformer_busy |
| GET | `/api/jetformer/status/<id>` | – | 上の job。不明な id は 404 |
| OPTIONS | 上の API パス | – | プリフライト。対応するメソッド 1 つと `Content-Type` ヘッダーだけを許可し、Origin があれば `Access-Control-Allow-Private-Network: true` を付ける。API 以外のパスは 404 |
| GET/HEAD | `/api` と `/api/*` のその他 | – | 404 JSON |
| GET/HEAD | それ以外のすべて | – | 静的ファイル(§5) |

## 3. 観測と行動のスキーマ(`/api/predict` の検証)

- 観測: `schema=="course-sim-observation-v1"`、`episode_id` は str か int、`frame` は 0 以上の int、`sim_time` は 0 以上、`dt` は正、`speed`/`steering` は有限値、`ranges` は 1〜1024 個の `{angle, distance>=0}`。`camera{width,height}` は (224,224) か旧形式の (320,180)、`encoding:"png-base64"`、`data` は接頭辞なしの base64 PNG で、IHDR が宣言した大きさと一致すること。それ以外のフィールド(`depth`, `imu`, `vehicle`, `actuator_limits`, `sensor_config`, `policy_identity`)はそのまま通す。ネストの奥まで含めて非有限の float は拒否する。
- `policy_identity`(任意)= `{model_sha256, model_config_sha256, effective_calibration_sha256}`。有効なモデルと違えば 409(別のタブでモデルを切り替えたときの保護)。
- 行動: `{steering(rad, 正は左), speed(m/s), [info]}`。JSON にして 2 MiB 以下。
- JetFormer アダプター(jetformer_policy)の追加条件:
  - `actuator_limits.max_speed_mps` と `max_steer_rad` が校正値と一致すること(UI の設定と校正を突き合わせる)
  - `depth.packed_data` が 54x42、imu が 10 サンプル以上で、最新サンプルが 100 ms 以内
  - 前処理は JPEG q92 の往復 → 224 → ImageNet 正規化。ToF は `(raw&0x1FFF)/tof_max_mm` と信頼度 `/7`。IMU の加速度は 9.80665 で割る。state は前回 chunk の PWM と eRPM の正規化、`command_input:false` ならゼロ
  - 出力 `actions[1,10,2]`*scale+center → `lookahead` 番目の行 → ゲイン → PWM を SI 単位に変換(前進は throttle が中立より**下**)
  - 同じフレームが重複したらキャッシュを返す。episode が変わるか frame/time が戻ったらリセット

## 4. セキュリティ確認

- **Host**: 小文字にして `{"127.0.0.1:<port>", "localhost:<port>"}` と完全一致で比べる。`[::1]` は含まれない。
- **Origin**: 無いか、`{"null", "http://127.0.0.1:<port>", "http://localhost:<port>"}` のどれかであること。CORS はその Origin をそのまま返し(`Vary: Origin`)、`*` は使わない。
- **リクエストのパス**: `/` で始まり、`//` で始まらないこと(絶対 URL 形式は拒否)。
- **静的ファイル**: strict UTF-8 で unquote し、NUL、`\`、`..` セグメントを拒否(403)。`resolve()` してから `relative_to(static_root)` で確認するので、**static_root の外を指すシンボリックリンクは 403**。ディレクトリなら `index.html`。
- **モデル ZIP**:
  - エントリ数 1〜1000、展開後の合計 512 MiB 以下
  - 絶対パス、`..`、`\`、`:`、NUL、重複(casefold)、シンボリックリンク、特殊ファイル、暗号化、stored/deflate 以外の圧縮は拒否
  - basename で `model_config.json` と `io_spec.json` がちょうど 1 つずつ、`io_spec.onnx` が指す ONNX と 3 つとも同じフォルダー
  - io_spec は `jetformer_v0` で、model_config と突き合わせる
  - **ZIP の中のコードは実行しない**(`executed_bundle_code:false`)
- **ONNX**:
  - バイト列から `InferenceSession(bytes, providers=["CPUExecutionProvider"])` で読む(外部テンソル参照はできない。カスタム op も読まない)
  - 入出力の名前と形状を検証してから 3 回ウォームアップする
  - 成功したときだけ runner を差し替える
- **`--policy`** だけが任意のコードを import する。CLI から指定するもので、HTTP からは指定できない。
- **JetFormer への転送先**:
  - http(s) のオリジンだけ(認証情報、パス、クエリ、フラグメントは不可)
  - `getaddrinfo` の結果が**すべて** 127/8、10/8、172.16/12、192.168/16、::1、fc00::/7 のどれかに入ること
  - 最初のアドレスに固定して接続する(DNS rebinding 対策。Host ヘッダーと SNI は元の名前のまま)
  - リダイレクトしない。PUT を自動で再送しない
- **プロキシでつないだときの注意**: ブリッジは「このマシンのユーザーだけが使う」前提で作られている。JetFormer 経由で LAN に出すと、JetFormer に届く人は誰でも、モデル ZIP のアップロード(300 MiB)、データ収集(1 セッション最大 2 GB)、ブリッジから LAN 内の任意のアドレスへの GET と PUT(`/api/jetformer/*`。転送先はプライベートアドレスだけ)を使えるようになる。JetFormer 自体に認証が無いので信頼の範囲は変わらないが、`/api/jetformer/*` の `url` を JetFormer 自身(例: 127.0.0.1:8080 と画面のオリジン)だけに絞るチェックをプロキシ側に入れておくと安全。

## 5. 静的ファイルと runtime/ の配信

- 静的ファイルは `static_root`(main では `ROOT/dist`)だけ。ファイル全体をメモリに読んでから返す(Range、ETag、圧縮には非対応)。MIME は `.js/.mjs` が `text/javascript`、`.wasm`、それ以外は `mimetypes`。text 系には `charset=utf-8` を付ける。**常に no-store**(three.core.js 1.4 MB などを毎回送り直す。LAN なら許容できる)。
- `dist/` の中身: `index.html, app.js, scene.js, world.js, sim-core.js, sensors.js, ego-sensors.js, jetformer-link.js, control-delay.js, speed-controller.js, rc-speed-controller.js(285 KB), optimized-reference.js(250 KB), sensor-mount-reference.js, styles.css, assets/*.jpg, vendor/three.{core,module}.js, vendor/addons/..., diagnosis/`。
- **`runtime/` は HTTP では配信しない。** 外から取り出せるのは `/api/datasets/download/<sid>` の ZIP だけ。
- **コース**: `dist/world.js`(`WORLD_VERSION='indoor-course-eval-3-turf-board-removed'`、route、required crossings)と `dist/scene.js`(形状)。app.js は `./world.js` と `./scene.js` を import する。旧 v2 は `qa/course-snapshots/indoor-course-eval-2-shortcut/{world.js,scene.js,physics-world.json,manifest.json}`。
  - 静的ルートの中だけに制限されるので、**コースを差し替えるにはシンボリックリンクではなく実ファイルのコピー**(コースごとの dist オーバーレイディレクトリ)が必要。または JetFormer 側が静的ファイルを自分で配信し、`/api/*` だけをブリッジに中継する(こちらの方が簡単)。
  - WORLD_VERSION は収集データの `context.world_version` に入り、まとめ ZIP の contract にも入る。コースを混ぜると、まとめるときに `contracts differ` で弾かれる。

## 6. モデルのアップロードと有効化

- **ファイルを個別に送る方法**: `upload?kind=onnx` → `kind=config` → (任意で)`kind=calibration` → `activate{tokens, mode_id}`。calibration が無いときは `ROOT/calibration.example.json` を使う(mode は 5 が学習済みなら 5、そうでなければ trained_modes の先頭)。
- **ZIP で送る方法**: `bundle` が ONNX、config、io_spec、合成した calibration の 4 つを `runtime/uploads/<token>.{onnx,json}` に保存する(`open("xb")`)。calibration は次の順に重ねて作る。
  1. `calibration.example.json`
  2. 既定値 `{mode, lookahead:3, steer_gain:1.3, throttle_gain:1.0, erpm_per_mps:2000, erpm_forward_sign:-1}`
  3. `model_config.simulator_calibration`(あれば)
  4. `model_config.recommended_inference`(steering_gain・throttle_gain・lookahead)

  ZIP に `calibration.json` が入っていても**使わない**(ignored_files に入る)。
- **activate** は `runner.lock` を待たずに取る(取れなければ 409)。`jetformer_policy.create_controller` が JSON の検証 → ORT セッション作成 → 3 回ウォームアップ → info(SHA256 を含む)を作り、成功したら runner の function、name、policy_info、capabilities を置き換える。`lookahead/steer_gain/throttle_gain` は上書きできる(gain は 0..2、lookahead は 1..10)。
- トークンは**メモリにしか持たない**(`server.uploads`)。再起動するとファイルは残ってもトークンは使えなくなる。`runtime/uploads` の合計が 2 GiB を超えると、以後のアップロードはすべて拒否される。**ラッパーの起動時に `runtime/uploads/` を空にしてよい**(どのみち再利用できないため)。
- **サーバー側からパスで有効化する HTTP API はない。** 代わりに次のどれかを使う。
  - (a) **推奨**: JetFormer のバックエンドが `model_export.iter_bundle_zip(run)`(= `/api/models/<run>/bundle.zip`)のバイト列を、`Host: 127.0.0.1:<port>`、`Content-Type: application/zip`、`Content-Length` 付きで `POST /api/models/bundle` に送り、返ったトークンで `POST /api/models/activate`。実測で動いた。JetFormer の bundle は ONNX を `ZIP_STORED`+zip64、`external_attr=0o644<<16` で作るが、ブリッジの検証を通る。
  - (b) 起動時に `--policy jetformer_policy.py` と、環境変数 `JETFORMER_MODEL`、`JETFORMER_CONFIG`、`JETFORMER_CALIBRATION` を渡す。calibration ファイルが必須で、JetFormer の run には calibration が無いので作る必要がある。
  - (c) ラッパーのプロセス内で `server.save_bundle(raw)` と `server.activate_model(tokens)` を `runner.lock` を取った状態で呼ぶ。
  - **UI への影響**: app.js の「走行モード・出力の調整」→「設定を反映してリセット」(`apply-model-settings`)は、ページが持つ `activeModelTokens` を使う。サーバー側で有効化すると、ページは「接続を確認」で `policy_info` を受け取れるが、トークンを持っていないのでゲインやモードの再設定ができない。ブラウザ側で JetFormer の bundle.zip を Blob として取得し、`#model-bundle-file` に DataTransfer で入れてから `#load-bundle` を押させる方法なら、app.js を変えずに元の流れ(トークンもモード選択も含む)がそのまま使える。ただし 56 MB が Mac を 2 回往復する。
- 同梱モデル: `models/modelB`(io_spec が無いので個別アップロードで使う)、`shortcut-finetuned`、`shortcut-rc-finetuned`(現行 v3 で 8.964 秒、trained [4]、simulator_calibration/physics/sensors/execution あり)、`shortcut-speed-finetuned`。どれも 56 MB で num_modes は 6。

## 7. データセットと ZIP の保存場所

runtime_root を R とする。
- `R/datasets/<sid32>/<slug>_<UTCstamp>_<sid8>/`: データセットの展開形
  - `aligned/<scene>.npz`
  - `samples/CAM_FRONT/<scene>/<ts>.jpg`
  - `samples/TOF_FRONT/<scene>/<ts>.png`
  - `can_bus/*.json`
  - `simulation/{session.json,frames.jsonl,[evaluation-report.json]}`
  - `v1.0-<name>/*.json`
  - `dataset_card.json`, `splits.json`, `manifest.json`, `DATASET_README.txt`
- `R/datasets/<sid32>/_frames/NNNNNN.npz`: フレームごとの作業ファイル。**finish の後も消えない。**
- `R/datasets/<sid32>/<dataset_name>.zip`(書き込み中は `.zip.part`)と `archive-info.json`(再起動後のダウンロードで使う印)。
  - **使う容量は ZIP + 展開形 + _frames で、ZIP の約 2 倍以上。**
- まとめ ZIP: `R/datasets/<sid32>/<title>.zip`、`archive-info.json`、`collection-summary.json`。
  - `qa/combine_scene_datasets.py` と `training/speed_physics.py` を **sys.path から namespace package として import する**ので、シミュレーターのフォルダー(qa/ と training/ を含む)が sys.path に入っている必要がある。
- `R/uploads/<token>.onnx|.json`: モデル。上限 2 GiB で、自動では消えない。
- `R/jetformer-transfers.json`(書き込みは `.tmp` → replace): 転送の記録。再起動すると、途中だった転送は `unknown` になる。
- 削除用の API は無い。配布物の `runtime/` には昔のセッションの `evaluation-report.json` が 1 つだけ残っている(ZIP が無いのでダウンロードはできない)。

## 8. JetFormer へのアップロード(jetformer_upload.py)

- 送信元は DatasetManager が作った ZIP(セッション ID で指定)だけ。条件:
  - `dataset_card.json` がちょうど 1 つで 8 MB 以下
  - 置き場所は `<name>/dataset_card.json`
  - name は ASCII の `[A-Za-z0-9_-]`
  - `card.synthetic==true`
- **check** は送り先に `GET /openapi.json`(`PUT /api/datasets/import` があること)、`GET /api/status`(`sensors, recorder, config, disk` を含むこと)、`GET /api/modes`、`GET /api/datasets` を投げ、同じ名前があれば件数と `import_source.sha256/bytes` を比べる。ID8 が「シミュレーター」でなければ警告を出す。
- **upload** はバックグラウンドのスレッドで動く。
  1. もう一度 check する。既にあれば送らない
  2. 空き容量を確認する(`bytes > free_gb*1e9*0.45` ならエラー)
  3. `PUT /api/datasets/import?filename=<zip>&simulator=1` を `Content-Type: application/octet-stream`、`Content-Length` 付きで、1 MiB ずつ送る
  4. 応答の `id` を見て、`GET /api/jobs` を 1 秒おきに最長 3600 秒ポーリングする。`state` が done なら `result.name==dataset名` を確認する
  - 状態は queued、checking、uploading、importing、done、error、unknown、already_imported、conflict。同時に動く転送は 1 本だけ。
- **送り先の URL** はユーザーが画面で入力する(`#jetformer-url`、既定は `http://192.168.55.1:8080`、localStorage の `course-jetformer-url` に保存)。Jetson の上でブリッジを動かす場合は、`http://127.0.0.1:8080`、`http://192.168.5.14:8080`、`http://192.168.55.1:8080`、`http://jetson.local:8080`、`http://localhost:8080` のどれでも check が通った(実測)。ただし「JetFormer Web UIを開く」リンクは同じ URL を Mac のブラウザで開くので、**画面のオリジン(Mac から見える LAN の URL)を既定にするのがよい**。
- 現在の JetFormer(8eac2ee)の `/api/jobs` の形 `{id, state, result{name,…}}` は、このコードの想定と合っている。
- **デッドロックに注意**: `/api/jetformer/check` はブリッジが同期的に JetFormer の `/openapi.json` などを呼ぶ。JetFormer 側のプロキシを `async def` の中のブロッキング HTTP 呼び出しで書くと、イベントループが止まり、30 秒でタイムアウトする。プロキシは `httpx.AsyncClient`(jetracer env に httpx 0.28.1 がある)で書くか、同期の `def` エンドポイント(スレッドプールで動く)にすること。

## 9. スレッドと状態

- `ThreadingHTTPServer`(daemon_threads、allow_reuse_address)。1 リクエストごとに 1 スレッドで、`Connection: close`。
- `PolicyRunner.lock` は**待たない**。推論中に別の predict が来ると 503 policy_busy、activate が来ると 409。**複数のタブが同時に AI 走行すると互いに弾き合う**(ブリッジ 1 つにつき有効なモデルは 1 つ)。
- `upload_lock` は**待たない**。upload と bundle で共有する。
- `DatasetManager._lock`(RLock)は**待つ**。収集は全体で同時に 1 本。finish とまとめ ZIP の作成はロックを持ったまま ZIP を作るので、その間、他の dataset 系リクエストは待たされる。DatasetManager と JetFormerUploadManager は、最初に使われたときに作られる。
- 再起動して残るもの: 完了した ZIP(download と collection で使える)、転送の記録。消えるもの: 有効なモデル、アップロードのトークン、収集中のセッション(README も「再開には未対応」と書いている)。

## 10. Jetson(aarch64, JetPack 7.2)で管理されたサブプロセスとして動かす

**確認したこと(scratchpad だけを使用。jetracer env には何も入れていない)**
- `conda activate jetracer` → `python -m venv <dir>`: Python 3.13.14 で成功し、pip 26.1.2 も入る。`--system-site-packages` も作れる。system の python3.12 は venv も ensurepip も無いので使えない。
- `pip download --only-binary=:all: -r requirements-model.txt`(PyPI): すべて取得できた。
  - `numpy-2.2.6-cp313-cp313-manylinux_2_17_aarch64` 14.0 MB
  - `opencv_python_headless-4.13.0.92-cp37-abi3-manylinux_2_28_aarch64` 35.0 MB
  - `onnxruntime-1.31.0-cp313-cp313-manylinux_2_28_aarch64` 21.4 MB
  - 依存: protobuf-7.36.2 (abi3 aarch64)、flatbuffers-25.12.19、packaging-26.3

  glibc は 2.39 で、2.28 以上の条件を満たす。オフラインで入れるときは `/tmp/claude-1000/-home-jetson-jetracer/f44d13ba-776f-4620-a46a-2574918483a9/scratchpad/wheels/` にある wheel を `--no-index --find-links` で使える。
- venv の中: `onnxruntime 1.31.0` の providers は `['AzureExecutionProvider','CPUExecutionProvider']`。ブリッジは CPU を明示して使う。
- jetracer env そのものには numpy 2.2.6 と opencv 4.13.0(CUDA ビルド)はあるが、**onnxruntime が無い**。その env で直接動かすと、手動、デモ、収集は動くが、モデルの有効化は失敗する。規則上、jetracer env には入れられない。→ **独立した venv(素の venv、約 221 MB)を JetFormer が持つ場所に作るのがよい。** 例: `~/jetformer/data/sim/venv`。`config.DATA_DIR` は `JETFORMER_DATA` の値、なければ `~/jetformer/data`。
  - 注意: この venv は conda の python を `home` として参照するので、jetracer env の Python を 3.14 に上げると壊れる。
- 試験の結果:
  - `bridge.py --self-test`: 13/13 OK
  - unittest(`PYTHONPATH=<simdir>` が必要): test_model_upload 14、test_dataset_recorder 13、test_dataset_collection 7、test_jetformer_upload 16、すべて OK
- **ラッパーで実行**(`probe_bridge.py`。sim_dir を sys.path に入れて `bridge.BridgeServer(0, sim/dist, PolicyRunner(), runtime_root=<別の場所>)`):
  - status 200、`GET /` 200、LAN の Host は 403、LAN の Origin は 403、X-Forwarded-For だけなら 200、`/sim/index.html` は 404
  - JetFormer の bundle を送る: 200 で 0.29 秒
  - activate: 200 で 1.02 秒、mode 5、max 1.6 m/s と 0.52 rad
  - predict を 20 回: 200。1 回目 110 ms、中央値 147 ms、最大 212 ms(アダプター内の ORT は 120 ms、前処理は 7 ms)。RSS は 290 MiB
  - 書き込まれたのは別の runtime だけで、ユーザーのフォルダーは無変更
- ORT 単体のスレッド数ごとの時間(中央値): 既定(6 コア)54 ms、4 スレッドで 59 ms、2 で 104 ms、1 で 197 ms。GPU の実験と並行して動かすなら、`nice -n 10` で動かすのがよい。スレッド数は jetformer_policy が SessionOptions を渡さないので変えられないが、ラッパーで `onnxruntime.InferenceSession` に `intra_op_num_threads` を差し込めば、ファイルを編集せずに 4 程度に絞れる。

**推奨する起動形(ラッパー案。ユーザーのファイルは無変更)**

```python
# 例: jetformer/sim_bridge_runner.py を sim venv の python で -I -B 実行
import sys
sim_dir, port, static_root, runtime_root = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
sys.path.insert(0, sim_dir)            # bridge / dataset_recorder / jetformer_policy / qa / training
import bridge
server = bridge.BridgeServer(port, static_root, bridge.PolicyRunner(), runtime_root=runtime_root)
print(f"[bridge] http://127.0.0.1:{server.server_address[1]}", flush=True)
try:
    server.serve_forever(poll_interval=0.2)
except KeyboardInterrupt:
    pass
finally:
    server.server_close()
```

- `-B`(または `PYTHONPYCACHEPREFIX`)で、ユーザーのフォルダーに `__pycache__` を作らないようにする。`-I` を付けると sys.path にスクリプトのディレクトリも cwd も入らないので、上のように sim_dir を明示的に入れる。
- `ROOT/calibration.example.json` は bridge.py と同じフォルダーにあることが前提。
- ポート: 固定(例 8766)にして、使用中なら失敗させる方が管理しやすい。`main()` の黙った自動切り替えは使わない。stdout の 1 行目からも取れる。
- ヘルスチェックは `GET http://127.0.0.1:<port>/api/status`(`Host` は自動で付く)。
- プロキシのタイムアウト: activate と bundle は 120 秒以上、finish は 120 秒以上、collection は 600 秒以上。本文の上限は bundle の 300 MiB。ダウンロードはストリームで返す(ZIP は 700 MB 級)。上りは `Content-Length` を付けて送ること(ブリッジは chunked を拒否する)。
- `/sim` → `/sim/` へリダイレクトする。
- jetformer-web と同じ cgroup で動くので、jetformer-web を再起動するとブリッジも止まり、収集中のデータは失われる。

**フロント側で必要な対応(§0-2 の詳細。どちらかを選ぶ)**
- (A) **index.html だけを差し替え、シムを入れる。** app.js より前に読む `<script>` で次を行う。
  - `fetch`、`XMLHttpRequest.open`、`<a>.href` の値が `http://127.0.0.1:8766`(`#endpoint` の既定値)で始まるなら、`location.origin + '/sim'` + パスに書き換える
  - `#jetformer-url` の既定値を `location.origin` にする
  - app.js 57 行は、ループバックでないページでは `#endpoint` を変えないので、既定の `http://127.0.0.1:8766` がそのまま残り、`bridgeURL()` の検査も通る
  - ダウンロードリンクは `href` を差し替えた後に設定される(352行、429行)ので、MutationObserver か、click をキャプチャして書き換える処理が必要
- (B) **app.js を JetFormer 側のコピーとして最小限だけ直す。** `bridgeURL()` を `location.origin+'/sim'` を返すようにし、351 行と 428 行の `url.origin!==bridgeURL()` と `new URL(download_url, base)` を、パス接頭辞を考えた形に直す。下流に `download_url` が絶対パスで返ってくることに注意(`new URL('/api/...', 'http://h/sim')` は `/sim` を落とす)。差し替えるときは元の文字列が見つかることを確かめ、見つからなければエラーにする方式にすると、ユーザーがシミュレーターを更新したときに気づける。
- localStorage のキー(`course-jetformer-url`、`course-jetformer-transfer`、`course-20s-collection`)は JetFormer のオリジンと共有になる(衝突は見当たらない)。

## 11. その他の気づき

- `/api/datasets/collection` の空き容量条件(20 GiB + 元データの 2 倍)は、今の Jetson(空き 26.7 GB ≈ 24.9 GiB)だと元データ約 2.4 GiB までが限度。60 シーン・20 分(約 700 MB)なら通るが、余裕は少ない。
- `simulator.html`(3.9 MB の単一ファイル)は `file://` 用(Origin null)。統合では `dist/` を使う方がよい。
- `qa/` と `training/` には `__init__.py` が無い(namespace package)。venv に同じ名前のパッケージがあると衝突する恐れがあるが、素の venv なら問題ない。
- `dataset-downloads/jetformer-simulator-id8-20min.zip`(709,547,352 bytes)は JetFormer に取り込み済み(`/api/modes` の ID8「シミュレーター」に 60 シーン、20.0 分)。
