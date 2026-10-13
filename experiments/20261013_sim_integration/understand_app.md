# CourseLab ブラウザアプリの調査（understand_app）

- 調査日: 2026-10-13
- 対象: `/home/jetson/sim/extracted/course-simulator/`（ユーザー提供。読み取りのみで、ファイルは変更していない）
- 読んだファイル: `dist/index.html`, `dist/app.js`（全行）, `dist/jetformer-link.js`, `dist/world.js`, `dist/scene.js`（全行）, `dist/sensors.js`, `dist/sim-core.js`（ワールドの正規化・ハッシュ・衝突処理）, `dist/speed-controller.js` / `rc-speed-controller.js` / `optimized-reference.js`（コース依存の部分）, `build-local-html.py`, `bridge.py`（Host/Origin/本文サイズの制約）, `dataset_recorder.py`（download_url）, `qa/course-snapshots/indoor-course-eval-2-shortcut/*`, `qa/load-course-world.mjs`, `qa/verify-learned-lap.mjs`（loadWorldSnapshot）
- 確認作業はすべて scratchpad（`/tmp/claude-1000/.../scratchpad/simtest`, `imtest`）で行った。テスト用サーバーとヘッドレス Firefox は終了済み。

---

## 0. 結論

1. **ブリッジの接続先は `#endpoint` 入力欄の値だけで決まる。** 既定値は index.html に書かれた `http://127.0.0.1:8766`。クエリパラメーター（`location.search` / `URLSearchParams`）も localStorage も参照しない。例外は app.js:57 で、ページ自体を `http://127.0.0.1:*` または `http://localhost:*` で開いた場合だけ、値が `location.origin` に置き換わる。
2. **app.js を変更せずに Jetson 上のブリッジやサブパスを指定する方法はない。** `bridgeURL()`（app.js:208）は、`http:` で hostname が `127.0.0.1` / `localhost` / `[::1]` のもの以外を例外で拒否し、戻り値は `u.origin` だけになる（パスは捨てられる）。Mac から `http://<jetson>:8080/...` で開いた場合、ブラウザがブリッジに到達できる値を入れる方法がない。
3. **ブリッジ側の最小で堅牢な方法:** JetFormer が配信する index.html の `<head>` に、JetFormer 側が管理する小さな**クラシック（非モジュール）スクリプト**を挿入する。このスクリプトは `window.fetch` と `XMLHttpRequest.prototype.open` をラップし、「現在の `#endpoint` の origin を宛先とし、パスが `/api/` で始まる絶対 URL」を、同一オリジンの `/sim/bridge/api/...` に書き換える。ダウンロード用の `<a href>` も同じ規則で書き換える。`bridgeURL()` は検証を通る偽のループバック origin を返し続けるので、アプリ内部の origin チェックも通る。**ユーザーのファイルは一切変更しない。** 動作は Jetson 上のヘッドレス Firefox 157 で確認済み（§3.3）。
4. **コースは `scene.js` と `world.js` の2ファイルだけで決まり、app.js はどちらも相対パスで import している。** そこで `/sim/c/<course_id>/` のようにコースごとに URL パスを分け、`scene.js` と `world.js`（任意で `assets/*`）だけをコースフォルダーから、それ以外は `dist/` から配信すれば、ページを読み込むたびに別のコースを使える。ユーザーの v2 スナップショット（`qa/course-snapshots/indoor-course-eval-2-shortcut/`）は、そのまま差し替え用コースとして使える。node でハッシュを確認した: Default は `0a464384`、v2 は `547d7e80`。
5. Default 以外のコースでは、**RC 教師・軌跡最適化教師・20分/60シーンの収集計画は使えない**（v3 の形状と照合して、日本語のエラーで拒否される）。`speed-fast` 教師は v3 専用の誘導経路を使う。手動・低速デモ・`speed` 教師・学習済み AI の評価は、どのコースでも使える。

---

## 1. ファイル構成と読み込みの仕組み

- `dist/index.html` の参照は相対パスだけで、絶対パスは含まない:
  - `<link rel="stylesheet" href="./styles.css" />`（styles.css には `url()` も `@import` もない）
  - `<script type="importmap">{"imports":{"three":"./vendor/three.module.js","three/addons/":"./vendor/addons/"}}</script>`
  - `<script type="module" src="./app.js"></script>`
  - したがって、ドキュメントの URL は**末尾が `/` のディレクトリ URL でなければならない**（例: `/sim/c/default/`）。末尾のスラッシュがないと、`/sim/c/app.js` や `/sim/c/vendor/...` を読みに行って壊れる。サーバー側でリダイレクトが必要。
- モジュールの依存関係は静的で、動的 `import()` も `import.meta` も使っていない（`build-local-html.py` もこれを前提としている）:
  - `app.js` → `three`, `three/addons/controls/OrbitControls.js`, `./scene.js`, `./sim-core.js`, `./speed-controller.js`, `./optimized-reference.js`, `./rc-speed-controller.js`, `./world.js`, `./sensors.js`, `./control-delay.js`, `./ego-sensors.js`, `./jetformer-link.js`, `./sensor-mount-reference.js`
  - `vendor/three.module.js` → `./three.core.js`、`OrbitControls.js` → `three`、`scene.js` → `three`
  - `speed-controller.js` → `./sim-core.js`。`optimized-reference.js` と `rc-speed-controller.js` → `./sim-core.js`, `./speed-controller.js`
  - `three` は importmap 経由で全モジュールが同じ URL に解決されるため、THREE のインスタンスは1つになる。これは重要で、`THREE.DefaultLoadingManager.onLoad` が scene.js のテクスチャ読み込みを検知できるのは、インスタンスが共通だからである（§4.4）。
- 通信が発生する箇所は次の3つだけ。app.js の fetch/XHR（ブリッジ宛て）、jetformer-link.js（app.js から渡された `post` / `get` を使うので、同じくブリッジ宛て）、scene.js の `TextureLoader.load('./assets/'+name)` によるテクスチャ6枚。
- localStorage を使う箇所: app.js の `course-20s-scene-ids` と `course-20s-collection`、jetformer-link.js の `course-jetformer-url` と `course-jetformer-transfer`。sessionStorage は使っていない。JetFormer 本体の UI は localStorage を使っていないため、キーの衝突はない。
- `window.courseSimulator` は凍結された読み取り専用の窓口で、`snapshot()`, `report()`, `world()`, `sensorSummary()` を持つ。挿入スクリプトから利用できる。app.js の最後のほうで定義されるため、WebGL の初期化に失敗した場合は存在しない。

## 2. ブリッジの見つけ方と、呼び出す URL

`{B}` は `bridgeURL()` の戻り値で、`#endpoint` の値から取り出した **origin だけ**（パスは含まない）。

| 呼び出し元 | 方法とパス | 本文 / 時間制限 |
|---|---|---|
| connect(), record-start | `GET {B}/api/status` | 5 s |
| 外部モデルで走行中、毎フレーム | `POST {B}/api/predict` | 観測 JSON（224×224 PNG の base64、ToF 54×42 の配列、IMU）。約100 KB/フレーム。5 s。結果を待ってから次のフレームに進む |
| モデル ZIP | XHR `POST {B}/api/models/bundle?name=<file>` | `application/zip`、300 MiB 以下、120 s |
| ONNX/JSON を個別に | XHR `POST {B}/api/models/upload?kind=onnx|config|calibration&name=` | `application/octet-stream` / `application/json` |
| モデルの有効化 | `POST {B}/api/models/activate` | 120 s |
| 収集 | `POST {B}/api/datasets/start`（15 s）、`/frame`（収集中は毎フレーム、15 s）、`/finish`（120 s）、`/collection`（600 s） | JSON |
| JetFormer への送信（jetformer-link.js） | `POST {B}/api/jetformer/check`（60 s）、`/api/jetformer/upload`（60 s）、`GET {B}/api/jetformer/status/<id>`（15 s） | JSON |
| ZIP のダウンロードリンク | `new URL(result.download_url, bridgeURL())`。ブリッジは `/api/datasets/download/<32hex>` を**ルート相対**で返す | `#dataset-download` と `#collection-download` の `href`（`download` 属性付き） |

### 2.1 サブパスを `bridgeURL()` に入れられない理由（重要）
- `bridgeURL()` はループバック以外のホストを拒否し、origin だけを返す。
- 仮に app.js を書き換えて「origin + `/sim/bridge`」を返すようにしても、次の2か所で失敗する:
  - `$('record-finish')`（app.js:351）は `new URL('/api/datasets/download/..', base)` を実行するが、ルート相対の URL ではベースのパスが無視される。その後の `url.origin!==bridgeURL()` チェックで「保存先URLが不正です。」の例外になる。
  - この例外は、ブリッジ側でセッションが閉じられた**後**、しかも `recording=null` の**前**に起きる。そのため UI は「収集中」のまま固まり、連続試験は `dataset_save_error` で中断する。`collectionLink()`（app.js:428）も正規表現が完全一致で、同じように失敗する。
- したがって、`bridgeURL()` にはパスを含めない。偽のループバック origin を返したまま、**通信の段階で書き換える**。

### 2.2 プロキシに関係するブリッジ側の制約（bridge.py）
- `Host` は `127.0.0.1:<port>` か `localhost:<port>` でなければ 403 になる。`Origin` は無し・`null`・`http://127.0.0.1:<port>`・`http://localhost:<port>` のいずれかでなければ 403 になる。
  - プロキシは Host を上流向けに設定し、ブラウザから届く `Origin`（POST では `http://<jetson>:8080` が付く）を**削除する**必要がある。
- `Content-Length` は必須で、`Transfer-Encoding`（chunked）は 400 で拒否される。`Content-Type` は完全一致でなければならない（`application/json` / `application/zip` / `application/octet-stream`）。
  - プロキシは Content-Length を引き継ぎ、chunked に変換してはならない。
- 本文の上限: JSON 2 MiB（finish は 16 MiB）、ONNX 256 MiB、モデル ZIP 300 MiB。ブリッジは本文をすべてメモリに読み込む。Jetson は 8 GB でスワップがないので、プロキシ側では本文をバッファせずストリーム転送するのが望ましい。

## 3. 推奨1: ブリッジの接続先をプレフィックスへ差し替える（挿入スクリプト）

### 3.1 配置
- JetFormer が index.html を返すときに、`<meta charset>` の直後かつ importmap の前に、次の2つを挿入する。クラシックスクリプトはモジュール（defer 扱い）より先に実行されるので、app.js が動き出す前にラップが完了する。
  - `<script>window.JETFORMER_SIM={...}</script>`
  - `<script src="/sim/static/sim-shim.js"></script>`（JetFormer が所有するファイル）

### 3.2 書き換えの規則
- **書き換えるのは、`http://` で始まる絶対 URL の文字列で**、origin が「現在の `#endpoint` の origin」（読めない場合は `http://127.0.0.1:8766`）と一致し、パスが `/api/` で始まるものだけ。
  - 相対 URL は一切書き換えない。これにより JetFormer 側のコードは `/api/...` や `/sim/...` を相対パスで安全に呼べる。ページを `http://localhost:8080/...` で開いて、app.js:57 が endpoint を `location.origin` にした場合でも混線しない。
  - origin を固定値ではなく「現在の endpoint の値」で判定するので、app.js:57 による上書きにも対応できる。
- 対象は `fetch` と `XMLHttpRequest.prototype.open`。クエリ文字列（`?name=` や `?kind=`）は保持する。
- `<a>` の href は MutationObserver（`attributes:true, attributeFilter:['href']`）で書き換える:
  - `#dataset-download` と `#collection-download` → `/sim/bridge/api/datasets/download/<id>`。同一オリジンになるので `download` 属性が有効になる。
  - `#jetformer-open` のホストがループバックなら `location.origin + '/'` にする。
  - 書き換え後の値はループバックではないので、再書き換えのループは起きない。
- `#endpoint.readOnly = true` にする。`syncControls()` が切り替えるのは `.disabled` だけで、`.readOnly` には触れない。あわせて、ヒント文を「JetFormer経由で接続」などに変えるとよい。
- モジュールより前に `localStorage['course-jetformer-url']='http://127.0.0.1:8080'` を書き込んでおく。
  - ブリッジは Jetson 上で動くため、送信先は自分自身の JetFormer になる。`jetformer_upload.py` はループバックと私設アドレスを許可している。
  - HTML の既定値 `http://192.168.55.1:8080` は l4tbr0（USB）のアドレスで、そのインターフェースが有効なときしか使えない。
- 任意の追加:
  - 読み込み後に `#connect` を自動でクリックする（GET /api/status だけなので無害）。
  - `course-20s-*` の2キーをコースごとに分ける。`Storage.prototype.getItem` / `setItem` / `removeItem` をこの2キーに限ってラップし、末尾に `:<course>` を付ける。分けないと、別のコースで集めた20秒シーンの ID が1つの集約 ZIP に混ざる恐れがある。

### 3.3 コード案と確認結果
次のコードのうち、fetch と XHR の書き換えはテスト版でブラウザ確認済み。テスト版では相対 URL も判定の対象にしていたが、本案では絶対 URL に限定している。

```js
(() => {
  'use strict';
  const cfg = window.JETFORMER_SIM || {};              // {bridge:'/sim/bridge', course:'default', ...}
  const BRIDGE = cfg.bridge || '/sim/bridge';
  const LOOP = new Set(['127.0.0.1', 'localhost', '[::1]']);
  const endpointOrigin = () => { try { return new URL(document.getElementById('endpoint').value).origin; }
                                 catch { return 'http://127.0.0.1:8766'; } };
  function rewrite(raw) {
    if (typeof raw !== 'string' || !/^http:\/\//i.test(raw)) return null;   // 相対URLは触らない
    let u; try { u = new URL(raw); } catch { return null; }
    if (!LOOP.has(u.hostname) || u.origin !== endpointOrigin() || !u.pathname.startsWith('/api/')) return null;
    return location.origin + BRIDGE + u.pathname + u.search;
  }
  const origFetch = window.fetch.bind(window);
  window.fetch = (input, init) => {
    const to = rewrite(typeof input === 'string' ? input : input instanceof URL ? input.href : null);
    return origFetch(to || input, init);
  };
  const open = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function (m, url, ...rest) { return open.call(this, m, rewrite(String(url)) || url, ...rest); };
  try { localStorage.setItem('course-jetformer-url', location.protocol + '//127.0.0.1:' + (cfg.jetformerPort || 8080)); } catch {}
  document.addEventListener('DOMContentLoaded', () => {
    const ep = document.getElementById('endpoint'); if (ep) ep.readOnly = true;
    const fix = a => { const to = rewrite(a.getAttribute('href') || '');
      if (to) a.setAttribute('href', to);
      else if (a.id === 'jetformer-open') { try { if (LOOP.has(new URL(a.href).hostname)) a.setAttribute('href', location.origin + '/'); } catch {} } };
    for (const id of ['dataset-download', 'collection-download', 'jetformer-open']) {
      const a = document.getElementById(id); if (!a) continue; fix(a);
      new MutationObserver(() => fix(a)).observe(a, {attributes: true, attributeFilter: ['href']});
    }
  });
})();
```

**確認結果（Jetson、ヘッドレス Firefox 157、scratchpad のテストサーバー）:**
- `/sim/c/v2/` でページを開くと、17個のモジュールがすべて `/sim/c/v2/...` で要求された。`scene.js` と `world.js` はコースフォルダー（v2 スナップショット）から返り、`document.baseURI` は `http://127.0.0.1:18977/sim/c/v2/` だった。
- `fetch('http://127.0.0.1:8766/api/status')` は、サーバーに `/sim/bridge/api/status` として届き、偽の状態 JSON がそのまま返った。
- 制約: ヘッドレス Firefox では WebGL が使えず、app.js はレンダラーの生成で停止した（「WebGL 2を利用できません」）。そのため、走行・テクスチャ読み込み・ダウンロードリンクの書き換えはブラウザで確認できていない。**実機の Mac ブラウザでの確認が必要**。

## 4. コースの定義と読み込み

### 4.1 scene.js — 見た目・衝突・路面・範囲
- `export function buildCourse(scene)` は、THREE のメッシュ（レール、支柱、マット、暗幕、照明ゲート、LED 矢印ゲート、芝、ショートカット、部屋）をシーンに追加すると同時に、衝突用の要素を集める。戻り値は `{course, room, curtains, animated, landmarks, dimensions, physics}` で、**app.js が使うのは `physics` だけ**。
  - `segments`: レールはカプセル形（`radius .0095`, `height .119`, `kind:'rail'`）、暗幕は `radius .045`、ゲートの脚は `gate-foot`
  - `disks`: 支柱の脚（sim-core 側で 0.16×0.15 の矩形として扱われる）、支柱、ゲートの支柱
  - `surfaces`: carpet μ.8、blue-mat .65、turf .6、ptfe .22、shortcut .7
  - `bounds`: `{minX:-5.425, maxX:5.425, minZ:-3.25, maxZ:3.55}`
- 座標: 表示座標は `x = X - 5.15`, `z = 2.8 - Z`（公式図面の原点は左下の外周支柱）。
- 他のエクスポート（`COURSE`, `MAT_SHORT`, `LANDMARKS`）は app.js では使われていない。

### 4.2 world.js — ゲートと開始位置
- `WORLD_VERSION = 'indoor-course-eval-3-turf-board-removed'`
- `ROUTE`: 33ゲート。`REQUIRED_CROSSINGS`: `[{id:'shortcut', surface_id:'shortcut', minX:1.26, maxX:2.36, z:0, direction:-1}]`
- `makeWorld(physics)` の戻り値は `{...physics, route, requiredCrossings, spawn:{x,z,yaw:0}}`。
- app.js が使うのは `makeWorld` と `WORLD_VERSION` だけ。`WORLD_VERSION` はレポートの `context.world_version` に入り、収集データセットにも記録される。
- app.js は開始ゲート・左右のずれ・角度に合わせて `route` を回転させ、`spawn` を作り直す（`activeWorld`）。開始ゲートの上限は `route.length-1`。

### 4.3 world hash（sim-core.js）
- `normalizedWorld()`（segments, disks, surfaces, bounds, route, spawn, requiredCrossings）を `JSON.stringify` した文字列に対する FNV-1a 32bit で、`report().world_hash` に入る。
- レポートのハッシュは `activeWorld` から計算される。コース本来のハッシュと一致するのは、開始ゲート0・ずれ0で、向きが0に正規化される場合だけ（v2 と v3 は該当する）。
- **node で確認した結果**（scratchpad の `worldhash.mjs`。ユーザーの `sim-core.js` と vendor の three を使い、`qa/load-course-world.mjs` と同じ方法で読み込んだ）:

| コース | WORLD_VERSION | world hash | 線分 / 円 / 路面 / ゲート | テクスチャ |
|---|---|---|---|---|
| Default（`dist/`） | indoor-course-eval-3-turf-board-removed | **0a464384** | 51 / 94 / 5 / 33 | `./assets/` の6枚 |
| v2 スナップショット | indoor-course-eval-2-shortcut | **547d7e80**（manifest と一致） | 52 / 98 / 5 / 33 | 同じ6枚 |

- v2 の `physics-world.json` は、`makeWorld(buildCourse().physics)` の出力と JSON として完全に一致した。v2 と v3 の scene.js の差は1行だけ（`R([9.52,5.56],[9.65,4.62],0)` の白い板）。

### 4.4 テクスチャと assetsReady（注意点）
- scene.js の `textureLoader.load('./assets/'+name)` は、**モジュールの URL ではなくドキュメントの URL** を基準に解決される。読み込むのは carpet.jpg、carpet-normal.jpg、turf.jpg、turf-normal.jpg、blue-pad.jpg、black-mat.jpg の6枚。読み込めない場合は、手続き生成のキャンバステクスチャで代用される。
- `assetsReady` が true になるのは `THREE.DefaultLoadingManager.onLoad` が呼ばれたときだけ（app.js の末尾）。**TextureLoader で1枚も読み込まないコースでは onLoad が呼ばれず、「評価を開始」「1ステップ」「連続試験」が押せないままになる。** アップロードするコースの要件として明記する（`photoTexture()` を1回以上使うこと）。

### 4.5 scene.js / world.js 以外でコースに依存する部分
- 教師:
  - `slow`（`sim.demoAction()`）と `speed` は汎用で、どのコースでも使える。
  - `speed-fast` は speed-controller.js にある v3 専用の `guideRoute` を使うため、他のコースでは失敗するか、おかしな走行になる。
  - `speed-rc` と `speed-optimized` は v3 の形状と完全一致するかを検査し、違えば「RC教師は監査済みv3コース・33ゲート専用です。」などのエラーで拒否する。
  - 「20分・60シーン」の収集計画は RC 教師を使うので、v3 専用。
- index.html の固定文言: 「COURSE / 01」、「10.30 × 5.60 m＋α」、BENCHMARK の説明（黒いシート）、「黒いシート：未通過／通過済み」。
- `setView()` のカメラ位置は、このコースの大きさに合わせた固定値（他のコースでもドラッグで回せば見られる）。
- これらは挿入スクリプトで表示文言を差し替えられる（例: `.view-label` をコース名にする）。機能面の対応は不要で、v3 専用の教師はアプリ自体がエラーで止める。

### 4.6 ユーザーが既に使っているコースパッケージ形式
`qa/course-snapshots/<world_version>/` に次のファイルを置く形式。そのままアップロード用の形式に使える:
- `manifest.json`: `{world_version, reason, files:{scene.js, world.js, physics-world.json の sha256}, world_hash, segments, disks}`
- `scene.js`, `world.js`
- `physics-world.json`: `makeWorld(buildCourse().physics)` の JSON
- 任意: `assets/*.jpg|png`

`qa/verify-learned-lap.mjs` の `loadWorldSnapshot()` は、sha256 と world_hash（sim-core で計算）を検査している。

## 5. 推奨2: ページを読み込むたびにコースを切り替える

### 方式A（推奨）: コースごとにパスを分ける `/sim/c/<course_id>/`
JetFormer 側の経路:
- `GET /sim` → `/sim/` へリダイレクト。`GET /sim/` → コース選択画面、または `/sim/c/default/` へリダイレクト。
- `GET /sim/c/{cid}` → `/sim/c/{cid}/` へリダイレクト（**末尾スラッシュ必須**。§1）
- `GET /sim/c/{cid}/` → `dist/index.html` に設定と挿入スクリプトを加えたもの。ヘッダーにコース切り替えの `<select>` を足してもよい。
- `GET /sim/c/{cid}/scene.js`, `/world.js` → `courses/{cid}/scene.js` と `world.js`（`cid == 'default'` のときは `dist/` から）
- `GET /sim/c/{cid}/assets/{name}` → `courses/{cid}/assets/{name}` があればそれを、なければ `dist/assets/{name}` を返す。
- `GET /sim/c/{cid}/{path}` → `dist/{path}`。パスを安全に結合し、`..`、`\`、NUL を拒否し、`resolve()` 後も dist 内に収まることを確認する。`diagnosis/` は除外してもよい。
- `/sim/bridge/{path}` → `127.0.0.1:8766/{path}` へプロキシ（`/api/*` に限る。§2.2 の Host / Origin / Content-Length の扱いを守る）。
- コース一覧と登録: 例として `GET /sim/courses`、`POST /sim/courses`（ZIP）。JetFormer の `/api/sim/...` に置いてもよい。どちらにしても、シミュレーターの API を JetFormer の `/api` 直下には置かない。
- `cid` は `^[a-z0-9][a-z0-9-]{0,47}$` に制限し、`default` は予約語にする。`.js` は `text/javascript`（または `application/javascript`）で返す。モジュールスクリプトは MIME を厳しく検査する。
- キャッシュは JetFormer の `_NoCacheStatic` と同じく `no-cache` + ETag / Last-Modified にして、更新後に古いモジュールが残らないようにする。

利点:
- ユーザーのファイルを変更しない。
- コース固有のテクスチャを置けて、ない場合は Default にフォールバックできる。
- URL がコースごとに異なるので、古いモジュールのキャッシュと取り違えない。
- 別のコースを別タブで同時に開ける。

欠点:
- `vendor/three.*`（約2 MB）がコースの URL ごとに1回ずつ取得される。LAN 内なので許容できる。
- 気になる場合は、配信する importmap の `three` を共通の `/sim/vendor/three.module.js` に向ける。importmap は1つにまとめ、THREE のインスタンスは1つに保つこと。

### 方式B（確認済みの代替案）: URL は1つで importmap を差し替える
- `/sim/?course=<id>` でページを返すときに、既存の importmap に URL 形式のキーを追加する:
  - `"./scene.js":"./courses/<id>/scene.js"`
  - `"./world.js":"./courses/<id>/world.js"`
- **ヘッドレス Firefox 157 で確認済み:** `{"imports":{"./a.js":"./alt/a.js"}}` で、`main.js` 内の `import ... from './a.js'` が `/sim/alt/a.js` に置き換わった。
- 制約:
  - テクスチャはドキュメントの `./assets/` から読まれるため、コース固有のテクスチャは使えない。
  - 古いブラウザでは importmap を1つしか持てないので、既存のものに統合する必要がある。
  - 再読み込みのときに `?course` を保持しなければならない。
- したがって方式Aのほうが単純。

### 方式C（非推奨）: simulator.html
- 単一ファイル版で、v3 のコースとデータ URI のテクスチャが埋め込まれている。
- `build-local-html.py` は ROOT と DIST が固定で、jpg がちょうど6枚、scene.js 内の `'./assets/'+...` がちょうど2か所であることを前提にしている。コースごとに作り直すにはスクリプトの改変が必要になるので、統合には使わない。

## 6. 整合性チェック（推奨）
- **アップロード時（サーバー側）:**
  - manifest の sha256 とファイルを照合する。
  - `physics-world.json`（データだけ）について、ユーザーの `sim-core.js` を node で実行して world_hash を計算し、manifest と照合する。
  - アップロードされた `scene.js` を Jetson 上で実行してはならない。
  - Jetson には `/usr/bin/node` v24.19.0 がある。
- **表示時（ブラウザ側）:** JetFormer 側のモジュール（例: `/sim/c/<id>/course-check.js`。dist ではなく JetFormer から配信）で `./sim-core.js` を import する。`createSimulation(window.courseSimulator.world()).report().world_hash` を manifest の world_hash と比べ、違えば警告帯を表示する。

## 7. リスクと補足
- **アップロードされた scene.js / world.js は、JetFormer と同じオリジンの JavaScript として実行される**。そのため JetFormer の走行 API などを呼べてしまう。
  - 対策案: `/sim/c/*` に CSP を付ける。`connect-src` をパス付きで `<origin>/sim/bridge/` とコース API だけに絞る。`script-src 'self' 'nonce-…'` とし、挿入する inline スクリプトと **inline の importmap の両方に nonce を付ける**。`img-src 'self' data: blob:` も指定する。
  - 導入する場合は、実機ブラウザで画面が壊れないかの確認が必要。
- iframe に埋め込む場合:
  - キー操作は iframe にフォーカスがあるときだけ効く。
  - フォーカスが外れると走行は一時停止する（app.js の `blur` 処理）。
  - 別タブで開くほうが簡単。
- 収集中や処理中は `beforeunload` で確認が出るため、ページ遷移でコースを切り替えても安全。
- 描画（WebGL）は Mac のブラウザが行い、Jetson が担うのはブリッジ（onnxruntime の CPU 推論）だけなので、GPU 使用中という条件に反しない。外部モデルでは毎フレーム約100 KB の観測データを送って応答を待つ（収集中はさらに frame の POST が加わる）。LAN とプロキシを経由しても動くが、走行速度は応答時間で決まる。
- **JetFormer のモデルをシミュレーターに読み込む方法（アプリ無改変）:** 挿入スクリプトで次を行えば、アプリの検証・有効化の手順をそのまま使える。
  1. 同一オリジンの `/api/models/<run>/bundle.zip` を fetch する。
  2. `new File([blob], name)` を作り、`DataTransfer` 経由で `#model-bundle-file.files` に設定する。
  3. `#load-bundle.click()` を実行する。
  - DataTransfer のコンストラクターは Chrome / Firefox / Safari 14.1 以降で使える。
- 画面の案内文（start.command / start.bat を起動、など）は単体配布向けのまま。必要なら挿入スクリプトで差し替える。
