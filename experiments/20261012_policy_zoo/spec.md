# 方策の比較(policy zoo)の実装仕様

作成 2026-10-12 ・ 対象 `/home/jetson/jetformer`(main、7239fea)・ この文書は仕様だけ(リポジトリのファイルは変えていない。GPU は使っていない)

依頼(2026-10-12): 「ACT 型・TCP/TransFuser++ 型・本家 TransFuser に近い多段融合・DINOv2 の画像特徴 の Policy を追加して比較し、
どのモデルがベストモデルかを評価する」。JetFormer v0 は `jetformer_v0` として残し、古い学習結果の既定とする。どの方策も
CLI(`tools/train.py`)と管理画面の学習タブで学習でき、ONNX → TensorRT FP16 に書き出せ、走行タブから同じ安全の仕組みで走れること。

元にした資料(同じフォルダー):

| 資料 | この仕様での扱い |
|---|---|
| [`interfaces.md`](interfaces.md) | 差し込み口(file:line)と後方互換の規則。行番号は 2b9774e のもの。その後の3コミットで `server.py`・`app.js` の行が約 50 行ずれた(この文書では 7239fea の行で書く) |
| [`design_research.md`](design_research.md) | 4つの方策の元の研究と置き換え方、TCP のラベル・較正・制御器、比べ方の下書き |
| [`design_bench.md`](design_bench.md)・`bench/` | 試作の ONNX → TensorRT FP16 の **実測**(速さ・メモリ・学習の1ステップ・書き出しの落とし穴) |

3つの資料で食い違う所は §0.3 で決める。速さの数字はすべて `bench/` の実測(乱数の重み。速さは重みに依存しない)、精度の見込みは推定。

---

## 0. 要約

### 0.1 決めたこと

1. **登録簿** `jetformer/policies/`(torch を読まない `__init__.py` + torch の `runtime.py`)。`build_policy(name, num_modes, cfg)`、
   `model_config.json` の `policy`・`policy_cfg`(無ければ `jetformer_v0` と v0 の凍結した既定値)。`jetformer_v0` は
   `jetformer/model.py` の `JetFormer` を **包まずに** 返すので、今の `best.pth`・`last.pth`・エンジンはそのまま使える。
2. **5つの方策**(大きさ・速さは実測。`run_numpy` は走行ループと同じ呼び方の中央値):

   | `policy` | 中身(既定) | パラメータ | TRT FP16 エンジンだけ | `run_numpy` | 学習 30 エポック |
   |---|---|---|---|---|---|
   | `jetformer_v0` | 今のまま(モデル B と同じ形) | 14.1M | 1.47 ms | 2.33 ms | 17 分 |
   | `act` | v0 と同じトークン → Encoder 4層 → 行動クエリ 10 の Decoder 2層 | 15.6M | 1.64 ms | 2.48 ms | 18 分 |
   | `tcp` | v0 と同じトークン → Encoder 4層 → クエリ 21(通過点 10・速度 1・操作 10)の Decoder 2層 + numpy の制御器 | 15.6M | 1.69 ms | 2.50 ms(+ 制御器 約0.2 ms) | 18 分 |
   | `msfusion` | ResNet-18 の4段と ToF の4段の CNN を各段で融合 → 最終段のトークンを v0 型で融合 | 18.9M | 2.18 ms | 3.12 ms | 26 分 |
   | `dinov2` | DINOv2 ViT-S/14(凍結)→ 64 + CLS トークン → v0 型の融合 4層 | 24.5M(学習 2.9M) | 3.88 ms | 5.06 ms | 16 分 |

3. **学習**: どれもモデル B と同じ条件(clean_driving_20261011・今の操作値なし・衝突 2.5G の除外・stride 3・30 エポック・lr 3e-4 cosine・
   バッチ 64・AdamW wd 1e-4・AMP FP16・色の揺らぎ)、**乱数の種 0・1 の2本ずつ**(`jetformer_v0` も同じコードで2本)。
   GPU の時間は本体 約4.1 時間、余り 約1.9 時間で条件つきの3本目の種(§4)。
4. **評価**: 主指標は **保留した人のきれいな走行の L1 [µs]**(clean の検証 11 シーン、全フレーム、2.5G の前後を除く、4,670 フレーム。
   モデル B は 49.14 µs)。種の平均と範囲で比べ、ばらつきの幅 δ = max(10 µs, v0 の3本(B・種0・種1)の範囲)を超えないと差と言わない。
   副指標は R1・DE の保留の通過(通過ごとのブートストラップ)、直線の前進の強さ、前進の偏り、ずれへの強さ、速さ。
   判定の規則は §5.4 に固定し、学習を始める前に `zoo_rules.json` に書き出して変えない。
5. **走行**: どの方策も出力 `actions [1,10,2]` を持つので、先読み・ゲイン・ToF ガード(記録だけ)・補助/リカバリー・ラップ・CH3 の交代・
   センサーの途切れで中立、はそのまま効く。`tcp` だけ、推論の直後・先読みの前に numpy の制御器(`jetformer/policies/controller.py`)を通せる。
   **既定の `tcp_mode` は 0(操作の枝だけ = 比較で評価した出力そのもの)**。速度の上限(`speed_cap`)は 0 = なし。
6. **正直な見込み**: どの方策もモデル B より L1 で 10 µs 以上良くなる見込みは低い(`design_research.md` §7.4)。
   「机上ではモデル B より良いと言えない」も正当な結論として先に書いておく(§5.4 の 4)。

### 0.2 作業の順番と持ち主(並行して進めるための約束)

```
段階1  F  枠組み(登録簿・runtime・train.py・データセット・エンジン・training.py・後処理と重ね合わせの枠)   ← 最初に1人で
            │ 終わりの確認: 既存のテストが全部通る + tests/test_policies.py・test_policy_post.py
            ▼
段階2  P-act │ P-tcp │ P-msf │ P-dino   (4人が並行。自分の方策のモジュールとテストだけ)
       I  走行・管理画面・記録・書き出し(並行可。方策のモジュールには触らない)
       E  比較の道具 tools/policy_zoo.py と既存の評価の道具の置き換え(並行可)
            ▼
段階3  M  (主の作業)レビュー → 全テスト → CPU での各方策の試し学習(2ステップ)→ git commit
            ▼
段階4  M  GPU の比較(tools/policy_zoo.py、1プロセスずつ・続きから再開できる)→ 評価 → 報告(日本語)
```

| 持ち主 | 作る・変えるファイル(ほかの人は触らない) |
|---|---|
| **F**(枠組み) | 新: `jetformer/policies/__init__.py`・`runtime.py`・`blocks.py`・`jetformer_v0.py`・`post.py`・`ensemble.py`、`tests/test_policies.py`・`tests/test_policy_post.py`。変更: `tools/train.py`・`jetformer/train_data.py`・`tools/build_engine.py`・`jetformer/trt_runtime.py`・`jetformer/training.py` |
| **P-act** | 新: `jetformer/policies/act.py`、`tests/test_policy_act.py` |
| **P-tcp** | 新: `jetformer/policies/tcp.py`・`tcp_labels.py`(numpy)・`controller.py`(numpy)、`tools/make_tcp_labels.py`、`tests/test_policy_tcp.py`・`tests/test_policy_tcp_controller.py` |
| **P-msf** | 新: `jetformer/policies/msfusion.py`、`tests/test_policy_msfusion.py` |
| **P-dino** | 新: `jetformer/policies/dinov2.py`・`vit.py`、`tests/test_policy_dinov2.py` |
| **I**(結合) | 変更: `jetformer/drive.py`・`jetformer/recorder.py`・`jetformer/web/server.py`・`static/app.js`・`static/index.html`・`static/style.css`・`jetformer/model_export.py`・`README.md`、`tests/test_drive.py`(末尾に足す)・`tests/test_model_export.py`(末尾に足す)。新: `tests/test_policy_ui.py` |
| **E**(評価) | 新: `tools/policy_zoo.py`、`tests/test_policy_zoo.py`。変更: `tools/compare_models.py`・`tools/evaluate.py`・`tools/rl_evaluate.py`(モデルの読み込みだけ)・`tools/rl_policies.py`(`GPU_PATTERNS` と読み込み)・`tools/offline_rl.py`(`embed` の確認)・`tools/benchmark_model.py` |
| **M**(主) | レビュー・commit・GPU の実行・報告。`experiments/` とこのフォルダーの `results/` |

約束:
- **ほかの持ち主のファイルは変えない**。必要なら最後の報告の `shared_changes_needed` に書く(M が決める)。
- 登録簿の5つの項目(`__init__.py` の `register(...)`)は **F が §3 の表のとおりに全部書く**。P は自分の項目の既定値を変えたくなったら M に報告する。
- P・I・E は F の API(§2)だけを使う。F の API で足りなければ、その場で回避策を作らず M に報告する。
- I と E は、`tcp` の制御器を §3.2.6 の約束(`TCPPost` の形)だけで呼ぶ。P-tcp の完成前は、テストで偽の後処理を使う(§2.8)。
- **commit はしない**(M がレビューしてから)。GPU は使わない(段階4の M だけ)。CPU の試しでも `CUDA_VISIBLE_DEVICES=` を付ける。

各持ち主の終わりの確認:

| 持ち主 | 終わりの確認 |
|---|---|
| F | §7 の自分のテスト + 既存の `tests/` が全部通る。`tools/train.py` を今の引数で動かしたコマンド・`model_config.json` の今の欄・`--resume` の引数の比較が変わらない。モデル B を `runtime.load_run` で読んだ予測(CPU、数フレーム)が `JetFormer` + `load_state_dict` と同じ |
| P-* | 自分のテスト + §7 の最後の CPU の試し学習(2ステップ)で `jetformer.onnx` と `model_config.json` ができる。§3 の大きさの範囲 |
| I | 自分のテスト + `test_drive.py`・`test_model_export.py`。モデル B(古い設定)が走行の一覧に今と同じに出て、`drivable = True` |
| E | 自分のテスト + `test_rl_evaluate.py`・`test_offline_rl.py`。`policy_zoo.py plan --dry-run` が CPU で動き、規則を表示する(本物の `zoo_rules.json` は M が段階4の最初に書く。学習・予測はしない) |

### 0.3 資料どうしの食い違いの決着

| 項目 | 資料 | この仕様 | 理由 |
|---|---|---|---|
| `tcp` の走行の既定 | research: hybrid | **0 = control(操作の枝だけ)** | 比較で評価するのは操作の枝(§5.2)。走行の既定は評価した出力と同じにする。hybrid・速度の上限は走行タブで選ぶ(別の A/B/A、§5.7) |
| TCP のラベルの置き場所 | research: `data/datasets/<ds>/derived/` | **`data/derived/<ds>/`** | データセットのフォルダーの中のファイルは全部、管理画面のダウンロード(tar)・エクスポート・Hugging Face に入る(`dataset.iter_dataset_tar` は `rglob('*')`) |
| 制御器のステア | bench: 10 行とも同じ値・直線の較正(正解を入れて 62 µs) | **research: ステップごとの pure pursuit + 速度帯ごとの表**(34〜40 µs) | 正解を入れたときの下限が小さい方 |
| pure pursuit の `T_la`・`L_min` | research: 予測した通過点で検証の L1 が最小になる値 | **先に固定: 0.25 s・0.3 m** | 検証 11 シーンで調整しない(資料自身の規則)。学習のシーンの正解での格子(0.15 s 35 µs、0.25 s 40 µs)と、通過点の 5 cm の横ずれの増幅(0.15 s で約180 µs → 0.25 s で約65 µs)の折り合い |
| 積分の項(PI の I) | 両方: 既定は使わない | **使わない** | 開ループの評価で飽和し、走行と評価が食い違う |
| `msfusion` の空きメモリ | research 4000 MB | **3500 MB** | 実測のピーク 1.8 GB |
| `act` の CVAE の符号化器 | research 2層・[CLS]+ジャンル+行動、bench 4層・[CLS]+state+行動 | **2層・[CLS]+ジャンル+行動** | state は今の操作値なしで eRPM しか入らない。CVAE は比較の本体には入れない(参考) |
| ジャンルの層の名前 | bench の試作は `tokens.mode_embed` | **一番上の `mode_embed`・`heads`** | `--init-mode-copy` と、v0 からの部分的な初期値のため |
| DINOv2 の版 | research: 通常版 / レジスタ版 | **通常版 `vits14` だけ** | `data/pretrained/` にあるのは通常版だけ |
| `msfusion` の `aux_depth` | research: 選択肢 | **この作業では作らない** | GPU の予算。将来の候補として §9 に書く |
| `hints`(学習の速さ) | research の推定 | **bench の実測**(§3.0) | — |

### 0.4 段階3(結合)で変えたこと

実装者の `shared_changes_needed` のうち、筋の通るものを結合のときに入れた(2026-10-12)。

| 所 | 変更 | 理由 |
|---|---|---|
| §3.2.3 | スロットルの係数の例を、この規則(前進のフレームだけ)の結果 1490.7・−72.1・−13.6 に直した。表の端の区切りの扱いを書いた | 例の値は research の全フレームの当てはめで、規則の文と食い違っていた |
| §3.2.4 | pure pursuit の向きを「区間(弦)の向き」→「頂点の向きの補間」 | 正解の通過点でのステアの誤差 56 / 52 µs(弦 70 / 65 µs、k3 で 88 µs) |
| §3.2.4 | `branch = 'fallback'`: 道が短い・通過点が数値でない(trajectory のステアだけ操作の枝)、目標速度・今の速さが無い(mode 1〜3 でも操作の枝 + 上限) | 例外を出さずに走り続けるため(§3.2.6 の約束) |
| §3.3 | ONNX のノード数を直した。`scales` の選択肢に `[1, 2, 3]` を足した(登録簿) | 既定の実測は 1,710。4 ms を超えたときの代わりを管理画面・CLI から選べるように |
| README | DINOv2 の重みの取得の手順 | 登録簿・エラーの文が「取得の手順は README」と案内している |
| §3.4 | エンジンの精度の代わりの順(bf16 → fp32)に注記: BF16 は FP16 より精度が低く、差のゲートの代わりにならない | 段階3の GPU の試し(`integration_smoke.md` §5) |

---

## 1. 用語と、変えないもの

- **方策(policy)** = モデルの構造。`tools/rl_policies.py` の「方策」(オフライン RL のデータの使い方)とは別。管理画面の表示は「モデルの種類」。
- **基本の入力** = `image [1,3,224,224]`・`tof [1,2,42,54]`・`imu [1,10,6]`・`state [1,3]`・`mode [1,num_modes]`(float32、バッチ1・固定の形)。
  この作業の方策は全部これだけを使う(フレームの履歴は使わない)。
- **出力の約束** = `actions [1,10,2]`((µs − 1500)/500、33 ms 刻みの 1〜10 フレーム先の [ステア, スロットル])。どの方策も必ず出す。
- 変えないもの: ファイル名 `jetformer.onnx`・`jetformer_fp16.engine`・`best.pth`・`last.pth`・`model_config.json`、
  `PARAM_LIMITS` の今の項目、`apply_gains`・`predicted_now`・`vehicle_state`・ToF ガード・`assist`・`LapCounter`、`DRIVE_CLIENT_VERSION = 6`、
  `tools/train.py` の既定の動き(引数を足さなければ今と同じ学習・同じファイル)、`data/rl/ai_mix_20261011/eval/` の `heldout.json`・`evaluation.json`。

---

## 2. 枠組み(持ち主 F)

### 2.1 `jetformer/policies/__init__.py`(torch を読まない)

`drive.py`・`web/server.py`・`training.py`・`model_export.py` から読むので、**import の中で torch・torchvision・onnx を読まない**(テストで確かめる)。

```python
"""方策(モデルの構造)の登録簿。drive.py・web/server.py・training.py も読むので torch を import しない。
構築(build)のときだけ、方策のモジュール(jetformer.policies.<名前>)を遅延 import する"""

CONTRACT = 1                       # 入出力の約束の版(model_config.json の 'contract')
DEFAULT = 'jetformer_v0'           # 'policy' が無い学習結果 = 今までのもの
FUTURE_STEPS = 10
BASE_INPUTS = ('image', 'tof', 'imu', 'state', 'mode')
BASE_SHAPES = {'image': [1, 3, 224, 224], 'tof': [1, 2, 42, 54], 'imu': [1, 10, 6], 'state': [1, 3]}   # mode は [1, num_modes]
ROOT = Path(__file__).resolve().parents[2]          # リポジトリの一番上(相対パスの基準)


class PolicySpec(object):
    def __init__(self, name, title, module, slug, description, status='experimental', defaults=None, options=(),
                 hints=None, genre_keys=None, outputs=('actions',), postprocess=None, requires=(),
                 requires_pretrained=False, train_only_keys=()):
        ...
    def resolve(self, cfg=None, ui=False):   # → 既定値 + 指定(dict)。§2.1.1
    def build(self, num_modes, cfg=None, pretrained=False):   # importlib で self.module を読み、module.build(num_modes, pretrained, **resolved)
    def module_obj(self):                    # importlib.import_module(self.module)
    def ready(self):                         # → (True, '') / (False, 理由)。モジュールがあるか(importlib.util.find_spec。実行しない)と
                                             #   requires のファイルがあるか(存在だけ。速い)
    def describe(self):                      # GET /api/policies 用(name, title, slug, description, status, defaults, options,
                                             #  hints, outputs, postprocess, requires_pretrained, ready, why_not)

REGISTRY = {}
def register(spec): ...
def get(name):          # 無ければ KeyError('知らないモデルの種類です: ...')
def list_specs():       # [describe()] 登録の順
def title_of(name):     # 登録簿に無ければ名前のまま(走行・一覧は登録簿に無い方策も表示できる)
def from_config(cfg):   # model_config(dict)→ (spec, resolved_cfg)。'policy' 無し → jetformer_v0 と {'dim':256,'layers':4,'heads':8}
def inputs_of(cfg):     # cfg['inputs'](ONNX から読み直した名前)。無ければ BASE_INPUTS
def outputs_of(cfg):    # cfg['outputs']。無ければ ['actions']
def check_io(inputs, outputs, num_modes):
    """inputs/outputs = {名前: 形(list)}。→ (True, '') / (False, 理由)。
    入力の名前がすべて BASE_INPUTS にあり、形が BASE_SHAPES(mode は [1,num_modes])と同じ。
    出力に actions [1,10,2] がある。すべての形が正の整数(-1・None なし)"""
def drivable(cfg):      # → (bool, 理由)。inputs_of が BASE_INPUTS の中、outputs_of に 'actions'。登録簿は見ない
def build_policy(name, num_modes, cfg=None, pretrained=False):
    return get(name).build(num_modes, cfg, pretrained)
```

`check_io` を torch の無いここに置くのは、`build_engine.py`(TensorRT だけ)と `drive.py` の両方が使うため。

#### 2.1.1 `resolve(cfg, ui=False)` の規則

1. `cfg` が None・`{}` → `defaults` の写し。
2. `defaults` に無いキー → `ValueError('知らない設定です: <キー>(使えるもの: ...)')`。
3. 値は既定値の型に直す(int・float・bool・str・list)。直せなければ `ValueError`。
4. `options` にあるキーは、`choices` に入っているか・`min`〜`max` の中か確かめる(外なら `ValueError`)。
5. `ui=True`(管理画面から)のときは、`options` に無いキーを変えたら `ValueError`(CLI の `--policy-cfg` はすべてのキーを変えられる)。
6. 戻り値は **全部のキー** を持つ dict(`model_config.json` の `policy_cfg` にはこれを書く)。

`train_only_keys` は「構造を変えない学習だけの設定」(例 `dinov2` の `finetune_last`・`lr_mult`)。`--init` の一致の確認(§2.6)で比べない。

#### 2.1.2 登録する5つ(F が書く。値は §3)

| name | title(画面) | slug | status | outputs | postprocess | genre_keys の heads(行/ジャンル) | requires |
|---|---|---|---|---|---|---|---|
| `jetformer_v0` | JetFormer v0(今の標準) | `v0` | stable | actions | — | `heads: 20` | — |
| `act` | ACT 型(行動クエリの Decoder) | `act` | experimental | actions | — | `heads: 2` | — |
| `tcp` | TCP / TransFuser++ 型(通過点 + 目標速度) | `tcp` | experimental | actions, waypoints, speed_logits | `tcp_controller` | `heads: 2, heads_wp: 2, heads_v: 10` | — |
| `msfusion` | 多段融合(TransFuser 型) | `msf` | experimental | actions | — | `heads: 20` | — |
| `dinov2` | DINOv2 の画像特徴(ViT-S/14 凍結) | `dino` | experimental | actions | — | `heads: 20` | `data/pretrained/dinov2_vits14_pretrain.pth` |

どれも `genre_keys = {'embed': 'mode_embed.weight', 'heads': {...}}`。`dinov2` だけ `requires_pretrained=True`(`--no-pretrained` では学習しない)。
`hints` の既定は `{'train_rate': 200, 'val_rate': 650, 'min_free_mb': 3500, 'amp': True, 'engine_precision': 'fp16', 'infer_ms': 2.3, 'params_m': 14.1}`
で、方策ごとの値は §3.0 の表(`params_m` は §3.0 の実測の表のパラメータ数)。

### 2.2 方策のモジュールの約束

| 項目 | 必須 | 内容 |
|---|---|---|
| `build(num_modes, pretrained=False, **cfg)`(モジュールの関数) | 必須 | `nn.Module` を返す。`pretrained=False` のときは **ネット・大きな重みのファイルを読まない**(`load_run`・評価の道具はこれで作ってから `best.pth` を読む) |
| `forward(image, tof, imu, state, mode)` | 必須 | **キーワードで呼ばれる**(`runtime.forward`)。戻り値は `Tensor [B,10,2]` か `{'actions': [B,10,2], ...}` |
| `EXPORT_OUTPUTS`(クラスの属性) | 任意 | 書き出す出力の名前の並び。既定 `('actions',)`。先頭は必ず `actions`。どれも固定の形 |
| `USES_TARGET`(属性) | 任意 | True なら学習のときだけ `forward(..., target=[B,10,2])` で呼ぶ(`act` の CVAE) |
| `policy_loss(out, target, extras)`(メソッド) | 任意 | サンプルごとの損失 `[B]`(float32)。無ければ L1。`extras` は追加の正解の dict(無ければ None)。内訳を `self.loss_parts = {'kl': スカラー, ...}` に置くと学習の記録に残る |
| `param_groups(lr)`(メソッド) | 任意 | AdamW に渡す群。無ければ `requires_grad` の全パラメータの1群 |
| `train(mode)` の上書き | 任意 | 凍結した部分を eval に保つ(`dinov2`) |
| `embed(...)` | 任意 | `tools/offline_rl.py features` 用。無い方策ではその道具は止まる(E) |
| `prepare(data_root, cfg)`(モジュールの関数) | 任意 | 学習の前に CPU で1回。何度呼んでもよい(済んでいれば何もしない)。足りなければ `SystemExit(理由)`(`tcp` のラベル、`dinov2` の重みの確認) |
| `extra_targets(data_root, cfg)`(モジュールの関数) | 任意 | 追加の正解 `{シーン名: {キー: np.ndarray [シーンのフレーム数, ...]}}`(numpy)。データセットが最後の dict で返す(§2.6) |
| `finalize(run_dir, cfg, data_root)`(モジュールの関数) | 任意 | 書き出しの後に CPU で。`model_config.json` に足す dict を返す(`tcp` の `postprocess`) |

守ること: import のときに torch の設定(`torch.backends.*`)を変えない、ネットにつながない、大きなファイルを読まない。
state_dict のジャンルの層は一番上の `mode_embed.weight`・`heads.*`(`genre_keys`)。ONNX に出せない演算(§8)を使わない。

### 2.3 `jetformer/policies/runtime.py`(torch)

```python
def as_outputs(y):                                   # Tensor → {'actions': y}。dict は 'actions' が無ければ ValueError
def forward(model, inputs, target=None, train=False):    # inputs: {名前: Tensor}。キーワードで呼ぶ → dict
                                                         # train かつ getattr(model, 'USES_TARGET', False) なら target= を渡す
def l1_per_sample(actions, target):                  # (actions.float() - target).abs().mean(dim=(1, 2)) → [B]
def sample_loss(model, out, target, extras):         # model.policy_loss があればそれ、無ければ l1_per_sample
def param_groups(model, lr):                         # model.param_groups(lr) があればそれ、無ければ
                                                     # [p for p in model.parameters() if p.requires_grad](v0 では今の AdamW と同じ並び)
def example_inputs(num_modes, batch=1, seed=0, device='cpu'):   # {5つ} 書き出し・確認用(mode は one-hot、ジャンル 5)
def check_forward(model, num_modes, device='cpu', batch=2):     # 学習の前に1回。actions [batch,10,2]・有限・EXPORT_OUTPUTS が全部ある。
                                                                # 違えば RuntimeError(30 分の学習の後で書き出しに失敗しないように)
class ExportWrapper(nn.Module):                      # forward(image, tof, imu, state, mode) → EXPORT_OUTPUTS の順のタプル
def export_onnx(model, path, num_modes):
    """model.float().eval().cpu()、torch.backends.mha.set_fastpath_enabled(False)、opset 17、dynamo=False、バッチ1・固定の形、
    input_names = BASE_INPUTS、output_names = EXPORT_OUTPUTS。path + '.tmp' に書いてから os.replace。
    書いた ONNX を読み直して(onnx.load(load_external_data=False))実際の入出力を返す:
      {'inputs': [...], 'input_shapes': {名前: 形}, 'outputs': [...], 'output_shapes': {名前: 形}}
    check_io が通らなければ RuntimeError(使っていない入力はグラフから消える: interfaces.md §4。消えるのは良い)"""
def read_onnx_io(path):                              # 上の dict だけ(書き出さない)
def load_state(path):                                # best.pth(state_dict)・last.pth({'model': ..., 'optimizer': ...})のどちらも
def load_run(run_dir, ckpt='best', device='cpu'):
    """model_config.json → policies.from_config → build(pretrained=False)→ <ckpt>.pth を strict=True で読む → eval。
    戻り値 (model, cfg, spec, resolved_cfg)"""
def expand_modes(state, model_state, genre_keys, copy_from=None):
    """train.expand_mode_state の一般形。genre_keys = {'embed': 'mode_embed.weight', 'heads': {層の名前: 1ジャンルの行の数}}。
    元のジャンルを写し、増えたジャンルを copy_from {新: 元} で初期化。heads の重みは [ジャンル×行, 入力] を view(ジャンル, 行, -1)"""
@torch.no_grad()
def predict(model, inputs):                          # FP32(autocast なし)・eval → {名前: np.ndarray float32}(actions は正規化のまま)
def count_params(model):                             # (全体, 学習するもの)
```

### 2.4 `jetformer/policies/blocks.py`(torch。新しい方策が共有する部品)

`act`・`tcp` は v0 と **同じトークン化** を使い、state_dict のキーも v0 と同じにする(将来の部分的な初期値のため)。
そのため部品は「子のモジュール」ではなく、**呼んだモジュールに直接、v0 と同じ名前の層を足す関数** にする。

```python
def select_mode(all_modes, mode):          # [B,M,...] と one-hot [B,M] → [B,...](掛け算と和だけ。TensorRT で特別な処理なし)
def mlp2(cin, dim):                        # Linear → ReLU → Linear(v0 の imu_mlp・state_mlp と同じ)
def encoder(dim, heads, ffn, layers, dropout=0.1):   # 前置き LayerNorm・batch_first、enable_nested_tensor=False
def decoder(dim, heads, ffn, layers, dropout=0.1):   # nn.TransformerDecoder(norm_first=True, batch_first=True)
def add_sensor_tokens(module, dim, num_modes, pretrained):
    """module に v0 と同じ名前・同じ初期化で image_backbone(ResNet-18 の children()[:-2])・image_proj(Conv 1x1 512→dim)・
    tof_encoder(model.ToFEncoder)・imu_mlp・state_mlp・mode_embed(Linear(num_modes, dim, bias=False)、normal std 0.02)を足す"""
def sensor_tokens(module, image, tof, imu, state, mode):
    """v0 の forward の 88〜92 行と同じ計算 → [B,94,dim](画像 49・ToF 42・IMU 1・操作値 1・ジャンル 1)"""
```

### 2.5 `model_config.json`・`result.json`・`run.json` に足す欄

古い学習結果を読むときは、すべて右の既定値で読む(`test_rl_review.py:100` の `{"num_modes": 6, "trained_modes": [4, 5]}` だけでも動くこと)。

| ファイル | 欄 | 内容 | 無いとき |
|---|---|---|---|
| model_config | `policy` | 登録簿の名前 | `jetformer_v0` |
| | `policy_cfg` | `resolve` の結果(全部のキー) | `{'dim': 256, 'layers': 4, 'heads': 8}` |
| | `contract` | 1 | 1 |
| | `inputs` | **書き出した ONNX から読み直した** 入力の名前(今もある欄。決め方だけ変える) | 基本の5つ |
| | `input_shapes`・`outputs`・`output_shapes` | ONNX から | 基本の形・`['actions']`・`{'actions': [1,10,2]}` |
| | `engine` | `{'precision': 'fp16', 'workspace_gb': 1.0}`(`hints.engine_precision`) | 同じ |
| | `postprocess` | `tcp` だけ(§3.2.5) | 無し |
| | `params_m` | `{'total': 15.6, 'trainable': 15.6}` | — |
| result | `policy`・`policy_cfg`・`seed`・`params_m`・`peak_mem_mb` | `peak_mem_mb` は `torch.cuda.max_memory_allocated`(CPU なら None) | — |
| | `history[].train_loss`・`history[].train_parts` | `policy_loss` を持つ方策だけ(v0 の `history` は今と同じ) | — |
| run.json(管理画面・比較の道具が書く) | `policy`・`policy_cfg`・`impact_g`・`seed` | | `jetformer_v0` ほか |

### 2.6 学習(`tools/train.py`・`jetformer/train_data.py`)

**引数**: `--policy NAME`(既定 `jetformer_v0`、選択肢は登録簿)、`--policy-cfg JSON`(既定 None。`'{}'` は None と同じ)。
`POLICY_ARGS = {'policy': 'jetformer_v0', 'policy_cfg': None}` を `RL_ARGS` と同じ方式で `same_args` に入れる
(**既定値なら記録しない** → 前からの `last.pth` で `--resume` できる。`policy_cfg` は `json.dumps(sort_keys=True)` で比べる)。

**流れ**(変更点だけ。行は 7239fea):

1. `spec = policies.get(args.policy)`、`cfg = spec.resolve(json.loads(args.policy_cfg or '{}'))`。
   `spec.requires_pretrained and args.no_pretrained` → `SystemExit`。`ok, why = spec.ready()` が False → `SystemExit(why)`。
2. `mod = spec.module_obj()`。`prepare` があれば `mod.prepare(args.data, cfg)`(CPU)。
3. データセット(235-239): `extra_targets = mod.extra_targets(args.data, cfg)` があれば学習・検証の両方に渡す。
4. 構築(271): `model = policies.build_policy(args.policy, num_modes, cfg, pretrained=not args.no_pretrained).to(device)`、
   `runtime.check_forward(model, num_modes, device)`。
5. 最適化(272): `torch.optim.AdamW(runtime.param_groups(model, args.lr), lr=args.lr, weight_decay=1e-4)`。cosine はそのまま。
6. AMP: `amp = spec.hints['amp']` を `run_epoch` に渡す。
7. `--init`(296-301): `Path(args.init).parent / 'model_config.json'` があれば読み、`policy` と、`train_only_keys` を除いた `policy_cfg` が
   今と同じでなければ `SystemExit`(関数 `check_init_compat(src_cfg, policy, cfg, spec)` にしてテストする)。重みのジャンルの拡張は
   `runtime.expand_modes(..., spec.genre_keys, ...)`。`train.expand_mode_state(state, model_state, copy_from)` は **名前と動き(同じ例外・同じ文言)を残し**、
   中で v0 の `genre_keys`(`{'embed': 'mode_embed.weight', 'heads': {'heads': 20}}`)を使って `expand_modes` を呼ぶ(`test_rl_policy_training.py` が使う)。`genre_keys` が無い方策の `--init-mode-copy` は `SystemExit`。
8. 書き出し(353-366): `io = runtime.export_onnx(model, out/'jetformer.onnx', num_modes)` → `model_config.json` に今の欄 + §2.5 の欄。
   `inputs` は `io['inputs']`。`finalize` があれば `model_config.update(mod.finalize(args.out, cfg, args.data))`。
   `model_config.json` は `.tmp` に書いてから置き換える。
9. `result.json` に §2.5 の欄。`seed` は `args.seed`(既存の `rl_options` もそのまま)。

**`run_epoch`**(62-103。引数と戻り値は今と同じ):

```python
image, tof, imu, state, mode, target = (t.to(device, non_blocking=True) for t in batch[:6])
extras = ({k: v.to(device, non_blocking=True) for k, v in batch[-1].items()}
          if isinstance(batch[-1], dict) else None)                       # 追加の正解(最後の dict)
weight = batch[6] if len(batch) > 6 and not isinstance(batch[6], dict) else None
inputs = dict(image=image, tof=tof, imu=imu, state=state, mode=mode)
with torch.set_grad_enabled(train), torch.autocast(..., enabled=amp and device.type == 'cuda'):
    out = runtime.forward(model, inputs, target=target, train=train)
    l1 = runtime.l1_per_sample(out['actions'], target)                    # 報告する誤差: どの方策でも同じ L1
    per = runtime.sample_loss(model, out, target, extras) if train else l1   # 学習の損失(v0 は l1 と同じ)
    loss = per.mean() if weight is None else (per * weight).mean() / weight_scale
# 集計: total += l1.detach().sum()(今の loss.item()*n と同じ値。重みつきの集計・ジャンル別は今と同じく l1 で)
# train のとき: stats['loss'] = 学習の損失の平均、stats['parts'] = loss_parts の平均(あれば)
```

v0 では `per == l1` なので、学習・報告の数字は今と同じ(浮動小数の足す順の差 1e-7 程度は許す)。

**データセット**(`JetFormerDataset.__init__` に `extra_targets=None` を足す):
- 渡されたら、各シーンの配列を `a['_y_<キー>']` として持つ。split のシーンが `extra_targets` に無ければ `KeyError('<シーン> の追加の正解がありません')`。
- `__getitem__` の最後(重みの後)に `{'y_<キー>': torch.from_numpy(配列[i])}` を足す。**無ければ今と同じ6個(+重み)**。
  既定の DataLoader の collate は dict のテンソルをまとめるので、そのまま使える。

### 2.7 ONNX とエンジン(`tools/build_engine.py`・`jetformer/trt_runtime.py`)

- `trt_runtime.build_engine(onnx_path, engine_path, fp16=True, workspace_gb=1.0, bf16=False)`(`bf16` を足すだけ。既定の動きは同じ)。
  `TRTRunner` に `self.shapes = {名前: tuple(形)}`(入力と出力)を足す。ほかは変えない。
- `build_engine.py <onnx> <engine>`(引数は今と同じ):
  1. 同じフォルダーの `model_config.json` があれば `engine.precision`(`fp16` 既定・`bf16`・`fp32`)と `workspace_gb` を読む。無ければ FP16・1GB。
  2. `<engine>.tmp` に作る → TensorRT だけで読み直して入出力の名前と形を取り、`policies.check_io(...)`(`num_modes` は model_config、
     無ければ ONNX の `mode` の形)→ 通れば `os.replace(tmp, engine)`、通らなければ tmp を消して理由を出し **終了コード 2**
     (管理画面は「エンジンの作成に失敗」と表示し、走行の一覧に壊れたモデルが出ない)。電源断でも半端なエンジンが残らない。

### 2.8 後処理と予測の重ね合わせ(numpy。`post.py`・`ensemble.py`)

**`jetformer/policies/post.py`**(torch を読まない。走行と評価の両方が使う):

```python
KINDS = {'tcp_controller': ('jetformer.policies.controller', 'TCPPost')}   # テストでは monkeypatch で偽物を足す

def make(cfg, ref):
    """cfg = model_config(または drive.model_info。'postprocess' を見る)、ref = (ステアの中心, スロットルの中立) µs。
    → (後処理のオブジェクト or None, 説明)。'postprocess' が無い → (None, '')。
    知らない kind・読み込めない → (None, '後処理 <kind> を使えないため、操作の枝(actions)だけで走ります')"""
```

後処理のオブジェクトの約束(`TCPPost` が満たす):

```python
obj.kind                                   # 'tcp_controller'
obj.reset()                                # 中立で止めた後・シーンの切れ目
obj.step(outputs, v_now, params, t_ns)     # → (pred_us [10,2] float32, info dict)
    # outputs: {名前: np.ndarray}(バッチの次元を外したもの。actions は正規化のまま [10,2])
    # v_now: 今の速さ [m/s] = -eRPM × 0.5e-3(後退は負)、params: 走行のパラメータ(dict)、t_ns: 撮像時刻
    # info: {'mode': int, 'branch': str, 'v_hat': float, 'v_cmd': float, 'v_curv': float(nan 可), 'cap': float}
```

**`jetformer/policies/ensemble.py`**(予測の重ね合わせ。どの方策にも使える。**既定オフ、比較の判定には使わない**):

```python
class TemporalEnsembler(object):
    def __init__(self, horizon=10, fps=30): ...
    def reset(self): ...
    def push(self, t_ns, pred_us, j_max, m):
        """pred_us [10,2](µs)を時刻つきで貯め、出力の行 s = Σ_j w_j·p_{t-j}[s+j] / Σ w_j(s + j ≤ 9)。
        j = round((t_now - t_past)·fps)(フレームが抜けても時刻で数える)、w_j = exp(-m·j)(j = 0 が一番新しい = 一番重い)。
        j_max = 0 なら pred_us の写しをそのまま返す。j > j_max の古い予測は捨てる。戻り値 [10,2] float32"""
```

ACT の元の式(古いほど重い、m = 0.01)は使わない(30fps の車で反応が遅れる。`design_research.md` §2)。

### 2.9 走行からの呼び方(`jetformer/drive.py`、持ち主 I)

`drive.py` は torch を読まずに一覧を作る(H20)。`from . import policies` と `from .policies import post, ensemble` は numpy だけ。

- `model_info`(87-109)に足す: `policy`(無ければ `jetformer_v0`)・`policy_title`(`title_of`)・`policy_cfg`・`inputs`・`outputs`・
  `postprocess`(cfg のまま)・`drivable`・`why_not`(`policies.drivable(cfg)`)。エンジンが無ければ今までどおり None。
- `Driver.start`: `info['drivable']` が False なら `ValueError(why_not)`。
- `_run`(414-420): `TRTRunner` を作った直後に `policies.check_io({n: list(runner.shapes[n]) for n in runner.inputs}, {n: list(runner.shapes[n]) for n in runner.outputs}, num_modes)`。
  通らなければ `RuntimeError`(phase=error、中立のまま)。空回しの入力は `runner.shapes` から作る(416-418 の直書きをやめる)。
- 収集の `drive_info`(444): `policy`・`policy_cfg`・`postprocess`(kind と version だけ)を足す。
- `_loop`(496): **位置の引数は変えない**(`test_drive.py:85-86`)。中で:

```python
post_obj, post_note = post.make(st.get('model') or {}, ref)        # tcp 以外は None
ens = ensemble.TemporalEnsembler()
...
feed = {'image': img[None], 'tof': tof[None], 'imu': imu_n[None], 'state': state[None], 'mode': mode_cache[mode_id]}
res = runner.run_numpy(**feed)                     # 基本の5つを必ず渡す(エンジンが使わない入力は読まれない)
model_pred = res['actions'][0] * PWM_SCALE + PWM_CENTER          # [10,2] µs(今の 630 行)
infer_ms = ...
if not np.all(np.isfinite(model_pred)): hold(...)                  # 今と同じ
pred, pinfo = model_pred, None
if post_obj is not None:
    try:
        pred, pinfo = post_obj.step({k: v[0] for k, v in res.items()}, -float(veh['erpm']) * 0.5e-3, p, cam.t_ns)
    except Exception as e:
        hold('後処理でエラー: %s' % e, cam, ...); continue                    # 中立(推論の異常と同じ扱い)
    if not np.all(np.isfinite(pred)): hold('後処理の結果が数値ではありません', ...); continue
if p['ensemble_j'] > 0:
    pred = ens.push(cam.t_ns, pred, int(p['ensemble_j']), float(p['ensemble_m']))
k = min(max(p['lookahead'], 1), len(pred)) - 1      # ここから先(636 行〜)は今のまま: 先読み → ゲイン → ToF ガード → 補助 → 書き込み
```

- `prev_pred`(680)には **後処理・重ね合わせの後、ゲインの前** の `pred` を入れる(drive.py の冒頭の「車に届いている操作に合わせる」)。
  今の操作値を入れないモデル(この比較の全部)では使われない。
- `hold()` の中で `post_obj.reset()`・`ens.reset()` も呼ぶ(`helper.reset()` と同じ所)。
- `st['now']` に `'post': {'mode', 'branch', 'v_hat', 'v_cmd', 'v_now', 'v_curv', 'cap'}`(後処理があるとき)と `'ens_j'` を足す。
- `recorder.add_ai`(702)に足す: `model_pred`(後処理・重ね合わせの前の µs [10,2])、`post_mode`(int、無ければ −1)、`v_hat`・`v_cmd`(無ければ nan)、
  `ens_j`。`recorder.py` の `AI_CMD.npz` はこれらを `x.get(..., 既定)` で書く(中立の行・古い呼び方でも壊れない)。
- `PARAM_LIMITS` に足す: `speed_cap (0.0, 5.0)`(0 = なし)・`speed_scale (0.5, 1.5)`・`tcp_mode (0, 3)`・`ensemble_j (0, 3)`・`ensemble_m (0.0, 2.0)`。
  `DEFAULT_PARAMS`: 0.0・1.0・0・0・0.5。`_clean_params` で `tcp_mode`・`ensemble_j` を整数に丸める。古い `drive_settings.json` は既定値で読める。

### 2.10 評価の道具からの呼び方(持ち主 E)

```python
model, cfg, spec, pcfg = runtime.load_run(run_dir, ckpt='best', device='cuda')    # JetFormer(num_modes=...) の直書きをやめる
out = runtime.predict(model, {'image': ..., 'tof': ..., 'imu': ..., 'state': ..., 'mode': ...})   # FP32 → {名前: np}
pred_us = out['actions'] * 500 + 1500
post_obj, _ = post.make(cfg, (cfg['steer_center_us'], cfg['throttle_neutral_us']))   # tcp だけ。シーンごとに reset し、
# フレームの時刻の順に post_obj.step(...)(走行と同じ関数。v_now は aligned の vehicle[:, 3] の eRPM から)
```

置き換える所(interfaces.md §2.10 の9か所のうち、`tools/train.py:271`(F)とテストを除く7か所): `compare_models.py:36`・`evaluate.py:117`・`rl_evaluate.py:519`・
`rl_policies.py:501, 589`・`offline_rl.py:534`(`embed` が無い方策は止める。v0 は今の `model.norm` のフックのまま)・`benchmark_model.py:50`
(`--policy`・`--policy-cfg`・`--num-modes` を足す。既定 8 は変えない)。`rl_evaluate.py` の `CONFIGS`・`RULES`・`heldout`・`evaluate` は **変えない**。
`rl_policies.GPU_PATTERNS` に `benchmark_model.py`・`policy_zoo.py train`・`policy_zoo.py predict`・`policy_zoo.py engine-check` を足す。

### 2.11 管理画面の学習(`jetformer/training.py`、持ち主 F)

- `start(dataset, stride=3, epochs=30, batch=64, lr=3e-4, pretrained=True, build_engine=True, command_input=True, mask_impacts=True,
  policy='jetformer_v0', policy_cfg=None, impact_g=None, seed=None)`。
- `spec.resolve(policy_cfg, ui=True)`(だめなら `ValueError` → 400)、`spec.ready()`(だめなら `ValueError`)、
  `requires_pretrained` なのに `pretrained=False` → `ValueError`。空きメモリ `max(MIN_FREE_MB, spec.hints['min_free_mb'])`。
- 学習結果の名前: v0 は今のまま `'%s-s%d-%s'`、ほかは `'%s-%s-s%d-%s' % (dataset, spec.slug, stride, 日時)`。
- `run.json` に `policy`・`policy_cfg`(resolve の結果)・`impact_g`・`seed`。コマンドには **既定以外のときだけ** `--policy`・`--policy-cfg`・
  `--impact-g`(None でないとき)・`--seed`(None でないとき)を足す。**v0・引数なしなら今とまったく同じコマンド**(テストで確かめる)。
- `list_runs`: `params` に `policy`(無ければ `jetformer_v0`)・`impact_g`・`seed`、項目に `policy_title`。

---

## 3. 4つの方策(+ v0)

### 3.0 共通と、登録簿の値

- 入力は基本の5つ。出力 `actions [B,10,2]` は全部が持つ(`tcp` は追加の出力も)。ジャンルは `mode_embed`(トークン)+ ジャンル別の `heads`。
- `act`・`tcp` は v0 と同じトークン化(`blocks.add_sensor_tokens`)なので、v0 との違いが「出力の作り方」だけになる。
- 学習の条件は全部同じ(§4)。方策ごとに違うのは構造・損失・凍結だけ。検証の誤差はどれも `actions` の L1。

| policy | defaults(凍結する既定値) | options(管理画面) | hints: train_rate / val_rate / min_free_mb / infer_ms | train_only_keys |
|---|---|---|---|---|
| `jetformer_v0` | dim 256・layers 4・heads 8 | なし(比べる基準) | 200 / 650 / 3500 / 2.3 | — |
| `act` | dim 256・enc_layers 4・dec_layers 2・heads 8・ffn 512・dropout 0.1・cvae false・latent_dim 32・kl_weight 10.0・cvae_layers 2 | dec_layers(1〜4)・cvae(bool)・kl_weight(1.0/10.0) | 190 / 620 / 3500 / 2.5 | kl_weight |
| `tcp` | dim 256・enc_layers 4・dec_layers 2・heads 8・ffn 512・dropout 0.1・wp_steps 10・wp_dt 0.1・speed_bins 10・speed_max 4.5・w_wp 0.5・w_speed 0.1・cumsum 'op'・labels 'tcp_labels_v1' | なし(比べる基準を固定) | 185 / 600 / 3500 / 2.7 | w_wp・w_speed |
| `msfusion` | dim 256・scales [1,2,3,4]・ex_layers [2,2,2,1]・ex_heads 4・ex_ffn [4,4,4,2]・late_layers 2・late_heads 8・head 'tokens'・dropout 0.1 | scales([1,2,3,4] / [2,3,4] / [3,4])・head(tokens / pooled) | 135 / 430 / 3500 / 3.1 | — |
| `dinov2` | variant 'vits14'・weights 'data/pretrained/dinov2_vits14_pretrain.pth'・pool 2・use_cls true・finetune_last 0・lr_mult 0.1・dim 256・layers 4・heads 8・vit_depth 12 | pool(1/2)・finetune_last(0/2) | 230 / 300 / 3500 / 5.1 | finetune_last・lr_mult |

実測(`bench/results/summary_table.md`。乱数の重み・バッチ1・TensorRT 10.16 FP16、作業メモリ 1GB):

| policy(既定の形) | パラメータ | エンジン | エンジンだけ 中央 / p99 | `run_numpy` | 学習 1ステップ(64) | 学習のピーク | 30エポックの見込み | エンジンの作成 |
|---|---|---|---|---|---|---|---|---|
| モデル B(学習済み、既存のエンジン) | 14.1M | 29 MB | 1.50 / 1.71 ms | 2.35 ms | — | — | — | — |
| `jetformer_v0` | 14.1M | 26 MB | 1.47 / 1.66 | 2.33 | 310 ms | 1.3 GB | 17 分 | 69 秒 |
| `act`(Decoder 2層) | 15.6M | 28 MB | 1.64 / 1.82 | 2.48 | 323 ms | 1.4 GB | 18 分 | 96 秒 |
| `tcp`(Decoder 2層・21 クエリ) | 15.6M | 28 MB | 1.69 / 2.13 | 2.50 | 334 ms | 1.4 GB | 18 分 | 95 秒 |
| `msfusion`(既定) | 18.9M | 36 MB | 2.18 / 2.37 | 3.12 | 463 ms | 1.8 GB | 26 分 | 210 秒 |
| `dinov2`(pool 2・凍結) | 24.5M(2.9M) | 49 MB | 3.88 / 4.13 | 5.06 | 270 ms | 0.5 GB | 16 分 | 105 秒 |

揺れ: 別のプロセス・作り直したエンジンで ±0.2 ms、`run_numpy` は同じエンジンで最大 0.6 ms。0.2 ms 以下の差は意味がない。
PyTorch FP16 のままだと 13〜34 ms なので、走行は必ずエンジンで。

### 3.1 `act` — ACT 型(持ち主 P-act、`jetformer/policies/act.py`)

```
blocks.sensor_tokens → [B,94,256] (+ CVAE のときだけ潜在トークン 1)+ pos [1,94(+1),256]
  → encoder(256, 8, 512, enc_layers=4) → enc_norm(LayerNorm) = memory
queries [1,10,256](学習できる。1 クエリ = 1 フレーム先)→ decoder(256, 8, 512, dec_layers=2)(自己注意 + memory への交差注意)
  → dec_norm → heads: Linear(256, num_modes×2)(10 クエリで共有)→ view(B,10,M,2).transpose(1,2) → select_mode → actions [B,10,2]
```

- 層の名前: `image_backbone`・`image_proj`・`tof_encoder`・`imu_mlp`・`state_mlp`・`mode_embed`(v0 と同じ)、`pos`・`encoder`・`enc_norm`・
  `queries`・`decoder`・`dec_norm`・`heads`。初期化: `pos`・`queries` は trunc_normal(std 0.02)。DETR の「クエリの位置埋め込みを毎層足す」形ではなく、
  学習できるクエリをそのまま Decoder の入力にする(試作と同じ)。
- `heads` の並びは出力の index = ジャンル×2 + [ステア, スロットル] → `genre_keys.heads = {'heads': 2}`。
- **CVAE(既定オフ)**: `cvae=True` のとき `USES_TARGET = True`。学習で `target` があれば、トークン [`cvae_cls`(学習できる 1)、
  `mode_embed(mode)`、`cvae_action(target)`(Linear(2,256) × 10)] + `cvae_pos [1,12,256]` → `cvae_encoder`(cvae_layers=2、同じ大きさ)→
  先頭 → `cvae_out` Linear(256, 2×32) → μ・logσ²(logσ² は [−10, 10] に切る)→ z = μ + ε·exp(logσ²/2)、
  `kl_i = −0.5·Σ(1 + logσ² − μ² − exp(logσ²))`(サンプルごと [B])。推論・検証・書き出しは z = 0(潜在トークン = `latent_proj` のバイアスの定数)。
- `policy_loss(out, target, extras)` = `l1 + kl_weight·kl_i`(CVAE で学習中のときだけ。ほかは l1)。`loss_parts = {'kl': kl.mean()}`。
  KL が 0 に張り付いていたら(最後のエポックの `train_parts.kl` < 0.01)、z が使われていない = 普通の ACT と同じ、と報告に書く。
- 期待する大きさ: 既定 15.6M(±0.3M)、CVAE で +約1.1M(学習のときだけ)。

### 3.2 `tcp` — TCP / TransFuser++ 型(持ち主 P-tcp)

#### 3.2.1 モデル(`jetformer/policies/tcp.py`)

```
blocks.sensor_tokens + pos [1,94,256] → encoder 4層 → enc_norm = memory
queries [1,21,256] = 通過点 10・速度 1・操作 10 → decoder 2層 → dec_norm → hs [B,21,256]
  hs[:, 0:10]  → heads_wp: Linear(256, M×2) → select_mode → Δ(x, y) [B,10,2] → 累積和 → waypoints [B,10,2](m、自車の座標)
  hs[:, 10]    → heads_v : Linear(256, M×10) → select_mode → speed_logits [B,10](0, 0.5, …, 4.5 m/s のクラス)
  hs[:, 11:21] → heads   : Linear(256, M×2) → select_mode → actions [B,10,2](操作の枝。v0 と同じ意味)
```

- `EXPORT_OUTPUTS = ('actions', 'waypoints', 'speed_logits')`。累積和は `cumsum='op'`(ONNX の CumSum。TensorRT 10.16 で通った)、
  `'matmul'`(下三角の定数行列)でも同じ結果になること(テスト)。GRU は使わない(TensorRT でループを作らない)。
- 自車の座標: **x = 前、y = 左**、単位 m、原点 = 撮像時刻 t₀ の車、通過点 k = t₀ + 0.1·k 秒(k = 1..10、1秒先まで)。
  操作の `actions` は 33 ms 刻みの 0.33 秒先まで(時間の刻みが違う)。
- 損失(`policy_loss`。重みは先に固定、検証で調整しない):

```
l1 = |actions − target|.mean((1,2))                                    [B]
v  = y_wp_valid.float()                                                [B]
wp = |waypoints − y_waypoints|.mean((1,2)) · v                         [B](m)
q  = twohot(y_v_tgt, bins)                                             [B,10](隣り合う2クラスに線形で分ける。0 未満・4.5 超は端に切る)
ce = −(q · log_softmax(speed_logits.float())).sum(1) · v               [B]
loss = l1 + 0.5·wp + 0.1·ce          loss_parts = {'wp_m': wp.mean(), 'ce': ce.mean()}
```

  TCP の特徴の損失・価値の頭・今の速さの補助の頭は使わない(教師が人、eRPM は入力にある)。
- 期待する大きさ: 15.6M(±0.3M)。

#### 3.2.2 ラベル(`jetformer/policies/tcp_labels.py`、numpy だけ)

元データ: `<データセット>/can_bus/<scene>_ms_imu.json`(約100 Hz、`rotation_rate[2]` rad/s、z 正 = 左旋回)と
`<scene>_vehicle_monitor.json`(約100 Hz、`erpm`・`steering_us`・`throttle_us`・`valid`・`ai_driving`)。手順は `research_src/tcp_label_check.py` の
`load_scene`・`dead_reckon`・`waypoints` と同じ:

1. ジャイロのバイアス b: ログ(シーン名の先頭の日時)ごとに、止まっている間(|eRPM| < 1 かつ |gz| < 5°/s)の gz の中央値。50 サンプル未満なら 0。
2. 速さ v(t) = −eRPM(t) × 0.5e-3 m/s(IMU の時刻へ線形補間。後退は負)。
3. IMU の時刻で積分: ψ = ∫(gz − b)dt、X = ∫ v·cos ψ dt、Y = ∫ v·sin ψ dt(dt は 0〜50 ms に切る)。
4. フレーム i(aligned の `timestamps_us[i]` = t₀)の自車座標へ: `x_k = cos ψ₀·ΔX + sin ψ₀·ΔY`、`y_k = −sin ψ₀·ΔX + cos ψ₀·ΔY`(k = 1..10、0.1 秒刻み)。
5. 目標速度 `v_tgt` = t₀ + 0.2〜0.5 秒の v の平均(0〜4.5 に切る)。`v_now` = t₀ の v。
6. 有効 `wp_valid`: 1秒の窓がシーンの中、窓の中の IMU の間隔 < 50 ms、窓の中で v ≥ −0.1、窓が衝突(水平 2.5G)の前後に掛からない、
   窓の中の |gz| ≤ 15 rad/s(衝突でジャイロが飽和した窓を外す)。無効のフレームも `actions` の L1 は今までどおり学習する。

キャッシュ: `data/derived/<データセット名>/tcp_labels_v1.npz`(`scene [N]`・`frame [N]`・`waypoints [N,10,2]`・`v_tgt [N]`・`v_now [N]`・
`wp_valid [N]`・`version`・`splits_sha1`)。split の全部のシーン(学習と検証)の全フレーム。`.tmp.npz` に書いてから置き換える。
`splits.json` の sha1 が変わっていたら作り直す。CPU で1分以内(clean_driving: 52 シーン)。`tools/make_tcp_labels.py <データセット>` でも作れる。

- `tcp.prepare(data_root, cfg)`: キャッシュと較正(§3.2.3)が無い・古ければ作る。`can_bus` が無ければ `SystemExit('TCP 型には can_bus の IMU・車両の記録が要ります')`。
- `tcp.extra_targets(data_root, cfg)`: npz を読み、split のシーンごとに `{'waypoints': [n,10,2], 'v_tgt': [n], 'wp_valid': [n]}`(n = aligned のフレーム数。
  npz に無いフレームは 0 と無効)。データセットは `y_waypoints`・`y_v_tgt`・`y_wp_valid` として返す。

確かめた数字(`research_src/tcp_label_check.json`): 1秒の窓が取れるのはジャンル 4 で 96.1%・5 で 93.3%、符号の確認は3つとも合格、
目標速度 p10 / p50 / p90 / p99 = 1.42 / 2.49 / 3.75 / 4.46 m/s、ジャイロのバイアス −0.0016 rad/s。

#### 3.2.3 較正(`tcp_labels.fit_calibration`、学習のシーンだけ、人の操作のフレームだけ)

- **ステア → 曲率の表**: 遅れ ∈ {0, 40, 80, 120} ms のうち、|ステア − 中心| ≤ 250 µs・v ≥ 0.8 m/s で R² が最大のもの(今のデータで 40 ms)。
  κ(t) = (gz(t + 遅れ) − b) / v(t + 遅れ)。ステアのずれ(µs − 中心)を −600〜+600 の 50 µs の区切りに分け、速度帯(v < 2・2 ≤ v < 3・3 ≤ v m/s)ごとの κ の中央値
  (30 個未満の区切りは全速度の表の値)。ずれについて単調に増える形にそろえる(`np.maximum.accumulate`、逆引きのため 1e-6 ずつ足して狭義単調に)。
  逆引き: `ずれ = np.interp(κ, κ_表, ずれ_表)`(表の外は端 = いっぱいに切った値)、ステア = 中心 + ずれ(1000〜2000 に切る)。
- **スロットルの前向きの写像**: `thr = c0 + c1·v + c2·(v_tgt − v)/0.33`(最小二乗。v ≥ 0.3、前進のフレーム thr < 中立 − 10)。
  今のデータで(この前進のフレームだけの規則)c0 1490.7・c1 −72.1・c2 −13.6(学習の誤差 31.4 µs、`fwd_max_us` 361 µs)。
  〔段階3で訂正: 以前ここにあった 1530.2・−82.7・−25.4 は research の「全フレーム」の当てはめの値で、この規則の結果ではない。
  全フレームでやり直すと 1529.1・−82.5・−24.7 になり research と合う。定常 2.75 m/s での差は約 10 µs。実装は上の規則のまま〕
  `fwd_max_us` = 同じフレームの (中立 − thr) の 99% 点。
  表の端で **どの速度帯にもサンプルの無い区切り**(±525・±575 µs)は、全速度の表の値ではなく、その速度帯の端の値をそのまま延ばす(= いっぱい。段階3で追加)。
- 結果は `data/derived/<データセット>/tcp_calib_v1.json`。`tcp.finalize(run_dir, cfg, data_root)` がこれを読み、§3.2.5 の `postprocess` を返す
  (学習結果ごとに当てはめ直さない。検証のシーンは使わない)。

#### 3.2.4 制御器(`jetformer/policies/controller.py`、numpy だけ、torch を読まない)

1フレームの処理(`TCPPost.step(outputs, v_now, params, t_ns)`、ref = (中心 c, 中立 n)):

1. **目標速度**: `v_hat = Σ softmax(speed_logits)·bins`。上限 `cap_eff`: `speed_cap > 0` ならその値、0 なら上限なし(∞)。上限が **上がった・外れた** ときだけ、
   `cap_eff` は前の値から 1.5 m/s² で上がり、5.0 m/s を超えたら「なし」になる(下がるときはすぐ。最初と `reset()` ではすぐ目標の値)。
   `v_cmd = min(speed_scale·v_hat, cap_eff)`。上限が一定なら状態を持たない(評価をフレームごとにできる)。
2. **前向きのスロットル** `thr_ff(v_cmd)`: `v_cmd < 0.3` → 中立 n。そうでなければ `thr = c0 + c1·max(v_now,0) + c2·(v_cmd − v_now)/0.33`、
   前進の量 `n − thr` を [0, fwd_max_us] に切る(**中立より上 = ブレーキ・後退側は出さない**)。例外のブレーキ: `v_now − v_cmd > 0.3` かつ `v_now > 1.0` なら
   中立 + 150 µs(ESC は前進中の後退側 = ブレーキ。止まってから R → N → R でしか後退しないので、1 m/s 以上なら後退に入らない)。
3. **通過点を追うステア**(k = 1..10、時刻 τ_k = k/30 秒): 通過点の前に (0,0)(時刻 0)を足した折れ線で、τ_k の位置 P_k を時間で線形補間し、
   向き θ_k は **頂点の向きを時間で補間したもの**(原点 = 0 = 自車の向き、途中の頂点 = 隣り合う区間の向きの円の平均、最後の頂点 = 最後の区間の向き。
   長さ 1e-3 m 未満の区間は次の区間)。〔段階3で変更: 以前は「その区間(弦)の向き」。弦の向きのままだと円の上でも τ_k が区間のどこにあるかで
   追う曲率が 0.6〜1.4 倍に揺れ、正解の通過点を入れたステアの誤差が 学習 69.8 / 検証 65.1 µs(k3 で 88 µs に跳ねる)。頂点の向きでは 56.2 / 52.2 µs。
   `tcp_notes.md` §5・§6〕その点の速さ v_k = max(区間の長さ/0.1, 0.3)、見る距離 L_k = max(0.3, 0.25·v_k)。
   折れ線の上で P_k から弧長 L_k 先の点 G_k(端を超えたら最後の区間の向きに延ばす)を、P_k・θ_k の座標へ直して (x, y)、κ_k = 2y/(x² + y²)。
   ステア = 表の逆引き(速度帯は v_now)。**道の全長が 0.2 m 未満なら操作の枝のステア**(止まっているときは手がかりが無い)。
4. **混ぜ方** `tcp_mode`(走行のパラメータ):

   | 値 | 名前 | ステア(10 行) | スロットル(10 行) |
   |---|---|---|---|
   | **0(既定)** | control | 操作の枝 | 操作の枝。**`cap_eff` が有限のときだけ** 各行を `max(操作の枝, thr_ff(cap_eff))`(µs が大きい = 前進が弱い方。上限で弱めるだけで速くしない) |
   | 1 | hybrid | 操作の枝 | `thr_ff(v_cmd)`(10 行同じ) |
   | 2 | fusion(試験的) | 過去 1 秒(このオブジェクトが出したステア)の半分以上が |ステア − c| > 100 µs なら 0.7·操作 + 0.3·通過点、でなければ 0.3·操作 + 0.7·通過点 | 0.5·操作 + 0.5·`thr_ff(v_cmd)` |
   | 3 | trajectory(試験的) | 通過点を追うステア | `thr_ff(v_cmd)` |

   1〜3 でも最後に「`thr_ff(cap_eff)` より前進を強くしない」を掛ける。**`tcp_mode = 0` かつ `speed_cap = 0` なら、出力は `actions` の µs とビット単位で同じ**(テスト)。
5. `info`: `mode`・`branch`('control' / 'hybrid' / 'fusion:ctrl' / 'fusion:traj' / 'trajectory' / 'fallback')・`v_hat`・`v_cmd`・`cap`(cap_eff、なしは 0)・
   `v_curv`(表示だけ: 予測の道の最大の |κ| に対し、表の「いっぱいに切った曲率」が 1.1·|κ| 以上になる最大の速度帯の上限。判定には使わない)。

時間の目安: 1回 0.1〜0.3 ms(試作 32 µs + ステップごとの pure pursuit 約0.2 ms)。1 ms を超えたらベクトル化する。

#### 3.2.5 `postprocess`(`model_config.json` に `finalize` が書く)

```json
"postprocess": {"kind": "tcp_controller", "version": 1, "pwm_center": 1500.0, "pwm_scale": 500.0,
  "speed_bins": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], "wp_dt": 0.1, "fps": 30,
  "steer_table": {"lag_ms": 40, "offset_us": [-575, ..., 575], "bands_mps": [2.0, 3.0], "curv": [[...], [...], [...]]},
  "throttle_ff": {"c0": 1490.7, "c1": -72.1, "c2": -13.6, "tau_s": 0.33, "fwd_max_us": 361.0},
  "rules": {"stop_v": 0.3, "brake_us": 150.0, "brake_dv": 0.3, "brake_vmin": 1.0, "cap_ramp_mps2": 1.5},
  "pursuit": {"t_look_s": 0.25, "l_min_m": 0.3, "min_path_m": 0.2},
  "fusion": {"turn_us": 100.0, "turn_frac": 0.5, "w_major": 0.7, "window_s": 1.0},
  "calib_source": {"dataset": "clean_driving_20261011", "scenes": "train", "created_at": "...", "labels_sha1": "..."}}
```

(数字は今のデータの値の例。`fwd_max_us` は当てはめの値。)`drive.model_info` が kind を知らなければ「操作の枝だけで走る」(§2.8)。

#### 3.2.6 他の持ち主との約束(I・E はこれだけを前提にする)

`from jetformer.policies.controller import TCPPost` → `TCPPost(postprocess: dict, ref: (float, float))`、`.kind`・`.reset()`・
`.step(outputs, v_now, params, t_ns) -> (np.ndarray [10,2] float32 µs, info dict)`(§2.8)。`params` は `tcp_mode`・`speed_cap`・`speed_scale`
が無ければ既定(0・0.0・1.0)で動く。例外は出さない(入力が足りなければ `branch='fallback'` で操作の枝)。

### 3.3 `msfusion` — 本家 TransFuser に近い多段融合(持ち主 P-msf、`jetformer/policies/msfusion.py`)

`bench/protos.py` の `MSFusionRDProto`(実測した既定の形)を本実装にする。

| 段 | 画像(ResNet-18、`pretrained` で ImageNet) | ToF(新しい CNN、BN) | 交換の Transformer(段ごと) |
|---|---|---|---|
| 入力 | 3×224×224 → `img_stem`(conv1・bn1・relu・maxpool) | [距離, 信頼度, 有効の印] 3×42×54 → **`F.pad(..., (1,1,3,3))` で 48×56**(詰めた所 = 無効 0)→ `stem` conv_bn(3→32) | — |
| 1 | `img_stages[0]`(layer1)64×56×56 → 平均 8×8 → 7×7 | conv_bn(32→64, s2)・conv_bn(64→64)→ 64×24×28 → 平均 4×4 → 6×7 | d64・4ヘッド・2層・FFN×4 |
| 2 | layer2 128×28×28 → 平均 4×4 | conv_bn(64→128, s2)・conv_bn → 128×12×14 → 平均 2×2 | d128・4ヘッド・2層・FFN×4 |
| 3 | layer3 256×14×14 → 平均 2×2 | conv_bn(128→256, s2)・conv_bn → 256×6×7 | d256・4ヘッド・2層・FFN×4 |
| 4 | layer4 512×7×7 | conv_bn(256→256, s1)・Conv 1×1(256→512)+BN+ReLU → 512×6×7 | d512・4ヘッド・**1層・FFN×2** |

- 各段の交換(`StageFusion`): 画像 49 + ToF 42 = 91 トークン + 段ごとの `pos [1,91,ch]` → `encoder(ch, 4, ch×ffn, layers)` → LayerNorm →
  分けて元の格子 → **双線形(`scale_factor` = 窓の大きさ、`align_corners=False`)で拡大して各枝に足す**(残差)。`scales` に無い段は交換しない。
- 平均は **固定の `F.avg_pool2d`**(窓は上の表)。`adaptive_avg_pool2d`・antialias の縮小は使わない(opset 17 で失敗する。§8)。
- 最後(`head='tokens'`、既定): 4段目の画像 49 トークン → `img_proj` Linear(512→256)、ToF 42 → `tof_proj` Linear(512→256)、`imu_mlp`・`state_mlp`・`mode_embed`
  の 3 トークン、`query` 1 → + `pos [1,95,256]` → `fusion` = encoder(256, 8, 512, late_layers=2) → `norm` → 最後のトークン → `trunk`(Linear+ReLU)→
  `heads` Linear(256, M×20)(v0 と同じ並び)→ select_mode。
- `head='pooled'`(選択肢、TransFuser どおり): `global_proj`(Linear 512→256)(画像と ToF の4段目の平均の和)と IMU・操作値・ジャンルをつなぐ(4×256)→
  `trunk` MLP → `heads`。
- 層の名前: `img_stem`・`img_stages`・`tof`・`fuse`・`img_proj`・`tof_proj`・`imu_mlp`・`state_mlp`・`mode_embed`・`query`・`pos`・`fusion`・`norm`・`trunk`・`heads`。
- 期待する大きさ: 18.9M(±0.5M)。ONNX は既定で約 1,710 ノード(v0 の 488 の約 3.5 倍。既定の試作 `msfusion_rd` は 1,786。
  以前ここにあった 951〜1619 は既定でない試作の値)。エンジンの作成 約3.5 分。
- `scales` の選択肢に `[1, 2, 3]`(4段目の交換なし)を段階3で足した(design_research §5.3: エンジンが 4 ms を超えたときの代わり)。

### 3.4 `dinov2` — DINOv2 の画像特徴(持ち主 P-dino、`jetformer/policies/dinov2.py`・`vit.py`)

- **`vit.py`**: `bench/protos.py` の `DinoViTS14` と部品(`_PatchEmbed`・`_Attention`・`_LayerScale`・`_Mlp`・`_Block`)を移す。キーの名前は公式と同じ。
  `DinoViTS14(img_size=224, dim=384, depth=12, heads=6)`、LayerNorm eps 1e-6、qkv にバイアス、LayerScale、GELU(erf)、MLP 4 倍、レジスタなし。
  `load_official(path)`: `torch.load(weights_only=True)` → `mask_token` を除く → `pos_embed` [1,1370,384](37×37)を **読み込むときに1回だけ**
  bicubic・antialias なし・`scale_factor = (16 + 0.1)/37` で 16×16 に補間 → `strict=True`。注意は `F.scaled_dot_product_attention`(書き出しで MatMul + Softmax に分かれる)。
- **`dinov2.py`**: `bench/protos.py` の `DinoProto`(pool 2)と同じ。

```
image [B,3,224,224](今の ImageNet 正規化のまま。DINOv2 と同じ平均・標準偏差)
  → vit(凍結)→ 最後の norm → CLS [B,1,384] + パッチ [B,256,384] → パッチを 16×16 に並べて 2×2 の平均 → 64(pool=1 なら 256)
  → image_proj(LayerNorm(384) → Linear(384, 256))→ 65 トークン
tof_encoder(v0 の ToFEncoder)42・imu_mlp・state_mlp・mode_embed 各1・query 1 → + pos [1,111,256]
  → fusion = encoder(256, 8, 512, 4) → norm → 最後のトークン → trunk → heads Linear(256, M×20) → select_mode
```

- 凍結: `finetune_last = 0`(既定)で ViT の全パラメータ `requires_grad=False`、学習でも ViT は `no_grad`・`eval`(`train()` を上書き)。
  `finetune_last = 2`: 最後の2ブロックと `vit.norm` を学習し、`param_groups(lr)` で学習率 `lr × lr_mult`(0.1)の群にする(LP-FT の2段目。
  凍結で学習した `best.pth` から `--init` で始める。`finetune_last`・`lr_mult` は `train_only_keys` なので `--init` の確認を通る)。
- 重み: `build(pretrained=True)` のときだけ `ROOT / weights` を読む。無ければ `FileNotFoundError('DINOv2 の重みがありません: <パス>(取得の手順は README)')`。
  **ネットから自動で取らない**。`build(pretrained=False)`(`load_run`・評価・テスト)は重みのファイルに触れない(`best.pth` に ViT の重みも入る、約98 MB)。
  `prepare()` で sha256 を確かめる(`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`、違えば警告して続ける)。
- `vit_depth`(既定 12)はテストで小さい ViT を作るためだけ(`pretrained=True` では 12 以外を断る)。
- エンジンの精度: FP16(CPU・TRT での確認でコサイン類似度 ≥ 0.9979、あふれなし)。学習した重みで §5.1 の差 ≤ 2.5 µs を満たさなければ、
  その学習結果の `model_config.engine.precision` を `bf16` → `fp32` の順に変えて作り直す(M。`fp32` で `run_numpy` > 6 ms なら走行の候補から外す)。
  〔段階3の GPU の試しで分かったこと(`integration_smoke.md` §5): **BF16 は FP16 より仮数が短く(7 ビット対 10 ビット)、精度のゲートの代わりにならない**。
  試し学習の dinov2 の重みで FP16 1.43 µs → BF16 4.31 µs(不合格、速さは同じ 3.95 ms)、FP32 0.18 µs だがエンジンだけ 8.5 ms・`run_numpy` 9.5 ms
  (速さのゲート 6 ms に不合格)。BF16 が効くのは FP16 であふれる(NaN・Inf)ときだけ。差のゲートに落ちたときの現実的な代わりは
  「FP16 + LayerNorm・Softmax だけ FP32」の候補(`smoke/build_mixed_experiment.py`。同じ重みで 1.32 µs・エンジン 4.08 ms・`run_numpy` 5.05 ms。
  この重みは FP16 でも合格なので改善は小さく、FP16 で落ちる重みで効くかは未確認)。必要になったら `build_engine.py` の精度の選択肢に足す(まだ足していない)〕
- 期待する大きさ: 24.5M(学習するのは 2.9M、±0.2M)。

---

## 4. 学習の計画と GPU の予算(実行は M、道具は E の `tools/policy_zoo.py`)

### 4.1 共通の設定(モデル B と同じ)

`nice -n 10 python -u tools/train.py --data data/datasets/clean_driving_20261011 --out data/runs/<名前> --policy <P> --seed <S>
--stride 3 --epochs 30 --batch 64 --lr 3e-4 --workers 4 --no-command-input --impact-g 2.5 [--resume]`(`--policy-cfg` なし = 既定の形)。
学習 5,912・検証 1,559 サンプル。名前: `clean_driving_20261011-zoo-<slug>-seed<S>-nocmd-s3-e30`(例 `...-zoo-act-seed0-...`)。
各学習の後、別のプロセスで `tools/build_engine.py`(学習とエンジンの作成を同時に動かさない。作成のプロセスは最大 3〜4 GB)。

### 4.2 本体(10 本。種を先に回す順 = 途中で止まっても全方策に1本ずつある)

| # | 学習 | 学習(分) | エンジン(分) |
|---|---|---|---|
| 1〜5 | v0・act・tcp・msfusion・dinov2 の種 0 | 17 + 18 + 18 + 26 + 16 = 95 | 1.2 + 1.6 + 1.6 + 3.5 + 1.8 = 9.7 |
| 6〜10 | 同じ順で種 1 | 95 | 9.7 |
| — | 起動・データの読み込み・CPU での ONNX の書き出し(1本 約1分) | 10 | — |
| — | 予測(PyTorch FP32、保留の 10,985 フレーム + ずれの試験 5×1,557 + last.pth の clean 4,670。1本 約1.5 分) | 16 | — |
| — | エンジンの確認・速さ(1本 約1分、別のプロセス) | 10 | — |
| | **合計** | **約 4.1 時間** | |

最初の1本(v0 種 0)が終わったら、実際の時間と見込み(18 分)を比べる。1.3 倍を超えていたら、残りの見込みを直して §4.3 の任意を減らす。

### 4.3 任意(本体が終わって時間が残るときだけ。この順に)

| 段 | 中身 | 条件(結果を見る前に決めた) | 時間 |
|---|---|---|---|
| O1 | 種 2: v0 と、候補の方策(最大2つ) | §5.4 の判定で「良い」または「同等で副指標が良い」方策がある(L̄ の小さい順に2つまで)。3本で判定をやり直し、3本の判定を採る | 最大 約85 分 |
| O2 | `act`(CVAE、β = 10)種 0 | 参考(判定に入れない。1本しかないため) | 約22 分 |
| O3 | `dinov2`(LP-FT: `finetune_last 2`、`--init` = dinov2 種 0 の best.pth、30 エポック)種 0 | 参考 | 約22 分 |

削る順: O3 → O2 → O1。**本体(2つの種・30 エポック)は削らない**。電源断などで本体が予算に収まらないときは、あるもので判定し「暫定」と書き、
残りは後で続ける(`--resume`)。

### 4.4 電源断への備え

- `policy_zoo.py train` は1本ずつ: `result.json` が `done`・30 エポック・`jetformer.onnx` あり → 飛ばす。`last.pth` があれば `--resume`。
  エンジンは tmp → 置き換え(§2.7)。状態は `data/zoo/policy_zoo_20261012/zoo_state.json`(tmp → 置き換え)。
- 予測・確認は1本ずつ npz・json に書き、あれば飛ばす。評価(CPU)は何度でもやり直せる。
- `--resume` で再開した学習は乱数の状態が戻らないので、ビット単位では同じにならない。`zoo_state.json` に再開したエポックを残し、報告に書く
  (モデル B も 26 エポック目で再開している)。

### 4.5 `tools/policy_zoo.py` の形(持ち主 E)

`tools/rl_policies.py`・`tools/rl_evaluate.py` と同じ作り(numpy の規則は関数にしてテスト、GPU の小コマンドは1つずつ、出来たものは飛ばす)。

| 小コマンド | GPU | 中身 | 書くもの |
|---|---|---|---|
| `plan [--dry-run]` | — | 計画(§4.2・§4.3)と規則(§5)を表示し、`zoo_rules.json` を1回だけ書く(あって中身が違えば止まる。`--dry-run` は書かない) | `results/zoo_rules.json`・`data/zoo/policy_zoo_20261012/zoo_rules.json` |
| `prepare` | — | `tcp` のラベル・較正(`tcp.prepare`)、DINOv2 の重みの sha256、`heldout.json` の `pred_table_key` の一致 | `data/derived/clean_driving_20261011/`、`zoo_state.json` |
| `train [--only 名前 ...] [--optional O1 ...]` | ○ | §4.1 のコマンドを1本ずつ(済んだものは飛ばす、`last.pth` があれば `--resume`)→ 別のプロセスで `build_engine.py`。各学習結果に `run.json`(`policy`・`policy_cfg`・`seed`・`impact_g` 2.5、管理画面の一覧に出るように) | `data/runs/<名前>/`、`zoo_state.json`(時間・再開の記録) |
| `predict` | ○ | 各学習結果の best.pth(と clean の last.pth)を `runtime.load_run` → `runtime.predict`(FP32)で、保留の表(`rl_evaluate.pred_table`)とずれの試験のフレーム。`tcp` は waypoints・speed_logits も | `data/zoo/policy_zoo_20261012/pred/<名前>.npz`・`pert/<名前>.npz` |
| `engine-check` | ○ | 1本ずつ **別のプロセス** で: `drive.py` と同じ手順でエンジンを読み(`check_io`・空回し)、実フレーム 64 個で TensorRT と FP32 の差、`run_numpy` の中央値・p99(100 回の空回しの後 600 回) | `engine/<名前>.json` |
| `evaluate` | — | §5 の全部(ゲート・主指標・副指標・`tcp` の追加・判定)。ゲートの「学習のメモリ」は `result.json` の `peak_mem_mb` | `zoo_eval.json`(と `results/zoo_eval.json` に写す) |
| `figures --fig-dir <dir>` | — | 主指標の種ごとの点と範囲・δ の帯、R1・DE の区間、速さと L1 の散布、学習曲線(日本語・大きい文字) | 図(PNG) |

GPU の小コマンドは始める前に `rl_policies._check_gpu_free`(`GPU_PATTERNS` に新しいコマンドを足したもの)と空きメモリ 3.5 GB 以上を確かめ、
走行中・管理画面の学習中なら始めない。結果のファイルはどれも tmp → 置き換え。

---

## 5. 評価の手順(結果を見る前に固定する)

`tools/policy_zoo.py plan` が、学習を始める前に下の規則を `/home/jetson/jetformer_reports/policy_zoo_20261012/results/zoo_rules.json`
(と `data/zoo/policy_zoo_20261012/zoo_rules.json`)に書く。**同じ名前のファイルがあって中身が違えば止まる**(`heldout.json` と同じ考え方)。
フレームは `rl_evaluate.py` の保留の表(`pred_table`、`heldout.json` の `pred_table_key` と一致を確かめる)と同じものを使う。
`clean_driving_20261011` で学習した方策にとって、この表のフレームは全部学習に使っていない(ただし人の 11 シーンは best.pth を選ぶのに使った検証と同じ)。

### 5.1 ゲート(満たさなければ走行の候補にしない。方策は2つの種の両方が満たすこと)

| 項目 | 合格 |
|---|---|
| エンジン | `build_engine.py` が終了コード 0(`check_io` 合格) |
| 精度 | 実フレーム 64 個で、TensorRT と PyTorch FP32(best.pth)の `actions` の差の最大 ≤ 0.005(**2.5 µs**)。`tcp` は `waypoints` ≤ 0.02 m、`speed_logits` の期待値 ≤ 0.05 m/s も |
| 速さ | 別のプロセスで 100 回の空回しの後 600 回の `run_numpy`(走行と同じ呼び方)の中央値 ≤ **6.0 ms**、p99 ≤ 8.0 ms |
| 学習のメモリ | `result.json` の `peak_mem_mb` ≤ 4,096 |
| 走行の確認 | `drive.py` と同じ手順(`check_io`・空回し・基本の5つを渡す)でエンジンを読める(`engine-check`) |
| 数値 | 保留の 10,985 フレームで NaN・Inf なし |

### 5.2 主指標

**L(r) = 保留した人のきれいな走行の L1 [µs]**: 学習結果 r の best.pth(PyTorch FP32)の予測と人の操作の差の絶対値を、10 ステップ × [ステア, スロットル] で
平均し、フレームで平均したもの。フレーム = ai_mix の val のうち人の 11 シーン(clean_driving の val と同じ中身)、全フレーム(stride 1)、
`future_valid & vehicle_valid`、水平 2.5G の衝突の前後を除く(4,670 フレーム)。`evaluation.json` の `human_clean_l1_us` と同じ定義(モデル B = 49.14 µs)。
`tcp` は操作の枝(`actions`)で計算する(走行の既定 `tcp_mode = 0` の出力と同じ)。

- 方策 P の値: L̄_P = 種の平均、範囲 [min_P, max_P]。
- 基準 R = `jetformer_v0` の種 0・1(同じコードで学習したもの)。モデル B は別の行に出す(旧コード・種なし・途中で再開)。
- ばらつきの幅 **δ = max(10 µs, {L(B), L(v0 種0), L(v0 種1)} の範囲)**。
- Δ_P = L̄_P − L̄_R。
  - **良い**: Δ_P ≤ −δ かつ max_P < min_R
  - **悪い**: Δ_P ≥ +δ かつ min_P > max_R
  - それ以外は **同等(区別できない)**
- チェックポイントの確認: 同じ計算を last.pth でもする。best.pth と last.pth で分類が変わる方策は「良い」と言わない(「同等」に下げる)。

### 5.3 副指標(全部の学習結果で計算。数字は報告に全部載せる)

計算は `rl_evaluate.py` の関数(`drive_outputs`・`heldout_passes`・`_pass_values`・`entry_estimate`・`_erpm_map`・`straight_mask`・
`cluster_bootstrap`・`bootstrap_mean`)と `compare_models.metrics` をそのまま使う。走行の出力はゲート D の固定設定(ステアのゲイン 1.3・速度のゲイン 1.0・
先読み 3・トリム 0、各学習結果の `steer_center_us`・`throttle_neutral_us` の周り)。

| 名前 | 中身 | 「良い」 | 「悪い」 |
|---|---|---|---|
| R1 | 保留の R1 の通過 17 本(AI 4 本)で、窓(起点の 0.4 秒前〜起点)の出力のスロットルの平均。方策の値 = 種ごとの値の平均。d = P − R の通過ごとのブートストラップ(2,000 回、種 0)の 95% 区間 | 区間の下限 > 0(進入が遅い)かつ **2つの種の両方で** d の平均 > 0 | 区間の上限 < 0 |
| DE | 保留の DE の通過 14 本(AI 4 本)、緑マット出口〜0.8 秒の出力のステア。d = P − R(+ = 左) | 同上 | 同上 |
| 崩れ | 保留の直線(542 フレーム)の前進の強さ(1498 − 出力のスロットル)の低下が R の 5% 以内、前進の偏り(3 フレーム先の予測 − 人)≥ −30 µs、向きの正解率 ≥ R − 0.01(どれも点推定) | — | どれか1つでも外れたら「崩れ」 |
| ずれへの強さ | clean のフレームのうち 3 フレームに1つ(フレーム番号 % 3 == 0、約1,557)で、5 つのずれ(画像を 3 行下へ・3 行上へ(端の行を複製)、明るさ ×0.8・×1.2(0〜255 に切る)、ToF を 1 行下へ(上の行は無効 0))の L1 の増え方の平均 ΔL1 | ΔL1 が R の 0.8 倍以下、かつ2つの種の両方で R 以下 | — |
| 参考 | ジャンル別(4・5)の L1、k1/k3/k10 のステア・スロットル、交代の後の人の修正(intervention_val、ログ単位で保留の 218 フレームは別に)、AI のフレームとの一致、速さ、パラメータ数 | 判定に使わない | |

`tcp` の追加(判定には使わない。§5.5): 通過点の ADE/FDE [m]・目標速度の誤差 [m/s](clean_driving の val のラベル、`wp_valid` のフレーム)。

### 5.4 判定の規則

1. 候補 C = v0 以外で、ゲートを満たし、主指標が「悪い」でなく、「崩れ」でなく、R1・DE のどちらも「悪い」でない方策。
2. C の中に主指標が **「良い」** のものがあれば、ベスト = その中で L̄ が最小のもの(L̄ の差が δ/2 未満なら `run_numpy` の速い方)。
3. 2 が無く、C の中に **「同等」で R1・DE のどちらかが「良い」** ものがあれば、ベスト = {R1 良い, DE 良い, ずれに強い} の数が多いもの
   (同じなら L̄ が小さい方 → 速い方)。報告では「机上の候補(L1 は同等、R1/DE で良い)」と書く。
4. どちらも無ければ、**ベスト = `jetformer_v0`(モデル B のまま)**。「机上では v0 より良いと言えない」と書く。
5. どの場合も、全方策の L̄ と範囲・δ・副指標を表に出す(順位は L̄ の順。δ の中の順位には意味が無いと明記)。
6. O1(3本目の種)を回したときは、3本で 1〜5 をやり直し、3本の結果を採る(δ も v0 の4本(B + 種 0〜2)の範囲)。
7. 机上の結論は実車の改善を示さない。ベストが v0 以外なら、§5.7 のカードで実車の A/B/A をするまで走行の標準はモデル B のまま。

### 5.5 `tcp` の追加の評価(参考。判定は 5.4 の操作の枝で行う)

保存した予測(`actions`・`waypoints`・`speed_logits`)から、CPU で走行と同じ `TCPPost` を **シーンごとに時刻の順に**(フレームが 2 つ以上あいたら `reset`)回す。
v_now は aligned の `vehicle[:, 3]`(eRPM)。比べる出力:

| 出力 | `tcp_mode` | `speed_cap` |
|---|---|---|
| control | 0 | 0(= actions と同じ) |
| control + 上限 | 0 | 2.75 m/s(R1 の進入の上限 5500 eRPM) |
| hybrid | 1 | 0 |
| hybrid + 上限 | 1 | 2.75 |
| trajectory | 3 | 0 |

出すもの: 各出力の L1(スロットルは L1 で不利になるのが分かっている: 正解の速さを入れても 51.7 µs、`design_research.md` §4.6)、R1 の窓の出力のスロットル
(上限ありとなしの差の通過ごとのブートストラップ)、直線の前進の強さ(上限なしの hybrid が control より 5% 以上弱くならないか)、v_hat の誤差。
「速度の上限が使える」と書いてよいのは、hybrid + 上限 または control + 上限 で R1 の窓のスロットルの区間の下限 > 0(弱まる)、かつ上限なしの直線の前進の強さの低下が 5% 以内のとき。
**机上で言えるのは「指令がどう変わるか」まで**(ログの車は指令に反応しない。上限の効き目は実車でしか分からない)。

### 5.6 ばらつきの書き方

- 報告の最初の表に、v0 の3本(B・種0・種1)の L とその範囲を出し、「この幅より小さい差は区別できない」と書く。
- 種の平均だけでなく、各種の値を全部出す。p 値は使わない(2本では意味が無い)。
- best.pth は検証(同じ 11 シーン)で選んだエポックなので、主指標は全方策で同じだけ楽観的。last.pth の値も並べる。
- 保留の R1・DE は 17・14 通過しかない。区間が広いのは当然で、「良い」が出なくても「差が無い」とは言えない、と書く。

### 5.7 実車の A/B/A カード(この作業では走らせない。M が人と行う)

```
目的     : 机上のベスト(候補 X)が、モデル B より実車で良いか
前提     : X が §5.1 のゲートに合格。エンジンはこの Jetson で作ったもの。推論の確認(dry run)を 60 秒以上、推論による中立 0 回
走らせる順: A(モデル B)→ B(X)→ A(モデル B)。各ブロック 5 周(または 3 分)、同じコース・同じジャンル(まず 5 低速走行。合格したら 4)
同じにするもの: 走行のパラメータ(ステア・速度のゲイン、先読み、トリム、ToF ガード = 1 記録だけ、リカバリー オン、後退の前に中立 オン、
          重ね合わせ オフ)。X が tcp なら tcp_mode = 0・speed_cap = 0。電池は各ブロックの前に電圧を記録し、
          A と B で同じ状態にそろえる(下がっていたら替えてから次のブロック)。
          照明・時間帯。収集 オン(AI_CMD に方策・model_pred が残る)
安全     : 人が送信機を持ち、CH3 でいつでも取り返す。速度ゲインは低い方から。安全の仕組みは1つも切らない。
          異常(連続の衝突・制御の発散・推論の中立が続く)があれば、その場で A に戻して終わり
記録する : 周回数とラップタイム(LapCounter)、1周あたりの交代(CH3)・衝突(水平 2.5G 以上)・リカバリー、R1 の進入 eRPM の中央値、
          中立(hold)の回数と理由、infer_ms・latency_ms
良いと言える: B の「1周あたりの交代 + 衝突」が両方の A より少なく、ラップタイムが A の平均の 5% より遅くない
別のカード: tcp の hybrid・速度の上限(speed_cap 2.75 m/s)は、A/B/A に合格した後に別に(A = tcp_mode 0、B = tcp_mode 1 + 上限)
```

---

## 6. 管理画面(持ち主 I。`server.py` の API の追加も I)

### 6.1 学習タブ

- `index.html` 220 行(`#tr-ds-info`)の後: `<label>モデルの種類<select id="tr-policy"></select></label>`・`<div id="tr-policy-info" class="small">`・
  `<div id="tr-policy-opts">`。
- 選択肢は `GET /api/policies`(`policies.list_specs()`、torch なし)から: 表示名、`status = experimental` なら「(試験的)」、`ready = false` なら選べず理由を出す
  (例「DINOv2 の重み data/pretrained/dinov2_vits14_pretrain.pth がありません」)。既定は `jetformer_v0`。
- 説明欄: `description`、推論の目安 `hints.infer_ms` ms、パラメータ数 `hints.params_m`、`requires_pretrained` の注意。
- オプション欄: `options` から作る(int → 数値、bool → チェック、choice → 選択)。送るのは既定から変えた項目だけ(`policy_cfg`)。
- 詳細(`<details>`): 「衝突とみなす加速度」`#tr-impact-g`(4.0 = 既定 / 2.5 = モデル B と同じ)、「乱数の種」`#tr-seed`(空 = 決めない)。
  237 行の文言「水平4G以上」は「水平の加速度が『衝突とみなす加速度』以上」に変える。
- `updateEstimate`(829): `TRAIN_RATE`・`VAL_RATE`(817)の代わりに選んだ方策の `hints.train_rate`・`val_rate`。
- 開始(847): body に `policy`・`policy_cfg`・`impact_g`・`seed`。`server.py:670` は `body.get('policy', 'jetformer_v0')` などで `trainer.start` へ
  (古い画面のキャッシュからは今と同じ v0 の学習)。`ValueError` は 400。
- 学習結果の一覧(`loadRuns` 898): 方策の印(v0 は目立たない灰色、ほかは色つき `policy_title`)、`seed`、`impact_g` が 2.5 なら「衝突 2.5G」の印。

### 6.2 走行タブ

- モデルの選択肢(`loadDrive` 947): `「<policy_title>」<名前>(検証誤差 xx µs)`。`drivable = false` は `disabled` と理由。
- `#dr-model-info`(`renderDriveModel` 963): 「モデルの種類 <b>…</b> ・ 出力 actions[, waypoints, speed_logits] ・ 後処理: TCP の制御器 / なし」。
- **TCP の欄** `#dr-tcp`(ゲインの欄の後。`m.postprocess && m.postprocess.kind === 'tcp_controller'` のときだけ表示):
  - `#dr-tcp-mode`(選択): 0「操作の枝だけ(既定・比較で評価した出力)」/ 1「ハイブリッド(ステア = 操作の枝、スロットル = 目標速度 → 制御器)」/
    2「混ぜる(試験的)」/ 3「通過点を追う(試験的)」。変えたらすぐ `POST /api/drive/params`(`tof_guard` と同じ)。
  - `#dr-speed-cap`(0〜5 m/s、0.05 刻み): 0 は「上限なし」、ほかは「2.75 m/s(約 5500 eRPM)」のように eRPM(= v / 0.5e-3)も出す。
  - `#dr-speed-scale`(0.5〜1.5): 「モード 1〜3 だけで効く」と添える。
  - 走行中の表示 `#dr-tcp-live`(`renderDrive` 1083 で `st.now.post` から): 予測の目標速度 v_hat・上限の後 v_cmd・今の速さ v_now(m/s と eRPM)・
    曲がれる速さの目安 v_curv・出力を決めた枝。
  - `tcp_mode ≥ 1` で速度ゲインが 1.0 でなければ「速度ゲインは制御器の後にも掛かります(1.0 を勧めます)」と出す。
- **予測の重ね合わせ**(全モデル、詳細の欄の先読みの近く): `#dr-ens-j`(0〜3、0 = オフ)・`#dr-ens-m`(0〜2、0.1 刻み)。
- `DR_PARAMS`・`readDriveParams` に新しい項目を足す。走行の開始の body の意味は変わらないので **`DRIVE_CLIENT_VERSION` は 6 のまま**。

### 6.3 書き出し(エミュレーター用、`model_export.py`)

`io_spec` の出力の説明に `waypoints`(自車の座標 x 前・y 左 [m]、0.1 秒刻みで 1 秒先まで)と `speed_logits`(0〜4.5 m/s の 10 クラス、期待値 = 目標速度)を足し、
`postprocess` があれば「制御器は jetformer/policies/controller.py(numpy)。走行の既定(tcp_mode 0)は actions をそのまま使う」と書く。README にも1段落。

---

## 7. テスト(CPU だけ、1ファイル 30 秒未満、GPU・実データ・ネットなしで通ること)

実行: `cd ~/jetformer && CUDA_VISIBLE_DEVICES= nice python -m pytest -q tests/<ファイル>`。実データ(モデル B の ONNX など)を使うテストは、無ければ skip。

| ファイル(持ち主) | 中身 |
|---|---|
| `tests/test_policies.py`(F) | (1) 新しいプロセスで `import jetformer.policies, jetformer.policies.post, jetformer.policies.ensemble, jetformer.drive` の後 `'torch' not in sys.modules`。(2) `from_config({'num_modes': 6})` → v0・`{dim:256, layers:4, heads:8}`、`drivable` が True、知らない方策の `title_of` は名前のまま。(3) `resolve`: 知らないキー・UI からの options 外のキー・範囲外 → ValueError、型の変換、全部のキーを返す。(4) `build_policy('jetformer_v0', 6)` が `JetFormer` のインスタンスで、state_dict のキーが `JetFormer(num_modes=6)` と同じ。(5) v0 の `export_onnx`(tmp)の入出力が `image/tof/imu/state/mode`・`actions [1,10,2]`、モデル B の ONNX があれば同じ名前と形。(6) state・mode しか使わない小さな方策を登録簿に足す(monkeypatch)→ ONNX の入力が `state, mode` だけ、`check_io` 合格。(7) `check_io` の不合格: 知らない入力・形の違い・actions 無し・-1 の形。(8) `run_epoch`: 6個・6個+重み・6個+重み+dict・6個+dict のバッチ、`policy_loss` と `loss_parts` を持つ偽の方策で「学習は loss、報告は L1」、v0 型の偽物で今と同じ数字。(9) 既定の方策では `same_args` に `policy` が入らない、既定以外では入る。(10) `check_init_compat`: 方策・構造の設定が違えば SystemExit、`train_only_keys` の違いは通る。(11) `expand_modes`: 行 2・10 の合成の state、v0 では `train.expand_mode_state` と同じ結果。(12) `param_groups`: 凍結したパラメータが入らない、v0 は全パラメータの1群。(13) `load_state` が best・last の両方の形を読む、`load_run`(tmp の学習結果、v0)の予測が `JetFormer` + `load_state_dict` と同じ。(14) `JetFormerDataset(extra_targets=...)`(tmp の合成データセット)の最後が `y_` の dict、無いシーンは KeyError、渡さなければ今と同じ6個。(15) `TrainingManager.start`(`subprocess.Popen`・空きメモリを monkeypatch)で、v0・引数なし → **今とまったく同じコマンド**、`act` → `--policy act`、`impact_g=2.5`・`seed=0` → `--impact-g 2.5 --seed 0`、準備できていない方策 → ValueError、名前に slug。(16) `build_engine` の設定の読み方(`engine.precision` → fp16/bf16/fp32)を関数にしてテスト |
| `tests/test_policy_post.py`(F) | 重ね合わせ: `j_max = 0` で入力と同じ、新しいほど重い、時刻で j を数える(フレームの抜け)、`j_max` より古いものは使わない、`reset`。`post.make`: postprocess 無し → None、知らない kind → None と説明、偽の kind(monkeypatch `KINDS`)→ そのオブジェクト |
| `tests/test_policy_act.py`(P-act) | 既定の大きさ 15.6M ± 0.3M、forward [2,10,2]、ジャンルの選択(one-hot を変えると heads の該当の行だけが効く)、`expand_modes`(`rows=2`)で増やしたジャンルが元と同じ出力、CVAE: 学習中に target ありで KL > 0・`loss_parts['kl']`、eval では target の有無で出力が同じ(z = 0)、`USES_TARGET` は cvae のときだけ、小さい設定で ONNX に書き出せて入出力が約束どおり・禁止の演算が無い(§8) |
| `tests/test_policy_tcp.py`(P-tcp) | 出力の形(actions・waypoints・speed_logits)、`cumsum` の op と matmul が同じ、two-hot(2.3 → 2.0 に 0.4・2.5 に 0.6、4.6 → 最後のクラス 1.0、−0.1 → 最初 1.0)、`policy_loss` が無効のフレームの通過点・速度の損失を 0 にする、書き出しで3つの出力。ラベル: 合成のログ(左へ一定のヨーレート・一定の速さ)で通過点が円の上・y > 0、まっすぐなら y ≈ 0、後退・シーンの終わり・IMU の切れ目・衝突・ジャイロの飽和で無効、キャッシュの作り直し(splits の sha1)。較正: 合成のデータで傾きが戻る、表が単調、逆引きが表の外で端に張り付く |
| `tests/test_policy_tcp_controller.py`(P-tcp) | 左へ曲がる通過点 → ステア > 中心、v_cmd > v_now → スロットル < 中立、v_cmd < 0.3 → 中立、v_now 0.5・v_cmd 0 → 中立より上を出さない、ブレーキは v_now > 1.0 のときだけ、上限を下げるとスロットルの µs が減らない(前進が強くならない)、上限を外すと cap_eff が 1.5 m/s² で上がる、`tcp_mode 0`・上限 0 で actions とビット単位で同じ、hybrid のステアの行 = actions、fusion の過去 1 秒の判定、道が短いと fallback、乱数の入力 200 通りで出力が有限、1回の時間を表示(参考) |
| `tests/test_policy_msfusion.py`(P-msf) | 既定の大きさ 18.9M ± 0.5M、forward [2,10,2]、詰め物の後の各段の格子が割り切れる(画像 7×7・ToF 6×7)、`scales` を減らすと交換の層が減る、`head='pooled'`、小さい設定で ONNX に書き出せて `AdaptiveAveragePool` 系・antialias が無い |
| `tests/test_policy_dinov2.py`(P-dino) | 公式と同じキーと形の合成の重み(`mask_token`・`pos_embed [1,1370,384]` を含む)を tmp に保存して `load_official` が `strict=True` で通る・`pos_embed` が [1,257,384]、凍結: ViT の `requires_grad` が False・`param_groups` に入らない、`finetune_last=2` で2群(学習率 ×0.1)、`model.train()` の後も ViT が eval、`build(pretrained=False)` が重みのファイルを開かない(monkeypatch)、`pretrained=True` で重みが無ければ分かる例外、トークン数 111(pool 2)・303(pool 1)、`vit_depth=1` で ONNX に書き出せる |
| `tests/test_drive.py`(I、末尾に足す) | 最小の古い設定の `model_info` に `policy = jetformer_v0`・`drivable`、`inputs` に知らない名前 → `drivable = False` で `Driver.start` が ValueError、`_loop` を今の位置の引数のまま呼べる、偽の runner が追加の出力を返し偽の後処理(monkeypatch `post.KINDS`)を通すと、先読みに渡る値が後処理の出力・記録に `model_pred` と `post_mode`、`hold` で後処理が reset、`ensemble_j = 0` で今と同じ出力、偽の runner が基本の5つを全部受け取る、新しいパラメータの丸めと古い設定ファイル |
| `tests/test_policy_ui.py`(I) | `GET /api/policies` の中身(関数を直接呼んでもよい)、学習の開始が `policy`・`policy_cfg`・`impact_g`・`seed` を `trainer.start` に渡す(monkeypatch)、古い body は v0、`io_spec` の tcp の出力の説明 |
| `tests/test_policy_zoo.py`(E) | 計画の名前と順、`zoo_rules.json` は1回だけ書き中身が違えば止まる、判定の関数(合成の数字で 良い・同等・悪い、δ が v0 の範囲から決まる、ゲートの不合格、best/last で分類が変わると下がる、副指標の「良い」は2つの種の両方が要る、O1 の3本の判定)、ずれの関数(形・端の複製・明るさの切り詰め・ToF の無効)、種をまとめた通過ごとの値、`GPU_PATTERNS` に新しいコマンド、tcp の順の評価が `TCPPost` を使う(制御器が無ければ skip) |

既存のテスト(`tests/` 全部)は M が段階3で通す。各持ち主は自分のテストと、関係する既存のテスト(F は `test_rl_policy_training.py`・`test_rl_review.py`・`test_ai_dataset.py`、
I は `test_drive.py`・`test_model_export.py`、E は `test_rl_evaluate.py`・`test_offline_rl.py`)を通してから終える。

各 P の終わりの確認(テストに加えて): `CUDA_VISIBLE_DEVICES= nice python tools/train.py --policy <P> --data data/datasets/clean_driving_20261011
--out <スクラッチのフォルダー> --epochs 1 --max-steps 2 --batch 4 --workers 0 --stride 30 --no-command-input --impact-g 2.5` が通り、
`jetformer.onnx`・`model_config.json`(§2.5 の欄)ができる(F が終わっていること。`data/runs/` には書かない = 管理画面の一覧に出さない)。

---

## 8. やらないこと

- **車を走らせない。PCA9685・車両の基板に出力しない。再起動しない。** `jetformer-web` は M が言うまで再起動しない(管理画面の変更は再起動まで反映されないので、テストで確かめる)。
- GPU を2つのプロセスで同時に使わない(学習とエンジンの作成も同時にしない)。段階2までは GPU を使わない。バッチ ≤ 64、workers ≤ 4、`nice`。
- git commit しない(M がレビューしてから)。
- 既存の動きを変えない: `tools/train.py` を引数なしで動かしたときの学習・ファイル、`model_config.json` の今の欄、ファイル名、`PARAM_LIMITS` の今の項目、
  安全の仕組み(ゲイン・トリム・先読み・補助・ToF ガード・ラップ・CH3・センサーの途切れで中立)、`DRIVE_CLIENT_VERSION`、`benchmark_model.py` の既定。
- 古い学習結果・モデル B のファイルを書き換えない。`data/rl/ai_mix_20261011/eval/` の `heldout.json`・`evaluation.json`・`pred_*.npz` を書き換えない(比較は別のフォルダー)。
- 検証の 11 シーン・保留のフレームで、損失の重み・制御器の定数・エポック数・チェックポイントを調整しない。結果を見てから規則(§5)・種・エポックを変えない。
  1つの方策だけエポックを増やさない。
- 予測の重ね合わせ・速度の上限を、比較の判定に入れない(参考の行だけ)。
- ONNX: `dynamo=True` にしない、`adaptive_avg_pool2d` を割り切れない大きさで使わない、antialias の縮小を使わない、`nn.GRU`(層)・ループ・データで変わる分岐・
  可変の形を使わない、MHA の高速経路を切らずに書き出さない、入力を基本の5つ以外に増やさない(この作業では履歴を使わない)。
- 走行で基本の5つより少ない入力を渡さない(エンジンが使わない入力は読まれないだけ)。後処理を ONNX の中に入れない。
- 制御器に積分の項を入れない。制御器が自分で作るスロットルで、中立より上(ブレーキ・後退側)を §3.2.4 のブレーキの規則のほかに出さない
  (操作の枝がもともと出す値はそのまま通す。後退の手順は今までどおり補助(`assist`)に任せる)。
- DINOv2 の重みを学習・管理画面から自動でダウンロードしない。DINOv2 の特徴をキャッシュしない(色の揺らぎと両立しない)。
- TCP のラベルをデータセットのフォルダーの中に書かない(ダウンロード・エクスポートに入る)。
- `tof_guard` の `K_LEFT`(データの傾きより約3割小さい)をこの作業で直さない(別の課題)。
- `tools/rl_policies.py` の「方策」(オフライン RL)と、この登録簿を混ぜない。`experiments/` の過去の記録は変えない。

---

## 9. 未解決・リスク

- **L1 の差は出ない見込みが高い**(§0.1 の 6)。δ = 10 µs は 49 µs の 20%。「良い」が出なくても、それが一番ありそうな結果。
- `tcp` の速度の上限の効き目は机上では測れない(指令の変化だけ)。走行の既定は `tcp_mode = 0` なので、`tcp` を選んでも上限を使わなければ動きは操作の枝そのもの。
- `dinov2` は `run_numpy` 5.06 ms で 6 ms の予算に近い(揺れ ±0.6 ms)。FP16 の精度がゲートを満たさず BF16・FP32 になると予算を超えうる。
  走行ループ全体は今 5.6 ms → 約8.3 ms の見込み(30fps の 33 ms には入る)。
- `msfusion` の ToF は LiDAR の鳥瞰図ではない(カメラとほぼ同じ向きの距離画像)ので、早い段の交換で新しく分かることは少ないかもしれない。
  補助タスク(深度など、TransFuser で一番効いたもの)はこの作業では入れない(将来の候補 `aux_depth`)。
- TCP のラベルの速さは eRPM からで空転を含む(±6%)。芝(GM)では曲率が小さく出る。較正は電池・タイヤ・路面で変わる(今日のデータだけ)。
- 保留の R1・DE は 17・14 通過で、区間は広い。best.pth は検証で選んでいる(全方策で同じ偏り)。
- 学習の時間は v0 の実測(34 秒/エポック)からの見込み。電源断で `--resume` すると乱数が戻らない。
- 管理画面からの学習は既定で衝突 4.0G・種なしなので、そのままではモデル B と比べられない(§6.1 の詳細の欄で 2.5G・種を選べるようにする)。
- 将来(この作業の外): フレームの履歴の入力(interfaces.md の段階2)、`msfusion` の補助タスク、DINOv2 のレジスタ版、v0 からの部分的な初期値、
  速度の上限を `tcp` 以外にも(制御器の `thr_ff` は較正だけで動くので、どの方策にも掛けられる)。
