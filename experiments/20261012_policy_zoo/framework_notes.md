# 方策の比較: 枠組み(段階1)の実装メモ

作成 2026-10-12 ・ 対象 `/home/jetson/jetformer`(main 7239fea の作業ツリー。commit はしていない)・ GPU は使っていない(CPU だけ)

spec.md の §2(枠組み)を実装した。このタスクの指示に合わせて、spec で I(結合)と E(評価)の持ち主になっている
走行・管理画面・評価の道具の配線も入れた。ただし `tools/policy_zoo.py` は作っていない。4つの方策のモジュールは、まだ仮のもの。

## 1. できたもの

| ファイル | 中身 |
|---|---|
| `jetformer/policies/__init__.py`(torch なし) | 登録簿。`PolicySpec`(`resolve`・`changed`・`build`・`ready`・`describe`)、`get`・`names`・`list_specs`・`title_of`・`policy_of`・`from_config`・`inputs_of`・`outputs_of`・`check_io`・`drivable`・`build_policy`。5つの方策を §2.1.2・§3.0 の値で登録してある |
| `jetformer/policies/runtime.py`(torch) | `as_outputs`・`forward`・`l1_per_sample`・`sample_loss`・`loss_parts`・`param_groups`・`example_inputs`・`check_forward`・`ExportWrapper`・`export_onnx`・`read_onnx_io`・`load_state`・`load_run`・`expand_modes`・`predict`・`count_params`・`params_m` |
| `jetformer/policies/blocks.py` | `select_mode`・`mlp2`・`encoder`・`decoder`・`add_sensor_tokens`・`sensor_tokens`(v0 と同じ層の名前・同じトークン。テストで v0 の forward と一致を確かめた) |
| `jetformer/policies/jetformer_v0.py` | `JetFormer` をそのまま返す |
| `jetformer/policies/post.py`(numpy) | `make`・`KINDS`・**`Stage`**(後処理 → 重ね合わせ。走行と評価で同じ手順)・**`replay`**(保存した予測を時刻の順に Stage へ通す)・`PostError`・`speed_of` |
| `jetformer/policies/ensemble.py`(numpy) | `TemporalEnsembler`(§2.8) |
| `jetformer/policies/act.py`・`tcp.py`・`msfusion.py`・`dinov2.py` | **仮のモジュール**。`POLICY_STUB = True` の行がある間は `ready()` が「まだ実装されていません」を返す。管理画面では選べず、`tools/train.py` は出力のフォルダーを作る前に止まる |
| `tools/train.py` | `--policy`・`--policy-cfg`、`POLICY_ARGS`・`policy_args_used`・`norm_policy_cfg`・`check_init_compat`。`prepare`・`extra_targets`・`finalize` のフック、`check_forward`、`param_groups`、`hints.amp`。`run_epoch` は方策の損失で学習し、L1 を報告する。`model_config.json` と `result.json` は §2.5 の欄つきで、tmp に書いてから置き換える |
| `jetformer/train_data.py` | `JetFormerDataset(extra_targets=...)`。各サンプルの最後に `{'y_<キー>': Tensor}` を足す。渡さなければ今までと同じ |
| `tools/build_engine.py`・`jetformer/trt_runtime.py` | `engine.precision`(fp16・bf16・fp32)、`<engine>.tmp` に作ってから `check_io` → 置き換え。合わなければ終了コード 2。`build_engine(..., bf16=)`・`engine_io()`・`TRTRunner.shapes` |
| `jetformer/training.py` | `start(..., policy, policy_cfg, impact_g, seed)`。v0 で引数を変えなければ今と同じコマンド。名前に slug が入る。`start_args(body)` は server が使う。`list_runs` に `policy`・`policy_title`・`impact_g`・`seed` |
| `jetformer/drive.py` | `model_info` に方策の欄・`drivable`。`start` で走れないモデルを断る。エンジンを読んだ後に `check_io`。空回しの入力は `runner.shapes` から作る。`_loop` は `post.Stage` を通す(位置の引数は同じ)。`hold` で reset。`PARAM_LIMITS` に `speed_cap`・`speed_scale`・`tcp_mode`・`ensemble_j`・`ensemble_m`。`st.now.post` と `ens_j` |
| `jetformer/recorder.py` | `AI_CMD.npz` に `model_pred`・`post_mode`・`v_hat`・`v_cmd`・`ens_j`(古い行は既定値) |
| `jetformer/web/server.py`・`static/*` | `GET /api/policies`。学習タブ: モデルの種類・options・詳細欄(衝突の加速度・乱数の種)、方策の目安から見積もる。一覧に方策の印。走行タブ: 「方策名」の選択肢(走れないものは選べない)・モデルの情報・TCP の欄(出し方・速度の上限と eRPM・倍率・走行中の v_hat / v_cmd)・重ね合わせのスライダー。`DRIVE_CLIENT_VERSION` は 6 のまま |
| `jetformer/model_export.py`・`README.md` | `waypoints`・`speed_logits` の説明・制御器の注記と、README の1段落 |
| 評価の道具 | `compare_models.py`・`evaluate.py`: `runtime.load_run` → `runtime.predict` → `post.Stage`(走行の既定のパラメータ。`compare_models --post-params` で変えられる)。`rl_evaluate.py`・`rl_policies.py`・`offline_rl.py`: 読み込みを `runtime.load_run` にした(`offline_rl features` は `embed` の無い方策では止まる)。`rl_policies.GPU_PATTERNS` に4つ足した。`benchmark_model.py` に `--policy`・`--policy-cfg`・`--num-modes`・`--json`(既定は今と同じ 8) |

## 2. 後方互換で確かめたこと(CPU)

- `tests/fixtures/v0_reference.npz` は変更の**前**のコードで作った(`tests/data/` は .gitignore の `data/` で除外されるため `fixtures`)。乱数の種 1234 の v0 の重みの和と、決めた入力への出力が入っている。登録簿から作った v0 で、重みの和(1e-9)と出力(1e-6)が一致した(`test_v0_outputs_match_reference_before_the_change`)。
- モデル B の `best.pth` を `runtime.load_run` で読んだ予測(検証の実フレーム 4 個)は、`JetFormer` + `load_state_dict` と最大差 0 で同じだった。`drive.model_info(モデル B)` は `policy = jetformer_v0`・`drivable = True`、走行の一覧は 11 本で、今までと同じ。
- 実データの CPU 試し学習(v0、2 ステップ)で書き出した ONNX は、モデル B と比べてノード数(427)・演算の並び・初期化子の名前が同じ。入出力の名前と形も同じだった。
- 管理画面の学習: v0 を既定の引数で始めたときのコマンドは、今と完全に同じ(テストで文字列を比べている)。前からの `last.pth` の `args` には `policy` が入らないので、`--resume` の比較は変わらない。
- 全テスト: **261 passed, 2 skipped(CPU、約 50 秒)**。変更前は 202 passed, 2 skipped。新しいテストファイルはどれも 1 本 20 秒未満。

## 3. spec からの追加・変更(M の判断用)

1. `post.Stage` と `post.replay` を足した。走行の `_loop` と `compare_models`・`evaluate` は同じ `Stage` を使う(spec §2.9 の擬似コードを1つのクラスにまとめた)。後処理の例外は `PostError` で、走行は中立にする。
2. `ensemble_j = 0` のときは重ね合わせの履歴を貯めない。途中でオンにすると、その時から貯め始める。
3. `ready()` は、モジュールのファイルの先頭 4 KB に `POLICY_STUB = True` があれば「未実装」とする。**P の担当者は実装を終えたらこの行を消す。**
4. `hints.engine_s`(エンジンの作成の秒数、bench の実測)を足した。学習タブの見積もりに使う。
5. `runtime.predict` は `EXPORT_OUTPUTS` の出力だけを返す。エンジンの `run_numpy` と同じ形にそろえるため。
6. v0(`type(model) is JetFormer`)は包まずに書き出す。今までの ONNX と同じグラフにするため。ほかの方策は `ExportWrapper` を使う。
7. server の学習の開始の引数は `training.start_args(body)` に移した。テストで server を import しなくて済むようにするため(server を import すると、センサーやスピーカーの物を作る)。
8. `--init` は `runtime.load_state` で読む。`last.pth` の形も読めるようになった。

## 4. P・I・E の担当者へ(使い方)

- 方策のモジュール: `build(num_modes, pretrained=False, **cfg)`。キーは登録簿の `defaults` の全部。forward はキーワードで呼ばれる。`act`・`tcp` は `blocks.add_sensor_tokens(self, dim, num_modes, pretrained)` と `blocks.sensor_tokens(self, ...)` を使う。
- 試し学習: `CUDA_VISIBLE_DEVICES= nice python tools/train.py --policy <P> --data data/datasets/clean_driving_20261011 --out <スクラッチ> --epochs 1 --max-steps 2 --batch 4 --workers 0 --stride 30 --no-command-input --impact-g 2.5`
- `tcp`: モジュールの関数 `prepare(data_root, cfg)`・`extra_targets(data_root, cfg)`(`{シーン: {'waypoints','v_tgt','wp_valid'}}` → データセットの `y_waypoints` など)・`finalize(run_dir, cfg, data_root)`(`{'postprocess': {...}}` を返す)。制御器は `jetformer/policies/controller.py` の `TCPPost`(`post.KINDS` が読む)。読めないときは、走行も評価も操作の枝だけになる。
- `dinov2`: `train()` の上書きと `param_groups(lr)` は `runtime.param_groups` が使う。`requires` の重みが無いと `ready()` が False になる。
- 評価: `runtime.load_run(run_dir, ckpt)` → `runtime.predict(model, 5つ)`。後処理は `post.Stage(cfg, ref)` か `post.replay(...)`。`policy_zoo.py` の GPU の小コマンドは、もう `GPU_PATTERNS` に入っている。

## 5. 残り・確かめていないこと

- **GPU が要るので確かめていない**: `build_engine.py` の `engine_io()`(TensorRT でエンジンを読み直す所)と BF16 のフラグ、走行の `check_io`(本物の `TRTRunner.shapes`)。偽の TensorRT で流れ(tmp → 置き換え、終了コード 2)はテストした。段階4の最初に、v0 の学習結果で `build_engine.py` を一度動かして確かめること。
- `tools/policy_zoo.py`(E)はまだ無い。
- 4つの方策の中身(P)、`controller.py`・`tcp_labels.py`・`vit.py` もまだ。
- `jetformer-web` は再起動していないので、管理画面の変更はまだ動いているサーバーに出ていない。画面は、モックのサーバー(一覧の API は新しいコード、ほかの GET は本物へ転送、変更の要求は断る)とヘッドレスの Firefox で目視した。学習タブ・方策の options・走行タブの TCP の欄が出ることを見た。
