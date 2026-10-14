#!/bin/bash
# 経路いろいろ(line)の作り直し(protocol allowed_tuning.retry)を、cond の本番(speed の段の train/speed/cond-main)のあとに lock rl を持ったまま行う。
# 2026-10-14、学習の段 line。
#   jf-heavy --mem 2G --lock rl --need-mb 1200 -- nice -n 15 bash rl/line_retry.sh
# 1. cond-main が終わっている(done.json)ことを確かめる(終わっていなければ何もしないで終わる = 終了コード 3)
# 2. 検証の組(evals/line)から protocol selection で line のチェックポイントを選ぶ(rl/select_line.py)→ train/line/select_main.json
# 3. 選んだ方策で試験の組(LN-P60・LN-FAM6、元の sim-core.js)→ train/line/eval_test_main.json(protocol selection.test_runs の 1 回目)
# 4. 合格なら作り直さない(終了コード 0)。不合格なら作り直し: 種 42(= 41 + 1)・調整は train/line/tuning_retry.json(modes.json tuning_ranges の中)・
#    500 万ステップ(modes.json line.training.retry_steps_max)・初期化は 2 で選んだ本番のチェックポイント・検証の組つき → train/line/cond-retry/
#    止めたあとも同じコマンドで last.pt から続く(2・3 は結果のファイルがあれば飛ばす)
set -u
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/train/line
MAIN=/home/jetson/jetformer_reports/sim_rl_modes_20261013/train/speed/cond-main
source /home/jetson/miniconda/etc/profile.d/conda.sh
conda activate jetracer
cd "$EXP"
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
log() { echo "[$(date -Is)] $*" >> "$OUT/retry.log"; }
log "lock rl を取りました(pid $$、nice $(nice))"
if [ ! -f "$MAIN/done.json" ]; then log "cond-main がまだ終わっていないので何もしません"; exit 3; fi
if [ ! -f "$OUT/select_main.json" ]; then
  python rl/select_line.py "$MAIN" > "$OUT/select_main.json" || { log "選び方が失敗"; exit 4; }
fi
POL=$(python -c "import json;print(json.load(open('$OUT/select_main.json'))['selected']['policy'])")
log "本番の選んだ方策: $POL"
if [ ! -f "$OUT/eval_test_main.json" ]; then
  log "試験の組(元のコア)を始めます"
  node rl/eval_modes.mjs --mode line --policy "$POL" --set test --core original --jobs 2 --out "$OUT/eval_test_main.json" >> "$OUT/retry.log" 2>&1 || { log "試験が失敗"; exit 5; }
fi
PASS=$(python -c "import json;print(json.load(open('$OUT/eval_test_main.json'))['all_judged_pass'])")
log "本番の試験の組: all_judged_pass=$PASS"
if [ "$PASS" = "True" ]; then log "合格なので作り直しはしません"; exit 0; fi
CK=${POL/\/policies\//\/checkpoints\/}; CK=${CK%.json}.pt
[ -f "$CK" ] || { log "チェックポイント $CK がありません"; exit 6; }
[ -f "$OUT/tuning_retry.json" ] || { log "tuning_retry.json がありません"; exit 7; }
log "作り直しを始めます: 種 42・500 万ステップ・初期化 $CK・調整 $(cat "$OUT/tuning_retry.json")"
echo "[$(date -Is)] 開始 cond-retry run=cond steps=5000000 seed=42 init=$CK tuning=$(cat "$OUT/tuning_retry.json")" >> "$OUT/cond-retry.log"
python -u rl/train_modes.py --run cond --steps 5000000 --seed 42 --init "$CK" --tuning "$OUT/tuning_retry.json" --eval --tag retry \
  --out "$OUT/cond-retry" >> "$OUT/cond-retry.log" 2>&1
code=$?
echo "[$(date -Is)] 終了 cond-retry 終了コード $code" >> "$OUT/cond-retry.log"
log "作り直し 終了コード $code"
exit $code
