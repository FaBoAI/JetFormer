#!/bin/bash
# 経路いろいろ r4 の学習(2026-10-14、line r4 の段。protocol_line_r4.json training のとおり)。
#   /home/jetson/bin/jf-heavy --mem 2G --lock rl --need-mb 1200 -- nice -n 15 bash /home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/line4/train/run_line_r4.sh
# lock rl を取ってから動く(jf-heavy の中で nice 15。子プロセスも nice 15)。止めたあとも同じコマンドで last.pt から続く。
# 100 万ステップごとに検証の組 LN-V60-R4 + LN-VHEAD40-R4 + LN-VCORNER-R4 + LN-VBOX40-R4 + LN-FAM6(高速版・シールド v5、eval_line_r4.mjs、2 本)を裏で評価する。
# 見張り: 200 万以降、続けて 2 つのチェックポイントが L1_r4〜L10 を全部満たしたら止める合図を送る(training.early_stop)。
# 止めたあと(または 500 万を終えたあと)、評価の残ったチェックポイントを評価する(選び方の候補は 100 万ごとの全部 + 初期化元)。
set -u
REP=/home/jetson/jetformer_reports/sim_rl_modes_20261013
OUT=$REP/r3/line_r4/train
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
L=$EXP/r3/line4
STEPS=${STEPS:-5000000}
source /home/jetson/miniconda/etc/profile.d/conda.sh
conda activate jetracer
cd "$EXP"
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 JF_TMP=/tmp
mkdir -p "$OUT/logs"
log() { echo "[$(date -Is)] $*" >> "$OUT/logs/chain.log"; }
log "lock rl を取りました(pid $$、nice $(nice))"
( cd "$REP/r3" && sha256sum -c protocol_line_r4.sha256 ) >> "$OUT/logs/chain.log" 2>&1 || { log "protocol_line_r4.json が固めたものと違うので止めます"; exit 3; }
if [ ! -f "$OUT/main/done.json" ] && [ ! -f "$OUT/main/early_stop.json" ]; then
  python -u "$L/train/train_line_r4.py" --run line_r4 --steps "$STEPS" --out "$OUT/main" --seed 46 --init line_r3_5M --eval --tag line-r4 \
    --lr-start 4e-5 --lr-end 2e-5 >> "$OUT/logs/main.log" 2>&1 &
  PID=$!
  log "学習を始めました(pid $PID、予定 $STEPS ステップ)"
  while kill -0 "$PID" 2>/dev/null; do
    sleep 60
    if res=$(python3 "$L/train/early_stop_r4.py" "$OUT/main"); then
      log "早めの終わりの条件を満たしました: $res"
      echo "{\"at\": \"$(date -Is)\", \"result\": $res}" > "$OUT/main/early_stop.json"
      kill -TERM "$PID"
      break
    fi
  done
  wait "$PID"; code=$?
  log "学習 終了コード $code"
else
  log "学習はもう終わっています"
fi
# 評価の残ったチェックポイントを評価する(1 つずつ、高速版、2 本)
for p in "$OUT"/main/policies/step_*.json; do
  [ -f "$p" ] || continue
  e="$OUT/main/evals/line/$(basename "$p")"
  if [ ! -f "$e" ]; then
    log "残りの検証: $(basename "$p")"
    node "$L/eval_line_r4.mjs" --set validation --core fast --jobs 2 --policy "$p" --out "$e" > "${e%.json}.log" 2>&1 || log "検証が失敗: $p"
  fi
done
log "終わり"
