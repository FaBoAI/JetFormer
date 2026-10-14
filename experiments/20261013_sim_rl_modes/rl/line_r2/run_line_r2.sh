#!/bin/bash
# 経路いろいろ r2 の学習(2026-10-14、学習の段 line r2。protocol_line_r2.json training のとおり)。
#   /home/jetson/bin/jf-heavy --mem 2G --lock rl --need-mb 1200 -- nice -n 15 bash rl/line_r2/run_line_r2.sh
# lock rl を取ってから動く(nice は jf-heavy の中で起動のときに付く。子プロセスも nice 15)。止めたあとも同じコマンドで last.pt から続く。
# 学習の中で 100 万ステップごとに検証の組 LN-V60-R2 + LN-FAM6(高速版、--protocol line_r2)を裏で評価する。
# 見張り: 800 万以降、続けて 3 つのチェックポイントが L1〜L6 を全部満たしたら止める合図を送る(training.early_stop)。
set -u
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/train/line/r2
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
source /home/jetson/miniconda/etc/profile.d/conda.sh
conda activate jetracer
cd "$EXP"
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$OUT/logs"
log() { echo "[$(date -Is)] $*" >> "$OUT/logs/chain.log"; }
log "lock rl を取りました(pid $$、nice $(nice))"
( cd /home/jetson/jetformer_reports/sim_rl_modes_20261013 && sha256sum -c protocol_line_r2.sha256 ) >> "$OUT/logs/chain.log" 2>&1 || { log "protocol_line_r2.json が固めたものと違うので止めます"; exit 3; }
if [ -f "$OUT/main/done.json" ] || [ -f "$OUT/main/early_stop.json" ]; then log "もう終わっています"; exit 0; fi
python -u rl/train_modes.py --run line --steps 20000000 --out "$OUT/main" --seed 44 --init line_r1_retry_5M --eval --tag line-r2 >> "$OUT/logs/main.log" 2>&1 &
PID=$!
log "学習を始めました(pid $PID)"
while kill -0 "$PID" 2>/dev/null; do
  sleep 60
  if res=$(python3 rl/line_r2/early_stop.py "$OUT/main"); then
    log "早めの終わりの条件を満たしました: $res"
    echo "{\"at\": \"$(date -Is)\", \"result\": $res}" > "$OUT/main/early_stop.json"
    kill -TERM "$PID"
    break
  fi
done
wait "$PID"; code=$?
log "学習 終了コード $code"
exit $code
