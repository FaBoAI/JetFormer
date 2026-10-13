#!/bin/bash
# RL の 1 本を jf-heavy(メモリ上限 2G・lock rl)の下で動かす。電源断のあとも同じコマンドで last.pt から続く(2026-10-13)
#   rl/run_rl.sh <名前> <variant> <seed> <steps> [train_rl.py の追加の引数...]
#   例: rl/run_rl.sh main-s0 main 0 15000000
set -u
NAME=$1; VARIANT=$2; SEED=$3; STEPS=$4; shift 4
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap
OUTROOT=/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/rl
mkdir -p "$OUTROOT/logs"
source /home/jetson/miniconda/etc/profile.d/conda.sh
conda activate jetracer
cd "$EXP"
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "[$(date -Is)] 開始 $NAME variant=$VARIANT seed=$SEED steps=$STEPS $*" >> "$OUTROOT/logs/$NAME.log"
/home/jetson/bin/jf-heavy --mem 2G --lock rl --need-mb 1200 -- \
  python -u rl/train_rl.py --variant "$VARIANT" --seed "$SEED" --steps "$STEPS" --out "$OUTROOT/$NAME" "$@" >> "$OUTROOT/logs/$NAME.log" 2>&1
code=$?
echo "[$(date -Is)] 終了 $NAME 終了コード $code" >> "$OUTROOT/logs/$NAME.log"
exit $code
