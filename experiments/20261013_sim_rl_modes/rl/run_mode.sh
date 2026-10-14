#!/bin/bash
# 強化学習モードの学習 1 本を jf-heavy(メモリ上限 2G・lock rl・nice 15)の下で動かす。電源断・止めたあとも同じコマンドで last.pt から続く(2026-10-13)
#   rl/run_mode.sh <名前> <run: no_wall|recover|cond|obstacle> <steps> [train_modes.py の追加の引数...]
#   例: rl/run_mode.sh no_wall-main no_wall 10000000 --eval
# nice はスレッドごとに効くので、起動のときに nice -n 15 で始める(あとから renice しない)。lock rl でベストラップの作業の学習と順番待ちになる。
set -u
NAME=$1; RUN=$2; STEPS=$3; shift 3
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
OUTROOT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/rl
mkdir -p "$OUTROOT/logs"
source /home/jetson/miniconda/etc/profile.d/conda.sh
conda activate jetracer
cd "$EXP"
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "[$(date -Is)] 開始 $NAME run=$RUN steps=$STEPS $*" >> "$OUTROOT/logs/$NAME.log"
/home/jetson/bin/jf-heavy --mem 2G --lock rl --need-mb 1200 -- nice -n 15 \
  python -u rl/train_modes.py --run "$RUN" --steps "$STEPS" --out "$OUTROOT/$NAME" "$@" >> "$OUTROOT/logs/$NAME.log" 2>&1
code=$?
echo "[$(date -Is)] 終了 $NAME 終了コード $code" >> "$OUTROOT/logs/$NAME.log"
exit $code
