#!/bin/bash
# 離脱復活(recover)の学習 1 本を jf-heavy(メモリ上限 2G・lock rl・nice 15)の下で動かす(2026-10-13、学習の段 recover)。
# rl/run_mode.sh と同じ包みで、出力だけ ~/jetformer_reports/sim_rl_modes_20261013/train/recover/<名前> に置く。止めたあとも同じコマンドで last.pt から続く。
#   rl/run_train_recover.sh <名前> <steps> [train_modes.py の追加の引数...]
#   例: rl/run_train_recover.sh main 10000000 --eval
set -u
NAME=$1; STEPS=$2; shift 2
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
OUTROOT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/train/recover
mkdir -p "$OUTROOT"
source /home/jetson/miniconda/etc/profile.d/conda.sh
conda activate jetracer
cd "$EXP"
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "[$(date -Is)] 開始 $NAME run=recover steps=$STEPS $*" >> "$OUTROOT/$NAME.log"
/home/jetson/bin/jf-heavy --mem 2G --lock rl --need-mb 1200 -- nice -n 15 \
  python -u rl/train_modes.py --run recover --steps "$STEPS" --out "$OUTROOT/$NAME" "$@" >> "$OUTROOT/$NAME.log" 2>&1
code=$?
echo "[$(date -Is)] 終了 $NAME 終了コード $code" >> "$OUTROOT/$NAME.log"
exit $code
