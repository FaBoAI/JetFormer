#!/bin/bash
# 離脱復活 r3 の学習 1 本を jf-heavy(メモリ上限 2G・lock rl・nice 15)の下で動かす(2026-10-14、学習の段 recover r3)。
# 止めたあとも同じコマンドで last.pt から続く。出力は ~/jetformer_reports/sim_rl_modes_20261013/r3/rc3/train/<名前>
#   r3/rc/run_train_rc3.sh <名前> <steps> [train_rc3.py の追加の引数...]
#   例: r3/rc/run_train_rc3.sh main 10000000 --eval
set -u
NAME=$1; STEPS=$2; shift 2
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
OUTROOT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/rc3/train
mkdir -p "$OUTROOT"
source /home/jetson/miniconda/etc/profile.d/conda.sh
conda activate jetracer
cd "$EXP"
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "[$(date -Is)] 開始 $NAME run=recover_r3 steps=$STEPS $*" >> "$OUTROOT/$NAME.log"
/home/jetson/bin/jf-heavy --mem 2G --lock rl --need-mb 1200 -- nice -n 15 \
  python -u r3/rc/train/train_rc3.py --run recover_r3 --steps "$STEPS" --out "$OUTROOT/$NAME" "$@" >> "$OUTROOT/$NAME.log" 2>&1
code=$?
echo "[$(date -Is)] 終了 $NAME 終了コード $code" >> "$OUTROOT/$NAME.log"
exit $code
