#!/bin/bash
# 強化学習モードの学習 1 本を、出力の置き場所を指定して jf-heavy(メモリ上限 2G・lock rl・nice 15)の下で動かす(2026-10-13、学習の段)。
# rl/run_mode.sh と同じ中身で、--out だけ引数で渡す(学習の段の決まり: 記録は train/<モード>/ の下)。
#   rl/run_mode_to.sh <出力のディレクトリ> <run: no_wall|recover|cond|obstacle> <steps> [train_modes.py の追加の引数...]
#   例: rl/run_mode_to.sh ~/jetformer_reports/sim_rl_modes_20261013/train/line/cond-main cond 20000000 --eval
# 止めたあとも同じコマンドで last.pt から続く。nice は起動のときに付ける(スレッドごとに効くため)。
set -u
OUT=$1; RUN=$2; STEPS=$3; shift 3
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
mkdir -p "$OUT"
LOG="$OUT.log"
source /home/jetson/miniconda/etc/profile.d/conda.sh
conda activate jetracer
cd "$EXP"
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "[$(date -Is)] 開始 $OUT run=$RUN steps=$STEPS $*" >> "$LOG"
/home/jetson/bin/jf-heavy --mem 2G --lock rl --need-mb 1200 -- nice -n 15 \
  python -u rl/train_modes.py --run "$RUN" --steps "$STEPS" --out "$OUT" "$@" >> "$LOG" 2>&1
code=$?
echo "[$(date -Is)] 終了 $OUT 終了コード $code" >> "$LOG"
exit $code
