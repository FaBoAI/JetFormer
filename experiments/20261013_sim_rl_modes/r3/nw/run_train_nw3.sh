#!/bin/bash
# no_wall r3 の学習(2026-10-14、protocol_no_wall_r3.json training)。種 23・600 万ステップ・初期化 = r2 の選んだ点(step_004014080)・lr 5e-5 → 3e-5。
# jf-heavy --mem 2G --lock rl -- nice -n 15(lock rl の順番待ち)。止めたあとは同じコマンドで last.pt から続く。
set -u
E=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
O=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/no_wall_r3
mkdir -p "$O/logs"
source /home/jetson/miniconda/etc/profile.d/conda.sh
conda activate jetracer
cd "$E/r3/nw/train"
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "[$(date -Is)] 開始(lock rl の順番待ちから)" >> "$O/logs/train.log"
/home/jetson/bin/jf-heavy --mem 2G --lock rl --need-mb 1200 -- nice -n 15 \
  python -u train_nw3.py --run no_wall_r3 --steps 6000000 --out "$O/train" --seed 23 --init no_wall_r2_4M --lr-start 5e-5 --tag r3 >> "$O/logs/train.log" 2>&1
code=$?
echo "[$(date -Is)] 終了 終了コード $code" >> "$O/logs/train.log"
exit $code
