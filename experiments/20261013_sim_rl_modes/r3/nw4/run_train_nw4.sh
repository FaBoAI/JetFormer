#!/bin/bash
# no_wall r4 の学習(2026-10-14、protocol_no_wall_r4.json training)。種 24・最大 300 万ステップ・初期化 = r3 の選んだ点(step_006012928、方策 3b099f9b…)・lr 3e-5(一定)・
# チェックポイント 50 万ステップごと。jf-heavy --mem 2G --lock rl -- nice -n 15(lock rl の順番待ち)。止めたあとは同じコマンドで last.pt から続く。
set -u
E=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
O=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/no_wall_r4
mkdir -p "$O/logs"
source /home/jetson/miniconda/etc/profile.d/conda.sh
conda activate jetracer
cd "$E/r3/nw4/train"
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "[$(date -Is)] 開始(lock rl の順番待ちから)" >> "$O/logs/train.log"
/home/jetson/bin/jf-heavy --mem 2G --lock rl --need-mb 1200 -- nice -n 15 \
  python -u train_nw4.py --run no_wall_r4 --steps 3000000 --out "$O/train" --seed 24 --init no_wall_r3_6M --lr-start 3e-5 --lr-end 3e-5 \
  --ckpt-every 500000 --tag r4 >> "$O/logs/train.log" 2>&1
code=$?
echo "[$(date -Is)] 終了 終了コード $code" >> "$O/logs/train.log"
exit $code
