#!/bin/bash
# no_wall r4c の学習(2026-10-14 21 時、protocol_no_wall_r4c.json training)。種 26・最大 150 万ステップ・初期化 = r4b の終わりの点(step_003014656、方策 22042f96…)・
# lr 3e-5(一定)・チェックポイント 50 万ステップごと・last.pt は 20 万ごと。jf-heavy --mem 2G --lock rl -- nice -n 15(lock rl の順番待ち)。止めたあとは同じコマンドで続きから。
set -u
E=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
O=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/no_wall_r4
mkdir -p "$O/logs"
source /home/jetson/miniconda/etc/profile.d/conda.sh
conda activate jetracer
cd "$E/r3/nw4/train_c"
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "[$(date -Is)] 開始(lock rl の順番待ちから)" >> "$O/logs/train_c.log"
/home/jetson/bin/jf-heavy --mem 2G --lock rl --need-mb 1200 -- nice -n 15 \
  python -u train_nw4.py --run no_wall_r4c --steps 1500000 --out "$O/train_c" --seed 26 --init no_wall_r4b_3M --lr-start 3e-5 --lr-end 3e-5 \
  --ckpt-every 500000 --save-every 200000 --tag r4c >> "$O/logs/train_c.log" 2>&1
code=$?
echo "[$(date -Is)] 終了 終了コード $code" >> "$O/logs/train_c.log"
exit $code
