#!/bin/bash
# 段 Desk の GPU の仕事を順に(1 つずつ・jf-heavy --lock big)。出来たものは飛ばす。車・装置には触れない
#   bash experiments/20261021_bench_prep/desk/run_desk_gpu.sh  >> experiments/20261021_bench_prep/desk/logs/run_desk_gpu.log 2>&1
set -u
cd /home/jetson/jetformer
PY=/home/jetson/miniconda/envs/jetracer/bin/python
J="/home/jetson/bin/jf-heavy --mem 3G --lock big --need-mb 3200 -- nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 $PY -u"
D=experiments/20261021_bench_prep/desk
B=clean_driving_20261011-nocmd-s3-20261011-024626
echo "[$(date +%T)] 始め"
if [ ! -s $D/logs/crosscheck_B.json ] || ! grep -q '"pass": true' $D/logs/crosscheck_B.json; then
  $J $D/desk_gpu.py crosscheck || { echo "[$(date +%T)] crosscheck 失敗"; exit 1; }
fi
$J $D/desk_gpu.py predict || { echo "[$(date +%T)] predict 失敗"; exit 1; }
if [ ! -s $D/logs/engine_check_B.json ]; then
  $J experiments/20261013_sim_cotrain/train/engine_load_check.py data/runs/$B $D/logs/engine_check_B.json || echo "[$(date +%T)] B の engine_load_check が門に不合格か失敗(rc $?)"
fi
/home/jetson/bin/jf-heavy --lock none --mem 1500M -- $PY -I $D/write_engine_check.py B || echo "[$(date +%T)] B の engine_check.json は書いていない"
$J $D/desk_gpu.py sanity || { echo "[$(date +%T)] sanity 失敗"; exit 1; }
echo "[$(date +%T)] 終わりました"
