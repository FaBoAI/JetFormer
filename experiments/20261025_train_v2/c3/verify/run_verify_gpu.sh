#!/bin/bash
# verify_c3 の GPU の仕事(2 回。どちらも jf-heavy --lock big で学習のキューと順番に分け合う。あいだ 60 秒)
cd /home/jetson/jetformer || exit 1
V=experiments/20261025_train_v2/c3/verify
P=/home/jetson/miniconda/envs/jetracer/bin/python
G="/home/jetson/bin/jf-heavy --lock big --mem 3000M --need-mb 2000 -- nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2"
echo "[$(date '+%F %T')] 仕事 1 を待ちます(B)"
$G $P -I $V/repredict_c3_gpu.py clean_driving_20261011-nocmd-s3-20261011-024626 tab,shift,trt
echo "[$(date '+%F %T')] 仕事 1 rc=$?"
sleep 60
echo "[$(date '+%F %T')] 仕事 2 を待ちます(v2h-e60-s1 と bfast-s0 の tab)"
$G bash -c "$P -I $V/repredict_c3_gpu.py v2_20261025-v2h-e60-s1 tab,shift,trt && $P -I $V/repredict_c3_gpu.py bfast_20261025-e15-s0 tab"
echo "[$(date '+%F %T')] 仕事 2 rc=$?"
touch $V/run_verify_gpu.done
