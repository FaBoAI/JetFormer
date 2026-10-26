#!/bin/bash
# 確認 V-decide の GPU の仕事(1 つずつ・jf-heavy --lock big)。車・装置には触れない
set -u
cd /home/jetson/jetformer
PY=/home/jetson/miniconda/envs/jetracer/bin/python
V=experiments/20261022_prep/verify_decide
R=/home/jetson/jetformer_reports/prep_20261022/verify_decide_raw
J="/home/jetson/bin/jf-heavy --mem 3G --lock big --need-mb 3200 -- nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 $PY -u -B"
echo "[$(date +%T)] start"
$J $V/vd_gpu.py; echo "[$(date +%T)] vd_gpu rc=$?"
$J experiments/20261013_sim_cotrain/train/engine_load_check.py data/runs/bench_20261022-slow-sec-s1 $R/engine_recheck_SLOW-SEC1.json; echo "[$(date +%T)] engine_load_check rc=$?"
echo "[$(date +%T)] done"
