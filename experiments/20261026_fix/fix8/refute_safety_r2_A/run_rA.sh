#!/bin/bash
# 票 A の広さを流す: bash run_rA.sh PART SHARD NSHARD
cd /home/jetson/jetformer/experiments/20261026_fix/fix8/refute_safety_r2_A
PYTHONDONTWRITEBYTECODE=1 RA_TAG="$1_$2" /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
  /home/jetson/miniconda/envs/jetracer/bin/python -I -B sweep_rA.py "$1" "$2" "$3" > "out/log_$1_$2of$3.txt" 2>&1
