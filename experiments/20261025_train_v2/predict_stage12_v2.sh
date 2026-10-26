#!/bin/bash
# 段 1・2 と B-FAST の 16 本の机上の予測(lock2 のあと)。1 本ずつ jf-heavy --lock big で。あいだ 30 秒。
cd /home/jetson/jetformer || exit 1
L=/home/jetson/jetformer/experiments/20261025_train_v2/logs/predict_stage12
[ -f /home/jetson/jetformer_reports/train_v2_20261025/protocol_v2_lock2.json ] || { echo "lock2 が無い"; exit 1; }
for n in v2_20261025-v2h-e30-s0 v2_20261025-v2m-e30-s0 v2_20261025-v2h-e30-s1 v2_20261025-v2m-e30-s1 \
         v2_20261025-v2h-e30-s2 v2_20261025-v2m-e30-s2 v2_20261025-v2h-e30-s3 v2_20261025-v2m-e30-s3 \
         v2_20261025-v2h-e60-s0 v2_20261025-v2h-e60-s1 v2_20261025-v2h-e60-s2 v2_20261025-v2h-e60-s3 v2_20261025-v2h-e100-s0 \
         bfast_20261025-e15-s0 bfast_20261025-e15-s1 bfast_20261025-e15-s2; do
  echo "[$(date '+%F %T')] $n 開始"
  /home/jetson/bin/jf-heavy --lock big --mem 3872M --need-mb 3200 -- nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MALLOC_ARENA_MAX=2 /home/jetson/miniconda/envs/jetracer/bin/python -u experiments/20261025_train_v2/eval_v2.py predict --runs "$n" > "$L/$n.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] $n rc=$rc $(tail -1 "$L/$n.log")"
  if [ $rc -ne 0 ]; then
    echo "[$(date '+%F %T')] もう 1 回だけ"
    sleep 30
    /home/jetson/bin/jf-heavy --lock big --mem 3872M --need-mb 3200 -- nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      MALLOC_ARENA_MAX=2 /home/jetson/miniconda/envs/jetracer/bin/python -u experiments/20261025_train_v2/eval_v2.py predict --runs "$n" >> "$L/$n.log" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] $n やり直し rc=$rc"
  fi
  sleep 30
done
echo "[$(date '+%F %T')] 全部おわり"
