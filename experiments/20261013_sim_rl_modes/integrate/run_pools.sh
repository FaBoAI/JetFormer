#!/bin/bash
# 記録に使うシーンの組(scene pool)を 4 モード分作る(統合の段)。jf-heavy --lock none・nice 15・ワーカー 2
set -u
cd /home/jetson/jetformer/experiments/20261013_sim_rl_modes
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/integrate/pools
R=/home/jetson/jetformer_reports/sim_rl_modes_20261013
run() {  # mode policy mode_json count max
  /home/jetson/bin/jf-heavy --mem 1200M --lock none --need-mb 400 -- nice -n 15 node integrate/make_scene_pool.mjs \
    --mode "$1" --policy "$2" --mode-json "$3" --count "$4" --max-candidates "$5" --workers 2 --out "$OUT/pool_$1.json" >> "$OUT/run_pools.log" 2>&1
  echo "$(date -Is) $1 exit $?" >> "$OUT/run_pools.log"
}
run fastest bundle/fastest/policy.json bundle/fastest/mode.json 90 300
run speed $R/train/speed/r2/export/policy.json $R/train/speed/r2/export/mode.json 90 300
run no_wall bundle/no_wall/policy.json bundle/no_wall/mode.json 90 300
run recover bundle/recover/policy.json bundle/recover/mode.json 75 600
echo "$(date -Is) all done" >> "$OUT/run_pools.log"
