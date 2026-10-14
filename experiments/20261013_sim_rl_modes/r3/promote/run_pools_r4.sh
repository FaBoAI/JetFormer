#!/bin/bash
# 昇格の段: r4 の 3 モードの記録のシーンの組(共有の rl/runtime = まとめた runtime、mode.json の runtime を渡す)。jf-heavy --lock none・nice 15・ワーカー 2 × 3
set -u
cd /home/jetson/jetformer/experiments/20261013_sim_rl_modes
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/promote/pools
run() {  # mode count max
  /home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 500 -- nice -n 15 node r3/promote/make_scene_pool_r4.mjs \
    --mode "$1" --policy bundle/$1/policy.json --mode-json bundle/$1/mode.json --count "$2" --max-candidates "$3" --workers 2 --out "$OUT/pool_$1.json" >> "$OUT/run_pools_$1.log" 2>&1
  echo "$(date -Is) $1 exit $?" >> "$OUT/run_pools.log"
}
run no_wall 90 300 &
run recover 90 400 &
run line 90 300 &
wait
echo "$(date -Is) all done" >> "$OUT/run_pools.log"
