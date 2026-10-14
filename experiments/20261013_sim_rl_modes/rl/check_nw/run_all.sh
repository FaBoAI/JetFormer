#!/bin/bash
# no_wall の確認 r1 の全部の組(順に)。jf-heavy --lock none -- nice -n 15 で動かす
set -e
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_no_wall_r1/runs
cd /home/jetson/jetformer/experiments/20261013_sim_rl_modes/rl/check_nw
for S in FNOM,FAST,LOWMU FDR,CORNER REPLAY FDROFF; do
  echo "[$(date +%T)] 組 $S" >&2
  node verify_nw.mjs --sets $S --jobs 3 --out $OUT
done
echo "[$(date +%T)] 全部終わり" >&2
