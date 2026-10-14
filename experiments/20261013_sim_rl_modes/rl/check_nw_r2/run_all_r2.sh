#!/bin/bash
# no_wall の確認 r2 の全部の組(順に)。jf-heavy --mem 1500M --lock none -- nice -n 15 で動かす
set -e
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_no_wall_r2/runs
cd /home/jetson/jetformer/experiments/20261013_sim_rl_modes/rl/check_nw_r2
JOBS=${JOBS:-3}
for S in ${SETS:-REPLAY2 FN3,FD3 VX,NG LONG,D2 FN3OFF,FD3OFF,VXOFF,FAST3}; do
  echo "[$(date +%T)] 組 $S" >&2
  node verify_nw_r2.mjs --sets $S --jobs $JOBS --out $OUT
done
echo "[$(date +%T)] 全部終わり" >&2
