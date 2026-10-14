#!/bin/bash
# 2 回目のブラウザの確認(F-ID を全モード・各モードを sim の 10 秒走らせる・モード未選択の RC 教師)。jf-heavy --lock big
set -u
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/integrate
cd /home/jetson/jetformer/experiments/20261013_sim_rl_modes
echo "$(date -Is) browser_check2 待ち" >> $OUT/run_checks.log
/home/jetson/bin/jf-heavy --mem 2500M --lock big --need-mb 1800 -- /home/jetson/miniconda/envs/jetracer/bin/python -u integrate/browser_check.py \
  --skip-plan --drive-sim-seconds 10 --out $OUT/browser2 > $OUT/browser_check2.log 2>&1
echo "$(date -Is) browser_check2 exit $?" >> $OUT/run_checks.log
