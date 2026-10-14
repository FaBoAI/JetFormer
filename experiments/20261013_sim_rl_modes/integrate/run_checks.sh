#!/bin/bash
# 統合の段の重い確認: ブラウザ(8080・Firefox)→ pytest 一式。どちらも jf-heavy --lock big(ベストラップの作業と順番待ち)
set -u
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/integrate
cd /home/jetson/jetformer/experiments/20261013_sim_rl_modes
echo "$(date -Is) browser_check 待ち" >> $OUT/run_checks.log
/home/jetson/bin/jf-heavy --mem 2500M --lock big --need-mb 1800 -- /home/jetson/miniconda/envs/jetracer/bin/python -u integrate/browser_check.py \
  --plan-minutes "${PLAN_MINUTES:-fastest=1,no_wall=1,recover=1,speed=1}" > $OUT/browser_check.log 2>&1
echo "$(date -Is) browser_check exit $?" >> $OUT/run_checks.log
if [ "${SKIP_PYTEST:-0}" != 1 ]; then
  cd /home/jetson/jetformer
  /home/jetson/bin/jf-heavy --mem 2500M --lock big -- /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider tests > $OUT/pytest_full.log 2>&1
  echo "$(date -Is) pytest exit $?" >> $OUT/run_checks.log
fi
