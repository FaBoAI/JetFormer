#!/bin/bash
# 離脱復活 r3 の確認: JetFormer の写し(元のまま / 昇格の段の差分を当てたもの)で pytest 一式(jf-heavy --lock big、GPU は使わない)
S=/tmp/claude-1000/-home-jetson-jetracer/f44d13ba-776f-4620-a46a-2574918483a9/scratchpad/ck_rc3_promote
R=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_recover_r3
for v in full_patched full_orig; do
  cd $S/$v
  CUDA_VISIBLE_DEVICES= JETFORMER_DATA=$S/pytest_data_$v /home/jetson/bin/jf-heavy --mem 2500M --lock big -- nice -n 10 /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider tests > $R/pytest_full_$v.log 2>&1
  echo "$(date -Is) $v exit $?" >> $R/pytest_full.status
done
