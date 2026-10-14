#!/bin/bash
# 離脱復活 r4: JetFormer の写し 2 つ(差分なし + 新しい試験 / 差分 3 つ + 新しい試験)で pytest 一式(jf-heavy --lock big、GPU は使わない)。
# 写しは scratchpad(git archive HEAD、experiments・data を除く)。JETFORMER_DATA も一時フォルダー(本物の data に触れない)
S=/tmp/claude-1000/-home-jetson-jetracer/f44d13ba-776f-4620-a46a-2574918483a9/scratchpad/rc4
R=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/rc4/promote
for v in jf_patched jf_orig; do
  rm -rf $S/pytest_data_$v; mkdir -p $S/pytest_data_$v
  cd $S/$v
  CUDA_VISIBLE_DEVICES= JETFORMER_DATA=$S/pytest_data_$v /home/jetson/bin/jf-heavy --mem 2500M --lock big -- nice -n 10 /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider tests > $R/pytest_full_$v.log 2>&1
  echo "$(date -Is) $v exit $?" >> $R/pytest_full.status
done
