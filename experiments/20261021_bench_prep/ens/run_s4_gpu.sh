#!/bin/bash
# 段 S4 の GPU・torch の分(1 回の順番で): ENS を置く → エンジンの確認 → 確かめ → S4 の試験(torch のもの)
# cd ~/jetformer && /home/jetson/bin/jf-heavy --mem 3G --lock big --need-mb 2500 -- bash experiments/20261021_bench_prep/ens/run_s4_gpu.sh
cd /home/jetson/jetformer || exit 1
PY=/home/jetson/miniconda/envs/jetracer/bin/python
echo "[$(date '+%F %T')] place_ens all を始めます"
nice -n 10 $PY -u experiments/20261021_bench_prep/ens/place_ens.py all
rc=$?
echo "[$(date '+%F %T')] place_ens all rc=$rc"
echo "[$(date '+%F %T')] S4 の試験(test_policy_ens・登録簿に触れる既存の試験)を始めます"
CUDA_VISIBLE_DEVICES= nice -n 10 $PY -m pytest -q -p no:cacheprovider tests/test_policy_ens.py tests/test_policies.py tests/test_policy_ui.py tests/test_drive_model_select.py
trc=$?
echo "[$(date '+%F %T')] test_policy_ens rc=$trc"
exit $(( rc != 0 ? rc : trc ))
