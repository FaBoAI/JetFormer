#!/bin/bash
# fix8 係 safety-proposal: 前からの安全の試験(Recovery を見るファイル)を、案の写しに差し替えて流す(-p p8swap。本物の jetformer/ は
# 書かない)。どの決まりが変わるかを数えるだけ(nice 19)。出力 out/pytest_<案>.txt
cd /home/jetson/jetformer
D=/home/jetson/jetformer/experiments/20261026_fix/fix8/proposal
T=/home/jetson/jetformer/experiments/20261026_fix/fix8/tmp_proposal
FILES="tests/test_rec_fix7_safety_r2.py tests/test_rec_fix7_safety.py tests/test_rec_fix6_safety_r2.py tests/test_rec_fix6_safety.py
 tests/test_rec_fix5_safety.py tests/test_rec_hold_keep.py tests/test_recovery_back_guard.py tests/test_fix3_r1_1.py tests/test_fix3_r2_1.py
 tests/test_fix3_r3_1.py tests/test_assist.py"
for v in "$@"; do
  P8_SWAP=$v PYTHONPATH=$D/swap PYTHONDONTWRITEBYTECODE=1 nice -n 5 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
    /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider -p p8swap -rf --basetemp=$T/pt_$v $FILES \
    > $D/out/pytest_$v.txt 2>&1
  echo "exit $?" >> $D/out/pytest_$v.txt
  rm -rf $T/pt_$v
done
