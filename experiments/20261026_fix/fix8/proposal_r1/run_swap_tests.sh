#!/bin/bash
# fix8 係 safety 直す 8-1 回目: 前からの Recovery の試験 11 ファイル(fix8 safety-proposal と同じ組)+ この係の試験を、案の写し
# ($1 = qA / qC)に差し替えて流す。本物の jetformer/ には書かない。出力 out/pytest_swap_$1.txt
set -u
P=$1
JF=/home/jetson/jetformer
D=$JF/experiments/20261026_fix/fix8/proposal_r1
BT=$JF/experiments/20261026_fix/fix8/tmp_safety_r1/swap_$P
mkdir -p "$BT"
cd $JF
PYTHONDONTWRITEBYTECODE=1 R1_SWAP=$P PYTHONPATH=$D/swap /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
  /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider -p r1swap -o tmp_path_retention_policy=failed \
  --basetemp="$BT/bt" -W ignore::DeprecationWarning \
  tests/test_rec_fix7_safety_r2.py tests/test_rec_fix7_safety.py tests/test_rec_fix6_safety_r2.py tests/test_rec_fix6_safety.py \
  tests/test_rec_fix5_safety.py tests/test_rec_hold_keep.py tests/test_recovery_back_guard.py tests/test_fix3_r1_1.py \
  tests/test_fix3_r2_1.py tests/test_fix3_r3_1.py tests/test_assist.py tests/test_rec_fix8_safety_r1.py \
  > $D/out/pytest_swap_$P.txt 2>&1
echo "exit $?" >> $D/out/pytest_swap_$P.txt
rm -rf "$BT"
