#!/bin/bash
# fix8 係 safety 直す 8-1 回目: assist・drive を読む tests/ の試験(out/existing_tests_list.txt = grep -lE で assist・drive を読むもの。
# この係の tests/test_rec_fix8_safety_r1.py も入る)を本物のコードで流す。出力 out/pytest_existing12.txt
set -u
JF=/home/jetson/jetformer
D=$JF/experiments/20261026_fix/fix8/fix_safety_r1
BT=$JF/experiments/20261026_fix/fix8/tmp_safety_r1/existing
mkdir -p "$BT"
cd $JF
PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
  /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider -o tmp_path_retention_policy=failed \
  --basetemp="$BT/bt" -W ignore::DeprecationWarning $(cat $D/out/existing_tests_list.txt) > $D/out/pytest_existing12.txt 2>&1
echo "exit $?" >> $D/out/pytest_existing12.txt
rm -rf "$BT"
