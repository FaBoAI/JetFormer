#!/bin/bash
# 直す 7-2 回目・係 safety: 対象の試験ファイルだけを流す。$1 = 出力の名前、残り = 試験ファイル(pytest の引数)
cd /home/jetson/jetformer
name="$1"; shift
/home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=/home/jetson/jetformer/experiments/20261026_fix/fix7/tmp_safety_r2/pt_$name "$@" \
  > experiments/20261026_fix/fix7/safety_r2/out/pytest_$name.txt 2>&1
echo "exit $?" >> experiments/20261026_fix/fix7/safety_r2/out/pytest_$name.txt
rm -rf /home/jetson/jetformer/experiments/20261026_fix/fix7/tmp_safety_r2/pt_$name
