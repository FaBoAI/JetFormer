#!/bin/bash
# 直す 7-1 回目・係 safety: 対象の試験ファイルだけを流す。$1 = 出力の名前、残り = 試験ファイル
cd /home/jetson/jetformer
name="$1"; shift
/home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=/home/jetson/jetformer/experiments/20261026_fix/fix7/tmp_safety/pt_$name "$@" \
  > experiments/20261026_fix/fix7/safety/out/pytest_$name.txt 2>&1
rm -rf /home/jetson/jetformer/experiments/20261026_fix/fix7/tmp_safety/pt_$name
