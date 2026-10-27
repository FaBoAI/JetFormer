#!/bin/bash
# tools:F6 の確かめの砂場(本物の tools・tests の写し)。終わったら rm -rf で消す。
#   bash setup_sandbox.sh && jf-heavy --lock none --mem 1500M -- python3 mutate.py && rm -rf 砂場
set -e; JF=/home/jetson/jetformer; S=$JF/experiments/20261026_fix/fix2/tmp_refute_F6
mkdir -p $S/tools $S/tests $S/diag_launch
cp $JF/tools/*.py $S/tools/; cp $JF/tests/test_standing_3lap.py $JF/tests/test_quick_eval_by_params.py $JF/tests/test_lap_anchor.py $S/tests/
ln -sfn $JF/jetformer $S/jetformer
cp $JF/experiments/20261026_fix/diag_launch/collect.py $S/diag_launch/
sed -e "s#sys.path.insert(0, str(JF / 'tests'))#sys.path.insert(0, '$S/tests')#" \
    -e "s#sys.path.insert(0, str(JF / 'experiments/20261026_fix/diag_launch'))#sys.path.insert(0, '$S/diag_launch')#" \
    $JF/experiments/20261026_fix/fix2/test_tools_collect.py > $S/tests/test_tools_collect.py
