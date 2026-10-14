#!/bin/bash
# 記録の途中で、合格したシーンから先に検証する(記録の段、2026-10-14)。記録が終わる(done.txt)まで 1 分おきに繰り返す。
# 結果は <OUT>/verify/scenes/<シーン>/row.json に残り、記録のあとの verify_dart_collection.py --reuse がそのまま使う。CPU だけ。
#   bash incremental_verify.sh D1 <記録の結果のフォルダー>
set -u
D=$1; OUT=$2
E=/home/jetson/jetformer/experiments/20261014_sim_dart_margin
source /home/jetson/miniconda/etc/profile.d/conda.sh && conda activate jetracer
mkdir -p $OUT/verify_partial
while :; do
  FIN=0; [ -f $OUT/done.txt ] && FIN=1
  python -I -c "
import json,glob,os
acc=[json.load(open(f))['name'] for f in sorted(glob.glob('$OUT/sidecars/*.json')) if 'attempt' not in os.path.basename(f)]
json.dump({'accepted':acc,'rejected':[],'retried':{}},open('$OUT/verify_partial/result.json','w'))
print(len(acc))" > $OUT/verify_partial/n.txt
  # sidecars を同じ場所から読めるように、仮の result.json の隣に sidecars へのリンクを置く
  ln -sfn $OUT/sidecars $OUT/verify_partial/sidecars
  if [ "$(cat $OUT/verify_partial/n.txt)" != 0 ]; then
    /home/jetson/bin/jf-heavy --mem 1500M --lock none -- nice -n 12 python -I $E/record/verify_dart_collection.py --dataset $D \
      --result $OUT/verify_partial/result.json --out-dir $OUT/verify_partial/out --workers 2 --reuse > $OUT/verify_partial/last.log 2>&1
    # 合格したシーンの row.json を本番の検証フォルダーへ写す(本番の --reuse が使う)
    for d in $OUT/verify_partial/out/scenes/*/; do n=$(basename $d); mkdir -p $OUT/verify/scenes/$n; cp -n $d/* $OUT/verify/scenes/$n/ 2>/dev/null; done
  fi
  [ $FIN = 1 ] && break
  sleep 60
done
echo "終わり $(date +%H:%M:%S)" >> $OUT/verify_partial/n.txt
