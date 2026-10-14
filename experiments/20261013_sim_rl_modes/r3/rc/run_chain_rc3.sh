#!/bin/bash
# 離脱復活 r3: 表づくり(run_table.sh)が終わったあと、lock rl を取って (1) 表をまとめて sha256 を固定し(まだなら)、(2) 学習(1,000 万まで・検証つき)を続ける。
# 止まったあとも同じコマンドで続きから(表は固定済みなら作り直さない。学習は last.pt から)。
#   r3/rc/run_chain_rc3.sh [学習の名前 main]
set -u
NAME=${1:-main}
E=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
R3=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3
T=$R3/rc3/table
OUTROOT=$R3/rc3/train
mkdir -p $OUTROOT
echo "[$(date -Is)] chain: lock rl を待ちます" >> $OUTROOT/$NAME.log
/home/jetson/bin/jf-heavy --mem 2G --lock rl --need-mb 1200 -- nice -n 15 bash -c "
  set -u
  source /home/jetson/miniconda/etc/profile.d/conda.sh && conda activate jetracer
  cd $E
  export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
  echo \"[\$(date -Is)] chain: lock rl を取りました\"
  for k in 0 1 2 3; do tail -1 $T/rows_\$k.jsonl | grep -q '\"done\":true' || { echo \"表の shard \$k が終わっていません\"; exit 3; }; done
  if [ ! -f $R3/protocol_recover_r3_table.sha256 ]; then
    python3 -I r3/rc/merge_recoverable_table.py $T/rows_0.jsonl $T/rows_1.jsonl $T/rows_2.jsonl $T/rows_3.jsonl \
      --out-json r3/rc/recoverable_table_v1.json --out-mjs r3/rc/rt/recoverable_table.mjs \
      --sha-file $R3/protocol_recover_r3_table.sha256 --summary $T/table_summary.json || exit 4
    cp -p r3/rc/recoverable_table_v1.json $T/recoverable_table_v1.json
    echo \"[\$(date -Is)] 表を固定しました: \$(cat $R3/protocol_recover_r3_table.sha256 | head -1)\"
  fi
  python -u r3/rc/train/train_rc3.py --run recover_r3 --steps 10000000 --out $OUTROOT/$NAME --eval --tag $NAME
" >> $OUTROOT/$NAME.log 2>&1
code=$?
echo "[$(date -Is)] chain: 終わり(終了コード $code)" >> $OUTROOT/$NAME.log
exit $code
