#!/bin/bash
# 離脱復活 r4(2026-10-14、学習の段 recover r4): lock rl を 1 回取って、
#   (1) 立て直せる表 v2 を作る(4 本を並べる。build_recoverable_table_v2.mjs は (driver, 開始) の組ごとに続きから)、
#   (2) 表をまとめて sha256 を固定する(まだなら。r3/protocol_recover_r4_table.sha256)、
#   (3) 追補の決まり protocol_recover_r4.json の固定(r3/protocol_recover_r4.sha256)を待つ(学習の前に固定する決まり)、
#   (4) 学習(train_rc4.py、検証つき)を last.pt から続ける。
# 止まったあとも同じコマンドで続きから(表の終わった組・固定した表は作り直さない。学習は last.pt から)。
#   r3/rc4/run_chain_rc4.sh [学習の名前 main]
set -u
NAME=${1:-main}
E=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
R3=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3
T=$R3/rc4/table
OUTROOT=$R3/rc4/train
mkdir -p $OUTROOT $T
echo "[$(date -Is)] chain: lock rl を待ちます" >> $OUTROOT/$NAME.log
/home/jetson/bin/jf-heavy --mem 2G --lock rl --need-mb 1200 -- nice -n 15 bash -c "
  set -u
  source /home/jetson/miniconda/etc/profile.d/conda.sh && conda activate jetracer
  cd $E
  export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
  echo \"[\$(date -Is)] chain: lock rl を取りました\"
  if [ ! -f $R3/protocol_recover_r4_table.sha256 ]; then
    for k in 0 1 2 3; do
      if [ -f $T/rows_\$k.jsonl ] && tail -1 $T/rows_\$k.jsonl | grep -q '\"done\":true' && [ \$(grep -c pair_done $T/rows_\$k.jsonl) -ge 8 ]; then echo \"表の shard \$k はもう終わっています\"; continue; fi
      node r3/rc4/build_recoverable_table_v2.mjs --shard \$k/4 --out $T/rows_\$k.jsonl 2>> $T/shard_\$k.log &
    done
    wait
    for k in 0 1 2 3; do [ \$(grep -c pair_done $T/rows_\$k.jsonl) -ge 8 ] || { echo \"表の shard \$k が終わっていません\"; exit 3; }; done
    python3 -I r3/rc4/merge_recoverable_table_v2.py $T/rows_0.jsonl $T/rows_1.jsonl $T/rows_2.jsonl $T/rows_3.jsonl \
      --out-json r3/rc4/recoverable_table_v2.json --out-mjs r3/rc4/rt/recoverable_table.mjs \
      --sha-file $R3/protocol_recover_r4_table.sha256 --summary $T/table_summary.json || exit 4
    cp -p r3/rc4/recoverable_table_v2.json $T/recoverable_table_v2.json
    echo \"[\$(date -Is)] 表を固定しました: \$(head -1 $R3/protocol_recover_r4_table.sha256)\"
  fi
  n=0
  while [ ! -f $R3/protocol_recover_r4.sha256 ]; do
    [ \$((n % 30)) -eq 0 ] && echo \"[\$(date -Is)] 追補の決まり r4 の固定を待っています\"
    n=\$((n + 1)); sleep 20
    [ \$n -gt 1080 ] && { echo '6 時間待っても固定されません'; exit 5; }
  done
  python -u r3/rc4/train/train_rc4.py --run recover_r4 --steps 6000000 --out $OUTROOT/$NAME --eval --tag $NAME
" >> $OUTROOT/$NAME.log 2>&1
code=$?
echo "[$(date -Is)] chain: 終わり(終了コード $code)" >> $OUTROOT/$NAME.log
exit $code
