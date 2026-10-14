#!/bin/bash
# no_wall 確認 r4: 組を K 本の子で走らせる(続きから。出力の行にある id は飛ばす)。使い方: K=3 ./run_all_ck4.sh <jobs.jsonl> <出力のフォルダー>
set -u
K=${K:-3}; JOBS=$1; OUT=$2
mkdir -p "$OUT"
cd "$(dirname "$0")"
for k in $(seq 0 $((K-1))); do
  /home/jetson/bin/jf-heavy --mem 1200M --lock none --need-mb 700 -- nice -n 15 node run_ck4.mjs --jobs "$JOBS" --shard $k --nshards $K --out "$OUT/rows_$k.jsonl" &
done
wait
echo "全部終わり $(date +%T)"
