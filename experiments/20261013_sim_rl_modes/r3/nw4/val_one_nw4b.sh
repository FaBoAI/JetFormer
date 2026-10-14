#!/bin/bash
# no_wall r4b/r4c: 選びの候補の 1 つを検証の組 NW-V-R4(シールド v5)で走らせて判定する(val_loop_nw4b.py の 1 回ぶんと同じ。2026-10-14 21 時、
# 候補を 2 つ並べて調べるために足した)。止めても同じコマンドで続きから。
#   val_one_nw4b.sh <名前(val/ の下のフォルダー: step_* / a_step_* / c_step_*)> <policy.json> [子の数 3]
set -eu
E=/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/nw4
O=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/no_wall_r4
N=$1; P=$2; J=${3:-3}
V=$O/val/$N
mkdir -p "$V"
echo "[$(date +%T)] 検証 $N $P"
cd "$E"
/home/jetson/bin/jf-heavy --mem 1800M --lock none --need-mb 700 -- nice -n 15 node run_sets_nw4.mjs --which validation --policy "$P" --out "$V" --jobs "$J"
python3 judge_nw4.py --rows "$V/rows_validation.jsonl" --which validation --out "$V/eval_validation.json" > "$V/judge.txt" 2>&1
head -c 900 "$V/judge.txt"; echo
