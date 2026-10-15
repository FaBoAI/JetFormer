#!/bin/bash
# 確認(sim_diverse_train_20261015 の Q1、2026-10-15。10/14 の review/replay_all.sh の写し。出力の場所だけ今日にした):
# ユーザーの replay-report.mjs(有効な版の dist/sim-core.js をそのまま使う)で、走行の記録を再生して、軌跡・集計・終わり方が記録と同じかを見る
# (読むだけ。何も書き換えない)。引数なしなら今日の closed_loop の全部の走行(単周・連続周回)
#   bash replay_all.sh [<走行の JSON> ...]
REL=/home/jetson/jetformer/data/sim/releases/$(python3 -c "import json;print(json.load(open('/home/jetson/jetformer/data/sim/active.json'))['release'])")
OUT=/home/jetson/jetformer_reports/sim_diverse_train_20261015/review/replay_user_tool
CL=/home/jetson/jetformer_reports/sim_diverse_train_20261015/eval/closed_loop
mkdir -p "$OUT"
sha256sum "$REL/dist/sim-core.js" "$REL/replay-report.mjs" > "$OUT/tool_sha256.txt"
if [ $# -eq 0 ]; then set -- $(ls "$CL"/*/*-single*.json "$CL"/*/*-flying.json 2>/dev/null | grep -v user_verifier | grep -v '/plan_'); fi
P=0; F=0
for f in "$@"; do
  n=$(basename "$f" .json)
  if node "$REL/replay-report.mjs" "$f" > "$OUT/$n.json" 2> "$OUT/$n.err"; then echo "PASS $n"; P=$((P+1)); else echo "FAIL $n"; F=$((F+1)); fi
done
echo "合計 PASS $P / FAIL $F"
