#!/bin/bash
# 確認の段(review、2026-10-14): ユーザーの replay-report.mjs(有効な版の dist/sim-core.js をそのまま使う)で、
# 走行の記録を再生して、軌跡・集計・終わり方が記録と同じかを見る(読むだけ。何も書き換えない)
REL=/home/jetson/jetformer/data/sim/releases/$(python3 -c "import json;print(json.load(open('/home/jetson/jetformer/data/sim/active.json'))['release'])")
OUT=/home/jetson/jetformer_reports/sim_dart_20261014/review/replay_user_tool
mkdir -p "$OUT"
sha256sum "$REL/dist/sim-core.js" "$REL/replay-report.mjs" > "$OUT/tool_sha256.txt"
for f in "$@"; do
  n=$(basename "$f" .json)
  if node "$REL/replay-report.mjs" "$f" > "$OUT/$n.json" 2> "$OUT/$n.err"; then echo "PASS $n"; else echo "FAIL $n"; fi
done
