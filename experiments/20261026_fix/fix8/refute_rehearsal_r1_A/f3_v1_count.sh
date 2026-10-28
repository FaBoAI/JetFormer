#!/bin/bash
# 指摘 rehearsal-v1-count の確かめ(打ち消しの票 A)。本物の data/bench_candidates.json は写しを読むだけ(変えない)。
set -u
HERE=/home/jetson/jetformer/experiments/20261026_fix/fix8/refute_rehearsal_r1_A
P=/home/jetson/miniconda/envs/jetracer/bin/python
REAL=/home/jetson/jetformer/data/bench_candidates.json
T=$HERE/tmp_f3; rm -rf "$T"; mkdir -p "$T"
echo "本物の sha256(前): $(sha256sum $REAL | cut -c1-16)"
cp "$REAL" "$T/bench_candidates.json"
cd /home/jetson/jetformer
PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- $P -I -B experiments/20261026_fix/fix3/add_v1_block.py \
  --write --bench "$T/bench_candidates.json" --run v3_20261026-v3d-aug2-e60-s1 --label V1 --allow-not-drivable \
  --user-approval '作り物(打ち消しの票 A。写しにだけ)' 2>&1 | grep -v '^[-+ @]' | head -20
echo "--- 写しの 2026-11-03 の blocks の数"
$P -I -B -c "import json,sys; d=json.load(open(sys.argv[1])); print([len(x['blocks']) for x in d['days'] if x['date']=='2026-11-03'][0])" "$T/bench_candidates.json"
echo "--- 本物の app.js の renderBenchPick に写しの日を渡した選ぶ欄"
node "$HERE/f3_pick.js" "$T/bench_candidates.json"
echo "本物の sha256(後): $(sha256sum $REAL | cut -c1-16)"
echo "--- 手順書・todo の「1〜23」"
grep -n "1〜23" /home/jetson/jetformer_reports/fix_20261026/runbook_20261103.md /home/jetson/jetformer_reports/fix_20261026/todo_user_1026.md | cut -c1-220
echo "--- README 6.2 の 10"
awk '/^### 6.2/{f=1} f' /home/jetson/jetformer_reports/fix_20261026/README.md | grep "^| 10 " | cut -c1-400
rm -rf "$T"
