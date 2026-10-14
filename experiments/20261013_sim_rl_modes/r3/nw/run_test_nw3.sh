#!/bin/bash
# no_wall r3 の試験(protocol_no_wall_r3.json sets.test)を、選んだ 1 つの点にだけ 1 回走らせて判定する(2026-10-14)。
# selection.json の選んだ方策だけを受け付け、別の方策で試験の印(test/TEST_POLICY)があれば止まる(試験を見て選び直さない)。
# 途中で止めたら同じコマンドで続きから(同じ方策の書き終えた回は飛ばす)。jf-heavy --lock none -- nice -n 15、子 JOBS 本。
set -eu
E=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
O=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/no_wall_r3
JOBS=${JOBS:-3}
cd "$E/r3/nw"
sha256sum -c --status <(grep protocol_no_wall_r3.json /home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/protocol_r3.sha256 | sed "s#  #  /home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/#") || { echo "protocol_no_wall_r3.json の sha256 が違います"; exit 1; }
P=$(python3 -c "import json;print(json.load(open('$O/selection.json'))['selected']['policy'])")
PS=$(sha256sum "$P" | cut -d' ' -f1)
mkdir -p "$O/test"
if [ -f "$O/test/TEST_POLICY" ] && [ "$(cat "$O/test/TEST_POLICY")" != "$PS" ]; then echo "別の方策で試験を始めています。試験は 1 回だけ"; exit 1; fi
echo "$PS" > "$O/test/TEST_POLICY"
echo "[$(date +%T)] 試験 方策 $P($PS)"
/home/jetson/bin/jf-heavy --mem 1800M --lock none --need-mb 900 -- nice -n 15 node run_sets_nw3.mjs --which test --policy "$P" --out "$O/test" --jobs "$JOBS"
python3 judge_nw3.py --rows "$O/test/rows_test.jsonl" --which test --out "$O/test/eval_test.json" | tee "$O/test/judge.txt"
