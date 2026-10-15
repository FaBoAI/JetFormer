#!/bin/bash
# Q1 の評価のあとの集計(CPU。sim_diverse_train_20261015、2026-10-15。10/14 の post_eval.sh の写し)
#   bash post_eval.sh
# 1) verify_eval.mjs(再生の数え直し)→ 2) teacher_eval.mjs(先生の参照・線からの距離・引き継ぎ)→ 3) analyze_q1.py(判定・図・eval_sim.json)
# → 4) 別の道具での数え直し(review/recount_q1.py)と、ユーザーの replay-report.mjs での再生(review/replay_all.sh)→ 5) 突き合わせ(check_recount.py)
# → 6) render_q1_md.py(eval_sim.md)
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
R=/home/jetson/jetformer_reports/sim_diverse_train_20261015
LOG=$R/eval/logs
mkdir -p "$LOG" "$R/review"
J="/home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 800 -- nice -n 10"
source /home/jetson/miniconda/etc/profile.d/conda.sh && conda activate jetracer
cd "$HERE"
$J node verify_eval.mjs > "$LOG/verify_eval.log" 2>&1
python -I analyze_q1.py --prepare
$J node teacher_eval.mjs --runs "$R/eval/teacher_runs.json" --out "$R/eval/teacher_eval.json" > "$LOG/teacher_eval.log" 2>&1
$J python -I analyze_q1.py "$@" > "$LOG/analyze_q1.log" 2>&1
$J python3 -I ../review/recount_q1.py > "$LOG/recount_q1.log" 2>&1
$J bash ../review/replay_all.sh > "$LOG/replay_all.log" 2>&1 || true
python -I check_recount.py | tee "$LOG/check_recount.log"
python -I render_q1_md.py
tail -1 "$LOG/replay_all.log"
