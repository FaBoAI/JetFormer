#!/bin/bash
# 評価のあとの集計(CPU。sim_dart_20261014 の評価の段、2026-10-14)
#   bash post_eval.sh
# 1) verify_eval.mjs(再生の数え直し)→ 2) teacher_eval.mjs(先生の参照・線からの距離・引き継ぎ)→ 3) analyze_eval.py(判定・図・evaluation.json)→ 4) render_md.py
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
R=/home/jetson/jetformer_reports/sim_dart_20261014
LOG=$R/eval/logs
mkdir -p "$LOG"
J="/home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 800 -- nice -n 10"
source /home/jetson/miniconda/etc/profile.d/conda.sh && conda activate jetracer
cd "$HERE"
$J node verify_eval.mjs > "$LOG/verify_eval.log" 2>&1
python -I analyze_eval.py --prepare
$J node teacher_eval.mjs --runs "$R/eval/teacher_runs.json" --out "$R/eval/teacher_eval.json" > "$LOG/teacher_eval.log" 2>&1
$J python -I analyze_eval.py "$@" > "$LOG/analyze_eval.log" 2>&1
python -I render_md.py
tail -40 "$LOG/analyze_eval.log"
