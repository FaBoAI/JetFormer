#!/bin/bash
# A4(DAgger)の学習 → 一式 → 閉ループ評価 → 集計(sim_diverse_train_20261015、2026-10-15。protocol.json dagger.procedure 6・7)
# 電源断のあとは同じコマンドで続きから(学習は last.pth から、評価は済んだ走行を飛ばす)。
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
X=$(cd "$HERE/.." && pwd)
R=/home/jetson/jetformer_reports/sim_diverse_train_20261015
log() { echo "[$(date +%H:%M:%S)] $*"; }
source /home/jetson/miniconda/etc/profile.d/conda.sh && conda activate jetracer
[ -f "$R/train/a4_plan.json" ] || { log "a4_plan.json がありません"; exit 2; }
cd /home/jetson/jetformer
log "A4 の学習を始める"
python -u experiments/20261015_sim_diverse_train/train/diverse_train.py train A4/s0 --mem-cap 3800M >> "$R/train/logs/driver.log" 2>&1
log "学習の道具の終わり $?"
python -I -c "
import json,sys,os
r=json.load(open('$R/train/runs.json'))['runs'].get('A4/s0') or {}
sys.exit(0 if r.get('bundle_zip') and os.path.exists(r['bundle_zip']) and r.get('http_load_ok') else 1)" || { log "A4 の一式がありません"; exit 3; }
log "A4 の一式あり"
bash "$X/eval/run_eval.sh" --jobs "A4-s0:core,A4-s0:ext"
log "A4 の評価の終わり"
bash "$X/eval/post_eval.sh" > "$R/eval/logs/post_eval_a4.log" 2>&1
python -I "$HERE/dagger_report.py" > "$R/dagger/logs/dagger_report.log" 2>&1
python -I "$X/eval/render_q1_md.py"
log "集計の終わり"
