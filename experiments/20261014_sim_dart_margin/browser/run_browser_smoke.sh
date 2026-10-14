#!/bin/bash
# 準備の段の試し(ブラウザ): D1・D2・D3 の train-00 を 1 シーンずつ、Firefox(DISPLAY :0)でアプリの連続試験と収集器で記録し、
# データセットごとに 1 つの ZIP にまとめる(2026-10-14、sim_dart_20261014)。
#   jf-heavy --mem 2500M --lock big -- bash run_browser_smoke.sh <結果のフォルダー> ["D1 D2 D3"]
# ブリッジ(ポート 8811)はこのスクリプトの中で起動し、最後に止める。実車・I2C には触れない。
set -u
OUT=${1:?結果のフォルダー}
SETS=${2:-D1 D2 D3}
HERE=$(cd "$(dirname "$0")" && pwd)
R=/home/jetson/jetformer_reports/sim_dart_20261014
mkdir -p "$OUT"
source /home/jetson/miniconda/etc/profile.d/conda.sh && conda activate jetracer
free -m > "$OUT/mem_before.txt"
python -I "$HERE/run_bridge.py" --port 8811 --no-browser --policy calibration_only_policy.py > "$OUT/bridge_8811.log" 2>&1 &
BRIDGE=$!
echo $BRIDGE > "$OUT/bridge.pid"
trap 'kill $BRIDGE 2>/dev/null; wait $BRIDGE 2>/dev/null' EXIT
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:8811/d1.html && break; sleep 1; done
for D in $SETS; do
  d=$(echo $D | tr 'D' 'd')
  case $D in D1) NAME=smoke-rl-dart-fast;; D2) NAME=smoke-rl-margin-line;; D3) NAME=smoke-rl-margin-dart;; esac
  python -I "$HERE/browser_collect.py" --page ${d}.html --plan $R/record_prep/plan_${D}.json --max-scenes 1 \
    --k-table $R/record_prep/prediction_${D}.json --collection-name $NAME --out-dir "$OUT/$D" \
    --screens-dir "$OUT/$D/screens" --screens-max 1 --progress-file "$OUT/$D/progress.jsonl" > "$OUT/$D.log" 2>&1
  echo "$D 終了 $?" >> "$OUT/done.txt"
done
free -m > "$OUT/mem_after.txt"
