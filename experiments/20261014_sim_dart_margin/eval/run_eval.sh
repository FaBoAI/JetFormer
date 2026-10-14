#!/bin/bash
# 閉ループ評価(sim_dart_20261014 の評価の段、2026-10-14): ブリッジ(口 8812・専用の venv の onnxruntime CPU)を立て、
# Firefox(DISPLAY :0)でアプリの index.html の「連続試験」を走らせる。終わったらブリッジを止める。実車・I2C には触れない。
#   bash run_eval.sh [--jobs EXT:core_ext_e0,A1-s0:core,...] [ほかの browser_eval.py の引数]
# ブリッジは jf-heavy --mem 1500M --lock none、ブラウザは jf-heavy --mem 2500M --lock big(Firefox は lock big で順番)。
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
X=$(cd "$HERE/.." && pwd)
R=/home/jetson/jetformer_reports/sim_dart_20261014
LOG=$R/eval/logs
mkdir -p "$LOG"
TS=$(date +%H%M%S)
free -m > "$LOG/mem_before_$TS.txt"
/home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 1000 -- "$X/.venv-bridge/bin/python" -I "$X/browser/run_bridge.py" --port 8812 --no-browser \
  > "$LOG/bridge_8812_$TS.log" 2>&1 &
BRIDGE=$!
echo $BRIDGE > "$LOG/bridge.pid"
cleanup() { pkill -f "run_bridge.py --port 8812" 2>/dev/null; kill $BRIDGE 2>/dev/null; wait $BRIDGE 2>/dev/null; }
trap cleanup EXIT
for i in $(seq 1 90); do curl -s -o /dev/null http://127.0.0.1:8812/index.html && break; sleep 1; done
curl -s -o /dev/null -w "ブリッジ 8812: HTTP %{http_code}\n" http://127.0.0.1:8812/index.html
source /home/jetson/miniconda/etc/profile.d/conda.sh && conda activate jetracer
/home/jetson/bin/jf-heavy --mem 2500M --lock big --need-mb 2048 -- python -I "$HERE/browser_eval.py" "$@"
RC=$?
echo "browser_eval 終了 $RC"
free -m > "$LOG/mem_after_$TS.txt"
exit $RC
