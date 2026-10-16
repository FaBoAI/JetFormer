#!/bin/bash
# S5(2026-10-16、シミュレーター較正): ブラウザの S-DIV(10/15 の JetFormer)を、写しのシミュレーターの 3 つのページで走らせる。
#   index.html(元の物理)・calib.html(較正の物理 P_cal)・calib_cap.html(P_cal + 区間の速度上限 v1.1、位置の区間)
# ブリッジ(口 8831、10/14 の専用の venv の onnxruntime CPU)を立て、Firefox(DISPLAY :0、geckodriver 4451)でアプリの「連続試験」を走らせる。
# 終わったらブリッジを止め、ブリッジが受け取ったモデルの写し(sim/runtime/uploads)を消す。実車・I2C には触れない。
#   bash run_s5.sh "index.html=SDIV-s0:e0" "calib.html=SDIV-s0:core" ...
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PREV=/home/jetson/jetformer/experiments/20261014_sim_dart_margin
R=/home/jetson/jetformer_reports/sim_calibration_20261016
PORT=8831
LOG=$R/browser/logs
mkdir -p "$LOG"
TS=$(date +%H%M%S)
free -m > "$LOG/mem_before_$TS.txt"
/home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 1000 -- "$PREV/.venv-bridge/bin/python" -I "$HERE/run_bridge.py" --port $PORT --no-browser \
  > "$LOG/bridge_${PORT}_$TS.log" 2>&1 &
BRIDGE=$!
cleanup() {
  pkill -f "run_bridge.py --port $PORT" 2>/dev/null; kill $BRIDGE 2>/dev/null; wait $BRIDGE 2>/dev/null
  if [ -d "$R/sim/runtime/uploads" ]; then
    du -sb "$R/sim/runtime/uploads" | awk '{print "uploads を消す: " $1 " バイト"}' >> "$LOG/uploads_cleanup.log"
    rm -rf "$R/sim/runtime/uploads"
  fi
}
trap cleanup EXIT
for i in $(seq 1 90); do curl -s -o /dev/null http://127.0.0.1:$PORT/index.html && break; sleep 1; done
for p in index.html calib.html calib_cap.html; do curl -s -o /dev/null -w "ブリッジ $PORT $p: HTTP %{http_code}\n" http://127.0.0.1:$PORT/$p; done
source /home/jetson/miniconda/etc/profile.d/conda.sh && conda activate jetracer
RC=0
for spec in "$@"; do
  page=${spec%%=*}; jobs=${spec#*=}
  echo "== $page: $jobs"
  /home/jetson/bin/jf-heavy --mem 2500M --lock big --need-mb 2048 -- python -I "$HERE/browser_eval.py" --page "$page" --jobs "$jobs" --view follow --shots 8,60,120,200,300 || RC=$?
done
free -m > "$LOG/mem_after_$TS.txt"
exit $RC
