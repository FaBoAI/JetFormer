#!/bin/bash
# 閉ループ評価(sim_diverse_train_20261015 の Q1 の評価の段、2026-10-15。10/14 の run_eval.sh の写し):
# ブリッジ(口 8821・10/14 の専用の venv の onnxruntime CPU。読むだけで使う)を立て、Firefox(DISPLAY :0)でアプリの index.html の
# 「連続試験」を走らせる。終わったらブリッジを止め、ブリッジが受け取ったモデルの写し(sim/runtime/uploads)を消す。実車・I2C には触れない。
#   bash run_eval.sh [--jobs EXT:e0,A1-s0:e0,SDIV-s0:core,...] [ほかの browser_eval.py の引数]
# ブリッジは jf-heavy --mem 1500M --lock none、ブラウザは jf-heavy --mem 2500M --lock big --need-mb 2048(Firefox は lock big で順番)。
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
X=$(cd "$HERE/.." && pwd)
PREV=/home/jetson/jetformer/experiments/20261014_sim_dart_margin
R=/home/jetson/jetformer_reports/sim_diverse_train_20261015
PORT=8821
LOG=$R/eval/logs
mkdir -p "$LOG"
TS=$(date +%H%M%S)
free -m > "$LOG/mem_before_$TS.txt"
/home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 1000 -- "$PREV/.venv-bridge/bin/python" -I "$X/browser/run_bridge.py" --port $PORT --no-browser \
  > "$LOG/bridge_${PORT}_$TS.log" 2>&1 &
BRIDGE=$!
echo $BRIDGE > "$LOG/bridge.pid"
cleanup() {
  pkill -f "run_bridge.py --port $PORT" 2>/dev/null; kill $BRIDGE 2>/dev/null; wait $BRIDGE 2>/dev/null
  # ブリッジが受け取ったモデルの写しを消す(protocol eval_Q1.uploads_cleanup)
  if [ -d "$R/sim/runtime/uploads" ]; then
    du -sb "$R/sim/runtime/uploads" | awk '{print "uploads を消す: " $1 " バイト"}' >> "$LOG/uploads_cleanup.log"
    ls "$R/sim/runtime/uploads" | wc -l | awk '{print "  ファイル数 " $1}' >> "$LOG/uploads_cleanup.log"
    rm -rf "$R/sim/runtime/uploads"
  fi
  rm -f "$LOG/bridge.pid"
}
trap cleanup EXIT
for i in $(seq 1 90); do curl -s -o /dev/null http://127.0.0.1:$PORT/index.html && break; sleep 1; done
curl -s -o /dev/null -w "ブリッジ $PORT: HTTP %{http_code}\n" http://127.0.0.1:$PORT/index.html
source /home/jetson/miniconda/etc/profile.d/conda.sh && conda activate jetracer
/home/jetson/bin/jf-heavy --mem 2500M --lock big --need-mb 2048 -- python -I "$HERE/browser_eval.py" "$@"
RC=$?
echo "browser_eval 終了 $RC"
free -m > "$LOG/mem_after_$TS.txt"
exit $RC
