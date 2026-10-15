#!/bin/bash
# DAgger の記録(sim_diverse_train_20261015、2026-10-15。protocol.json dagger.procedure 4。10/14 の browser/run_record.sh の写しを作り替えたもの):
# ブリッジ(口 8822、PWM 換算だけを渡す calibration_only_policy.py。10/14 の DART・余裕の線の記録と同じ 1442 / 1498 µs)を立て、
# Firefox(DISPLAY :0)で dagger.html の連続試験と収集器で記録し、合格したシーンを 1 つの ZIP にまとめる。
#   bash run_dagger_record.sh <計画.json> <結果のフォルダー> <まとめの名前> [ほかの dagger_collect.py の引数]
# 始める前に: dagger/prepare_record.py(dagger.html・dist/dagger/)。実車・I2C には触れない。
set -u
PLAN=${1:?計画}
OUT=${2:?結果のフォルダー}
NAME=${3-}   # 空ならブラウザでまとめない(post_dagger.sh の build_collection.py でまとめる)
shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
X=$(cd "$HERE/.." && pwd)
PORT=8822
mkdir -p "$OUT"
source /home/jetson/miniconda/etc/profile.d/conda.sh && conda activate jetracer
free -m > "$OUT/mem_before.txt"
/home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 1000 -- python -I "$X/browser/run_bridge.py" --port $PORT --no-browser \
  --policy calibration_only_policy.py > "$OUT/bridge_$PORT.log" 2>&1 &
BRIDGE=$!
echo $BRIDGE > "$OUT/bridge.pid"
cleanup() { pkill -f "run_bridge.py --port $PORT" 2>/dev/null; kill $BRIDGE 2>/dev/null; wait $BRIDGE 2>/dev/null; rm -f "$OUT/bridge.pid"; }
trap cleanup EXIT
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:$PORT/dagger.html && break; sleep 1; done
curl -s -o /dev/null -w "ブリッジ $PORT: HTTP %{http_code}\n" http://127.0.0.1:$PORT/dagger.html
/home/jetson/bin/jf-heavy --mem 2500M --lock big --need-mb 2048 -- python -I "$HERE/dagger_collect.py" --page dagger.html --plan "$PLAN" \
  --collection-name "$NAME" --out-dir "$OUT" --resume --screens-dir "$OUT/screens" --screens-max 4 --screens-every-s 120 \
  --progress-file "$OUT/progress.jsonl" "$@" >> "$OUT/collect.log" 2>&1
RC=$?
echo "終了 $RC $(date +%T)" >> "$OUT/done.txt"
free -m > "$OUT/mem_after.txt"
exit $RC
