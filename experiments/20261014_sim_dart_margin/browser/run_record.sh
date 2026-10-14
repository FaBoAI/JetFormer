#!/bin/bash
# 記録の段で使う: D1・D2・D3 のどれか 1 つの 30 シーンを、Firefox(DISPLAY :0)でアプリの連続試験と収集器で記録し、
# 合格したシーンを 1 つの ZIP にまとめる(2026-10-14、sim_dart_20261014)。だめなシーンは Node の予想で通った次の k で取り直す(2 回まで)。
#   jf-heavy --mem 2500M --lock big -- bash run_record.sh D1 <結果のフォルダー>
# ブリッジ(ポート 8811)はこのスクリプトの中で起動し、最後に止める。実車・I2C には触れない。
# 始める前に: record/install_dart.py(k の表を dist に写す)。終わったら: record/verify_dart_collection.py → record/import_dataset.py →(D1・D3)relabel/relabel_dataset.py --mode write
set -u
D=${1:?D1・D2・D3}
OUT=${2:?結果のフォルダー}
HERE=$(cd "$(dirname "$0")" && pwd)
R=/home/jetson/jetformer_reports/sim_dart_20261014
case $D in D1) NAME=rl-dart-fast-10min;; D2) NAME=rl-margin-line-10min;; D3) NAME=rl-margin-dart-10min;; *) echo "D1・D2・D3 のどれか"; exit 2;; esac
d=$(echo $D | tr 'D' 'd')
mkdir -p "$OUT"
source /home/jetson/miniconda/etc/profile.d/conda.sh && conda activate jetracer
free -m > "$OUT/mem_before.txt"
python -I "$HERE/run_bridge.py" --port 8811 --no-browser --policy calibration_only_policy.py > "$OUT/bridge_8811.log" 2>&1 &
BRIDGE=$!
echo $BRIDGE > "$OUT/bridge.pid"
trap 'kill $BRIDGE 2>/dev/null; wait $BRIDGE 2>/dev/null' EXIT
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:8811/${d}.html && break; sleep 1; done
# 記録の段(14:30)で足した: 6 シーンずつの連続試験(--chunk 6)と、途中で止まったときの続き(--resume。合格済みのシーンは飛ばす)。
# Firefox が途中で落ちたら(result.json に error)、同じ結果のフォルダーで 2 回まで続きから記録する
for TRY in 1 2 3; do
  python -I "$HERE/browser_collect.py" --page ${d}.html --plan $R/record_prep/plan_${D}.json \
    --k-table $R/record_prep/prediction_${D}.json --collection-name $NAME --out-dir "$OUT" --chunk 6 --resume \
    --screens-dir "$OUT/screens" --screens-max 6 --screens-every-s 420 --progress-file "$OUT/progress.jsonl" >> "$OUT/collect.log" 2>&1
  RC=$?
  echo "終了 $RC(試し $TRY)" >> "$OUT/done.txt"
  python -I -c "import json,sys;r=json.load(open('$OUT/result.json'));sys.exit(1 if r.get('error') else 0)" && break
  sleep 10
done
# アプリのまとめが空き容量の取り置き(20 GiB)で止まったときは、取り置きだけ小さくした同じ処理でまとめる(record/build_collection.py)
if ! python -I -c "import json,sys;r=json.load(open('$OUT/result.json'));sys.exit(0 if r.get('collection') else 1)"; then
  python -I "$HERE/../record/build_collection.py" --result "$OUT/result.json" --name $NAME --out "$OUT/collection.json" > "$OUT/collection.log" 2>&1
  echo "まとめ(build_collection) $?" >> "$OUT/done.txt"
fi
free -m > "$OUT/mem_after.txt"
