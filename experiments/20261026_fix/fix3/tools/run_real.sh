#!/bin/bash
# 係 tools(fix3、2026-10-26 夜): 本物の記録 10/23・10/24 で夜の道具を流し、直す前と後の出力を比べるための台本。読むだけ。
#   bash run_real.sh <道具の根(tools/ と diag_launch/collect.py がある)> <collect.py の場所> <出力の場所>
# 出力の場所にだけ書く(data/raw・data/labels・~/jetformer_reports には書かない)。GPU なし(jf-heavy --lock none --mem 1500M)。
set -u
ROOTT=$1; COL=$2; OUT=$3
PY=/home/jetson/miniconda/envs/jetracer/bin/python
H="/home/jetson/bin/jf-heavy --lock none --mem 1500M --"
mkdir -p "$OUT"
cd "$OUT" || exit 1
for d in 20261023 20261024; do
  $H $PY -I "$ROOTT/tools/standing_3lap.py" --day $d --json "$OUT/standing_$d.json" --md "$OUT/standing_$d.md" > "$OUT/standing_$d.stdout" 2> "$OUT/standing_$d.stderr"; echo "standing $d rc=$?"
  $H $PY -I "$ROOTT/tools/quick_eval_by_params.py" --day $d --out "$OUT/qp_$d" > "$OUT/qp_$d.stdout" 2> "$OUT/qp_$d.stderr"; echo "qp $d rc=$?"
  $H $PY -I "$COL" --day $d --out "$OUT/launches_$d.json" > "$OUT/launches_$d.stdout" 2> "$OUT/launches_$d.stderr"; echo "collect $d rc=$?"
done
$H $PY -I "$COL" --out "$OUT/launches.json" > "$OUT/launches.stdout" 2> "$OUT/launches.stderr"; echo "collect default rc=$?"
$H $PY -I "$COL" --day 20261024 --first-only --out "$OUT/launches_20261024_first.json" > "$OUT/launches_20261024_first.stdout" 2> "$OUT/launches_20261024_first.stderr"; echo "collect 1024 first-only rc=$?"
