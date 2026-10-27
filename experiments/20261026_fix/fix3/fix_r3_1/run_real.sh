#!/bin/bash
# 直す 3 回目・1 組目(2026-10-27): 本物の記録 10/23・10/24 で ⑤ collect.py と car_check --latest を、直す前(tests/fixtures/fix3_r3_1_before)と
# 後で流す。読むだけ。作業場所にだけ書く(data/raw・data/labels・data/reports・diag_launch/out には書かない)。GPU なし。
#   bash run_real.sh <作業場所(消してよい)>
set -u
W=$1
JF=/home/jetson/jetformer
B=$JF/tests/fixtures/fix3_r3_1_before
PY=/home/jetson/miniconda/envs/jetracer/bin/python
H="/home/jetson/bin/jf-heavy --lock none --mem 1500M --"
mkdir -p "$W/old_root/tools" "$W/out_old" "$W/out_new"
for f in $JF/tools/*.py; do ln -sf "$f" "$W/old_root/tools/$(basename $f)"; done
rm -f "$W/old_root/tools/car_check.py"; cp "$B/car_check.py" "$W/old_root/tools/car_check.py"
ln -sfn $JF/jetformer "$W/old_root/jetformer"
cp "$B/collect.py" "$W/old_root/collect.py"
run() {   # run <old|new> <root> <collect>
  local tag=$1 R=$2 COL=$3 O=$W/out_$1
  cd "$O" || exit 1
  for d in 20261023 20261024; do
    for v in "" "--first-only" "--ignore-gain"; do
      n=${v#--}; n=${n:-plain}
      $H $PY -I "$COL" --day $d $v --out "$O/launches_${d}_$n.json" > "$O/launches_${d}_$n.stdout" 2> "$O/launches_${d}_$n.stderr"; echo "$tag collect $d $n rc=$?"
    done
  done
  $H $PY -I "$COL" --out "$O/launches.json" > "$O/launches.stdout" 2> "$O/launches.stderr"; echo "$tag collect default rc=$?"
  for d in 20261023 20261024; do
    $H $PY -I "$R/tools/car_check.py" --latest --day $d --no-json > "$O/cc_latest_$d.stdout" 2> "$O/cc_latest_$d.stderr"; echo "$tag car_check --latest $d rc=$?" | tee "$O/cc_latest_$d.rc"
  done
}
run old "$W/old_root" "$W/old_root/collect.py"
run new "$JF" "$JF/experiments/20261026_fix/diag_launch/collect.py"
