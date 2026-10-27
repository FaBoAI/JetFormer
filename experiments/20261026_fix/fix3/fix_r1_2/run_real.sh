#!/bin/bash
# 直す 1 回目・2 組目(2026-10-26 夜): 本物の記録 10/23・10/24 で夜の道具を、直す前(before/*.orig)と後で流して比べる台本。読むだけ。
#   bash run_real.sh <作業場所(消してよい)>
# 作業場所にだけ書く(data/raw・data/labels・data/reports・~/jetformer_reports には書かない。car_check は --json で作業場所へ)。
# GPU なし(jf-heavy --lock none --mem 1500M)。
set -u
W=$1
JF=/home/jetson/jetformer
B=$JF/experiments/20261026_fix/fix3/fix_r1_2/before
PY=/home/jetson/miniconda/envs/jetracer/bin/python
H="/home/jetson/bin/jf-heavy --lock none --mem 1500M --"
mkdir -p "$W/old_root/tools" "$W/out_old" "$W/out_new"
# 直す前の根: tools は本物への symlink、直したファイルだけ写し(resolve() で根が作業場所になるように)
for f in $JF/tools/*.py; do ln -sf "$f" "$W/old_root/tools/$(basename $f)"; done
for f in standing_3lap quick_eval_by_params car_check; do rm -f "$W/old_root/tools/$f.py"; cp "$B/$f.py.orig" "$W/old_root/tools/$f.py"; done
ln -sfn $JF/jetformer "$W/old_root/jetformer"
cp "$B/collect.py.orig" "$W/old_root/collect.py"
cp "$B/runs_1103.py.orig" "$W/old_root/runs_1103.py"
run() {   # run <old|new> <root> <collect> <runs_1103>
  local tag=$1 R=$2 COL=$3 RUNS=$4 O=$W/out_$1
  cd "$O" || exit 1
  for d in 20261023 20261024; do
    $H $PY -I "$R/tools/standing_3lap.py" --day $d --json "$O/standing_$d.json" --md "$O/standing_$d.md" > "$O/standing_$d.stdout" 2> "$O/standing_$d.stderr"; echo "$tag standing $d rc=$?"
    $H $PY -I "$R/tools/quick_eval_by_params.py" --day $d --out "$O/qp_$d" > "$O/qp_$d.stdout" 2> "$O/qp_$d.stderr"; echo "$tag qp $d rc=$?"
    $H $PY -I "$COL" --day $d --out "$O/launches_$d.json" > "$O/launches_$d.stdout" 2> "$O/launches_$d.stderr"; echo "$tag collect $d rc=$?"
    $H $PY -I "$RUNS" --day $d --json > "$O/runs_$d.stdout" 2> "$O/runs_$d.stderr"; echo "$tag runs_1103 $d rc=$?"
  done
  $H $PY -I "$COL" --out "$O/launches.json" > "$O/launches.stdout" 2> "$O/launches.stderr"; echo "$tag collect default rc=$?"
  $H $PY -I "$COL" --day 20261024 --first-only --out "$O/launches_20261024_first.json" > "$O/launches_20261024_first.stdout" 2> "$O/launches_20261024_first.stderr"; echo "$tag collect 1024 first-only rc=$?"
  mkdir -p "$O/car_check"
  for lg in $(ls $JF/data/raw | grep -E '^2026102[34]-' ); do
    [ -f "$JF/data/raw/$lg/log.json" ] || continue
    $H $PY -I "$R/tools/car_check.py" "$lg" --json "$O/car_check/$lg.json" > "$O/car_check/$lg.txt" 2>&1; echo "$lg $?" >> "$O/car_check/rc.txt"
  done
  echo "$tag car_check done $(wc -l < $O/car_check/rc.txt)"
}
run old "$W/old_root" "$W/old_root/collect.py" "$W/old_root/runs_1103.py"
run new "$JF" "$JF/experiments/20261026_fix/diag_launch/collect.py" "$JF/experiments/20261026_fix/plan/runs_1103.py"
