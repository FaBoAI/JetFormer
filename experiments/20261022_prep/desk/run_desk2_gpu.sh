#!/bin/bash
# 段 Desk(10/23・A10)の GPU の仕事を順に(1 つずつ・jf-heavy --lock big)。出来たものは飛ばす。車・装置には触れない
#   setsid nohup bash experiments/20261022_prep/desk/run_desk2_gpu.sh >> experiments/20261022_prep/desk/logs/run_desk2_gpu.log 2>&1 &
# 10/21 の run_desk_gpu.sh と同じ形。エンジンの確認(engine_check.json)は学習の段で済んでいる(作り直さない)
set -u
cd /home/jetson/jetformer
PY=/home/jetson/miniconda/envs/jetracer/bin/python
D=experiments/20261022_prep/desk
NEW="SLOW-SEC0 SLOW1 SLOW-SEC1"
run() {   # メモリ上限 3G で落ちたら(rc 137)3872M で 1 回だけやり直す(学習の段と同じ)
  local rc
  /home/jetson/bin/jf-heavy --mem 3G --lock big --need-mb 3200 -- nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 $PY -u "$@"
  rc=$?
  if [ $rc -eq 137 ]; then
    echo "[$(date +%T)] rc 137 → 3872M でやり直す"
    /home/jetson/bin/jf-heavy --mem 3872M --lock big --need-mb 3200 -- nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 $PY -u "$@"
    rc=$?
  fi
  return $rc
}
echo "[$(date +%T)] 始め"
if [ ! -s $D/logs/crosscheck_B.json ] || ! grep -q '"pass": true' $D/logs/crosscheck_B.json; then
  run $D/desk_gpu2.py crosscheck || { echo "[$(date +%T)] crosscheck 失敗"; exit 1; }
fi
run $D/desk_gpu2.py predict --ids $NEW || { echo "[$(date +%T)] predict 失敗"; exit 1; }
if [ ! -s $D/logs/sanity.json ] || ! $PY -c "import json,sys;d=json.load(open('$D/logs/sanity.json'));sys.exit(0 if all(d['candidates'].get(c,{}).get('sanity_ok') is not None for c in '$NEW'.split()) else 1)"; then
  run $D/desk_gpu2.py sanity --ids $NEW || { echo "[$(date +%T)] sanity 失敗"; exit 1; }
fi
echo "[$(date +%T)] 終わりました"
