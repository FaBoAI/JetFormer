#!/bin/bash
# C3(desk_v2.py)の GPU の仕事を、1 回ずつ jf-heavy --lock big で流す(学習のキューと順番に分け合う。あいだ 60 秒)。
#   bash run_c3_gpu.sh tab   <学習の名前> ...     # PyTorch FP32 の AI の記録の集合(1 本ずつ・約 1〜2 分)
#   bash run_c3_gpu.sh shift <学習の名前> ...     # H1 のずれの試し 7 つ(1 本ずつ・約 3 分)
#   bash run_c3_gpu.sh trt   <学習の名前> ...     # TensorRT(B と 2 本ずつ・約 3〜5 分)。名前なしなら B だけ
# 終わったら CPU で: /home/jetson/bin/jf-heavy --lock none --mem 2500M --need-mb 1500 -- \
#   /home/jetson/miniconda/envs/jetracer/bin/python -I experiments/20261025_train_v2/c3/desk_v2.py report
cd /home/jetson/jetformer || exit 1
what="$1"; shift
L=/home/jetson/jetformer/experiments/20261025_train_v2/c3/logs
mkdir -p "$L"
P=/home/jetson/miniconda/envs/jetracer/bin/python
C3=experiments/20261025_train_v2/c3/desk_v2.py
G=(/home/jetson/bin/jf-heavy --lock big --mem 3872M --need-mb 3200 -- nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 "$P" -u "$C3")
one() {   # $1 = ログの名前、残り = desk_v2.py の引数
  local name="$1"; shift
  echo "[$(date '+%F %T')] $name 開始: $*"
  "${G[@]}" "$@" >> "$L/$name.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] $name rc=$rc $(tail -1 "$L/$name.log")"
  sleep 60
  return $rc
}
case "$what" in
  tab|shift)
    for n in "$@"; do one "${what}_$n" predict --what "$what" --runs "$n"; done ;;
  trt)
    if [ $# -eq 0 ]; then one "trt_B" trt; fi
    while [ $# -gt 0 ]; do
      if [ $# -ge 2 ]; then one "trt_$1+$2" trt --runs "$1" "$2"; shift 2
      else one "trt_$1" trt --runs "$1"; shift; fi
    done ;;
  *) echo "使い方: bash run_c3_gpu.sh tab|shift|trt <学習の名前> ..." >&2; exit 2 ;;
esac
echo "[$(date '+%F %T')] 全部おわり"
