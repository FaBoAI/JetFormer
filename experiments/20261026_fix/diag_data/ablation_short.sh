#!/bin/bash
# 原因 A の確かめの短い学習(この係は起動しない。main session が決めて起動する)。
# 同じ種 0・15 エポック・v2-H と同じ引数で、表だけを変えた腕を順に学習し、最後に probe_gates.py で best・last を見る。
# 1 本 約 11 分(v2h-e30 は 30 エポック 1,328 秒)。既定の 2 腕で 約 25 分、4 腕で 約 50 分。GPU は jf-heavy --lock big で 1 つずつ。
# 書くのは experiments/20261026_fix/diag_data/out/ablation/runs/ の下だけ(data/runs・既定値・protocol は触らない)。
#   bash experiments/20261026_fix/diag_data/ablation_short.sh            # same・noTK
#   bash experiments/20261026_fix/diag_data/ablation_short.sh same noTK noTK_noLB tkB
set -u
cd /home/jetson/jetformer
P=/home/jetson/miniconda/envs/jetracer/bin/python
D=experiments/20261026_fix/diag_data
A=$D/out/ablation
EPOCHS=${EPOCHS:-15}
ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || ARMS=(same noTK)
G="/home/jetson/bin/jf-heavy --mem 3872M --lock big --need-mb 3200 -- nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2"
for arm in "${ARMS[@]}"; do
  case "$arm" in
    same)      EXTRA="--drop-mask $A/drop_v2h_same.npz";;
    noTK)      EXTRA="--drop-mask $A/drop_v2h_noTK.npz";;
    noTK_noLB) EXTRA="--drop-mask $A/drop_v2h_noTK_noLB.npz";;
    tkB)       EXTRA="--drop-mask $A/drop_v2h_same.npz --label-override-npz $A/override_tk_thr_B.npz";;
    *) echo "知らない腕: $arm" >&2; exit 2;;
  esac
  OUTD=$A/runs/abl-$arm-e$EPOCHS-s0
  if [ -f "$OUTD/result.json" ] && grep -q '"state": "done"' "$OUTD/result.json"; then echo "$arm: 済み"; continue; fi
  R=""; [ -f "$OUTD/last.pth" ] && R="--resume"
  mkdir -p "$OUTD"
  $G choom -n 1000 -- $P -I experiments/20261015_real_accuracy/train/nothp.py $P -u tools/train.py --policy jetformer_v0 \
     --data data/datasets/dataset_v2_20261025 $EXTRA --epochs $EPOCHS --lr 3e-4 --stride 3 --batch 64 --workers 2 \
     --no-command-input --impact-g 2.5 --seed 0 --out "$OUTD" $R >> "$OUTD/train.log" 2>&1 || { echo "$arm: 学習が止まった(rc $?)"; exit 1; }
  sleep 30
done
RUNS=()
for arm in "${ARMS[@]}"; do RUNS+=("$A/runs/abl-$arm-e$EPOCHS-s0"); done
for ck in best last; do
  $G $P -u $D/probe_gates.py run --ckpt $ck --runs "${RUNS[@]}" --out $A/probe_$ck.json
done
echo "見るところ: $A/probe_best.json・probe_last.json(st23_thr_vs_b_us・stand_brake_side_frac・low10_brake_frac)"
