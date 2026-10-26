#!/bin/bash
# C3 を終わった 16 本(B-FAST 3・v2h-e30 4・v2m-e30 4・v2h-e60 4・v2h-e100 1)と 5 本(protocol の five。B を含む)で回す(2026-10-26)。
# GPU の仕事は run_c3_gpu.sh(1 回ごとに jf-heavy --lock big・あいだ 60 秒)で 1 本ずつ流し、学習のキュー(段 3)と順番に分け合う。
# 群ごとに report(CPU・numpy)を流して desk/desk_v2.json・desk_v2.md を書き直す(途中でも見られるように)。
# 段 3(v2h-aug2-e60-s0〜s3)はここでは回さない(学習が終わってから main session が回す)。
#   setsid nohup bash run_c3_all_20261026.sh > logs/run_all_20261026.log 2>&1 &
cd /home/jetson/jetformer || exit 1
C=experiments/20261025_train_v2/c3
P=/home/jetson/miniconda/envs/jetracer/bin/python
L=$C/logs
DONE=$L/run_all_20261026.done
rm -f "$DONE"
ts() { date '+%F %T'; }
report() {
  echo "[$(ts)] report($1)"
  /home/jetson/bin/jf-heavy --lock none --mem 2500M --need-mb 1500 -- "$P" -I $C/desk_v2.py report >> $L/report_20261026.log 2>&1
  echo "[$(ts)] report rc=$? $(tail -1 $L/report_20261026.log)"
}
G=v2_20261025
BF=bfast_20261025
A0=clean_driving_20261011-jetformer_v0-s0-nocmd-stride3-e30
A1=clean_driving_20261011-jetformer_v0-s1-nocmd-stride3-e30
V2=realacc_20261015-v0-s2
V3=realacc_20261015-v0-s3
B=clean_driving_20261011-nocmd-s3-20261011-024626

echo "[$(ts)] 群 A: D3 の拡張なし側(v2h-e60)と最終の腕の候補"
bash $C/run_c3_gpu.sh shift $G-v2h-e60-s1 $G-v2h-e60-s2 $G-v2h-e60-s3
bash $C/run_c3_gpu.sh tab $G-v2h-e60-s0 $G-v2h-e60-s1 $G-v2h-e60-s2 $G-v2h-e60-s3
bash $C/run_c3_gpu.sh trt $G-v2h-e60-s1 $G-v2h-e60-s2 $G-v2h-e60-s3 $G-v2h-e100-s0
report A

echo "[$(ts)] 群 B: B-FAST(参考。腕は拒否権)"
bash $C/run_c3_gpu.sh tab $BF-e15-s0 $BF-e15-s1 $BF-e15-s2
bash $C/run_c3_gpu.sh trt $BF-e15-s0 $BF-e15-s1 $BF-e15-s2 $G-v2h-e30-s0
report B

echo "[$(ts)] 群 C: 段 1・2 の残り(v2h-e30・v2m-e30・v2h-e100)"
bash $C/run_c3_gpu.sh tab $G-v2h-e30-s0 $G-v2h-e30-s1 $G-v2h-e30-s2 $G-v2h-e30-s3 \
  $G-v2m-e30-s0 $G-v2m-e30-s1 $G-v2m-e30-s2 $G-v2m-e30-s3 $G-v2h-e100-s0
bash $C/run_c3_gpu.sh trt $G-v2h-e30-s1 $G-v2h-e30-s2 $G-v2h-e30-s3 $G-v2m-e30-s0 $G-v2m-e30-s1 $G-v2m-e30-s2 $G-v2m-e30-s3
report C

echo "[$(ts)] 群 D: 基準の 5 本(B の tab は済み。v0-s2・v0-s3 はエンジンが無いので TensorRT なし)"
bash $C/run_c3_gpu.sh tab $A0 $A1 $V2 $V3
bash $C/run_c3_gpu.sh trt $A0 $A1
report D

echo "[$(ts)] 群 E: ずれの試しの残り(参考。D3 には効かない)"
bash $C/run_c3_gpu.sh shift $BF-e15-s0 $BF-e15-s1 $BF-e15-s2 $G-v2h-e100-s0 \
  $G-v2h-e30-s0 $G-v2h-e30-s1 $G-v2h-e30-s2 $G-v2h-e30-s3 \
  $G-v2m-e30-s0 $G-v2m-e30-s1 $G-v2m-e30-s2 $G-v2m-e30-s3 \
  $B $A0 $A1 $V2 $V3
report E
echo "[$(ts)] 全部おわり"
date '+%F %T' > "$DONE"
