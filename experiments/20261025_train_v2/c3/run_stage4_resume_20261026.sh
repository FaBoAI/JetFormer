#!/bin/bash
# 段 4 の C3 の続き(2026-10-26 08:4x。最後の評価の係)。07:40 に shift s1 と trt が Terminated(そのあと 07:42 に電源が落ちた)ので、
# 残りだけ流す: shift v2h-msf-aug2-e60-s1 → trt msf s0+s1(B は毎回入る)→ report --ens(D4 の 2 組)→ gates_table。
cd /home/jetson/jetformer || exit 1
C=experiments/20261025_train_v2/c3
P=/home/jetson/miniconda/envs/jetracer/bin/python
L=$C/logs
DONE=$L/run_stage4_resume_20261026.done
rm -f "$DONE"
ts() { date '+%F %T'; }
G=v2_20261025
echo "[$(ts)] 段 4 の C3 の続き 開始"
bash $C/run_c3_gpu.sh shift $G-v2h-msf-aug2-e60-s1 || echo "[$(ts)] shift rc!=0"
bash $C/run_c3_gpu.sh trt $G-v2h-msf-aug2-e60-s0 $G-v2h-msf-aug2-e60-s1 || echo "[$(ts)] trt rc!=0"
echo "[$(ts)] report --ens"
/home/jetson/bin/jf-heavy --lock none --mem 2500M --need-mb 1500 -- "$P" -I $C/desk_v2.py report \
  --ens $G-v2h-aug2-e60-s0+$G-v2h-msf-aug2-e60-s0 $G-v2h-aug2-e60-s1+$G-v2h-msf-aug2-e60-s1 >> $L/report_stage4_20261026.log 2>&1
echo "[$(ts)] report rc=$? $(tail -1 $L/report_stage4_20261026.log)"
python3 -I $C/gates_table.py >> $L/report_stage4_20261026.log 2>&1
echo "[$(ts)] gates_table rc=$? $(tail -1 $L/report_stage4_20261026.log)"
echo "[$(ts)] 段 4 の C3 の続き おわり"
date '+%F %T' > "$DONE"
