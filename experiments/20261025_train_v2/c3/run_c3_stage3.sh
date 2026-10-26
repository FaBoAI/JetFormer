#!/bin/bash
# 段 3(v2h-aug2-e60-s0〜s3)の C3 を回す(main session が、段 3 の学習・エンジン・H1 と介入の予測がキューで全部終わってから流す)。
# 2026-10-26 に用意しただけで、まだ流していない。GPU の仕事は run_c3_gpu.sh(1 回ごとに jf-heavy --lock big・あいだ 60 秒)。
#   cd ~/jetformer && setsid nohup bash experiments/20261025_train_v2/c3/run_c3_stage3.sh \
#       > experiments/20261025_train_v2/c3/logs/run_stage3.log 2>&1 < /dev/null &
# 順番: ① D3 の robust_dl1_us(shift 4 本・約 3 分 × 4)→ report → ② tab 4 本(約 2 分 × 4)→ ③ TensorRT 2 回(約 5 分 × 2)→ report・門の表
# 「そのまま」は eval/pred の保存した best(キューの予測)を使うので、4 本とも eval/pred/<名前>.npz が無ければ始めない。
cd /home/jetson/jetformer || exit 1
C=experiments/20261025_train_v2/c3
P=/home/jetson/miniconda/envs/jetracer/bin/python
L=$C/logs
REP=/home/jetson/jetformer_reports/train_v2_20261025
N=(v2_20261025-v2h-aug2-e60-s0 v2_20261025-v2h-aug2-e60-s1 v2_20261025-v2h-aug2-e60-s2 v2_20261025-v2h-aug2-e60-s3)
for n in "${N[@]}"; do
  for f in data/runs/$n/best.pth data/runs/$n/jetformer_fp16.engine data/runs/$n/engine_check.json $REP/eval/pred/$n.npz; do
    [ -f "$f" ] || { echo "まだ: $f(段 3 のキューが終わってから流す)" >&2; exit 3; }
  done
done
ts() { date '+%F %T'; }
report() {
  echo "[$(ts)] report($1)"
  /home/jetson/bin/jf-heavy --lock none --mem 2500M --need-mb 1500 -- "$P" -I $C/desk_v2.py report >> $L/report_stage3.log 2>&1
  echo "[$(ts)] report rc=$? $(tail -1 $L/report_stage3.log)"
  python3 -I $C/gates_table.py
}
bash $C/run_c3_gpu.sh shift "${N[@]}"
report shift
bash $C/run_c3_gpu.sh tab "${N[@]}"
bash $C/run_c3_gpu.sh trt "${N[@]}"
report all
echo "[$(ts)] 段 3 の C3 おわり。続けて: /home/jetson/bin/jf-heavy --lock none --mem 1500M -- $P -I experiments/20261025_train_v2/eval_v2.py judge"
