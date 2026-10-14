#!/bin/bash
# recover の試しの学習(50 万ステップ)を、途中で止める合図 → 同じコマンドで last.pt から続く、の形で動かす(実装の段の再開の確かめ。2026-10-13)
set -u
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
R=/home/jetson/jetformer_reports/sim_rl_modes_20261013/rl
cd "$EXP"
rm -rf "$R/smoke-recover"
rl/run_mode.sh smoke-recover recover 500000 --ckpt-every 250000 --save-every 100000 --tag smoke &
W=$!
# lock rl の順番待ちは長いことがあるので、時間で打ち切らない(学習が 14 回更新するまで待つ)
while kill -0 $W 2>/dev/null; do
  sleep 2
  if [ -f "$R/smoke-recover/progress.jsonl" ] && [ "$(wc -l < "$R/smoke-recover/progress.jsonl")" -ge 14 ]; then
    pkill -TERM -f "python -u rl/train_modes.py --run recover --steps 500000 --out $R/smoke-recover" ; break
  fi
done
wait $W
echo "[$(date -Is)] 止める合図のあと: last.pt $(ls -la $R/smoke-recover/last.pt 2>/dev/null | awk '{print $6,$7,$8}') progress $(wc -l < $R/smoke-recover/progress.jsonl)" >> "$R/logs/smoke-recover.log"
rl/run_mode.sh smoke-recover recover 500000 --ckpt-every 250000 --save-every 100000 --tag smoke
echo "[$(date -Is)] resumes: $(python3 -c "import json;print(json.load(open('$R/smoke-recover/run.json'))['resumes'])")" >> "$R/logs/smoke-recover.log"
