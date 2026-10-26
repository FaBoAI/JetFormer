#!/bin/bash
# 段 Train(2026-10-23、A10): gpu_queue2.tsv の 3 本の学習とエンジンを 1 本ずつ回す。experiments/20261021_bench_prep/gpu_queue.sh の写し
# (変えたのは表・手伝い・ログの置き場所と締め切り 2026-10-23 05:20 だけ。メモリの上限は表のコマンドで最初から 3872M)。もとの説明:
# 段 G2(2026-10-21): gpu_queue.tsv の学習とエンジンを 1 本ずつ回す。電源断のあとは同じコマンドをもう一度動かせば続きから。
#
#   起動(見守りは logs/gpu_queue2.log と python -I gpu_queue2.py status):
#     cd /home/jetson/jetformer && setsid nohup bash experiments/20261022_prep/gpu_queue2.sh \
#         >> experiments/20261022_prep/logs/gpu_queue2.log 2>&1 < /dev/null &
#
# 1 本ごとに:
#   ① 学習: 済み(result.json の state が done・30 エポック・best.pth・ONNX)なら飛ばす。last.pth があれば --resume で続きから。
#      キューのコマンドのまま(jf-heavy --lock big --mem 3G --need-mb 3200・nothp.py・choom・nice)。
#      失敗(終わったのに済んでいない)は 1 回だけやり直す(メモリの上限で止められた終了コード 137 のときだけ上限を 3872M に)。
#      電源断で途切れた試み(終わりの記録が無い)は失敗に数えない。GPU_QUEUE_CUTOFF(既定 2026-10-23 05:20)を過ぎたら学習を始めない。
#   ② エンジン: tools/build_engine.py(FP16、jetformer_fp16.engine。tmp → check_io → 置き換え)。1 回だけやり直す。
#   ③ 確認: experiments/20261013_sim_cotrain/train/engine_load_check.py(drive.model_info・実フレーム 64 個で TensorRT と
#      PyTorch FP32 の差 ≤ 2.5 µs・速さ)→ engine_check.json → drive.model_info の drivable を確かめる。
#   GPU の手順のあとは毎回 GPU_QUEUE_GAP_S 秒(既定 60)あけて、ほかの段(試験・エンジンの確認)に GPU の順番を回す。
#   管理画面が学習・走行・収集中なら待つ。車は動かさない・装置に触れない・jetformer-web は再起動しない。
set -u
REPO=/home/jetson/jetformer
HERE=$REPO/experiments/20261022_prep
LOGS=$HERE/logs
PY=/home/jetson/miniconda/envs/jetracer/bin/python
QUEUE=$HERE/gpu_queue2.tsv
CUTOFF=${GPU_QUEUE_CUTOFF:-"2026-10-23 05:20"}
GAP_S=${GPU_QUEUE_GAP_S:-60}
GPU="/home/jetson/bin/jf-heavy --lock big --mem 3G --need-mb 3200 --"
LOAD_CHECK=$REPO/experiments/20261013_sim_cotrain/train/engine_load_check.py
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
mkdir -p "$LOGS"
cd "$REPO" || exit 1

h() { "$PY" -I "$HERE/gpu_queue2.py" "$@"; }
log() { echo "[$(date '+%F %T')] $*"; }
past_cutoff() { [ "$(date +%s)" -ge "$(date -d "$CUTOFF" +%s)" ]; }
gap() { log "  ${GAP_S} 秒あけます(ほかの段に GPU の順番を回す)"; sleep "$GAP_S"; }
wait_web() {
  local n=0
  while h webbusy >/dev/null; do
    [ $((n % 10)) -eq 0 ] && log "  管理画面が使用中のため待っています($(h webbusy))"
    n=$((n + 1)); sleep 30
  done
}

# 同時に 2 つ動かさない
exec 8>"$LOGS/gpu_queue2.lock" < /dev/null
if ! flock -n 8; then
  log "gpu_queue2.sh はもう動いています。終わります"
  exit 0
fi

log "==== 始めます(学習を始めてよいのは $CUTOFF まで。あける時間 ${GAP_S} 秒)===="
h status

train_one() {  # 名前 コマンド → 0 = 学習が済んだ
  local name=$1 cmd=$2 run=$REPO/data/runs/$1 c rc f
  while ! h trained "$name"; do
    f=$(h fails "$name" train)
    if [ "$f" -ge 2 ]; then
      log "$name: 学習が 2 回失敗しました。次へ進みます(logs/train_$name.log を見てください)"
      return 1
    fi
    if past_cutoff; then
      log "$name: $CUTOFF を過ぎたので学習を始めません"
      return 2
    fi
    c=$cmd
    if [ "$f" -ge 1 ] && [ "$(h last-rc "$name" train)" = "137" ]; then
      c=${c/--mem 3G/--mem 3872M}
      log "$name: 前回はメモリの上限で止められた(137)ので、やり直しは上限 3872M"
    fi
    [ -f "$run/last.pth" ] && c="$c --resume"
    wait_web
    mkdir -p "$run"
    log "$name: 学習を始めます(失敗の数 $f)"
    h begin "$name" train
    echo "==== $(date '+%F %T') 開始(cd ~/jetformer && $c)====" | tee -a "$run/train.log" >> "$LOGS/train_$name.log"
    bash -c "$c" < /dev/null 2>&1 | tee -a "$run/train.log" >> "$LOGS/train_$name.log"
    rc=${PIPESTATUS[0]}
    echo "==== $(date '+%F %T') 終了 rc=$rc ====" | tee -a "$run/train.log" >> "$LOGS/train_$name.log"
    log "$name: $(h end "$name" train "$rc")"
    gap
  done
  return 0
}

engine_one() {  # 名前 → 0 = エンジンがある
  local name=$1 run=$REPO/data/runs/$1 rc f
  while h need-engine "$name"; do
    f=$(h fails "$name" engine)
    if [ "$f" -ge 2 ]; then
      log "$name: エンジンの作成が 2 回失敗しました(logs/engine_$name.log)"
      return 1
    fi
    wait_web
    log "$name: エンジンを作ります"
    h begin "$name" engine
    echo "==== $(date '+%F %T') 開始 ====" >> "$LOGS/engine_$name.log"
    $GPU nice -n 10 "$PY" -u tools/build_engine.py "$run/jetformer.onnx" "$run/jetformer_fp16.engine" >> "$LOGS/engine_$name.log" 2>&1
    rc=$?
    echo "==== $(date '+%F %T') 終了 rc=$rc ====" >> "$LOGS/engine_$name.log"
    log "$name: $(h end "$name" engine "$rc")"
    gap
  done
  return 0
}

check_one() {  # 名前 → 0 = engine_check.json が合格で drivable
  local name=$1 run=$REPO/data/runs/$1 rc f
  while h need-check "$name"; do
    f=$(h fails "$name" check)
    if [ "$f" -ge 2 ]; then
      log "$name: エンジンの確認が 2 回失敗しました(logs/check_$name.log)"
      return 1
    fi
    wait_web
    log "$name: エンジンを確かめます(走行タブと同じ手順で読む・実フレーム 64 個・速さ)"
    h begin "$name" check
    echo "==== $(date '+%F %T') 開始 ====" >> "$LOGS/check_$name.log"
    $GPU nice -n 10 "$PY" -u "$LOAD_CHECK" "$run" "$LOGS/engine_check_$name.json" >> "$LOGS/check_$name.log" 2>&1
    rc=$?
    echo "==== $(date '+%F %T') 終了 rc=$rc ====" >> "$LOGS/check_$name.log"
    log "$name: $(h end "$name" check "$rc")"
    if [ "$rc" -eq 0 ]; then
      log "$name: engine_check.json: $(h enginecheck "$name")"
    fi
    gap
  done
  log "$name: 走行の一覧: $(/home/jetson/bin/jf-heavy --lock none --mem 1500M --need-mb 500 -- "$PY" -I "$HERE/gpu_queue2.py" drivable "$name")"
  return 0
}

# キューは先に全部読む(中で動かすコマンドが標準入力を読んでも行を食べないように)
mapfile -t LINES < <(grep -v '^#' "$QUEUE" | grep -v '^[[:space:]]*$')
for line in "${LINES[@]}"; do
  name=${line%%$'\t'*}; cmd=${line#*$'\t'}
  log "---- $name ----"
  train_one "$name" "$cmd"; trc=$?
  if [ "$trc" -ne 0 ]; then
    continue
  fi
  h runjson "$name" "$cmd"
  engine_one "$name" || continue
  check_one "$name" || continue
done

log "==== 終わりました ===="
h status
h summary > "$LOGS/gpu_queue2_summary.md"
cat "$LOGS/gpu_queue2_summary.md"
