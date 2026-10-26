#!/bin/bash
# キュー v2(C4、2026-10-25): gpu_queue_v2.tsv の学習 22 本(B-FAST 3 本 → v2 19 本)とエンジンを 1 本ずつ回す。
# experiments/20261021_bench_prep/gpu_queue.sh(10/21 段 G2)・20261022_prep/gpu_queue2.sh の写し。電源断のあとは同じコマンドを
# もう一度動かせば続きから(済んだ学習は飛ばす・途中は --resume)。
#
#   起動(main session が Bash の run_in_background で):
#     GPU_QUEUE_ACCEPT_DATA_FAILS=train_samples_v2m_within_1pct,bfast_teacher_within_1pct \
#         bash /home/jetson/jetformer/experiments/20261025_train_v2/gpu_queue_v2.sh
#   見守り: logs/gpu_queue_v2.log・python -I gpu_queue_v2.py status
#   試し(先頭の 1 本を --epochs 1・--max-steps 5、名前 _smoke_*、終わったら消す): GPU_QUEUE_SMOKE=1 bash gpu_queue_v2.sh
#
# 始める前: gpu_queue_v2.py preflight(protocol の固定・封・表・B の重み・ジャンル 9・表の行・データの確かめの了承・ディスク)。
#   NG なら何もせずに 1 で終わる。
# 1 本ごとに:
#   ⓪ 行の型({x}・{E}・{A})をその時の decisions.json で埋める。段 3・4 の答えが無い・train.py にまだ無い引数(C1)なら、そこで止まる
#   ① 学習: 済み(result.json が done・予定のエポック・best.pth・last.pth・ONNX・model_config)なら飛ばす。
#      last.pth が読めれば --resume(deviations.md に書き足す)。読めない last.pth は脇へ動かして最初から。
#      見込み × 1.2 が締め切り(GPU_QUEUE_DEADLINE、既定 2026-10-31 06:00)を超える・空きディスク 5 GB 未満なら始めない。
#      失敗(終わったのに済んでいない)は 1 回だけやり直す。メモリの上限で止められた(137)ときだけ、やり直しの上限を 4608M に
#      (protocol common_recipe.memory_retry。deviations.md に書き足す)。電源断で途切れた試み(終わりの記録が無い)は失敗に数えない
#   ② エンジン: tools/build_engine.py(jetformer_fp16.engine。tmp → check_io → 置き換え)。1 回だけやり直す
#   ③ 確認: experiments/20261013_sim_cotrain/train/engine_load_check.py(実フレーム 64 個で TensorRT と PyTorch FP32 の差
#      ≤ 2.5 µs・速さ)→ engine_check.json → drive.model_info の drivable
#   ④ 予測(protocol_v2_lock2.json があるときだけ。GPU_QUEUE_PREDICT=0 で止める): eval_v2.py predict(H1・介入の best と last)
#   GPU の手順のあとは毎回 GPU_QUEUE_GAP_S 秒(既定 60)あけて、ほかの仕事(別の workflow の試験など)に GPU の順番を回す。
#   走行中・収集中(GET /api/drive/status)・管理画面の学習中(GET /api/status)は待つ。
#   車は動かさない・装置に触れない・jetformer-web は再起動しない・既定値(drive.py・推奨モデル・drive_settings.json)は変えない。
#   動いている間に gpu_queue_v2.sh・gpu_queue_v2.py を直しても、動いている分には効かない(全体を main で先に読み、手伝いは
#   起動のときに logs/.gpu_queue_v2.run.py へ写して使う)。表(gpu_queue_v2.tsv)と decisions.json は行ごとに読み直す
#   (直すときは別の名前に書いてから mv で置き換える)。
set -u
main() {
REPO=/home/jetson/jetformer
HERE=$REPO/experiments/20261025_train_v2
SMOKE=${GPU_QUEUE_SMOKE:-0}
export GPU_QUEUE_SMOKE=$SMOKE
if [ "$SMOKE" = 1 ]; then
  LOGS=$HERE/logs/smoke
  GAP_S=${GPU_QUEUE_GAP_S:-5}
else
  LOGS=$HERE/logs
  GAP_S=${GPU_QUEUE_GAP_S:-60}
fi
PY=/home/jetson/miniconda/envs/jetracer/bin/python
MEM=3872M
MEM_RETRY=4608M                 # protocol common_recipe.memory_retry: 1 回だけ・最大 4.5G
GPU="/home/jetson/bin/jf-heavy --lock big --mem $MEM --need-mb 3200 --"
LOAD_CHECK=$REPO/experiments/20261013_sim_cotrain/train/engine_load_check.py
LOCK2=/home/jetson/jetformer_reports/train_v2_20261025/protocol_v2_lock2.json
PREDICT=${GPU_QUEUE_PREDICT:-1}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
mkdir -p "$LOGS"
cd "$REPO" || exit 1
exec >> >(tee -a "$LOGS/gpu_queue_v2.log") 2>&1

SNAP=$LOGS/.gpu_queue_v2.run.py
h() { "$PY" -I "$SNAP" "$@"; }
log() { echo "[$(date '+%F %T')] $*"; }
gap() { log "  ${GAP_S} 秒あけます(ほかの仕事に GPU の順番を回す)"; sleep "$GAP_S"; }
wait_web() {
  local n=0
  while h webbusy >/dev/null; do
    [ $((n % 10)) -eq 0 ] && log "  走行・収集・管理画面の学習のどれかが使用中のため待っています($(h webbusy))"
    n=$((n + 1)); sleep 30
  done
}

# 同時に 2 つ動かさない
exec 8>"$LOGS/gpu_queue_v2.lock" < /dev/null
if ! flock -n 8; then
  log "gpu_queue_v2.sh はもう動いています。終わります"
  exit 0
fi
cp -f "$HERE/gpu_queue_v2.py" "$SNAP.tmp" && mv -f "$SNAP.tmp" "$SNAP" || { log "手伝いを写せません"; exit 1; }
log "手伝い: $SNAP(sha256 $(sha256sum "$SNAP" | cut -c1-16)…)"

log "==== 始めます($([ "$SMOKE" = 1 ] && echo '試し・')あける時間 ${GAP_S} 秒・締め切り ${GPU_QUEUE_DEADLINE:-2026-10-31 06:00})===="
if ! h preflight; then
  log "始める前の確かめに NG があります。何もせずに終わります"
  exit 1
fi
h status

train_one() {  # 名前 コマンド → 0 = 学習が済んだ・1 = 2 回失敗・2 = 始めない(締め切り・ディスク)
  local name=$1 cmd=$2 run=$REPO/data/runs/$1 c rc f lp why
  while ! h trained "$name" "$cmd"; do
    f=$(h fails "$name" train)
    if [ "$f" -ge 2 ]; then
      log "$name: 学習が 2 回失敗しました。次へ進みます(logs/train_$name.log を見てください)"
      return 1
    fi
    if ! why=$(h fits "$name" "$cmd"); then
      log "$name: 学習を始めません($why)"
      return 2
    fi
    c=$cmd
    if [ "$f" -ge 1 ] && [ "$(h last-rc "$name" train)" = "137" ]; then
      c=${c/--mem $MEM/--mem $MEM_RETRY}
      log "$name: 前回はメモリの上限で止められた(137)ので、やり直しは上限 $MEM_RETRY"
      h deviation "$name の学習のメモリの上限を $MEM_RETRY に上げた" \
        "前回の学習がメモリの上限 $MEM で止められた(rc 137)。protocol common_recipe.memory_retry のとおり 1 回だけ上げてやり直す。判定への影響: なし(学習の中身は同じ)"
    fi
    lp=$(h lastpth "$name")
    case "$lp" in
      resume)
        c="$c --resume"
        h deviation "$name を --resume で再開した" \
          "last.pth から続きを学習する(途切れたのは $(grep -o '^epoch *[0-9]*' "$run/train.log" 2>/dev/null | tail -1 | tr -s ' ' | cut -d' ' -f2) エポック目のあと)。シャッフル・拡張の乱数の状態は保存していないので、続けて学習したときと完全には同じにならない(plan.md 5 章)。判定への影響: そのまま使う(protocol resume)"
        ;;
      broken*)
        log "$name: last.pth が読めないので最初から(${lp#broken$'\t'})"
        ;;
    esac
    wait_web
    mkdir -p "$run"
    log "$name: 学習を始めます(失敗の数 $f・$why)"
    h begin "$name" train "$lp"
    echo "==== $(date '+%F %T') 開始(cd ~/jetformer && $c)====" | tee -a "$run/train.log" >> "$LOGS/train_$name.log"
    bash -c "$c" < /dev/null 2>&1 | tee -a "$run/train.log" >> "$LOGS/train_$name.log"
    rc=${PIPESTATUS[0]}
    echo "==== $(date '+%F %T') 終了 rc=$rc ====" | tee -a "$run/train.log" >> "$LOGS/train_$name.log"
    log "$name: $(h end "$name" train "$rc" "$cmd")"
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
    if [ -s "$LOGS/engine_check_$name.json" ]; then
      log "$name: engine_check.json: $(h enginecheck "$name")"
    fi
    gap
  done
  local d
  if d=$(/home/jetson/bin/jf-heavy --lock none --mem 1500M --need-mb 500 -- "$PY" -I "$SNAP" drivable "$name"); then
    log "$name: 走行の一覧: $d"
  else
    log "$name: 走行の一覧で drivable になっていません: $d"
    return 1
  fi
  return 0
}

predict_one() {  # 名前 → 机上の予測(lock2 のあとだけ)
  local name=$1 rc
  [ "$PREDICT" = 1 ] || return 0
  if [ "$SMOKE" = 1 ] || [ ! -f "$LOCK2" ] || [ ! -f "$HERE/eval_v2.py" ]; then
    log "$name: 予測はしません(protocol_v2_lock2.json のあと。eval_v2.py predict --runs $name をあとで)"
    return 0
  fi
  if h predicted "$name"; then
    log "$name: 予測はあります(eval/pred)"
    return 0
  fi
  wait_web
  log "$name: H1・介入の予測(best・last)"
  $GPU nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 \
    "$PY" -u "$HERE/eval_v2.py" predict --runs "$name" >> "$LOGS/predict_$name.log" 2>&1
  rc=$?
  log "$name: 予測 rc=$rc(logs/predict_$name.log)"
  gap
  return 0
}

stopped=""
# 行の名前は先に全部読む(中で動かすコマンドが標準入力を読んでも行を食べないように)。型を埋めるのはその行に来たとき
mapfile -t TEMPLATES < <(h templates)
for tmpl in "${TEMPLATES[@]}"; do
  res=$(h resolve "$tmpl")
  kind=${res%%$'\t'*}
  rest=${res#*$'\t'}
  case "$kind" in
    SKIP) [ "$SMOKE" = 1 ] || log "---- $tmpl: 飛ばします($rest)"; continue ;;
    STOP) log "---- $tmpl: ここで止めます。$rest"; stopped=$tmpl; break ;;
    RUN) name=${rest%%$'\t'*}; cmd=${rest#*$'\t'} ;;
    *) log "---- $tmpl: 行を読めません: $res"; stopped=$tmpl; break ;;
  esac
  log "---- $name ----"
  train_one "$name" "$cmd"; trc=$?
  [ "$trc" -eq 0 ] || continue
  h runjson "$name" "$cmd"
  engine_one "$name" || continue
  check_one "$name" || continue
  predict_one "$name"
done

if [ -n "$stopped" ]; then
  log "==== $stopped の前で止まりました(答えか引数を足したあとに、同じコマンドをもう一度)===="
else
  log "==== 終わりました ===="
fi
h status
h summary > "$LOGS/gpu_queue_v2_summary.md"
cat "$LOGS/gpu_queue_v2_summary.md"
}

cleanup_smoke() {   # 試しの出力(data/runs/_smoke_*)を消す。途中で止まっても消すように EXIT で呼ぶ
  [ "${GPU_QUEUE_SMOKE:-0}" = 1 ] && [ "${GPU_QUEUE_SMOKE_KEEP:-0}" != 1 ] || return 0
  local d
  for d in /home/jetson/jetformer/data/runs/_smoke_*; do
    [ -d "$d" ] || continue
    echo "[$(date '+%F %T')] 試しの出力を消します: $d"
    rm -rf -- "$d"
  done
}
trap cleanup_smoke EXIT

main "$@"; exit $?
