#!/bin/bash
# キュー v3(段 data、2026-10-26): gpu_queue_v3.tsv の学習(B-FAST2 6 本 → 段 0 の短い学習 4 本 → 段 1 の 12 本)を 1 本ずつ回す。
# experiments/20261025_train_v2/gpu_queue_v2.sh の写しを protocol_v3 に合わせたもの。電源断のあとは同じコマンドをもう一度
# 動かせば続きから(済んだ学習は飛ばす・途中は --resume)。
#
#   起動(main session が Bash の run_in_background で。この係は起動しない):
#     bash /home/jetson/jetformer/experiments/20261026_fix/gpu_queue_v3.sh
#   見守り: logs/gpu_queue_v3.log・python3 -I gpu_queue_v3.py status
#   試し(段 1 の最初の行を --epochs 1・--max-steps 5、名前 _smoke_*、出力 data/runs/_smoke_*、終わったら消す):
#     GPU_QUEUE_SMOKE=1 bash gpu_queue_v3.sh
#
# 流れ:
#   始める前: gpu_queue_v3.py preflight(protocol_v3・v2・lock2 の固定・封・v3 の表・データの確かめ・物差しの sha256・表の行・
#     train.py の引数・ディスク)。NG なら何もせずに 1 で終わる。
#   B-FAST2 の 6 行: BF2-0 が「学習しない」(今の答え。上限 A −0.05・B +2.42 µs < 28.6)なら飛ばす。
#   段 0 の 4 行: 短い学習(diag_data/ablation_short.sh と同じコマンド・同じ出力先)→ 物差し(best・last)。エンジンは作らない。
#   段 1 の 12 行: protocol_v3_lock2.json と decisions_v3.json の D0.order が無ければ、そこで止まる(段 0 の答えを書いてから
#     同じコマンドをもう一度)。順は D0.order の腕の順で 種 0 → 1 → 2 → 3。種 2・3 は途中でやめる決まり(futility)を見る。
# 1 本ごとに:
#   ① 学習: 済み(result.json が done・予定のエポック・best.pth・last.pth・ONNX・model_config)なら飛ばす。last.pth が読めれば
#      --resume(deviations_v3.md に書き足す)。読めない last.pth は脇へ動かして最初から。見込み × 1.2 が締め切り
#      (GPU_QUEUE_DEADLINE、既定 2026-10-30 06:00)を超える・空きディスク 5 GB 未満なら始めない。失敗は 1 回だけやり直す。
#      メモリの上限(137)のときだけ、やり直しの上限を 4608M に(protocol_v3 common_recipe.memory_retry)
#   ② エンジン(段 1・B-FAST2): tools/build_engine.py → ③ 確認: engine_load_check.py → engine_check.json → 走行の一覧
#   ④ 物差し: diag_data/probe_gates.py run --ckpt best・last → logs/probe/<名前>_best.json・_last.json(学習中の門・途中でやめる決まり)
#   ⑤ 机上の予測(段 1・B-FAST2): desk_v3/eval_v3.py があって lock2_v3 のあとだけ(まだ無い。無ければ書いて進む)
#   GPU の手順のあとは毎回 GPU_QUEUE_GAP_S 秒(既定 60)あける。走行中・収集中・管理画面の学習中は待つ。
#   車は動かさない・装置に触れない・jetformer-web は再起動しない・既定値(drive.py・推奨モデル・drive_settings.json)は変えない。
#   動いている間に gpu_queue_v3.sh・gpu_queue_v3.py を直しても、動いている分には効かない(手伝いは起動のときに
#   logs/.gpu_queue_v3.run.py へ写して使う)。表(gpu_queue_v3.tsv)と decisions_v3.json は行ごとに読み直す。
set -u
main() {
REPO=/home/jetson/jetformer
HERE=$REPO/experiments/20261026_fix
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
MEM_RETRY=4608M
GPU="/home/jetson/bin/jf-heavy --lock big --mem $MEM --need-mb 3200 --"
LOAD_CHECK=$REPO/experiments/20261013_sim_cotrain/train/engine_load_check.py
PROBE_GATES=$HERE/diag_data/probe_gates.py
EVAL_V3=$HERE/desk_v3/eval_v3.py
LOCK2=/home/jetson/jetformer_reports/fix_20261026/protocol_v3_lock2.json
PREDICT=${GPU_QUEUE_PREDICT:-1}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
mkdir -p "$LOGS/probe"
cd "$REPO" || exit 1
exec >> >(tee -a "$LOGS/gpu_queue_v3.log") 2>&1

SNAP=$LOGS/.gpu_queue_v3.run.py
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

exec 8>"$LOGS/gpu_queue_v3.lock" < /dev/null
if ! flock -n 8; then
  log "gpu_queue_v3.sh はもう動いています。終わります"
  exit 0
fi
cp -f "$HERE/gpu_queue_v3.py" "$SNAP.tmp" && mv -f "$SNAP.tmp" "$SNAP" || { log "手伝いを写せません"; exit 1; }
log "手伝い: $SNAP(sha256 $(sha256sum "$SNAP" | cut -c1-16)…)"
log "==== 始めます($([ "$SMOKE" = 1 ] && echo '試し・')あける時間 ${GAP_S} 秒・締め切り ${GPU_QUEUE_DEADLINE:-2026-10-30 06:00})===="
if ! h preflight; then
  log "始める前の確かめに NG があります。何もせずに終わります"
  exit 1
fi
h status

train_one() {  # 名前 コマンド → 0 = 学習が済んだ・1 = 2 回失敗・2 = 始めない(締め切り・ディスク)
  local name=$1 cmd=$2 run c rc f lp why
  run=$(h rundir "$name")
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
        "前回の学習がメモリの上限 $MEM で止められた(rc 137)。protocol_v3 common_recipe.memory_retry のとおり 1 回だけ上げてやり直す。判定への影響: なし(学習の中身は同じ)"
    fi
    lp=$(h lastpth "$name")
    case "$lp" in
      resume)
        c="$c --resume"
        h deviation "$name を --resume で再開した" \
          "last.pth から続きを学習する(途切れたのは $(grep -o '^epoch *[0-9]*' "$run/train.log" 2>/dev/null | tail -1 | tr -s ' ' | cut -d' ' -f2) エポック目のあと)。シャッフル・拡張の乱数の状態は保存していないので、続けて学習したときと完全には同じにならない。判定への影響: そのまま使う(protocol_v3 resume)"
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

engine_one() {
  local name=$1 run rc f
  run=$(h rundir "$name")
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

check_one() {
  local name=$1 run rc f d
  run=$(h rundir "$name")
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
  if d=$(/home/jetson/bin/jf-heavy --lock none --mem 1500M --need-mb 500 -- "$PY" -I "$SNAP" drivable "$name"); then
    log "$name: 走行の一覧: $d"
  else
    log "$name: 走行の一覧で drivable になっていません: $d"
    return 1
  fi
  return 0
}

probe_one() {  # 名前 → logs/probe/<名前>_best.json・_last.json(probe_gates.py run。学習中の門・途中でやめる決まりに使う)
  local name=$1 run ck rc f out
  run=$(h rundir "$name")
  for ck in best last; do
    while h need-probe "$name" "$ck"; do
      f=$(h fails "$name" "probe_$ck")
      if [ "$f" -ge 2 ]; then
        log "$name: 物差し($ck)が 2 回失敗しました(logs/probe_$name.log)"
        break
      fi
      wait_web
      out=$(h probe-out "$name" "$ck")
      log "$name: 物差し($ck.pth。probe_gates.py run)"
      h begin "$name" "probe_$ck"
      $GPU nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 \
        "$PY" -u "$PROBE_GATES" run --ckpt "$ck" --runs "$run" --out "$out" >> "$LOGS/probe_$name.log" 2>&1
      rc=$?
      log "$name: $(h end "$name" "probe_$ck" "$rc")  $(tail -1 "$LOGS/probe_$name.log" | cut -c1-220)"
      gap
    done
  done
  return 0
}

predict_one() {
  local name=$1 rc
  [ "$PREDICT" = 1 ] || return 0
  if [ "$SMOKE" = 1 ] || [ ! -f "$LOCK2" ] || [ ! -f "$EVAL_V3" ]; then
    log "$name: 机上の予測はしません(desk_v3/eval_v3.py がまだ無いか lock2_v3 の前。あとで eval_v3.py predict --runs $name)"
    return 0
  fi
  if h predicted "$name"; then
    log "$name: 机上の予測はあります"
    return 0
  fi
  wait_web
  log "$name: 机上の予測(eval_v3.py predict)"
  $GPU nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 \
    "$PY" -u "$EVAL_V3" predict --runs "$name" >> "$LOGS/predict_$name.log" 2>&1
  rc=$?
  log "$name: 予測 rc=$rc(logs/predict_$name.log)"
  gap
  return 0
}

stopped=""
mapfile -t TEMPLATES < <(h templates)
for tmpl in "${TEMPLATES[@]}"; do
  res=$(h resolve "$tmpl")
  kind=${res%%$'\t'*}
  rest=${res#*$'\t'}
  case "$kind" in
    SKIP)
      log "---- $tmpl: 飛ばします($rest)"
      case "$rest" in
        途中でやめる決まり*) h deviation "$tmpl を始めなかった(途中でやめる決まり)" \
          "$rest。protocol_v3 futility のとおり、その腕は n = 2 で判定する(閾値は n2)。学習の途中では止めていない" ;;
      esac
      continue ;;
    STOP) log "---- $tmpl: ここで止めます。$rest"; stopped=$tmpl; break ;;
    RUN) name=${rest%%$'\t'*}; cmd=${rest#*$'\t'} ;;
    *) log "---- $tmpl: 行を読めません: $res"; stopped=$tmpl; break ;;
  esac
  log "---- $name ----"
  train_one "$name" "$cmd"; trc=$?
  [ "$trc" -eq 0 ] || continue
  post=$(h post "$name")
  case "$post" in
    *engine*)
      h runjson "$name" "$cmd"
      engine_one "$name" && check_one "$name"
      probe_one "$name"
      predict_one "$name"
      ;;
    *) probe_one "$name" ;;
  esac
done

if [ -n "$stopped" ]; then
  log "==== $stopped の前で止まりました(答えか lock2_v3 を書いたあとに、同じコマンドをもう一度)===="
else
  log "==== 終わりました ===="
fi
h status
h summary > "$LOGS/gpu_queue_v3_summary.md"
cat "$LOGS/gpu_queue_v3_summary.md"
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
