#!/bin/bash
# C1 の試し(2026-10-25): キュー v2 の段 3 の行(v2_20261025-{x}-aug2-e{E}-s0)を、試しの答え(x = v2h・E = 60・AUG2)と
# GPU_QUEUE_WORKERS=2 で gpu_queue_v2.py resolve に埋めてもらい、そのコマンドの --out を data/runs/_smoke_c1、
# --epochs 1、--max-steps $STEPS に変えて 1 回だけ学習する(GPU。コマンドの中の jf-heavy --lock big --mem 3872M のまま)。
# 本番の decisions.json・状態・deviations.md には触れない(試しの答えは一時ファイル)。終わったら(途中で止まっても)出力を消す。
#
#   bash /home/jetson/jetformer/experiments/20261025_train_v2/smoke_c1.sh            # 既定 40 ステップ
#   SMOKE_C1_STEPS=5 bash .../smoke_c1.sh
#   SMOKE_C1_NOAUG=1 bash .../smoke_c1.sh      # 比べ: 同じ行から --aug-* を全部外す(拡張なしの 1 エポックの時間・メモリ)
# ログ: experiments/20261025_train_v2/logs/smoke/train__smoke_c1.log・smoke_c1.json
set -u
REPO=/home/jetson/jetformer
HERE=$REPO/experiments/20261025_train_v2
LOGS=$HERE/logs/smoke
PY=/home/jetson/miniconda/envs/jetracer/bin/python
STEPS=${SMOKE_C1_STEPS:-40}
NOAUG=${SMOKE_C1_NOAUG:-0}
NAME=_smoke_c1
RUN=$REPO/data/runs/$NAME
ROW='v2_20261025-{x}-aug2-e{E}-s0'
TAG=$([ "$NOAUG" = 1 ] && echo _noaug)
LOG=$LOGS/train_$NAME$TAG.log
OUTJ=$LOGS/smoke_c1$TAG.json
mkdir -p "$LOGS"
cd "$REPO" || exit 1
TMPD=$(mktemp -d "$LOGS/.smoke_c1.XXXXXX")
cleanup() {
  [ -d "$RUN" ] && { echo "[$(date '+%F %T')] 試しの出力を消します: $RUN"; rm -rf -- "$RUN"; }
  rm -rf -- "$TMPD"
}
trap cleanup EXIT
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

[ -e "$RUN" ] && { echo "$RUN がもうあります。消してからもう一度"; exit 1; }
printf '{"D1":{"x":"v2h"},"D2":{"E":60},"D3":{"aug":"AUG2"}}' > "$TMPD/decisions.json"
res=$(GPU_QUEUE_DECISIONS=$TMPD/decisions.json GPU_QUEUE_WORKERS=2 "$PY" -I "$HERE/gpu_queue_v2.py" resolve "$ROW")
kind=${res%%$'\t'*}
rest=${res#*$'\t'}
[ "$kind" = RUN ] || { log "resolve が RUN ではありません: $res"; exit 1; }
qname=${rest%%$'\t'*}
qcmd=${rest#*$'\t'}
cmd=${qcmd/--out data\/runs\/$qname/--out data\/runs\/$NAME}
cmd=$(printf '%s' "$cmd" | sed -E 's/--epochs [0-9]+/--epochs 1/')" --max-steps $STEPS"
[ "$NOAUG" = 1 ] && cmd=$(printf '%s' "$cmd" | sed -E 's/ --aug-[a-z-]+ [0-9.]+//g')
[ "$cmd" != "$qcmd" ] && [[ "$cmd" == *"--out data/runs/$NAME "* || "$cmd" == *"--out data/runs/$NAME" ]] || { log "--out を変えられません"; exit 1; }
log "キューの行: $qname"
log "キューのコマンド: $qcmd"
log "試しのコマンド: $cmd"
mkdir -p "$RUN"
t0=$(date +%s)
bash -c "$cmd" < /dev/null >> "$LOG" 2>&1 &
bpid=$!
peak=0; peak_anon=0; peak_file=0; cg=""
while kill -0 "$bpid" 2>/dev/null; do
  if [ -z "$cg" ]; then     # jf-heavy の systemd-run --scope(run-*.scope)。jf-heavy の bash 自身はセッションの scope なので使わない
    for tp in $(pgrep -f -- "tools/train.py .*--out data/runs/$NAME"); do
      c=$(cut -d: -f3- "/proc/$tp/cgroup" 2>/dev/null | head -1)
      case "$c" in */run-*.scope) cg=/sys/fs/cgroup$c; break ;; esac
    done
  fi
  if [ -n "$cg" ] && [ -r "$cg/memory.peak" ]; then
    p=$(cat "$cg/memory.peak" 2>/dev/null || echo 0)
    [ "$p" -gt "$peak" ] && peak=$p
    a=$(awk '$1=="anon"{print $2}' "$cg/memory.stat" 2>/dev/null); f=$(awk '$1=="file"{print $2}' "$cg/memory.stat" 2>/dev/null)
    [ -n "$a" ] && [ "$a" -gt "$peak_anon" ] && peak_anon=$a
    [ -n "$f" ] && [ "$f" -gt "$peak_file" ] && peak_file=$f
  fi
  sleep 1
done
wait "$bpid"; rc=$?
wall=$(( $(date +%s) - t0 ))
log "学習 rc=$rc・$wall 秒・メモリの最大 $((peak / 1048576)) MB(anon の最大 $((peak_anon / 1048576)) MB・ページキャッシュの最大 $((peak_file / 1048576)) MB。cgroup $cg)"
GPU_QUEUE_SMOKE=1 "$PY" -I "$HERE/gpu_queue_v2.py" runjson "$NAME" "$cmd" >> "$LOG" 2>&1
"$PY" -I - "$RUN" "$OUTJ" "$rc" "$wall" "$peak" "$peak_anon" "$peak_file" "$qname" "$qcmd" "$cmd" <<'EOF'
import json, os, sys, zipfile
run, outj, rc, wall, peak, anon, filec, qname, qcmd, cmd = sys.argv[1:]
def load(p):
    try:
        return json.load(open(p, encoding='utf-8'))
    except Exception:
        return None
res = load(os.path.join(run, 'result.json')) or {}
mc = load(os.path.join(run, 'model_config.json')) or {}
rj = load(os.path.join(run, 'run.json')) or {}
args = {}
try:      # last.pth の args(torch を読まずに: pickle の中の文字列だけ見る)
    with zipfile.ZipFile(os.path.join(run, 'last.pth')) as z:
        pk = [n for n in z.namelist() if n.endswith('data.pkl')][0]
        raw = z.read(pk)
    args['aug_top_brightness_in_last_pth'] = b'aug_top_brightness' in raw
except Exception as e:
    args['last_pth_error'] = repr(e)
out = {'queue_row': qname, 'queue_cmd': qcmd, 'smoke_cmd': cmd, 'rc': int(rc), 'wall_s': int(wall),
       'mem_peak_mb': round(int(peak) / 2 ** 20), 'anon_peak_mb': round(int(anon) / 2 ** 20),
       'file_peak_mb': round(int(filec) / 2 ** 20), 'mem_limit': '3872M',
       'state': res.get('state'), 'epochs_done': res.get('epochs_done'), 'train_samples': res.get('train_samples'),
       'val_samples': res.get('val_samples'), 'best_val_us': res.get('best_val_us'), 'gpu_peak_mem_mb': res.get('peak_mem_mb'),
       'bench_options_aug': (res.get('bench_options') or {}).get('aug'), 'duration_s': res.get('duration_s'),
       'files': {f: os.path.exists(os.path.join(run, f)) for f in ('best.pth', 'last.pth', 'jetformer.onnx', 'model_config.json',
                                                                  'result.json', 'run.json')},
       'model_config_engine': mc.get('engine'), 'model_config_policy': mc.get('policy'),
       'run_json_aug': rj.get('aug'), 'steer_center_us': res.get('steer_center_us'),
       'throttle_neutral_us': res.get('throttle_neutral_us'), **args}
out['ok'] = bool(out['rc'] == 0 and out['state'] == 'done' and all(out['files'].values())
                 and (out['bench_options_aug'] or {}).get('top_brightness') == 0.15
                 and (out['run_json_aug'] or {}).get('top_brightness') == 0.15
                 and args.get('aug_top_brightness_in_last_pth'))
json.dump(out, open(outj, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
EOF
ok=$?
tail -40 "$LOG" | grep -E "学習 |学習の拡張|epoch|保存|Error|error|Killed" | tail -12
exit $(( rc != 0 ? rc : ok ))
