#!/bin/bash
# S-MOD の評価と DAgger の記録を順につなぐ(sim_diverse_train_20261015、2026-10-15)。電源断のあとはそのまま同じコマンドで続きから(済んだ段は飛ばす)。
# 1) S-MOD の一式ができるまで待つ(train/runs.json の S-MOD/s0 の ZIP。学習の道具が作る)
# 2) S-MOD の E0〜E11 + フライング(eval/run_eval.sh。protocol eval_Q1.sessions_and_order 4)→ 集計(eval/post_eval.sh)
# 3) 画面を撮るための走行: S-DIV の E2(完走した開始)を窓 1400×900・追従の視点でもう一度(closed_loop_screens/。数えない)
# 4) 一番良い腕が S-DIV のままなら: DAgger の G2(記録 1 シーンの試し)→ 合格なら 12 シーンの記録 → post_dagger.sh(取り込み・付け替え・A4 のデータ)
#    S-MOD のほうが良ければ、ここで止める(S-MOD で状態を集め直す)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
X=$(cd "$HERE/.." && pwd)
R=/home/jetson/jetformer_reports/sim_diverse_train_20261015
D=$R/dagger
log() { echo "[$(date +%H:%M:%S)] $*"; }
source /home/jetson/miniconda/etc/profile.d/conda.sh && conda activate jetracer
# 1)
for i in $(seq 1 720); do
  python -I -c "
import json,sys,os
r=json.load(open('$R/train/runs.json'))['runs'].get('S-MOD/s0') or {}
sys.exit(0 if r.get('bundle_zip') and os.path.exists(r['bundle_zip']) and r.get('http_load_ok') else 1)" 2>/dev/null && break
  sleep 10
done
log "S-MOD の一式あり"
# 2)
if [ ! -f "$R/eval/closed_loop/SMOD-s0/SMOD-s0-E0-flying.json" ]; then
  bash "$X/eval/run_eval.sh" --jobs "SMOD-s0:core,SMOD-s0:ext,SMOD-s0:flying"
  log "S-MOD の評価の終わり"
fi
bash "$X/eval/post_eval.sh" > "$R/eval/logs/post_eval_smod.log" 2>&1
log "集計の終わり: $(tail -1 $R/eval/logs/post_eval_smod.log)"
# 3)
if [ ! -f "$R/eval/closed_loop_screens/SDIV-s0/SDIV-s0-E2-single.json" ]; then
  bash "$X/eval/run_eval.sh" --jobs "SDIV-s0:core" --only SDIV-s0-E2-single --out "$R/eval/closed_loop_screens" \
    --screens "$R/screens/lap_SDIV_E2" --window 1400x900 --view follow --shots 20,70,120,170,220
  log "画面の走行の終わり"
fi
# 4)
BEST=$(python -I -c "import json;print(json.load(open('$R/eval_sim.json'))['verdicts']['best_arm'].get('best_today_arm'))")
log "今日の一番良い腕: $BEST"
if [ "$BEST" != "S-DIV" ]; then log "S-DIV ではないので、DAgger の記録はここで止める(状態を集め直す)"; exit 3; fi
NAME=jetformer-simulator-rl-dagger-sdiv-4min
if [ ! -f "$D/rec_g2test/result.json" ]; then
  bash "$HERE/run_dagger_record.sh" "$D/record_plan.json" "$D/rec_g2test" "" --max-scenes 1 --chunk 1
fi
SID=$(python -I -c "
import json,glob
for f in glob.glob('$D/rec_g2test/sidecars/*.json'):
    if 'attempt' in f: continue
    print(json.load(open(f))['session_id'])" | head -1)
SC=$(ls -d $R/sim/runtime/datasets/$SID/*/simulation/scenes/*/ 2>/dev/null | head -1)
python3 -I "$HERE/verify_dagger_scenes.py" --scenes "$SC" --out "$D/g2.json"
python3 -I -c "import json,sys;sys.exit(0 if json.load(open('$D/g2.json'))['all_pass'] else 1)" || { log "G2 不合格 → DAgger は止める"; exit 4; }
log "G2 合格"
if [ ! -f "$D/rec/collection.json" ] && ! python3 -I -c "import json,sys;r=json.load(open('$D/rec/result.json'));sys.exit(0 if len(r.get('accepted',[]))==12 else 1)" 2>/dev/null; then
  bash "$HERE/run_dagger_record.sh" "$D/record_plan.json" "$D/rec" "" --chunk 6
fi
python3 -I -c "import json;r=json.load(open('$D/rec/result.json'));print('合格', len(r.get('accepted',[])), '不合格', r.get('rejected'), 'エラー', r.get('error'))"
log "記録の終わり"
bash "$HERE/post_dagger.sh" "$D/rec" "$NAME" S-DIV
log "post_dagger の終わり $?"
