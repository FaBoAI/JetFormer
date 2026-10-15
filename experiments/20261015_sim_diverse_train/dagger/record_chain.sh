#!/bin/bash
# DAgger の G2(記録 1 シーンの試し)→ 合格なら 12 シーンの記録 → post_dagger.sh(取り込み・付け替え・A4 のデータ)→ a4_chain.sh(A4 の学習・評価)
# (sim_diverse_train_20261015、2026-10-15。afternoon_chain.sh の 4) を、引数の誤り(まとめの名前が空)を直して別に分けたもの)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
R=/home/jetson/jetformer_reports/sim_diverse_train_20261015
D=$R/dagger
log() { echo "[$(date +%H:%M:%S)] $*"; }
source /home/jetson/miniconda/etc/profile.d/conda.sh && conda activate jetracer
NAME=jetformer-simulator-rl-dagger-sdiv-4min
if [ ! -f "$D/rec_g2test/result.json" ]; then
  bash "$HERE/run_dagger_record.sh" "$D/record_plan.json" "$D/rec_g2test" "" --max-scenes 1 --chunk 1
fi
SID=$(python -I -c "
import json,glob
for f in sorted(glob.glob('$D/rec_g2test/sidecars/*.json')):
    if 'attempt' in f: continue
    print(json.load(open(f))['session_id'])" | head -1)
SC=$(ls -d $R/sim/runtime/datasets/$SID/*/simulation/ 2>/dev/null | head -1)   # 生の記録はシーン 1 つ: simulation/frames.jsonl
log "G2 の試しのシーン: $SID $SC"
python3 -I "$HERE/verify_dagger_scenes.py" --scenes "$SC" --out "$D/g2.json"
python3 -I -c "import json,sys;sys.exit(0 if json.load(open('$D/g2.json'))['all_pass'] else 1)" || { log "G2 不合格 → DAgger は止める"; exit 4; }
log "G2 合格"
if ! python3 -I -c "import json,sys;r=json.load(open('$D/rec/result.json'));sys.exit(0 if len(r.get('accepted',[]))==12 else 1)" 2>/dev/null; then
  bash "$HERE/run_dagger_record.sh" "$D/record_plan.json" "$D/rec" "" --chunk 6
fi
python3 -I -c "import json,sys;r=json.load(open('$D/rec/result.json'));print('合格', len(r.get('accepted',[])), '不合格', r.get('rejected'), 'エラー', r.get('error'));sys.exit(0 if len(r.get('accepted',[]))==12 else 1)" \
  || { log "12 シーンがそろわない → 止める"; exit 5; }
log "記録の終わり"
bash "$HERE/post_dagger.sh" "$D/rec" "$NAME" S-DIV || { log "post_dagger が失敗"; exit 6; }
log "post_dagger の終わり"
bash "$HERE/a4_chain.sh"
log "a4_chain の終わり $?"
