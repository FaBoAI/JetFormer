#!/bin/bash
# no_wall r2 の学習のあとの流れ(2026-10-14、学習の段 no_wall r2)。各段は結果のファイルがあれば飛ばす(やり直しても同じところから続く)。
#   rl/nw2/after_r2.sh select      検証の組の評価(rl/nw/eval_checkpoints.sh が書いたもの)から protocol selection で選ぶ → train/no_wall/r2/selection.json
#   rl/nw2/after_r2.sh probes      選んだ方策で試験でない組: 確認 r1 の道具そのもの(rl/check_nw: probe_g16.mjs・verify_nw.mjs --sets ABL,CORNER。
#                                  bundle/no_wall に「未試験」で置いてから)と、rl/nw2 の組(VAL・G16・ABL・CORNER・FNOM・FDR とシールドなし)
#   rl/nw2/after_r2.sh test        試験の組(元の sim-core.js、protocol selection.test_runs の 2 回目 = 最後)→ eval/no_wall/no_wall_test_r2.json
#   rl/nw2/after_r2.sh publish     bundle/no_wall に eval つきで書き出し + F-ID の列(標準の STD と、遅い舵 + 遅延の列)
# Node は jf-heavy --mem 2G --lock none -- nice -n 15。
set -u
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
REP=/home/jetson/jetformer_reports/sim_rl_modes_20261013
R2=$REP/train/no_wall/r2
NW2=$EXP/rl/nw2
H="/home/jetson/bin/jf-heavy --mem 2G --lock none --need-mb 900 -- nice -n 15"
source /home/jetson/miniconda/etc/profile.d/conda.sh
conda activate jetracer
cd "$EXP"
log() { echo "[$(date -Is)] $*" | tee -a "$REP/train/no_wall/logs/after_r2.log"; }
POL() { python -c "import json;print(json.load(open('$R2/selection.json'))['policy'])"; }
OVR=$REP/train/no_wall/r2/mode_overrides.json
case "${1:-}" in
  select)
    python rl/nw2/select_r2.py "$R2" "$R2/selection.json" | tee -a "$REP/train/no_wall/logs/after_r2.log"
    ;;
  probes)
    P=$(POL)
    [ -d "$R2/diag/bundle_r1" ] || cp -r "$EXP/bundle/no_wall" "$R2/diag/bundle_r1"
    # 確認 r1 の道具は bundle/no_wall/policy.json と web/rl を読むので、選んだ方策を「未試験」で置く(試験のあと publish で eval つきに置き換える)
    python rl/publish_mode.py --mode no_wall --policy "$P" --source "no_wall r2(未試験): $(basename "$P")" --overrides "$OVR" | tee -a "$REP/train/no_wall/logs/after_r2.log"
    O=$R2/probes_checker_tools; mkdir -p "$O"
    O2=$R2/probes_r2policy; mkdir -p "$O2"
    # 2 本を並べて走らせる(どちらも --jobs 3)。1 本目 = 確認 r1 の道具そのもの、2 本目 = rl/nw2 の組(シールドなしの組を含む)
    (
      [ -s "$O/probe_g16.json" ] || (cd rl/check_nw && $H node probe_g16.mjs "$O/probe_g16.json" > "$O/probe_g16.log" 2>&1)
      log "確認 r1 の probe_g16.mjs: $(python -c "import json;d=json.load(open('$O/probe_g16.json'));print('衝突',sum(1 for x in d if x['col']),'/',len(d),'最小',min(x['min'] for x in d))")"
      for S in ABL,CORNER CORNEROFF; do
        [ -s "$O/rows_${S//,/_}.jsonl" ] || (cd rl/check_nw && $H node verify_nw.mjs --sets $S --jobs 3 --out "$O" >> "$O/verify.log" 2>&1)
        log "確認 r1 の verify_nw.mjs --sets $S: 終わり"
      done
    ) &
    (
      for S in VAL,G16 FNOM G16OFF,ABLOFF FDR FDROFF LOWMU; do
        f="$O2/rows_${S//,/_}.jsonl"; [ -s "$f" ] && continue
        NW2_RT=$EXP/rl/runtime NW2_POLICY=$P $H node rl/nw2/verify_nw2.mjs --sets "$S" --jobs 3 --out "$O2" >> "$O2/run.log" 2>&1
        log "nw2 の組 $S: 終わり"
      done
    ) &
    wait
    log "試験でない組: 全部終わり"
    ;;
  test)
    P=$(POL); OUT=$REP/eval/no_wall/no_wall_test_r2.json
    if [ -s "$OUT" ]; then log "試験の結果がもうあります: $OUT(2 回目の試験はしない)"; exit 0; fi
    log "試験の組(元のコア、作り直しの 1 回だけ)を始めます: $P"
    $H node rl/eval_modes.mjs --mode no_wall --policy "$P" --set test --core original --jobs 3 --out "$OUT" > "${OUT%.json}.log" 2>&1
    log "試験の組: $(tail -1 "${OUT%.json}.log" | cut -c1-600)"
    ;;
  publish)
    P=$(POL); OUT=$REP/eval/no_wall/no_wall_test_r2.json
    S=$(python -c "import json;d=json.load(open('$R2/selection.json'));print(d['selected'])")
    python rl/publish_mode.py --mode no_wall --policy "$P" --eval "$OUT" --overrides "$OVR" \
      --source "no_wall r2(作り直し): train/no_wall/r2 $S(r1 の 1,000 万から種 22・600 万ステップ、遅い舵 + 遅延の回を増やし速さの見張りつき。deviations.md 5)。走行中のシールド jf-shield/v3(deviations.md 4)" \
      | tee -a "$REP/train/no_wall/logs/after_r2.log"
    # F-ID(標準の物理・STD の連続 600 フレーム。eval_modes.mjs の writeFid だけを呼ぶ = 試験の組は走らせない)
    $H node --input-type=module -e "const m = await import('$EXP/rl/eval_modes.mjs'); const fs = await import('node:fs'); const c = await import('node:crypto');
      const p = '$EXP/bundle/no_wall/policy.json'; const w = JSON.parse(fs.readFileSync(p, 'utf8'));
      await m.writeFid('no_wall', w, c.createHash('sha256').update(fs.readFileSync(p)).digest('hex'), '$REP/eval/no_wall/fid_std_no_wall_r2.json');"
    $H node rl/nw2/write_fid_slow.mjs --policy "$EXP/bundle/no_wall/policy.json" --out "$REP/eval/no_wall/fid_slow_no_wall_r2.json" | tee -a "$REP/train/no_wall/logs/after_r2.log"
    for F in fid_std_no_wall_r2 fid_slow_no_wall_r2; do
      node rl/check_fid.mjs --fid "$REP/eval/no_wall/$F.json" --policy "$EXP/bundle/no_wall/policy.json" --runtime "$EXP/web/rl/mode_runtime.mjs" | tee -a "$REP/train/no_wall/logs/after_r2.log"
    done
    ;;
  *) echo "使い方: $0 select|probes|test|publish"; exit 2;;
esac
