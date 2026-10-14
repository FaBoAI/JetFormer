// 検証の組でゲートを外した回を選んだ方策で走らせ直し、外したフレームと直前の乱しの関係を見る(報告だけ。試験の組は使わない)
import {buildJobs} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/rc/eval_rc3.mjs';
import {runModeEpisode, readJson} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/rc/episode_rc3.mjs';
import {LOOP_LENGTH} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/rc/rt/features.mjs';
const [polPath, evalPath] = process.argv.slice(2);
const w = readJson(polPath), ev = readJson(evalPath);
const ids = new Set(ev.reported.gate_miss.ids);
const jobs = buildJobs('validation').filter(j => j.test === 'RC-V100-R3' && ids.has(j.id));
const out = [];
for (const j of jobs) {
  const r = await runModeEpisode({mode: 'recover', weights: w, start: j.spec.start, params: j.spec.params ?? {}, coreKind: 'fast', audit: false});
  for (const gf of r.gate_miss_frames ?? []) {
    const prev = (r.events ?? []).filter(e => e.ps <= gf).at(-1);
    out.push({id: j.id, gf, s: +((r.log[gf]?.s ?? 0) % LOOP_LENGTH).toFixed(2), ev: prev ? {type: prev.type, kind: prev.kind ?? null, cls: prev.cls, sign: prev.sign, fallback: prev.fallback,
      s: +((prev.s ?? 0) % LOOP_LENGTH).toFixed(2), dt_after_pe_s: +((gf - prev.pe) / 30).toFixed(2)} : null});
  }
}
console.log(JSON.stringify(out));
