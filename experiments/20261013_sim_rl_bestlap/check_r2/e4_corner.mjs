#!/usr/bin/env node
// 確認 r2 の 4: 学習の範囲の「角」(遅い舵 × 弱いブレーキ)を細かく調べる。
//   node check_r2/e4_corner.mjs <selected|r2_5M|r2_1M|teacher>
// e3 の rec60x(開始 ±3 cm・±3° × 学習の範囲のランダムな物理)で、r2 の 500 万が落ちた 4 本はすべて「舵の速さ < 1.0 かつ ブレーキ < 0.9」だった。
// その角を、標準の開始(単周・連続 20 秒)と記録の 60 条件の開始(連続 20 秒)で調べる。ほかの物理は標準、遅延 0。
import {PLAN60, loadMlp, runPolicy, runTeacher, writeJson} from './common_r2.mjs';

const which = process.argv[2] ?? 'r2_5M';   // selected | r2_5M | r2_1M | teacher(並べて走らせるため 1 本ずつ)
const ctrls = [[which, which === 'teacher' ? null : loadMlp(which)]];
const STEER = [.85, .9, .95, 1.0, 1.1], BRAKE = [.8, .9, 1.0, 1.2];
const out = {schema: 'rl-check-r2-corner/v1', created_at: new Date().toISOString(), steer: STEER, brake: BRAKE, results: {}};
for (const [key, pol] of ctrls) {
  const run = o => pol ? runPolicy(pol, o) : runTeacher(o);
  const cells = [];
  for (const sr of STEER) for (const br of BRAKE) {
    const phys = {steerRate: sr, brakeDeceleration: br};
    const s = run({mode: 'single', phys}).summary, c = run({mode: 'continuous', phys}).summary;
    let recOk = 0, recN = 0;
    for (const p of PLAN60) {   // 記録の 60 条件の開始(連続 20 秒)
      const r = run({plan: p, gate: p['start-gate'], offset: p['spawn-offset'], yawDeg: p['spawn-yaw'], mode: 'continuous', phys}).summary;
      recN++; if (r.valid) recOk++;
    }
    cells.push({steerRate: sr, brake: br, std_single: s.valid, std_single_lap: s.laps[0] ?? null, std_cont: c.valid, rec60_valid: recOk, rec60_n: recN});
  }
  out.results[key] = cells;
  console.log(key);
  for (const sr of STEER) console.log('  steer', sr, cells.filter(c => c.steerRate === sr).map(c => `b${c.brake}:${c.std_single ? 'S' : 's'}${c.std_cont ? 'C' : 'c'} ${c.rec60_valid}/60`).join('  '));
}
writeJson(`e4_corner_${which}.json`, out);
