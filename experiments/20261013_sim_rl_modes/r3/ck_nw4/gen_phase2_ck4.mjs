// no_wall 確認 r4 の第 2 段の組(第 1 段の結果を見て、危なそうな所のまわりを細かく)。種 7,410,000 + i、乱数 mulberry32(0xC4A10E21〜)。
//   LS : 標準の物理の第 1 段の回のうち、シールドの「どれも Q に合格しない」選び(e_brake・max_e)・nominal_pass・抜け出し・長く続けて替えた回・
//        しきい値ちょうど(8.10〜8.11 cm)の回から危なさの順に 25 個を選び、開始のまわり(左右 ±0.006 m・向き ±0.6° の 5 × 5、真ん中を除く 24 点)
//   VXS: 遅い舵(0.8)の頂点の全部の組(ほかの 6 つの値の端 64 × 遅延 0/1 = 128)× ゲート 20〜25 の開始のすき間 12〜15 cm の開始 2(検証の VV9 の形)
//   使い方: node gen_phase2_ck4.mjs <runs のフォルダー> <出力 jobs.jsonl>
import fs from 'node:fs';
import {mulberry32, startClear} from './own_ck4.mjs';
const [dir, outp] = process.argv.slice(2);
const rows = [];
for (const f of fs.readdirSync(dir).filter(f => /^rows_\d+\.jsonl$/.test(f))) for (const l of fs.readFileSync(`${dir}/${f}`, 'utf8').split('\n')) if (l.trim()) rows.push(JSON.parse(l));
const isDr = r => !!r.phys || r.fc || r.delay;
const std = rows.filter(r => !r.error && !r.invalid_spawn && !isDr(r) && r.shield && ['G17F', 'G2023', 'CORN', 'LOWC', 'U'].includes(r.set));
const score = r => {
  const fb = r.fb ?? {};
  const k = s => Object.entries(fb).filter(([n]) => n.includes(s)).reduce((a, [, v]) => a + v, 0);
  return 100 * (k('max_e') + k('e_brake')) + 10 * k('nominal_pass') + 20 * (r.escapes ?? 0) + (r.max_run_sh ?? 0) + (r.min_clearance < .0811 ? 5 : 0);
};
std.sort((a, b) => score(b) - score(a));
const top = std.slice(0, 25);
const jobs = [];
let seed = 7410000;
const add = (set, spec) => jobs.push({id: `${set}:${jobs.filter(j => j.set === set).length}`, set, seed: seed++, ...spec});
const r6 = x => Math.round(x * 1e6) / 1e6;
for (const r of top) {
  for (const dO of [-.006, -.003, 0, .003, .006]) for (const dY of [-.6, -.3, 0, .3, .6]) {
    if (dO === 0 && dY === 0) continue;
    const off = r6(r.start.off + dO), yaw = r6(r.start.yaw + dY);
    if (Math.abs(off) > .2 || Math.abs(yaw) > 15) continue;
    const c = startClear(r.start.gate, off, yaw);
    if (c >= .12) add('LS', {start: {gate: r.start.gate, off, yaw, c0: +c.toFixed(5)}, of: r.id, of_score: score(r)});
  }
}
// VXS
const R = mulberry32(0xC4A10E21);
const ends = {brakeDeceleration: [.7, 2.2], driveTau: [.28, .55], driveAccelerationCap: [3.5, 6.0], frictionMultiplier: [.85, 1.1], sg: [.94, 1.06], vg: [.96, 1.04]};
function lowcGate() {
  for (let t = 0; t < 100000; t++) {
    const gate = 20 + Math.floor(R() * 6), off = r6((R() * 2 - 1) * .2), yaw = r6((R() * 2 - 1) * 15), c = startClear(gate, off, yaw);
    if (c >= .12 && c < .15) return {gate, off, yaw, c0: +c.toFixed(5)};
  }
  throw new Error('開始が見つかりません');
}
for (let b = 0; b < 64; b++) for (const delay of [0, 1]) for (let k = 0; k < 2; k++) {
  const e = (key, i) => ends[key][(b >> i) & 1];
  add('VXS', {start: lowcGate(), phys: {steerRate: .8, brakeDeceleration: e('brakeDeceleration', 0), driveTau: e('driveTau', 1), driveAccelerationCap: e('driveAccelerationCap', 2),
    frictionMultiplier: e('frictionMultiplier', 3)}, sg: e('sg', 4), vg: e('vg', 5), delay, fc: R() < .3});
}
fs.writeFileSync(outp, jobs.map(j => JSON.stringify(j)).join('\n') + '\n');
console.log(JSON.stringify({std_rows: std.length, top: top.map(r => ({id: r.id, score: score(r), min_cm: +(100 * r.min_clearance).toFixed(3), fb: r.fb, esc: r.escapes})).slice(0, 25),
  jobs: jobs.length, LS: jobs.filter(j => j.set === 'LS').length, VXS: jobs.filter(j => j.set === 'VXS').length}));
