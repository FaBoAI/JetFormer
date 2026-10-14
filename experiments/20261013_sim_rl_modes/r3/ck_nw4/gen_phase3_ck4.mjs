// no_wall 確認 r4 の第 3 段の組(物理のずれの側の危なそうな所のまわり)。種 7,420,000 + i、乱数 mulberry32(0xC4A10E31)。
//   VVN: 学習の段の検証で止まった VV9(ゲート 23・左 +0.0183 m・左向き 14.80°・舵 0.8・ブレーキ 2.2・時定数 0.28・加速の上限 6・μ × 0.85・舵のゲイン 1.06・前進のゲイン 0.96・
//        遅延 0・摩擦円なし)そのもの(この段の約束と学習の段の約束の 2 通り)、開始のまわり 5 × 5(±0.005 m・±0.5°)、物理の 1 つずつの裏返し 9
//   DLS: 第 1・2 段の物理のずれの回のうち最小すき間が小さい 12 個のまわり(開始 ±0.004 m・±0.4° の 8 点 + 物理を範囲の幅の ±3% でずらした 4 点)
//   使い方: node gen_phase3_ck4.mjs <出力 jobs.jsonl> <runs のフォルダー>...
import fs from 'node:fs';
import {mulberry32, startClear} from './own_ck4.mjs';
const [outp, ...dirs] = process.argv.slice(2);
const rows = [];
for (const d of dirs) for (const f of fs.readdirSync(d).filter(f => /^rows_\d+\.jsonl$/.test(f))) for (const l of fs.readFileSync(`${d}/${f}`, 'utf8').split('\n')) if (l.trim()) rows.push(JSON.parse(l));
const isDr = r => !!r.phys || r.fc || r.delay;
const DRT = {steerRate: [.8, 1.8], brakeDeceleration: [.7, 2.2], driveTau: [.28, .55], driveAccelerationCap: [3.5, 6.0], frictionMultiplier: [.85, 1.1], sg: [.94, 1.06], vg: [.96, 1.04]};
const jobs = [];
let seed = 7420000;
const add = (set, spec) => jobs.push({id: `${set}:${jobs.filter(j => j.set === set).length}`, set, seed: seed++, ...spec});
const r6 = x => Math.round(x * 1e6) / 1e6;
const VV9 = {start: {gate: 23, off: 0.018328845407813793, yaw: 14.797822062391788}, phys: {steerRate: .8, brakeDeceleration: 2.2, driveTau: .28, driveAccelerationCap: 6, frictionMultiplier: .85},
  sg: 1.06, vg: .96, delay: 0, fc: false};
add('VVN', {...VV9, tag: 'VV9'});
add('VVN', {...VV9, tag: 'VV9-applied00', applied0: [0, 0]});
for (const dO of [-.005, -.0025, 0, .0025, .005]) for (const dY of [-.5, -.25, 0, .25, .5]) {
  if (dO === 0 && dY === 0) continue;
  const off = r6(VV9.start.off + dO), yaw = r6(Math.min(15, VV9.start.yaw + dY)), c = startClear(23, off, yaw);
  if (c >= .12) add('VVN', {...VV9, start: {gate: 23, off, yaw, c0: +c.toFixed(5)}, tag: 'start'});
}
const flip = (k, v) => DRT[k][0] === v ? DRT[k][1] : DRT[k][0];
for (const k of ['steerRate', 'brakeDeceleration', 'driveTau', 'driveAccelerationCap', 'frictionMultiplier']) add('VVN', {...VV9, phys: {...VV9.phys, [k]: flip(k, VV9.phys[k])}, tag: 'flip_' + k});
add('VVN', {...VV9, sg: flip('sg', VV9.sg), tag: 'flip_sg'});
add('VVN', {...VV9, vg: flip('vg', VV9.vg), tag: 'flip_vg'});
add('VVN', {...VV9, delay: 1, tag: 'flip_delay'});
add('VVN', {...VV9, fc: true, tag: 'flip_fc'});
// DLS
const R = mulberry32(0xC4A10E31);
const dr = rows.filter(r => !r.error && !r.invalid_spawn && isDr(r) && r.shield && !['LOWMU', 'DELAY2', 'VVN', 'DLS'].includes(r.set)).sort((a, b) => a.min_clearance - b.min_clearance).slice(0, 12);
for (const r of dr) {
  for (const [dO, dY] of [[-.004, 0], [.004, 0], [0, -.4], [0, .4], [-.004, -.4], [-.004, .4], [.004, -.4], [.004, .4]]) {
    const off = r6(r.start.off + dO), yaw = r6(r.start.yaw + dY);
    if (Math.abs(off) > .2 || Math.abs(yaw) > 15) continue;
    const c = startClear(r.start.gate, off, yaw);
    if (c >= .12) add('DLS', {start: {gate: r.start.gate, off, yaw, c0: +c.toFixed(5)}, phys: r.phys, sg: r.sg, vg: r.vg, delay: r.delay, fc: r.fc, of: r.id});
  }
  for (let k = 0; k < 4; k++) {
    const j = (key, v) => Math.min(DRT[key][1], Math.max(DRT[key][0], v + (R() * 2 - 1) * .03 * (DRT[key][1] - DRT[key][0])));
    const p = r.phys;
    add('DLS', {start: r.start, phys: {steerRate: j('steerRate', p.steerRate), brakeDeceleration: j('brakeDeceleration', p.brakeDeceleration), driveTau: j('driveTau', p.driveTau),
      driveAccelerationCap: j('driveAccelerationCap', p.driveAccelerationCap), frictionMultiplier: j('frictionMultiplier', p.frictionMultiplier)}, sg: j('sg', r.sg), vg: j('vg', r.vg),
      delay: r.delay, fc: r.fc, of: r.id});
  }
}
fs.writeFileSync(outp, jobs.map(j => JSON.stringify(j)).join('\n') + '\n');
console.log(JSON.stringify({jobs: jobs.length, VVN: jobs.filter(j => j.set === 'VVN').length, DLS: jobs.filter(j => j.set === 'DLS').length,
  dls_of: dr.map(r => ({id: r.id, min_cm: +(100 * r.min_clearance).toFixed(3), phys: r.phys, sg: r.sg, vg: r.vg, delay: r.delay, fc: r.fc}))}));
