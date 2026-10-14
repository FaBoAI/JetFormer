// 開始の分布(ゲート U{0..32}・±0.2 m・±15°・開始のすき間 ≥ 0.12 m)の中の |Δψ| の帯・開始のすき間の帯の割合(自前の幾何、一様 20,000 回)
import fs from 'node:fs';
import {startInfo} from './ck_run.mjs';
function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const rnd = mulberry32(0x6c6e33f0);
let n = 0, ok = 0; const band = {lt06: 0, b06_08: 0, b08_10: 0, ge10: 0}, near = {c12_14: 0, c14_15: 0}, byGate = {};
for (let i = 0; i < 20000; i++) {
  const g = Math.floor(rnd() * 33), off = -.2 + .4 * rnd(), yaw = -15 + 30 * rnd(), I = startInfo(g, off, yaw);
  n++; if (!(I.clearance >= .12)) continue; ok++;
  const a = Math.abs(I.dpsi);
  if (a < .6) band.lt06++; else if (a < .8) band.b06_08++; else if (a < 1) band.b08_10++; else { band.ge10++; byGate[g] = (byGate[g] ?? 0) + 1; }
  if (I.clearance < .14) near.c12_14++; else if (I.clearance < .15) near.c14_15++;
}
const out = {drawn: n, clearance_ok: ok, share: Object.fromEntries(Object.entries(band).map(([k, v]) => [k, v / ok])), near: Object.fromEntries(Object.entries(near).map(([k, v]) => [k, v / ok])), ge10_by_gate: byGate};
fs.writeFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3/share.json', JSON.stringify(out, null, 1));
console.log(JSON.stringify(out));
