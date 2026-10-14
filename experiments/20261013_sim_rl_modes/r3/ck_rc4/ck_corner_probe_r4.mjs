// 離脱復活 r4 の確認: 角の開始(ゲート 17・7 の右)と P27 を r4・r3・最速のモードで乱しなしに走らせて比べる(2026-10-15、報告だけ)
import {runOne, LOOP} from './ck_run.mjs';
const GM = 2.019;
// ゲートの外し(自前の射影): 1 フレームで 0.5 m より跳んだところは進んだ距離に数えない(ck_analyze_r4.py と同じ直し)。jumps = 跳んだフレーム
function gm(F) { let lastS = F[0].s, lastK = `${F[0].g}|${F[0].lap}`, n = 0, at = [], prev = F[0].s; for (const q of F) { const k = `${q.g}|${q.lap}`; if (k !== lastK) { lastK = k; lastS = q.s; } else if (Math.abs(q.s - prev) > .5) { lastS = q.s - (prev - lastS); } else if (q.s - lastS > GM) { n++; at.push(q.f); lastS = q.s; } prev = q.s; } return [n, at]; }
const jumps = F => F.filter((q, i) => i && Math.abs(q.s - F[i - 1].s) > .5).map(q => q.f);
function stall(F) { let run = 0, mx = 0; for (const q of F) { if (q.f < 30) continue; if (Math.abs(q.v) < .2) { run++; mx = Math.max(mx, run); } else run = 0; } return mx; }
for (const [name, st] of [['C17_R+', {gate: 17, left_offset_m: -.2, left_yaw_deg: 15}], ['C7_R+', {gate: 7, left_offset_m: -.2, left_yaw_deg: 15}], ['P27', {gate: 7, left_offset_m: -0.1970645847798721, left_yaw_deg: 3.268876800000001}]]) {
  for (const [pol, mode] of [['r4', 'recover'], ['r3', 'recover'], ['fastest', 'fastest']]) {
    const r = await runOne({policy: pol, mode, start: {...st, seed: 9625000}, perturb: false});
    const F = r.frames_data, [n, at] = gm(F);
    console.log(name, pol, mode, r.reason, r.frames, 'coll', r.collisions, 'gate_miss', n, at.slice(0, 3), 'proj_jumps', jumps(F).slice(0, 3), 'first_gate_frame', F.find(q => q.g > 0)?.f, 'stall_max', stall(F), 'shield', F.filter(q => q.sh).length, 'unstick', F.filter(q => q.fb === 'unstick').length, 'min_clr', r.min_clearance.toFixed(4), 'ey@20', F[20]?.ey.toFixed(3));
  }
}
