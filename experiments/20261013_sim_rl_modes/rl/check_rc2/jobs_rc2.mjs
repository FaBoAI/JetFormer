// 離脱復活の確認 r2: 新しい種・新しい開始の一覧(2026-10-14、確認の段)。
//   node rl/check_rc2/jobs_rc2.mjs <out.json>
// 種は学習・検証・試験(903000〜)・記録(703000〜)・確認 r1(1830000〜1850199)のどれとも重ならない 2710000〜 を使う。
// 開始は記録の開始の分布(ゲート U{0..32}・左右 ±0.2 m・向き ±15°)から一様乱数(確認 r1 と別の乱数の種)。重なる開始は飛ばす。
import fs from 'node:fs';
import {startClearance} from '../check_rc/run_rc.mjs';
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const {createRng} = await import(`${EXP}/web/rl/rng.mjs`);
const J = [];
const add = (set, id, spec) => J.push({set, id, spec});
const r = createRng(0x14102055);
const fresh = [];
while (fresh.length < 390) {
  const st = {gate: Math.min(32, Math.floor(r.uniform() * 33)), left_offset_m: r.uniform(-.2, .2), left_yaw_deg: r.uniform(-15, 15)};
  if (startClearance(st.gate, st.left_offset_m, st.left_yaw_deg) > 1e-9) fresh.push(st);
}
for (let i = 0; i < 150; i++) add('F2-P', `Q${i}`, {mode: 'recover', start: {...fresh[i], seed: 2710000 + i}});
for (let i = 0; i < 240; i++) {
  const kind = i % 2 ? 'lateral' : 'yaw', sign = r.uniform() < .5 ? 1 : -1, frame = Math.round(r.uniform(1, 19) * 30);
  add(kind === 'yaw' ? 'F2-KY' : 'F2-KL', `K${i}`, {mode: 'recover', perturb: false, kind, kick: {frame, [kind]: sign * (kind === 'yaw' ? .3 : .15)},
    start: {...fresh[150 + i], seed: 2720000 + i}});
}
fs.writeFileSync(process.argv[2], JSON.stringify(J));
console.log('jobs', J.length, Object.entries(J.reduce((a, j) => (a[j.set] = (a[j.set] ?? 0) + 1, a), {})).map(([k, v]) => `${k}:${v}`).join(' '));
