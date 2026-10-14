// 離脱復活の確認 r1: 走らせる回の一覧を作る(2026-10-14)。
//   node rl/check_rc/jobs_rc.mjs <out.json>
// 試験の組の作り直し(REPLAY)は protocol の作り方(Halton 20,001〜・種 903000 + i・重なる開始は飛ばす)をここで作り直し、
// 新しい組(FRESH)は試験・検証・記録・線で使っていない種と、一様乱数の開始を使う。
import fs from 'node:fs';
import {startClearance} from './run_rc.mjs';
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const {haltonStart, createRng, mix32} = await import(`${EXP}/web/rl/rng.mjs`);
const modes = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/modes.json', 'utf8'));
const dist = modes.modes.find(m => m.key === 'recover').recording.start_distribution;
const J = [];
const add = (set, id, spec) => J.push({set, id, spec});
// 1) 乱しなしの線(protocol definitions.nominal_line)
for (const [mode, pol] of [['recover', 'recover'], ['fastest', 'fastest']]) for (let i = 0; i < 8; i++) {
  const seed = 990000 + i, rng = createRng(mix32(seed));
  add(`NOM-${mode}`, `N${i}`, {mode, policy: pol, perturb: false, start: {gate: 0, left_offset_m: rng.uniform(-.03, .03), left_yaw_deg: rng.uniform(-3, 3), seed}});
}
// 2) RC-P100 の作り直し
{
  let i = 20001, k = 0;
  while (k < 100) {
    const st = haltonStart(i, dist);
    if (startClearance(st.gate, st.left_offset_m, st.left_yaw_deg) > 1e-9) { add('REPLAY-P100', `P${k}`, {mode: 'recover', start: {...st, seed: 903000 + k}}); k++; }
    i++;
  }
}
// 3) RC-KICK の作り直し(STD・種 1・乱しなし、1.0〜19.0 秒の 0.5 秒おき × 左右 × 向き 0.3 rad / 横 0.15 m)
for (let i = 0; i < 37; i++) { const t = 1 + .5 * i; for (const sign of [1, -1]) for (const kind of ['yaw', 'lateral'])
  add('REPLAY-KICK', `${kind}${sign > 0 ? '+' : '-'}${t.toFixed(1)}`, {mode: 'recover', perturb: false, kind, start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1},
    kick: {frame: Math.round(t * 30), [kind]: sign * (kind === 'yaw' ? .3 : .15)}}); }
// 4) RC-STD・RC-REC60
add('REPLAY-STD', 'single', {mode: 'recover', perturb: false, episodeMode: 'single', start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}});
add('FAST-STD', 'single', {mode: 'fastest', perturb: false, episodeMode: 'single', start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}});
const plan = JSON.parse(fs.readFileSync(`${REL}/qa/rc-collection-plan-60.json`, 'utf8'));
plan.forEach(c => add('REPLAY-REC60', c.name, {mode: 'recover', perturb: false, start: {gate: c['start-gate'] ?? 0, left_offset_m: c['spawn-offset'] ?? 0, left_yaw_deg: c['spawn-yaw'] ?? 0, seed: c.seed}}));
// 5) 新しい開始(一様乱数、記録の開始の分布: ゲート U{0..32}・左右 ±0.2 m・向き ±15°、重なる開始は飛ばす)と新しい種
const r = createRng(0x14102027);
const fresh = [];
while (fresh.length < 200) {
  const st = {gate: Math.min(32, Math.floor(r.uniform() * 33)), left_offset_m: r.uniform(-.2, .2), left_yaw_deg: r.uniform(-15, 15)};
  if (startClearance(st.gate, st.left_offset_m, st.left_yaw_deg) > 1e-9) fresh.push(st);
}
fresh.forEach((st, i) => {
  add('FRESH-P', `F${i}`, {mode: 'recover', start: {...st, seed: 1830000 + i}});
  add('FRESH-P-FAST', `F${i}`, {mode: 'recover', policy: 'fastest', start: {...st, seed: 1830000 + i}});
  if (i < 100) {
    add('FRESH-U', `F${i}`, {mode: 'recover', perturb: false, start: {...st, seed: 1830000 + i}});
    add('FRESH-U-FAST', `F${i}`, {mode: 'fastest', perturb: false, start: {...st, seed: 1830000 + i}});
    add('FRESH-WM', `F${i}`, {mode: 'recover', classes: ['弱', '中'], start: {...st, seed: 1840000 + i}});
  }
  // キック: 1〜19 秒の一様な時刻(フレーム)・左右・向き/横を交互
  const kind = i % 2 ? 'lateral' : 'yaw', sign = r.uniform() < .5 ? 1 : -1, frame = Math.round(r.uniform(1, 19) * 30);
  const kick = {frame, [kind]: sign * (kind === 'yaw' ? .3 : .15)};
  add('FRESH-KICK', `K${i}`, {mode: 'recover', perturb: false, kind, kick, start: {...st, seed: 1850000 + i}});
  add('FRESH-KICK-FAST', `K${i}`, {mode: 'fastest', perturb: false, kind, kick, start: {...st, seed: 1850000 + i}});
});
fs.writeFileSync(process.argv[2], JSON.stringify(J));
console.log('jobs', J.length, Object.entries(J.reduce((a, j) => (a[j.set] = (a[j.set] ?? 0) + 1, a), {})).map(([k, v]) => `${k}:${v}`).join(' '));
