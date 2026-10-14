// 離脱復活 r3 の確認: 足した回の一覧(2026-10-14 20:30、停電のあとの続き)。
//   1. 決めた試験の組(RC-P100-R3・RC-KICK-R3・RC-U60-R3)を自前の作り方で作り直す(学習の段の eval_test.json と 1 回ずつ比べる)。
//      開始は自前の Halton(基数 2・3・5)と自前の開始のすき間(ck_geom.mjs)で作り、eval_test.json の開始と照合する
//   2. 強さを固定した記録の設定(画面の「乱しの強さ」弱・中。強は jobs.json の FSTRONG)
//   3. シールドを切った比べ(報告だけ。立て直しが方策の力かシールドの力か)
//   node ck_jobs_extra.mjs <out.json>
import fs from 'node:fs';
import {startClearance} from './ck_run.mjs';

const EV = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/rc3/eval_test.json', 'utf8'));
const J = [], add = (set, id, spec) => J.push({set, id, spec});
function halton(i, b) { let f = 1, r = 0, n = i; while (n > 0) { f /= b; r += f * (n % b); n = Math.floor(n / b); } return r; }
function haltonFromOwn(from, count) {
  const out = []; let i = from;
  for (; out.length < count; i++) {
    const st = {gate: Math.min(32, Math.floor(halton(i, 2) * 33)), left_offset_m: -.2 + .4 * halton(i, 3), left_yaw_deg: -15 + 30 * halton(i, 5), halton_index: i};
    const c = startClearance(st.gate, st.left_offset_m, st.left_yaw_deg);
    if (!(c > 1e-9)) continue;
    out.push({...st, start_clearance: c});
  }
  return {starts: out, next: i};
}
const P = haltonFromOwn(28001, 100), U = haltonFromOwn(P.next, 60);
// 学習の段の開始と照合
const tP = EV.tests['RC-P100-R3'], tU = EV.tests['RC-U60-R3'];
let mism = 0;
const cmp = (a, b) => a.gate === b.gate && Math.abs(a.left_offset_m - b.left_offset_m) < 1e-12 && Math.abs(a.left_yaw_deg - b.left_yaw_deg) < 1e-9 && a.halton_index === b.halton_index;
P.starts.forEach((st, i) => { if (!cmp(st, tP[i].start)) mism++; });
U.starts.forEach((st, i) => { if (!cmp(st, tU[i].start)) mism++; });
P.starts.forEach((st, i) => add('RP', `P${i}`, {policy: 'r3', start: {...st, seed: 3120000 + i}, perturb: true}));
U.starts.forEach((st, i) => add('RU', `U${i}`, {policy: 'r3', start: {...st, seed: 3122000 + i}, perturb: false}));
// RC-KICK-R3: STD・時刻 1.25 + 0.5k・左右・向き/横・種 3,121,000 + i(eval_rc3.mjs と同じ並び)
let ki = 0;
const STD = {gate: 0, left_offset_m: 0, left_yaw_deg: 0};
for (let k = 0; k <= 35; k++) for (const sign of [1, -1]) for (const kind of ['yaw', 'lateral']) {
  const t = 1.25 + .5 * k;
  add('RK', `${kind}${sign > 0 ? '+' : '-'}${t.toFixed(2)}`, {policy: 'r3', start: {...STD, seed: 3121000 + ki++}, perturb: false,
    kick: {frame: Math.round(t * 30), [kind]: sign * (kind === 'yaw' ? .3 : .15)}, kind});
}
// 強さ固定(弱・中)と、シールドを切った比べ: jobs.json の FP の最初の 60 の開始と同じ(種は新しく)
const FPJ = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_recover_r3/jobs.json', 'utf8')).filter(j => j.set === 'FP').slice(0, 60);
FPJ.forEach((j, i) => {
  const st = {...j.spec.start};
  add('FWEAK', `P${i}`, {policy: 'r3', start: {...st, seed: 7380000 + i}, perturb: true, params: {perturb_class: '弱'}});
  add('FMED', `P${i}`, {policy: 'r3', start: {...st, seed: 7381000 + i}, perturb: true, params: {perturb_class: '中'}});
  add('FNS', `P${i}`, {policy: 'r3', start: {...st}, perturb: true, overrides: {shield: false}});
});
fs.writeFileSync(process.argv[2], JSON.stringify(J));
const cnt = {}; for (const j of J) cnt[j.set] = (cnt[j.set] ?? 0) + 1;
console.log(JSON.stringify(cnt), '開始の食い違い', mism, 'P の次の番号', P.next, 'U の次の番号', U.next);
