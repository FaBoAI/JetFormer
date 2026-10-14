// 離脱復活 r3 の確認: 最速のモードそのもの(mode 'fastest'、最速の方策・最速の走行中の仕組み)との比べ(2026-10-14 20:50)。
//   NOMF: 乱しなしの線の 8 回(jobs.json の NOM-fastest と同じ開始・種)。FUF: jobs.json の FU の最初の 100 の開始・同じ種で乱しなし
//   node ck_jobs_extra2.mjs <out.json>
import fs from 'node:fs';
const J0 = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_recover_r3/jobs.json', 'utf8'));
const J = [];
for (const j of J0.filter(j => j.set === 'NOM-fastest')) J.push({set: 'NOMF', id: j.id, spec: {...j.spec, policy: 'fastest', mode: 'fastest'}});
for (const j of J0.filter(j => j.set === 'FU').slice(0, 100)) J.push({set: 'FUF', id: j.id, spec: {...j.spec, policy: 'fastest', mode: 'fastest'}});
fs.writeFileSync(process.argv[2], JSON.stringify(J));
console.log(J.length);
