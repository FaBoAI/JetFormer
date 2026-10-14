// 確認 line r3(再実行): 組をまとめて走らせる。node batch_b.mjs <jobs.json> <k> <K> <out.jsonl> [shield|noshield|fastest]
// k 番目の働き手は index % K == k の回だけ。出力に既にある id は飛ばす(途中から続けられる)
import fs from 'node:fs';
import {run} from './run_b.mjs';
const [jf, k, K, out, how = 'shield'] = process.argv.slice(2);
const {jobs} = JSON.parse(fs.readFileSync(jf, 'utf8'));
const done = new Set();
if (fs.existsSync(out)) for (const l of fs.readFileSync(out, 'utf8').split('\n')) if (l.trim()) done.add(JSON.parse(l).id);
let n = 0;
const t0 = Date.now();
for (let i = 0; i < jobs.length; i++) {
  if (i % +K !== +k) continue;
  const j = {...jobs[i]};
  if (how === 'noshield') { j.shield = false; }
  if (how === 'fastest') { j.shield = false; j.policy = 'fastest'; }
  if (done.has(j.id)) continue;
  const r = run(j);
  r.dpsi_gen = j.dpsi; r.corner = j.corner ?? null; r.edge = j.edge ?? null;
  fs.appendFileSync(out, JSON.stringify(r) + '\n');
  n++;
}
console.log(`働き手 ${k}/${K}: ${n} 回、${((Date.now() - t0) / 1000).toFixed(0)} 秒`);
