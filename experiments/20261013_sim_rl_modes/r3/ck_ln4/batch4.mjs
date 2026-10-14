// 確認 line r4: 組をまとめて走らせる(続きから再開できる。出力は 1 回 1 行の JSONL)
// 使い方: node batch4.mjs <jobs.json> <out.jsonl> <shard> <nshards> [上書きの JSON(例 {"shield":false})]
import fs from 'node:fs';
import {runJob} from './run4.mjs';
const [jobsPath, outPath, shard, nsh, ov] = process.argv.slice(2);
const all = JSON.parse(fs.readFileSync(jobsPath, 'utf8'));
const jobs = (Array.isArray(all) ? all : all.jobs).filter((_, i) => i % +nsh === +shard);
const over = ov ? JSON.parse(ov) : {};
const done = new Set();
if (fs.existsSync(outPath)) for (const line of fs.readFileSync(outPath, 'utf8').split('\n')) { if (!line.trim()) continue; try { done.add(JSON.parse(line).id); } catch { /* 途中で切れた行は走らせ直す */ } }
let n = 0;
const t0 = Date.now();
for (const j of jobs) {
  if (done.has(j.id)) continue;
  const r = runJob({...j, ...over});
  fs.appendFileSync(outPath, JSON.stringify(r) + '\n');
  n++;
  if (n % 20 === 0) console.log(`[${shard}] ${n} 回 ${((Date.now() - t0) / 1000 / n).toFixed(2)} 秒/回`);
}
console.log(`[${shard}] 終わり ${n} 回(飛ばした ${done.size})`);
