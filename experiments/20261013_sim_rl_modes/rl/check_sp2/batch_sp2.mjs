// 速さいろいろ(speed)の確認 r2: 組を走らせて JSONL に書く(2026-10-14、確認の段)
//   node batch_sp2.mjs --set REPLAY [--shard k/n] [--limit N] --out 結果.jsonl     (CHK_RT=web で web/rl の写しを使う)
import fs from 'node:fs';
import {runOne} from './run_sp2.mjs';
import {buildSet} from './jobs_sp2.mjs';

const arg = k => { const i = process.argv.indexOf(k); return i >= 0 ? process.argv[i + 1] : null; };
const set = arg('--set'), out = arg('--out'), [sk, sn] = (arg('--shard') ?? '0/1').split('/').map(Number), limit = +(arg('--limit') ?? Infinity);
const jobs = buildSet(set === 'REPLAY_WEB' ? 'REPLAY' : set.startsWith('HOT') ? 'HOT' : set).filter((_, i) => i % sn === sk).slice(0, limit);
const fd = fs.openSync(out, 'w');
const t0 = Date.now();
for (const j of jobs) {
  const t1 = Date.now();
  const r = await runOne(j.spec);
  r.set = set; r.id = j.id; r.wall_s = (Date.now() - t1) / 1000;
  fs.writeSync(fd, JSON.stringify(r) + '\n');
}
fs.closeSync(fd);
console.log(`${set} ${sk}/${sn}: ${jobs.length} 回 ${((Date.now() - t0) / 1000).toFixed(1)} 秒`);
