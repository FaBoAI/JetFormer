// 確認 line r3(再実行): 崩れた回をフレームつきで走らせ直す(図と、後ろへの指令・シールドの替え方を見るため)
// node failframes_b.mjs <id,id,...>
import fs from 'node:fs';
import {run} from './run_b.mjs';
const OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3b';
const want = new Set(process.argv[2].split(','));
const all = ['jobs.json', 'testjobs.json', 'probejobs.json'].flatMap(f => JSON.parse(fs.readFileSync(`${OUT}/${f}`, 'utf8')).jobs);
const out = [];
for (const j of all.filter(j => want.has(j.id))) {
  for (const sh of [true, false]) {
    const r = run({...j, shield: sh, keep: true});
    const neg = r.fr.filter(f => f[6] < 0).length;
    out.push({id: j.id, shield: sh, reason: r.reason, collision: r.collision, min_c: r.min_c, min_id: r.min_id, stall_first: r.stall_first, sh_frames: r.sh_frames, fb: r.fb,
      neg_speed_cmd_frames: neg, laps: r.laps.map(l => l.t), fr: r.fr});
    console.log(j.id, sh ? 'シールドあり' : 'シールドなし', r.reason, r.min_c, r.min_id, 'stall', r.stall_first, 'sh', r.sh_frames, JSON.stringify(r.fb), 'neg', neg);
  }
}
fs.writeFileSync(`${OUT}/fail_frames.json`, JSON.stringify(out));
