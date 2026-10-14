// 確認 line r3: sets.json の組を走らせる(ワーカー w / W、行ごとに JSONL)。
//   node ck_batch.mjs <variant> <w> <W>
//   variant: cand = 候補(シールドあり、r3 の写し)・nosh = 候補のシールドなし(r3 の写し)・web = 候補を共有の web/rl で(シールドなし)・fast = 最速(bundle/fastest、web/rl)
import fs from 'node:fs';
import {runOne} from './ck_run.mjs';
const OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3';
const [variant, w, W] = [process.argv[2], +process.argv[3], +process.argv[4]];
const S = JSON.parse(fs.readFileSync(`${OUT}/sets.json`, 'utf8'));
let jobs = S.jobs;
const V = {cand: {policy: 'line', rt: 'r3', shield: true}, nosh: {policy: 'line', rt: 'r3', shield: false}, web: {policy: 'line', rt: 'web', shield: false},
  fast: {policy: 'fastest', rt: 'web', shield: false}}[variant];
if (!V) throw new Error('variant?');
if (variant === 'web') jobs = jobs.filter(j => j.set === 'REPLAY');
// シールドなし: シールドが 1 フレームも替えなかった回は、シールドなしでも同じ走りになる(決定的)。替えた回と、照らし合わせ用に REPLAY の全部を走らせる
if (variant === 'nosh') {
  const acted = new Set();
  for (const f of fs.readdirSync(`${OUT}/runs`).filter(f => f.startsWith('cand_')))
    for (const l of fs.readFileSync(`${OUT}/runs/${f}`, 'utf8').split('\n')) if (l.trim()) { const r = JSON.parse(l); if (r.shield_frames > 0) acted.add(`${r.set}|${r.id}`); }
  jobs = jobs.filter(j => j.set === 'REPLAY' || acted.has(`${j.set}|${j.id}`));
}
if (variant === 'fast') jobs = [...jobs.filter(j => j.set === 'U300').slice(0, 60), ...jobs.filter(j => j.set === 'REPLAY' && j.test === 'LN-P60-R3'),
  {set: 'FAST_STD', id: 'STD', start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 4719000}, params: {}, family: null}];
const mine = jobs.filter((_, k) => k % W === w);
const file = `${OUT}/runs/${variant}_${w}.jsonl`;
const done = new Set();
if (fs.existsSync(file)) for (const l of fs.readFileSync(file, 'utf8').split('\n')) if (l.trim()) { const r = JSON.parse(l); done.add(`${r.set}|${r.id}`); }
const t0 = Date.now();
let n = 0;
for (const j of mine) {
  if (done.has(`${j.set}|${j.id}`)) continue;
  const r = runOne({...j, ...V, keepFrames: true, sampleCheck: true});
  r.test = j.test ?? null; r.variant = variant; r.job_dpsi = j.dpsi ?? null; r.variant_tag = j.variant ?? null;
  const interesting = r.set === 'REPLAY' || r.collisions > 0 || r.stall || r.shield_frames > 0 || r.min_clearance < .05 || !r.valid_r3 || r.set === 'FAST_STD';
  if (!interesting) delete r.fr;
  fs.appendFileSync(file, JSON.stringify(r) + '\n');
  n++;
}
console.log(JSON.stringify({variant, w, W, ran: n, of: mine.length, s: (Date.now() - t0) / 1000}));
