#!/usr/bin/env node
// 参考(合否に使わない): 各 run の一番良い選べるチェックポイントと最後のチェックポイントを、制御遅延 1 フレーム(アプリの固定遅延)で
// STD の停止から 1 周に走らせる(design.md 3.3 節。教師は遅延 1 でゲート 16 のあと衝突する)。2026-10-13、RL の学習の段
//
//   node rl/delay1_seeds.mjs --out rl/final/delay1_seeds.json [--core original]
import fs from 'node:fs';
import path from 'node:path';
import {runEpisode, policyController, teacherController, STD} from './eval_sets.mjs';
import {loadPolicy} from './policy_forward.mjs';

const args = process.argv.slice(2);
const opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const out = opt('--out'), coreKind = opt('--core', 'original');
const RL = '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/rl';
const sel = JSON.parse(fs.readFileSync(path.join(RL, 'selection.json'), 'utf8'));
const rows = [];
const one = async (label, controller, variant) => {
  const r = {};
  for (const d of [0, 1]) {
    const e = await runEpisode({start: STD, controller, coreKind, variant, episodeMode: 'single', timeLimit: 15, delayFrames: d});
    r[`delay${d}`] = {reason: e.reason, lap_s: e.lap_complete ? e.laps[0].elapsed : null, passed_gates: e.passed_gates, collisions: e.collisions,
      min_clearance: e.min_clearance};
  }
  rows.push({label, variant, ...r});
  console.log(label, JSON.stringify(r));
};
await one('RC 教師', await teacherController(), 'main');
for (const [name, run] of Object.entries(sel.runs)) {
  const pols = fs.readdirSync(path.join(RL, name, 'policies')).filter(f => f.endsWith('.json')).sort();
  if (!pols.length) continue;
  const picks = new Set([pols[pols.length - 1]]);
  if (run.best) picks.add(path.basename(run.best.policy));
  for (const p of picks) {
    const fn = await loadPolicy(path.join(RL, name, 'policies', p));
    await one(`${name} ${p.replace('.json', '')}${run.best && path.basename(run.best.policy) === p ? '(選べる最良)' : '(最後)'}`, policyController(fn), run.variant);
  }
}
if (out) fs.writeFileSync(out, JSON.stringify({schema: 'rl-bestlap-delay1/v1', core: coreKind, note: '参考。合否に使わない(design.md 3.3 節)', rows}, null, 1) + '\n');
