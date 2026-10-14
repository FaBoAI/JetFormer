#!/usr/bin/env node
// 試験用の「記録」を Node で作る(2026-10-14、sim_dart_20261014)。ブラウザの記録と同じ形のうち、付け替えが読むところだけを書く:
//   <out>/session.json(context.start_variation・context.teacher・calibration・model_config・display_name)
//   <out>/evaluation-report.json(actions・trace・config・seed・scenario)
//   <out>/frames.jsonl(frame・command_si・pose_pre_action・state_post_action・action_info)
// 走らせるのは dist/dart/entry-Dn.js(ブラウザと同じ入口)。app.js と同じ順(教師 → sim.step)。--frames n で先頭の n フレームだけ書く。
//   node make_fixture.mjs --dataset D1 --index 0 --frames 150 --out <dir>(乱しなしの試験には D2 か、A0 の本物の記録を使う)
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {runCondition, baseWorld, SIM_ROOT} from '../lib/app_emulation.mjs';

const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const D = opt('--dataset'), idx = Number(opt('--index', 0)), nFrames = Number(opt('--frames', 600)), out = opt('--out');
const plan = JSON.parse(fs.readFileSync(`/home/jetson/jetformer_reports/sim_dart_20261014/record_prep/plan_${D}.json`, 'utf8'));
const A0 = '/home/jetson/jetformer/data/datasets/jetformer-simulator-rl-bestlap-20min_20261013-174515_7f3650b0/simulation/scenes/20261013-161744-f330ca5f-0001';
const a0s = JSON.parse(fs.readFileSync(A0 + '/session.json', 'utf8'));
const mod = await import(pathToFileURL(`${SIM_ROOT}/dist/dart/entry-${D}.js`).href);
const c = plan[idx];
globalThis.__dartOverride = {};
const rows = [];
let teacherMeta = null;
const wrapped = {createRcTrainingTeacher: (w, cfg, v) => { const t = mod.createRcTrainingTeacher(w, cfg, v); teacherMeta = t.metadata; return t; }};
const {report, variation, sim} = await runCondition(c, wrapped, {world: await baseWorld(), onFrame: (before, action, after) => {
  const cfg = after.config;
  const bounded = {...action, steering: Math.max(-cfg.maxSteer, Math.min(cfg.maxSteer, action.steering)), speed: Math.max(-cfg.maxSpeed, Math.min(cfg.maxSpeed, action.speed))};
  rows.push({frame: before.state.frame, sim_time: before.state.sim_time, command_si: {steering: bounded.steering, speed: bounded.speed},
    action_info: action.info ?? {}, pose_pre_action: {x: before.state.x, z: before.state.z, yaw: before.state.yaw}, state_post_action: {...after.state}});
}});
fs.mkdirSync(out, {recursive: true});
const n = Math.min(nFrames, rows.length);
fs.writeFileSync(path.join(out, 'session.json'), JSON.stringify({schema: 'fixture', display_name: `${c.name}-fixture`,
  context: {start_variation: variation, teacher: teacherMeta}, calibration: a0s.calibration, model_config: a0s.model_config}));
fs.writeFileSync(path.join(out, 'evaluation-report.json'), JSON.stringify({actions: report.actions.slice(0, n), trace: report.trace.slice(0, n),
  config: report.config, seed: report.seed, scenario: report.scenario, outcome: report.outcome, metrics: {collisions: report.metrics.collisions}}));
fs.writeFileSync(path.join(out, 'frames.jsonl'), rows.slice(0, n).map(r => JSON.stringify(r)).join('\n') + '\n');
console.log(JSON.stringify({out, frames: n, noisy: rows.slice(0, n).filter(r => r.action_info?.dart?.noise).length, reason: report.outcome.reason}));
