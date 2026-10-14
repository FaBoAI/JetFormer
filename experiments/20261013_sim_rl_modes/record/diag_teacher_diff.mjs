// 記録の段の診断: 取り込んだデータセットの 1 シーンで、記録した操作と Node の教師(同じ snapshot)の操作が違うフレームを並べる
//   node record/diag_teacher_diff.mjs <データセット> <plan_scene>
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {readJson, BASE_WORLD, EXP} from '../rl/episode.mjs';
import {activeWorldFor, simConfig} from '../snapshot/rl/env_core.mjs';
import {createModeTeacher} from '../rl/runtime/mode_runtime.mjs';
const [root, want] = process.argv.slice(2);
const coll = readJson(path.join(root, 'simulation/collection.json'));
const card = readJson(path.join(root, 'dataset_card.json'));
const key = card.rl.mode;
const weights = readJson('/home/jetson/jetformer/data/sim_rl/modes/' + key + '/policy.json');
const core = await import(pathToFileURL(path.join(EXP, 'snapshot/sim/dist/sim-core.js')).href);
const sc = coll.scenes.find(s => s.plan_scene === want);
const dir = path.join(root, 'simulation/scenes', sc.scene);
const sess = readJson(path.join(dir, 'session.json'));
const rows = fs.readFileSync(path.join(dir, 'frames.jsonl'), 'utf8').split('\n').filter(Boolean).map(JSON.parse);
const st0 = sc.plan_start;
const {world, variation} = activeWorldFor(BASE_WORLD, st0.gate, st0.left_offset_m, st0.left_yaw_deg);
const cfg = simConfig({episodeMode: 'continuous', timeLimit: 20, seed: sess.config.seed, noiseStd: sess.config.noiseStd ?? 0});
const sim = core.createSimulation(world, cfg);
let snap = sim.reset({seed: cfg.seed, scenario: sess.config.noiseStd ? 'noise' : 'baseline'});
const cfgT = JSON.parse(JSON.stringify(snap.config));
const t = createModeTeacher(world, cfgT, variation, {mode: key, weights, params: sc.params ?? {}, policySha256: card.rl.policy_sha256});
const out = [];
for (const r of rows) {
  const a = t.action({state: snap.state, metrics: snap.metrics, config: cfgT});
  const c = r.command_si;
  const ds = Math.abs(a.steering - c.steering), dv = Math.abs(a.speed - c.speed);
  if (ds > 1e-9 || dv > 1e-9) out.push({frame: r.frame, node: {steering: a.steering, speed: a.speed}, rec: c, d_steer: ds, d_speed: dv, jf_node: a.info?.jf, jf_rec: r.action_info?.jf,
    state: {speed: snap.state.speed, steering: snap.state.steering}});
  snap = sim.step({steering: c.steering, speed: c.speed});
}
console.log(JSON.stringify({scene: sc.scene, plan_scene: want, diff_frames: out.length, frames: out}, null, 1));
