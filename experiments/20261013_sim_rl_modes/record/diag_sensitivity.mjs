// 記録の段の診断: 記録と Node の教師の操作が違うフレームが「状態の 1e-13 m ほどの違いで観測が跳ぶ所」かどうかを見る。
// 同じ開ループの再生の snapshot を、そのまま(A)と、x・z を ±eps ずらしたもの(B±)で、別々の教師に与えて操作を比べる
//   node record/diag_sensitivity.mjs <データセット> <plan_scene> [eps]
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {readJson, BASE_WORLD, EXP} from '../rl/episode.mjs';
import {activeWorldFor, simConfig} from '../snapshot/rl/env_core.mjs';
import {createModeTeacher} from '../rl/runtime/mode_runtime.mjs';
const [root, want, epsArg] = process.argv.slice(2);
const eps = Number(epsArg ?? 1e-13);
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
const mk = () => createModeTeacher(world, cfgT, variation, {mode: key, weights, params: sc.params ?? {}, policySha256: card.rl.policy_sha256});
const variants = [[0, 0], [eps, 0], [-eps, 0], [0, eps], [0, -eps], [eps, eps], [-eps, -eps]];
const ts = variants.map(mk);
const res = [];
for (const r of rows) {
  const acts = variants.map(([dx, dz], i) => ts[i].action({state: {...snap.state, x: snap.state.x + dx, z: snap.state.z + dz}, metrics: snap.metrics, config: cfgT}));
  const c = r.command_si;
  const dRec = Math.abs(acts[0].speed - c.speed) + Math.abs(acts[0].steering - c.steering);
  const spread = Math.max(...acts.map(a => Math.abs(a.speed - acts[0].speed) + Math.abs(a.steering - acts[0].steering)));
  const recMatchesVariant = acts.findIndex(a => Math.abs(a.speed - c.speed) + Math.abs(a.steering - c.steering) < 1e-9);
  if (dRec > 1e-9 || spread > 1e-6) res.push({frame: r.frame, d_rec_vs_A: dRec, variant_spread: spread, rec_matches_variant: recMatchesVariant,
    speeds: acts.map(a => +a.speed.toFixed(6)), rec_speed: +c.speed.toFixed(6), x: snap.state.x, z: snap.state.z});
  snap = sim.step({steering: c.steering, speed: c.speed});
}
console.log(JSON.stringify({scene: sc.scene, plan_scene: want, eps, frames: res}, null, 1));
