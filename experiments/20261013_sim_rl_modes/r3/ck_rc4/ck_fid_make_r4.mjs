// 離脱復活 r4 の確認: ブラウザと Node で同じ操作になるか(F-ID)の材料(2026-10-15。確認 r3 の ck_fid_make.mjs の写しを r4 に)。
// 元の sim-core.js + 候補 r4 の runtime で走らせ、各フレームに教師へ渡した {state, metrics, config} と教師の出力 {steering, speed, info} を残す。
//   node ck_fid_make_r4.mjs <jobs.json> <出力のフォルダ> <組/番号> [<組/番号> …]
import fs from 'node:fs';
import crypto from 'node:crypto';
import {BASE_WORLD, activeWorld, NOMINAL, CAND, POLICY_PATHS} from './ck_run.mjs';
const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const core = await import(`${REL}/dist/sim-core.js`);
const RT = await import(`${CAND}/runtime/mode_runtime.mjs`);
const weights = JSON.parse(fs.readFileSync(POLICY_PATHS.r4, 'utf8'));
const psha = crypto.createHash('sha256').update(fs.readFileSync(POLICY_PATHS.r4)).digest('hex');
const [jobsPath, out, ...names] = process.argv.slice(2);
const JOBS = JSON.parse(fs.readFileSync(jobsPath, 'utf8'));
for (const nm of names) {
  const [set, id] = nm.split('/');
  const j = JOBS.find(q => q.set === set && q.id === id);
  if (!j || j.spec.kick || j.spec.policy !== 'r4') throw new Error('使えない回: ' + nm);
  const st = j.spec.start, seed = st.seed;
  const {world, variation} = activeWorld(BASE_WORLD, st.gate, st.left_offset_m, st.left_yaw_deg);
  const sim = core.createSimulation(world, {...NOMINAL, seed, timeLimit: 20, episodeMode: 'continuous-laps-v1'});
  let snap = sim.reset({seed, scenario: 'baseline'});
  const config = JSON.parse(JSON.stringify(snap.config));
  const params = j.spec.params ?? {};
  const overrides = j.spec.perturb === false ? {perturbation: false} : {};
  const teacher = RT.createModeTeacher(world, snap.config, variation, {mode: 'recover', weights, policySha256: psha, params, overrides});
  const frames = [];
  while (!snap.state.terminal) {
    const s = JSON.parse(JSON.stringify({state: snap.state, metrics: snap.metrics, config: snap.config}));
    const a = teacher.action({state: snap.state, metrics: snap.metrics, config: snap.config});
    frames.push({snapshot: s, action: {steering: a.steering, speed: a.speed, info: a.info}});
    snap = sim.step({steering: a.steering, speed: a.speed});
  }
  const unstick = teacher.log.filter(q => q.shield?.fb === 'unstick' && q.sh === 1).length;
  const fid = {name: nm.replace('/', '_'), world, config, variation, params, overrides, policy_sha256: psha, metadata: teacher.metadata, frames,
    events: teacher.events.length, shield_frames: teacher.shieldFrames, unstick_frames: unstick, reason: snap.state.reason};
  fs.writeFileSync(`${out}/fid_${fid.name}.json`, JSON.stringify(fid));
  console.log(nm, frames.length, 'events', fid.events, 'shield', fid.shield_frames, 'unstick', unstick, snap.state.reason);
}
