// 診断(2026-10-13、no_wall の学習の段): シールドなしで方策を走らせた軌跡の各フレームで、シールドの検査(提案 1 フレーム + 予備の操縦 24 フレーム)が
// 通るかを、予備の操縦の先読みの距離の決め方を変えて数える(誤報の割合 = N7 の目安)。予備の操縦だけで 25 フレーム安全かも数える。
//   node rl/nw/shield_probe.mjs --policy <policy.json> [--set validation] [--tests NW-N60,NW-STD,NW-R200] [--max-r200 20] [--every 1]
import {buildJobs} from '../eval_modes.mjs';
import {readJson, BASE_WORLD, loadCore} from '../episode.mjs';
import {activeWorldFor, simConfig, PHYSICS, PERTURB_KEYS} from '../../snapshot/rl/env_core.mjs';
import {createModeTeacher} from '../runtime/mode_runtime.mjs';
import {createShield} from './shield_variant.mjs';
import {prepareGeometry} from '../runtime/features.mjs';
const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const tests = opt('--tests', 'NW-N60,NW-STD,NW-R200').split(',');
let J = buildJobs('no_wall', opt('--set', 'validation')).filter(j => tests.includes(j.test));
const maxR = +opt('--max-r200', '20'); let nr = 0;
J = J.filter(j => j.test !== 'NW-R200' || nr++ < maxR);
const weights = readJson(opt('--policy')), every = +opt('--every', '1');
const variants = JSON.parse(opt('--variants', '[{"name":"0.6","p":{}},{"name":"0.6+0.3v","p":{"lookahead_k":0.3}},{"name":"0.6+0.5v","p":{"lookahead_k":0.5}},{"name":"0.4+0.7v","p":{"lookahead_m":0.4,"lookahead_k":0.7}}]'));
const res = variants.map(v => ({name: v.name, n: 0, fail: 0, failNom: 0, failPess: 0, replaced: 0, byS: {}}));
for (const job of J) {
  const sp = job.spec, start = sp.start, dr = sp.dr ?? null;
  const core = await loadCore(dr?.frictionCircle ? 'fastfc' : 'fast');
  const {world, variation} = activeWorldFor(BASE_WORLD, start.gate, start.left_offset_m, start.left_yaw_deg);
  const cfg = {...simConfig({episodeMode: sp.episodeMode ?? 'continuous', timeLimit: 20, seed: start.seed}), ...(dr?.physics ?? {}), ...(dr?.frictionCircle ? {frictionCircle: true} : {})};
  const sim = core.createSimulation(world, cfg);
  sim.reset({seed: cfg.seed, scenario: 'baseline'});
  let snap = sim.peek();
  const cfgN = {...JSON.parse(JSON.stringify(snap.config)), ...Object.fromEntries(PERTURB_KEYS.map(k => [k, PHYSICS[k]]))}; delete cfgN.frictionCircle;
  const teacher = createModeTeacher(world, cfgN, variation, {mode: 'no_wall', weights, overrides: {shield: false}});
  const geom = prepareGeometry(world);
  const shields = variants.map(v => createShield({geometry: geom, config: cfgN, params: v.p}));
  const delay = dr?.delay ?? 0, queue = []; let appliedPrev = null, prevCmd = {steering: 0, speed: 0};
  while (!snap.state.terminal) {
    const st = {...snap.state};
    const out = teacher.action({state: snap.state, metrics: snap.metrics, config: cfgN}, delay ? {applied: appliedPrev ?? [0, 0]} : null);
    const lg = teacher.log.at(-1);
    if (st.frame % every === 0) {
      shields.forEach((sh, k) => {
        const r = sh.check(st, lg.s, prevCmd, {steering: out.steering, speed: out.speed});
        res[k].n++; if (!r.pass) { res[k].fail++; const r2 = sh.check(st, lg.s, prevCmd, {steering: out.steering, speed: out.speed}, false); if (r2.nominal < sh.params.c_nominal) res[k].failNom++; if (r2.pessimistic < sh.params.c_pessimistic) res[k].failPess++; const b = Math.floor((((lg.s % 26.12) + 26.12) % 26.12)); res[k].byS[b] = (res[k].byS[b] ?? 0) + 1; }
      });
    }
    prevCmd = {steering: out.steering, speed: out.speed};
    let ap = {steering: out.steering, speed: out.speed, a: [lg.a0, lg.a1]};
    if (delay) { queue.push(ap); ap = queue.length > delay ? queue.shift() : {steering: 0, speed: 0, a: [0, -1]}; appliedPrev = ap.a; }
    sim.step({steering: ap.steering * (dr?.steerGain ?? 1), speed: ap.speed * (dr?.speedGain ?? 1)}, true); snap = sim.peek();
  }
  process.stderr.write(`${job.test} ${job.id} ${snap.state.reason} ${res.map(r => `${r.name}:${(r.fail / r.n * 100).toFixed(1)}%`).join(' ')}\n`);
}
console.log(JSON.stringify(res.map(r => ({...r, fail_frac: r.fail / r.n}))));
