// 摩擦円の物理: 自前の置き換え(own_ck3.mjs coreFc)と高速版(r3/nw/fast_core_nw3.mjs)を、同じ指令の列で開いたまま比べる
import {coreFc, BASE_WORLD, startWorld} from './own_ck3.mjs';
import {NOMINAL} from './run_ck3.mjs';
const fast = await import('/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/nw/fast_core_nw3.mjs');
let rnd = 12345; const R = () => (rnd = (rnd * 1103515245 + 12345) % 2147483648) / 2147483648;
let maxd = 0, firstNonzero = null, maxdStep1 = 0;
for (let trial = 0; trial < 20; trial++) {
  const {world} = startWorld(BASE_WORLD, trial % 33, 0, 0);
  const cfg = {...NOMINAL, seed: 1, timeLimit: 20, episodeMode: 'continuous-laps-v1', brakeDeceleration: .7, frictionMultiplier: .85, steerRate: 1.8};
  const a = coreFc.createSimulation(world, {...cfg, ckFrictionCircle: true}), b = fast.createSimulation(world, {...cfg, frictionCircle: true});
  a.reset({seed: 1}); b.reset({seed: 1});
  let st = 0, sp = 0;
  for (let f = 0; f < 600; f++) {
    // 強く曲がりながら加減速(摩擦円が効く指令)
    st = Math.max(-.52, Math.min(.52, st + (R() - .5) * .3)); sp = Math.max(0, Math.min(4.5, sp + (R() - .5) * 2));
    const sa = a.step({steering: st, speed: sp}); b.step({steering: st, speed: sp}, true); const sb = b.peek();
    const d = Math.max(Math.abs(sa.state.x - sb.state.x), Math.abs(sa.state.z - sb.state.z), Math.abs(sa.state.speed - sb.state.speed));
    if (d > 0 && firstNonzero === null) firstNonzero = {trial, f, d};
    if (f === 0) maxdStep1 = Math.max(maxdStep1, d);
    maxd = Math.max(maxd, d);
    if (sa.state.terminal || sb.state.terminal) { if (sa.state.reason !== sb.state.reason) console.log('reason differs', trial, f, sa.state.reason, sb.state.reason); break; }
  }
}
console.log(JSON.stringify({maxd, firstNonzero}));
