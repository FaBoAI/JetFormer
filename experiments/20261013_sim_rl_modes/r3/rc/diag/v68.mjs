import {runModeEpisode, readJson} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/rc/episode_rc3.mjs';
const w = readJson('/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/candidates/recover_r3/policy.json');
const start = {gate: 23, left_offset_m: 0.12200375958949344, left_yaw_deg: 13.147584000000002, seed: 3220068};
for (const sh of [true, false]) {
  const r = await runModeEpisode({mode: 'recover', weights: w, start, overrides: sh ? {} : {shield: false}, coreKind: 'fast', audit: false});
  const fb = {}; for (const l of r.log) { const k = l.shield?.fb ?? '-'; fb[k] = (fb[k] ?? 0) + 1; }
  console.log('shield', sh, r.reason, 'frames', r.frames, 'collisions', r.collisions, 'progress', (r.log.at(-1).s - r.log[0].s).toFixed(2), 'min', r.min_clearance?.toFixed?.(3), 'fb', JSON.stringify(fb), 'first cmds', r.log.slice(0, 3).map(l => `${l.steering?.toFixed(2)}/${l.speed?.toFixed(2)}`).join(' '));
}
