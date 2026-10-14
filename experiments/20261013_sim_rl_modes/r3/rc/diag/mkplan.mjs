import {buildJobs} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/rc/eval_rc3.mjs';
import fs from 'node:fs';
const J = buildJobs('validation').filter(j => ['V68', 'V1', 'V2'].includes(j.id) && j.test === 'RC-V100-R3');
const plan = {mode: 'recover', scenes: J.map(j => ({name: j.id, start: {gate: j.spec.start.gate, left_offset_m: j.spec.start.left_offset_m, left_yaw_deg: j.spec.start.left_yaw_deg}, seed: j.spec.start.seed}))};
fs.writeFileSync(process.argv[2], JSON.stringify(plan));
