// ブラウザ(teacher.js は shieldParams を渡さない = shield.mjs の既定)と mode.json の shield_params を渡したときで、シールドの値が同じか
import fs from 'node:fs';
const C = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/candidates/no_wall_r3';
const {createShield, SHIELD_DEFAULTS, SHIELD_VERSION} = await import(`${C}/runtime/shield.mjs`);
const {prepareGeometry} = await import(`${C}/runtime/features.mjs`);
const {BASE_WORLD} = await import('./own_ck3.mjs');
const m = JSON.parse(fs.readFileSync(`${C}/mode.json`, 'utf8'));
const g = prepareGeometry(BASE_WORLD), cfg = {dt: 1 / 30, maxSpeed: 7.24, maxSteer: .52, steerRate: 1.2, brakeDeceleration: 1.2, wheelbase: .26, vehicleLength: .42, vehicleWidth: .19, frictionMultiplier: 1, longitudinalModel: 'real-drive-response-estimate-v1', driveTau: .4, driveAccelerationCap: 6};
const a = createShield({geometry: g, config: cfg}).params, b = createShield({geometry: g, config: cfg, params: m.runtime.shield_params}).params;
const canon = o => JSON.stringify(o, (k, v) => v && typeof v === 'object' && !Array.isArray(v) ? Object.fromEntries(Object.keys(v).sort().map(x => [x, v[x]])) : v);
console.log(JSON.stringify({version: SHIELD_VERSION, mode_json_version: m.runtime.shield_version, equal: canon(a) === canon(b)}));
if (canon(a) !== canon(b)) { console.log(canon(a)); console.log(canon(b)); }
