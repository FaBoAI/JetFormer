// 確認 r4: シールドの既定の値(ブラウザが使う)= 候補の mode.json の shield_params = 決まり(protocol_no_wall_r4.json runtime.shield.params)か。
// createShield が実際に使う値 P(深い合成のあと)で比べる
import fs from 'node:fs';
import {createShield, SHIELD_DEFAULTS, SHIELD_VERSION} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/candidates/no_wall_r4/runtime/shield.mjs';
import {prepareGeometry} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/candidates/no_wall_r4/runtime/features.mjs';
import {BASE_WORLD, NOMINAL_CFG} from './own_ck4.mjs';
const mode = JSON.parse(fs.readFileSync('/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/candidates/no_wall_r4/mode.json', 'utf8'));
const prot = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/protocol_no_wall_r4.json', 'utf8'));
const g = prepareGeometry(BASE_WORLD);
const norm = o => JSON.parse(JSON.stringify(o, (k, v) => v));
const sortKeys = o => Array.isArray(o) ? o.map(sortKeys) : (o && typeof o === 'object') ? Object.fromEntries(Object.keys(o).sort().map(k => [k, sortKeys(o[k])])) : o;
const Pdef = sortKeys(norm(createShield({geometry: g, config: NOMINAL_CFG, params: {}}).params));
const Pmode = sortKeys(norm(createShield({geometry: g, config: NOMINAL_CFG, params: mode.runtime.shield_params}).params));
const Pprot = sortKeys(norm(createShield({geometry: g, config: NOMINAL_CFG, params: prot.runtime.shield.params}).params));
const a = JSON.stringify(Pdef), b = JSON.stringify(Pmode), c = JSON.stringify(Pprot);
console.log(JSON.stringify({shield_version: SHIELD_VERSION, default_eq_mode_json: a === b, default_eq_protocol: a === c, mode_version: mode.runtime.shield_version, n_keys: Object.keys(Pdef).length}));
if (a !== b || a !== c) { console.log(a); console.log(b); console.log(c); process.exit(1); }
