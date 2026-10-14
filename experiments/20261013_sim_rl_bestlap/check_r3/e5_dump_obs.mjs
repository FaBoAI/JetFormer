#!/usr/bin/env node
// 確認 r3 の 5 の準備: 自前の閉ループの観測と、自前の MLP の行動を書き出す(学習のチェックポイントとの照合用)
//   node check_r3/e5_dump_obs.mjs <方策> <出力.json>
import fs from 'node:fs';
import {PLAN, loadPolicy, drive} from './common_r3.mjs';
const [key, outPath] = process.argv.slice(2);
const pol = loadPolicy(key), obs = [], act = [];
const rec = o => { const f = x => { const a = pol(x); obs.push(Array.from(x)); act.push(a); return a; }; return drive(f, o); };
rec({}); rec({continuous: true, row: PLAN[7]}); rec({continuous: true, row: PLAN[40], phys: {steerRate: .85, brakeDeceleration: .8}, delay: 1});
rec({continuous: true, row: PLAN[23], phys: {frictionMultiplier: .88, driveTau: .5}});
fs.writeFileSync(outPath, JSON.stringify({key, sha: pol.sha, obs, act}));
console.log(key, obs.length);
