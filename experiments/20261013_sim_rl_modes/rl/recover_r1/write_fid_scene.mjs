// 離脱復活の F-ID(ブラウザと Node の操作の一致の確かめに使う 600 フレームの列)を、乱しが入って 20 秒を走り切るシーンで書く(学習の段 recover r1)。
//   node rl/recover_r1/write_fid_scene.mjs --policy <policy.json> --start '{"gate":12,...,"seed":903005}' --out <fid.json>
import fs from 'node:fs';
import crypto from 'node:crypto';
import {writeFid} from '../eval_modes.mjs';
const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, arr) => (x.startsWith('--') ? a.concat([[x.slice(2), arr[i + 1]]]) : a), []));
const weights = JSON.parse(fs.readFileSync(args.policy, 'utf8')), sha = crypto.createHash('sha256').update(fs.readFileSync(args.policy)).digest('hex');
await writeFid('recover', weights, sha, args.out, {start: JSON.parse(args.start)});
const f = JSON.parse(fs.readFileSync(args.out, 'utf8'));
console.log('frames', f.frames.length, 'learn=0', f.frames.filter(x => x.action.info.jf.l === 0).length, 'events', new Set(f.frames.map(x => x.action.info.jf.ev).filter(Boolean)).size);
