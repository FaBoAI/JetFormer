// 離脱復活の「立て直した」の判定に使う乱しなしの線(protocol definitions.nominal_line)を、登録簿の方策で作り直す(記録の段、2026-10-14)
//   node record/nominal_recover.mjs <out.json>
import fs from 'node:fs';
import crypto from 'node:crypto';
import {computeNominalLine} from '../rl/build_nominal_line.mjs';
const p = '/home/jetson/jetformer/data/sim_rl/modes/recover/policy.json';
const sha = crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const line = await computeNominalLine({mode: 'recover', weights: JSON.parse(fs.readFileSync(p, 'utf8')), policySha256: sha});
fs.writeFileSync(process.argv[2], JSON.stringify(line));
console.log('書いた', process.argv[2], sha, line.runs_detail.map(r => r.reason + ' ' + r.laps.map(x => x.toFixed(2)).join('/')).join(' | '));
