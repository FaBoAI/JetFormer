#!/usr/bin/env node
// 確認 r3 の 4b: 摩擦円の反事実(縦を優先・e4 と同じコピー)での RC 教師。STD 単周・連続と、約束の中の REC60
import {pathToFileURL} from 'node:url';
import {PLAN, driveTeacher, save} from './common_r3.mjs';
const fc = await import(pathToFileURL(new URL('./sim_core_fc.mjs', import.meta.url).pathname).href);
const inC = r => r['start-gate'] === 0 && Math.abs(r['spawn-offset']) <= .03 + 1e-12 && Math.abs(r['spawn-yaw']) <= 3 + 1e-12;
const s = driveTeacher({coreMod: fc}).judge, c = driveTeacher({coreMod: fc, continuous: true}).judge;
const rec = PLAN.filter(inC).map(row => driveTeacher({row, continuous: true, coreMod: fc}).judge);
const out = {single: s, cont: c, rec60_valid: rec.filter(j => j.ok).length, rec60_n: rec.length};
console.log('teacher', s.ok ? s.laps[0].toFixed(4) : `${s.reason} ${JSON.stringify(s.first_collision)}`, c.ok ? c.laps.map(x => x.toFixed(4)) : c.reason, `REC60 ${out.rec60_valid}/${out.rec60_n}`);
save('e4b_teacher_fc.json', out);
