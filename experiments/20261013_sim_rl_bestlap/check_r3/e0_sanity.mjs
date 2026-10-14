#!/usr/bin/env node
// 確認 r3 の 0: 新しく書いた道具が、記録済みの値(確認 r1・r2 の sha256・周回時間)を作り直せるかの確かめ
import {loadPolicy, drive, driveTeacher, hashJson} from './common_r3.mjs';
const t0 = Date.now();
for (const k of ['selected', 'r2_5M', 'r3_s21', 'r3_s22']) {
  const p = loadPolicy(k), s = drive(p), c = drive(p, {continuous: true});
  console.log(k, s.judge.ok, s.judge.laps.map(x => x.toFixed(6)), hashJson(s.report.actions).slice(0, 12), '| cont', c.judge.ok, c.judge.laps.map(x => x.toFixed(6)), hashJson(c.report.actions).slice(0, 12));
}
const t = driveTeacher(); console.log('teacher', t.judge.ok, t.judge.laps.map(x => x.toFixed(6)));
console.log('wall', (Date.now() - t0) / 1000);
