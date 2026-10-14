#!/usr/bin/env node
// 確認 r3 の 8: 遅延 1 の角(舵 0.85 × ブレーキ 0.8、ほかは標準)を、観測の約束 2 通り・前進ゲイン 1.01(ONNX の記録の条件)でも。REC60 の 60 の開始
import {PLAN, loadPolicy, drive, save} from './common_r3.mjs';
const out = {};
for (const key of ['r3_s21', 'r3_s22']) {
  const pol = loadPolicy(key); out[key] = {};
  for (const [lab, o] of [['fresh', {prev: 'fresh'}], ['applied_gain101', {speedGain: 1.01}], ['fresh_gain101', {prev: 'fresh', speedGain: 1.01}]]) {
    const c = {phys: {steerRate: .85, brakeDeceleration: .8}, delay: 1, ...o};
    const s = drive(pol, c).judge;
    const ok = PLAN.filter(row => drive(pol, {...c, row, continuous: true}).judge.ok).length;
    out[key][lab] = {std_single: s.ok, lap: s.laps[0] ?? null, reason: s.reason, rec60: ok};
    console.log(key, lab, s.ok ? s.laps[0].toFixed(3) : s.reason, `${ok}/60`);
  }
}
save('e8_corner_cell.json', out);
