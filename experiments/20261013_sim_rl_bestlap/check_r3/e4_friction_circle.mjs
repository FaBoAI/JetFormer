#!/usr/bin/env node
// 確認 r3 の 4: 摩擦円の反事実を、確認 r1 の自前の積分とは別のやり方で(ユーザーの sim-core.js のコピーの横の上限 1 行だけを変える。縦を優先)。
// r3 の段の主張「摩擦円の反事実で r3 の 2 本は完走(8.655・8.677 秒)、選んだ方策・r2・教師は衝突」は、確認 r1 の積分(横を優先して縦を削る)の値。
// やり方が違っても同じ向きになるかを見る。STD の単周・連続と REC60 の 60 の開始。
//   node check_r3/e4_friction_circle.mjs
import {pathToFileURL} from 'node:url';
import {PLAN, loadPolicy, drive, save} from './common_r3.mjs';

const fc = await import(pathToFileURL(new URL('./sim_core_fc.mjs', import.meta.url).pathname).href);
const out = {schema: 'rl-check-r3-friction-circle/v1', created_at: new Date().toISOString(), variant: '縦を優先(横の上限を √((μg)² − a_long²) に下げる)', rows: {}};
for (const key of ['selected', 'r2_5M', 'r3_s21', 'r3_s22']) {
  const pol = loadPolicy(key);
  const single = drive(pol, {coreMod: fc}).judge, cont = drive(pol, {coreMod: fc, continuous: true}).judge;
  const rec = PLAN.map(row => drive(pol, {row, continuous: true, coreMod: fc}).judge);
  out.rows[key] = {single, cont, rec60_valid: rec.filter(j => j.ok).length, rec60_n: rec.length,
    rec60_fail_colliders: rec.filter(j => !j.ok).reduce((a, j) => (a[j.first_collision?.collider ?? j.reason] = (a[j.first_collision?.collider ?? j.reason] ?? 0) + 1, a), {})};
  console.log(key, single.ok ? single.laps[0].toFixed(4) : `${single.reason} ${JSON.stringify(single.first_collision)}`, cont.ok ? cont.laps.map(x => x.toFixed(4)) : cont.reason,
    `REC60 ${out.rows[key].rec60_valid}/${rec.length}`, JSON.stringify(out.rows[key].rec60_fail_colliders));
  save('e4_friction_circle.json', out);
}
