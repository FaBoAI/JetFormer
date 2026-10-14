#!/usr/bin/env node
// 乱し v2(rt/perturb_v2.mjs)の試験(2026-10-14、離脱復活 r3)。node r3/rc/test_perturb_v2.mjs [検証の結果.json]
//   1. 替え方の順番(protocol_recover_r3.json perturbation_r3.scheduler.fallback_order)
//   2. 全部使える表・散らし 1.0 なら、v1(perturb.mjs)と同じ乱数の列・同じ出来事(種類・強さ・向き・始まり・中身)になる(わざと外すの幅が足りない回を除く)
//   3. 決定的: 同じ列を 2 回与えると同じ出来事
//   4. (結果のファイルを渡したとき)走った回の全部の出来事の「当てた組」が、始まりの s・速さで表の使えるマス(≥ 0.95)
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createPerturberV2, fallbackOrder, comboKey, usable} from './rt/perturb_v2.mjs';
import {createPerturber, RECORD_SPEC, TRAIN_SPEC} from './rt/perturb.mjs';
import {RECOVERABLE_TABLE} from './rt/recoverable_table.mjs';
import {tableAt} from './rt/line_profiles.mjs';

let n = 0;
const ok = (name, f) => { f(); n++; console.log('ok', name); };

ok('替え方の順番', () => {
  const o = fallbackOrder({type: 'steer_pulse', cls: '強', sign: -1}).map(c => `${comboKey(c.type, c.cls, c.sign, c.kind)}@${c.step}`);
  assert.deepEqual(o.slice(0, 5), ['steer_pulse:強:R@requested', 'steer_pulse:強:L@reverse', 'excursion:強:R@excursion', 'excursion:強:L@excursion_reverse', 'brake:強@brake']);
  assert.equal(o[5], 'steer_pulse:中:R@weaker1+requested');
  assert.equal(o.at(-1), 'brake:弱@weaker2+brake');
  const b = fallbackOrder({type: 'speed_pulse', kind: 'burst', cls: '中', sign: 1}).map(c => `${comboKey(c.type, c.cls, c.sign, c.kind)}@${c.step}`);
  assert.deepEqual(b.slice(0, 4), ['burst:中@requested', 'brake:中@reverse', 'excursion:中:L@excursion', 'excursion:中:R@excursion_reverse']);
  assert.ok(!b.includes('brake:中@brake'));   // 重なる組は 1 度だけ
  const w = fallbackOrder({type: 'excursion', cls: '中', sign: 1}, ['弱', '中']);
  assert.equal(w.at(-1).cls, '弱');
});

// 全部使える表(マスを全部 [10, 10])
const ALL = {...RECOVERABLE_TABLE, cells: new Proxy({}, {get: () => [10, 10]})};
function drive(p, frames = 600, ctxOf = f => ({s: 2 + .1 * f, ey: 0, dmin: .3, x: 0, z: 0, yaw: 0, v: 3, speedCap: 4.5, policy: {steering: 0, speed: 3}})) {
  const out = [];
  for (let f = 0; f < frames; f++) out.push(JSON.stringify(p.step(f, ctxOf(f))));
  return out;
}
ok('全部使える表・散らし 1 なら v1 と同じ', () => {
  let compared = 0;
  for (const spec of [RECORD_SPEC, TRAIN_SPEC]) for (let seed = 1; seed <= 40; seed++) {
    // 幅の広い所(わざと外すが入れ替わらない)だけを走る列: s を幅 0.6 m 以上の区間でゆっくり進める
    const ctxOf = f => ({s: 9 + .004 * f, ey: 0, dmin: .3, x: 0, z: 0, yaw: 0, v: 3, speedCap: 4.5, policy: {steering: 0, speed: 3}});
    const a = createPerturber({spec, seed}), b = createPerturberV2({spec, seed, table: ALL, spreadP: 1});
    const ra = drive(a, 600, ctxOf), rb = drive(b, 600, ctxOf);
    const ea = a.events.map(e => [e.type, e.cls, e.sign, e.ps, e.pe, e.frames, e.k ?? null, e.d ?? null, e.kind ?? null]);
    const eb = b.events.map(e => [e.type, e.cls, e.sign, e.ps, e.pe, e.frames, e.k ?? null, e.d ?? null, e.kind ?? null]);
    // v1 でわざと外すが入れ替わった・舵のパルスに替わった回は比べない(v2 は表の上で使えない組として替える)
    const flipped = a.events.some((e, i) => b.events[i] && e.type !== b.events[i].type);
    if (!flipped) { assert.deepEqual(eb, ea, `seed ${seed}`); assert.deepEqual(rb, ra); compared++; }
  }
  assert.ok(compared >= 20, '比べた回が少なすぎます ' + compared);
  console.log('  比べた回', compared);
});
ok('決定的', () => {
  const mk = () => createPerturberV2({spec: RECORD_SPEC, seed: 77, table: RECOVERABLE_TABLE});
  const ctxOf = f => ({s: (2 + .1 * f) % 26, ey: .05 * Math.sin(f / 20), dmin: .3, x: 0, z: 0, yaw: 0, v: 2 + Math.sin(f / 50), speedCap: 4.5, policy: {steering: 0, speed: 3}});
  assert.deepEqual(drive(mk(), 600, ctxOf), drive(mk(), 600, ctxOf));
});
const res = process.argv[2];
if (res) ok('当てた組は表の使えるマス', () => {
  const d = JSON.parse(fs.readFileSync(res, 'utf8'));
  let k = 0;
  for (const rows of Object.values(d.tests)) for (const r of rows) for (const e of r.events ?? []) {
    if (!e.req) continue;   // v1 の回
    const key = comboKey(e.type, e.cls, e.sign, e.kind);
    assert.ok(usable(RECOVERABLE_TABLE, e.s, e.v, key, .95), `${r.id} 出来事 ${e.n} ${key} s ${e.s} v ${e.v}`);
    if (e.type === 'excursion') {
      let m = Infinity; for (let q = 0; q <= 1.5 + 1e-9; q += .25) m = Math.min(m, e.sign > 0 ? tableAt('rhoL', e.s + q) : tableAt('rhoR', e.s + q));
      assert.ok(Number.isFinite(m));
    }
    k++;
  }
  console.log('  出来事', k);
});
console.log(`全部 ${n} 本合格`);
