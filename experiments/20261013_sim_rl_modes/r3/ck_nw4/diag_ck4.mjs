// no_wall 確認 r4: 1 回をフレームごとに見る(崩れの調べ)。run_ck4.mjs の runOne と同じ作りで、シールドの内訳・速さ・すき間を出す。
// 使い方: node diag_ck4.mjs '<spec の JSON>' [フレームの始め] [終わり]
import {runOne} from './run_ck4.mjs';
import {coreH, makeStart, obstacles, clearOwn} from './own_ck4.mjs';
const spec = JSON.parse(process.argv[2]);
const f0 = +(process.argv[3] ?? 0), f1 = +(process.argv[4] ?? 9999);
// runOne の中の教師の log を見るため、ここでは同じ作りを短く書き直さず、runOne に keepPath と diag の印を渡す
spec.keepPath = true; spec.diag = true;
const r = await runOne(spec);
const {world} = makeStart(spec.start.gate, spec.start.off, spec.start.yaw);
const O = obstacles(world);
console.log(JSON.stringify({id: r.id, reason: r.reason, coll: r.collision, min_cm: +(100 * r.min_clearance).toFixed(3), min_frame: r.min_frame, kind: r.min_kind,
  stall: r.stall, stall_first: r.stall_first, valid_laps: r.valid_laps, escapes: r.escapes, rev: r.reverse_frames, fb: r.fb}));
for (let i = 0; i < r.path.length; i++) {
  const fr = i + 1;
  if (fr < f0 || fr > f1) continue;
  const [x, z, v, sh] = r.path[i];
  const d = r.diag?.[i];
  console.log(`f${fr} x=${x} z=${z} v=${v} sh=${sh} c=${(100 * clearOwn(O, x, z, d?.yaw ?? 0)).toFixed(2)}cm ${d ? JSON.stringify(d) : ''}`);
}
