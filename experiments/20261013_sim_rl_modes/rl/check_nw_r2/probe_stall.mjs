// 確認 r2: 標準の物理で止まったままになった開始(FN581)と、DR で止まった FD73・FD108 を、シールドあり / なしで走らせて中身を見る
import {runOne} from './run_nw_r2.mjs';
const cases = [
  {id: 'FN581', start: {gate: 17, left_offset_m: -0.19765207832679155, left_yaw_deg: 14.691235923673958, seed: 2400581}},
  {id: 'FD73', start: {gate: 23, left_offset_m: 0.03225647350773214, left_yaw_deg: 13.315470144152641, seed: 2400673}, dr: null},
];
for (const c of cases) for (const sh of [true, false]) {
  const r = await runOne({mode: 'no_wall', start: c.start, dr: c.dr ?? undefined, shield: sh, keepPath: true});
  const p = r.path;
  const pick = [0, 30, 60, 120, 200, 280, 300, 320, 330, 340, 400, 500, 599].filter(i => i < p.length).map(i => ({f: i, x: +p[i][0].toFixed(2), z: +p[i][1].toFixed(2), v: +p[i][2].toFixed(2), s: +p[i][3].toFixed(2), st: +p[i][5].toFixed(3), sp: +p[i][6].toFixed(2), sh: p[i][7]}));
  console.log(JSON.stringify({id: c.id, shield: sh, reason: r.reason, frames: r.frames, col: r.collisions, coll: r.collision, laps: r.valid_laps, min: r.min_clearance, minf: r.min_frame, kind: r.min_kind, sh: r.shield_frames, fb: r.labels.fb}));
  console.log(JSON.stringify(pick));
}
