// 評価に足す開始 E6〜E11 が正しい開始(invalid_spawn でない)で、先生(最速・余裕の線)が完走できるかを Node で確かめる
import fs from 'node:fs';
const M = await import('./measure_teachers.mjs');
const NEW = {E6: [0, .05, 2], E7: [0, -.05, -2], E8: [0, .02, -4], E9: [0, -.02, 4], E10: [10, 0, 0], E11: [28, 0, 0]};
const out = [];
for (const teacher of ['fastest', 'line_l3', 'line_c0', 'no_wall']) for (const [k, st] of Object.entries({...M.STARTS, ...NEW})) {
  const r = M.runEpisode({teacher, start: st, episodeMode: 'single', timeLimit: 15});
  if (r.invalid_spawn) { out.push({teacher, k, invalid_spawn: true}); console.log(teacher, k, 'INVALID'); continue; }
  const lap = r.reason === 'route_complete' && !r.collisions ? +r.elapsed.toFixed(3) : null;
  let fsMin = null; if (st[0] === 0) { fsMin = 9; for (const q of r.rows) { if (q.x > 2.6) break; fsMin = Math.min(fsMin, q.dmin); } }
  out.push({teacher, k, start: st, reason: r.reason, lap, min_clearance: Math.min(...r.rows.map(q => q.dmin)), first_straight: fsMin});
  console.log(teacher, k, r.reason, lap, Math.min(...r.rows.map(q => q.dmin)).toFixed(3), fsMin?.toFixed(3));
}
fs.writeFileSync(process.argv[2], JSON.stringify(out, null, 1));
