// 経路いろいろ(line r2 候補)の一定の寄せ c0 を変えて、記録と同じ範囲の開始(ゲート 0・±3 cm・±3°、10 本。E0〜E5 とは別)で単周のすき間を測る
import fs from 'node:fs';
const M = await import('./measure_teachers.mjs');
const out = [];
for (const c0 of [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]) {
  M.TEACHERS['sweep'] = {mode: 'line', weights: '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/candidates/line_r2/policy.json', params: {line: {family: 'const', c0}, c_v: 1.0}};
  const rows = [];
  for (let i = 0; i < 10; i++) {
    const off = ((i * 0.6180339887) % 1 * 2 - 1) * .03, yaw = ((i * 0.7548776662 + .3) % 1 * 2 - 1) * 3;
    const r = M.runEpisode({teacher: 'sweep', start: [0, off, yaw], episodeMode: 'single', timeLimit: 15, seed: 100 + i});
    let fsMin = 9; for (const q of r.rows) { if (q.x > 2.6) break; fsMin = Math.min(fsMin, q.dmin); }
    rows.push({lap: r.reason === 'route_complete' && !r.collisions ? r.elapsed : null, min: Math.min(...r.rows.map(q => q.dmin)), fs: fsMin});
  }
  const ok = rows.filter(r => r.lap);
  const s = {c0, ok: ok.length, lap_mean: ok.reduce((a, r) => a + r.lap, 0) / Math.max(1, ok.length), min_worst: Math.min(...rows.map(r => r.min)), fs_worst: Math.min(...rows.map(r => r.fs))};
  out.push(s); console.log(JSON.stringify(s));
}
fs.writeFileSync(process.argv[2], JSON.stringify(out, null, 1));
