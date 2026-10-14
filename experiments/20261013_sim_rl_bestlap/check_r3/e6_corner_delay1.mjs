#!/usr/bin/env node
// 確認 r3 の 6(e3 の箱の角で見つけた弱さの確かめ): r3 の段の「角の 20 マス」は遅延 0 だけで測った。
// 同じ 20 マス(舵の速さ 0.85〜1.1 × ブレーキ 0.8〜1.2、ほかの物理は標準)を、制御遅延 1 フレーム(アプリの固定遅延・学習済み ONNX の記録の条件。
// 学習では「標準でない回」の半分が遅延 1)で走らせる。観測の「前回の行動」は約束 2 通り(applied = RL の段の約束、fresh = 方策の出力)。
//   node check_r3/e6_corner_delay1.mjs <方策> [--every 1] [--prev applied|fresh] [--gain 1.01]
import {PLAN, loadPolicy, drive, save} from './common_r3.mjs';

const args = process.argv.slice(2), key = args[0];
const opt = (n, d) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const every = +opt('--every', 1), prev = opt('--prev', 'applied'), gain = +opt('--gain', 1);
const pol = loadPolicy(key), rows = PLAN.filter((_, i) => i % every === 0);
const out = {schema: 'rl-check-r3-corner-delay1/v1', created_at: new Date().toISOString(), policy: key, sha256: pol.sha, every, prev, speedGain: gain, n_starts: rows.length, cells: []};
for (const sr of [.85, .9, .95, 1.0, 1.1]) for (const br of [.8, .9, 1.0, 1.2]) {
  const o = {phys: {steerRate: sr, brakeDeceleration: br}, delay: 1, prev, speedGain: gain};
  const s = drive(pol, o).judge, c = drive(pol, {...o, continuous: true}).judge;
  let ok = 0; const cols = {};
  for (const row of rows) { const j = drive(pol, {...o, row, continuous: true}).judge; if (j.ok) ok++; else { const w = j.first_collision?.collider ?? j.reason; cols[w] = (cols[w] ?? 0) + 1; } }
  out.cells.push({sr, br, std_single: s.ok, std_single_lap: s.laps[0] ?? null, std_cont: c.ok, rec: ok, n: rows.length, fail_where: cols});
  console.log(key, prev, sr, br, s.ok ? s.laps[0].toFixed(3) : s.reason, c.ok ? 'C' : 'c', `${ok}/${rows.length}`, JSON.stringify(cols));
  save(`e6_corner_delay1_${key}_${prev}${gain !== 1 ? '_g' + gain : ''}${every !== 1 ? '_e' + every : ''}.json`, out);
}
out.total = out.cells.reduce((a, c) => a + c.rec, 0); out.total_n = out.cells.reduce((a, c) => a + c.n, 0);
out.cells_all = out.cells.filter(c => c.rec === c.n).length; out.cells_ge95pct = out.cells.filter(c => c.rec >= .95 * c.n).length;
save(`e6_corner_delay1_${key}_${prev}${gain !== 1 ? '_g' + gain : ''}${every !== 1 ? '_e' + every : ''}.json`, out);
console.log('total', key, out.total, '/', out.total_n, 'cells all', out.cells_all);
