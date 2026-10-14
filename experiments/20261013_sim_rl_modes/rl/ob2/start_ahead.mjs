#!/usr/bin/env node
// 障害物よけ r2 の下調べ(判定に使わない): 選びの組・試験の組の各回で、開始から最初の障害物までの R 上の距離(障害物の s_ref − 開始の s)を出す。2026-10-14
//   node rl/ob2/start_ahead.mjs <eval_ob2 か eval_modes の結果.json> [selection|test]
import fs from 'node:fs';
import {buildJobs as buildSel} from './eval_ob2.mjs';
import {buildJobs as buildProt} from '../eval_modes.mjs';
import {BASE_WORLD} from '../episode.mjs';
import {activeWorldFor} from '../../snapshot/rl/env_core.mjs';
import {projectWindow, LOOP_LENGTH} from '../runtime/features.mjs';
const [resPath, set = 'selection'] = process.argv.slice(2);
const res = JSON.parse(fs.readFileSync(resPath, 'utf8'));
const rows = new Map((res.rows ?? Object.values(res.tests ?? {}).flat()).map(r => [r.id, r]));
const jobs = (set === 'selection' ? buildSel('selection') : buildProt('obstacle', 'test')).filter(j => j.spec.layout);
const bins = [[1.5, 2.5], [2.5, 3.5], [3.5, 5], [5, 99]], out = {};
for (const j of jobs) {
  const st = j.spec.start, {world} = activeWorldFor(BASE_WORLD, st.gate, st.left_offset_m, st.left_yaw_deg);
  const s0 = projectWindow(world.spawn.x, world.spawn.z, 0, LOOP_LENGTH).s;
  const ahead = Math.min(...j.spec.layout.obstacles.map(o => ((o.s_ref - s0) % LOOP_LENGTH + LOOP_LENGTH) % LOOP_LENGTH));
  const b = bins.find(([a, c]) => ahead >= a && ahead < c) ?? ['<1.5'];
  const k = `${j.id.endsWith('-STD') ? 'STD' : 'H'} ${b[0]}-${b[1] ?? ''}`, r = rows.get(j.id);
  out[k] ??= {n: 0, valid: 0}; out[k].n++; if (r?.valid_episode) out[k].valid++;
}
console.log(JSON.stringify(Object.fromEntries(Object.entries(out).sort())));
