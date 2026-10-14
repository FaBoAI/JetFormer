// 離脱復活の確認 r2: 記録の決まりの開始(Halton 1〜・開始の衝突なし・種 703000 + i)の最初の 60 候補(2026-10-14)。mode.json の recording の数字の確かめ
import fs from 'node:fs';
import {startClearance} from '../check_rc/run_rc.mjs';
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const {haltonStart} = await import(`${EXP}/web/rl/rng.mjs`);
const modes = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/modes.json', 'utf8'));
const dist = modes.modes.find(m => m.key === 'recover').recording.start_distribution;
const J = [];
for (let h = 1, i = 0; J.length < 60; h++) {
  const st = haltonStart(h, dist);
  if (!(startClearance(st.gate, st.left_offset_m, st.left_yaw_deg) > 1e-9)) continue;
  J.push({set: 'REC', id: `R${i}`, spec: {mode: 'recover', start: {...st, seed: 703000 + i}}, halton: h}); i++;
}
fs.writeFileSync(process.argv[2], JSON.stringify(J));
console.log('jobs', J.length);
