// 離脱復活 r3 の確認: ブラウザの閉ループと Node の予想の食い違いを数えるための開始の一覧(FP の最初の n、2026-10-14 21:40)
//   node ck_cl_specs.mjs <jobs.json> <n> <出力.json>
import fs from 'node:fs';
import {BASE_WORLD, activeWorld, NOMINAL} from './ck_run.mjs';
const [jobsPath, n, outPath] = process.argv.slice(2);
const J = JSON.parse(fs.readFileSync(jobsPath, 'utf8')).filter(j => j.set === 'FP').slice(0, +n);
const out = J.map(j => {
  const st = j.spec.start, {world, variation} = activeWorld(BASE_WORLD, st.gate, st.left_offset_m, st.left_yaw_deg);
  return {id: j.id, world, variation, config: {...NOMINAL, seed: st.seed, timeLimit: 20, episodeMode: 'continuous-laps-v1'}, params: j.spec.params ?? {}};
});
fs.writeFileSync(outPath, JSON.stringify(out));
console.log(out.length);
