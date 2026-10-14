// 離脱復活 r4 の確認: ブラウザの閉ループと Node の予想の食い違いを数えるための開始の一覧(2026-10-15。確認 r3 の ck_cl_specs.mjs の写し)
//   node ck_cl_specs_r4.mjs <jobs.json> <組> <from> <n> <出力.json>
import fs from 'node:fs';
import {BASE_WORLD, activeWorld, NOMINAL} from './ck_run.mjs';
const [jobsPath, set, from, n, outPath] = process.argv.slice(2);
const J = JSON.parse(fs.readFileSync(jobsPath, 'utf8')).filter(j => j.set === set).slice(+from, +from + +n);
const out = J.map(j => {
  const st = j.spec.start, {world, variation} = activeWorld(BASE_WORLD, st.gate, st.left_offset_m, st.left_yaw_deg);
  return {id: j.id, world, variation, config: {...NOMINAL, seed: st.seed, timeLimit: 20, episodeMode: 'continuous-laps-v1'}, params: j.spec.params ?? {},
    overrides: j.spec.perturb === false ? {perturbation: false} : {}};
});
fs.writeFileSync(outPath, JSON.stringify(out));
console.log(out.length);
