// 確認の段(review): ユーザーの qa/verify-learned-lap.mjs の verifyLearnedReport を RL の単周のレポートに当て、
// 失敗の項目を「周の項目」(完走・衝突・ゲート・黒シート・物理の再生)と「ONNX・開始・センサーの約束の項目」(RL の周には当てはまらない)に分ける
//   node user_verifier_lap_items.mjs <出力 JSON> <レポート.json> [...]
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const USER_SIM = '/home/jetson/sim/extracted/course-simulator';
const {verifyLearnedReport} = await import(pathToFileURL(USER_SIM + '/qa/verify-learned-lap.mjs').href);
const {loadCourseWorld} = await import(pathToFileURL(USER_SIM + '/qa/load-course-world.mjs').href);
const {WORLD_VERSION} = await import(pathToFileURL(USER_SIM + '/dist/world.js').href);
const world = await loadCourseWorld();
const LAP = [/route_complete/, /^Collision/, /Finished flag/, /ordered gates/, /mandatory crossing/i, /^replay\./, /Replay error/, /Trace\/actions/, /Missing saved controls/, /stopOnCollision/, /benchmark physics/];
const [out, ...files] = process.argv.slice(2);
const res = {};
for (const f of files) {
  const rep = JSON.parse(fs.readFileSync(f, 'utf8'));
  rep.context = {...(rep.context ?? {}), world_version: WORLD_VERSION};   // RL のレポートには世界の版の欄が無いので、今のコースの版を書く(周の判定には効かない)
  const r = verifyLearnedReport(rep, {expectedWorld: world, expectedWorldVersion: WORLD_VERSION, expectedSha256: 'f'.repeat(64)});
  const lapFailures = r.failures.filter(m => LAP.some(re => re.test(m)));
  res[f.split('/').pop()] = {lap_items_pass: lapFailures.length === 0, lap_failures: lapFailures, replay: r.replay ?? null,
    other_failures_not_applicable_to_rl: r.failure_count - lapFailures.length, examples_other: r.failures.filter(m => !lapFailures.includes(m)).slice(0, 4)};
  console.log(f.split('/').pop(), 'lap_items_pass', lapFailures.length === 0, lapFailures.slice(0, 3), 'replay', JSON.stringify(r.replay?.outcome), r.replay?.elapsed);
}
fs.writeFileSync(out, JSON.stringify(res, null, 1));
