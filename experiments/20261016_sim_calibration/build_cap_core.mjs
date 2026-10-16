#!/usr/bin/env node
// 上限の層つきの較正版のコア(ブラウザ用)と calib_cap.html を作る(2026-10-16、シミュレーター較正 S4/S5)。ユーザーのファイルは変えない。
//
//   node build_cap_core.mjs
//
//   - patchCapCore(元の sim-core.js, P_cal, 上限の設定) = patchSimCore(…) に、さらに 3 か所の差分を当てたもの:
//       (1) 先頭に cap_runtime.mjs(export を外して 1 つの関数で包む)・参照ループ R・上限の設定を埋め込む
//       (2) シミュレーションを作るときに上限の層のフィルター(createCapFilter、位置の区間)を作り、calibControl({filter}) の口に差す
//       (3) reset で、較正の実行部を戻したあと(フィルターも外れる)にフィルターを戻して差し直す
//     上限の層の計算は Node の上限の試験(cap_test.mjs)と同じ関数(cap_runtime.mjs の createCapFilter + calib_runtime.mjs の governorStep)。
//     参照ループ R の route を回したものでない world(別のコース)では、上限の層を付けずに較正の物理だけで動く(console.warn)。
//   - dist/_calib/sim-core-<名前>-cap-<版>.js(静的なファイル。fast なし)
//   - dist/calib_cap.html = index.html と比べて <title> と import map の 1 行だけが違うページ(calib.html と同じ作り方)
//   - dist/_calib/MANIFEST_cap.json
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {patchSimCore, loadProfile, replaceOnce, sha256, sha256File, ORIGINAL_SIM_CORE, ORIGINAL_SIM_CORE_SHA256, REPORT, RUNTIME_PATH} from './calib_core.mjs';
import {calibPageHtml, ensureCopy, DIST} from './build_browser_core.mjs';
import {GOVERNOR_V1_SIM} from './calib_runtime.mjs';
import {REFERENCE_LOOP_JSON} from './pp_driver.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
export const CAP_RUNTIME_PATH = path.join(here, 'cap_runtime.mjs');
export const CAP_SPEC_O25 = Object.freeze({name: 'cap-v1.1-O25', zoneMode: 'oracle', params: {...GOVERNOR_V1_SIM}});

export function capRuntimeBlock(text) {
  const body = text.replace(/^export (const|function) /gm, '$1 ');
  if (/^\s*import\s/m.test(body) || /\bexport\b/.test(body.replace(/\/\/.*$/gm, '')))
    throw new Error('cap-core: cap_runtime.mjs に import / export が残っています');
  return `const __capRuntime = (() => {\n${body}\nreturn { createLoop, createCapFilter, CAP_RUNTIME_VERSION };\n})();\n`;
}

export function loopData() {
  const rl = JSON.parse(fs.readFileSync(REFERENCE_LOOP_JSON, 'utf8'));
  return {n: rl.n, length_m: rl.length_m, x: rl.x, z: rl.z, s: rl.s, route: rl.route, gate_s: rl.gate_s};
}

export function patchCapCore(source, profile, spec = CAP_SPEC_O25, options = {}) {
  let p = patchSimCore(source, profile, {...options, fast: false});
  const r = (a, b) => { p = replaceOnce(p, a, b, 'cap-core'); };
  // (2) 作るとき
  r('  const __CAL = __CAL_PROFILE ? __calib.createCalibRuntime(__CAL_PROFILE, config, __CAL_META) : null;\n',
    `  const __CAL = __CAL_PROFILE ? __calib.createCalibRuntime(__CAL_PROFILE, config, __CAL_META) : null;
  let __CAP = null;
  if (__CAL && __CAP_SPEC) {
    try {
      __CAP = __capRuntime.createCapFilter({loop: __capRuntime.createLoop(__CAP_LOOP), route: world.route, governorStep: __calib.governorStep,
        governorInitialState: __calib.governorInitialState, params: __CAP_SPEC.params, zoneMode: __CAP_SPEC.zoneMode, name: __CAP_SPEC.name, record: false});
    } catch (e) { __CAP = null; console.warn('上限の層を付けません(参照ループ R のコースではありません):', e.message); }
    if (__CAP) __CAL.control({filter: __CAP.filter, filterName: __CAP.name});
  }
`);
  // (3) reset のあと
  r('    if (__CAL) __CAL.reset();', '    if (__CAL) __CAL.reset();\n    if (__CAP) { __CAP.reset(); __CAL.control({filter: __CAP.filter, filterName: __CAP.name}); }');
  // (1) 先頭に埋め込む
  const capText = options.capRuntimeText ?? fs.readFileSync(CAP_RUNTIME_PATH, 'utf8');
  const head = `// 上限の層つき(${spec.name})。生成: experiments/20261016_sim_calibration/build_cap_core.mjs patchCapCore。このファイルを手で直さない。\n` +
    capRuntimeBlock(capText) +
    `const __CAP_SPEC = ${JSON.stringify(spec)};\n` +
    `const __CAP_LOOP = ${JSON.stringify(loopData())};\n`;
  const marker = 'const __CAL_PROFILE = ';
  const i = p.indexOf(marker);
  if (i < 0 || p.indexOf(marker, i + 1) >= 0) throw new Error('cap-core: 埋め込む場所が 1 つではありません');
  return p.slice(0, i) + head + p.slice(i);
}

export function buildCap({profileKey = 'P_cal', spec = CAP_SPEC_O25} = {}) {
  const copy = ensureCopy();
  const {profile, sha256: psha} = loadProfile(profileKey);
  const source = fs.readFileSync(ORIGINAL_SIM_CORE, 'utf8');
  const name = profile.name;
  const coreFile = `sim-core-${name}-${spec.name}.js`;
  const text = patchCapCore(source, profile, spec, {name, profileSha256: psha});
  fs.mkdirSync(path.join(DIST, '_calib'), {recursive: true});
  fs.writeFileSync(path.join(DIST, '_calib', coreFile), text);
  const index = fs.readFileSync(path.join(DIST, 'index.html'), 'utf8');
  const title = `Course Lab — AI走行評価(較正の物理 ${name} + 区間の速度上限 ${spec.name})`;
  const html = calibPageHtml(index, coreFile, title);
  fs.writeFileSync(path.join(DIST, 'calib_cap.html'), html);
  const manifest = {schema: 'jf-calib-browser-build-cap/v1', created_at: new Date().toISOString(),
    original_sim_core: {path: ORIGINAL_SIM_CORE, sha256: ORIGINAL_SIM_CORE_SHA256},
    runtime: {path: RUNTIME_PATH, sha256: sha256File(RUNTIME_PATH)}, cap_runtime: {path: CAP_RUNTIME_PATH, sha256: sha256File(CAP_RUNTIME_PATH)},
    reference_loop: {path: REFERENCE_LOOP_JSON, sha256: sha256File(REFERENCE_LOOP_JSON)},
    profile: {key: profileKey, name, sha256: psha}, cap_spec: spec,
    files: {[`_calib/${coreFile}`]: sha256(text), 'calib_cap.html': sha256(html), 'index.html': sha256File(path.join(DIST, 'index.html'))},
    copy, page_diff: 'index.html と比べて <title> と import map の 1 行(./sim-core.js → ./_calib/' + coreFile + ')だけ'};
  fs.writeFileSync(path.join(DIST, '_calib', 'MANIFEST_cap.json'), JSON.stringify(manifest, null, 1));
  return manifest;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const m = buildCap();
  console.log(JSON.stringify(m, null, 1));
}
