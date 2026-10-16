#!/usr/bin/env node
// ブラウザ用の較正版のコアとページを作る(2026-10-16、シミュレーター較正 S2)。ユーザーのファイルは変えない。
//
//   node build_browser_core.mjs            写し(sim/)が無ければ作り、_calib/ と calib.html を(作り直して)書く
//
// 1) 10/15 のシミュレーターの写し(sim_diverse_train_20261015/sim)を、レポートのフォルダーの sim/ に写す(__pycache__ は除く)。
//    ユーザーの展開物(data/sim/releases/20261013-144013-6f610dd5)の dist の全ファイルと、ブリッジなどの 5 つが、
//    写しと sha256 で同じことを確かめる(違えば止まる)。
// 2) dist/_calib/sim-core-<名前>.js = patchSimCore(元の sim-core.js, P_cal)(静的なファイル。fast なし)
// 3) dist/calib.html = index.html と比べて、<title> と import map の 1 行(./sim-core.js の差し替え)だけが違うページ。
//    app.js だけでなく、先生のモジュール(speed-controller.js・rc-speed-controller.js・optimized-reference.js)も
//    同じ較正版のコアを読む(import map は './sim-core.js' に解決されるすべての import に効く)。
// 4) dist/_calib/MANIFEST.json に、元・プロファイル・生成物の sha256 を書く。
import fs from 'node:fs';
import path from 'node:path';
import {patchSimCore, loadProfile, sha256, sha256File, ORIGINAL_SIM_CORE, ORIGINAL_SIM_CORE_SHA256, REPORT, RUNTIME_PATH} from './calib_core.mjs';

export const RELEASE = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
export const COPY_FROM = '/home/jetson/jetformer_reports/sim_diverse_train_20261015/sim';
export const SIM_DIR = path.join(REPORT, 'sim');
export const DIST = path.join(SIM_DIR, 'dist');
const ROOT_FILES = ['bridge.py', 'jetformer_policy.py', 'dataset_recorder.py', 'jetformer_upload.py', 'calibration.example.json'];

function listFiles(dir, base = dir) {
  const out = [];
  for (const e of fs.readdirSync(dir, {withFileTypes: true})) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...listFiles(p, base)); else if (e.isFile()) out.push(path.relative(base, p));
  }
  return out.sort();
}

export function ensureCopy() {
  if (!fs.existsSync(SIM_DIR)) {
    fs.cpSync(COPY_FROM, SIM_DIR, {recursive: true, filter: src => !src.includes('__pycache__')});
  }
  // 確かめる: ユーザーの展開物の dist の全ファイルとブリッジなど
  const rel = listFiles(path.join(RELEASE, 'dist'));
  const bad = [];
  for (const f of rel) {
    const a = sha256File(path.join(RELEASE, 'dist', f)), cp = path.join(DIST, f);
    if (!fs.existsSync(cp) || sha256File(cp) !== a) bad.push(f);
  }
  for (const f of ROOT_FILES) if (sha256File(path.join(RELEASE, f)) !== sha256File(path.join(SIM_DIR, f))) bad.push(f);
  if (bad.length) throw new Error('build: 写しが元と違います: ' + bad.join(', '));
  return {release_dist_files: rel.length, root_files: ROOT_FILES.length, identical: true};
}

export function calibPageHtml(indexHtml, coreFile, title) {
  const lines = indexHtml.split('\n');
  const ti = lines.findIndex(l => l.startsWith('<title>'));
  const mi = lines.findIndex(l => l.startsWith('<script type="importmap">'));
  if (ti < 0 || mi < 0 || lines.filter(l => l.startsWith('<title>')).length !== 1 || lines.filter(l => l.startsWith('<script type="importmap">')).length !== 1)
    throw new Error('build: index.html の title / import map の行が 1 つではありません');
  const t = lines[ti].replace(/<title>[^<]*<\/title>/, `<title>${title}</title>`);
  if (t === lines[ti]) throw new Error('build: title を置き換えられません');
  const m = lines[mi].match(/^<script type="importmap">(.*)<\/script>$/);
  if (!m) throw new Error('build: import map を読めません');
  const map = JSON.parse(m[1]);
  if (map.imports['./sim-core.js']) throw new Error('build: index.html の import map に既に sim-core.js があります');
  map.imports['./sim-core.js'] = `./_calib/${coreFile}`;
  const out = [...lines];
  out[ti] = t;
  out[mi] = `<script type="importmap">${JSON.stringify(map)}</script>`;
  return out.join('\n');
}

export function build({profileKey = 'P_cal'} = {}) {
  const copy = ensureCopy();
  const {profile, sha256: psha} = loadProfile(profileKey);
  const source = fs.readFileSync(ORIGINAL_SIM_CORE, 'utf8');
  const name = profile.name;
  const coreFile = `sim-core-${name}.js`;
  const text = patchSimCore(source, profile, {fast: false, name, profileSha256: psha});
  fs.mkdirSync(path.join(DIST, '_calib'), {recursive: true});
  fs.writeFileSync(path.join(DIST, '_calib', coreFile), text);
  const index = fs.readFileSync(path.join(DIST, 'index.html'), 'utf8');
  const title = `Course Lab — AI走行評価(較正の物理 ${name})`;
  const html = calibPageHtml(index, coreFile, title);
  fs.writeFileSync(path.join(DIST, 'calib.html'), html);
  const manifest = {schema: 'jf-calib-browser-build/v1', created_at: new Date().toISOString(),
    original_sim_core: {path: ORIGINAL_SIM_CORE, sha256: ORIGINAL_SIM_CORE_SHA256},
    runtime: {path: RUNTIME_PATH, sha256: sha256File(RUNTIME_PATH)},
    profile: {key: profileKey, name, sha256: psha},
    files: {[`_calib/${coreFile}`]: sha256(text), 'calib.html': sha256(html), 'index.html': sha256File(path.join(DIST, 'index.html'))},
    copy, page_diff: 'index.html と比べて <title> と import map の 1 行(./sim-core.js → ./_calib/' + coreFile + ')だけ'};
  fs.writeFileSync(path.join(DIST, '_calib', 'MANIFEST.json'), JSON.stringify(manifest, null, 1));
  return manifest;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const m = build();
  console.log(JSON.stringify(m, null, 1));
}
