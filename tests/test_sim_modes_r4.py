"""強化学習モードの第 4 回(r4・r4b)の昇格(2026-10-15)の試験。

  - ページの teacher.js は、登録簿の mode.json の runtime(RL.modes[key].runtime)を mode_runtime.mjs に渡す。
    経路いろいろ(line)はこれでシールド v5b が入り(無ければ入らない = catalogue のまま)、絶対壁にぶつからない(no_wall)は shield.mjs の既定(v5)、
    離脱復活(recover)は試験に使った jf-shield/v4-proto(shield_v4p.mjs)と RECOVER_R4 の値で固定(mode.json の値は使わない)。
  - 配るファイルの組(static/rl/manifest.json)に、r4 で足したファイルが入っていて sha256 が合う。
  - 登録簿の mode.json の ui_note_ja(手で走らせるときの注意)が画面用の dict に出る。
node が無ければ JS の試験は skip。ユーザーのシミュレーターの JS は実行しない(元の rc-speed-controller.js は小さな偽物)。
"""

import hashlib
import json
import shutil
import subprocess

import pytest

from jetformer import config
from jetformer.sim import paths, rl_modes

NODE = shutil.which('node')

STUB = r'''
export const RC_COLLECTION_PLAN = Object.freeze([{"maxspeed": 7.24, "controller": "demo", "teacher-profile": "speed-rc", "seed": 1}]);
export function assertRcTeacherContract(world, config, variation = {}) { return {...config, checked: true}; }
export function createRcTrainingTeacher(world, config, variation = {}) { return {marker: 'orig'}; }
'''

MAIN = r'''
import fs from 'node:fs';
const RL = globalThis.JETFORMER_RL = {mode: 'rc', weights: {}, modes: {}};
const T = await import('./_jf/rl/teacher.js');
const world = JSON.parse(fs.readFileSync('./world.json', 'utf8'));
function active(gate, off, yawDeg) {
  const route = [...world.route.slice(gate), ...world.route.slice(0, gate)], first = route[0], next = route[1];
  const heading = Math.atan2(-(next[1] - first[1]), next[0] - first[0]), turn = yawDeg * Math.PI / 180;
  return {...world, route, spawn: {x: first[0] + off * Math.sin(heading), z: first[1] - off * Math.cos(heading), yaw: heading + turn}};
}
function tiny(seed, dim) {
  let s = seed; const r = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296 - .5; };
  return {schema: 'rl-bestlap-policy/v1', obs_dim: dim, obs_mean: Array(dim).fill(0), obs_std: Array(dim).fill(1), obs_clip: 5, action_clip: 1,
    layers: [{n_in: dim, n_out: 4, act: 'tanh', w: Array.from({length: 4 * dim}, r), b: [0, 0, 0, 0]},
             {n_in: 4, n_out: 2, act: 'linear', w: Array.from({length: 8}, r), b: [0.1, 0.3]}]};
}
const cfg = (seed) => ({seed, maxSteer: .52, wheelbase: .26, vehicleLength: .42, vehicleWidth: .19, frictionMultiplier: 1,
  dt: 1 / 30, episodeMode: 'continuous-laps-v1', timeLimit: 20, steerRate: 1.2, brakeDeceleration: 1.2, acceleration: 1.2, maxSpeed: 7.24,
  driveTau: .4, driveAccelerationCap: 6, longitudinalModel: 'real-drive-response-estimate-v1', substep: .01});
const snap = (w, f) => ({state: {x: w.spawn.x, z: w.spawn.z, yaw: w.spawn.yaw, speed: 0, steering: 0, frame: f, sim_time: f / 30},
  metrics: {route_index: 0}, config: {}});
const w = active(5, .05, 4);
const run = (key, runtime) => {
  RL.mode = key;
  RL.modes[key] = {name: key, policy_sha256: 'ab'.repeat(32), planned_time_s: 9, ...(runtime === undefined ? {} : {runtime})};
  const t = T.createRcTrainingTeacher(w, cfg(21), {gate: 5, left_offset_m: .05, left_yaw_deg: 4});
  const a = t.action(snap(w, 0));
  return {shield: t.metadata.shield ?? null, perturbation: t.metadata.perturbation ?? null, jf: a.info.jf};
};
RL.weights.line = tiny(5, 97); RL.weights.no_wall = tiny(6, 79); RL.weights.recover = tiny(7, 79);
const LINE_RT = {shield: true, shield_params: {H: 24, c_nominal: .04, c_pessimistic: -1, identify: {enabled: false}, governor: {enabled: false},
  worst: {enabled: false}, select: 'v5b', stop: {enabled: true, E: true, Q: false, H_max: 240, v_stop: .02}}};
const out = {
  line_with: run('line', LINE_RT),
  line_without: run('line', undefined),
  no_wall: run('no_wall', {shield: true, shield_params: 'jf-shield/v3 の説明(文字列は使わない)'}),
  recover: run('recover', {shield: true, shield_params: {c_nominal: .99, unstick: {enabled: false}}}),
};
console.log(JSON.stringify(out));
'''


@pytest.fixture
def page(tmp_path):
    d = tmp_path / 'page'
    (d / '_jf').mkdir(parents=True)
    shutil.copytree(paths.STATIC / 'rl', d / '_jf' / 'rl')
    (d / 'package.json').write_text('{"type": "module"}')
    (d / 'rc-speed-controller.js').write_text(STUB)
    shutil.copy(paths.PKG_DATA / 'default-v3-physics-world.json', d / 'world.json')
    return d


@pytest.mark.skipif(NODE is None, reason='node がありません')
def test_teacher_passes_mode_runtime_and_each_mode_gets_its_tested_shield(page):
    (page / 'main.mjs').write_text(MAIN)
    r = subprocess.run([NODE, str(page / 'main.mjs')], capture_output=True, text=True, timeout=180, cwd=str(page))
    assert r.returncode == 0, r.stderr
    o = json.loads(r.stdout.strip().splitlines()[-1])
    # 経路いろいろ: mode.json の runtime があればシールド v5b(値は mode.json から)、無ければ catalogue のまま(シールドなし)
    lw = o['line_with']
    assert lw['shield']['version'] == 'jf-shield/v5' and lw['shield']['select'] == 'v5b' and lw['shield']['c_nominal'] == .04
    assert lw['jf']['sh'] in (0, 1) and lw['jf']['m'] == 'line'
    assert o['line_without']['shield'] is None and 'sh' not in o['line_without']['jf']
    # 絶対壁にぶつからない: shield.mjs の既定(v5、select 'v5'・c_nominal 0.081)。文字列の shield_params は使わない
    nw = o['no_wall']['shield']
    assert nw['version'] == 'jf-shield/v5' and nw['select'] == 'v5' and nw['c_nominal'] == .081 and nw['escape']['enabled'] is True
    # 離脱復活: 試験に使った jf-shield/v4-proto と RECOVER_R4 の値(mode.json の値は使わない)・乱し v2(立て直せる表 v2)
    rc = o['recover']
    assert rc['shield']['version'] == 'jf-shield/v4-proto' and rc['shield']['c_nominal'] == .035 and rc['shield']['unstick']['enabled'] is True
    assert rc['perturbation']['id'] == 'jf-perturb-record/v2' and rc['perturbation']['table_sha256'].startswith('8735365c')
    assert rc['jf']['p']['perturb_version'] == 'v2'


def test_static_rl_bundle_has_r4_files_with_matching_sha256():
    d = paths.STATIC / 'rl'
    man = json.loads((d / 'manifest.json').read_text())
    for name in ('mode_runtime.mjs', 'shield.mjs', 'shield_v4p.mjs', 'perturb_v2.mjs', 'recoverable_table.mjs'):
        assert name in man['files'], name
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == man['files'][name]['sha256']
    assert man['runtime_version'] == 'jf-sim-rl-mode-runtime/v1'


def test_load_mode_returns_ui_note(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path)
    d = rl_modes.modes_dir() / 'line'
    d.mkdir(parents=True)
    pol = json.dumps({'schema': 'rl-bestlap-policy/v1', 'obs_dim': 97, 'layers': []}).encode()
    (d / 'policy.json').write_bytes(pol)
    mode = {'schema': rl_modes.MODE_SCHEMA, 'key': 'line', 'name_ja': '強化学習モード(経路いろいろ)', 'order': 4, 'course': 'default',
            'runtime': {'shield': True, 'shield_params': {'select': 'v5b'}}, 'policy_sha256': hashlib.sha256(pol).hexdigest(),
            'ui_note_ja': '経路の形はシーンの種から決まります'}
    (d / 'mode.json').write_text(json.dumps(mode, ensure_ascii=False), encoding='utf-8')
    (d / 'status.json').write_text(json.dumps({'schema': rl_modes.STATUS_SCHEMA, 'key': 'line', 'state': 'accepted', 'failed': [], 'pending': []}))
    m = rl_modes.load_mode('line')
    assert m['selectable'] and m['ui_note'] == '経路の形はシーンの種から決まります'
    assert m['runtime']['shield'] is True and m['runtime']['shield_params']['select'] == 'v5b'


def test_collected_datasets_counts_superseded_separately(tmp_path, monkeypatch):
    """前の版の方策で記録したデータセット(カードに rl_superseded)は、集めたデータの数に入れず別に数える"""
    monkeypatch.setattr(config, 'DATASET_DIR', tmp_path / 'datasets')
    for name, sup in (('sim-rl-recover_a', False), ('sim-rl-recover_old', True)):
        d = config.DATASET_DIR / name
        d.mkdir(parents=True)
        card = {'name': name, 'scenes': 3, 'frames': 1800, 'tags': ['simulator', 'sim-rl', 'sim-rl-recover']}
        if sup:
            card['rl_superseded'] = {'note_ja': '前の版'}
        (d / 'dataset_card.json').write_text(json.dumps(card, ensure_ascii=False), encoding='utf-8')
    ds = rl_modes.collected_datasets()['recover']
    assert ds['count'] == 1 and ds['scenes'] == 3 and ds['names'] == ['sim-rl-recover_a']
    assert ds['superseded'] == {'count': 1, 'names': ['sim-rl-recover_old']}
