"""強化学習モードのブラウザ側(jetformer/sim/static/rl/teacher.js・sim-rl.js)を node で試す(node がなければ skip)。

teacher.js は、元の ./rc-speed-controller.js(ここでは同じ名前の小さな偽物)を ../../rc-speed-controller.js として読み、
モードを選んでいなければ元の教師をそのまま返し、選んでいれば Node の試験と同じ mode_runtime.mjs で走ることを確かめる。
ユーザーのシミュレーターの JS は実行しない。観測には JetFormer が持つ Default(v3)の physics-world を使う。
"""

import json
import shutil
import subprocess

import pytest

from jetformer.sim import paths

NODE = shutil.which('node')
pytestmark = pytest.mark.skipif(NODE is None, reason='node がありません')

STUB = r'''
export const calls = [];
export const OTHER_EXPORT = 'そのまま';
export const RC_COLLECTION_PLAN = Object.freeze([{"maxspeed": 7.24, "controller": "demo", "teacher-profile": "speed-rc", "seed": 1, "name": "rc20-00"}]);
export function assertRcTeacherContract(world, config, variation = {}) {
  calls.push({route0: world.route[0], spawn: world.spawn, variation});
  if ((config.frictionMultiplier ?? 1) !== 1) throw Error('RC教師の物理設定が監査条件と違います: frictionMultiplier');
  return {...config, checked: true};
}
export function createRcTrainingTeacher(world, config, variation = {}) { assertRcTeacherContract(world, config, variation); return {marker: 'orig'}; }
'''

MAIN = r'''
import fs from 'node:fs';
const RL = globalThis.JETFORMER_RL = {mode: 'rc', weights: {}, modes: {}};
const T = await import('./_jf/rl/teacher.js');
const stub = await import('./rc-speed-controller.js');
const world = JSON.parse(fs.readFileSync('./world.json', 'utf8'));
// アプリの reset() と同じ式で、開始ゲートで回した world を作る
function active(gate, off, yawDeg) {
  const route = [...world.route.slice(gate), ...world.route.slice(0, gate)], first = route[0], next = route[1];
  const heading = Math.atan2(-(next[1] - first[1]), next[0] - first[0]), turn = yawDeg * Math.PI / 180;
  return {...world, route, spawn: {x: first[0] + off * Math.sin(heading), z: first[1] - off * Math.cos(heading), yaw: heading + turn}};
}
function tiny(seed) {
  let s = seed; const r = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296 - .5; };
  return {schema: 'rl-bestlap-policy/v1', obs_dim: 79, obs_mean: Array(79).fill(0), obs_std: Array(79).fill(1), obs_clip: 5, action_clip: 1,
    layers: [{n_in: 79, n_out: 4, act: 'tanh', w: Array.from({length: 316}, r), b: [0, 0, 0, 0]},
             {n_in: 4, n_out: 2, act: 'linear', w: Array.from({length: 8}, r), b: [0.1, 0.5]}]};
}
const cfg = (seed, extra = {}) => ({seed, maxSteer: .52, wheelbase: .26, vehicleLength: .42, vehicleWidth: .19, frictionMultiplier: 1,
  dt: 1 / 30, episodeMode: 'continuous-laps-v1', timeLimit: 20, ...extra});
const snap = (w, f) => ({state: {x: w.spawn.x, z: w.spawn.z, yaw: w.spawn.yaw, speed: 0, steering: 0, frame: f, sim_time: f / 30},
  metrics: {route_index: 0}, config: {}});
const err = (fn) => { try { fn(); return null; } catch (e) { return e.message; } };
const out = {};
// 1. モードを選んでいない → 元の教師・元の名前はそのまま
out.delegate = T.createRcTrainingTeacher(active(0, 0, 0), cfg(1), {gate: 0}).marker;
out.reexport = T.OTHER_EXPORT; out.planLen = T.RC_COLLECTION_PLAN.length; out.basePlan = RL.basePlan;
out.teacherModule = RL.teacherModule;
// 2. 選んでいるが重みがまだ → 日本語のエラー
RL.mode = 'fastest';
out.noWeights = err(() => T.createRcTrainingTeacher(active(0, 0, 0), cfg(1), {gate: 0}));
RL.weights.fastest = tiny(7); RL.modes.fastest = {name: '最速', policy_sha256: 'ab'.repeat(32), planned_time_s: 8.09};
const w5 = active(5, .1, 7);
stub.calls.length = 0;
const t = T.createRcTrainingTeacher(w5, cfg(11), {gate: 5, left_offset_m: .1, left_yaw_deg: 7});
out.contractCall = stub.calls[0];
out.canonical0 = world.route[0];
const a = t.action(snap(w5, 0)), b = t.action(snap(w5, 0));
out.same = JSON.stringify(a) === JSON.stringify(b);
out.a = a; out.metadata = t.metadata; out.reference = t.reference; out.options = t.options;
out.skip = err(() => t.action(snap(w5, 2)));
out.live = RL.live;
// 種が違っても metadata(context.teacher)は同じ。シーンの条件は info の p にだけ入る
const t2 = T.createRcTrainingTeacher(active(9, -.2, -12), cfg(12), {gate: 9, left_offset_m: -.2, left_yaw_deg: -12});
const a2 = t2.action(snap(active(9, -.2, -12), 0));
out.metaSame = JSON.stringify(t2.metadata) === JSON.stringify(t.metadata);
out.p2 = a2.info.jf.p;
// 3. 検査: 連続試験の計画の検査(回していない world・ゲートは別の番号)は通す。範囲の外・低摩擦・回し方の違い・途中状態は止める
out.planCheck = err(() => T.assertRcTeacherContract(active(0, .1, 7), {...cfg(1), seed: undefined}, {gate: 5, left_offset_m: .1, left_yaw_deg: 7}));
out.wide = err(() => T.assertRcTeacherContract(active(0, .35, 0), cfg(1), {gate: 0, left_offset_m: .35, left_yaw_deg: 0}));
out.lowmu = err(() => T.assertRcTeacherContract(active(0, 0, 0), cfg(1, {frictionMultiplier: .4}), {gate: 0}));
out.rot = err(() => T.assertRcTeacherContract(active(5, 0, 0), cfg(1), {gate: 3}));
out.spawn = err(() => T.assertRcTeacherContract(active(5, 0, 0), cfg(1), {gate: 5, left_offset_m: .1}));
out.init = err(() => T.assertRcTeacherContract(active(0, 0, 0), cfg(1), {gate: 0, initialState: {}}));
out.badRoute = err(() => T.assertRcTeacherContract({...active(0, 0, 0), route: world.route.slice(1)}, cfg(1), {gate: 0}));
// 4. 速さいろいろ: シーンの条件(プランの指定)は p に入り、速さの上限 4.5·c_v を守る
RL.mode = 'speed'; RL.modes.speed = {name: '速さ', policy_sha256: 'cd'.repeat(32)};
const t3w = tiny(3); t3w.obs_dim = 97; t3w.obs_mean = Array(97).fill(0); t3w.obs_std = Array(97).fill(1); t3w.layers[0].n_in = 97;
t3w.layers[0].w = Array.from({length: 388}, (_, i) => Math.sin(i)); RL.weights.speed = t3w;
RL.sceneParams = {mode: 'speed', params: {c_v: .5}};
const t3 = T.createRcTrainingTeacher(w5, cfg(13), {gate: 5, left_offset_m: .1, left_yaw_deg: 7});
const a3 = t3.action(snap(w5, 0));
out.speed = {p: a3.info.jf.p, speed: a3.speed, meta_cap: t3.metadata.speed_cap_mps};
// 5. 'rc' に戻すと元の教師
RL.mode = 'rc';
out.back = T.createRcTrainingTeacher(active(0, 0, 0), cfg(1), {gate: 0}).marker;
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


def run_node(d, name, script):
    (d / name).write_text(script)
    r = subprocess.run([NODE, str(d / name)], capture_output=True, text=True, timeout=120, cwd=str(d))
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_teacher_delegates_and_runs_rl_modes(page):
    o = run_node(page, 'main.mjs', MAIN)
    assert o['delegate'] == 'orig' and o['back'] == 'orig'                        # 選んでいなければ今までと同じ教師
    assert o['reexport'] == 'そのまま' and o['planLen'] == 1 and o['basePlan']['maxspeed'] == 7.24
    assert o['teacherModule']['loaded'] and 'fastest' in o['teacherModule']['modes']
    assert '読み込まれていません' in o['noWeights']
    # 元の検査には「ゲート 0 に回し戻した world」(経路の先頭 = 標準の先頭・開始 = ゲート 0・ずれ 0)を渡す
    assert o['contractCall']['route0'] == o['canonical0'] and o['contractCall']['variation'] == {'gate': 0, 'left_offset_m': 0, 'left_yaw_deg': 0}
    a = o['a']
    jf = a['info']['jf']
    assert o['same'] and jf['v'] == 1 and jf['m'] == 'fastest' and jf['f'] == 0 and jf['l'] == 1 and jf['ph'] == 'drive'
    assert jf['p']['seed'] == 11 and jf['p']['policy'] == 'ab' * 6 and jf['p']['runtime'] == 'jf-sim-rl-mode-runtime/v1'
    assert set(a) == {'steering', 'speed', 'info'} and abs(a['steering']) <= .52 + 1e-12 and 0 <= a['speed'] <= 4.5
    assert 'steering_pwm' not in json.dumps(a) and 'throttle_pwm' not in json.dumps(a)
    md = o['metadata']
    assert md['schema'] == 'jetformer-sim-rl-teacher/v1' and md['id'] == 'jf-rl-fastest-v1' and md['policy_sha256'] == 'ab' * 32
    assert md['episode_mode'] == 'continuous-laps-v1' and md['teacher_slot'] == 'jf-sim-rl-teacher-slot/v1'
    assert o['metaSame'] and o['p2']['seed'] == 12                                 # context.teacher はシーンの間で同じ
    assert o['reference']['planned_time_s'] == 8.09 and o['options'] == {'mode': 'fastest'}
    assert 'フレームが飛びました' in o['skip']
    assert o['live']['mode'] == 'fastest' and o['live']['frame'] == 0
    assert o['planCheck'] is None                                                  # 連続試験の計画の検査(回していない world)
    assert '±0.3 m' in o['wide'] and '低摩擦' in o['lowmu'] and '回し方' in o['rot'] and '開始姿勢' in o['spawn']
    assert 'initialState' in o['init'] and 'Default' in o['badRoute']
    sp = o['speed']
    assert sp['p']['c_v'] == .5 and sp['speed'] <= 4.5 * .5 + 1e-12 and sp['meta_cap'] == '4.5·c_v'


SIMRL_SMOKE = r'''
import fs from 'node:fs';
import vm from 'node:vm';
const code = fs.readFileSync(process.argv[2], 'utf8');
const out = {};
for (const enabled of [false, true]) {
  const warns = [];
  const ctx = {console: {warn: (...a) => warns.push(a.join(' ')), log() {}, info() {}}, setTimeout, setInterval: () => 0, URLSearchParams,
    location: {search: '?rl=fastest'}, fetch: async () => { throw new Error('通信しない'); },
    document: {readyState: 'complete', getElementById: () => null, addEventListener() {}}};
  ctx.window = ctx;
  ctx.JETFORMER_SIM = {rl: {enabled, course_ok: true}};
  vm.createContext(ctx);
  vm.runInContext(code, ctx);
  await new Promise((r) => setTimeout(r, 50));
  out[enabled] = {rl: Object.keys(ctx.JETFORMER_RL).sort(), ui: !!ctx.JETFORMER_RL.ui, warns};
}
console.log(JSON.stringify(out));
'''


def test_sim_rl_page_script_is_safe_without_the_app(tmp_path):
    """sim-rl.js: 無効なら何もしない。有効でもページの要素が無ければ欄を出さずに止まる(例外を出さない)"""
    (tmp_path / 'smoke.mjs').write_text(SIMRL_SMOKE)
    r = subprocess.run([NODE, str(tmp_path / 'smoke.mjs'), str(paths.STATIC / 'sim-rl.js')], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    o = json.loads(r.stdout.strip().splitlines()[-1])
    assert o['false']['ui'] is False and 'mode' in o['false']['rl']
    assert o['true']['ui'] is True and any('teacher-profile' in w for w in o['true']['warns'])
