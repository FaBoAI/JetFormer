"""JetFormer が持つシミュレーター用の JS(汎用描画・shim)を node で試す(node がなければ skip)。
ユーザーのシミュレーターの JS は実行しない"""

import json
import shutil
import subprocess

import pytest

import _sim_testlib as L
from jetformer.sim import courses, paths, worldhash

NODE = shutil.which('node')
pytestmark = pytest.mark.skipif(NODE is None, reason='node がありません')

THREE_STUB = r'''
export const SRGBColorSpace = 'srgb', RepeatWrapping = 1000, DoubleSide = 2;
export const calls = {textures: [], meshes: 0};
class Obj { constructor() { this.position = {set: (...a) => { this.pos = a; }}; this.rotation = {}; this.children = []; }
  add(o) { this.children.push(o); } }
export class Group extends Obj {}
export class Mesh extends Obj { constructor(g, m) { super(); this.geometry = g; this.material = m; calls.meshes++; } }
export class BoxGeometry { constructor(...a) { this.args = a; } }
export class CylinderGeometry { constructor(...a) { this.args = a; } }
export class PlaneGeometry { constructor(...a) { this.args = a; } }
export class MeshStandardMaterial { constructor(o) { this.o = o; this.normalScale = {set() {}}; } }
export class TextureLoader { load(url, cb) { calls.textures.push(url); return {}; } }
'''


def run_node(tmp_path, script):
    (tmp_path / 'main.mjs').write_text(script)
    r = subprocess.run([NODE, str(tmp_path / 'main.mjs')], capture_output=True, text=True, timeout=60, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    return r.stdout


@pytest.fixture
def course_dir(tmp_path, monkeypatch):
    L.install_fake_release(tmp_path, monkeypatch)
    p = tmp_path / 'c.zip'
    p.write_bytes(L.course_zip({'course.json': {'format': 'jetformer-sim-course/v1', 'name': 'テスト', 'world_version': 'js-1',
                                                'visual': {'floor_texture': 'turf.jpg'}},
                                'physics-world.json': L.simple_world()}))
    c = courses.import_package(p, 'c.zip')['course']
    d = tmp_path / 'page'
    (d / '_jf').mkdir(parents=True)
    (d / 'package.json').write_text('{"type": "module"}')
    shutil.copy(paths.STATIC / 'course-scene.js', d / 'scene.js')
    shutil.copy(paths.STATIC / 'course-world.js', d / 'world.js')
    (d / '_jf' / 'course-data.js').write_text(courses.course_data_module(c['id']))
    nm = d / 'node_modules' / 'three'
    nm.mkdir(parents=True)
    (nm / 'package.json').write_text('{"name": "three", "type": "module", "main": "index.js"}')
    (nm / 'index.js').write_text(THREE_STUB)
    return d, c


def test_generic_course_world_hash_matches_registration(course_dir):
    course_dir, course = course_dir
    out = run_node(course_dir, '''
import * as THREE from 'three';
import { buildCourse } from './scene.js';
import { makeWorld, WORLD_VERSION } from './world.js';
const scene = new THREE.Group();
const model = buildCourse(scene);
const world = makeWorld(model.physics);
console.log(JSON.stringify({world, version: WORLD_VERSION, textures: THREE.calls.textures, meshes: THREE.calls.meshes,
  keys: Object.keys(model)}));
''')
    r = json.loads(out)
    assert r['version'] == 'js-1'
    assert worldhash.world_hash(r['world']) == course['world_hash']     # ブラウザの report().world_hash と同じになる
    assert './assets/turf.jpg' in r['textures']                                     # 1 枚以上読む(assetsReady)
    w = L.simple_world()
    assert r['meshes'] >= len(w['segments']) + len(w['disks']) + len(w['surfaces']) + 1
    assert 'physics' in r['keys']


def test_shim_rewrites_only_bridge_urls(tmp_path):
    out = run_node(tmp_path, '''
import fs from 'fs';
import vm from 'vm';
const store = {};
class Storage { getItem(k) { return k in store ? store[k] : null; } setItem(k, v) { store[k] = String(v); } removeItem(k) { delete store[k]; } }
const ls = new Storage();
const opened = [];
class XMLHttpRequest { open(m, u) { opened.push(u); } }
const fetched = [];
const ctx = { console, URL, URLSearchParams, setTimeout, setInterval, Storage, localStorage: ls, XMLHttpRequest,
  fetch: (u) => { fetched.push(String(u)); return Promise.resolve({ ok: false, json: async () => ({}) }); },
  location: { origin: 'http://192.168.5.14:8080', search: '', pathname: '/sim/c/x/', hash: '' },
  document: { readyState: 'loading', addEventListener() {}, getElementById() { return null; } },
  JETFORMER_SIM: { course: { id: 'x', world_version: 'wv-1' }, bridge: '/sim/bridge', jetformer_port: 8080 } };
ctx.window = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[2] || '%s', 'utf8'), ctx);
const rw = ctx.JETFORMER_SIM_REWRITE;
const res = {
  a: rw('http://127.0.0.1:8766/api/status'),
  b: rw('http://127.0.0.1:8766/api/models/bundle?name=x.zip'),
  c: rw('/api/models/run/bundle.zip'),
  d: rw('http://127.0.0.1:9999/api/status'),
  e: rw('http://127.0.0.1:8766/other'),
  f: rw('http://example.com:8766/api/status'),
};
ctx.window.fetch('http://127.0.0.1:8766/api/datasets/start');
new XMLHttpRequest().open('POST', 'http://127.0.0.1:8766/api/models/upload?kind=onnx&name=a');
ls.setItem('course-20s-scene-ids', '["a"]');
res.fetched = fetched; res.opened = opened; res.store = store;
res.scoped = ls.getItem('course-20s-scene-ids');
console.log(JSON.stringify(res));
''' % str(paths.STATIC / 'sim-shim.js').replace('\\', '/'))
    r = json.loads(out)
    assert r['a'] == 'http://192.168.5.14:8080/sim/bridge/api/status'
    assert r['b'] == 'http://192.168.5.14:8080/sim/bridge/api/models/bundle?name=x.zip'
    assert r['c'] is None and r['d'] is None and r['e'] is None and r['f'] is None
    assert r['fetched'][0] == 'http://192.168.5.14:8080/sim/bridge/api/datasets/start'
    assert r['opened'] == ['http://192.168.5.14:8080/sim/bridge/api/models/upload?kind=onnx&name=a']
    assert r['store']['course-jetformer-url'] == 'http://127.0.0.1:8080'
    assert r['store']['course-20s-scene-ids:wv-1'] == '["a"]' and 'course-20s-scene-ids' not in r['store']
    assert r['scoped'] == '["a"]'


SHIM_DOM = r'''
import fs from 'fs';
import vm from 'vm';
const ORIGIN = process.argv[2];
const sets = {}, observers = [];
let callbacks = 0;
class El {
  constructor(id) { this.id = id; this.attrs = {}; this.children = []; this.style = {}; this.hidden = false; this._t = '';
    this.classList = { toggle() {}, add() {} }; this.parentNode = { appendChild() {} }; }
  setAttribute(k, v) {   // DOM と同じく、同じ値を入れても変化の記録が作られる
    this.attrs[k] = String(v); sets[this.id] = (sets[this.id] || 0) + 1;
    for (const o of observers) if (o.target === this && (!o.filter || o.filter.includes(k))) o.queue();
  }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  get href() { const v = this.getAttribute('href'); try { return new URL(v, ORIGIN + '/sim/c/default/').href; } catch (e) { return ''; } }
  set href(v) { this.setAttribute('href', v); }
  appendChild(c) { this.children.push(c); return c; }
  insertAdjacentElement() {}
  addEventListener() {}
  set textContent(v) { this._t = String(v); } get textContent() { return this._t; }
}
class MutationObserver {
  constructor(cb) { this.cb = cb; this.pending = false; }
  observe(t, o) { this.target = t; this.filter = o.attributeFilter; observers.push(this); }
  queue() {
    if (this.pending) return;
    this.pending = true;
    queueMicrotask(() => { this.pending = false; if (++callbacks > 200) return; this.cb([]); });
  }
}
const els = {};
for (const id of ['endpoint', 'jetformer-open', 'dataset-download', 'collection-download']) els[id] = new El(id);
// app.js はループバックで開かれたとき #endpoint を location.origin にする。LAN から開くと既定の 127.0.0.1:8766 のまま
const LOOPBACK = /^http:\/\/(127\.0\.0\.1|localhost)(:|$)/.test(ORIGIN);
els.endpoint.value = LOOPBACK ? ORIGIN : 'http://127.0.0.1:8766';
let start = null;
const store = {};
class Storage { getItem(k) { return k in store ? store[k] : null; } setItem(k, v) { store[k] = String(v); } removeItem(k) { delete store[k]; } }
const body = new El('body'); body.firstChild = null; body.insertBefore = (c) => c;
const ctx = { console: { warn() {}, log: console.log }, URL, URLSearchParams, setTimeout, setInterval: () => 0, queueMicrotask,
  Storage, localStorage: new Storage(), MutationObserver, XMLHttpRequest: class { open() {} },
  fetch: () => Promise.resolve({ ok: false, status: 503, json: async () => ({}) }),
  location: { origin: ORIGIN, search: '', pathname: '/sim/c/default/', hash: '' },
  document: { readyState: 'loading', title: 'Course Lab', body, addEventListener: (t, f) => { start = f; },
    getElementById: (id) => els[id] || null, querySelector: () => null, createElement: (t) => new El(t) },
  courseSimulator: {},
  JETFORMER_SIM: { course: { id: 'default', world_version: 'wv', name: 'Default' }, bridge: '/sim/bridge', jetformer_port: 8080 } };
ctx.window = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[3], 'utf8'), ctx);
// アプリの初期化(createJetFormerLink の refreshLink)は shim の start より先に動く
els['jetformer-open'].href = 'http://127.0.0.1:8080';
start();
await new Promise((r) => setTimeout(r, 50));
const before = { ...sets };
// 収集の終了(setArchive → invalidate → refreshLink)・「接続・取り込み状況を確認」で、アプリが href を入れ直す
els['jetformer-open'].href = 'http://127.0.0.1:8080';
els['dataset-download'].href = els.endpoint.value + '/api/datasets/download/0123456789abcdef0123456789abcdef';   // new URL(download_url, bridgeURL())
await new Promise((r) => setTimeout(r, 50));
console.log(JSON.stringify({ callbacks, before, sets, open: els['jetformer-open'].getAttribute('href'),
  download: els['dataset-download'].getAttribute('href') }));
'''


@pytest.mark.parametrize('origin', ['http://127.0.0.1:8080', 'http://192.168.5.14:8080'])
def test_shim_link_fix_does_not_loop(tmp_path, origin):
    """#jetformer-open などの書き換えが MutationObserver で自分自身を呼び続けない(127.0.0.1 で開くとページが固まっていた)"""
    (tmp_path / 'main.mjs').write_text(SHIM_DOM)
    r = subprocess.run([NODE, str(tmp_path / 'main.mjs'), origin, str(paths.STATIC / 'sim-shim.js')],
                       capture_output=True, text=True, timeout=60, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out['callbacks'] <= 6, out                          # 直す前は 200 で打ち切り(無限ループ)
    assert out['open'] == origin + '/'
    assert out['download'] == origin + '/sim/bridge/api/datasets/download/0123456789abcdef0123456789abcdef'
    assert out['sets']['jetformer-open'] - out['before']['jetformer-open'] <= 2


def test_shim_adds_random_uuid_on_insecure_pages(tmp_path):
    """http://<Jetson の IP>:8080 は安全な接続ではないので crypto.randomUUID が無い。app.js が最初に呼ぶので shim が補う"""
    out = run_node(tmp_path, '''
import fs from 'fs';
import vm from 'vm';
import { webcrypto } from 'crypto';
const insecure = { getRandomValues: (a) => webcrypto.getRandomValues(a) };          // randomUUID の無い crypto
const secure = { getRandomValues: (a) => webcrypto.getRandomValues(a), randomUUID: () => 'native' };
const res = {};
for (const [name, crypto] of [['insecure', insecure], ['secure', secure]]) {
  class Storage { getItem() { return null; } setItem() {} removeItem() {} }
  const ctx = { console: { warn() {} }, URL, URLSearchParams, setTimeout, setInterval: () => 0, crypto,
    Storage, localStorage: new Storage(), XMLHttpRequest: class { open() {} },
    fetch: () => Promise.resolve({ ok: false, json: async () => ({}) }),
    location: { origin: 'http://192.168.55.1:8080', search: '', pathname: '/sim/c/default/', hash: '' },
    document: { readyState: 'loading', addEventListener() {}, getElementById() { return null; } },
    JETFORMER_SIM: { course: { id: 'default', world_version: 'wv' }, bridge: '/sim/bridge', jetformer_port: 8080 } };
  ctx.window = ctx;
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync('%s', 'utf8'), ctx);
  res[name] = [crypto.randomUUID(), crypto.randomUUID()];
}
console.log(JSON.stringify(res));
''' % str(paths.STATIC / 'sim-shim.js').replace('\\', '/'))
    r = json.loads(out)
    a, b = r['insecure']
    assert a != b
    for u in (a, b):
        assert len(u) == 36 and u[14] == '4' and u[19] in '89ab' and [len(x) for x in u.split('-')] == [8, 4, 4, 4, 12]
    assert r['secure'] == ['native', 'native']                 # もともと使えるページでは触らない
