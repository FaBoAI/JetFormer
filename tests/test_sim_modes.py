"""シミュレーターの強化学習モード(jetformer/sim/rl_modes.py・rl_import.py と web.py の /api/sim/rl/*・ページの差し込み)のテスト

登録簿(data/sim_rl/modes/<key>/)の読み込みと確かめ、方策の配信、import map の差し替え、収集プラン(モード × 分)と進み具合・替え、
シーン ZIP のまとめての取り込み(学習除外・タグ・契約の確かめ)、ジャンルを付け直しても学習除外が残ること。
"""

import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import _sim_testlib as L
from test_sim_web import FakeBridge, FakeJobs
from jetformer import config, dataset
from jetformer.sim import paths, rl_import, rl_modes, web

POLICY = {'schema': 'rl-bestlap-policy/v1', 'obs_dim': 79, 'layers': [{'w': [[0.0]], 'b': [0.0]}]}


def write_mode(key, state='accepted', order=1, policy=POLICY, pool_n=30, pool_sha=None, tags=None, course='default'):
    d = rl_modes.modes_dir() / key
    d.mkdir(parents=True, exist_ok=True)
    pol = json.dumps(policy).encode()
    sha = hashlib.sha256(pol).hexdigest()
    if state != 'not_published':
        (d / 'policy.json').write_bytes(pol)
    mode = {'schema': rl_modes.MODE_SCHEMA, 'key': key, 'name_ja': '強化学習モード(%s)' % key, 'short_ja': '説明 %s' % key, 'order': order,
            'priority': 1, 'course': course, 'obs_wrapper': 'base79', 'runtime': {}, 'tags': tags or ['simulator', 'sim-rl', 'sim-rl-%s' % key],
            'policy_sha256': sha, 'recording': {'start_distribution': {'gate': 'U{0..32}'}}}
    (d / 'mode.json').write_text(json.dumps(mode, ensure_ascii=False), encoding='utf-8')
    (d / 'status.json').write_text(json.dumps({'schema': rl_modes.STATUS_SCHEMA, 'key': key, 'state': state, 'failed': [], 'pending': []}))
    (d / 'card.json').write_text(json.dumps({'schema': rl_modes.CARD_SCHEMA, 'key': key, 'headline_ja': '見出し', 'lap_from_stop_s': 8.5}))
    (d / 'eval.json').write_text(json.dumps({'created_at': 'x', 'judged': {}}))
    if pool_n:
        scenes = [{'c': i, 'start': {'gate': i % 33, 'left_offset_m': -0.2 + 0.4 * i / pool_n, 'left_yaw_deg': -15 + 30 * i / pool_n},
                   'seed': 700000 + i, 'params': {'c_v': 1.0} if key == 'speed' else {}, 'min_clearance': 0.1, 'valid_laps': 2,
                   'actions_sha256': '%064x' % i} for i in range(pool_n)]
        (d / 'scenes.json').write_text(json.dumps({'schema': rl_modes.POOL_SCHEMA, 'mode': key, 'policy_sha256': pool_sha or sha,
                                                   'scenes': scenes, 'acceptance_rate': 0.9}))
    return sha


@pytest.fixture
def client(tmp_path, monkeypatch):
    L.install_fake_release(tmp_path, monkeypatch)
    monkeypatch.setattr(config, 'DATASET_DIR', config.DATA_DIR / 'datasets')
    monkeypatch.setattr(config, 'RAW_DIR', config.DATA_DIR / 'raw')
    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv('JETFORMER_SIM_RL', raising=False)
    jobs, fb = FakeJobs(), FakeBridge()
    orig = web.bridge
    web.init(jobs=jobs, busy=lambda: {'training': False, 'driving': False, 'recording': False}, bridge=fb)
    web._index_cache.clear()
    app = FastAPI()
    app.include_router(web.router)
    c = TestClient(app)
    c.jobs = jobs
    yield c
    web.init(bridge=orig)
    web._index_cache.clear()


# ---------------------------------------------------------------- 登録簿・方策
def test_registry_states_and_checks(client):
    write_mode('fastest', 'accepted', 1)
    write_mode('recover', 'not_accepted', 3)
    write_mode('line', 'not_published', 4, pool_n=0)
    sha = write_mode('no_wall', 'accepted', 2)
    (rl_modes.modes_dir() / 'no_wall' / 'policy.json').write_text('{"changed": 1}')      # 差し替えの途中など
    write_mode('speed', 'accepted', 5, pool_sha='0' * 64)                                    # 別の方策で作ったシーンの組
    r = client.get('/api/sim/rl/modes')
    assert r.status_code == 200
    m = {x['key']: x for x in r.json()['modes']}
    assert [x['key'] for x in r.json()['modes']] == ['fastest', 'no_wall', 'recover', 'line', 'speed']
    assert m['fastest']['selectable'] and m['fastest']['plan_ok'] and m['fastest']['label'] == '合格'
    assert m['recover']['selectable'] and m['recover']['label'] == '不合格(参考)'
    assert not m['line']['selectable'] and m['line']['state'] == 'not_published' and m['line']['policy'] is None
    assert not m['no_wall']['selectable'] and 'sha256' in m['no_wall']['problems'][0] and m['no_wall']['policy_sha256'] == sha
    assert m['speed']['selectable'] and not m['speed']['plan_ok'] and m['speed']['pool'] is None and '別の方策' in m['speed']['warnings'][0]
    assert 'low-friction' in m['fastest']['scenario_blocklist']
    assert m['fastest']['datasets'] == {'count': 0, 'scenes': 0, 'minutes': 0.0, 'names': []}


def test_policy_endpoint(client):
    sha = write_mode('fastest')
    write_mode('line', 'not_published', pool_n=0)
    r = client.get('/api/sim/rl/modes/fastest/policy.json')
    assert r.status_code == 200 and r.json() == POLICY and r.headers['x-policy-sha256'] == sha
    assert r.headers['cache-control'] == 'no-cache'
    assert client.get('/api/sim/rl/modes/fastest/policy.json', headers={'If-None-Match': r.headers['etag']}).status_code == 304
    assert client.get('/api/sim/rl/modes/line/policy.json').status_code == 409
    assert client.get('/api/sim/rl/modes/nothing/policy.json').status_code == 404
    assert client.get('/api/sim/rl/modes/BAD..x/policy.json').status_code == 404


def test_disabled_by_env(client, monkeypatch):
    write_mode('fastest')
    monkeypatch.setenv('JETFORMER_SIM_RL', '0')
    web._index_cache.clear()
    assert client.get('/api/sim/rl/modes').status_code == 404
    html = client.get('/sim/c/default/').text
    assert 'sim-rl.js' not in html and '_jf/rl/teacher.js' not in html
    assert client.get('/sim/c/default/_jf/rl/teacher.js').status_code == 404


# ---------------------------------------------------------------- ページの差し込み
def test_index_rewrites_importmap_and_adds_rl_script(client):
    html = client.get('/sim/c/default/').text
    head = html.split('<meta charset="utf-8">', 1)[1]
    assert head.index('sim-shim.js') < head.index('/sim/static/sim-rl.js?v=') < head.index('<script type="importmap">') < head.index('src="./app.js"')
    a = html.index('<script type="importmap">') + len('<script type="importmap">')
    im = json.loads(html[a:html.index('</script>', a)])
    assert im['imports']['three'] == './vendor/three.module.js'                      # 元の対応はそのまま
    assert im['imports']['./rc-speed-controller.js'] == './_jf/rl/teacher.js'
    assert im['scopes']['./_jf/'] == {'./rc-speed-controller.js': './rc-speed-controller.js'}
    assert client.get('/sim/static/sim-rl.js').status_code == 200
    assert client.get('/sim/static/sim-rl.css').status_code == 200


def test_rl_static_files_and_config(client):
    r = client.get('/sim/c/default/_jf/rl/teacher.js')
    assert r.status_code == 200 and r.headers['content-type'].startswith('text/javascript')
    man = json.loads((paths.STATIC / 'rl' / 'manifest.json').read_text())
    for name, meta in man['files'].items():                                           # Node の試験と同じファイル(sha256)
        assert hashlib.sha256((paths.STATIC / 'rl' / name).read_bytes()).hexdigest() == meta['sha256'], name
    assert client.get('/sim/c/default/_jf/rl/mode_runtime.mjs').status_code == 200
    assert client.get('/sim/c/default/_jf/rl/features.mjs').headers['content-type'].startswith('text/javascript')
    for bad in ('_jf/rl/nothing.mjs', '_jf/rl/%2e%2e/config.js', '_jf/rl', '_jf/rl/x/teacher.js', '_jf/rl/.manifest.json.tmp'):
        assert client.get('/sim/c/default/' + bad).status_code == 404, bad
    cfg = json.loads(client.get('/sim/c/default/_jf/config.js').text.split('freeze(', 1)[1][:-3])
    assert cfg['rl']['enabled'] is True and cfg['rl']['course_ok'] is True and cfg['rl']['modes_api'] == '/api/sim/rl/modes'
    other = 'indoor-course-eval-2-shortcut'
    cfg2 = json.loads(client.get('/sim/c/%s/_jf/config.js' % other).text.split('freeze(', 1)[1][:-3])
    assert cfg2['rl']['course_ok'] is False and 'Default' in cfg2['rl']['reason']


# ---------------------------------------------------------------- 収集プラン
def test_plan_create_scenes_splits_and_fields(client):
    write_mode('fastest', order=1)
    write_mode('speed', order=5)
    r = client.post('/api/sim/rl/plans', json={'minutes': {'fastest': 5, 'speed': 1}, 'val_ratio': 0.2, 'seed': 3})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s['scenes'] == 18 and s['url'].startswith('/sim/c/default/?rlplan=')
    plan = client.get('/api/sim/rl/plans/%s' % s['id']).json()
    assert plan['schema'] == rl_modes.PLAN_SCHEMA and plan['state'] == 'ready'
    fast = [x for x in plan['scenes'] if x['mode'] == 'fastest']
    assert len(fast) == 15 and sum(x['split'] == 'val' for x in fast) == 3                 # 5 分 = 15 シーン = train 12・val 3
    assert len({x['name'] for x in plan['scenes']}) == 18
    starts = [(x['fields']['start-gate'], x['fields']['spawn-offset'], x['fields']['spawn-yaw']) for x in plan['scenes'] if x['mode'] == 'fastest']
    assert len(set(starts)) == 15                                                          # 開始が重ならない(train と val をまたがない)
    allowed = {'name', 'dataset-split', 'seed', 'start-gate', 'spawn-offset', 'spawn-yaw', 'scenario', 'controller', 'teacher-profile',
               'episode-mode', 'limit', 'dataset-mode'}
    for x in plan['scenes']:
        f = x['fields']
        assert set(f) == allowed and f['controller'] == 'demo' and f['teacher-profile'] == 'speed-rc' and f['dataset-mode'] == 8
        assert f['episode-mode'] == 'continuous-laps-v1' and f['limit'] == 20 and f['scenario'] in ('baseline', 'noise', 'dim')
        assert f['name'] == x['name'] and f['dataset-split'] == x['split']
    assert [x['scenario'] for x in fast[:4]] == ['baseline', 'baseline', 'noise', 'dim']
    assert plan['modes']['fastest']['dataset_name'].startswith('sim-rl-fastest_') and len(plan['modes']['fastest']['spares']) == 5
    assert [x['params'] for x in plan['scenes'] if x['mode'] == 'speed'] == [{'c_v': 1.0}] * 3
    lst = client.get('/api/sim/rl/plans').json()['plans']
    assert lst[0]['id'] == s['id'] and lst[0]['modes']['fastest']['planned'] == 15


def test_plan_rejects_bad_requests(client):
    write_mode('fastest', pool_n=6)
    write_mode('line', 'not_published', pool_n=0)
    assert client.post('/api/sim/rl/plans', json={'minutes': {'fastest': 0}}).status_code == 400
    assert client.post('/api/sim/rl/plans', json={'minutes': {'fastest': 21}}).status_code == 400
    assert client.post('/api/sim/rl/plans', json={'minutes': {'fastest': 3}}).status_code == 409        # シーンの組は 6 = 2 分まで
    assert client.post('/api/sim/rl/plans', json={'minutes': {'line': 1}}).status_code == 409
    assert client.post('/api/sim/rl/plans', json={'minutes': {'fastest': 1}, 'course': 'indoor-course-eval-2-shortcut'}).status_code == 400
    assert client.post('/api/sim/rl/plans', json={'minutes': {'nothing': 1}}).status_code == 404
    assert client.get('/api/sim/rl/plans/../../x').status_code == 404
    assert client.get('/api/sim/rl/plans/20261014-000000-abcdef').status_code == 404


def test_progress_and_spare_replacement(client):
    write_mode('fastest', pool_n=12)
    pid = client.post('/api/sim/rl/plans', json={'minutes': {'fastest': 1}}).json()['id']
    plan = client.get('/api/sim/rl/plans/%s' % pid).json()
    first = plan['order'][0]
    r = client.post('/api/sim/rl/plans/%s/progress' % pid, json={'scene': first, 'status': 'failed', 'frames': 120, 'collisions': 1,
                                                                   'replace': True})
    assert r.status_code == 200
    rep = r.json()['replacement']
    p = r.json()['plan']
    assert rep['id'] == first + 'r1' and p['order'][1] == rep['id'] and rep['split'] == p['scenes'][0]['split']
    assert p['progress'][first]['replaced_by'] == rep['id'] and p['modes']['fastest']['spares_used'] == 1
    assert rep['fields']['seed'] == p['modes']['fastest']['spares'][0]['scene']['seed']
    sid = 'a' * 32
    p = client.post('/api/sim/rl/plans/%s/progress' % pid, json={'scene': rep['id'], 'status': 'done', 'session_id': sid, 'frames': 600,
                                                                   'collisions': 0, 'laps': [8.1, 7.7], 'plan_state': 'running'}).json()['plan']
    assert p['state'] == 'running' and p['progress'][rep['id']]['laps'] == [8.1, 7.7]
    assert rl_modes.mode_session_ids(p, 'fastest') == [(p['scenes'][-1], sid)]
    assert client.post('/api/sim/rl/plans/%s/progress' % pid, json={'scene': 's999', 'status': 'done'}).status_code == 404
    assert client.post('/api/sim/rl/plans/%s/progress' % pid, json={'scene': first, 'status': 'odd'}).status_code == 400
    assert client.post('/api/sim/rl/plans/%s/progress' % pid, json={'scene': first, 'status': 'done', 'session_id': '../x'}).status_code == 400


def test_spare_name_keeps_mode_name(client):
    """替えのシーンの名前は「元の名前-r<k>」(名前に「-r」を含む recover でも。前は rl-recover-train-00 の替えが rl-r1 になっていた)"""
    write_mode('recover', 'not_accepted', pool_n=12)
    pid = client.post('/api/sim/rl/plans', json={'minutes': {'recover': 1}}).json()['id']
    plan = client.get('/api/sim/rl/plans/%s' % pid).json()
    first = plan['order'][0]
    orig = next(s for s in plan['scenes'] if s['id'] == first)['name']
    assert orig.startswith('rl-recover-')
    rep = client.post('/api/sim/rl/plans/%s/progress' % pid, json={'scene': first, 'status': 'failed', 'frames': 225, 'collisions': 1,
                                                                     'replace': True}).json()['replacement']
    assert rep['name'] == orig + '-r1' and rep['fields']['name'] == rep['name']
    rep2 = client.post('/api/sim/rl/plans/%s/progress' % pid, json={'scene': rep['id'], 'status': 'failed', 'frames': 10, 'collisions': 1,
                                                                      'replace': True}).json()['replacement']
    assert rep2['name'] == orig + '-r2' and rep2['id'] == first + 'r2'


# ---------------------------------------------------------------- まとめての取り込み
N = 60


def scene_zip(sid, scene, split, l0=(), teacher_mode='recover', start=(1.0, 2.0, 0.1), model_config=None, collisions=0):
    """ブリッジの 1 シーンの ZIP(本物と同じ形の小さいもの)を data/sim/runtime/datasets/<sid>/ に置く"""
    name = '%s_20261014-000000_%s' % (scene.lower(), sid[:8])
    files = {}
    t0 = 1_800_000_000_000_000
    ts = t0 + np.rint(np.arange(N) * 1e6 / 30).astype(np.int64)
    cams = ['samples/CAM_FRONT/%s/%d.jpg' % (scene, t) for t in ts]
    jpg = L.tiny_jpg()
    for c in cams:
        files[c] = jpg
    fv = np.arange(N) < N - 10
    veh = np.tile(np.array([1442, 1600, 1500, -3000], np.float32), (N, 1))
    fa = np.zeros((N, 10, 2), np.float32)
    for h in range(1, 11):
        fa[:N - h, h - 1] = veh[h:, :2]
    buf = io.BytesIO()
    np.savez(buf, timestamps_us=ts, cam_files=np.array(cams), tof=np.zeros((N, 42, 54), np.uint16), tof_age_ms=np.zeros(N, np.float32),
             imu=np.zeros((N, 10, 6), np.float32), vehicle=veh, vehicle_age_ms=np.zeros(N, np.float32), vehicle_valid=np.ones(N, bool),
             erpm_valid=np.ones(N, bool), future_actions=fa, future_valid=fv, mode=np.full(N, 8, np.int16), human_control=np.zeros(N, bool),
             synthetic=np.ones(N, bool), imu_timestamps_us=np.tile(ts[:, None], (1, 10)))
    files['aligned/%s.npz' % scene] = buf.getvalue()
    modes = [{'id': i, 'name': n} for i, n in enumerate(['未設定', '駐車(1)', '駐車(2)', '駐車(3)', '高速走行', '低速走行', '未使用', '未使用', 'シミュレーター'])]
    card = {'name': name, 'version': 'v1.0-' + name, 'scenes': 1, 'train_scenes': int(split == 'train'), 'val_scenes': int(split == 'val'),
            'frames': N, 'synthetic': True, 'controller': 'demo', 'training_label_mode': 'synthetic_controller', 'training_frames': N - 10,
            'modes': modes, 'num_modes': 9, 'mode_frames': {'8': N}, 'notes': ['合成']}
    files['dataset_card.json'] = json.dumps(card).encode()
    files['splits.json'] = json.dumps({'train': [scene] if split == 'train' else [], 'val': [scene] if split == 'val' else [], 'test': []}).encode()
    cfg = {'seed': 7, 'noiseStd': 0, 'frictionMultiplier': 1, 'episodeMode': 'continuous-laps-v1', 'maxSpeed': 7.24}
    teacher = {'schema': 'jetformer-sim-rl-teacher/v1', 'id': 'jf-rl-%s-v1' % teacher_mode, 'mode': teacher_mode, 'policy_sha256': 'p' * 64}
    files['simulation/session.json'] = json.dumps({'schema': rl_import.SCHEMA, 'context': {'world_version': 'v3', 'route': 'r', 'teacher': teacher},
                                                   'config': cfg, 'calibration': {'max_speed_mps': 7.24},
                                                   'model_config': model_config or {'steer_center_us': 1441, 'throttle_neutral_us': 1496}}).encode()
    rows = [{'frame': i, 'sim_time': i / 30, 'timestamp_us': int(ts[i]), 'action_info': {'jf': {'v': 1, 'm': teacher_mode, 'f': i, 'l': 0 if i in l0 else 1}},
             'pose_pre_action': {'x': start[0], 'z': start[1], 'yaw': start[2]} if i == 0 else {'x': 0, 'z': 0, 'yaw': 0}} for i in range(N)]
    files['simulation/frames.jsonl'] = ''.join(json.dumps(r) + '\n' for r in rows).encode()
    files['simulation/evaluation-report.json'] = json.dumps({'outcome': {'terminal': True, 'success': False, 'reason': 'collection_duration_complete'},
                                                             'metrics': {'collisions': collisions, 'completed_laps': 2}, 'scenario': 'baseline',
                                                             'config': cfg}).encode()
    files['v1.0-%s/sample.json' % name] = json.dumps([{'token': 'tok-%s-%d' % (sid, i)} for i in range(3)]).encode()
    files['v1.0-%s/sensor.json' % name] = json.dumps([{'token': 'sensor-cam', 'channel': 'CAM_FRONT'}]).encode()
    files['manifest.json'] = json.dumps({'schema': rl_import.SCHEMA, 'frames': N, 'files': [
        {'path': p, 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()} for p, b in files.items()]}).encode()
    d = paths.runtime_dir() / 'datasets' / sid
    d.mkdir(parents=True)
    zp = d / (name + '.zip')
    with zipfile.ZipFile(zp, 'w') as z:
        for p, b in files.items():
            z.writestr(name + '/' + p, b)
    (d / 'archive-info.json').write_text(json.dumps({'schema': 'course-sim-jetformer-dataset-v1', 'session_id': sid, 'filename': zp.name,
                                                     'bytes': zp.stat().st_size}))
    return sid


def plan_with_done(client, key, sids):
    pid = client.post('/api/sim/rl/plans', json={'minutes': {key: 1}, 'val_ratio': 0.34}).json()['id']
    plan = client.get('/api/sim/rl/plans/%s' % pid).json()
    for sc_id, sid in zip(plan['order'], sids):
        client.post('/api/sim/rl/plans/%s/progress' % pid, json={'scene': sc_id, 'status': 'done', 'session_id': sid, 'frames': 600})
    return pid, plan


def test_import_merges_scenes_with_mask_and_tags(client, monkeypatch):
    monkeypatch.setattr(rl_import, 'FRAMES', N)
    write_mode('recover', 'not_accepted', tags=['simulator', 'sim-rl', 'sim-rl-recover', 'sim-recover'])
    sids = [scene_zip('%032x' % 1, '20261014-000000-00000001-0001', 'train', l0=range(20, 25), start=(1, 2, .1)),
            scene_zip('%032x' % 2, '20261014-000000-00000002-0001', 'train', start=(1.5, 2, .1)),
            scene_zip('%032x' % 3, '20261014-000000-00000003-0001', 'val', l0=(40,), start=(2, 2, .1))]
    pid, plan = plan_with_done(client, 'recover', sids)
    bad = scene_zip('%032x' % 4, '20261014-000000-00000004-0001', 'train', collisions=1, start=(9, 9, 0))   # 外れたシーン(衝突)
    pl = rl_modes.get_plan(pid)                                     # 外れたシーンが 1 つあり、取り込んだら元の ZIP を消すプランにする
    pl['scenes'].append(dict(pl['scenes'][0], id='s009'))
    pl['progress']['s009'] = {'status': 'failed', 'session_id': bad}
    pl['delete_sources'] = True
    rl_modes._save_plan(pl)
    r = client.post('/api/sim/rl/import', json={'plan': pid, 'mode': 'recover'})
    assert r.status_code == 200, r.text
    job = client.jobs.jobs[-1]
    assert job['state'] == 'done', job.get('error')
    name = job['result']['name']
    assert name == plan['modes']['recover']['dataset_name'] and name.startswith('sim-rl-recover_')
    root = config.DATASET_DIR / name
    card = json.loads((root / 'dataset_card.json').read_text())
    assert card['scenes'] == 3 and card['frames'] == 3 * N and card['train_scenes'] == 2 and card['val_scenes'] == 1
    assert set(card['tags']) >= {'simulator', 'sim-rl', 'sim-rl-recover', 'sim-recover'} and card['source'] == 'simulator'
    assert card['rl']['mode'] == 'recover' and card['rl']['plan'] == pid and card['rl']['scenes'] == 3
    lm = card['label_mask']
    assert lm['rule_id'] == 'k ∈ [j − 10, j]' and lm['masked_frames'] == 15 + 11
    with np.load(root / 'aligned' / '20261014-000000-00000001-0001.npz') as z:
        assert np.array_equal(np.nonzero(z['mode'] == 0)[0], np.arange(10, 25))     # [ps − 10, pe − 1]
        assert not z['future_valid'][10:25].any() and z['future_valid'][25:N - 10].all()
    for s in ('00000002', '00000003'):
        assert (root / 'simulation' / 'scenes' / ('20261014-000000-%s-0001' % s) / 'frames.jsonl').is_file()
    coll = json.loads((root / 'simulation' / 'collection.json').read_text())
    assert [x['source_session_id'] for x in coll['scenes']] == sids and coll['scenes'][0]['plan_scene'] == plan['order'][0]
    tables = json.loads((root / ('v1.0-' + name) / 'sensor.json').read_text())
    assert len(tables) == 1                                                                 # 同じ token は 1 つに重ねる
    assert len(json.loads((root / ('v1.0-' + name) / 'sample.json').read_text())) == 9
    p = client.get('/api/sim/rl/plans/%s' % pid).json()
    assert p['imports']['recover']['state'] == 'done' and p['imports']['recover']['dataset'] == name
    assert sorted(job['result']['deleted_sources']) == sorted(sids + [bad])                    # 元のシーンと外れたシーンの ZIP を消した
    assert not (paths.runtime_dir() / 'datasets' / bad).exists()
    assert client.post('/api/sim/rl/import', json={'plan': pid, 'mode': 'recover'}).status_code == 409   # 2 回目は取り込まない
    ds = client.get('/api/sim/rl/modes').json()['modes'][0]['datasets']
    assert ds['count'] == 1 and ds['scenes'] == 3
    # シーンタブに展開したシーンのジャンルを付け直しても、学習から外した区間(未設定)は残す(確認 recover r2)
    sc = [s for s in dataset.list_scenes() if s['name'].startswith('ds-' + name)]
    assert len(sc) == 3
    first = json.loads((config.RAW_DIR / ('ds-' + name) / sc[0]['name'] / 'scene.json').read_text())
    assert first['label_mask']['unset_segments']
    dataset.update_scene(sc[0]['name'], mode_id=4)
    after = json.loads((config.RAW_DIR / ('ds-' + name) / sc[0]['name'] / 'scene.json').read_text())
    unset = [g for g in after['mode_segments'] if g['mode_id'] == 0]
    assert unset == first['label_mask']['unset_segments'] and {g['mode_id'] for g in after['mode_segments']} == {0, 4}
    t = np.round(np.arange(N) / 30, 3)
    marr = dataset._frame_modes(after, (t * 1e6).astype(np.int64))
    assert np.array_equal(np.nonzero(marr == 0)[0], np.arange(10, 25)) and (np.delete(marr, np.arange(10, 25)) == 4).all()


def test_import_rejects_contract_mismatch_and_collisions(client, monkeypatch):
    monkeypatch.setattr(rl_import, 'FRAMES', N)
    write_mode('recover', 'not_accepted')
    a = scene_zip('%032x' % 11, '20261014-000000-0000000b-0001', 'train')
    b = scene_zip('%032x' % 12, '20261014-000000-0000000c-0001', 'train', model_config={'steer_center_us': 1375, 'throttle_neutral_us': 1333},
                  start=(3, 3, 0))
    with pytest.raises(rl_import.ImportError_, match='model_config'):
        rl_import.import_scenes([a, b], 'sim-rl-recover_x')
    c = scene_zip('%032x' % 13, '20261014-000000-0000000d-0001', 'train', collisions=1, start=(4, 4, 0))
    with pytest.raises(rl_import.ImportError_, match='衝突'):
        rl_import.import_scenes([a, c], 'sim-rl-recover_y')
    d = scene_zip('%032x' % 14, '20261014-000000-0000000e-0001', 'val')                     # a と同じ開始姿勢
    with pytest.raises(rl_import.ImportError_, match='開始姿勢'):
        rl_import.import_scenes([a, d], 'sim-rl-recover_z')
    assert not [p for p in config.DATASET_DIR.iterdir() if not p.name.startswith('.')]         # 半端なデータセットは残らない
    assert not [p for p in config.DATASET_DIR.iterdir() if p.name.startswith('.import-')]
