"""走行タブの「走る日の準備」(基準の設定・走る前のチェック)・ベンチマークの候補・区間の速度上限のスイッチのテスト。
サーバー全体は起動しない(センサー・PCA9685 に触れない)。web/bench.py の Router だけを FastAPI に付けて試し、
画面の計算(app.js の benchFieldValues・benchDiff・BENCH_FALLBACK)は node で確かめる(node が無ければ飛ばす)

    cd ~/jetformer && python -m pytest -q tests/test_bench_ui.py
"""

import json
import os
import re
import shutil
import subprocess

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jetformer import drive
from jetformer.web import bench

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
STATIC = os.path.join(ROOT, 'jetformer', 'web', 'static')
APP_JS = os.path.join(STATIC, 'app.js')
NODE = shutil.which('node')
B = 'clean_driving_20261011-nocmd-s3-20261011-024626'
MSF = 'clean_driving_20261011-msfusion-s0-nocmd-stride3-e30'
TPL = ('bench=real_bench_20261023 day={date} block={block} cand={id} model={run} '
       'battery=#__ V_before=__ V_after=__ check=__ memo=')


def _run(runs, name, trained=(4, 5), cfg=None, check=None):
    d = runs / name
    d.mkdir(parents=True)
    (d / drive.ENGINE_FILE).write_bytes(b'engine-' + name.encode())
    (d / 'model_config.json').write_text(json.dumps(dict(cfg or {}, num_modes=6, trained_modes=list(trained),
                                                         steer_center_us=1441.0, throttle_neutral_us=1496.0)))
    (d / 'run.json').write_text(json.dumps({'created_at': '2026-10-21T10:00:00'}))
    if check is not None:                                  # 今のエンジンの確認(大きさ・更新時刻を合わせる)
        st = (d / drive.ENGINE_FILE).stat()
        (d / drive.ENGINE_CHECK_FILE).write_text(json.dumps(dict(check, engine={'size': st.st_size,
                                                                                'mtime_ns': st.st_mtime_ns})))
    return d


def _bench_doc(**over):
    doc = {
        'version': 1, 'bench': 'real_bench_20261023', 'protocol_sha256': '1dbb279e' + '0' * 56,
        'created_at': '2026-10-21T13:00:00',
        'baseline': {'mode_id': 5, 'params': dict(drive.DEFAULT_PARAMS, steer_gain=1.0, throttle_gain=1.0, lookahead=3,
                                                  tof_guard=1, ensemble_j=0, speed_cap=0, zone_caps=0)},
        'candidates': [
            {'id': 'B', 'label': 'モデルB(基準)', 'run': B, 'hypothesis': '基準', 'risk': 'low'},
            {'id': 'MSF', 'label': '多段型', 'run': MSF, 'hypothesis': '融合を何段も', 'risk': 'mid'},
            {'id': 'ENS', 'label': '2本の平均', 'run': 'bench_20261021-ens-p2', 'hypothesis': '机上で一番良い', 'risk': 'mid'},
            {'id': 'BAD', 'label': '精度の確認に落ちた', 'run': 'bad-engine', 'hypothesis': '-', 'risk': 'high'},
            {'id': 'M4', 'label': 'ジャンル 4 だけ', 'run': 'only-mode4', 'hypothesis': '-', 'risk': 'low'},
        ],
        'days': [
            {'date': '2026-10-23', 'blocks': [{'block': 1, 'id': 'B'}, {'block': 2, 'id': 'MSF'}, {'block': 3, 'id': 'ENS'},
                                              {'block': 4, 'id': 'B'}, {'block': 5, 'id': 'XX'}]},
            {'date': '2026-10-24', 'blocks': [], 'note': '10/23 の夜に決める'},
        ],
        'description_template': TPL,
    }
    doc.update(over)
    return doc


@pytest.fixture
def env(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    runs.mkdir()
    monkeypatch.setattr(drive, 'RUNS_DIR', runs)
    monkeypatch.setattr(drive, 'RECOMMENDED_PATH', tmp_path / 'drive_recommended.json')
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    monkeypatch.setattr(bench, 'BENCH_PATH', tmp_path / 'bench_candidates.json')
    _run(runs, B)
    _run(runs, MSF, cfg={'policy': 'msfusion'})
    _run(runs, 'bad-engine', check={'ok': False, 'accuracy_ok': False, 'actions_max_us': 9.0, 'gate_us': 2.5})
    _run(runs, 'only-mode4', trained=(4,))
    app = FastAPI()
    app.include_router(bench.router)
    c = TestClient(app)
    c.tmp, c.path = tmp_path, tmp_path / 'bench_candidates.json'
    return c


def _write(c, doc):
    c.path.write_text(doc if isinstance(doc, str) else json.dumps(doc, ensure_ascii=False), encoding='utf-8')


# ---------------------------------------------------------------- 基準の値(10/18 plan.md 3.4)

def test_fallback_baseline_is_plan_3_4_and_does_not_touch_defaults():
    before = dict(drive.DEFAULT_PARAMS)
    fb = bench.fallback_baseline()
    assert fb['run'] == B and fb['mode_id'] == 5
    p = fb['params']
    assert (p['steer_gain'], p['throttle_gain'], p['lookahead'], p['tof_guard'], p['ensemble_j'], p['speed_cap'],
            p['zone_caps']) == (1.0, 1.0, 3, 1, 0, 0.0, 0)
    for k in ('steer_trim_us', 'osc_percent', 'esc_reverse', 'recovery', 'rev_us', 'rev_s', 'speed_scale', 'tcp_mode',
              'ensemble_m'):
        assert p[k] == drive.DEFAULT_PARAMS[k], k                          # ほかは DEFAULT_PARAMS
    assert set(p) == set(drive.PARAM_LIMITS) == set(drive.DEFAULT_PARAMS)  # 画面のどの欄も埋まる
    p['steer_gain'] = 9.0
    assert drive.DEFAULT_PARAMS == before and bench.fallback_baseline()['params']['steer_gain'] == 1.0
    assert drive.DEFAULT_PARAMS['steer_gain'] == 1.0 and drive.DEFAULT_PARAMS['throttle_gain'] == 0.5   # 既定値は変えていない
    assert drive.DEFAULT_PARAMS['lookahead'] == 1 and drive.DEFAULT_PARAMS['zone_caps'] == 0


def test_missing_file_hides_panel_and_gives_fallback(env):
    r = env.get('/api/bench/candidates')
    assert r.status_code == 200
    d = r.json()
    assert d['exists'] is False and d['candidates'] == [] and d['days'] == []
    assert d['baseline'] == json.loads(json.dumps(bench.fallback_baseline()))
    assert not env.path.exists() and not (env.tmp / 'drive_settings.json').exists()     # 何も書かない


def test_broken_file_reports_error(env):
    _write(env, '{"version": 1, "candidates": [')
    d = env.get('/api/bench/candidates').json()
    assert d['exists'] is True and d['ok'] is False and 'JSON' in d['error']
    assert d['baseline']['mode_id'] == 5 and d['baseline']['run'] == B
    _write(env, '[1, 2]')
    d = env.get('/api/bench/candidates').json()
    assert d['exists'] and not d['ok'] and d['error']


# ---------------------------------------------------------------- 候補・ブロック・走れるか

def test_candidates_drivable_and_reasons(env):
    _write(env, _bench_doc())
    d = env.get('/api/bench/candidates').json()
    json.dumps(d, allow_nan=False)
    assert d['exists'] and d['ok'] and d['bench'] == 'real_bench_20261023' and d['protocol_sha256'].startswith('1dbb279e')
    by = {c['id']: c for c in d['candidates']}
    assert by['B']['drivable'] and by['B']['why_not'] is None and by['B']['risk'] == 'low'
    assert by['MSF']['drivable'] and by['MSF']['policy_title'].startswith('TransFuser多段型')
    assert not by['ENS']['drivable'] and 'TensorRT' in by['ENS']['why_not']          # 学習結果・エンジンが無い
    assert not by['BAD']['drivable'] and '精度の確認に落ちて' in by['BAD']['why_not']   # drivable False(drive と同じ理由)
    assert not by['M4']['drivable'] and 'ジャンル 5 を学習していません' in by['M4']['why_not']
    # drive.start と同じ確かめ: 走れると出したものは、model_info でも走れる
    for c in d['candidates']:
        info = drive.model_info(c['run'])
        assert c['drivable'] == bool(info and info['drivable'] and 5 in info['trained_modes']), c['id']


def test_blocks_carry_model_and_description_template(env):
    _write(env, _bench_doc())
    d = env.get('/api/bench/candidates').json()
    day1, day2 = d['days']
    assert [b['id'] for b in day1['blocks']] == ['B', 'MSF', 'ENS', 'B', 'XX']
    b1, b2, b3, b4, b5 = day1['blocks']
    assert b1['run'] == B and b1['drivable'] and b1['block'] == 1
    assert b1['description'] == ('bench=real_bench_20261023 day=2026-10-23 block=1 cand=B model=%s '
                                 'battery=#__ V_before=__ V_after=__ check=__ memo=' % B)       # 電池・電圧は空欄
    assert b2['description'].startswith('bench=real_bench_20261023 day=2026-10-23 block=2 cand=MSF model=%s ' % MSF)
    assert not b3['drivable'] and b3['why_not'] and b4['run'] == B and 'block=4 cand=B' in b4['description']
    assert not b5['drivable'] and '候補の表' in b5['why_not'] and b5['description'] == ''
    assert day2['blocks'] == [] and day2['note'] == '10/23 の夜に決める'
    assert d['baseline']['run'] == B                    # 基準の設定のモデル = 候補 B


def test_baseline_from_file_merges_defaults_and_never_turns_zone_caps_on(env):
    doc = _bench_doc(baseline={'mode_id': 5, 'params': {'steer_gain': 1.3, 'lookahead': 3.2, 'throttle_gain': 7,
                                                         'zone_caps': 1, 'no_such': 1, 'rev_s': 'x'}})
    _write(env, doc)
    d = env.get('/api/bench/candidates').json()
    p = d['baseline']['params']
    assert p['steer_gain'] == 1.3 and p['lookahead'] == 3 and p['throttle_gain'] == 2.0      # 丸める
    assert p['tof_guard'] == drive.DEFAULT_PARAMS['tof_guard'] and p['rev_s'] == drive.DEFAULT_PARAMS['rev_s']
    assert 'no_such' not in p and d['baseline']['source'] == 'data/bench_candidates.json'
    w = ' '.join(d['warnings'])
    assert 'no_such' in w and 'throttle_gain' in w and 'rev_s' in w and 'zone_caps' in w
    # zone_caps 1 はそのまま返す(画面の benchFieldValues が 0 にする。下の node の試験)


def test_custom_template_and_missing_baseline(env):
    doc = _bench_doc(description_template='day={date} b={block} c={id} m={run} battery=#__')
    del doc['baseline']
    _write(env, doc)
    d = env.get('/api/bench/candidates').json()
    assert d['days'][0]['blocks'][1]['description'] == 'day=2026-10-23 b=2 c=MSF m=%s battery=#__' % MSF
    assert d['baseline']['params'] == bench.fallback_baseline()['params'] and d['baseline']['run'] == B


# ---------------------------------------------------------------- 画面(静的なファイル・node)

def test_server_includes_bench_router_and_client_version_unchanged():
    src = open(os.path.join(ROOT, 'jetformer', 'web', 'server.py'), encoding='utf-8').read()
    assert 'from . import bench as bench_web' in src and 'app.include_router(bench_web.router)' in src
    assert 'DRIVE_CLIENT_VERSION = 6' in src
    assert "@app.post('/api/bench" not in src                              # 読むだけ(書く API は無い)
    bsrc = open(os.path.join(ROOT, 'jetformer', 'web', 'bench.py'), encoding='utf-8').read()
    assert '@router.post' not in bsrc and 'open(path, \'w\'' not in bsrc and 'driver.start' not in bsrc


def test_static_ui_has_prep_bench_and_zone_controls():
    html = open(os.path.join(STATIC, 'index.html'), encoding='utf-8').read()
    for el in ('dr-prep', 'btn-dr-baseline', 'dr-base-steer', 'dr-base-info', 'dr-base-note', 'btn-dr-base-undo',
               'dr-check-count', 'dr-check-disk', 'dr-check-heavy', 'dr-check-base', 'btn-dr-check-clear',
               'bench-card', 'bench-days', 'bench-cands', 'bench-error', 'dr-zone-caps', 'dr-zone-note'):
        assert 'id="%s"' % el in html, el
    assert html.count('class="dr-check"') == 6                              # 10/18 plan.md 3.5 の 6 項目
    for words in ('ip_status', '10 GB', '重い処理', 'CH3 が人側', '基準の設定', '電池の番号と電圧'):
        assert words in html, words
    assert 'G1 に合格し、ユーザーが了承したあと' in html                      # 区間の速度上限の注意書き
    zone = re.search(r'<input type="checkbox" id="dr-zone-caps"[^>]*>', html).group(0)
    assert 'checked' not in zone                                            # 既定 OFF
    assert 'autocomplete="off"' in zone                                   # 再読み込みで前の ON を戻さない(Firefox のフォームの復元)
    assert re.search(r'<div class="card hidden" id="bench-card">', html)     # ファイルが無ければ出さない
    js = open(APP_JS, encoding='utf-8').read()
    assert "'/api/bench/candidates'" in js
    assert re.search(r'client: 6\b', js)                                    # DRIVE_CLIENT_VERSION のまま
    assert 'randomUUID' not in js and 'crypto.' not in js                    # http(安全でない接続)でも動く
    assert "p.zone_caps = $('#dr-zone-caps').checked ? 1 : 0" in js
    # 基準の設定・候補のボタンは走行を始めない(/api/drive/start を呼ぶのは走行開始のボタンだけ)
    assert js.count("'/api/drive/start'") == 1
    fn = re.search(r'^function benchApply\(.*?^}\n', js, re.S | re.M).group(0)
    assert '/api/drive/' not in fn and 'api(' not in fn                      # 欄に入れるだけ(送らない)
    if NODE:
        r = subprocess.run([NODE, '--check', APP_JS], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def _node(code):
    js = open(APP_JS, encoding='utf-8').read()
    parts = [re.search(r'^const BENCH_FALLBACK = \{.*?^};\n', js, re.S | re.M).group(0)]
    for name in ('benchFieldValues', 'benchDiff'):
        m = re.search(r'^function %s\(.*?^}\n' % name, js, re.S | re.M)
        assert m, name
        parts.append(m.group(0))
    r = subprocess.run([NODE, '-e', '\n'.join(parts) + '\n' + code], capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@pytest.mark.skipif(NODE is None, reason='node がありません')
def test_js_fallback_matches_server_fallback():
    got = _node('process.stdout.write(JSON.stringify(BENCH_FALLBACK));')
    fb = bench.fallback_baseline()
    assert got['run'] == fb['run'] and got['mode_id'] == fb['mode_id'] and got['params'] == fb['params']


@pytest.mark.skipif(NODE is None, reason='node がありません')
def test_js_field_values_force_zone_caps_off_and_steer_of_the_day():
    base = dict(bench.fallback_baseline())
    on = dict(base, params=dict(base['params'], zone_caps=1))
    cases = [[base, ''], [base, '1.3'], [on, ''], [{'params': {}}, None], [base, 'x']]
    got = _node('const cases = %s;\nprocess.stdout.write(JSON.stringify(cases.map(([b, s]) => benchFieldValues(b, s))));'
                % json.dumps(cases))
    a, b, c, d, e = got
    assert a['run'] == B and a['mode_id'] == 5 and a['params'] == base['params'] and a['notes'] == []
    assert b['params']['steer_gain'] == 1.3 and b['params']['throttle_gain'] == 1.0          # この日のステアだけ替わる
    assert c['params']['zone_caps'] == 0 and c['notes'] and 'ON にしていません' in c['notes'][0]   # 基準の設定は ON にしない
    assert d['params'] == {'zone_caps': 0} and d['mode_id'] == 5 and d['run'] == ''
    assert e['params']['steer_gain'] == 1.0                                                   # 数でなければ基準どおり


@pytest.mark.skipif(NODE is None, reason='node がありません')
def test_js_diff_lists_only_changed_fields():
    saved = dict(drive.DEFAULT_PARAMS, steer_gain=1.5, throttle_gain=1.0, lookahead=3)       # 今の保存値(ステア 1.5)
    want = bench.fallback_baseline()['params']
    got = _node('process.stdout.write(JSON.stringify([benchDiff(%s, %s), benchDiff(%s, %s), benchDiff({steer_gain: 1}, %s)]));'
                % (json.dumps(saved), json.dumps(want), json.dumps(want), json.dumps(want), json.dumps(want)))
    assert got[0] == [['steer_gain', 1.5, 1.0]]
    assert got[1] == [] and got[2] == []                    # 画面に無い項目は比べない
