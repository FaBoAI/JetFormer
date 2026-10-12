"""走行タブのモデルの選び方のテスト(推奨のモデル・精度の確認に落ちたエンジン)。CPU だけ・サーバーは起動しない

- 推奨のモデル(data/drive_recommended.json)は一覧の先頭に出し、走行タブは初回にこれを選ぶ(前回のモデルより先)。
  前は「前回のモデル → 一番新しい学習結果」で、評価の前に使った古いモデルや比較の実験で作ったモデルが選ばれていた
- 精度の確認(engine_check.json の accuracy_ok)に落ちた今のエンジンは drivable False(画面で選べない・走行を始めない)。
  理由を出す。速さだけの不合格・古いエンジンの確認は今まで通り走れる
- 画面の選び方(app.js の pickDriveModel)は node で確かめる(node が無ければ飛ばす)

    cd ~/jetformer && python -m pytest -q tests/test_drive_model_select.py
"""

import json
import os
import re
import shutil
import subprocess
import sys

import pytest

from jetformer import drive

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
APP_JS = os.path.join(ROOT, 'jetformer', 'web', 'static', 'app.js')
sys.path.insert(0, os.path.join(ROOT, 'tools'))

B = 'clean_driving_20261011-nocmd-s3-20261011-024626'


class FakeRecorder:
    recording = False


def _run(runs, name, created_at, cfg=None, check=None):
    d = runs / name
    d.mkdir(parents=True)
    (d / drive.ENGINE_FILE).write_bytes(b'engine-' + name.encode())
    (d / 'model_config.json').write_text(json.dumps(dict(cfg or {}, num_modes=6, trained_modes=[4, 5],
                                                         steer_center_us=1441.0, throttle_neutral_us=1496.0)))
    (d / 'run.json').write_text(json.dumps({'created_at': created_at}))
    if check is not None:                                  # 今のエンジンの確認(大きさ・更新時刻を合わせる)
        st = (d / drive.ENGINE_FILE).stat()
        (d / drive.ENGINE_CHECK_FILE).write_text(json.dumps(dict(check, engine={'size': st.st_size,
                                                                                'mtime_ns': st.st_mtime_ns})))
    return d


@pytest.fixture
def runs(monkeypatch, tmp_path):
    r = tmp_path / 'runs'
    r.mkdir()
    monkeypatch.setattr(drive, 'RUNS_DIR', r)
    monkeypatch.setattr(drive, 'RECOMMENDED_PATH', tmp_path / 'drive_recommended.json')
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    _run(r, 'first_trial', '2026-10-10T15:36:29')
    _run(r, B, '2026-10-11T02:46:26')
    _run(r, 'zoo-act-s1', '2026-10-12T17:00:00', {'policy': 'act'})
    _run(r, 'zoo-dinov2-s1', '2026-10-12T18:00:00', {'policy': 'dinov2'})
    return r


ACC_FAIL = {'ok': False, 'accuracy_ok': False, 'speed_ok': True, 'actions_max_us': 2.854, 'gate_us': 2.5,
            'run_numpy_median_ms': 4.98, 'checked_at': '2026-10-12 18:27:24', 'note': 'FP16 のまま'}
SPEED_FAIL = {'ok': False, 'accuracy_ok': True, 'speed_ok': False, 'actions_max_us': 1.2, 'gate_us': 2.5,
              'run_numpy_median_ms': 6.8, 'checked_at': '2026-10-12 18:27:24', 'note': ''}


# ---------------------------------------------------------------- サーバー側(drive.list_models・model_info・start)

def test_without_recommendation_order_is_unchanged(runs):
    ms = drive.list_models()
    assert [m['name'] for m in ms] == ['zoo-dinov2-s1', 'zoo-act-s1', B, 'first_trial']     # 今まで通り新しい順
    assert not any(m['recommended'] for m in ms) and all('recommended_reason' not in m for m in ms)
    assert drive.recommended() is None


def test_recommended_model_is_listed_first(runs):
    rec = drive.save_recommended(B, '評価: ベスト = JetFormer v0(規則 4)。モデル B のまま', 'zoo_eval.json')
    assert rec['run'] == B and rec['decided_at']
    assert drive.recommended() == {'run': B, 'reason': rec['reason'], 'source': 'zoo_eval.json',
                                   'decided_at': rec['decided_at']}
    ms = drive.list_models()
    assert [m['name'] for m in ms] == [B, 'zoo-dinov2-s1', 'zoo-act-s1', 'first_trial']     # 推奨を先頭、残りは新しい順
    assert ms[0]['recommended'] is True and ms[0]['recommended_reason'].startswith('評価')
    assert [m['recommended'] for m in ms[1:]] == [False] * 3
    json.dumps(ms, allow_nan=False)                                                         # API でそのまま返せる


def test_recommendation_missing_or_broken_falls_back(runs):
    drive.save_recommended('deleted_run', 'x')                     # 消えた学習結果を指している → 誰も推奨にならない
    ms = drive.list_models()
    assert not any(m['recommended'] for m in ms) and ms[0]['name'] == 'zoo-dinov2-s1'
    drive.RECOMMENDED_PATH.write_text('{broken')
    assert drive.recommended() is None and drive.list_models()[0]['name'] == 'zoo-dinov2-s1'
    drive.RECOMMENDED_PATH.write_text(json.dumps({'run': ''}))
    assert drive.recommended() is None
    for bad in ('', '../x', 'a/b'):
        with pytest.raises(ValueError):
            drive.save_recommended(bad)


def test_engine_failing_accuracy_gate_is_not_drivable(runs):
    _run(runs, 'dino-fp16', '2026-10-12T19:00:00', {'policy': 'dinov2'}, check=ACC_FAIL)
    info = drive.model_info('dino-fp16')
    assert info['drivable'] is False and info['engine_check']['accuracy_ok'] is False
    assert '精度の確認に落ちています' in info['why_not'] and '最大 2.85 µs > 2.5 µs' in info['why_not']
    assert 'dino-fp16' in [m['name'] for m in drive.list_models()]          # 一覧には出す(画面で選べなくし、理由を出す)
    d = drive.Driver(hub=None, recorder=FakeRecorder())
    d._run = lambda *a: pytest.fail('走行のスレッドを始めてはいけない')
    with pytest.raises(ValueError, match='精度の確認'):
        d.start('dino-fp16', 5)                                              # 古い画面から送られても始めない
    assert d.thread is None and d.last['run'] is None


def test_speed_only_failure_and_stale_check_stay_drivable(runs):
    _run(runs, 'slow', '2026-10-12T19:00:00', check=SPEED_FAIL)
    info = drive.model_info('slow')
    assert info['drivable'] is True and info['why_not'] == '' and info['engine_check']['speed_ok'] is False
    d = _run(runs, 'rebuilt', '2026-10-12T19:10:00', {'policy': 'dinov2'}, check=ACC_FAIL)
    (d / drive.ENGINE_FILE).write_bytes(b'engine rebuilt with fp16_norm_fp32')   # 作り直した → 古い不合格は当てない
    info = drive.model_info('rebuilt')
    assert info['engine_check'] is None and info['drivable'] is True
    _run(runs, 'passed', '2026-10-12T19:20:00', check=dict(ACC_FAIL, ok=True, accuracy_ok=True, actions_max_us=2.05))
    assert drive.model_info('passed')['drivable'] is True


def test_config_reason_wins_over_engine_gate(runs):
    """model_config で走れない(入力が作れない)ものは、その理由のまま(エンジンの確認の理由で上書きしない)"""
    _run(runs, 'hist', '2026-10-12T19:00:00', {'policy': 'act', 'inputs': ['image', 'history']}, check=ACC_FAIL)
    info = drive.model_info('hist')
    assert info['drivable'] is False and 'history' in info['why_not']


def test_disabled_rl_engines_stay_hidden(runs):
    d = _run(runs, 'ai_mix-rl-a', '2026-10-12T20:00:00')
    (d / drive.ENGINE_FILE).rename(d / (drive.ENGINE_FILE + '.do_not_drive'))
    drive.save_recommended('ai_mix-rl-a', '名前を変えたエンジンを指しても出さない')
    ms = drive.list_models()
    assert 'ai_mix-rl-a' not in [m['name'] for m in ms] and not any(m['recommended'] for m in ms)


# ---------------------------------------------------------------- tools/policy_zoo.py recommend

def test_policy_zoo_recommend_writes_eval_recommendation(runs, monkeypatch, tmp_path):
    import policy_zoo as pz
    res = {'recommended': {'name': 'B', 'run': B}, 'verdict': {'best': 'jetformer_v0', 'rule': 4}}
    run, reason, _ = pz.recommendation_from_eval(res, 'zoo_eval.json')
    assert run == B and '規則 4' in reason and 'モデル B のまま' in reason
    with pytest.raises(SystemExit):
        pz.recommendation_from_eval({'verdict': {'best': 'act'}})
    zoo = tmp_path / 'zoo'
    zoo.mkdir()
    (zoo / 'zoo_eval.json').write_text(json.dumps(res))
    monkeypatch.setattr(pz, 'zoo_dir', lambda: zoo)
    pz.cmd_recommend(type('A', (), {'run': '', 'reason': ''})())
    assert drive.recommended()['run'] == B and drive.list_models()[0]['name'] == B
    pz.cmd_recommend(type('A', (), {'run': 'zoo-act-s1', 'reason': '試す'})())   # 手で指定
    assert drive.recommended()['run'] == 'zoo-act-s1' and drive.recommended()['reason'] == '試す'
    with pytest.raises(SystemExit):
        pz.cmd_recommend(type('A', (), {'run': 'no_such_run', 'reason': ''})())   # 走行タブに出ないものは書かない
    assert drive.recommended()['run'] == 'zoo-act-s1'


# ---------------------------------------------------------------- 画面(app.js の pickDriveModel)

def _pick_cases(cases):
    js = open(APP_JS, encoding='utf-8').read()
    m = re.search(r'^function pickDriveModel\(.*?^}\n', js, re.S | re.M)
    assert m, 'app.js に pickDriveModel がありません'
    code = m.group(0) + '\nconst cases = %s;\nprocess.stdout.write(JSON.stringify(cases.map(([ms, o]) => pickDriveModel(ms, o))));' \
        % json.dumps(cases, ensure_ascii=False)
    r = subprocess.run(['node', '-e', code], capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@pytest.mark.skipif(not shutil.which('node'), reason='node がありません')
def test_ui_picks_recommended_before_last_and_newest():
    rec = {'name': B, 'recommended': True, 'drivable': True}
    newest = {'name': 'zoo-dinov2-s1', 'drivable': True}
    act = {'name': 'zoo-act-s1', 'drivable': True}
    old = {'name': 'first_trial', 'drivable': True}
    bad = {'name': 'dino-fp16', 'drivable': False}
    ms = [rec, bad, newest, act, old]                                # list_models の順(推奨が先頭)
    no_rec = [bad, newest, act, dict(rec, recommended=False), old]
    cases = [
        [ms, {'first': True, 'last': 'first_trial'}],                # 前回のモデルが推奨でない → 推奨
        [ms, {'first': True, 'last': 'deleted_run'}],                # 前回のモデルが消えた → 推奨(一番新しいものではない)
        [ms, {'first': True}],                                       # 前回なし → 推奨
        [ms, {'first': True, 'last': 'dino-fp16'}],                  # 前回が確認に落ちたエンジン → 推奨
        [ms, {'first': True, 'last': 'first_trial', 'running': 'zoo-act-s1'}],   # 走行中のモデルが最優先
        [ms, {'first': False, 'cur': 'zoo-act-s1'}],                 # 2 回目以降は人の選択を残す
        [ms, {'first': False, 'cur': 'deleted_run'}],                # 選んでいたものが消えた → 推奨
        [ms, {'first': False, 'cur': 'dino-fp16'}],                  # 選べないもの → 推奨
        [no_rec, {'first': True, 'last': 'first_trial'}],            # 推奨なし → 今まで通り前回のモデル
        [no_rec, {'first': True, 'last': 'deleted_run'}],            # 推奨なし・前回なし → 一番新しい走れるもの
        [[dict(rec, drivable=False), newest, old], {'first': True, 'last': 'first_trial'}],  # 推奨が走れない → 前回
        [[bad], {'first': True, 'last': 'dino-fp16'}],               # 走れるものが無い → 選ばない
        [[], {'first': True}],
    ]
    assert _pick_cases(cases) == [B, B, B, B, 'zoo-act-s1', 'zoo-act-s1', B, B, 'first_trial', 'zoo-dinov2-s1',
                                  'first_trial', '', '']


def test_ui_uses_pick_and_shows_recommended_and_gate_reason():
    js = open(APP_JS, encoding='utf-8').read()
    assert 'pickDriveModel(state.drive.models,' in js                # loadDrive が同じ関数で選ぶ
    assert '★推奨' in js and 'recommended_reason' in js
    assert "m.drivable === false ? 'disabled' : ''" in js            # 走れないもの(精度の確認に落ちたエンジン)は選べない
    assert '走れません: ${esc(m.why_not' in js                       # 理由を一覧に出す
    assert '走れますが、モデルの比較では走行の候補から外しました' not in js   # 精度の不合格はもう「走れます」と出さない
