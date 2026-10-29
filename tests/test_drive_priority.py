"""走行タブの「優先のモデル」のテスト(2026-10-29。ユーザーの指示「セレクトボックスの上から 5 つ目までは優先モデルを並べる」)。
CPU だけ・サーバーは起動しない

- data/drive_priority.json(推奨のモデルと同じフォルダ)に書いたモデルを、書いた順に一覧の先頭に並べる(5 つまで)。
  番号 ①〜⑤・名前・理由を付ける。そのあとに推奨のモデル、残りは新しい順(前と同じ)
- ファイルが無い・壊れている・1 つも無いときは、前とまったく同じ一覧(新しい項目を付けない)
- 画面(app.js の fillDriveModels)は、①〜⑤ と名前を選ぶ欄の先に出す(node で確かめる。node が無ければ飛ばす)

    cd ~/jetformer && python -m pytest -q tests/test_drive_priority.py
"""

import json
import os
import re
import shutil
import subprocess

import pytest

from jetformer import drive

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
APP_JS = os.path.join(ROOT, 'jetformer', 'web', 'static', 'app.js')

B = 'clean_driving_20261011-nocmd-s3-20261011-024626'
NAMES = [('first_trial', '2026-10-10T15:36:29'), (B, '2026-10-11T02:46:26'), ('slow-sec-s1', '2026-10-23 04:06:41'),
         ('ens-p2', '2026-10-21 11:16:39'), ('v3d-s1', '2026-10-26 23:13:03'), ('v3n-s1', '2026-10-27 00:24:59'),
         ('v3b-s3', '2026-10-27 08:39:20')]


def _run(runs, name, created_at):
    d = runs / name
    d.mkdir(parents=True)
    (d / drive.ENGINE_FILE).write_bytes(b'engine-' + name.encode())
    (d / 'model_config.json').write_text(json.dumps({'num_modes': 6, 'trained_modes': [4, 5],
                                                     'steer_center_us': 1441.0, 'throttle_neutral_us': 1496.0}))
    (d / 'run.json').write_text(json.dumps({'created_at': created_at}))


@pytest.fixture
def runs(monkeypatch, tmp_path):
    r = tmp_path / 'runs'
    r.mkdir()
    monkeypatch.setattr(drive, 'RUNS_DIR', r)
    monkeypatch.setattr(drive, 'RECOMMENDED_PATH', tmp_path / 'drive_recommended.json')
    for name, at in NAMES:
        _run(r, name, at)
    return r


def _old_list_models():
    """直す前の list_models(推奨を先頭・残りは新しい順)"""
    models = [m for m in (drive.model_info(d.name) for d in drive.RUNS_DIR.iterdir()) if m]
    models.sort(key=lambda m: (m['created_at'] or '', m['name']), reverse=True)
    rec = drive.recommended()
    for m in models:
        m['recommended'] = bool(rec) and m['name'] == rec['run']
        if m['recommended']:
            m['recommended_reason'] = rec.get('reason') or ''
    return [m for m in models if m['recommended']] + [m for m in models if not m['recommended']]


FIVE = [{'run': B, 'label': 'モデルB(基準)', 'reason': '今の標準'},
        {'run': 'v3d-s1', 'label': 'V1', 'reason': '学習 v3 の候補 1'},
        {'run': 'v3n-s1', 'label': 'V2', 'reason': '学習 v3 の候補 2'},
        {'run': 'slow-sec-s1', 'label': '遅く入る型', 'reason': ''},
        {'run': 'ens-p2', 'label': '2本の平均', 'reason': ''}]


def test_no_file_is_exactly_the_same_as_before(runs):
    assert drive.priority() is None
    assert drive._priority_path() == drive.RECOMMENDED_PATH.with_name('drive_priority.json')
    assert drive.list_models() == _old_list_models()
    drive.save_recommended(B, '評価で決めた')
    ms = drive.list_models()
    assert ms == _old_list_models() and ms[0]['name'] == B
    assert not any('priority' in m for m in ms)


@pytest.mark.parametrize('text', ['{broken', json.dumps({'models': []}), json.dumps({'models': 'x'}),
                                  json.dumps([1, 2]), json.dumps({'models': [{'run': ''}, {'label': 'x'}, 3]})])
def test_broken_or_empty_file_is_ignored(runs, text):
    drive.save_recommended(B, '評価で決めた')
    drive._priority_path().write_text(text)
    assert drive.priority() is None
    assert drive.list_models() == _old_list_models()


def test_priority_models_come_first_in_written_order(runs):
    drive.save_recommended(B, '評価で決めた')
    pri = drive.save_priority(FIVE, title='10/29 の試験で確かめる 5 つ', reason='ユーザーの指示')
    assert drive.priority() == {'title': pri['title'], 'reason': 'ユーザーの指示', 'decided_at': pri['decided_at'],
                                'models': FIVE}
    ms = drive.list_models()
    assert [m['name'] for m in ms] == [B, 'v3d-s1', 'v3n-s1', 'slow-sec-s1', 'ens-p2', 'v3b-s3', 'first_trial']
    assert [m['priority'] for m in ms[:5]] == [1, 2, 3, 4, 5]
    assert [m['priority_mark'] for m in ms[:5]] == list('①②③④⑤')
    assert [m['priority_label'] for m in ms[:5]] == [x['label'] for x in FIVE]
    assert ms[1]['priority_reason'] == '学習 v3 の候補 1' and ms[1]['priority_title'] == '10/29 の試験で確かめる 5 つ'
    assert not any('priority' in m for m in ms[5:])
    assert ms[0]['recommended'] is True and ms[0]['recommended_reason'] == '評価で決めた'
    old = {m['name']: m for m in _old_list_models()}             # 足したのは priority の項目だけ(ほかの値は同じ)
    for m in ms:
        assert {k: v for k, v in m.items() if not k.startswith('priority')} == old[m['name']]


def test_recommended_not_in_priority_comes_right_after(runs):
    drive.save_recommended('first_trial', '古い推奨')
    drive.save_priority(FIVE[1:3])
    ms = drive.list_models()
    assert [m['name'] for m in ms][:4] == ['v3d-s1', 'v3n-s1', 'first_trial', 'v3b-s3']
    assert ms[2]['recommended'] is True and 'priority' not in ms[2]


def test_missing_runs_are_skipped_and_numbers_close_up(runs):
    rows = [{'run': 'deleted_run', 'label': '消えた'}, {'run': 'v3n-s1', 'label': 'V2'},
            {'run': 'v3n-s1', 'label': '同じ名前の 2 つ目'}, {'run': 'ens-p2', 'label': '2本の平均'}]
    drive._priority_path().write_text(json.dumps({'models': rows}, ensure_ascii=False))
    pri = drive.priority()
    assert [x['run'] for x in pri['models']] == ['deleted_run', 'v3n-s1', 'ens-p2']   # 同じ名前の 2 つ目は捨てる
    ms = drive.list_models()
    assert [(m['name'], m['priority'], m['priority_mark']) for m in ms[:2]] == [('v3n-s1', 1, '①'), ('ens-p2', 2, '②')]
    assert 'deleted_run' not in [m['name'] for m in ms]


def test_more_than_five_in_the_file_keeps_the_first_five(runs):
    rows = [{'run': n} for n, _ in NAMES]
    drive._priority_path().write_text(json.dumps({'models': rows}))
    assert [x['run'] for x in drive.priority()['models']] == [n for n, _ in NAMES][:5]
    ms = drive.list_models()
    assert [m.get('priority') for m in ms] == [1, 2, 3, 4, 5, None, None]


@pytest.mark.parametrize('bad', [[], [{'run': n} for n, _ in NAMES[:6]], [{'run': '../x'}], [{'run': 'a/b'}],
                                 [{'run': ''}], ['B'], [{'run': B}, {'run': B}]])
def test_save_priority_rejects_bad_lists(runs, bad):
    with pytest.raises(ValueError):
        drive.save_priority(bad)
    assert not drive._priority_path().exists()


def test_save_priority_writes_the_file_next_to_recommended(runs):
    drive.save_priority(FIVE, title='t')
    p = drive.RECOMMENDED_PATH.parent / 'drive_priority.json'
    assert p.exists() and not (p.parent / 'drive_priority.json.tmp').exists()
    assert json.loads(p.read_text(encoding='utf-8'))['models'] == FIVE


# ---------------------------------------------------------------- 画面(app.js の fillDriveModels)

def _fill(models, rendered):
    js = open(APP_JS, encoding='utf-8').read()
    m = re.search(r'^function fillDriveModels\(\) \{.*?^}\n', js, re.S | re.M)
    assert m, 'app.js に fillDriveModels がありません'
    stub = """
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const sel = { value: '', innerHTML: '' };
const $ = (q) => sel;
const state = { drive: { models: %s, st: null, restored: true } };
const pickDriveModel = (ms) => ms.length ? ms[0].name : '';
const renderDriveModel = () => {};
""" % json.dumps(models, ensure_ascii=False)
    code = stub + m.group(0) + '\nfillDriveModels();\nprocess.stdout.write(JSON.stringify(sel.innerHTML));'
    r = subprocess.run(['node', '-e', code], capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@pytest.mark.skipif(not shutil.which('node'), reason='node がありません')
def test_ui_shows_marks_and_labels_first(runs):
    drive.save_recommended(B, '評価で決めた')
    drive.save_priority(FIVE, title='10/29 の試験で確かめる 5 つ')
    ms = drive.list_models()
    t = ms[0]['policy_title']                                       # 「TransFuser軽量型(JetFormer v0・今の標準)」など
    html = _fill(ms, True)
    opts = re.findall(r'<option value="([^"]*)"[^>]*>(.*?)</option>', html)
    assert [v for v, _ in opts][:5] == [x['run'] for x in FIVE]
    assert opts[0][1].startswith('①モデルB(基準) ★推奨 「%s」%s' % (t, B))
    assert opts[1][1].startswith('②V1 「%s」v3d-s1' % ms[1]['policy_title'])
    assert opts[4][1].startswith('⑤2本の平均 ')
    assert not opts[5][1].startswith(tuple('①②③④⑤'))


@pytest.mark.skipif(not shutil.which('node'), reason='node がありません')
def test_ui_without_priority_is_the_same_text_as_before(runs):
    drive.save_recommended(B, '評価で決めた')
    ms = drive.list_models()
    html = _fill(ms, True)
    first = re.findall(r'<option value="([^"]*)"[^>]*>(.*?)</option>', html)[0]
    assert first == (B, '★推奨 「%s」%s' % (ms[0]['policy_title'], B))


def test_ui_info_shows_priority_reason():
    js = open(APP_JS, encoding='utf-8').read()
    assert 'm.priority_mark' in js and 'm.priority_label' in js and 'm.priority_reason' in js and 'm.priority_title' in js
    assert '★推奨のモデル' in js                                    # 推奨の表示は前のまま残す
