"""シミュレーターで集めたデータセットの一覧・取り込み・削除(jetformer/sim/datasets.py と /api/sim/datasets)のテスト"""

import json
import os
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jetformer import config, dataset_import
from jetformer.sim import datasets, paths, web

SID = 'a' * 32


def publish(sid=SID, name='sim-run_20261013_aaaaaaaa', collection=False, frames=600, size_override=None):
    d = paths.runtime_dir() / 'datasets' / sid
    d.mkdir(parents=True, exist_ok=True)
    z = d / (name + '.zip')
    with zipfile.ZipFile(z, 'w') as zf:
        zf.writestr(name + '/dataset_card.json', json.dumps({'name': name, 'frames': frames, 'scenes': 3 if collection else 1}))
        zf.writestr(name + '/aligned/s.npz', b'x' * 100)
    info = {'schema': datasets.SCHEMA, 'session_id': sid, 'filename': z.name, 'sha256': 'x', 'bytes': size_override or z.stat().st_size}
    (d / 'archive-info.json').write_text(json.dumps(info))
    if collection:
        (d / 'collection-summary.json').write_text(json.dumps({'scenes': 3, 'frames': 1800, 'duration_s': 60}))
    (d / '_frames').mkdir(exist_ok=True)
    (d / '_frames' / '000001.npz').write_bytes(b'y' * 50)
    return z


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path / 'data')
    monkeypatch.setattr(config, 'DATASET_DIR', tmp_path / 'data' / 'datasets')
    config.DATASET_DIR.mkdir(parents=True)
    return tmp_path


def test_list_finished_archives(data):
    publish()
    publish('b' * 32, name='coll_1', collection=True)
    (paths.runtime_dir() / 'datasets' / ('c' * 32)).mkdir()                       # まだ保存していない(archive-info なし)
    (paths.runtime_dir() / 'datasets' / ('c' * 32) / 'x.zip.part').write_bytes(b'p' * 10)
    (paths.runtime_dir() / 'datasets' / 'not-a-session').mkdir()
    r = datasets.list_sim_datasets(active_session='c' * 32)
    by = {d['session_id']: d for d in r['datasets']}
    assert set(by) == {SID, 'b' * 32}
    assert by[SID]['kind'] == 'scene' and by[SID]['dataset_name'] == 'sim-run_20261013_aaaaaaaa' and by[SID]['frames'] == 600
    assert by[SID]['duration_s'] == 20.0 and by[SID]['scenes'] == 1 and by[SID]['imported'] is False
    assert by['b' * 32]['kind'] == 'collection' and by['b' * 32]['duration_s'] == 60 and by['b' * 32]['frames'] == 1800
    assert r['incomplete'] == [{'session_id': 'c' * 32, 'bytes': 10, 'recording': True}]
    assert r['free_gb'] is not None


def test_archive_checks(data):
    z = publish()
    assert datasets.archive_path(SID)[0] == z
    publish('d' * 32, size_override=1)                                               # 大きさが合わない
    with pytest.raises(datasets.DatasetError):
        datasets.archive_path('d' * 32)
    for bad in ('../x', 'A' * 32, 'a' * 31):
        with pytest.raises(datasets.DatasetError):
            datasets.archive_path(bad)
    e = paths.runtime_dir() / 'datasets' / ('e' * 32)
    os.symlink(z.parent, e)                                                          # シンボリックリンクは見ない
    with pytest.raises(datasets.DatasetError):
        datasets.archive_path('e' * 32)
    info = json.loads((z.parent / 'archive-info.json').read_text())
    info['filename'] = '../escape.zip'
    (z.parent / 'archive-info.json').write_text(json.dumps(info))
    with pytest.raises(datasets.DatasetError):
        datasets.archive_path(SID)


def test_imported_flag(data):
    z = publish()
    d = config.DATASET_DIR / 'sim-run_20261013_aaaaaaaa'
    d.mkdir()
    (d / 'import_source.json').write_text(json.dumps({'file': z.name, 'bytes': z.stat().st_size}))
    assert datasets.list_sim_datasets()['datasets'][0]['imported'] is True


def test_delete_rules(data):
    publish()
    with pytest.raises(datasets.DatasetError) as e:
        datasets.delete(SID, recording_sid=SID)
    assert e.value.status == 409
    with pytest.raises(datasets.DatasetError) as e:
        datasets.delete(SID, in_flight=1)
    assert e.value.status == 409
    datasets.delete(SID)
    assert not (paths.runtime_dir() / 'datasets' / SID).exists()
    with pytest.raises(datasets.DatasetError) as e:
        datasets.delete(SID)
    assert e.value.status == 404


class Jobs(object):
    def __init__(self):
        self.submitted = []

    def submit(self, kind, title, fn):
        self.submitted.append((kind, title))
        return {'kind': kind, 'title': title, 'result': fn(lambda p, m='': None)}


def test_import_api_uses_existing_import_job(data, monkeypatch):
    z = publish()
    calls = []

    def fake_import(path, simulator=True, source_name=None, progress=None, to_scenes=True):
        calls.append((path, simulator, source_name))
        return {'name': 'sim-run_20261013_aaaaaaaa'}
    monkeypatch.setattr(dataset_import, 'import_archive', fake_import)
    jobs = Jobs()
    web.init(jobs=jobs)
    app = FastAPI()
    app.include_router(web.router)
    c = TestClient(app)
    r = c.post('/api/sim/datasets/%s/import' % SID)
    assert r.status_code == 200 and r.json()['kind'] == 'import'
    assert calls == [(z, True, z.name)]                       # 写しを作らずにそのまま
    assert 'シミュレーターから取り込み' in jobs.submitted[0][1]
    assert c.post('/api/sim/datasets/%s/import' % ('9' * 32)).status_code == 404
    lst = c.get('/api/sim/datasets').json()
    assert lst['datasets'][0]['session_id'] == SID and lst['in_flight'] == 0
    monkeypatch.setattr(web.shutil, 'disk_usage', lambda p: type('U', (), {'free': 10})())
    r = c.post('/api/sim/datasets/%s/import' % SID)
    assert r.status_code == 507
    assert c.delete('/api/sim/datasets/%s' % SID).json() == {'ok': True}


class LaterJobs(object):
    """submit してもすぐには動かさない(動いている途中の状態を作る)"""

    def __init__(self):
        self.pending = []

    def submit(self, kind, title, fn):
        self.pending.append(fn)
        return {'kind': kind, 'title': title, 'state': 'running'}


def test_import_twice_and_delete_while_importing_are_409(data, monkeypatch):
    """review で追加: 同じデータセットの「取り込む」を 2 回押しても 2 本目は動かない。取り込み中は作業フォルダーを消せない"""
    publish()
    monkeypatch.setattr(dataset_import, 'import_archive', lambda *a, **k: {'name': 'x'})
    jobs = LaterJobs()
    web.init(jobs=jobs)
    app = FastAPI()
    app.include_router(web.router)
    c = TestClient(app)
    assert c.post('/api/sim/datasets/%s/import' % SID).status_code == 200
    r = c.post('/api/sim/datasets/%s/import' % SID)
    assert r.status_code == 409 and '取り込んでいます' in r.json()['detail']
    r = c.delete('/api/sim/datasets/%s' % SID)
    assert r.status_code == 409
    assert len(jobs.pending) == 1
    jobs.pending.pop()(lambda p, m='': None)                  # 取り込みが終わる
    assert c.post('/api/sim/datasets/%s/import' % SID).status_code == 200
    jobs.pending.pop()(lambda p, m='': None)
    assert c.delete('/api/sim/datasets/%s' % SID).json() == {'ok': True}


def test_init_sweeps_stale_temp_files(data):
    """review で追加: 電源断などで残ったモデル受け渡しの ZIP・自己診断のフォルダーは、起動時に(古ければ)片付ける"""
    import time
    d = paths.sim_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / '.bundle-deadbeef.zip').write_bytes(b'x' * 10)
    (d / 'selftest-abc').mkdir()
    (d / '.bundle-fresh000.zip').write_bytes(b'y')
    old = time.time() - 3600
    for n in ('.bundle-deadbeef.zip', 'selftest-abc'):
        os.utime(d / n, (old, old))
    web.init(jobs=Jobs())
    assert sorted(os.listdir(d)) == ['.bundle-fresh000.zip']
