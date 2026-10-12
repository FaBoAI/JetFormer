"""別の環境(シミュレーターなど)で作ったデータセットの取り込み(jetformer/dataset_import.py・tools/upload_datasets.py)のテスト"""

import importlib.util
import io
import json
import os
import tarfile
import zipfile

import numpy as np
import pytest

from jetformer import config, dataset, dataset_import

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path / 'data')
    monkeypatch.setattr(config, 'DATASET_DIR', tmp_path / 'data' / 'datasets')
    config.DATASET_DIR.mkdir(parents=True)
    return tmp_path


def make_sim_dataset(base, name='sim_run_01', n=6, drop=None, mode=True):
    """シミュレーターが作った形のデータセット(学習が読むものだけ)"""
    root = base / name
    (root / 'aligned').mkdir(parents=True)
    (root / 'samples' / 'CAM_FRONT').mkdir(parents=True)
    card = {'name': name, 'scenes': 1, 'frames': n, 'modes': [{'id': 0, 'name': '未設定'}, {'id': 5, 'name': '低速走行'}]}
    (root / 'dataset_card.json').write_text(json.dumps(card, ensure_ascii=False), encoding='utf-8')
    (root / 'splits.json').write_text(json.dumps({'train': ['s1'], 'val': []}), encoding='utf-8')
    cams = []
    for i in range(n):
        rel = 'samples/CAM_FRONT/s1_%03d.jpg' % i
        (root / rel).write_bytes(b'\xff\xd8jpeg')
        cams.append(rel)
    arrays = {'timestamps_us': np.arange(n, dtype=np.int64) * 33333, 'cam_files': np.array(cams),
              'tof': np.zeros((n, 42, 54), np.uint16), 'imu': np.zeros((n, 10, 6), np.float32),
              'vehicle': np.tile(np.array([1500, 1450, 2000, -3000], np.float32), (n, 1)),
              'vehicle_valid': np.ones(n, bool), 'future_actions': np.full((n, 10, 2), 1500, np.float32),
              'future_valid': np.ones(n, bool)}
    if mode:
        arrays['mode'] = np.full(n, 5, np.int16)
    for k in drop or []:
        arrays.pop(k)
    np.savez(root / 'aligned' / 's1.npz', **arrays)
    return root


def tar_of(folder, path, top=True, gz=False):
    with tarfile.open(path, 'w:gz' if gz else 'w') as t:
        for f in sorted(folder.rglob('*')):
            if f.is_file():
                rel = f.relative_to(folder.parent if top else folder)
                t.add(f, arcname=str(rel))
    return path


def zip_of(folder, path, mac=False):
    with zipfile.ZipFile(path, 'w') as z:
        for f in sorted(folder.rglob('*')):
            if f.is_file():
                rel = str(f.relative_to(folder.parent))
                z.write(f, rel)
                if mac:                                                        # Finder の「圧縮」が入れるもの
                    z.writestr('__MACOSX/' + os.path.dirname(rel) + '/._' + f.name, b'apple')
        if mac:
            z.writestr(folder.name + '/.DS_Store', b'ds')
    return path


@pytest.mark.parametrize('kind', ['tar', 'targz', 'zip', 'zip_mac', 'tar_no_top'])
def test_import_formats(dirs, kind):
    src = make_sim_dataset(dirs / 'src')
    arch = {'tar': lambda: tar_of(src, dirs / 'a.tar'), 'targz': lambda: tar_of(src, dirs / 'a.tar.gz', gz=True),
            'zip': lambda: zip_of(src, dirs / 'a.zip'), 'zip_mac': lambda: zip_of(src, dirs / 'a.zip', mac=True),
            'tar_no_top': lambda: tar_of(src, dirs / 'a.tar', top=False)}[kind]()
    steps = []
    info = dataset_import.import_archive(arch, simulator=True, progress=lambda p, m='': steps.append(p))
    root = config.DATASET_DIR / 'sim_run_01'
    assert info['name'] == 'sim_run_01' and info['source'] == 'simulator' and 'simulator' in info['tags']
    assert (root / 'aligned' / 's1.npz').exists() and not list(root.rglob('._*')) and not (root / '.DS_Store').exists()
    assert json.loads((root / 'import_source.json').read_text(encoding='utf-8'))['simulator'] is True
    assert info['summary']['usable_frames'] == 6 and steps[-1] == 1.0
    assert not [p for p in config.DATASET_DIR.iterdir() if p.name.startswith('.import-')]
    listed = {d['name']: d for d in dataset.list_datasets()}
    assert listed['sim_run_01']['source'] == 'simulator' and listed['sim_run_01']['import_source']['file'] == arch.name


def test_import_as_real(dirs):
    src = make_sim_dataset(dirs / 'src')
    info = dataset_import.import_archive(tar_of(src, dirs / 'a.tar'), simulator=False)
    assert info['source'] == 'real' and info['tags'] == []


def test_import_rejects_missing_arrays_and_leaves_nothing(dirs):
    src = make_sim_dataset(dirs / 'src', drop=['future_actions'])
    with pytest.raises(ValueError, match='future_actions'):
        dataset_import.import_archive(tar_of(src, dirs / 'a.tar'))
    assert not list(config.DATASET_DIR.iterdir())


def test_import_warns_without_mode(dirs):
    src = make_sim_dataset(dirs / 'src', mode=False)
    with pytest.raises(ValueError, match='学習に使えるフレームがありません'):      # mode が無いと全部「未設定」で使えない
        dataset_import.import_archive(tar_of(src, dirs / 'a.tar'))
    errors, warnings, _ = dataset_import.validate_dataset(src)
    assert any('mode' in w for w in warnings)


def test_import_rejects_name_conflict(dirs):
    src = make_sim_dataset(dirs / 'src')
    arch = tar_of(src, dirs / 'a.tar')
    dataset_import.import_archive(arch)
    with pytest.raises(FileExistsError):
        dataset_import.import_archive(arch)


def test_import_rejects_two_datasets_in_one_file(dirs):
    a = make_sim_dataset(dirs / 'src', 'one')
    make_sim_dataset(dirs / 'src', 'two')
    with tarfile.open(dirs / 'both.tar', 'w') as t:
        t.add(a.parent, arcname='.')
    with pytest.raises(ValueError, match='1つ'):
        dataset_import.import_archive(dirs / 'both.tar')


@pytest.mark.parametrize('bad', ['sim_run_01/../../evil.txt', '/etc/evil.txt'])
def test_import_rejects_unsafe_paths(dirs, bad):
    src = make_sim_dataset(dirs / 'src')
    arch = tar_of(src, dirs / 'a.tar')
    with tarfile.open(arch, 'a') as t:
        ti = tarfile.TarInfo(bad)
        ti.size = 1
        t.addfile(ti, io.BytesIO(b'x'))
    with pytest.raises(ValueError):
        dataset_import.import_archive(arch)
    assert not (dirs / 'evil.txt').exists() and not list(config.DATASET_DIR.iterdir())


def test_import_rejects_links(dirs):
    src = make_sim_dataset(dirs / 'src')
    arch = tar_of(src, dirs / 'a.tar')
    with tarfile.open(arch, 'a') as t:
        ln = tarfile.TarInfo('sim_run_01/passwd')
        ln.type = tarfile.SYMTYPE
        ln.linkname = '/etc/passwd'
        t.addfile(ln)
    with pytest.raises(ValueError, match='リンク'):
        dataset_import.import_archive(arch)


def test_import_rejects_non_archive(dirs):
    p = dirs / 'x.tar'
    p.write_bytes(b'not an archive')
    with pytest.raises(ValueError):
        dataset_import.import_archive(p)


def _uploader():
    spec = importlib.util.spec_from_file_location('upload_datasets', os.path.join(HERE, '..', 'tools', 'upload_datasets.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_uploader_folder_tar_round_trip(dirs):
    """tools/upload_datasets.py がフォルダからその場で作る tar を、そのまま取り込めること(._* や .DS_Store は入れない)"""
    up = _uploader()
    src = make_sim_dataset(dirs / 'src')
    (src / '.DS_Store').write_bytes(b'ds')
    (src / 'aligned' / '._s1.npz').write_bytes(b'apple')
    assert up.check_folder(str(src)) == []
    data = b''.join(up.folder_chunks(str(src), chunk=4096))
    p = dirs / 'stream.tar'
    p.write_bytes(data)
    with tarfile.open(p) as t:
        names = t.getnames()
    assert all(n.startswith('sim_run_01/') for n in names) and not [n for n in names if '._' in n or 'DS_Store' in n]
    info = dataset_import.import_archive(p)
    assert info['name'] == 'sim_run_01'


def test_uploader_check_folder_reports_missing(dirs):
    up = _uploader()
    src = make_sim_dataset(dirs / 'src')
    os.remove(src / 'aligned' / 's1.npz')
    assert any('s1.npz' in x for x in up.check_folder(str(src)))
