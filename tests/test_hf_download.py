"""Hugging Face からのダウンロード(hf_export.download / extract_dataset_tar)のテスト。Hub には接続しない"""

import hashlib
import io
import json
import tarfile

import pytest

from jetformer import config, dataset, hf_export


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path / 'data')
    monkeypatch.setattr(config, 'DATASET_DIR', tmp_path / 'data' / 'datasets')
    monkeypatch.setattr(config, 'RAW_DIR', tmp_path / 'data' / 'raw')        # シーンへの展開も一時フォルダに
    monkeypatch.setattr(config, 'EXPORT_DIR', tmp_path / 'data' / 'exports')
    config.DATASET_DIR.mkdir(parents=True)
    return tmp_path


def make_dataset(name):
    root = config.DATASET_DIR / name
    (root / 'aligned').mkdir(parents=True)
    (root / ('v1.0-%s' % name)).mkdir()
    (root / 'dataset_card.json').write_text(json.dumps({'name': name, 'scenes': 1}), encoding='utf-8')
    (root / 'splits.json').write_text(json.dumps({'train': ['s1'], 'val': []}), encoding='utf-8')
    (root / 'aligned' / 's1.npz').write_bytes(b'x' * 1000)
    (root / ('v1.0-%s' % name) / 'scene.json').write_text('[]', encoding='utf-8')
    return root


def publish(monkeypatch, tar_path, name, sha256=None):
    """Hub の代わりに、手元の tar を nuscenes/<name>.tar として見せる"""
    data = tar_path.read_bytes()
    files = [{'path': 'nuscenes/%s.tar' % name, 'name': name, 'size': len(data),
              'sha256': sha256 if sha256 is not None else hashlib.sha256(data).hexdigest(), 'exists': False}]
    monkeypatch.setattr(hf_export, 'hub_dataset_files', lambda repo_id, revision=None: [dict(f) for f in files])
    monkeypatch.setattr(hf_export, '_open_remote', lambda repo_id, path, revision=None: io.BytesIO(data))


def exported_tar(name, tmp_path):
    make_dataset(name)
    tar = dataset.export_dataset(name)
    copy = tmp_path / ('%s_copy.tar' % name)
    copy.write_bytes(tar.read_bytes())
    dataset.delete_dataset(name)                     # Jetson にはまだ無い状態にする
    return copy


def test_download_extracts_dataset_and_records_source(dirs, monkeypatch):
    tar = exported_tar('tiny_ds', dirs)
    publish(monkeypatch, tar, 'tiny_ds')
    steps = []
    info = hf_export.download('FaBoAI/jetformer-tiny_ds', progress=lambda p, m='': steps.append(p))
    root = config.DATASET_DIR / 'tiny_ds'
    assert (root / 'dataset_card.json').exists() and (root / 'aligned' / 's1.npz').read_bytes() == b'x' * 1000
    assert info['name'] == 'tiny_ds' and info['repo_id'] == 'FaBoAI/jetformer-tiny_ds' and info['bytes'] == tar.stat().st_size
    src = json.loads((root / 'hf_source.json').read_text(encoding='utf-8'))
    assert src['repo_id'] == 'FaBoAI/jetformer-tiny_ds'
    assert steps[-1] == 1.0
    assert not list((config.DATA_DIR / 'downloads').iterdir())          # 途中のファイルは残さない
    assert not [p for p in config.DATASET_DIR.iterdir() if p.name.startswith('.download-')]
    listed = {d['name']: d for d in dataset.list_datasets()}
    assert listed['tiny_ds']['hf_source']['repo_id'] == 'FaBoAI/jetformer-tiny_ds'
    assert 'scenes' in info                                  # シーンへの展開を試みた(この小さな形では npz が無いので注意だけ)


def test_download_refuses_existing_dataset(dirs, monkeypatch):
    tar = exported_tar('tiny_ds', dirs)
    publish(monkeypatch, tar, 'tiny_ds')
    make_dataset('tiny_ds')
    marker = config.DATASET_DIR / 'tiny_ds' / 'mine.txt'
    marker.write_text('keep', encoding='utf-8')
    with pytest.raises(FileExistsError):
        hf_export.download('FaBoAI/jetformer-tiny_ds')
    assert marker.read_text(encoding='utf-8') == 'keep'


def test_download_rejects_hash_mismatch(dirs, monkeypatch):
    tar = exported_tar('tiny_ds', dirs)
    publish(monkeypatch, tar, 'tiny_ds', sha256='0' * 64)
    with pytest.raises(IOError):
        hf_export.download('FaBoAI/jetformer-tiny_ds')
    assert not (config.DATASET_DIR / 'tiny_ds').exists()
    assert not list((config.DATA_DIR / 'downloads').iterdir())


def test_download_without_training_tar(dirs, monkeypatch):
    monkeypatch.setattr(hf_export, 'hub_dataset_files', lambda repo_id, revision=None: [])
    with pytest.raises(ValueError):
        hf_export.download('someone/parquet-only')


def _tar_with(path, members):
    with tarfile.open(path, 'w') as tar:
        for name, data in members:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
    return path


@pytest.mark.parametrize('bad', ['tiny_ds/../evil.txt', 'other_ds/dataset_card.json', '/abs/evil.txt'])
def test_extract_rejects_unsafe_members(dirs, bad):
    tar = _tar_with(dirs / 'bad.tar', [('tiny_ds/dataset_card.json', b'{}'), ('tiny_ds/splits.json', b'{}'), (bad, b'x')])
    with pytest.raises(ValueError):
        hf_export.extract_dataset_tar(tar, 'tiny_ds')
    assert not (config.DATASET_DIR / 'tiny_ds').exists()
    assert not (dirs / 'evil.txt').exists() and not (config.DATASET_DIR / 'evil.txt').exists()


def test_extract_rejects_links(dirs):
    path = dirs / 'link.tar'
    with tarfile.open(path, 'w') as tar:
        for name in ('tiny_ds/dataset_card.json', 'tiny_ds/splits.json'):
            ti = tarfile.TarInfo(name)
            ti.size = 2
            tar.addfile(ti, io.BytesIO(b'{}'))
        ln = tarfile.TarInfo('tiny_ds/passwd')
        ln.type = tarfile.SYMTYPE
        ln.linkname = '/etc/passwd'
        tar.addfile(ln)
    with pytest.raises(ValueError):
        hf_export.extract_dataset_tar(path, 'tiny_ds')
    assert not (config.DATASET_DIR / 'tiny_ds').exists()


def test_extract_requires_card_and_splits(dirs):
    tar = _tar_with(dirs / 'nocard.tar', [('tiny_ds/aligned/s1.npz', b'x')])
    with pytest.raises(ValueError):
        hf_export.extract_dataset_tar(tar, 'tiny_ds')
    assert not (config.DATASET_DIR / 'tiny_ds').exists()


@pytest.mark.parametrize('repo', ['', 'noslash', 'a/b/c', '/b', 'a/', 'a/../b', 'a b/c'])
def test_repo_id_check(repo):
    with pytest.raises(ValueError):
        hf_export._check_repo_id(repo)


@pytest.mark.parametrize('name', ['', '.hidden', 'a/b', 'a b', '../x'])
def test_name_check(name):
    with pytest.raises(ValueError):
        hf_export._check_name(name)


def test_iter_dataset_tar_round_trip(dirs):
    """その場で作る tar(ブラウザへのダウンロード)が、export_dataset の tar と同じ中身で、取り込みにも使えること"""
    root = make_dataset('tiny_ds')
    (root / 'samples' / 'CAM_FRONT').mkdir(parents=True)
    long_name = 'x' * 150 + '.jpg'                                   # 100 文字を超える名前(PAX)
    (root / 'samples' / 'CAM_FRONT' / long_name).write_bytes(bytes(range(256)) * 50)
    stream = b''.join(dataset.iter_dataset_tar('tiny_ds', chunk=1024))
    exported = dataset.export_dataset('tiny_ds').read_bytes()

    def contents(data):
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            return {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    assert contents(stream) == contents(exported)
    assert 'tiny_ds/samples/CAM_FRONT/' + long_name in contents(stream)
    assert dataset.dataset_bytes(root) == sum(len(v) for v in contents(stream).values())
    path = dirs / 'stream.tar'
    path.write_bytes(stream)
    dataset.delete_dataset('tiny_ds')
    hf_export.extract_dataset_tar(path, 'tiny_ds')                   # PC から戻すときも同じ形で展開できる
    assert (config.DATASET_DIR / 'tiny_ds' / 'samples' / 'CAM_FRONT' / long_name).exists()


@pytest.mark.parametrize('bad', ['../etc', 'a/b', 'missing'])
def test_iter_dataset_tar_rejects_bad_names(dirs, bad):
    with pytest.raises(KeyError):
        next(dataset.iter_dataset_tar(bad))
