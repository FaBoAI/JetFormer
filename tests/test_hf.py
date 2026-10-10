"""Hugging Face への書き出し(Parquet・データセットカード・アップロード手順)のテスト。実際には上げない"""

import json
import sys
import types

import cv2
import numpy as np
import pytest

from jetformer import config, hf_export


def make_dataset(root, name='tiny', frames=(5, 3)):
    """splits の train/val に1シーンずつの、小さな nuScenes 形式データセットを作る"""
    d = root / name
    scenes = {'train': ['s-0001'], 'val': ['s-0002']}
    (d / ('v1.0-%s' % name)).mkdir(parents=True)
    (d / 'aligned').mkdir()
    table = []
    for (split, names), n in zip(scenes.items(), frames):
        for s in names:
            (d / 'samples' / 'CAM_FRONT' / s).mkdir(parents=True)
            files = []
            for i in range(n):
                f = 'samples/CAM_FRONT/%s/%d.jpg' % (s, i)
                cv2.imwrite(str(d / f), np.full((224, 224, 3), 40 * i, np.uint8))
                files.append(f)
            tof = np.full((n, 42, 54), 500, np.uint16)
            tof[:, 0, 0] = 0x1FFF
            np.savez(d / 'aligned' / (s + '.npz'), timestamps_us=np.arange(n, dtype=np.int64) * 33333,
                     cam_files=np.array(files), tof=tof, tof_age_ms=np.zeros(n, np.float32),
                     imu=np.zeros((n, 10, 6), np.float32), vehicle=np.tile([1450, 1500, 1000, 0], (n, 1)).astype(np.float32),
                     vehicle_age_ms=np.zeros(n, np.float32), vehicle_valid=np.ones(n, bool),
                     future_actions=np.full((n, 10, 2), 1500, np.float32), future_valid=np.ones(n, bool),
                     mode=np.full(n, 4, np.int16))
            table.append({'name': s, 'description': '説明 %s' % s, 'mode_id': 4, 'mode_name': '高速走行', 'duration_s': n / 30})
    (d / ('v1.0-%s' % name) / 'scene.json').write_text(json.dumps(table, ensure_ascii=False), encoding='utf-8')
    (d / 'splits.json').write_text(json.dumps(scenes))
    (d / 'dataset_card.json').write_text(json.dumps({
        'name': name, 'created_at': '2026-10-10T12:00:00', 'scenes': 2, 'frames': sum(frames),
        'modes': [{'id': 0, 'name': '未設定'}, {'id': 4, 'name': '高速走行'}], 'mode_frames': {'4': sum(frames)}},
        ensure_ascii=False), encoding='utf-8')
    return d


@pytest.fixture
def ds_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'DATASET_DIR', tmp_path / 'datasets')
    monkeypatch.setattr(config, 'EXPORT_DIR', tmp_path / 'exports')
    make_dataset(tmp_path / 'datasets')
    return tmp_path


def test_stage_builds_viewer_friendly_parquet(ds_dirs):
    import datasets
    out, counts = hf_export.stage('tiny', 'me/jetformer-tiny', include_nuscenes=False)
    assert counts == {'train': 5, 'validation': 3}
    ds = datasets.load_dataset('parquet', data_files={'train': str(out / 'data' / 'train-*.parquet'),
                                                      'validation': str(out / 'data' / 'validation-*.parquet')})
    r = ds['train'][2]
    assert r['image'].size == (224, 224) and r['tof_image'].size == (216, 168)
    assert r['mode_name'] == '高速走行' and r['description'] == '説明 s-0001' and r['frame_index'] == 2
    assert np.array(r['tof_raw']).shape == (42, 54) and r['tof_raw'][0][0] == 0x1FFF
    assert len(r['future_actions_us']) == 10 and len(r['imu'][0]) == 6
    card = (out / 'README.md').read_text(encoding='utf-8')
    assert 'path: data/train-*.parquet' in card and 'path: data/validation-*.parquet' in card
    assert 'me/jetformer-tiny' in card and 'nuscenes/' not in card


def test_upload_creates_repo_and_records_it(ds_dirs, monkeypatch):
    calls = []

    class FakeApi(object):
        def create_repo(self, repo_id, **kw):
            calls.append(('create', repo_id, kw))

        def upload_folder(self, **kw):
            import os
            files = sorted(os.path.relpath(os.path.join(r, f), kw['folder_path'])
                           for r, _, fs in os.walk(kw['folder_path']) for f in fs)
            calls.append(('upload', kw['repo_id'], kw['repo_type'], files, kw.get('delete_patterns')))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(HfApi=FakeApi))
    info = hf_export.upload('tiny', 'me/jetformer-tiny', private=True, include_nuscenes=True, license_id='cc-by-4.0')
    assert calls[0] == ('create', 'me/jetformer-tiny', {'repo_type': 'dataset', 'private': True, 'exist_ok': True})
    assert calls[1][2] == 'dataset' and calls[1][4] == ['data/*.parquet']
    assert calls[1][3] == ['README.md', 'data/train-00000-of-00001.parquet', 'data/validation-00000-of-00001.parquet',
                           'nuscenes/tiny.tar']
    rec = json.loads((config.DATASET_DIR / 'tiny' / 'hf_upload.json').read_text(encoding='utf-8'))
    assert rec['url'] == 'https://huggingface.co/datasets/me/jetformer-tiny' and rec['private'] and rec['license'] == 'cc-by-4.0'
    assert info['counts'] == {'train': 5, 'validation': 3}
    assert not (config.EXPORT_DIR / 'hf_tiny').exists()                # 作業用フォルダは消す


def test_repo_name_must_have_owner(ds_dirs):
    with pytest.raises(ValueError):
        hf_export.upload('tiny', 'no-owner')
