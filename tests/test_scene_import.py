"""データセットのシーンをシーンタブの形に展開する(jetformer/scene_import.py)のテスト"""

import json

import cv2
import numpy as np
import pytest

from jetformer import config, dataset, scene_import


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path / 'data')
    monkeypatch.setattr(config, 'DATASET_DIR', tmp_path / 'data' / 'datasets')
    monkeypatch.setattr(config, 'RAW_DIR', tmp_path / 'data' / 'raw')
    config.DATASET_DIR.mkdir(parents=True)
    config.RAW_DIR.mkdir(parents=True)
    return tmp_path


def make_dataset(name='sim_run_01', scenes=('s1', 's2'), n=45, sim=True):
    """aligned と画像だけのデータセット(シミュレーターが作る形)。操作値は時間で変わる"""
    root = config.DATASET_DIR / name
    (root / 'aligned').mkdir(parents=True)
    card = {'name': name, 'scenes': len(scenes), 'frames': n * len(scenes),
            'modes': [{'id': 0, 'name': '未設定'}, {'id': 4, 'name': '高速走行'}, {'id': 5, 'name': '低速走行'}],
            'tags': ['simulator'] if sim else [], 'source': 'simulator' if sim else 'real'}
    (root / 'dataset_card.json').write_text(json.dumps(card, ensure_ascii=False), encoding='utf-8')
    (root / 'splits.json').write_text(json.dumps({'train': list(scenes[:-1]), 'val': list(scenes[-1:])}), encoding='utf-8')
    for si, s in enumerate(scenes):
        (root / 'samples' / 'CAM_FRONT').mkdir(parents=True, exist_ok=True)
        t = 1_000_000 * (si + 1) + np.arange(n, dtype=np.int64) * 33333
        cams = []
        for i in range(n):
            rel = 'samples/CAM_FRONT/%s_%03d.jpg' % (s, i)
            cv2.imwrite(str(root / rel), np.full((224, 224, 3), (i * 5) % 255, np.uint8))
            cams.append(rel)
        steer = 1500 + 200 * np.sin(np.arange(n) / 7.0)
        thr = np.full(n, 1400.0)
        veh = np.stack([steer, thr, np.full(n, 2000.0), np.full(n, -4000.0)], axis=1).astype(np.float32)
        fut = np.stack([veh[np.clip(np.arange(n) + k + 1, 0, n - 1), :2] for k in range(10)], axis=1)
        imu = np.zeros((n, 10, 6), np.float32)
        imu[:, :, 1] = 1.5
        imu[:, :, 5] = 0.8                                            # rad/s(左回り)
        mode = np.where(np.arange(n) < n // 2, 5, 4).astype(np.int16)
        tof = np.full((n, 42, 54), (4 << 13) | 1200, np.uint16)
        np.savez(root / 'aligned' / ('%s.npz' % s), timestamps_us=t, cam_files=np.array(cams), tof=tof,
                 tof_age_ms=np.full(n, 5.0, np.float32), imu=imu, vehicle=veh, vehicle_valid=np.ones(n, bool),
                 vehicle_age_ms=np.full(n, 3.0, np.float32), future_actions=fut.astype(np.float32),
                 future_valid=np.r_[np.ones(n - 10, bool), np.zeros(10, bool)], mode=mode)
    return root


def test_expand_creates_viewable_scenes(dirs):
    make_dataset()
    info = scene_import.dataset_to_scenes('sim_run_01')
    assert info['log'] == 'ds-sim_run_01' and info['created'] == ['ds-sim_run_01-0001', 'ds-sim_run_01-0002']
    names = [s['name'] for s in dataset.list_scenes()]
    assert names == info['created']
    meta = dataset.load_scene_meta('ds-sim_run_01-0001')
    assert meta['tags'] == ['simulator'] and meta['source'] == 'simulator' and meta['complete']
    assert meta['imported_from'] == {'dataset': 'sim_run_01', 'scene': 's1'}
    assert [sg['mode_id'] for sg in meta['mode_segments']] == [5, 4] and meta['mode_id'] in (4, 5)
    series = dataset.scene_timeseries('ds-sim_run_01-0001')               # シーン詳細のグラフ
    assert series['frames'] == 45 and 'vehicle' in series and 'imu' in series and 'tof' in series
    img, depth, ts = dataset.scene_frame('ds-sim_run_01-0001', 10)       # 1フレームの表示
    assert img is not None and img.shape == (224, 224, 3) and depth is not None
    d = dataset.scene_path('ds-sim_run_01-0001')
    assert (d / 'thumb.jpg').exists()
    src = config.DATASET_DIR / 'sim_run_01' / 'samples' / 'CAM_FRONT' / 's1_000.jpg'
    assert (d / 'CAM_FRONT').stat().st_dev == src.stat().st_dev
    assert json.loads((config.DATASET_DIR / 'sim_run_01' / 'dataset_card.json').read_text(encoding='utf-8'))['scene_log'] == 'ds-sim_run_01'
    listed = {x['name']: x for x in dataset.list_datasets()}
    assert listed['sim_run_01']['scene_count'] == 2


def test_imu_and_vehicle_round_trip_through_build_dataset(dirs):
    """展開したシーンから作り直したデータセットの入力と正解が、元のデータセットとほぼ同じ"""
    make_dataset(scenes=('s1',), n=60)
    scene_import.dataset_to_scenes('sim_run_01')
    dataset.build_dataset('rebuilt', ['ds-sim_run_01-0001'], val_ratio=0.0, min_duration=0.0)
    a = np.load(config.DATASET_DIR / 'rebuilt' / 'aligned' / 'ds-sim_run_01-0001.npz')
    b = np.load(config.DATASET_DIR / 'sim_run_01' / 'aligned' / 's1.npz')
    assert len(a['timestamps_us']) == len(b['timestamps_us'])
    assert np.allclose(a['imu'][20:], b['imu'][20:], atol=1e-4)            # gyro は deg/s → rad/s で戻る
    assert np.allclose(a['vehicle'][5:, :2], b['vehicle'][5:, :2], atol=1.0)
    ok = a['future_valid'] & b['future_valid']
    assert ok.sum() > 30 and np.abs(a['future_actions'][ok] - b['future_actions'][ok]).max() <= 1.0
    assert (a['tof'] == b['tof']).all()
    assert set(np.unique(a['mode'])) == {4, 5}
    card = json.loads((config.DATASET_DIR / 'rebuilt' / 'dataset_card.json').read_text(encoding='utf-8'))
    assert card['source'] == 'simulator' and 'simulator' in card['tags']      # シーンのタグから自動で


def test_expand_again_replaces_and_skips_local_scenes(dirs):
    make_dataset()
    scene_import.dataset_to_scenes('sim_run_01')
    info = scene_import.dataset_to_scenes('sim_run_01')                      # 作り直し
    assert len(info['created']) == 2 and len(dataset.list_scenes()) == 2
    make_dataset('mine', scenes=('20261010-200916-0001', 'x2'), sim=False)
    (config.RAW_DIR / '20261010-200916' / '20261010-200916-0001').mkdir(parents=True)
    (config.RAW_DIR / '20261010-200916' / '20261010-200916-0001' / 'scene.json').write_text(
        json.dumps({'name': '20261010-200916-0001', 'log': '20261010-200916'}), encoding='utf-8')
    info = scene_import.dataset_to_scenes('mine')
    assert info['skipped'] == ['20261010-200916-0001'] and info['created'] == ['ds-mine-0002']
    assert dataset.load_scene_meta('ds-mine-0002')['source'] == 'dataset'


def test_expand_refuses_foreign_log_and_bad_names(dirs):
    make_dataset()
    (config.RAW_DIR / 'ds-sim_run_01').mkdir()
    with pytest.raises(FileExistsError):
        scene_import.dataset_to_scenes('sim_run_01')
    for bad in ('../x', 'a/b', 'missing'):
        with pytest.raises(KeyError):
            scene_import.dataset_to_scenes(bad)


def test_build_dataset_uses_a_temporary_folder(dirs, monkeypatch):
    """データセットは .building-<名前> に作ってから名前を付ける(途中で止まっても半端なものが一覧に出ない)"""
    make_dataset(scenes=('s1',), n=40)
    scene_import.dataset_to_scenes('sim_run_01')
    seen = []
    real_replace = dataset.os.replace

    def spy(src, dst):
        seen.append((str(src), str(dst)))
        return real_replace(src, dst)
    monkeypatch.setattr(dataset.os, 'replace', spy)
    dataset.build_dataset('rebuilt2', ['ds-sim_run_01-0001'], val_ratio=0.0, min_duration=0.0)
    assert any(src.endswith('.building-rebuilt2') and dst.endswith('/rebuilt2') for src, dst in seen)
    assert (config.DATASET_DIR / 'rebuilt2' / 'dataset_card.json').stat().st_size > 0
    assert not (config.DATASET_DIR / '.building-rebuilt2').exists()
