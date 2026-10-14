"""シミュレーターの強化学習モードのシーンの学習除外(jetformer/sim/label_mask.py と dataset_import.import_archive)のテスト

乱した操作(frames.jsonl の action_info.jf.l = 0)を正解に含む観測を future_valid = False・mode = 0 にすること、
jf の無い行・frames.jsonl の無いシーンを外すこと(安全側)、強化学習モードでないシーンを変えないことを確かめる。
"""

import json
import zipfile

import numpy as np
import pytest

from jetformer import config, dataset_import
from jetformer.sim import label_mask

N = 60


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path / 'data')
    monkeypatch.setattr(config, 'DATASET_DIR', tmp_path / 'data' / 'datasets')
    monkeypatch.setattr(config, 'RAW_DIR', tmp_path / 'data' / 'raw')
    config.DATASET_DIR.mkdir(parents=True)
    return tmp_path


def rl_teacher(mode='recover'):
    return {'schema': label_mask.TEACHER_SCHEMA, 'id': 'jf-rl-%s-v1' % mode, 'mode': mode}


def make_dataset(base, scenes, name='sim_rl_recover_test'):
    """scenes: {シーン名: {'teacher': dict|None, 'rows': [frames.jsonl の行]|None}}"""
    root = base / name
    (root / 'aligned').mkdir(parents=True)
    (root / 'samples' / 'CAM_FRONT').mkdir(parents=True)
    card = {'name': name, 'scenes': len(scenes), 'frames': N * len(scenes), 'training_frames': (N - 10) * len(scenes),
            'mode_frames': {'0': 0, '8': N * len(scenes)}, 'modes': [{'id': 0, 'name': '未設定'}, {'id': 8, 'name': 'シミュレーター'}]}
    (root / 'dataset_card.json').write_text(json.dumps(card, ensure_ascii=False), encoding='utf-8')
    (root / 'splits.json').write_text(json.dumps({'train': list(scenes), 'val': []}), encoding='utf-8')
    for s, spec in scenes.items():
        cams = []
        for i in range(N):
            rel = 'samples/CAM_FRONT/%s_%03d.jpg' % (s, i)
            (root / rel).write_bytes(b'\xff\xd8jpeg')
            cams.append(rel)
        fv = np.ones(N, bool)
        fv[-10:] = False
        np.savez(root / 'aligned' / ('%s.npz' % s), timestamps_us=np.arange(N, dtype=np.int64) * 33333, cam_files=np.array(cams),
                 tof=np.zeros((N, 42, 54), np.uint16), imu=np.zeros((N, 10, 6), np.float32),
                 vehicle=np.tile(np.array([1442, 1450, 1500, -3000], np.float32), (N, 1)), vehicle_valid=np.ones(N, bool),
                 future_actions=np.full((N, 10, 2), 1500, np.float32), future_valid=fv, mode=np.full(N, 8, np.int16))
        d = root / 'simulation' / 'scenes' / s
        d.mkdir(parents=True)
        if spec.get('teacher') is not None:
            (d / 'session.json').write_text(json.dumps({'context': {'teacher': spec['teacher']}}), encoding='utf-8')
        if spec.get('rows') is not None:
            (d / 'frames.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in spec['rows']), encoding='utf-8')
    return root


def rows_with(l0=(), missing=(), drop=(), mode='recover'):
    out = []
    for f in range(N):
        if f in drop:
            continue
        info = {} if f in missing else {'jf': {'v': 1, 'm': mode, 'f': f, 'l': 0 if f in l0 else 1}}
        out.append({'frame': f, 'action_info': info})
    return out


def zip_of(folder, path):
    with zipfile.ZipFile(path, 'w') as z:
        for f in sorted(folder.rglob('*')):
            if f.is_file():
                z.write(f, str(f.relative_to(folder.parent)))
    return path


def load(name, scene):
    with np.load(config.DATASET_DIR / name / 'aligned' / ('%s.npz' % scene)) as z:
        return z['future_valid'].copy(), z['mode'].copy()


def test_mask_rule_interval():
    learn = np.ones(600, int)
    learn[100:110] = 0                                              # 乱しの区間 [ps, pe) = [100, 110)
    # k ∈ [ps − 10, pe − 1](2026-10-14 統合の段: シーンからの作り直しで正解が 1 フレーム前にずれても乱した操作が入らないよう 1 つ広げた)
    assert np.array_equal(np.nonzero(label_mask.mask_from_learn(learn))[0], np.arange(90, 110))


def test_learn_is_fail_closed():
    rows = [{'frame': 0, 'action_info': {'jf': {'l': 1}}}, {'frame': 1, 'action_info': {'jf': {'l': 0}}},
            {'frame': 2, 'action_info': None}, {'frame': 3}, {'frame': 4, 'action_info': {'jf': {'l': 'x'}}},
            {'frame': 5, 'action_info': {'jf': {}}}, {'frame': 6, 'action_info': {'jf': {'l': True}}}]
    learn, st = label_mask.learn_from_rows(rows, 9)                  # フレーム 7・8 は行が無い
    assert learn.tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert st['missing_jf'] == 2 and st['bad_l'] == 3 and st['missing_frames'] == 2 and st['l0'] == 1


def test_import_masks_rl_scene_and_keeps_others(dirs):
    src = make_dataset(dirs / 'src', {
        'rl1': {'teacher': rl_teacher(), 'rows': rows_with(l0=range(20, 25), missing=(40,))},
        'rc1': {'teacher': {'schema': 'course-external-teacher-slot/v1', 'id': 'rc'}, 'rows': [{'frame': f, 'action_info': {}} for f in range(N)]},
    })
    info = dataset_import.import_archive(zip_of(src, dirs / 'a.zip'), to_scenes=False)
    name = info['name']
    fv, md = load(name, 'rl1')
    masked = np.zeros(N, bool)
    masked[10:25] = True                                            # l = 0 の 20〜24 → k 10〜24
    masked[30:41] = True                                            # jf の無いフレーム 40 → k 30〜40(安全側)
    assert np.array_equal(md == 0, masked)
    assert np.array_equal(fv, ~masked & (np.arange(N) < N - 10))    # 最後の 10 フレームの False は元のまま
    fv2, md2 = load(name, 'rc1')                                    # 強化学習モードでないシーンは変えない
    assert (md2 == 8).all() and fv2[:N - 10].all() and not fv2[N - 10:].any()
    card = json.loads((config.DATASET_DIR / name / 'dataset_card.json').read_text(encoding='utf-8'))
    lm = card['label_mask']
    assert lm['schema'] == 'jetformer-sim-label-mask/v1' and lm['horizon'] == 10 and lm['fail_closed'] is True
    assert lm['masked_frames'] == 26 and list(lm['per_scene']) == ['rl1']
    assert lm['per_scene']['rl1']['missing_jf'] == 1 and lm['per_scene']['rl1']['l0'] == 5
    assert card['mode_frames']['0'] == 26 and card['mode_frames']['8'] == 2 * N - 26
    assert card['training_frames'] == 2 * (N - 10) - (26 - 0)       # 外した 26 はどれも最後の 10 フレームの外
    assert info['summary']['usable_frames'] == 2 * (N - 10) - 26 and info['summary']['label_masked_frames'] == 26
    assert set(info['tags']) >= {'simulator', 'sim-rl', 'sim-rl-recover'}    # 強化学習モードのタグ(2026-10-14 統合の段)


def test_rl_scene_without_frames_is_fully_masked(dirs):
    src = make_dataset(dirs / 'src', {'rl1': {'teacher': rl_teacher(), 'rows': None},
                                      'rl2': {'teacher': rl_teacher(), 'rows': rows_with()}})
    info = dataset_import.import_archive(zip_of(src, dirs / 'a.zip'), to_scenes=False)
    fv, md = load(info['name'], 'rl1')
    assert not fv.any() and (md == 0).all()                         # frames.jsonl が無い → 全部外す
    fv2, md2 = load(info['name'], 'rl2')
    assert (md2 == 8).all() and fv2[:N - 10].all()                  # 乱しの無い強化学習モードのシーンは外さない
    assert any('rl1' in w for w in info['warnings'])


def test_jf_without_session_and_missing_rows(dirs):
    src = make_dataset(dirs / 'src', {'rl1': {'teacher': None, 'rows': rows_with(drop=(50,))},
                                      'rc1': {'teacher': None, 'rows': None}})
    info = dataset_import.import_archive(zip_of(src, dirs / 'a.zip'), to_scenes=False)
    fv, md = load(info['name'], 'rl1')
    assert np.array_equal(np.nonzero(md == 0)[0], np.arange(40, 51))   # jf があるので強化学習モード。行の無い 50 → k 40〜50
    fv2, md2 = load(info['name'], 'rc1')
    assert (md2 == 8).all()


def test_rl_only_dataset_without_labels_is_rejected(dirs):
    src = make_dataset(dirs / 'src', {'rl1': {'teacher': rl_teacher(), 'rows': None}})
    with pytest.raises(ValueError):
        dataset_import.import_archive(zip_of(src, dirs / 'a.zip'), to_scenes=False)
    assert not any(config.DATASET_DIR.iterdir())                    # 半端なデータセットは残らない
