"""強化学習モード(離脱復活)のシーンの学習除外を、取り込み → シーンタブへの展開 → 作り直し → ジャンルの付け直し → 作り直し まで通すテスト(案)。

離脱復活 r3(2026-10-14、学習の段 recover r3)が昇格の段のために用意したもの(protocol_recover_r3.json label_mask_fixes_for_promote)。
確認 recover r2 の e2e_label_mask.py と同じ流れを、乱し v2 の形のシーン(出来事 = l = 0 の区間。立て直せる表で替わった組を含む)で確かめる:
  - 取り込んだデータセット・展開したシーンから作り直したデータセット・ジャンルを付け直してから作り直したデータセットのどれでも、
    学習に使う観測(future_valid かつ mode ≠ 0)の正解(k+1〜k+10)に l = 0 の操作が入らない
  - 外した観測(mode = 0)は付け直しのあとも同じ
  - r3/rc/promote/dataset_frame_index.patch を当てると、展開したシーンからの作り直しで正解が 1 フレームもずれない(future_actions[k] = vehicle[k+1..k+10])
  - 強化学習モードでないシーン(RC 教師)は外さない
"""

import json
import zipfile

import numpy as np
import pytest

from jetformer import config, dataset, dataset_import, modes

N = 180
EVENTS = [(40, 52), (95, 104), (140, 158)]          # 乱しの区間 [ps, pe)(v2 の替えた組を含む長さ)


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path / 'data')
    monkeypatch.setattr(config, 'DATASET_DIR', tmp_path / 'data' / 'datasets')
    monkeypatch.setattr(config, 'RAW_DIR', tmp_path / 'data' / 'raw')
    config.DATASET_DIR.mkdir(parents=True)
    config.RAW_DIR.mkdir(parents=True)
    # ジャンルの表も一時フォルダーに(本物の data/modes.json に触れない)。シミュレーターのジャンル 8 を足す
    mf = tmp_path / 'data' / 'modes.json'
    mf.write_text(json.dumps([dict(m) for m in modes.DEFAULT_MODES] + [{'id': 8, 'name': 'シミュレーター', 'description': ''}], ensure_ascii=False),
                  encoding='utf-8')
    monkeypatch.setattr(modes, 'MODES_FILE', mf)
    return tmp_path


def sim_t(n=N):
    return (np.arange(n) * 1e6 / 30).astype(np.int64)          # シミュレーターのフレーム: 33,333/33,334 µs おき


def steer_of(frame):
    return 1100 + 3 * frame                                     # 舵の値からフレームの番号が分かる(どれも整数の µs)


def make_source(base, name='sim-rl-recover_e2e'):
    root = base / name
    (root / 'aligned').mkdir(parents=True)
    (root / 'samples' / 'CAM_FRONT').mkdir(parents=True)
    scenes = {'rl1': 'rl', 'rl2': 'rl', 'rc1': 'rc'}
    card = {'name': name, 'scenes': len(scenes), 'frames': N * len(scenes), 'training_frames': (N - 10) * len(scenes), 'tags': ['simulator'],
            'mode_frames': {'0': 0, '8': N * len(scenes)}, 'modes': [{'id': 0, 'name': '未設定'}, {'id': 8, 'name': 'シミュレーター'}]}
    (root / 'dataset_card.json').write_text(json.dumps(card, ensure_ascii=False), encoding='utf-8')
    (root / 'splits.json').write_text(json.dumps({'train': list(scenes), 'val': []}), encoding='utf-8')
    t = sim_t()
    veh = np.zeros((N, 4), np.float32)
    veh[:, 0] = [steer_of(f) for f in range(N)]
    veh[:, 1] = 1500
    veh[:, 2] = 1000
    fut = np.zeros((N, 10, 2), np.float32)
    for k in range(N):
        for i in range(10):
            j = min(N - 1, k + 1 + i)
            fut[k, i] = veh[j, :2]
    fv = np.arange(N) < N - 10
    for s, kind in scenes.items():
        cams = []
        for i in range(N):
            rel = 'samples/CAM_FRONT/%s_%03d.jpg' % (s, i)
            (root / rel).write_bytes(b'\xff\xd8jpeg')
            cams.append(rel)
        np.savez(root / 'aligned' / ('%s.npz' % s), timestamps_us=t, cam_files=np.array(cams), tof=np.zeros((N, 42, 54), np.uint16),
                 imu=np.zeros((N, 10, 6), np.float32), vehicle=veh, vehicle_valid=np.ones(N, bool), vehicle_age_ms=np.zeros(N, np.float32),
                 future_actions=fut, future_valid=fv, mode=np.full(N, 8, np.int16))
        d = root / 'simulation' / 'scenes' / s
        d.mkdir(parents=True)
        if kind == 'rl':
            teacher = {'schema': 'jetformer-sim-rl-teacher/v1', 'id': 'jf-rl-recover-v1', 'mode': 'recover',
                       'perturbation': {'id': 'jf-perturb-record/v2'}}
            l0 = {f for a, b in EVENTS for f in range(a, b)}
            rows = [{'frame': f, 'action_info': {'jf': {'v': 1, 'm': 'recover', 'f': f, 'l': 0 if f in l0 else 1,
                                                       'ph': 'perturb' if f in l0 else 'drive'}}} for f in range(N)]
        else:
            teacher = {'schema': 'course-external-teacher-slot/v1', 'id': 'rc'}
            rows = [{'frame': f, 'action_info': {}} for f in range(N)]
        (d / 'session.json').write_text(json.dumps({'context': {'teacher': teacher}}), encoding='utf-8')
        (d / 'frames.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    return root


def zip_of(folder, path):
    with zipfile.ZipFile(path, 'w') as z:
        for f in sorted(folder.rglob('*')):
            if f.is_file():
                z.write(f, str(f.relative_to(folder.parent)))
    return path


def label_frames(fa):
    """future_actions の舵の値 → 正解のフレームの番号"""
    return np.rint((fa[..., 0] - 1100) / 3).astype(int)


def check_no_leak(name, scene, rl=True):
    with np.load(config.DATASET_DIR / name / 'aligned' / ('%s.npz' % scene)) as z:
        fv, md, fa = z['future_valid'].copy(), z['mode'].copy(), z['future_actions'].copy()
    use = fv & (md != 0)
    lf = label_frames(fa)
    if rl:
        bad = [(k, f) for k in np.nonzero(use)[0] for f in lf[k] if any(a <= f < b for a, b in EVENTS)]
        assert not bad, bad[:5]
    return fv, md, lf


def test_e2e_import_expand_rebuild_relabel(dirs):
    src = make_source(dirs / 'src')
    info = dataset_import.import_archive(zip_of(src, dirs / 'a.zip'), to_scenes=True)
    name = info['name']
    _, md0, _ = check_no_leak(name, 'rl1')
    masked = np.zeros(N, bool)
    for a, b in EVENTS:
        masked[a - 10:b] = True                                 # [ps − 10, pe − 1]
    assert np.array_equal(md0 == 0, masked)
    _, mdrc, _ = check_no_leak(name, 'rc1', rl=False)
    assert (mdrc == 8).all()
    # シーンタブに展開したシーン → 作り直し
    sc = sorted(s['name'] for s in dataset.list_scenes() if s['name'].startswith('ds-' + name))
    assert len(sc) == 3
    rl_scene = sc[0]
    dataset.build_dataset('rebuild1', sc, val_ratio=0.0)
    fv1, md1, lf1 = check_no_leak('rebuild1', rl_scene)
    assert np.array_equal(md1 == 0, masked)
    # 正解をフレームの番号で引く(dataset_frame_index.patch): 学習に使う観測の正解はちょうど k+1〜k+10
    use = fv1 & (md1 != 0)
    for k in np.nonzero(use)[0]:
        assert np.array_equal(lf1[k], np.arange(k + 1, k + 11)), (k, lf1[k])
    # ジャンルを付け直してから作り直し: 外した区間(未設定)は残る
    for s in sc:
        dataset.update_scene(s, mode_id=8)
    dataset.build_dataset('rebuild2', sc, val_ratio=0.0)
    fv2, md2, lf2 = check_no_leak('rebuild2', rl_scene)
    assert np.array_equal(md2 == 0, masked) and np.array_equal(fv2, fv1) and np.array_equal(lf2, lf1)


def test_e2e_rc_scene_rebuild_unchanged_by_patch(dirs):
    """RC 教師のシーン(強化学習モードでない)は、展開・作り直しで外す観測が無く、正解もフレームの番号どおり"""
    src = make_source(dirs / 'src')
    info = dataset_import.import_archive(zip_of(src, dirs / 'a.zip'), to_scenes=True)
    sc = sorted(s['name'] for s in dataset.list_scenes() if s['name'].startswith('ds-' + info['name']))
    dataset.build_dataset('rebuild_rc', [sc[2]], val_ratio=0.0)
    fv, md, lf = check_no_leak('rebuild_rc', sc[2], rl=False)
    assert (md == 8).all() and fv[:N - 10].all() and not fv[N - 10:].any()
