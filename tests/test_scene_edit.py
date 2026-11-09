"""シーンの分割と、範囲の削除(jetformer/scene_edit.py・web/server.py・app.js。2026-11-07)。作り物のシーンだけで確かめる(本物の data/ は触らない)。

- 分割: 前は元の名前・後ろは「元の名前.ミリ秒」。カメラの絵とセンサーのサンプルは 1 つも落とさず、重ならない。絵は消えない
- 範囲の削除: 始め・終わり・途中。途中なら前と後ろの 2 つのシーン(時刻の穴を作らない)。消した絵はディスクから消える
- 断る場合(シーン全部・範囲の外・知らないファイル・音声・同じ名前)。サーバーは収集中・走行中・確認なしを断る
- 分けたシーンからデータセットを作ると、中ほどのフレームの正解・入力が元のシーンと同じ
- 画面の範囲の計算(node)
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from jetformer import config, dataset, scene_edit as E

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'jetformer' / 'web' / 'static'
NODE = shutil.which('node')
LOG = '20261108-111500'
FRAME_US = 33333


def make_scene(raw, name=LOG + '-0001', n=600, t0=1_800_000_000_000_000, ai=False, segs=None, extra=None):
    """30 fps のカメラ n 枚・100 Hz の IMU と車両・30 Hz の ToF の作り物のシーン"""
    d = Path(raw) / name.rsplit('-', 1)[0] / name
    (d / 'CAM_FRONT').mkdir(parents=True)
    cam_t = t0 + np.arange(n, dtype=np.int64) * FRAME_US
    files = np.array(['CAM_FRONT/%d.jpg' % t for t in cam_t])                 # 本物と同じく、シーンのフォルダからの場所
    for i, f in enumerate(files):
        (d / f).write_bytes(b'JPEG-%06d' % i)
    np.savez(d / 'CAM_FRONT.npz', timestamps_us=cam_t, seq=np.arange(n), files=files)
    tof_t = cam_t + 5000
    np.savez_compressed(d / 'TOF_FRONT.npz', timestamps_us=tof_t, seq=np.arange(n), depth=np.full((n, 42, 54), 800, np.uint16))
    m = int((cam_t[-1] - t0) / 10000) + 30
    it = t0 - 50000 + np.arange(m, dtype=np.int64) * 10000
    rng = np.random.default_rng(0)
    np.savez(d / 'IMU.npz', timestamps_us=it, accel=rng.normal(0, 1, (m, 3)).astype(np.float32),
             gyro=rng.normal(0, 1, (m, 3)).astype(np.float32), mag=np.zeros((m, 3), np.float32), temp=np.zeros(m, np.float32))
    pwm = np.stack([1500 + 300 * np.sin(np.arange(m) / 50.0), 1400 + 50 * np.cos(np.arange(m) / 30.0), np.full(m, 975)], 1).astype(np.int32)
    np.savez(d / 'VEHICLE.npz', timestamps_us=it, pwm=pwm, valid=np.ones((m, 3), bool), select=np.zeros(m, bool),
             moving=np.ones(m, bool), erpm=np.full(m, -3000, np.int32), seq=np.arange(m), quality=np.array('snapshot'),
             vbat_v=np.full(m, 8.1, np.float32), vbat_scale=np.array(4.3, np.float32))
    if ai:
        np.savez(d / 'AI_CMD.npz', timestamps_us=cam_t + 2000, pred=np.zeros((n, 10, 2), np.float32), mode_id=np.full(n, 4, np.int16))
    dur = round((cam_t[-1] - cam_t[0]) / 1e6, 2)
    sj = {'token': 'tok', 'name': name, 'log': name.rsplit('-', 1)[0], 'description': 'd', 'tags': ['x'], 'mode_id': 6,
          'mode_segments': segs or [{'start_s': 0.0, 'end_s': dur, 'mode_id': 6}], 'start_us': int(t0 - 20000),
          'end_us': int(cam_t[-1]), 'duration_s': dur, 'complete': True, 'counts': {}, 'rates': {}, 'stats': {'s': 1},
          'vehicle_quality': 'snapshot', 'drive': ({'run': 'm'} if ai else None)}
    (d / 'scene.json').write_text(json.dumps(sj, ensure_ascii=False), encoding='utf-8')
    (d / 'thumb.jpg').write_bytes(b'THUMB')
    lj = d.parent / 'log.json'
    if not lj.exists():
        lj.write_text(json.dumps({'token': 'log-' + sj['log'], 'logfile': sj['log'], 'vehicle': 'jetformer-01', 'date_captured': '2026-11-08',
                                  'location': '場所', 'mode_id': 6, 'drive': None}, ensure_ascii=False), encoding='utf-8')
    for f in extra or []:
        (d / f).write_bytes(b'x')
    return d, cam_t


@pytest.fixture
def env(tmp_path, monkeypatch):
    raw, data = tmp_path / 'data' / 'raw', tmp_path / 'data'
    raw.mkdir(parents=True)
    monkeypatch.setattr(config, 'RAW_DIR', raw)
    monkeypatch.setattr(config, 'DATA_DIR', data)
    monkeypatch.setattr(config, 'DATASET_DIR', data / 'datasets')
    return raw, data


def npz(d, s):
    with np.load(Path(d) / ('%s.npz' % s), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sj(d):
    return json.loads((Path(d) / 'scene.json').read_text(encoding='utf-8'))


def test_split_keeps_every_frame_and_sample(env):
    raw, data = env
    d, cam_t = make_scene(raw, ai=True)
    before = {s: npz(d, s) for s in ('CAM_FRONT', 'TOF_FRONT', 'IMU', 'VEHICLE', 'AI_CMD')}
    inode = {f: os.stat(d / 'CAM_FRONT' / f).st_ino for f in os.listdir(d / 'CAM_FRONT')}
    r = E.split_scene(LOG + '-0001', 157)
    off = int(round((cam_t[157] - cam_t[0]) / 1000))
    assert r['scenes'] == [LOG + '-0001', '%s-0001.%06d' % (LOG, off)] and r['deleted_frames'] == 0
    p1, p2 = raw / LOG / r['scenes'][0], raw / LOG / r['scenes'][1]
    for s, arr in before.items():
        a, b = npz(p1, s), npz(p2, s)
        t = np.concatenate([a['timestamps_us'], b['timestamps_us']])
        assert np.array_equal(t, arr['timestamps_us']), s                  # 1 つも落とさず、重ならない
        for k in arr:
            if arr[k].ndim == 0:
                assert a[k] == arr[k] and b[k] == arr[k]                   # 0 次元(quality・vbat_scale)はそのまま
            else:
                assert np.array_equal(np.concatenate([a[k], b[k]]), arr[k]), (s, k)
    assert np.array_equal(npz(p2, 'CAM_FRONT')['timestamps_us'], cam_t[157:])
    assert all(os.stat(p / 'CAM_FRONT' / f).st_ino == inode[f] for p in (p1, p2) for f in os.listdir(p / 'CAM_FRONT'))
    assert len(os.listdir(p1 / 'CAM_FRONT')) + len(os.listdir(p2 / 'CAM_FRONT')) == 600
    m1, m2 = sj(p1), sj(p2)
    assert (m1['counts']['CAM_FRONT'], m2['counts']['CAM_FRONT']) == (157, 443)
    assert m1['start_us'] == int(cam_t[0] - 20000) and m2['start_us'] == int(cam_t[157])     # 前は元の始め・後ろは最初の絵の時刻
    assert m2['duration_s'] == round((cam_t[-1] - cam_t[157]) / 1e6, 2) and m2['token'] != 'tok' and m1['token'] not in ('tok', m2['token'])
    assert m2['mode_segments'][0]['start_s'] == 0.0 and m2['mode_segments'][0]['mode_id'] == 6
    assert m1['complete'] is False and m2['edit']['op'] == 'split' and m2['edit_origin']['name'] == LOG + '-0001'
    assert (p1 / 'thumb.jpg').read_bytes() == b'THUMB' and (p2 / 'thumb.jpg').read_bytes() == b'JPEG-%06d' % (157 + 29)
    assert not (data / '.edit').exists() or not any((data / '.edit').iterdir())          # 作業のフォルダは残さない
    assert [s['name'] for s in dataset.list_scenes()] == r['scenes']                      # 名前の順 = 時刻の順
    assert dataset.scene_path(r['scenes'][1]) == p2                                       # 点の入った名前でも引ける


def test_delete_head_tail_middle(env):
    raw, data = env
    d, cam_t = make_scene(raw)
    jpg = sorted(os.listdir(d / 'CAM_FRONT'))
    r = E.delete_range(LOG + '-0001', 0, 59)                                  # 始めの 60 枚
    assert r['scenes'] == [LOG + '-0001'] and r['deleted_frames'] == 60
    assert np.array_equal(npz(d, 'CAM_FRONT')['timestamps_us'], cam_t[60:])
    assert npz(d, 'IMU')['timestamps_us'].min() >= cam_t[60] and npz(d, 'VEHICLE')['timestamps_us'].min() >= cam_t[60]
    assert not any((d / 'CAM_FRONT' / f).exists() for f in jpg[:60]) and all((d / 'CAM_FRONT' / f).exists() for f in jpg[60:])
    assert sj(d)['start_us'] == int(cam_t[60])
    r = E.delete_range(LOG + '-0001', 500, 539)                               # 終わりの 40 枚(今のシーンの番号)
    assert r['scenes'] == [LOG + '-0001'] and np.array_equal(npz(d, 'CAM_FRONT')['timestamps_us'], cam_t[60:560])
    assert npz(d, 'VEHICLE')['timestamps_us'].max() < cam_t[560]
    r = E.delete_range(LOG + '-0001', 100, 199)                               # 途中 → 前と後ろの 2 つ
    off = int(round((cam_t[60 + 200] - cam_t[0]) / 1000))                      # 名前は最初の元のシーンの始めから
    assert r['scenes'] == [LOG + '-0001', '%s-0001.%06d' % (LOG, off)]
    a, b = raw / LOG / r['scenes'][0], raw / LOG / r['scenes'][1]
    assert np.array_equal(npz(a, 'CAM_FRONT')['timestamps_us'], cam_t[60:160])
    assert np.array_equal(npz(b, 'CAM_FRONT')['timestamps_us'], cam_t[260:560])
    for p in (a, b):                                                           # どちらのシーンにも時刻の穴が無い
        assert np.diff(npz(p, 'CAM_FRONT')['timestamps_us']).max() == FRAME_US
        assert np.diff(npz(p, 'IMU')['timestamps_us']).max() == 10000
    gone = set(jpg) - set(os.listdir(a / 'CAM_FRONT')) - set(os.listdir(b / 'CAM_FRONT'))
    assert len(gone) == 60 + 40 + 100 and not any((d / 'CAM_FRONT' / f).exists() for f in gone if (d / 'CAM_FRONT').exists())
    v = np.concatenate([npz(a, 'VEHICLE')['timestamps_us'], npz(b, 'VEHICLE')['timestamps_us']])
    assert not np.any((v >= cam_t[160]) & (v < cam_t[260]))                   # 消した範囲のサンプルは無い
    assert len(sj(b)['edit_history']) == 3


def test_segments_shift_and_second_split_names(env):
    raw, data = env
    segs = [{'start_s': 0.0, 'end_s': 5.0, 'mode_id': 6}, {'start_s': 5.002, 'end_s': 8.0, 'mode_id': 0},
            {'start_s': 8.002, 'end_s': 19.97, 'mode_id': 1}]
    d, cam_t = make_scene(raw, segs=segs)
    r = E.split_scene(LOG + '-0001', 180)                                      # 6.0 秒
    m2 = sj(raw / LOG / r['scenes'][1])
    t180 = (cam_t[180] - cam_t[0]) / 1e6
    assert [(g['mode_id'], g['start_s'], g['end_s']) for g in m2['mode_segments']] == [
        (0, 0.0, round(8.0 - t180, 3)), (1, round(8.002 - t180, 3), m2['duration_s'])]
    assert m2['mode_id'] == 0                                                    # 元の主(6)が後ろに無いので、最初の区間のジャンル
    cam2 = npz(raw / LOG / r['scenes'][1], 'CAM_FRONT')['timestamps_us']
    fm = dataset._frame_modes(m2, cam2)                                        # 学習のフレームのジャンルが元の時刻と同じ
    fm0 = dataset._frame_modes(sj(d) | {'mode_segments': segs, 'mode_id': 6}, cam_t)
    assert np.array_equal(fm, fm0[180:])
    r2 = E.split_scene(r['scenes'][1], 120)                                    # 後ろの部分をさらに分ける(10 秒の所)
    off = int(round((cam_t[300] - cam_t[0]) / 1000))
    assert r2['scenes'] == [r['scenes'][1], '%s-0001.%06d' % (LOG, off)]
    names = [s['name'] for s in dataset.list_scenes()]
    assert names == sorted(names) == [LOG + '-0001', r['scenes'][1], r2['scenes'][1]]


def test_refuses(env):
    raw, data = env
    d, cam_t = make_scene(raw, n=200)
    for kw in ({'a': 0, 'b': 199}, {'a': -1, 'b': 5}, {'a': 10, 'b': 200}, {'a': 10, 'b': 150}, {'a': 40, 'b': 185}):
        with pytest.raises(E.EditError):                   # 全部・外・残る部分が 30 枚より短い(前 10 枚・後ろ 14 枚)
            E.delete_range(LOG + '-0001', kw['a'], kw['b'])
    for i in (0, 10, 171, 200):
        with pytest.raises(E.EditError):
            E.split_scene(LOG + '-0001', i)
    with pytest.raises(KeyError):
        E.split_scene(LOG + '-0009', 50)
    with pytest.raises(KeyError):
        E.split_scene('../' + LOG + '-0001', 50)
    make_scene(raw, name=LOG + '-0002', n=100, extra=['AUDIO.npz'])
    with pytest.raises(E.EditError):
        E.split_scene(LOG + '-0002', 50)
    make_scene(raw, name=LOG + '-0003', n=100, extra=['note.txt'])
    with pytest.raises(E.EditError):
        E.split_scene(LOG + '-0003', 50)
    r = E.split_scene(LOG + '-0001', 100)
    nm = '%s-0001.%06d' % (LOG, int(round((cam_t[50] - cam_t[0]) / 1000)))
    (raw / LOG / nm).mkdir()                                                   # 分けると同じ名前ができる → 断る
    with pytest.raises(E.EditError):
        E.split_scene(LOG + '-0001', 50)
    assert np.array_equal(npz(raw / LOG / (LOG + '-0001'), 'CAM_FRONT')['timestamps_us'], cam_t[:100])   # 断ったら元のまま
    assert sorted(os.listdir(raw / LOG / nm)) == [] and not any((data / '.edit').iterdir())


def test_stale_and_lock(env):
    raw, data = env
    d, cam_t = make_scene(raw, n=200)
    tok = sj(d)['token']
    with pytest.raises(E.StaleError):
        E.delete_range(LOG + '-0001', 0, 59, expect_token='other')
    with pytest.raises(E.StaleError):
        E.delete_range(LOG + '-0001', 0, 59, expect_token=tok, expect_frames=199)
    r = E.delete_range(LOG + '-0001', 0, 59, expect_token=tok, expect_frames=200)
    with pytest.raises(E.StaleError):                                         # 同じ頼みの 2 度目(古い画面)は断る
        E.delete_range(LOG + '-0001', 0, 59, expect_token=tok, expect_frames=200)
    assert len(npz(d, 'CAM_FRONT')['timestamps_us']) == 140
    E._LOCK.acquire()
    try:
        with pytest.raises(E.EditError):                                      # ほかの分割・削除の最中は断る
            E.split_scene(LOG + '-0001', 70)
    finally:
        E._LOCK.release()


def test_ai_cmd_missing_in_a_part_is_not_written(env):
    raw, data = env
    d, cam_t = make_scene(raw, n=200, ai=True)
    a = npz(d, 'AI_CMD')
    keep = a['timestamps_us'] < cam_t[80]                                     # AI の指令は前の 80 枚の間だけ
    np.savez(d / 'AI_CMD.npz', **{k: v[keep] for k, v in a.items()})
    r = E.split_scene(LOG + '-0001', 100)
    back = raw / LOG / r['scenes'][1]
    assert not (back / 'AI_CMD.npz').exists() and 'AI_CMD' not in sj(back)['counts']
    assert dataset.scene_timeseries(r['scenes'][1])['frames'] == 100            # 開ける(空のファイルで 500 にならない)


def test_recover_pending(env):
    raw, data = env
    d, cam_t = make_scene(raw, n=200)
    # (1) 元をよけた直後に止まった: raw に元の名前が無い → 元に戻す
    st = data / '.edit' / 'x1'
    st.mkdir(parents=True)
    os.rename(d, st / '__before__')
    assert E.recover_pending() == ['元に戻した: %s' % (LOG + '-0001')] and (d / 'scene.json').exists() and not st.exists()
    # (2) 前の部分は raw に入り、後ろの部分は作業のフォルダに残ったまま止まった → 入れ替えを終える
    r = E.split_scene(LOG + '-0001', 100)
    back = raw / LOG / r['scenes'][1]
    st = data / '.edit' / 'x2'
    st.mkdir(parents=True)
    os.rename(back, st / back.name)
    shutil.copytree(d, st / '__before__', copy_function=os.link)
    out = E.recover_pending()
    assert out == ['入れ替えを終えた: %s' % back.name, '元を消した: %s' % (LOG + '-0001')] and back.exists() and not st.exists()
    # (3) 作りかけ(元は raw にそのまま) → 作業のフォルダを消すだけ
    st = data / '.edit' / 'x3' / (LOG + '-0001')
    st.mkdir(parents=True)
    assert E.recover_pending() == ['作りかけを消した: x3'] and d.exists() and not (data / '.edit' / 'x3').exists()


def test_swap_and_rollback_both_fail_keep_the_original(env, monkeypatch):
    raw, data = env
    d, cam_t = make_scene(raw, n=200)
    real = os.rename
    calls = []

    def flaky(src, dst):
        calls.append((str(src), str(dst)))
        if len(calls) in (3, 4):                    # 3 回目 = 後ろの部分を raw へ・4 回目 = 戻し(前の部分を作業のフォルダへ)
            raise OSError('作り物の失敗')
        return real(src, dst)
    monkeypatch.setattr(E.os, 'rename', flaky)
    with pytest.raises(E.EditError) as ei:
        E.split_scene(LOG + '-0001', 100)
    assert '__before__' in str(ei.value)
    stages = list((data / '.edit').iterdir())
    assert len(stages) == 1 and (stages[0] / '__before__' / 'scene.json').exists()   # 元は消さずに残す
    monkeypatch.setattr(E.os, 'rename', real)
    E.recover_pending()                                                         # 起動のときに入れ替えを終える
    assert not (data / '.edit').exists() or not any((data / '.edit').iterdir())
    names = [x['name'] for x in dataset.list_scenes()]
    assert names[0] == LOG + '-0001' and len(names) == 2


def test_labels_marked_stale(env):
    raw, data = env
    make_scene(raw, n=100)
    (data / 'labels').mkdir()
    (data / 'labels' / (LOG + '.json')).write_text('{}')
    (data / 'labels' / (LOG + '.npz')).write_bytes(b'x')
    r = E.split_scene(LOG + '-0001', 50)
    assert not (data / 'labels' / (LOG + '.json')).exists() and len(r['labels_stale']) == 2
    assert all('.stale-' in f for f in r['labels_stale'])


def test_dataset_from_parts_matches_original(env):
    """分けたシーンからデータセットを作ると、中ほどのフレームの入力・正解が、元のシーンから作った物と同じ"""
    raw, data = env
    d, cam_t = make_scene(raw, n=600)
    shutil.copytree(d, raw / '20261108-120000' / '20261108-120000-0001', copy_function=os.link)   # 元の写し(比べる用)
    (raw / '20261108-120000' / 'log.json').write_text(json.dumps({'token': 'log-ref', 'logfile': '20261108-120000', 'vehicle': 'v',
                                                                  'date_captured': '2026-11-08'}), encoding='utf-8')
    ref = raw / '20261108-120000' / '20261108-120000-0001'
    m = sj(ref)
    m.update(name='20261108-120000-0001', log='20261108-120000')
    (ref / 'scene.json').unlink()                                               # hardlink を切ってから書く(元を変えない)
    (ref / 'scene.json').write_text(json.dumps(m), encoding='utf-8')
    assert sj(d)['name'] == LOG + '-0001'
    r = E.delete_range(LOG + '-0001', 200, 259)
    c0 = dataset.build_dataset('orig', ['20261108-120000-0001'], val_ratio=0.0)
    c1 = dataset.build_dataset('parts', r['scenes'], val_ratio=0.0)
    assert c0['frames'] == 600 and c1['frames'] == 540
    o = np.load(config.DATASET_DIR / 'orig' / 'aligned' / '20261108-120000-0001.npz')
    a = np.load(config.DATASET_DIR / 'parts' / 'aligned' / ('%s.npz' % r['scenes'][0]))
    b = np.load(config.DATASET_DIR / 'parts' / 'aligned' / ('%s.npz' % r['scenes'][1]))
    for k in ('tof', 'imu', 'vehicle', 'future_actions', 'mode'):
        assert np.array_equal(a[k][:185], o[k][:185]), k                      # 前: 終わりの 0.5 秒より前は同じ
        assert np.array_equal(b[k][5:], o[k][265:]), k                       # 後ろ: 始めの 5 枚より後は同じ
    assert not a['future_valid'][-5:].any()                                    # 前の終わりは未来が無いので正解なし
    assert b['vehicle_valid'][1:].all()


def test_server_endpoints(env, monkeypatch):
    raw, data = env
    from fastapi import HTTPException
    from jetformer.web import server
    d, cam_t = make_scene(raw, n=100)
    monkeypatch.setattr(server.recorder, 'recording', True, raising=False)
    with pytest.raises(HTTPException) as ei:
        server.api_scene_split(LOG + '-0001', {'index': 10})
    assert ei.value.status_code == 409
    monkeypatch.setattr(server.recorder, 'recording', False, raising=False)
    with pytest.raises(HTTPException) as ei:
        server.api_scene_delete_range(LOG + '-0001', {'start': 0, 'end': 5})       # 確認なし
    assert ei.value.status_code == 400
    with pytest.raises(HTTPException) as ei:
        server.api_scene_delete_range(LOG + '-0001', {'start': 0, 'end': 99, 'confirm': 'delete'})   # 全部
    assert ei.value.status_code == 400
    with pytest.raises(HTTPException) as ei:
        server.api_scene_split(LOG + '-0404', {'index': 10})
    assert ei.value.status_code == 404
    with pytest.raises(HTTPException) as ei:
        server.api_scene_delete_range(LOG + '-0001', {'start': 0, 'end': 9, 'confirm': 'delete', 'expect_token': 'old'})
    assert ei.value.status_code == 409                                         # 古い画面
    r = server.api_scene_delete_range(LOG + '-0001', {'start': 0, 'end': 9, 'confirm': 'delete', 'expect_token': 'tok', 'expect_frames': 100})
    assert r['scenes'] == [LOG + '-0001'] and r['deleted_frames'] == 10
    r = server.api_scene_split(LOG + '-0001', {'index': 40})
    assert len(r['scenes']) == 2


# ---------------------------------------------------------------- 画面(node)
def _defs(js, names):
    out = []
    for n in names:
        m = re.search(r'^(?:async )?function %s\(.*?^}\n' % n, js, re.S | re.M) or \
            re.search(r'^const %s = [^\n]*;[^\n]*\n' % n, js, re.M)           # 1 行の const(行の終わりの説明つき)
        assert m, n
        out.append(m.group(0))
    return '\n'.join(out)


@pytest.mark.skipif(NODE is None, reason='node がありません')
def test_js_range_plan_and_text():
    js = (STATIC / 'app.js').read_text(encoding='utf-8')
    code = _defs(js, ['SHORT_SCENE_S', 'MIN_PART_FRAMES', 'rangeEditPlan', 'partLine', 'rangeEditText']) + r"""
const state = { modal: { series: { frame_t: Array.from({ length: 600 }, (_, i) => i / 30) } } };
function frameTime(i) { const t = state.modal.series.frame_t; return t.length ? t[Math.min(i, t.length - 1)] : 0; }
const n = 600;
const out = {
  none: rangeEditPlan(n, null, null), half: rangeEditPlan(n, 5, null), all: rangeEditPlan(n, 599, 0),
  head: rangeEditPlan(n, 0, 59), tail: rangeEditPlan(n, 599, 500), mid: rangeEditPlan(n, 300, 100),
  short: rangeEditPlan(n, 10, 300), shortTail: rangeEditPlan(n, 100, 580),
};
out.midText = rangeEditText(n, out.mid);
out.headText = rangeEditText(n, out.head);
process.stdout.write(JSON.stringify(out));
"""
    r = subprocess.run([NODE, '-e', code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    o = json.loads(r.stdout)
    assert not o['none']['ok'] and not o['half']['ok'] and not o['all']['ok'] and o['all']['kind'] == 'all'
    assert (o['head']['kind'], o['tail']['kind'], o['mid']['kind']) == ('head', 'tail', 'middle')
    assert not o['short']['ok'] and o['short']['kind'] == 'short' and '10 枚' in o['short']['text']    # 前に 10 枚だけ残る
    assert not o['shortTail']['ok'] and '19 枚' in o['shortTail']['text']
    assert (o['mid']['a'], o['mid']['b'], o['mid']['frames']) == (100, 300, 201)            # 始めと終わりが逆でもよい
    assert '前と後ろの 2 つのシーンに分かれます' in o['midText'] and '101〜301 枚目' in o['midText']
    assert '前(元の名前): 1〜100 枚目' in o['midText'] and '5 秒より短いので学習に使われません' in o['midText']
    assert '残る(元の名前): 61〜600 枚目' in o['headText']


def test_static_ui():
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    for el in ('m-edit', 'm-range-bar', 'm-range-fill', 'm-range-cursor', 'm-mark-start', 'm-mark-end', 'm-mark-clear', 'm-range-info',
               'm-split', 'm-delete-range'):
        assert 'id="%s"' % el in html, el
    js = (STATIC / 'app.js').read_text(encoding='utf-8')
    assert "confirm: 'delete'" in js and "/delete_range`" in js and "/split`" in js and '戻せません' in js
    assert "frame/${i}.jpg?v=${encodeURIComponent(s.token || '')}" in js and "thumb.jpg?v=${encodeURIComponent(s.token || '')}" in js
    assert 'expect_token: s.token, expect_frames: s.series.frames' in js and 'setEditBusy(true)' in js
    if NODE:
        r = subprocess.run([NODE, '--check', str(STATIC / 'app.js')], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_thumb_has_tof_half(env):
    """分けた後ろの部分・始めを消した部分のサムネイルも、カメラの絵 + ToF の色の絵(recorder.py と同じ。11/7 にユーザーが見つけた)"""
    import cv2
    raw, data = env
    d, cam_t = make_scene(raw, n=200)
    for i, f in enumerate(np.load(d / 'CAM_FRONT.npz')['files']):              # 本物の絵(224×224)にする
        ok, buf = cv2.imencode('.jpg', np.full((224, 224, 3), i % 255, np.uint8))
        (d / str(f)).write_bytes(buf.tobytes())
    r = E.split_scene(LOG + '-0001', 100)
    back = raw / LOG / r['scenes'][1]
    assert cv2.imread(str(back / 'thumb.jpg')).shape == (224, 448, 3)          # 左がカメラ・右が ToF
    E.delete_range(LOG + '-0001', 0, 39)                                         # 始めを消した前の部分も
    assert cv2.imread(str(d / 'thumb.jpg')).shape == (224, 448, 3)
    (back / 'thumb.jpg').write_bytes(open(str(back / str(np.load(back / 'CAM_FRONT.npz')['files'][0])), 'rb').read())
    tok = sj(back)['token']
    E.refresh_thumb(r['scenes'][1])                                              # 直す道具: ToF つきに作り直し、版を新しく
    assert cv2.imread(str(back / 'thumb.jpg')).shape == (224, 448, 3) and sj(back)['token'] != tok

