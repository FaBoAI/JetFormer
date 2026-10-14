"""展開したシミュレーターのシーンからデータセットを作り直すときの正解の引き方(dataset._aligned_arrays)のテスト(案)。

離脱復活 r4(2026-10-14、学習の段 recover r4)が r3 の案を直したもの。r3/rc4/promote/dataset_frame_index_v2.patch を当てた
jetformer/dataset.py で通る。r4 で足したもの(確認 recover r3 の指摘): 実機の VEHICLE.npz も quality = 'snapshot'(recorder.py、100 Hz)なので、
その形(quality 'snapshot'・VEHICLE 100 Hz・カメラ 30 Hz)が今までどおり時刻で引かれ、差分の前と同じ値になる試験(test_real_car_snapshot_100hz_unchanged)。
以下は r3 の説明: 確認 recover r2 の指摘(a): 展開したシーン(VEHICLE の quality = 'snapshot')から build_dataset で作り直すと、
正解を時刻(int(33333.33 µs) おき)で引くため、シミュレーターの 33,333/33,334 µs おきのフレームでは観測の一部で正解が 1 フレーム前にずれる。
直し: snapshot のシーンだけ未来の操作をフレームの番号で引く(future_actions[k] = vehicle[k+1..k+10])。実機のシーンは今までどおり。
"""

import numpy as np

from jetformer import dataset

N = 90


def sim_times(n=N):
    """シミュレーターのフレームの時刻(µs): 0, 33333, 66666, 100000, …(33,333/33,334 µs おき)"""
    return (np.arange(n) * 1e6 / 30).astype(np.int64)


def veh_stream(t, quality=None, invalid=()):
    n = len(t)
    pwm = np.zeros((n, 3), np.int32)
    pwm[:, 0] = 1000 + np.arange(n)            # 舵の値 = フレームの番号がわかる値
    pwm[:, 1] = 1500 - np.arange(n)
    pwm[:, 2] = 1000
    valid = np.ones((n, 3), bool)
    for k in invalid:
        valid[k] = False
    v = {'timestamps_us': t.copy(), 'pwm': pwm, 'valid': valid, 'select': np.zeros(n, bool), 'erpm': np.zeros(n, np.int32)}
    if quality is not None:
        v['quality'] = np.array(quality)
    return v


def time_based_reference(cam_t, veh):
    """パッチ前の _aligned_arrays の未来の操作(時刻で引く)。実機のシーンが変わらないことの比べに使う"""
    ok = veh['valid'][:, 0] & veh['valid'][:, 1] & ~veh['select']
    state = np.concatenate([veh['pwm'][:, :3].astype(np.float32), veh['erpm'][:, None].astype(np.float32)], axis=1)
    vt, st = veh['timestamps_us'][ok], state[ok]
    n = len(cam_t)
    fut = np.zeros((n, dataset.FUTURE_STEPS, 2), np.float32)
    fut_ok = np.ones(n, bool)
    frame_us = 1e6 / dataset.config.CAMERA['fps']
    for k in range(dataset.FUTURE_STEPS):
        tk = cam_t + int((k + 1) * frame_us)
        jk = np.searchsorted(vt, tk, side='right') - 1
        jkc = np.clip(jk, 0, len(vt) - 1)
        fut[:, k] = st[jkc, :2]
        fut_ok &= (jk >= 0) & (tk - vt[jkc] <= dataset.VEHICLE_MAX_AGE_MS * 1000)
    horizon = cam_t + int(dataset.FUTURE_STEPS * frame_us)
    return fut, fut_ok & (horizon <= veh['timestamps_us'][-1])


def test_snapshot_scene_uses_frame_index():
    t = sim_times()
    out = dataset._aligned_arrays(t, None, None, veh_stream(t, 'snapshot'))
    fa, fv = out['future_actions'], out['future_valid']
    for k in range(N):
        if k + dataset.FUTURE_STEPS <= N - 1:
            assert fv[k]
            assert np.array_equal(fa[k, :, 0], 1000 + k + 1 + np.arange(dataset.FUTURE_STEPS)), k
        else:
            assert not fv[k]
    # 時刻で引くと 1 フレーム前にずれる観測がある(直す前の性質。確認 recover r2 の 67%)
    old, _ = time_based_reference(t, veh_stream(t, 'snapshot'))
    shifted = sum(1 for k in range(N - 10) if not np.array_equal(old[k, :, 0], fa[k, :, 0]))
    assert shifted > 0


def test_real_scene_unchanged():
    """実機のシーン(quality が無い・snapshot でない)は今までどおり時刻で引く(パッチ前と同じ値)"""
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.integers(30000, 37000, N)).astype(np.int64)
    for q in (None, 'v2', 'legacy'):
        veh = veh_stream(t, q, invalid=(15, 40))
        out = dataset._aligned_arrays(t, None, None, veh)
        fut, ok = time_based_reference(t, veh)
        assert np.array_equal(out['future_actions'], fut) and np.array_equal(out['future_valid'], ok), q


def test_snapshot_invalid_frames_drop_labels():
    """snapshot でも、正解に入る操作が無効(valid が False)の観測は future_valid = False(次の有効な値で詰めない)"""
    t = sim_times()
    out = dataset._aligned_arrays(t, None, None, veh_stream(t, 'snapshot', invalid=(30,)))
    fv = out['future_valid']
    assert not fv[20:30].any() and fv[19] and fv[30]


def test_snapshot_not_one_to_one_falls_back_to_time():
    """snapshot でも操作がカメラのフレームと 1 対 1 でなければ(足りない・ずれている)時刻で引く"""
    t = sim_times()
    veh = veh_stream(t[::2], 'snapshot')
    out = dataset._aligned_arrays(t, None, None, veh)
    fut, ok = time_based_reference(t, veh)
    assert np.array_equal(out['future_actions'], fut) and np.array_equal(out['future_valid'], ok)


def test_masked_range_keeps_perturbed_actions_out_of_labels():
    """乱しの区間 [ps, pe) の操作は、外す範囲 [ps − 10, pe − 1] の外の観測の正解に入らない(フレームの番号で引くとき)"""
    t = sim_times()
    out = dataset._aligned_arrays(t, None, None, veh_stream(t, 'snapshot'))
    ps, pe = 40, 52
    masked = set(range(ps - 10, pe))
    for k in range(N):
        if k in masked or not out['future_valid'][k]:
            continue
        frames = out['future_actions'][k, :, 0] - 1000
        assert not any(ps <= f < pe for f in frames), k


def real_car_streams(seconds=6.0, seed=1, invalid_every=0):
    """実機の形: カメラ 30 Hz(33,000〜34,000 µs おき)・VEHICLE 100 Hz(9,500〜10,500 µs おき、quality = 'snapshot'、recorder.py と同じ)"""
    rng = np.random.default_rng(seed)
    t0 = 1_791_000_000_000_000
    cam = t0 + np.cumsum(rng.integers(33000, 34000, int(seconds * 30))).astype(np.int64)
    vt = t0 - 50_000 + np.cumsum(rng.integers(9500, 10500, int(seconds * 100) + 10)).astype(np.int64)
    n = len(vt)
    pwm = np.zeros((n, 3), np.int32)
    pwm[:, 0] = 1000 + np.arange(n)
    pwm[:, 1] = 1500 - (np.arange(n) % 300)
    pwm[:, 2] = 1000
    valid = np.ones((n, 3), bool)
    if invalid_every:
        valid[::invalid_every] = False
    veh = {'timestamps_us': vt, 'pwm': pwm, 'valid': valid, 'select': np.zeros(n, bool), 'erpm': np.zeros(n, np.int32),
           'quality': np.array('snapshot')}
    return cam, veh


def test_real_car_snapshot_100hz_unchanged():
    """実機の VEHICLE.npz(quality = 'snapshot'・100 Hz)とカメラ 30 Hz: 1 対 1 でないので時刻で引き、差分の前と同じ値(確認 recover r3 の指摘)"""
    for seed, inv in ((1, 0), (2, 37), (3, 5)):
        cam, veh = real_car_streams(seed=seed, invalid_every=inv)
        assert str(np.asarray(veh['quality'])) == 'snapshot'
        if hasattr(dataset, '_frame_indexed'):                 # 差分を当てた本体だけにある(当てる前の本体でもこの試験は通る = 前と同じ値)
            assert not dataset._frame_indexed(veh, cam)
        out = dataset._aligned_arrays(cam, None, None, veh)
        fut, ok = time_based_reference(cam, veh)
        assert np.array_equal(out['future_actions'], fut) and np.array_equal(out['future_valid'], ok), (seed, inv)


def test_real_car_snapshot_same_count_but_not_aligned():
    """実機の形で操作の数がたまたまカメラと同じでも、各フレームの最新の操作が同じ番号でなければ時刻で引く"""
    cam, veh = real_car_streams(seed=4)
    n = len(cam)
    v = {k: (val[:n] if isinstance(val, np.ndarray) and val.ndim >= 1 and len(val) == len(veh['timestamps_us']) else val) for k, val in veh.items()}
    v['timestamps_us'] = cam - 20_000                      # 1 つ前のフレームの時刻に近い操作(番号が 1 対 1 にならない)
    v['timestamps_us'][1::2] = cam[1::2] - 40_000
    assert len(v['timestamps_us']) == n
    if hasattr(dataset, '_frame_indexed'):
        assert not dataset._frame_indexed(v, cam)
    out = dataset._aligned_arrays(cam, None, None, v)
    fut, ok = time_based_reference(cam, v)
    assert np.array_equal(out['future_actions'], fut) and np.array_equal(out['future_valid'], ok)
