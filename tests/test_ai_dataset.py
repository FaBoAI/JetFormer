"""AI 側のデータセット(tools/build_ai_dataset.py)と介入区間の検証セット(tools/make_intervention_set.py)のテスト。
実機・GPU不要(合成した配列と、小さな合成ログ)

    cd ~/jetformer && python -m pytest -q tests/test_ai_dataset.py
"""

import json
import os
import sys

import cv2
import numpy as np
import pytest

from jetformer import config, drive, modes

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import build_ai_dataset as bad  # noqa: E402
import label_events as le  # noqa: E402
import make_intervention_set as mis  # noqa: E402

REF = (1443.0, 1498.0)
G = le.G
FU = 1e6 / 30.0
AI_LOG, HUMAN_LOG = '20261010-210000', '20261010-110000'


# ---------------------------------------------------------------- 小さな部品
def test_eval_actions_inverts_drive_apply_gains():
    """評価時のゲイン(D3)で割り戻した値を drive.apply_gains に通すと、出した値 out に戻る(張り付きも含む)"""
    rng = np.random.default_rng(0)
    out = np.stack([rng.uniform(1000, 2000, 300), rng.uniform(1000, 2000, 300)], axis=1)
    out[:5, 0] = 1000.0                                  # 上下限に張り付いた値
    lab = bad.eval_actions(out, REF, bad.D3)
    p = dict(bad.D3)
    for i in range(len(out)):
        assert drive.apply_gains(lab[i], REF, p, 1000, 2000) == pytest.approx(tuple(out[i]), abs=1e-6)
    assert np.allclose(lab[:, 1], out[:, 1])            # 速度のゲイン 1.0 ならスロットルはそのまま
    g = {'steer_gain': 1.0, 'throttle_gain': 2.0, 'steer_trim_us': 0.0}
    lab2 = bad.eval_actions(np.array([[1443.0, 1398.0], [1443.0, 1600.0]]), REF, g)
    assert lab2[0, 1] == pytest.approx(1448.0) and lab2[1, 1] == pytest.approx(1600.0)   # 前進側だけ割る


def test_horizon_uses_consecutive_frames_and_breaks_at_masks():
    t = (np.arange(40) * FU).astype(np.int64)
    t[20:] += int(FU)                                    # 19 と 20 の間で1フレーム取りこぼし
    label = np.column_stack([np.arange(40) * 1.0, 1000 + np.arange(40) * 1.0])
    valid = np.ones(40, bool)
    seg = np.zeros(40, int)
    fut, ok = bad.horizon(t, label, valid, seg, 10)
    assert ok[0] and np.allclose(fut[0, :, 0], np.arange(10))           # k=0 はこのフレーム、k はフレーム i+k
    # 取りこぼし: 撮像から k+0.5 フレーム以内の最新 → 19 の値が2回続く
    assert ok[15] and fut[15, :, 0].tolist() == [15, 16, 17, 18, 19, 19, 20, 21, 22, 23]
    assert ok[29] and not ok[30:].any()                 # 終わりの近く: 先が足りない(最後のフレームの先は分からない)
    valid2 = valid.copy()
    valid2[8] = False                                    # 8 が使えない → 0〜8 の未来は使えない
    _, ok2 = bad.horizon(t, label, valid2, seg, 10)
    assert not ok2[:9].any() and ok2[9]
    seg2 = np.where(np.arange(40) < 12, 0, 1)            # 区間が 12 で切れる
    _, ok3 = bad.horizon(t, label, valid, seg2, 10)
    assert ok3[2] and not ok3[3:12].any() and ok3[12]
    t4 = t.copy()
    t4[6:] += 200_000                                    # 0.2秒の途切れ(100ms より古い値になる)
    _, ok4 = bad.horizon(t4, label, valid, seg, 10)
    assert not ok4[:6].any() and ok4[6]


def test_human_horizon_ok_ignores_camera_gaps():
    """人の正解は車両データから作るので、カメラの抜けでは切らず、マスクされたフレーム・AI のフレームで切る"""
    t = (np.arange(60) * FU).astype(np.int64)
    t[30:] += 150_000                                    # カメラが 0.15秒止まった
    ok = np.ones(60, bool)
    assert bad.human_horizon_ok(t, ok, 11)[:49].all()
    ok[40] = False                                       # 40 は使えない(交代の前後など)
    h = bad.human_horizon_ok(t, ok, 11)
    assert not h[30:41].any() and h[41:].all()
    assert h[:30].all()                                  # 29 の先 10.5 フレームの時間には 40 が入らない(途切れの分)


def test_previous_command():
    t = (np.array([0, 1, 2, 3, 12]) * FU).astype(np.int64)
    label = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, 5.0], [7.0, 8.0], [9.0, 9.0]])
    prev, ok, age = bad.previous_command(t, label)
    assert ok.tolist() == [False, True, True, False, False]   # 3: 前が NaN、4: 0.2秒より前
    assert prev[1].tolist() == [1.0, 2.0] and age[1] == pytest.approx(33.333, abs=0.01)


@pytest.mark.parametrize('erpm,thr,expect', [
    (-2000, 1498, 0),          # 前へ走っている(ブレーキでも前進)
    (-2000, 1700, 0),
    (0, 1400, 0),              # 止まっていて前進の指令
    (800, 1400, 1),            # 後ろへ動いている
    (0, 1700, 1),              # 止まっていて後退側の指令
    (0, 1500, 2),              # 停止
])
def test_correction_kind(erpm, thr, expect):
    k = mis.correction_kind(np.array([erpm]), np.full((1, 10), thr), REF[1])
    assert k.tolist() == [expect]


# ---------------------------------------------------------------- 小さな合成ログ
def _scene(d, log, si, a, b, ft, vt, it, cam_extra=None, mode_id=5, t0=1_800_000_000_000_000):
    """1シーン分の CAM_FRONT(画像つき)・TOF_FRONT・IMU・VEHICLE を書く。戻り値 (シーンのディレクトリ, 各マスク, 時刻関数)"""
    s = d / ('%s-%04d' % (log, si + 1))
    (s / 'CAM_FRONT').mkdir(parents=True)
    us = lambda x: (t0 + np.round(x * 1e6)).astype(np.int64)      # noqa: E731
    fm, vm, im = (ft >= a) & (ft < b), (vt >= a) & (vt < b), (it >= a) & (it < b)
    ct = us(ft[fm])
    files = []
    for t in ct:
        f = 'CAM_FRONT/%d.jpg' % t
        cv2.imwrite(str(s / f), np.full((16, 16, 3), int(t // 1000) % 200, np.uint8))
        files.append(f)
    np.savez(s / 'CAM_FRONT.npz', timestamps_us=ct, seq=np.arange(len(ct)), files=np.array(files))
    np.savez(s / 'TOF_FRONT.npz', timestamps_us=ct - 2000, seq=np.arange(len(ct)),
             depth=np.full((len(ct), 42, 54), 800, np.uint16))
    meta = {'token': 'tok-%s-%d' % (log, si), 'name': s.name, 'log': log, 'log_token': 'log-' + log,
            'description': 'test', 'tags': [], 'mode_id': mode_id,
            'mode_segments': [{'start_s': 0.0, 'end_s': b - a, 'mode_id': mode_id}],
            'start_us': int(ct[0]), 'end_us': int(ct[-1]), 'duration_s': round(b - a, 3), 'vehicle_quality': 'snapshot'}
    (s / 'scene.json').write_text(json.dumps(meta), encoding='utf-8')
    return s, fm, vm, im, us


def _log_json(d, log, drive_info=None):
    d.mkdir(parents=True)
    m = {'token': 'log-' + log, 'logfile': log, 'vehicle': 'test', 'date_captured': '2026-10-10', 'location': '',
         'description': 'ai_run_test' if drive_info else 'human', 'tags': [], 'mode_id': 5}
    if drive_info:
        m['drive'] = drive_info
    (d / 'log.json').write_text(json.dumps(m), encoding='utf-8')


def write_ai_log(raw_root):
    """12秒・2シーンの AI 走行(記録時のゲイン ステア 1.5・速度 1.2、前進側だけ)。
    1.0〜1.3s は hold。5.5s に正面から衝突して止まる → 6.5s に人へ交代 → 人は 7.0s まで後退、その後 右へ切って前進
    → 9.0s に AI へ戻す。戻り値 (フレーム時刻, 出した値 out)"""
    d = raw_root / AI_LOG
    _log_json(d, AI_LOG, {'run': 'm', 'params': {}, 'steer_center_us': REF[0], 'throttle_neutral_us': REF[1]})
    ft = np.arange(0, 12, 1 / 30.0)
    vt = np.arange(0, 12, 0.01)
    it = vt + 0.003
    human_v = (vt >= 6.5) & (vt < 9.0)
    sel_f = ~((ft >= 6.5) & (ft < 9.0))
    raw = np.stack([1443 + 80 * np.sin(ft), 1440 + 30 * np.sin(ft * 1.7)], axis=1)
    gains = np.tile([1.5, 1.2], (len(ft), 1))
    out = le.apply_gains_vec(raw, gains, np.zeros(len(ft)), REF, forward_only=True)
    hold = (ft >= 1.0) & (ft < 1.3)
    out[hold] = REF
    raw[hold] = np.nan
    erpm = np.full(len(vt), -3000)
    erpm[(vt > 5.55) & (vt < 6.5)] = 0
    erpm[(vt >= 6.5) & (vt < 7.0)] = 800
    erpm[(vt >= 7.0) & (vt < 9.0)] = -2500
    pwm = np.tile([1440, 1500, 1900], (len(vt), 1))
    pwm[human_v, 0] = 1643
    pwm[human_v, 1] = np.where(vt[human_v] < 7.0, 1700, 1400)
    pwm[human_v, 2] = 1100
    rng = np.random.default_rng(0)
    acc = np.zeros((len(it), 3))
    acc[:, 2] = G
    acc[:, :2] = rng.normal(0, 0.2, (len(it), 2))
    k = np.searchsorted(it, 5.5)
    acc[k:k + 3, 1] = [-3.0 * G, -6.0 * G, -2.0 * G]
    for si, (a, b) in enumerate([(0, 6.0), (6.0, 12.0)]):
        s, fm, vm, im, us = _scene(d, AI_LOG, si, a, b, ft, vt, it)
        n = int(fm.sum())
        np.savez(s / 'AI_CMD.npz', timestamps_us=us(ft[fm]), pred=np.zeros((n, 10, 2), np.float32),
                 raw=raw[fm].astype(np.float32), out=out[fm].astype(np.float32), gains=gains[fm].astype(np.float32),
                 trim=np.zeros(n, np.float32), lookahead=np.full(n, 3, np.int16), mode_id=np.full(n, 5, np.int16),
                 hold=hold[fm], select=sel_f[fm], latency_ms=np.full(n, 30, np.float32), infer_ms=np.zeros(n, np.float32),
                 dry_run=np.zeros(n, bool), assist=np.zeros(n, np.int8), model_out=out[fm].astype(np.float32))
        np.savez(s / 'VEHICLE.npz', timestamps_us=us(vt[vm]), pwm=pwm[vm].astype(np.int32),
                 valid=np.ones((vm.sum(), 3), bool), select=~human_v[vm], erpm=erpm[vm].astype(np.int32),
                 board_t_us=us(vt[vm]) - 1_800_000_000_000_000 + 10_000_000)
        np.savez(s / 'IMU.npz', timestamps_us=us(it[im]), accel=acc[im].astype(np.float32),
                 gyro=np.zeros((im.sum(), 3), np.float32), temp=np.full(im.sum(), 30.0, np.float32))
    return ft, out


def write_human_log(raw_root):
    """8秒・1シーンの人の走行(スティックは揺れるステア・一定の前進)"""
    d = raw_root / HUMAN_LOG
    _log_json(d, HUMAN_LOG)
    ft = np.arange(0, 8, 1 / 30.0)
    vt = np.arange(0, 8, 0.01)
    it = vt + 0.003
    s, fm, vm, im, us = _scene(d, HUMAN_LOG, 0, 0, 8.0, ft, vt, it, t0=1_700_000_000_000_000)
    us = lambda x: (1_700_000_000_000_000 + np.round(x * 1e6)).astype(np.int64)      # noqa: E731
    pwm = np.column_stack([1443 + 100 * np.sin(vt), np.full(len(vt), 1450), np.full(len(vt), 1100)]).astype(np.int32)
    np.savez(s / 'VEHICLE.npz', timestamps_us=us(vt), pwm=pwm, valid=np.ones((len(vt), 3), bool),
             select=np.zeros(len(vt), bool), erpm=np.full(len(vt), -2000, np.int32),
             board_t_us=us(vt) - 1_700_000_000_000_000 + 10_000_000)
    acc = np.zeros((len(it), 3))
    acc[:, 2] = G
    np.savez(s / 'IMU.npz', timestamps_us=us(it), accel=acc.astype(np.float32), gyro=np.zeros((len(it), 3), np.float32),
             temp=np.full(len(it), 30.0, np.float32))


@pytest.fixture
def raw_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(config, 'RAW_DIR', tmp_path / 'raw')
    monkeypatch.setattr(config, 'DATASET_DIR', tmp_path / 'datasets')
    monkeypatch.setattr(modes, 'MODES_FILE', tmp_path / 'modes.json')
    ft, out = write_ai_log(tmp_path / 'raw')
    write_human_log(tmp_path / 'raw')
    return {'ft': ft, 'out': out, 'root': tmp_path}


def _load(root, scene):
    with np.load(root / 'aligned' / (scene + '.npz'), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- 通しで: AI 側のデータセット
def test_build_ai_dataset_end_to_end(raw_logs):
    card = bad.build('mix', human_scenes=[HUMAN_LOG + '-0001'], human_val=[HUMAN_LOG + '-0001'],
                     ai_logs=[AI_LOG], val_ai_logs=(), log=lambda *a: None)
    root = config.DATASET_DIR / 'mix'
    splits = json.loads((root / 'splits.json').read_text())
    assert splits == {'train': [AI_LOG + '-0001', AI_LOG + '-0002'], 'val': [HUMAN_LOG + '-0001']}
    assert card['built_by'] == bad.TOOL and card['ai_mix']['label_gains'] == 'eval'
    assert (root / 'v1.0-mix' / 'sample.json').exists()                  # build_dataset の表もある
    lab = card['ai_mix']['labels'][AI_LOG]
    assert lab['eval_roundtrip_max_abs_us'] == pytest.approx(0.0, abs=1e-3)

    s1, s2 = _load(root, AI_LOG + '-0001'), _load(root, AI_LOG + '-0002')
    for k in bad.EXTRA_KEYS:
        assert k in s1
    for k in ('timestamps_us', 'cam_files', 'tof', 'imu', 'vehicle', 'vehicle_valid', 'future_actions',
              'future_valid', 'mode', 'erpm_valid'):
        assert k in s1
    A = {k: np.concatenate([s1[k], s2[k]]) for k in s1}
    t = (A['timestamps_us'] - A['timestamps_us'][0]) / 1e6
    exp = bad.eval_actions(raw_logs['out'], REF, bad.D3)
    ai = A['source'] == 1
    assert ai[(t > 2.0) & (t < 6.4)].all() and not ai[(t > 6.6) & (t < 8.9)].any()
    # AI のフレーム: k フレーム先 = 続くフレームの out を D3 で割り戻した値、今の操作値 = 1つ前のフレームの値
    i = int(np.argmin(np.abs(t - 3.0)))
    assert A['future_valid'][i] and A['vehicle_valid'][i] and A['mode'][i] == 5
    assert np.allclose(A['future_actions'][i], exp[i:i + 10], atol=1e-3)
    assert np.allclose(A['vehicle'][i, :2], exp[i - 1], atol=1e-3)
    assert A['vehicle'][i, 3] == -3000 and A['vehicle'][i, 2] == 1900
    assert np.allclose(A['action_label'][i], exp[i], atol=1e-3) and np.allclose(A['action_exec'][i], raw_logs['out'][i])
    assert np.isfinite(A['reward'][i]) and np.isfinite(A['mc_return'][i]) and A['label_valid'][i]
    # シーンの境目(6.0s)をまたいでも、ラベルはログ全体でつながっている
    j = int(np.argmin(np.abs(t - 5.9)))
    assert A['future_valid'][j - 30] and np.allclose(A['future_actions'][j - 30], exp[j - 30:j - 20], atol=1e-3)
    # hold の区間と、その手前 9 フレームは使えない。交代の前(0.3秒のマスク + 先の 10フレーム)も使えない
    assert not A['future_valid'][(t >= 1.0) & (t < 1.3)].any()
    assert not A['future_valid'][(t > 0.69) & (t < 1.0)].any() and A['future_valid'][(t > 0.3) & (t < 0.65)].all()
    assert not A['future_valid'][(t > 5.9) & (t < 6.5)].any()
    # AI のログの中の人のフレームは、既定では使わない
    assert not A['future_valid'][A['source'] == 0].any()
    # 失敗による交代の罰 −2 が、交代の直前の AI のフレームに入る
    k_tk = np.nonzero(A['failure_takeover'])[0]
    assert len(k_tk) == 1 and A['source'][k_tk[0]] == 1 and A['reward'][k_tk[0]] < -1.5
    assert A['collision'].sum() == 1 and A['end_code'][k_tk[0]] == 1
    # 人のログ: build_dataset のスティックのまま、source=0
    h = _load(root, HUMAN_LOG + '-0001')
    assert (h['source'] == 0).all() and h['future_valid'].sum() > 150
    m = h['future_valid']
    assert np.allclose(h['future_actions'][m, 0, 1], 1450)
    # JetFormerDataset で読める(train = AI のフレームだけ)
    from jetformer.train_data import JetFormerDataset
    ds = JetFormerDataset(str(root), 'train')
    usable = int((A['future_valid'] & A['vehicle_valid'] & (A['mode'] != 0)).sum())
    assert len(ds) == usable == card['ai_mix']['usable_frames']['train']['ai'] > 200
    img, tof, imu, state, mode, target = ds[0]
    assert tuple(target.shape) == (10, 2) and int(mode.argmax()) == 5
    assert card['ai_mix']['usable_frames']['val']['human'] == int((m & h['vehicle_valid']).sum())


def test_build_ai_dataset_options(raw_logs):
    card = bad.build('mix2', ai_logs=[AI_LOG], val_ai_logs=[AI_LOG], keep_ai_log_human=True, label_gains='recorded',
                     log=lambda *a: None)
    root = config.DATASET_DIR / 'mix2'
    assert json.loads((root / 'splits.json').read_text())['train'] == []
    A = {k: np.concatenate([v, _load(root, AI_LOG + '-0002')[k]]) for k, v in _load(root, AI_LOG + '-0001').items()}
    t = (A['timestamps_us'] - A['timestamps_us'][0]) / 1e6
    i = int(np.argmin(np.abs(t - 3.0)))
    raw = np.stack([1443 + 80 * np.sin(raw_logs['ft']), 1440 + 30 * np.sin(raw_logs['ft'] * 1.7)], axis=1)
    assert np.allclose(A['future_actions'][i], raw[i:i + 10], atol=1e-2)       # 記録時のゲイン = raw
    hum = (A['source'] == 0) & A['future_valid']
    assert hum.any() and (t[hum] > 6.79).all() and (t[hum] < 8.7).all()      # 交代・戻しの前後 0.3秒は使わない
    with pytest.raises(ValueError):
        bad.build('mix2', ai_logs=[AI_LOG], log=lambda *a: None)              # 同名
    bad.build('mix2', ai_logs=[AI_LOG], overwrite=True, log=lambda *a: None)
    (config.DATASET_DIR / 'other').mkdir()
    (config.DATASET_DIR / 'other' / 'dataset_card.json').write_text('{}')
    with pytest.raises(ValueError):
        bad.build('other', ai_logs=[AI_LOG], overwrite=True, log=lambda *a: None)   # ほかのツールのものは消さない
    assert card['ai_mix']['keep_ai_log_human']


# ---------------------------------------------------------------- 通しで: 介入区間の検証セット
def test_intervention_set_and_compare_models(raw_logs, monkeypatch):
    card = mis.build('iv', logs=[AI_LOG], lead_s=2.0, max_s=5.0, log=lambda *a: None)
    root = config.DATASET_DIR / 'iv'
    splits = json.loads((root / 'splits.json').read_text())
    name = AI_LOG + '-tk01'
    assert splits == {'train': [], 'val': [name]}
    w = card['intervention']['windows'][0]
    assert w['class'] == 'after_collision' and w['included'] and w['scenes'] == [AI_LOG + '-0001', AI_LOG + '-0002']
    a = _load(root, name)
    ts = a['t_since_takeover_s']
    assert ts[0] == pytest.approx(-2.0, abs=0.04) and ts[-1] < 2.5           # 前置き 2秒 〜 戻し(9.0s)の手前
    sc = a['future_valid']
    # 採点は人が前進している部分だけ(6.5〜7.0s は後退)。正解はスティック(右へ 1643・前進 1400)
    assert sc.sum() == w['scored'] > 30
    assert (ts[sc] >= 0.5 - 1e-3).all() and (ts[sc] < 2.5 - 0.3).all()
    assert np.allclose(a['future_actions'][sc, :, 0], 1643) and np.allclose(a['future_actions'][sc, :, 1], 1400)
    assert (a['correction_kind'][(ts > 0.05) & (ts < 0.45)] == 1).all()
    assert w['correction_kinds']['reverse'] >= 10 and w['correction_kinds']['forward'] >= 40
    # 前置きの AI のフレームは採点しないが、今の操作値は AI の出した値(D3 で割り戻し)
    lead = ts < -0.1
    assert not sc[lead].any() and (a['source'][lead] == 1).all()
    for f in a['cam_files']:
        assert (root / str(f)).exists()
    b = card['intervention']['baselines']['all']
    assert b['n'] == int(sc.sum()) and b['hold_current_k3'][1] == pytest.approx(0.0, abs=1e-3)
    # 後退も含めると、交代の直後(交代の前後 0.3秒のマスクは使わない)から採点する
    card2 = mis.build('iv2', logs=[AI_LOG], kinds=mis.KINDS, lead_s=2.0, log=lambda *a: None)
    a2 = _load(config.DATASET_DIR / 'iv2', name)
    assert a2['future_valid'].sum() > sc.sum() and a2['t_since_takeover_s'][a2['future_valid']].min() < 0.05
    assert card2['intervention']['counts']['scored_frames'] == int(a2['future_valid'].sum())
    # 運用上の交代だけにすると窓は無い(交代の情報は残る)
    card3 = mis.build('iv3', logs=[AI_LOG], classes=['operational'], log=lambda *a: None)
    assert card3['val_scenes'] == 0 and len(card3['intervention']['windows']) == 1

    # JetFormerDataset と tools/compare_models.py で読める(モデルは偽物。GPU は使わない)
    from jetformer.train_data import JetFormerDataset
    ds = JetFormerDataset(str(root), 'val')
    assert len(ds) == int(sc.sum())
    import compare_models as cm
    monkeypatch.setattr(cm, 'load_model', lambda run: (None, {'num_modes': 6, 'command_input': True}, [5], {}))

    def fake_predict(model, cfg, trained, root_, a_):
        p = a_['future_actions'].astype(np.float32) + 10.0
        return p, p, np.zeros(len(p), bool)
    monkeypatch.setattr(cm, 'predict_scene', fake_predict)
    monkeypatch.setattr(sys, 'argv', ['compare_models.py', '--dataset', 'iv', '--runs', 'fake'])
    cm.main()
    rep = json.loads((config.DATA_DIR / 'reports' / 'compare_iv' / 'compare.json').read_text())
    r = rep['models']['fake']['sf']['all']
    assert r['n'] == int(sc.sum()) and r['steer_mae_k3'] == pytest.approx(10.0) and r['throttle_mae_k3'] == pytest.approx(10.0)
