"""tools/tof_floor_ref.py(ToF ガードの床の基準を自由走行の記録から作り直す。2026-10-21 Fix-all)の試験。numpy だけ

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python -m pytest -q tests/test_tof_floor_ref.py

確かめること:
  - まっすぐ速いフレーム(|eRPM| > 1,500・|ジャイロ Z| < 60)の信頼度 4 の画素の 80 パーセンタイル。曲がる・遅いフレームは使わない
  - --install なしでは何も書かない。--install は今の基準を控えてから、条件を満たすときだけ入れ替える
  - フレームが足りない・今の基準から ±15% を超えてずれる → 入れ替えない(終了コード 2)
  - 10/10 の人の自由走行の記録(あるときだけ)から作ると、今の data/tof_floor_ref.npy(10/10 18:41)と ±3% で合う
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import tof_floor_ref as tfr  # noqa: E402

CONF4 = 4 << 13


def write_log(raw, log, floor_mm, n=300, erpm=-3000, gz=5.0, curve_frames=0, curve_floor=None):
    """1 シーンの記録(TOF_FRONT・VEHICLE・IMU)。curve_frames 個のフレームは曲がっていて、床が curve_floor"""
    s = raw / log / ('%s-0001' % log)
    s.mkdir(parents=True)
    t = (np.arange(n) * 33_333).astype(np.int64) + 1_000_000
    rng = np.random.default_rng(0)
    depth = np.zeros((n, 42, 54), np.uint16)
    g = np.full(n, gz)
    for i in range(n):
        f = curve_floor if i < curve_frames else floor_mm
        d = np.clip(f + rng.normal(0, 5, (42, 54)), 1, 8000).astype(np.uint16)
        depth[i] = d | CONF4
        if i < curve_frames:
            g[i] = 150.0
    np.savez(s / 'TOF_FRONT.npz', timestamps_us=t, depth=depth, seq=np.arange(n))
    np.savez(s / 'VEHICLE.npz', timestamps_us=t, erpm=np.full(n, erpm, np.int32))
    gyro = np.zeros((n, 3), np.float32)
    gyro[:, 2] = g
    np.savez(s / 'IMU.npz', timestamps_us=t, gyro=gyro)
    (raw / log / 'log.json').write_text('{}', encoding='utf-8')


def test_build_uses_straight_fast_frames_only(tmp_path):
    floor = np.full((42, 54), 600.0)
    write_log(tmp_path, 'L1', floor, n=400, curve_frames=150, curve_floor=np.full((42, 54), 300.0))
    old = tmp_path / 'ref.npy'
    np.save(old, np.full((42, 54), 610.0, np.float32))
    rep, code = tfr.run('L1', tmp_path, old)
    assert code == 0 and rep['ok'] and rep['frames_used'] == 250 and rep['fill_guard_rows'] == 1.0
    assert abs(rep['compare']['guard_rows_median_shift_pct'] - (604.2 / 610 - 1) * 100) < 1.0     # 80 パーセンタイル ≈ 600 + 0.84σ
    assert not rep['installed'] and 'candidate' not in rep
    assert np.all(np.load(old) == 610.0)                                         # --install なしでは変えない


def test_install_backs_up_and_replaces(tmp_path):
    write_log(tmp_path, 'L2', np.full((42, 54), 600.0))
    ref = tmp_path / 'data' / 'tof_floor_ref.npy'
    ref.parent.mkdir()
    np.save(ref, np.full((42, 54), 590.0, np.float32))
    rep, code = tfr.run('L2', tmp_path, ref, install=True, stamp='20261023-090000')
    assert code == 0 and rep['installed']
    bak = ref.with_name('tof_floor_ref.20261023-090000.bak.npy')
    assert np.all(np.load(bak) == 590.0)
    new = np.load(ref)
    assert new.shape == (42, 54) and new.dtype == np.float32 and abs(float(np.median(new)) - 604) < 3
    assert Path(rep['candidate']).exists()


def test_refuses_too_few_frames_or_big_shift(tmp_path):
    ref = tmp_path / 'ref.npy'
    np.save(ref, np.full((42, 54), 600.0, np.float32))
    write_log(tmp_path, 'slow', np.full((42, 54), 600.0), erpm=-800)                # 遅い = まっすぐ速いフレームが 0
    rep, code = tfr.run('slow', tmp_path, ref, install=True)
    assert code == 2 and not rep['installed'] and rep['reasons']
    write_log(tmp_path, 'short', np.full((42, 54), 600.0), n=100)                    # 3 秒だけ
    rep, code = tfr.run('short', tmp_path, ref, install=True)
    assert code == 2 and not rep['installed']
    write_log(tmp_path, 'tilted', np.full((42, 54), 450.0))                          # 25% 近い = ToF が下を向いた
    rep, code = tfr.run('tilted', tmp_path, ref, install=True)
    assert code == 2 and not rep['installed'] and any('取り付け' in r for r in rep['reasons'])
    assert np.all(np.load(ref) == 600.0)


def test_main_exit_codes(tmp_path, capsys):
    assert tfr.main(['nolog', '--raw-dir', str(tmp_path), '--ref', str(tmp_path / 'r.npy')]) == 2
    write_log(tmp_path, 'L3', np.full((42, 54), 600.0))
    assert tfr.main(['L3', '--raw-dir', str(tmp_path), '--ref', str(tmp_path / 'r.npy')]) == 0


def test_real_1010_free_drive_matches_current_ref():
    from jetformer import config
    log = '20261010-150424'
    if not (config.RAW_DIR / log).is_dir() or not tfr.FLOOR_REF_PATH.exists():
        pytest.skip('10/10 の記録か床の基準がありません')
    before = tfr.sha256(tfr.FLOOR_REF_PATH)
    rep, code = tfr.run(log)
    assert code == 0 and rep['frames_used'] > 500
    assert abs(rep['compare']['guard_rows_median_shift_pct']) < 3.0
    assert tfr.sha256(tfr.FLOOR_REF_PATH) == before and not rep['installed']
