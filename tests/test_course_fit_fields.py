"""tools/course_fit.py の LogData が使う欄だけを読む試験(2026-10-26 fix3・係 tools、反証 1 回目 code:F1)。

前(fix2): npz の欄を全部読んだので、使わない欄(AI_CMD の pred・IMU の mag・VEHICLE の seq・CAM_FRONT の files)の
Bad CRC だけでシーンの AI_CMD を捨てた・シーンごと飛ばした。AI_CMD を捨てたシーンの時間は、ai_at が前のシーンの最後の
select を返した(人の時間を AI と数えた)。
今: 使う欄(course_fit.USED_FIELDS)だけを読む。使う欄が壊れたら今まで通り注意を出して飛ばす。AI_CMD を飛ばしたシーンの
時間の ai_at は VEHICLE の select。

作り物の小さな記録だけ(tmp)。本物の data/raw は読まない。GPU・torch は使わない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_course_fit_fields.py
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import course_fit as cf  # noqa: E402

LOG = '20001024-130000'
T0 = 1_800_000_000_000_000
N = 300                                  # 1 シーンのコマ(30 Hz で 10 秒)


def _scene(d, k, ai_sel):
    """k 番目のシーン(10 秒)。ai_sel: このシーンの AI 側(AI_CMD と VEHICLE の select)"""
    d.mkdir(parents=True)
    t = T0 + k * 10_000_000 + np.arange(N, dtype=np.int64) * 33_333
    rng = np.random.default_rng(k)
    sel = np.full(N, ai_sel, bool)
    np.savez(d / 'IMU.npz', timestamps_us=t, accel=np.tile([0.0, 0.0, cf.G], (N, 1)).astype(np.float32),
             gyro=np.zeros((N, 3), np.float32), mag=rng.normal(size=(N, 3)).astype(np.float32),
             temp=np.full(N, 30.0, np.float32))
    np.savez(d / 'VEHICLE.npz', timestamps_us=t, pwm=np.tile([1443, 1496], (N, 1)).astype(np.int32),
             erpm=np.full(N, -3000, np.int32), select=sel, hall_count=np.arange(N, dtype=np.int64) * 3,
             board_t_us=t - T0, seq=np.arange(N, dtype=np.int64), hall_state=rng.integers(0, 6, N).astype(np.int8))
    np.savez(d / 'CAM_FRONT.npz', timestamps_us=t, files=np.array(['CAM_FRONT/%06d.jpg' % i for i in range(N)]))
    np.savez(d / 'AI_CMD.npz', timestamps_us=t, select=sel, out=np.tile([1443.0, 1300.0], (N, 1)).astype(np.float32),
             pred=rng.normal(size=(N, 8, 2)).astype(np.float32), gains=np.tile([1.0, 0.9], (N, 1)).astype(np.float32))


def make_log(raw):
    """3 シーン: 1 = AI 側・2 = 人(CH3 を人側)・3 = AI 側"""
    root = raw / LOG
    root.mkdir(parents=True)
    (root / 'log.json').write_text(json.dumps({'drive': {'run': 'x', 'steer_center_us': 1443}}), encoding='utf-8')
    for k, sel in ((1, True), (2, False), (3, True)):
        _scene(root / ('%s-%04d' % (LOG, k)), k, sel)
    return root


def corrupt_member(npz, member):
    """npz(zip・圧縮なし)の中の 1 つの欄 member.npy の中身だけを潰す(開けるが、その欄を読むと Bad CRC-32)"""
    with zipfile.ZipFile(npz) as zf:
        info = zf.getinfo(member + '.npy')
    b = bytearray(npz.read_bytes())
    off = info.header_offset
    n_name = int.from_bytes(b[off + 26:off + 28], 'little')
    n_extra = int.from_bytes(b[off + 28:off + 30], 'little')
    start = off + 30 + n_name + n_extra + 128            # .npy の頭(128 バイト)のあとの数の部分
    assert start + 64 < off + 30 + n_name + n_extra + info.compress_size
    for i in range(start, start + 64):
        b[i] ^= 0xFF
    npz.write_bytes(bytes(b))
    with np.load(npz) as z:                                 # 確かめ: その欄だけ読めない
        with pytest.raises(zipfile.BadZipFile):
            z[member]
        for k in z.files:
            if k != member:
                z[k]


def arrays(d):
    keys = ('cam_us', 'ct', 'cam_scene', 'it', 'acc', 'gyr', 'vt', 'pwm', 'erpm', 'vsel', 'hall', 'board', 'at', 'a_select',
            'a_out', 'g', 'hg')
    return {k: np.asarray(getattr(d, k)) for k in keys}


def same(a, b):
    return set(a) == set(b) and all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and np.array_equal(a[k], b[k])
                                    for k in a)


@pytest.mark.parametrize('name, member', [('AI_CMD', 'pred'), ('AI_CMD', 'gains'), ('IMU', 'mag'), ('IMU', 'temp'),
                                          ('VEHICLE', 'seq'), ('VEHICLE', 'hall_state'), ('CAM_FRONT', 'files')])
def test_unused_field_broken_reads_the_same(tmp_path, capsys, name, member):
    """使わない欄だけが壊れた: シーンも配列も壊れていない写しとビット単位で同じ。注意も出さない"""
    good_raw, bad_raw = tmp_path / 'good', tmp_path / 'bad'
    make_log(good_raw)
    root = make_log(bad_raw)
    corrupt_member(root / ('%s-0002' % LOG) / (name + '.npz'), member)
    g, b = cf.LogData(LOG, str(good_raw)), cf.LogData(LOG, str(bad_raw))
    assert b.scenes == g.scenes == ['%s-%04d' % (LOG, k) for k in (1, 2, 3)]
    assert same(arrays(b), arrays(g))
    assert b.ai_bad == [] and len(b.at) == 3 * N
    assert '注意' not in capsys.readouterr().err


def test_ai_cmd_used_field_broken_skips_and_uses_vehicle_select(tmp_path, capsys):
    """AI_CMD の使う欄(out)が壊れた: 注意を出してそのシーンの AI_CMD を飛ばす。そのシーンの時間の ai_at は VEHICLE の
    select(人)。前は 1 つ前のシーンの最後の select(AI)を返し、人の 10 秒を AI と数えた"""
    root = make_log(tmp_path)
    corrupt_member(root / ('%s-0002' % LOG) / 'AI_CMD.npz', 'out')
    d = cf.LogData(LOG, str(tmp_path))
    err = capsys.readouterr().err
    assert '%s/%s-0002/AI_CMD.npz が読めません(BadZipFile)' % (LOG, LOG) in err
    assert d.scenes == ['%s-%04d' % (LOG, k) for k in (1, 2, 3)]          # シーンは残す(IMU・VEHICLE・カメラは読める)
    assert len(d.at) == 2 * N and len(d.vt) == 3 * N
    assert d.ai_bad == [(pytest.approx(10.0, abs=1e-6), pytest.approx(10.0 + (N - 1) * 0.033333, abs=1e-6))]
    t = np.array([5.0, 12.0, 15.0, 19.9, 25.0])
    assert d.ai_at(t).tolist() == [True, False, False, False, True]
    assert bool(d.ai_at(15.0)) is False and bool(d.ai_at(5.0)) is True    # 1 つの時刻でも
    grid = np.arange(0.0, 30.0, 0.05)
    assert int(np.sum(d.ai_at(grid))) == int(np.sum((grid < 10.0 - 1e-9) | (grid >= 20.0 - 1e-9)))


def test_imu_used_field_broken_skips_the_scene(tmp_path, capsys):
    """IMU の使う欄(accel)が壊れた: 今まで通りシーンごと飛ばす(注意つき)"""
    root = make_log(tmp_path)
    corrupt_member(root / ('%s-0002' % LOG) / 'IMU.npz', 'accel')
    d = cf.LogData(LOG, str(tmp_path))
    assert d.scenes == ['%s-0001' % LOG, '%s-0003' % LOG] and len(d.cam_us) == 2 * N
    assert '%s/%s-0002 が読めません(BadZipFile)' % (LOG, LOG) in capsys.readouterr().err


def test_good_record_unchanged_and_used_fields_listed(tmp_path):
    """壊れていない記録: AI_CMD の時間はそのまま(ai_bad は空・前と同じ道)。USED_FIELDS は LogData が読む欄と同じ"""
    make_log(tmp_path)
    d = cf.LogData(LOG, str(tmp_path))
    assert d.ai_bad == [] and d.has_ai
    grid = np.arange(0.0, 30.0, 0.05)
    i = np.clip(np.searchsorted(d.at, grid, side='right') - 1, 0, len(d.at) - 1)
    assert np.array_equal(d.ai_at(grid), d.a_select[i])
    import re
    src = (ROOT / 'tools' / 'course_fit.py').read_text(encoding='utf-8')
    used = set(re.findall(r"cat\('(\w+)', '(\w+)'", src))                # LogData が cat で使う欄
    assert used == {(n, k) for n, ks in cf.USED_FIELDS.items() for k in ks}
