"""0 バイト・壊れた記録(電源断)で tools/ の台本が落ちずに、飛ばして注意を出す試験(2026-10-26 fix2)。

本物の記録には 0 バイトの npz がある(20261024-162305 の 0007・0008、20261010-163247 の 0021・0022、20261023-204825 は
log.json・scene.json も 0 バイト)。前に落ちていたもの(experiments/20261026_fix/fix2/probe_broken_npz.py で探した):
  - bench_report.py・car_check.py: log.json が 0 バイトの記録で JSONDecodeError(section_scorecard._labels → label_log)
  - tof_floor_ref.load: 0 バイト(EOFError)・壊れた zip(BadZipFile)を except (OSError, ValueError) が捕まえない
  - drive_report._load: try が無い(0 バイト・切れた・Bad CRC で落ちる)
  - course_fit.LogData: 開くだけで中身を読まないので、Bad CRC の npz は try の外で落ちる
  - build_ai_dataset.scene_problem: scene.json が 0 バイト・中身の Bad CRC を見ない(choose_ai_scenes・make_intervention_set が落ちる)
  - make_intervention_set.load_raw: 壊れた AI_CMD.npz で落ちる

本物の data/raw・data/labels には書かない(全部 tmp)。本物の記録 20261024-185926 の写しを使う試験は、無ければ飛ばす。
GPU・torch は使わない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q tests/test_broken_records.py
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))
SRC = ROOT / 'data' / 'raw' / '20261024-185926'      # 3 シーンの AI の記録(約 5 MB。画像は写さない)
need_src = pytest.mark.skipif(not (SRC / 'log.json').exists(), reason='写しの元の記録 20261024-185926 が無い')


def _corrupt(p):
    """npz の中身を 0 で潰す(開けるが、読むと Bad CRC-32)"""
    b = bytearray(p.read_bytes())
    n = len(b)
    b[n // 4: n // 4 + 256] = b'\x00' * 256
    p.write_bytes(bytes(b))


def _truncate(p):
    b = p.read_bytes()
    p.write_bytes(b[:len(b) // 2])


# ---------------------------------------------------------------- 小さな偽物の記録
def _fake_scene(d, n=200, t0=1_800_000_000_000_000):
    d.mkdir(parents=True)
    t = t0 + np.arange(n, dtype=np.int64) * 33_333
    rng = np.random.default_rng(0)
    np.savez(d / 'TOF_FRONT.npz', timestamps_us=t, depth=rng.integers(0, 2 ** 15, (n, 42, 54)).astype(np.uint16))
    np.savez(d / 'VEHICLE.npz', timestamps_us=t, erpm=np.full(n, -3000, np.int32), pwm=np.full((n, 2), 1500, np.int32),
             valid=np.ones(n, bool), select=np.ones(n, bool))
    np.savez(d / 'IMU.npz', timestamps_us=t, gyro=np.zeros((n, 3), np.float32), accel=np.zeros((n, 3), np.float32))
    np.savez(d / 'CAM_FRONT.npz', timestamps_us=t, files=np.array(['CAM_FRONT/%06d.jpg' % i for i in range(n)]))
    (d / 'scene.json').write_text(json.dumps({'duration_s': n / 30.0, 'description': ''}))


def test_tof_floor_ref_skips_zero_and_corrupt_npz(tmp_path, capsys):
    import tof_floor_ref as tfr
    log = '20001024-110000'
    root = tmp_path / log
    for i in (1, 2, 3, 4):
        _fake_scene(root / ('%s-%04d' % (log, i)), t0=1_800_000_000_000_000 + i * 10_000_000)
    good = tfr.load(log, tmp_path)
    for n in ('TOF_FRONT', 'VEHICLE', 'IMU'):
        (root / (log + '-0002') / (n + '.npz')).write_bytes(b'')          # 0 バイト(EOFError)
    _corrupt(root / (log + '-0003') / 'VEHICLE.npz')                     # Bad CRC(BadZipFile)
    _truncate(root / (log + '-0004') / 'IMU.npz')                        # 切れた zip(BadZipFile)
    out = tfr.load(log, tmp_path)
    probs = out[-1]
    assert sorted(probs) == sorted(['%s-0002/TOF_FRONT: 読めません' % log, '%s-0002/VEHICLE: 読めません' % log,
                                    '%s-0002/IMU: 読めません' % log, '%s-0003/VEHICLE: 読めません' % log,
                                    '%s-0004/IMU: 読めません' % log])
    assert len(out[0]) == len(good[0]) - 200                             # 0002 の ToF だけ減る(0003・0004 の ToF は読める)
    err = capsys.readouterr().err
    assert '注意:' in err and 'EOFError' in err and 'BadZipFile' in err


def test_drive_report_load_skips_broken(tmp_path, capsys):
    import drive_report as dr
    d = tmp_path / 's'
    _fake_scene(d)
    good = dr._load(d, 'VEHICLE')
    with np.load(d / 'VEHICLE.npz', allow_pickle=True) as z:
        ref = {k: z[k] for k in z.files}
    assert sorted(good) == sorted(ref) and all(np.array_equal(good[k], ref[k]) for k in ref)   # 読めるものは前と同じ
    assert dr._load(d, 'AI_CMD') is None                                 # 無いものは今まで通り None
    (d / 'IMU.npz').write_bytes(b'')
    _corrupt(d / 'VEHICLE.npz')
    _truncate(d / 'TOF_FRONT.npz')
    assert dr._load(d, 'IMU') is None and dr._load(d, 'VEHICLE') is None and dr._load(d, 'TOF_FRONT') is None
    err = capsys.readouterr().err
    assert err.count('注意:') == 3 and 'EOFError' in err and 'BadZipFile' in err


def test_drive_report_main_skips_unreadable_log(tmp_path, monkeypatch, capsys):
    import drive_report as dr
    from jetformer import config
    log = '20001024-110001'
    (tmp_path / 'raw' / log).mkdir(parents=True)
    (tmp_path / 'raw' / log / 'log.json').write_bytes(b'')               # 20261023-204825 と同じ形
    monkeypatch.setattr(config, 'RAW_DIR', tmp_path / 'raw')
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path / 'data')
    monkeypatch.setattr(sys, 'argv', ['drive_report.py', log])
    with pytest.raises(SystemExit) as e:
        dr.main()
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert '%s を評価できません' % log in err and '評価できなかった記録 1 本' in err
    assert not (tmp_path / 'data').exists()                              # 何も書かない


def test_scene_problem_broken_scene_json_and_corrupt_member(tmp_path):
    import build_ai_dataset as bad
    for k in ('ok', 'sj', 'crc', 'zero'):
        _fake_scene(tmp_path / k)
    assert bad.scene_problem(tmp_path / 'ok') is None
    (tmp_path / 'sj' / 'scene.json').write_bytes(b'')
    assert bad.scene_problem(tmp_path / 'sj') == 'scene.json が読めません(JSONDecodeError)'
    _corrupt(tmp_path / 'crc' / 'TOF_FRONT.npz')                          # timestamps_us の外が壊れても見つける
    assert bad.scene_problem(tmp_path / 'crc') == 'TOF_FRONT が読めません(BadZipFile)'
    (tmp_path / 'zero' / 'IMU.npz').write_bytes(b'')
    assert bad.scene_problem(tmp_path / 'zero') == 'IMU が読めません(EOFError)'


def test_scorecard_labels_unreadable_log_json(tmp_path, capsys):
    import section_scorecard as ss
    log = '20001024-110002'
    (tmp_path / 'raw' / log).mkdir(parents=True)
    (tmp_path / 'raw' / log / 'log.json').write_bytes(b'')
    (tmp_path / 'labels').mkdir()
    info, lab, src = ss._labels(log, tmp_path / 'labels', tmp_path / 'raw')
    assert lab is None and src == 'computed' and info['error'].startswith('記録の log.json が読めません(JSONDecodeError')
    res = ss.score_log(log, tmp_path / 'labels', tmp_path / 'raw')
    assert res['error'] == info['error'] and res['log'] == log
    # ラベルのファイルが 0 バイト → 記録から作り直す(ここでは記録も読めないので error)
    (tmp_path / 'labels' / (log + '.json')).write_bytes(b'')
    (tmp_path / 'labels' / (log + '.npz')).write_bytes(b'')
    info2, lab2, _ = ss._labels(log, tmp_path / 'labels', tmp_path / 'raw')
    assert lab2 is None and info2['error'] == info['error']
    err = capsys.readouterr().err
    assert 'ラベルのファイルが読めません' in err and '記録から作り直します' in err and '飛ばします' in err
    # 記録の dir はあるが log.json が無い → 飛ばす。記録の dir が無い(名前の間違い)は今まで通り例外(bench_report.car_checks・
    # section_scorecard.build が捕まえる)
    (tmp_path / 'raw' / '20001024-110003').mkdir()
    info3, lab3, _ = ss._labels('20001024-110003', tmp_path / 'labels', tmp_path / 'raw')
    assert lab3 is None and 'FileNotFoundError' in info3['error']
    with pytest.raises(FileNotFoundError):
        ss._labels('20001024-110009', tmp_path / 'labels', tmp_path / 'raw')


def test_car_check_unreadable_log_json(tmp_path):
    import car_check as cc
    log = '20001024-110004'
    (tmp_path / 'raw' / log).mkdir(parents=True)
    (tmp_path / 'raw' / log / 'log.json').write_bytes(b'')
    (tmp_path / 'labels').mkdir()
    r = cc.check_log(log, tmp_path / 'labels', tmp_path / 'raw')
    assert r['judge']['code'] == 3 and 'log.json が読めません' in r['error']
    assert '読めません' in cc.to_text(r)


# ---------------------------------------------------------------- 本物の記録の写し(壊した形 7 つ)
VARIANTS = {
    'good': '20001024-100000', 'zero_scene': '20001024-100001', 'trunc_ai': '20001024-100002', 'crc_veh': '20001024-100003',
    'zero_logjson': '20001024-100004', 'zero_imu_tof': '20001024-100005', 'zero_scenejson': '20001024-100006'}


@pytest.fixture(scope='module')
def broken_raw(tmp_path_factory):
    raw = tmp_path_factory.mktemp('broken') / 'raw'
    raw.mkdir()
    for key, log in VARIANTS.items():
        dst = raw / log
        dst.mkdir()
        meta = json.loads((SRC / 'log.json').read_text(encoding='utf-8'))
        meta['logfile'] = log
        (dst / 'log.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
        for s in sorted(d for d in SRC.iterdir() if d.is_dir()):
            ds = dst / s.name.replace(SRC.name, log)
            ds.mkdir()
            for f in s.iterdir():
                if f.is_file():
                    shutil.copy2(f, ds / f.name)
        s2 = dst / (log + '-0002')
        if key == 'zero_scene':                       # 20261024-162305-0007 と同じ形のシーンを足す
            s4 = dst / (log + '-0004')
            (s4 / 'CAM_FRONT').mkdir(parents=True)
            for n in ('AI_CMD', 'CAM_FRONT', 'IMU', 'TOF_FRONT', 'VEHICLE'):
                (s4 / (n + '.npz')).write_bytes(b'')
            (s4 / 'scene.json').write_bytes(b'')
        elif key == 'trunc_ai':
            _truncate(s2 / 'AI_CMD.npz')
        elif key == 'crc_veh':
            _corrupt(s2 / 'VEHICLE.npz')
        elif key == 'zero_logjson':
            (dst / 'log.json').write_bytes(b'')
        elif key == 'zero_imu_tof':
            for n in ('IMU', 'TOF_FRONT', 'CAM_FRONT'):
                (s2 / (n + '.npz')).write_bytes(b'')
        elif key == 'zero_scenejson':
            (s2 / 'scene.json').write_bytes(b'')
    return raw


@need_src
def test_bench_report_cli_skips_broken_logs(broken_raw, tmp_path):
    """前は log.json が 0 バイトの記録で JSONDecodeError で落ちた。今は外れたこと(deviations)に「読めなかった」と出す"""
    (tmp_path / 'labels').mkdir()
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, str(ROOT / 'tools' / 'bench_report.py'), '--logs'] + list(VARIANTS.values()) +
                       ['--raw', str(broken_raw), '--labels', str(tmp_path / 'labels'), '--out', str(tmp_path / 'br'),
                        '--boot', '50', '--min-laps', '1', '--k-first', '0', '--k-later', '0'],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=600, env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert 'Traceback' not in r.stderr
    dv = (tmp_path / 'br' / 'deviations.md').read_text(encoding='utf-8')
    assert '20001024-100004' in dv and '読めなかった: 記録の log.json が読めません(JSONDecodeError' in dv
    assert '注意: 20001024-100004 の記録の log.json が読めません' in r.stderr
    rep = json.loads((tmp_path / 'br' / 'report.json').read_text(encoding='utf-8'))
    assert rep['days']
    assert not list((tmp_path / 'labels').iterdir())                    # ラベルは書き出さない(その場で作るだけ)


@need_src
def test_loaders_on_broken_copies(broken_raw, tmp_path, capsys):
    import build_ai_dataset as bad
    import course_fit as cf
    import drive_report as dr
    import make_intervention_set as mis
    from jetformer import config
    V = VARIANTS
    old = config.RAW_DIR
    config.RAW_DIR = broken_raw
    try:
        rr = {k: dr.analyze(lg)[0] for k, lg in V.items() if k != 'zero_logjson'}
    finally:
        config.RAW_DIR = old
    assert rr['zero_scene']['frames'] == rr['good']['frames']             # 0 バイトのシーンは飛ばすだけ
    assert rr['trunc_ai']['frames'] < rr['good']['frames']
    # course_fit: Bad CRC の VEHICLE のシーンは飛ばす(前は try の外で落ちた)
    d = cf.LogData(V['crc_veh'], str(broken_raw))
    assert d.scenes == [V['crc_veh'] + '-0001', V['crc_veh'] + '-0003']
    assert cf.LogData(V['good'], str(broken_raw)).scenes == [V['good'] + '-%04d' % i for i in (1, 2, 3)]
    # build_ai_dataset・make_intervention_set: scene.json が 0 バイトのシーン・Bad CRC のシーンを外す
    lbl = tmp_path / 'labels'
    lbl.mkdir()
    for k in ('zero_scenejson', 'crc_veh', 'zero_scene'):
        lg = V[k]
        info, arr = bad.load_labels(lg, lbl, broken_raw)
        use, sk = bad.choose_ai_scenes(lg, bad.LogLabels(lg, info, arr), raw_dir=broken_raw)
        why = dict(sk)
        if k == 'zero_scenejson':
            assert why[lg + '-0002'] == 'scene.json が読めません(JSONDecodeError)' and lg + '-0001' in use
        elif k == 'crc_veh':
            assert why[lg + '-0002'] == 'VEHICLE が読めません(BadZipFile)'
        else:
            assert why[lg + '-0004'] == 'CAM_FRONT が読めません(EOFError)'
        raw_mis, sk2 = mis.load_raw(lg, broken_raw)
        assert raw_mis is not None
    raw_mis, sk2 = mis.load_raw(V['trunc_ai'], broken_raw)               # 壊れた AI_CMD は注意を出して無いものとする
    assert raw_mis is not None and sk2 == []
    err = capsys.readouterr().err
    assert 'AI_CMD.npz が読めません(BadZipFile)' in err
    assert 'Traceback' not in err
