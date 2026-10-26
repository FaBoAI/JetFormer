#!/usr/bin/env python3
"""保留の val のシーンのフレームごとの区間の判定(v1.1)。make_bench_tables.py(10/21 段 G1)と同じやり方(CPU・torch を読まない)

    cd ~/jetformer
    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- nice -n 10 /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261022_prep/desk/zones2.py

やり方(make_bench_tables.main の区間の行をそのまま写した。関数 phase_of_log・prev_idx・定数は make_bench_tables から読む):
  - ログごとに、clean_driving_20261011 のそのログのシーン(train ∪ val)のフレームを時刻の順につなぐ
  - ジャイロの位相: ログ全体の IMU に phase_step(PhaseParams() = v1.1)を因果的に当てる(make_bench_tables.phase_of_log)
  - 区間 = combine_zone(フレームの時刻の直前の IMU の区間(0.1 秒より古ければ unknown), 右旋回中, 合図 (b), v, pred_min_speed=1.0)
    合図 (b) = 人の 1〜10 フレーム先のステア(ラベル)のどれかが 1150 µs 以下 & future_valid
確かめ: 学習のシーンの区間ごとのフレーム数が tables_card.json(10/21 10:36)の zone_frames と同じか(同じでなければ止める)
出力: /home/jetson/jetformer_reports/prep_20261022/desk/pred/zones_heldout.npz(scene・frame・zone・zone_gyro・v・t_us)
      と logs/zones2.json(数・確かめ)
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
MBT_DIR = JF / 'experiments' / '20261021_bench_prep' / 'data'
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(MBT_DIR))
import make_bench_tables as M  # noqa: E402
import label_events as le  # noqa: E402
from jetformer import safety_caps as sc  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = Path('/home/jetson/jetformer_reports/prep_20261022/desk/pred')
CARD = JF / 'data' / 'derived' / 'bench_20261021' / 'tables_card.json'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def zones_of_log(lg, sc_names):
    """make_bench_tables.main の区間の行と同じ → {scene: (zone[n], zone_gyro[n], v[n], t_us[n])}"""
    L = le.load_log(lg)
    it, zone_g, turning, anchors = M.phase_of_log(L)
    A = {s: dict(np.load(M.DATA / 'aligned' / (s + '.npz'))) for s in sc_names}
    t_all = np.concatenate([A[s]['timestamps_us'].astype(np.int64) for s in sc_names]) / 1e6
    owner = np.concatenate([np.full(len(A[s]['timestamps_us']), j) for j, s in enumerate(sc_names)])
    fidx = np.concatenate([np.arange(len(A[s]['timestamps_us'])) for s in sc_names])
    o = np.argsort(t_all, kind='stable')
    t_all, owner, fidx = t_all[o], owner[o], fidx[o]
    veh = np.stack([A[sc_names[w]]['vehicle'][f] for w, f in zip(owner, fidx)])
    fut = np.stack([A[sc_names[w]]['future_actions'][f] for w, f in zip(owner, fidx)])
    fvalid = np.array([bool(A[sc_names[w]]['future_valid'][f]) for w, f in zip(owner, fidx)])
    v = -veh[:, 3].astype(np.float64) * M.KV
    ki = M.prev_idx(it, t_all)
    imu_fresh = np.abs(t_all - it[ki]) < 0.1
    pred_sig = (fut[:, :, 0] <= M.PRED_TH).any(axis=1) & fvalid
    zg = zone_g[ki]
    zone = np.array([sc.combine_zone(z if f else 'unknown', bool(tr) and f, bool(ps), v=float(vv),
                                     pred_min_speed=sc.PRED_MIN_SPEED)
                     for z, tr, ps, vv, f in zip(zg, turning[ki], pred_sig, v, imu_fresh)], dtype=object)
    zone_gyro = np.array([z if f else 'unknown' for z, f in zip(zg, imu_fresh)], dtype=object)
    out = {}
    for j, s in enumerate(sc_names):
        m = owner == j
        order = np.argsort(fidx[m])
        gi = np.nonzero(m)[0][order]
        n = len(A[s]['timestamps_us'])
        assert np.array_equal(fidx[gi], np.arange(n))
        out[s] = (zone[gi].astype(str), zone_gyro[gi].astype(str), v[gi], A[s]['timestamps_us'].astype(np.int64))
    return out, len(anchors)


def main():
    t0 = time.time()
    splits = json.loads((M.DATA / 'splits.json').read_text())
    train_set, val_set = set(splits['train']), set(splits['val'])
    scenes_all = sorted(train_set | val_set)
    logs = sorted({s.rsplit('-', 1)[0] for s in scenes_all})
    zone_train = {z: 0 for z in sc.ZONES}
    zone_gyro_train = {z: 0 for z in sc.ZONES}
    zone_val = {z: 0 for z in sc.ZONES}
    rows = []
    for lg in logs:
        sc_names = sorted(s for s in scenes_all if s.rsplit('-', 1)[0] == lg)
        Z, na = zones_of_log(lg, sc_names)
        for s, (z, zg, v, t) in Z.items():
            if s in val_set:
                rows.append((s, z, zg, v, t))
                for name in sc.ZONES:
                    zone_val[name] += int((z == name).sum())
            else:
                for name in sc.ZONES:
                    zone_train[name] += int((z == name).sum())
                    zone_gyro_train[name] += int((zg == name).sum())
        print(time.strftime('%H:%M:%S'), lg, len(sc_names), 'シーン', flush=True)
    card = json.loads(CARD.read_text())
    want, want_g = card['stats']['zone_frames'], card['stats']['zone_gyro_only_frames']
    same = all(int(want.get(z, 0)) == zone_train[z] for z in sc.ZONES)
    same_g = all(int(want_g.get(z, 0)) == zone_gyro_train[z] for z in sc.ZONES)
    rep = {'created_at': time.strftime('%Y-%m-%dT%H:%M:%S+09:00'), 'script': str(Path(__file__).resolve()),
           'way': 'make_bench_tables.py と同じ(phase_of_log・combine_zone・合図 (b) = 人の先のステア ≤ 1150 µs)',
           'dataset': 'clean_driving_20261011', 'val_scenes': sorted(val_set), 'logs': logs,
           'zone_frames_val': zone_val, 'zone_frames_train': zone_train, 'zone_frames_train_card': want,
           'zone_gyro_only_frames_train': zone_gyro_train, 'zone_gyro_only_frames_train_card': want_g,
           'check_train_counts_same_as_card': bool(same and same_g),
           'card': {'file': str(CARD), 'created_at': card.get('created_at')},
           'safety_caps_sha256': sha256(JF / 'jetformer' / 'safety_caps.py'),
           'make_bench_tables_sha256': sha256(MBT_DIR / 'make_bench_tables.py')}
    if not (same and same_g):
        (HERE / 'logs').mkdir(parents=True, exist_ok=True)
        (HERE / 'logs' / 'zones2.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        raise SystemExit('学習のシーンの区間の数が tables_card.json と違います: %s / %s' % (zone_train, want))
    scene = np.concatenate([np.full(len(r[1]), r[0], dtype='<U20') for r in rows])
    frame = np.concatenate([np.arange(len(r[1]), dtype=np.int32) for r in rows])
    zone = np.concatenate([r[1] for r in rows]).astype('<U8')
    zone_g = np.concatenate([r[2] for r in rows]).astype('<U8')
    v = np.concatenate([r[3] for r in rows]).astype(np.float32)
    t_us = np.concatenate([r[4] for r in rows])
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / 'zones_heldout.tmp.npz'
    np.savez_compressed(tmp, scene=scene, frame=frame, zone=zone, zone_gyro=zone_g, v=v, t_us=t_us,
                        note=np.array('保留の val のシーン(clean_driving_20261011)のフレームごとの v1.1 の区間。zones2.py'))
    os.replace(tmp, OUT / 'zones_heldout.npz')
    rep['file'] = str(OUT / 'zones_heldout.npz')
    rep['sha256'] = sha256(OUT / 'zones_heldout.npz')
    rep['frames_val'] = int(len(scene))
    rep['seconds'] = round(time.time() - t0, 1)
    (HERE / 'logs').mkdir(parents=True, exist_ok=True)
    (HERE / 'logs' / 'zones2.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: rep[k] for k in ('zone_frames_val', 'zone_frames_train', 'check_train_counts_same_as_card',
                                          'frames_val', 'seconds')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
