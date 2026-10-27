#!/usr/bin/env python3
"""ToF ガードの床の基準(data/tof_floor_ref.npy)を、人の自由走行の記録から作り直す(2026-10-21 Fix-all、V-protocol major 6)

    python tools/tof_floor_ref.py 20261023-090512                 # 作って比べるだけ(何も書かない。--out-dir DIR で作った基準を置く)
    python tools/tof_floor_ref.py 20261023-090512 --install       # 今の基準を控えてから入れ替える(走行の開始のたびに読み直す)

作り方(2026-10-11 の夜の解析 experiments/20261011_overnight/runs/src/tof_recalc.py の build_ref と同じ):
  まっすぐ速く走っているフレーム(|eRPM| > 1,500・|ジャイロ Z| < 60 deg/s)の、信頼度 4 の画素の距離の 80 パーセンタイル [mm]。
  値の無い画素は今の基準の値で埋める(無ければ 0)。

入れ替えるのは、次を全部満たすときだけ(満たさなければ入れ替えずに理由を出し、終了コード 2):
  - まっすぐ速いフレームが MIN_FRAMES(150 = 5 秒)以上
  - 値のある画素が、ガードの使う行(4〜27 行)の FILL_MIN(90%)以上
  - ガードの使う行の中央値が、今の基準から ±MAX_SHIFT_PCT(15%)の中(大きくずれるなら取り付けを先に見る)

注意:
  - ライブ・収集タブの「点検する」(ToF ±6%)は、学習データの向きの基準 data/tof_check_ref.npy と比べる。この床の基準とは別の
    ファイルで、この道具は変えない(床の基準を作り直しても「点検する」の結果は変わらない)。
  - 床の基準を使うのは ToF ガードだけ。ベンチマークの基準の値は tof_guard 1(記録だけ)なので、車の動きは変わらない。
  - 読み取りだけ(numpy、GPU なし)。--install のときだけ data/tof_floor_ref.npy を書く(控え: data/tof_floor_ref.<時刻>.bak.npy、
    作った基準の写し: data/derived/tof_floor_ref/)。
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jetformer import config, tof_format  # noqa: E402

FLOOR_REF_PATH = config.DATA_DIR / 'tof_floor_ref.npy'
ERPM_MIN = 1500.0
GZ_MAX_DPS = 60.0
PCTL = 80.0
GOOD_CONF = 4
GUARD_ROWS = slice(4, 28)          # jetformer/tof_guard.ToFGuard.ROWS
MIN_FRAMES = 150
FILL_MIN = 0.90
MAX_SHIFT_PCT = 15.0
SHAPE = (42, 54)


def _npz(p):
    try:
        with np.load(p, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    except Exception as e:     # 壊れたファイル(電源断など)は飛ばす。0 バイトは EOFError・壊れた zip は BadZipFile(2026-10-26 fix2)
        return e


def load(log, raw_dir=None):
    """記録の TOF_FRONT・VEHICLE・IMU をつなぐ → (tof_t_us, depth, veh_t_us, erpm, imu_t_us, gz, problems)"""
    root = Path(raw_dir or config.RAW_DIR) / log
    if not root.is_dir():
        raise FileNotFoundError('記録がありません: %s' % root)
    parts = {'TOF_FRONT': [], 'VEHICLE': [], 'IMU': []}
    problems = []
    for s in sorted(d for d in root.iterdir() if d.is_dir()):
        for n in parts:
            p = s / (n + '.npz')
            if not p.exists():
                continue
            d = _npz(p)
            if isinstance(d, Exception) or 'timestamps_us' not in d:
                problems.append('%s/%s: 読めません' % (s.name, n))
                print('注意: %s/%s/%s.npz が読めません(%s)。飛ばします' % (log, s.name, n, type(d).__name__ if isinstance(
                    d, Exception) else '形式'), file=sys.stderr)
                continue
            parts[n].append(d)
    if not parts['TOF_FRONT'] or not parts['VEHICLE'] or not parts['IMU']:
        raise ValueError('TOF_FRONT・VEHICLE・IMU のどれかがありません(%s)' % log)
    cat = lambda n, k: np.concatenate([d[k] for d in parts[n]])      # noqa: E731
    return (cat('TOF_FRONT', 'timestamps_us'), cat('TOF_FRONT', 'depth'), cat('VEHICLE', 'timestamps_us'),
            cat('VEHICLE', 'erpm').astype(np.float64), cat('IMU', 'timestamps_us'),
            cat('IMU', 'gyro')[:, 2].astype(np.float64), problems)


def build(tof_t, depth, veh_t, erpm, imu_t, gz, old_ref=None):
    """床の基準 [mm] (42, 54) float32 と、使ったフレームの数・値のあった画素の割合"""
    tof_t = np.asarray(tof_t, np.float64)
    er = np.interp(tof_t, np.asarray(veh_t, np.float64), erpm)
    g = np.interp(tof_t, np.asarray(imu_t, np.float64), gz)
    m = (np.abs(er) > ERPM_MIN) & (np.abs(g) < GZ_MAX_DPS)
    n = int(m.sum())
    if n == 0:
        return None, 0, 0.0
    raw = np.asarray(depth)[m]
    dist = tof_format.distance_mm(raw).astype(np.float64)
    conf = tof_format.confidence(raw)
    x = np.where((conf == GOOD_CONF) & (dist > 0), dist, np.nan)
    with warnings.catch_warnings():               # 値の無い画素(全部 NaN)の警告は出さない(下で今の基準で埋める)
        warnings.simplefilter('ignore', RuntimeWarning)
        fl = np.nanpercentile(x, PCTL, axis=0)
    have = np.isfinite(fl)
    fill = float(have[GUARD_ROWS].mean())
    fallback = np.asarray(old_ref, np.float64) if old_ref is not None else np.zeros(SHAPE)
    fl = np.where(have, fl, fallback)
    return fl.astype(np.float32), n, fill


def compare(new, old):
    """ガードの使う行の中央値のずれ [%](新 / 今 − 1)と、行 9・18 の中央の列(「点検する」と同じ場所)のずれ"""
    if old is None:
        return {}
    old = np.asarray(old, np.float64)
    new = np.asarray(new, np.float64)
    ok = (old[GUARD_ROWS] > 0) & (new[GUARD_ROWS] > 0)
    out = {'guard_rows_median_shift_pct': round(float(np.median(new[GUARD_ROWS][ok] / old[GUARD_ROWS][ok]) - 1.0) * 100, 2)
           if ok.any() else None}
    for r in (9, 18):
        a, b = float(np.median(new[r, 20:35])), float(np.median(old[r, 20:35]))
        out['row%d_center_shift_pct' % r] = round((a / b - 1.0) * 100, 2) if b > 0 else None
    return out


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def run(log, raw_dir=None, ref_path=FLOOR_REF_PATH, out_dir=None, install=False, stamp=None):
    """返り値 (report dict, 終了コード)。install=True で、条件を満たせば ref_path を入れ替える(先に控える)"""
    ref_path = Path(ref_path)
    old = None
    if ref_path.exists():
        try:
            old = np.load(ref_path)
        except (OSError, ValueError):
            old = None
    tof_t, depth, veh_t, erpm, imu_t, gz, problems = load(log, raw_dir)
    new, n, fill = build(tof_t, depth, veh_t, erpm, imu_t, gz, old)
    rep = {'log': log, 'frames_tof': int(len(tof_t)), 'frames_used': n, 'fill_guard_rows': round(fill, 4),
           'rule': {'erpm_min': ERPM_MIN, 'gz_max_dps': GZ_MAX_DPS, 'percentile': PCTL, 'conf': GOOD_CONF},
           'problems': problems, 'current_ref': str(ref_path) if old is not None else None,
           'current_sha256': sha256(ref_path) if old is not None else None, 'installed': False, 'reasons': []}
    if new is None or n < MIN_FRAMES:
        rep['reasons'].append('まっすぐ速いフレームが %d(%d 以上が要る。|eRPM| > %d・|ジャイロ Z| < %d deg/s の直線を 5 秒以上)' % (
            n, MIN_FRAMES, ERPM_MIN, GZ_MAX_DPS))
    if new is not None and fill < FILL_MIN:
        rep['reasons'].append('値のある画素がガードの行の %.0f%%(%.0f%% 以上が要る)' % (fill * 100, FILL_MIN * 100))
    if new is not None:
        rep['compare'] = compare(new, old)
        sh = rep['compare'].get('guard_rows_median_shift_pct')
        if sh is not None and abs(sh) > MAX_SHIFT_PCT:
            rep['reasons'].append('今の基準から %+.1f%% ずれている(±%.0f%% まで)。取り付け(ToF の向き)を先に見る' % (sh, MAX_SHIFT_PCT))
    stamp = stamp or datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if new is not None and (out_dir or install):
        od = Path(out_dir) if out_dir else ref_path.parent / 'derived' / 'tof_floor_ref'
        od.mkdir(parents=True, exist_ok=True)
        cand = od / ('tof_floor_ref_%s.npy' % log)
        np.save(cand, new)
        rep['candidate'] = str(cand)
        rep['candidate_sha256'] = sha256(cand)
    ok = new is not None and not rep['reasons']
    rep['ok'] = ok
    if install and ok:
        if old is not None:
            bak = ref_path.with_name('tof_floor_ref.%s.bak.npy' % stamp)
            bak.write_bytes(ref_path.read_bytes())
            rep['backup'] = str(bak)
        tmp = ref_path.with_name('.tof_floor_ref.tmp.npy')
        np.save(tmp, new)
        os.replace(tmp, ref_path)                       # 途中で電源が落ちても、古いか新しいかのどちらか
        rep['installed'] = True
        rep['installed_sha256'] = sha256(ref_path)
    return rep, (0 if ok else 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('log', help='人の自由走行の記録(data/raw/<log>)。収集オンで 60〜90 秒、直線をふつうの速さで')
    ap.add_argument('--raw-dir', default=None)
    ap.add_argument('--ref', default=str(FLOOR_REF_PATH), help='入れ替える床の基準(既定 data/tof_floor_ref.npy)')
    ap.add_argument('--out-dir', default=None, help='作った基準の置き場(既定: 置かない。--install なら data/derived/tof_floor_ref/)')
    ap.add_argument('--install', action='store_true', help='条件を満たせば --ref を入れ替える(今のものを控える)')
    a = ap.parse_args(argv)
    try:
        rep, code = run(a.log, a.raw_dir, a.ref, a.out_dir, a.install)
    except (OSError, ValueError) as e:
        print(json.dumps({'ok': False, 'error': str(e)}, ensure_ascii=False))
        return 2
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if rep['reasons']:
        print('入れ替えません: ' + ' / '.join(rep['reasons']), file=sys.stderr)
    elif a.install:
        print('入れ替えました(控え %s)。次の走行の開始から使います' % rep.get('backup', 'なし'), file=sys.stderr)
    else:
        print('作れました。入れ替えるときは --install を付けて同じコマンド', file=sys.stderr)
    return code


if __name__ == '__main__':
    sys.exit(main())
