"""反証 8-1 回目・係 safety: 本物の記録(data/raw の AI 走行。読むだけ)を、本物の drive._loop に開いたループで流し直し、
fix7 の直し(s4-a・s4-b・s4-c)の前後で出す値が変わらないかを、記録の始めから終わりまで(発進の区間だけでなく)比べる。
流し方は fix8/decided_safety/rec_replay10.py と同じ(その RecFeeder・RecRunner・load_rec を読み込んで使う。開いたループ:
車は記録のとおり動いたまま)。版: now(今の jetformer/)・f7b(tests/fixtures/fix7_safety_before = fix7 の前)。
スイッチ: ON(rec_back_guard 1・rec_hold_keep 1)・OFF(両方 0)。そのほかの設定は記録の log.json の params
(launch_boost 0・天井 0・自動合わせ 0・launch_no_kick 0。区間の上限は記録の値)。
比べるもの: PCA9685 に書いた値・記録の行(AI_CMD の全部の欄。infer_ms のほか)・出来事。
  now OFF と f7b OFF(全部 OFF のビット単位の同じ)・now ON と f7b ON(fix7 の直しが本物の記録で出す値を変えるか)・
  now ON と now OFF(参考: 見張り・残すが本物の記録で出す値を変える所)
出力 out/rec_replay11.json。装置・GPU・API には触れない
    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -I -B experiments/20261026_fix/fix8/review_safety_r1/rec_replay11.py [i/n]
"""
import collections
import json
import shutil
import sys
import tempfile
import time as _time
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'decided_safety'))
import h11  # noqa: E402
import rec_replay10 as RR  # noqa: E402

from jetformer import clock, config  # noqa: E402
from test_drive_caps import FakeRecorder, fake_normalize  # noqa: E402

RAW = h11.ROOT / 'data' / 'raw'
TMP = h11.TMP.parent / 'rec'
A = h11.A


def run_v(name, rec, params, t_from_us, t_to_us, floor):
    mod, amod = h11.load(name)
    TMP.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(TMP)) as td:
        tmp_path = Path(td)
        box = {'ns': 0, 'm': 1000.0}
        mp.setattr(clock, 'now_ns', lambda: box['ns'])
        mp.setattr(mod, 'time', types.SimpleNamespace(monotonic=lambda: box['m'], perf_counter=_time.perf_counter,
                                                      time=lambda: 1.0, sleep=lambda s: None, strftime=_time.strftime))
        mp.setattr(mod, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
        fp = tmp_path / 'floor_ref.npy'
        if floor is not None:
            np.save(fp, floor)
        mp.setattr(mod, 'FLOOR_REF_PATH', fp if floor is not None else tmp_path / 'none.npy')
        recd = FakeRecorder()
        d = mod.Driver(hub=None, recorder=recd)
        d.params = dict(params)
        d.hb_token, d.hb_count = 'tok', 1
        d.st = {'phase': 'driving', 'mode_id': 4}
        pca = RR.TPCA(box)
        ref = (float(params['_ref'][0]), float(params['_ref'][1]))
        out = mod._Output(pca, config.PCA9685, ref, lambda: d.params)
        bufs = (collections.deque(maxlen=10), collections.deque(maxlen=60), collections.deque(maxlen=50))
        cam_slot = [None]
        fd = RR.RecFeeder(d, rec, bufs, cam_slot, box, t_from_us, t_to_us)
        d._loop(d.st, RR.RecRunner(fd), fake_normalize, pca, out, ref, 6, 0.0, cam_slot, fd, *bufs, False)
        d._stop.clear()
        stats = dict(d.st.get('stats') or {})
    return fd, recd, pca, stats


def summary(r):
    rows = r[1].rows
    return {'rows': len(rows), 'pca_writes': len(r[2].w),
            'rec_frames': sum(1 for t, x in rows if int(x['assist']) == A.ASSIST_RECOVERY),
            'giveup_frames': sum(1 for t, x in rows if int(x['assist']) == A.ASSIST_GIVEUP),
            'guards': collections.Counter(e[1].get('what') for e in r[1].events if e[0] == 'recovery_guard'),
            'recoveries': int(r[3].get('recoveries', 0))}


def main(part='0/1'):
    h11.check_now()
    pi, pn = (int(x) for x in part.split('/'))
    RR._EPOCH_OFF[0] = int(clock.to_epoch_us(0)) * 1000
    logs = sorted({p.parent.parent.name for p in RAW.glob('*/*/AI_CMD.npz')})
    fpath = Path(h11.drive.FLOOR_REF_PATH)
    floor = np.load(fpath) if fpath.exists() else None
    res = {'sha': h11.check_now(), 'part': part, 'logs': {}, 'started': _time.strftime('%F %T')}
    t_start = _time.time()
    try:
        for li, log in enumerate(logs):
            if li % pn != pi:
                continue
            try:
                meta, rec, problems = RR.load_rec(log)
            except Exception as e:      # noqa: BLE001
                res['logs'][log] = {'error': repr(e)[:200]}
                continue
            if not all(k in rec for k in ('AI_CMD', 'VEHICLE', 'IMU', 'TOF_FRONT')) or not len(rec['AI_CMD']['timestamps_us']):
                res['logs'][log] = {'error': 'センサーが足りない %s' % sorted(rec), 'problems': problems}
                print('skip', log, sorted(rec), flush=True)
                continue
            dr = meta.get('drive') or {}
            p = dict(h11.drive.DEFAULT_PARAMS)
            p.update({k: v for k, v in (dr.get('params') or {}).items() if k in h11.drive.DEFAULT_PARAMS})
            p.update(speed_ceiling=0, speed_auto=0, launch_no_kick=0, launch_boost=0)
            p['_ref'] = (float(dr.get('steer_center_us', 1441)), float(dr.get('throttle_neutral_us', 1496)))
            Z = rec['AI_CMD']
            t_lo, t_hi = int(Z['timestamps_us'][0]), int(Z['timestamps_us'][-1])
            rr = {}
            for ver in ('now', 'f7b'):
                for sw, fn in (('on', h11.ON), ('off', h11.OFF)):
                    rr[(ver, sw)] = run_v(ver, rec, fn(p), t_lo, t_hi, floor)
            ent = {'problems': problems, 'n_frames': int(len(Z['timestamps_us']))}
            for a, b, key in ((('now', 'off'), ('f7b', 'off'), 'off_now_vs_f7b'), (('now', 'on'), ('f7b', 'on'), 'on_now_vs_f7b'),
                              (('now', 'on'), ('now', 'off'), 'now_on_vs_off')):
                s_, d_ = RR.first_diff(rr[a], rr[b])
                ent[key] = {'same': s_, 'diff': d_}
                if d_ is not None and 't_ns' in d_:
                    ent[key]['diff_s'] = round((d_['t_ns'] / 1000 - t_lo) / 1e6, 3)
            # PCA9685 の値だけ(出来事は除く)の比べも
            for a, b, key in ((('now', 'on'), ('f7b', 'on'), 'on_now_vs_f7b_pca'),):
                ent[key] = rr[a][2].w == rr[b][2].w
            ent['sum'] = {'%s_%s' % k: summary(v) for k, v in rr.items()}
            try:
                ent['fidelity_off_now'] = RR.fidelity(rr[('now', 'off')], Z)
            except KeyError as e:          # 10/10 の記録には assist の欄が無い
                ent['fidelity_off_now'] = {'skipped': repr(e)}
            res['logs'][log] = ent
            print(log, 'off_eq', ent['off_now_vs_f7b']['same'], 'on_eq', ent['on_now_vs_f7b']['same'],
                  'on_pca_eq', ent['on_now_vs_f7b_pca'], 'on_vs_off', ent['now_on_vs_off']['same'],
                  ent['now_on_vs_off'].get('diff_s'), dict(ent['sum']['now_on']['guards']),
                  '%.0fs' % (_time.time() - t_start), flush=True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
        h11.cleanup()
    res['elapsed_s'] = round(_time.time() - t_start, 1)
    (HERE / 'out' / ('rec_replay11_%s.json' % part.replace('/', 'of'))).write_text(
        json.dumps(res, ensure_ascii=False, default=str, indent=0))
    print('done', len(res['logs']), res['elapsed_s'])


if __name__ == '__main__':
    main(*sys.argv[1:2])
