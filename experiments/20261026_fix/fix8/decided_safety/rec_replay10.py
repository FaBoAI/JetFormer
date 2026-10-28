"""fix8 係 safety-decided: 本物の記録(10/23・10/24 の AI 走行。data/raw を読むだけ)を、本物の drive._loop に開いたループで
流し直し、ON(rec_back_guard 1・rec_hold_keep 1)と OFF(両方 0)の PCA9685 に書く値・記録の行(AI_CMD の全部の欄)・出来事を
フレームごとに比べる。

流し方(開いたループ。車は記録のとおり動いたまま = 出す値が違っても次の入力は同じ):
  - カメラ: 記録の AI_CMD の行(= 走行ループが処理したフレーム)の撮像時刻に、5 ms 遅れで届く。走行ループの処理の終わりは
    撮像時刻 + 記録の latency_ms(中立止めの行など無いときは 5 ms)。次のフレームが 0.1 秒より来なければ、走行ループは
    本物と同じく中立止め(カメラの映像が来ていません)
  - モデル: 記録の model_pred(後処理・重ね合わせの前の [10,2] µs)をそのまま返す。記録で中立止めだった行(model_pred が NaN)は
    「推論結果が数値ではありません」の中立止めになる(本物と同じ行で中立止め。理由の文だけ違うことがある)
  - センサー: VEHICLE(100 Hz)・IMU(100 Hz。gyro は deg/s → rad/s)・TOF_FRONT を記録の時刻で、その時刻になったら入れる
  - 設定: その記録の log.json の drive.params(ゲイン・先読み・ToF ガード・start_kick・rev_s・rev_us など)に、
    launch_boost 0 / 1 × 見張り・残す ON / OFF。区間の上限は記録の値、天井・自動合わせは 0(その日はまだ無い)。フレームごとの
    ゲイン・トリム・先読みは記録の AI_CMD の値。床の基準は本物の drive.FLOOR_REF_PATH(data/tof_floor_ref.npy)の写し
比べる区間: 発進(diag_launch/collect.py --day の行: t_ai)から最初の周の終わり(first_lap_end_s。無ければ AI の区間の終わり + 0.5 秒)。
記録全体の初めて違ったフレームも出す。
    python -I rec_replay10.py [--logs a,b,...]   → out/rec_replay10.json
"""
import argparse
import collections
import hashlib
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
import h10  # noqa: E402

from jetformer import clock, config, drive  # noqa: E402
from jetformer import assist as A  # noqa: E402
from jetformer.sensors.base import Sample  # noqa: E402
from test_drive_caps import FakeRecorder, fake_normalize  # noqa: E402

ROOT = Path('/home/jetson/jetformer')
RAW = ROOT / 'data' / 'raw'
LAB = ROOT / 'data' / 'labels'
OUT = HERE / 'out'
TMP = h10.TMP.parent / 'rec'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_rec(log):
    """記録の全シーンをつなぐ。壊れた・無い npz は飛ばして problems に"""
    root = RAW / log
    meta = json.loads((root / 'log.json').read_text(encoding='utf-8'))
    parts = collections.defaultdict(list)
    problems = []
    for s in sorted(d.name for d in root.iterdir() if d.is_dir() and d.name.startswith(log + '-')):
        for n in ('AI_CMD', 'VEHICLE', 'IMU', 'TOF_FRONT'):
            p = root / s / (n + '.npz')
            if not p.exists() or p.stat().st_size == 0:
                problems.append('%s/%s なし' % (s, n))
                continue
            try:
                z = np.load(p)
                d = {k: z[k] for k in z.files if k != 'quality'}
            except Exception as e:      # noqa: BLE001
                problems.append('%s/%s 読めない %r' % (s, n, e))
                continue
            if 'timestamps_us' in d and len(d['timestamps_us']):
                parts[n].append(d)
    out = {}
    for n, ds in parts.items():
        keys = set.intersection(*[set(d) for d in ds])
        out[n] = {k: np.concatenate([d[k] for d in ds]) for k in keys}
        o = np.argsort(out[n]['timestamps_us'], kind='stable')
        out[n] = {k: v[o] for k, v in out[n].items()}
    return meta, out, problems


class RecFeeder(object):
    """cam_evt の代わり(wait / clear / set)。記録の時刻で時計・センサー・カメラを進める"""

    def __init__(self, d, rec, bufs, cam_slot, box, t_from_us, t_to_us):
        self.d, self.box, self.cam_slot = d, box, cam_slot
        self.tof_buf, self.imu_buf, self.veh_buf = bufs
        Z = rec['AI_CMD']
        m = (Z['timestamps_us'] >= t_from_us) & (Z['timestamps_us'] <= t_to_us)
        self.ft = Z['timestamps_us'][m].astype(np.int64) * 1000
        self.fp = Z['model_pred'][m].astype(np.float64) if 'model_pred' in Z else Z['pred'][m].astype(np.float64)
        lat = Z['latency_ms'][m].astype(np.float64)
        # フレームごとのゲイン・トリム・先読み(記録の値。走行中に変えた本も記録どおりに流す。走行ループは各回の始めに params を読むので、
        # wait() が「次に渡すフレーム」の値を入れておく)
        self.fg = Z['gains'][m].astype(np.float64) if 'gains' in Z else None
        self.ftrim = Z['trim'][m].astype(np.float64) if 'trim' in Z else None
        self.flook = Z['lookahead'][m].astype(np.int64) if 'lookahead' in Z else None
        self.fend = self.ft + np.where(np.isfinite(lat), lat * 1e6, 5e6).astype(np.int64)
        V, I, T = rec['VEHICLE'], rec['IMU'], rec['TOF_FRONT']
        self.vt = V['timestamps_us'].astype(np.int64) * 1000
        self.V = V
        self.it = I['timestamps_us'].astype(np.int64) * 1000
        self.imu = np.concatenate([I['accel'].astype(np.float32), np.deg2rad(I['gyro'].astype(np.float32))], axis=1)
        self.tt = T['timestamps_us'].astype(np.int64) * 1000
        self.T = T['depth']
        self.i = 0
        self.kv = self.ki = self.kt = 0
        self.now = int(self.ft[0]) - 200_000_000 if len(self.ft) else 0
        self.cur = -1
        self.timeouts = []
        self._set_clock()
        self._params_for(0)

    def _params_for(self, j):
        if self.fg is None or j >= len(self.ft):
            return
        p = dict(self.d.params)
        if np.all(np.isfinite(self.fg[j])):
            p['steer_gain'], p['throttle_gain'] = float(self.fg[j][0]), float(self.fg[j][1])
        if self.ftrim is not None and np.isfinite(self.ftrim[j]):
            p['steer_trim_us'] = float(self.ftrim[j])
        if self.flook is not None:
            p['lookahead'] = int(self.flook[j])
        self.d.params = p

    def _set_clock(self):
        self.box['ns'] = int(self.now)
        self.box['m'] = self.now / 1e9 - self.ft[0] / 1e9 + 1000.0 if len(self.ft) else 1000.0
        self.d.heartbeat_t = self.box['m']

    def _push(self, upto):
        while self.kv < len(self.vt) and self.vt[self.kv] <= upto:
            k = self.kv
            self.veh_buf.append((int(self.vt[k]), {'pwm': [int(x) for x in self.V['pwm'][k]],
                                                  'valid': [bool(x) for x in self.V['valid'][k]],
                                                  'select': bool(self.V['select'][k]), 'moving': bool(self.V['moving'][k]),
                                                  'erpm': int(self.V['erpm'][k])}))
            self.kv += 1
        while self.ki < len(self.it) and self.it[self.ki] <= upto:
            self.imu_buf.append((int(self.it[self.ki]), self.imu[self.ki]))
            self.ki += 1
        while self.kt < len(self.tt) and self.tt[self.kt] <= upto:
            self.tof_buf.append((int(self.tt[self.kt]), self.T[self.kt]))
            self.kt += 1

    def wait(self, timeout):
        if self.cur >= 0:                     # 前のフレームの処理の終わり
            self.now = max(self.now, int(self.fend[self.cur]))
        if self.i >= len(self.ft):
            self.d._stop.set()
            return False
        arrive = int(self.ft[self.i]) + 5_000_000
        if arrive - self.now > timeout * 1e9 + 1:
            self.now += int(timeout * 1e9)
            self._push(self.now)
            self._set_clock()
            self.cur = -1
            self.timeouts.append(self.now)
            self._params_for(self.i)
            return False
        self.now = max(self.now, arrive)
        self._push(self.now)
        self.cam_slot[0] = Sample(int(self.ft[self.i]), {'image': np.zeros((8, 8, 3), np.uint8), 'seq': self.i})
        self.cur = self.i
        self.i += 1
        self._set_clock()
        self._params_for(self.i)
        return True

    def clear(self):
        pass

    def set(self):
        pass


class RecRunner(object):
    def __init__(self, feeder):
        self.f = feeder

    def run_numpy(self, **kw):
        p = self.f.fp[self.f.cur]
        return {'actions': ((np.asarray(p, np.float32) - drive.PWM_CENTER) / drive.PWM_SCALE)[None]}


class TPCA(object):
    def __init__(self, box):
        self.box, self.w = box, []

    def set_pair_us(self, sc_, s, tc, t):
        self.w.append((int(self.box['ns']), float(s), float(t)))

    def set_us(self, ch, us):
        self.w.append((int(self.box['ns']), 'ch%d' % ch, float(us)))

    def set_osc_scale(self, scale):
        pass


def run(rec, params, t_from_us, t_to_us, floor):
    TMP.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(TMP)) as td:
        tmp_path = Path(td)
        box = {'ns': 0, 'm': 1000.0}
        mp.setattr(clock, 'now_ns', lambda: box['ns'])
        mp.setattr(drive, 'time', types.SimpleNamespace(monotonic=lambda: box['m'], perf_counter=_time.perf_counter,
                                                        time=lambda: 1.0, sleep=lambda s: None, strftime=_time.strftime))
        mp.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
        fp = tmp_path / 'floor_ref.npy'
        if floor is not None:
            np.save(fp, floor)
        mp.setattr(drive, 'FLOOR_REF_PATH', fp if floor is not None else tmp_path / 'none.npy')
        recd = FakeRecorder()
        d = drive.Driver(hub=None, recorder=recd)
        d.params = dict(params)
        d.hb_token, d.hb_count = 'tok', 1
        d.st = {'phase': 'driving', 'mode_id': int(params.get('_mode_id', 4))}
        pca = TPCA(box)
        ref = (float(params['_ref'][0]), float(params['_ref'][1]))
        out = drive._Output(pca, config.PCA9685, ref, lambda: d.params)
        bufs = (collections.deque(maxlen=10), collections.deque(maxlen=60), collections.deque(maxlen=50))
        cam_slot = [None]
        fd = RecFeeder(d, rec, bufs, cam_slot, box, t_from_us, t_to_us)
        d._loop(d.st, RecRunner(fd), fake_normalize, pca, out, ref, 6, 0.0, cam_slot, fd, *bufs, False)
        d._stop.clear()
        stats = dict(d.st.get('stats') or {})
    return fd, recd, pca, stats


def first_diff(a, b, t_lo_ns=None, t_hi_ns=None):
    """(feeder, rec, pca, stats) 2 つを比べる → (同じか, 最初の違い)。t_lo_ns〜t_hi_ns のフレーム・書き込みだけ"""
    def inwin(t):
        return (t_lo_ns is None or t >= t_lo_ns) and (t_hi_ns is None or t <= t_hi_ns)
    wa = [w for w in a[2].w if inwin(w[0])]
    wb = [w for w in b[2].w if inwin(w[0])]
    if wa != wb:
        i = next((k for k, (x, y) in enumerate(zip(wa, wb)) if x != y), min(len(wa), len(wb)))
        return False, {'what': 'pca', 't_ns': (wa[i] if i < len(wa) else wb[i])[0], 'on': wa[i] if i < len(wa) else None,
                       'off': wb[i] if i < len(wb) else None}
    ra = [(t, r) for t, r in a[1].rows if inwin(t)]
    rb = [(t, r) for t, r in b[1].rows if inwin(t)]
    for (t0, r0), (t1, r1) in zip(ra, rb):
        bad = [c for c in r0 if c != 'infer_ms' and (c not in r1 or not h10._same(r0[c], r1[c]))]
        if t0 != t1 or set(r0) != set(r1) or bad:
            return False, {'what': 'row', 't_ns': t0, 'cols': bad, 'on': {c: h10._short(r0[c]) for c in bad},
                           'off': {c: h10._short(r1.get(c)) for c in bad}}
    if len(ra) != len(rb):
        return False, {'what': 'rows_len', 'on': len(ra), 'off': len(rb)}
    ea = [e for e in a[1].events if _ev_in(e, inwin)]
    eb = [e for e in b[1].events if _ev_in(e, inwin)]
    if len(ea) != len(eb) or any(x[0] != y[0] or not h10._same(dict(x[1]), dict(y[1])) for x, y in zip(ea, eb)):
        k = next((k for k, (x, y) in enumerate(zip(ea, eb)) if x[0] != y[0] or not h10._same(dict(x[1]), dict(y[1]))),
                 min(len(ea), len(eb)))
        return False, {'what': 'event', 'on': [h10._short(list(e)) for e in ea[k:k + 2]],
                       'off': [h10._short(list(e)) for e in eb[k:k + 2]]}
    return True, None


def _ev_in(e, inwin):
    us = (e[1] or {}).get('cam_epoch_us') if isinstance(e[1], dict) else None
    if us is None:
        return True
    # cam_epoch_us = clock.to_epoch_us(cam.t_ns)。ここでの t_ns は記録の µs × 1000 なので、その差だけずらして戻す
    return inwin(int(us) * 1000 - _EPOCH_OFF[0])


_EPOCH_OFF = [0]


def windows_for(log, rows, t0_us):
    """collect.py --day の行(発進ごと)→ [(t_ai_us, t_end_us, 行の要約)]"""
    out = []
    for r in rows:
        if r['log'] != log:
            continue
        t_ai = float(r['t_ai'])
        te = r.get('first_lap_end_s')
        if te is None:
            te = t_ai + float(r.get('ai_seg_s') or 15.0) + 0.5
        out.append((t0_us + int(round(t_ai * 1e6)), t0_us + int(round(float(te) * 1e6)),
                    {k: r.get(k) for k in ('t_ai', 'first_lap_end_s', 'first_lap_end', 'to_r1_s', 'loss_s', 'move_0p2_s',
                                           'move_0p4_s', 'slip', 'v_launch_at_lh', 'kick', 'recovery', 'collisions_first_lap',
                                           'ai_seg_s')}))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default=None)
    a = ap.parse_args(argv)
    _EPOCH_OFF[0] = int(clock.to_epoch_us(0)) * 1000
    shas = {f: sha(ROOT / 'jetformer' / f) for f in ('drive.py', 'assist.py')}
    rows = []
    for day in ('20261023', '20261024'):
        p = OUT / ('collect_launches_%s.json' % day)
        rows += json.loads(p.read_text(encoding='utf-8'))['launches']
    logs = sorted({r['log'] for r in rows}) if not a.logs else a.logs.split(',')
    fpath = Path(drive.FLOOR_REF_PATH)          # 本物の床の基準(読むだけ。走りごとに一時の場所へ写す)
    floor = np.load(fpath) if fpath.exists() else None
    res = {'sha256': shas, 'floor_ref': str(fpath) if floor is not None else None, 'logs': {}, 'started': _time.strftime('%F %T')}
    t_start = _time.time()
    try:
        for log in logs:
            try:
                meta, rec, problems = load_rec(log)
            except Exception as e:      # noqa: BLE001
                res['logs'][log] = {'error': repr(e)[:200]}
                print('skip', log, repr(e)[:120], flush=True)
                continue
            if not all(k in rec for k in ('AI_CMD', 'VEHICLE', 'IMU', 'TOF_FRONT')):
                res['logs'][log] = {'error': 'センサーが足りない %s' % sorted(rec), 'problems': problems}
                print('skip', log, 'missing', sorted(rec), flush=True)
                continue
            lab = json.loads((LAB / (log + '.json')).read_text(encoding='utf-8'))
            t0_us = int(lab['t0_us'])
            dr = meta.get('drive') or {}
            p = dict(drive.DEFAULT_PARAMS)
            p.update({k: v for k, v in (dr.get('params') or {}).items() if k in drive.DEFAULT_PARAMS})
            # 区間の上限(zone_caps)は記録の値のまま(10/23 は ON の本がある)。速さの天井・自動合わせ・launch_no_kick は
            # 10/23・10/24 にはまだ無い(既定 0)
            p.update(speed_ceiling=0, speed_auto=0, launch_no_kick=0)
            p['_ref'] = (float(dr.get('steer_center_us', 1441)), float(dr.get('throttle_neutral_us', 1496)))
            wins = windows_for(log, rows, t0_us)
            ent = {'problems': problems, 'params': {k: v for k, v in p.items() if not k.startswith('_')},
                   'n_frames': int(len(rec['AI_CMD']['timestamps_us'])), 'launches': []}
            Z = rec['AI_CMD']
            t_lo, t_hi = int(Z['timestamps_us'][0]), int(Z['timestamps_us'][-1])
            for lb in (0, 1):
                pp = dict(p, launch_boost=lb)
                ron = run(rec, h10.ON(pp), t_lo, t_hi, floor)
                roff = run(rec, h10.OFF(pp), t_lo, t_hi, floor)
                same_all, d_all = first_diff(ron, roff)
                hold_rows_rec = int(np.sum(Z['hold'] & Z['select']))
                if lb == 0:                # 確かめ: OFF の流し直しの出す値が、記録の出した値と同じ行の数(記録のあとにコードが変わった所は違ってよい)
                    ent['fidelity_off'] = fidelity(roff, Z)
                hold_rows_on = sum(1 for t, r in ron[1].rows if r['hold'] and r['select'])
                k = 'lb%d' % lb
                ev_s = None
                if d_all is not None and d_all.get('what') == 'event':
                    evs = [e for e in (d_all.get('on') or []) + (d_all.get('off') or []) if isinstance(e, list) and len(e) > 1
                           and isinstance(e[1], dict) and e[1].get('cam_epoch_us') is not None]
                    if evs:
                        ev_s = round(((int(evs[0][1]['cam_epoch_us']) * 1000 - _EPOCH_OFF[0]) / 1000 - t0_us) / 1e6, 3)
                ent[k] = {'same_all': same_all, 'first_diff_all': d_all, 'first_diff_event_s': ev_s,
                          'first_diff_s': None if d_all is None or 't_ns' not in d_all else round((d_all['t_ns'] / 1000 - t0_us) / 1e6, 3),
                          'pca_writes': len(ron[2].w), 'rows': len(ron[1].rows),
                          'ai_hold_rows_record': hold_rows_rec, 'ai_hold_rows_replay': hold_rows_on,
                          'cam_timeouts': len(ron[0].timeouts),
                          'ai_cam_timeouts': sum(1 for t in ron[0].timeouts if _sel_at(rec, t)),
                          'guards_on': [e[1].get('what') for e in ron[1].events if e[0] == 'recovery_guard'],
                          'rec_frames_on': sum(1 for t, r in ron[1].rows if int(r['assist']) in (A.ASSIST_RECOVERY, A.ASSIST_GIVEUP)),
                          'rec_frames_off': sum(1 for t, r in roff[1].rows if int(r['assist']) in (A.ASSIST_RECOVERY, A.ASSIST_GIVEUP)),
                          'launch_frames_on': sum(1 for t, r in ron[1].rows if int(r['assist']) == A.ASSIST_LAUNCH)}
                for wi, (wa, wb, info) in enumerate(wins):
                    s_, d_ = first_diff(ron, roff, wa * 1000, wb * 1000)
                    holds_in = [round((t / 1000 - t0_us) / 1e6, 3) for t, r in ron[1].rows
                                if r['hold'] and r['select'] and wa * 1000 <= t <= wb * 1000]
                    to_in = [round((t / 1000 - t0_us) / 1e6, 3) for t in ron[0].timeouts
                             if wa * 1000 <= t <= wb * 1000 and _sel_at(rec, t)]
                    rec_in = sum(1 for t, r in roff[1].rows if wa * 1000 <= t <= wb * 1000
                                 and int(r['assist']) in (A.ASSIST_RECOVERY, A.ASSIST_GIVEUP))
                    if lb == 0:
                        ent['launches'].append(dict(info))
                    ent['launches'][wi]['lb%d' % lb] = {
                        'same': s_, 'diff': d_,
                        'diff_s': None if d_ is None or 't_ns' not in d_ else round((d_['t_ns'] / 1000 - t0_us) / 1e6, 3),
                        'ai_holds_rows': holds_in, 'ai_cam_timeouts': to_in, 'rec_frames_off': rec_in}
                print(log, k, 'same_all', same_all, 'diff_s', ent[k]['first_diff_s'],
                      'launches', [(round(w[2]['t_ai'], 2), ent['launches'][i]['lb%d' % lb]['same']) for i, w in enumerate(wins)],
                      'hold_rows rec/replay', hold_rows_rec, hold_rows_on, '%.0fs' % (_time.time() - t_start), flush=True)
            res['logs'][log] = ent
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
        h10.cleanup()
    res['elapsed_s'] = round(_time.time() - t_start, 1)
    (OUT / 'rec_replay10.json').write_text(json.dumps(res, ensure_ascii=False, default=str, indent=0))
    print('done', len(res['logs']), res['elapsed_s'])


def fidelity(r, Z):
    """OFF の流し直しの行と記録の行を撮像時刻で合わせ、出した値(out)が 0.6 µs 未満・補助の種類が同じ行の数"""
    idx = {int(t): i for i, t in enumerate(Z['timestamps_us'].astype(np.int64) * 1000)}
    n = same = 0
    bad = []
    for t, row in r[1].rows:
        i = idx.get(int(t))
        if i is None:
            continue
        n += 1
        o = np.asarray(row['out'], float)
        ok = bool(np.all(np.abs(o - Z['out'][i].astype(float)) < 0.6)) and int(row['assist']) == int(Z['assist'][i]) \
            and bool(row['hold']) == bool(Z['hold'][i])
        same += ok
        if not ok and len(bad) < 3:
            bad.append({'i': i, 'replay': [round(float(x), 1) for x in o] + [int(row['assist']), bool(row['hold'])],
                        'record': [round(float(x), 1) for x in Z['out'][i]] + [int(Z['assist'][i]), bool(Z['hold'][i])]})
    return {'rows_matched': n, 'rows_same': same, 'first_bad': bad}


def _sel_at(rec, t_ns):
    V = rec['VEHICLE']
    k = np.searchsorted(V['timestamps_us'], t_ns // 1000, side='right') - 1
    return bool(V['select'][k]) if k >= 0 else False


if __name__ == '__main__':
    main()
