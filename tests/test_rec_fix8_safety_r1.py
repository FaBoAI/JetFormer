"""10/28 fix8 係 safety 直す 8-1 回目(反証 8-1 回目の指摘 rs8-1-real-record-backing-missed・rs8-1-backearly-rear-faster)の試験。

jetformer/ のコード(assist.py・drive.py)は変えない(main session の決め 10/28: 11/3 の安全のコードは変えない。直しは試験・台本・
文書に入れる文・11/3 のあとの案だけ)。この試験は、今のコードの 2 つの形を固定する(文書に書く数の元。コードや流し方が変わったら
落ちて、文書の直しを知らせる)。

1. 本物の記録の再生は「1 本(log)の場面(scene)をつないで、1 つの Assist / 1 つの drive._loop で流す」(rs8-1-real-record-backing-missed)。
   fix7 の再生(experiments/20261026_fix/fix7/safety/replay_all.py・safety_r2/replay_all9.py)は場面のディレクトリごとに Assist を
   作り直していた。10/24 20261024-160847 では、前の物を押して止まった車のリカバリーの 'brake' が場面 0002 の最後のフレーム
   (39.97 秒)で始まり、場面 0003 は 40.003 秒から。作り直した Assist は armed でなくリカバリーが始まらず、fix7 は「235 本で出す値が
   1 フレームも違わない」「'brake'・'wait'・'gap' で後ろへ動いたのは 10/23 の 1 つだけ」と書いた(235 は記録の本数でなく場面の数)。
   つないで流すと、見張り ON の fix7 の前(tests/fixtures/fix7_safety_before)と今は、160847 の 40.07〜40.37 秒の 10 フレームだけで
   違う(fix7 の前は back_early で 'gap' を飛ばして後退側 1796 µs を出し続け、今は back_no_esc で 'brake' → 'gap' = 見張り OFF と
   同じ値)。そこで車は本当に後ろへ動いていた(hall_count が一方向に進む = 跳ねではない。eRPM はどの段でもほぼ同じ速さ = ESC の
   後退ではなく受け身の動き)。今の ON はそこで OFF と同じ値を書くので、OFF より悪くはない(approvals 14 の「ESC の見立て」の形)。
   この違いは 7-1 回目(s4-a)で入った: 7-2 回目(s4-c)の前(tests/fixtures/fix7_safety_r2_before)と今は、つないでも全部同じ
   (= fix7/fix_safety_r2.md の「1 フレームも違わない」は成り立つ。直すのは「235 本」が場面の数なことだけ)。
   - 流し方は 2 つ(どちらも開いたループ = 車は記録のとおりに動いたまま):
     (A) Assist だけ(fix7 の再生と同じ形: 見張り ON・中立止めの行で launch_hold と reset・書いた値を EscMayReverse に入れて
         esc_rev を渡す)。10/10・10/23・10/24 の全部の記録を、場面ごとと、つないだ形の両方で。
     (D) 本物の drive._loop(両方 ON = rec_back_guard 1・rec_hold_keep 1 と、両方 OFF)。記録の時刻でカメラ・センサーを入れ、
         モデルは記録の model_pred(fix8 decided_safety の rec_replay10.py の RecFeeder と同じ形)。ふだんは 160847 の 1 本、
         JF_FIX8S1_ALL=1 のときは全部の記録(4 つの形で約 14 分)。
   - 場面ごとに作り直す前からの試験(tests/test_recovery_back_guard.py の _replay など)はそのまま残す。
2. 10/23 の形(止まる前にブレーキ側 → 止まって ESC が後退を待つ → リカバリー)で、見張り ON は back_early で 'gap' の中立を
   飛ばして後退側を出し続ける(fix2 の見張りからの形。fix7 の前と同じ)。近い後ろの物には ON のほうが速く当たり、遠い物には
   ON のほうが遅く当たるか当たらない(rs8-1-backearly-rear-faster。どちらの向きかは ESC の模型と距離しだい = 文書は両方の向きを
   書く。「ON のほうが長く下がる」とは書かない)。tests/test_rec_fix7_safety.py の閉じたループ(場面 brake)で固定する。

data/raw は読むだけ(10/10・10/23・10/24 の記録だけを見る。11/3 からの記録は入れない)。記録が無い環境では 1 を skip する。
実機・GPU・torch・jetformer-web は使わない。台本と出力: experiments/20261026_fix/fix8/fix_safety_r1/。

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider \\
        -o tmp_path_retention_policy=failed --basetemp=<消せる場所> tests/test_rec_fix8_safety_r1.py
"""
import collections
import json
import os
import tempfile
import time as _time
import types
from pathlib import Path

import numpy as np
import pytest

from jetformer import assist as A
from jetformer import clock, config, drive
from jetformer.sensors.base import Sample

import test_rec_fix7_safety as F7
import test_rec_fix7_safety_r2 as F7R2
from test_drive_caps import FakeRecorder, fake_normalize

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
DAYS = ('20261010', '20261023', '20261024')      # 見る記録の日(11/3 からの記録は入れない = この試験の数は変わらない)
TABLES = ('AI_CMD', 'VEHICLE', 'IMU')
LOG = '20261024-160847'                          # 場面のさかい目で始まったリカバリー(10/24。前の物を押して止まったあと)
LOG_1023 = '20261023-205458'                     # 10/23 No.6(ESC が後退を待っていた形。back_early の本物)
# 160847 で違うフレーム(撮像の時刻。記録の最初の場面の始め scene.json の start_us から)
DIFF_S = (40.070, 40.103, 40.136, 40.170, 40.203, 40.236, 40.270, 40.303, 40.336, 40.369)
ALL_DRIVE = os.environ.get('JF_FIX8S1_ALL') == '1'


# ================================================================ 記録を読む(読むだけ)
def raw_logs():
    """見る記録(DAYS の日・log.json あり・AI_CMD・VEHICLE・IMU のそろった場面が 1 つ以上)"""
    if not RAW.is_dir():
        return []
    return sorted(p.name for p in RAW.iterdir()
                  if p.is_dir() and p.name[:8] in DAYS and (p / 'log.json').exists() and scene_dirs(p.name))


def scene_dirs(log):
    """1 本の記録の場面のディレクトリ(名前の順 = 時刻の順。AI_CMD・VEHICLE・IMU のファイルがある場面。中身が空でも入れる)"""
    root = RAW / log
    return sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith(log + '-')
                  and all((d / (n + '.npz')).exists() for n in TABLES))


def _read(p):
    if p.stat().st_size == 0:
        raise ValueError('空のファイル %s' % p.name)
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files if z[k].ndim > 0}


def read_scene(d):
    out = {n: _read(d / (n + '.npz')) for n in TABLES}
    if 'timestamps_us' not in out['AI_CMD'] or not len(out['AI_CMD']['timestamps_us']):
        raise ValueError('AI_CMD が空')
    return out


def log_t0_us(log):
    """記録の時刻の起点: 最初の場面の scene.json の start_us(無い・読めなければ None)"""
    for d in sorted(x for x in (RAW / log).iterdir() if x.is_dir() and x.name.startswith(log + '-')):
        try:
            return int(json.loads((d / 'scene.json').read_text(encoding='utf-8'))['start_us'])
        except Exception:       # noqa: BLE001  空・壊れた scene.json
            return None
    return None


def load_unit(log, scenes):
    """場面 scenes(ディレクトリ)をつないだ表(表ごとに時刻の順・全部の場面にある欄だけ)。読めない場面は飛ばす。
    戻り値 (表, log.json, 飛ばした場面 [(名前, 理由)])。読めた場面が無ければ表は None"""
    parts, skipped = [], []
    for d in scenes:
        try:
            parts.append(read_scene(d))
        except Exception as e:      # noqa: BLE001  空の npz(電源断の記録)など
            skipped.append((d.name, repr(e)[:80]))
    try:
        meta = json.loads((RAW / log / 'log.json').read_text(encoding='utf-8'))
    except Exception as e:          # noqa: BLE001  0 バイトの log.json(電源断の記録 10/23 204825)
        return None, None, skipped + [(log + '/log.json', repr(e)[:80])]
    if not parts:
        return None, meta, skipped
    cat = {}
    for n in TABLES:
        keys = set.intersection(*[set(p[n]) for p in parts])
        c = {k: np.concatenate([p[n][k] for p in parts]) for k in keys}
        o = np.argsort(c['timestamps_us'], kind='stable')
        cat[n] = {k: v[o] for k, v in c.items()}
    return cat, meta, skipped


def units(log, how):
    """how = 'scene'(場面ごと。fix7 の再生の形)/ 'concat'(1 本の場面をつなぐ)→ [[場面のディレクトリ], ...]"""
    ss = scene_dirs(log)
    return [[d] for d in ss] if how == 'scene' else [ss]


# ================================================================ (A) Assist だけの開いたループ(fix7 の再生の形)
FORMS = ('now_on', 'now_off', 'f7b_on', 'f7b_off')


def assist_replay(cat, meta, mods=None, reset_at=None, spec=None):
    """表 cat を 4 つの形(今 / fix7 の前 × 見張り ON / OFF)の Assist に同じ入力で流す(開いたループ)。
    fix7 の replay_all9 と同じ: 車両の値は撮像の時刻より前の最新・IMU は処理の時刻(撮像 + latency)の前 0.15 秒の水平 G の最大・
    中立止めの行(hold)は launch_hold と reset(= 残す OFF)・見張り ON の今の形は書いた値を EscMayReverse に入れて esc_rev を渡す。
    mods = {'now': assist のモジュール, 'f7b': …}(わざと壊す試験が差し替える)。reset_at = この撮像時刻 [µs] の集合のフレームの前で
    4 つの Assist を作り直す(わざと壊す試験: 場面ごとの作り直しをつないだ表の中でまねる)。
    戻り値 dict: t_us・erpm・hall(処理したフレームごと)、out[形] = (n, 3)(舵・スロットル・補助の種類)、phase0・phase(今の ON の
    段の前・後)、ev[形] = [(フレームの番号, 出来事)]、rec_starts(今の ON)。
    spec = {形: (mods の名前, 見張り, esc_rev を渡すか)}(既定は FORMS の 4 つ。台本が案の写しを並べるときに使う。'now_on' は要る)"""
    mods = dict({'now': A, 'f7b': F7.load_before()[0]}, **(mods or {}))
    z, v, im = cat['AI_CMD'], cat['VEHICLE'], cat['IMU']
    dr = meta.get('drive') or {}
    p = dr.get('params') or {}
    ref = (float(dr.get('steer_center_us', 1441)), float(dr.get('throttle_neutral_us', 1496)))
    t = z['timestamps_us'].astype(np.int64)
    sel = z['select'].astype(bool)
    hold = z['hold'].astype(bool)
    dry = z['dry_run'].astype(bool) if 'dry_run' in z else np.zeros(len(t), bool)
    mo = z['model_out'].astype(float) if 'model_out' in z else z['out'].astype(float)
    lat = z['latency_ms'].astype(float) if 'latency_ms' in z else np.full(len(t), np.nan)
    vt = v['timestamps_us'].astype(np.int64)
    ve = v['erpm'].astype(float)
    vh = v['hall_count'].astype(np.int64) if 'hall_count' in v else np.zeros(len(vt), np.int64)
    it = im['timestamps_us'].astype(np.int64)
    acc = im['accel'].astype(float)
    hg = np.hypot(acc[:, 0], acc[:, 1]) / 9.80665
    kw = dict(esc_on=p.get('esc_reverse', 1) == 1, rec_on=p.get('recovery', 1) == 1, rev_us=float(p.get('rev_us', 300)),
              rev_s=float(p.get('rev_s', 0.4)), kick_on=int(p.get('start_kick', 0) or 0) == 1)
    spec = spec or {'now_on': ('now', True, True), 'now_off': ('now', False, False), 'f7b_on': ('f7b', True, False),
                    'f7b_off': ('f7b', False, False)}
    forms = tuple(spec)

    def fresh():
        # 書いた値の EscMayReverse は esc_rev を渡す形ごとに 1 つ(その形が書いた値を入れる。drive と同じ)
        return ({f: mods[m].Assist(ref) for f, (m, g, x) in spec.items()},
                {f: A.EscMayReverse(ref[1]) for f, (m, g, x) in spec.items() if x})
    hp, gates = fresh()
    q = None
    reset_at = set(reset_at or ())
    res = {'t_us': [], 'erpm': [], 'hall': [], 'out': {f: [] for f in forms}, 'phase0': [], 'phase': [],
           'ev': {f: [] for f in forms}, 'rec_starts': []}
    for i in range(len(t)):
        if int(t[i]) in reset_at:
            hp, gates = fresh()
            q = None
        now_us = t[i] + (lat[i] * 1000 if np.isfinite(lat[i]) else 0)
        ns = int(now_us * 1000)
        if hold[i]:
            for a in hp.values():
                a.launch_hold()
                a.reset()
            for gate in gates.values():
                if not sel[i] or dry[i]:
                    gate.unknown(ns)
                else:
                    gate.write(ns, ref[1])
            continue
        jv = np.searchsorted(vt, t[i], side='right') - 1
        e = ve[jv] if jv >= 0 else 0.0
        ji = np.searchsorted(it, now_us, side='right')
        idx = np.arange(max(0, ji - 10), ji)
        idx = idx[it[idx] > t[i] - 150_000]
        ig = float(hg[idx].max()) if len(idx) else 0.0
        er = {f: (None if q is None else gate.rev_s(q, ns) > 0.0) for f, gate in gates.items()}
        q = ns
        s_ = bool(sel[i]) and not dry[i]
        k_now = len(res['t_us'])
        ph0 = hp['now_on'].rec.phase
        outs = {}
        for f, (m, g, x) in spec.items():
            extra = {'esc_rev': er[f]} if x else {}
            o, k = hp[f].step((mo[i, 0], mo[i, 1]), e, s_, now_us / 1e6, impact_g=ig, rec_guard=g, **kw, **extra)
            ev, hp[f].rec.guard_event = hp[f].rec.guard_event, None
            if ev:
                res['ev'][f].append((k_now, ev[0]))
            outs[f] = (float(o[0]), float(o[1]), int(k))
        for f, gate in gates.items():
            if s_:
                gate.write(ns + 1, outs[f][1])
            else:
                gate.unknown(ns + 1)
        if ph0 is None and hp['now_on'].rec.phase == 'brake':
            res['rec_starts'].append(k_now)
        res['t_us'].append(int(t[i]))
        res['erpm'].append(float(e))
        res['hall'].append(int(vh[jv]) if jv >= 0 else 0)
        res['phase0'].append(ph0)
        res['phase'].append(hp['now_on'].rec.phase)
        for f in forms:
            res['out'][f].append(outs[f])
    res['t_us'] = np.asarray(res['t_us'], np.int64)
    for f in forms:
        res['out'][f] = np.asarray(res['out'][f], float).reshape(-1, 3)
    return res


def diff_idx(r, a, b):
    """形 a と b の出す値(舵・スロットル・補助の種類)が違うフレームの番号"""
    return [int(i) for i in np.nonzero(np.any(r['out'][a] != r['out'][b], axis=1))[0]]


def backing_bwg(r):
    """今の ON で、段 'brake'・'wait'・'gap' の中(このフレームの前の段)なのに後ろへ動いていた(eRPM > BACKING_ERPM)フレームの番号"""
    return [i for i, (ph, e) in enumerate(zip(r['phase0'], r['erpm'])) if ph in ('brake', 'wait', 'gap') and e > A.BACKING_ERPM]


def replay_all_assist(how, logs=None, mods=None):
    """全部の記録を (A) で流してまとめる。how = 'scene' / 'concat'"""
    tot = {'how': how, 'logs': 0, 'units': 0, 'frames': 0, 'rec_starts': 0, 'skipped': [], 'diff_on': [], 'diff_off': [],
           'backing_bwg': [], 'on_vs_off_at_diff_on': [], 'events_now_on': collections.Counter(),
           'events_f7b_on': collections.Counter()}
    for log in (logs if logs is not None else raw_logs()):
        t0 = log_t0_us(log)
        tot['logs'] += 1
        for u in units(log, how):
            cat, meta, sk = load_unit(log, u)
            tot['skipped'] += [s[0] for s in sk]
            if cat is None:
                continue
            r = assist_replay(cat, meta, mods)
            base = t0 if t0 is not None else int(r['t_us'][0])
            name = u[0].name if how == 'scene' else log
            tot['units'] += 1
            tot['frames'] += len(r['t_us'])
            tot['rec_starts'] += len(r['rec_starts'])
            tot['events_now_on'].update(e for _, e in r['ev']['now_on'])
            tot['events_f7b_on'].update(e for _, e in r['ev']['f7b_on'])

            def rel(i, r=r, base=base):
                return round((int(r['t_us'][i]) - base) / 1e6, 3)
            d_on = diff_idx(r, 'now_on', 'f7b_on')
            tot['diff_on'] += [{'unit': name, 's': rel(i), 'erpm': r['erpm'][i], 'hall': r['hall'][i],
                                'now_on': r['out']['now_on'][i].tolist(), 'f7b_on': r['out']['f7b_on'][i].tolist(),
                                'now_off': r['out']['now_off'][i].tolist(),
                                'ev_now': [e for k, e in r['ev']['now_on'] if k == i],
                                'ev_f7b': [e for k, e in r['ev']['f7b_on'] if k == i]} for i in d_on]
            tot['diff_off'] += [{'unit': name, 's': rel(i)} for i in diff_idx(r, 'now_off', 'f7b_off')]
            tot['on_vs_off_at_diff_on'] += [bool(np.all(r['out']['now_on'][i] == r['out']['now_off'][i])) for i in d_on]
            tot['backing_bwg'] += [{'unit': name, 's': rel(i), 'phase': r['phase0'][i], 'erpm': r['erpm'][i]}
                                   for i in backing_bwg(r)]
    tot['events_now_on'] = dict(tot['events_now_on'])
    tot['events_f7b_on'] = dict(tot['events_f7b_on'])
    return tot


# ================================================================ (D) 本物の drive._loop の開いたループ
class RecFeeder(object):
    """cam_evt の代わり(wait / clear / set)。記録の時刻で時計・センサー・カメラを進める(fix8 decided_safety の rec_replay10.py と
    同じ形: カメラは撮像の時刻の 5 ms あとに届く・処理の終わりは撮像 + latency_ms・次が 0.1 秒来なければ本物と同じ中立止め・
    モデルは記録の model_pred・フレームごとのゲイン・トリム・先読みは記録の値)"""

    def __init__(self, d, rec, bufs, box):
        self.d, self.box = d, box
        self.tof_buf, self.imu_buf, self.veh_buf = bufs
        self.cam_slot = [None]
        Z = rec['AI_CMD']
        self.ft = Z['timestamps_us'].astype(np.int64) * 1000
        self.fp = Z['model_pred'].astype(np.float64) if 'model_pred' in Z else Z['pred'].astype(np.float64)
        lat = Z['latency_ms'].astype(np.float64)
        self.fg = Z['gains'].astype(np.float64) if 'gains' in Z else None
        self.ftrim = Z['trim'].astype(np.float64) if 'trim' in Z else None
        self.flook = Z['lookahead'].astype(np.int64) if 'lookahead' in Z else None
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
        self.now = int(self.ft[0]) - 200_000_000
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
        self.box['m'] = self.now / 1e9 - self.ft[0] / 1e9 + 1000.0
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
        if self.cur >= 0:
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


def load_rec_drive(log):
    """drive._loop の流し直しの記録: 全部の場面をつなぐ(AI_CMD・VEHICLE・IMU・TOF_FRONT。空・壊れた npz は飛ばす)"""
    root = RAW / log
    parts = collections.defaultdict(list)
    problems = []
    try:
        meta = json.loads((root / 'log.json').read_text(encoding='utf-8'))
    except Exception as e:          # noqa: BLE001  0 バイトの log.json(電源断の記録 10/23 204825)
        return {}, {}, ['log.json %s' % repr(e)[:60]]
    for s in sorted(d.name for d in root.iterdir() if d.is_dir() and d.name.startswith(log + '-')):
        for n in TABLES + ('TOF_FRONT',):
            p = root / s / (n + '.npz')
            try:
                d = _read(p)
            except Exception as e:      # noqa: BLE001
                problems.append('%s/%s %s' % (s, n, repr(e)[:60]))
                continue
            if 'timestamps_us' in d and len(d['timestamps_us']):
                parts[n].append(d)
    out = {}
    for n, ds in parts.items():
        keys = set.intersection(*[set(d) for d in ds])
        c = {k: np.concatenate([d[k] for d in ds]) for k in keys}
        o = np.argsort(c['timestamps_us'], kind='stable')
        out[n] = {k: v[o] for k, v in c.items()}
    return meta, out, problems


def drive_params(meta, guard, keep, mod=drive):
    """記録の log.json の drive.params(DEFAULT_PARAMS にある名前だけ)+ 天井・自動合わせ・補助なし・発進の前進 0(その日はまだ無い)
    + 見張り・残す"""
    dr = meta.get('drive') or {}
    p = dict(mod.DEFAULT_PARAMS)
    p.update({k: v for k, v in (dr.get('params') or {}).items() if k in mod.DEFAULT_PARAMS})
    p.update(speed_ceiling=0, speed_auto=0, launch_no_kick=0, launch_boost=0, rec_back_guard=guard, rec_hold_keep=keep)
    p['_ref'] = (float(dr.get('steer_center_us', 1441)), float(dr.get('throttle_neutral_us', 1496)))
    return p


def drive_replay(mod, rec, params, tmp_dir, floor):
    """本物の drive._loop(mod = jetformer.drive か写し)に記録を開いたループで流す → (feeder, recorder, pca)"""
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(tmp_dir)) as td:
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
        pca = TPCA(box)
        ref = (float(params['_ref'][0]), float(params['_ref'][1]))
        out = mod._Output(pca, config.PCA9685, ref, lambda: d.params)
        bufs = (collections.deque(maxlen=10), collections.deque(maxlen=60), collections.deque(maxlen=50))
        fd = RecFeeder(d, rec, bufs, box)
        d._loop(d.st, RecRunner(fd), fake_normalize, pca, out, ref, 6, 0.0, fd.cam_slot, fd, *bufs, False)
        d._stop.clear()
    return fd, recd, pca


def floor_ref():
    """本物の床の基準(data/tof_floor_ref.npy。読むだけ。走りごとに一時の場所へ写す)"""
    fpath = Path(drive.FLOOR_REF_PATH)
    return np.load(fpath) if fpath.exists() else None


def drive_rows(r):
    """記録の行 → {撮像の時刻 ns: (舵, スロットル, 補助の種類, 中立止め)}"""
    return {int(t): (float(x['out'][0]), float(x['out'][1]), int(x['assist']), bool(x['hold'])) for t, x in r[1].rows}


def drive_compare(log, tmp_dir, floor=None, forms=(('now', 1, 1), ('now', 0, 0), ('f7b', 1, 1), ('f7b', 0, 0))):
    """1 本の記録を drive._loop で 4 つの形に流し、今の ON と fix7 の前の ON の違うフレーム(撮像の時刻 [s]。起点は log_t0_us)、
    そのフレームで今の ON と今の OFF が同じか、今の OFF と fix7 の前の OFF が同じか(PCA9685 に書いた値・行)を返す"""
    meta, rec, problems = load_rec_drive(log)
    if not all(k in rec for k in TABLES + ('TOF_FRONT',)) or not len(rec['AI_CMD']['timestamps_us']):
        return {'log': log, 'skipped': 'センサーが足りない %s' % sorted(rec), 'problems': problems}
    mods = {'now': drive, 'f7b': F7.load_before()[1]}
    rr = {}
    for ver, g, k in forms:
        rr[(ver, g)] = drive_replay(mods[ver], rec, drive_params(meta, g, k, mods[ver]), tmp_dir, floor)
    t0 = log_t0_us(log)
    base = t0 if t0 is not None else int(rec['AI_CMD']['timestamps_us'][0])
    rows = {key: drive_rows(r) for key, r in rr.items()}

    def rel(tn):
        return round((tn / 1000 - base) / 1e6, 3)
    out = {'log': log, 'problems': problems, 'rows': len(rows[('now', 1)]), 'pca_writes': len(rr[('now', 1)][2].w)}
    on, onb = rows[('now', 1)], rows[('f7b', 1)]
    keys = sorted(set(on) | set(onb))
    d_on = [tn for tn in keys if on.get(tn) != onb.get(tn)]
    out['diff_on_s'] = [rel(tn) for tn in d_on]
    out['diff_on'] = [{'s': rel(tn), 'now_on': on.get(tn), 'f7b_on': onb.get(tn), 'now_off': rows[('now', 0)].get(tn)}
                      for tn in d_on]
    out['on_eq_off_at_diff'] = [on.get(tn) == rows[('now', 0)].get(tn) for tn in d_on]
    out['pca_on_same'] = rr[('now', 1)][2].w == rr[('f7b', 1)][2].w
    wa, wb = rr[('now', 1)][2].w, rr[('f7b', 1)][2].w
    out['pca_on_diff_writes'] = (sum(1 for x, y in zip(wa, wb) if x != y) + abs(len(wa) - len(wb)))
    if ('f7b', 0) in rr:
        out['off_same_rows'] = rows[('now', 0)] == rows[('f7b', 0)]
        out['off_same_pca'] = rr[('now', 0)][2].w == rr[('f7b', 0)][2].w
        out['off_same_events'] = rr[('now', 0)][1].events == rr[('f7b', 0)][1].events
    out['guards_now_on'] = [(e[1].get('what'), e[1].get('phase')) for e in rr[('now', 1)][1].events if e[0] == 'recovery_guard']
    out['guards_f7b_on'] = [(e[1].get('what'), e[1].get('phase')) for e in rr[('f7b', 1)][1].events if e[0] == 'recovery_guard']
    return out


# ================================================================ 固定する形(文書の数の元)
def check_concat(tot):
    """(A) のつないだ流し方の結果が、固定した形かを照らす(わざと壊す試験もこれを使う)。違えば AssertionError"""
    d = tot['diff_on']
    assert [x['unit'] for x in d] == [LOG] * len(DIFF_S), [(x['unit'], x['s']) for x in d]
    assert tuple(round(x['s'], 3) for x in d) == DIFF_S, [x['s'] for x in d]
    assert all(tot['on_vs_off_at_diff_on'])                                  # そこで今の ON = 今の OFF
    assert d[0]['ev_f7b'] == ['back_early'] and d[0]['ev_now'] == ['back_no_esc']
    # fix7 の前は 'reverse'(舵 1291・後退側 1796)を出し続けた。今は 'brake'(舵 1441・1796)のあと 40.236 秒から 'gap'(中立の帯)
    assert all(x['f7b_on'][:2] == [1291.0, 1796.0] for x in d)
    assert [x['now_on'][:2] for x in d] == [[1441.0, 1796.0]] * 5 + [[1291.0, 1506.0]] * 5
    # どのフレームも後ろへ動いていた(eRPM > 0。40.103 秒の 1 つだけ 211 で BACKING_ERPM より遅い)・hall_count は戻らない
    assert all(x['erpm'] > 0 for x in d) and sum(x['erpm'] > A.BACKING_ERPM for x in d) == 9
    assert all(b['hall'] >= a['hall'] for a, b in zip(d, d[1:])) and d[-1]['hall'] - d[0]['hall'] >= 15
    assert tot['diff_off'] == []                                             # 見張り OFF は fix7 の前とビット単位で同じ
    b = collections.Counter(x['unit'] for x in tot['backing_bwg'])
    assert b == {LOG_1023: 1, LOG: 10}, b      # 'brake'・'wait'・'gap' で後ろへ: 10/23 の 1 と 160847 の 10(40.07〜40.403 秒)


def check_scene(tot):
    """(A) の場面ごとの流し方(fix7 の形)の結果: 違い 0・'brake'・'wait'・'gap' で後ろへ動いたのは 10/23 の 1 フレームだけ
    (= fix7 の文書の数。見落としの元)"""
    assert tot['diff_on'] == [] and tot['diff_off'] == []
    assert [x['unit'] for x in tot['backing_bwg']] == [LOG_1023 + '-0002']


needs_raw = pytest.mark.skipif(not (RAW / LOG / 'log.json').exists() or not (RAW / LOG_1023 / 'log.json').exists(),
                               reason='本物の記録(data/raw の 10/23・10/24)がありません')

_CACHE = {}


def _tot(how):
    if how not in _CACHE:
        _CACHE[how] = replay_all_assist(how)
    return _CACHE[how]


# ---------------------------------------------------------------- 1. (A) 全部の記録
@needs_raw
def test_assist_concat_all_records_differs_only_at_160847():
    """つないだ流し方: 今の見張り ON と fix7 の前の見張り ON が違うのは 160847 の 40.07〜40.37 秒の 10 フレームだけ。そこで今の ON は
    今の OFF と同じ値。見張り OFF は全部同じ。'brake'・'wait'・'gap' で後ろへ動いたフレームは 11(10/23 の 1・160847 の 10)"""
    tot = _tot('concat')
    check_concat(tot)
    assert tot['skipped'] == ['20261010-163247-0021', '20261010-163247-0022', '20261023-204825-0001',
                              '20261023-204825/log.json']                # 空の npz・0 バイトの log.json(電源断の記録)
    assert tot['logs'] == 60 and tot['units'] == 59                       # 59 本(場面ごとの流し方では 235 の場面)
    # 場面ごとと同じフレーム・同じリカバリーの始めの数(160847 の始めは場面 0002 の最後のフレーム 39.97 秒で、場面ごとでも
    # 数えられている。違うのは、その続きの場面 0003 を作り直した Assist で流すか、同じ Assist で流すかだけ)
    assert tot['frames'] == 122767 and tot['rec_starts'] == 16
    assert tot['events_now_on'].get('back_no_esc') == 1 and tot['events_f7b_on'].get('back_early') == 2


@needs_raw
def test_assist_per_scene_reproduces_fix7_count():
    """場面ごとに作り直す流し方(fix7 の replay_all・replay_all9 の形)では、同じフレームを流しても違いが 0 = fix7 の「235 本で
    1 フレームも違わない」は、この流し方の数(235 は場面の数。つなぐと 59 本)"""
    tot = _tot('scene')
    check_scene(tot)
    assert tot['units'] == 235 and tot['frames'] == 122767 and tot['rec_starts'] == 16
    # 238 の場面(AI_CMD・VEHICLE・IMU のファイルがある)のうち空の 3 つを飛ばした 235(fix7 の「238 本のうち読めた 235 本」と同じ数)
    assert sum(len(scene_dirs(x)) for x in raw_logs()) == 238
    assert [x for x in tot['skipped'] if '/' not in x] == ['20261010-163247-0021', '20261010-163247-0022', '20261023-204825-0001']
    assert _tot('concat')['frames'] == tot['frames']                      # 同じフレームを流した(違いは作り直しだけ)


@needs_raw
def test_assist_concat_now_equals_fix7_r1():
    """7-2 回目(s4-c)の前 = 7-1 回目のあと(tests/fixtures/fix7_safety_r2_before)と今の見張り ON は、つないで流しても全部の
    フレームで同じ(fix7/fix_safety_r2.md の「直す前の写しと今に流した: 1 フレームも違わない」は、つないでも成り立つ。違うのは
    「235 本」が場面の数なことだけ)。160847 の違いは 7-1 回目(s4-a)で入った"""
    r1 = F7R2.load_r1()[0]
    spec = {'now_on': ('now', True, True), 'r1_on': ('r1', True, True), 'now_off': ('now', False, False),
            'r1_off': ('r1', False, False)}
    frames = 0
    for log in raw_logs():
        cat, meta, _ = load_unit(log, scene_dirs(log))
        if cat is None:
            continue
        r = assist_replay(cat, meta, mods={'r1': r1}, spec=spec)
        frames += len(r['t_us'])
        assert diff_idx(r, 'now_on', 'r1_on') == [] and diff_idx(r, 'now_off', 'r1_off') == [], log
    assert frames == 122767


@needs_raw
def test_160847_backing_is_real_and_passive():
    """160847 の後ろへの動きは本当(hall_count が一方向に進む。跳ねは ±1〜2 の行き来)で、受け身(ESC の後退ではない):
    'brake' の前は前の物を押して止まっていた(モデルは前進・eRPM 0)。AI 側の 40.03〜40.45 秒('brake'・'gap'。記録の走りは
    見張り OFF)に +20 カウントより多く進み、人が CH3 を人側にした 40.45 秒のあとも、人のスロットルが中立の帯のまま
    (|入力 − 1496| < 20 µs)後ろへ動き続けた(+10 カウントより多く)"""
    log = LOG
    cat, meta, sk = load_unit(log, scene_dirs(log))
    t0 = log_t0_us(log)
    v = cat['VEHICLE']
    s = (v['timestamps_us'].astype(np.int64) - t0) / 1e6
    hall = v['hall_count'].astype(np.int64)
    sel = v['select'].astype(bool)
    pre = (s >= 39.44) & (s <= 40.03)
    assert np.all(v['erpm'][pre] == 0) and np.all(sel[pre])                # その前は止まっていた(AI 側)
    ai = (s >= 40.03) & (s <= 40.45)
    h = hall[ai]
    assert np.all(sel[ai]) and h[-1] - h[0] >= 20 and np.all(np.diff(h) >= 0)  # AI 側の間に一方向に進む(跳ねではない)
    hu = (s >= 40.46) & (s <= 41.0)
    assert not np.any(sel[hu]) and np.all(np.abs(v['pwm'][hu][:, 1].astype(float) - 1496.0) < 20.0)   # 人側・スロットル中立
    assert hall[hu][-1] - hall[hu][0] >= 10 and np.all(np.diff(hall[hu]) >= -1)    # 中立のまま後ろへ動き続けた = 受け身
    z = cat['AI_CMD']
    zs = (z['timestamps_us'].astype(np.int64) - t0) / 1e6
    before = (zs >= 39.6) & (zs < 39.97)
    assert np.all(z['model_out'][before][:, 1] < 1496.0 - 60)            # モデルは前進(前の物を押していた)
    assert int(z['assist'][np.argmin(np.abs(zs - 39.97))]) == A.ASSIST_RECOVERY   # 39.97 秒にリカバリーの 'brake'(場面 0002 の最後)
    sc = scene_dirs(log)
    assert round((int(read_scene(sc[1])['AI_CMD']['timestamps_us'][-1]) - t0) / 1e6, 2) == 39.97
    assert round((int(read_scene(sc[2])['AI_CMD']['timestamps_us'][0]) - t0) / 1e6, 3) == 40.003


# ---------------------------------------------------------------- 2. (D) 本物の drive._loop
@needs_raw
def test_drive_loop_concat_160847(tmp_path):
    """本物の drive._loop(両方 ON / 両方 OFF)で 160847 をつないで流す: 今の ON と fix7 の前の ON の行が違うのは 40.07〜40.37 秒の
    10 フレームだけで、そこで今の ON = 今の OFF。PCA9685 に書く値も 10 個だけ違う。見張り OFF は fix7 の前と同じ"""
    r = drive_compare(LOG, tmp_path, floor_ref())
    assert tuple(r['diff_on_s']) == DIFF_S, r['diff_on_s']
    assert all(r['on_eq_off_at_diff'])
    assert r['pca_on_diff_writes'] == 10 and not r['pca_on_same']
    assert r['off_same_rows'] and r['off_same_pca'] and r['off_same_events']
    assert ('back_no_esc', 'brake') in r['guards_now_on'] and ('back_early', 'brake') in r['guards_f7b_on']
    # fix7 の前: 'reverse'(舵 1291・後退側 1796)を 10 フレーム。今: 'brake'(舵 1441・1796)5 フレームのあと 'gap'(1291・1506)
    assert [tuple(x['f7b_on'][:2]) for x in r['diff_on']] == [(1291.0, 1796.0)] * 10
    assert [tuple(x['now_on'][:2]) for x in r['diff_on']] == [(1441.0, 1796.0)] * 5 + [(1291.0, 1506.0)] * 5


@pytest.mark.skipif(not ALL_DRIVE, reason='全部の記録の drive._loop の流し直しは JF_FIX8S1_ALL=1 のときだけ(約 14 分)')
@needs_raw
def test_drive_loop_concat_all_records(tmp_path):
    """全部の記録を drive._loop でつないで流す: 今の ON と fix7 の前の ON が違うのは 160847 の 10 フレームだけ。OFF は全部同じ"""
    floor = floor_ref()
    diff = {}
    for log in raw_logs():
        r = drive_compare(log, tmp_path, floor)
        if 'skipped' in r:
            assert log == '20261023-204825', r                           # 電源断で全部が空の記録だけ
            continue
        assert r['off_same_rows'] and r['off_same_pca'] and r['off_same_events'], log
        if r['diff_on_s']:
            diff[log] = r
    assert list(diff) == [LOG]
    assert tuple(diff[LOG]['diff_on_s']) == DIFF_S and all(diff[LOG]['on_eq_off_at_diff'])


# ---------------------------------------------------------------- 3. わざと壊すと落ちる
@needs_raw
def test_broken_per_scene_misses_160847():
    """場面ごとに作り直した結果を check_concat に入れると落ちる(見落としを試験が見つける)"""
    with pytest.raises(AssertionError):
        check_concat(_tot('scene'))
    with pytest.raises(AssertionError):
        check_scene(_tot('concat'))


@needs_raw
def test_broken_reset_at_scene_boundary_in_concat_misses_160847():
    """つないだ表の中でも、場面のさかい目(0003 の最初のフレーム)で Assist を作り直すと 160847 の違いが消える(= 見落としの仕組み)"""
    log = LOG
    ss = scene_dirs(log)
    cat, meta, _ = load_unit(log, ss)
    first3 = int(read_scene(ss[2])['AI_CMD']['timestamps_us'][0])
    ok = assist_replay(cat, meta)
    bad = assist_replay(cat, meta, reset_at={first3})
    assert len(diff_idx(ok, 'now_on', 'f7b_on')) == 10
    assert diff_idx(bad, 'now_on', 'f7b_on') == []


@needs_raw
def test_broken_now_reverted_to_f7b_fails_on_eq_off():
    """「今」を fix7 の前の写しに戻すと(s4-a が無い)、160847 で今の ON が OFF と違い、check_concat が落ちる"""
    amod = F7.load_before()[0]
    tot = replay_all_assist('concat', logs=[LOG, LOG_1023], mods={'now': _NowLike(amod)})
    with pytest.raises(AssertionError):
        check_concat(tot)
    assert tot['diff_on'] == []                                          # 今 = fix7 の前なので違いは無く、10 フレームを見つけない
    # その 10 フレームで「今」の ON が今の OFF と違う(= 固定した「そこで今の ON = OFF」が、戻した版では成り立たない)
    cat, meta, _ = load_unit(LOG, scene_dirs(LOG))
    r = assist_replay(cat, meta, mods={'now': _NowLike(amod)})
    t0 = log_t0_us(LOG)
    idx = [i for i, t in enumerate(r['t_us']) if round((int(t) - t0) / 1e6, 3) in DIFF_S]
    assert len(idx) == 10 and all(np.any(r['out']['now_on'][i] != r['out']['now_off'][i]) for i in idx)


@needs_raw
def test_broken_drive_loop_without_esc_rev_fails(monkeypatch, tmp_path):
    """drive._loop の見張りに esc_rev を渡さない(s4-a を外した)形にすると、160847 で今の ON が fix7 の前の ON と同じになり、
    その 10 フレームで OFF と違う(drive の流し方の試験が s4-a の効きを見ている)"""
    meta, rec, _ = load_rec_drive(LOG)
    floor = floor_ref()
    off = drive_rows(drive_replay(drive, rec, drive_params(meta, 0, 0), tmp_path, floor))
    orig = A.Recovery.step

    def step(self, *a, **kw):
        return orig(self, *a, **dict(kw, esc_rev=None))
    monkeypatch.setattr(A.Recovery, 'step', step)
    on = drive_rows(drive_replay(drive, rec, drive_params(meta, 1, 1), tmp_path, floor))
    monkeypatch.setattr(A.Recovery, 'step', orig)
    t0 = log_t0_us(LOG)
    at = [tn for tn in sorted(on) if round((tn / 1000 - t0) / 1e6, 3) in DIFF_S]
    assert len(at) == 10 and all(on[tn] != off[tn] for tn in at)
    assert [on[tn][:2] for tn in at] == [(1291.0, 1796.0)] * 10            # fix7 の前と同じ back_early の 'reverse'


class _NowLike(object):
    """fix7 の前の Assist を「今」の形(esc_rev を受ける)で呼ぶ包み(esc_rev は捨てる = fix7 s4-a が無い形)"""

    def __init__(self, amod):
        self.amod = amod

    def Assist(self, ref):       # noqa: N802
        a = self.amod.Assist(ref)
        orig = a.step

        def step(*args, esc_rev=None, **kw):
            return orig(*args, **kw)
        a.step = step
        return a


@needs_raw
def test_broken_window_or_off_rule_is_caught():
    """固定した数を 1 つずつ変えた結果を check_concat に入れると落ちる(時刻 1 つ・OFF の違い 1 つ・今の ON ≠ OFF 1 つ・
    後ろへ動いたフレームの数)"""
    import copy
    base = _tot('concat')
    check_concat(base)
    muts = []
    m = copy.deepcopy(base)
    m['diff_on'] = m['diff_on'][:-1]
    m['on_vs_off_at_diff_on'] = m['on_vs_off_at_diff_on'][:-1]
    muts.append(m)
    m = copy.deepcopy(base)
    m['diff_on'][3]['s'] = round(m['diff_on'][3]['s'] + 0.033, 3)
    muts.append(m)
    m = copy.deepcopy(base)
    m['on_vs_off_at_diff_on'][5] = False
    muts.append(m)
    m = copy.deepcopy(base)
    m['diff_off'] = [{'unit': LOG, 's': 40.07}]
    muts.append(m)
    m = copy.deepcopy(base)
    m['backing_bwg'] = m['backing_bwg'][1:]
    muts.append(m)
    for k, m in enumerate(muts):
        with pytest.raises(AssertionError):
            check_concat(m)


# ================================================================ 4. 10/23 の形の後ろの物(rs8-1-backearly-rear-faster)
def _brake_run(monkeypatch, tmp_path, guard, rear=None, esc=None, dur=6.0):
    s = F7.scene_brake(dur=dur, rear=rear, veh_dt=0.01)
    if esc is not None:
        s['esc'] = esc
    sim, rec, pca = F7.run(drive, monkeypatch, tmp_path, F7.P(guard=guard, keep=guard), s)
    return sim, rec, pca


def _phases(pca, t_from):
    """'brake'(後退側 中立 + 300)を書いた時刻からの、書いた値の並び(続く同じ値は 1 つ): 'R' 後退側・'G' 中立の帯・'F' 前進"""
    tb = next(t for t, s, th in pca.w if t > t_from and abs(th - F7.REV) < 1e-6)
    seq = []
    for t, s, th in pca.w:
        if t < tb or t > tb + 1.5:
            continue
        c = 'R' if th > N + 40 else ('F' if th < N - 20 else 'G')
        if not seq or seq[-1] != c:
            seq.append(c)
    return tb, ''.join(seq)


def _rear_hit(sim):
    h = [x for x in sim.car.hits if x[1] == 'rear']
    return None if not h else round(-h[0][2], 2)


N = F7.N


@pytest.mark.parametrize('esc', [None, (0.12, 0.10)])
def test_1023_form_on_skips_gap(monkeypatch, tmp_path, esc):
    """10/23 の形: 見張り ON は 'brake' の中で下がり始めると back_early で 'gap' の中立を飛ばし、後退側を続けて出す(R → G の前に
    1 回の続いた R)。OFF は 'brake' 0.25 秒 → 'gap'(中立)→ 'reverse'(R G R)。fix7 の前の ON も同じ形(fix2 の見張りから)"""
    sim1, rec1, pca1 = _brake_run(monkeypatch, tmp_path, 1, esc=esc)
    sim0, rec0, pca0 = _brake_run(monkeypatch, tmp_path, 0, esc=esc)
    _, s1 = _phases(pca1, 1.5)
    _, s0 = _phases(pca0, 1.5)
    assert s1.startswith('RG') and s1.count('R') == 1 and s1.endswith('F'), s1      # 後退側 → 惰性(中立)→ escape
    assert s0.startswith('RGR'), s0
    assert [e[1]['what'] for e in rec1.events if e[0] == 'recovery_guard'][:1] == ['back_early']
    _, before = F7.load_before()
    simb, recb, pcab = F7.run(before, monkeypatch, tmp_path, F7.P(guard=1, keep=1),
                              dict(F7.scene_brake(dur=6.0, veh_dt=0.01), **({'esc': esc} if esc else {})))
    assert pcab.w == pca1.w                                              # fix7 の前の ON と同じ値(fix7 の直しではない)


# 後ろの物の距離(止まった所から [m])ごとの最初の後ろの当たりの速さ [m/s](None = 当たらない)。閉じたループの偽物の ESC は既定の
# 模型(中立 0.1 秒・後退側から 0.08 秒で下がり始める)と (0.12, 0.10)。台本 experiments/20261026_fix/fix8/fix_safety_r1/brake_rear12.py
# → out/brake_rear12.json の 0.1〜1.8 m の表から選んだ 4 点(既定: ON 1.85 / OFF 1.05・2.06 / 1.65・当たらない / 2.02・当たらない /
# 1.95 m/s。(0.12, 0.10): 1.85 / 1.15・2.19 / 1.7・1.01 / 1.71・当たらない / 1.15)
REAR_D = (0.35, 0.5, 1.15, 1.25)


@pytest.mark.parametrize('esc', [None, (0.12, 0.10)])
def test_1023_form_rear_hit_both_directions(monkeypatch, tmp_path, esc):
    """10/23 の形で後ろに物を置く: 近い物(0.35・0.5 m)には ON のほうが 0.1 m/s より速く当たり、遠い物(1.15・1.25 m)には ON は
    当たらないか OFF より遅い。どちらの向きもあるので、文書は「近い物には ON のほうが強く当たることがある・遠い物には ON のほうが
    遅く当たるか当たらない」と両方を書く(「ON のほうが長く下がる」とは書かない。下がる距離の大小は ESC の模型しだい)"""
    sim, _, _ = _brake_run(monkeypatch, tmp_path, 1, esc=esc)
    stop_x = _stop_x(sim)
    assert stop_x == 2.1711
    got = {}
    for d in REAR_D:
        rear_from_wall = round(2.4 - (stop_x - d), 4)
        on = _rear_hit(_brake_run(monkeypatch, tmp_path, 1, rear=rear_from_wall, esc=esc)[0])
        off = _rear_hit(_brake_run(monkeypatch, tmp_path, 0, rear=rear_from_wall, esc=esc)[0])
        got[d] = (on, off)
    near = [got[d] for d in REAR_D[:2]]
    far = [got[d] for d in REAR_D[2:]]
    assert all(on is not None and off is not None and on > off + 0.1 for on, off in near), got
    assert all(off is not None and (on is None or on < off - 0.1) for on, off in far), got


def _stop_x(sim):
    """リカバリーの前に止まった位置(前の壁 2.4 m の手前。ブレーキ側で止まった所)"""
    xs = [x for t, x, v, th in sim.log if 1.5 < t < F7.T_BRAKE and abs(v) < 1e-9]
    return round(xs[-1], 4)
