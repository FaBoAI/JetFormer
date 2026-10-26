"""反証 V-replay の共通の読み込み(2026-10-25)。読むだけ。生の記録・ラベル・設定は変えない。

段 drive の replay_speed_ctrl.py とは別に作った(入力の作り方・走行の一覧・区間・衝突・周を独立に):
  - 走行: data/raw/2026102[34]-* の全部(AI のフレームが 3 秒以上。dry run・中立止めは除く)。段 drive は tt_data の 28 本だけ
  - ラベル: tools/label_events.label_log をその場で呼ぶ(偽の起点を外した今の決まり。data/labels は使わない・書かない)
  - 周: tools/section_scorecard.score_log(labels=その場のラベル)の laps(R1 の起点 → 次の起点、偽の起点を外したあと)
  - 天井の入力: drive.py の speed_inputs と同じ決まり(撮像の時刻までの最新 3 つの eRPM の中央値・0.1 秒前との差・
    最新 5 つの gz の平均・舵 = 車に出した舵 − (中心 + フレームごとのトリム))を、numpy で作り直す
  - 天井そのものは実装の純粋な関数 jetformer.safety_caps.ceil_step(仕様の判定ではなく、作ったものの効き目を数える)
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = '/home/jetson/jetformer'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import label_events as le  # noqa: E402
import section_scorecard as ssc  # noqa: E402
from jetformer import safety_caps as sc  # noqa: E402

KV = sc.KV
G0 = 9.80665
REP24 = '/home/jetson/jetformer_reports/bench_20261024_report'
OUT_DIR = '/home/jetson/jetformer_reports/speed_control_20261025'
HERE = os.path.dirname(os.path.abspath(__file__))
CAM = json.load(open(os.path.join(ROOT, 'experiments', '20261024_report', 'col', 'cam_check.json')))['events']
COL = json.load(open(os.path.join(ROOT, 'experiments', '20261024_report', 'col', 'col_data.json')))
TT = json.load(open(os.path.join(REP24, 'tt_data.json')))
TT_RUNS = {r['log']: r for r in TT['runs']}
LOC_DETAIL = ['?', 'R1', 'GM_in', 'GM_out', 'DE', 'SW', 'HS', 'LH']


def desc_tok(desc, key):
    for tok in (desc or '').split():
        if tok.startswith(key + '='):
            return tok.split('=', 1)[1]
    return None


def session_of(log, no):
    """回の名前。10/24 は tt_data のまま。10/23 は 公式 8 本(No.8〜15)・速さ上げ(No.16〜26)・その前(No.1〜6)"""
    if log in TT_RUNS:
        return TT_RUNS[log]['session']
    if log.startswith('20261023'):
        n = int(no) if no and no.isdigit() else 0
        return '10/23 速さ上げ' if n >= 16 else '10/23 その前'
    return '10/24 ほか'


def all_logs():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, 'data', 'raw', '2026102[34]-*'))
                  if len(os.path.basename(p)) == 15)


def _median3(vt, ve, tq):
    j = np.searchsorted(vt, tq, side='right') - 1
    jj = np.clip(np.stack([j - 2, j - 1, j]), 0, len(vt) - 1)     # 3 つ無ければ古い方で埋める(drive と同じ)
    v = np.median(-ve[jj] * KV, axis=0)
    v = np.where(j >= 0, v, np.nan)
    return v, j


def load_run(log):
    """1 本を読む。戻り値 dict(フレームの配列・周・衝突)。AI のフレームが足りなければ None"""
    info, fr = le.label_log(log)
    if fr is None:
        return None
    L = le.load_log(log)
    V, I = L['VEHICLE'], L['IMU']
    if V is None or I is None:
        return None
    t0 = int(info['t0_us'])
    t = fr['t_s'].astype(float)
    if np.any(np.diff(t) < 0):
        raise RuntimeError('フレームの時刻が逆に並んでいる: ' + log)
    vt = (V['timestamps_us'].astype(np.int64) - t0) / 1e6
    ve = V['erpm'].astype(float)
    ov = np.argsort(vt, kind='stable')
    vt, ve = vt[ov], ve[ov]
    it = (I['timestamps_us'].astype(np.int64) - t0) / 1e6
    oi = np.argsort(it, kind='stable')
    it = it[oi]
    gz = I['gyro'][oi, 2].astype(float)
    acc = I['accel'][oi].astype(float)
    v, j = _median3(vt, ve, t)
    v_old, j_old = _median3(vt, ve, t - 0.1)
    a = np.where(j_old >= 0, (v - v_old) / 0.1, np.nan)
    fresh = (j >= 0) & (t - vt[np.clip(j, 0, None)] < 0.1)
    k = np.searchsorted(it, t, side='right') - 1
    kk = np.clip(np.stack([k - m for m in range(4, -1, -1)]), 0, len(it) - 1)   # 古い順。5 つ無ければ古い方で埋める
    gz5 = np.where(k >= 0, gz[kk].mean(axis=0), np.nan)
    hg = np.hypot(acc[:, 0], acc[:, 1]) / G0
    a3 = np.abs(np.linalg.norm(acc, axis=1) / G0 - 1.0)
    # フレームごとの IMU の衝撃の最大(前のフレームの後 〜 このフレームまで)
    kc = np.clip(k + 1, 0, len(it))
    hg_f = np.zeros(len(t))
    a3_f = np.zeros(len(t))
    prev = 0
    for i in range(len(t)):
        e_ = kc[i]
        if e_ > prev:
            hg_f[i] = hg[prev:e_].max()
            a3_f[i] = a3[prev:e_].max()
            prev = e_
    center = float(info['ref_us']['steer_center'])
    neutral = float(info['ref_us']['throttle_neutral'])
    out = fr['out'].astype(float)
    steer = out[:, 0] - (center + fr['trim'].astype(float))
    thr = out[:, 1].copy()
    ai = fr['ai'].astype(bool) & ~fr['dry_run'].astype(bool) & ~fr['hold'].astype(bool)
    active = ai & fresh & np.isfinite(v)
    desc = info.get('description', '')
    no = desc_tok(desc, 'ai_no') or ''
    # 周(偽の起点を外したあと)・衝突(場所は外したあとの起点で)
    ex = {}
    try:
        sco = ssc.score_log(log, labels=(info, fr, 'fresh'), loaded=L, extras=ex)
    except Exception as e:  # noqa: BLE001
        sco = {'laps': [], 'collision_events': [], 'error': repr(e)}
    loc_code = ex.get('loc_code', fr['loc_code']).astype(int)
    loc_detail = fr['loc_detail'].astype(int)
    tg = float(np.median(fr['gains'][ai, 1])) if ai.any() else float('nan')
    return dict(log=log, no=no, session=session_of(log, no), desc=desc, t0_us=t0, t=t, ai=ai, active=active, v=v, a=a,
                gz5=gz5, steer=steer, thr=thr, neutral=neutral, center=center, assist=fr['assist'].astype(int),
                guard=fr['guard_applied'].astype(bool), hg=hg_f, a3=a3_f, loc=loc_code, locd=loc_detail,
                gains=fr['gains'].astype(float), tg=tg, laps=sco.get('laps', []), coll=sco.get('collision_events', []),
                anchors=np.asarray(ex.get('anchors', []), float), rejected=sco.get('r1_anchors_rejected', []),
                vt=vt, ve=ve, it=it, gz=gz, ai_s=float(fr['dt_s'][ai].sum()), kap=np.where(
                    v > 0.3, np.deg2rad(gz5) / np.maximum(v, 0.3), 0.0), dist=_dist(t, v))


def _dist(t, v):
    d = np.zeros(len(t))
    vv = np.nan_to_num(np.maximum(v, 0.0))
    dt = np.clip(np.diff(t), 0, 0.2)
    d[1:] = np.cumsum(0.5 * (vv[1:] + vv[:-1]) * dt)
    return d


def open_loop(d, p):
    """記録の入力を ceil_step に通す(記録の車は動きを変えない)"""
    st = sc.CeilState()
    n = len(d['t'])
    thr_out = np.zeros(n)
    mode = np.full(n, -1, int)
    left = np.zeros(n, bool)
    cap = np.full(n, np.nan)
    vpred = np.full(n, np.nan)
    for i in range(n):
        st, o = sc.ceil_step(st, float(d['t'][i]), float(d['v'][i]), float(d['a'][i]), float(d['steer'][i]),
                             float(d['gz5'][i]), float(d['thr'][i]), d['neutral'], bool(d['active'][i]), p)
        thr_out[i], mode[i], left[i], cap[i], vpred[i] = o.thr_us, o.mode, o.left, o.cap, o.v_pred
    return dict(thr_out=thr_out, mode=mode, left=left, cap=cap, vpred=vpred)


def P(variant='impl', cap_all=4.5, cap_left=4.0, **kw):
    if variant == 'spec':
        kw = dict(dict(start_from_hold=False, release_bumpless=False), **kw)
    return sc.ceil_params(cap_all, cap_left, **kw)


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=conv)
