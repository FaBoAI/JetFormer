#!/usr/bin/env python3
"""データセット v2 の棚卸し(2026-10-25): data/raw の実車の記録(10/10・10/23・10/24)をすべて数える。

読むだけ(numpy・OpenCV。torch・GPU は使わない)。data/ は変えない。出力は次の 2 つだけ:
  /home/jetson/jetformer_reports/dataset_v2_plan_20261025/inventory.json
  /home/jetson/jetformer_reports/dataset_v2_plan_20261025/inventory.md

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        python3 -I /home/jetson/jetformer/experiments/20261025_dataset_v2/inventory.py

数え方(この台本で決めたもの。md の 8 章にも書く):
  フレーム       ラベル(data/labels/<記録>.npz)のフレーム(AI 走行の記録 = AI_CMD、人だけの記録 = カメラ。30 fps)
  AI の秒        CH3 が AI 側・dry run でないフレームの dt の和。人の秒 = CH3 が人側。dry run の秒 = AI 側で dry run
  時間帯         記録の始まりの時刻(日本時間): 昼 〜16:59・夕方 17:00〜18:59・夜 19:00〜。回 = 60 分を超えて空いたら次
  明るさ         カメラの画像(224×224)の灰色の平均(0〜255)。3 枚に 1 枚(学習の間引き 3 と同じ)。上 1/3 の平均も出す
  直線の最高速   前へ走っている(eRPM < −1000)・まっすぐ(ヨーの 11 サンプル平均 |gz| < 40 deg/s)フレームの前進 eRPM の 90% 点
  周             R1 の起点 → 次の起点(5〜16 秒・ずっと同じ操作者・0.2 秒を超える途切れなし)。tools/section_scorecard.py と同じ
  きれいな周     AI: その間に AI の衝突・交代・リカバリー・基板の再起動が無い(section_scorecard と同じ)
                 人: その間に衝突(水平 2.5G、ラベル)・交代・基板の再起動が無い
  (a) 人の走行   CH3 が人側・操作値が有効(ラベルの valid_action)・前へ走っている(eRPM < −200)・
                 スロットルを 1 秒以内に入れている(押している・惰性を除く)・持ち上げていない
                 (1 秒の平均で 傾き > 30° かつ |gz| < 40 deg/s を除く)
  (b) 交代の直後 AI → 人の交代から 3 秒の、人側のフレーム
  (c) AI のきれいな周  上のきれいな周のフレーム(9 秒未満の周も別に数える)
  (d) AI の衝突の前後  AI 側のフレームで、衝突の 1 秒前〜2 秒後(ラベルの near_collision。学習の --mask-impacts と同じ幅)
  サンプル       間引き 3(シーンの中のフレーム番号 % 3 == 0。jetformer/train_data.py と同じ)
"""

import glob
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import label_events as le  # noqa: E402  (numpy だけ。torch は読まない)

RAW = ROOT / 'data' / 'raw'
LABELS = ROOT / 'data' / 'labels'
DATASETS = ROOT / 'data' / 'datasets'
OUT = Path('/home/jetson/jetformer_reports/dataset_v2_plan_20261025')
DAYS = ('20261010', '20261023', '20261024')
DAY_NAMES = {'20261010': '10/10', '20261023': '10/23', '20261024': '10/24'}
BANDS = ('昼', '夕方', '夜')
JST = timezone(timedelta(hours=9))
STRIDE = 3
TK_WINDOW_S = 3.0
SUB9_S = 9.0
LAP_MIN_S, LAP_MAX_S = 5.0, 16.0
LAP_SUSPECT_S = 7.0          # これより速い周は区切りの誤り(R1 の起点の取り違え)とみなす(人のベスト 7.79 秒・上位チーム 約 7.7 秒)
FWD_ERPM = -200.0            # これより小さい(前へ速い) = 前へ走っている
REV_ERPM = 200.0
STRAIGHT_ERPM = -1000.0
STRAIGHT_GZ = 40.0           # deg/s
THR_ACTIVE_US = 20.0         # 人のスロットルが中立からこれより離れている = 入れている
COAST_S = 1.0                # これより長くスロットルを入れていない = 押している・惰性
LIFT_TILT_DEG = 30.0
LIFT_GZ = 40.0
BASE_DS = 'clean_driving_20261011'
SC_REF = {'20261023': Path('/home/jetson/jetformer_reports/real_bench_20261023/nightly_20261023/scorecard/scorecard.json'),
          '20261024': Path('/home/jetson/jetformer_reports/bench_20261024_tune/since_1300/nightly/scorecard/scorecard.json')}

MODEL_NAMES = {
    'clean_driving_20261011-nocmd-s3-20261011-024626': 'B',
    'bench_20261022-slow-sec-s1': '遅く入る型',
    'bench_20261021-ens-p2': '2本の平均',
    'bench_20261021-aug-s0': 'AUG',
    'bench_20261021-wgt-s0': 'WGT',
    'clean_driving_20261011-act-s0-nocmd-stride3-e30': 'ACT型',
    'clean_driving_20261011-tcp-s1-nocmd-stride3-e30': 'TCP型',
    'clean_driving_20261011-dinov2-s0-nocmd-stride3-e30': 'DINOv2型',
    'first_trial_20261010-s3-20261010-153629': '最初のモデル',
}
POLICY_NAMES = {'jetformer_v0': 'TransFuser軽量型', 'act': 'ACT型', 'tcp': 'TCP型', 'dinov2': 'DINOv2型',
                'ens_average': 'TransFuser軽量型 2 本の平均', None: 'TransFuser軽量型'}
CAT_NAMES = {'drive': '走行', 'stop': '停止', 'reverse': '後退', 'coast': '押し・惰性', 'lift': '持ち上げ',
             'invalid': '操作値なし'}
CATS = ('drive', 'stop', 'reverse', 'coast', 'lift', 'invalid')


# ---------------------------------------------------------------- 小道具
def read_json(p):
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def r(x, n=1):
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    return round(x, n)


def pct(x, q, min_n=1):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) >= min_n else None


def movavg(x, k):
    if len(x) < k:
        return np.full(len(x), np.mean(x) if len(x) else np.nan)
    return np.convolve(x, np.ones(k) / k, 'same')


def band_of(dt):
    """昼 〜16:59(日の入りの前)・夕方 17:00〜18:59・夜 19:00〜"""
    h = dt.hour + dt.minute / 60.0
    return '昼' if h < 17 else ('夕方' if h < 19 else '夜')


def sessions(logs):
    """記録の始まりの間が 60 分を超えたら次の回。{記録: '10/24 2 回目(16:01〜)'}"""
    out = {}
    for day in DAYS:
        L = sorted(x for x in logs if x[:8] == day)
        k, first, prev = 0, None, None
        groups = []
        for x in L:
            t = log_local(x)
            if prev is None or (t - prev).total_seconds() > 3600:
                groups.append([])
            groups[-1].append(x)
            prev = t
        for i, g in enumerate(groups):
            lab = '%s %d 回目(%s〜)' % (DAY_NAMES[day], i + 1, log_local(g[0]).strftime('%H:%M'))
            for x in g:
                out[x] = lab
    return out


def local(us):
    return datetime.fromtimestamp(us / 1e6, tz=JST)


def log_local(log):
    return datetime.strptime(log, '%Y%m%d-%H%M%S').replace(tzinfo=JST)


def samples(mask, idx_in_scene):
    return int(np.sum(mask & (idx_in_scene % STRIDE == 0)))


def battery_note(desc):
    out = []
    for m in re.finditer(r'(\d\.\d+)\s*V\s*の(?:バッテリー|電池)', desc or ''):   # 「電圧8.0V以下のとき」などの予定は入れない
        out.append(m.group(1) + 'V')
    if '充電100%' in (desc or '') or '満充電' in (desc or ''):
        out.append('満充電')
    for k in ('V_before', 'V_after', 'battery'):
        m = re.search(k + r'=([^\s]+)', desc or '')
        if m and '_' not in m.group(1) and m.group(1) not in ('#__',):
            out.append('%s=%s' % (k, m.group(1)))
    return '・'.join(out) or None


def desc_short(desc):
    d = desc or ''
    if 'bench=' not in d:
        return d[:40]
    parts = []
    for k in ('cand', 'tune'):
        m = re.search(k + r'=(\S+)', d)
        if m:
            parts.append('%s=%s' % (k, m.group(1)))
    m = re.search(r'official=(\d)', d)
    if m:
        parts.append('公式' if m.group(1) == '1' else '試し')
    m = re.search(r'memo=(.*)$', d)
    if m and m.group(1).strip():
        parts.append('memo:' + m.group(1).strip()[:30])
    free = re.sub(r'(ai_no|bench|day|block|cand|model|battery|V_before|V_after|check|tune|throttle_gain|official|'
                  r'steer_gain|steer|plan_laps)=\S*', '', d.split('memo=')[0]).strip()
    if free:
        parts.append(free[:30])
    return ' '.join(parts)


def mode_table():
    m = read_json(ROOT / 'data' / 'modes.json') or []
    return {int(x['id']): x['name'] for x in m}


def dataset_membership():
    """実車のシーン → [[データセット, 分け方], ...]"""
    mem = {}
    for d in sorted(DATASETS.iterdir()):
        c = read_json(d / 'dataset_card.json')
        if not c:
            continue
        s = read_json(d / 'splits.json') or {}
        split_of = {}
        for k, v in s.items():
            if isinstance(v, list):
                for x in v:
                    split_of[x] = k
        for sc in c.get('source_scenes') or []:
            if sc[:8] in DAYS:
                mem.setdefault(sc, []).append([d.name, split_of.get(sc)])
    return mem


def scene_brightness(sdir):
    """3 枚に 1 枚の (時刻 µs, 平均, 上 1/3 の平均)。CAM_FRONT.npz が壊れていれば jpg の名前の時刻で"""
    try:
        with np.load(sdir / 'CAM_FRONT.npz', allow_pickle=False) as z:
            ts = z['timestamps_us'].astype(np.int64)
            files = [str(x) for x in z['files']]
    except Exception:
        names = sorted(os.listdir(sdir / 'CAM_FRONT')) if (sdir / 'CAM_FRONT').is_dir() else []
        names = [x for x in names if x.endswith('.jpg') and x[:-4].isdigit()]
        ts = np.array([int(x[:-4]) for x in names], np.int64)
        files = ['CAM_FRONT/' + x for x in names]
    idx = np.arange(0, len(ts), STRIDE)
    m = np.full(len(idx), np.nan)
    top = np.full(len(idx), np.nan)
    for j, i in enumerate(idx):
        im = cv2.imread(str(sdir / files[i]), cv2.IMREAD_REDUCED_GRAYSCALE_2)
        if im is None:
            continue
        m[j] = float(im.mean())
        top[j] = float(im[:im.shape[0] // 3].mean())
    span = (int(ts[0]), int(ts[-1])) if len(ts) else None
    return ts[idx] if len(ts) else ts, m, top, len(ts), span


def gain_summary(g, mask):
    """AI 側のフレームのゲインの {中央値, 値ごとの割合}"""
    if not np.any(mask):
        return None
    v = np.round(g[mask].astype(float), 2)
    u, c = np.unique(v, return_counts=True)
    order = np.argsort(-c)
    return {'median': r(np.median(v), 2), 'values': {('%.2f' % u[i]): r(c[i] / len(v), 3) for i in order}}


# ---------------------------------------------------------------- 1 本の記録
def process_log(log, membership, modes, session=None):
    root = RAW / log
    scenes = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name.startswith(log + '-'))
    meta = read_json(root / 'log.json')
    smeta = [read_json(root / s / 'scene.json') for s in scenes]
    info = read_json(LABELS / (log + '.json'))
    try:
        with np.load(LABELS / (log + '.npz'), allow_pickle=False) as z:
            lab = {k: z[k] for k in z.files}
    except Exception:
        lab = None
    start = log_local(log)
    rec = {'log': log, 'day': log[:8], 'date': DAY_NAMES[log[:8]], 'start': start.strftime('%H:%M:%S'),
           'band': band_of(start), 'session': session, 'n_scenes': len(scenes), 'problems': [], 'usable': True}
    desc = (meta or {}).get('description') or next((m.get('description') for m in smeta if m and m.get('description')), '')
    rec['description'] = desc
    rec['desc_short'] = desc_short(desc)
    m_ai = re.search(r'ai_no=(\d+)', desc or '')
    rec['ai_no'] = int(m_ai.group(1)) if m_ai else None
    if rec['ai_no'] is None:
        m_run = re.search(r'(?:ai_run_|run_?)(\d+)', desc or '')
        rec['run_no'] = m_run.group(1) if m_run else None
    rec['battery_note'] = battery_note(desc)
    tags = (meta or {}).get('tags') or next((m.get('tags') for m in smeta if m), []) or []
    rec['ai_log'] = 'ai-drive' in tags or any(m and m.get('drive') for m in smeta)

    # ---- 明るさ(3 枚に 1 枚)
    bt, bm, btop, bsc = [], [], [], []
    cam_frames, spans = [], []
    for si, s in enumerate(scenes):
        t_, m_, top_, ncam, span = scene_brightness(root / s)
        spans.append(span)
        bt.append(t_)
        bm.append(m_)
        btop.append(top_)
        bsc.append(np.full(len(t_), si))
        cam_frames.append(ncam)
    bt = np.concatenate(bt) if bt else np.zeros(0, np.int64)
    bm = np.concatenate(bm) if bm else np.zeros(0)
    btop = np.concatenate(btop) if btop else np.zeros(0)
    bsc = np.concatenate(bsc) if bsc else np.zeros(0, int)
    rec['camera_frames'] = int(sum(cam_frames))

    # ---- シーンの表(scene.json)
    srows = []
    for si, (s, m) in enumerate(zip(scenes, smeta)):
        row = {'scene': s, 'camera_frames': cam_frames[si], 'datasets': membership.get(s, [])}
        if m:
            st = local(m['start_us']) if m.get('start_us') else None
            row.update(start=st.strftime('%H:%M:%S') if st else None, band=band_of(st) if st else rec['band'],
                       duration_s=m.get('duration_s'), complete=m.get('complete'))
            segs = m.get('mode_segments') or [{'start_s': 0.0, 'end_s': m.get('duration_s') or 0, 'mode_id': m.get('mode_id')}]
            row['genres'] = sorted({modes.get(int(x['mode_id']), str(x['mode_id'])) for x in segs if x.get('mode_id') is not None})
            row['_segs'] = segs
            row['_start_us'] = m.get('start_us')
            dr = m.get('drive') or {}
            row['run'] = dr.get('run')
            row['model'] = MODEL_NAMES.get(dr.get('run'), dr.get('run')) if dr.get('run') else None
            row['policy'] = POLICY_NAMES.get(dr.get('policy'), dr.get('policy')) if dr.get('run') else None
            row['dry_run_flag'] = dr.get('dry_run') if dr else None
            if m.get('description') and m.get('description') != desc:
                row['description'] = m.get('description')
        else:
            row.update(start=None, band=rec['band'], duration_s=None, genres=[], run=None, model=None, policy=None)
            row['problem'] = 'scene.json が読めない(空)'
        b = bsc == si
        row['brightness_p50'] = r(pct(bm[b], 50))
        row['brightness_top_p50'] = r(pct(btop[b], 50))
        srows.append(row)
    rec['scenes'] = srows
    rec['brightness'] = {'p10': r(pct(bm, 10)), 'p50': r(pct(bm, 50)), 'p90': r(pct(bm, 90)),
                         'top_p50': r(pct(btop, 50)), 'n': int(np.isfinite(bm).sum())}
    rec['_bright'] = (bt, bm, btop)

    if meta is None or info is None or lab is None:
        rec['usable'] = False
        rec['problems'].append('log.json・ラベルが無い(ファイルが空。記録中の電源断か)。カメラの画像だけ')
        rec['duration_s'] = None
        return rec, None

    # ---- フレームごとの値(ラベル)
    t = lab['t_s'].astype(np.float64)
    n = len(t)
    t_us = lab['t_us'].astype(np.int64)
    # シーンは時刻で決める(10/11 にラベルを作った後で消したシーンがある: 181504・193638・200916・204605)
    scene = np.full(n, -1, int)
    for si, sp in enumerate(spans):
        if sp is not None:
            scene[(t_us >= sp[0] - 50000) & (t_us <= sp[1] + 50000)] = si
    ondisk = scene >= 0
    ai_sel = lab['ai'].astype(bool) & ondisk
    dry = lab['dry_run'].astype(bool)
    ai = ai_sel & ~dry
    dryf = ai_sel & dry
    hum = ~lab['ai'].astype(bool) & ondisk
    dt = lab['dt_s'].astype(np.float64)
    erpm = lab['erpm'].astype(np.float64)
    valid = lab['valid_action'].astype(bool)
    near = lab['near_collision'].astype(bool)
    gains = lab['gains'].astype(np.float64)
    assist = lab['assist'].astype(int)
    idx_in = np.zeros(n, int)
    for s in np.unique(scene):
        w = np.nonzero(scene == s)[0]
        idx_in[w] = np.arange(len(w))
    L = le.load_log(log)
    lab_scenes = list(scenes)
    sc_name = np.array([scenes[s] if s >= 0 else '' for s in scene])
    rec['frames_not_on_disk'] = int((~ondisk).sum())
    if rec['frames_not_on_disk']:
        rec['problems'].append('ラベルの %d フレームは消したシーンのもの(数に入れない)' % rec['frames_not_on_disk'])
    rec['problems'] += list(info.get('problems') or [])

    # 明るさをフレームへ
    fin = np.isfinite(bm)
    bf = np.interp(t_us, bt[fin], bm[fin]) if fin.sum() >= 2 else np.full(n, np.nan)
    bft = np.interp(t_us, bt[fin], btop[fin]) if fin.sum() >= 2 else np.full(n, np.nan)

    # ジャンル(シーンの mode_segments)
    name2row = {x['scene']: x for x in srows}
    mode = np.zeros(n, int)
    for s in np.unique(scene):
        w = scene == s
        row = name2row.get(lab_scenes[s]) if 0 <= s < len(lab_scenes) else None
        if not row or not row.get('_segs'):
            continue
        rel = (t_us[w] - (row.get('_start_us') or t_us[w][0])) / 1e6
        mm = np.full(w.sum(), int(row['_segs'][0].get('mode_id') or 0))
        for sg in row['_segs']:
            mm[(rel >= sg['start_s'] - 0.05) & (rel <= sg['end_s'] + 0.05)] = int(sg.get('mode_id') or 0)
        mode[w] = mm

    # ---- IMU・車両: ヨー・直線・持ち上げ・押し
    I, V = L['IMU'], L['VEHICLE']
    t0 = info['t0_us']
    gzs_f = np.full(n, np.nan)
    lifted = np.zeros(n, bool)
    coast_flag = np.zeros(n, bool)
    anchors = np.zeros(0)
    if I is not None and V is not None and len(I['timestamps_us']) > 20:
        it = (I['timestamps_us'].astype(np.int64) - t0) / 1e6
        vt = (V['timestamps_us'].astype(np.int64) - t0) / 1e6
        verpm = V['erpm'].astype(np.float64)
        gz = I['gyro'][:, 2].astype(np.float64)
        acc = I['accel'].astype(np.float64)
        still = (np.abs(np.interp(it, vt, verpm)) < 50) & (np.abs(gz) < 5)
        hd, gzu = le.heading_deg(it, gz, still)
        anchors, gm_exit, gzs = le.course_anchors(it, hd, gzu, vt, verpm)
        anchors = np.asarray(anchors, float)
        gzs_f = np.interp(t, it, gzs)
        gref = np.median(acc[still], axis=0) if still.sum() >= 50 else np.array([0.0, 0.0, le.G])
        alp = np.stack([movavg(acc[:, k], 100) for k in range(3)], 1)
        cosang = (alp @ gref) / (np.linalg.norm(alp, axis=1) * np.linalg.norm(gref) + 1e-9)
        tilt = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        gz1 = movavg(gzu, 100)
        lift_i = (tilt > LIFT_TILT_DEG) & (np.abs(gz1) < LIFT_GZ)
        lifted = np.interp(t, it, lift_i.astype(float)) > 0.5
        thr = V['pwm'][:, 1].astype(np.float64)
        vval = V['valid'][:, 1].astype(bool) if 'valid' in V else np.ones(len(thr), bool)
        z0 = vval & (verpm == 0)
        neutral_h = float(np.median(thr[z0])) if z0.sum() >= 20 else 1500.0
        active = vval & (np.abs(thr - neutral_h) > THR_ACTIVE_US)
        last_act = np.maximum.accumulate(np.where(active, vt, -1e9))
        iv = np.clip(le.latest_index(vt, t), 0, len(vt) - 1)
        coast_flag = (t - last_act[iv]) > COAST_S
        rec['human_throttle_neutral_us'] = r(neutral_h, 0)
    else:
        rec['problems'].append('IMU・車両が無いので、直線・持ち上げ・押し・周は出せない')

    fwd = erpm < FWD_ERPM
    rev = erpm > REV_ERPM
    # 人のフレームの分類(上から順に決める)
    cat = np.full(n, 'drive', dtype=object)
    cat[coast_flag] = 'coast'
    cat[rev] = 'reverse'
    cat[~(fwd | rev)] = 'stop'
    cat[lifted] = 'lift'
    cat[~valid] = 'invalid'
    a_mask = hum & (cat == 'drive')
    straight = (erpm < STRAIGHT_ERPM) & (np.abs(gzs_f) < STRAIGHT_GZ)

    ev = info.get('events') or {}

    def on_disk(e, key='t_s'):
        k = int(np.clip(np.searchsorted(t, e[key], side='right') - 1, 0, n - 1))
        return bool(ondisk[k])
    colls = [e for e in ev.get('collisions', []) if on_disk(e)]
    tks = [e for e in ev.get('takeovers', []) if on_disk(e)]
    rbs = [e for e in ev.get('board_reboots', []) if on_disk(e, 't_last_s')]
    bumps = [e for e in ev.get('bumps', []) if on_disk(e)]
    coll_ai_t = np.array([c['t_s'] for c in colls if c.get('ai')])
    coll_all_t = np.array([c['t_s'] for c in colls])
    tk_t = np.array([x['t_s'] for x in tks])
    rb_t = np.array([b['t_last_s'] for b in rbs])

    # (b) 交代の直後 3 秒
    b_mask = np.zeros(n, bool)
    b_by_class = {}
    for e in tks:
        w = hum & (t >= e['t_s']) & (t < e['t_s'] + TK_WINDOW_S)
        b_mask |= w
        c = b_by_class.setdefault(e.get('class') or '?', {'takeovers': 0, 'frames': 0})
        c['takeovers'] += 1
        c['frames'] += int(w.sum())

    # ---- 周(R1 の起点 → 次の起点)
    laps = []
    c_mask = np.zeros(n, bool)
    c9_mask = np.zeros(n, bool)
    hl_mask = np.zeros(n, bool)
    for j in range(len(anchors) - 1):
        a, b = float(anchors[j]), float(anchors[j + 1])
        m = (t >= a) & (t < b)
        if not m.any():
            continue
        dur = b - a
        tm = t[m]
        gaps = np.any(np.diff(tm) > le.GAP_S) or (tm[0] - a > le.GAP_S) or (b - tm[-1] > le.GAP_S)
        if not (LAP_MIN_S <= dur <= LAP_MAX_S) or gaps or not ondisk[m].all():
            continue
        if ai[m].all():
            who = 'ai'
            clean = (not np.any((coll_ai_t >= a) & (coll_ai_t < b)) and not np.any((tk_t >= a) & (tk_t < b))
                     and not np.any(assist[m] == 2) and not np.any((rb_t >= a - 0.5) & (rb_t < b)))
        elif hum[m].all():
            who = 'human'
            clean = (not np.any((coll_all_t >= a) & (coll_all_t < b)) and not np.any((tk_t >= a) & (tk_t < b))
                     and not np.any((rb_t >= a - 0.5) & (rb_t < b)))
        else:
            continue
        si = int(scene[m][0])
        row = name2row.get(lab_scenes[si])
        st = straight & m
        clean_raw = bool(clean)
        suspect = dur < LAP_SUSPECT_S
        clean = clean_raw and not suspect
        lp = {'who': who, 't_start': r(a, 3), 'lap_s': r(dur, 3), 'clean': bool(clean), 'clean_raw': clean_raw,
              'suspect': bool(suspect), 'scene': lab_scenes[si],
              'frames': int(m.sum()), 'samples': samples(m, idx_in),
              'bumps': sum(1 for x in bumps if a <= x['t_s'] < b),
              'straight_erpm_p90': r(pct(-erpm[st], 90, 10), 0), 'brightness': r(np.nanmean(bf[m])),
              'model': row.get('model') if row else None}
        if who == 'ai':
            lp['throttle_gain'] = r(np.median(gains[m, 1]), 2)
            lp['steer_gain'] = r(np.median(gains[m, 0]), 2)
            if clean:
                c_mask |= m
                if dur < SUB9_S:
                    c9_mask |= m
        elif clean:
            hl_mask |= m
        laps.append(lp)

    d_mask = ai & near

    def ssum(mask):
        return r(dt[mask].sum(), 1)

    def cand(mask):
        return {'frames': int(mask.sum()), 'samples': samples(mask, idx_in), 'seconds': ssum(mask)}

    # ---- 記録の数
    rec['duration_s'] = r(sum((x.get('duration_s') or 0) for x in srows), 1)
    rec['span_s'] = info.get('duration_s')
    rec['frames'] = int(ondisk.sum())
    rec['ai_s'] = ssum(ai)
    rec['human_s'] = ssum(hum)
    rec['dry_s'] = ssum(dryf)
    rec['dry_run'] = bool(dryf.any()) or any(x.get('dry_run_flag') for x in srows)
    rec['models'] = sorted({x['model'] for x in srows if x.get('model')})
    rec['policies'] = sorted({x['policy'] for x in srows if x.get('policy')})
    rec['throttle_gain'] = gain_summary(gains[:, 1], ai)
    rec['steer_gain'] = gain_summary(gains[:, 0], ai)
    gm = {}
    for mid in np.unique(mode[ondisk]):
        gm[modes.get(int(mid), str(mid))] = ssum((mode == mid) & ondisk)
    rec['genres_s'] = gm
    rec['collisions'] = {'ai': int(sum(1 for c in colls if c.get('ai'))),
                         'human': int(sum(1 for c in colls if not c.get('ai'))),
                         'strong': int(sum(1 for c in colls if c.get('strong'))),
                         'bumps': len(bumps)}
    rec['takeovers'] = {'all': len(tks), 'by_class': {k: v['takeovers'] for k, v in b_by_class.items()}}
    rec['board_reboots'] = len(rbs)
    al = [x for x in laps if x['who'] == 'ai']
    hl = [x for x in laps if x['who'] == 'human']
    ac = sorted(x['lap_s'] for x in al if x['clean'])
    hc = sorted(x['lap_s'] for x in hl if x['clean'])
    rec['laps'] = {'ai': len(al), 'ai_clean': len(ac), 'ai_clean_s': ac, 'ai_sub9': sum(1 for x in ac if x < SUB9_S),
                   'suspect': [x['lap_s'] for x in laps if x['suspect']],
                   'human': len(hl), 'human_clean': len(hc), 'human_clean_s': hc,
                   'lapcounter': {k: (info.get('lap_summary') or {}).get(k) for k in ('laps', 'clean_laps', 'best_ai_s', 'median_ai_s')}}
    rec['lap_list'] = laps
    rec['straight_erpm_p90'] = {'ai': r(pct(-erpm[straight & ai], 90, 30), 0), 'human': r(pct(-erpm[straight & hum], 90, 30), 0),
                                'ai_n': int((straight & ai).sum()), 'human_n': int((straight & hum).sum())}
    rec['human_categories_s'] = {CAT_NAMES[k]: ssum(hum & (cat == k)) for k in CATS}
    base_tr = np.array([any(d == BASE_DS and sp == 'train' for d, sp in membership.get(s, [])) for s in sc_name])
    base_va = np.array([any(d == BASE_DS and sp == 'val' for d, sp in membership.get(s, [])) for s in sc_name])
    rec['candidates'] = {
        'a_human_drive': cand(a_mask), 'a_human_drive_no_coll': cand(a_mask & ~near),
        'a_in_base_train': cand(a_mask & base_tr), 'a_in_base_val': cand(a_mask & base_va),
        'a_new': cand(a_mask & ~base_tr & ~base_va),
        'a_nc_in_base_train': cand(a_mask & ~near & base_tr), 'a_nc_in_base_val': cand(a_mask & ~near & base_va),
        'a_nc_new': cand(a_mask & ~near & ~base_tr & ~base_va),
        'a_after_takeover': cand(a_mask & b_mask), 'a_other_than_takeover': cand(a_mask & ~b_mask),
        'a_new_track': cand(a_mask & ~base_tr & ~base_va & np.isin(mode, (4, 5))),
        'a_nc_new_track': cand(a_mask & ~near & ~base_tr & ~base_va & np.isin(mode, (4, 5))),
        'human_all': cand(hum), 'human_valid': cand(hum & valid),
        'b_takeover_3s': cand(b_mask), 'b_takeover_3s_drive': cand(b_mask & (cat == 'drive')),
        'b_by_class': b_by_class,
        'c_ai_clean_laps': cand(c_mask), 'c_ai_clean_sub9': cand(c9_mask),
        'c_ai_clean_laps_valid': cand(c_mask & valid),
        'd_ai_near_collision': cand(d_mask),
        'ai_all': cand(ai), 'ai_valid': cand(ai & valid),
        'ai_other': cand(ai & ~c_mask & ~d_mask),
        'human_clean_laps': cand(hl_mask),
    }
    rec['in_base'] = {'train': int(base_tr.sum()), 'val': int(base_va.sum())}

    # ---- シーンごと
    for row in srows:
        s = row['scene']
        if s not in lab_scenes:
            row['labelled'] = False
            continue
        w = sc_name == s
        row['labelled'] = True
        row['frames'] = int(w.sum())
        row['ai_s'] = ssum(ai & w)
        row['human_s'] = ssum(hum & w)
        row['dry_s'] = ssum(dryf & w)
        row['throttle_gain'] = gain_summary(gains[:, 1], ai & w)
        row['steer_gain'] = gain_summary(gains[:, 0], ai & w)
        tw0, tw1 = (float(t[w][0]), float(t[w][-1]) + 1 / 30.0) if w.any() else (0, 0)
        inw = lambda x: tw0 <= x < tw1  # noqa: E731
        row['collisions_ai'] = sum(1 for c in colls if c.get('ai') and inw(c['t_s']))
        row['collisions_human'] = sum(1 for c in colls if not c.get('ai') and inw(c['t_s']))
        row['bumps'] = sum(1 for x in bumps if inw(x['t_s']))
        row['takeovers'] = sum(1 for x in tks if inw(x['t_s']))
        sl = [x for x in laps if x['scene'] == s]
        row['laps_ai'] = sum(1 for x in sl if x['who'] == 'ai')
        row['laps_ai_clean_s'] = sorted(x['lap_s'] for x in sl if x['who'] == 'ai' and x['clean'])
        row['laps_human'] = sum(1 for x in sl if x['who'] == 'human')
        row['laps_human_clean_s'] = sorted(x['lap_s'] for x in sl if x['who'] == 'human' and x['clean'])
        row['straight_erpm_p90_ai'] = r(pct(-erpm[straight & ai & w], 90, 30), 0)
        row['straight_erpm_p90_human'] = r(pct(-erpm[straight & hum & w], 90, 30), 0)
        row['candidates'] = {'a': cand(a_mask & w), 'b': cand(b_mask & w), 'c': cand(c_mask & w),
                             'c_sub9': cand(c9_mask & w), 'd': cand(d_mask & w)}
    for row in srows:
        row.pop('_segs', None)
        row.pop('_start_us', None)

    # 分布のためのフレームの表(間引き 3 のフレームだけ)
    samp = (idx_in % STRIDE == 0) & ondisk
    band_f = np.full(n, rec['band'], dtype=object)
    frames = {'band': band_f[samp], 'bright': bf[samp], 'top': bft[samp], 'ai': ai[samp], 'hum': hum[samp],
              'a': a_mask[samp], 'c': c_mask[samp], 'straight_ai': (straight & ai)[samp],
              'straight_hum': (straight & hum)[samp], 'erpm': erpm[samp], 'tg': gains[samp, 1],
              'model': np.array([name2row.get(s, {}).get('model') or '' for s in sc_name])[samp],
              'mode': mode[samp], 'session': np.full(int(samp.sum()), rec.get('session') or '', dtype=object)}
    del L
    return rec, frames


# ---------------------------------------------------------------- まとめ
def add(d, k, v):
    d[k] = d.get(k, 0) + (v or 0)


def summarize(recs):
    tot = {}
    for day in DAYS:
        R = [x for x in recs if x['day'] == day]
        U = [x for x in R if x['usable']]
        s = {'logs': len(R), 'logs_ai': sum(1 for x in R if x['ai_log']), 'scenes': sum(x['n_scenes'] for x in R),
             'camera_frames': sum(x['camera_frames'] for x in R), 'unusable_logs': [x['log'] for x in R if not x['usable']]}
        for k in ('duration_s', 'ai_s', 'human_s', 'dry_s'):
            s[k] = r(sum(x.get(k) or 0 for x in U), 1)
        s['frames'] = sum(x.get('frames') or 0 for x in U)
        for k in ('ai', 'human', 'strong', 'bumps'):
            s['collisions_' + k] = sum(x['collisions'][k] for x in U)
        s['takeovers'] = sum(x['takeovers']['all'] for x in U)
        for k in ('ai', 'ai_clean', 'ai_sub9', 'human', 'human_clean'):
            s['laps_' + k] = sum(x['laps'][k] for x in U)
        hc = {}
        for x in U:
            for k, v in x['human_categories_s'].items():
                add(hc, k, v)
        s['human_categories_s'] = {k: r(v, 1) for k, v in hc.items()}
        cs = {}
        for x in U:
            for k, v in x['candidates'].items():
                if k == 'b_by_class':
                    for kk, vv in v.items():
                        c = cs.setdefault('b_by_class', {}).setdefault(kk, {'takeovers': 0, 'frames': 0})
                        c['takeovers'] += vv['takeovers']
                        c['frames'] += vv['frames']
                    continue
                c = cs.setdefault(k, {'frames': 0, 'samples': 0, 'seconds': 0.0})
                c['frames'] += v['frames']
                c['samples'] += v['samples']
                c['seconds'] = r(c['seconds'] + (v['seconds'] or 0), 1)
        s['candidates'] = cs
        tot[day] = s
    return tot


def distributions(recs, frames_by_log, modes):
    out = {'brightness': [], 'straight': [], 'straight_by_gain': [], 'straight_human_by_genre': []}
    logs = [x for x in recs if x['usable'] and x['log'] in frames_by_log]
    F = [frames_by_log[x['log']] for x in logs]

    def cat(k):
        return np.concatenate([f[k] for f in F])
    day_f = np.concatenate([np.full(len(f['bright']), x['day'], dtype=object) for x, f in zip(logs, F)])
    band, br, top, ai, a = cat('band'), cat('bright'), cat('top'), cat('ai'), cat('a')
    er, sa, sh, tg, mdl, mode, sess = cat('erpm'), cat('straight_ai'), cat('straight_hum'), cat('tg'), cat('model'), cat('mode'), cat('session')
    for day in DAYS:
        for b in BANDS:
            bm = (day_f == day) & (band == b)
            if not bm.any():
                continue
            row = {'day': DAY_NAMES[day], 'band': b, 'samples': int(bm.sum()),
                   'logs': sum(1 for x in logs if x['day'] == day and x['band'] == b)}
            for name, m in (('all', bm), ('ai', bm & ai), ('human_drive', bm & a)):
                row[name] = {'n': int(m.sum()), 'p10': r(pct(br[m], 10)), 'p50': r(pct(br[m], 50)),
                             'p90': r(pct(br[m], 90)), 'top_p50': r(pct(top[m], 50))}
            out['brightness'].append(row)
            srow = {'day': DAY_NAMES[day], 'band': b}
            for name, m in (('ai', bm & sa), ('human', bm & sh)):
                srow[name] = {'n': int(m.sum()), 'p50': r(pct(-er[m], 50, 30), 0), 'p90': r(pct(-er[m], 90, 30), 0)}
            lp = [x['straight_erpm_p90']['ai'] for x in logs if x['day'] == day and x['band'] == b and x['straight_erpm_p90']['ai']]
            srow['ai_log_p90_range'] = [min(lp), max(lp)] if lp else None
            out['straight'].append(srow)
            for g_id in sorted(set(mode[bm & sh].tolist())):
                m = bm & sh & (mode == g_id)
                if m.sum() >= 30:
                    out['straight_human_by_genre'].append({'day': DAY_NAMES[day], 'band': b, 'genre': modes.get(int(g_id), str(g_id)),
                                                           'n': int(m.sum()), 'p50': r(pct(-er[m], 50), 0), 'p90': r(pct(-er[m], 90), 0)})
    tgr = np.round(tg.astype(float), 2)
    for se in sorted(set(sess[sa].tolist())):
        for model in sorted(set(mdl[sa & (sess == se)].tolist())):
            for g in sorted(set(tgr[sa & (sess == se) & (mdl == model)].tolist())):
                m = sa & (sess == se) & (mdl == model) & (tgr == g)
                if m.sum() < 30:
                    continue
                b0 = band[m][0]
                out['straight_by_gain'].append({'session': se, 'band': b0, 'model': model, 'throttle_gain': float(g),
                                                'n': int(m.sum()), 'p50': r(pct(-er[m], 50), 0), 'p90': r(pct(-er[m], 90), 0)})
    return out


def model_gain_table(recs):
    rows = {}
    for x in recs:
        if not x['usable']:
            continue
        for lp in x['lap_list']:
            if lp['who'] != 'ai':
                continue
            k = (x['date'], lp['model'], lp.get('throttle_gain'), lp.get('steer_gain'))
            d = rows.setdefault(k, {'logs': set(), 'laps': 0, 'clean': [], 'frames_clean': 0, 'samples_clean': 0,
                                    'sub9_frames': 0, 'sub9_samples': 0})
            d['logs'].add(x['ai_no'] if x['ai_no'] is not None else ('run%s' % x['run_no'] if x.get('run_no') else x['log'][9:]))
            d['laps'] += 1
            if lp['clean']:
                d['clean'].append(lp['lap_s'])
                d['frames_clean'] += lp['frames']
                d['samples_clean'] += lp['samples']
                if lp['lap_s'] < SUB9_S:
                    d['sub9_frames'] += lp['frames']
                    d['sub9_samples'] += lp['samples']
    out = []
    for (date, model, tg, sg), d in sorted(rows.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2] or 0, kv[0][3] or 0)):
        c = sorted(d['clean'])
        out.append({'date': date, 'model': model, 'throttle_gain': tg, 'steer_gain': sg,
                    'logs': sorted(d['logs'], key=str), 'laps': d['laps'], 'clean': len(c),
                    'sub9': sum(1 for v in c if v < SUB9_S), 'best': c[0] if c else None,
                    'median': r(np.median(c), 2) if c else None, 'frames_clean': d['frames_clean'],
                    'samples_clean': d['samples_clean'], 'sub9_frames': d['sub9_frames'], 'sub9_samples': d['sub9_samples']})
    return out


def checks(recs):
    """(1) clean_driving_20261011 の学習のサンプル数の再現 (2) AI の周を今までの成績表と比べる"""
    out = {}
    root = DATASETS / BASE_DS
    sp = read_json(root / 'splits.json') or {}
    n_tr = n_tr_mask = 0
    for s in sp.get('train', []):
        try:
            with np.load(root / 'aligned' / (s + '.npz'), allow_pickle=False) as z:
                ok = z['future_valid'] & z['vehicle_valid']
                if 'mode' in z.files:
                    ok &= z['mode'] != 0
                # 学習の --mask-impacts(train_data.impact_mask と同じ: 直前 10 サンプルの水平の加速度の最大 ≥ 2.5G の 1 秒前〜2 秒後)
                tt = z['timestamps_us'].astype(np.float64) / 1e6
                acc = z['imu'][:, :, :2].astype(np.float32)
                hg = np.hypot(acc[..., 0], acc[..., 1]).max(axis=1) / 9.80665
                im = np.zeros(len(tt), bool)
                for ti in tt[hg >= 2.5]:
                    im |= (tt >= ti - 1.0) & (tt <= ti + 2.0)
                st = (np.arange(len(ok)) % STRIDE) == 0
                n_tr += int((ok & st).sum())
                n_tr_mask += int((ok & ~im & st).sum())
        except Exception:
            pass
    lab_tr = sum(x['in_base']['train'] for x in recs if x['usable'])
    out['base_train_samples_aligned'] = n_tr
    out['base_train_samples_aligned_mask_impacts'] = n_tr_mask
    out['base_train_frames_in_labels'] = lab_tr
    out['base_train_a_nc_samples'] = sum(x['candidates']['a_nc_in_base_train']['samples'] for x in recs if x['usable'])
    cmp_ = {}
    for day, p in SC_REF.items():
        sc = read_json(p)
        if not sc:
            continue
        same = diff = 0
        ex = []
        for x in recs:
            if x['day'] != day or not x['usable'] or x['log'] not in sc.get('per_log', {}):
                continue
            ref = [(round(lp['t_start'], 2), round(lp['lap_s'], 2), lp['clean']) for lp in sc['per_log'][x['log']].get('laps', [])]
            mine = [(round(lp['t_start'], 2), round(lp['lap_s'], 2), lp['clean_raw']) for lp in x['lap_list'] if lp['who'] == 'ai']
            if ref == mine:
                same += 1
            else:
                diff += 1
                ex.append({'log': x['log'], 'ref': len(ref), 'mine': len(mine)})
        cmp_[DAY_NAMES[day]] = {'scorecard': str(p), 'same_logs': same, 'different_logs': diff, 'examples': ex[:5]}
    out['ai_laps_vs_scorecard'] = cmp_
    return out


# ---------------------------------------------------------------- Markdown
def fmt(x, f='%.0f'):
    return '—' if x is None else (f % x)


def k_(n):
    return '{:,}'.format(int(n or 0))


def by_session(recs):
    out = []
    names = []
    for x in recs:
        if x.get('session') and x['session'] not in names:
            names.append(x['session'])
    for se in names:
        R = [x for x in recs if x.get('session') == se]
        U = [x for x in R if x['usable']]
        out.append({'session': se, 'band': '・'.join(sorted({x['band'] for x in R}, key=BANDS.index)),
                    'logs': len(R), 'ai_logs': sum(1 for x in R if x['ai_log']),
                    'human_s': r(sum(x['human_s'] or 0 for x in U), 1), 'ai_s': r(sum(x['ai_s'] or 0 for x in U), 1),
                    'a_samples': sum(x['candidates']['a_human_drive']['samples'] for x in U),
                    'c_samples': sum(x['candidates']['c_ai_clean_laps']['samples'] for x in U),
                    'ai_laps': sum(x['laps']['ai'] for x in U), 'ai_clean': sum(x['laps']['ai_clean'] for x in U),
                    'ai_sub9': sum(x['laps']['ai_sub9'] for x in U),
                    'collisions_ai': sum(x['collisions']['ai'] for x in U),
                    'battery': sorted({x['battery_note'] for x in R if x.get('battery_note')}),
                    'brightness_p50': r(np.median([x['brightness']['p50'] for x in R if x['brightness']['p50'] is not None]), 1),
                    'top_p50': r(np.median([x['brightness']['top_p50'] for x in R if x['brightness']['top_p50'] is not None]), 1)})
    return out


def md(res):
    T = res['totals']
    R = res['records']
    C = {d: T[d]['candidates'] for d in DAYS}
    ck = res['checks']
    L = []
    w = L.append

    def cs(key, days=DAYS, f='samples'):
        return sum(C[d].get(key, {}).get(f, 0) for d in days)
    w('# 実車の記録の棚卸し(10/10・10/23・10/24)')
    w('')
    w('2026-10-25 作成。読むだけ(データセットは作っていない・学習していない・data/ は変えていない)。')
    w('台本: `/home/jetson/jetformer/experiments/20261025_dataset_v2/inventory.py`。シーンごと・周ごとの数は `inventory.json`(records[].scenes・records[].lap_list)。')
    w('')
    allf = sum(T[d]['camera_frames'] for d in DAYS)
    base_nc = ck['base_train_a_nc_samples']
    new_nc_track = cs('a_nc_new_track')
    new_nc = cs('a_nc_new')
    hum2324 = cs('a_human_drive', ('20261023', '20261024'), 'seconds')
    c_all, c9 = cs('c_ai_clean_laps'), cs('c_ai_clean_sub9')
    susp = [v for x in R if x['usable'] for v in x['laps']['suspect']]
    susp_clean = sorted(lp['lap_s'] for x in R if x['usable'] for lp in x['lap_list'] if lp['suspect'] and lp['clean_raw'])
    susp_ai = sum(1 for x in R if x['usable'] for lp in x['lap_list'] if lp['suspect'] and lp['who'] == 'ai')
    w('## まとめ')
    w('')
    w('- 実車の記録は **%d 本・%d シーン・カメラ %s フレーム(%.1f 時間)**。10/10 が %d 本(%s フレーム)、10/23 が %d 本(%s)、10/24 が %d 本(%s)。'
      % (len(R), sum(T[d]['scenes'] for d in DAYS), k_(allf), allf / 30 / 3600, T['20261010']['logs'], k_(T['20261010']['camera_frames']),
         T['20261023']['logs'], k_(T['20261023']['camera_frames']), T['20261024']['logs'], k_(T['20261024']['camera_frames'])))
    w('- 使えない記録は %s だけ(log.json・npz が 0 バイト。カメラの画像だけ)。' % ('・'.join(sum((T[d]['unusable_logs'] for d in DAYS), [])) or 'なし'))
    w('- **人の走行はほとんど増えていない。** 10/23・10/24 の人の走行(止まる・押す・持ち上げを除く)は合わせて %.0f 秒(%s サンプル)だけ。'
      '10/23〜24 の記録は AI 走行が中心(AI %.1f 分・人 %.1f 分で、人の時間の多くは止まっている・置き直し)。'
      % (hum2324, k_(cs('a_human_drive', ('20261023', '20261024'))), (T['20261023']['ai_s'] + T['20261024']['ai_s']) / 60,
         (T['20261023']['human_s'] + T['20261024']['human_s']) / 60))
    w('- (a) 人の走行: 全部で %s サンプル(間引き 3)。clean_driving_20261011 と重なる %s、新しい %s(うち 10/10 の駐車(1)〜(3) の 3 本が %s)。'
      % (k_(cs('a_human_drive')), k_(cs('a_in_base_train') + cs('a_in_base_val')), k_(cs('a_new')),
         k_(cs('a_new') - cs('a_new_track'))))
    w('- 同じ数え方(人の走行・衝突の前後を除く)で比べると: 今の学習 41 シーン = %s、新しい分(高速・低速のジャンル)= %s → **人の走行だけでは 約 %.2f 倍**。'
      '駐車も入れて 約 %.2f 倍。今の学習のサンプル 5,912 に当てはめると 約 %s。'
      % (k_(base_nc), k_(new_nc_track), (base_nc + new_nc_track) / max(base_nc, 1), (base_nc + new_nc) / max(base_nc, 1),
         k_(5912 * (base_nc + new_nc_track) / max(base_nc, 1))))
    short = max(base_nc - new_nc_track, 0)
    w('- 人の走行だけで 2 倍(10/15 の量の実験では 2 倍で L1 が 2〜4 µs 良くなる見込み)にするには、あと 約 %s サンプル'
      '= 前へ走っている時間で 約 %.1f 分(衝突の前後を除いた時間)の人の走行が要る。' % (k_(short), short * STRIDE / 30.0 / 60.0))
    w('- 量を 2 倍にできるのは AI の記録: (c) AI のきれいな周 = **%s サンプル(%d 周)**、うち 9 秒未満の周 %s サンプル(%d 周、10/23 %d・10/24 %d)。'
      '人の走行の新しい分と (c) を足すと 5,912 + %s + %s ≈ %s で 約 %.1f 倍(ただし (c) は AI の操作値。8 章)。'
      % (k_(c_all), sum(T[d]['laps_ai_clean'] for d in DAYS), k_(c9), sum(T[d]['laps_ai_sub9'] for d in DAYS),
         T['20261023']['laps_ai_sub9'], T['20261024']['laps_ai_sub9'], k_(new_nc_track), k_(c_all),
         k_(5912 + new_nc_track + c_all), (5912 + new_nc_track + c_all) / 5912))
    w('- (b) 交代の直後 3 秒: %s サンプル(交代 %d 回。前へ走っているのは %s)。(d) AI の衝突の前後(外す候補): %s サンプル。AI の全フレームは %s サンプル。'
      % (k_(cs('b_takeover_3s')), sum(T[d]['takeovers'] for d in DAYS), k_(cs('b_takeover_3s_drive')),
         k_(cs('d_ai_near_collision')), k_(cs('ai_all'))))
    p50s = [x['all']['p50'] for x in res['distributions']['brightness'] if x['all']['p50'] is not None]
    tops_ = [x['all']['top_p50'] for x in res['distributions']['brightness'] if x['all']['top_p50'] is not None]
    w('- 明るさ: カメラは自動露出(目標 110)・自動の白の調整なので、平均の明るさはどの日・時間帯も中央値 %.0f〜%.0f(3 章)。'
      '外の光の差は画像の上 1/3 に少しだけ出る(中央値 %.0f〜%.0f)。露出・ゲインの値は記録に無い。'
      % (min(p50s), max(p50s), min(tops_), max(tops_)))
    bb = [x for x in res['distributions']['straight_by_gain'] if x['model'] == 'B' and abs(x['throttle_gain'] - 0.9) < 1e-6]
    if bb:
        w('- 直線の最高速(電池の代わり): B・速さ 0.90 の直線の eRPM の 90% 点は ' + '、'.join('%s %s' % (x['session'], k_(x['p90'])) for x in bb)
          + ('(%s が %.0f%% 低い。満充電 → 8.49V。10/24 のまとめでは 3 回目は後輪の修理の後)' % (bb[-1]['session'][:10], 100 * (1 - bb[-1]['p90'] / bb[0]['p90']))
             if len(bb) >= 2 and bb[0]['p90'] else '')
          + '。電圧の記録は説明欄の「満充電」「8.49V」だけ(4 章)。')
    if susp:
        w('- 7 秒未満の周 %d 本(AI %d・人 %d。%.2f〜%.2f 秒)は R1 の起点の取り違え(衝突で回ったときなど)とみなした。'
          'きれいな周の扱いだったのは %d 本(%s 秒)で、きれいな周・9 秒未満から外した。'
          % (len(susp), susp_ai, len(susp) - susp_ai, min(susp), max(susp), len(susp_clean),
             '・'.join('%.2f' % v for v in susp_clean) or '—'))
    w('')

    # 1
    w('## 1. 日ごとの量')
    w('')
    w('| 日 | 記録(AI 走行) | シーン | カメラのフレーム | 長さ 分 | 人 分 | AI 分 | dry run 分 | 衝突 AI / 人 | 交代 | AI の周(きれい・9 秒未満) | 人の周(きれい) |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for d in DAYS:
        s = T[d]
        w('| %s | %d(%d) | %d | %s | %.1f | %.1f | %.1f | %.1f | %d / %d | %d | %d(%d・%d) | %d(%d) |'
          % (DAY_NAMES[d], s['logs'], s['logs_ai'], s['scenes'], k_(s['camera_frames']), s['duration_s'] / 60,
             s['human_s'] / 60, s['ai_s'] / 60, s['dry_s'] / 60, s['collisions_ai'], s['collisions_human'], s['takeovers'],
             s['laps_ai'], s['laps_ai_clean'], s['laps_ai_sub9'], s['laps_human'], s['laps_human_clean']))
    w('')
    w('- 長さ = ディスクにあるシーンの長さ(scene.json)の和。人・AI = ラベルのフレームの dt の和。')
    w('- 人の周は R1 の起点が取れた周だけ(10/10 の人の記録は 20 秒のシーンの間が空くので、周がシーンをまたぐと数えない)。')
    w('')
    w('人の時間の内わけ(分):')
    w('')
    w('| 日 | ' + ' | '.join(CAT_NAMES[k] for k in CATS) + ' |')
    w('|---|' + '---|' * len(CATS))
    for d in DAYS:
        hc = T[d]['human_categories_s']
        w('| %s | ' % DAY_NAMES[d] + ' | '.join('%.1f' % ((hc.get(CAT_NAMES[k]) or 0) / 60) for k in CATS) + ' |')
    w('')
    w('回ごと(記録の始まりが 60 分を超えて空いたら次の回):')
    w('')
    w('| 回 | 時間帯 | 記録(AI) | 人 分 | AI 分 | (a) サンプル | (c) サンプル | AI の周(きれい・9 秒未満) | AI の衝突 | 明るさ・上 1/3 | 電池(説明欄) |')
    w('|---|---|---|---|---|---|---|---|---|---|---|')
    for x in res['by_session']:
        w('| %s | %s | %d(%d) | %.1f | %.1f | %s | %s | %d(%d・%d) | %d | %s・%s | %s |'
          % (x['session'], x['band'], x['logs'], x['ai_logs'], (x['human_s'] or 0) / 60, (x['ai_s'] or 0) / 60, k_(x['a_samples']),
             k_(x['c_samples']), x['ai_laps'], x['ai_clean'], x['ai_sub9'], x['collisions_ai'], fmt(x['brightness_p50']),
             fmt(x['top_p50']), '・'.join(x['battery']) or '—'))
    w('')

    # 2
    w('## 2. 学習に使える候補の量')
    w('')
    w('フレーム数 / 間引き 3 のサンプル数。')
    w('')
    w('| 候補 | 10/10 | 10/23 | 10/24 | 合計 |')
    w('|---|---|---|---|---|')

    def crow(label, key):
        vals = [C[d].get(key, {'frames': 0, 'samples': 0}) for d in DAYS]
        w('| %s | ' % label + ' | '.join('%s / %s' % (k_(v['frames']), k_(v['samples'])) for v in vals)
          + ' | **%s / %s** |' % (k_(sum(v['frames'] for v in vals)), k_(sum(v['samples'] for v in vals))))
    crow('(a) 人の走行', 'a_human_drive')
    crow('　うち clean_driving の学習 41 シーン', 'a_in_base_train')
    crow('　うち clean_driving の検証 11 シーン', 'a_in_base_val')
    crow('　**うち新しい分**', 'a_new')
    crow('　　新しい分のうち 高速・低速のジャンル', 'a_new_track')
    crow('　うち交代の直後 3 秒の中', 'a_after_takeover')
    crow('(a) から衝突の前後(1 秒前〜2 秒後)を除く', 'a_human_drive_no_coll')
    crow('　うち clean_driving の学習 41 シーン', 'a_nc_in_base_train')
    crow('　うち新しい分', 'a_nc_new')
    crow('　　新しい分のうち 高速・低速のジャンル', 'a_nc_new_track')
    crow('(参考)人側の全フレーム(止まる・押すも入れる)', 'human_all')
    crow('(b) 交代の直後 3 秒(人側)', 'b_takeover_3s')
    crow('　うち前へ走っている', 'b_takeover_3s_drive')
    crow('(c) AI のきれいな周', 'c_ai_clean_laps')
    crow('　うち 9 秒未満の周', 'c_ai_clean_sub9')
    crow('(d) AI の衝突の前後(外す候補)', 'd_ai_near_collision')
    crow('(参考)AI の全フレーム', 'ai_all')
    crow('(参考)AI のそのほか(c・d 以外)', 'ai_other')
    crow('(参考)人のきれいな周', 'human_clean_laps')
    w('')
    w('- clean_driving_20261011 と重なるのは 10/10 の人の記録 9 本(52 シーン)だけ。10/23・10/24 の記録はどのデータセットにも入っていない。')
    w('- 10/10 の新しい分 = AI 走行の記録(11 本)の中の人の区間と、ジャンル「駐車(1)〜(3)」の人の記録 3 本(192141・192654・193130。'
      '遅い走り: 直線の 90% 点 約 3,700〜3,900 eRPM、R1 の起点が取れない)。')
    w('- (b)・(a) の重なり: (a) のうち交代の直後 3 秒の中にあるものは上の表の「うち交代の直後 3 秒の中」。')
    w('')
    w('(b) 交代の分類(交代の回数 / 3 秒のフレーム):')
    w('')
    w('| 日 | ' + ' | '.join(le.TAKEOVER_CLASSES) + ' |')
    w('|---|' + '---|' * len(le.TAKEOVER_CLASSES))
    for d in DAYS:
        bc = C[d].get('b_by_class', {})
        w('| %s | ' % DAY_NAMES[d] + ' | '.join('%d / %s' % (bc.get(k, {}).get('takeovers', 0), k_(bc.get(k, {}).get('frames', 0)))
                                                for k in le.TAKEOVER_CLASSES) + ' |')
    w('')
    w('(a) 人の走行のジャンル(間引き 3 のサンプル。ジャンルはシーンの印。AI 走行の記録の中の人の区間は、走らせたときのジャンル):')
    w('')
    gen = {}
    for x in R:
        if not x['usable']:
            continue
        for sc in x['scenes']:
            if not sc.get('labelled'):
                continue
            g = '・'.join(sc.get('genres') or ['?'])
            in_base = any(dd == BASE_DS for dd, _ in sc.get('datasets', []))
            k = (x['date'], g, 'AI 走行の記録' if x['ai_log'] else '人の記録', '重なる' if in_base else '新しい')
            add(gen, k, sc['candidates']['a']['samples'])
    w('| 日 | ジャンル | 記録 | clean_driving と | (a) サンプル |')
    w('|---|---|---|---|---|')
    for (dd, g, kind, ov), v in sorted(gen.items()):
        if v:
            w('| %s | %s | %s | %s | %s |' % (dd, g, kind, ov, k_(v)))
    w('')
    w('### AI のきれいな周(モデル・速さごと)')
    w('')
    w('| 日 | モデル | 速さ | ステア | 記録(No.) | AI の周 | きれい | 9 秒未満 | ベスト | 中央値 | きれいな周のサンプル | 9 秒未満のサンプル |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for x in res['ai_laps_by_model']:
        w('| %s | %s | %s | %s | %s | %d | %d | %d | %s | %s | %s | %s |'
          % (x['date'], x['model'], fmt(x['throttle_gain'], '%.2f'), fmt(x['steer_gain'], '%.2f'),
             '・'.join(str(v) for v in x['logs']), x['laps'], x['clean'], x['sub9'], fmt(x['best'], '%.2f'),
             fmt(x['median'], '%.2f'), k_(x['samples_clean']), k_(x['sub9_samples'])))
    w('')
    w('- 速さ・ステア = 周の中の AI のフレームのゲインの中央値(走行中に変えたものは周ごとに分かれる)。No. = 説明欄の ai_no(10/10 は ai_run の番号)。')
    w('- 10/10 の「最初のモデル」= first_trial_20261010-s3(TransFuser軽量型)。10/23・10/24 のモデルは 8 章。')
    w('')

    # 3
    D = res['distributions']
    w('## 3. 明るさ(カメラの平均の明るさ)の分布')
    w('')
    w('灰色の平均(0〜255)。間引き 3 のフレーム。上 1/3 = 画像の上の 1/3 の平均(壁・窓・天井の光が入るところ)。')
    w('')
    w('| 日 | 時間帯 | 記録 | サンプル | 全部 10% / 50% / 90% | 上 1/3 の中央値 | AI の中央値 | 人の走行の中央値 |')
    w('|---|---|---|---|---|---|---|---|')
    for x in D['brightness']:
        a = x['all']
        w('| %s | %s | %d | %s | %s / %s / %s | %s | %s | %s |'
          % (x['day'], x['band'], x['logs'], k_(x['samples']), fmt(a['p10']), fmt(a['p50']), fmt(a['p90']), fmt(a['top_p50']),
             fmt(x['ai']['p50']), fmt(x['human_drive']['p50'])))
    w('')
    w('- 自動露出が平均を目標 110 に寄せるので、時間帯の差は平均にはほとんど出ない。')
    tops = {}
    for x in D['brightness']:
        tops.setdefault(x['band'], []).append(x['all']['top_p50'])
    w('- 上 1/3 の中央値: ' + '・'.join('%s %s' % (b_, '〜'.join(fmt(v) for v in sorted({min(vs), max(vs)})))
                                    for b_, vs in ((b_, [v for v in tops.get(b_, []) if v is not None]) for b_ in BANDS) if vs)
      + '。外の光の差は平均よりここに出る。')
    w('- 時間帯でデータを分ける理由は平均の明るさには無い。暗いときのノイズ・光の向きの違いは平均では見えない(露出・ゲインは記録が無い)。')
    w('')

    # 4
    w('## 4. 直線の最高速(eRPM の 90% 点)の分布')
    w('')
    w('前へ走っている(eRPM < −1000)・まっすぐ(|ヨー| < 40 deg/s)のフレームの前進 eRPM。0.5e-3 を掛けると m/s(8,000 eRPM = 4.0 m/s)。')
    w('')
    w('| 日 | 時間帯 | AI のフレーム | AI 50% / 90% | 記録ごとの AI の 90% 点 | 人のフレーム | 人 50% / 90% |')
    w('|---|---|---|---|---|---|---|')
    for x in D['straight']:
        rg = x['ai_log_p90_range']
        w('| %s | %s | %s | %s / %s | %s | %s | %s / %s |'
          % (x['day'], x['band'], k_(x['ai']['n']), fmt(x['ai']['p50']), fmt(x['ai']['p90']),
             '%s〜%s' % (k_(rg[0]), k_(rg[1])) if rg else '—', k_(x['human']['n']), fmt(x['human']['p50']), fmt(x['human']['p90'])))
    w('')
    w('人(ジャンルごと):')
    w('')
    w('| 日 | 時間帯 | ジャンル | フレーム | 50% | 90% |')
    w('|---|---|---|---|---|---|')
    for x in D['straight_human_by_genre']:
        w('| %s | %s | %s | %s | %s | %s |' % (x['day'], x['band'], x['genre'], k_(x['n']), k_(x['p50']), k_(x['p90'])))
    w('')
    w('AI(回・モデル・速さごと。同じモデル・同じ速さで比べると電池の差が見える):')
    w('')
    w('| 回 | 時間帯 | モデル | 速さ | フレーム | 50% | 90% |')
    w('|---|---|---|---|---|---|---|')
    for x in D['straight_by_gain']:
        w('| %s | %s | %s | %.2f | %s | %s | %s |' % (x['session'], x['band'], x['model'], x['throttle_gain'], k_(x['n']),
                                                   k_(x['p50']), k_(x['p90'])))
    w('')

    # 5
    w('## 5. 記録ごと')
    w('')
    w('No. = 説明欄の ai_no(10/10 は ai_run の番号)。長さ・人・AI・dry は秒。速さ・ステア = AI のフレームのゲインの中央値。'
      '衝突 = AI / 人(ラベル。水平 2.5G)。周 = AI / 人。きれいな周 = AI(人)。直線 = eRPM の 90% 点(AI / 人)。明るさ = 中央値。(a)・(c) = 間引き 3 のサンプル。')
    w('')
    w('| 日 | No. | 記録 | 時刻 | 帯 | シーン | 長さ | 人 | AI | dry | モデル | 速さ | ステア | ジャンル | 衝突 | 周 | きれいな周 | きれいな周のラップ | 直線 | 明るさ | (a) | (c) | 説明欄 |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for x in R:
        no = x['ai_no'] if x['ai_no'] is not None else (x.get('run_no') or '')
        if not x['usable']:
            w('| %s | %s | %s | %s | %s | %d | — | — | — | — | — | — | — | — | — | — | — | — | — | %s | — | — | 壊れた記録(log.json・npz が空。画像 %s 枚) |'
              % (x['date'], no, x['log'][9:], x['start'][:5], x['band'], x['n_scenes'], fmt(x['brightness']['p50']), k_(x['camera_frames'])))
            continue
        lp = x['laps']
        cl = lp['ai_clean_s']
        lap_txt = ('%.2f〜%.2f' % (cl[0], cl[-1]) if len(cl) > 1 else ('%.2f' % cl[0] if cl else '—'))
        if lp['human_clean_s']:
            hc = lp['human_clean_s']
            lap_txt += '(人 %.2f〜%.2f)' % (hc[0], hc[-1]) if len(hc) > 1 else '(人 %.2f)' % hc[0]
        gen = '・'.join(k for k, v in x['genres_s'].items() if v and v > 1)
        st = x['straight_erpm_p90']
        w('| %s | %s | %s | %s | %s | %d | %.0f | %.0f | %.0f | %s | %s | %s | %s | %s | %d / %d | %d / %d | %d(%d) | %s | %s / %s | %s | %s | %s | %s |'
          % (x['date'], no, x['log'][9:], x['start'][:5], x['band'], x['n_scenes'], x['duration_s'] or 0,
             x['human_s'] or 0, x['ai_s'] or 0, '%.0f' % x['dry_s'] if x['dry_s'] else '—', '・'.join(x['models']) or '人',
             fmt((x['throttle_gain'] or {}).get('median'), '%.2f'), fmt((x['steer_gain'] or {}).get('median'), '%.2f'), gen,
             x['collisions']['ai'], x['collisions']['human'], lp['ai'], lp['human'], lp['ai_clean'], lp['human_clean'], lap_txt,
             fmt(st['ai']), fmt(st['human']), fmt(x['brightness']['p50']), k_(x['candidates']['a_human_drive']['samples']),
             k_(x['candidates']['c_ai_clean_laps']['samples']), (x['desc_short'] or '').replace('|', '/')))
    w('')

    # 6
    w('## 6. 数え方')
    w('')
    for s_ in res['rules']:
        w('- ' + s_)
    w('')

    # 7
    w('## 7. 確かめたこと')
    w('')
    w('- clean_driving_20261011 の学習 41 シーンを aligned から数え直した: 間引き 3 で %s、学習と同じ衝突の前後の除外(2.5G・1 秒前〜2 秒後)をすると **%s**'
      '(モデル B の result.json の train_samples 5,912 と比べる)。同じシーンのこの台本の (a) から衝突の前後を除いた数は %s(止まっている・押しているも除くので少ない)。'
      % (k_(ck['base_train_samples_aligned']), k_(ck['base_train_samples_aligned_mask_impacts']), k_(ck['base_train_a_nc_samples'])))
    for d, v in ck['ai_laps_vs_scorecard'].items():
        w('- %s の AI の周(始まり・ラップ・きれいか)を今までの成績表(%s)と比べた: 同じ %d 本・違う %d 本%s。'
          % (d, v['scorecard'].replace('/home/jetson/', '~/'), v['same_logs'], v['different_logs'],
             ('(%s)' % '・'.join('%s: 成績表 %d 周 / ここ %d 周' % (e['log'], e['ref'], e['mine']) for e in v['examples'])) if v['examples'] else ''))
    n9 = next((x for x in R if x['log'] == '20261024-162033' and x['usable']), None)
    if n9 and n9['laps']['ai_clean_s']:
        c9_ = n9['laps']['ai_clean_s']
        w('- 10/24 No.9(162033)のきれいな周は %d 周・%.2f〜%.2f 秒、9 秒未満 %d 周(10/24 のまとめのベスト 8.25 と同じ)。'
          % (len(c9_), c9_[0], c9_[-1], n9['laps']['ai_sub9']))
    w('')

    # 8
    w('## 8. 気づいたこと・注意')
    w('')
    for s_ in res['notes']:
        w('- ' + s_)
    w('')
    return '\n'.join(L)


RULES = [
    'フレーム = ラベル(data/labels/<記録>.npz)のフレーム。AI 走行の記録は AI_CMD、人だけの記録はカメラ(30 fps)。ファイルが壊れたシーンはラベルに入っていない。',
    'AI の秒 = CH3 が AI 側・dry run でないフレームの dt の和。人の秒 = CH3 が人側。dry run = AI 側で出力が中立のまま。',
    '時間帯 = 記録の始まりの時刻(日本時間): 昼 〜16:59(日の入りの前)・夕方 17:00〜18:59・夜 19:00〜(10/24 の 18:31〜 はレポートでも「夕方」)。回 = 記録の始まりの間が 60 分を超えたら次の回。',
    '明るさ = カメラの画像の灰色の平均(0〜255)。3 枚に 1 枚を読む。上 1/3 の平均も。',
    '直線の最高速 = 前へ走っている(eRPM < −1000)・まっすぐ(ヨーの 11 サンプル平均 |gz| < 40 deg/s)のフレームの前進 eRPM の 90% 点(30 フレーム未満は出さない)。',
    '周 = R1 の起点 → 次の起点(5〜16 秒・ずっと同じ操作者・0.2 秒を超える途切れなし)。tools/section_scorecard.py のラップと同じ。',
    'AI のきれいな周 = その周に AI の衝突(水平 2.5G)・交代・リカバリー・基板の再起動が無い(section_scorecard と同じ)。人のきれいな周 = 衝突・交代・再起動が無い。',
    '(a) 人の走行 = CH3 が人側・操作値が有効(valid_action: スティックを受信中・交代/戻しの前後 0.3 秒と基板の再起動の前後でない)・前へ走っている(eRPM < −200)・スロットルを 1 秒以内に入れている(中立から 20µs 以上。押している・惰性を除く)・持ち上げていない(1 秒の平均で傾き > 30° かつ |ヨー| < 40 deg/s を除く)。',
    '(b) 交代の直後 = AI → 人の交代(ラベルの takeovers)から 3 秒の人側のフレーム。',
    '(c) AI のきれいな周 = 上のきれいな周のフレーム。9 秒未満の周は別にも数える。',
    '(d) AI の衝突の前後 = AI 側のフレームで、衝突の 1 秒前〜2 秒後(ラベルの near_collision。学習の --mask-impacts と同じ幅)。',
    'サンプル = 間引き 3(シーンの中のフレーム番号 % 3 == 0。jetformer/train_data.py と同じ)。学習では未来の操作値が揃わないシーンの終わりの約 10 フレームも外れる。',
    'clean_driving_20261011 と重なる = そのデータセットの source_scenes にあるシーン(学習 41・検証 11)。',
]


CAND_MODEL = {'B': 'B', 'SLOW-SEC1': '遅く入る型', 'ENS': '2本の平均', 'AUG': 'AUG', 'WGT': 'WGT', 'ACT': 'ACT型',
              'TCP': 'TCP型', 'DINOV2': 'DINOv2型'}


def no_of(x):
    if x.get('ai_no') is not None:
        return 'No.%s' % x['ai_no']
    if x.get('run_no'):
        return 'run%s' % x['run_no']
    return x['log'][9:]


def notes(res):
    T = res['totals']
    R = res['records']
    U = [x for x in R if x['usable']]
    out = []
    bad = sum((T[d]['unusable_logs'] for d in DAYS), [])
    if bad:
        out.append('壊れた記録 %s: log.json・npz・scene.json が 0 バイトで、画像だけ残っている。操作値が無いので学習には使えない。' % '・'.join(bad))
    deleted = [(x['log'], x['frames_not_on_disk']) for x in U if (x.get('frames_not_on_disk') or 0) > 10]
    if deleted:
        out.append('ラベルを作った後(10/11)にシーンを消した記録: %s。消したシーンのフレームは数に入れていない。'
                   % '・'.join('%s(%d フレーム)' % d for d in deleted))
    broken = sorted({x['log'] for x in U for p in x['problems'] if '読めません' in p or 'ありません' in p})
    if broken:
        out.append('一部のシーンのファイルが壊れている・無い記録: %s。読めないシーンはラベルに無い(車両データが無いシーンは「操作値なし」)。'
                   % '・'.join(broken))
    mism = []
    for x in U:
        m = re.search(r'cand=(\S+)', x['description'] or '')
        if not m or m.group(1).upper() not in CAND_MODEL:
            continue
        want = CAND_MODEL[m.group(1).upper()]
        if x['models'] and want not in x['models']:
            mism.append('%s No.%s(説明欄 %s・実際 %s)' % (x['date'], x['ai_no'], m.group(1), '・'.join(x['models'])))
    if mism:
        out.append('説明欄の cand= と実際に走ったモデルが違う記録: %s。この表のモデルはシーンの drive.run。' % '、'.join(mism))
    inval = [(x, x['human_categories_s'].get('操作値なし') or 0) for x in U]
    inval = [(x, v) for x, v in inval if v >= 20]
    if inval:
        out.append('人側の「操作値なし」(スティック・車両データが無効)が 20 秒以上の記録: %s。'
                   % '、'.join('%s %s %.0f 秒' % (x['date'], no_of(x), v) for x, v in inval))
    dry = [x for x in U if x['dry_s']]
    if dry:
        out.append('dry run(AI 側で出力は中立)の記録: %s。dry run のフレームは AI にも人にも数えない。'
                   % '・'.join('%s(%.0f 秒)' % (x['log'], x['dry_s']) for x in dry))
    for d in ('20261023', '20261024'):
        hc = T[d]['human_categories_s']
        tot = sum(v or 0 for v in hc.values())
        top = sorted([x for x in U if x['day'] == d], key=lambda x: -x['candidates']['a_human_drive']['samples'])[:3]
        out.append('%s の人側の時間 %.1f 分のうち、走行は %.1f 分(%.0f%%)。止まっている %.1f 分・後退 %.1f 分。(a) が多い記録: %s。'
                   '(a) のうち交代の直後 3 秒の中は %s サンプル / %s。'
                   % (DAY_NAMES[d], tot / 60, (hc.get('走行') or 0) / 60, 100 * (hc.get('走行') or 0) / max(tot, 1),
                      (hc.get('停止') or 0) / 60, (hc.get('後退') or 0) / 60,
                      '・'.join('%s %s' % (no_of(x), k_(x['candidates']['a_human_drive']['samples'])) for x in top),
                      k_(T[d]['candidates']['a_after_takeover']['samples']), k_(T[d]['candidates']['a_human_drive']['samples'])))
    out.append('(c) の AI のきれいな周の操作値は、そのとき走ったモデル(B など)の出力をゲインで割り戻した値。そのまま足すとモデルの癖を強める'
               '(10/11 のオフライン強化学習の結論: AI の記録を足すと右寄りになった)。足すなら 9 秒未満の周だけ・重みを下げる・衝突の前後を外すなど。')
    out.append('ジャンル「駐車(1)〜(3)」(10/10 19:21〜19:31 の 3 本)は遅い走り(直線の 90% 点 約 3,700〜3,900 eRPM)。'
               'タイムアタックの学習では別のジャンルとして扱うか外す候補。')
    susp = [(x['log'], v) for x in U for v in x['laps']['suspect']]
    if susp:
        by_day = {}
        for lg, v in susp:
            by_day[lg[:8]] = by_day.get(lg[:8], 0) + 1
        out.append('7 秒未満の周(%s)は R1 の起点の取り違えとみなした(人のきれいな周のベストは %s 秒・上位チームは 1 周 約 7.7 秒)。'
                   '一覧は inventory.json の records[].laps.suspect。成績表(section_scorecard)はこれらも周として数えている。'
                   % ('・'.join('%s %d 本' % (DAY_NAMES[d], c) for d, c in sorted(by_day.items())),
                      fmt(min([v for x in U for v in x['laps']['human_clean_s']] or [None]), '%.2f')))
    diff = [x for x in U if x['laps']['lapcounter'].get('clean_laps') is not None and x['laps']['ai']
            and x['laps']['lapcounter']['clean_laps'] != x['laps']['ai_clean']]
    if diff:
        out.append('ラベルの lap_summary(走行画面と同じ LapCounter。ジャイロで 360° ごと)のきれいな周の数は、ここ(R1 の起点ごと)と %d 本で違う'
                   '(例: 10/24 No.9 は LapCounter 8 周・ここ 9 周)。この棚卸しは R1 の起点の周(成績表と同じ)を使う。' % len(diff))
    for d in DAYS:
        L = sorted(x['start'][:5] for x in R if x['day'] == d)
        out.append('%s の記録の時刻: %s〜%s(記録の始まり)。' % (DAY_NAMES[d], L[0], L[-1]))
    return out


# ---------------------------------------------------------------- 本体
def main():
    modes = mode_table()
    membership = dataset_membership()
    logs = sorted(d for d in os.listdir(RAW) if d[:8] in DAYS and (RAW / d).is_dir())
    sess = sessions(logs)
    recs, frames_by_log = [], {}
    for log in logs:
        rec, fr = process_log(log, membership, modes, sess.get(log))
        rec.pop('_bright', None)
        recs.append(rec)
        if fr is not None:
            frames_by_log[log] = fr
        print('%s %s 人 %s 秒 AI %s 秒 (a) %s (c) %s' % (log, '使える' if rec['usable'] else '壊れた', rec.get('human_s'),
                                                    rec.get('ai_s'), (rec.get('candidates') or {}).get('a_human_drive', {}).get('samples'),
                                                    (rec.get('candidates') or {}).get('c_ai_clean_laps', {}).get('samples')), flush=True)
    res = {'schema': 'jf-dataset-v2-inventory/v1', 'created_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
           'tool': 'experiments/20261025_dataset_v2/inventory.py', 'rules': RULES,
           'params': {'stride': STRIDE, 'takeover_window_s': TK_WINDOW_S, 'sub9_s': SUB9_S, 'fwd_erpm': FWD_ERPM,
                      'straight_erpm': STRAIGHT_ERPM, 'straight_gz_dps': STRAIGHT_GZ, 'thr_active_us': THR_ACTIVE_US,
                      'coast_s': COAST_S, 'lift_tilt_deg': LIFT_TILT_DEG, 'lift_gz_dps': LIFT_GZ, 'base_dataset': BASE_DS},
           'totals': None, 'records': recs}
    res['totals'] = summarize(recs)
    res['distributions'] = distributions(recs, frames_by_log, modes)
    res['ai_laps_by_model'] = model_gain_table(recs)
    res['by_session'] = by_session(recs)
    res['checks'] = checks(recs)
    res['notes'] = notes(res)
    OUT.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, set):
            return sorted(o, key=str)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(OUT / 'inventory.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=default)
    text = md(res)
    with open(OUT / 'inventory.md', 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print('書いた: %s・%s' % (OUT / 'inventory.json', OUT / 'inventory.md'))


if __name__ == '__main__':
    main()
