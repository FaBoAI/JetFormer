#!/usr/bin/env python3
"""B2(選び直した実機のデータで学習したモデル B)の机上の評価: 表(CPU)・crosscheck(GPU)・予測(GPU)・エンジン(GPU)・
評価(CPU)。図と文書は b2_report.py。

    cd ~/jetformer
    PY=/home/jetson/miniconda/envs/jetracer/bin/python
    CPU="/home/jetson/bin/jf-heavy --mem 1500M --lock none -- nice -n 10 env CUDA_VISIBLE_DEVICES="
    GPU="/home/jetson/bin/jf-heavy --mem 3G --lock big --need-mb 3200 -- nice -n 10"
    $CPU $PY -u experiments/20261016_b2_curation/eval/b2_eval.py tables       # 窓の表・ジャンルの表(数が protocol と違えば止める)
    $CPU $PY -u experiments/20261016_b2_curation/eval/b2_eval.py repro        # 既にある A・B の予測で、固定した基準値を作り直せるか
    $GPU $PY -u experiments/20261016_b2_curation/eval/b2_eval.py crosscheck   # B の best を予測し直して 10/12 の B.npz と比べる
    $GPU $PY -u experiments/20261016_b2_curation/eval/b2_eval.py predict      # 予測(出来たものは飛ばす)
    $GPU $PY -u experiments/20261016_b2_curation/eval/b2_eval.py engine       # B2 種 0 の FP16 エンジン・確認・走行ループの時間
    $PY experiments/20261016_b2_curation/eval/b2_eval.py freeze               # 道具の sha256 を eval_sha256.txt に書く(評価の前)
    $CPU $PY -u experiments/20261016_b2_curation/eval/b2_eval.py evaluate     # 判定 → evaluation.json

protocol.json(b2_curation_20261016)の evaluation の手順をそのまま実装したもの:
  - 予測は policy_zoo._Frames・_predict_frames と同じ計算(PyTorch FP32・eval・バッチ 32・workers 3)にジャンルの指定を足す
  - 保留の表 = rl_evaluate.pred_table()(key 3d339512…、10,985 行)。B2 は own_labels(付け替えた 2 シーンの行 = 4)。
    aligned は付け替えた 2 シーンの行だけ予測し、ほかは own_labels の値を写す。A・B は 10/12・10/15 の予測を読むだけ
  - 窓の表 = R1 の [ta − 2, ta]・DE の [tg, tg + 0.8](H・AI・T の通過)+ 直線 542 行(+ 参考の SW・HS の直線)。
    行の順を固定して sha1 の key を作り、予測の前に eval_sha256.txt に書く
  - 走行の出力 = rl_evaluate.drive_outputs(apply_gains と同じ式)、先読み 3・速度のゲイン 1.0・ステアはゲイン前(1.0)
GPU を使うのは crosscheck・predict・engine だけ(jf-heavy --lock big で 1 つずつ。管理画面の走行・学習が動いていたら始めない)。
電源が落ちても、同じコマンドをもう一度実行すれば続きから(予測は 1 本ずつ npz に書き、key が合うものは飛ばす)。
車は動かさない・PCA9685・I2C に触れない(走行ループの確認は偽の PCA9685)・drive.py と走行の既定は変えない。
"""

import argparse
import datetime
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = Path('/home/jetson/jetformer')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tools'))
sys.path.insert(0, str(HERE))
from jetformer import config  # noqa: E402
import policy_zoo as pz  # noqa: E402
import rl_evaluate as ev  # noqa: E402
import rl_policies as rp  # noqa: E402
import b2_lib as L  # noqa: E402

REPORT = Path('/home/jetson/jetformer_reports/b2_curation_20261016')
PROTOCOL = REPORT / 'protocol.json'
PROTOCOL_SHA256 = 'bd54dbbdf31f610064baa8e401b197287a5ec1e23110ec0ba8f45cbf62f86e86'
EVAL_SHA = REPORT / 'eval_sha256.txt'
DERIVED = REPO / 'data' / 'derived' / 'b2_curation_20261016'
ZOO = REPO / 'data' / 'zoo' / 'b2_curation_20261016'
PRED = ZOO / 'pred'
WT = ZOO / 'window_table.npz'
WJ = ZOO / 'windows.json'
GENRES = ZOO / 'pred_genres.npz'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
PRED_TABLE_KEY = '3d339512828b17195d568f1b93ee4a28208f2e85'
DS_NAMES = ('clean_driving_20261011', 'ai_mix_20261011')
CLEAN = DS_NAMES[0]
R1_W1, R1_W04, R1_W2, DE_W = 1.0, 0.4, 2.0, 0.8
RUNS = {'B': 'clean_driving_20261011-nocmd-s3-20261011-024626',
        'A_s0': 'clean_driving_20261011-jetformer_v0-s0-nocmd-stride3-e30',
        'A_s1': 'clean_driving_20261011-jetformer_v0-s1-nocmd-stride3-e30',
        'A_s2': 'realacc_20261015-v0-s2', 'A_s3': 'realacc_20261015-v0-s3',
        'B2_s0': 'b2cur_20261016-s0', 'B2_s1': 'b2cur_20261016-s1',
        'DROP_s0': 'b2cur_20261016-droponly-s0', 'RELABEL_s0': 'b2cur_20261016-relabelonly-s0'}
OLD_PRED = {'A_s0': ('policy_zoo_20261012', 'jetformer_v0_s0'), 'A_s1': ('policy_zoo_20261012', 'jetformer_v0_s1'),
            'B': ('policy_zoo_20261012', 'B'), 'A_s2': ('real_accuracy_20261015', 'realacc_20261015-v0-s2'),
            'A_s3': ('real_accuracy_20261015', 'realacc_20261015-v0-s3')}
A_NAMES = ('A_s0', 'A_s1', 'A_s2', 'A_s3')
B2_NAMES = ('B2_s0', 'B2_s1')
ARM_NAMES = ('DROP_s0', 'RELABEL_s0')
CKPTS = ('best', 'last')
BATCH, WORKERS = 32, 3


def now_jst():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_protocol():
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise SystemExit('protocol.json の sha256 が固定の値と違います。止めます')
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def _r(x, nd=3):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    return round(xf, nd) if np.isfinite(xf) else None


def rnd(x, nd=3):
    if isinstance(x, dict):
        return {str(k): rnd(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, nd) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return _r(x, nd)
    if isinstance(x, np.ndarray):
        return rnd(x.tolist(), nd)
    return x


def run_dir(name):
    return rp.RUNS_DIR / RUNS[name]


def old_pred_path(name, ck):
    zoo, stem = OLD_PRED[name]
    return config.DATA_DIR / 'zoo' / zoo / 'pred' / ('%s%s.npz' % (stem, '' if ck == 'best' else '_last'))


def pred_path(name, table, ck, spec):
    """table = 'pred'(保留の表)・'win'(窓の表)、spec = 'own' | 'alignedrows' | 'g5' | 'g4'"""
    return PRED / ('%s__%s__%s__%s.npz' % (name, table, ck, spec))


def table_key(ds, scene, frame):
    return hashlib.sha1('|'.join('%d:%s:%d' % (int(d), s, int(i)) for d, s, i in zip(ds, scene, frame)).encode()).hexdigest()


# ---------------------------------------------------------------- tables(CPU)

def _remap_scenes():
    with np.load(DERIVED / 'remap_b2.npz') as z:
        sc, md = z['scene'].astype(str), z['mode']
    out = sorted({s for s in np.unique(sc) if (md[sc == s] == 4).all()})
    part = sorted({s for s in np.unique(sc) if (md[sc == s] >= 0).any()})
    if out != part:
        raise SystemExit('remap_b2.npz にシーンの一部だけを付け替えた行があります(想定外)')
    return out


def _dropped_pass_keys():
    card = json.loads((DERIVED / 'curation_card.json').read_text(encoding='utf-8'))
    dp = card.get('dropped_passes')
    if dp is None:
        dp = json.loads(PROTOCOL.read_text(encoding='utf-8'))['curation']['dropped_passes']
    return {(q['log'], round(float(q['ta_s']), 3)): q.get('reason') for q in dp}


def build_tables(proto):
    """窓の表(行の順を固定)・通過の一覧・保留の表のジャンル(own_labels・aligned)"""
    passes = json.loads((DERIVED / 'passes_b2.json').read_text(encoding='utf-8'))
    root = config.DATASET_DIR / CLEAN
    splits = json.loads((root / 'splits.json').read_text())
    split_of = {s: k for k in ('train', 'val') for s in splits[k]}
    human = {}
    for s in sorted(split_of):
        with np.load(root / 'aligned' / (s + '.npz')) as z:
            human.setdefault(s[:15], []).append((s, z['timestamps_us'].astype(np.int64), z['vehicle_valid'].astype(bool)))
    with np.load(config.DATA_DIR / 'rl' / 'ai_mix_20261011' / 'frames.npz') as z:
        f = {k: z[k] for k in ('log', 'timestamps_us', 'scene', 'frame', 'vehicle_valid', 'source')}
    dropped = _dropped_pass_keys()
    rowid = {}
    rows = []

    def rid(ds, s, i):
        k = (int(ds), str(s), int(i))
        if k not in rowid:
            rowid[k] = None
            rows.append(k)
        return k

    entries = []
    for j, p in enumerate(passes):
        if not p['ai'] and p['log'] not in human:
            continue                          # AI のログの中の人の区間は使わない(下調べ survey2 と同じ)
        for kind in ('r1', 'de'):
            if kind == 'r1':
                lo, hi, lo_wide = p['ta_s'] - R1_W1, p['ta_s'], p['ta_s'] - R1_W2
            else:
                if p['tg_s'] is None:
                    continue
                lo, hi = p['tg_s'], p['tg_s'] + DE_W
                lo_wide = lo
            got = []                          # (行の key, 1 秒の窓か, 0.4 秒の窓か)
            sp = set()
            if not p['ai']:
                for s, t, vv in human[p['log']]:
                    tt = (t - p['t0_us']) / 1e6
                    w1 = (tt >= lo) & (tt <= hi) & vv
                    ww = (tt >= lo_wide) & (tt <= hi) & vv
                    w04 = (tt >= p['ta_s'] - R1_W04) & (tt <= p['ta_s']) & vv if kind == 'r1' else w1
                    if w1.any():
                        sp.add(split_of[s])
                    for i in np.nonzero(ww)[0]:
                        got.append((rid(0, s, i), bool(w1[i]), bool(w04[i])))
                if not any(g[1] for g in got):
                    continue
                st = 'H' if sp == {'val'} else ('T' if sp == {'train'} else 'mixed')
            else:
                m = np.nonzero(f['log'] == p['log'])[0]
                tt = (f['timestamps_us'][m].astype(np.int64) - p['t0_us']) / 1e6
                vv = f['vehicle_valid'][m].astype(bool)
                w1 = (tt >= lo) & (tt <= hi) & vv
                ww = (tt >= lo_wide) & (tt <= hi) & vv
                w04 = (tt >= p['ta_s'] - R1_W04) & (tt <= p['ta_s']) & vv if kind == 'r1' else w1
                if not w1.any():
                    continue
                for k in np.nonzero(ww)[0]:
                    r = m[k]
                    got.append((rid(1, str(f['scene'][r]), int(f['frame'][r])), bool(w1[k]), bool(w04[k])))
                st = 'AI'
            e = {'pass_index': j, 'kind': kind, 'set': st, 'log': p['log'], 'ai': bool(p['ai']), 'ta_s': p['ta_s'],
                 'tg_s': p['tg_s'], 'entry_erpm': p['entry_erpm'], 'r1_collision': bool(p['r1_collision']),
                 'r1_strong': bool(p['r1_strong']), 'de_collision': bool(p['de_collision']), 'de_yaw_dps': p.get('de_yaw_dps'),
                 'dropped_b2': (p['log'], round(float(p['ta_s']), 3)) in dropped,
                 'dropped_reason': dropped.get((p['log'], round(float(p['ta_s']), 3))),
                 '_rows': got}
            entries.append(e)
    # 直線(10/12 の 542 行 = rl_evaluate の item4 と同じ)と、参考の SW・HS の直線(計画 G3 の字義)
    tab = ev.pred_table()
    if tab['key'] != PRED_TABLE_KEY:
        raise SystemExit('pred_table の key が違います')
    cols, fr = ev._val_arrays(tab)
    m0 = tab['part'] == 0
    frow = cols['frow']
    stm = np.zeros(len(frow), bool)
    stm[m0] = ev.straight_mask(fr['loc_code'][frow[m0]], fr['gz'][frow[m0]], fr['erpm'][frow[m0]],
                               fr['near_collision'][frow[m0]])
    sws = np.zeros(len(frow), bool)
    sws[m0] = ev.straight_mask(fr['loc_code'][frow[m0]], fr['gz'][frow[m0]], fr['erpm'][frow[m0]],
                               fr['near_collision'][frow[m0]], locs=('other', 'HS'))
    straight_keys = [rid(1, tab['scene'][k], tab['frame'][k]) for k in np.nonzero(stm)[0]]
    swhs_keys = [rid(1, tab['scene'][k], tab['frame'][k]) for k in np.nonzero(sws)[0]]
    # 行の順を固定: (データセット, シーン, フレーム) の並び
    rows = sorted(rows)
    index = {k: n for n, k in enumerate(rows)}
    ds = np.array([k[0] for k in rows], np.int8)
    scene = np.array([k[1] for k in rows])
    frame = np.array([k[2] for k in rows], np.int32)
    key = table_key(ds, scene, frame)
    # 行ごとの aligned の mode・人/AI の実際の指令(action_exec。clean_driving の行は ai_mix の同じシーンを時刻で引く)
    amode = np.zeros(len(rows), np.int16)
    exe = np.full((len(rows), 2), np.nan)
    src = np.full(len(rows), -1, np.int8)
    for d in (0, 1):
        for s in np.unique(scene[ds == d]):
            r = np.nonzero((ds == d) & (scene == s))[0]
            a = ev._aligned(DS_NAMES[d], s, ['mode', 'timestamps_us'])
            amode[r] = a['mode'][frame[r]]
            b = ev._aligned(DS_NAMES[1], s, ['action_exec', 'timestamps_us', 'source'])
            pos = {int(t): i for i, t in enumerate(b['timestamps_us'].tolist())}
            for rr in r:
                i = pos.get(int(a['timestamps_us'][frame[rr]]))
                if i is not None:
                    exe[rr] = b['action_exec'][i]
                    src[rr] = b['source'][i]
    for e in entries:
        g = e.pop('_rows')
        e['rows_w2'] = [index[k] for k, _, _ in g]
        e['rows_w1'] = [index[k] for k, a1, _ in g if a1]
        e['rows_w04'] = [index[k] for k, a1, a04 in g if a04]
        e['rows'] = e['rows_w1']
    straight = [index[k] for k in straight_keys]
    swhs = [index[k] for k in swhs_keys]
    # 数を protocol と比べる
    ce = proto['evaluation']['window_sets']['counts_expected']
    counts = {}
    for kind in ('r1', 'de'):
        for st in ('H', 'AI', 'T', 'mixed'):
            es = [e for e in entries if e['kind'] == kind and e['set'] == st]
            c = {'n': len(es), 'ge5500': sum(e['entry_erpm'] is not None and e['entry_erpm'] >= L.FAST_ERPM for e in es),
                 'ge6200': sum(e['entry_erpm'] is not None and e['entry_erpm'] >= L.VERY_FAST_ERPM for e in es),
                 'r1_strong': sum(e['r1_strong'] for e in es), 'no_entry': sum(e['entry_erpm'] is None for e in es),
                 'frames_w1': sum(len(e['rows_w1']) for e in es), 'frames_w2': sum(len(e['rows_w2']) for e in es),
                 'dropped_b2': sum(e['dropped_b2'] for e in es)}
            counts.setdefault(kind, {})[st] = c
    diffs = []
    for kind, d in ce.items():
        if kind not in ('r1', 'de'):
            continue
        for st, want in d.items():
            for k, v in want.items():
                if counts[kind][st][k] != v:
                    diffs.append('%s/%s/%s: %s(protocol %s)' % (kind, st, k, counts[kind][st][k], v))
    if len(straight) != ce['straight_frames']:
        diffs.append('straight_frames: %d(protocol %d)' % (len(straight), ce['straight_frames']))
    # 保留の表のジャンル(own_labels・aligned)
    remap = _remap_scenes()
    modes = np.zeros(len(tab['part']), np.int64)
    for part, dsn in ((0, ev.DATASET), (1, ev.IV_DATASET)):
        for s in np.unique(tab['scene'][tab['part'] == part]):
            r = np.nonzero((tab['part'] == part) & (tab['scene'] == s))[0]
            modes[r] = ev._aligned(dsn, s, ['mode'])['mode'][tab['frame'][r]]
    trained = np.array([4, 5])
    g_al, sub = ev._genres(modes, None, trained)
    hit = np.isin(tab['scene'], remap) & (tab['part'] == 0)
    g_own = g_al.copy()
    g_own[hit] = 4
    hit_scenes = sorted(set(tab['scene'][hit].tolist()))
    if hit_scenes != ['20261010-152405-0002', '20261010-152707-0001']:
        diffs.append('付け替えの対象の保留のシーン: %s' % hit_scenes)
    if (tab['part'][np.isin(tab['scene'], remap)] != 0).any():
        diffs.append('付け替えのシーンが介入の集合にある')
    info = {'created_at': now_jst(), 'window_key': key, 'window_rows': len(rows), 'pred_table_key': tab['key'],
            'counts': counts, 'straight_frames': len(straight), 'straight_sw_hs_frames': len(swhs),
            'remap_scenes': remap, 'own_label_rows': int(hit.sum()), 'own_label_scenes': hit_scenes,
            'aligned_modes_of_own_label_rows': sorted(set(modes[hit].tolist())),
            'genre_substituted_rows': int(sub.sum()), 'exec_found_rows': int(np.isfinite(exe[:, 1]).sum()),
            'diffs': diffs}
    return {'rows': (ds, scene, frame, key, amode, exe, src), 'entries': entries, 'straight': straight, 'swhs': swhs,
            'g_al': g_al, 'g_own': g_own, 'own_rows': np.nonzero(hit)[0], 'info': info}


def cmd_tables(args):
    proto = load_protocol()
    if WT.exists() and WJ.exists() and GENRES.exists() and not args.force:
        print('作成済み: %s' % WT)
        print(json.dumps(json.loads(WJ.read_text())['info'], ensure_ascii=False, indent=1))
        return
    t0 = time.time()
    T = build_tables(proto)
    ds, scene, frame, key, amode, exe, src = T['rows']
    info = T['info']
    print(json.dumps(info, ensure_ascii=False, indent=1))
    if info['diffs']:
        raise SystemExit('窓の数が protocol と違います(止めて deviations.md へ): %s' % info['diffs'])
    rp._save_npz(WT, ds=ds, scene=scene, frame=frame, key=np.array(key), aligned_mode=amode, exec=exe, source=src)
    rp._save_npz(GENRES, aligned=T['g_al'].astype(np.int8), own=T['g_own'].astype(np.int8), own_rows=T['own_rows'],
                 key=np.array(PRED_TABLE_KEY))
    rp._save_json(WJ, {'info': info, 'entries': T['entries'], 'straight_rows': T['straight'], 'straight_sw_hs_rows': T['swhs']})
    lines = ['# B2 の評価の固定(%s)。tables の段で書いた。予測の前に固定する' % now_jst(),
             'window_table_key %s' % key, 'pred_table_key %s' % PRED_TABLE_KEY,
             '%s  %s' % (sha256(WT), WT), '%s  %s' % (sha256(WJ), WJ), '%s  %s' % (sha256(GENRES), GENRES)]
    EVAL_SHA.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('窓の表: %d 行・key %s(%.0f 秒)。eval_sha256.txt に書きました' % (len(ds), key, time.time() - t0))


def load_tables():
    with np.load(WT) as z:
        wt = {k: z[k] for k in z.files}
    wj = json.loads(WJ.read_text(encoding='utf-8'))
    with np.load(GENRES) as z:
        gn = {k: z[k] for k in z.files}
    if table_key(wt['ds'], wt['scene'], wt['frame']) != str(wt['key']):
        raise SystemExit('窓の表の key が合いません')
    lines = EVAL_SHA.read_text(encoding='utf-8').splitlines()
    if 'window_table_key %s' % str(wt['key']) not in lines:
        raise SystemExit('eval_sha256.txt の window_table_key と違います')
    return wt, wj, gn


# ---------------------------------------------------------------- 予測(GPU)

class _RowFrames(object):
    """窓の表の行(データセット・シーン・フレーム)。policy_zoo._Frames と同じ __getitem__(ずれの試験なし)"""

    def __init__(self, ds, scene, frame, use_command=False):
        self.use_command, self.pert = use_command, None
        self.items = []
        cache = {}
        for d, s, i in zip(ds, scene, frame):
            dsn = DS_NAMES[int(d)]
            if (dsn, s) not in cache:
                cache[(dsn, s)] = ev._aligned(dsn, str(s), ['cam_files', 'tof', 'imu', 'vehicle', 'mode'])
            a = cache[(dsn, s)]
            i = int(i)
            self.items.append((str(config.DATASET_DIR / dsn / str(a['cam_files'][i])), a['tof'][i], a['imu'][i],
                               a['vehicle'][i], int(a['mode'][i])))

    __len__ = pz._Frames.__len__
    __getitem__ = pz._Frames.__getitem__


def _web_busy():
    import urllib.request
    why = []
    try:
        with urllib.request.urlopen('http://127.0.0.1:8080/api/status', timeout=3) as r:
            st = json.load(r)
        for k, v in (('training', st.get('training')), ('driving', st.get('driving')),
                     ('recording', (st.get('recorder') or {}).get('recording')),
                     ('session', (st.get('session') or {}).get('active'))):
            if v:
                why.append('管理画面: %s' % k)
    except Exception:          # noqa: BLE001  管理画面が止まっていれば確認しない
        pass
    return why


def _check_gpu_ready():
    busy = _web_busy()
    if busy:
        raise SystemExit('始めません: %s' % busy)
    rp._check_gpu_free(min_gb=2.0)


def _pred_ok(path, key, n_rows):
    try:
        with np.load(path) as z:
            return str(z['key']) == key and z['pred'].shape == (n_rows, 10, 2) and bool(np.isfinite(z['pred']).all())
    except Exception:          # noqa: BLE001  壊れた npz は作り直す
        return False


def _model(name, ck):
    from jetformer.policies import runtime
    d = run_dir(name)
    cfg = rp._read_json(d / 'model_config.json')
    model, _, spec, _ = runtime.load_run(d, ck, device='cuda')
    total, trainable = runtime.count_params(model)
    meta = dict(run=np.array(d.name), ckpt=np.array(ck), policy=np.array(spec.name), params_total=np.array(total),
                params_trainable=np.array(trainable))
    return model, cfg, meta


def _free(model):
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _crosscheck_one(tab, which):
    """B の best を予測し直して 10/12 の B.npz と比べる。which = 'part0'(protocol の字義)| 'full'(10/12 と同じ行・同じバッチの切り方)"""
    t0 = time.time()
    rows = np.nonzero(tab['part'] == 0)[0] if which == 'part0' else np.arange(len(tab['part']))
    model, cfg, _ = _model('B', 'best')
    fs = pz._Frames(tab, rows, use_command=bool(cfg.get('command_input', True)))
    trained = np.array([int(x) for x in cfg['trained_modes']])
    genres, _ = ev._genres([it[4] for it in fs.items], None, trained)
    pred = pz._predict_frames(model, fs, genres, cfg['num_modes'], BATCH, WORKERS)['actions'].astype(np.float64)
    _free(model)
    with np.load(old_pred_path('B', 'best')) as z:
        ref = z['pred'].astype(np.float64)[rows]
        same_key = str(z['key']) == tab['key']
        same_genre = bool(np.array_equal(z['genre'][rows], genres.astype(np.int8)))
    rp._save_npz(ZOO / ('crosscheck_B_%s.npz' % which), pred=pred.astype(np.float32), rows=rows, key=np.array(tab['key']))
    d = np.abs(pred - ref).max(axis=(1, 2))
    mx = float(d.max())
    nb = (len(rows) + BATCH - 1) // BATCH
    last_batch = np.arange(len(rows)) >= (nb - 1) * BATCH
    bad = np.nonzero(d > 0.01)[0]
    return {'name': 'B', 'run': RUNS['B'], 'ckpt': 'best', 'rows': which, 'frames': int(len(rows)),
            'same_key': same_key, 'same_genre': same_genre, 'max_abs_us': _r(mx, 5),
            'mean_abs_us': _r(np.abs(pred - ref).mean(), 6), 'limit_us': 0.01,
            'pass': bool(same_key and same_genre and mx <= 0.01), 'batch': BATCH, 'workers': WORKERS,
            'rows_over_limit': int(len(bad)), 'rows_over_limit_in_last_batch': int(last_batch[bad].sum()),
            'last_batch_size': int(len(rows) - (nb - 1) * BATCH),
            'max_abs_us_outside_last_batch': _r(float(d[~last_batch].max()), 5),
            'seconds': _r(time.time() - t0, 1), 'checked_at': now_jst(), 'tool_sha256': sha256(__file__)}


def cmd_crosscheck(args):
    """protocol の crosscheck(part 0)。21:13 の 1 回目は part 0 だけを予測し、最後の半端なバッチ(7 行)で
    0.080 µs の差が出た(deviations.md)。原因を確かめるため、part 0 と全部の行(10/12 と同じ行・同じバッチの切り方)の
    両方を予測して保存し、どこで差が出るかを記録する。続けてよいのは「全部の行で 0.01 µs 以内」かつ
    「part 0 の差が最後の半端なバッチの中だけ」のとき"""
    out_path = ZOO / 'crosscheck.json'
    old = rp._read_json(out_path)
    if old and old.get('pass') and not args.force:
        print('作成済み: %s' % out_path)
        print(json.dumps(old, ensure_ascii=False, indent=1))
        return
    tab = ev.pred_table()
    if tab['key'] != PRED_TABLE_KEY:
        raise SystemExit('pred_table の key が違います')
    _check_gpu_ready()
    res = {w: _crosscheck_one(tab, w) for w in ('part0', 'full')}
    p0, fu = res['part0'], res['full']
    ok = bool(fu['pass'] and p0['same_key'] and p0['same_genre']
              and p0['rows_over_limit'] == p0['rows_over_limit_in_last_batch']
              and (p0['max_abs_us_outside_last_batch'] or 0.0) <= 0.01)
    out = {'pass': ok, 'protocol_literal_part0_pass': p0['pass'], 'part0': p0, 'full': fu,
           'rule': '全部の行(10/12 と同じ行・同じバッチの切り方)で最大の差 ≤ 0.01 µs かつ part 0 の差が最後の半端なバッチの中だけ'
                   '(deviations.md 2026-10-16 21:2x)', 'checked_at': now_jst(), 'tool_sha256': sha256(__file__)}
    rp._save_json(out_path, out)
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    if not out['pass']:
        raise SystemExit('crosscheck が合いません(予測を止めて deviations.md へ)')


def predict_jobs(include_arms=False):
    """(名前, 表, ckpt, spec) の一覧(protocol の gpu_order の 4。+ last のジャンル 4・A・B のジャンル 4 は足した参考)"""
    jobs = []
    for n in B2_NAMES:
        for ck in CKPTS:
            jobs.append((n, 'pred', ck, 'own'))
            jobs.append((n, 'pred', ck, 'alignedrows'))
    for n in ('B',) + A_NAMES:
        jobs.append((n, 'win', 'best', 'g5'))
    for n in B2_NAMES:
        for ck in CKPTS:
            jobs.append((n, 'win', ck, 'g5'))
        jobs.append((n, 'win', 'best', 'g4'))
    for n in B2_NAMES:                     # 足した: last のジャンル 4(desk_go(4) の R1-held の last の条件に要る)
        jobs.append((n, 'win', 'last', 'g4'))
    for n in ('B',) + A_NAMES:             # 足した(参考): aligned のジャンルの窓の値を作るため
        jobs.append((n, 'win', 'best', 'g4'))
    if include_arms:
        for n in ARM_NAMES:
            jobs.append((n, 'pred', 'best', 'own'))
            jobs.append((n, 'pred', 'last', 'own'))
            jobs.append((n, 'win', 'best', 'g5'))
            jobs.append((n, 'win', 'best', 'g4'))
    return jobs


def cmd_predict(args):
    cc = rp._read_json(ZOO / 'crosscheck.json')
    if not (cc and cc.get('pass')):
        raise SystemExit('先に crosscheck を通してください')
    tab = ev.pred_table()
    if tab['key'] != PRED_TABLE_KEY:
        raise SystemExit('pred_table の key が違います')
    wt, wj, gn = load_tables()
    wkey = str(wt['key'])
    jobs = predict_jobs(include_arms=args.arms)
    if args.only:
        jobs = [j for j in jobs if j[0] in args.only]
    todo = []
    for n, table, ck, spec in jobs:
        res = rp._read_json(run_dir(n) / 'result.json') or {}
        if res.get('state') != 'done':
            print('%s: 学習が済んでいないので飛ばします' % n)
            continue
        key, nrows = (PRED_TABLE_KEY, len(tab['part'])) if table == 'pred' else (wkey, len(wt['ds']))
        if spec == 'alignedrows':
            nrows = len(gn['own_rows'])
        if not _pred_ok(pred_path(n, table, ck, spec), key, nrows):
            todo.append((n, table, ck, spec))
    if not todo:
        print('予測は作成済み')
        return
    _check_gpu_ready()
    for n, table, ck, spec in todo:
        t0 = time.time()
        model, cfg, meta = _model(n, ck)
        if [int(x) for x in cfg['trained_modes']] != [4, 5] or cfg['num_modes'] != 6:
            raise SystemExit('%s: trained_modes・num_modes が想定と違います: %s %s' % (n, cfg['trained_modes'], cfg['num_modes']))
        use_cmd = bool(cfg.get('command_input', True))
        if table == 'pred':
            if spec == 'own':
                rows = np.arange(len(tab['part']))
                genres = gn['own'].astype(np.int64) if n.startswith('B2') or n == 'RELABEL_s0' else gn['aligned'].astype(np.int64)
            else:
                rows = gn['own_rows']
                genres = gn['aligned'].astype(np.int64)[rows]
            fs = pz._Frames(tab, rows, use_command=use_cmd)
            key = PRED_TABLE_KEY
        else:
            fs = _RowFrames(wt['ds'], wt['scene'], wt['frame'], use_command=use_cmd)
            genres = np.full(len(wt['ds']), 5 if spec == 'g5' else 4, np.int64)
            rows = np.arange(len(wt['ds']))
            key = wkey
        res = pz._predict_frames(model, fs, genres, cfg['num_modes'], BATCH, WORKERS)
        _free(model)
        pred = res.pop('actions')
        p = pred_path(n, table, ck, spec)
        rp._save_npz(p, pred=pred, genre=genres.astype(np.int8), rows=rows, key=np.array(key), table=np.array(table),
                     spec=np.array(spec), ref=np.array([cfg['steer_center_us'], cfg['throttle_neutral_us']]), **meta)
        ok = _pred_ok(p, key, len(rows))
        print('[%s] %s %s %s %s: %d 行 %.0f 秒・読み直し %s' % (time.strftime('%H:%M:%S'), n, table, ck, spec, len(pred),
                                                         time.time() - t0, 'OK' if ok else 'NG'), flush=True)
        if not ok:
            raise SystemExit('%s を読み直せません' % p)


# ---------------------------------------------------------------- エンジン(GPU)

ENGINE_DIR = ZOO / 'engine'
LOAD_CHECK = REPO / 'experiments' / '20261013_sim_cotrain' / 'train' / 'engine_load_check.py'
LOOP_CHECK = REPO / 'experiments' / '20261012_policy_zoo' / 'smoke' / 'engine_check.py'


def cmd_engine(args):
    names = ['B2_s%d' % args.seed]
    _check_gpu_ready()
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    log = {}
    for n in names:
        d = run_dir(n)
        eng = d / 'jetformer_fp16.engine'
        if not eng.exists():
            t0 = time.time()
            rc = subprocess.call([PY, '-u', str(REPO / 'tools' / 'build_engine.py'), str(d / 'jetformer.onnx'), str(eng)],
                                 cwd=str(REPO))
            log[n] = {'build_rc': rc, 'build_s': round(time.time() - t0, 1)}
            print('[%s] %s: エンジン rc=%s %.0f 秒' % (time.strftime('%H:%M:%S'), n, rc, time.time() - t0), flush=True)
            if rc != 0:
                raise SystemExit('エンジンを作れませんでした')
    for n in names + ['B']:
        d = run_dir(n)
        for tag, script in (('load', LOAD_CHECK), ('loop', LOOP_CHECK)):
            out = ENGINE_DIR / ('%s_%s.json' % (tag, n))
            old = rp._read_json(out)
            eng = d / 'jetformer_fp16.engine'
            if old and old.get('engine_mtime_ns') == eng.stat().st_mtime_ns and not args.force:
                print('作成済み: %s' % out)
                continue
            t0 = time.time()
            rc = subprocess.call([PY, '-u', str(script), str(d), str(out)], cwd=str(REPO))
            j = rp._read_json(out) or {}
            j['engine_mtime_ns'] = eng.stat().st_mtime_ns
            j['rc'] = rc
            j['seconds_total'] = round(time.time() - t0, 1)
            rp._save_json(out, j)
            print('[%s] %s %s: rc=%s %.0f 秒' % (time.strftime('%H:%M:%S'), n, tag, rc, time.time() - t0), flush=True)
    # 走行タブの一覧(drive.list_models。読むだけ)
    code = ('import json,sys; sys.path.insert(0, %r); from jetformer import drive; '
            'print(json.dumps([{k: m.get(k) for k in ("name","drivable","trained_modes","steer_center_us",'
            '"throttle_neutral_us","policy")} for m in drive.list_models()], ensure_ascii=False))' % str(REPO))
    out = subprocess.run([PY, '-c', code], capture_output=True, text=True, cwd=str(REPO))
    try:
        lm = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:          # noqa: BLE001
        lm = {'error': out.stderr[-2000:]}
    rp._save_json(ENGINE_DIR / 'list_models.json', {'checked_at': now_jst(), 'models': lm, 'log': log})
    print('list_models: %s' % [m.get('name') for m in lm] if isinstance(lm, list) else lm)


# ---------------------------------------------------------------- freeze(道具の sha256)

def cmd_freeze(args):
    files = [HERE / 'b2_eval.py', HERE / 'b2_lib.py', HERE / 'test_b2_lib.py']
    lines = EVAL_SHA.read_text(encoding='utf-8').splitlines()
    lines = [ln for ln in lines if not any(ln.endswith('  %s' % f) for f in files) and not ln.startswith('# 道具')]
    lines.append('# 道具(%s。評価の前に固定)' % now_jst())
    lines += ['%s  %s' % (sha256(f), f) for f in files]
    EVAL_SHA.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


def check_frozen():
    lines = EVAL_SHA.read_text(encoding='utf-8').splitlines()
    bad = []
    for ln in lines:
        if ln.startswith('#') or '  ' not in ln:
            continue
        h, p = ln.split('  ', 1)
        if Path(p).exists() and sha256(p) != h:
            bad.append(p)
    return bad


# ---------------------------------------------------------------- 評価の材料(CPU)

def _drop_windows(tab, held):
    """主指標のフレームのうち drop_b2 の reason > 0(b2_baseline.py と同じ。ai_mix の行と clean_driving の行を時刻で対応)"""
    with np.load(DERIVED / 'drop_b2.npz') as z:
        reason_of = {}
        sc, fr, rs = z['scene'].astype(str), z['frame'], z['reason']
        for s in np.unique(sc):
            m = sc == s
            reason_of[s] = dict(zip(fr[m].tolist(), rs[m].tolist()))
    m0 = tab['part'] == 0
    human = np.isin(tab['scene'], held['human_clean_scenes']) & m0
    in_win = np.zeros(len(tab['part']), bool)
    for s in np.unique(tab['scene'][human]):
        rows = np.nonzero(human & (tab['scene'] == s))[0]
        a_mix = ev._aligned('ai_mix_20261011', s, ['timestamps_us'])['timestamps_us']
        a_cd = ev._aligned(CLEAN, s, ['timestamps_us'])['timestamps_us']
        pos = {int(t): i for i, t in enumerate(a_cd.tolist())}
        for r in rows:
            in_win[r] = reason_of[s][pos[int(a_mix[tab['frame'][r]])]] > 0
    return in_win


def _iv_class(tab):
    """介入の集合の行ごとの交代の分類(label_events.TAKEOVER_CLASSES の番号)"""
    out = np.full(len(tab['part']), -1, int)
    for s in np.unique(tab['scene'][tab['part'] == 1]):
        r = np.nonzero((tab['part'] == 1) & (tab['scene'] == s))[0]
        a = ev._aligned(ev.IV_DATASET, s, ['takeover_class'])
        out[r] = a['takeover_class'][tab['frame'][r]]
    return out


def load_base():
    """保留の表の正解・集合・既にある A・B の予測"""
    tab = ev.pred_table()
    held = ev._heldout()
    if tab['key'] != PRED_TABLE_KEY or held['pred_table_key'] != PRED_TABLE_KEY:
        raise SystemExit('pred_table の key が違います')
    cols, f = ev._val_arrays(tab)
    m0 = tab['part'] == 0
    clean = np.isin(tab['scene'], held['human_clean_scenes']) & m0 & cols['fv'] & ~cols['impact']
    in_win = _drop_windows(tab, held)
    iv = (tab['part'] == 1) & cols['fv']
    sets = {'primary': clean, 'curation_windows': clean & in_win, 'rest': clean & ~in_win, 'intervention_iv_all': iv}
    old = {ck: {} for ck in CKPTS}
    for n in OLD_PRED:
        for ck in CKPTS:
            with np.load(old_pred_path(n, ck)) as z:
                if str(z['key']) != PRED_TABLE_KEY:
                    raise SystemExit('key が違います: %s' % old_pred_path(n, ck))
                old[ck][n] = z['pred'].astype(np.float64)
    return tab, held, cols, f, sets, old


def cmd_repro(args):
    """既にある A・B の予測で、固定した基準値(baseline.json の L・陰性対照の窓の値)を作り直せるか"""
    proto = load_protocol()
    tab, held, cols, f, sets, old = load_base()
    base = proto['baseline_values']
    y = cols['y']
    rep, worst = {}, 0.0
    for ck in CKPTS:
        for n in OLD_PRED:
            l1 = pz.l1_rows(old[ck][n], y)
            for k, m in sets.items():
                a, b = float(l1[m].mean()), float(base['L_us'][ck][n][k])
                rep['%s/%s/%s' % (ck, n, k)] = [round(a, 4), b]
                worst = max(worst, abs(round(a, 3) - b))
    frames = {k: int(m.sum()) for k, m in sets.items()}
    # 陰性対照(保留の人の R1 の 1 秒・DE の窓、aligned のジャンル、既にある予測)を窓の表の通過で作り直す
    wt, wj, gn = load_tables()
    key = {(s, int(i)): k for k, (s, i) in enumerate(zip(tab['scene'], tab['frame'])) if tab['part'][k] == 0}
    outs = {n: ev.drive_outputs(old['best'][n], (1441.0, 1496.0), 1.0, 1.0, 3, 0.0) for n in OLD_PRED}
    nc = {}
    for kind, ch in (('r1', 1), ('de', 0)):
        vals = []
        for e in wj['entries']:
            if e['kind'] != kind or e['set'] != 'H':
                continue
            rr = [key[(str(wt['scene'][r]), int(wt['frame'][r]))] for r in e['rows_w1']
                  if (str(wt['scene'][r]), int(wt['frame'][r])) in key]
            v = {n: float(o[rr, ch].mean()) for n, o in outs.items()}
            v['entry_erpm'] = e['entry_erpm']
            vals.append(v)
        res = {'passes': len(vals)}
        for sub, sel in (('all', lambda q: True), ('ge5500', lambda q: q['entry_erpm'] >= L.FAST_ERPM),
                         ('lt5500', lambda q: q['entry_erpm'] < L.FAST_ERPM)):
            vs = [q for q in vals if sel(q)]
            d = {('%s-B' % n): round(float(np.mean([q[n] - q['B'] for q in vs])), 2) for n in A_NAMES}
            d['A_s1-A_s0'] = round(float(np.mean([q['A_s1'] - q['A_s0'] for q in vs])), 2)
            d['n'] = len(vs)
            d['mean_B'] = round(float(np.mean([q['B'] for q in vs])), 1)
            res[sub] = d
        nc[kind] = res
    want = base['negative_control_existing_aligned_genre']
    nc_diff = []
    for kind, wk in (('r1', 'r1_1s_H'), ('de', 'de_H')):
        for sub in ('all', 'ge5500', 'lt5500'):
            for k, v in want[wk][sub].items():
                if k == 'mean_human_exec':
                    continue
                if abs(nc[kind][sub][k] - v) > 0.011:
                    nc_diff.append('%s/%s/%s: %s(固定 %s)' % (wk, sub, k, nc[kind][sub][k], v))
    out = {'checked_at': now_jst(), 'L_max_abs_diff_rounded': worst, 'L_pass': worst <= 0.0015, 'frames': frames,
           'frames_pass': frames == base['frames'], 'L_values': rep, 'negative_control': nc, 'negative_control_diffs': nc_diff,
           'negative_control_pass': not nc_diff}
    out['pass'] = bool(out['L_pass'] and out['frames_pass'] and out['negative_control_pass'])
    rp._save_json(ZOO / 'repro.json', out)
    print(json.dumps({k: out[k] for k in ('L_max_abs_diff_rounded', 'L_pass', 'frames', 'frames_pass', 'negative_control_diffs',
                                          'negative_control_pass', 'pass')}, ensure_ascii=False, indent=1))
    if not out['pass']:
        raise SystemExit('固定した基準値を作り直せません(止めて調べる)')


# ---------------------------------------------------------------- 評価(CPU)

def _load_new(name, table, ck, spec, key):
    p = pred_path(name, table, ck, spec)
    with np.load(p) as z:
        if str(z['key']) != key:
            raise SystemExit('key が違います: %s' % p)
        return z['pred'].astype(np.float64), tuple(float(x) for x in z['ref'])


def _have(name, table, ck, spec):
    return pred_path(name, table, ck, spec).exists()


def evaluate_all(proto):
    bad = check_frozen()
    tab, held, cols, f, sets, old = load_base()
    wt, wj, gn = load_tables()
    wkey = str(wt['key'])
    y = cols['y']
    base = proto['baseline_values']
    th = base['thresholds_us']
    res = {'created_at': now_jst(), 'protocol_sha256': PROTOCOL_SHA256, 'tools_changed_after_freeze': bad,
           'window_table_key': wkey, 'pred_table_key': PRED_TABLE_KEY, 'tables_info': wj['info'],
           'crosscheck': rp._read_json(ZOO / 'crosscheck.json'), 'repro': {k: v for k, v in (rp._read_json(ZOO / 'repro.json')
                                                                                            or {}).items() if k != 'L_values'}}
    # ---- 保留の表の予測(B2: own_labels と aligned)
    P = {'own': {ck: {} for ck in CKPTS}, 'aligned': {ck: {} for ck in CKPTS}}
    for n in B2_NAMES + ARM_NAMES:
        for ck in CKPTS:
            if not _have(n, 'pred', ck, 'own'):
                continue
            p_own, _ = _load_new(n, 'pred', ck, 'own', PRED_TABLE_KEY)
            P['own'][ck][n] = p_own
            if _have(n, 'pred', ck, 'alignedrows'):
                p_al = p_own.copy()
                pr, _ = _load_new(n, 'pred', ck, 'alignedrows', PRED_TABLE_KEY)
                p_al[gn['own_rows']] = pr
                P['aligned'][ck][n] = p_al
            elif n in ARM_NAMES:
                P['aligned'][ck][n] = p_own if n == 'DROP_s0' else None
    for lab in P:
        for ck in CKPTS:
            P[lab][ck].update(old[ck])
            P[lab][ck] = {k: v for k, v in P[lab][ck].items() if v is not None}
    missing = [n for n in B2_NAMES for ck in CKPTS if n not in P['own'][ck]]
    if missing:
        raise SystemExit('B2 の予測が足りません: %s' % missing)
    # ---- L1(10/13 の規則)
    L1 = {lab: {ck: {n: {k: float(pz.l1_rows(p, y)[m].mean()) for k, m in sets.items()} for n, p in P[lab][ck].items()}
                for ck in CKPTS} for lab in P}
    s_floor = {k: {ck: float(th[ck][k]['s_A_three']) for ck in CKPTS} for k in sets}

    def vals(lab, names, metric):
        return {ck: [L1[lab][ck][n][metric] for n in names] for ck in CKPTS}

    l1 = {}
    for lab in ('own', 'aligned'):
        l1[lab] = {}
        for metric in sets:
            l1[lab][metric] = {
                'vs_A': L.compare(vals(lab, B2_NAMES, metric), vals(lab, ('A_s0', 'A_s1'), metric), s_floor[metric]),
                'vs_B': L.compare(vals(lab, B2_NAMES, metric), vals(lab, ('B',), metric), s_floor[metric])}
            for ref in ('vs_A', 'vs_B'):
                c = l1[lab][metric][ref]
                band = th['best'][metric]['2s_five_fresh']
                c['fresh_band_best_us'] = band
                c['within_fresh_band'] = c['worsening_best_us'] <= band
    res['L1_values'] = rnd(L1, 3)
    res['L1'] = rnd(l1, 4)
    # ---- 崩れなし(主指標、own_labels、best)
    import compare_models as cm
    clean = sets['primary']
    met = {}
    for n, p in P['own']['best'].items():
        mt = cm.metrics(p[clean], y[clean], cols['cur'][clean], 1441.0, 1496.0)
        met[n] = {'fwd_bias_us': mt['fwd_bias_us'], 'steer_sign_acc': mt['steer_sign_acc'],
                  'fwd_bias_exact': float(np.mean(p[clean, 2, 1] - y[clean, 2, 1]))}
    nocol = L.gate_no_collapse([met[n]['fwd_bias_exact'] for n in B2_NAMES], [met[n]['steer_sign_acc'] for n in B2_NAMES],
                               met['B']['steer_sign_acc'])
    res['no_collapse_metrics'] = rnd(met, 4)
    # ---- 介入(own_labels = aligned。付け替えのシーンは介入の集合に無い)
    ivc = _iv_class(tab)
    iv = sets['intervention_iv_all']
    by_class = {}
    for cid, cname in ((1, 'after_collision'), (3, 'preemptive'), (2, 'stuck')):
        m = iv & (ivc == cid)
        by_class[cname] = {'frames': int(m.sum())}
        for n in ('A_s0', 'A_s1', 'B') + B2_NAMES:
            by_class[cname][n] = float(pz.l1_rows(P['own']['best'][n], y)[m].mean()) if m.any() else None
    res['intervention_by_class'] = rnd(by_class, 3)
    # ---- 窓の表の予測 → 走行の出力
    W = {}
    refs = {}
    for n in ('B',) + A_NAMES + B2_NAMES + ARM_NAMES:
        for ck in CKPTS:
            for spec in ('g5', 'g4'):
                if _have(n, 'win', ck, spec):
                    p, ref = _load_new(n, 'win', ck, spec, wkey)
                    refs[n] = ref
                    W[(n, ck, spec)] = ev.drive_outputs(p, ref, 1.0, 1.0, 3, 0.0)
    res['refs_us'] = refs
    bad_ref = {n: r for n, r in refs.items() if tuple(r) != (1441.0, 1496.0)}
    entries = wj['entries']
    amode = wt['aligned_mode'].astype(int)

    def out_of(n, ck, genre):
        """genre = 5・4・'aligned'(行の aligned の mode)・'own'(B2 だけ付け替え)"""
        if genre in (5, 4):
            return W.get((n, ck, 'g%d' % genre))
        a5, a4 = W.get((n, ck, 'g5')), W.get((n, ck, 'g4'))
        if a5 is None or a4 is None:
            return None
        g = amode.copy()
        if genre == 'own' and (n.startswith('B2') or n.startswith('RELABEL')):
            g[np.isin(wt['scene'], wj['info']['remap_scenes'])] = 4
        return np.where((g == 4)[:, None], a4, a5)

    def sel(kind, sets_, speed=None, extra=None):
        es = [e for e in entries if e['kind'] == kind and e['set'] in sets_]
        if speed == 'fast':
            es = [e for e in es if e['entry_erpm'] is not None and e['entry_erpm'] >= L.FAST_ERPM]
        elif speed == 'slow':
            es = [e for e in es if e['entry_erpm'] is not None and e['entry_erpm'] < L.FAST_ERPM]
        elif speed == 'vfast':
            es = [e for e in es if e['entry_erpm'] is not None and e['entry_erpm'] >= L.VERY_FAST_ERPM]
        if extra is not None:
            es = [e for e in es if extra(e)]
        return es

    def summary(es, ch, b2_genre, win='rows_w1', ck_b2='best', arms=B2_NAMES, ref_genre=5, with_last=True,
                a_genre=5):
        rows = [e[win] for e in es]
        oB = out_of('B', 'best', ref_genre)
        uB = L.window_means(oB, rows, ch)
        ds = {}
        for k, n in enumerate(arms):
            o = out_of(n, ck_b2, b2_genre)
            if o is None:
                return None
            ds[k] = L.window_means(o, rows, ch) - uB
        dA = {}
        for n in A_NAMES:
            o = out_of(n, 'best', a_genre)
            if o is not None:
                dA[n] = L.window_means(o, rows, ch) - uB
        dl = None
        if with_last:
            dl = {}
            for k, n in enumerate(arms):
                o = out_of(n, 'last', b2_genre)
                if o is None:
                    dl = None
                    break
                dl[k] = L.window_means(o, rows, ch) - uB
        keep = np.isfinite(uB) & np.all([np.isfinite(v) for v in ds.values()], axis=0)
        s = L.set_summary({k: v[keep] for k, v in ds.items()}, {k: v[keep] for k, v in dA.items()} if dA else None,
                          {k: v[keep] for k, v in dl.items()} if dl else None)
        s['dropped_nan_passes'] = int((~keep).sum())
        s['u_B_mean'] = float(np.nanmean(uB))
        s['u_arm_mean'] = float(np.mean([np.nanmean(L.window_means(out_of(n, ck_b2, b2_genre), rows, ch)) for n in arms]))
        s['passes'] = [{'log': e['log'], 'ta_s': e['ta_s'], 'set': e['set'], 'entry_erpm': e['entry_erpm'],
                        'dropped_b2': e['dropped_b2'], 'r1_strong': e['r1_strong'], 'frames': len(e[win])}
                       for e, k in zip(es, keep) if k]
        return s

    gates_by_g = {}
    detail = {}
    for g in (5, 4):
        dg = {}
        dg['R1_held'] = summary(sel('r1', ('H', 'AI'), 'fast'), 1, g)
        dg['R1_train'] = summary(sel('r1', ('T',), 'fast'), 1, g, with_last=False)
        dg['R1_slow'] = summary(sel('r1', ('H', 'AI'), 'slow'), 1, g, with_last=False)
        dg['DE_steer'] = summary(sel('de', ('H', 'AI')), 0, g, win='rows', with_last=False)
        dg['DE_throttle'] = summary(sel('de', ('H', 'AI')), 1, g, win='rows', with_last=False)
        st_rows = np.array(wj['straight_rows'])
        fwd_b = L.fwd_strength(out_of('B', 'best', 5)[st_rows, 1])
        fwd_b2 = [L.fwd_strength(out_of(n, 'best', g)[st_rows, 1]) for n in B2_NAMES]
        gts = {
            'G3-R1-held': L.gate_r1_held(dg['R1_held']),
            'G3-R1-train': L.gate_r1_train(dg['R1_train']),
            'G3-R1-slow': L.gate_r1_slow(dg['R1_slow']),
            'G3-DE': L.gate_de(dg['DE_steer'], dg['DE_throttle']),
            'G3-straight': L.gate_straight(fwd_b2, fwd_b),
            'G3-no-collapse': nocol,
            'G3-L1': L.gate_l1(l1['own']['primary']['vs_A'], l1['own']['rest']['vs_A']),
            'G3-intervention': L.gate_intervention(l1['own']['intervention_iv_all']['vs_A']),
        }
        go, failed = L.desk_go(gts)
        gates_by_g[g] = {'gates': gts, 'desk_go': go, 'failed': failed}
        detail[g] = dg
    g_drive = L.drive_genre(gates_by_g[5]['desk_go'], gates_by_g[4]['desk_go'])
    res['gates'] = rnd({str(g): v for g, v in gates_by_g.items()}, 4)
    res['gate_detail'] = rnd({str(g): {k: {kk: vv for kk, vv in v.items()} for k, v in d.items()} for g, d in detail.items()}, 3)
    res['drive_genre'] = g_drive
    # 感度: G3-L1 を vs B で見たら
    res['sensitivity_L1_vs_B'] = rnd({'gate': L.gate_l1(l1['own']['primary']['vs_B'], l1['own']['rest']['vs_B']),
                                      'intervention': L.gate_intervention(l1['own']['intervention_iv_all']['vs_B'])}, 4)
    # ---- エンジン(走らせる種 = 種 0。不合格のときだけ種 1)
    eng = {}
    lm = rp._read_json(ENGINE_DIR / 'list_models.json') or {}
    listed = {m.get('name'): m for m in (lm.get('models') or []) if isinstance(m, dict)}
    for n in ('B',) + B2_NAMES:
        e = rp._read_json(ENGINE_DIR / ('load_%s.json' % n))
        lp = rp._read_json(ENGINE_DIR / ('loop_%s.json' % n))
        if not e:
            continue
        rj = rp._read_json(run_dir(n) / 'result.json') or {}
        e = dict(e, peak_mem_mb=rj.get('peak_mem_mb'), listed=RUNS[n] in listed)
        eng[n] = {'load': e, 'loop': lp}
    b_med_same_session = ((eng.get('B') or {}).get('load') or {}).get('latency_ms', {}).get('run_numpy_median')
    b_med = base['B_run_numpy_median_ms']
    g_for_engine = g_drive if g_drive is not None else 5
    eng_gate = {}
    for n in B2_NAMES:
        if n in eng:
            eng_gate[n] = L.gate_engine(eng[n]['load'], b_med, g_for_engine)
            eng_gate[n]['vs_B_same_session'] = L.gate_engine(eng[n]['load'], b_med_same_session, g_for_engine) \
                if b_med_same_session else None
    car_seed = None
    if 'B2_s0' in eng_gate:
        car_seed = 'B2_s0' if eng_gate['B2_s0']['pass'] else ('B2_s1' if eng_gate.get('B2_s1', {}).get('pass') else None)
    loop_cmp = {}
    for n in eng:
        lp = eng[n]['loop'] or {}
        c = ((lp.get('drive_loop') or {}).get('cases') or {}).get('default') or {}
        loop_cmp[n] = {'infer_ms_median': c.get('infer_ms_median'), 'holds': c.get('holds'), 'driven_rows': c.get('driven_rows'),
                       'error': c.get('error'), 'loop_vs_direct_max_us': c.get('loop_vs_direct_max_us'),
                       'stats': c.get('stats'), 'pca_writes': c.get('pca_writes'),
                       'run_numpy_median': (eng[n]['load'].get('latency_ms') or {}).get('run_numpy_median'),
                       'run_numpy_p99': (eng[n]['load'].get('latency_ms') or {}).get('run_numpy_p99'),
                       'engine_median': (eng[n]['load'].get('latency_ms') or {}).get('engine_median')}
    for n in loop_cmp:
        if 'B' in loop_cmp and loop_cmp[n]['run_numpy_median'] and loop_cmp['B']['run_numpy_median']:
            dlt = loop_cmp[n]['run_numpy_median'] - loop_cmp['B']['run_numpy_median']
            loop_cmp[n]['loop_estimate_ms'] = pz.CAR_LOOP_B_MS + dlt
            loop_cmp[n]['cam_to_pwm_estimate_ms'] = pz.CAR_CAM_TO_PWM_B_MS + dlt
    res['engine'] = rnd({'checks': {n: {'accuracy': v['load'].get('accuracy'), 'latency_ms': v['load'].get('latency_ms'),
                                        'model_info': v['load'].get('model_info'), 'peak_mem_mb': v['load'].get('peak_mem_mb'),
                                        'listed': v['load'].get('listed'), 'engine_mb': v['load'].get('engine_mb')}
                                    for n, v in eng.items()},
                         'gate': eng_gate, 'genre_used_for_gate': g_for_engine, 'B_run_numpy_fixed_ms': b_med,
                         'B_run_numpy_same_session_ms': b_med_same_session, 'car_run': RUNS.get(car_seed) if car_seed else None,
                         'loop': loop_cmp,
                         'budget_note': '30 fps のカメラ 1 フレーム = 33.3 ms。走行ループ・カメラ → PWM は、B の実車の値(%.1f ms・%.0f ms)'
                                        'に同じ道具での run_numpy の差を足した見積もり(実測ではない)'
                                        % (pz.CAR_LOOP_B_MS, pz.CAR_CAM_TO_PWM_B_MS)}, 3)
    if eng_gate:
        res['car_go'] = {'desk_go': g_drive is not None, 'engine': car_seed is not None,
                         'all_machine_checks': bool(g_drive is not None and car_seed is not None),
                         'note': 'car_go = desk_go(g) ∧ G3-engine ∧ ユーザーの了承 ∧(なるべく)カード 1 が済んでいる。了承は人が決める'}
    # ---- 参考の値(判定しない)
    sec = {}
    gs = g_drive if g_drive is not None else 5
    sec['genre_for_reference'] = gs
    sec['all_passes'] = {'HAI_r1': summary(sel('r1', ('H', 'AI')), 1, gs), 'T_r1': summary(sel('r1', ('T',)), 1, gs,
                                                                                            with_last=False)}
    sec['ge6200'] = {'HAI': summary(sel('r1', ('H', 'AI'), 'vfast'), 1, gs), 'T': summary(sel('r1', ('T',), 'vfast'), 1, gs,
                                                                                          with_last=False)}
    sec['H_vs_AI_fast'] = {'H': summary(sel('r1', ('H',), 'fast'), 1, gs), 'AI': summary(sel('r1', ('AI',), 'fast'), 1, gs)}
    sec['T_dropped_vs_kept_fast'] = {
        'dropped': summary(sel('r1', ('T',), 'fast', lambda e: e['dropped_b2']), 1, gs, with_last=False),
        'kept': summary(sel('r1', ('T',), 'fast', lambda e: not e['dropped_b2']), 1, gs, with_last=False)}
    sec['windows'] = {'w04_HAI_fast': summary(sel('r1', ('H', 'AI'), 'fast'), 1, gs, win='rows_w04'),
                      'w2_HAI_fast': summary(sel('r1', ('H', 'AI'), 'fast'), 1, gs, win='rows_w2'),
                      'w04_T_fast': summary(sel('r1', ('T',), 'fast'), 1, gs, win='rows_w04', with_last=False),
                      'w2_T_fast': summary(sel('r1', ('T',), 'fast'), 1, gs, win='rows_w2', with_last=False)}
    sec['genres_HAI_fast'] = {}
    for lab, b2g, rg, ag in (('B2_aligned_vs_B_aligned', 'aligned', 'aligned', 'aligned'),
                             ('B2_own_vs_B_aligned', 'own', 'aligned', 'aligned'),
                             ('B2_g4_vs_B_g5', 4, 5, 5), ('B2_g5_vs_B_g5', 5, 5, 5), ('B2_g4_vs_B_g4', 4, 4, 4)):
        sec['genres_HAI_fast'][lab] = summary(sel('r1', ('H', 'AI'), 'fast'), 1, b2g, ref_genre=rg, a_genre=ag,
                                              with_last=False)
    sec['last_HAI_fast'] = summary(sel('r1', ('H', 'AI'), 'fast'), 1, gs, ck_b2='last', with_last=False)
    # 見積もった進入 eRPM(対応表 = protocol の erpm_map_steady)
    em = base['erpm_map_steady']
    est = {}
    for setname, sets_ in (('HAI', ('H', 'AI')), ('T', ('T',))):
        es = sel('r1', sets_)
        rows = [e['rows_w1'] for e in es]
        ent = np.array([e['entry_erpm'] for e in es], np.float64)
        u_act = np.array([np.nanmean(wt['exec'][r, 1]) if np.isfinite(wt['exec'][r, 1]).any() else np.nan for r in rows])
        ok = np.isfinite(u_act) & np.isfinite(ent)
        row = {'passes': int(ok.sum()), 'actual_share_ge5500': float(np.mean(ent[ok] >= 5500)),
               'actual_share_ge6200': float(np.mean(ent[ok] >= 6200))}
        for lab, n, g in (('B', 'B', 5), ('B2_s0', 'B2_s0', gs), ('B2_s1', 'B2_s1', gs)):
            u = L.window_means(out_of(n, 'best', g), rows, 1)
            e2 = L.entry_estimate(ent, u, u_act, em['x_us'], em['y_erpm'])[ok]
            row[lab] = {'median': float(np.median(e2)), 'share_ge5500': float(np.mean(e2 >= 5500)),
                        'share_ge6200': float(np.mean(e2 >= 6200))}
        est[setname] = row
    sec['entry_estimate'] = est
    # DE の出口の予測ステアと人の差(人の通過だけ。H と T)
    de_h = {}
    for setname in ('H', 'T'):
        es = sel('de', (setname,))
        rows = [e['rows'] for e in es]
        hum = np.array([np.nanmean(wt['exec'][r, 0]) if np.isfinite(wt['exec'][r, 0]).any() else np.nan for r in rows])
        ok = np.isfinite(hum)
        row = {'passes': int(ok.sum()), 'human_exec_steer_mean': float(np.mean(hum[ok]))}
        for lab, n, g in (('B', 'B', 5), ('B2_s0', 'B2_s0', gs), ('B2_s1', 'B2_s1', gs)):
            u = L.window_means(out_of(n, 'best', g), rows, 0)
            row[lab + '_minus_human'] = float(np.mean(u[ok] - hum[ok]))
        de_h[setname] = row
    sec['de_vs_human'] = de_h
    # SW + HS の直線(計画 G3 の字義)
    sw = np.array(wj['straight_sw_hs_rows'])
    fb = L.fwd_strength(out_of('B', 'best', 5)[sw, 1])
    sec['straight_sw_hs'] = {}
    for g in (5, 4):
        sec['straight_sw_hs']['g%d' % g] = L.gate_straight([L.fwd_strength(out_of(n, 'best', g)[sw, 1]) for n in B2_NAMES], fb)
        sec['straight_sw_hs']['g%d' % g]['frames'] = int(len(sw))
    st_rows = np.array(wj['straight_rows'])
    sec['straight_542_A'] = {n: 1.0 - L.fwd_strength(out_of(n, 'best', 5)[st_rows, 1]) / L.fwd_strength(
        out_of('B', 'best', 5)[st_rows, 1]) for n in A_NAMES}
    # ジャンル別(aligned の mode)の L1・fwd_bias・シーン別の L1
    by_genre = {}
    for gid in (4, 5):
        m = clean & (cols['mode'] == gid)
        by_genre['genre%d' % gid] = {'frames': int(m.sum())}
        for n in ('A_s0', 'A_s1', 'B') + B2_NAMES:
            p = P['own']['best'][n]
            by_genre['genre%d' % gid][n] = {'L1': float(pz.l1_rows(p, y)[m].mean()),
                                            'fwd_bias': float(np.mean(p[m, 2, 1] - y[m, 2, 1]))}
    sec['by_genre_aligned'] = by_genre
    per_scene = {}
    for s in held['human_clean_scenes']:
        m = clean & (tab['scene'] == s)
        per_scene[s] = {'frames': int(m.sum())}
        for n in ('A_s0', 'A_s1', 'B') + B2_NAMES:
            per_scene[s][n] = float(pz.l1_rows(P['own']['best'][n], y)[m].mean())
    sec['per_scene_L1_own_best'] = per_scene
    # 学習の曲線
    curves = {}
    for n in ('B', 'A_s0', 'A_s1') + B2_NAMES + ARM_NAMES:
        rj = rp._read_json(run_dir(n) / 'result.json')
        if not rj:
            continue
        curves[n] = {k: rj.get(k) for k in ('best_epoch', 'best_val_us', 'epochs_done', 'duration_s', 'peak_mem_mb',
                                             'train_samples', 'val_samples', 'train_mode_counts', 'impact_masked_frames')}
        hist = rj.get('history') or []
        if hist:
            curves[n]['last_val_us'] = hist[-1].get('val_us') if isinstance(hist[-1], dict) else None
        curves[n]['L1_primary_best_minus_last_own'] = (L1['own']['best'][n]['primary'] - L1['own']['last'][n]['primary']
                                                       if n in L1['own']['last'] and n in L1['own']['best'] else None)
    sec['training'] = curves
    # 参考の腕(drop-only・relabel-only の種 0)
    arms = {}
    for n in ARM_NAMES:
        if n not in P['own']['best'] or out_of(n, 'best', 5) is None:
            continue
        a = {}
        for g in (5, 4):
            if out_of(n, 'best', g) is None:
                continue
            a['R1_held_fast_g%d' % g] = summary(sel('r1', ('H', 'AI'), 'fast'), 1, g, arms=(n,), with_last=False)
            a['R1_train_fast_g%d' % g] = summary(sel('r1', ('T',), 'fast'), 1, g, arms=(n,), with_last=False)
        a['L1_primary_own_best'] = L1['own']['best'][n]['primary']
        a['L1_rest_own_best'] = L1['own']['best'][n]['rest']
        arms[n] = a
    sec['optional_arms'] = arms
    res['secondary'] = rnd(sec, 3)
    res['bad_refs'] = bad_ref
    # ---- 予想との答え合わせ
    g5 = res['gate_detail']['5']
    res['predictions_vs_results'] = {
        'G3-R1-held': {'predicted': proto['predictions_before_results']['G3-R1-held'], 'D_g5': g5['R1_held']['D'],
                       'D_g4': res['gate_detail']['4']['R1_held']['D'], 'pass_g5': res['gates']['5']['gates']['G3-R1-held']['pass'],
                       'pass_g4': res['gates']['4']['gates']['G3-R1-held']['pass']},
        'G3-R1-train': {'predicted': proto['predictions_before_results']['G3-R1-train'], 'D_g5': g5['R1_train']['D'],
                        'D_g4': res['gate_detail']['4']['R1_train']['D']},
        'G3-R1-slow': {'predicted': proto['predictions_before_results']['G3-R1-slow'], 'D_g5': g5['R1_slow']['D'],
                       'D_g4': res['gate_detail']['4']['R1_slow']['D']},
        'G3-DE': {'predicted': proto['predictions_before_results']['G3-DE'], 'steer_D_g5': g5['DE_steer']['D'],
                  'thr_D_g5': g5['DE_throttle']['D']},
        'G3-straight': {'predicted': proto['predictions_before_results']['G3-straight'],
                        'drop_g5': res['gates']['5']['gates']['G3-straight']['drop'],
                        'drop_g4': res['gates']['4']['gates']['G3-straight']['drop']},
        'G3-L1': {'predicted': proto['predictions_before_results']['G3-L1'],
                  'class_vs_A': res['L1']['own']['primary']['vs_A']['final_class'],
                  'worsening_best_us': res['L1']['own']['primary']['vs_A']['worsening_best_us'],
                  'rest_class_vs_A': res['L1']['own']['rest']['vs_A']['final_class']},
        'G3-intervention': {'predicted': proto['predictions_before_results']['G3-intervention'],
                            'class_vs_A': res['L1']['own']['intervention_iv_all']['vs_A']['final_class']},
        'G3-engine': {'predicted': proto['predictions_before_results']['G3-engine'],
                      'run_numpy_median': ((res['engine']['checks'].get('B2_s0') or {}).get('latency_ms') or {}).get(
                          'run_numpy_median')},
        'desk_go_any_g': {'predicted': proto['predictions_before_results']['desk_go_any_g'], 'result': g_drive}}
    return res


def cmd_evaluate(args):
    proto = load_protocol()
    res = evaluate_all(proto)
    out = REPORT / 'evaluation.json'
    rp._save_json(out, res)
    g = res['gates']
    print(json.dumps({'desk_go(5)': g['5']['desk_go'], 'failed(5)': g['5']['failed'], 'desk_go(4)': g['4']['desk_go'],
                      'failed(4)': g['4']['failed'], 'drive_genre': res['drive_genre'],
                      'R1_held_D(5)': res['gate_detail']['5']['R1_held']['D'],
                      'R1_held_D(4)': res['gate_detail']['4']['R1_held']['D'],
                      'engine': res['engine']['gate'], 'tools_changed_after_freeze': res['tools_changed_after_freeze']},
                     ensure_ascii=False, indent=1))
    print('書きました: %s' % out)


def cmd_status(args):
    wt_ok = WT.exists()
    print('窓の表: %s' % ('あり' if wt_ok else 'なし'))
    print('crosscheck: %s' % ((rp._read_json(ZOO / 'crosscheck.json') or {}).get('pass')))
    if wt_ok:
        wt, wj, gn = load_tables()
        for j in predict_jobs(include_arms=True):
            p = pred_path(*j)
            print('%-55s %s' % (p.name, 'あり' if p.exists() else '-'))
    for f in sorted(ENGINE_DIR.glob('*.json')) if ENGINE_DIR.exists() else []:
        print('engine: %s' % f.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for c in ('tables', 'repro', 'crosscheck', 'predict', 'engine', 'freeze', 'evaluate', 'status'):
        s = sub.add_parser(c)
        s.add_argument('--force', action='store_true')
        if c == 'predict':
            s.add_argument('--arms', action='store_true', help='参考の腕(drop-only・relabel-only)も予測する')
            s.add_argument('--only', nargs='*', help='この名前だけ(B2_s0 など)')
        if c == 'engine':
            s.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    {'tables': cmd_tables, 'repro': cmd_repro, 'crosscheck': cmd_crosscheck, 'predict': cmd_predict, 'engine': cmd_engine,
     'freeze': cmd_freeze, 'evaluate': cmd_evaluate, 'status': cmd_status}[args.cmd](args)


if __name__ == '__main__':
    main()
