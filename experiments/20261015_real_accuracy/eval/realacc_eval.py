#!/usr/bin/env python3
"""実機の精度の試し(real_accuracy_20261015)の評価: crosscheck・予測(GPU)・評価(CPU)。図は realacc_figures.py。

    cd ~/jetformer && jf-heavy --mem 3G --lock big --need-mb 3200 -- nice -n 10 python -u \\
        experiments/20261015_real_accuracy/eval/realacc_eval.py crosscheck        # GPU: A_s0 の best を予測し直して 10/12 と比べる
    ... realacc_eval.py predict [--names <学習結果> ...]                           # GPU: best・last の予測(出来たものは飛ばす)
    CUDA_VISIBLE_DEVICES= jf-heavy --mem 1500M --lock none -- nice -n 10 python \\
        experiments/20261015_real_accuracy/eval/realacc_eval.py evaluate          # CPU: 判定 → evaluation.json・evaluation.md

10/13 の tools/sim_cotrain_eval.py の写しを、この作業の腕・名前に合わせて直したもの(protocol.json の prediction_and_evaluation):
  - 予測するフレーム = rl_evaluate.pred_table()(key 3d339512…)の 10,985 フレーム。予測は policy_zoo._Frames・_predict_frames
    (PyTorch FP32・eval・バッチ 32・workers 3)。ジャンル = aligned の mode(rl_evaluate._genres)。ずれの試験(pert)はしない
  - A・B・10/12 の msfusion の予測は data/zoo/policy_zoo_20261012/pred を読むだけ。新しい予測は data/zoo/real_accuracy_20261015/pred
  - 判定は realacc_lib.compare(= tools/sim_cotrain.py の judge_primary・final_verdict。test_realacc_lib.py で一致を確かめる)
  - 主指標・副指標の集合と計算は protocol/baseline_secondary.py と同じ(基準 A・B の値が protocol.json と一致するかを毎回確かめる)
GPU を使うのは crosscheck・predict だけ(1 プロセスずつ。ほかの GPU の処理・管理画面の走行や学習が動いていたら始めない)。
"""

import argparse
import datetime
import gc
import hashlib
import importlib.util
import json
import os
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
import realacc_lib as L  # noqa: E402

EVAL_NAME = 'real_accuracy_20261015'
REPORT = Path('/home/jetson/jetformer_reports/real_accuracy_20261015')
PROTOCOL = REPORT / 'protocol.json'
PROTOCOL_SHA256 = '5941a6f6335a3ef95ed90ee66b7f9867f2089b8e81a103a731d15fa4208f72d8'
PREFIX = 'realacc_20261015-'
BATTERY_SCENE = '20261010-193638-0003'
CLEAN_DS = 'clean_driving_20261011'
CKPTS = ('best', 'last')
OLD = {'A_s0': 'jetformer_v0_s0', 'A_s1': 'jetformer_v0_s1', 'B': 'B', 'msf_s0': 'msfusion_s0', 'msf_s1': 'msfusion_s1'}
OLD_RUNS = {'A_s0': 'clean_driving_20261011-jetformer_v0-s0-nocmd-stride3-e30',
            'A_s1': 'clean_driving_20261011-jetformer_v0-s1-nocmd-stride3-e30',
            'B': 'clean_driving_20261011-nocmd-s3-20261011-024626',
            'msf_s0': 'clean_driving_20261011-msfusion-s0-nocmd-stride3-e30',
            'msf_s1': 'clean_driving_20261011-msfusion-s1-nocmd-stride3-e30'}
NEW = ['msf-s2', 'v0-s2', 'msf-s3', 'v0-s3', 'sizeH-s0', 'sizeH-s1', 'sizeQ-s0', 'sizeQ-s1', 'rot-s0', 'rot-s1',
       'sizeT-s0', 'sizeT-s1', 'gainl-s0', 'gainl-s1',
       'sizeHeq-s0', 'sizeHeq-s1', 'sizeQeq-s0', 'sizeQeq-s1']      # 最後の 4 本 = 固定の外の参考の腕 SIZE-EQ(deviations.md 21:55)
SETS = ('primary', 'S1_no_battery_scene', 'S3_intervention_part1', 'battery_scene_only', 'genre4', 'genre5')
METRICS = SETS + ('S2_k3_steer', 'k3_throttle')
SECONDARY_JUDGED = ('S1_no_battery_scene', 'S2_k3_steer', 'S3_intervention_part1')
CLASS_JA = {'better': '良い', 'equivalent': '同等', 'worse': '悪い', None: '—'}


def eval_dir():
    return config.DATA_DIR / 'zoo' / EVAL_NAME


def run_dir(name):
    return rp.RUNS_DIR / (OLD_RUNS[name] if name in OLD_RUNS else PREFIX + name)


def pred_file(name, ckpt='best'):
    if ckpt not in CKPTS:
        raise ValueError(ckpt)
    if name in OLD:
        return pz.pred_path(OLD[name], ckpt)
    return eval_dir() / 'pred' / ('%s%s%s.npz' % (PREFIX, name, '' if ckpt == 'best' else '_last'))


def short(name):
    return name[len(PREFIX):] if name.startswith(PREFIX) else name


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _r(x, nd=3):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    return round(xf, nd) if np.isfinite(xf) else None


def _round_tree(x, nd=4):
    if isinstance(x, dict):
        return {k: _round_tree(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round_tree(v, nd) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return _r(x, nd)
    return x


# ---------------------------------------------------------------- GPU の前の確認

def _web_busy():
    import urllib.request
    busy = []
    for path in ('/api/drive/status', '/api/train/status'):
        try:
            with urllib.request.urlopen('http://127.0.0.1:8080' + path, timeout=3) as r:
                if json.load(r).get('running'):
                    busy.append('管理画面 %s が running' % path)
        except Exception:        # noqa: BLE001
            pass
    return busy


def _check_gpu_ready():
    busy = _web_busy()
    if busy:
        raise SystemExit('始めません: %s' % busy)
    rp._check_gpu_free(min_gb=2.0)


def _table():
    held = ev._heldout()
    tab = ev.pred_table()
    want = json.loads(PROTOCOL.read_text(encoding='utf-8'))['data']['pred_table_key']
    if tab['key'] != held['pred_table_key'] or tab['key'] != want:
        raise SystemExit('予測するフレームの表が heldout.json・protocol.json と違います')
    return tab, held


def _pred_ok(path, key, ckpt):
    try:
        with np.load(path) as z:
            return (str(z['key']) == key and str(z['ckpt']) == ckpt and z['pred'].shape == (10985, 10, 2)
                    and bool(np.isfinite(z['pred']).all()))
    except Exception:          # noqa: BLE001  0 バイト・壊れた npz は作り直す
        return False


# ---------------------------------------------------------------- crosscheck・predict(GPU)

def _predict_one(d, ck, tab, batch, workers):
    import torch
    from jetformer.policies import runtime
    cfg = rp._read_json(d / 'model_config.json')
    trained = np.array([int(x) for x in cfg['trained_modes']])
    model, _, spec, _ = runtime.load_run(d, ck, device='cuda')
    total, trainable = runtime.count_params(model)
    fs = pz._Frames(tab, np.arange(len(tab['part'])), use_command=bool(cfg.get('command_input', True)))
    genres, sub = ev._genres([it[4] for it in fs.items], None, trained)
    res = pz._predict_frames(model, fs, genres, cfg['num_modes'], batch, workers)
    meta = dict(key=np.array(tab['key']), run=np.array(d.name), ckpt=np.array(ck), policy=np.array(spec.name),
                params_total=np.array(total), params_trainable=np.array(trainable))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res, genres, sub, meta


def cmd_crosscheck(args):
    out_path = eval_dir() / 'crosscheck.json'
    old = rp._read_json(out_path)
    if old and old.get('pass') and not args.force:
        print('作成済み: %s' % out_path)
        print(json.dumps(old, ensure_ascii=False, indent=1))
        return
    tab, held = _table()
    _check_gpu_ready()
    t0 = time.time()
    res, genres, _, _ = _predict_one(run_dir('A_s0'), 'best', tab, args.batch, args.workers)
    pred = res['actions'].astype(np.float64)
    with np.load(pred_file('A_s0', 'best')) as z:
        ref = z['pred'].astype(np.float64)
        same_key = str(z['key']) == tab['key']
        same_genre = bool(np.array_equal(z['genre'], genres.astype(np.int8)))
    cols, _ = ev._val_arrays(tab)
    clean = np.isin(tab['scene'], held['human_clean_scenes']) & (tab['part'] == 0) & cols['fv'] & ~cols['impact']
    mx = float(np.abs(pred - ref).max())
    out = {'name': 'A_s0', 'run': OLD_RUNS['A_s0'], 'ckpt': 'best', 'frames': int(len(pred)), 'same_key': same_key,
           'same_genre': same_genre, 'max_abs_us': _r(mx, 5), 'mean_abs_us': _r(np.abs(pred - ref).mean(), 6),
           'L_new_us': _r(L.l1_rows(pred, cols['y'])[clean].mean(), 4), 'L_1012_us': _r(L.l1_rows(ref, cols['y'])[clean].mean(), 4),
           'limit_us': 0.01, 'pass': bool(same_key and same_genre and mx <= 0.01), 'batch': args.batch, 'workers': args.workers,
           'seconds': _r(time.time() - t0, 1), 'checked_at': time.strftime('%Y-%m-%d %H:%M:%S'),
           'tool_sha256': _sha256(__file__)}
    rp._save_json(out_path, out)
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    if not out['pass']:
        raise SystemExit('crosscheck が合いません(予測を止めて調べる。deviations.md へ)')


def cmd_predict(args):
    cc = rp._read_json(eval_dir() / 'crosscheck.json')
    if not (cc and cc.get('pass')):
        raise SystemExit('先に crosscheck を通してください')
    tab, _ = _table()
    names = [short(n) for n in (args.names or [PREFIX + n for n in NEW])]
    todo = []
    for n in names:
        if n not in NEW:
            raise SystemExit('知らない学習結果: %s' % n)
        res = rp._read_json(run_dir(n) / 'result.json') or {}
        if res.get('state') != 'done':
            print('%s: 学習が済んでいないので飛ばします' % n)
            continue
        for ck in CKPTS:
            if not _pred_ok(pred_file(n, ck), tab['key'], ck):
                if ck == 'last' and not (run_dir(n) / 'last.pth').exists():
                    raise SystemExit('%s: last.pth が無いのに last の予測がありません' % n)
                todo.append((n, ck))
    if not todo:
        print('予測は作成済み')
        return
    _check_gpu_ready()
    for n, ck in todo:
        t0 = time.time()
        res, genres, sub, meta = _predict_one(run_dir(n), ck, tab, args.batch, args.workers)
        pred = res.pop('actions')
        extra = {'out_' + k: v for k, v in res.items()}
        rp._save_npz(pred_file(n, ck), pred=pred, genre=genres.astype(np.int8), substituted=sub, **extra, **meta)
        ok = _pred_ok(pred_file(n, ck), tab['key'], ck)
        print('[%s] %s(%s): %d フレーム %.0f 秒・読み直し %s' % (time.strftime('%H:%M:%S'), n, ck, len(pred), time.time() - t0,
                                                         'OK' if ok else 'NG'), flush=True)
        if not ok:
            raise SystemExit('%s の予測を読み直せません' % pred_file(n, ck))


# ---------------------------------------------------------------- GAIN の g_t(CPU)

def _probe_gain():
    spec = importlib.util.spec_from_file_location('probe_gain', str(EXP / 'ideas_data' / 'probe_gain.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gain_inputs(tab, held, W=5.0):
    """行ごとの g_t(probe_gain.py の ratio_series・running_median。part 0 = 人のきれいな 11 シーン(clean_driving の aligned)、
    part 1 = intervention_val の aligned)と、学習の 41 シーンの g_ref(確かめ用)"""
    pg = _probe_gain()
    gt = np.full(len(tab['part']), np.nan)
    root = config.DATASET_DIR
    for part, ds, scenes in ((0, CLEAN_DS, held['human_clean_scenes']),
                             (1, ev.IV_DATASET, sorted(set(tab['scene'][tab['part'] == 1].tolist())))):
        for s in scenes:
            a = dict(np.load(root / ds / 'aligned' / (s + '.npz')))
            run = pg.running_median(pg.ratio_series(a), a['timestamps_us'].astype(np.float64), W)
            rows = np.nonzero((tab['part'] == part) & (tab['scene'] == s))[0]
            gt[rows] = run[tab['frame'][rows]]
    return gt


def gain_ref_recomputed():
    pg = _probe_gain()
    root = config.DATASET_DIR / CLEAN_DS
    sp = json.loads((root / 'splits.json').read_text())
    gs = {4: [], 5: []}
    for s in sp['train']:
        a = dict(np.load(root / 'aligned' / (s + '.npz')))
        g = pg.ratio_series(a)
        for k in (4, 5):
            gs[k].append(g[(a['mode'] == k) & np.isfinite(g)])
    return {k: float(np.median(np.concatenate(v))) for k, v in gs.items()}


# ---------------------------------------------------------------- 評価(CPU)

def _load_pred(name, ck, key):
    p = pred_file(name, ck)
    if not _pred_ok(p, key, ck):
        return None
    with np.load(p) as z:
        return z['pred'].astype(np.float64)


def evaluate_all():
    proto = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    proto_sha = _sha256(PROTOCOL)
    tab, held = _table()
    cols, f = ev._val_arrays(tab)
    part, scene = tab['part'], tab['scene']
    yfin = np.isfinite(cols['y']).all(axis=(1, 2))
    clean = np.isin(scene, held['human_clean_scenes']) & (part == 0) & cols['fv'] & ~cols['impact']
    sets = {'primary': clean, 'S1_no_battery_scene': clean & (scene != BATTERY_SCENE),
            'S3_intervention_part1': (part == 1) & cols['fv'] & yfin, 'battery_scene_only': clean & (scene == BATTERY_SCENE),
            'genre4': clean & (cols['mode'] == 4), 'genre5': clean & (cols['mode'] == 5)}
    want_n = proto['secondary']['frames']
    for k, m in sets.items():
        if int(m.sum()) != int(want_n[k]):
            raise SystemExit('集合 %s のフレーム数 %d が protocol の %d と違います' % (k, int(m.sum()), want_n[k]))
    y = np.nan_to_num(cols['y'].astype(np.float64))
    key = tab['key']
    # ---- 予測(ある分だけ)
    names = list(OLD) + [n for n in NEW]
    P = {ck: {} for ck in CKPTS}
    for n in names:
        for ck in CKPTS:
            p = _load_pred(n, ck, key)
            if p is not None:
                P[ck][n] = p
    have = [n for n in names if all(n in P[ck] for ck in CKPTS)]
    missing_old = [n for n in OLD if n not in have]
    if missing_old:
        raise SystemExit('10/12 の予測が読めません: %s' % missing_old)

    def has(*ns):
        return all(n in have for n in ns)

    # ---- 後処理・組み合わせ(V = 変種の名前 → {ck: 予測})
    V = {n: {ck: P[ck][n] for ck in CKPTS} for n in have}

    def mean_of(*ns):
        return {ck: np.mean([V[n][ck] for n in ns], axis=0) for ck in CKPTS}

    combos = {'P2': ('v0-s2', 'msf-s2'), 'P3': ('v0-s3', 'msf-s3'), 'SAME_A': ('A_s0', 'A_s1'), 'SAME_new': ('v0-s2', 'v0-s3'),
              'PH0': ('A_s0', 'msf_s0'), 'PH1': ('A_s1', 'msf_s1')}
    for c, ns in combos.items():
        if has(*ns):
            V[c] = mean_of(*ns)
    # GAIN(既定: 5 秒・clip 0.8〜2.0・g_ref 固定)と参考の行
    gz = proto['arms']['Z_zero_training']['GAIN']
    g_ref = {int(k): float(v) for k, v in gz['g_ref'].items()}
    gref_row = np.array([g_ref.get(int(m), np.nan) for m in cols['mode']])
    gt5 = gain_inputs(tab, held, float(gz['window_s']))
    gt10 = gain_inputs(tab, held, 10.0)
    scale = L.gain_scale(gt5, gref_row, *gz['clip'])
    scale10 = L.gain_scale(gt10, gref_row, *gz['clip'])
    scale13 = L.gain_scale(gt5, gref_row, 0.8, 1.3)
    gain_targets = [n for n in ('A_s0', 'A_s1', 'v0-s2', 'v0-s3', 'B', 'P2', 'P3', 'gainl-s0', 'gainl-s1') if n in V]
    for n in gain_targets:
        V['GAIN(%s)' % n] = {ck: L.gain_correct(V[n][ck], scale) for ck in CKPTS}
        if n in ('A_s0', 'A_s1', 'v0-s2', 'v0-s3'):
            V['GAIN10(%s)' % n] = {ck: L.gain_correct(V[n][ck], scale10) for ck in CKPTS}
            V['GAIN13(%s)' % n] = {ck: L.gain_correct(V[n][ck], scale13) for ck in CKPTS}
    # TE(j = 2・m = 0)
    te = proto['arms']['Z_zero_training']['TE']
    prev = L.prev_index(part, scene, tab['frame'], 2)
    for n in ('A_s0', 'A_s1', 'v0-s2', 'v0-s3'):
        if n in V:
            V['TE(%s)' % n] = {ck: L.temporal_ensemble(V[n][ck], prev, 0.0) for ck in CKPTS}
    # ---- 指標
    M = {v: {ck: L.metric_values(V[v][ck], y, sets, clean) for ck in CKPTS} for v in V}

    def vals(vnames, metric='primary'):
        return {ck: [M[v][ck][metric] for v in vnames] for ck in CKPTS}

    # ---- 10/12 と同じ値か(基準)
    base = proto['secondary']['baseline_values']
    rep = {}
    ok_rep = True
    for n in ('A_s0', 'A_s1', 'B'):
        for ck in CKPTS:
            for m in METRICS:
                if m in base[ck][n]:
                    a, b = M[n][ck][m], base[ck][n][m]
                    rep['%s/%s/%s' % (n, ck, m)] = [_r(a, 3), b]
                    ok_rep &= abs(a - b) <= 0.0015
    s_a = {ck: float(proto['baseline_A']['spread_s_A'][ck]) for ck in CKPTS}
    thr_sec = proto['secondary']['thresholds_2xsA']
    s_sec = {m: {ck: float(thr_sec[m][ck]) / 2.0 for ck in CKPTS} for m in thr_sec}
    s_sec['primary'] = s_a

    def question(qid, label, arm, ref, kind='verdict', metrics=SECONDARY_JUDGED + ('battery_scene_only', 'genre4', 'genre5',
                                                                                  'k3_throttle')):
        if not all(v in V for v in arm + ref):
            return {'id': qid, 'label': label, 'arm': list(arm), 'ref': list(ref), 'pending': True, 'kind': kind}
        q = {'id': qid, 'label': label, 'arm': list(arm), 'ref': list(ref), 'pending': False, 'kind': kind,
             'primary': L.compare(vals(arm), vals(ref), s_a), 'secondary': {}}
        for m in metrics:
            q['secondary'][m] = L.compare(vals(arm, m), vals(ref, m), s_sec[m])
        q['final_class'] = q['primary']['final_class']
        q['veto_S3'] = q['secondary']['S3_intervention_part1']['final_class'] == 'worse'
        return q

    A = ('A_s0', 'A_s1')
    Q = {}
    Q['ENS_vs_A'] = question('ENS_vs_A', 'ENS(v0 + msfusion の平均・新しい種 2・3)vs A', ('P2', 'P3'), A)
    Q['SAME_vs_A'] = question('SAME_vs_A', 'SAME(同じ形 v0 の 2 本の平均)vs A', ('SAME_A', 'SAME_new'), A)
    Q['ENS_vs_SAME'] = question('ENS_vs_SAME', 'ENS vs SAME(形の違いの上乗せ)', ('P2', 'P3'), ('SAME_A', 'SAME_new'))
    Q['FRESH_A'] = question('FRESH_A', 'FRESH-A(v0 の新しい種 2・3)vs A', ('v0-s2', 'v0-s3'), A)
    Q['FRESH_MSF'] = question('FRESH_MSF', 'FRESH-MSF(msfusion の新しい種 2・3)vs A', ('msf-s2', 'msf-s3'), A)
    Q['ROT_vs_A'] = question('ROT_vs_A', 'ROT(間引きの位相を回す)vs A', ('rot-s0', 'rot-s1'), A)
    Q['GAIN_i'] = question('GAIN_i', 'GAIN 組 (i): GAIN(A_s0・A_s1)vs A_s0・A_s1', ('GAIN(A_s0)', 'GAIN(A_s1)'), A)
    Q['GAIN_ii'] = question('GAIN_ii', 'GAIN 組 (ii): GAIN(v0-s2・v0-s3)vs v0-s2・v0-s3', ('GAIN(v0-s2)', 'GAIN(v0-s3)'),
                            ('v0-s2', 'v0-s3'))
    Q['TE_i'] = question('TE_i', 'TE 組 (i): TE(A_s0・A_s1)vs A', ('TE(A_s0)', 'TE(A_s1)'), A)
    Q['TE_ii'] = question('TE_ii', 'TE 組 (ii): TE(v0-s2・v0-s3)vs v0-s2・v0-s3', ('TE(v0-s2)', 'TE(v0-s3)'), ('v0-s2', 'v0-s3'))
    Q['GAINL_vs_Apost'] = question('GAINL_vs_Apost', 'GAINL + 後処理 vs A + 後処理(任意)', ('GAIN(gainl-s0)', 'GAIN(gainl-s1)'),
                                   ('GAIN(A_s0)', 'GAIN(A_s1)'))
    ref_q = {}
    ref_q['GAINL_vs_A'] = question('GAINL_vs_A', '参考: GAINL + 後処理 vs A', ('GAIN(gainl-s0)', 'GAIN(gainl-s1)'), A, 'reference')
    ref_q['GAINL_raw_vs_A'] = question('GAINL_raw_vs_A', '参考: GAINL(後処理なし)vs A', ('gainl-s0', 'gainl-s1'), A, 'reference')
    ref_q['ENS_GAIN_vs_A'] = question('ENS_GAIN_vs_A', '参考: ENS + GAIN vs A', ('GAIN(P2)', 'GAIN(P3)'), A, 'reference')
    ref_q['GAIN10_i'] = question('GAIN10_i', '参考: GAIN 窓 10 秒(後から見た値)組 (i)', ('GAIN10(A_s0)', 'GAIN10(A_s1)'), A, 'reference')
    ref_q['GAIN13_i'] = question('GAIN13_i', '参考: GAIN clip 0.8〜1.3(実車のカードの上限)組 (i)', ('GAIN13(A_s0)', 'GAIN13(A_s1)'), A,
                                 'reference')
    ref_q['GAIN13_ii'] = question('GAIN13_ii', '参考: GAIN clip 0.8〜1.3 組 (ii)', ('GAIN13(v0-s2)', 'GAIN13(v0-s3)'),
                                  ('v0-s2', 'v0-s3'), 'reference')
    ref_q['PH_vs_A'] = question('PH_vs_A', '参考(見たあと): 下調べの組 (v0_s0+ms_s0)・(v0_s1+ms_s1)vs A', ('PH0', 'PH1'), A, 'reference')
    ref_q['SIZEH_vs_A'] = question('SIZEH_vs_A', '参考(判定しない): sizeH vs A', ('sizeH-s0', 'sizeH-s1'), A, 'reference')
    ref_q['SIZEQ_vs_A'] = question('SIZEQ_vs_A', '参考(判定しない): sizeQ vs A', ('sizeQ-s0', 'sizeQ-s1'), A, 'reference')
    ref_q['SIZEHEQ_vs_A'] = question('SIZEHEQ_vs_A', '参考(固定の外): sizeH をステップ数をそろえて学習 vs A',
                                     ('sizeHeq-s0', 'sizeHeq-s1'), A, 'reference')
    ref_q['SIZEQEQ_vs_A'] = question('SIZEQEQ_vs_A', '参考(固定の外): sizeQ をステップ数をそろえて学習 vs A',
                                     ('sizeQeq-s0', 'sizeQeq-s1'), A, 'reference')
    ref_q['SIZET_vs_A'] = question('SIZET_vs_A', '参考(判定しない): sizeT vs A', ('sizeT-s0', 'sizeT-s1'), A, 'reference')
    ref_q['MODEL_B_vs_A'] = question('MODEL_B_vs_A', '参考: モデル B vs A', ('B',), A, 'reference')
    # ---- 感度(conclusion_rules 3): FRESH-A が同等でなければ s_A' = {A_s0〜s3, B} の範囲で同じ判定
    sens = None
    if not Q['FRESH_A']['pending'] and Q['FRESH_A']['final_class'] != 'equivalent':
        s_a2 = {ck: max(M[n][ck]['primary'] for n in ('A_s0', 'A_s1', 'v0-s2', 'v0-s3', 'B'))
                - min(M[n][ck]['primary'] for n in ('A_s0', 'A_s1', 'v0-s2', 'v0-s3', 'B')) for ck in CKPTS}
        sens = {'s_A_prime_us': s_a2, 'rows': {}}
        for qid, q in Q.items():
            if not q['pending'] and q['kind'] == 'verdict':
                sens['rows'][qid] = L.compare(vals(q['arm']), vals(q['ref']), {ck: max(s_a2[ck], s_a[ck]) for ck in CKPTS})
    # ---- GAIN・TE の判定(Z_zero_training の verdict)
    def zfinal(qid, metric):
        q = Q[qid]
        if q['pending']:
            return None
        return q['primary']['final_class'] if metric == 'primary' else q['secondary'][metric]['final_class']

    gain_v = None
    if not (Q['GAIN_i']['pending'] or Q['GAIN_ii']['pending']):
        g1 = zfinal('GAIN_i', 'primary') == 'better' and zfinal('GAIN_ii', 'primary') == 'better'
        g2 = 'worse' not in (zfinal('GAIN_i', 'S1_no_battery_scene'), zfinal('GAIN_ii', 'S1_no_battery_scene'))
        g3 = 'worse' not in (zfinal('GAIN_i', 'S3_intervention_part1'), zfinal('GAIN_ii', 'S3_intervention_part1'))
        gain_v = {'G1': bool(g1), 'G2': bool(g2), 'G3': bool(g3), 'card': bool(g1 and g2 and g3),
                  'battery_scene_share_of_gain': {}}
        for qid, (a, r) in (('GAIN_i', (('GAIN(A_s0)', 'GAIN(A_s1)'), A)),
                            ('GAIN_ii', (('GAIN(v0-s2)', 'GAIN(v0-s3)'), ('v0-s2', 'v0-s3')))):
            tot = np.mean([M[x]['best']['primary'] for x in r]) - np.mean([M[x]['best']['primary'] for x in a])
            n_all, n_bat = int(sets['primary'].sum()), int(sets['battery_scene_only'].sum())
            bat = (np.mean([M[x]['best']['battery_scene_only'] for x in r])
                   - np.mean([M[x]['best']['battery_scene_only'] for x in a])) * n_bat / n_all
            gain_v['battery_scene_share_of_gain'][qid] = {'gain_us': tot, 'from_battery_scene_us': bat,
                                                          'share': bat / tot if tot else None}
    te_v = None
    if not (Q['TE_i']['pending'] or Q['TE_ii']['pending']):
        t1 = zfinal('TE_i', 'primary') == 'better' and zfinal('TE_ii', 'primary') == 'better'
        t3 = 'worse' not in (zfinal('TE_i', 'S3_intervention_part1'), zfinal('TE_ii', 'S3_intervention_part1'))
        te_v = {'T1': bool(t1), 'T3': bool(t3), 'turn_on': bool(t1 and t3)}
    # ---- SIZE
    size = None
    if has('sizeH-s0', 'sizeH-s1', 'sizeQ-s0', 'sizeQ-s1'):
        N = {'Q': 1682, 'H': 3046, 'T': 4682, 'full': 5912}
        arms = {'Q': ('sizeQ-s0', 'sizeQ-s1'), 'H': ('sizeH-s0', 'sizeH-s1'), 'full': A, 'T': ('sizeT-s0', 'sizeT-s1')}
        size = {'N': N, 'by_metric': {}}
        for metric in ('primary', 'S1_no_battery_scene'):
            for ck in CKPTS:
                Ls = {k: float(np.mean([M[x][ck][metric] for x in v])) for k, v in arms.items() if all(x in V for x in v)}
                sl = L.size_slopes(Ls['Q'], Ls['H'], Ls['full'], N['Q'], N['H'], N['full'])
                row = {'L': Ls, 'slopes': sl, 'reading': L.size_reading(sl, s_sec[metric][ck] * 2 if metric != 'primary'
                                                                        else 0.626)}
                if 'T' in Ls:
                    row['s2_T'] = (Ls['T'] - Ls['full']) / np.log2(N['full'] / N['T'])
                size['by_metric']['%s/%s' % (metric, ck)] = row
        # ブートストラップ(val 11 シーンをシーン単位で選び直す・2,000 回・種 0。2 つの種の平均)
        scenes = list(held['human_clean_scenes'])
        nsc = np.array([int((clean & (scene == s)).sum()) for s in scenes])
        sums = {}
        for k, v in arms.items():
            if k == 'T' or not all(x in V for x in v):
                continue
            per = []
            for x in v:
                rows_l1 = L.l1_rows(V[x]['best'], y)
                per.append([rows_l1[clean & (scene == s)].sum() for s in scenes])
            sums[k] = np.mean(per, axis=0)
        bs = L.scene_bootstrap(sums, nsc, 2000, 0)
        s1b = (bs['Q'] - bs['H']) / np.log2(N['H'] / N['Q'])
        s2b = (bs['H'] - bs['full']) / np.log2(N['full'] / N['H'])
        gpow = np.where(s1b > 0, s2b * np.clip(s2b / np.where(s1b > 0, s1b, 1), 0, 1), 0.0)
        pc = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
        size['bootstrap_primary_best'] = {'n_boot': 2000, 'seed': 0, 'LH_minus_Lfull_ci95': pc(bs['H'] - bs['full']),
                                          's1_ci95': pc(s1b), 'G2_lin_ci95': pc(s2b), 'G2_pow_ci95': pc(gpow),
                                          'reading_share': {k: float(np.mean([L.size_reading(
                                              {'LH_minus_Lfull': h - fl, 'G2_lin': b2, 'G2_pow': gp}) == k
                                              for h, fl, b2, gp in zip(bs['H'], bs['full'], s2b, gpow)]))
                                              for k in ('low_value', 'middle', 'high_value')}}
        main = size['by_metric']['primary/best']
        size['reading'] = main['reading']
        size['reading_text'] = L.SIZE_TEXT[main['reading']]
        # 固定の外の参考の腕 SIZE-EQ(勾配のステップ数を全部とそろえた。deviations.md 2026-10-15 21:55。読みは変えない)
        eq = {'Q': ('sizeQeq-s0', 'sizeQeq-s1'), 'H': ('sizeHeq-s0', 'sizeHeq-s1')}
        if all(x in V for x in eq['H']):
            size['equal_steps_reference'] = {'note': '固定の外の参考(結果を見た後に足した)。protocol の SIZE の読みは変えない',
                                             'epochs': {'H': 59, 'Q': 106, 'full': 30}, 'by_metric': {}}
            for metric in ('primary', 'S1_no_battery_scene'):
                for ck in CKPTS:
                    Le = {k: float(np.mean([M[x][ck][metric] for x in v])) for k, v in eq.items() if all(x in V for x in v)}
                    Le['full'] = float(np.mean([M[x][ck][metric] for x in A]))
                    row = {'L': Le, 's2_eq': (Le['H'] - Le['full']) / np.log2(N['full'] / N['H'])}
                    if 'Q' in Le:
                        row['slopes'] = L.size_slopes(Le['Q'], Le['H'], Le['full'], N['Q'], N['H'], N['full'])
                        row['reading_same_formula'] = L.size_reading(row['slopes'])
                    size['equal_steps_reference']['by_metric']['%s/%s' % (metric, ck)] = row
            s2e = size['equal_steps_reference']['by_metric']['primary/best']['s2_eq']
            size['equal_steps_reference']['s2_eq_primary_best'] = s2e
            size['equal_steps_reference']['text'] = (
                'ステップ数をそろえてもデータの量の効果が残る(s2 = %.2f µs / 2 倍)' % s2e if s2e >= 1.0 else
                'データの量の効果は、ステップ数の差で説明できる部分が大きい(ステップ数をそろえると s2 = %.2f µs / 2 倍)' % s2e)
    # ---- 参考: 4×4 の組み合わせ・シーン別・食い違い
    refs = {}
    v0s = [n for n in ('A_s0', 'A_s1', 'v0-s2', 'v0-s3') if n in V]
    msfs = [n for n in ('msf_s0', 'msf_s1', 'msf-s2', 'msf-s3') if n in V]
    pairs = {}
    for a in v0s:
        for b in msfs:
            pairs['%s+%s' % (a, b)] = {ck: L.metric_values((V[a][ck] + V[b][ck]) / 2, y, {'primary': clean}, clean)['primary']
                                       for ck in CKPTS}
    if pairs:
        refs['pairs_4x4'] = {'pairs': pairs, 'mean_best': float(np.mean([v['best'] for v in pairs.values()])),
                             'mean_last': float(np.mean([v['last'] for v in pairs.values()]))}
    scenes = list(held['human_clean_scenes'])
    per_scene = {}
    for v in ('A_s0', 'A_s1', 'B', 'P2', 'P3', 'SAME_A', 'SAME_new', 'v0-s2', 'v0-s3', 'msf-s2', 'msf-s3', 'rot-s0', 'rot-s1',
              'GAIN(A_s0)', 'GAIN(A_s1)', 'TE(A_s0)', 'TE(A_s1)'):
        if v in V:
            r = L.l1_rows(V[v]['best'], y)
            per_scene[v] = {s: float(r[clean & (scene == s)].mean()) for s in scenes}
    refs['per_scene_best'] = per_scene
    refs['per_scene_frames'] = {s: int((clean & (scene == s)).sum()) for s in scenes}
    dis = {}
    for a, b in (('A_s0', 'A_s1'), ('rot-s0', 'rot-s1'), ('v0-s2', 'v0-s3'), ('P2', 'P3'), ('msf-s2', 'msf-s3')):
        if a in V and b in V:
            dis['%s|%s' % (a, b)] = float(np.abs(V[a]['best'] - V[b]['best'])[clean].mean())
    refs['seed_disagreement_us'] = dis
    all10 = [n for n in V if n in ('A_s0', 'A_s1', 'msf_s0', 'msf_s1')]
    refs['g_ref_recomputed'] = gain_ref_recomputed()
    refs['g_ref_fixed'] = g_ref
    refs['gain_scale_by_scene'] = {s: float(np.mean(scale[clean & (scene == s)])) for s in scenes}
    refs['gain_scale_intervention_mean'] = float(np.mean(scale[sets['S3_intervention_part1']]))
    refs['gain_scale_rows_changed'] = {'primary': int((np.abs(scale[clean] - 1) > 1e-12).sum()),
                                       'S3': int((np.abs(scale[sets['S3_intervention_part1']] - 1) > 1e-12).sum())}
    # ---- 崩れ(参考。10/13 の規則)
    import compare_models as cm
    rules = pz.load_rules()
    dset = rules['parameters']['drive_settings']
    cfgs = {n: rp._read_json(run_dir(n) / 'model_config.json') for n in have}
    refpwm = {n: (float(cfgs[n]['steer_center_us']), float(cfgs[n]['throttle_neutral_us'])) for n in have}
    m0 = part == 0
    fr = cols['frow']
    stm = np.zeros(len(fr), bool)
    stm[m0] = ev.straight_mask(f['loc_code'][fr[m0]], f['gz'][fr[m0]], f['erpm'][fr[m0]], f['near_collision'][fr[m0]])

    def base_of(v):
        for n in have:
            if v == n or v.endswith('(%s)' % n):
                return n
        return {'P2': 'v0-s2', 'P3': 'v0-s3', 'SAME_A': 'A_s0', 'SAME_new': 'v0-s2', 'PH0': 'A_s0', 'PH1': 'A_s1'}.get(
            v.split('(')[-1].rstrip(')'), 'A_s0')

    def coll_stats(v):
        rf = refpwm[base_of(v)]
        o = ev.drive_outputs(V[v]['best'], rf, dset['steer_gain'], dset['throttle_gain'], dset['lookahead'], dset['steer_trim_us'])
        met = cm.metrics(V[v]['best'][clean], cols['y'][clean], cols['cur'][clean], rf[0], rf[1])
        return float((pz.NEUTRAL_US - o[stm, 1]).mean()), met['fwd_bias_us'], met['steer_sign_acc']

    CS = {}
    collapse = {'straight_frames': int(stm.sum()), 'per_question': {}}
    for qid, q in list(Q.items()) + list(ref_q.items()):
        if q['pending']:
            continue
        for v in q['arm'] + q['ref']:
            if v not in CS:
                CS[v] = coll_stats(v)
        fp = np.mean([CS[v][0] for v in q['arm']])
        frr = np.mean([CS[v][0] for v in q['ref']])
        bp = np.mean([CS[v][1] for v in q['arm']])
        sp = np.mean([CS[v][2] for v in q['arm']])
        sr = np.mean([CS[v][2] for v in q['ref']])
        c, parts, drop = pz.judge_collapse(fp, frr, bp, sp, sr)
        collapse['per_question'][qid] = {'collapse': bool(c), 'parts': parts, 'straight_drop': drop, 'fwd_strength_us': fp,
                                         'ref_fwd_strength_us': frr, 'fwd_bias_us': bp, 'sign_acc': sp, 'ref_sign_acc': sr}
    # ---- ゲート(ENS のエンジン)
    gate = rp._read_json(EXP / 'ens' / 'gate.json')
    # ---- 学習の記録
    train = {}
    for n in have:
        res = rp._read_json(run_dir(n) / 'result.json') or {}
        hist = res.get('history') or []
        train[n] = {'best_val_us': res.get('best_val_us'), 'best_epoch': res.get('best_epoch'),
                    'last_val_us': hist[-1]['val_us'] if hist else None, 'train_samples': res.get('train_samples'),
                    'duration_s': res.get('duration_s'), 'last_train_us': hist[-1]['train_us'] if hist else None}
    # ---- まとめ
    out = {'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
           'protocol': {'path': str(PROTOCOL), 'sha256': proto_sha, 'frozen_sha256': PROTOCOL_SHA256,
                        'unchanged': proto_sha == PROTOCOL_SHA256},
           'tool': {'path': __file__, 'sha256': _sha256(__file__), 'lib_sha256': _sha256(HERE / 'realacc_lib.py')},
           'crosscheck': rp._read_json(eval_dir() / 'crosscheck.json'),
           'frames': {k: int(m.sum()) for k, m in sets.items()}, 'pred_table_key': key,
           'reproduce_baseline': {'all_match': bool(ok_rep), 'values': rep},
           'have_predictions': have, 's_A_us': s_a, 'thresholds_secondary_us': thr_sec,
           'per_variant': {v: {ck: M[v][ck] for ck in CKPTS} for v in M},
           'per_run': {}, 'questions': Q, 'reference_questions': ref_q, 'sensitivity_rule3': sens,
           'gain': {'verdict': gain_v, 'g_ref': g_ref, 'window_s': gz['window_s'], 'clip': gz['clip']},
           'te': {'verdict': te_v, 'j': 2, 'm': 0.0}, 'size': size, 'reference': refs, 'collapse_reference': collapse,
           'ens_gate': gate, 'training': train}
    for n in have:
        k = n if n in OLD else PREFIX + n
        out['per_run'][k] = {'L_best': M[n]['best']['primary'], 'L_last': M[n]['last']['primary'],
                             'S1_best': M[n]['best']['S1_no_battery_scene'], 'S3_best': M[n]['best']['S3_intervention_part1']}
    out['verdicts'] = {qid: {'final_class': q.get('final_class'), 'pending': q['pending']} for qid, q in Q.items()}
    out['verdicts_table'] = [{'question': q['label'], 'best': None if q['pending'] else CLASS_JA[q['primary']['best']['class']],
                              'last': None if q['pending'] else CLASS_JA[q['primary']['last']['class']],
                              'final': '未' if q['pending'] else CLASS_JA[q['final_class']],
                              'gain_best_us': None if q['pending'] else _r(q['primary']['best']['gain_us'], 3)}
                             for q in Q.values()]
    out['conclusions'] = conclusions(out)
    # 10/16 の電源断で 2 本目の種を削った腕(GAINL・SIZE-EQ)の、種 1 本だけの参考(判定はしない。deviations.md 10/16)
    import single_seed_ref
    out['single_seed_reference'] = single_seed_ref.single_seed_reference(out['per_variant'])
    return _round_tree(out)


# ---------------------------------------------------------------- 結論(conclusion_rules)

def conclusions(res):
    Q = res['questions']
    c = {}
    e = Q['ENS_vs_A']
    if not e['pending']:
        es = Q['ENS_vs_SAME']
        if e['final_class'] == 'better':
            if not es['pending'] and es['final_class'] == 'better':
                c['ENS'] = '形の違う 2 本の平均で精度が上がった。同じ形の平均より上'
            else:
                c['ENS'] = '平均で上がった。形の違いの上乗せは言えない'
        elif e['final_class'] == 'equivalent':
            c['ENS'] = '下調べの −2.3 µs は新しい種では確かめられなかった(同等)'
        else:
            c['ENS'] = 'ENS は A より悪かった'
    fa = Q['FRESH_A']
    if not fa['pending'] and fa['final_class'] != 'equivalent':
        c['FRESH_A'] = 'A の 2 本は種のばらつきを小さく見ていた可能性(s_A\' の感度を並べる。主の判定は変えない)'
    if res['size']:
        c['SIZE'] = res['size']['reading_text']
    g = res['gain']['verdict']
    if g is not None:
        c['GAIN'] = ('G1・G2・G3 を満たす → 実車の A/B/A のカードの候補(電池が新しいとき・減ったとき)。' if g['card'] else
                     'G1〜G3 のどれかを満たさない → 実車のカードには載せない。') + '主指標の改善はほぼ 1 シーン(電池の減ったシーン)から'
    t = res['te']['verdict']
    if t is not None:
        c['TE'] = 'T1・T3 を満たす(走行の既定を見直す候補)' if t['turn_on'] else '走行の既定(オフ)のまま'
    r = Q['ROT_vs_A']
    if not r['pending']:
        c['ROT'] = {'better': '捨てていた 2/3 のフレームを回して使うと良くなった(机上の小さな改善)',
                    'equivalent': '位相を回しても同等(捨てていたフレームを使っても精度は変わらない)',
                    'worse': '位相を回すと悪くなった'}[r['final_class']]
    cands = []
    for qid in ('ENS_vs_A', 'ROT_vs_A'):
        q = Q[qid]
        if not q['pending'] and q['final_class'] == 'better' and not q['veto_S3']:
            cands.append(qid)
    if g is not None and g['card']:
        cands.append('GAIN')
    change = [x for x in cands if x != 'GAIN' and Q[x]['primary']['practical']]
    c['recommendation'] = ('走行の推奨はモデル B のまま。' + (
        '「良い」で拒否権の無い腕 %s は 10 µs 未満の机上の小さな改善なので、実車の A/B/A のカードの候補にする。' % '・'.join(cands)
        if cands else '実車で試す候補は無い。')) if not change else '実用上の差(≥ 10 µs)のある「良い」: %s(ゲート・崩れを確かめて推奨を見直す)' % change
    c['aba_candidates'] = cands
    c['open_loop'] = 'どの結論も開いたループ(保留の記録)の結果で、実車の走りの改善は示さない'
    c['multiple_comparisons'] = ('判定する問いは ENS・ROT・GAIN・TE(・GAINL)。どれも 2 つの種と best・last の両方を求めるので偽陽性は'
                                 '小さいが 0 ではない')
    return c


def cmd_evaluate(args):
    t0 = time.time()
    res = evaluate_all()
    res['seconds'] = round(time.time() - t0, 1)
    rp._save_json(eval_dir() / 'evaluation.json', res)
    rp._save_json(REPORT / 'evaluation.json', res)
    import realacc_report
    realacc_report.write_md(res, REPORT / 'evaluation.md')
    print(json.dumps({'verdicts': res['verdicts'], 'reproduce': res['reproduce_baseline']['all_match'],
                      'protocol_unchanged': res['protocol']['unchanged'], 'have': res['have_predictions'],
                      'seconds': res['seconds']}, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('predict')
    p.add_argument('--names', nargs='*', help='学習結果の名前(既定: 学習が済んだ今夜の全部)')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--workers', type=int, default=3)
    p = sub.add_parser('crosscheck')
    p.add_argument('--force', action='store_true')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--workers', type=int, default=3)
    sub.add_parser('evaluate')
    args = ap.parse_args()
    if getattr(args, 'batch', 0) > 64 or getattr(args, 'workers', 0) > 4:
        raise SystemExit('バッチは 64 以下・workers は 4 以下')
    {'predict': cmd_predict, 'crosscheck': cmd_crosscheck, 'evaluate': cmd_evaluate}[args.cmd](args)


if __name__ == '__main__':
    main()
