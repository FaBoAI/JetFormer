#!/usr/bin/env python3
"""シミュレーターと実機を混ぜた学習(2026-10-13、案 C)の評価: 予測(GPU)・評価(CPU)・図(CPU)

    cd ~/jetformer && nice -n 10 python -u tools/sim_cotrain_eval.py crosscheck   # GPU: 10/12 の v0 種 0 を同じ道具で予測し直して一致を確かめる
    cd ~/jetformer && nice -n 10 python -u tools/sim_cotrain_eval.py predict      # GPU: 6 本の best・last・ずれ + シミュレーターの 12 シーン
    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python tools/sim_cotrain_eval.py evaluate   # CPU: 判定(protocol.json の規則)→ evaluation.json
    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python tools/sim_cotrain_eval.py figures    # CPU: 図(日本語・16 pt 以上・2160×1350)

10/12 の policy zoo(tools/policy_zoo.py)の道具をそのまま使う(このファイルは名前の表と保存先だけを持つ薄い包み):
  - 予測するフレーム = rl_evaluate.pred_table()(key 3d339512…)の 10,985 フレーム。予測は policy_zoo._Frames・_predict_frames
    (PyTorch FP32・eval)。ジャンル = aligned の mode(rl_evaluate._genres)
  - 基準 A(v0 の種 0・1)とモデル B は 10/12 の予測(data/zoo/policy_zoo_20261012/pred・pert)をそのまま読む(書き換えない)
  - 新しい 6 本(C・C+FT・A+FT の種 0・1)の予測は data/zoo/sim_cotrain_20261013/ に書く
  - 判定は tools/sim_cotrain.py の judge_primary・final_verdict(protocol.json の primary)、R1・DE・崩れ・ずれは policy_zoo の関数
  - シミュレーターの L1(参考)は元のシミュレーターのデータセットの val の 12 シーン。ジャンルは one-hot で直接与える
    (頭 8 は C・C+FT だけ。A・A+FT・B はジャンル数 6 なので頭 4・5)

GPU を使うのは crosscheck・predict だけ(1 プロセスずつ。ほかの GPU の処理・管理画面の走行や学習が動いていたら始めない)。
電源が落ちても、同じコマンドをもう一度実行すれば続きから(予測は 1 本・1 種類ずつ npz に書き、出来たものは飛ばす)。
"""

import argparse
import collections
import datetime
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from jetformer import config  # noqa: E402
import policy_zoo as pz  # noqa: E402
import rl_evaluate as ev  # noqa: E402
import rl_policies as rp  # noqa: E402
import sim_cotrain as sc  # noqa: E402

EVAL_NAME = 'sim_cotrain_20261013'
REPORT_DIR = sc.REPORT_DIR
PROTOCOL_SHA256 = '91a50d410a9232528fae7e09bb647b8292edb65a9fee473c135f5319046b4f05'   # 学習の前に固定した protocol.json
CKPTS = ('best', 'last')
SIM_HEADS = (4, 5, 8)                    # シミュレーターの L1 で与えるジャンル(頭)。頭 8 はジャンル数 9 の学習結果だけ
ARMS = ('A', 'C', 'C+FT', 'A+FT')        # 'A' = 基準(10/12 の v0 の種 0・1)
ARM_KEY = {'A': 'A', 'C': 'C', 'C+FT': 'CFT', 'A+FT': 'AFT'}
ARM_LABELS = {'A': 'A(実機だけ)', 'B': 'モデル B(今の走行)', 'C': 'C(混ぜて学習)', 'C+FT': 'C+FT(C → 実機で微調整)',
              'A+FT': 'A+FT(A → 実機で微調整・対照)'}
ARM_COLORS = {'A': '#2a78d6', 'B': '#52514e', 'C': '#eb6834', 'C+FT': '#1baf7a', 'A+FT': '#8a6bd1'}
SPEED_HABIT_US = 10.0                    # protocol secondary_metrics.speed_habit
GENRE_NAMES = {4: '高速(ジャンル 4)', 5: '低速(ジャンル 5)', 8: 'シミュレーター(ジャンル 8)'}


def eval_dir():
    return config.DATA_DIR / 'zoo' / EVAL_NAME


# ---------------------------------------------------------------- 名前の表(numpy だけ。テストする)

def name_table():
    """{名前: {'arm', 'seed', 'run', 'zoo'}}。zoo = 10/12 の予測の名前(A・モデル B だけ。新しい 6 本は None)"""
    out = collections.OrderedDict()
    for s in sc.SEEDS:
        out['A_s%d' % s] = {'arm': 'A', 'seed': s, 'run': sc.run_name('A', s), 'zoo': '%s_s%d' % (pz.REF_POLICY, s)}
    out['B'] = {'arm': 'B', 'seed': None, 'run': sc.run_name('B', None), 'zoo': 'B'}
    for arm in ('C', 'C+FT', 'A+FT'):
        for s in sc.SEEDS:
            out['%s_s%d' % (ARM_KEY[arm], s)] = {'arm': arm, 'seed': s, 'run': sc.run_name(arm, s), 'zoo': None}
    return out


def arm_names(arm):
    """腕 → その腕の学習結果の名前(種の順)。'B' はモデル B だけ"""
    if arm == 'B':
        return ['B']
    if arm not in ARM_KEY:
        raise ValueError('知らない腕: %s' % arm)
    return ['%s_s%d' % (ARM_KEY[arm], s) for s in sc.SEEDS]


def new_names():
    """この評価で予測する学習結果(10/12 に予測が無いもの)"""
    return [n for n, v in name_table().items() if v['zoo'] is None]


def pred_file(name, ckpt='best'):
    """保留の 10,985 フレームの予測の npz(A・モデル B は 10/12 のもの = 読むだけ)"""
    if ckpt not in CKPTS:
        raise ValueError('ckpt は %s のどれか: %s' % (CKPTS, ckpt))
    z = name_table()[name]['zoo']
    if z is not None:
        return pz.pred_path(z, ckpt)
    return eval_dir() / 'pred' / ('%s.npz' % name if ckpt == 'best' else '%s_last.npz' % name)


def pert_file(name):
    z = name_table()[name]['zoo']
    return pz.pert_path(z) if z is not None else eval_dir() / 'pert' / ('%s.npz' % name)


def sim_file(name):
    """シミュレーターの 12 シーンの予測(全部の学習結果。この評価で作る)"""
    return eval_dir() / 'sim' / ('%s.npz' % name)


def heads_for(num_modes, heads=SIM_HEADS):
    """ジャンル数 → 与えられる頭(one-hot の長さより小さい ID だけ)"""
    return [int(h) for h in heads if int(h) < int(num_modes)]


def onehot(n, head, num_modes):
    """[n, num_modes] の one-hot(float32)"""
    out = np.zeros((int(n), int(num_modes)), np.float32)
    out[:, int(head)] = 1.0
    return out


def masked_l1(pred, y, mask):
    """[N,10,2] µs と正解 → mask の行の L1(10 ステップ × 2 の平均)の平均・ステア・スロットル・前進の偏り(3 フレーム先)"""
    m = np.asarray(mask, bool)
    p, t = np.asarray(pred, np.float64)[m], np.asarray(y, np.float64)[m]
    if not len(p):
        return {'n': 0, 'l1_us': None, 'l1_steer_us': None, 'l1_throttle_us': None, 'fwd_bias_us': None}
    d = np.abs(p - t)
    return {'n': int(len(p)), 'l1_us': float(d.mean()), 'l1_steer_us': float(d[:, :, 0].mean()),
            'l1_throttle_us': float(d[:, :, 1].mean()), 'fwd_bias_us': float((p[:, 2, 1] - t[:, 2, 1]).mean())}


def own_spread_large(vals, s_a, mult=sc.SPREAD_MULT):
    """protocol secondary_comparisons.own_spread: 腕の種の範囲 > 2 × s_A なら「ばらつきが大きい」"""
    v = np.asarray(vals, np.float64)
    return bool(float(v.max() - v.min()) > mult * float(s_a))


def compare(arm_vals, ref_vals, s_floor):
    """best・last の両方で judge_primary(spread_floor = s_A)→ final_verdict。
    arm_vals・ref_vals = {'best': [種], 'last': [種]}、s_floor = {'best': s_A, 'last': s_A(last)}"""
    out = {ck: sc.judge_primary(arm_vals[ck], ref_vals[ck], sc.SPREAD_MULT, s_floor[ck], sc.PRACTICAL_US) for ck in CKPTS}
    out['final_class'] = sc.final_verdict(out['best'], out['last'])
    out['practical'] = bool(out['best']['practical'])
    out['paired_best_us'] = [float(a - b) for a, b in zip(arm_vals['best'], ref_vals['best'])]
    out['paired_last_us'] = [float(a - b) for a, b in zip(arm_vals['last'], ref_vals['last'])]
    return out


def speed_habit(fb_arm, fb_ref, th=SPEED_HABIT_US):
    """ジャンル別の前進の偏り(種の平均どうし)のずれ。|ずれ| ≥ th なら「速さの癖の移り」"""
    shift = float(np.mean(fb_arm)) - float(np.mean(fb_ref))
    return {'fwd_bias_us': float(np.mean(fb_arm)), 'ref_fwd_bias_us': float(np.mean(fb_ref)), 'shift_us': shift,
            'moved': bool(abs(shift) >= th)}


def forgetting(l_c, l_cft, l_ref):
    """シミュレーターの L1 で、微調整が C の得を何割返したか(記述。固定した規則は無い)。
    l_c・l_cft = 頭 8 の L1(種の平均)、l_ref = 実機だけの学習の L1(A。頭 5)。
    → {'increase_us': C+FT − C, 'gain_c_us': A − C, 'share_lost': 増えた分 / C の得(得が 0 以下なら None)}"""
    inc = float(l_cft) - float(l_c)
    gain = float(l_ref) - float(l_c)
    return {'increase_us': inc, 'gain_c_us': gain, 'share_lost': (inc / gain) if gain > 0 else None}


def conclusion(cls_cft, cls_c, cls_cft_vs_aft, practical_cft=False, practical_c=False, ok_cft=True, ok_c=True):
    """protocol.json の conclusion_rules 1〜4 → {'key', 'text', 'recommend_change', 'candidate'}。
    cls_* = final_class('better'|'equivalent'|'worse')、ok_* = 崩れなし かつ ゲート合格"""
    names = {'better': '良い', 'equivalent': '同等', 'worse': '悪い'}
    head = 'C+FT は A と比べて「%s」、C は「%s」。' % (names[cls_cft], names[cls_c])
    if cls_cft == 'better':
        if cls_cft_vs_aft == 'better':
            key, body = 'sim_helped', 'シミュレーターを混ぜた事前学習 + 微調整で実機の精度が上がった。'
        else:
            key, body = 'not_separable', ('C+FT は A より良いが、A+FT との比較が「%s」なので、微調整を足した効果と区別できない'
                                          '(シミュレーターの効果とは言えない)。' % names[cls_cft_vs_aft])
    elif cls_c == 'better':
        key, body = 'mix_only', '混ぜるだけで上がった(微調整なしの C だけが良い)。'
    elif cls_cft == 'equivalent' and cls_c == 'equivalent':
        key, body = 'no_gain', 'シミュレーターを混ぜても実機の保留データの精度は上がらなかった(A と同じ)。'
    else:
        worse = [a for a, c in (('C+FT', cls_cft), ('C', cls_c)) if c == 'worse']
        key, body = 'worse', '%s は A より悪かった。精度は上がらなかった。' % '・'.join(worse)
    cand = None
    if cls_cft == 'better' and ok_cft:
        cand = 'C+FT'
    elif cls_c == 'better' and ok_c:
        cand = 'C'
    prac = {'C+FT': practical_cft, 'C': practical_c}.get(cand, False)
    rec = bool(cand and prac)
    if cand and not prac:
        body += ' 差は 10 µs 未満の小さな改善なので、推奨はモデル B のまま(実車の A/B/A の候補)。'
    elif not cand:
        body += ' 走行の推奨はモデル B のまま。'
    return {'key': key, 'text': head + body, 'recommend_change': rec, 'candidate': cand}


def _r(x, nd=3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- GPU の前の確認

def _web_busy():
    """管理画面が走行中・学習中なら、その説明(止まっていれば空)"""
    import urllib.request
    busy = []
    for path in ('/api/drive/status', '/api/train/status'):
        try:
            with urllib.request.urlopen('http://127.0.0.1:8080' + path, timeout=3) as r:
                if json.load(r).get('running'):
                    busy.append('管理画面 %s が running' % path)
        except Exception:        # noqa: BLE001  管理画面が止まっていれば ps の確認だけ
            pass
    return busy


def _check_gpu_ready():
    busy = _web_busy()
    if busy:
        raise SystemExit('始めません: %s' % busy)
    rp._check_gpu_free(min_gb=2.0)


# ---------------------------------------------------------------- シミュレーターのフレーム

class _SimFrames(object):
    """元のシミュレーターのデータセットの val の全フレーム(policy_zoo._Frames と同じ前処理)。torch の Dataset"""

    def __init__(self, use_command=False, dataset=sc.SIM_DATASET):
        from jetformer.train_data import impact_mask
        root = config.DATASET_DIR / dataset
        self.use_command = use_command
        self.scenes = list(ev._splits(dataset)['val'])
        self.items, cols = [], collections.defaultdict(list)
        for s in self.scenes:
            a = ev._aligned(dataset, s)
            imp = impact_mask(a['timestamps_us'], a['imu'], sc.TRAIN_KW['impact_g'])
            for i in range(len(a['timestamps_us'])):
                self.items.append((str(root / str(a['cam_files'][i])), a['tof'][i], a['imu'][i], a['vehicle'][i]))
            cols['scene'].append(np.full(len(imp), s))
            cols['frame'].append(np.arange(len(imp), dtype=np.int32))
            cols['y'].append(a['future_actions'].astype(np.float32))
            cols['ok'].append(a['future_valid'] & a['vehicle_valid'] & ~imp)
            cols['impact'].append(imp)
            cols['mode'].append(a['mode'].astype(np.int16))
        self.cols = {k: np.concatenate(v) for k, v in cols.items()}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        import cv2
        import torch
        from jetformer.train_data import normalize_inputs
        path, tof, imu, veh = self.items[n]
        img, t, im, st = normalize_inputs(cv2.imread(path), tof, imu, veh, self.use_command)
        return torch.from_numpy(img), torch.from_numpy(t), torch.from_numpy(im), torch.from_numpy(st)


def _predict_heads(model, fs, heads, num_modes, batch, workers):
    """1 回の読み込みで、各バッチを頭ごとに予測する(policy_zoo._predict_frames と同じ計算)→ {頭: actions µs [N,10,2]}"""
    import torch
    from torch.utils.data import DataLoader
    from jetformer.policies import runtime
    dl = DataLoader(fs, batch_size=batch, shuffle=False, num_workers=workers)
    outs = {h: [] for h in heads}
    with torch.no_grad():
        for image, tof, imu, state in dl:
            x = dict(image=image.cuda(), tof=tof.cuda(), imu=imu.cuda(), state=state.cuda())
            for h in heads:
                x['mode'] = torch.from_numpy(onehot(len(image), h, num_modes)).cuda()
                outs[h].append(runtime.forward(model, x)['actions'].float().cpu().numpy())
    return {h: (np.concatenate(v).astype(np.float32) * 500.0 + 1500.0) for h, v in outs.items()}


# ---------------------------------------------------------------- predict・crosscheck(GPU)

def _todo(names, no_pert=False):
    todo = []
    for n in names:
        if name_table()[n]['zoo'] is None:
            for ck in CKPTS:
                if not pred_file(n, ck).exists():
                    todo.append((n, 'table', ck))
            if not no_pert and not pert_file(n).exists():
                todo.append((n, 'pert', 'best'))
        if not sim_file(n).exists():
            todo.append((n, 'sim', 'best'))
    return todo


def cmd_predict(args):
    held = ev._heldout()
    tab = ev.pred_table()
    if tab['key'] != held['pred_table_key']:
        raise SystemExit('予測するフレームの表が heldout.json と違います')
    names = args.names or list(name_table())
    todo = _todo(names, args.no_pert)
    if not todo:
        print('予測は作成済み')
        return
    _check_gpu_ready()
    import torch
    from jetformer.policies import runtime
    all_rows = np.arange(len(tab['part']))
    cols = sim_fs = None
    for n in names:
        jobs = [j for j in todo if j[0] == n]
        if not jobs:
            continue
        d = rp.RUNS_DIR / name_table()[n]['run']
        cfg = rp._read_json(d / 'model_config.json')
        use_cmd = bool(cfg.get('command_input', True))
        trained = np.array([int(x) for x in cfg['trained_modes']])
        for _, kind, ck in jobs:
            t0 = time.time()
            model, _, spec, _ = runtime.load_run(d, ck, device='cuda')
            total, trainable = runtime.count_params(model)
            meta = dict(key=np.array(tab['key']), run=np.array(d.name), ckpt=np.array(ck), policy=np.array(spec.name),
                        params_total=np.array(total), params_trainable=np.array(trainable))
            if kind == 'table':
                fs = pz._Frames(tab, all_rows, use_command=use_cmd)
                genres, sub = ev._genres([it[4] for it in fs.items], None, trained)
                res = pz._predict_frames(model, fs, genres, cfg['num_modes'], args.batch, args.workers)
                pred = res.pop('actions')
                extra = {'out_' + k: v for k, v in res.items()}
                rp._save_npz(pred_file(n, ck), pred=pred, genre=genres.astype(np.int8), substituted=sub, **extra, **meta)
                msg = '%d フレーム' % len(pred)
            elif kind == 'pert':
                if cols is None:
                    cols, _ = ev._val_arrays(tab)
                human = np.isin(tab['scene'], held['human_clean_scenes']) & (tab['part'] == 0)
                clean = human & cols['fv'] & ~cols['impact']
                rows = pz.pert_rows(tab['frame'], clean)
                arrs = {}
                for pk in pz.PERTS:
                    fs = pz._Frames(tab, rows, use_command=use_cmd, pert=pk)
                    genres, _ = ev._genres([it[4] for it in fs.items], None, trained)
                    arrs['pred_' + pk] = pz._predict_frames(model, fs, genres, cfg['num_modes'], args.batch,
                                                            args.workers)['actions']
                rp._save_npz(pert_file(n), rows=rows, **arrs, **meta)
                msg = 'ずれ %d × %d フレーム' % (len(pz.PERTS), len(rows))
            else:
                if sim_fs is None or sim_fs.use_command != use_cmd:
                    sim_fs = _SimFrames(use_command=use_cmd)
                heads = heads_for(cfg['num_modes'])
                out = _predict_heads(model, sim_fs, heads, cfg['num_modes'], args.batch, args.workers)
                meta['key'] = np.array('sim_val:' + ','.join(sim_fs.scenes))
                rp._save_npz(sim_file(n), heads=np.array(heads, np.int8), scene=sim_fs.cols['scene'],
                             frame=sim_fs.cols['frame'], **{'pred_h%d' % h: v for h, v in out.items()}, **meta)
                msg = 'シミュレーター %d フレーム × 頭 %s' % (len(sim_fs), heads)
            print('[%s] %s(%s・%s): %s %.0f秒' % (time.strftime('%H:%M:%S'), n, kind, ck, msg, time.time() - t0), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()


def cmd_crosscheck(args):
    """10/12 の v0 種 0(best)を、この評価の予測の道具で作り直し、10/12 の npz と比べる(同じフレーム・同じ計算か)"""
    out_path = eval_dir() / 'crosscheck.json'
    old = rp._read_json(out_path)
    if old and not args.force:
        print('作成済み: %s' % out_path)
        print(json.dumps(old, ensure_ascii=False, indent=1))
        return
    held = ev._heldout()
    tab = ev.pred_table()
    if tab['key'] != held['pred_table_key']:
        raise SystemExit('予測するフレームの表が heldout.json と違います')
    _check_gpu_ready()
    import torch
    from jetformer.policies import runtime
    n = 'A_s0'
    d = rp.RUNS_DIR / name_table()[n]['run']
    cfg = rp._read_json(d / 'model_config.json')
    trained = np.array([int(x) for x in cfg['trained_modes']])
    t0 = time.time()
    model, _, _, _ = runtime.load_run(d, 'best', device='cuda')
    fs = pz._Frames(tab, np.arange(len(tab['part'])), use_command=bool(cfg.get('command_input', True)))
    genres, _ = ev._genres([it[4] for it in fs.items], None, trained)
    pred = pz._predict_frames(model, fs, genres, cfg['num_modes'], args.batch, args.workers)['actions'].astype(np.float64)
    with np.load(pred_file(n, 'best')) as z:
        ref = z['pred'].astype(np.float64)
        same_key = str(z['key']) == tab['key']
        same_genre = bool(np.array_equal(z['genre'], genres.astype(np.int8)))
    cols, _ = ev._val_arrays(tab)
    human = np.isin(tab['scene'], held['human_clean_scenes']) & (tab['part'] == 0)
    clean = human & cols['fv'] & ~cols['impact']
    res = {'name': n, 'run': d.name, 'ckpt': 'best', 'frames': int(len(pred)), 'same_key': same_key,
           'same_genre': same_genre, 'max_abs_us': _r(np.abs(pred - ref).max(), 4),
           'mean_abs_us': _r(np.abs(pred - ref).mean(), 5),
           'L_new_us': _r(pz.l1_rows(pred, cols['y'])[clean].mean(), 4),
           'L_1012_us': _r(pz.l1_rows(ref, cols['y'])[clean].mean(), 4), 'seconds': _r(time.time() - t0, 1),
           'checked_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    rp._save_json(out_path, res)
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- evaluate(CPU)

def _load_preds(names, key):
    preds, extras = {ck: {} for ck in CKPTS}, {}
    for n in names:
        for ck in CKPTS:
            p = pred_file(n, ck)
            if not p.exists():
                raise SystemExit('予測がありません: %s(predict を実行)' % p)
            with np.load(p) as z:
                if str(z['key']) != key:
                    raise SystemExit('%s は別のフレームの表で作られています' % p)
                preds[ck][n] = z['pred'].astype(np.float64)
                if ck == 'best':
                    extras[n] = {k: z[k] for k in z.files if k != 'pred'}
    return preds, extras


def _r1_de(names, outs, tab, cols, f, held, prm, scene, m0):
    """R1・DE(10/12 の evaluate_all と同じ計算。P = 腕の 2 つの種、R = A の 2 つの種)"""
    with open(rp.rl_dir(ev.DATASET) / 'passes.json') as fh:
        passes = json.load(fh)
    r1p, dep = ev.heldout_passes(passes, f['log'], f['timestamps_us'], f['split'])
    if len(r1p) != held['counts']['r1_passes'] or len(dep) != held['counts']['de_passes']:
        raise SystemExit('保留の通過の数が heldout.json と違います')
    prow = np.full(len(f['scene']), -1, int)
    prow[cols['frow'][m0]] = np.nonzero(m0)[0]
    nb = int(prm['bootstrap_n'])
    items = {}
    for item, ps, ch in (('r1', r1p, 1), ('de', dep, 0)):
        idx = pz.bootstrap_index(len(ps), nb, prm['bootstrap_seed'])
        wv = {ck: pz._window_values(ps, prow, outs[ck], ch) for ck in CKPTS}
        rows = wv['best'][1]

        def judge(p_names, r_names, ck='best'):
            P = np.array([wv[ck][0][x] for x in p_names])
            R = np.array([wv[ck][0][x] for x in r_names])
            d_row = (np.mean([outs[ck][x][:, ch] for x in p_names], axis=0)
                     - np.mean([outs[ck][x][:, ch] for x in r_names], axis=0))
            adj = pz._adjusted_windows(d_row, scene, m0, rows)
            return judge_pass(P, R, adj, idx, prm)

        it = {'passes': len(ps), 'ai_passes': int(sum(p['ai'] for p in ps)),
              'window_mean_us': {ck: {n: _r(float(np.mean(v)), 1) for n, v in wv[ck][0].items()} for ck in CKPTS},
              'per_arm': {}, 'per_arm_last': {}}
        for arm in ('C', 'C+FT', 'A+FT'):
            it['per_arm'][arm] = pz._round_tree(judge(arm_names(arm), arm_names('A')))
            it['per_arm_last'][arm] = pz._round_tree(judge(arm_names(arm), arm_names('A'), 'last'))
        it['CFT_vs_C'] = pz._round_tree(judge(arm_names('C+FT'), arm_names('C')))
        it['CFT_vs_AFT'] = pz._round_tree(judge(arm_names('C+FT'), arm_names('A+FT')))
        it['model_B_vs_A'] = pz._round_tree(judge(['B'], arm_names('A')))
        it['nc_1012'] = {'NC-A': pz._round_tree(judge(['A_s1'], ['A_s0', 'B'])),
                         'NC-B': pz._round_tree(judge(['A_s0'], ['A_s1', 'B']))}
        it['pass_values'] = [dict({k: p[k] for k in ('pass_id', 'log', 'ai', 'entry_erpm') if k in p},
                                  **{n: _r(float(wv['best'][0][n][j]), 1) for n in names}) for j, p in enumerate(ps)]
        items[item] = it
    return items


def judge_pass(P, R, adj, idx, prm):
    """policy_zoo.judge_pass_item に 10/12 の固定の値(99.375%・95%・4 組 ≥ 10 µs)を入れる"""
    return pz.judge_pass_item(P, R, adj, idx, prm['ci_good'], prm['ci_bad'], prm['r1_de_good_min_pair_d_us'])


def _sim_eval(names, cfgs):
    """シミュレーターの 12 シーンの L1(参考)。フレーム = future_valid & vehicle_valid & 2.5G の衝突の前後でない"""
    from jetformer.train_data import impact_mask
    ds = sc.SIM_DATASET
    scenes = list(ev._splits(ds)['val'])
    ys, oks, sc_, frs = [], [], [], []
    for s in scenes:
        a = ev._aligned(ds, s, ['timestamps_us', 'imu', 'future_actions', 'future_valid', 'vehicle_valid', 'mode'])
        imp = impact_mask(a['timestamps_us'], a['imu'], sc.TRAIN_KW['impact_g'])
        ys.append(a['future_actions'])
        oks.append(a['future_valid'] & a['vehicle_valid'] & ~imp)
        sc_.append(np.full(len(imp), s))
        frs.append(np.arange(len(imp)))
        if not (a['mode'] == sc.SIM_MODE).all():
            raise SystemExit('シミュレーターのシーン %s にジャンル 8 以外のフレームがあります' % s)
    y, ok, scene, frame = np.concatenate(ys), np.concatenate(oks), np.concatenate(sc_), np.concatenate(frs)
    out = {'dataset': ds, 'scenes': scenes, 'frames_all': int(len(y)), 'frames': int(ok.sum()),
           'impact_frames': int((~ok).sum()), 'per_run': {}, 'per_arm': {}}
    for n in names:
        p = sim_file(n)
        if not p.exists():
            raise SystemExit('シミュレーターの予測がありません: %s(predict を実行)' % p)
        with np.load(p) as z:
            if not (np.array_equal(z['scene'], scene) and np.array_equal(z['frame'], frame)):
                raise SystemExit('%s のフレームが今のシミュレーターのデータセットと合いません' % p)
            heads = [int(h) for h in z['heads']]
            row = {}
            for h in heads:
                m = masked_l1(z['pred_h%d' % h], y, ok)
                m['per_scene_l1_us'] = {s: _r(masked_l1(z['pred_h%d' % h], y, ok & (scene == s))['l1_us'], 2) for s in scenes}
                row['h%d' % h] = {k: (_r(v, 3) if isinstance(v, float) else v) for k, v in m.items()}
            row['num_modes'] = int(cfgs[n]['num_modes'])
            row['trained_modes'] = [int(x) for x in cfgs[n]['trained_modes']]
            out['per_run'][n] = row
    for arm in ARMS + ('B',):
        nm = arm_names(arm)
        row = {}
        for h in SIM_HEADS:
            key = 'h%d' % h
            if all(key in out['per_run'][x] for x in nm):
                v = [out['per_run'][x][key]['l1_us'] for x in nm]
                row[key] = {'l1_mean_us': _r(np.mean(v), 3), 'l1_seeds_us': v,
                            'fwd_bias_mean_us': _r(np.mean([out['per_run'][x][key]['fwd_bias_us'] for x in nm]), 2)}
        out['per_arm'][arm] = row
    pa = out['per_arm']
    out['forgetting'] = {
        'C_h8_us': pa['C']['h8']['l1_mean_us'], 'CFT_h8_us': pa['C+FT']['h8']['l1_mean_us'],
        'A_h5_us': pa['A']['h5']['l1_mean_us'], 'A_h4_us': pa['A']['h4']['l1_mean_us'],
        'paired_CFT_minus_C_h8_us': [_r(out['per_run'][a]['h8']['l1_us'] - out['per_run'][b]['h8']['l1_us'], 3)
                                     for a, b in zip(arm_names('C+FT'), arm_names('C'))],
        'vs_A_h5': {k: _r(v, 4) for k, v in forgetting(pa['C']['h8']['l1_mean_us'], pa['C+FT']['h8']['l1_mean_us'],
                                                        pa['A']['h5']['l1_mean_us']).items()},
        'vs_A_h4': {k: _r(v, 4) for k, v in forgetting(pa['C']['h8']['l1_mean_us'], pa['C+FT']['h8']['l1_mean_us'],
                                                        pa['A']['h4']['l1_mean_us']).items()},
        'note': ('記述(固定した規則は無い)。C・C+FT の頭 8 は学習に入ったシーン(保留ではない)。A・A+FT・B はジャンル数 6 で頭 8 が無い'
                 'ので、頭 5(低速。指示どおり)と頭 4(高速)で予測した値と比べる')}
    return out


def evaluate_all():
    import compare_models as cm
    proto_path = REPORT_DIR / 'protocol.json'
    proto = json.loads(proto_path.read_text(encoding='utf-8'))
    proto_sha = _sha256(proto_path)
    rules = pz.load_rules()
    prm = rules['parameters']
    table = name_table()
    names = list(table)
    held = ev._heldout()
    tab = ev.pred_table()
    if tab['key'] != held['pred_table_key'] or tab['key'] != proto['baseline_A']['frames']['pred_table_key']:
        raise SystemExit('予測するフレームの表が heldout.json・protocol.json と違います')
    preds, extras = _load_preds(names, tab['key'])
    cols, f = ev._val_arrays(tab)
    cfgs = {n: rp._read_json(rp.RUNS_DIR / v['run'] / 'model_config.json') for n, v in table.items()}
    resj = {n: rp._read_json(rp.RUNS_DIR / v['run'] / 'result.json') or {} for n, v in table.items()}
    refs = {n: (float(cfgs[n]['steer_center_us']), float(cfgs[n]['throttle_neutral_us'])) for n in names}
    dset = prm['drive_settings']
    outs = {ck: {n: ev.drive_outputs(preds[ck][n], refs[n], dset['steer_gain'], dset['throttle_gain'], dset['lookahead'],
                                     dset['steer_trim_us']) for n in names} for ck in CKPTS}
    m0 = tab['part'] == 0
    human = np.isin(tab['scene'], held['human_clean_scenes']) & m0
    clean = human & cols['fv'] & ~cols['impact']
    scene = tab['scene']
    res = {'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
           'protocol': {'path': str(proto_path), 'sha256': proto_sha, 'frozen_sha256': PROTOCOL_SHA256,
                        'unchanged': proto_sha == PROTOCOL_SHA256, 'written_at': proto['written_at']},
           'zoo_rules': {'path': str(pz.rules_path()), 'written_at': rules['written_at']},
           'names': {n: dict(v, label=ARM_LABELS[v['arm']],
                             pred_source=('10/12 data/zoo/%s' % pz.ZOO_NAME) if v['zoo'] else 'data/zoo/%s' % EVAL_NAME)
                     for n, v in table.items()},
           'refs_us': refs, 'drive_settings': dset,
           'frames': {'pred_table': int(len(tab['part'])), 'human_clean': int(clean.sum()), 'pred_table_key': tab['key'],
                      'human_clean_genre4': int((clean & (cols['mode'] == 4)).sum()),
                      'human_clean_genre5': int((clean & (cols['mode'] == 5)).sum())},
           'genre_substituted': {n: int(np.asarray(extras[n]['substituted']).sum()) for n in names}}
    finite = {n: bool(all(np.isfinite(preds[ck][n]).all() for ck in CKPTS)) for n in names}
    res['finite'] = finite
    cc = rp._read_json(eval_dir() / 'crosscheck.json')
    if cc:
        res['crosscheck'] = cc
    # ---- 主指標
    l1 = {ck: {n: pz.l1_rows(preds[ck][n], cols['y']) for n in names} for ck in CKPTS}
    L = {ck: {n: float(l1[ck][n][clean].mean()) for n in names} for ck in CKPTS}
    vals = {arm: {ck: [L[ck][x] for x in arm_names(arm)] for ck in CKPTS} for arm in ARMS}
    s_calc = {ck: max(L[ck][x] for x in ('A_s0', 'A_s1', 'B')) - min(L[ck][x] for x in ('A_s0', 'A_s1', 'B')) for ck in CKPTS}
    base = proto['baseline_A']
    s_a = {ck: float(base['spread_us'][ck]) for ck in CKPTS}         # 固定した s_A(protocol.json。丸めた L から計算した値)
    prim = {'L_us': {ck: {n: _r(v, 3) for n, v in L[ck].items()} for ck in CKPTS},
            's_A_us': {ck: _r(v, 3) for ck, v in s_a.items()},
            's_A_recomputed_us': {ck: _r(v, 4) for ck, v in s_calc.items()},
            'threshold_us': {ck: _r(sc.SPREAD_MULT * v, 3) for ck, v in s_a.items()},
            'arm_mean_us': {arm: {ck: _r(np.mean(vals[arm][ck]), 3) for ck in CKPTS} for arm in ARMS},
            'seed_range_us': {arm: {ck: _r(max(vals[arm][ck]) - min(vals[arm][ck]), 3) for ck in CKPTS} for arm in ARMS},
            'own_spread_large': {arm: own_spread_large(vals[arm]['best'], s_a['best']) for arm in ARMS},
            'best_minus_last_us': {n: _r(L['best'][n] - L['last'][n], 3) for n in names},
            'vs_A': {}, 'secondary': {}}
    for arm in ('C', 'C+FT', 'A+FT'):
        prim['vs_A'][arm] = pz._round_tree(compare(vals[arm], vals['A'], s_a))
        # 記述(判定は変えない): 2 つの種 × {best, last} の全部で A の 2 本より上(下)か(10/12 の consistent_direction)
        prim['vs_A'][arm]['consistent_direction'] = pz.consistent_direction(vals[arm], vals['A'])
        prim['vs_A'][arm]['diff_pct'] = _r(-100.0 * prim['vs_A'][arm]['best']['gain_us'] / prim['vs_A'][arm]['best']['ref_mean'], 2)
    # 副: 基準を C・A+FT にする(s = max(その腕の種の範囲, s_A)= judge_primary の spread_floor)
    prim['secondary']['C+FT_vs_C'] = pz._round_tree(compare(vals['C+FT'], vals['C'], s_a))
    prim['secondary']['C+FT_vs_A+FT'] = pz._round_tree(compare(vals['C+FT'], vals['A+FT'], s_a))
    prim['secondary']['note'] = ('A+FT と A の比較は vs_A の A+FT。C+FT と A+FT はどちらも A の best.pth・C の best.pth から同じ微調整'
                                 '(lr 1e-4・12 エポック)をしたもの')
    prim['per_scene_L_us'] = {n: {s: _r(l1['best'][n][clean & (scene == s)].mean(), 2) for s in held['human_clean_scenes']}
                              for n in names}
    prim['train_val_us'] = {n: resj[n].get('best_val_us') for n in names}
    res['primary'] = prim
    # ---- 10/12 と同じ値になるか(A・モデル B)
    rep = {'L_best': {n: [_r(L['best'][n], 3), base['L_us']['best'][table[n]['zoo']]] for n in ('A_s0', 'A_s1', 'B')},
           'L_last': {n: [_r(L['last'][n], 3), base['L_us']['last'][table[n]['zoo']]] for n in ('A_s0', 'A_s1', 'B')}}
    rep['s_A'] = [_r(s_calc['best'], 3), _r(base['spread_us']['best'], 3), _r(s_calc['last'], 3), _r(base['spread_us']['last'], 3)]
    # ---- ジャンル別・交代の後の人の修正(policy_zoo._accuracy と同じ)
    acc = pz._accuracy(preds, cols, tab, held, refs, l1, names)
    res['accuracy'] = acc
    hab = {}
    for arm in ('C', 'C+FT', 'A+FT'):
        hab[arm] = {}
        for g in (4, 5):
            k = 'human_clean_genre%d' % g
            hab[arm][str(g)] = speed_habit([acc['metrics'][x][k]['fwd_bias_us'] for x in arm_names(arm)],
                                           [acc['metrics'][x][k]['fwd_bias_us'] for x in arm_names('A')])
    res['speed_habit'] = pz._round_tree(hab)
    rep['per_genre'] = {n: {k: [acc['metrics'][n][k]['l1_us'], base['per_genre'][table[n]['zoo']][k]['l1_us'],
                                acc['metrics'][n][k]['fwd_bias_us'], base['per_genre'][table[n]['zoo']][k]['fwd_bias_us']]
                            for k in ('human_clean', 'human_clean_genre4', 'human_clean_genre5')}
                        for n in ('A_s0', 'A_s1', 'B')}
    # ---- R1・DE
    items = _r1_de(names, outs, tab, cols, f, held, prm, scene, m0)
    res['r1'], res['de'] = items['r1'], items['de']
    rep['r1_window_best'] = {n: [items['r1']['window_mean_us']['best'][n], base['r1_window_mean_us']['best'][table[n]['zoo']]]
                             for n in ('A_s0', 'A_s1', 'B')}
    rep['de_window_best'] = {n: [items['de']['window_mean_us']['best'][n], base['de_window_mean_us']['best'][table[n]['zoo']]]
                             for n in ('A_s0', 'A_s1', 'B')}
    rep['nc_d_mean'] = {it: {k: [items[it]['nc_1012'][k]['d_mean'], round(base['nc_d_mean_us'][it][k], 4)]
                             for k in ('NC-A', 'NC-B')} for it in ('r1', 'de')}
    # ---- 崩れ
    fr = cols['frow']
    stm = np.zeros(len(fr), bool)
    stm[m0] = ev.straight_mask(f['loc_code'][fr[m0]], f['gz'][fr[m0]], f['erpm'][fr[m0]], f['near_collision'][fr[m0]])
    met = {n: cm.metrics(preds['best'][n][clean], cols['y'][clean], cols['cur'][clean], refs[n][0], refs[n][1]) for n in names}
    fwd = {n: float((pz.NEUTRAL_US - outs['best'][n][stm, 1]).mean()) for n in names}
    col = {'straight_frames': int(stm.sum()), 'per_run': {}, 'per_arm': {}}
    for n in names:
        col['per_run'][n] = {'straight_fwd_strength_us': _r(fwd[n], 1), 'fwd_bias_us': met[n]['fwd_bias_us'],
                             'steer_sign_acc': met[n]['steer_sign_acc']}

    def collapse_of(p_names, r_names):
        fp = np.mean([fwd[x] for x in p_names])
        fr_ = np.mean([fwd[x] for x in r_names])
        bp = np.mean([met[x]['fwd_bias_us'] for x in p_names])
        sp = np.mean([met[x]['steer_sign_acc'] for x in p_names])
        sr = np.mean([met[x]['steer_sign_acc'] for x in r_names])
        c, parts, drop = pz.judge_collapse(fp, fr_, bp, sp, sr)
        return {'collapse': c, 'parts': parts, 'straight_drop': _r(drop, 4), 'fwd_strength_us': _r(fp, 1),
                'ref_fwd_strength_us': _r(fr_, 1), 'fwd_bias_us': _r(bp, 1), 'sign_acc': _r(sp, 4), 'ref_sign_acc': _r(sr, 4)}
    for arm in ('C', 'C+FT', 'A+FT'):
        col['per_arm'][arm] = collapse_of(arm_names(arm), arm_names('A'))
    col['model_B_vs_A'] = collapse_of(['B'], arm_names('A'))
    res['collapse'] = pz._round_tree(col)
    rep['straight_fwd'] = {n: [_r(fwd[n], 1), base['collapse'][table[n]['zoo']]['straight_fwd_strength_us']]
                           for n in ('A_s0', 'A_s1', 'B')}
    # ---- ずれへの強さ
    rob = {'per_run': {}, 'per_arm': {}}
    dl1, per_all, rows0 = {}, {}, None
    for n in names:
        p = pert_file(n)
        if not p.exists():
            raise SystemExit('ずれの予測がありません: %s' % p)
        with np.load(p) as z:
            rows = z['rows']
            if rows0 is None:
                rows0 = rows
            elif not np.array_equal(rows, rows0):
                raise SystemExit('%s のずれの行が 10/12 と違います' % p)
            b = pz.l1_rows(preds['best'][n][rows], cols['y'][rows])
            per = {k: float(pz.l1_rows(z['pred_' + k].astype(np.float64), cols['y'][rows]).mean() - b.mean()) for k in pz.PERTS}
        per_all[n] = per
        dl1[n] = float(np.mean(list(per.values())))
        rob['per_run'][n] = {'frames': int(len(rows)), 'L_clean_us': _r(b.mean(), 3), 'dL1_us': _r(dl1[n], 3),
                             'per_pert_dL1_us': {k: _r(v, 3) for k, v in per.items()}}
    dl1_r = float(np.mean([dl1[x] for x in arm_names('A')]))
    for arm in ('C', 'C+FT', 'A+FT'):
        v = [dl1[x] for x in arm_names(arm)]
        rob['per_arm'][arm] = {'robust_good': pz.judge_robust(v, dl1_r), 'dL1_mean_us': _r(np.mean(v), 3),
                               'dL1_seeds_us': [_r(x, 3) for x in v], 'ref_dL1_us': _r(dl1_r, 3),
                               'ratio_to_ref': _r(np.mean(v) / dl1_r, 3) if dl1_r else None,
                               'per_pert_mean_us': {k: _r(np.mean([per_all[x][k] for x in arm_names(arm)]), 3)
                                                    for k in pz.PERTS}}
    rob['per_arm']['A'] = {'dL1_mean_us': _r(dl1_r, 3), 'dL1_seeds_us': [_r(dl1[x], 3) for x in arm_names('A')],
                           'per_pert_mean_us': {k: _r(np.mean([per_all[x][k] for x in arm_names('A')]), 3) for k in pz.PERTS}}
    # 微調整でずれへの強さを失ったか(記述): C → C+FT の ΔL1 の変化(種ごと)
    rob['ft_change'] = {'CFT_minus_C_dL1_us': [_r(dl1[a] - dl1[b], 3) for a, b in zip(arm_names('C+FT'), arm_names('C'))],
                        'AFT_minus_A_dL1_us': [_r(dl1[a] - dl1[b], 3) for a, b in zip(arm_names('A+FT'), arm_names('A'))],
                        'C_robust_good': rob['per_arm']['C']['robust_good'],
                        'CFT_robust_good': rob['per_arm']['C+FT']['robust_good'],
                        'lost': bool(rob['per_arm']['C']['robust_good'] and not rob['per_arm']['C+FT']['robust_good'])}
    rob['ref_dL1_us'] = _r(dl1_r, 3)
    res['robustness'] = rob
    rep['robust_dL1'] = {n: [_r(dl1[n], 3), base['robust_dL1_us'][table[n]['zoo']]] for n in ('A_s0', 'A_s1', 'B')}
    # ---- シミュレーターの L1(参考)
    res['sim'] = _sim_eval(names, cfgs)
    # ---- ゲート・速さ(学習の段のエンジンの確認 = engine_load_check.py。A・B は 10/12 の値)
    gates, lat = {'per_run': {}, 'per_arm': {}}, {'per_run': {}}
    chk_dir = REPORT_DIR / 'train' / 'engine_check'
    rnB = base['latency']['B']['run_numpy_median_ms']
    for n in names:
        v = table[n]
        if v['zoo'] is not None:
            b = base['latency'][v['zoo']]
            lat['per_run'][n] = dict(b, source='10/12(protocol.json の baseline_A.latency)')
            g12 = ((rp._read_json(pz.zoo_dir() / 'zoo_eval.json') or {}).get('gates') or {}).get('per_run', {}).get(v['zoo'], {})
            gates['per_run'][n] = {'pass': bool(g12.get('pass')), 'actions_max_us': g12.get('actions_max_us'),
                                   'peak_mem_mb': g12.get('peak_mem_mb'), 'source': '10/12 の評価(zoo_eval.json の gates)'}
            continue
        chk = rp._read_json(chk_dir / (v['run'] + '.json'))
        ok, parts = pz.gate_of(chk, resj[n], finite[n], 'jetformer_v0')
        lm, ac = (chk or {}).get('latency_ms') or {}, (chk or {}).get('accuracy') or {}
        eng = rp.RUNS_DIR / v['run'] / 'jetformer_fp16.engine'
        same_engine = bool(chk and eng.exists() and ((chk.get('engine') or {}).get('mtime_ns') in (None, eng.stat().st_mtime_ns)))
        gates['per_run'][n] = {'pass': ok, 'parts': parts, 'actions_max_us': ac.get('actions_max_us'),
                               'actions_p99_us': ac.get('actions_p99_us'), 'peak_mem_mb': resj[n].get('peak_mem_mb'),
                               'checked_at': (chk or {}).get('checked_at'), 'same_engine_file': same_engine,
                               'source': str(chk_dir / (v['run'] + '.json'))}
        rn = lm.get('run_numpy_median')
        lat['per_run'][n] = {'run_numpy_median_ms': rn, 'run_numpy_p99_ms': lm.get('run_numpy_p99'),
                             'engine_median_ms': lm.get('engine_median'), 'engine_p99_ms': lm.get('engine_p99'),
                             'loop_estimate_ms': _r(pz.CAR_LOOP_B_MS + rn - rnB, 2) if rn is not None else None,
                             'cam_to_pwm_estimate_ms': _r(pz.CAR_CAM_TO_PWM_B_MS + rn - rnB, 1) if rn is not None else None,
                             'params_total_m': _r(float(extras[n]['params_total']) / 1e6, 2),
                             'engine_mb': (chk or {}).get('engine_mb')}
    for arm in ARMS:
        gates['per_arm'][arm] = bool(all(gates['per_run'][x]['pass'] for x in arm_names(arm)))
    lat['per_arm'] = {arm: {'run_numpy_median_ms_mean': _r(np.mean([lat['per_run'][x]['run_numpy_median_ms']
                                                                    for x in arm_names(arm)]), 3)} for arm in ARMS}
    lat['note'] = ('run_numpy・エンジンだけは同じ道具(100 回の空回しの後 600 回)。新しい 6 本は 2026-10-13 の学習の段、A・B は 10/12 に測った値'
                   '(測り直すと ~0.15 ms 揺れる)。走行ループ・カメラ → PWM はモデル B の実車の値(%.1f ms・%.0f ms)に run_numpy の差を足した'
                   '見積もりで、実測ではない' % (pz.CAR_LOOP_B_MS, pz.CAR_CAM_TO_PWM_B_MS))
    res['gates'], res['latency'] = gates, lat
    # ---- 学習(学習の段の summary.json)
    summ = rp._read_json(REPORT_DIR / 'train' / 'summary.json') or {}
    tr = {}
    for key, v in (summ.get('runs') or {}).items():
        tr[key] = {k: v.get(k) for k in ('run', 'arm', 'seed', 'epochs', 'best_val_us', 'best_epoch', 'last_val_us',
                                         'best_minus_last_us', 'epoch_s_median', 'duration_s', 'peak_mem_mb')}
    res['training'] = tr
    # ---- 10/12 と同じか(まとめ)
    ok = all(abs(a - b) <= 0.0015 for a, b in list(rep['L_best'].values()) + list(rep['L_last'].values()))
    ok &= abs(rep['s_A'][0] - rep['s_A'][1]) <= 0.002 and abs(rep['s_A'][2] - rep['s_A'][3]) <= 0.002
    ok &= all(abs(a - b) <= 0.051 for a, b in list(rep['r1_window_best'].values()) + list(rep['de_window_best'].values()))
    ok &= all(abs(a - b) <= 0.051 for a, b in rep['straight_fwd'].values())
    ok &= all(abs(a - b) <= 0.0015 for a, b in rep['robust_dL1'].values())
    ok &= all(abs(v[0] - v[1]) <= 0.0015 and abs(v[2] - v[3]) <= 0.051 for d in rep['per_genre'].values() for v in d.values())
    ok &= all(abs(a - b) <= 0.0002 for d in rep['nc_d_mean'].values() for a, b in d.values())
    rep['all_match'] = bool(ok)
    rep['note'] = '[この評価の値, 10/12 の値(protocol.json の baseline_A)]。A・モデル B は 10/12 の予測を読み直して同じ道具で計算した'
    res['reproduce_1012'] = rep
    # ---- 判定(protocol.json の conclusion_rules)
    ok_arm = {arm: bool(gates['per_arm'][arm] and not col['per_arm'][arm]['collapse']) for arm in ('C', 'C+FT', 'A+FT')}
    concl = conclusion(prim['vs_A']['C+FT']['final_class'], prim['vs_A']['C']['final_class'],
                       prim['secondary']['C+FT_vs_A+FT']['final_class'], prim['vs_A']['C+FT']['practical'],
                       prim['vs_A']['C']['practical'], ok_arm['C+FT'], ok_arm['C'])
    res['verdict'] = {'C+FT_vs_A': prim['vs_A']['C+FT']['final_class'], 'C_vs_A': prim['vs_A']['C']['final_class'],
                      'A+FT_vs_A': prim['vs_A']['A+FT']['final_class'],
                      'C+FT_vs_C': prim['secondary']['C+FT_vs_C']['final_class'],
                      'C+FT_vs_A+FT': prim['secondary']['C+FT_vs_A+FT']['final_class'],
                      'accuracy_went_up': concl['key'] in ('sim_helped', 'not_separable', 'mix_only'),
                      'sim_effect_shown': concl['key'] == 'sim_helped',
                      'gate_and_no_collapse': ok_arm, 'conclusion': concl,
                      'recommended': 'モデル B のまま' if not concl['recommend_change'] else concl['candidate']}
    return pz._round_tree(res)


def cmd_evaluate(args):
    t0 = time.time()
    res = evaluate_all()
    res['seconds'] = round(time.time() - t0, 1)
    rp._save_json(eval_dir() / 'evaluation.json', res)
    rp._save_json(Path(args.report_dir) / 'evaluation.json', res)
    print(json.dumps({'verdict': res['verdict'], 'L_best': res['primary']['L_us']['best'],
                      'reproduce_1012': res['reproduce_1012']['all_match'],
                      'protocol_unchanged': res['protocol']['unchanged']}, ensure_ascii=False, indent=1))


def cmd_figures(args):
    import sim_cotrain_figures
    res = rp._read_json(Path(args.report_dir) / 'evaluation.json')
    if not res:
        raise SystemExit('先に evaluate を実行してください')
    for p in sim_cotrain_figures.make_all(res, Path(args.report_dir) / 'fig'):
        print(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('predict')
    p.add_argument('--names', nargs='*', choices=list(name_table()), help='学習結果の名前(既定: 全部。A・B はシミュレーターだけ)')
    p.add_argument('--no-pert', action='store_true', help='ずれの試験の予測をしない')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--workers', type=int, default=3)
    p = sub.add_parser('crosscheck')
    p.add_argument('--force', action='store_true')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--workers', type=int, default=3)
    for c in ('evaluate', 'figures'):
        p = sub.add_parser(c)
        p.add_argument('--report-dir', default=str(REPORT_DIR))
    args = ap.parse_args()
    if getattr(args, 'batch', 0) > 64 or getattr(args, 'workers', 0) > 4:
        raise SystemExit('バッチは 64 以下・workers は 4 以下')
    {'predict': cmd_predict, 'crosscheck': cmd_crosscheck, 'evaluate': cmd_evaluate, 'figures': cmd_figures}[args.cmd](args)


if __name__ == '__main__':
    main()
