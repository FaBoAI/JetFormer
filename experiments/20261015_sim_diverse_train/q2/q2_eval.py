#!/usr/bin/env python3
"""Q2(実機の保留データの精度)の評価: C-DIV(実機 + いろいろな走りのシミュレーターのデータ)を A・C・モデル B と比べる

    cd ~/jetformer && jf-heavy --mem 3G --lock big --need-mb 3200 -- nice -n 10 python -u experiments/20261015_sim_diverse_train/q2/q2_eval.py predict
    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python experiments/20261015_sim_diverse_train/q2/q2_eval.py evaluate   # CPU: 判定 → eval_real.json
    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python experiments/20261015_sim_diverse_train/q2/q2_eval.py figures    # CPU: 図

10/13 の tools/sim_cotrain_eval.py の写し(orig_1013/ に元のまま置いた)に、名前の表(C-DIV)と保存先、5 モードの保留のシーンを足したもの。
判定の関数(judge_primary・final_verdict)は 10/13 の sim_cotrain.py の写し(同じフォルダー。sha256 は元と同じ)。
  - 予測するフレーム = rl_evaluate.pred_table()(key 3d339512…)の 10,985 フレーム。予測は policy_zoo._Frames・_predict_frames(PyTorch FP32・eval)
  - A(10/12 の v0 の種 0・1)・モデル B は 10/12 の予測(data/zoo/policy_zoo_20261012)、C は 10/13 の予測(data/zoo/sim_cotrain_20261013)を
    読むだけ(走らせ直さない)。C-DIV の予測は data/zoo/sim_diverse_train_20261015/ に書く
  - crosscheck: この道具で C の種 0 の best を予測し直し、10/13 の npz と同じ(最大の差 ≤ 0.01 µs)ことを確かめる
  - シミュレーターの L1(参考): (a) 前の教師のデータの val 12 シーン(10/13 と同じ。A・B・C は 10/13 の npz)、
    (b) 5 モードの見本の val 15 シーン(C-DIV の学習に入っていない = 保留。全部の学習結果をここで予測する)

GPU を使うのは predict・crosscheck だけ(jf-heavy --lock big で 1 つずつ)。電源が落ちても同じコマンドで続きから
(予測は 1 本・1 種類ずつ npz に書き、出来たもの(読めるもの)は飛ばす。0 バイト・壊れた npz は作り直す)。
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
REPO = '/home/jetson/jetformer'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tools'))
sys.path.insert(0, HERE)                     # sim_cotrain は 10/13 の写し(このフォルダー)を使う
from jetformer import config  # noqa: E402
import policy_zoo as pz  # noqa: E402
import rl_evaluate as ev  # noqa: E402
import rl_policies as rp  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location('sim_cotrain', os.path.join(HERE, 'sim_cotrain.py'))   # 10/13 の写しを名指しで読む
sc = importlib.util.module_from_spec(_spec)
sys.modules['sim_cotrain'] = sc
_spec.loader.exec_module(sc)

EVAL_NAME = 'sim_diverse_train_20261015'
REPORT_DIR = Path('/home/jetson/jetformer_reports/sim_diverse_train_20261015')
PROTO_1013 = Path('/home/jetson/jetformer_reports/sim_cotrain_20261013/protocol.json')     # baseline_A(10/12 の値)
EVAL_1013 = Path('/home/jetson/jetformer_reports/sim_cotrain_20261013/evaluation.json')    # C の値(同じになるかの確かめ)
ZOO_1013 = config.DATA_DIR / 'zoo' / 'sim_cotrain_20261013'
CKPTS = ('best', 'last')
SIM_HEADS = (4, 5, 8)
ARMS = ('A', 'C', 'C-DIV')
ARM_KEY = {'A': 'A', 'C': 'C', 'C-DIV': 'CDIV'}
ARM_LABELS = {'A': 'A(実機だけ)', 'B': 'モデル B(今の走行)', 'C': 'C(前の教師のくり返しを混ぜた・10/13)',
              'C-DIV': 'C-DIV(いろいろな走りを混ぜた・今日)'}
ARM_SHORT = {'A': 'A(実機だけ)', 'B': 'モデル B', 'C': 'C(前の教師)', 'C-DIV': 'C-DIV(いろいろな走り)'}
ARM_COLORS = {'A': '#2a78d6', 'B': '#52514e', 'C': '#eb6834', 'C-DIV': '#1baf7a'}
SPEED_HABIT_US = 10.0
RUN_CDIV = 'simdiv_20261015-CDIV-s%d'
MODE_DATASETS = ('sim-rl-fastest_20261014-1125', 'sim-rl-speed_20261014-1125', 'sim-rl-no_wall_20261015-0302',
                 'sim-rl-recover_20261015-0302', 'sim-rl-line_20261015-0302')
MODE_LABELS = {'sim-rl-fastest_20261014-1125': '最速', 'sim-rl-speed_20261014-1125': '速さいろいろ',
               'sim-rl-no_wall_20261015-0302': '絶対壁にぶつからない', 'sim-rl-recover_20261015-0302': '離脱復活',
               'sim-rl-line_20261015-0302': '経路いろいろ'}
C_BAD_SCENES = ('20261010-152707-0001', '20261010-193638-0003')     # 10/13 で C が悪かった 2 シーン


def eval_dir():
    return config.DATA_DIR / 'zoo' / EVAL_NAME


# ---------------------------------------------------------------- 名前の表

def name_table():
    """{名前: {'arm', 'seed', 'run', 'src'}}。src = 予測の出どころ('1012' = policy zoo、'1013' = 10/13、None = ここで予測)"""
    out = collections.OrderedDict()
    for s in sc.SEEDS:
        out['A_s%d' % s] = {'arm': 'A', 'seed': s, 'run': sc.run_name('A', s), 'src': '1012', 'zoo': '%s_s%d' % (pz.REF_POLICY, s)}
    out['B'] = {'arm': 'B', 'seed': None, 'run': sc.run_name('B', None), 'src': '1012', 'zoo': 'B'}
    for s in sc.SEEDS:
        out['C_s%d' % s] = {'arm': 'C', 'seed': s, 'run': sc.run_name('C', s), 'src': '1013', 'zoo': None}
    for s in sc.SEEDS:
        out['CDIV_s%d' % s] = {'arm': 'C-DIV', 'seed': s, 'run': RUN_CDIV % s, 'src': None, 'zoo': None}
    return out


def run_ready(name):
    """学習が 30 エポックまで済んでいるか(result.json の state = done)"""
    d = rp.RUNS_DIR / name_table()[name]['run']
    res = rp._read_json(d / 'result.json') or {}
    return bool(res.get('state') == 'done' and (d / 'best.pth').exists() and (d / 'last.pth').exists()
                and (d / 'model_config.json').exists())


def available_names():
    """評価に使える名前(C-DIV は学習が済んだ種だけ)"""
    return [n for n, v in name_table().items() if v['src'] is not None or run_ready(n)]


def arm_names(arm, names=None):
    if arm == 'B':
        return ['B']
    if arm not in ARM_KEY:
        raise ValueError('知らない腕: %s' % arm)
    out = ['%s_s%d' % (ARM_KEY[arm], s) for s in sc.SEEDS]
    return [n for n in out if names is None or n in names]


def pred_file(name, ckpt='best'):
    if ckpt not in CKPTS:
        raise ValueError(ckpt)
    v = name_table()[name]
    if v['src'] == '1012':
        return pz.pred_path(v['zoo'], ckpt)
    base = ZOO_1013 if v['src'] == '1013' else eval_dir()
    return base / 'pred' / ('%s.npz' % name if ckpt == 'best' else '%s_last.npz' % name)


def pert_file(name):
    v = name_table()[name]
    if v['src'] == '1012':
        return pz.pert_path(v['zoo'])
    base = ZOO_1013 if v['src'] == '1013' else eval_dir()
    return base / 'pert' / ('%s.npz' % name)


def sim_file(name):
    """(a) 前の教師のデータの val 12 シーン。A・B・C は 10/13 のもの"""
    v = name_table()[name]
    base = ZOO_1013 if v['src'] in ('1012', '1013') else eval_dir()
    return base / 'sim' / ('%s.npz' % name)


def sim5_file(name):
    """(b) 5 モードの見本の val 15 シーン(全部ここで予測する)"""
    return eval_dir() / 'sim5' / ('%s.npz' % name)


def heads_for(num_modes, heads=SIM_HEADS):
    return [int(h) for h in heads if int(h) < int(num_modes)]


def onehot(n, head, num_modes):
    out = np.zeros((int(n), int(num_modes)), np.float32)
    out[:, int(head)] = 1.0
    return out


def npz_ok(path, need=()):
    """読める npz か(0 バイト・途中で切れたものは False)"""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        with np.load(p) as z:
            files = set(z.files)
            for k in need:
                if k not in files:
                    return False
                _ = z[k].shape
        return True
    except Exception:          # noqa: BLE001
        return False


def masked_l1(pred, y, mask):
    m = np.asarray(mask, bool)
    p, t = np.asarray(pred, np.float64)[m], np.asarray(y, np.float64)[m]
    if not len(p):
        return {'n': 0, 'l1_us': None, 'l1_steer_us': None, 'l1_throttle_us': None, 'fwd_bias_us': None}
    d = np.abs(p - t)
    return {'n': int(len(p)), 'l1_us': float(d.mean()), 'l1_steer_us': float(d[:, :, 0].mean()),
            'l1_throttle_us': float(d[:, :, 1].mean()), 'fwd_bias_us': float((p[:, 2, 1] - t[:, 2, 1]).mean())}


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


def _wait_gpu_ready(max_wait_s=1800):
    """ほかの GPU の処理(学習など)・管理画面の走行や学習が無くなるまで待つ(lock big の中なので、普通は待たない)"""
    t0 = time.time()
    said = None
    while True:
        busy = _web_busy() + ['%d %s' % j for j in rp.gpu_jobs()]
        if not busy and rp.mem_available_gb() >= 2.0:
            return
        why = str(busy) if busy else '空きメモリ %.1f GB' % rp.mem_available_gb()
        if why != said:
            print('[%s] 待ちます: %s' % (time.strftime('%H:%M:%S'), why), flush=True)
            said = why
        if time.time() - t0 > max_wait_s:
            raise SystemExit('GPU が空かないので止めます: %s' % why)
        time.sleep(15)


# ---------------------------------------------------------------- シミュレーターのフレーム

class _SimFrames(object):
    """シミュレーターのデータセット(複数可)の val の全フレーム(policy_zoo._Frames と同じ前処理)。torch の Dataset。
    ok = future_valid & vehicle_valid & 2.5G の衝突の前後でない & ジャンル 8(離脱復活の外したフレームはジャンル 0)"""

    def __init__(self, datasets, use_command=False):
        from jetformer.train_data import impact_mask
        self.use_command = use_command
        self.datasets = list(datasets)
        self.scenes = []
        self.items, cols = [], collections.defaultdict(list)
        for ds in self.datasets:
            root = config.DATASET_DIR / ds
            for s in ev._splits(ds)['val']:
                self.scenes.append('%s/%s' % (ds, s))
                a = ev._aligned(ds, s)
                imp = impact_mask(a['timestamps_us'], a['imu'], sc.TRAIN_KW['impact_g'])
                for i in range(len(a['timestamps_us'])):
                    self.items.append((str(root / str(a['cam_files'][i])), a['tof'][i], a['imu'][i], a['vehicle'][i]))
                n = len(imp)
                cols['dataset'].append(np.full(n, ds))
                cols['scene'].append(np.full(n, s))
                cols['frame'].append(np.arange(n, dtype=np.int32))
                cols['y'].append(a['future_actions'].astype(np.float32))
                cols['ok'].append(a['future_valid'] & a['vehicle_valid'] & ~imp & (a['mode'] == sc.SIM_MODE))
                cols['impact'].append(imp)
                cols['mode'].append(a['mode'].astype(np.int16))
        self.cols = {k: np.concatenate(v) for k, v in cols.items()}

    def key(self):
        return 'sim_val:' + ','.join(self.scenes)

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
    """1 回の読み込みで、各バッチを頭ごとに予測する(10/13 と同じ計算)→ {頭: actions µs [N,10,2]}"""
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

def _todo(names, kinds):
    todo = []
    for n in names:
        new = name_table()[n]['src'] is None
        if new and not run_ready(n):
            print('%s: 学習がまだ済んでいないので飛ばします' % n, flush=True)
            continue
        if new and 'table' in kinds:
            for ck in CKPTS:
                if not npz_ok(pred_file(n, ck), ('pred', 'genre', 'key')):
                    todo.append((n, 'table', ck))
        if new and 'pert' in kinds and not npz_ok(pert_file(n), ['rows'] + ['pred_' + k for k in pz.PERTS]):
            todo.append((n, 'pert', 'best'))
        if new and 'sim' in kinds and not npz_ok(sim_file(n), ('heads', 'scene', 'frame')):
            todo.append((n, 'sim', 'best'))
        if 'sim5' in kinds and not npz_ok(sim5_file(n), ('heads', 'scene', 'frame', 'dataset')):
            todo.append((n, 'sim5', 'best'))
    return todo


def cmd_predict(args):
    held = ev._heldout()
    tab = ev.pred_table()
    if tab['key'] != held['pred_table_key']:
        raise SystemExit('予測するフレームの表が heldout.json と違います')
    names = args.names or list(name_table())
    kinds = args.kinds or ['table', 'pert', 'sim', 'sim5']
    todo = _todo(names, kinds)
    if not todo:
        print('予測は作成済み', flush=True)
        return
    print('[%s] 予測: %s' % (time.strftime('%H:%M:%S'), todo), flush=True)
    _wait_gpu_ready()
    import torch
    from jetformer.policies import runtime
    all_rows = np.arange(len(tab['part']))
    cols = sim_fs = sim5_fs = None
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
            elif kind == 'sim':
                if sim_fs is None or sim_fs.use_command != use_cmd:
                    sim_fs = _SimFrames([sc.SIM_DATASET], use_command=use_cmd)
                heads = heads_for(cfg['num_modes'])
                out = _predict_heads(model, sim_fs, heads, cfg['num_modes'], args.batch, args.workers)
                meta['key'] = np.array('sim_val:' + ','.join(ev._splits(sc.SIM_DATASET)['val']))
                rp._save_npz(sim_file(n), heads=np.array(heads, np.int8), scene=sim_fs.cols['scene'],
                             frame=sim_fs.cols['frame'], **{'pred_h%d' % h: v for h, v in out.items()}, **meta)
                msg = '前の教師の val %d フレーム × 頭 %s' % (len(sim_fs), heads)
            else:
                if sim5_fs is None or sim5_fs.use_command != use_cmd:
                    sim5_fs = _SimFrames(MODE_DATASETS, use_command=use_cmd)
                heads = heads_for(cfg['num_modes'])
                out = _predict_heads(model, sim5_fs, heads, cfg['num_modes'], args.batch, args.workers)
                meta['key'] = np.array(sim5_fs.key())
                rp._save_npz(sim5_file(n), heads=np.array(heads, np.int8), dataset=sim5_fs.cols['dataset'],
                             scene=sim5_fs.cols['scene'], frame=sim5_fs.cols['frame'],
                             **{'pred_h%d' % h: v for h, v in out.items()}, **meta)
                msg = '5 モードの val %d フレーム × 頭 %s' % (len(sim5_fs), heads)
            print('[%s] %s(%s・%s): %s %.0f秒' % (time.strftime('%H:%M:%S'), n, kind, ck, msg, time.time() - t0), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()


def cmd_crosscheck(args):
    """C の種 0(best)を、この道具で予測し直し、10/13 の npz と比べる(同じフレーム・同じ計算か)。≤ 0.01 µs で合格"""
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
    _wait_gpu_ready()
    import torch
    from jetformer.policies import runtime
    n = 'C_s0'
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
        same_run = str(z['run']) == d.name
    cols, _ = ev._val_arrays(tab)
    human = np.isin(tab['scene'], held['human_clean_scenes']) & (tab['part'] == 0)
    clean = human & cols['fv'] & ~cols['impact']
    mx = float(np.abs(pred - ref).max())
    res = {'name': n, 'run': d.name, 'ckpt': 'best', 'frames': int(len(pred)), 'same_key': same_key,
           'same_genre': same_genre, 'same_run': same_run, 'max_abs_us': _r(mx, 5),
           'mean_abs_us': _r(np.abs(pred - ref).mean(), 6),
           'L_new_us': _r(pz.l1_rows(pred, cols['y'])[clean].mean(), 4),
           'L_1013_us': _r(pz.l1_rows(ref, cols['y'])[clean].mean(), 4),
           'tolerance_us': 0.01, 'pass': bool(mx <= 0.01 and same_key and same_genre and same_run),
           'seconds': _r(time.time() - t0, 1), 'checked_at': time.strftime('%Y-%m-%d %H:%M:%S'),
           'batch': args.batch, 'workers': args.workers}
    rp._save_json(out_path, res)
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if not res['pass']:
        raise SystemExit('crosscheck が不合格です(C-DIV の予測は始めません)')


def cmd_gpu(args):
    """GPU の 1 回分(lock big の中で): crosscheck → C-DIV の予測 → (--sim5-all なら)A・B・C の 5 モードの val"""
    cc = rp._read_json(eval_dir() / 'crosscheck.json')
    if not cc or not cc.get('pass'):
        a = argparse.Namespace(force=bool(cc), batch=args.batch, workers=args.workers)
        cmd_crosscheck(a)
    names = [n for n in name_table() if name_table()[n]['src'] is None]
    cmd_predict(argparse.Namespace(names=names, kinds=None, batch=args.batch, workers=args.workers))
    if args.sim5_all:
        others = [n for n in name_table() if name_table()[n]['src'] is not None]
        cmd_predict(argparse.Namespace(names=others, kinds=['sim5'], batch=args.batch, workers=args.workers))
    print('[%s] GPU の分は済みました' % time.strftime('%H:%M:%S'), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('predict')
    p.add_argument('--names', nargs='*', choices=list(name_table()))
    p.add_argument('--kinds', nargs='*', choices=['table', 'pert', 'sim', 'sim5'])
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--workers', type=int, default=2)
    p = sub.add_parser('crosscheck')
    p.add_argument('--force', action='store_true')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--workers', type=int, default=2)
    p = sub.add_parser('gpu')
    p.add_argument('--sim5-all', action='store_true')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--workers', type=int, default=2)
    for c in ('evaluate', 'figures', 'status'):
        sub.add_parser(c)
    args = ap.parse_args()
    if getattr(args, 'batch', 0) > 64 or getattr(args, 'workers', 0) > 4:
        raise SystemExit('バッチは 64 以下・workers は 4 以下')
    if args.cmd in ('evaluate', 'figures', 'status'):
        import q2_evaluate
        return {'evaluate': q2_evaluate.cmd_evaluate, 'figures': q2_evaluate.cmd_figures,
                'status': q2_evaluate.cmd_status}[args.cmd](args)
    {'predict': cmd_predict, 'crosscheck': cmd_crosscheck, 'gpu': cmd_gpu}[args.cmd](args)


if __name__ == '__main__':
    main()
