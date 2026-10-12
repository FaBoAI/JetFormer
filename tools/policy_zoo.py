#!/usr/bin/env python3
"""方策の比較(policy zoo)の評価: 保留データの予測・エンジンの確認・評価・図(spec.md §4.5・§5、results/zoo_rules.json)

    python tools/policy_zoo.py plan                       # 計画と、固定した規則(zoo_rules.json)を表示する(書かない)
    python tools/policy_zoo.py predict                    # GPU: 10 本 + モデル B の予測(best・last・ずれの試験)。出来たものは飛ばす
    python tools/policy_zoo.py crosscheck                 # GPU: tools/compare_models.py の predict_scene(走行と同じ手順)と同じ予測か
    python tools/policy_zoo.py engine-check --names B ... # GPU: 1 本ずつ別のプロセスで、走行と同じ手順でエンジンを読み、精度・速さを測る
    python tools/policy_zoo.py evaluate [--report-dir D]  # CPU: ゲート・主指標・副指標・陰性対照・tcp の追加・判定 → zoo_eval.json
    python tools/policy_zoo.py figures --fig-dir D        # CPU: 図(日本語・大きい文字)
    python tools/policy_zoo.py recommend                  # CPU: 評価の推奨のエンジンを走行タブの「推奨のモデル」にする
                                                          #      (data/drive_recommended.json。走行タブが最初に選び、一覧の先頭に出す)

判定の規則は data/zoo/policy_zoo_20261012/zoo_rules.json(保留データの予測の前に固定。spec §5 + 逸脱 D1〜D13)に従い、
このファイルでは変えない。計算の仕方は eval_protocol.json(予測の前に書いた)。
フレームは tools/rl_evaluate.py の保留の表(pred_table。heldout.json の pred_table_key と一致を確かめる)と同じ。
予測は PyTorch FP32(best.pth・last.pth)。tcp は操作の枝(actions)で判定する(走行の既定 tcp_mode 0 と同じ出力)。

GPU を使うのは predict・crosscheck・engine-check だけ(1つずつ。ほかの GPU の処理が動いていたら始めない)。
電源が落ちても、同じコマンドをもう一度実行すれば続きから(予測は1本・1種類ずつ npz に書き、出来たものは飛ばす)。
"""

import argparse
import collections
import datetime
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from jetformer import config  # noqa: E402
import rl_evaluate as ev  # noqa: E402
import rl_policies as rp  # noqa: E402

ZOO_NAME = 'policy_zoo_20261012'
POLICIES = ['jetformer_v0', 'act', 'tcp', 'msfusion', 'dinov2']
REF_POLICY = 'jetformer_v0'
SEEDS = (0, 1)
CKPTS = ('best', 'last')
PERTS = ('img_down3', 'img_up3', 'bright_0.8', 'bright_1.2', 'tof_down1')
PERT_LABELS = {'img_down3': '画像を 3 行下へ', 'img_up3': '画像を 3 行上へ', 'bright_0.8': '明るさ ×0.8',
               'bright_1.2': '明るさ ×1.2', 'tof_down1': 'ToF を 1 行下へ'}
CROSS_SCENE = '20261010-202633-0003'
D8_BOUNDARIES = (('20261010-152405-0002', 'start'), ('20261010-152405-0002', 'end'), ('20261010-152707-0001', 'end'))
D8_SECONDS = 1.0
NEUTRAL_US = ev.NEUTRAL_US            # 前進の強さ(1498 − 出力のスロットル)の基準
GATES = {'actions_max_us': 2.5, 'waypoints_max_m': 0.02, 'speed_exp_max_mps': 0.05, 'run_numpy_median_ms': 6.0,
         'run_numpy_p99_ms': 8.0, 'peak_mem_mb': 4096.0}
COLLAPSE = {'straight_drop_max': 0.05, 'fwd_bias_min_us': -30.0, 'sign_acc_margin': 0.01}
ROBUST_RATIO = 0.8
# 走行ループの見積もり(参考。実測ではない): モデル B の実車の値 + 同じ道具で測った run_numpy の差
CAR_LOOP_B_MS = 5.6
CAR_CAM_TO_PWM_B_MS = 28.0
# tcp の §5.5 の出し方(名前 → (tcp_mode, speed_cap m/s))
TCP_SETTINGS = collections.OrderedDict([('control', (0, 0.0)), ('control_cap', (0, 2.75)), ('hybrid', (1, 0.0)),
                                        ('hybrid_cap', (1, 2.75)), ('trajectory', (3, 0.0))])
LABELS = {'B': 'モデル B(今の走行)', 'jetformer_v0': 'TransFuser軽量型(JetFormer v0)', 'act': 'ACT型', 'tcp': 'TCP型',
          'msfusion': 'TransFuser多段型', 'dinov2': 'DINOv2型'}
COLORS = {'B': '#52514e', 'jetformer_v0': '#2a78d6', 'act': '#eb6834', 'tcp': '#1baf7a', 'msfusion': '#eda100',
          'dinov2': '#e87ba4'}


def zoo_dir():
    return config.DATA_DIR / 'zoo' / ZOO_NAME


def rules_path():
    return zoo_dir() / 'zoo_rules.json'


def load_rules(path=None):
    p = Path(path or rules_path())
    if not p.exists():
        raise SystemExit('判定の規則がありません: %s(results/write_zoo_rules.py で予測の前に書く)' % p)
    return json.loads(p.read_text(encoding='utf-8'))


def run_names(rules):
    """{'B': モデル B, '<方策>_s<種>': 学習結果}(B が最初、あとは方策・種の順)"""
    out = collections.OrderedDict([('B', rules['runs']['model_B'])])
    for p in POLICIES:
        for s in SEEDS:
            out['%s_s%d' % (p, s)] = rules['runs']['policy']['%s_s%d' % (p, s)]
    return out


def policy_of(name):
    """'act_s1' → 'act'、'B' → 'B'"""
    return name if name == 'B' else name.rsplit('_s', 1)[0]


def seed_names(policy):
    return ['%s_s%d' % (policy, s) for s in SEEDS]


def pred_path(name, ckpt='best'):
    if ckpt not in CKPTS:
        raise ValueError('ckpt は %s のどれか: %s' % (CKPTS, ckpt))
    return zoo_dir() / 'pred' / ('%s.npz' % name if ckpt == 'best' else '%s_last.npz' % name)


def pert_path(name):
    return zoo_dir() / 'pert' / ('%s.npz' % name)


def engine_json(name):
    return zoo_dir() / 'engine' / ('%s.json' % name)


def _r(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# ---------------------------------------------------------------- numpy だけの規則(テストする)

def perturb(kind, img, tof):
    """ずれの試験(spec §5.3)。img = BGR uint8 [H,W,3](224×224)、tof = ToF の生の値 [42,54](uint16)→ (img, tof) の写し。
    img_down3: 3 行下へ(上の 3 行は元の 1 行目を複製)、img_up3: 3 行上へ(下の 3 行は最後の行を複製)、
    bright_0.8・bright_1.2: float で掛けて四捨五入し 0〜255 に切る、tof_down1: 1 行下へ(上の 1 行は 0 = 無効)"""
    img = np.asarray(img)
    tof = np.asarray(tof)
    if kind == 'img_down3':
        return np.concatenate([np.repeat(img[:1], 3, axis=0), img[:-3]], axis=0), tof.copy()
    if kind == 'img_up3':
        return np.concatenate([img[3:], np.repeat(img[-1:], 3, axis=0)], axis=0), tof.copy()
    if kind in ('bright_0.8', 'bright_1.2'):
        f = float(kind.split('_')[1])
        return np.clip(np.rint(img.astype(np.float32) * f), 0, 255).astype(np.uint8), tof.copy()
    if kind == 'tof_down1':
        out = np.zeros_like(tof)
        out[1:] = tof[:-1]
        return img.copy(), out
    raise ValueError('知らないずれ: %s' % kind)


def pert_rows(frame, clean):
    """ずれの試験の行: 主指標のフレーム(clean)のうちフレーム番号 % 3 == 0"""
    return np.nonzero(np.asarray(clean, bool) & (np.asarray(frame) % 3 == 0))[0]


def l1_rows(pred, y):
    """[N,10,2] µs → 行ごとの L1(10 ステップ × 2 の平均)[N]"""
    return np.abs(np.asarray(pred, np.float64) - np.nan_to_num(np.asarray(y, np.float64))).mean(axis=(1, 2))


def delta_of(values, floor=10.0):
    """δ = max(10 µs, 値の範囲)(spec §5.2)"""
    v = [float(x) for x in values]
    return max(float(floor), max(v) - min(v))


def classify_primary(p_vals, r_vals, delta):
    """主指標の分類(spec §5.2)。p_vals・r_vals = 種ごとの L。→ {'class': good|bad|equivalent, 'mean', 'min', 'max', 'diff', ...}
    良い: Δ ≤ −δ かつ max_P < min_R / 悪い: Δ ≥ +δ かつ min_P > max_R / ほかは同等"""
    p, r = np.asarray(p_vals, np.float64), np.asarray(r_vals, np.float64)
    d = float(p.mean() - r.mean())
    if d <= -delta and p.max() < r.min():
        c = 'good'
    elif d >= delta and p.min() > r.max():
        c = 'bad'
    else:
        c = 'equivalent'
    return {'class': c, 'mean': float(p.mean()), 'min': float(p.min()), 'max': float(p.max()), 'ref_mean': float(r.mean()),
            'ref_min': float(r.min()), 'ref_max': float(r.max()), 'diff': d, 'delta': float(delta)}


def final_primary_class(best_cls, last_cls):
    """best.pth と last.pth で分類が変わる方策は「良い」と言わない(同等に下げる)。「悪い」は best.pth のまま"""
    if best_cls == 'good' and last_cls != 'good':
        return 'equivalent'
    return best_cls


def consistent_direction(p_by_ckpt, r_by_ckpt):
    """D5: 全部の種 × {best, last} で向きがそろうか。p_by_ckpt = {ckpt: [種ごとの L]}。→ 'worse' | 'better' | None"""
    worse = all(min(p_by_ckpt[c]) > max(r_by_ckpt[c]) for c in p_by_ckpt)
    better = all(max(p_by_ckpt[c]) < min(r_by_ckpt[c]) for c in p_by_ckpt)
    return 'worse' if worse else ('better' if better else None)


def bootstrap_index(n, n_boot, seed=0):
    """通過の選び直しの添え字 [n_boot, n](全部の学習結果で同じものを使う)"""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def ci_of(values, idx, level):
    """平均と、添え字 idx での選び直しの平均の両側 level の区間 → [平均, 下, 上]"""
    v = np.asarray(values, np.float64)
    b = v[idx].mean(axis=1)
    a = (1.0 - level) / 2.0
    return [float(v.mean()), float(np.percentile(b, 100 * a)), float(np.percentile(b, 100 * (1 - a)))]


def judge_pass_item(P, R, P_adj, idx, ci_good, ci_bad, min_pair):
    """R1・DE の項目(spec §5.3 + D1〜D3・D6)。
    P = [種, 通過] 方策の窓の値、R = [基準の本, 通過] 基準の窓の値、P_adj = [通過] 差の差(行ごとの P̄ − R̄ からシーンの平均を
    引いて窓で平均。NaN の通過は除く)、idx = 選び直しの添え字。
    良い: d̄ の ci_good 区間の下限 > 0、各種の d̄_i > 0、全部の組 d_ij ≥ min_pair、差の差の ci_good 区間の下限 > 0
    悪い: d̄ の ci_bad 区間の上限 < 0"""
    P, R = np.atleast_2d(np.asarray(P, np.float64)), np.atleast_2d(np.asarray(R, np.float64))
    d = P.mean(axis=0) - R.mean(axis=0)
    ci_g = ci_of(d, idx, ci_good)
    ci_b = ci_of(d, idx, ci_bad)
    d_i = [float((P[i] - R.mean(axis=0)).mean()) for i in range(len(P))]
    d_ij = [[float((P[i] - R[j]).mean()) for j in range(len(R))] for i in range(len(P))]
    adj = np.asarray(P_adj, np.float64)
    ok = np.isfinite(adj)
    if ok.all():
        ci_adj = ci_of(adj, idx, ci_good)
    else:                           # 窓にフレームの無い通過がある: その通過を除いて選び直す(添え字は同じ乱数の種で作り直す)
        ci_adj = ci_of(adj[ok], bootstrap_index(int(ok.sum()), idx.shape[0]), ci_good)
    parts = {'ci_lower_gt0': ci_g[1] > 0, 'each_seed_gt0': all(x > 0 for x in d_i),
             'all_pairs_ge_min': all(x >= min_pair for row in d_ij for x in row), 'adj_ci_lower_gt0': ci_adj[1] > 0}
    return {'d_mean': d.mean(), 'ci_good': ci_g, 'ci_bad': ci_b, 'd_seed': d_i, 'd_pairs': d_ij, 'adj_ci_good': ci_adj,
            'n_pass': int(P.shape[1]), 'good': bool(all(parts.values())), 'good_parts': parts, 'bad': bool(ci_b[2] < 0)}


def judge_collapse(fwd_p, fwd_r, bias_p, sign_p, sign_r, th=None):
    """崩れ(spec §5.3、点推定、D6 = 種の平均どうし)。fwd = 直線の前進の強さ、bias = 前進の偏り、sign = 向きの正解率。
    → (崩れたか, {項目: 合格か}, 直線の低下)"""
    th = th or COLLAPSE
    drop = 1.0 - float(fwd_p) / float(fwd_r)
    parts = {'straight': drop <= th['straight_drop_max'], 'fwd_bias': float(bias_p) >= th['fwd_bias_min_us'],
             'sign_acc': float(sign_p) >= float(sign_r) - th['sign_acc_margin']}
    return (not all(parts.values())), parts, drop


def judge_robust(dl1_p, dl1_r):
    """ずれへの強さ: ΔL1 の種の平均 ≤ 0.8 × ΔL1_R かつ全部の種で ΔL1 ≤ ΔL1_R(R = v0 の種の平均)"""
    p = np.asarray(dl1_p, np.float64)
    return bool(p.mean() <= ROBUST_RATIO * float(dl1_r) and np.all(p <= float(dl1_r)))


def pick_by_speed(cands, lbar, speed, delta, min_ms=0.3):
    """規則 2・3 の同点の決め方(D11)。cands の中で L̄ が最小のものと、L̄ の差が δ/2 未満のものの間では、run_numpy(種の平均)が
    min_ms 以上速いものを選ぶ。それより小さい差なら L̄ の小さい方"""
    if not cands:
        return None
    order = sorted(cands, key=lambda c: lbar[c])
    best = order[0]
    for c in order[1:]:
        if lbar[c] - lbar[best] < delta / 2.0 and speed.get(c) is not None and speed.get(best) is not None \
                and speed[best] - speed[c] >= min_ms:
            best = c
    return best


def decide(policies, gate, primary, collapse, r1, de, robust, nc_invalidates, lbar, speed, delta):
    """判定の規則(zoo_rules.json verdict_rules_effective の 1〜4)。各引数は {方策: 値}。
    primary: 'good'|'equivalent'|'bad'(best/last の確認の後)、r1・de: {'good', 'bad'}、robust: bool。
    → {'candidates', 'rule', 'best', 'rule2', 'rule3', 'note'}"""
    cands = [p for p in policies if p != REF_POLICY and gate[p] and primary[p] != 'bad' and not collapse[p]
             and not r1[p]['bad'] and not de[p]['bad']]
    good = [p for p in cands if primary[p] == 'good']
    out = {'candidates': cands, 'rule2': good, 'rule3': [], 'rule3_valid': not nc_invalidates}
    if good:
        out.update(rule=2, best=pick_by_speed(good, lbar, speed, delta),
                   note='主指標で v0 より δ 以上良い方策がある')
        return out
    r3 = [p for p in cands if primary[p] == 'equivalent' and (r1[p]['good'] or de[p]['good'])]
    out['rule3'] = r3
    if r3 and not nc_invalidates:
        score = {p: int(r1[p]['good']) + int(de[p]['good']) + int(bool(robust[p])) for p in r3}
        top = max(score.values())
        tied = [p for p in r3 if score[p] == top]
        out.update(rule=3, best=pick_by_speed(tied, lbar, speed, delta), score=score,
                   note='机上の候補(L1 は同等、R1/DE で良い)')
        return out
    out.update(rule=4, best=REF_POLICY, note='机上では v0 より良いと言えない(ベスト = jetformer_v0 = モデル B のまま)')
    return out


def gate_of(check, result, finite, policy):
    """1 本の学習結果のゲート(spec §5.1)。check = エンジンの確認の json、result = result.json、finite = 保留の予測が有限か。
    → (合格か, {項目: 合格か})"""
    acc, lat = (check or {}).get('accuracy') or {}, (check or {}).get('latency_ms') or {}
    mi = (check or {}).get('model_info') or {}
    parts = {
        'engine': bool(check) and bool(((check or {}).get('check_io') or {}).get('ok')),
        'drivable': bool(mi.get('drivable')),
        'accuracy': acc.get('actions_max_us') is not None and acc['actions_max_us'] <= GATES['actions_max_us']
                    and bool(acc.get('finite', True)),
        'speed': lat.get('run_numpy_median') is not None and lat['run_numpy_median'] <= GATES['run_numpy_median_ms']
                 and lat.get('run_numpy_p99', 99) <= GATES['run_numpy_p99_ms'],
        'train_mem': (result or {}).get('peak_mem_mb') is not None and result['peak_mem_mb'] <= GATES['peak_mem_mb'],
        'finite': bool(finite)}
    if policy == 'tcp':
        parts['tcp_waypoints'] = acc.get('waypoints_max_m') is not None and acc['waypoints_max_m'] <= GATES['waypoints_max_m']
        parts['tcp_speed'] = acc.get('speed_exp_max_mps') is not None and acc['speed_exp_max_mps'] <= GATES['speed_exp_max_mps']
    return all(parts.values()), parts


def boundary_mask(scene, t_us, boundaries=D8_BOUNDARIES, seconds=D8_SECONDS):
    """D8: 境目の検証の側 seconds 秒の行(scene・t_us は行ごと)。boundaries = [(シーン, 'start'|'end')]"""
    scene, t_us = np.asarray(scene), np.asarray(t_us, np.int64)
    out = np.zeros(len(scene), bool)
    for s, side in boundaries:
        m = scene == s
        if not m.any():
            continue
        t = t_us[m]
        if side == 'start':
            out[m] |= t < t.min() + seconds * 1e6
        else:
            out[m] |= t > t.max() - seconds * 1e6
    return out


def softmax_expectation(logits, bins):
    z = np.asarray(logits, np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    return (p * np.asarray(bins, np.float64)).sum(-1)


def wp_errors(pred, label):
    """通過点 [N,W,2] m → (ADE [N], FDE [N])"""
    d = np.linalg.norm(np.asarray(pred, np.float64) - np.asarray(label, np.float64), axis=-1)
    return d.mean(axis=1), d[:, -1]


# ---------------------------------------------------------------- plan(CPU。表示だけ)

def cmd_plan(args):
    rules = load_rules()
    other = Path('/home/jetson/jetformer_reports/policy_zoo_20261012/results/zoo_rules.json')
    same = None
    if other.exists():
        strip = lambda d: {k: v for k, v in d.items() if k != 'written_at'}  # noqa: E731
        same = strip(json.loads(other.read_text(encoding='utf-8'))) == strip(rules)
    print(json.dumps({'runs': run_names(rules), 'parameters': rules['parameters'],
                      'verdict_rules_effective': rules['verdict_rules_effective'],
                      'deviations': [d['id'] for d in rules['deviations']], 'report_copy_identical': same},
                     ensure_ascii=False, indent=1))
    print('規則は %s(%s に固定。このコマンドは書かない)' % (rules_path(), rules['written_at']))


# ---------------------------------------------------------------- predict(GPU)

class _Frames(object):
    """予測するフレーム(rl_evaluate._FrameSet と同じ中身 + ずれの試験)。torch の Dataset として使う"""

    def __init__(self, tab, rows, use_command=False, pert=None):
        self.use_command, self.pert = use_command, pert
        self.items = []
        cache = {}
        for j in rows:
            part, s, i = int(tab['part'][j]), str(tab['scene'][j]), int(tab['frame'][j])
            ds = ev.DATASET if part == 0 else ev.IV_DATASET
            if (ds, s) not in cache:
                cache[(ds, s)] = ev._aligned(ds, s, ['cam_files', 'tof', 'imu', 'vehicle', 'mode'])
            a = cache[(ds, s)]
            self.items.append((str(config.DATASET_DIR / ds / str(a['cam_files'][i])), a['tof'][i], a['imu'][i],
                               a['vehicle'][i], int(a['mode'][i])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        import cv2
        import torch
        from jetformer.train_data import normalize_inputs
        path, tof, imu, veh, _ = self.items[n]
        img = cv2.imread(path)
        if self.pert:
            img, tof = perturb(self.pert, img, tof)
        img, t, im, st = normalize_inputs(img, tof, imu, veh, self.use_command)
        return torch.from_numpy(img), torch.from_numpy(t), torch.from_numpy(im), torch.from_numpy(st)


def _predict_frames(model, fs, genres, num_modes, batch, workers):
    """→ {'actions' µs [N,10,2], ほかの出力(waypoints・speed_logits)はそのまま}"""
    import torch
    from torch.utils.data import DataLoader
    from jetformer.policies import runtime
    dl = DataLoader(fs, batch_size=batch, shuffle=False, num_workers=workers)
    outs, k = collections.defaultdict(list), 0
    with torch.no_grad():
        for image, tof, imu, state in dl:
            g = torch.from_numpy(genres[k:k + len(image)].astype(np.int64))
            k += len(image)
            onehot = torch.nn.functional.one_hot(g, num_modes).float()
            x = dict(image=image.cuda(), tof=tof.cuda(), imu=imu.cuda(), state=state.cuda(), mode=onehot.cuda())
            o = runtime.forward(model, x)
            for name, v in o.items():
                outs[name].append(v.float().cpu().numpy())
    res = {n: np.concatenate(v).astype(np.float32) for n, v in outs.items()}
    res['actions'] = res['actions'] * 500.0 + 1500.0                 # policies.PWM_SCALE・PWM_CENTER
    return res


def cmd_predict(args):
    rules = load_rules()
    names = run_names(rules)
    held = ev._heldout()
    tab = ev.pred_table()
    if tab['key'] != held['pred_table_key']:
        raise SystemExit('予測するフレームの表が heldout.json と違います')
    want = args.names or list(names)
    todo = []
    for n in want:
        for ck in args.ckpt:
            if not pred_path(n, ck).exists():
                todo.append((n, 'table', ck))
        if not args.no_pert and not pert_path(n).exists():
            todo.append((n, 'pert', 'best'))
    if not todo:
        print('予測は作成済み')
        return
    rp._check_gpu_free(min_gb=2.0)
    import torch
    from jetformer.policies import runtime
    cols = None
    all_rows = np.arange(len(tab['part']))
    for n in want:
        jobs = [j for j in todo if j[0] == n]
        if not jobs:
            continue
        d = rp.RUNS_DIR / names[n]
        cfg = rp._read_json(d / 'model_config.json')
        use_cmd = bool(cfg.get('command_input', True))
        trained = np.array([int(x) for x in cfg['trained_modes']])
        for _, kind, ck in jobs:
            t0 = time.time()
            model, _, spec, _ = runtime.load_run(d, ck, device='cuda')
            total, trainable = runtime.count_params(model)
            meta = dict(key=np.array(tab['key']), run=np.array(names[n]), ckpt=np.array(ck), policy=np.array(spec.name),
                        params_total=np.array(total), params_trainable=np.array(trainable))
            if kind == 'table':
                fs = _Frames(tab, all_rows, use_command=use_cmd)
                genres, sub = ev._genres([it[4] for it in fs.items], None, trained)
                res = _predict_frames(model, fs, genres, cfg['num_modes'], args.batch, args.workers)
                pred = res.pop('actions')
                extra = {'out_' + k: v for k, v in res.items()}
                rp._save_npz(pred_path(n, ck), pred=pred, genre=genres.astype(np.int8), substituted=sub, **extra, **meta)
                msg = '%d フレーム' % len(pred)
            else:
                if cols is None:
                    cols, _ = ev._val_arrays(tab)
                human = np.isin(tab['scene'], held['human_clean_scenes']) & (tab['part'] == 0)
                clean = human & cols['fv'] & ~cols['impact']
                rows = pert_rows(tab['frame'], clean)
                arrs = {}
                for pk in PERTS:
                    fs = _Frames(tab, rows, use_command=use_cmd, pert=pk)
                    genres, _ = ev._genres([it[4] for it in fs.items], None, trained)
                    arrs['pred_' + pk] = _predict_frames(model, fs, genres, cfg['num_modes'], args.batch,
                                                         args.workers)['actions']
                rp._save_npz(pert_path(n), rows=rows, **arrs, **meta)
                msg = 'ずれ %d × %d フレーム' % (len(PERTS), len(rows))
            print('%s(%s・%s): %s %.0f秒' % (n, kind, ck, msg, time.time() - t0), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()


def load_pred(name, ckpt='best', key=None):
    p = pred_path(name, ckpt)
    if not p.exists():
        raise SystemExit('予測がありません: %s(predict を実行)' % p)
    with np.load(p) as z:
        if key is not None and str(z['key']) != key:
            raise SystemExit('%s は別のフレームの表で作られています' % p)
        out = {k: z[k] for k in z.files}
    out['pred'] = out['pred'].astype(np.float64)
    return out


# ---------------------------------------------------------------- crosscheck(GPU。走行と同じ手順の compare_models と同じか)

def cmd_crosscheck(args):
    out_path = zoo_dir() / 'crosscheck.json'
    res = rp._read_json(out_path) or {}
    names = run_names(load_rules())
    want = args.names or ['B'] + ['%s_s0' % p for p in POLICIES]
    todo = [n for n in want if n not in res]
    if not todo:
        print('作成済み: %s' % out_path)
        return
    rp._check_gpu_free(min_gb=2.0)
    import torch
    import compare_models as cm
    tab = ev.pred_table()
    a = ev._aligned(ev.DATASET, args.scene)
    m = (tab['part'] == 0) & (tab['scene'] == args.scene)
    fr = tab['frame'][m]
    root = config.DATASET_DIR / ev.DATASET
    for n in todo:
        cached = load_pred(n, 'best', tab['key'])['pred'][m]
        model, cfg, trained, _ = cm.load_model(names[n])
        t0 = time.time()
        tf, sf, sub = cm.predict_scene(model, cfg, trained, root, a)
        res[n] = {'scene': args.scene, 'frames': int(m.sum()),
                  'max_abs_tf_vs_cached_us': _r(np.abs(tf[fr] - cached).max(), 4),
                  'max_abs_sf_vs_cached_us': _r(np.abs(sf[fr] - cached).max(), 4),
                  'max_abs_tf_vs_sf_us': _r(np.nanmax(np.abs(tf - sf)), 4), 'seconds': _r(time.time() - t0, 1)}
        print(n, res[n], flush=True)
        rp._save_json(out_path, res)
        del model
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- engine-check(GPU。1 本ずつ別のプロセス)

ENGINE_N_FRAMES = 64
ENGINE_WARMUP, ENGINE_TIMED = 100, 600


def _engine_frames(cfg):
    """clean_driving の検証から等間隔に 64 個(2.5G の衝突の前後を除く・学習と同じ正規化)→ {名前: [64,...]}"""
    from jetformer.train_data import JetFormerDataset
    ds = JetFormerDataset(str(config.DATASET_DIR / ev.CLEAN_DATASET), 'val', stride=1,
                          use_command=cfg.get('command_input', True), mask_impacts=True, impact_g=2.5)
    idx = np.linspace(0, len(ds) - 1, ENGINE_N_FRAMES).round().astype(int)
    rows = [ds[int(i)] for i in idx]
    out = {n: np.stack([c[k].numpy() for c in rows]).astype(np.float32)
           for k, n in enumerate(('image', 'tof', 'imu', 'state', 'mode'))}
    if out['mode'].shape[1] != cfg['num_modes']:
        raise SystemExit('ジャンル数が違います: データ %d / モデル %d' % (out['mode'].shape[1], cfg['num_modes']))
    return out


def engine_check_one(run_dir, out_path):
    """1 本のエンジンを走行タブと同じ手順で確かめる(train/engine_load_check.py と同じ計算。PWM は出さない・走らせない)。
    1. drive.model_info(走行タブの一覧と同じ関数)、2. 実フレーム 64 個で PyTorch FP32(CPU、best.pth)と TensorRT の差、
    3. drive._run と同じ順(CUDA の初期化 → TRTRunner → check_io → 0 で空回し 3 回)、4. 100 回の空回しの後 600 回の run_numpy"""
    import torch
    from jetformer import drive, policies
    from jetformer.policies import post as post_mod
    from jetformer.policies import runtime
    from jetformer.trt_runtime import TRTRunner
    run_dir = Path(run_dir).resolve()
    res = {'run': run_dir.name, 'checked_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    cfg = runtime.read_config(run_dir)
    res['policy'] = cfg.get('policy', 'jetformer_v0')
    res['engine_precision'] = (cfg.get('engine') or {}).get('precision', 'fp16')
    if run_dir.parent.resolve() != Path(drive.RUNS_DIR).resolve():
        drive.RUNS_DIR = run_dir.parent
    info = drive.model_info(run_dir.name)
    if info is None:
        raise SystemExit('drive.model_info が None(エンジンか model_config.json が無い)')
    res['model_info'] = {k: info.get(k) for k in ('policy', 'policy_title', 'drivable', 'why_not', 'inputs', 'outputs',
                                                  'num_modes', 'trained_modes', 'command_input', 'steer_center_us',
                                                  'throttle_neutral_us', 'best_val_us', 'created_at', 'engine_precision')}
    res['model_info']['postprocess_kind'] = post_mod.kind_of(info)
    frames = _engine_frames(cfg)
    torch.set_num_threads(4)
    model, _, _, _ = runtime.load_run(run_dir, 'best', device='cpu')        # CPU(TF32 が入らない)
    ref = collections.defaultdict(list)
    t = time.time()
    for s in range(0, ENGINE_N_FRAMES, 8):
        for k, v in runtime.predict(model, {k: v[s:s + 8] for k, v in frames.items()}).items():
            ref[k].append(v)
    ref = {k: np.concatenate(v) for k, v in ref.items()}
    res['torch_ref'] = {'device': 'cpu', 'frames': ENGINE_N_FRAMES, 'seconds': round(time.time() - t, 1)}
    del model
    gc.collect()
    torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    t = time.time()
    eng = Path(drive.RUNS_DIR) / info['name'] / drive.ENGINE_FILE
    runner = TRTRunner(str(eng))
    ok, why = policies.check_io({n: list(runner.shapes[n]) for n in runner.inputs},
                                {n: list(runner.shapes[n]) for n in runner.outputs}, info['num_modes'])
    res['check_io'] = {'ok': ok, 'why': why, 'inputs': {n: list(runner.shapes[n]) for n in runner.inputs},
                       'outputs': {n: list(runner.shapes[n]) for n in runner.outputs}}
    if not ok:
        rp._save_json(out_path, res)
        raise SystemExit('check_io: ' + why)
    zeros = {n: np.zeros(runner.shapes[n], np.float32) for n in runner.inputs}
    for _ in range(3):
        runner.run_numpy(**zeros)
    st = eng.stat()
    res.update(load_s=round(time.time() - t, 2), engine_mb=round(st.st_size / 2 ** 20, 1),
               engine={'size': st.st_size, 'mtime_ns': st.st_mtime_ns})
    got = collections.defaultdict(list)
    for i in range(ENGINE_N_FRAMES):
        for k, v in runner.run_numpy(**{n: frames[n][i:i + 1] for n in runner.inputs}).items():
            got[k].append(v)
    got = {k: np.concatenate(v) for k, v in got.items()}
    da = np.abs(got['actions'] - ref['actions']) * policies.PWM_SCALE
    acc = {'actions_max_us': round(float(da.max()), 3), 'actions_mean_us': round(float(da.mean()), 4),
           'actions_p99_us': round(float(np.percentile(da, 99)), 3), 'gate_actions_2p5us': bool(da.max() <= 2.5),
           'finite': bool(all(np.isfinite(v).all() for v in got.values()))}
    if 'waypoints' in got:
        dw = np.abs(got['waypoints'] - ref['waypoints'])
        acc.update(waypoints_max_m=round(float(dw.max()), 5), gate_waypoints_0p02m=bool(dw.max() <= 0.02))
    if 'speed_logits' in got:
        bins = (cfg.get('postprocess') or {}).get('speed_bins') or list(np.arange(10) * 0.5)
        dv = np.abs(softmax_expectation(got['speed_logits'], bins) - softmax_expectation(ref['speed_logits'], bins))
        acc.update(speed_exp_max_mps=round(float(dv.max()), 5), gate_speed_0p05mps=bool(dv.max() <= 0.05))
    res['accuracy'] = acc
    feed = {n: frames[n][:1] for n in runner.inputs}
    for _ in range(ENGINE_WARMUP):
        runner.run_numpy(**feed)
    ts = []
    for _ in range(ENGINE_TIMED):
        t0 = time.perf_counter()
        runner.run_numpy(**feed)
        ts.append((time.perf_counter() - t0) * 1000)
    en = []
    for _ in range(ENGINE_WARMUP):
        runner.context.execute_async_v3(runner.stream.cuda_stream)
        runner.stream.synchronize()
    for _ in range(ENGINE_TIMED):
        t0 = time.perf_counter()
        runner.context.execute_async_v3(runner.stream.cuda_stream)
        runner.stream.synchronize()
        en.append((time.perf_counter() - t0) * 1000)
    pct = lambda x, q: round(float(np.percentile(np.asarray(x), q)), 3)  # noqa: E731
    res['latency_ms'] = {'run_numpy_median': pct(ts, 50), 'run_numpy_p99': pct(ts, 99), 'engine_median': pct(en, 50),
                         'engine_p99': pct(en, 99),
                         'gate_run_numpy': bool(np.median(ts) <= GATES['run_numpy_median_ms']
                                                and np.percentile(ts, 99) <= GATES['run_numpy_p99_ms'])}
    gates = [res['model_info']['drivable'], ok, acc['gate_actions_2p5us'], acc['finite'], res['latency_ms']['gate_run_numpy']]
    gates += [acc[k] for k in ('gate_waypoints_0p02m', 'gate_speed_0p05mps') if k in acc]
    res['all_gates'] = bool(all(gates))
    rp._save_json(out_path, res)
    print(json.dumps({'run': res['run'], 'all_gates': res['all_gates'], 'accuracy': acc, 'latency_ms': res['latency_ms']},
                     ensure_ascii=False), flush=True)


def cmd_engine_check(args):
    if args.one:
        engine_check_one(args.one, args.out)
        return
    names = run_names(load_rules())
    for n in args.names:
        out = engine_json(n)
        run_dir = rp.RUNS_DIR / names[n]
        eng = run_dir / 'jetformer_fp16.engine'
        old = rp._read_json(out)
        if old and (old.get('engine') or {}).get('mtime_ns') == eng.stat().st_mtime_ns and not args.force:
            print('作成済み(同じエンジン): %s' % out)
            continue
        rp._check_gpu_free(min_gb=2.0)
        cmd = [sys.executable, '-u', os.path.abspath(__file__), 'engine-check', '--one', str(run_dir), '--out', str(out)]
        print('[%s] %s' % (time.strftime('%H:%M:%S'), n), flush=True)
        rc = subprocess.call(['nice', '-n', '10'] + cmd)
        if rc != 0:
            print('  終了コード %d' % rc, flush=True)


# ---------------------------------------------------------------- evaluate(CPU)

def _engine_checks(names, report_dir):
    """ゲートに使うエンジンの確認 → (判定用 {名前: json}, 事後の dinov2 {名前: json})。
    判定用: data/zoo/.../engine/<名前>.json があればそれ(モデル B)、10 本は学習の段の train/engine_check/<学習結果>.json(同じ計算)。
    dinov2 は D7 のとおり学習の段の FP16 のエンジンの確認(作り直したエンジンの確認は事後の参考)"""
    pre, post = {}, {}
    tr = Path(report_dir) / 'train' / 'engine_check'
    for n, run in names.items():
        z = rp._read_json(engine_json(n))
        t = rp._read_json(tr / (run + '.json'))
        if policy_of(n) == 'dinov2':
            pre[n] = t
            post[n] = z
        else:
            pre[n] = t or z
            if n == 'B':
                pre[n] = z
    return pre, post


def _window_values(ps, prow, outs, ch):
    """通過ごとの窓の平均 → {名前: [通過]}(ch 0 = ステア、1 = スロットル)と、各通過の行"""
    rows = []
    for p in ps:
        r = prow[p['rows']]
        rows.append(r[r >= 0])
    return {n: np.array([o[r, ch].mean() for r in rows]) for n, o in outs.items()}, rows


def _adjusted_windows(d_row, scene, m0, rows):
    """差の差: 行ごとの差からシーン(part 0 の全フレーム)の平均を引き、窓で平均 [通過]"""
    import rl_review
    dm = rl_review.minus_group_mean(d_row, scene, m0)
    return np.array([np.nanmean(dm[r]) if len(r) else np.nan for r in rows])


def evaluate_all(report_dir):
    import compare_models as cm
    rules = load_rules()
    prm = rules['parameters']
    names = run_names(rules)
    held = ev._heldout()
    tab = ev.pred_table()
    if tab['key'] != held['pred_table_key']:
        raise SystemExit('予測するフレームの表が heldout.json と違います')
    preds = {ck: {} for ck in CKPTS}
    extras = {}
    for n in names:
        for ck in CKPTS:
            z = load_pred(n, ck, tab['key'])
            preds[ck][n] = z['pred']
            if ck == 'best':
                extras[n] = z
    cols, f = ev._val_arrays(tab)
    cfgs = {n: rp._read_json(rp.RUNS_DIR / r / 'model_config.json') for n, r in names.items()}
    resj = {n: rp._read_json(rp.RUNS_DIR / r / 'result.json') or {} for n, r in names.items()}
    refs = {n: (float(cfgs[n]['steer_center_us']), float(cfgs[n]['throttle_neutral_us'])) for n in names}
    ds = prm['drive_settings']
    outs = {ck: {n: ev.drive_outputs(preds[ck][n], refs[n], ds['steer_gain'], ds['throttle_gain'], ds['lookahead'],
                                     ds['steer_trim_us']) for n in names} for ck in CKPTS}
    m0 = tab['part'] == 0
    human = np.isin(tab['scene'], held['human_clean_scenes']) & m0
    clean = human & cols['fv'] & ~cols['impact']
    scene = tab['scene']
    scene_id = np.unique(scene, return_inverse=True)[1]
    res = {'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
           'rules': {'path': str(rules_path()), 'written_at': rules['written_at']},
           'protocol': str(zoo_dir() / 'eval_protocol.json'),
           'runs': dict(names), 'refs_us': refs, 'drive_settings': ds,
           'frames': {'pred_table': int(len(tab['part'])), 'human_clean': int(clean.sum()), 'pred_table_key': tab['key']}}
    pol_all = POLICIES
    # ---- 数値(NaN・Inf)
    finite = {n: bool(all(np.isfinite(preds[ck][n]).all() for ck in CKPTS)
                      and all(np.isfinite(extras[n][k]).all() for k in extras[n] if k.startswith('out_'))) for n in names}
    # ---- B の予測が前回のキャッシュと同じか
    oldB = ev.pred_path('B')
    if oldB.exists():
        with np.load(oldB) as z:
            if str(z['key']) == tab['key']:
                res['model_B_vs_previous_cache_max_us'] = _r(np.abs(z['pred'].astype(np.float64) - preds['best']['B']).max(), 4)
    # ---- 主指標
    l1 = {ck: {n: l1_rows(preds[ck][n], cols['y']) for n in names} for ck in CKPTS}
    L = {ck: {n: float(l1[ck][n][clean].mean()) for n in names} for ck in CKPTS}
    delta = {ck: delta_of([L[ck]['B']] + [L[ck][x] for x in seed_names(REF_POLICY)]) for ck in CKPTS}
    prim = {}
    for p in pol_all:
        row = {}
        for ck in CKPTS:
            row[ck] = classify_primary([L[ck][x] for x in seed_names(p)], [L[ck][x] for x in seed_names(REF_POLICY)],
                                       delta[ck])
        row['final_class'] = final_primary_class(row['best']['class'], row['last']['class'])
        row['consistent'] = consistent_direction({ck: [L[ck][x] for x in seed_names(p)] for ck in CKPTS},
                                                 {ck: [L[ck][x] for x in seed_names(REF_POLICY)] for ck in CKPTS})
        row['diff_pct'] = 100.0 * row['best']['diff'] / row['best']['ref_mean']
        prim[p] = row
    t_us = _t_us(tab)
    bmask = boundary_mask(scene, t_us)
    res['primary'] = {
        'L_us': {ck: {n: _r(v, 3) for n, v in L[ck].items()} for ck in CKPTS},
        'delta_us': delta, 'per_policy': prim,
        'seed_range_us': {p: {ck: _r(max(L[ck][x] for x in seed_names(p)) - min(L[ck][x] for x in seed_names(p)), 3)
                              for ck in CKPTS} for p in pol_all},
        'best_minus_last_us': {n: _r(L['best'][n] - L['last'][n], 3) for n in names},
        'v0_three_runs_range_us': _r(max(L['best'][n] for n in ['B'] + seed_names(REF_POLICY))
                                     - min(L['best'][n] for n in ['B'] + seed_names(REF_POLICY)), 3),
        'd8_boundary_excluded': {'frames': int((clean & ~bmask).sum()),
                                 'L_us': {n: _r(l1['best'][n][clean & ~bmask].mean(), 3) for n in names},
                                 'note': '参考(判定に使わない)。境目 3 か所の検証の側 1 秒を除く(D8)'},
        'train_val_us': {n: resj[n].get('best_val_us') for n in names},
        'per_scene_L_us': {n: {s: _r(l1['best'][n][clean & (scene == s)].mean(), 2) for s in held['human_clean_scenes']}
                           for n in names},
    }
    # ---- R1・DE(D1〜D3・D6)
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
        idx = bootstrap_index(len(ps), nb, prm['bootstrap_seed'])
        wv = {ck: _window_values(ps, prow, outs[ck], ch) for ck in CKPTS}
        rows = wv['best'][1]

        def judge(p_names, r_names, ck='best'):
            P = np.array([wv[ck][0][x] for x in p_names])
            R = np.array([wv[ck][0][x] for x in r_names])
            d_row = (np.mean([outs[ck][x][:, ch] for x in p_names], axis=0)
                     - np.mean([outs[ck][x][:, ch] for x in r_names], axis=0))
            adj = _adjusted_windows(d_row, scene, m0, rows)
            return judge_pass_item(P, R, adj, idx, prm['ci_good'], prm['ci_bad'], prm['r1_de_good_min_pair_d_us'])

        it = {'passes': len(ps), 'ai_passes': int(sum(p['ai'] for p in ps)),
              'window_mean_us': {ck: {n: _r(float(np.mean(v)), 1) for n, v in wv[ck][0].items()} for ck in CKPTS},
              'per_policy': {}, 'per_policy_last': {}}
        for p in pol_all:
            if p == REF_POLICY:
                continue
            it['per_policy'][p] = _round_tree(judge(seed_names(p), seed_names(REF_POLICY)))
            it['per_policy_last'][p] = _round_tree(judge(seed_names(p), seed_names(REF_POLICY), 'last'))
        it['model_B_vs_v0'] = _round_tree(judge(['B'], seed_names(REF_POLICY)))
        it['nc'] = {'NC-A': _round_tree(judge(['jetformer_v0_s1'], ['jetformer_v0_s0', 'B'])),
                    'NC-B': _round_tree(judge(['jetformer_v0_s0'], ['jetformer_v0_s1', 'B']))}
        it['pass_values'] = [dict({k: p[k] for k in ('pass_id', 'log', 'ai', 'entry_erpm') if k in p},
                                  **{n: _r(float(wv['best'][0][n][j]), 1) for n in names}) for j, p in enumerate(ps)]
        items[item] = it
    # R1 の進入の見積もり(参考)
    emap, emap_info = ev._erpm_map(f)
    pv = ev._pass_values(r1p, prow, outs['best'], cols)
    ent = np.array([q['entry_erpm'] for q in pv], np.float64)
    ai_r1 = np.array([q['ai'] for q in pv], bool)
    fit_act = ev.erpm_from_throttle(emap, np.array([q['exec_thr'] for q in pv]))
    emap_info['validation_heldout_r1'] = {        # rl_evaluate.evaluate_all と同じ(対応表の当てはまりの目安)
        'passes': len(pv), 'spearman_map_vs_entry': _r(ev._spearman(fit_act, ent), 3),
        'mae_erpm': _r(np.mean(np.abs(fit_act - ent)), 0), 'bias_erpm': _r(np.mean(fit_act - ent), 0)}
    entry = {'actual': {'median_erpm': _r(np.median(ent), 0), 'share_lt5500': _r(np.mean(ent < ev.ENTRY_BAND_ERPM), 3),
                        'median_erpm_human': _r(np.median(ent[~ai_r1]), 0),
                        'exec_thr_us_mean': _r(np.mean([q['exec_thr'] for q in pv]), 1)},
             'per_run': {}, 'map': emap_info}
    for n in names:
        e = np.array([float(ev.entry_estimate(q['entry_erpm'], q['thr_' + n], q['exec_thr'], emap)) for q in pv])
        entry['per_run'][n] = {'median_erpm': _r(np.median(e), 0), 'share_lt5500': _r(np.mean(e < ev.ENTRY_BAND_ERPM), 3),
                               'median_erpm_human_passes': _r(np.median(e[~ai_r1]), 0),
                               'mean_minus_actual_erpm': _r(np.mean(e - ent), 0),
                               'thr_out_us_mean': _r(np.mean([q['thr_' + n] for q in pv]), 1)}
    items['r1']['entry_estimate'] = entry
    res['r1'], res['de'] = items['r1'], items['de']
    # ---- 崩れ
    fr = cols['frow']
    stm = np.zeros(len(fr), bool)
    stm[m0] = ev.straight_mask(f['loc_code'][fr[m0]], f['gz'][fr[m0]], f['erpm'][fr[m0]], f['near_collision'][fr[m0]])
    met = {n: cm.metrics(preds['best'][n][clean], cols['y'][clean], cols['cur'][clean], refs[n][0], refs[n][1]) for n in names}
    fwd = {n: float((NEUTRAL_US - outs['best'][n][stm, 1]).mean()) for n in names}
    col = {'straight_frames': int(stm.sum()), 'per_run': {}, 'per_policy': {}, 'nc': {}}
    for n in names:
        col['per_run'][n] = {'straight_fwd_strength_us': _r(fwd[n], 1), 'fwd_bias_us': met[n]['fwd_bias_us'],
                             'steer_sign_acc': met[n]['steer_sign_acc']}

    def collapse_of(p_names, r_names):
        fp = np.mean([fwd[x] for x in p_names])
        fr_ = np.mean([fwd[x] for x in r_names])
        bp = np.mean([met[x]['fwd_bias_us'] for x in p_names])
        sp = np.mean([met[x]['steer_sign_acc'] for x in p_names])
        sr = np.mean([met[x]['steer_sign_acc'] for x in r_names])
        c, parts, drop = judge_collapse(fp, fr_, bp, sp, sr)
        return {'collapse': c, 'parts': parts, 'straight_drop': _r(drop, 4), 'fwd_strength_us': _r(fp, 1),
                'ref_fwd_strength_us': _r(fr_, 1), 'fwd_bias_us': _r(bp, 1), 'sign_acc': _r(sp, 4), 'ref_sign_acc': _r(sr, 4)}
    for p in pol_all:
        if p != REF_POLICY:
            col['per_policy'][p] = collapse_of(seed_names(p), seed_names(REF_POLICY))
    col['model_B_vs_v0'] = collapse_of(['B'], seed_names(REF_POLICY))
    col['nc'] = {'NC-A': collapse_of(['jetformer_v0_s1'], ['jetformer_v0_s0', 'B']),
                 'NC-B': collapse_of(['jetformer_v0_s0'], ['jetformer_v0_s1', 'B'])}
    res['collapse'] = col
    # ---- ずれへの強さ
    rob = {'per_run': {}, 'per_policy': {}, 'nc': {}}
    dl1 = {}
    for n in names:
        p = pert_path(n)
        if not p.exists():
            raise SystemExit('ずれの予測がありません: %s' % p)
        with np.load(p) as z:
            rows = z['rows']
            base = l1_rows(preds['best'][n][rows], cols['y'][rows])
            per = {k: float(l1_rows(z['pred_' + k].astype(np.float64), cols['y'][rows]).mean() - base.mean()) for k in PERTS}
        dl1[n] = float(np.mean(list(per.values())))
        rob['per_run'][n] = {'frames': int(len(rows)), 'L_clean_us': _r(base.mean(), 3), 'dL1_us': _r(dl1[n], 3),
                             'per_pert_dL1_us': {k: _r(v, 3) for k, v in per.items()}}
    dl1_r = float(np.mean([dl1[x] for x in seed_names(REF_POLICY)]))
    for p in pol_all:
        if p != REF_POLICY:
            vals = [dl1[x] for x in seed_names(p)]
            rob['per_policy'][p] = {'robust_good': judge_robust(vals, dl1_r), 'dL1_mean_us': _r(np.mean(vals), 3),
                                    'dL1_seeds_us': [_r(v, 3) for v in vals], 'ref_dL1_us': _r(dl1_r, 3),
                                    'ratio_to_ref': _r(np.mean(vals) / dl1_r, 3) if dl1_r else None}
    for nc, (pn, rn) in {'NC-A': ('jetformer_v0_s1', ['jetformer_v0_s0', 'B']),
                         'NC-B': ('jetformer_v0_s0', ['jetformer_v0_s1', 'B'])}.items():
        rr = float(np.mean([dl1[x] for x in rn]))
        rob['nc'][nc] = {'robust_good': judge_robust([dl1[pn]], rr), 'dL1_us': _r(dl1[pn], 3), 'ref_dL1_us': _r(rr, 3)}
    rob['ref_dL1_us'] = _r(dl1_r, 3)
    res['robustness'] = rob
    # ---- 陰性対照(D4)
    delta_b = delta['best']
    nc = {}
    for nm, (pn, rn) in {'NC-A': ('jetformer_v0_s1', ['jetformer_v0_s0', 'B']),
                         'NC-B': ('jetformer_v0_s0', ['jetformer_v0_s1', 'B'])}.items():
        pb = classify_primary([L['best'][pn]], [L['best'][x] for x in rn], delta_b)
        pl = classify_primary([L['last'][pn]], [L['last'][x] for x in rn], delta['last'])
        pc = final_primary_class(pb['class'], pl['class'])
        flags = {'primary_good': pc == 'good', 'r1_good': items['r1']['nc'][nm]['good'],
                 'de_good': items['de']['nc'][nm]['good'], 'robust_good': rob['nc'][nm]['robust_good'],
                 'collapse': col['nc'][nm]['collapse']}
        nc[nm] = {'policy': pn, 'ref': rn, 'primary': {'best': _round_tree(pb), 'last': _round_tree(pl), 'final_class': pc},
                  'flags': flags, 'invalidates_rule3': bool(any(flags.values())),
                  'bad_any': bool(pc == 'bad' or items['r1']['nc'][nm]['bad'] or items['de']['nc'][nm]['bad'])}
    nc_inval = any(v['invalidates_rule3'] for v in nc.values())
    res['negative_controls'] = dict(nc, invalidates_rule3=nc_inval)
    # ---- ゲート
    pre, post_chk = _engine_checks(names, report_dir)
    gates = {'per_run': {}, 'per_policy': {}, 'dinov2_rebuilt': {}}
    for n in names:
        ok, parts = gate_of(pre.get(n), resj[n] if n != 'B' else {'peak_mem_mb': 0.0}, finite[n], policy_of(n))
        gates['per_run'][n] = {'pass': ok, 'parts': parts, 'source': 'train/engine_check' if n != 'B' else 'zoo/engine',
                               'engine_precision': (pre.get(n) or {}).get('engine_precision', 'fp16'),
                               'actions_max_us': ((pre.get(n) or {}).get('accuracy') or {}).get('actions_max_us'),
                               'run_numpy_median_ms': ((pre.get(n) or {}).get('latency_ms') or {}).get('run_numpy_median'),
                               'run_numpy_p99_ms': ((pre.get(n) or {}).get('latency_ms') or {}).get('run_numpy_p99'),
                               'engine_median_ms': ((pre.get(n) or {}).get('latency_ms') or {}).get('engine_median'),
                               'peak_mem_mb': resj[n].get('peak_mem_mb')}
    for p in pol_all:
        gates['per_policy'][p] = bool(all(gates['per_run'][x]['pass'] for x in seed_names(p)))
    for n in seed_names('dinov2'):
        c = post_chk.get(n)
        if c:
            ok, parts = gate_of(c, resj[n], finite[n], 'dinov2')
            gates['dinov2_rebuilt'][n] = {'pass': ok, 'parts': parts, 'engine_precision': c.get('engine_precision'),
                                          'actions_max_us': c['accuracy']['actions_max_us'],
                                          'actions_p99_us': c['accuracy']['actions_p99_us'],
                                          'run_numpy_median_ms': c['latency_ms']['run_numpy_median'],
                                          'run_numpy_p99_ms': c['latency_ms']['run_numpy_p99'],
                                          'engine_median_ms': c['latency_ms']['engine_median'], 'checked_at': c['checked_at']}
    rebuilt_ok = bool(gates['dinov2_rebuilt']) and all(v['pass'] for v in gates['dinov2_rebuilt'].values())
    gates['dinov2_drivable_after_rebuild'] = rebuilt_ok
    gates['note'] = ('判定のゲート(D7)は学習の段の FP16 のエンジン。dinov2 の作り直し(fp16_norm_fp32)は走行タブで走れるかの印だけで、'
                     '判定の候補には入れない')
    res['gates'] = gates
    # ---- 速さ・大きさ
    speed = {}
    lat = {'per_run': {}, 'per_policy': {}}
    rnB = gates['per_run']['B']['run_numpy_median_ms']
    for n in names:
        g = gates['per_run'][n]
        if policy_of(n) == 'dinov2' and n in gates['dinov2_rebuilt']:
            g = dict(g, **{k: gates['dinov2_rebuilt'][n][k] for k in ('run_numpy_median_ms', 'run_numpy_p99_ms',
                                                                      'engine_median_ms')})
        rn = g['run_numpy_median_ms']
        lat['per_run'][n] = {'run_numpy_median_ms': rn, 'run_numpy_p99_ms': g['run_numpy_p99_ms'],
                             'engine_median_ms': g['engine_median_ms'],
                             'loop_estimate_ms': _r(CAR_LOOP_B_MS + rn - rnB, 2) if rn is not None and rnB else None,
                             'cam_to_pwm_estimate_ms': _r(CAR_CAM_TO_PWM_B_MS + rn - rnB, 1) if rn is not None and rnB else None,
                             'params_total_m': _r(float(extras[n]['params_total']) / 1e6, 2),
                             'params_trainable_m': _r(float(extras[n]['params_trainable']) / 1e6, 2)}
    for p in pol_all:
        v = [gates['per_run'][x]['run_numpy_median_ms'] for x in seed_names(p)]       # 判定の速さ = 判定のゲートのエンジン
        speed[p] = float(np.mean(v)) if all(x is not None for x in v) else None
        lat['per_policy'][p] = {'run_numpy_median_ms_mean': _r(speed[p], 3)}
    lat['note'] = ('run_numpy・エンジンだけは同じ道具(100 回の空回しの後 600 回)。走行ループ・カメラ → PWM は、モデル B の実車の値'
                   '(%.1f ms・%.0f ms)に同じ道具での run_numpy の差を足した見積もりで、実測ではない。dinov2 は作り直した'
                   'エンジン(fp16_norm_fp32)の値。測り直すと ~0.15 ms 揺れる(D11)。tcp は制御器の時間を含まない'
                   % (CAR_LOOP_B_MS, CAR_CAM_TO_PWM_B_MS))
    res['latency'] = lat
    # ---- 判定
    lbar = {p: prim[p]['best']['mean'] for p in pol_all}
    r1j = {p: res['r1']['per_policy'].get(p, {'good': False, 'bad': False}) for p in pol_all}
    dej = {p: res['de']['per_policy'].get(p, {'good': False, 'bad': False}) for p in pol_all}
    colj = {p: col['per_policy'].get(p, {'collapse': False})['collapse'] for p in pol_all}
    robj = {p: rob['per_policy'].get(p, {'robust_good': False})['robust_good'] for p in pol_all}
    primj = {p: prim[p]['final_class'] for p in pol_all}
    verdict = decide(pol_all, gates['per_policy'], primj, colj, r1j, dej, robj, nc_inval, lbar, speed, delta_b)
    res['verdict'] = verdict
    # 参考(判定に使わない): dinov2 を作り直したエンジンで候補に入れたら
    g2 = dict(gates['per_policy'], dinov2=rebuilt_ok)
    res['verdict_sensitivity_dinov2_rebuilt'] = dict(
        decide(pol_all, g2, primj, colj, r1j, dej, robj, nc_inval, lbar, speed, delta_b),
        note='事後の参考。dinov2 を作り直したエンジン(fp16_norm_fp32)のゲートで候補に入れたときの判定(D7 により判定には使わない)')
    # 推奨のエンジン
    if verdict['best'] == REF_POLICY:
        rec_name = 'B'
    else:
        rec_name = min(seed_names(verdict['best']), key=lambda x: L['best'][x])
    res['recommended'] = {'name': rec_name, 'run': names[rec_name],
                          'engine': str(rp.RUNS_DIR / names[rec_name] / 'jetformer_fp16.engine'),
                          'note': 'ベストが jetformer_v0 ならモデル B のエンジン(今の走行の標準)'}
    # ---- 参考: 走行の精度(compare_models.metrics。今の操作値を入れないので教師あり = 自己入力)
    res['accuracy'] = _accuracy(preds, cols, tab, held, refs, l1, names)
    # ---- tcp の追加(§5.5)
    res['tcp'] = _tcp_extra(names, cfgs, refs, extras, preds, cols, tab, held, f, prow, r1p, emap, stm, clean, ds, nb, prm)
    cc = rp._read_json(zoo_dir() / 'crosscheck.json')
    if cc:
        res['crosscheck'] = cc
    return _round_tree(res)


def _t_us(tab):
    """行ごとの撮像時刻 [µs]"""
    t = np.zeros(len(tab['part']), np.int64)
    for part, ds in ((0, ev.DATASET), (1, ev.IV_DATASET)):
        for s in np.unique(tab['scene'][tab['part'] == part]):
            m = np.nonzero((tab['part'] == part) & (tab['scene'] == s))[0]
            a = ev._aligned(ds, s, ['timestamps_us'])
            t[m] = a['timestamps_us'][tab['frame'][m]]
    return t


def _accuracy(preds, cols, tab, held, refs, l1, names):
    import compare_models as cm
    m0 = tab['part'] == 0
    human = np.isin(tab['scene'], held['human_clean_scenes']) & m0
    ai_val = np.isin(tab['scene'], held['ai_val_scenes']) & m0
    iv = tab['part'] == 1
    win = {w['name']: w for w in held['iv_windows']}
    iv_held = iv & np.array([win.get(s, {}).get('log_heldout_policies', False) for s in tab['scene']])
    iv_class = np.array([win.get(s, {}).get('class', '') for s in tab['scene']])
    fv, imp = cols['fv'], cols['impact']
    sets = {'human_clean': human & fv & ~imp, 'human_clean_genre4': human & fv & ~imp & (cols['mode'] == 4),
            'human_clean_genre5': human & fv & ~imp & (cols['mode'] == 5), 'human_all': human & fv,
            'ai_val_clean_ref': ai_val & fv & ~imp & (cols['source'] == 1), 'iv_all': iv & fv,
            'iv_log_heldout': iv_held & fv, 'iv_train_logs': iv & ~iv_held & fv}
    for c in ('after_collision', 'stuck', 'preemptive'):
        sets['iv_' + c] = iv & fv & (iv_class == c)
    out = {'sets': {k: int(v.sum()) for k, v in sets.items()}, 'metrics': {}, 'metrics_last_human_clean': {},
           'note': 'compare_models.metrics(k1〜k10 の MAE・向きの正解率・前進の偏り)。今の操作値を入れないモデルなので教師あり = '
                   '自己入力(crosscheck で確かめた)。iv_* は交代の後の人の修正で、ログ単位で保留は iv_log_heldout(3 窓)だけ。'
                   'D12: 学習データの中の参考。ai_val_clean_ref は AI 自身の出力(ゲイン前)との一致で精度ではない'}
    for n in names:
        out['metrics'][n] = {}
        for k, m in sets.items():
            if m.any():
                r = cm.metrics(preds['best'][n][m], cols['y'][m], cols['cur'][m], refs[n][0], refs[n][1])
                r['l1_us'] = _r(l1['best'][n][m].mean(), 3)
                out['metrics'][n][k] = r
        m = sets['human_clean']
        r = cm.metrics(preds['last'][n][m], cols['y'][m], cols['cur'][m], refs[n][0], refs[n][1])
        r['l1_us'] = _r(l1['last'][n][m].mean(), 3)
        out['metrics_last_human_clean'][n] = r
    return out


def _tcp_extra(names, cfgs, refs, extras, preds, cols, tab, held, f, prow, r1p, emap, stm, clean, ds, nb, prm):
    """§5.5: tcp の通過点・目標速度の誤差と、制御器(post.replay = 走行と同じ Stage・TCPPost)の出し方ごとの出力(参考)"""
    from jetformer.policies import post as post_mod
    from jetformer.policies import tcp_labels
    from jetformer.drive import DEFAULT_PARAMS
    m0 = tab['part'] == 0
    croot = config.DATASET_DIR / ev.CLEAN_DATASET
    hs = list(held['human_clean_scenes'])
    lab = tcp_labels.labels_by_scene(croot, scenes=hs, keys=('waypoints', 'v_tgt', 'wp_valid'))
    # ai_mix の人のシーンと clean_driving のシーンのフレームが同じか(撮像時刻で確かめる)
    same = all(np.array_equal(tcp_labels.aligned_times(croot, s), ev._aligned(ev.DATASET, s, ['timestamps_us'])['timestamps_us'])
               for s in hs)
    rows_h = np.nonzero(m0 & np.isin(tab['scene'], hs))[0]
    wp_lab = np.zeros((len(tab['part']), 10, 2))
    v_lab = np.full(len(tab['part']), np.nan)
    wp_ok = np.zeros(len(tab['part']), bool)
    for j in rows_h:
        s, i = str(tab['scene'][j]), int(tab['frame'][j])
        wp_lab[j], v_lab[j], wp_ok[j] = lab[s]['waypoints'][i], lab[s]['v_tgt'][i], bool(lab[s]['wp_valid'][i])
    # 行ごとの eRPM と時刻(part 0)
    erpm = np.zeros(len(tab['part']))
    t_us = _t_us(tab)
    for s in np.unique(tab['scene'][m0]):
        mm = np.nonzero(m0 & (tab['scene'] == s))[0]
        erpm[mm] = ev._aligned(ev.DATASET, s, ['vehicle'])['vehicle'][tab['frame'][mm], 3]
    r0 = np.nonzero(m0)[0]
    out = {'labels_same_frames_as_ai_mix': bool(same), 'wp_valid_frames': int(wp_ok.sum()),
           'wp_valid_clean_frames': int((wp_ok & clean).sum()), 'per_run': {}}
    idx = bootstrap_index(len(r1p), nb, prm['bootstrap_seed'])
    for n in [x for x in names if policy_of(x) == 'tcp']:
        z = extras[n]
        pp = cfgs[n]['postprocess']
        bins = np.asarray(pp['speed_bins'], np.float64)
        wp, sl = z['out_waypoints'].astype(np.float64), z['out_speed_logits'].astype(np.float64)
        ade, fde = wp_errors(wp[wp_ok], wp_lab[wp_ok])
        vh = softmax_expectation(sl, bins)
        dv = vh[wp_ok] - v_lab[wp_ok]
        row = {'waypoints': {'ade_m': _r(ade.mean(), 4), 'fde_m': _r(fde.mean(), 4),
                             'ade_m_clean': _r(wp_errors(wp[wp_ok & clean], wp_lab[wp_ok & clean])[0].mean(), 4)},
               'target_speed': {'mae_mps': _r(np.abs(dv).mean(), 4), 'bias_mps': _r(dv.mean(), 4),
                                'v_label_mean_mps': _r(v_lab[wp_ok].mean(), 3), 'v_hat_mean_mps': _r(vh[wp_ok].mean(), 3)},
               'settings': {}}
        outputs = {'actions': ((preds['best'][n] - 1500.0) / 500.0).astype(np.float32)[r0],
                   'waypoints': z['out_waypoints'][r0], 'speed_logits': z['out_speed_logits'][r0]}
        sett = {}
        for sname, (mode, cap) in TCP_SETTINGS.items():
            params = dict(DEFAULT_PARAMS, tcp_mode=mode, speed_cap=cap, steer_gain=ds['steer_gain'],
                          throttle_gain=ds['throttle_gain'], lookahead=ds['lookahead'], ensemble_j=0)
            pr, _, infos = post_mod.replay(cfgs[n], refs[n], outputs, erpm[r0], t_us[r0] * 1000, params, scene=tab['scene'][r0])
            full = np.full((len(tab['part']), 10, 2), np.nan)
            full[r0] = pr
            o = ev.drive_outputs(np.nan_to_num(full), refs[n], ds['steer_gain'], ds['throttle_gain'], ds['lookahead'],
                                 ds['steer_trim_us'])
            sett[sname] = o
            l1c = l1_rows(full[clean], cols['y'][clean])
            branches = collections.Counter((inf or {}).get('branch', 'none') for inf in infos)
            row['settings'][sname] = {'tcp_mode': mode, 'speed_cap_mps': cap, 'L1_human_clean_us': _r(l1c.mean(), 3),
                                      'L1_steer_us': _r(np.abs(full[clean, :, 0] - cols['y'][clean, :, 0]).mean(), 2),
                                      'L1_throttle_us': _r(np.abs(full[clean, :, 1] - cols['y'][clean, :, 1]).mean(), 2),
                                      'straight_fwd_strength_us': _r((NEUTRAL_US - o[stm, 1]).mean(), 1),
                                      'branches': dict(branches)}
        # R1 の窓のスロットルと進入の見積もり
        wv, _ = _window_values(r1p, prow, sett, 1)
        pv_exec = np.array([q['exec_thr'] for q in ev._pass_values(r1p, prow, {'x': sett['control']}, cols)])
        ent = np.array([q['entry_erpm'] for q in ev._pass_values(r1p, prow, {'x': sett['control']}, cols)])
        for sname in TCP_SETTINGS:
            e = ev.entry_estimate(ent, wv[sname], pv_exec, emap)
            row['settings'][sname].update(r1_window_thr_us=_r(wv[sname].mean(), 1), r1_entry_est_median_erpm=_r(np.median(e), 0),
                                          r1_entry_share_lt5500=_r(np.mean(e < ev.ENTRY_BAND_ERPM), 3))
        caps = {}
        for a_, b_ in (('control_cap', 'control'), ('hybrid_cap', 'hybrid'), ('hybrid', 'control')):
            caps['%s-%s' % (a_, b_)] = {'r1_d_thr_us_ci95': [_r(x, 2) for x in ci_of(wv[a_] - wv[b_], idx, 0.95)],
                                        'straight_drop': _r(1 - (NEUTRAL_US - sett[a_][stm, 1]).mean()
                                                            / (NEUTRAL_US - sett[b_][stm, 1]).mean(), 4)}
        hyb_drop = caps['hybrid-control']['straight_drop']
        usable = {}
        for a_, b_ in (('control_cap', 'control'), ('hybrid_cap', 'hybrid')):
            lo = caps['%s-%s' % (a_, b_)]['r1_d_thr_us_ci95'][1]
            base_drop = 0.0 if b_ == 'control' else hyb_drop
            usable[a_] = bool(lo is not None and lo > 0 and base_drop <= 0.05)
        row['cap_effect'] = caps
        row['speed_cap_usable_offline'] = usable
        out['per_run'][n] = row
    out['note'] = ('参考(判定に使わない)。tcp_mode 0・上限 0 は actions と同じ。tcp_mode 1・3 のスロットルの写像は今の較正ではまだ使えない'
                   '精度(D13、tcp_check/oracle_modes.json)。机上で言えるのは指令がどう変わるかまで(ログの車は指令に反応しない)')
    return out


def _round_tree(x, nd=4):
    """json に書く値を丸める(float・numpy の数・bool)"""
    if isinstance(x, dict):
        return {k: _round_tree(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round_tree(v, nd) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), nd)
    return x


def cmd_evaluate(args):
    t0 = time.time()
    res = evaluate_all(args.report_dir)
    res['seconds'] = round(time.time() - t0, 1)
    rp._save_json(zoo_dir() / 'zoo_eval.json', res)
    if args.report_dir:
        rp._save_json(Path(args.report_dir) / 'eval' / 'zoo_eval.json', res)
    print(json.dumps({'verdict': res['verdict'], 'primary_L_best': res['primary']['L_us']['best'],
                      'delta': res['primary']['delta_us']}, ensure_ascii=False, indent=1))


def cmd_figures(args):
    import zoo_figures
    res = rp._read_json(zoo_dir() / 'zoo_eval.json')
    if not res:
        raise SystemExit('先に evaluate を実行してください')
    zoo_figures.make_all(res, Path(args.fig_dir))


def recommendation_from_eval(res, source=''):
    """zoo_eval.json の判定・推奨のエンジン → drive.save_recommended の引数 (run, reason, source)"""
    rec, verdict = (res or {}).get('recommended') or {}, (res or {}).get('verdict') or {}
    if not rec.get('run'):
        raise SystemExit('評価の結果に推奨のエンジン(recommended.run)がありません。先に evaluate を実行してください')
    best = verdict.get('best')
    reason = '方策の比較(%s)の評価: ベスト = %s(規則 %s)' % (ZOO_NAME, LABELS.get(best, best), verdict.get('rule'))
    if rec.get('name') == 'B':
        reason += '。モデル B のまま(今の走行の標準)'
    return rec['run'], reason, source


def cmd_recommend(args):
    from jetformer import drive
    if args.run:
        run, reason, source = args.run, args.reason or '手で指定', 'tools/policy_zoo.py recommend --run'
    else:
        path = zoo_dir() / 'zoo_eval.json'
        run, reason, source = recommendation_from_eval(rp._read_json(path), str(path))
    if drive.model_info(run) is None:
        raise SystemExit('走行タブに出ない学習結果です(エンジンか model_config.json が無い): %s' % run)
    out = drive.save_recommended(run, reason, source)
    print(json.dumps(dict(out, path=str(drive.RECOMMENDED_PATH)), ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('plan')
    p = sub.add_parser('predict')
    p.add_argument('--names', nargs='*', help='B・<方策>_s<種>(既定: 全部)')
    p.add_argument('--ckpt', nargs='+', choices=CKPTS, default=list(CKPTS))
    p.add_argument('--no-pert', action='store_true', help='ずれの試験の予測をしない')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--workers', type=int, default=3)
    p = sub.add_parser('crosscheck')
    p.add_argument('--names', nargs='*')
    p.add_argument('--scene', default=CROSS_SCENE)
    p = sub.add_parser('engine-check')
    p.add_argument('--names', nargs='*', default=[])
    p.add_argument('--force', action='store_true')
    p.add_argument('--one', default='', help=argparse.SUPPRESS)
    p.add_argument('--out', default='', help=argparse.SUPPRESS)
    p = sub.add_parser('evaluate')
    p.add_argument('--report-dir', default='/home/jetson/jetformer_reports/policy_zoo_20261012')
    p = sub.add_parser('figures')
    p.add_argument('--fig-dir', required=True)
    p = sub.add_parser('recommend')
    p.add_argument('--run', default='', help='評価の結果ではなく、この学習結果を推奨のモデルにする')
    p.add_argument('--reason', default='', help='--run のときの推奨の理由(走行タブに出る)')
    args = ap.parse_args()
    {'plan': cmd_plan, 'predict': cmd_predict, 'crosscheck': cmd_crosscheck, 'engine-check': cmd_engine_check,
     'evaluate': cmd_evaluate, 'figures': cmd_figures, 'recommend': cmd_recommend}[args.cmd](args)


if __name__ == '__main__':
    main()
