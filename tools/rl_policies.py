#!/usr/bin/env python3
"""オフライン強化学習の最小形 その2: 方策の学習(offline_rl.md 5.3、ロードマップ 4.3〜4.4 のゲートC 用の4本)

    python tools/rl_policies.py modes       # ジャンル「低速走行・良」「高速走行・良」を data/modes.json に登録(あれば何もしない)
    python tools/rl_policies.py prepare     # 4本のフレームの表(drop・重み・ジャンルの付け替え)→ data/rl/<データセット>/policy/
    python tools/rl_policies.py run --epochs 12 [--variants c b a n]   # 1本ずつ学習 → TensorRT エンジン。続きから再開できる
    python tools/rl_policies.py check-engines   # エンジンを読み込んで PyTorch(best.pth)と同じ出力か確かめる(GPU)
    python tools/rl_policies.py common-val      # 参考: モデル B と4本を同じ検証フレーム(人・AI・AI の失敗の前後)で比べる(GPU)
    python tools/rl_policies.py curves --fig-dir <dir>     # 学習曲線の図
    python tools/rl_policies.py summary     # 4本の結果をまとめる → data/rl/<データセット>/policy/summary.json

4本は「オフラインのデータの使い方」だけが違う。共通: 初期値 = モデル B の best.pth、今の操作値を入れない、同じデータセット
(ai_mix_20261011 の train/val)・間引き 1/3・エポック数・学習率の予定(1e-4 から cosine)・乱数の種・バッチ 64、
衝突の前後の除外(水平 2.5G の 1秒前〜2秒後)。
  (c) fbc    filtered BC: 使えるフレーム(advantages.npz の eligible で A が有限)から、衝突の前後(near_collision)と
             失敗による交代の 2秒前まで(AI のフレーム)を外し、ふつうの L1 の模倣
  (b) awr    AWR: (c) のフレーム + 失敗の前後の AI のフレーム(AI のフレームは衝突の前後の除外からも戻す)。損失に
             w = clip(exp(A/β), 0, 20)、β = A の標準偏差(使えるフレーム全体)を掛ける(学習の平均が1になるよう割る)
  (a) recap  アドバンテージ条件付き(RECAP 型。ネットワークは変えない): (b) と同じフレームで、印 I=1 のフレームを
             新しいジャンル「低速走行・良」「高速走行・良」で学習し、ほかは元のジャンル。学習中は 30% の確率で I=1 を
             元のジャンルに戻す(分類器なし誘導の形)。走行では「〜・良」のジャンルを使う
  (n) negctl 陰性対照: (a) と同じで、印を逆にしたもの(I_neg: A の下位 35% を「良い」)。ゲートCで落ちるはず。実車では使わない
新しいジャンルのヘッド・埋め込みは、元のジャンル(低速走行・良 ← 低速走行、高速走行・良 ← 高速走行)の値で初期化する。

GPU を使うのは run・check-engines・common-val だけ(学習・エンジン作成を1つずつ。ほかの GPU の処理が動いていたら始めない)。
電源が落ちても、同じコマンドをもう一度実行すれば続きから(学習は train.py --resume、出来たものは飛ばす)。
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from jetformer import config  # noqa: E402

DATASET = 'ai_mix_20261011'
BASE_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'     # 段階Bの基準(モデル B)
RUNS_DIR = config.DATA_DIR / 'runs'
FAIL_WINDOW_S = 2.0              # 失敗による交代の何秒前までを「失敗の前後」とするか
AWR_CLIP = 20.0
REMAP_DROPOUT = 0.3
# 登録の順 = 新しい ID の順(modes.add は最大の ID + 1 を付ける)。(名前, 元のジャンル ID, 説明)
GOOD_MODES = [('低速走行・良', 5, 'オフライン強化学習(アドバンテージ条件付き)用: 低速走行のうち「良い」の印のフレーム。収集には使わない'),
              ('高速走行・良', 4, 'オフライン強化学習(アドバンテージ条件付き)用: 高速走行のうち「良い」の印のフレーム。収集には使わない')]
TRAIN = {'stride': 3, 'batch': 64, 'workers': 4, 'lr': 1e-4, 'seed': 0, 'impact_g': 2.5}
ORDER = ['c', 'b', 'a', 'n']
VARIANTS = {
    'c': {'slug': 'fbc', 'title': '(c) filtered BC', 'frames': 'clean', 'weights': False, 'labels': None},
    'b': {'slug': 'awr', 'title': '(b) AWR', 'frames': 'all', 'weights': True, 'labels': None},
    'a': {'slug': 'recap', 'title': '(a) 印つき(RECAP 型)', 'frames': 'all', 'weights': False, 'labels': 'I'},
    'n': {'slug': 'negctl', 'title': '(n) 陰性対照(印を逆)', 'frames': 'all', 'weights': False, 'labels': 'I_neg'},
}
GPU_PATTERNS = ('tools/train.py', 'build_engine.py', 'offline_rl.py features', 'trtexec', 'compare_models.py',
                'tools/evaluate.py', 'rl_policies.py check-engines', 'rl_policies.py common-val',
                'rl_evaluate.py predict', 'rl_evaluate.py crosscheck', 'rl_evaluate.py engine-check',
                # 方策の比較(policy zoo)・推論の速さの計測
                'benchmark_model.py', 'policy_zoo.py train', 'policy_zoo.py predict', 'policy_zoo.py engine-check',
                # シミュレーターと実機を混ぜた学習(2026-10-13)の評価の予測
                'sim_cotrain_eval.py predict', 'sim_cotrain_eval.py crosscheck')
DISABLED_ENGINE_SUFFIX = '.do_not_drive'   # 走らせない方策のエンジン(管理画面の一覧から外すために名前を変えたもの)


def engine_disabled(out):
    """エンジンの名前を変えて走らせないようにしてあるか(jetformer_fp16.engine.do_not_drive がある)"""
    return (Path(out) / ('jetformer_fp16.engine' + DISABLED_ENGINE_SUFFIX)).exists()


def rl_dir(dataset=DATASET):
    return config.DATA_DIR / 'rl' / dataset


def policy_dir(dataset=DATASET):
    return rl_dir(dataset) / 'policy'


def run_name(v, epochs, dataset=DATASET):
    return '%s-rl-%s-%s-nocmd-s%d-e%d' % (dataset, v, VARIANTS[v]['slug'], TRAIN['stride'], epochs)


def _save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _read_json(path, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


# ---------------------------------------------------------------- フレームの規則(numpy だけ。テストする)

def failure_window(scene, t_s, source, flag, window_s=FAIL_WINDOW_S):
    """失敗による交代のフレーム(flag)の window_s 秒前〜そのフレームまでの、同じシーンの AI のフレーム"""
    scene = np.asarray(scene)
    t_s = np.asarray(t_s, np.float64)
    out = np.zeros(len(t_s), bool)
    ai = np.asarray(source) == 1
    for i in np.nonzero(np.asarray(flag, bool))[0]:
        out |= (scene == scene[i]) & ai & (t_s >= t_s[i] - window_s) & (t_s <= t_s[i])
    return out


def frame_sets(eligible, A, I, source, near_collision, fail_window):
    """4本のフレームの規則。戻り値(すべて [N] bool):
    valid       方策に使える(eligible・A が有限・印がある)
    drop_clean  (c) で外す: 使えない・衝突の前後・失敗の前後
    drop_all    (b)(a)(n) で外す: 使えない・人のフレームの衝突の前後(AI の失敗の前後は使う)
    keep_impact (b)(a)(n) で、衝突の前後の除外(train.py の 2.5G)から戻す: 使える AI のフレーム
    fail_adj_ai (b) で足される AI のフレームのうち、ラベルで分かる失敗の前後(数える用。2.5G の除外の分は学習時に数える)"""
    eligible = np.asarray(eligible, bool)
    I = np.asarray(I)
    valid = eligible & np.isfinite(np.asarray(A, np.float64)) & (I >= 0)
    ai = np.asarray(source) == 1
    nc = np.asarray(near_collision, bool)
    fw = np.asarray(fail_window, bool)
    return {'valid': valid,
            'drop_clean': ~valid | nc | fw,
            'drop_all': ~valid | (~ai & (nc | fw)),
            'keep_impact': valid & ai,
            'fail_adj_ai': valid & ai & (nc | fw)}


def awr_weights(A, valid, beta=None, clip=AWR_CLIP):
    """w = clip(exp(A/β), 0, clip)。β は使えるフレームの A の標準偏差。使えないフレームは NaN。(w, β) を返す"""
    A = np.asarray(A, np.float64)
    valid = np.asarray(valid, bool)
    if beta is None:
        beta = float(np.std(A[valid]))
    w = np.full(len(A), np.nan)
    w[valid] = np.clip(np.exp(A[valid] / beta), 0.0, clip)
    return w.astype(np.float32), beta


def remap_modes(I, genre, valid, good_of):
    """印 I=1 の使えるフレームを「〜・良」のジャンルへ(good_of {元のID: 良のID})。ほかは -1(そのまま)"""
    I = np.asarray(I)
    genre = np.asarray(genre)
    out = np.full(len(I), -1, np.int16)
    for base, good in good_of.items():
        out[np.asarray(valid, bool) & (I == 1) & (genre == base)] = good
    return out


def register_good_modes(modes_module=None, specs=GOOD_MODES):
    """「〜・良」のジャンルを data/modes.json に登録する(同じ名前があればその ID を使う)。{元のID: 良のID}"""
    if modes_module is None:
        from jetformer import modes as modes_module
    good_of = {}
    for name, base, desc in specs:
        found = [m for m in modes_module.load() if m['name'] == name]
        m = found[0] if found else modes_module.add(name, desc)
        good_of[base] = int(m['id'])
    return good_of


def effective_sample_size(w):
    w = np.asarray(w, np.float64)
    return float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12))


# ---------------------------------------------------------------- prepare

def _load_inputs(dataset):
    d = rl_dir(dataset)
    adv = dict(np.load(d / 'advantages.npz', allow_pickle=False))
    fr = np.load(d / 'frames.npz', allow_pickle=False)
    fr = {k: fr[k] for k in ('scene', 'frame', 'timestamps_us', 'source', 'near_collision', 'failure_takeover', 'split',
                             'mode', 'loc_code')}
    if not (np.array_equal(adv['scene'], fr['scene']) and np.array_equal(adv['frame'], fr['frame'])):
        raise SystemExit('advantages.npz と frames.npz のフレームの並びが違います')
    return adv, fr


def _impact_and_stride(dataset, scene, frame, impact_g):
    """データセットと同じ規則の衝突の前後(train_data.impact_mask)と、データセットの usable(train.py が使う条件)"""
    from jetformer.train_data import impact_mask
    root = config.DATASET_DIR / dataset
    imp = np.zeros(len(scene), bool)
    usable = np.zeros(len(scene), bool)
    for s in np.unique(scene):
        idx = np.nonzero(scene == s)[0]
        a = np.load(root / 'aligned' / (s + '.npz'))
        if len(a['timestamps_us']) != len(idx) or not np.array_equal(frame[idx], np.arange(len(idx))):
            raise SystemExit('シーン %s: aligned とフレームの表の長さ・番号が違います' % s)
        imp[idx] = impact_mask(a['timestamps_us'], a['imu'], impact_g)
        usable[idx] = a['future_valid'] & a['vehicle_valid'] & (a['mode'] != 0)
    return imp, usable


def _count(mask, fr, adv, label=None, good_ids=None):
    split = fr['split']
    src = fr['source']
    st = (fr['frame'] % TRAIN['stride']) == 0
    out = {}
    for sp in ('train', 'val'):
        m = mask & (split == sp)
        e = {'frames': int(m.sum()), 'samples_stride': int((m & st).sum()),
             'human': int((m & st & (src == 0)).sum()), 'ai': int((m & st & (src == 1)).sum()),
             'genre': {str(g): int((m & st & (adv['genre'] == g)).sum()) for g in (4, 5)}}
        if label is not None:
            lab = adv[label]
            e['good_samples'] = int((m & st & (lab == 1)).sum())
            e['good_share'] = round(float((lab[m & st] == 1).mean()), 4) if (m & st).any() else None
            e['good_by_genre'] = {str(g): int((m & st & (lab == 1) & (adv['genre'] == g)).sum()) for g in (4, 5)}
        out[sp] = e
    return out


def cmd_modes(args):
    good_of = register_good_modes()
    print('「〜・良」のジャンル(元のID → 良のID):', good_of)
    from jetformer import modes
    for m in modes.load():
        print('  %d %s %s' % (m['id'], m['name'], m.get('description', '')))
    return good_of


def cmd_prepare(args):
    out = policy_dir(args.dataset)
    if (out / 'policy_sets.json').exists() and not args.force:
        print('作成済み: %s(--force で作り直す)' % (out / 'policy_sets.json'))
        return
    good_of = register_good_modes()
    adv, fr = _load_inputs(args.dataset)
    t_s = fr['timestamps_us'] / 1e6
    fw = failure_window(fr['scene'], t_s, fr['source'], fr['failure_takeover'])
    sets = frame_sets(adv['eligible'], adv['A'], adv['I'], fr['source'], fr['near_collision'], fw)
    imp, usable = _impact_and_stride(args.dataset, fr['scene'], fr['frame'], TRAIN['impact_g'])
    valid = sets['valid']
    if (valid & ~usable).any():
        raise SystemExit('使えるフレームなのにデータセットで使えないフレームが %d 個あります' % int((valid & ~usable).sum()))
    w, beta = awr_weights(adv['A'], valid)
    remap = {v: remap_modes(adv[VARIANTS[v]['labels']], adv['genre'], valid, good_of) for v in ('a', 'n')}
    key = {'scene': fr['scene'].astype('U'), 'frame': fr['frame'].astype(np.int32)}
    _save_npz(out / 'drop_clean.npz', drop=sets['drop_clean'], **key)
    _save_npz(out / 'drop_all.npz', drop=sets['drop_all'], keep_impact=sets['keep_impact'], **key)
    _save_npz(out / 'awr_weights.npz', weight=w, A=adv['A'].astype(np.float32), **key)
    for v in ('a', 'n'):
        _save_npz(out / ('modes_%s.npz' % v), mode=remap[v], **key)
    # train.py が実際に使うフレーム(衝突の前後の除外 2.5G と keep_impact を入れた後)
    used_clean = ~sets['drop_clean'] & ~imp
    used_all = ~sets['drop_all'] & ~(imp & ~sets['keep_impact'])
    added = used_all & ~used_clean
    st = (fr['frame'] % TRAIN['stride']) == 0
    wt = w[used_all & st & (fr['split'] == 'train')].astype(np.float64)
    counts = {'c': _count(used_clean, fr, adv), 'b': _count(used_all, fr, adv),
              'a': _count(used_all, fr, adv, 'I'), 'n': _count(used_all, fr, adv, 'I_neg'),
              'b_added_vs_c': _count(added, fr, adv),
              'b_added_vs_c_by_reason': {
                  'near_collision_or_fail_window': int((added & st & sets['fail_adj_ai'] & (fr['split'] == 'train')).sum()),
                  'impact_2.5G_only': int((added & st & ~sets['fail_adj_ai'] & (fr['split'] == 'train')).sum())}}
    added_tr = added & st & (fr['split'] == 'train')
    rep = {
        'dataset': args.dataset, 'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'created_by': 'tools/rl_policies.py prepare',
        'good_modes': {str(k): v for k, v in good_of.items()},
        'rules': {
            'valid': 'advantages.npz の eligible(future_valid & vehicle_valid & ジャンルあり & label_valid & 未来の操作 950〜2050µs)'
                     ' で A が有限・印がある',
            'fail_window': '失敗による交代(failure_takeover)の %.1f 秒前〜そのフレームの、同じシーンの AI のフレーム' % FAIL_WINDOW_S,
            'c': '使えるフレーム − 衝突の前後(near_collision: 2.5G の 1秒前〜2秒後)− 失敗の前後。さらに train.py の衝突の前後の除外(2.5G)',
            'b': '(c) + AI の失敗の前後(near_collision・失敗の前後・2.5G の除外にかかる AI のフレームも使う)。人のフレームは (c) と同じ',
            'a': '(b) と同じフレーム。I=1 → 「〜・良」、学習中は %.0f%% で元のジャンルに戻す' % (REMAP_DROPOUT * 100),
            'n': '(a) と同じで I_neg(A の下位 35%% を「良い」)',
            'awr': 'w = clip(exp(A/β), 0, %g)、β = std(A)(使えるフレーム全体)。train.py が学習の平均で割る' % AWR_CLIP},
        'awr': {'beta': round(beta, 4), 'clip': AWR_CLIP,
                'train_weights': {'mean': round(float(wt.mean()), 4), 'median': round(float(np.median(wt)), 4),
                                  'p90': round(float(np.percentile(wt, 90)), 3), 'max': round(float(wt.max()), 3),
                                  'clipped_share': round(float((wt >= AWR_CLIP).mean()), 4),
                                  'ess': round(effective_sample_size(wt), 1), 'n': int(len(wt))},
                'mean_weight_added_ai_train': round(float(w[added_tr].mean()), 4) if added_tr.any() else None,
                'mean_weight_ai_clean_train': round(float(w[used_clean & st & (fr['source'] == 1) & (fr['split'] == 'train')]
                                                      .mean()), 4),
                'mean_weight_human_train': round(float(w[used_clean & st & (fr['source'] == 0) & (fr['split'] == 'train')]
                                                   .mean()), 4)},
        'counts': counts,
        'files': {k: str(out / k) for k in ('drop_clean.npz', 'drop_all.npz', 'awr_weights.npz', 'modes_a.npz', 'modes_n.npz')},
        'note': '数は間引き(1/3)の後の「学習サンプル」と、間引き前の frames。train.py の表示(学習 / 検証 サンプル)と一致するはず'}
    _save_json(out / 'policy_sets.json', rep)
    for v in ORDER:
        c = counts[v]
        print('%s %-22s 学習 %5d(人 %5d・AI %5d)検証 %4d%s' % (
            v, VARIANTS[v]['title'], c['train']['samples_stride'], c['train']['human'], c['train']['ai'],
            c['val']['samples_stride'],
            '  良 %d (%.1f%%)' % (c['train']['good_samples'], 100 * c['train']['good_share']) if 'good_samples' in c['train']
            else ''))
    print('AWR: β=%.3f  学習の重み 平均 %.2f・中央値 %.2f・最大 %.1f・上限に張り付き %.1f%%・有効サンプル数 %.0f / %d' % (
        beta, rep['awr']['train_weights']['mean'], rep['awr']['train_weights']['median'], rep['awr']['train_weights']['max'],
        100 * rep['awr']['train_weights']['clipped_share'], rep['awr']['train_weights']['ess'], len(wt)))
    print('書き出し:', out)


# ---------------------------------------------------------------- run(GPU。1つずつ)

def is_gpu_cmd(argv):
    """GPU を使う処理のコマンドか。実行ファイルが python か trtexec のものだけを見る(シェルのコマンド文字列に
    「tools/train.py」が含まれているだけのものは数えない。2026-10-12、待ち合わせのシェルを誤って数えた)"""
    if not argv:
        return False
    exe = os.path.basename(argv[0])
    if not (re.match(r'^python[0-9.]*$', exe) or exe == 'trtexec'):
        return False
    cmd = ' '.join(argv)
    return exe == 'trtexec' or (any(k in cmd for k in GPU_PATTERNS) and 'pytest' not in cmd)


def gpu_jobs(self_pid=None):
    """GPU を使うほかの処理(学習・エンジン作成・特徴抽出など)の [(pid, コマンド)]"""
    self_pid = self_pid or os.getpid()
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) in (self_pid, os.getppid()):
            continue
        try:
            argv = [x.decode('utf-8', 'replace') for x in (p / 'cmdline').read_bytes().split(b'\0') if x]
        except OSError:
            continue
        if is_gpu_cmd(argv):
            found.append((int(p.name), ' '.join(argv)[:160]))
    return found


def mem_available_gb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def _check_gpu_free(min_gb=3.0):
    jobs = gpu_jobs()
    if jobs:
        raise SystemExit('ほかの GPU の処理が動いています(8GB・スワップなしなので同時に動かさない): %s' % jobs)
    gb = mem_available_gb()
    if gb < min_gb:
        raise SystemExit('空きメモリが %.1f GB しかありません(%.1f GB 以上で始める)' % (gb, min_gb))


def train_command(v, out, epochs, dataset=DATASET, resume=False, good_of=None):
    pdir = policy_dir(dataset)
    spec = VARIANTS[v]
    cmd = ['nice', '-n', '10', sys.executable, '-u', os.path.join(HERE, 'train.py'),
           '--data', str(config.DATASET_DIR / dataset), '--out', str(out), '--epochs', str(epochs),
           '--batch', str(TRAIN['batch']), '--lr', str(TRAIN['lr']), '--stride', str(TRAIN['stride']),
           '--workers', str(TRAIN['workers']), '--no-command-input', '--impact-g', str(TRAIN['impact_g']),
           '--seed', str(TRAIN['seed']), '--init', str(RUNS_DIR / BASE_RUN / 'best.pth'),
           '--drop-mask', str(pdir / ('drop_clean.npz' if spec['frames'] == 'clean' else 'drop_all.npz'))]
    if spec['weights']:
        cmd += ['--sample-weights-npz', str(pdir / 'awr_weights.npz')]
    if spec['labels']:
        good_of = good_of or {int(k): int(g) for k, g in _read_json(pdir / 'policy_sets.json')['good_modes'].items()}
        cmd += ['--mode-remap-npz', str(pdir / ('modes_%s.npz' % v)), '--mode-remap-dropout', str(REMAP_DROPOUT),
                '--init-mode-copy', ','.join('%d:%d' % (g, b) for b, g in sorted(good_of.items(), key=lambda x: x[1]))]
    if resume:
        cmd.append('--resume')
    return cmd


def training_done(out, epochs):
    res = _read_json(Path(out) / 'result.json') or {}
    return res.get('state') == 'done' and res.get('epochs_done') == epochs and (Path(out) / 'jetformer.onnx').exists()


def _append_wall(out, entry):
    p = Path(out) / 'rl_wall.json'
    rows = _read_json(p, [])
    rows.append(entry)
    _save_json(p, rows)


class RunLock(object):
    """run を2つ同時に動かさない(policy/run.lock に pid。持ち主が終わっていれば取り直す)"""

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        if self.path.exists():
            try:
                pid = int(self.path.read_text().strip())
                os.kill(pid, 0)
                raise SystemExit('run がもう動いています(pid %d、%s)' % (pid, self.path))
            except (ValueError, ProcessLookupError):
                pass
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except OSError:
            pass


def cmd_run(args):
    pdir = policy_dir(args.dataset)
    if not (pdir / 'policy_sets.json').exists():
        raise SystemExit('先に prepare を実行してください')
    with RunLock(pdir / 'run.lock'):
        _run_all(args)


def _run_all(args):
    for v in args.variants:
        out = RUNS_DIR / run_name(v, args.epochs, args.dataset)
        out.mkdir(parents=True, exist_ok=True)
        if not (out / 'run.json').exists():
            _save_json(out / 'run.json', {
                'dataset': args.dataset, 'stride': TRAIN['stride'], 'epochs': args.epochs, 'batch': TRAIN['batch'],
                'lr': TRAIN['lr'], 'pretrained': True, 'build_engine': True, 'command_input': False, 'mask_impacts': True,
                'impact_g': TRAIN['impact_g'], 'name': out.name,
                'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
                'rl_variant': v, 'init': BASE_RUN,
                'note': 'オフラインRLの方策 %s(tools/rl_policies.py)。モデル B から微調整。%s' % (
                    VARIANTS[v]['title'],
                    '走行では「〜・良」のジャンル(6・7)を使う' if v == 'a' else
                    '陰性対照: 実車では走らせない' if v == 'n' else '走行では元のジャンル(4・5)')})
        if training_done(out, args.epochs):
            print('[%s] 学習済み: %s' % (v, out.name), flush=True)
        else:
            _check_gpu_free()
            resume = (out / 'last.pth').exists()
            cmd = train_command(v, out, args.epochs, args.dataset, resume=resume)
            print('[%s] 学習%s: %s' % (v, '(再開)' if resume else '', ' '.join(cmd)), flush=True)
            t0 = time.time()
            started = datetime.datetime.now().isoformat(timespec='seconds')
            with open(out / 'train.log', 'a', encoding='utf-8') as log:
                log.write('\n==== %s %s\n' % (started, ' '.join(cmd)))
                log.flush()
                rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(config.ROOT))
            _append_wall(out, {'step': 'train', 'start': started, 'seconds': round(time.time() - t0, 1),
                               'resume': resume, 'returncode': rc})
            if rc != 0 or not training_done(out, args.epochs):
                raise SystemExit('[%s] 学習が終わっていません(終了コード %d)。%s/train.log を見て、同じコマンドで再開してください'
                                 % (v, rc, out))
            print('[%s] 学習 %.0f 秒' % (v, time.time() - t0), flush=True)
        engine = out / 'jetformer_fp16.engine'
        onnx = out / 'jetformer.onnx'
        if engine.exists() and engine.stat().st_mtime >= onnx.stat().st_mtime:
            print('[%s] エンジン作成済み' % v, flush=True)
            continue
        if engine_disabled(out):           # 走らせないために名前を変えたもの(tools/rl_review.py disable-engines)。作り直さない
            print('[%s] エンジンは走らせない設定(%s%s)。作り直さない' % (v, engine.name, DISABLED_ENGINE_SUFFIX), flush=True)
            continue
        _check_gpu_free()
        tmp = out / 'jetformer_fp16.engine.tmp'
        cmd = ['nice', '-n', '10', sys.executable, '-u', os.path.join(HERE, 'build_engine.py'), str(onnx), str(tmp)]
        print('[%s] エンジン作成: %s' % (v, ' '.join(cmd)), flush=True)
        t0 = time.time()
        started = datetime.datetime.now().isoformat(timespec='seconds')
        with open(out / 'engine.log', 'a', encoding='utf-8') as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(config.ROOT))
        _append_wall(out, {'step': 'engine', 'start': started, 'seconds': round(time.time() - t0, 1), 'returncode': rc})
        if rc != 0 or not tmp.exists():
            raise SystemExit('[%s] エンジンの作成に失敗しました(%s/engine.log)' % (v, out))
        os.replace(tmp, engine)
        print('[%s] エンジン %.0f 秒' % (v, time.time() - t0), flush=True)


# ---------------------------------------------------------------- check-engines(GPU)

def cmd_check_engines(args):
    """エンジンを TRTRunner で読み込み、検証データの数サンプルで PyTorch(best.pth、FP32)と比べる。管理画面の
    drive.model_info でも読めるか確かめる"""
    _check_gpu_free(min_gb=2.0)
    import torch
    from jetformer import drive
    from jetformer.model import PWM_SCALE
    from jetformer.policies import runtime
    from jetformer.train_data import JetFormerDataset
    from jetformer.trt_runtime import TRTRunner
    ds = JetFormerDataset(str(config.DATASET_DIR / args.dataset), 'val', stride=30, use_command=False)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(ds), size=min(8, len(ds)), replace=False)
    rows = {}
    for v in args.variants:
        out = RUNS_DIR / run_name(v, args.epochs, args.dataset)
        info = drive.model_info(out.name)
        model, cfg, _, _ = runtime.load_run(out, 'best', device='cuda')
        runner = TRTRunner(str(out / 'jetformer_fp16.engine'))
        diffs = []
        for n in pick:
            image, tof, imu, state, mode, _ = ds[int(n)]
            for g in cfg['trained_modes']:
                onehot = torch.zeros(1, cfg['num_modes'])
                onehot[0, g] = 1.0
                x = dict(image=image[None], tof=tof[None], imu=imu[None], state=state[None], mode=onehot)
                ref = runtime.predict(model, x)['actions']
                got = runner.run_numpy(**{k: t.numpy() for k, t in x.items()})['actions']
                diffs.append(float(np.abs(got - ref).max() * PWM_SCALE))
        rows[v] = {'run': out.name, 'ui_model_info': info is not None,
                   'trained_modes': info['trained_modes'] if info else None, 'num_modes': cfg['num_modes'],
                   'engine_mb': round((out / 'jetformer_fp16.engine').stat().st_size / 1e6, 1),
                   'trt_vs_torch_max_abs_us': round(max(diffs), 2), 'trt_vs_torch_median_abs_us': round(float(np.median(diffs)), 2),
                   'checks': len(diffs)}
        print(v, rows[v], flush=True)
        del runner, model
        torch.cuda.empty_cache()
    _save_json(policy_dir(args.dataset) / 'engine_check.json', rows)


# ---------------------------------------------------------------- common-val(GPU。参考の共通の物差し)

INFER_GENRE_NOTE = 'モデル B・(c)・(b) はフレームのジャンル(4・5)、(a)・(n) は「〜・良」(4→7、5→6)で予測'


def val_groups(dataset=DATASET):
    """検証のフレーム(間引き 1/3)を3つに分ける: 0 = 人のきれいな走行((c) と同じ規則)、1 = AI の失敗の前後以外、
    2 = AI の失敗の前後((b)(a)(n) だけが使う)。-1 = どれでもない。{シーン: [n] int8}"""
    adv, fr = _load_inputs(dataset)
    t_s = fr['timestamps_us'] / 1e6
    fw = failure_window(fr['scene'], t_s, fr['source'], fr['failure_takeover'])
    sets = frame_sets(adv['eligible'], adv['A'], adv['I'], fr['source'], fr['near_collision'], fw)
    imp, _ = _impact_and_stride(dataset, fr['scene'], fr['frame'], TRAIN['impact_g'])
    used_c = ~sets['drop_clean'] & ~imp
    used_all = ~sets['drop_all'] & ~(imp & ~sets['keep_impact'])
    g = np.full(len(imp), -1, np.int8)
    g[used_c & (fr['source'] == 0)] = 0
    g[used_c & (fr['source'] == 1)] = 1
    g[used_all & ~used_c & (fr['source'] == 1)] = 2
    out = {}
    for s in np.unique(fr['scene'][fr['split'] == 'val']):
        idx = np.nonzero(fr['scene'] == s)[0]
        out[str(s)] = (fr['frame'][idx], g[idx])
    return out


def cmd_common_val(args):
    """モデル B と4本を、同じ検証フレーム(人のきれいな走行・AI・AI の失敗の前後)で比べる(重みなしの L1)。
    今の操作値を入れないモデルなので、教師あり = 自己入力。ゲートC の判定そのものではない(参考)"""
    out_path = policy_dir(args.dataset) / 'common_val.json'
    if out_path.exists() and not args.force:
        print('作成済み: %s' % out_path)
        return
    _check_gpu_free(min_gb=2.0)
    import torch
    from torch.utils.data import DataLoader
    from jetformer.model import PWM_CENTER, PWM_SCALE
    from jetformer.policies import runtime
    from jetformer.train_data import JetFormerDataset
    groups = val_groups(args.dataset)
    ds = JetFormerDataset(str(config.DATASET_DIR / args.dataset), 'val', stride=TRAIN['stride'], use_command=False)
    names = json.load(open(config.DATASET_DIR / args.dataset / 'splits.json'))['val']
    scene_names = []                          # JetFormerDataset と同じ規則で読み込まれるシーン(ds.scenes の順)
    for s in names:
        with np.load(config.DATASET_DIR / args.dataset / 'aligned' / (s + '.npz')) as z:
            if all(k in z.files for k in ('vehicle', 'tof', 'imu')):
                scene_names.append(s)
    if len(scene_names) != len(ds.scenes):
        raise SystemExit('検証のシーンの対応が取れません')
    grp = np.array([groups[scene_names[k]][1][i] if scene_names[k] in groups else -1 for k, i in ds.index])
    keep = np.nonzero(grp >= 0)[0]
    ds.index = [ds.index[j] for j in keep]
    grp = grp[keep]
    cluster = np.array([k for k, _ in ds.index])          # ブートストラップの単位(シーン)
    dl = DataLoader(ds, 64, shuffle=False, num_workers=TRAIN['workers'])
    good_of = {int(k): int(g) for k, g in _read_json(policy_dir(args.dataset) / 'policy_sets.json')['good_modes'].items()}
    models = [('B', RUNS_DIR / BASE_RUN, False)] + [(v, RUNS_DIR / run_name(v, args.epochs, args.dataset), v in ('a', 'n'))
                                                   for v in args.variants]
    res = {'note': INFER_GENRE_NOTE, 'groups': {'0': '人のきれいな走行', '1': 'AI(失敗の前後以外)', '2': 'AI の失敗の前後'},
           'n': {str(k): int((grp == k).sum()) for k in range(3)}, 'models': {}}
    label_cache = None
    per_sample = {}
    for name, d, use_good in models:
        model, cfg, _, _ = runtime.load_run(d, 'best', device='cuda')
        preds, labels = [], []
        with torch.no_grad():
            for image, tof, imu, state, mode, target in dl:
                g = mode.argmax(dim=1)
                if use_good:
                    g = torch.tensor([good_of.get(int(x), int(x)) for x in g])
                onehot = torch.nn.functional.one_hot(g, cfg['num_modes']).float()
                x = dict(image=image.cuda(), tof=tof.cuda(), imu=imu.cuda(), state=state.cuda(), mode=onehot.cuda())
                p = runtime.forward(model, x)['actions'].float().cpu().numpy()
                preds.append(p * PWM_SCALE + PWM_CENTER)
                labels.append(target.numpy() * PWM_SCALE + PWM_CENTER)
        pred = np.concatenate(preds)
        lab = np.concatenate(labels)
        label_cache = lab
        err = np.abs(pred - lab)
        row = {}
        for k in range(3):
            m = grp == k
            row[str(k)] = {'l1_us': round(float(err[m].mean()), 1), 'steer_mae_us': round(float(err[m, :, 0].mean()), 1),
                           'throttle_mae_us': round(float(err[m, :, 1].mean()), 1),
                           'throttle_bias_us': round(float((pred[m, :, 1] - lab[m, :, 1]).mean()), 1),
                           'steer_bias_us': round(float((pred[m, :, 0] - lab[m, :, 0]).mean()), 1),
                           'pred_throttle_us': round(float(pred[m, :, 1].mean()), 1)}
        res['models'][name] = row
        per_sample[name] = {'l1': err.mean(axis=(1, 2)), 'thr': pred[:, :, 1].mean(axis=1), 'steer': pred[:, :, 0].mean(axis=1)}
        print(name, json.dumps(row, ensure_ascii=False), flush=True)
        del model
        torch.cuda.empty_cache()
    res['label_throttle_us'] = {str(k): round(float(label_cache[grp == k, :, 1].mean()), 1) for k in range(3)}
    # 差の 95% 区間(シーン単位の対応のあるブートストラップ)。基準はモデル B と (c)
    diffs = {}
    for ref in ('B', 'c'):
        if ref not in per_sample:
            continue
        for name in per_sample:
            if name == ref:
                continue
            for k in range(3):
                m = grp == k
                for key in ('l1', 'thr', 'steer'):
                    d = per_sample[name][key][m] - per_sample[ref][key][m]
                    diffs['%s-%s/%d/%s' % (name, ref, k, key)] = cluster_bootstrap_mean(d, cluster[m])
    res['diffs'] = diffs
    res['diffs_note'] = '名前 = 方策−基準/グループ/量(l1 = L1 誤差、thr = 予測スロットル(+ = 遅い)、steer = 予測ステア(+ = 左))。' \
                        '[平均, 2.5%, 97.5%] µs。検証のシーン単位で 2000回のブートストラップ'
    _save_npz(policy_dir(args.dataset) / 'common_val_samples.npz', group=grp, cluster=cluster,
              **{'%s_%s' % (n, k): v[k].astype(np.float32) for n, v in per_sample.items() for k in v})
    _save_json(out_path, res)


def cluster_bootstrap_mean(d, cluster, n_boot=2000, seed=0):
    """平均と、クラスタ(シーン)単位のブートストラップの 95% 区間 [平均, 下, 上]"""
    d = np.asarray(d, np.float64)
    cluster = np.asarray(cluster)
    ids = np.unique(cluster)
    sums = np.array([d[cluster == c].sum() for c in ids])
    cnts = np.array([(cluster == c).sum() for c in ids])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    boot = sums[pick].sum(axis=1) / np.maximum(cnts[pick].sum(axis=1), 1)
    return [round(float(d.mean()), 2), round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)]


# ---------------------------------------------------------------- summary / curves

def collect(dataset=DATASET, epochs=12, variants=ORDER):
    rows = {}
    sets = _read_json(policy_dir(dataset) / 'policy_sets.json', {})
    for v in variants:
        out = RUNS_DIR / run_name(v, epochs, dataset)
        res = _read_json(out / 'result.json')
        if not res:
            continue
        cfg = _read_json(out / 'model_config.json', {})
        wall = _read_json(out / 'rl_wall.json', [])
        rows[v] = {
            'run': out.name, 'title': VARIANTS[v]['title'], 'state': res.get('state'),
            'epochs_done': res.get('epochs_done'), 'best_epoch': res.get('best_epoch'), 'best_val_us': res.get('best_val_us'),
            'best_val_per_mode_us': res.get('best_val_per_mode_us'),
            'best_val_l1_us': next((h.get('val_l1_us') for h in res['history'] if h['epoch'] == res.get('best_epoch')), None),
            'train_samples': res.get('train_samples'), 'val_samples': res.get('val_samples'),
            'train_mode_counts': res.get('train_mode_counts'), 'trained_modes': cfg.get('trained_modes'),
            'num_modes': cfg.get('num_modes'), 'impact_masked_frames': res.get('impact_masked_frames'),
            'rl_options': res.get('rl_options'), 'steer_center_us': res.get('steer_center_us'),
            'throttle_neutral_us': res.get('throttle_neutral_us'),
            'train_seconds_sum_epochs': round(sum(h['seconds'] for h in res['history']), 1),
            'duration_s_last_process': res.get('duration_s'),
            'wall_train_s': round(sum(w['seconds'] for w in wall if w['step'] == 'train'), 1),
            'wall_engine_s': round(sum(w['seconds'] for w in wall if w['step'] == 'engine'), 1),
            'resumed': any(w.get('resume') for w in wall if w['step'] == 'train'),
            'engine': (out / 'jetformer_fp16.engine').exists(), 'history': res['history'],
            'expected_samples': (sets.get('counts', {}).get(v, {}).get('train', {}).get('samples_stride'),
                                 sets.get('counts', {}).get(v, {}).get('val', {}).get('samples_stride'))}
    return rows


def cmd_summary(args):
    rows = collect(args.dataset, args.epochs, args.variants)
    _save_json(policy_dir(args.dataset) / 'summary.json', rows)
    print('| 方策 | 学習 / 検証 サンプル | 最良の検証 µs (エポック) | 学習の壁時計 | エンジン |')
    print('|---|---|---|---|---|')
    for v, r in rows.items():
        print('| %s | %s / %s | %s (%s) | %.1f 分 | %s |' % (r['title'], r['train_samples'], r['val_samples'], r['best_val_us'],
                                                       r['best_epoch'], r['wall_train_s'] / 60, '有' if r['engine'] else '無'))


def cmd_curves(args):
    sys.path.insert(0, HERE)
    import figures_offline_rl as fo
    plt = fo._setup()
    rows = collect(args.dataset, args.epochs, args.variants)
    colors = {'B': fo.GRAY_LIGHT, 'c': fo.GRAY, 'b': fo.ORANGE, 'a': fo.BLUE, 'n': fo.RED}
    os.makedirs(args.fig_dir, exist_ok=True)

    def save(fig, name, bottom):
        fig.tight_layout(rect=(0, bottom, 1, 1))
        path = os.path.join(args.fig_dir, name)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print('図:', path)

    # 1) 学習曲線
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2))
    for v, r in rows.items():
        ep = [h['epoch'] for h in r['history']]
        axes[0].plot(ep, [h['train_us'] for h in r['history']], '-o', color=colors[v], lw=2.4, ms=5, label=r['title'])
        axes[1].plot(ep, [h['val_us'] for h in r['history']], '-o', color=colors[v], lw=2.4, ms=5)
        if any('val_l1_us' in h for h in r['history']):
            axes[1].plot(ep, [h.get('val_l1_us') for h in r['history']], '--', color=colors[v], lw=1.8)
        if r['best_epoch']:
            axes[1].plot([r['best_epoch']], [r['best_val_us']], marker='*', ms=18, color=colors[v], mec=fo.INK, zorder=5)
    axes[0].set_title('学習の誤差((b) は重み付き)')
    axes[1].set_title('検証の誤差(★ = 選んだエポック、点線 = (b) の重みなし)')
    for ax in axes:
        ax.set_xlabel('エポック')
        ax.set_ylabel('L1 誤差 [µs]')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.06))
    fig.text(0.5, 0.015, '検証のフレームとジャンルは方策ごとに違う((c) は失敗の前後なし、(b)(a)(n) は含む。(a)(n) は印のフレームを'
             '「〜・良」で評価)。この図の高さで優劣は決められない', ha='center', fontsize=13, color=fo.INK2)
    save(fig, 'policy_training_curves.png', 0.12)

    # 2) ジャンル別の検証の誤差
    fig, ax = plt.subplots(figsize=(15, 6.4))
    names = {4: '高速', 5: '低速', 6: '低速・良', 7: '高速・良'}
    styles = {4: '-', 5: '--', 6: ':', 7: '-.'}
    for v, r in rows.items():
        ep = [h['epoch'] for h in r['history']]
        for g in sorted({int(k) for h in r['history'] for k in h['val_per_mode_us']}):
            ax.plot(ep, [h['val_per_mode_us'].get(str(g)) for h in r['history']], styles.get(g, '-'), color=colors[v], lw=2,
                    label='%s %s' % (r['title'].split(' ')[0], names.get(g, g)))
    ax.set_xlabel('エポック')
    ax.set_ylabel('検証の L1 誤差 [µs](重みなし)')
    ax.set_title('ジャンル別の検証の誤差((a)(n) の「〜・良」は、印の付いた検証フレームだけ)')
    ax.legend(frameon=False, ncol=1, fontsize=12, loc='center left', bbox_to_anchor=(1.01, 0.5))
    save(fig, 'policy_val_by_genre.png', 0)

    # 3) 同じ検証フレームでの比較(common-val)
    cv = _read_json(policy_dir(args.dataset) / 'common_val.json')
    if cv:
        order = [m for m in ['B'] + list(args.variants) if m in cv['models']]
        titles = {'B': 'モデル B(元)', **{v: VARIANTS[v]['title'] for v in VARIANTS}}
        groups = [('0', '人のきれいな走行'), ('1', 'AI(失敗の前後以外)'), ('2', 'AI の失敗の前後')]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.8))
        xx = np.arange(len(groups))
        wd = 0.8 / len(order)
        for j, m in enumerate(order):
            l1 = [cv['models'][m][g]['l1_us'] for g, _ in groups]
            bias = [cv['models'][m][g]['throttle_bias_us'] for g, _ in groups]
            axes[0].bar(xx + (j - (len(order) - 1) / 2) * wd, l1, wd, color=colors[m], label=titles[m])
            axes[1].bar(xx + (j - (len(order) - 1) / 2) * wd, bias, wd, color=colors[m])
            for x, val in zip(xx + (j - (len(order) - 1) / 2) * wd, l1):
                axes[0].text(x, val + 1, '%.0f' % val, ha='center', fontsize=11, color=fo.INK2)
        for ax in axes:
            ax.set_xticks(xx)
            ax.set_xticklabels(['%s\n(%s)' % (n, cv['n'][g]) for g, n in groups])
            ax.grid(axis='x', visible=False)
        axes[0].set_ylabel('L1 誤差 [µs]')
        axes[0].set_title('検証の誤差(同じフレーム、間引き 1/3)')
        axes[1].set_ylabel('予測 − 実際 [µs]')
        axes[1].set_title('スロットルの偏り(+ = 実際より遅い)')
        axes[1].axhline(0, color=fo.INK2, lw=1)
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.05))
        fig.text(0.5, 0.012, 'モデル B・(c)・(b) はフレームのジャンル、(a)・(n) は「〜・良」で予測。今の操作値を入れないので教師あり = 自己入力。'
                 'AI の検証はログ2本(172505・204208)だけ', ha='center', fontsize=12.5, color=fo.INK2)
        save(fig, 'policy_common_val.png', 0.12)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['modes', 'prepare', 'run', 'check-engines', 'common-val', 'summary', 'curves'])
    ap.add_argument('--dataset', default=DATASET)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--variants', nargs='+', default=ORDER, choices=ORDER)
    ap.add_argument('--fig-dir', default='/home/jetson/jetformer_reports/offline_rl_20261011/fig')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    {'modes': cmd_modes, 'prepare': cmd_prepare, 'run': cmd_run, 'check-engines': cmd_check_engines,
     'common-val': cmd_common_val, 'summary': cmd_summary, 'curves': cmd_curves}[args.cmd](args)


if __name__ == '__main__':
    main()
