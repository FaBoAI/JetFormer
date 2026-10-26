#!/usr/bin/env python3
"""別の目の試験(2026-10-25 夜): C1(--aug-top-brightness)が既定 OFF なら今までと同じか。
C1 の係の試験(tests/test_train_aug_top.py・check_c1_default_same.py)は使わない・読まない。自分で書いた。

比べる相手: C1 の前の train_data.py・train.py(git c247daa。sha256 を protocol defaults_untouched の値と照らす)。
  1. augment_image(numpy): 既定(top_brightness = 0・引数なし)で、前と同じ画素・同じ乱数の状態
  2. augment_image(numpy): top_brightness > 0 の中身(上 1/3 だけ・足し算・直線の重み・乱数 1 回・順番)
  3. JetFormerDataset(torch・実データ dataset_v2 の v2-H): 段 1・2 の設定(拡張なし)と 10/21 の拡張で、
     前と新しいので index・全部の出力・乱数の状態が同じ。AUG2(ずれ・ToF の雑音を 0 にして)は上 1/3 の外が同じ
  4. train.py(AST で関数だけ取り出す): aug_of・bench_args_used が既定で前と同じ(記録しない = 再開できる)。
     --aug-top-brightness の既定は 0.0
  5. 段 1・2・B-FAST の 16 本の run.json の code_sha256 が C1 の前の train.py・train_data.py

torch を読むので jf-heavy --lock big で動かす(GPU は使わない):
    /home/jetson/bin/jf-heavy --lock big --mem 3000M --need-mb 2000 -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        /home/jetson/jetformer/experiments/20261025_train_v2/verify_d/test_c1_default_off.py
"""
import argparse
import ast
import atexit
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path('/home/jetson/jetformer')
REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
OUT = Path(__file__).resolve().parent / 'test_c1_default_off.json'
OLD_COMMIT = 'c247daa'
sys.path.insert(0, str(REPO))

proto = json.load(open(REP / 'protocol_v2.json'))
SHA_OLD = proto['defaults_untouched']['sha256_now']
results = []


def check(name, ok, info=None):
    results.append({'name': name, 'ok': bool(ok), 'info': info})
    print('%s  %s%s' % ('OK ' if ok else 'NG ', name, '' if info is None else '  ' + json.dumps(info, ensure_ascii=False)[:300]))


def git_show(path):
    return subprocess.run(['git', '-C', str(REPO), 'show', '%s:%s' % (OLD_COMMIT, path)], check=True,
                          capture_output=True).stdout


old_td_src = git_show('jetformer/train_data.py')
old_tr_src = git_show('tools/train.py')
new_td_src = (REPO / 'jetformer' / 'train_data.py').read_bytes()
new_tr_src = (REPO / 'tools' / 'train.py').read_bytes()
check('C1 の前の train_data.py の sha256 が protocol の値', hashlib.sha256(old_td_src).hexdigest() == SHA_OLD['train_data.py'])
check('C1 の前の train.py の sha256 が protocol の値', hashlib.sha256(old_tr_src).hexdigest() == SHA_OLD['train.py'])

tmpdir = Path(tempfile.mkdtemp(prefix='c1_old_'))
atexit.register(shutil.rmtree, str(tmpdir), True)       # C1 の前の写しは終わったら消す
(tmpdir / 'train_data_c247daa.py').write_bytes(old_td_src)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


import jetformer.train_data as new  # noqa: E402
old = load_module('jetformer.train_data_c247daa', tmpdir / 'train_data_c247daa.py')
check('新しい train_data は今のファイル', Path(new.__file__).resolve() == (REPO / 'jetformer' / 'train_data.py').resolve())


def rng_same(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


# ---------------------------------------------------------------- 1. augment_image の既定
ds_root = REPO / 'data' / 'datasets' / 'dataset_v2_20261025'
import cv2  # noqa: E402
_dirs = sorted(d for d in (ds_root / 'samples' / 'CAM_FRONT').iterdir() if d.is_dir())
cams = [sorted(d.glob('*.jpg'))[len(list(d.glob('*.jpg'))) // 2] for d in _dirs[::max(1, len(_dirs) // 8)]][:8]
imgs = [im for im in (cv2.imread(str(p)) for p in cams) if im is not None]
rs = np.random.RandomState(123)
imgs += [rs.randint(0, 256, size=(h, w, 3), dtype=np.uint8) for h, w in ((224, 224), (240, 320), (97, 131))]
SETTINGS = [{}, {'brightness': 0.15}, {'contrast': 0.2}, {'shift_px': 4},
            {'brightness': 0.15, 'contrast': 0.2, 'shift_px': 4}]
bad = []
for ii, img in enumerate(imgs):
    for st in SETTINGS:
        for s in range(12):
            np.random.seed(s)
            o = old.augment_image(img, **st)
            so = np.random.get_state()
            for extra in ({}, {'top_brightness': 0.0}):
                np.random.seed(s)
                n = new.augment_image(img, **st, **extra)
                sn = np.random.get_state()
                if not (np.array_equal(o, n) and o.dtype == n.dtype and rng_same(so, sn)):
                    bad.append((ii, st, s, extra))
check('1. augment_image: 既定(引数なし・top_brightness=0)で画素と乱数の状態が前と同じ',
      not bad, {'images': len(imgs), 'real_images': len(imgs) - 3, 'cases': len(imgs) * len(SETTINGS) * 12 * 2, 'bad': bad[:5]})

# ---------------------------------------------------------------- 2. top_brightness > 0 の中身
ramp_bad = []
for h in (224, 240, 480, 97):
    w = new.top_ramp(h)[:, 0, 0]
    cut = h / 3.0
    rows = np.arange(h)
    exp = np.clip(1.0 - rows / cut, 0.0, 1.0)
    if not (w[0] == 1.0 and np.allclose(w, exp, atol=1e-6) and (w[rows >= cut] == 0).all() and (np.diff(w) <= 0).all()):
        ramp_bad.append(h)
check('2a. top_ramp: 一番上 1.0、上 1/3 の境まで直線で 0、下 2/3 は 0', not ramp_bad, {'bad_heights': ramp_bad})

bad2, diffs = [], []
for ii, img in enumerate(imgs):
    h = img.shape[0]
    lower = np.arange(h) >= h / 3.0
    for s in range(12):
        st = {'brightness': 0.15, 'contrast': 0.2}
        np.random.seed(s)
        o = old.augment_image(img, **st)
        u = np.random.uniform(-0.15, 0.15)                 # 新しいのが 3 番目に引く値
        s_after = np.random.get_state()
        np.random.seed(s)
        n = new.augment_image(img, **st, top_brightness=0.15)
        # 自分で同じ式を書く(コントラスト → 明るさ → 上 1/3 の足し算)
        np.random.seed(s)
        f = img.astype(np.float32)
        c = np.random.uniform(0.8, 1.2)
        m = float(f.mean())
        f = (f - m) * c + m
        f = f + np.random.uniform(-0.15, 0.15) * 255.0
        u2 = np.random.uniform(-0.15, 0.15)
        f = f + np.float32(u2 * 255.0) * np.clip(1.0 - np.arange(h, dtype=np.float32) / (h / 3.0), 0, 1).reshape(h, 1, 1)
        e = np.clip(f, 0, 255).astype(np.uint8)
        ok = (np.array_equal(n, e) and np.array_equal(n[lower], o[lower]) and rng_same(np.random.get_state(), s_after)
              and u == u2)
        if not ok:
            bad2.append((ii, s, int(np.abs(n.astype(int) - e.astype(int)).max())))
        diffs.append(float(np.abs(n[0].astype(int) - o[0].astype(int)).mean()))
check('2b. top_brightness=0.15: 足し算 ±0.15×255×重み、下 2/3 は前と同じ、乱数は 1 回だけ多い', not bad2,
      {'bad': bad2[:5], 'top_row_mean_abs_change': [round(min(diffs), 2), round(max(diffs), 2)]})

bad3 = []
for s in range(20):
    img = imgs[0]
    np.random.seed(s)
    n = new.augment_image(img, brightness=0.15, contrast=0.2, shift_px=4, top_brightness=0.15)
    sn = np.random.get_state()
    np.random.seed(s)
    np.random.uniform(0.8, 1.2)
    np.random.uniform(-0.15, 0.15)
    np.random.uniform(-0.15, 0.15)
    np.random.randint(-4, 5, size=2)
    if not rng_same(sn, np.random.get_state()):
        bad3.append(s)
check('2c. 乱数の順番: コントラスト → 明るさ → 上 1/3 → ずれ', not bad3, {'bad_seeds': bad3})

# ---------------------------------------------------------------- 3. JetFormerDataset(実データ)
import torch  # noqa: E402,F401
drop_path = REPO / 'data' / 'derived' / 'dataset_v2_20261025' / 'drop_v2h.npz'
with np.load(drop_path) as z:
    has_keep = 'keep_impact' in z.files


def make(mod, aug):
    rl = {'drop': mod.load_frame_table(str(drop_path), 'drop')}
    if has_keep:
        rl['keep_impact'] = mod.load_frame_table(str(drop_path), 'keep_impact')
    kw = {} if aug is None else {'aug': aug}
    return mod.JetFormerDataset(str(ds_root), 'train', color_jitter=True, stride=3, use_command=False,
                                mask_impacts=True, impact_g=2.5, **rl, **kw)


def same_item(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, torch.Tensor):
            if not (x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)):
                return False
        elif x != y:
            return False
    return True


AUG1021 = {'brightness': 0.15, 'contrast': 0.2, 'shift_px': 4, 'tof_noise': 0.02}
ds_cases = [('段 1・2(拡張なし)', None, None), ('10/21 の拡張(上 1/3 なし)', AUG1021, AUG1021),
            ('新しい aug に top_brightness 0.0 を明示', AUG1021, dict(AUG1021, top_brightness=0.0))]
for label, aug_old, aug_new in ds_cases:
    do, dn = make(old, aug_old), make(new, aug_new)
    idx_same = do.index == dn.index and len(do) == len(dn)
    picks = np.linspace(0, len(dn) - 1, 24).astype(int)
    bad = []
    for j, n_i in enumerate(picks):
        np.random.seed(1000 + j)
        a = do[int(n_i)]
        sa = np.random.get_state()
        np.random.seed(1000 + j)
        b = dn[int(n_i)]
        sb = np.random.get_state()
        if not (same_item(a, b) and rng_same(sa, sb)):
            bad.append(int(n_i))
    check('3. JetFormerDataset %s: index・24 サンプルの全部の出力・乱数の状態が前と同じ' % label,
          idx_same and not bad, {'len': len(dn), 'aug_attr_new': dn.aug, 'bad': bad[:5]})

# AUG2(ずれ・ToF の雑音は 0 にして比べる): 上 1/3 の外の画素とほかの出力が前と同じ
A2 = {'brightness': 0.15, 'contrast': 0.2, 'shift_px': 0, 'tof_noise': 0.0}
do, dn = make(old, A2), make(new, dict(A2, top_brightness=0.15))
S = new.IMAGE_SIZE
low = int(np.ceil(S / 3.0))
bad, changed = [], 0
for j, n_i in enumerate(np.linspace(0, len(dn) - 1, 12).astype(int)):
    np.random.seed(2000 + j)
    a = do[int(n_i)]
    np.random.seed(2000 + j)
    b = dn[int(n_i)]
    rest = same_item(a[1:], b[1:])
    lower = torch.equal(a[0][:, low:, :], b[0][:, low:, :])
    changed += int(not torch.equal(a[0][:, :max(1, low // 4), :], b[0][:, :max(1, low // 4), :]))
    if not (rest and lower):
        bad.append(int(n_i))
check('3d. AUG2 の上 1/3: モデルの入力の行 %d より下と ToF・IMU・状態・ジャンル・正解は前と同じ、上は変わる' % low,
      not bad and changed > 0, {'bad': bad, 'top_changed_samples': changed, 'of': 12})

# ---------------------------------------------------------------- 4. train.py(AST で取り出す)
def extract(src, names):
    tree = ast.parse(src)
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names) or
            (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), '<extract>', 'exec'), ns)
    return ns


NAMES = {'BENCH_ARGS', 'bench_args_used', 'aug_of'}
ot, nt = extract(old_tr_src, NAMES), extract(new_tr_src, NAMES)


def ns_old(**kw):
    d = dict(label_override_npz=None, aug_brightness=0.0, aug_contrast=0.0, aug_shift_px=0, aug_tof_noise=0.0)
    d.update(kw)
    return argparse.Namespace(**d)


def ns_new(**kw):
    d = dict(label_override_npz=None, aug_brightness=0.0, aug_contrast=0.0, aug_shift_px=0, aug_tof_noise=0.0,
             aug_top_brightness=0.0)
    d.update(kw)
    return argparse.Namespace(**d)


a21 = dict(aug_brightness=0.15, aug_contrast=0.2, aug_shift_px=4, aug_tof_noise=0.02)
r = {
    '既定: aug_of': (ot['aug_of'](ns_old()), nt['aug_of'](ns_new())),
    '既定: bench_args_used': (ot['bench_args_used'](ns_old()), nt['bench_args_used'](ns_new())),
    '10/21: aug_of': (ot['aug_of'](ns_old(**a21)), nt['aug_of'](ns_new(**a21))),
    '10/21: bench_args_used': (ot['bench_args_used'](ns_old(**a21)), nt['bench_args_used'](ns_new(**a21))),
}
check('4a. train.py: 既定と 10/21 の拡張で aug_of・bench_args_used が前と同じ(記録が増えない = 前の last.pth で再開できる)',
      all(a == b for a, b in r.values()), {k: [a, b] for k, (a, b) in r.items()})
aug2 = nt['aug_of'](ns_new(aug_top_brightness=0.15, **a21))
used2 = nt['bench_args_used'](ns_new(aug_top_brightness=0.15, **a21))
check('4b. train.py: AUG2 なら aug に top_brightness 0.15 が入り、記録にも入る(違う引数の再開は断られる)',
      aug2 == dict(brightness=0.15, contrast=0.2, shift_px=4, tof_noise=0.02, top_brightness=0.15) and
      used2.get('aug_top_brightness') == 0.15, {'aug': aug2, 'used': used2})
add = None
for node in ast.walk(ast.parse(new_tr_src)):
    if isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'add_argument' and node.args and \
            isinstance(node.args[0], ast.Constant) and node.args[0].value == '--aug-top-brightness':
        add = {k.arg: ast.unparse(k.value) for k in node.keywords}
check('4c. --aug-top-brightness の既定は 0.0・type float', add is not None and add.get('default') == '0.0' and
      add.get('type') == 'float', add and {k: v for k, v in add.items() if k != 'help'})

# ---------------------------------------------------------------- 5. 済んだ 16 本は C1 の前のコード
names = (['bfast_20261025-e15-s%d' % k for k in range(3)] +
         ['v2_20261025-%s-e30-s%d' % (x, k) for x in ('v2h', 'v2m') for k in range(4)] +
         ['v2_20261025-v2h-e60-s%d' % k for k in range(4)] + ['v2_20261025-v2h-e100-s0'])
bad5 = []
for n in names:
    cs = json.load(open(REPO / 'data' / 'runs' / n / 'run.json')).get('code_sha256') or {}
    if cs.get('tools/train.py') != SHA_OLD['train.py'] or cs.get('jetformer/train_data.py') != SHA_OLD['train_data.py']:
        bad5.append(n)
check('5. 段 1・2・B-FAST の 16 本は C1 の前の train.py・train_data.py で学習した(run.json の code_sha256)',
      not bad5, {'runs': len(names), 'bad': bad5})

summary = {'all_ok': all(x['ok'] for x in results), 'n': len(results), 'results': results,
           'new_sha256': {'tools/train.py': hashlib.sha256(new_tr_src).hexdigest(),
                          'jetformer/train_data.py': hashlib.sha256(new_td_src).hexdigest()},
           'old_commit': OLD_COMMIT}
OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
print('all_ok', summary['all_ok'], '%d 件' % len(results))
sys.exit(0 if summary['all_ok'] else 1)
