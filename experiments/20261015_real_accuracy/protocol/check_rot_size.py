#!/usr/bin/env python3
"""ROT の位相と SIZE の drop の表を、学習なしで確かめる(CPU。Dataset を作るだけ。画像は読まない)。
  1. ROT の位相 0 の行 = A(stride 3)の学習の行と同じ。位相 0・1・2 は重ならず、合わせると stride 1 の行と同じ
  2. SIZE の Q・H・T: 学習のサンプル数が size_subsets.json と同じ、検証は A と同じ 1559
"""
import json, os, sys
REPO = '/home/jetson/jetformer'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], '--seed', '0', '--out', '/tmp/none']
sys.path.insert(0, REPO)
import importlib.util  # noqa
spec = importlib.util.spec_from_file_location('train_rot', os.path.join(HERE, '..', 'train_rot.py'))
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
from jetformer.train_data import JetFormerDataset, load_frame_table  # noqa
DS = REPO + '/data/datasets/clean_driving_20261011'
kw = dict(use_command=False, mask_impacts=True, impact_g=2.5)
A = JetFormerDataset(DS, 'train', stride=3, **kw)
F = JetFormerDataset(DS, 'train', stride=1, **kw)
Rt = R.RotDataset(DS, 'train', color_jitter=True, stride=3, **kw)
Rv = R.RotDataset(DS, 'val', stride=3, **kw)
Av = JetFormerDataset(DS, 'val', stride=3, **kw)
res = {'A_train': len(A), 'stride1_train': len(F), 'rot_phase_sizes': [len(v) for v in Rt.rot_index],
       'phase0_equals_A': Rt.rot_index[0] == A.index,
       'phases_disjoint_union_equals_stride1': sorted(sum(Rt.rot_index, [])) == sorted(F.index),
       'val_same_as_A': Rv.index == Av.index and len(Rv) == 1559}
sub = json.load(open(os.path.join(REPO, 'experiments/20261015_real_accuracy/size/size_subsets.json')))
for name in ('Q', 'H', 'T'):
    rl = {'drop': load_frame_table(os.path.join(REPO, 'experiments/20261015_real_accuracy/size', 'size_%s.npz' % name), 'drop')}
    t = JetFormerDataset(DS, 'train', stride=3, **kw, **rl)
    v = JetFormerDataset(DS, 'val', stride=3, **kw, **rl)
    res['size_%s' % name] = {'train': len(t), 'expected': sub['subsets'][name]['samples_total'], 'val': len(v),
                             'ok': len(t) == sub['subsets'][name]['samples_total'] and v.index == Av.index}
print(json.dumps(res, ensure_ascii=False, indent=1))
assert res['phase0_equals_A'] and res['phases_disjoint_union_equals_stride1'] and res['val_same_as_A']
assert all(res['size_%s' % n]['ok'] for n in ('Q', 'H', 'T'))
print('OK')
