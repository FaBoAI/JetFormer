#!/usr/bin/env python3
"""学習の前の確かめ: 合わせたフォルダーで読み込んだサンプル数が、元のデータセットそれぞれの合計と同じか
(tools/train.py と同じ JetFormerDataset・同じ設定: 間引き 3・衝突の前後を外す 2.5G・今の操作値なし)

    cd ~/jetformer && python -I experiments/20261014_sim_dart_margin/train/count_samples.py <出力 json> <合わせたフォルダー> <元 1> <元 2> ...
"""
import json
import os
import sys

sys.path.insert(0, '/home/jetson/jetformer')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
from jetformer.train_data import JetFormerDataset  # noqa: E402

KW = dict(stride=3, use_command=False, mask_impacts=True, impact_g=2.5)


def count(root):
    out = {}
    for split in ('train', 'val'):
        ds = JetFormerDataset(root, split, color_jitter=(split == 'train'), **KW)
        out[split] = {'samples': len(ds), 'scenes': len(ds.scenes), 'impact_masked': ds.impact_masked,
                      'num_modes': ds.num_modes, 'mode_counts': {str(k): v for k, v in ds.mode_counts().items()}}
    return out


def main(argv):
    if len(argv) < 4:
        raise SystemExit(__doc__)
    res = {'settings': KW, 'combined': {'path': argv[2], **count(argv[2])}, 'sources': []}
    for s in argv[3:]:
        res['sources'].append({'path': s, **count(s)})
    ok = True
    for split in ('train', 'val'):
        tot = sum(r[split]['samples'] for r in res['sources'])
        sc = sum(r[split]['scenes'] for r in res['sources'])
        im = sum(r[split]['impact_masked'] for r in res['sources'])
        same = (res['combined'][split]['samples'] == tot and res['combined'][split]['scenes'] == sc
                and res['combined'][split]['impact_masked'] == im)
        res['check_' + split] = {'combined': res['combined'][split]['samples'], 'sum_of_sources': tot, 'ok': same}
        ok &= same
    res['ok'] = ok
    with open(argv[1], 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k.startswith('check') or k == 'ok'}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
