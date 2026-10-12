"""自前の DinoViTS14(jetformer/policies/vit.py)と、公式の dinov2 のソース(github.com/facebookresearch/dinov2 7764ea0 の
生のファイルを取得したもの・hub と同じ設定)を同じ重みで比べる。
CPU だけ。python -I で動かす(引数: 公式のソースのフォルダー)"""
import glob
import json
import os
import sys
import time

official_dir = sys.argv[1]
sys.path.insert(0, official_dir)
sys.path.insert(0, '/home/jetson/jetformer')
os.environ['XFORMERS_DISABLED'] = '1'

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
from dinov2.models import vision_transformer as vits  # noqa: E402
from jetformer.policies.vit import DinoViTS14  # noqa: E402
from jetformer.train_data import normalize_inputs  # noqa: E402

W = '/home/jetson/jetformer/data/pretrained/dinov2_vits14_pretrain.pth'
off = vits.vit_small(patch_size=14, img_size=518, init_values=1.0, block_chunks=0, num_register_tokens=0,
                     interpolate_antialias=False, interpolate_offset=0.1)
off.load_state_dict(torch.load(W, map_location='cpu', weights_only=True), strict=True)
off.eval()
mine = DinoViTS14(img_size=224).load_official(W).eval()

files = sorted(glob.glob('/home/jetson/jetformer/data/datasets/clean_driving_20261011/samples/CAM_FRONT/*/*.jpg'))
rng = np.random.default_rng(0)
pick = [files[i] for i in rng.choice(len(files), 8, replace=False)]
imgs = []
for f in pick:
    im, _, _, _ = normalize_inputs(cv2.imread(f), np.zeros((42, 54), np.uint16), np.zeros((10, 6), np.float32),
                                   np.zeros(4, np.float32), False)
    imgs.append(im)
x = torch.from_numpy(np.stack(imgs)).float()
g = torch.Generator().manual_seed(1)
x = torch.cat([x, torch.randn(2, 3, 224, 224, generator=g)])
res = {}
with torch.no_grad():
    t = time.time()
    o = off.forward_features(x)
    ref = torch.cat([o['x_norm_clstoken'][:, None], o['x_norm_patchtokens']], dim=1)
    t1 = time.time()
    got = mine(x)
    t2 = time.time()
    # 位置の埋め込み(公式は毎回補間、自前は読み込みのとき1回)
    pe_off = off.interpolate_pos_encoding(torch.zeros(1, 257, 384), 224, 224)
d = (got - ref).abs()
cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
res = {'official_commit': open(os.path.join(official_dir, 'COMMIT')).read().strip() if os.path.exists(
    os.path.join(official_dir, 'COMMIT')) else None,
    'frames_real': len(pick), 'frames_random': 2, 'shape': list(got.shape),
    'max_abs_diff': float(d.max()), 'mean_abs_diff': float(d.mean()), 'ref_absmax': float(ref.abs().max()),
    'min_token_cos': float(cos.min()), 'cls_max_abs_diff': float(d[:, 0].max()),
    'pos_embed_max_abs_diff': float((mine.pos_embed - pe_off).abs().max()),
    'seconds_official': round(t1 - t, 2), 'seconds_mine': round(t2 - t1, 2),
    'state_dict_keys_official': len(off.state_dict()), 'state_dict_keys_mine': len(mine.state_dict())}
print(json.dumps(res, indent=1))
