import os, sys, json
sys.path.insert(0, sys.argv[1]); os.environ['XFORMERS_DISABLED'] = '1'
import torch
from dinov2.models import vision_transformer as vits
off = vits.vit_small(patch_size=14, img_size=518, init_values=1.0, block_chunks=0, num_register_tokens=0,
                     interpolate_antialias=False, interpolate_offset=0.1)
off.load_state_dict(torch.load('/home/jetson/jetformer/data/pretrained/dinov2_vits14_pretrain.pth', map_location='cpu', weights_only=True), strict=True)
off.eval()
g = torch.Generator().manual_seed(0)
x = torch.randn(1, 3, 224, 224, generator=g)
with torch.no_grad():
    o = off.forward_features(x)
cls = o['x_norm_clstoken'][0]; pt = o['x_norm_patchtokens'][0]
print(json.dumps({'cls8': [round(float(v), 5) for v in cls[:8]], 'p0_4': [round(float(v), 5) for v in pt[0, :4]],
                  'p255_4': [round(float(v), 5) for v in pt[255, :4]], 'patch_mean': round(float(pt.mean()), 6),
                  'patch_std': round(float(pt.std()), 6), 'cls_norm': round(float(cls.norm()), 5)}))
