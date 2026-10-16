#!/usr/bin/env python3
"""ROT 腕(2026-10-15): tools/train.py をそのまま使い、学習のフレームの間引き(stride 3)の位相をエポックごとに回す。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --mem 3G --lock big --need-mb 3200 -- nice -n 10 python -u \
        experiments/20261015_real_accuracy/train_rot.py <tools/train.py と同じ引数>

変えるのは学習の Dataset の行の選び方だけ(tools/train.py・jetformer/train_data.py は書き換えない。この中で差し替える):
  - 学習の Dataset は stride=1 で作る(future_valid・vehicle_valid・ジャンル未設定・2.5G の衝突の前後の除外は今と同じ)。
    そのあと「フレーム番号 % stride == 位相」の行だけを使う。位相 0 の行は A(--stride 3)の学習の行とまったく同じ
  - 位相 = (エポック − 1 + 種) % stride。30 エポックで位相 0・1・2 を 10 回ずつ。1 エポックのサンプル数は A とほぼ同じ
  - 検証の Dataset・最良のエポックの選び方・学習率・エポック数・バッチ・最適化は A と同じ(stride 3・位相 0)
  - --resume で再開しても、位相はエポック番号から決まるので同じ(--out に rot.json を書き、無い所への再開・ふつうの train.py での
    再開を止める)
  - 各エポックの位相とサンプル数を --out/rot_log.jsonl に書く
"""
import importlib.util
import json
import os
import random
import sys
import time

REPO = '/home/jetson/jetformer'
_spec = importlib.util.spec_from_file_location('jf_train', os.path.join(REPO, 'tools', 'train.py'))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)


def _argv_value(flag, default=None):
    a = sys.argv
    for i, x in enumerate(a):
        if x == flag and i + 1 < len(a):
            return a[i + 1]
        if x.startswith(flag + '='):
            return x.split('=', 1)[1]
    return default


SEED = int(_argv_value('--seed', '0'))
OUT = _argv_value('--out')
MAX_STEPS = int(_argv_value('--max-steps', '0'))
BATCH = int(_argv_value('--batch', '64'))
_Base = T.JetFormerDataset


class RotDataset(_Base):
    """学習のときだけ stride=1 で作り、位相ごとの行の表を持つ(検証はそのまま)"""

    def __init__(self, root, split='train', *args, stride=1, **kw):
        self.rot_stride = int(stride) if split == 'train' else 1
        super().__init__(root, split, *args, stride=(1 if self.rot_stride > 1 else stride), **kw)
        if self.rot_stride > 1:
            full = list(self.index)
            self.rot_index = [[(k, i) for (k, i) in full if i % self.rot_stride == p] for p in range(self.rot_stride)]
            if MAX_STEPS:                       # 動作確認用(train.py の --max-steps と同じ考え: ランダムに抜き出す)
                rng = random.Random(0)
                for v in self.rot_index:
                    rng.shuffle(v)
                self.rot_index = [v[:MAX_STEPS * BATCH] for v in self.rot_index]
            self.set_phase(0)

    def set_phase(self, p):
        self.rot_phase = int(p) % self.rot_stride
        self.index = self.rot_index[self.rot_phase]


_orig_run_epoch = T.run_epoch


def run_epoch(model, loader, device, optimizer=None, amp=True, epoch=0, phase='train', **kw):
    ds = loader.dataset
    if optimizer is not None and getattr(ds, 'rot_stride', 1) > 1:
        p = (int(epoch) - 1 + SEED) % ds.rot_stride
        ds.set_phase(p)            # DataLoader の workers はこの後の iter() で fork されるので、新しい行の表を使う
        print('[ROT] epoch %d: 位相 %d・学習 %d サンプル' % (epoch, p, len(ds)), flush=True)
        if OUT:
            with open(os.path.join(OUT, 'rot_log.jsonl'), 'a') as fh:
                fh.write(json.dumps({'epoch': int(epoch), 'phase': p, 'samples': len(ds),
                                     'time': time.strftime('%Y-%m-%d %H:%M:%S')}) + '\n')
    return _orig_run_epoch(model, loader, device, optimizer=optimizer, amp=amp, epoch=epoch, phase=phase, **kw)


def _marker():
    if not OUT:
        raise SystemExit('--out が必要です')
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'rot.json')
    info = {'arm': 'ROT', 'rule': 'phase = (epoch - 1 + seed) % stride', 'seed': SEED, 'wrapper': os.path.abspath(__file__)}
    if '--resume' in sys.argv:
        if not os.path.exists(path):
            raise SystemExit('--resume: %s がありません(ROT 以外の学習の続きにはできません)' % path)
        old = json.load(open(path))
        if old.get('seed') != SEED or old.get('rule') != info['rule']:
            raise SystemExit('--resume: rot.json が違います: %s' % old)
    else:
        if os.path.exists(os.path.join(OUT, 'last.pth')):
            raise SystemExit('%s に last.pth があります(続きなら --resume を付ける)' % OUT)
        info['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        json.dump(info, open(path, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    _marker()
    T.JetFormerDataset = RotDataset
    T.run_epoch = run_epoch
    T.main()
