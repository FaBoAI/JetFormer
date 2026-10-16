#!/usr/bin/env python3
"""GAINL 腕(任意・2026-10-15。protocol.json の arms.GAINL_optional): tools/train.py をそのまま使い、スロットルのラベルだけを
「電池の効きをそろえた」値に替える(固定の後に書いた。sha256 は eval_sha256.txt・deviations.md)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --mem 3G --lock big --need-mb 3200 -- nice -n 10 python -u \\
        experiments/20261015_real_accuracy/train_gainl.py <tools/train.py と同じ引数>

変えるのは学習と検証の Dataset のラベル(future_actions のスロットル)だけ(tools/train.py・jetformer/train_data.py は書き換えない):
  - 各フレームの r_i = clip(g_t / g_ref, 0.8, 2.0)。g_t = ideas_data/probe_gain.py の ratio_series・running_median(人の
    スロットルと前進 eRPM から、同じシーンの過去 5 秒・今のフレームを含む・30 フレーム未満なら r = 1)、g_ref = {4: 36.41, 5: 36.86}
    (GAIN の後処理と同じ式・同じ値。eval/realacc_lib.py の gain_scale)
  - スロットルのラベルのうち < 1498 を 1498 − (1498 − y) / r_i に(realacc_lib.gain_label。後処理 gain_correct の逆)。ステアは変えない
  - 評価では予測に同じ r の後処理をかけて元のラベルと比べる(realacc_eval.py の GAIN(gainl-s*))
  - --out に gainl.json(式・ラベルの確かめ: r = 1 のフレームのラベルが元と同じ)。--resume は gainl.json がある所だけ
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np

REPO = '/home/jetson/jetformer'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'eval'))
import realacc_lib as RL  # noqa: E402

_spec = importlib.util.spec_from_file_location('jf_train', os.path.join(REPO, 'tools', 'train.py'))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
_pg_spec = importlib.util.spec_from_file_location('probe_gain', os.path.join(HERE, 'ideas_data', 'probe_gain.py'))
PG = importlib.util.module_from_spec(_pg_spec)
_cvd = os.environ.get('CUDA_VISIBLE_DEVICES')      # probe_gain.py は読み込むと CUDA_VISIBLE_DEVICES='' にする(CPU の下調べ用)
_pg_spec.loader.exec_module(PG)
if _cvd is None:                                    # → 学習は GPU で動かすので元に戻す(22:43 の動作確認が CPU で動いて止まった)
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = _cvd

G_REF = {4: 36.41, 5: 36.86}
CLIP = (0.8, 2.0)
WINDOW_S = 5.0
NEUTRAL = 1498.0
STATS = {}


def _argv_value(flag, default=None):
    a = sys.argv
    for i, x in enumerate(a):
        if x == flag and i + 1 < len(a):
            return a[i + 1]
        if x.startswith(flag + '='):
            return x.split('=', 1)[1]
    return default


OUT = _argv_value('--out')
_Base = T.JetFormerDataset


def scene_ratio(a):
    """1 シーンの aligned(dict)→ 各フレームの r(GAIN と同じ式)"""
    g = PG.ratio_series(a)
    gt = PG.running_median(g, a['timestamps_us'].astype(np.float64), WINDOW_S)
    gref = np.array([G_REF.get(int(m), np.nan) for m in a['mode']])
    return RL.gain_scale(gt, gref, *CLIP)


class GainLDataset(_Base):
    """ラベルのスロットルを r でそろえる(学習と検証の両方)"""

    def __init__(self, root, split='train', *args, **kw):
        super().__init__(root, split, *args, **kw)
        used = {}
        for k, i in self.index:
            used.setdefault(k, []).append(i)
        st = {'frames_used': 0, 'r_not_1': 0, 'labels_changed': 0, 'r1_labels_same': True, 'r_mean_used': 0.0}
        rs = []
        for k, a in enumerate(self.scenes):
            r = scene_ratio(a)
            old = np.array(a['future_actions'], copy=True)
            new = RL.gain_label(old.astype(np.float64), r).astype(old.dtype)
            a['future_actions'] = new
            a['_gain_r'] = r.astype(np.float32)
            rows = np.asarray(used.get(k, []), int)
            if len(rows):
                rr = r[rows]
                rs.append(rr)
                st['frames_used'] += int(len(rows))
                st['r_not_1'] += int((np.abs(rr - 1) > 1e-12).sum())
                st['labels_changed'] += int((np.abs(new[rows] - old[rows]) > 0).any(axis=(1, 2)).sum())
                one = rows[np.abs(rr - 1) <= 1e-12]
                if len(one) and not np.array_equal(new[one], old[one]):
                    st['r1_labels_same'] = False
                if not np.array_equal(new[rows][..., 0], old[rows][..., 0]):
                    st['r1_labels_same'] = False          # ステアが変わっていれば失敗として扱う
        st['r_mean_used'] = float(np.concatenate(rs).mean()) if rs else None
        STATS[split] = st


def _marker():
    if not OUT:
        raise SystemExit('--out が必要です')
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'gainl.json')
    if '--resume' in sys.argv:
        if not os.path.exists(path):
            raise SystemExit('--resume: %s がありません(GAINL 以外の学習の続きにはできません)' % path)
    elif os.path.exists(os.path.join(OUT, 'last.pth')):
        raise SystemExit('%s に last.pth があります(続きなら --resume を付ける)' % OUT)


def _write_info():
    path = os.path.join(OUT, 'gainl.json')
    info = {'arm': 'GAINL', 'g_ref': G_REF, 'clip': CLIP, 'window_s': WINDOW_S, 'neutral_us': NEUTRAL,
            'label': '1498 − (1498 − y) / r(y < 1498 のスロットルだけ)', 'stats': STATS,
            'label_check_ok': bool(all(s.get('r1_labels_same') for s in STATS.values()) and STATS),
            'wrapper': os.path.abspath(__file__), 'written_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    print('[GAINL] %s' % json.dumps(STATS, ensure_ascii=False), flush=True)
    if not info['label_check_ok']:
        raise SystemExit('GAINL: ラベルの確かめが通りません(r = 1 のフレームのラベルが元と違う)')


if __name__ == '__main__':
    _marker()
    T.JetFormerDataset = GainLDataset
    _orig_run_epoch = T.run_epoch
    _wrote = {'done': False}

    def run_epoch(*a, **kw):          # Dataset を作った後・最初のエポックの前に gainl.json を書く
        if not _wrote['done']:
            _write_info()
            _wrote['done'] = True
        return _orig_run_epoch(*a, **kw)

    T.run_epoch = run_epoch
    T.main()
