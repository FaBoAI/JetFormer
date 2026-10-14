"""離脱復活の確認 r2: 取り込みの学習除外(jetformer/sim/label_mask.py)を、本物のシミュレーターのデータセットの形で端から端まで確かめる(2026-10-14)。

ベストラップの記録(data/datasets/jetformer-simulator-rl-bestlap-20min_…)の 2 シーンを写し、frames.jsonl に、確認 r2 の新しい開始で
離脱復活の方策が走った回の info.jf(l・ev)を入れ、session.json の teacher を強化学習モードのものにして tar にし、
一時フォルダの JetFormer(config を差し替え)で import_archive → シーンタブへの展開 → 展開したシーンから build_dataset で作り直し、を通す。
  python rl/check_rc2/e2e_label_mask.py <F2-P の rows.jsonl のフォルダ> <作業フォルダ> <out.json>
"""
import glob
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import config, dataset, dataset_import  # noqa: E402

RUNS, WORK, OUT = sys.argv[1:4]
WORK = Path(WORK)
if WORK.exists():
    shutil.rmtree(WORK)
(WORK / 'data' / 'datasets').mkdir(parents=True)
config.DATA_DIR = WORK / 'data'
config.DATASET_DIR = WORK / 'data' / 'datasets'
config.RAW_DIR = WORK / 'data' / 'raw'
config.RAW_DIR.mkdir(parents=True)

SRC = Path('/home/jetson/jetformer/data/datasets/jetformer-simulator-rl-bestlap-20min_20261013-174515_7f3650b0')
rows = [json.loads(line) for p in sorted(glob.glob(RUNS + '/rows_*.jsonl')) for line in open(p)]
valid = [r for r in rows if r.get('set') == 'F2-P' and r.get('valid_episode')]
valid.sort(key=lambda r: int(r['id'][1:]))
L = [valid[0]['pf']['l'], valid[1]['pf']['l']]
EV = [valid[0]['events'], valid[1]['events']]
splits = json.loads((SRC / 'splits.json').read_text())
scenes = [splits['train'][0], splits['val'][0]]
name = 'e2e_recover_mask'
root = WORK / 'src' / name
(root / 'aligned').mkdir(parents=True)
card = json.loads((SRC / 'dataset_card.json').read_text())
card['name'] = name
card['scenes'] = 2
card.pop('scene_log', None)
(root / 'dataset_card.json').write_text(json.dumps(card, ensure_ascii=False))
(root / 'splits.json').write_text(json.dumps({'train': [scenes[0]], 'val': [scenes[1]]}))
case_drop = {}
for i, s in enumerate(scenes):
    shutil.copy(SRC / 'aligned' / ('%s.npz' % s), root / 'aligned' / ('%s.npz' % s))
    z = np.load(SRC / 'aligned' / ('%s.npz' % s))
    for f in z['cam_files']:
        dst = root / str(f)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SRC / str(f), dst)
    sd = root / 'simulation' / 'scenes' / s
    sd.mkdir(parents=True)
    sess = json.loads((SRC / 'simulation' / 'scenes' / s / 'session.json').read_text())
    sess.setdefault('context', {})['teacher'] = {**(sess['context'].get('teacher') or {}), 'schema': 'jetformer-sim-rl-teacher/v1', 'id': 'jf-rl-recover-v1', 'mode': 'recover'}
    (sd / 'session.json').write_text(json.dumps(sess))
    out = []
    for line in open(SRC / 'simulation' / 'scenes' / s / 'frames.jsonl'):
        r = json.loads(line)
        f = r['frame']
        r['action_info'] = {'jf': {'v': 1, 'm': 'recover', 'f': f, 'l': int(L[i][f]), 'ev': 0}}
        out.append(r)
    # 2 つ目のシーン: ブラウザが info を落とした行を 1 つ作る(fail closed の確かめ)
    if i == 1:
        drop = 300
        out[drop]['action_info'] = {}
        case_drop = {'scene': s, 'frame': drop}
    (sd / 'frames.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in out))
tarp = WORK / 'src' / ('%s.tar' % name)
with tarfile.open(tarp, 'w') as t:
    t.add(root, arcname=name)

res = {'source_runs': [valid[0]['id'], valid[1]['id']], 'scenes': scenes, 'drop_case': case_drop}
info = dataset_import.import_archive(tarp, simulator=True, to_scenes=True)
res['import_summary'] = info['summary']
res['import_warnings'] = info['warnings']
D = config.DATASET_DIR / name
cardI = json.loads((D / 'dataset_card.json').read_text())
res['card_label_mask'] = {k: v for k, v in cardI['label_mask'].items() if k != 'per_scene'}
res['card_training_frames'] = cardI.get('training_frames')


def expected_mask(l, n, drop=None):
    learn = np.array(l[:n]) == 1
    if drop is not None:
        learn[drop] = False
    m = np.zeros(n, bool)
    for j in np.nonzero(~learn)[0]:
        m[max(0, j - 10):j] = True
    return m


checks = []
for i, s in enumerate(scenes):
    z = np.load(D / 'aligned' / ('%s.npz' % s))
    zo = np.load(SRC / 'aligned' / ('%s.npz' % s))
    n = len(z['timestamps_us'])
    m = expected_mask(L[i], n, case_drop['frame'] if i == 1 else None)
    fv, fvo, md = z['future_valid'].astype(bool), zo['future_valid'].astype(bool), z['mode']
    exp_fv = fvo & ~m
    # 学習に使う観測(future_valid・vehicle_valid・mode != 0)の正解 k+1〜k+10 に l = 0 の操作が入っていないか
    learn = np.array(L[i][:n]) == 1
    if i == 1:
        learn[case_drop['frame']] = False
    use = fv & z['vehicle_valid'].astype(bool) & (md != 0)
    leak = [int(k) for k in np.nonzero(use)[0] if not learn[k + 1:k + 11].all()]
    checks.append({'scene': s, 'frames': n, 'l0_frames': int((~learn).sum()), 'masked_expected': int(m.sum()),
                   'future_valid_matches_rule': bool((fv == exp_fv).all()), 'mode0_matches_mask': bool(((md == 0) == m).all()),
                   'last10_false_kept': bool((~fv[-10:]).all()), 'vehicle_valid_unchanged': bool((z['vehicle_valid'] == zo['vehicle_valid']).all()),
                   'other_arrays_unchanged': all(np.array_equal(z[k], zo[k]) for k in zo.files if k not in ('future_valid', 'mode')),
                   'usable': int(use.sum()), 'usable_with_l0_in_target': leak})
res['imported'] = checks

# シーンタブへの展開 → 展開したシーンから作り直したデータセット
exp = info.get('scenes') or {}
res['expanded'] = exp
rebuilt_checks = []
if exp.get('created'):
    for sc in exp['created']:
        meta = json.loads((config.RAW_DIR / exp['log'] / sc / 'scene.json').read_text())
        rebuilt_checks.append({'scene': sc, 'mode_segments': len(meta['mode_segments']), 'unset_segments': sum(1 for g in meta['mode_segments'] if g['mode_id'] == 0)})
    card2 = dataset.build_dataset('e2e_rebuilt', exp['created'], val_ratio=0.5, min_duration=5.0, seed=0)
    R2 = config.DATASET_DIR / 'e2e_rebuilt'
    sp2 = json.loads((R2 / 'splits.json').read_text())
    for sname in sp2['train'] + sp2['val']:
        z = np.load(R2 / 'aligned' / ('%s.npz' % sname))
        # どの元のシーンか(時刻で合わせる)
        t = z['timestamps_us']
        i = next(ii for ii, s in enumerate(scenes) if np.array_equal(np.load(SRC / 'aligned' / ('%s.npz' % s))['timestamps_us'], t))
        n = len(t)
        learn = np.array(L[i][:n]) == 1
        if i == 1:
            learn[case_drop['frame']] = False
        veh = np.load(config.RAW_DIR / exp['log'] / exp['created'][i] / 'VEHICLE.npz')
        vt = veh['timestamps_us']
        frame_us = 1e6 / config.CAMERA['fps']
        idx = np.stack([np.searchsorted(vt, t + int((k + 1) * frame_us), side='right') - 1 for k in range(10)], axis=1)
        use = z['future_valid'].astype(bool) & z['vehicle_valid'].astype(bool) & (z['mode'] != 0)
        leak = [int(k) for k in np.nonzero(use)[0] if not learn[np.clip(idx[k], 0, n - 1)].all()]
        shifted = int((idx[:n - 11] != (np.arange(n - 11)[:, None] + np.arange(1, 11)[None, :])).any(axis=1).sum())
        leak_detail = [{'k': k, 'l0_targets': int((~learn[np.clip(idx[k], 0, n - 1)]).sum()), 'first_target_frame': int(idx[k][0])} for k in leak[:20]]
        rebuilt_checks.append({'rebuilt_scene': sname, 'orig': scenes[i], 'usable': int(use.sum()), 'usable_masked_in_import': int((use & expected_mask(L[i], n, case_drop['frame'] if i == 1 else None)).sum()),
                               'obs_with_shifted_future_index': shifted, 'usable_with_l0_in_target': len(leak), 'events': len(EV[i]), 'leak_detail': leak_detail})
    res['rebuilt_card_training_frames'] = card2.get('training_frames')
res['rebuilt'] = rebuilt_checks

# 展開したシーンのジャンルをシーンタブで付け直した場合(update_scene(mode_id=8))
if exp.get('created'):
    sc = exp['created'][0]
    before = json.loads((config.RAW_DIR / exp['log'] / sc / 'scene.json').read_text())['mode_segments']
    dataset.update_scene(sc, mode_id=8)
    after = json.loads((config.RAW_DIR / exp['log'] / sc / 'scene.json').read_text())['mode_segments']
    res['relabel_scene_genre'] = {'segments_before': len(before), 'unset_before': sum(1 for g in before if g['mode_id'] == 0),
                                  'segments_after': len(after), 'unset_after': sum(1 for g in after if g['mode_id'] == 0)}
json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1)[:6000])
