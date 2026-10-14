"""離脱復活 r4 の確認: 取り込みの学習除外を、r4 の走り(この確認の新しい開始 FP の有効な回、乱し v2 + 表 v2・シールド r4)で端から端まで確かめる(2026-10-15)。
確認 r3 の e2e_label_mask_r3.py(sha256 d34db6ae…)の写しに、次を足した:
  - CK_GENRE8=0 で、ジャンルの表を既定の 0〜5 だけにする(ジャンル 8「シミュレーター」の無い Jetson。scene_import_unset_segments.patch の確かめ)。
    付け直すジャンルは、表に 8 があれば 8、無ければ 4
  - 学習に使える観測の数を、外すべき観測(l = 0 の操作 j に対して k ∈ [j − 10, j])とそれ以外に分けて数える(付け直したあと、外さない観測がジャンル付きに戻るか)
  - 値はいつもフレームの番号が分かる値(1000 + k)にする(CK_UNIQUE=1 と同じ)

確認 r2 の e2e_label_mask.py(rl/check_rc2/)を元に、次を変えた:
  - JetFormer の本体はその場では使わず、写し(JF_ROOT。元のまま / 昇格の段の差分を当てたもの)を読む。データの置き場は一時フォルダ
  - 正解のフレームは、作り直したデータセットの future_actions の値を VEHICLE.npz の操作の値と突き合わせて求める(本体の引き方を真似しない)
  - 取り込み → 展開 → 作り直し → 全シーンのジャンルを付け直し(update_scene(mode_id=8))→ 作り直し、の各段で「学習に使う観測の正解に l = 0 の操作」を数える
  python e2e_label_mask_r3.py <JF_ROOT> <rows.jsonl のフォルダ> <作業フォルダ> <out.json>
"""
import glob
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

JF_ROOT, RUNS, WORK, OUT = sys.argv[1:5]
import os  # noqa: E402
os.environ['JETFORMER_DATA'] = str(Path(WORK) / 'data')
sys.path.insert(0, JF_ROOT)
from jetformer import config, dataset, dataset_import, modes  # noqa: E402

assert Path(dataset.__file__).resolve().is_relative_to(Path(JF_ROOT).resolve()), dataset.__file__
WORK = Path(WORK)
if (WORK / 'src').exists() or (WORK / 'data' / 'datasets').exists():
    shutil.rmtree(WORK)
(WORK / 'data' / 'datasets').mkdir(parents=True, exist_ok=True)
config.DATA_DIR = WORK / 'data'
config.DATASET_DIR = WORK / 'data' / 'datasets'
config.RAW_DIR = WORK / 'data' / 'raw'
config.RAW_DIR.mkdir(parents=True, exist_ok=True)
# ジャンルの表は一時フォルダの写しにする(本物の data/modes.json に書かない)
GENRE8 = os.environ.get('CK_GENRE8', '1') == '1'
if hasattr(modes, 'MODES_FILE'):
    src_modes = Path('/home/jetson/jetformer/data/modes.json')
    dst = WORK / 'data' / 'modes.json'
    if GENRE8:
        shutil.copy(src_modes, dst)
    else:
        dst.write_text(json.dumps(modes.DEFAULT_MODES, ensure_ascii=False))
    modes.MODES_FILE = dst
RELABEL = 8 if any(m['id'] == 8 for m in modes.load()) else 4
os.environ['CK_UNIQUE'] = '1'

SRC = Path('/home/jetson/jetformer/data/datasets/jetformer-simulator-rl-bestlap-20min_20261013-174515_7f3650b0')
rows = [json.loads(line) for p in sorted(glob.glob(RUNS + '/rows_*.jsonl')) for line in open(p)]
fp = [r for r in rows if r.get('set') == 'FP' and r.get('valid_episode') and r['frames'] == 600 and r.get('policy_sha', '').startswith('68e321e40309')]
# 出来事が多く、シールドの替えのある回を先に(乱しのあとの l = 1 の替えた操作も通す)。
# 20:40 直し: 最初の版はシールドの数だけで選び、止まり(600 フレームずっとシールド・出来事 0)の回を選んでいた → 出来事 4 以上・シールド 300 フレーム未満に限る
fp = [r for r in fp if len(r['events']) >= 4 and sum(r['pf']['sh']) < 300]
fp.sort(key=lambda r: (-min(1, sum(r['pf']['sh'])), -len(r['events']), -sum(r['pf']['sh']), int(r['id'][1:])))
pick = fp[:3]
L = [r['pf']['l'] for r in pick]
EV = [r['events'] for r in pick]
splits = json.loads((SRC / 'splits.json').read_text())
scenes = [splits['train'][0], splits['train'][1], splits['val'][0]]
name = 'e2e_recover_r4_mask'
root = WORK / 'src' / name
(root / 'aligned').mkdir(parents=True)
card = json.loads((SRC / 'dataset_card.json').read_text())
card['name'] = name
card['scenes'] = len(scenes)
card.pop('scene_log', None)
(root / 'dataset_card.json').write_text(json.dumps(card, ensure_ascii=False))
(root / 'splits.json').write_text(json.dumps({'train': scenes[:2], 'val': scenes[2:]}))
for i, s in enumerate(scenes):
    shutil.copy(SRC / 'aligned' / ('%s.npz' % s), root / 'aligned' / ('%s.npz' % s))
    z = np.load(SRC / 'aligned' / ('%s.npz' % s))
    if os.environ.get('CK_UNIQUE') == '1':
        # 20:45 足した: 舵の値をフレームの番号が分かる値(1000 + k)にして、正解のフレームを値から一意に求める(同じ値のくり返しで
        # 「どのフレームか」があいまいになり、漏れを多めに数えるのを避ける)。正解は元の並びのまま future_actions[k, j] = vehicle[k + 1 + j]
        arr = {k2: z[k2] for k2 in z.files}
        n0 = len(arr['timestamps_us'])
        arr['vehicle'] = arr['vehicle'].copy()
        arr['vehicle'][:, 0] = 1000 + np.arange(n0)
        fa = arr['future_actions'].copy()
        for j in range(fa.shape[1]):
            fa[:, j, 0] = 1000 + np.minimum(np.arange(n0) + 1 + j, n0 - 1)
        arr['future_actions'] = fa
        np.savez(root / 'aligned' / ('%s.npz' % s), **arr)
    for f in z['cam_files']:
        dst = root / str(f)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SRC / str(f), dst)
    sd = root / 'simulation' / 'scenes' / s
    sd.mkdir(parents=True)
    sess = json.loads((SRC / 'simulation' / 'scenes' / s / 'session.json').read_text())
    sess.setdefault('context', {})['teacher'] = {**(sess['context'].get('teacher') or {}), 'schema': 'jetformer-sim-rl-teacher/v1', 'id': 'jf-rl-recover-v1', 'mode': 'recover',
                                                 'perturbation': {'id': 'jf-perturb-record/v2'}}
    (sd / 'session.json').write_text(json.dumps(sess))
    out = []
    for line in open(SRC / 'simulation' / 'scenes' / s / 'frames.jsonl'):
        r = json.loads(line)
        f = r['frame']
        ph = pick[i]['pf']['ph'][f] if f < len(pick[i]['pf']['ph']) else 0
        r['action_info'] = {'jf': {'v': 1, 'm': 'recover', 'f': f, 'l': int(L[i][f]), 'ev': int(pick[i]['pf']['ev'][f]),
                                   'ph': ['drive', 'perturb', 'shield', 'kick'][ph] if ph < 4 else 'drive', 'sh': int(pick[i]['pf']['sh'][f])}}
        out.append(r)
    (sd / 'frames.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in out))
tarp = WORK / 'src' / ('%s.tar' % name)
with tarfile.open(tarp, 'w') as t:
    t.add(root, arcname=name)

res = {'genre8': GENRE8, 'relabel_mode': RELABEL, 'modes_ids': [m['id'] for m in modes.load()], 'jf_root': JF_ROOT, 'dataset_py': dataset.__file__, 'source_runs': [r['id'] for r in pick], 'scenes': scenes,
       'l0_frames': [int((np.array(l) == 0).sum()) for l in L], 'shield_frames': [int(sum(r['pf']['sh'])) for r in pick], 'events': [len(e) for e in EV]}
info = dataset_import.import_archive(tarp, simulator=True, to_scenes=True)
res['import_summary'] = info.get('summary')
res['import_warnings'] = info.get('warnings')


def label_frames(fut, veh_pwm, k, n):
    """観測 k の正解 10 個が、VEHICLE のどのフレームの値か(値の突き合わせ。候補 k−2〜k+13)。あいまいなら全部の候補を返す"""
    out = []
    for j in range(fut.shape[1]):
        cand = [c for c in range(max(0, k - 2), min(n, k + 14)) if np.allclose(veh_pwm[c, :2], fut[k, j], atol=1e-3)]
        out.append(cand)
    return out


def should_mask(learn, n):
    m = np.zeros(n, bool)
    for j in np.nonzero(~np.asarray(learn[:n]))[0]:
        m[max(0, j - 10):j + 1] = True
    return m


def leak_scan(z, veh_pwm, learn, n):
    use = z['future_valid'].astype(bool) & z['vehicle_valid'].astype(bool) & (z['mode'] != 0)
    sm = should_mask(learn, len(use))
    tail_ok = np.ones(len(use), bool)
    tail_ok[max(0, len(use) - 10):] = False   # 終わりの 10 フレームは正解が足りない
    leak, amb, exact, shifted = [], 0, 0, 0
    for k in np.nonzero(use)[0]:
        lf = label_frames(z['future_actions'], veh_pwm, int(k), n)
        if any(len(c) != 1 for c in lf):
            amb += 1
        bad = any(any(not learn[c] for c in cs) for cs in lf) or any(len(c) == 0 for c in lf)
        if bad:
            leak.append(int(k))
        uniq = [c[0] for c in lf if len(c) == 1]
        if len(uniq) == 10:
            if uniq == list(range(k + 1, k + 11)):
                exact += 1
            else:
                shifted += 1
    return {'usable': int(use.sum()), 'usable_with_l0_or_unmatched_target': len(leak), 'leak_k': leak[:20], 'ambiguous_obs': amb,
            'frame_exact_obs': exact, 'shifted_obs': shifted,
            'should_mask': int(sm.sum()), 'usable_in_should_mask': int((use & sm).sum()),
            'not_masked_obs': int((~sm & tail_ok).sum()), 'usable_not_masked': int((use & ~sm & tail_ok).sum()),
            'mode_values': {int(k_): int(v_) for k_, v_ in zip(*np.unique(z['mode'], return_counts=True))}}


D = config.DATASET_DIR / name
checks = []
for i, s in enumerate(scenes):
    z = np.load(D / 'aligned' / ('%s.npz' % s))
    n = len(z['timestamps_us'])
    learn = np.array(L[i][:n]) == 1
    # 取り込んだ npz の vehicle(現在の操作)は frames.jsonl の操作と同じ並び: 正解の値を vehicle 列と突き合わせる
    veh = z['vehicle']
    c = leak_scan(z, veh, learn, n)
    c.update({'scene': s, 'frames': n})
    checks.append(c)
res['imported'] = checks

exp = info.get('scenes') or {}
res['expanded'] = {k: v for k, v in exp.items() if k != 'created'} | {'created': exp.get('created')}


def rebuilt_scan(tag, created):
    card2 = dataset.build_dataset(tag, created, val_ratio=0.34, min_duration=5.0, seed=0)
    R2 = config.DATASET_DIR / tag
    sp2 = json.loads((R2 / 'splits.json').read_text())
    out = []
    for sname in sp2['train'] + sp2['val']:
        z = np.load(R2 / 'aligned' / ('%s.npz' % sname))
        t = z['timestamps_us']
        i = next(ii for ii, s in enumerate(scenes) if np.array_equal(np.load(SRC / 'aligned' / ('%s.npz' % s))['timestamps_us'], t))
        n = len(t)
        learn = np.array(L[i][:n]) == 1
        vz = np.load(config.RAW_DIR / exp['log'] / created[i] / 'VEHICLE.npz')
        veh_pwm = vz['pwm'][:, :2].astype(np.float32)
        c = leak_scan(z, veh_pwm, learn, min(n, len(veh_pwm)))
        c.update({'rebuilt_scene': sname, 'orig': scenes[i], 'vehicle_quality': str(vz['quality']) if 'quality' in vz.files else None,
                  'veh_dt_us': sorted(set(np.diff(vz['timestamps_us']).tolist()))[:4]})
        out.append(c)
    return {'training_frames': card2.get('training_frames'), 'scenes': out}


if exp.get('created'):
    created = exp['created']
    segs = {}
    for sc in created:
        meta = json.loads((config.RAW_DIR / exp['log'] / sc / 'scene.json').read_text())
        segs[sc] = {'segments': len(meta['mode_segments']), 'unset': sum(1 for g in meta['mode_segments'] if g['mode_id'] == 0),
                    'label_mask_unset_segments': len((meta.get('label_mask') or {}).get('unset_segments', []))}
    res['expanded_segments'] = segs
    res['rebuilt'] = rebuilt_scan('e2e_rebuilt', created)
    # シーンタブでジャンルを付け直す(全シーン)→ 作り直す
    after = {}
    for sc in created:
        dataset.update_scene(sc, mode_id=RELABEL)
        meta = json.loads((config.RAW_DIR / exp['log'] / sc / 'scene.json').read_text())
        after[sc] = {'segments': len(meta['mode_segments']), 'unset': sum(1 for g in meta['mode_segments'] if g['mode_id'] == 0)}
    res['relabel_segments'] = after
    res['rebuilt_after_relabel'] = rebuilt_scan('e2e_rebuilt2', created)
json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1)[:5000])
