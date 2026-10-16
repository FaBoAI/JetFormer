#!/usr/bin/env python3
"""SIZE 腕(実機のデータの量と誤差の曲線)の学習シーンの部分集合と --drop-mask の npz を作る(CPU だけ。予測・誤差は見ない)。

選び方(決め打ち・乱数の種 20261015):
  - 学習の 41 シーンを、ジャンル(4 高速・5 低速)ごとに分ける(1 シーン = 1 ジャンル。混ざっていたら止める)
  - ジャンルごとに、ログの順を乱数で決め、各ログのシーンも乱数で並べ、ログを順に 1 つずつ回して取る(ログが偏らないように)
  - この並びの先頭から、ジャンルごとの学習サンプル数(A と同じ数え方: stride 3・future_valid & vehicle_valid・
    ジャンル未設定でない・2.5G の衝突の前後でない)が、そのジャンルの全体 × f 以上になるまで取る
  - f = 0.25(Q)・0.5(H)・0.75(T)。同じ並びの先頭なので Q ⊂ H ⊂ T ⊂ 全部(入れ子)
  - drop の表は 52 シーン全部の全フレーム(選ばなかった学習シーン = True、ほか = False)。train.py の _per_frame は
    表に無いシーンをエラーにするので、検証のシーンも False で入れる
出力: experiments/20261015_real_accuracy/size/size_{Q,H,T}.npz と size_subsets.json

  cd ~/jetformer && CUDA_VISIBLE_DEVICES= jf-heavy --mem 1500M --lock none -- nice -n 10 python experiments/20261015_real_accuracy/size/make_size_masks.py
"""
import hashlib, json, os, sys
import numpy as np
REPO = '/home/jetson/jetformer'
sys.path.insert(0, REPO)
from jetformer.train_data import impact_mask  # noqa
HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(REPO, 'data/datasets/clean_driving_20261011')
SEED = 20261015
FRACS = {'Q': 0.25, 'H': 0.5, 'T': 0.75}


def samples(a):
    ok = a['future_valid'] & a['vehicle_valid'] & (a['mode'] != 0)
    ok &= ~impact_mask(a['timestamps_us'], a['imu'], 2.5)
    ok &= (np.arange(len(ok)) % 3) == 0
    return int(ok.sum())


def main():
    sp = json.load(open(os.path.join(DS, 'splits.json')))
    info = {}
    for s in sp['train'] + sp['val']:
        a = dict(np.load(os.path.join(DS, 'aligned', s + '.npz')))
        modes = sorted(set(int(m) for m in np.unique(a['mode'][a['future_valid'] & a['vehicle_valid']]) if m != 0))
        info[s] = {'n_frames': int(len(a['mode'])), 'modes': modes, 'samples_stride3': samples(a) if s in sp['train'] else None}
    train = sp['train']
    for s in train:
        if len(info[s]['modes']) != 1:
            raise SystemExit('シーン %s のジャンルが 1 つではありません: %s' % (s, info[s]['modes']))
    total = {g: sum(info[s]['samples_stride3'] for s in train if info[s]['modes'][0] == g) for g in (4, 5)}
    rng = np.random.default_rng(SEED)
    order = {}
    for g in (4, 5):
        sc = [s for s in train if info[s]['modes'][0] == g]
        logs = sorted({s[:15] for s in sc})
        logs = [logs[i] for i in rng.permutation(len(logs))]
        per = {l: [x for x in sorted(sc) if x[:15] == l] for l in logs}
        per = {l: [v[i] for i in rng.permutation(len(v))] for l, v in per.items()}
        seq = []
        while any(per.values()):
            for l in logs:
                if per[l]:
                    seq.append(per[l].pop(0))
        order[g] = seq
    subsets = {}
    for name, f in FRACS.items():
        chosen = []
        for g in (4, 5):
            acc = 0
            for s in order[g]:
                if acc >= f * total[g]:
                    break
                chosen.append(s)
                acc += info[s]['samples_stride3']
        subsets[name] = sorted(chosen)
    assert set(subsets['Q']) <= set(subsets['H']) <= set(subsets['T']) <= set(train)
    out = {'seed': SEED, 'fractions': FRACS, 'train_samples_full': total, 'order': order, 'subsets': {}, 'files': {}}
    for name, chosen in subsets.items():
        sc_rows, fr_rows, dr_rows = [], [], []
        for s in train + sp['val']:
            n = info[s]['n_frames']
            drop = (s in train) and (s not in chosen)
            sc_rows += [s] * n; fr_rows.append(np.arange(n)); dr_rows.append(np.full(n, drop))
        path = os.path.join(HERE, 'size_%s.npz' % name)
        np.savez_compressed(path, scene=np.array(sc_rows), frame=np.concatenate(fr_rows).astype(np.int64),
                            drop=np.concatenate(dr_rows))
        smp = {g: sum(info[s]['samples_stride3'] for s in chosen if info[s]['modes'][0] == g) for g in (4, 5)}
        out['subsets'][name] = {'scenes': chosen, 'n_scenes': len(chosen), 'logs': sorted({s[:15] for s in chosen}),
                                'samples_stride3': smp, 'samples_total': sum(smp.values()),
                                'fraction_of_full': round(sum(smp.values()) / sum(total.values()), 3),
                                'genre4_share': round(smp[4] / max(sum(smp.values()), 1), 3)}
        out['files'][name] = {'path': path, 'sha256': hashlib.sha256(open(path, 'rb').read()).hexdigest()}
    out['full'] = {'n_scenes': len(train), 'samples_total': sum(total.values()),
                   'genre4_share': round(total[4] / sum(total.values()), 3)}
    json.dump(out, open(os.path.join(HERE, 'size_subsets.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'order'}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
