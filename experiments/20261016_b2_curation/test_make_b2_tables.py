"""make_b2_tables.py の試験(CPU)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= /home/jetson/bin/jf-heavy --mem 1500M --lock none -- nice -n 10 \
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q experiments/20261016_b2_curation/test_make_b2_tables.py

純粋な関数の試験(すぐ終わる)と、本物のデータでの試験(生ログ・ラベル・データセットを読むだけ。2〜3 分)。
本物の試験は、出力を一時フォルダーに書き、tools/train.py と同じ読み込み(JetFormerDataset)で数が合うかを確かめる。
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_b2_tables as M  # noqa: E402


def test_collided_end_uses_longest():
    assert M.collided_end(10.0, None, []) == pytest.approx(13.5)          # tg なし: ta + 3.5
    assert M.collided_end(10.0, 12.0, []) == pytest.approx(11.7)          # tg − 0.3
    assert M.collided_end(10.0, 10.5, []) == pytest.approx(11.0)          # 少なくとも ta + 1.0
    assert M.collided_end(10.0, 12.0, [10.4, 11.2]) == pytest.approx(13.2)  # 最後の衝突 + 2.0


def test_drop_reason_fast_and_collision():
    t = np.arange(0.0, 30.0, 0.1)
    passes = [{'ta_s': 5.0, 'tg_s': 7.0, 'entry_erpm': 6500.0, 'r1_collision': False},          # 速い: [3, 6]
              {'ta_s': 15.0, 'tg_s': 17.0, 'entry_erpm': 5000.0, 'r1_collision': False},        # 遅い: 残す
              {'ta_s': 25.0, 'tg_s': None, 'entry_erpm': 6300.0, 'r1_collision': True,           # 速い + 衝突: [23, 28.5]
               'r1_collision_t_s': [25.4]}]
    r = M.drop_reason(t, passes)
    assert set(np.unique(r)) == {0, 1, 2}
    assert (r[(t > 3.0 + 1e-6) & (t < 6.0 - 1e-6)] == 1).all()
    assert (r[(t > 6.05) & (t < 22.95)] == 0).all()
    assert (r[(t > 23.0 + 1e-6) & (t < 28.5 - 1e-6)] == 2).all()
    assert (r[t > 28.55] == 0).all()


def test_drop_reason_slow_collision_drops_whole_pass():
    t = np.arange(0.0, 12.0, 0.05)
    r = M.drop_reason(t, [{'ta_s': 5.0, 'tg_s': 9.0, 'entry_erpm': 4000.0, 'r1_collision': True, 'r1_collision_t_s': [5.3]}])
    assert (r[(t > 3.0 + 1e-6) & (t < 8.7 - 1e-6)] == 2).all()          # [ta − 2, tg − 0.3](境目の丸めは見ない)
    assert (r[t < 2.95] == 0).all() and (r[t > 8.75] == 0).all()


def test_drop_reason_entry_threshold_is_inclusive_and_none_safe():
    t = np.array([4.0, 5.0])
    assert M.drop_reason(t, [{'ta_s': 5.0, 'tg_s': 7.0, 'entry_erpm': 6200.0, 'r1_collision': False}]).tolist() == [1, 1]
    assert M.drop_reason(t, [{'ta_s': 5.0, 'tg_s': 7.0, 'entry_erpm': 6199.0, 'r1_collision': False}]).tolist() == [0, 0]
    assert M.drop_reason(t, [{'ta_s': 5.0, 'tg_s': 7.0, 'entry_erpm': None, 'r1_collision': False}]).tolist() == [0, 0]


def test_remap_value():
    assert M.remap_value('20261010-152405-0003') == 4
    assert M.remap_value('20261010-152707-0001') == 4
    assert M.remap_value('20261010-150424-0001') == -1
    assert M.remap_value('20261010-200916-0004') == -1


def test_used_samples_matches_rule():
    n = 9
    fv = np.ones(n, bool)
    vv = np.ones(n, bool)
    mode = np.array([5] * n)
    imp = np.zeros(n, bool)
    imp[3] = True
    drop = np.zeros(n, bool)
    drop[6] = True
    assert np.nonzero(M.used_samples(fv, vv, mode, imp))[0].tolist() == [0, 6]
    assert np.nonzero(M.used_samples(fv, vv, mode, imp, drop))[0].tolist() == [0]
    mode[0] = 0
    assert np.nonzero(M.used_samples(fv, vv, mode, imp))[0].tolist() == [6]


REAL = Path('/home/jetson/jetformer/data/datasets/clean_driving_20261011/splits.json')


@pytest.mark.skipif(not REAL.exists(), reason='本物のデータが無い')
def test_real_tables(tmp_path):
    from jetformer import config
    from jetformer.train_data import JetFormerDataset, load_frame_table
    card = M.build(out_dir=tmp_path)
    # 1) course_passes が 10/11 の passes.json と同じ(131 通過・同じ値)
    passes = json.loads((tmp_path / 'passes_b2.json').read_text())
    old = json.loads((config.DATA_DIR / 'rl' / 'ai_mix_20261011' / 'passes.json').read_text())
    key = lambda q: (q['log'], round(q['ta_s'], 3))   # noqa: E731
    o = {key(q): q for q in old}
    assert len(passes) == len(old) == 131
    for q in passes:
        for f in ('entry_erpm', 'r1_collision', 'r1_strong', 'tg_s', 'ai', 'de_collision'):
            assert o[key(q)][f] == q[f]
    # 2) B と同じ数え方で 5912 / 1559(モデル B・A の result.json の train_samples・val_samples)
    assert card['counts']['train']['used_B'] == 5912 and card['counts']['val']['used_B'] == 1559
    # 3) tools/train.py と同じ読み込みで、表を渡したときの数が card と同じ
    rl = {'drop': load_frame_table(tmp_path / 'drop_b2.npz', 'drop'),
          'mode_remap': load_frame_table(tmp_path / 'remap_b2.npz', 'mode')}
    root = str(config.DATASET_DIR / M.DATASET)
    for split in ('train', 'val'):
        ds = JetFormerDataset(root, split, stride=3, use_command=False, mask_impacts=True, impact_g=2.5, **rl)
        assert len(ds) == card['counts'][split]['used_B2']
        assert {str(k): v for k, v in ds.mode_counts().items()} == card['counts'][split]['used_B2_by_genre_b2']
    # 4) 付け替えは 152405・152707 だけ、外すのは人の R1 の通過の窓だけ
    z = np.load(tmp_path / 'remap_b2.npz')
    sc = z['scene'].astype(str)
    relabel = np.array([s[:15] in M.RELABEL_LOGS for s in sc])
    assert (z['mode'][relabel] == 4).all() and (z['mode'][~relabel] == -1).all()
    d = np.load(tmp_path / 'drop_b2.npz')
    assert d['drop'].tolist() == (d['reason'] > 0).tolist()
    splits = json.loads(REAL.read_text())
    assert set(np.unique(sc)) == set(splits['train'] + splits['val'])
