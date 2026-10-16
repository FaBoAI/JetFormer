"""prepare_b2.py の試験(CPU)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= /home/jetson/bin/jf-heavy --mem 1500M --lock none -- nice -n 10 \
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q experiments/20261016_b2_curation/test_prepare_b2.py

小さな作り物のデータセットで、ハードリンクのデータセットの作り方・確かめ方を試す(元のファイルが変わらないこと・
リンクと新しいファイルの区別・半端なフォルダーからの作り直し)。ログの読み方・外したフレームの通過の割り当て・
protocol のコマンドから CPU の動作確認のコマンドを作る関数も試す。本物の試験は、作ったデータセットがあるときだけ。
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_b2_tables as M  # noqa: E402
import prepare_b2 as P  # noqa: E402


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _fake_dataset(root):
    """元のデータセットの形の小さな作り物(付け替えるログ 152405 のシーン 1 つ + そのままのシーン 1 つ)"""
    root = Path(root)
    name = 'fake_ds'
    ver = 'v1.0-' + name
    modes = [{'id': 0, 'name': '未設定'}, {'id': 4, 'name': '高速走行'}, {'id': 5, 'name': '低速走行'}]
    scenes = ['20261010-152405-0001', '20261010-135449-0001']
    (root / 'aligned').mkdir(parents=True)
    (root / ver).mkdir()
    (root / 'can_bus').mkdir()
    for k, s in enumerate(scenes):
        n = 6
        mode = np.array([5, 5, 0, 5, 5, 5], np.int16)
        np.savez_compressed(root / 'aligned' / (s + '.npz'), timestamps_us=np.arange(n, dtype=np.int64) + 1000 * k,
                            tof=np.full((n, 2, 3), k, np.uint16), cam_files=np.array(['samples/CAM_FRONT/%s/%d.jpg' % (s, i)
                                                                                      for i in range(n)]), mode=mode)
        d = root / 'samples' / 'CAM_FRONT' / s
        d.mkdir(parents=True)
        for i in range(n):
            (d / ('%d.jpg' % i)).write_bytes(b'jpg%d%d' % (k, i))
        (root / 'can_bus' / (s + '_vehicle_monitor.json')).write_text('[]')
    (root / 'splits.json').write_text(json.dumps({'train': [scenes[1]], 'val': [scenes[0]]}))
    (root / 'dataset_card.json').write_text(json.dumps({'name': name, 'version': ver, 'created_at': '2026-10-11T02:18:28',
                                                        'modes': modes, 'mode_frames': {'5': 10}}, ensure_ascii=False))
    (root / ver / 'scene.json').write_text(json.dumps(
        [{'name': s, 'mode_id': 5, 'mode_name': '低速走行',
          'mode_segments': [{'start_s': 0.0, 'end_s': 1.0, 'mode_id': 5, 'mode_name': '低速走行'}]} for s in scenes],
        ensure_ascii=False))
    (root / ver / 'sample.json').write_text('[]')
    return root, scenes


def test_relabel_mode_keeps_unset():
    m = np.array([5, 0, 5, 4], np.int16)
    out = P.relabel_mode(m, 4)
    assert out.tolist() == [4, 0, 4, 4] and out.dtype == np.int16
    assert m.tolist() == [5, 0, 5, 4]                      # 元の配列は変えない


def test_build_derived_links_and_relabels(tmp_path):
    src, scenes = _fake_dataset(tmp_path / 'src')
    before = {str(p.relative_to(src)): _sha(p) for p in src.rglob('*') if p.is_file()}
    dst = tmp_path / 'out' / 'fake_ds_b2genre'
    v = P.build_derived(src, dst, relabel_logs=('20261010-152405',), to=4, manifest_path=tmp_path / 'out' / 'm.json',
                        created_at='2026-10-16T21:00:00')
    assert v['ok'] and v['splits_same_inode']
    # 元のファイルは 1 つも変わらない
    after = {str(p.relative_to(src)): _sha(p) for p in src.rglob('*') if p.is_file()}
    assert before == after
    # リンク(同じ inode)と新しいファイル(別の inode)
    same = lambda a, b: os.stat(a).st_ino == os.stat(b).st_ino      # noqa: E731
    assert same(src / 'splits.json', dst / 'splits.json')
    assert same(src / 'aligned' / (scenes[1] + '.npz'), dst / 'aligned' / (scenes[1] + '.npz'))
    assert same(src / 'samples/CAM_FRONT' / scenes[0] / '0.jpg', dst / 'samples/CAM_FRONT' / scenes[0] / '0.jpg')
    assert same(src / 'v1.0-fake_ds' / 'sample.json', dst / 'v1.0-fake_ds_b2genre' / 'sample.json')
    assert not same(src / 'aligned' / (scenes[0] + '.npz'), dst / 'aligned' / (scenes[0] + '.npz'))
    assert not (dst / 'v1.0-fake_ds').exists()
    # 付け替えたシーンの mode だけ変わる
    with np.load(dst / 'aligned' / (scenes[0] + '.npz')) as z, np.load(src / 'aligned' / (scenes[0] + '.npz')) as o:
        assert z['mode'].tolist() == [4, 4, 0, 4, 4, 4]
        assert list(z.files) == list(o.files)
        for k in o.files:
            if k != 'mode':
                assert np.array_equal(z[k], o[k])
    with np.load(dst / 'aligned' / (scenes[1] + '.npz')) as z:
        assert z['mode'].tolist() == [5, 5, 0, 5, 5, 5]
    sc = {s['name']: s for s in json.loads((dst / 'v1.0-fake_ds_b2genre' / 'scene.json').read_text(encoding='utf-8'))}
    assert sc[scenes[0]]['mode_id'] == 4 and sc[scenes[0]]['mode_name'] == '高速走行'
    assert sc[scenes[0]]['mode_segments'][0]['mode_id'] == 4
    assert sc[scenes[1]]['mode_id'] == 5
    card = json.loads((dst / 'dataset_card.json').read_text(encoding='utf-8'))
    assert card['name'] == 'fake_ds_b2genre' and card['version'] == 'v1.0-fake_ds_b2genre'
    assert card['mode_frames'] == {'0': 2, '4': 5, '5': 5}
    assert card['genre_fix']['scenes'] == [scenes[0]]
    m = json.loads((tmp_path / 'out' / 'm.json').read_text(encoding='utf-8'))
    assert m['linked_files'] == v['linked'] and len(m['new_files']) == 3
    # もう一度呼んでも作り直さない(確かめるだけ)
    ino = os.stat(dst / 'dataset_card.json').st_ino
    assert P.build_derived(src, dst, relabel_logs=('20261010-152405',), to=4, manifest_path=tmp_path / 'out' / 'm.json')['ok']
    assert os.stat(dst / 'dataset_card.json').st_ino == ino


def test_build_derived_recovers_from_partial(tmp_path):
    src, _ = _fake_dataset(tmp_path / 'src')
    dst = tmp_path / 'out' / 'fake_ds_b2genre'
    part = dst.with_name(dst.name + '.partial')
    (part / 'aligned').mkdir(parents=True)
    (part / 'aligned' / 'junk.npz').write_bytes(b'x')         # 電源断で残った半端なもの
    v = P.build_derived(src, dst, relabel_logs=('20261010-152405',), to=4, manifest_path=tmp_path / 'out' / 'm.json')
    assert v['ok'] and not part.exists() and not (dst / 'aligned' / 'junk.npz').exists()


def test_verify_derived_finds_problems(tmp_path):
    src, scenes = _fake_dataset(tmp_path / 'src')
    dst = tmp_path / 'out' / 'fake_ds_b2genre'
    P.build_derived(src, dst, relabel_logs=('20261010-152405',), to=4, manifest_path=tmp_path / 'out' / 'm.json')
    (dst / 'extra.txt').write_text('x')
    f = dst / 'samples/CAM_FRONT' / scenes[1] / '1.jpg'
    f.unlink()
    f.write_bytes(b'copy')                                    # リンクでなくコピー
    v = P.verify_derived(src, dst, relabel_logs=('20261010-152405',), to=4)
    assert not v['ok']
    kinds = {next(iter(p)) for p in v['problems']}
    assert {'extra', 'not_linked'} <= kinds
    with pytest.raises(SystemExit):                           # 確かめに通らないものは自動で消さずに止める
        P.build_derived(src, dst, relabel_logs=('20261010-152405',), to=4, manifest_path=tmp_path / 'out' / 'm.json')
    assert (dst / 'extra.txt').exists()


def test_attribute_drops_matches_rule():
    t = np.arange(0.0, 40.0, 0.1)
    passes = [{'ta_s': 5.0, 'tg_s': 7.0, 'entry_erpm': 6500.0, 'r1_collision': False},
              {'ta_s': 15.0, 'tg_s': 17.0, 'entry_erpm': 5000.0, 'r1_collision': False},
              {'ta_s': 25.0, 'tg_s': None, 'entry_erpm': 6300.0, 'r1_collision': True, 'r1_collision_t_s': [25.4]},
              {'ta_s': 33.0, 'tg_s': 36.0, 'entry_erpm': 4000.0, 'r1_collision': True, 'r1_collision_t_s': [33.2]}]
    r = M.drop_reason(t, passes)
    who = P.attribute_drops(t, r, passes)
    assert ((who >= 0) == (r > 0)).all()
    assert set(who[r > 0].tolist()) == {0, 2, 3}
    assert (who[(t > 3.05) & (t < 5.95)] == 0).all()
    assert (who[(t > 23.05) & (t < 28.45)] == 2).all()
    assert (who[(t > 31.05) & (t < 35.65)] == 3).all()
    assert [P.pass_label(p) for p in (passes[0], passes[2], passes[3])] == ['fast', 'fast+collision', 'collision']


LOG = """==== 2026-10-16 開始
衝突の前後として外したフレーム: 学習 2889 / 検証 923
フレームの表: drop で外した 学習 1639 / 検証 406、衝突の前後の除外から戻した 学習 0 / 検証 0
学習 5574 サンプル / 検証 1476 サンプル / 間引き 1/3 / ジャンル数 6 / device=cuda
ジャンル別の学習サンプル数: {'高速走行': 3295, '低速走行': 2279}
epoch   1  train 90µs  val 80µs  40s *  [高速走行 70µs  低速走行 85µs]
"""
EXP = {'train_samples': 5574, 'val_samples': 1476, 'dropped_printed': {'train': 1639, 'val': 406},
       'impact_masked_printed': {'train': 2889, 'val': 923}, 'genre_counts_train': {'5': 2279, '4': 3295}}
NAMES = {0: '未設定', 4: '高速走行', 5: '低速走行'}


def test_parse_and_compare_train_log():
    p = P.parse_train_log(LOG)
    assert p['impact'] == [2889, 923] and p['drop'] == [1639, 406, 0, 0]
    assert p['samples'] == [5574, 1476, 3, 6, 'cuda'] and p['genres'] == {'高速走行': 3295, '低速走行': 2279}
    assert P.compare_train_log(p, EXP, NAMES) == (True, [])
    bad = P.parse_train_log(LOG.replace('学習 5574 サンプル', '学習 5912 サンプル').replace("'高速走行': 3295", "'高速走行': 2228"))
    ok, diffs = P.compare_train_log(bad, EXP, NAMES)
    assert not ok and len(diffs) == 2
    # --max-steps の動作確認: サンプル数の行は見ない。drop の行が無いのは違い
    short = P.parse_train_log(LOG.replace('学習 5574 サンプル', '学習 16 サンプル'))
    assert P.compare_train_log(short, EXP, NAMES, full=False) == (True, [])
    ok, diffs = P.compare_train_log(P.parse_train_log(LOG.splitlines()[1]), EXP, NAMES, full=False)
    assert not ok and any('drop' in d for d in diffs)


CMD = ('cd ~/jetformer && /home/jetson/bin/jf-heavy --mem 3G --lock big --need-mb 3200 -- nice -n 10 env '
       'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 choom -n 1000 -- '
       '/home/jetson/miniconda/envs/jetracer/bin/python -I experiments/20261015_real_accuracy/train/nothp.py '
       '/home/jetson/miniconda/envs/jetracer/bin/python -u tools/train.py --policy jetformer_v0 '
       '--data data/datasets/clean_driving_20261011 --epochs 30 --lr 3e-4 --stride 3 --batch 64 --workers 4 '
       '--no-command-input --impact-g 2.5 --drop-mask data/derived/b2_curation_20261016/drop_b2.npz '
       '--mode-remap-npz data/derived/b2_curation_20261016/remap_b2.npz --seed 0 --out data/runs/b2cur_20261016-s0')


def test_smoke_cmd_from_protocol():
    proto = {'model': {'train_cmd': {'0': CMD}}}
    c = P.smoke_cmd(proto, out='/x/S1', epochs=0, workers=0)
    s = ' '.join(c)
    assert s.startswith('/home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 1500 -- nice -n 10 env CUDA_VISIBLE_DEVICES= ')
    assert 'choom -n 1000 --' in s and 'nothp.py' in s and 'MALLOC_ARENA_MAX=2' in s
    i = c.index('tools/train.py')
    args = c[i + 1:]
    assert args[args.index('--out') + 1] == '/x/S1' and args[args.index('--epochs') + 1] == '0'
    assert args[args.index('--workers') + 1] == '0' and args[args.index('--seed') + 1] == '0'
    for k in ('--drop-mask', '--mode-remap-npz', '--no-command-input', '--impact-g', '--stride', '--lr'):
        assert k in args
    c2 = P.smoke_cmd(proto, out='/x/S2', data='data/derived/x', drop_remap=False, epochs=0)
    a2 = c2[c2.index('tools/train.py') + 1:]
    assert '--mode-remap-npz' not in a2 and '--drop-mask' in a2 and a2[a2.index('--data') + 1] == 'data/derived/x'
    c3 = P.smoke_cmd(proto, out='/x/S3', epochs=2, max_steps=2, batch=8, workers=2, resume=True)
    a3 = c3[c3.index('tools/train.py') + 1:]
    assert a3[-1] == '--resume' and a3[a3.index('--max-steps') + 1] == '2' and a3[a3.index('--batch') + 1] == '8'
    # protocol の文字列は変えない(GPU の段はそのまま使う)
    assert proto['model']['train_cmd']['0'] == CMD


@pytest.mark.skipif(not P.DERIVED_ROOT.exists(), reason='ジャンルを直したデータセットをまだ作っていない')
def test_real_derived_equals_tables_val():
    """作ったデータセット + 外す表 = 元 + 外す表 + 付け替えの表(検証の分け方。学習も checks で確かめる)"""
    from jetformer import config
    from jetformer.train_data import JetFormerDataset, load_frame_table
    assert P.verify_derived(config.DATASET_DIR / P.SRC_NAME, P.DERIVED_ROOT)['ok']
    drop = load_frame_table(P.OUT_DIR / 'drop_b2.npz', 'drop')
    remap = load_frame_table(P.OUT_DIR / 'remap_b2.npz', 'mode')
    a = JetFormerDataset(str(config.DATASET_DIR / P.SRC_NAME), 'val', stride=3, use_command=False, mask_impacts=True,
                         impact_g=2.5, drop=drop, mode_remap=remap)
    b = JetFormerDataset(str(P.DERIVED_ROOT), 'val', stride=3, use_command=False, mask_impacts=True, impact_g=2.5, drop=drop)
    assert len(a) == len(b) == 1476
    assert [(k, i, a.mode_of(k, i)) for k, i in a.index] == [(k, i, b.mode_of(k, i)) for k, i in b.index]
    assert a.mode_counts() == b.mode_counts() == {5: 650, 4: 826}


# ---------------------------------------------------------------- train_b2.py(GPU の段の道具。ここでは GPU を使わない部分だけ)

import train_b2 as T  # noqa: E402


def test_train_targets_are_protocol_strings():
    proto = P.load_protocol()
    ts = T.targets(proto)
    assert set(ts) == {'0', '1', 'drop_only', 'relabel_only'}
    for s in ('0', '1'):
        assert ts[s]['cmd'] == proto['model']['train_cmd'][s] and ts[s]['resume'] == proto['model']['resume_cmd'][s]
        assert ts[s]['run'] == 'b2cur_20261016-s%s' % s and ts[s]['check']
    assert ts['drop_only']['resume'].endswith(' --resume') and not ts['drop_only']['check']


def test_with_mem_changes_only_cap():
    c = T.with_mem(CMD, '3872M')
    assert '--mem 3872M --lock big' in c and c.replace('3872M', '3G') == CMD
    with pytest.raises(ValueError):
        T.with_mem('python tools/train.py', '3G')


def test_stage(tmp_path):
    d = tmp_path / 'run'
    assert T.stage(d) == 'fresh'
    d.mkdir()
    (d / 'last.pth').write_bytes(b'x')
    assert T.stage(d) == 'resume'
    (d / 'result.json').write_text(json.dumps({'state': 'stopped', 'epochs_done': 12}))
    assert T.stage(d) == 'resume'
    (d / 'result.json').write_text(json.dumps({'state': 'done', 'epochs_done': 30}))
    assert T.stage(d) == 'done'


def test_run_json_and_check_result():
    proto = P.load_protocol()
    t = T.targets(proto)['0']
    rj = T.run_json(proto, t, '2026-10-17 01:00:00')
    assert rj['seed'] == 0 and rj['name'] == 'b2cur_20261016-s0' and rj['impact_g'] == 2.5 and rj['command_input'] is False
    assert set(rj['tables_sha256']) == {'drop_mask', 'mode_remap_npz'}
    assert rj['tables_sha256']['drop_mask'] == proto['files']['drop_b2']['sha256']
    assert rj['tables_sha256']['mode_remap_npz'] == proto['files']['remap_b2']['sha256']
    assert '机上の G3 に合格するまで走らせない' in rj['note']
    sha1 = {k: hashlib.sha1((P.REPO / v).read_bytes()).hexdigest()[:12] for k, v in T.TABLES.items()}
    good = {'state': 'done', 'epochs_done': 30, 'train_samples': 5574, 'val_samples': 1476,
            'impact_masked_frames': {'train': 2889, 'val': 923}, 'train_mode_counts': {'4': 3295, '5': 2279}, 'seed': 0,
            'impact_g': 2.5, 'command_input': False, 'policy': 'jetformer_v0', 'stride': 3,
            'rl_options': {'dropped_frames': {'train': 1639, 'val': 406}, 'sha1': sha1}}
    assert T.check_result(good, proto, t) == []
    bad = dict(good, train_samples=5912, rl_options={'dropped_frames': {'train': 0, 'val': 0}, 'sha1': {}})
    assert len(T.check_result(bad, proto, t)) == 3
    assert T.targets(proto)['drop_only']['check'] is False
    assert T.check_result(dict(good, train_samples=1), proto, T.targets(proto)['drop_only']) == []


def test_watch_stops_on_wrong_first_lines(tmp_path):
    """学習の最初の数行が違えば止める(GPU を使わない作り物のコマンドで)"""
    import time as _t
    proto = P.load_protocol()
    wrong = LOG.replace('学習 5574 サンプル', '学習 5912 サンプル')
    src = tmp_path / 'wrong.txt'
    src.write_text(wrong, encoding='utf-8')
    t0 = _t.time()
    rc, diffs, stopped = T._watch('cat %s; sleep 120' % src, tmp_path / 'train.log', True, proto)
    assert stopped and diffs and _t.time() - t0 < 60
    good = tmp_path / 'good.txt'
    good.write_text(LOG, encoding='utf-8')
    rc, diffs, stopped = T._watch('cat %s; sleep 1' % good, tmp_path / 'train2.log', True, proto)
    assert rc == 0 and diffs == [] and not stopped
    assert '==== ' in (tmp_path / 'train2.log').read_text(encoding='utf-8')
