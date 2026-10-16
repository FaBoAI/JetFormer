#!/usr/bin/env python3
"""realacc_driver.py の小さな関数のテスト(GPU なし・学習しない)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python -m pytest -q experiments/20261015_real_accuracy/train/test_realacc_driver.py
"""
import datetime
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realacc_driver as D  # noqa: E402


def test_parse_protocol_command():
    t = ('/home/jetson/bin/jf-heavy --mem 3800M --lock big --need-mb 4000 -- nice -n 10 python -u tools/train.py --policy msfusion '
         '--out /x/realacc_20261015-msf-s2 --seed 2')
    c = D.parse_protocol_command(t)
    assert c['mem'] == '3800M' and c['need_mb'] == 4000 and c['lock'] == 'big'
    assert c['args'] == ['tools/train.py', '--policy', 'msfusion', '--out', '/x/realacc_20261015-msf-s2', '--seed', '2']
    with pytest.raises(ValueError):
        D.parse_protocol_command('python -u tools/train.py')
    with pytest.raises(ValueError):
        D.parse_protocol_command('/home/jetson/bin/jf-heavy --mem 3G -- python -u tools/train.py')


def test_mem_and_raise():
    assert D.mem_to_mb('3G') == 3072 and D.mem_to_mb('3800M') == 3800
    assert D.raise_cap('3G') == '3872M' and D.raise_cap('3800M') == '4600M'


def test_with_resume():
    a = ['tools/train.py', '--seed', '0']
    assert D.with_resume(a, True) == a + ['--resume']
    assert D.with_resume(a + ['--resume'], False) == a
    assert D.with_resume(a + ['--resume'], True) == a + ['--resume']


def test_wrap_gpu_keeps_args():
    full = D.wrap_gpu(['tools/train.py', '--seed', '2'], '3G', 3200)
    assert full[:8] == [D.JF_HEAVY, '--mem', '3G', '--lock', 'big', '--need-mb', '3200', '--']
    assert full[-3:] == ['tools/train.py', '--seed', '2']
    assert 'choom' in full and str(D.NOTHP) in full


def test_fits_before():
    t = datetime.datetime(2026, 10, 16, 5, 30, tzinfo=D.TZ_JST)
    assert D.fits_before(t, 30, D.GPU_HARD_STOP)
    assert not D.fits_before(t, 31, D.GPU_HARD_STOP)


def test_build_steps_matches_protocol_order():
    proto, _ = D.load_protocol()
    st = D.build_steps(proto)
    ids = [s['id'] for s in st]
    want = ['train:realacc_20261015-msf-s2', 'train:realacc_20261015-v0-s2', 'train:realacc_20261015-msf-s3',
            'train:realacc_20261015-v0-s3', 'crosscheck', 'predict:ENS', 'train:realacc_20261015-sizeH-s0',
            'train:realacc_20261015-sizeH-s1', 'train:realacc_20261015-sizeQ-s0', 'train:realacc_20261015-sizeQ-s1',
            'rot_smoke', 'train:realacc_20261015-rot-s0', 'train:realacc_20261015-rot-s1', 'predict:SIX', 'gate',
            'train:realacc_20261015-sizeT-s0', 'train:realacc_20261015-sizeT-s1', 'predict:SIZE_T', 'gainl_smoke',
            'train:realacc_20261015-gainl-s0', 'train:realacc_20261015-gainl-s1', 'predict:GAINL']
    assert ids == want
    by = {s['id']: s for s in st}
    # protocol のコマンドの引数をそのまま使う(包みは別)
    c = by['train:realacc_20261015-sizeQ-s1']['cmd']
    assert c['args'][-2:] == ['--drop-mask', 'experiments/20261015_real_accuracy/size/size_Q.npz']
    assert by['train:realacc_20261015-msf-s2']['cmd']['mem'] == '3800M'
    assert by['train:realacc_20261015-v0-s2']['cmd']['mem'] == '3G'
    assert by['rot_smoke']['cmd']['args'][0] == 'experiments/20261015_real_accuracy/train_rot.py'
    assert '--max-steps' in by['rot_smoke']['cmd']['args']
    assert [s['optional'] for s in st].count(True) == 7
    # ROT は腕ごと: rot_smoke の時点の残り = 動作確認 + 2 本
    i = ids.index('rot_smoke')
    assert D.group_remaining_minutes(st, i, set()) == D.EST_MIN['rot_smoke'] + 2 * D.EST_MIN['jetformer_v0']
    assert D.group_remaining_minutes(st, i + 1, {'rot_smoke'}) == 2 * D.EST_MIN['jetformer_v0']


def test_check_run_result_rot(tmp_path):
    step = {'kind': 'ROT', 'cmd': {'args': ['x.py', '--seed', '1']}}
    run = tmp_path
    (run / 'result.json').write_text(json.dumps({'train_samples': 5912, 'val_samples': 1559}))
    rows = [{'epoch': e, 'phase': (e - 1 + 1) % 3, 'samples': D.ROT_PHASE_SIZES[(e - 1 + 1) % 3]} for e in range(1, 31)]
    (run / 'rot_log.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    ok, why = D.check_run_result(step, run)
    assert ok, why
    # 種 0 なら最後のエポックの位相は 2(5,879 行)
    step0 = {'kind': 'ROT', 'cmd': {'args': ['x.py', '--seed', '0']}}
    ok, why = D.check_run_result(step0, run)
    assert not ok
    # 再開で同じエポックが 2 回あれば後のほうを見る
    rows[4] = dict(rows[4], phase=9)
    (run / 'rot_log.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n' +
                                       json.dumps({'epoch': 5, 'phase': (5 - 1 + 1) % 3, 'samples': D.ROT_PHASE_SIZES[2]}) + '\n')
    ok, why = D.check_run_result(step, run)
    assert ok, why


def test_check_run_result_size(tmp_path):
    (tmp_path / 'result.json').write_text(json.dumps({'train_samples': 3046, 'val_samples': 1559}))
    assert D.check_run_result({'kind': 'SIZE_H', 'cmd': {'args': []}}, tmp_path)[0]
    assert not D.check_run_result({'kind': 'SIZE_Q', 'cmd': {'args': []}}, tmp_path)[0]


def test_posthoc_steps_equal_gradient_steps():
    proto, _ = D.load_protocol()
    st = D.build_posthoc_steps(proto)
    ids = [s['id'] for s in st]
    assert ids == ['train:realacc_20261015-sizeHeq-s0', 'train:realacc_20261015-sizeHeq-s1', 'predict:SIZEEQ_H',
                   'train:realacc_20261015-sizeQeq-s0', 'train:realacc_20261015-sizeQeq-s1', 'predict:SIZEEQ_Q']
    main = {s['id']: s for s in D.build_steps(proto)}
    for s in st:
        if s['type'] != 'train':
            continue
        sub = 'H' if 'sizeH' in s['name'] else 'Q'
        seed = s['name'][-1]
        base = main['train:realacc_20261015-size%s-s%s' % (sub, seed)]['cmd']['args']
        a = s['cmd']['args']
        ep = int(D.arg_value(a, '--epochs'))
        n = {'H': 3046, 'Q': 1682}[sub]
        assert abs(ep * (n // 64) - 30 * (5912 // 64)) <= n // 64          # ステップ数が全部とほぼ同じ
        # エポックと --out だけが違う
        diff = [i for i, (x, y) in enumerate(zip(a, base)) if x != y]
        assert [a[i - 1] for i in diff] == ['--epochs', '--out'] and len(a) == len(base)
        assert s['optional'] and s['kind'] == 'SIZE_' + sub


def test_run_status_ja():
    assert D.run_status_ja({'check_ok': True}) == '済み'
    assert D.run_status_ja({'check_ok': True, 'cut': 'x'}) == '済み'           # 済みが優先
    assert D.run_status_ja({'cut': '電源断で未完'}) == '削った(電源断で未完)'
    assert D.run_status_ja({'cut': True}) == '削った'
    assert D.run_status_ja({}) == '学習中・未完'
