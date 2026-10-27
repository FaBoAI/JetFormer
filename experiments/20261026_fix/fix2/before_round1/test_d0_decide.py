"""d0_decide.py の試験(作り物の入力。torch・GPU を使わない。本物の decisions_v3.json は書かない)。

    cd /home/jetson/jetformer/experiments/20261026_fix/fix2 && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
      /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider \
      --basetemp=/home/jetson/jetformer/experiments/20261026_fix/fix2/tmp_d0 --rootdir=. test_d0_decide.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d0_decide as D  # noqa: E402

QUEUE = HERE.parent / 'gpu_queue_v3.py'
REAL_DECISIONS = Path('/home/jetson/jetformer_reports/fix_20261026/decisions_v3.json')


def probe(st23, stand=0.02, low10=0.0, st10=0.0):
    return {'st10_thr_vs_b_us': st10, 'st23_thr_vs_b_us': st23, 'stand_brake_side_frac': stand, 'stand_go_frac': 0.2,
            'low10_brake_frac': low10, 'n': dict(D.FRAMES)}


def make(tmp, vals, bf2=None, b_st23=0.0, skip=(), state='done', older_probe=(), abl=None, n_train=None):
    """作り物の段 0: vals = {腕: probe}。→ main() に渡す引数"""
    runs, pdir = tmp / 'runs', tmp / 'probe'
    runs.mkdir(exist_ok=True)
    pdir.mkdir(exist_ok=True)
    for arm, spec in D.ARMS.items():
        if arm in skip:
            continue
        name = D.run_name(arm)
        rd = runs / name
        rd.mkdir(exist_ok=True)
        res = {'state': state, 'epochs_planned': 15, 'epochs_done': 15, 'seed': 0,
               'train_samples': (n_train or {}).get(arm, spec['n_train']), 'val_samples': 1559, 'best_epoch': 14,
               'best_val_us': 50.0, 'rl_options': {'drop_mask': 'experiments/20261026_fix/diag_data/out/ablation/' + spec['drop']},
               'bench_options': {'label_override_npz': ('experiments/20261026_fix/diag_data/out/ablation/' + spec['override'])
                                 if spec['override'] else None,
                                 'label_overridden_rows': {'all': spec['override_used'], 'used': spec['override_used']}}}
        (rd / 'result.json').write_text(json.dumps(res))
        (rd / 'best.pth').write_bytes(b'x' * 10)
        if arm in vals:
            pf = pdir / ('%s_best.json' % name)
            pf.write_text(json.dumps({str(rd): vals[arm]}))
            if arm in older_probe:
                t = time.time() - 100
                os.utime(pf, (t, t))
    pc = tmp / 'probe_check.json'
    pc.write_text(json.dumps({D.B_RUN: probe(b_st23, 0.0161, 0.0052)}))
    bf = tmp / 'bf2.json'
    bf.write_text(json.dumps(bf2 or {'made_at': '2026-10-26 11:53:03', 'bound_g090_us': {'current': -18.5, 'A': -0.049, 'B': 2.42},
                                     'need_us': 28.57, 'passed': {'A': False, 'B': False}, 'train_bfast2': False,
                                     'candidate': None}))
    args = ['--runs-dir', str(runs), '--probe-dir', str(pdir), '--probe-check', str(pc), '--bf2', str(bf),
            '--abl-probe', str(abl or (tmp / 'no_abl.json')), '--out', str(tmp / 'decisions_v3.json')]
    return args


def run(args, capsys):
    rc = D.main(args)
    out, err = capsys.readouterr()
    return rc, (json.loads(out) if out.strip() else None), err


FOUR = {'same': probe(60.0, 0.40, 0.08), 'noTK': probe(10.0, 0.05, 0.03), 'noTK_noLB': probe(9.0, 0.04, 0.0),
        'tkB': probe(12.0, 0.06, 0.05)}


def with_(**kw):
    v = {k: dict(x) for k, x in FOUR.items()}
    for arm, p in kw.items():
        v[arm] = p
    return v


@pytest.mark.parametrize('noTK,tkB,want,p1,pD', [
    (probe(10.0, 0.05), probe(12.0, 0.06), ['v3d', 'v3n', 'v3b'], True, True),       # 両方 → v3d → v3n → v3b
    (probe(10.0, 0.05), probe(30.0, 0.06), ['v3n', 'v3d', 'v3b'], True, False),      # 案 1 だけ
    (probe(30.0, 0.05), probe(12.0, 0.06), ['v3d', 'v3n', 'v3b'], False, True),      # 案 D だけ
    (probe(30.0, 0.05), probe(30.0, 0.06), ['v3b', 'v3d', 'v3n'], False, False),     # どちらも
    (probe(10.0, 0.30), probe(12.0, 0.25), ['v3b', 'v3d', 'v3n'], False, False),     # 後ろ側が 0.10 を超える
])
def test_order_by_rule(tmp_path, capsys, noTK, tkB, want, p1, pD):
    rc, out, err = run(make(tmp_path, with_(noTK=noTK, tkB=tkB)), capsys)
    assert rc == 0, err
    assert out['D0']['order'] == want
    c = out['D0']['confirmed']
    assert c['valid'] is True and c['案 1(noTK → v3n)'] is p1 and c['案 D(tkB → v3d)'] is pD
    assert out['BF2-0']['train'] is False and out['schema'] == 'jf-train-v3-decisions/v1'


def test_not_a_check_when_same_below_20(tmp_path, capsys):
    rc, out, _ = run(make(tmp_path, with_(same=probe(19.99, 0.4), noTK=probe(0.0, 0.0), tkB=probe(0.0, 0.0))), capsys)
    assert rc == 0
    assert out['D0']['confirmed']['valid'] is False
    assert out['D0']['order'] == ['v3b', 'v3d', 'v3n']
    assert '確かめにならない' in out['D0']['why']


def test_boundaries(tmp_path, capsys):
    # same 30 → 1/3 = 10。noTK 10.0 は ≤ で確かめた、tkB 10.01 は外れ。後ろ側 0.10 ちょうどは ≤
    rc, out, _ = run(make(tmp_path, with_(same=probe(30.0), noTK=probe(10.0, 0.10), tkB=probe(10.01, 0.0))), capsys)
    assert rc == 0 and out['D0']['order'] == ['v3n', 'v3d', 'v3b']
    rc, out, _ = run(make(tmp_path, with_(same=probe(20.0), noTK=probe(6.66, 0.1001), tkB=probe(6.66, 0.0))), capsys)
    assert rc == 0 and out['D0']['order'] == ['v3d', 'v3n', 'v3b']          # same ちょうど 20 は「確かめになる」
    assert out['D0']['confirmed']['案 1(noTK → v3n)'] is False


def test_b_reference_is_subtracted(tmp_path, capsys):
    rc, out, err = run(make(tmp_path, FOUR, b_st23=0.005), capsys)        # 0.01 以内なら引く
    assert rc == 0, err
    assert abs(out['D0']['values']['st23_minus_b_us']['same'] - 59.995) < 1e-9
    rc, _, err = run(make(tmp_path, FOUR, b_st23=1.0), capsys)
    assert rc == 2 and 'B の st10・st23 が 0' in err


@pytest.mark.parametrize('low10,want', [(0.01, True), (0.0101, False)])
def test_plan2_does_not_change_order(tmp_path, capsys, low10, want):
    rc, out, _ = run(make(tmp_path, with_(noTK_noLB=probe(9.0, 0.04, low10))), capsys)
    assert rc == 0
    assert out['D0']['confirmed']['案 2(noTK_noLB。順には効かない)'] is want
    assert out['D0']['order'] == ['v3d', 'v3n', 'v3b']


def test_stops_when_an_arm_is_missing(tmp_path, capsys):
    rc, out, err = run(make(tmp_path, FOUR, skip=('tkB',)), capsys)
    assert rc == 2 and out is None and 'abl-tkB-e15-s0 の result.json がありません' in err


def test_stops_when_not_done(tmp_path, capsys):
    rc, _, err = run(make(tmp_path, FOUR, state='running'), capsys)
    assert rc == 2 and 'done でない' in err


def test_stops_on_wrong_sample_count(tmp_path, capsys):
    rc, _, err = run(make(tmp_path, FOUR, n_train={'noTK': 6143}), capsys)
    assert rc == 2 and '学習サンプルが 6143' in err


def test_stops_on_zero_byte_best(tmp_path, capsys):
    args = make(tmp_path, FOUR)
    (tmp_path / 'runs' / 'abl-same-e15-s0' / 'best.pth').write_bytes(b'')
    rc, _, err = run(args, capsys)
    assert rc == 2 and '0 バイト' in err


def test_stops_on_broken_result_json(tmp_path, capsys):
    args = make(tmp_path, FOUR)
    (tmp_path / 'runs' / 'abl-noTK-e15-s0' / 'result.json').write_text('{"state": "do')
    rc, _, err = run(args, capsys)
    assert rc == 2 and '壊れている' in err


def test_stops_without_probe(tmp_path, capsys):
    v = dict(FOUR)
    del v['noTK_noLB']
    rc, _, err = run(make(tmp_path, v), capsys)
    assert rc == 2 and 'abl-noTK_noLB-e15-s0: best.pth の物差しがありません' in err


def test_stops_on_stale_probe(tmp_path, capsys):
    rc, _, err = run(make(tmp_path, FOUR, older_probe=('same',)), capsys)
    assert rc == 2 and 'best.pth より古い' in err


def test_stops_on_bad_probe_values(tmp_path, capsys):
    bad_n = probe(10.0)
    bad_n['n'] = dict(D.FRAMES, st23=580)
    rc, _, err = run(make(tmp_path, with_(noTK=bad_n)), capsys)
    assert rc == 2 and 'フレームの数' in err
    args = make(tmp_path, FOUR)
    pf = tmp_path / 'probe' / 'abl-tkB-e15-s0_best.json'
    pf.write_text(pf.read_text().replace('12.0', 'NaN'))
    rc, _, err = run(args, capsys)
    assert rc == 2 and '数でありません' in err


def test_two_probe_sources_must_agree(tmp_path, capsys):
    abl = tmp_path / 'abl_probe_best.json'
    args = make(tmp_path, FOUR, abl=abl)                                 # best.pth を書いたあとに物差しを書く
    abl.write_text(json.dumps({'experiments/x/runs/%s' % D.run_name(a): p for a, p in FOUR.items()}))
    rc, out, err = run(args, capsys)
    assert rc == 0, err
    assert set(out['D0']['runs']['same']['probe_sources']) == {'queue', 'ablation_short'}
    other = dict(FOUR, tkB=probe(12.5, 0.06, 0.05))
    abl.write_text(json.dumps({'experiments/x/runs/%s' % D.run_name(a): p for a, p in other.items()}))
    rc, _, err = run(args, capsys)
    assert rc == 2 and '出どころで st23_thr_vs_b_us が違います' in err
    t = time.time() - 100                                                # best.pth より古い物差しは止まる
    os.utime(abl, (t, t))
    rc, _, err = run(args, capsys)
    assert rc == 2 and 'ablation_short.sh の物差し' in err and 'best.pth より古い' in err


def test_write_once_never_overwrites(tmp_path, capsys):
    args = make(tmp_path, FOUR)
    dest = tmp_path / 'decisions_v3.json'
    rc, out, err = run(args + ['--write'], capsys)
    assert rc == 0 and dest.exists() and '書きました' in err
    first = dest.read_bytes()
    assert json.loads(first)['D0']['order'] == ['v3d', 'v3n', 'v3b']
    args2 = make(tmp_path, with_(noTK=probe(30.0), tkB=probe(30.0)))
    rc, out, err = run(args2 + ['--write'], capsys)
    assert rc == 3 and '上書きしません' in err and '違う' in err
    assert dest.read_bytes() == first
    rc, out, err = run(args2, capsys)                                   # 書かないなら出すだけ(rc 0)
    assert rc == 0 and out['D0']['order'] == ['v3b', 'v3d', 'v3n'] and dest.read_bytes() == first


def test_same_noTK_only_fallback(tmp_path, capsys):
    v = {k: FOUR[k] for k in ('same', 'noTK')}
    rc, out, err = run(make(tmp_path, v, skip=('tkB',)) + ['--same-noTK-only', 'tkB が 2 回止まった'], capsys)
    assert rc == 0, err
    assert out['D0']['order'] == ['v3n', 'v3d', 'v3b']
    assert out['D0']['confirmed']['案 D(tkB → v3d)'] is None and out['D0']['partial']['why'] == 'tkB が 2 回止まった'
    rc, _, err = run(make(tmp_path, FOUR) + ['--same-noTK-only', 'x'], capsys)
    assert rc == 2 and 'そろっています' in err


def test_bf2_recount_must_match(tmp_path, capsys):
    good = {'bound_g090_us': {'current': -18.5, 'A': 30.0, 'B': 2.42}, 'passed': {'A': True, 'B': False},
            'train_bfast2': True, 'candidate': 'A'}
    rc, out, err = run(make(tmp_path, FOUR, bf2=good), capsys)
    assert rc == 0, err
    assert out['BF2-0']['train'] is True and out['BF2-0']['values']['passed'] == {'A': True, 'B': False}
    lie = dict(good, train_bfast2=False, passed={'A': False, 'B': False})
    rc, _, err = run(make(tmp_path, FOUR, bf2=lie), capsys)
    assert rc == 2 and '数え直した答え' in err
    edge = {'bound_g090_us': {'A': 28.57, 'B': 28.58}, 'passed': {'A': False, 'B': True}, 'train_bfast2': True,
            'candidate': 'B'}                                           # 28.571… が境目
    rc, out, err = run(make(tmp_path, FOUR, bf2=edge), capsys)
    assert rc == 0, err


def test_real_bf2_0_answer(tmp_path, capsys):
    """本物の data/out/bf2_0.json(読むだけ)。台本の sha256 も確かめる"""
    args = make(tmp_path, FOUR)
    i = args.index('--bf2')
    args[i + 1] = str(D.BF2_0)
    rc, out, err = run(args, capsys)
    assert rc == 0, err
    b = out['BF2-0']
    assert b['train'] is False and b['values']['bound_g090_us'] == {'A': -0.049, 'B': 2.42}
    assert b['scripts']['diag_bfast.py']['sha256'].startswith('c07c9450')


def test_queue_reads_it(tmp_path, capsys):
    """gpu_queue_v3.py(読むだけの templates・resolve)が、この形の D0.order・BF2-0.train を読む"""
    args = make(tmp_path, with_(noTK=probe(10.0, 0.05), tkB=probe(30.0, 0.06)))
    rc, out, _ = run(args + ['--write'], capsys)
    assert rc == 0 and out['D0']['order'] == ['v3n', 'v3d', 'v3b']
    env = {k: v for k, v in os.environ.items() if not k.startswith('GPU_QUEUE_')}
    env.update({'GPU_QUEUE_DECISIONS': str(tmp_path / 'decisions_v3.json'), 'GPU_QUEUE_LOCK2': str(tmp_path / 'none.json'),
                'GPU_QUEUE_PROBE_DIR': str(tmp_path / 'qprobe')})
    r = subprocess.run([sys.executable, '-I', str(QUEUE), 'templates'], capture_output=True, text=True, env=env)
    names = r.stdout.split()
    st1 = [n for n in names if n.startswith('v3_')]
    assert st1[:3] == ['v3_20261026-v3n-aug2-e60-s0', 'v3_20261026-v3d-aug2-e60-s0', 'v3_20261026-v3b-aug2-e60-s0']
    assert st1[3:6] == ['v3_20261026-v3n-aug2-e60-s1', 'v3_20261026-v3d-aug2-e60-s1', 'v3_20261026-v3b-aug2-e60-s1']
    r = subprocess.run([sys.executable, '-I', str(QUEUE), 'resolve', 'bfast2_20261026-e15-s0'], capture_output=True, text=True,
                       env=env)
    assert r.stdout.startswith('SKIP\tBF2-0 が「学習しない」(decisions_v3.json)')
    r = subprocess.run([sys.executable, '-I', str(QUEUE), 'resolve', 'v3_20261026-v3n-aug2-e60-s0'], capture_output=True,
                       text=True, env=env)
    assert r.stdout.startswith('STOP\tprotocol_v3_lock2.json')            # lock2 が無いあいだは段 1 で止まる


def test_real_decisions_untouched():
    """試験は本物の decisions_v3.json を作らない"""
    assert not REAL_DECISIONS.exists() or REAL_DECISIONS.stat().st_mtime < time.time() - 1
