"""係 e5(2026-10-26): e5_open_seal.py の試験。本物の封の記録は読まない・開けない(作り物の封と、封でない記録だけ)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -m pytest \
        -q -p no:cacheprovider --basetemp=experiments/20261026_fix/fix3/tmp_e5 experiments/20261026_fix/fix3/test_e5_open_seal.py

- 本物の seal_opened.json(train_v2_20261025/)が試験の前にも後にも無いこと、を毎回確かめる。
- 封の記録を読む関数(ラベル・生の記録・inventory・aligned・T の窓)は、封の 14 本の名前が来たら試験を落とすように包む。
- 封でない記録(10/10・10/24 2 回目)を読む試験は 2 つ(作りの照らし。数秒)。GPU・torch は使わない。
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))


def _load():
    spec = importlib.util.spec_from_file_location('e5_open_seal_t', str(HERE / 'e5_open_seal.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _load()
SEAL = set(E.seal_logs())
REAL = E.REAL_SEAL_OPENED
V1N, V2N = 'v3_20261026-v3d-aug2-e60-s0', 'v3_20261026-v3d-aug2-e60-s1'


@pytest.fixture(autouse=True)
def guard(monkeypatch, tmp_path):
    """本物の封を読まない・本物の場所に書かない"""
    assert not REAL.exists(), '試験の前に本物の seal_opened.json がある(試験を止める)'
    monkeypatch.setattr(E, 'OUT', tmp_path / 'real_out')
    monkeypatch.setattr(E, 'SEAL_OPENED', tmp_path / 'real_out' / 'seal_opened.json')
    monkeypatch.setattr(E, 'DRY_ROOT', tmp_path / 'dry_root')
    monkeypatch.setattr(E, 'SELFCHECK_JSON', tmp_path / 'selfcheck.json')

    def no_seal(fn, pick):
        def wrap(*a, **k):
            for lg in pick(*a, **k):
                assert str(lg)[:15] not in SEAL, '封の記録を読もうとした: %s' % lg
            return fn(*a, **k)
        return wrap
    monkeypatch.setattr(E, 'extract_inventory', no_seal(E.extract_inventory, lambda logs, *a, **k: list(logs)))
    monkeypatch.setattr(E, 'ai_log_labels', no_seal(E.ai_log_labels, lambda lg, *a, **k: [lg]))
    monkeypatch.setattr(E, 'scene_aligned', no_seal(E.scene_aligned, lambda s, *a, **k: [s]))
    monkeypatch.setattr(E, 't_windows', no_seal(E.t_windows, lambda lg, *a, **k: [lg]))
    import build_ai_dataset as bad
    import make_intervention_set as mis
    monkeypatch.setattr(bad, 'load_labels', no_seal(bad.load_labels, lambda lg, *a, **k: [lg]))
    monkeypatch.setattr(mis, 'load_raw', no_seal(mis.load_raw, lambda lg, *a, **k: [lg]))
    yield
    assert not REAL.exists(), '試験のあとに本物の seal_opened.json がある'


def ns(**k):
    d = dict(user_approved_open_without_candidates=None, dry_logs=[], dry_logs1=[], dry_logs3=[], scenario='pass')
    d.update(k)
    return argparse.Namespace(**d)


def dec3(v1=V1N, v2=V2N, final=True, bf2_train=False, e4=None):
    d = {'E3': {'V1': v1, 'V2': v2, 'final': final}, 'BF2-0': {'train': bf2_train}}
    if e4 is not None:
        d['E4'] = e4
    return d


DEC2 = {'D5a': {'C1': None, 'C2': None, 'status': '決まった(無し)'}, 'BF': {'BF1': None, 'BF2': None}, 'D4': {'reserve': False}}


# ---------------------------------------------------------------- 候補と開けるか

def test_read_candidates_e3_missing_is_not_final():
    c = E.read_candidates({'BF2-0': {'train': False}}, DEC2)
    assert not c['final'] and any('E3' in p for p in c['problems'])
    pl = E.plan_of(c)
    assert not pl['open'] and pl['decision'].startswith('開けない')


def test_read_candidates_final_and_bf21_none():
    c = E.read_candidates(dec3(), DEC2)
    assert c['final'] and c['v3'] == {'BF2-1': None, 'V1': V1N, 'V2': V2N}
    pl = E.plan_of(c)
    assert pl['open'] and pl['predict']['v3']['models'] == E.FIVE_RUNS + [V1N, V2N]
    assert pl['predict']['v2']['models'] == E.FIVE_RUNS and pl['predict']['all'] == E.FIVE_RUNS + [V1N, V2N]


def test_read_candidates_bad_names_and_same():
    c = E.read_candidates(dec3(v1='v2_20261025-v2h-aug2-e60-s0', v2=None), DEC2)
    assert not c['final'] and any('V1' in p for p in c['problems'])
    c = E.read_candidates(dec3(v1=V1N, v2=V1N), DEC2)
    assert not c['final'] and any('同じ' in p for p in c['problems'])
    c = E.read_candidates(dec3(bf2_train=True), DEC2)                      # B-FAST2 を学習したのに E4 が無い
    assert not c['final'] and any('E4' in p for p in c['problems'])
    c = E.read_candidates(dec3(bf2_train=True, e4={'BF2-1': 'bfast2_20261026-e15-s1', 'final': True}), DEC2)
    assert c['final'] and c['v3']['BF2-1'] == 'bfast2_20261026-e15-s1'


def test_zero_candidates_b_stays_and_seal_not_opened(tmp_path):
    c = E.read_candidates(dec3(v1=None, v2=None), DEC2)
    pl = E.plan_of(c)
    assert c['final'] and pl['n_candidates'] == 0 and not pl['open'] and 'B のまま' in pl['decision'] and 'ユーザー' in pl['decision']
    P = E.Paths(tmp_path / 'o', tmp_path / 'o' / 'seal_opened.json', False)
    called = []
    rc = E.cmd_open(ns(), P, c, checks=lambda: called.append(1) or [], date_fn=lambda: E.OPEN_DATE)
    assert rc == 0 and not P.seal_opened.exists() and not called


def test_open_refused_before_nov1_and_after_run_day(tmp_path):
    c = E.read_candidates(dec3(), DEC2)
    P = E.Paths(tmp_path / 'o', tmp_path / 'o' / 'seal_opened.json', False)
    import datetime
    for d in (datetime.date(2026, 10, 31), datetime.date(2026, 11, 4)):
        with pytest.raises(SystemExit):
            E.cmd_open(ns(), P, c, checks=lambda: pytest.fail('確かめの前に止まるはず'), date_fn=lambda d=d: d)
        assert not P.seal_opened.exists()


def test_open_refused_when_e3_not_final_or_fix_check_fails(tmp_path, monkeypatch):
    P = E.Paths(tmp_path / 'o', tmp_path / 'o' / 'seal_opened.json', False)
    with pytest.raises(SystemExit):
        E.cmd_open(ns(), P, E.read_candidates(dec3(final=False), DEC2), checks=lambda: [], date_fn=lambda: E.OPEN_DATE)
    assert not P.seal_opened.exists()
    monkeypatch.setattr(E, 'weights_of', lambda names: {n: {'best_sha256': 'x'} for n in names})
    with pytest.raises(SystemExit) as ei:
        E.cmd_open(ns(), P, E.read_candidates(dec3(), DEC2), checks=lambda: [{'what': 'protocol_v3.sha256', 'rc': 1, 'out': 'FAILED'}],
                   date_fn=lambda: E.OPEN_DATE)
    assert '固定' in str(ei.value) and not P.seal_opened.exists()


def test_open_real_mode_writes_once_in_tmp_and_never_reads_seal_in_tests(tmp_path, monkeypatch):
    """本物の形(dry でない)の open を一時の場所で: seal_opened.json を 1 回だけ書き、集合の段(封を読む)は試験の中で止まる"""
    P = E.Paths(tmp_path / 'o', tmp_path / 'o' / 'seal_opened.json', False)
    monkeypatch.setattr(E, 'weights_of', lambda names: {n: {'best_sha256': 'sha-%s' % n} for n in names})
    c = E.read_candidates(dec3(), DEC2)
    with pytest.raises(SystemExit) as ei:
        E.cmd_open(ns(), P, c, checks=lambda: [{'what': 'x', 'rc': 0, 'out': 'OK'}], date_fn=lambda: E.OPEN_DATE)
    assert '試験の中で本物の封の記録を読もうとしました' in str(ei.value)
    rec = json.loads(P.seal_opened.read_text(encoding='utf-8'))
    assert rec['schema'] == E.SCHEMA and rec['dry_run'] is False and rec['seal_logs'] == E.seal_logs()
    assert rec['predict']['v3']['models'] == E.FIVE_RUNS + [V1N, V2N] and rec['predict']['v2']['models'] == E.FIVE_RUNS
    assert rec['predict']['v2']['candidates'] == {'BF1': None, 'BF2': None, 'C1': None, 'C2': None, '予備': None}
    assert rec['weights'][V1N]['best_sha256'] == 'sha-%s' % V1N and rec['date_note'] is None
    assert not os.access(P.seal_opened, os.W_OK) or os.geteuid() == 0           # 読み取りだけ
    with pytest.raises(SystemExit) as ei:                                       # 2 回目は止まる
        E.cmd_open(ns(), P, c, checks=lambda: [], date_fn=lambda: E.OPEN_DATE)
    assert 'もう開けました' in str(ei.value)


def test_dry_paths_cannot_point_to_real_seal():
    with pytest.raises(SystemExit):
        E.Paths(REAL.parent, REAL, True)


def test_main_all_needs_dry_run():
    with pytest.raises(SystemExit):
        E.main(['all'])


# ---------------------------------------------------------------- 拒否権と順

def st(tm=16.0, tsd=1.0, vm=0.5, vsd=4.0):
    return {'T': (tm, tsd), 'V': (vm, vsd)}


def test_veto_one_boundaries():
    s = st()
    assert E.veto_one(18.0, 0.5, s)['veto'] is False                            # T − 平均 = 2 sd ちょうどは外さない(> で外す)
    assert E.veto_one(18.0001, 0.5, s)['veto_T'] is True
    v = E.veto_one(16.0, 10.5, s)                                                # V の線は max(10, 2 × 4) = 10
    assert v['v_thr'] == 10.0 and v['veto'] is False
    assert E.veto_one(16.0, 10.5001, s)['veto_V'] is True
    v = E.veto_one(16.0, 12.0, st(vsd=6.0))                                      # 2 sd = 12 が 10 より大きい
    assert v['v_thr'] == 12.0 and v['veto'] is False
    assert E.veto_one(None, 0.0, s)['veto'] is None and E.veto_one(30.0, None, s)['veto'] is True   # T で外れれば V は要らない
    assert E.veto_one(16.0, 0.0, st(tsd=None))['veto'] is None


def test_choose_order_and_stops_at_undecided():
    cands = {'BF2-1': 'b', 'V1': 'x', 'V2': 'y'}
    assert E.choose(E.V3_ORDER, cands, {'BF2-1': {'veto': False}, 'V1': {'veto': False}, 'V2': {'veto': False}})[:2] == ('b', 'BF2-1')
    assert E.choose(E.V3_ORDER, cands, {'BF2-1': {'veto': True}, 'V1': {'veto': True}, 'V2': {'veto': False}})[:3] == \
        ('y', 'V2', '決まった')
    r = E.choose(E.V3_ORDER, cands, {'BF2-1': {'veto': True}, 'V1': {'veto': None}, 'V2': {'veto': False}})
    assert r[0] is None and r[2] == '決まらない'                                  # 後ろの V2 は見ない
    r = E.choose(E.V3_ORDER, cands, {k: {'veto': True} for k in cands})
    assert r[0] is None and r[2] == '無し(B のまま)'
    assert E.choose(E.V3_ORDER, {'BF2-1': None, 'V1': None, 'V2': None}, {})[2] == '無し(B のまま)'


def _fields(t, v):
    return {'seal_t_l1_us': t, 'seal_v_thr_vs_b_us': v}


def test_judge_core_v3_and_v2_rules():
    five_t, five_v = [15.0, 16.0, 17.0, 16.0, 16.0], [2.0, -3.0, 4.0, -1.0, 0.0]
    f = {n: _fields(t, v) for n, t, v in zip(E.FIVE_RUNS, five_t, five_v)}
    tm, tsd = float(np.mean(five_t)), float(np.std(five_t, ddof=1))
    f['v1'] = _fields(tm + 2 * tsd + 0.01, 0.0)                                  # T で外れる
    f['v2'] = _fields(tm, 30.0)                                                  # V で外れる
    f['c1'] = _fields(tm, 30.0)                                                  # v2 の候補: V は見ない(T だけ)
    J = E.judge_core(f, {'BF2-1': None, 'V1': 'v1', 'V2': 'v2'}, {'BF1': None, 'BF2': None, 'C1': 'c1', 'C2': None})
    assert J['v3']['vetoes']['V1']['veto_T'] and not J['v3']['vetoes']['V1']['veto_V']
    assert J['v3']['vetoes']['V2']['veto_V'] and not J['v3']['vetoes']['V2']['veto_T']
    assert J['v3']['slot'] is None and J['v3']['status'] == '無し(B のまま)'
    assert J['v2_D5b']['slot'] == 'c1' and J['v2_D5b']['vetoes']['C1']['veto'] is False
    assert abs(J['five']['T']['sd'] - tsd) < 1e-12 and J['five']['V']['mean'] == pytest.approx(0.4)
    J = E.judge_core(f, {'BF2-1': None, 'V1': 'v1', 'V2': 'v2'}, {'BF1': None, 'BF2': None, 'C1': None, 'C2': None})
    assert J['v2_D5b']['status'] == '無し(候補 0)'


def test_fields_by_hand():
    """V = 行 2 の車に出る値(ゲイン 0.90・前進側だけ・各モデルの中立)の 候補 − B。T = 10 行 × 2 の |予測 − 人| の平均"""
    n = 4
    R = {'y': np.zeros((n, 10, 2)), 'y_valid': np.ones(n, bool), 'model_out': np.zeros((n, 2)), 'gains': np.ones((n, 2)),
         'trim': np.zeros(n), 'lookahead': np.full(n, 3)}
    for m in E.MASKS:
        R[m] = np.zeros(n, bool)
    R['m_T'][:2] = True
    R['m_V'][2:] = True
    R['y'][:, :, 0], R['y'][:, :, 1] = 1440.0, 1450.0
    pc = np.zeros((n, 10, 2))
    pb = np.zeros((n, 10, 2))
    pc[:, :, 0], pc[:, :, 1] = 1450.0, 1430.0           # T: |10| と |−20| → 15
    pb[:, :, 0], pb[:, :, 1] = 1440.0, 1450.0
    pc[2:, 2, 1], pb[2:, 2, 1] = 1300.0, 1320.0
    pc[3, 2, 1] = 1560.0                                # 後ろ側(中立より上)はゲインを掛けない
    refc, refb = (1441.0, 1496.0), (1433.0, 1500.0)
    f = E.fields_of(pc, pb, refc, refb, R)
    assert f['seal_t_l1_us'] == pytest.approx(15.0)
    a = (1496 + 0.9 * (1300 - 1496)) - (1500 + 0.9 * (1320 - 1500))
    b = 1560.0 - (1500 + 0.9 * (1320 - 1500))
    assert f['seal_v_thr_vs_b_us'] == pytest.approx((a + b) / 2)
    assert f['seal_h_l1_us'] is None and f['seal_standstill_go_frac'] is None and f['seal_bfast_gb_fwd_vs_b_us'] is None


# ---------------------------------------------------------------- 作り物の封で全部を流す

@pytest.mark.parametrize('scenario,slot,label,veto', [
    ('pass', V1N, 'V1', None), ('veto_v', V2N, 'V2', 'veto_V'), ('veto_t', V2N, 'V2', 'veto_T'), ('veto_all', None, None, None)])
def test_dry_run_all_scenarios(tmp_path, scenario, slot, label, veto, capsys):
    d = tmp_path / scenario
    assert E.main(['--dry-run', 'all', '--scenario', scenario, '--dry-dir', str(d)]) == 0
    prop = json.loads((d / 'e5_proposed.json').read_text(encoding='utf-8'))
    e5 = prop['E5']
    assert prop['dry_run'] is True and e5['slot'] == slot and e5['slot_label'] == label
    assert e5['recount']['same'] is True
    assert e5['status'] == ('決まった' if slot else '無し(B のまま)')
    if veto:
        v = e5['vetoes']['V1']
        assert v['veto'] is True and v[veto] is True and v[{'veto_V': 'veto_T', 'veto_T': 'veto_V'}[veto]] is False
    if scenario == 'veto_all':
        assert e5['vetoes']['V1']['veto_V'] and e5['vetoes']['V2']['veto_T']
    assert prop['v2_D5b']['status'] == '無し(候補 0)'
    md = (d / 'e5.md').read_text(encoding='utf-8')
    assert '作り物の封' in md and '## 3. 候補' in md
    rec = json.loads((d / 'seal_opened.json').read_text(encoding='utf-8'))
    assert rec['dry_run'] is True and rec['fix_checks'] is None


def test_dry_run_none_does_not_open(tmp_path):
    d = tmp_path / 'none'
    assert E.main(['--dry-run', 'all', '--scenario', 'none', '--dry-dir', str(d)]) == 0
    assert not (d / 'seal_opened.json').exists()
    po = json.loads((d / 'plan_only.json').read_text(encoding='utf-8'))
    assert po['plan']['n_candidates'] == 0 and 'B のまま' in po['plan']['decision']


def test_dry_run_opens_only_once(tmp_path):
    d = tmp_path / 'once'
    assert E.main(['--dry-run', 'all', '--scenario', 'pass', '--dry-dir', str(d)]) == 0
    with pytest.raises(SystemExit) as ei:
        E.main(['--dry-run', 'all', '--scenario', 'pass', '--dry-dir', str(d)])
    assert 'もう開けました' in str(ei.value)


def test_reference_model_after_open_is_never_the_slot(tmp_path):
    d = tmp_path / 'ref'
    assert E.main(['--dry-run', 'all', '--scenario', 'veto_all', '--dry-dir', str(d)]) == 0
    extra = 'v3_20261026-v3n-aug2-e60-s0'
    with pytest.raises(SystemExit):                                             # 一覧に無い名前は --reference なしでは予測しない
        E.main(['--dry-run', 'predict', '--scenario', 'veto_all', '--dry-dir', str(d), '--runs', extra])
    assert E.main(['--dry-run', 'predict', '--scenario', 'veto_all', '--dry-dir', str(d), '--runs', extra, '--reference']) == 0
    assert E.main(['--dry-run', 'judge', '--scenario', 'veto_all', '--dry-dir', str(d)]) == 0
    o = json.loads((d / 'e5.json').read_text(encoding='utf-8'))
    assert o['reference_after_open'] == [extra] and extra in o['fields']
    assert o['judge']['v3']['slot'] is None and o['judge']['recount']['same'] is True


def test_judge_waits_when_predictions_missing(tmp_path):
    d = tmp_path / 'miss'
    P = E.dry_paths(d)
    c = E.read_candidates(E.dry_decisions('pass'), DEC2)
    E.cmd_open(ns(), P, c)
    R, _ = E.load_rows(P)
    fn = E.fake_predict_fn('pass', c['v3'], R)
    E.predict(P, E.FIVE_RUNS + [V1N], fn, check_b=False)                       # V2 が無い
    o = E.judge(P)
    assert o['missing'] == [V2N] and o['judge']['v3']['status'].startswith('まだ') and o['judge']['v3']['slot'] is None


def test_predict_real_mode_checks_weights_from_open(tmp_path, monkeypatch):
    """本物の形の予測: 封を開けたときの best.pth の sha256 と違えば止まる(開けたあとに作り直さない)"""
    P = E.Paths(tmp_path / 'o', tmp_path / 'o' / 'seal_opened.json', False)
    runs = tmp_path / 'runs'
    for n in E.FIVE_RUNS + [V1N, V2N]:
        (runs / n).mkdir(parents=True)
        (runs / n / 'best.pth').write_bytes(n.encode())
        (runs / n / 'model_config.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(E, 'run_dir', lambda n: runs / n)
    c = E.read_candidates(dec3(), DEC2)
    pl = E.plan_of(c)
    E.write_once(P.seal_opened, {'schema': E.SCHEMA, 'dry_run': False, 'predict': pl['predict'],
                                 'weights': E.weights_of(pl['predict']['all'])})
    R = E.synthetic_rows()
    E.save_rows(P, R, {'logs': ['作り物'], 'synthetic': True})
    fn = E.fake_predict_fn('pass', c['v3'], R)
    E.predict(P, [E.B_RUN], lambda *a: fn(a[0], a[1], len(R['y']), a[3]), check_b=False)
    (runs / V1N / 'best.pth').write_bytes(b'remade')
    with pytest.raises(SystemExit) as ei:
        E.predict(P, [V1N], lambda *a: fn(a[0], a[1], len(R['y']), a[3]), check_b=False)
    assert '封を開けたときと違います' in str(ei.value)


def test_dry_logs_must_not_be_seal_logs(tmp_path):
    with pytest.raises(SystemExit) as ei:
        E.main(['--dry-run', 'all', '--scenario', 'pass', '--dry-dir', str(tmp_path / 's'), '--dry-logs3', sorted(SEAL)[0]])
    assert '本物の封の記録は使いません' in str(ei.value)


def test_selfcheck_refuses_seal_logs():
    with pytest.raises(SystemExit):
        E.selfcheck(t_logs=(sorted(SEAL)[0],))


def test_commands_text():
    t = E.commands_text(E.plan_of(E.read_candidates(dec3(), DEC2)))
    assert t.count('jf-heavy --lock big') == 4 and 'predict --runs' in t and 'check_seal_v2.py' in t and 'judge' in t
    t0 = E.commands_text(E.plan_of(E.read_candidates(dec3(v1=None, v2=None), DEC2)))
    assert '候補 0 なら B のまま' in t0 and '今の答え: B のまま' in t0


# ---------------------------------------------------------------- 封でない記録で、作りが今の物と同じか(数秒)

def test_selfcheck_on_nonseal_logs_is_same():
    res = E.selfcheck(t_logs=('20261010-171639',), ai_logs=('20261024-160654', '20261024-162033'))
    assert res['same'], json.dumps({k: v.get('same') for k, v in res['parts'].items()}, ensure_ascii=False)
    m = res['parts']['masks_vs_sets_v2']['masks']
    assert m['pre_de']['mine'] > 0 and m['de_pass']['mine'] > 0                 # 空の照らしではない
    a = res['parts']['A_vs_stage0_2']
    assert a['rows'] > 1000 and a['b_reproduces_recorded']['median_abs_us_pooled'] < 1.0   # 段 0 の 2(0.109 µs)と同じくらい


def _rows_and_judge(tmp_path, sess_of, name):
    arrs, meta = E.build_rows(list(sess_of), sess_of, 'seal')
    P = E.dry_paths(tmp_path / name)
    c3 = E.read_candidates(E.dry_decisions('pass'), DEC2)
    pl = E.plan_of(c3)
    E.write_once(P.seal_opened, {'schema': E.SCHEMA, 'dry_run': True, 'predict': pl['predict'], 'weights': {}})
    E.save_rows(P, arrs, dict(meta, from_logs=True))
    R, _ = E.load_rows(P)
    E.predict(P, pl['predict']['all'], E.fake_predict_fn('pass', c3['v3'], R), check_b=False)
    return arrs, E.judge(P)


def test_build_rows_from_nonseal_log_without_straight_is_undecided(tmp_path):
    """封でない記録 1 本(直線の行が無い)から同じ作りで集合 → V が出せないので「決まらない」(足した決め c)"""
    lg = '20261024-160536'
    arrs, o = _rows_and_judge(tmp_path, {lg: 3}, 'nostraight')
    c = E.counts_of(arrs)
    assert c['m_T'] > 0 and c['T_windows'] == 2 and c['m_V'] == 0 and c['rows'] == len(arrs['img'])
    assert all(Path(p).exists() for p in arrs['img'][:50]) and arrs['tof'].shape[1:] == (42, 54)
    t = arrs['part'] == 1
    assert (arrs['genre_kind'][t] == 1).all() and arrs['y_valid'][t].all() and (arrs['genre_kind'][arrs['m_H']] == 1).all()
    J = o['judge']
    assert J['five']['V']['mean'] is None and J['v3']['slot'] is None and J['v3']['status'] == '決まらない'
    assert J['v3']['vetoes']['V1']['veto'] is None and J['recount']['same'] is None


def test_build_rows_from_two_nonseal_logs_decides(tmp_path):
    """封でない記録 2 本(1 本目を「1 回目」、2 本目を「3 回目」に)→ 集合・作り物の予測・判定が通る"""
    arrs, o = _rows_and_judge(tmp_path, {'20261024-160536': 1, '20261024-162033': 3}, 'two')
    c = E.counts_of(arrs)
    assert c['m_V'] > 0 and c['m_A'] > 0 and c['m_V_desk'] == 0                  # 10/24 2 回目の B の周は 9.5 秒未満(desk の決まりでは 0)
    assert set(arrs['sess'][arrs['m_V']].tolist()) == {3}
    assert o['judge']['v3']['slot'] == V1N and o['judge']['recount']['same'] is True
    assert o['b_reproduces_recorded'] is not None
