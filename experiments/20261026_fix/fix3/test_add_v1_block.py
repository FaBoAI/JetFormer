"""係 e5(2026-10-26): add_v1_block.py の試験。本物の data/bench_candidates.json は読むだけ(一時の写しに書く)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -m pytest \
        -q -p no:cacheprovider --basetemp=experiments/20261026_fix/fix3/tmp_e5 experiments/20261026_fix/fix3/test_add_v1_block.py
"""
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
REAL_BENCH = JF / 'data' / 'bench_candidates.json'
RUN = 'v3_20261026-v3d-aug2-e60-s0'


def _load():
    spec = importlib.util.spec_from_file_location('add_v1_block_t', str(HERE / 'add_v1_block.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture
def bench(tmp_path):
    before = sha(REAL_BENCH)
    p = tmp_path / 'bench_candidates.json'
    p.write_bytes(REAL_BENCH.read_bytes())
    yield p
    assert sha(REAL_BENCH) == before, '本物の bench_candidates.json が変わった'


def proposal(tmp_path, slot=RUN, label='V1', status='決まった', dry=False):
    p = tmp_path / 'e5_proposed.json'
    p.write_text(json.dumps({'dry_run': dry, 'E5': {'slot': slot, 'slot_label': label, 'status': status}}, ensure_ascii=False),
                 encoding='utf-8')
    return p


def test_dry_run_prints_diff_and_writes_nothing(bench, tmp_path, capsys):
    before = sha(bench)
    assert A.main(['--dry-run', '--bench', str(bench), '--proposed', str(proposal(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert sha(bench) == before and '書いていない' in out
    for s in ('24 5-1 V3', '25 5-2 V3', '26 5-3 B', '27 5-4 V3', '押したときの速度 0.85', '+   "id": "V3"'):
        assert s in out, s


def test_build_only_appends(bench):
    cur = json.loads(bench.read_text(encoding='utf-8'))
    new = A.build(cur, RUN, 'V1', '決まった')
    assert A.same_except_new(cur, new) == []
    assert new['candidates'][:-1] == cur['candidates'] and new['candidates'][-1]['id'] == 'V3' and new['candidates'][-1]['run'] == RUN
    od = [d for d in cur['days'] if d['date'] == '2026-11-03'][0]
    nd = [d for d in new['days'] if d['date'] == '2026-11-03'][0]
    assert nd['blocks'][:23] == od['blocks'] and [b['block'] for b in nd['blocks'][23:]] == [24, 25, 26, 27]
    assert [b['id'] for b in nd['blocks'][23:]] == ['V3', 'V3', 'B', 'V3']
    assert [b['params']['throttle_gain'] for b in nd['blocks'][23:]] == [0.85, 0.9, 0.9, 0.9]
    assert all(b['expect']['launch_boost'] == 0 and b['expect']['zone_caps'] == 0 and b['expect']['speed_auto'] == 0
               for b in nd['blocks'][23:])
    assert [d for d in new['days'] if d['date'] != '2026-11-03'] == [d for d in cur['days'] if d['date'] != '2026-11-03']
    assert new['changes'][:-1] == cur['changes'] and 'V3' in new['changes'][-1]['what']
    for k in cur:
        if k not in ('days', 'candidates', 'changes'):
            assert new[k] == cur[k], k


def test_same_except_new_catches_changes(bench):
    cur = json.loads(bench.read_text(encoding='utf-8'))
    new = A.build(cur, RUN, 'V1', '決まった')
    bad = copy.deepcopy(new)
    [d for d in bad['days'] if d['date'] == '2026-11-03'][0]['blocks'][0]['params']['throttle_gain'] = 1.0
    assert A.same_except_new(cur, bad)
    bad = copy.deepcopy(new)
    bad['candidates'][0]['run'] = 'x'
    assert A.same_except_new(cur, bad)
    bad = copy.deepcopy(new)
    bad['baseline']['params']['start_kick'] = 0
    assert A.same_except_new(cur, bad)
    bad = copy.deepcopy(new)
    bad['days'][0]['blocks'] = []
    assert A.same_except_new(cur, bad)


def test_screen_reads_new_blocks(bench, tmp_path):
    """画面と同じ読み方(web/bench.load_candidates): 警告が増えない・24〜27 が並ぶ・V3 は説明欄 cand=V3"""
    from jetformer.web import bench as wb
    old = wb.load_candidates(path=bench)
    cur = json.loads(bench.read_text(encoding='utf-8'))
    p2 = tmp_path / 'new.json'
    p2.write_text(A.dumps(A.build(cur, RUN, 'V1', '決まった')), encoding='utf-8')
    new = wb.load_candidates(path=p2)
    assert new['ok'] and new['warnings'] == old['warnings']
    day = [d for d in new['days'] if d['date'] == '2026-11-03'][0]
    bl = {b['block']: b for b in day['blocks']}
    assert sorted(bl)[-4:] == [24, 25, 26, 27] and bl[24]['run'] == RUN and bl[26]['run'] == wb.B_RUN
    assert 'cand=V3' in bl[24]['description'] and 'throttle_gain=0.85' in bl[24]['description']
    v3 = [c for c in new['candidates'] if c['id'] == 'V3'][0]
    ok, why = A.drivable(RUN)
    assert v3['drivable'] == bool(ok)
    assert [b for b in old['days'] if b['date'] == '2026-11-03'][0]['blocks'] == day['blocks'][:23]


def test_write_needs_approval_and_makes_backup(bench, tmp_path, monkeypatch):
    monkeypatch.setattr(A, 'drivable', lambda run: (True, '試験'))
    with pytest.raises(SystemExit):
        A.main(['--write', '--bench', str(bench), '--run', RUN, '--label', 'V1'])
    old_text = bench.read_text(encoding='utf-8')
    assert A.main(['--write', '--bench', str(bench), '--run', RUN, '--label', 'V1', '--user-approval', '試験の了承']) == 0
    baks = list(tmp_path.glob('bench_candidates.json.bak-*-e5'))
    assert len(baks) == 1 and baks[0].read_text(encoding='utf-8') == old_text
    new = json.loads(bench.read_text(encoding='utf-8'))
    assert new['changes'][-1]['why'].endswith('試験の了承') and new['candidates'][-1]['id'] == 'V3'
    with pytest.raises(SystemExit):                                              # 2 回目は止まる(もう V3 がある)
        A.main(['--write', '--bench', str(bench), '--run', RUN, '--label', 'V1', '--user-approval', '試験の了承'])


def test_write_refuses_not_drivable(bench, monkeypatch):
    monkeypatch.setattr(A, 'drivable', lambda run: (False, 'エンジンが無い'))
    before = sha(bench)
    with pytest.raises(SystemExit):
        A.main(['--write', '--bench', str(bench), '--run', RUN, '--user-approval', 'x'])
    assert sha(bench) == before


def test_no_slot_or_dry_proposal(bench, tmp_path, capsys):
    before = sha(bench)
    assert A.main(['--dry-run', '--bench', str(bench), '--proposed', str(proposal(tmp_path, slot=None, label=None,
                                                                                      status='無し(B のまま)'))]) == 0
    assert '1 枠は無し' in capsys.readouterr().out and sha(bench) == before
    with pytest.raises(SystemExit):
        A.main(['--dry-run', '--bench', str(bench), '--proposed', str(proposal(tmp_path, dry=True))])
    assert A.main(['--dry-run', '--bench', str(bench), '--proposed', str(proposal(tmp_path, dry=True)), '--allow-dry-proposal']) == 0
    assert A.main(['--dry-run', '--bench', str(bench), '--proposed', str(tmp_path / 'none.json')]) == 0
    assert sha(bench) == before
