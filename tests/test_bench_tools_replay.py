"""区間の速度上限の再生(tools/cap_replay.py)と夜の 1 コマンド(tools/bench_nightly.sh)の試験(段 S3、2026-10-21)

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python -m pytest -q tests/test_bench_tools_replay.py

10/10 の記録で、10/16 の G1(sim_calibration_20261016/g1_replay.json の v1.1)と同じ数が出ることを確かめる。
記録・ラベル・g1_replay.json が無い機械では飛ばす。GPU・torch は使わない(約 10〜60 秒)。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import cap_replay as cr  # noqa: E402

LABELS = ROOT / 'data' / 'labels'
RAW = ROOT / 'data' / 'raw'
G1_JSON = Path('/home/jetson/jetformer_reports/sim_calibration_20261016/g1_replay.json')
G1_LOGS = {'20261010-172910': 'ai_run_006', '20261010-173445': 'ai_run_007', '20261010-203924': 'ai_run_009',
           '20261010-204208': 'ai_run_010', '20261010-204605': 'ai_run_101'}
OTHER_AI = ['20261010-171639', '20261010-172505', '20261010-181504']
LOGS_1010 = sorted(p.stem for p in LABELS.glob('20261010-*.json') if (RAW / p.stem).exists())
HAVE = len(LOGS_1010) >= 20 and G1_JSON.exists() and cr.PCAL_DEFAULT.exists()
need = pytest.mark.skipif(not HAVE, reason='10/10 の記録・ラベル・g1_replay.json・P_cal が無い')


@pytest.fixture(scope='module')
def replay1010():
    return cr.replay(LOGS_1010, g1_logs=list(G1_LOGS), other_ai_logs=OTHER_AI, strong_logs=['20261010-173445'],
                     variants=('3zone', 'R1_2.75', 'R1GM', 'DE3.3'), labels_dir=LABELS, raw_dir=RAW)


@pytest.fixture(scope='module')
def ref():
    return json.loads(G1_JSON.read_text(encoding='utf-8'))['v1.1']['summary']


@need
def test_g1_six_items_match_g1_replay(replay1010, ref):
    ga, gb = ref['gate'], replay1010['summary']['gate']
    assert gb['R1_signal_ge_90pct']['value'] == ga['R1_signal_ge_90pct']['value'] == '10/11'
    assert gb['R1_signal_strong_ge_90pct']['value'] == ga['R1_signal_007_board_ge_90pct']['value'] == '8/9'
    assert {G1_LOGS[k]: v for k, v in gb['false_signal_le_1_per_min']['value'].items()} == ga['false_signal_le_1_per_min_all5']['value']
    assert gb['gyro_r1_detection_ge_95pct_false_0']['value'] == ga['gyro_r1_detection_ge_95pct_false_0']['value']
    assert gb['lap_loss_le_0.3_3zone']['value'] == ga['lap_loss_le_0.3_R1_2.5']['value'] == 0.152
    assert gb['lap_loss_le_0.3_R1_2.75']['value'] == ga['lap_loss_le_0.3_R1_2.75']['value']
    assert gb['brake_invariants_3zone']['pass'] == ga['brake_invariants_R1_2.5']['pass'] is True
    assert gb['need_decel_reported_3zone']['value'] == ga['need_decel_reported_R1_2.5']['value']
    assert gb['need_decel_reported_R1_2.75']['value'] == ga['need_decel_reported_R1_2.75']['value']
    # 落ちた項目も同じ(10/16: 007 の板 8/9 と、夜の 3 本の誤った合図)
    assert gb['G1_failed_items_3zone'] == ['R1_signal_strong_ge_90pct', 'false_signal_le_1_per_min']
    assert ga['G1_failed_items_R1_2.5'] == ['R1_signal_007_board_ge_90pct', 'false_signal_le_1_per_min_all5']
    assert gb['G1_pass_3zone'] is False


@need
def test_g1_protocol_four_items(replay1010):
    """protocol(day2.cap_plan_A)の G1 は 4 項目(R1 の合図・誤った合図・旋回中のブレーキ・ラップの損)。10/10 は 007 の板 8/9 が
    入らないので、落ちるのは誤った合図だけ(V-tools 4 章 6)"""
    g = replay1010['summary']['gate']
    assert g['turn_brake_0_3zone']['pass'] is True and g['turn_brake_0_3zone']['value'] == {'shadow_frames': 0, 'virtual_steps': 0}
    assert g['G1_protocol_failed_items_3zone'] == ['false_signal_le_1_per_min'] and g['G1_protocol_pass_3zone'] is False
    assert g['G1_protocol_undecided_items_3zone'] == []
    md = cr.to_markdown(replay1010)
    assert 'G1(4 項目)' in md and 'G1(6 項目、参考)' in md


@need
def test_details_match_g1_replay(replay1010, ref):
    S = replay1010['summary']
    assert S['gyro_r1_detection'] == ref['gyro_r1_detection']
    assert abs(S['false_signals_pooled']['combined_per_min_strict'] - ref['false_signals']['pooled_5_logs']['combined_per_min_strict']) < 1e-12
    assert {G1_LOGS[k]: v for k, v in S['brakes']['3zone']['shadow_brake_starts'].items()} == ref['brakes']['R1_2.5']['shadow_brake_starts']
    a, b = S['lap_loss']['3zone']['nominal']['G1_logs'], ref['lap_loss']['R1_2.5']['nominal']['G1_logs']
    for k in ('laps', 'loss_mean_s', 'loss_median_s', 'loss_max_s', 'r1_entry_rec_gt_cap', 'r1_entry_capped_gt_cap_plus_0.1'):
        assert a[k] == b[k], k


@need
def test_variants_are_ordered_sensibly(replay1010):
    """DE の上限を外す・緩めると、ラップの損は小さくなる(R1 と GM は同じ)"""
    g = replay1010['summary']['gate']
    l3, lr, l33 = g['lap_loss_le_0.3_3zone']['value'], g['lap_loss_le_0.3_R1GM']['value'], g['lap_loss_le_0.3_DE3.3']['value']
    assert lr <= l33 <= l3
    for v in ('R1GM', 'DE3.3'):
        assert g['brake_invariants_' + v]['pass'] is True


@need
def test_markdown_lists_six_items(replay1010):
    md = cr.to_markdown(replay1010)
    for k in cr.ITEM_TEXT.values():
        assert k in md
    assert '**G1**' in md and '3zone' in md and 'DE3.3' in md


def test_recorded_shadow_fields_are_read():
    """S1 が AI_CMD に書く影の判定の欄(build/S1.md: cap_zone・cap_mode は番号)を読み、区間・モードに直す"""
    A = {'cap_zone': np.array([0, 2, 3, -1], np.int8), 'cap_mode': np.array([0, 1, 3, -1], np.int8),
         'cap_in_us': np.array([1400, 1380, 1390, 1500], np.float32), 'cap_on': np.array([0, 1, 1, 0], bool),
         'timestamps_us': np.arange(4)}
    rec, used = cr.recorded_shadow(A, np.arange(4))
    assert used == {'zone': 'cap_zone', 'mode': 'cap_mode', 'thr_in': 'cap_in_us', 'active': 'cap_on'}
    assert list(rec['zone']) == ['none', 'R1_app', 'R1_turn', 'off']
    assert list(rec['mode']) == ['pass', 'cap', 'brake', 'off']
    rec, used = cr.recorded_shadow({'timestamps_us': np.arange(3)}, np.arange(3))
    assert rec == {} and used == {}


def test_recorded_counts_on_brake_in_turn():
    d = {'sel': np.ones(6, bool), 'at': np.arange(6) * 0.1, 'anchors': np.array([0.25])}
    rec = {'mode': np.array(['pass', 'cap', 'brake', 'brake', 'pass', 'brake'], object),
           'zone': np.array(['R1_app', 'R1_app', 'R1_app', 'R1_turn', 'R1_turn', 'GM'], object),
           'active': np.array([1, 1, 1, 1, 1, 0], bool), 'applied': np.array([0, 1, 1, 1, 0, 0], bool)}
    c = cr.recorded_counts(d, rec)
    assert c['shadow_brake_starts'] == 2 and c['on_frames'] == 5 and c['applied_frames'] == 3
    assert c['on_brake_frames_in_r1_turn_zone'] == 1          # 0.3 秒のフレーム(区間 R1_turn)
    assert c['on_brake_frames_in_r1_turn_0_0p5s'] == 1        # 起点 0.25 → 0.75 秒の中で ON のブレーキは 0.3 秒だけ


def test_gate_with_no_collisions_is_undecided():
    g = cr._gate_rate(0, 0)
    assert g['pass'] is None


# ---------------------------------------------------------------- 夜の 1 コマンド
def test_nightly_script_syntax():
    r = subprocess.run(['bash', '-n', str(ROOT / 'tools' / 'bench_nightly.sh')], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


MOCK = Path('/home/jetson/jetformer_reports/bench_prep_20261021/build/mock_bench')


@need
@pytest.mark.skipif(not (MOCK / 'mock_assign.json').exists(), reason='模擬の割り当てが無い')
def test_nightly_runs_on_1010(tmp_path):
    env = dict(os.environ)
    r = subprocess.run(['bash', str(ROOT / 'tools' / 'bench_nightly.sh'), '20261010', '--bench', str(MOCK / 'mock_bench_candidates.json'),
                        '--assign', str(MOCK / 'mock_assign.json'), '--out', str(tmp_path / 'n'), '--report', str(tmp_path / 'r'),
                        '--mock', '--', '--desk', str(MOCK / 'mock_desk_bench.json'), '--min-laps', '3', '--k-first', '1',
                        '--k-later', '1', '--boot', '300'], capture_output=True, text=True, timeout=600, env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for p in ('n/scorecard/scorecard.md', 'n/cap_replay/cap_replay.json', 'n/brake_table/brake_table.json', 'r/README.md', 'r/report.json'):
        assert (tmp_path / p).exists(), p
    cap = json.loads((tmp_path / 'n/cap_replay/cap_replay.json').read_text(encoding='utf-8'))
    # G1 は説明欄の cand=B の AI の記録だけで判定する
    assert sorted(cap['summary']['g1_logs']) == sorted(['20261010-172910', '20261010-181504', '20261010-160306',
                                                         '20261010-155820', '20261010-172505'])
    log = (tmp_path / 'n' / 'nightly.log').read_text(encoding='utf-8')
    assert 'ラベル: 全部あるので作らない' in log           # 10/10 のラベルは作り直さない(data/labels を変えない)


# ---------------------------------------------------------------- S1 の記録の欄(cap_*)がある記録
def _write_log_with_cap_fields(src_root, dst_root, log):
    """10/10 の記録の写し(画像などは記号リンク)に、S1 の純粋な包み(safety_caps.zone_imu・zone_frame)で計算した
    cap_* を AI_CMD.npz へ足す(zone_caps = 0 の走行で記録されるのと同じ影の判定)"""
    sys.path.insert(0, str(ROOT))
    from jetformer import safety_caps as sc
    import label_events as le
    L = le.load_log(log, src_root)
    A, I, V = L['AI_CMD'], L['IMU'], L['VEHICLE']
    at_us = A['timestamps_us'].astype(np.int64)
    o = np.argsort(at_us, kind='stable')
    t0 = int(min(at_us.min(), I['timestamps_us'].min()))
    sec = lambda x: (np.asarray(x, np.int64) - t0) / 1e6      # noqa: E731
    it, vt = sec(I['timestamps_us']), sec(V['timestamps_us'])
    gz = I['gyro'][:, 2].astype(float)
    v_all = -V['erpm'].astype(float) * 0.5e-3
    st = sc.ZoneState()
    n = len(o)
    cap = {k: np.zeros(n, dt) for k, dt in (('cap_zone', np.int8), ('cap_mode', np.int8))}
    cap['cap_on'] = np.zeros(n, bool)
    cap['cap_in_us'] = A['out'][:, 1].astype(np.float32)
    ii = 0
    zi = {z: i for i, z in enumerate(sc.ZONES)}
    mi = {'off': -1, 'pass': 0, 'cap': 1, 'neutral': 2, 'brake': 3}
    for j in o:
        t = sec(at_us[j])
        while ii < len(it) and it[ii] <= t:
            kv = np.searchsorted(vt, it[ii], side='right') - 1
            st = sc.zone_imu(st, float(it[ii]), float(gz[ii]), float(v_all[max(kv, 0)]))
            ii += 1
        kv = int(np.searchsorted(vt, t, side='right') - 1)
        fresh = kv >= 0 and t - vt[kv] < 0.1
        if A['hold'][j]:
            st = sc.zone_inactive(st)
            cap['cap_zone'][j], cap['cap_mode'][j] = -1, -1
            continue
        active = bool(A['select'][j]) and not bool(A['dry_run'][j]) and fresh
        st, z = sc.zone_frame(st, float(t), float(A['out'][j, 1]), float(A['out'][j, 0]), float(v_all[max(kv, 0)]),
                              A['pred'][j, :, 0], active)
        cap['cap_zone'][j] = zi[z.zone]
        cap['cap_mode'][j] = mi[z.mode]
    by_us = {int(u): i for i, u in enumerate(at_us)}
    src = Path(src_root) / log
    dst = Path(dst_root) / log
    dst.mkdir(parents=True)
    (dst / 'log.json').symlink_to(src / 'log.json')
    for sd in sorted(p for p in src.iterdir() if p.is_dir()):
        dd = dst / sd.name
        dd.mkdir()
        for f in sd.iterdir():
            if f.name != 'AI_CMD.npz':
                (dd / f.name).symlink_to(f)
        if (sd / 'AI_CMD.npz').exists():
            with np.load(sd / 'AI_CMD.npz', allow_pickle=True) as zz:
                d = {k: zz[k] for k in zz.files}
            idx = np.array([by_us[int(u)] for u in d['timestamps_us']], int)
            for k, arr in cap.items():
                d[k] = arr[idx]
            np.savez(dd / 'AI_CMD.npz', **d)


@need
def test_replay_reads_s1_cap_fields(tmp_path):
    log = '20261010-173445'
    _write_log_with_cap_fields(RAW, tmp_path, log)
    res = cr.replay([log], g1_logs=[log], variants=('3zone',), labels_dir=LABELS, raw_dir=tmp_path)
    e = res['logs'][log]
    assert e['recorded_fields']['zone'] == 'cap_zone' and e['recorded_fields']['mode'] == 'cap_mode'
    assert e['recorded_vs_replay']['zone'] > 0.999 and e['recorded_vs_replay']['mode'] > 0.999
    assert e['recorded']['on_frames'] == 0 and e['recorded']['on_brake_frames_in_r1_turn_0_0p5s'] == 0
    assert e['recorded']['shadow_brake_starts'] == res['summary']['brakes']['3zone']['shadow_brake_starts'][log]
    res2 = cr.replay([log], g1_logs=[log], variants=('3zone',), labels_dir=LABELS, raw_dir=tmp_path, use_recorded=True)
    assert res2['summary']['gate']['R1_signal_ge_90pct'] == res['summary']['gate']['R1_signal_ge_90pct']
    # 成績表も cap_* を数える(B+上限の「R1 の旋回中のブレーキ」は OFF の走行なので 0)
    import section_scorecard as ss
    s = ss.score_log(log, LABELS, tmp_path, warmup_anchors=1, counted_ai_s=240)
    c = s['window']['cap']
    assert c['ai_frames'] > 1000 and c['on_frames'] == 0 and c['on_brake_frames_in_r1_turn'] == 0
    assert c['shadow_brake_starts'] > 0
    # 旋回中のブレーキは時刻(後から見た起点 ta → +0.5 秒)で数える。層の区間 R1_turn ではブレーキは作りの上で出ない
    assert c['shadow_brake_in_r1_turn_zone'] == 0 and c['r1_anchors'] > 0
