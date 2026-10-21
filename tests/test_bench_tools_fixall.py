"""ベンチマークの道具の直し(2026-10-21 Fix-all、V-protocol の major 3・major 9・minor 11)の試験。読み取りだけ・GPU なし

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python -m pytest -q tests/test_bench_tools_fixall.py

確かめること:
  - 走ったあとにシーンの説明(scene.json)へ書いた V_after・「安全で中止」・「回れない」が、道具の説明欄に入る
    (section_scorecard.effective_description・bench_report.collect)。直しが無ければ記録の説明のまま。食い違いは checks に残す
  - 時間のずれの判定は直し A7(ラップも差の 90% 区間で判定)。protocol.json のままの判定は flag_protocol_v1 に残り、違えば note
  - 収集オンで録った dry run だけの記録(AI の時間 0)は、本番の記録を「走り直し(K = 2)」にしない
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import bench_report as br  # noqa: E402
import section_scorecard as ss  # noqa: E402

BASE = ('bench=real_bench_20261023 day=2026-10-23 block=2 cand=SLOW model=bench_20261021-slow-s0 battery=#1 '
        'V_before=8.30 V_after=__ check=bumper7_tof_ok memo=')


def make_log(raw, log, scene_descs, base=BASE, run='bench_20261021-slow-s0'):
    d = raw / log
    d.mkdir(parents=True)
    (d / 'log.json').write_text(json.dumps({'logfile': log, 'description': base, 'drive': {'run': run}}), encoding='utf-8')
    for i, (desc, mtime) in enumerate(scene_descs):
        s = d / ('%s-%04d' % (log, i + 1))
        s.mkdir()
        f = s / 'scene.json'
        f.write_text(json.dumps({'name': s.name, 'log': log, 'description': desc}), encoding='utf-8')
        os.utime(f, (mtime, mtime))
    return d


def test_no_edit_keeps_log_description(tmp_path):
    make_log(tmp_path, '20261023-100000', [(BASE, 1000), (BASE, 1001)])
    desc, info = ss.effective_description('20261023-100000', BASE, tmp_path)
    assert desc == BASE and info['edited_scenes'] == [] and not info['changed']
    # 記録のフォルダが無い・scene.json が壊れている → そのまま
    assert ss.effective_description('20261023-999999', BASE, tmp_path)[0] == BASE
    bad = make_log(tmp_path, '20261023-100100', [(BASE, 1000)])
    (bad / '20261023-100100-0001' / 'scene.json').write_text('{broken', encoding='utf-8')
    assert ss.effective_description('20261023-100100', BASE, tmp_path)[0] == BASE


def test_after_run_edit_reaches_tools(tmp_path):
    edited = BASE.replace('V_after=__', 'V_after=7.62').replace('memo=', 'memo=R1 で 3 周続けて止まった 安全で中止')
    make_log(tmp_path, '20261023-101000', [(BASE, 1000), (BASE, 1001), (edited, 2000)])
    desc, info = ss.effective_description('20261023-101000', BASE, tmp_path)
    d = ss.parse_description(desc)
    assert d['V_after'] == '7.62' and d['V_before'] == '8.30' and d['battery'] == '1' and d['cand'] == 'SLOW'
    assert d['block'] == 2 and d['day'] == '2026-10-23' and d['check'] == 'bumper7_tof_ok'
    assert '安全で中止' in desc and info['edited_scenes'] == ['20261023-101000-0003'] and info['changed']
    assert any(any(w in desc for w in words) for words in [br.FLAG_WORDS['safety_stop']])
    # 「回れない」を memo= の外に書いても落とさない
    make_log(tmp_path, '20261023-102000', [(BASE + ' ', 1000), (BASE.replace('memo=', '回れない memo='), 3000)])
    desc2, _ = ss.effective_description('20261023-102000', BASE, tmp_path)
    assert '回れない' in desc2


def test_conflicting_edits_use_newest_and_are_reported(tmp_path):
    e1 = BASE.replace('V_after=__', 'V_after=7.60')
    e2 = BASE.replace('V_after=__', 'V_after=7.70')
    make_log(tmp_path, '20261023-103000', [(e2, 3000), (e1, 2000)])        # 0001 が新しい直し
    desc, info = ss.effective_description('20261023-103000', BASE, tmp_path)
    assert ss.parse_description(desc)['V_after'] == '7.70'
    assert len(info['conflicts']) == 1 and 'V_after' in info['conflicts'][0]
    bench = {'bench': 'real_bench_20261023', 'candidates': [{'id': 'SLOW', 'run': 'bench_20261021-slow-s0'}],
             'days': [{'date': '2026-10-23', 'blocks': [{'block': 2, 'id': 'SLOW'}]}]}
    rows, un = br.collect(['20261023-103000'], bench, None, tmp_path / 'no_labels', tmp_path)
    assert not un and rows[0]['desc']['V_after'] == '7.70'
    assert any(k == 'scene' for k, _ in rows[0]['checks'])
    assert rows[0]['desc_scene_edits'] == ['20261023-103000-0001', '20261023-103000-0002']
    # --assign の説明欄は人の直しなので、シーンの説明では上書きしない
    rows2, _ = br.collect(['20261023-103000'], bench, {'20261023-103000': BASE}, tmp_path / 'no_labels', tmp_path)
    assert rows2[0]['desc']['V_after'] is None and not any(k == 'scene' for k, _ in rows2[0]['checks'])


def test_dry_run_only_log_does_not_make_rerun(monkeypatch):
    calls = []

    def fake_score(log, labels_dir, raw_dir, desc, bench, K, left):
        calls.append((log, K, left))
        if log == 'dry':
            return {'ai_s': 0.0, 'window': {'ai_s': 0.0}}
        return {'ai_s': 250.0, 'window': {'ai_s': 240.0}}
    monkeypatch.setattr(ss, 'score_log', fake_score)
    es = [{'log': 'dry', 'day': '2026-10-23', 'block': 2, 'cand': 'SLOW', 'description': BASE},
          {'log': 'real', 'day': '2026-10-23', 'block': 2, 'cand': 'SLOW', 'description': BASE},
          {'log': 'again', 'day': '2026-10-23', 'block': 2, 'cand': 'SLOW', 'description': BASE}]
    br.score_entries(es, None, None, None)
    assert es[0]['dry_only'] and not es[0]['rerun']
    assert es[1]['rerun'] is False and es[1]['K'] == 3 and calls[1] == ('real', 3, 240.0)      # 本番は K = 3・240 秒
    assert es[2]['rerun'] is True and es[2]['K'] == 2 and calls[2] == ('again', 2, 0.0)       # 本当の走り直しは今までどおり


def _rows(n, lap_fn, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append({'ta': 10 * i, 'tg': 10 * i + 2, 'incidents': 0, 'collisions': 0, 'strong': 0, 'tk_failure': 0, 'tk_all': 0,
                    'tk_stuck_preemptive': 0, 'r1_strong': False, 'r1_coll': False, 'gm_coll': False, 'de_coll': False,
                    'clean': True, 'lap_time_s': round(float(lap_fn(rng)), 3), 'reboot': False, 'n_reboots': 0,
                    'collision_locs': [], 'entry_erpm': 5000.0, 't_r1gm_s': 2.0, 't_rest_s': 7.0, 'holds': 0,
                    'ai_s_at_mid': 10.0 * i + 5, 'recoveries': 0})
    return out


def _entry(log, block, cand, rows):
    return {'log': log, 'block': block, 'cand': cand, 'day': '2026-10-23', 'desc': {'block': block, 'cand': cand},
            'K': 2, 'flags': [], 'rerun': False, 'score': {'window': {'ai_s': 10.0 * len(rows)}, 'laps_protocol': rows}}


def test_drift_rule_a7():
    # ラップのばらつきが大きい日(σ 0.9 秒)に、B の最初と最後の中央値が 0.3 秒より少し違う → A7 ではずれなし、元の決まりでは「ずれあり」
    es = [_entry('a', 1, 'B', _rows(20, lambda r: 9.0 + r.normal(0, 0.9), 1)),
          _entry('b', 2, 'C1', _rows(20, lambda r: 9.5 + r.normal(0, 0.9), 2)),
          _entry('c', 3, 'B', _rows(20, lambda r: 9.0 + r.normal(0, 0.9), 3))]
    br.apply_battery(es, {'used': False})
    dr = br.compare_day('2026-10-23', es, n_boot=500)['reference']['drift']
    lo, hi = dr['d_lap_median_ci90']
    assert dr['rule'] == 'A7' and dr['flag'] == bool(lo > 0 or hi < 0)
    assert dr['flag_protocol_v1'] == bool(abs(dr['d_lap_median_s']) > 0.3)
    if dr['flag'] != dr['flag_protocol_v1']:
        assert 'A7' in dr['note']
    # はっきりしたずれ(1 秒・σ 0.1)は A7 でも「ずれあり」
    es2 = [_entry('a', 1, 'B', _rows(20, lambda r: 9.0 + r.normal(0, 0.1), 1)),
           _entry('b', 2, 'C1', _rows(20, lambda r: 9.5 + r.normal(0, 0.1), 2)),
           _entry('c', 3, 'B', _rows(20, lambda r: 10.0 + r.normal(0, 0.1), 3))]
    br.apply_battery(es2, {'used': False})
    cm = br.compare_day('2026-10-23', es2, n_boot=500)
    assert cm['reference']['drift']['flag'] is True and cm['cands']['C1']['ref'] == 'interpolated'


def test_label_events_skips_bad_log_and_continues(tmp_path):
    """label_events.py: 壊れた記録(log.json が無い)があっても、ほかの記録のラベルを作る。終了コード 1(V-tools minor 7)"""
    import shutil
    import subprocess
    from jetformer import config
    good = '20261010-172505'
    if not (config.RAW_DIR / good).is_dir():
        import pytest
        pytest.skip('10/10 の記録がありません')
    raw = tmp_path / 'raw'
    shutil.copytree(config.RAW_DIR / good, raw / good)
    (raw / '20261023-000000').mkdir()                                # log.json の無い記録
    r = subprocess.run([sys.executable, str(ROOT / 'tools' / 'label_events.py'), '20261023-000000', good,
                        '--raw', str(raw), '--out', str(tmp_path / 'labels')], capture_output=True, text=True, timeout=300)
    assert r.returncode == 1, r.stderr[-500:]
    assert (tmp_path / 'labels' / (good + '.json')).exists() and (tmp_path / 'labels' / (good + '.npz')).exists()
    assert '20261023-000000' in r.stderr and '飛ばして続けます' in r.stderr
