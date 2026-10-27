"""発進の前進の記録の再生(T13)。2026-10-26
10/24 の B の発進 9 回の走行の記録(data/raw。あるときだけ)を、手を入れたあとの assist.Assist に OFF(launch = None)で
頭から流し、出した値が記録の out と同じかを見る。台本は experiments/20261026_fix/launch/replay_launch.py。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q tests/test_launch_replay.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from jetformer import assist as A

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'experiments' / '20261026_fix' / 'launch' / 'replay_launch.py'
RAW = ROOT / 'data' / 'raw'


def _mod():
    if not SCRIPT.exists():
        pytest.skip('台本がありません: %s' % SCRIPT)
    spec = importlib.util.spec_from_file_location('replay_launch', str(SCRIPT))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _need(m):
    missing = [log for log, _ in m.B9 if not (RAW / log / 'log.json').exists()]
    if missing:
        pytest.skip('10/24 の記録がありません: %s' % ', '.join(missing))


def test_t13_off_replay_equals_recorded_out():
    m = _mod()
    _need(m)
    tot = same = win_tot = win_same = n_launch = 0
    diffs = []
    for log, no in m.B9:
        r = m.replay_off(log)
        win = np.zeros(len(r['t_us']), bool)
        for k0, k1, stopped in r['sections']:
            if stopped:
                n_launch += 1
                win |= (r['t_us'] >= r['t_us'][k0]) & (r['t_us'] <= r['t_us'][k0] + int(m.WINDOW_S * 1e6))
        c = r['cmp']
        tot += int(c.sum())
        same += int((c & r['same']).sum())
        win_tot += int((c & win).sum())
        win_same += int((c & win & r['same']).sum())
        diffs += [(log, int(k), int(r['kind'][k])) for k in np.nonzero(c & ~r['same'])[0]]
        assert not (r['kind'][r['cmp']] == A.ASSIST_LAUNCH).any()           # OFF は発進の前進を出さない
    assert n_launch >= 9 and win_tot > 900
    assert win_same == win_tot, diffs[:10]                                    # 発進の場面(AI 側にしてから 3 秒)は全部同じ
    assert tot > 25000 and same >= tot - 5, diffs[:10]
    # 違うのはリカバリーの段の切り替わりの時刻だけ(再生の時刻は 撮像 + latency で、走行ループの monotonic と少しずれる)
    assert all(kind == A.ASSIST_RECOVERY for _, _, kind in diffs), diffs


def test_launch_estimate_on_recorded_starts():
    """閉ループの見積もり(diag_launch の車のモデル)。10/24 の B の 9 回は記録の条件で全部始まり、速さで終わる"""
    m = _mod()
    _need(m)
    if not (m.DIAG / 'out' / 'launches.json').exists():
        pytest.skip('diag_launch の out がありません')
    rows = m.estimate(sel_logs=set(m.B9))
    s = m.summarize(rows, lambda r: r['main_B_1024'])
    assert s['n'] == 9 and s['would_start_recorded'] == 9
    assert s['end_reasons'] == {A.LAUNCH_END_NAMES[1]: 9}
    assert 0.15 <= s['save_vs_base']['median'] <= 0.40 and s['save_vs_base']['min'] > 0.0
    assert s['dv_lh_on_minus_off']['max'] <= 0.3                              # LH の入口の速さは +0.3 m/s まで
