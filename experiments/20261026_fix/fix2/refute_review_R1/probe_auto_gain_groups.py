"""R1 の打ち消し(2026-10-26 fix2): 回 3 を「天井 + 自動合わせ ON」で走ると collect.py の summarize() が ON と OFF を
同じ組にするか。作り物の 11/3 回 3 の記録 4 本(OFF・ON 交互、V_before 8.22→8.04)を作り、AI_CMD の gains[:,1] を
走行ループと同じ式(drive._SpeedAuto.gain、基板の電圧なし → 説明欄の V_before)で書き替えて collect.main を流す。
実機・GPU 不要。書くのは引数の作業場所だけ。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 probe_auto_gain_groups.py <作業場所>
"""
import json
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tests'))
sys.path.insert(0, str(JF / 'experiments/20261026_fix/diag_launch'))
import collect  # noqa: E402
from test_standing_3lap import write_standing_log  # noqa: E402

work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=True)
raw = work / 'raw'
raw.mkdir()
bench = work / 'bench.json'
bench.write_text(json.dumps({'candidates': [{'id': 'B', 'run': 'runB'}],
                             'days': [{'date': '2026-11-03', 'blocks': [{'block': 12 + i, 'run_id': '3-%d' % (i + 1)}
                                                                         for i in range(8)]}]}), encoding='utf-8')
TG = 0.95
PRM = {'throttle_gain': TG, 'steer_gain': 1.0, 'speed_ceiling': 1, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 1,
       'auto_target_mps': 4.27, 'launch_us': 250.0, 'launch_mps': 2.5, 'launch_max_s': 1.2, 'zone_caps': 0, 'start_kick': 1}

try:
    from jetformer import drive  # noqa: E402
    SA = drive._SpeedAuto
    how = 'drive._SpeedAuto.gain(本物の走行ループのつなぎ)'
except Exception as e:      # noqa: BLE001
    SA = None
    how = 'speed_auto の純粋な関数(drive を読めない: %s)' % e
from jetformer import speed_auto  # noqa: E402


def run_gain(desc, n, t0_ns=1_800_000_000_000_000_000):
    """止まっている間(v = 0)に走行ループと同じ gain() を n フレーム呼んだ g(基板の電圧なし)"""
    if SA is not None:
        s = SA(dict(PRM), desc)
        gs = [s.gain(t0_ns + int(i * 1e9 / 30), TG, 0.0, []) for i in range(n)]
        return np.asarray(gs, float), s.src
    cfg = speed_auto.auto_params(PRM['auto_target_mps'], 8.35)
    v = speed_auto.v_before_of(desc, cfg)
    g = speed_auto.gain_of(TG, speed_auto.m_of(speed_auto.ff_of_voltage(v, cfg), 1.0, cfg), cfg)
    return np.full(n, g), speed_auto.SRC_DESC


plan = [(12, 8.22, 0), (13, 8.16, 1), (14, 8.10, 0), (15, 8.04, 1)]
desc_t = 'ai_no=%d bench=real_bench_20261023 day=2026-11-03 block=%d cand=EXP model=runB V_before=%.2f memo=x'
print('ゲインの計算:', how)
for k, (blk, vb, lb) in enumerate(plan):
    log = '20261103-13%02d00' % (k * 5)
    desc = desc_t % (blk, blk, vb)
    kw = dict(launch=(1.0, 0.5, 1), t_move=1.2, first_anchor=3.7) if lb else {}
    d, _, _ = write_standing_log(raw, log, params=dict(PRM, launch_boost=lb), run='runB', ramp_s=0.5, description=desc, **kw)
    for p in sorted(d.glob('*/AI_CMD.npz')):
        with np.load(p) as z:
            a = {kk: z[kk] for kk in z.files}
        n = len(a['timestamps_us'])
        g, src = run_gain(desc, n)
        gg = a['gains'].copy()
        gg[:, 1] = np.where(a['select'], g, TG)           # 掛けるのは AI のフレームだけ(drive.py の g_used)
        a['gains'] = gg.astype(np.float32)
        np.savez(p, **a)
    print('  %s block=%d V_before=%.2f launch_boost=%d  g(AI のフレーム)=%.4f  電圧の源=%s' % (log, blk, vb, lb, float(g[0]), src))

collect.RAW, collect.LAB, collect.BENCH, collect.OUT = raw, work / 'no_labels', bench, work / 'out'
collect.main(['--day', '20261103'])
j = json.loads((work / 'out' / 'launches_20261103.json').read_text(encoding='utf-8'))
print('\n== 発進ごと')
for r in j['launches']:
    print('  %s gain=%s launch_state=%s to_r1=%s fly_ref=%s fly_n=%s' % (r['log'], r['gain'], r.get('launch_state'), r.get('to_r1_s'),
                                                                     r.get('fly_ref'), r.get('fly_n')))
S = j['summary']
both = [g for g in S if 'on' in g['by_launch'] and 'off' in g['by_launch']]
print('\n== まとめ: 組の数 %d・ON と OFF の両方がある組 %d・compare のある組 %d' % (len(S), len(both), sum('compare' in g for g in S)))
for g in S:
    print('  gain=%s by_launch=%s compare=%s' % (g['gain'], sorted(g['by_launch']), 'compare' in g))
