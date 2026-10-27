"""gen_course の作り物が道具に正しく読まれるかの確かめ(③ standing_3lap・⑤ collect・car_check)。作り物は fix7/tmp_review_tools_r1 の下"""
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
for p in (JF, JF / 'tools', JF / 'tests', HERE):
    sys.path.insert(0, str(p))
import gen_course as gc  # noqa: E402
import standing_3lap as s3  # noqa: E402
import car_check as cc  # noqa: E402
from test_fix3_r1_2 import load_mod  # noqa: E402

TMP = JF / 'experiments/20261026_fix/fix7/tmp_review_tools_r1/sanity'
COLLECT = JF / 'experiments/20261026_fix/diag_launch/collect.py'
PRM = {'throttle_gain': 0.9, 'steer_gain': 1.0, 'speed_ceiling': 0, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 0,
       'auto_target_mps': 4.27, 'launch_boost': 0, 'launch_us': 250.0, 'launch_mps': 2.5, 'launch_max_s': 1.2, 'zone_caps': 0,
       'start_kick': 1, 'rec_back_guard': 1, 'rec_hold_keep': 1, 'launch_no_kick': 0}
DESC = 'ai_no=%d bench=real_bench_20261023 day=2026-11-03 block=%d cand=EXP model=runB memo=x'


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    raw = TMP / 'raw'
    raw.mkdir(parents=True)
    bench = TMP / 'bench.json'
    bench.write_text(json.dumps({'candidates': [{'id': 'B', 'run': 'runB'}],
                                 'days': [{'date': '2026-11-03', 'blocks': [{'block': b, 'run_id': '3-%d' % (b - 11)}
                                                                            for b in range(12, 20)]}]}), encoding='utf-8')
    dur = 40.0
    vk = gc.launch_keys(1.0, 0.2, 0.6, until=dur)
    gc.write_log(raw, '20261103-130000', dur, vk, [(1.0, 30.0)], PRM, DESC % (1, 12))
    vk2 = gc.launch_keys(1.0, 0.15, 0.3, until=dur)
    gc.write_log(raw, '20261103-130500', dur, vk2, [(1.0, 30.0)], dict(PRM, launch_boost=1), DESC % (2, 13), launch=[(1.0, 0.5)])
    for lg in ('20261103-130000', '20261103-130500'):
        r = s3.analyze_log(lg, raw, TMP / 'nolab', {'runB': 'B'})
        print(lg, 'anchors', r.get('r1_anchors'), [(s.get('t_ai'), s.get('to_r1_s'), s.get('from_move_s'), s.get('valid'), s.get('why'))
                                                     for s in r['standing']], r['settings_text'])
        c = cc.check_log(lg, TMP / 'nolab', raw)
        print('\n'.join(x for x in cc.to_text(c).splitlines() if '試験' in x or '衝突' in x or '判定' in x or '空転' in x))
    m = load_mod('col_sanity', COLLECT)
    m.RAW, m.LAB, m.BENCH, m.OUT = raw, TMP / 'nolab', bench, TMP / 'out'
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.main(['--day', '20261103', '--out', str(TMP / 'c.json'), '--ignore-gain'])
    j = json.loads((TMP / 'c.json').read_text())
    for r in j['launches']:
        print(r['log'], r['t_ai'], 'to_r1', r.get('to_r1_s'), r.get('launch_state'), 'slip', (r.get('slip') or {}).get('ratio'),
              'lh', r.get('v_launch_at_lh'), r.get('_lh_self'), r.get('lh_self'), 'fl', r.get('collisions_first_lap'), r.get('first_lap_end'))
    print('\n'.join(x for x in buf.getvalue().splitlines() if 'ON − OFF' in x or x.startswith('11/03')))
    shutil.rmtree(TMP)


if __name__ == '__main__':
    main()
