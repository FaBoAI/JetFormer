"""反証 7-1 回目・係 tools: 両方 ON の日(rec_back_guard = rec_hold_keep = 1)の「リカバリー中止」のあと、その場で CH3 を人側 → AI 側に
戻して走り直した本(fix3/approvals.md 3「困ること(ON): 壁の無い所で止まりかけのときにカメラが 0.5 秒より長く止まると、中止で車が
止まる(人が CH3 で戻す)」)を、⑤ collect.py・③ standing_3lap・④ quick_eval_by_params はどう読むか。

作り物の回 3(gen_course。1 周 18 m・2 m/s。置いた場所は R1 の起点の 5 m 手前):
  OFF 3-1・3-3・3-5・3-7  起点まで 3.013・3.023・3.033・3.043 秒
  ON  3-2・3-4・3-6       2.813・2.863・2.943 秒
  ON  3-8                 1 回目の発進 2.973 秒。2 周目の R1 の 1.5 m 手前(LH)でカメラが 0.8 秒止まり(フレームなし)、中立の惰性で
                          R1 の起点の 0.5 m 手前に止まる → リカバリー中止(assist 3・中立。events.jsonl の recovery_guard hold_long)
                          → 2 秒後に人が CH3 を人側 1.0 秒 → その場で AI 側(発進の前進がまた出る)→ 3 周走る
  A  3-8 は中止のあと走り直さない(人側にして止めた)
  B  3-8 は上のとおり、その場で走り直した
どちらも当たり 0。出力は ../out/probe_abort_resume.json。作り物は fix7/tmp_review_tools_r1 の下に作って最後に消す。本物は読まない。
"""
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
import quick_eval_by_params as qp  # noqa: E402
from test_fix3_r1_2 import load_mod  # noqa: E402

TMP = JF / 'experiments/20261026_fix/fix7/tmp_review_tools_r1/abort_resume'
CASES = [('A_no_resume', False, 0.5)] + [('B_resume_%.1fm' % x, True, x) for x in (0.5, 1.5, 2.5, 3.5, 4.5)]
COLLECT = JF / 'experiments/20261026_fix/diag_launch/collect.py'
OUT = HERE.parent / 'out' / 'probe_abort_resume.json'
PRM = {'throttle_gain': 0.9, 'steer_gain': 1.0, 'speed_ceiling': 0, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 0,
       'auto_target_mps': 4.27, 'launch_boost': 0, 'launch_us': 250.0, 'launch_mps': 2.5, 'launch_max_s': 1.2, 'zone_caps': 0,
       'start_kick': 1, 'rec_back_guard': 1, 'rec_hold_keep': 1, 'launch_no_kick': 0}
ON = dict(PRM, launch_boost=1)
DESC = 'ai_no=%d bench=real_bench_20261023 day=2026-11-03 block=%d cand=EXP model=runB memo=x'


def day(root, resume, stop_before_r1=0.5):
    raw = root / 'raw'
    raw.mkdir(parents=True)
    bench = root / 'bench.json'
    bench.write_text(json.dumps({'candidates': [{'id': 'B', 'run': 'runB'}],
                                 'days': [{'date': '2026-11-03', 'blocks': [{'block': b, 'run_id': '3-%d' % (b - 11)}
                                                                            for b in range(12, 20)]}]}), encoding='utf-8')
    n = [0]

    def add(block, prm, d_move, ramp, **kw):
        log = '20261103-13%02d00' % (n[0] * 5)
        n[0] += 1
        dur = kw.pop('dur', 30.0)
        vk = kw.pop('vk', None) or gc.launch_keys(1.0, d_move, ramp, until=dur)
        ai = kw.pop('ai', None) or [(1.0, dur - 2.0)]
        launch = kw.pop('launch', [(1.0, 0.5)] if prm.get('launch_boost') else [])
        gc.write_log(raw, log, dur, vk, ai, prm, DESC % (n[0], block), launch=launch, **kw)
        return log
    add(12, PRM, 0.20, 0.6)
    add(13, ON, 0.15, 0.3)
    add(14, PRM, 0.21, 0.6)
    add(15, ON, 0.20, 0.3)
    add(16, PRM, 0.22, 0.6)
    add(17, ON, 0.28, 0.3)
    add(18, PRM, 0.23, 0.6)
    # 3-8: 1 回目の発進(1.0 秒・動くのは 1.31 秒・0.3 秒で 2 m/s)。2 周目の R1(23 m)の 1.5 m 手前 = 21.5 m でカメラが止まる
    base = gc.launch_keys(1.0, 0.31, 0.3)
    t_st = gc.time_at_distance(base + [(200.0, 2.0)], 23.0 - stop_before_r1 - 1.0, 200.0)
    t_stop = t_st + 1.0                       # 2 m/s から 2 m/s² で 1.0 秒・1.0 m(R1 の起点の stop_before_r1 m 手前)で止まる
    t_flip = t_stop + 2.0                     # 中止(中立で人を待つ)を 2 秒
    t_res = t_flip + 1.0                      # 人側 1.0 秒 → その場で AI 側
    if resume:
        dur = t_res + 32.0
        vk = base + [(t_st, 2.0), (t_stop, 0.0), (t_res + 0.15, 0.0), (t_res + 0.45, 2.0), (dur, 2.0)]
        ai = [(1.0, t_flip), (t_res, dur - 1.0)]
        launch = [(1.0, 0.5), (t_res, 0.5)]
    else:
        dur = t_flip + 3.0
        vk = base + [(t_st, 2.0), (t_stop, 0.0), (dur, 0.0)]
        ai = [(1.0, t_flip)]
        launch = [(1.0, 0.5)]
    ev = [{'kind': 'recovery_guard', 'what': 'hold_long', 'phase': None, 'backed_s': 0.0, 'hold_s': 0.8, 'run': 'runB',
           'cam_epoch_us': int(gc.T0_US + round((t_st + 0.8) * 1e6)), 't_us': int(gc.T0_US + round((t_st + 0.8) * 1e6))}]
    log38 = add(19, ON, 0.31, 0.3, dur=dur, vk=vk, ai=ai, launch=launch, stall=[(t_st, t_st + 0.8)],
                giveup=[(t_st + 0.8, t_flip)], events=ev)
    return raw, bench, log38, {'t_stall': round(t_st, 3), 't_stop': round(t_stop, 3), 't_flip': round(t_flip, 3),
                               't_resume': round(t_res, 3)}


def run5(root, raw, bench, name, args):
    m = load_mod(name, COLLECT)
    m.RAW, m.LAB, m.BENCH, m.OUT = raw, root / 'nolab', bench, root / 'out'
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.main(['--day', '20261103', '--out', str(root / (name + '.json'))] + list(args))
    return json.loads((root / (name + '.json')).read_text()), buf.getvalue()


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    res = {}
    try:
        for case, resume, sb in CASES:
            root = TMP / case
            raw, bench, log38, tt = day(root, resume, sb)
            tt['stop_before_r1_m'] = sb
            r = {'times_3_8': tt}
            for nm, args in (('ig', ['--ignore-gain']), ('ig_first', ['--ignore-gain', '--first-only'])):
                j, o = run5(root, raw, bench, '%s_%s' % (case, nm), args)
                g = j['summary'][0]
                r['collect_' + nm] = {
                    'on_n': g['by_launch']['on']['n'], 'on_logs': g['by_launch']['on']['logs'],
                    'on_to_r1_median': g['by_launch']['on']['to_r1_median_s'], 'off_to_r1_median': g['by_launch']['off']['to_r1_median_s'],
                    'restarts_on': g['by_launch']['on']['restarts'], 'compare': g.get('compare'),
                    'rows_3_8': [{k: x.get(k) for k in ('t_ai', 'start_no', 'to_r1_s', 'dist_m', 'launch_state', 'start_loc',
                                                        'collisions_first_lap', 'hits_before')}
                                 for x in j['launches'] if x['log'] == log38],
                    'lines': [x for x in o.splitlines() if 'ON − OFF' in x or x.startswith('  on ') or x.startswith('  off ')]}
            logs, _ = s3.day_logs('20261103', raw)
            R = s3.run(logs, raw, root / 'nolab', None)
            r38 = [x for x in R if x['log'] == log38][0]
            r['standing_3_8'] = [{k: s.get(k) for k in ('start_no', 't_ai', 'valid', 'why', 'counted', 'coll_before', 'from_move_s',
                                                        'd_pre_m', 'start_frac_of_lap', 'note')} for s in r38['standing']]
            md = s3.to_markdown(R)
            r['standing_md_3_8'] = [x for x in md.splitlines() if log38 in x]
            qmd, runs, groups = qp.to_markdown(R, '20261103', None, ('gain',))
            r['by_params_groups'] = [{k: gq.get(k) for k in ('text', 'runs', 'standing_n', 'standing_best_s', 'standing_le25',
                                                             'standing_clean_le25', 'standing_clean_best_s', 'standing_list_s')}
                                     for gq in groups]
            res[case] = r
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for case, r in res.items():
        print('==', case, r['times_3_8'])
        for nm in ('ig', 'ig_first'):
            c = r['collect_' + nm]
            print('  ⑤ %-8s on n=%d restarts=%d on median %s off median %s → all_ok %s %s' % (
                nm, c['on_n'], c['restarts_on'], c['on_to_r1_median'], c['off_to_r1_median'], (c['compare'] or {}).get('all_ok'),
                (c['compare'] or {}).get('checks')))
            for x in c['rows_3_8']:
                print('     3-8 row', x)
        for s in r['standing_3_8']:
            print('  ③ 3-8', s)
        for g in r['by_params_groups']:
            print('  ④', g)


if __name__ == '__main__':
    main()
