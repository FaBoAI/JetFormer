"""(fix7 tools r2 が写した。before|after と作り物・出力の場所だけ変えた) 反証 7-2 回目・係 tools: fix7 の「リカバリー中止のあとの走り直し」(abort_resumes)は、区間の最後のフレームが assist 3 のときだけ
印を付ける。手順書 7 章の中立止めの行(「CH3: AI側 ・ 中立で停止中」が出たら **車に近づく前に CH3 を人側**)のとおりに、
中立止めの **途中で** CH3 を人側にすると、drive は中立止めの間 helper.step を呼ばないので assist 3 のフレームは 1 つも書かれない
(中止は明けた最初のフレームで決まる)。その場合、コースの途中で止まった車をその場から AI 側に戻した発進(fix7 r1 の指摘と同じ形)
は、⑤ ③ ④ でどう読まれるか。

作り物の回 3(fix7/tools_r1/probes/probe_abort_resume.py と同じ。1 周 18 m・2 m/s。置いた場所は R1 の起点の 5 m 手前):
  OFF 3-1・3-3・3-5・3-7  起点まで 3.013・3.023・3.033・3.043 秒
  ON  3-2・3-4・3-6       2.813・2.863・2.943 秒
  ON  3-8                 1 回目の発進 2.973 秒。2 周目の R1 の (sb + 1) m 手前で止まり始め、R1 の sb m 手前で止まる。
    B_abort  fix7 r1 の B そのもの(カメラ 0.8 秒止まる → リカバリー中止 assist 3 → 人側 1.0 秒 → その場で AI 側)。対照
    H_hold   中立止め(AI_CMD の hold = True のフレーム。処理の遅れ・車両の値が古いなど)が止まり始めから CH3 を人側にするまで続く
             → 人側 1.0 秒 → その場で AI 側(assist 3 のフレームは無い)
    S_stall  カメラが止まり始めから CH3 を人側にした 0.5 秒あとまで止まる(フレームが無い。人側のうちに戻る)→ 人側 → その場で AI 側
  sb = 0.5・1.5・2.5・3.5・4.5 m。どれも当たり 0。
出力は ../out/probe_hold_resume.json。作り物は fix7/tmp_review_tools_r2/hold_resume の下に作って最後に消す。本物は読まない。
"""
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
MODE = sys.argv[1] if len(sys.argv) > 1 else 'after'
assert MODE in ('before', 'after')
TMP = JF / ('experiments/20261026_fix/fix7/tmp_tools_r2/probe_hold_%s' % MODE) / 'hold_resume'
for p in (JF, JF / 'tools', JF / 'tests', HERE):
    sys.path.insert(0, str(p))
import gen_course2 as gc  # noqa: E402
from test_fix3_r1_2 import load_mod  # noqa: E402
BEFORE = JF / 'tests/fixtures/fix7_tools_before/r2'
if MODE == 'before':                 # 直す前(fix7 r1 の版)の写しを先に読む
    sys.modules['standing_3lap'] = load_mod('standing_3lap', BEFORE / 'standing_3lap.py.orig')
    sys.modules['quick_eval_by_params'] = load_mod('quick_eval_by_params', BEFORE / 'quick_eval_by_params.py.orig')
import standing_3lap as s3  # noqa: E402
import quick_eval_by_params as qp  # noqa: E402

COLLECT = (BEFORE / 'collect.py.orig') if MODE == 'before' else (JF / 'experiments/20261026_fix/diag_launch/collect.py')
OUT = HERE.parent / 'out' / ('probe_hold_resume_%s.json' % MODE)
PRM = {'throttle_gain': 0.9, 'steer_gain': 1.0, 'speed_ceiling': 0, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 0,
       'auto_target_mps': 4.27, 'launch_boost': 0, 'launch_us': 250.0, 'launch_mps': 2.5, 'launch_max_s': 1.2, 'zone_caps': 0,
       'start_kick': 1, 'rec_back_guard': 1, 'rec_hold_keep': 1, 'launch_no_kick': 0}
ON = dict(PRM, launch_boost=1)
DESC = 'ai_no=%d bench=real_bench_20261023 day=2026-11-03 block=%d cand=EXP model=runB memo=x'
SBS = (0.5, 1.5, 2.5, 3.5, 4.5)
CASES = [('A_no_resume', 'none', 0.5)] + [('%s_%.1fm' % (k, sb), k, sb) for k in ('B_abort', 'H_hold', 'S_stall') for sb in SBS]


def day(root, kind, sb):
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
    base = gc.launch_keys(1.0, 0.31, 0.3)
    t_st = gc.time_at_distance(base + [(200.0, 2.0)], 23.0 - sb - 1.0, 200.0)
    t_stop = t_st + 1.0
    t_flip = t_stop + 2.0
    t_res = t_flip + 1.0
    kw = {}
    if kind == 'none':
        dur = t_flip + 3.0
        vk = base + [(t_st, 2.0), (t_stop, 0.0), (dur, 0.0)]
        ai = [(1.0, t_flip)]
        launch = [(1.0, 0.5)]
        kw = dict(stall=[(t_st, t_st + 0.8)], giveup=[(t_st + 0.8, t_flip)])
    else:
        dur = t_res + 32.0
        vk = base + [(t_st, 2.0), (t_stop, 0.0), (t_res + 0.15, 0.0), (t_res + 0.45, 2.0), (dur, 2.0)]
        ai = [(1.0, t_flip), (t_res, dur - 1.0)]
        launch = [(1.0, 0.5), (t_res, 0.5)]
        if kind == 'B_abort':
            kw = dict(stall=[(t_st, t_st + 0.8)], giveup=[(t_st + 0.8, t_flip)],
                      events=[{'kind': 'recovery_guard', 'what': 'hold_long', 'phase': None, 'backed_s': 0.0, 'hold_s': 0.8,
                               'run': 'runB', 'cam_epoch_us': int(gc.T0_US + round((t_st + 0.8) * 1e6)),
                               't_us': int(gc.T0_US + round((t_st + 0.8) * 1e6))}])
        elif kind == 'H_hold':
            kw = dict(hold=[(t_st, t_flip)])
        elif kind == 'S_stall':
            kw = dict(stall=[(t_st, t_flip + 0.5)])
    log38 = add(19, ON, 0.31, 0.3, dur=dur, vk=vk, ai=ai, launch=launch, **kw)
    return raw, bench, log38, {'t_st': round(t_st, 3), 't_stop': round(t_stop, 3), 't_flip': round(t_flip, 3),
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
        for case, kind, sb in CASES:
            root = TMP / case
            raw, bench, log38, tt = day(root, kind, sb)
            r = {'times_3_8': tt}
            for nm, args in (('ig', ['--ignore-gain']), ('ig_first', ['--ignore-gain', '--first-only'])):
                j, o = run5(root, raw, bench, '%s_%s' % (case, nm), args)
                g = j['summary'][0]
                r['collect_' + nm] = {
                    'on_n': g['by_launch']['on']['n'], 'on_to_r1_median': g['by_launch']['on']['to_r1_median_s'],
                    'off_to_r1_median': g['by_launch']['off']['to_r1_median_s'], 'restarts_on': g['by_launch']['on']['restarts'],
                    'compare': g.get('compare'), 'abort_resume_excluded': g.get('abort_resume_excluded'),
                    'rows_3_8': [{k: x.get(k) for k in ('t_ai', 'start_no', 'to_r1_s', 'dist_m', 'launch_state', 'start_loc',
                                                        'after_abort')} for x in j['launches'] if x['log'] == log38],
                    'lines': [x for x in o.splitlines() if 'ON − OFF' in x or 'リカバリー中止のあと' in x]}
            logs, _ = s3.day_logs('20261103', raw)
            R = s3.run(logs, raw, root / 'nolab', None)
            r38 = [x for x in R if x['log'] == log38][0]
            r['standing_3_8'] = [{k: s.get(k) for k in ('start_no', 't_ai', 'valid', 'why', 'counted', 'coll_before', 'from_move_s',
                                                        'd_pre_m', 'after_abort')} for s in r38['standing']]
            md = s3.to_markdown(R)
            r['standing_md_3_8'] = [x for x in md.splitlines() if log38 in x]
            qmd, runs, groups = qp.to_markdown(R, '20261103', None, ('gain',))
            r['by_params_groups'] = [{k: gq.get(k) for k in ('text', 'runs', 'standing_n', 'standing_best_s', 'standing_list_s',
                                                             'standing_excluded_n', 'launch_to_r1_median_s',
                                                             'launch_to_r1_abort_excluded')} for gq in groups]
            res[case] = r
    finally:
        shutil.rmtree(TMP.parent, ignore_errors=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for case, r in res.items():
        c, cf = r['collect_ig'], r['collect_ig_first']
        cmp_ = c['compare'] or {}
        s2 = [s for s in r['standing_3_8'] if s['start_no'] == 2]
        g_on = [g for g in r['by_params_groups'] if '発進 250' in g['text']]
        print('%-16s ⑤ig on n=%d on med %s d=%s all_ok=%s | ⑤ig_first all_ok=%s | ③ 2nd counted=%s from_move=%s after_abort=%s | '
              '④ ON 止まってから3周 %s 数えない %s 動く→A1 %s' % (
                  case, c['on_n'], c['on_to_r1_median'], cmp_.get('d_to_r1_s'), cmp_.get('all_ok'),
                  (cf['compare'] or {}).get('all_ok'), s2[0]['counted'] if s2 else None, s2[0]['from_move_s'] if s2 else None,
                  bool(s2 and s2[0]['after_abort']), g_on[0]['standing_list_s'] if g_on else None,
                  g_on[0]['standing_excluded_n'] if g_on else None, g_on[0]['launch_to_r1_median_s'] if g_on else None))
        for x in c['lines']:
            print('      ', x[:220])


if __name__ == '__main__':
    main()
