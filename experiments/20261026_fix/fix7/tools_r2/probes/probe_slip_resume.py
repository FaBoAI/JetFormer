"""(fix7 tools r2 が写した。今・r1・fix7 の前の 3 つで流す。作り物の場所だけ変えた) 反証 7-2 回目・係 tools: fix7 r1 で ⑤ collect.py は「R1 に届いたリカバリー中止のあとの走り直し」を比べ(起点まで・失う・動く・
**空転**・LH)から外した。走り直しの ON の発進が空転した(比 > 1.3)とき、⑤ の「空転 ≤ 1.3」と、car_check の「発進の空転の比」の行
(「→ もう 200 µs 以下。⑤ collect.py は空転の条件を満たさない(回 4 は発進の前進 OFF)」)はどう出るか。直す前(fix7 の前の
tests/fixtures/fix7_tools_before/collect.py.orig)とも比べる。

作り物の回 3(gen_course2。1 周 18 m・2 m/s。置いた場所は R1 の起点の 5 m 手前。発進の前進 ON は 4 本とも launch_us 200):
  OFF 3-1・3-3・3-5・3-7  起点まで 3.013・3.023・3.033・3.043 秒
  ON  3-2・3-4・3-6       2.813・2.863・2.943 秒
  ON  3-8                 1 回目の発進 2.813 秒(d_move 0.15)。ここまでで ON の中央値 2.838 秒 = OFF − 0.190 秒
  3-8 の 2 周目に、置いた場所と同じ所(R1 の 5 m 手前 = HS。手順書のとおり置き直した形と同じ場所)で止まる:
    K0  走り直さない(対照)
    K1  カメラ 0.8 秒 → リカバリー中止(assist 3)→ 人側 1.0 秒 → AI 側(発進の前進がまた出る)。走り直しの発進で空転
        (動いてから 1.0 秒の IMU の前の加速を 0.55 倍 = 車輪 ÷ IMU ≈ 1.8)
    K2  K1 と同じで空転なし(対照)
    K3  K1 と同じ空転で、中止でなくカメラが止まったまま人側にした形(S_stall。abort_resumes は印を付けない)
    K4  K2(空転なし)で、3-8 のシーン 0002(20〜40 秒。走り直しの最初の周の後ろ半分)の IMU.npz が無い(記録が欠けた形)
    K5  K4 と同じ欠けで、中止でなくカメラが止まったまま人側にした形(対照: 走り直しが ON の組に入る)
出力は ../out/probe_slip_resume.json。作り物は fix7/tmp_review_tools_r2/slip_resume の下に作って最後に消す。本物は読まない。
"""
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
TMP = JF / 'experiments/20261026_fix/fix7/tmp_tools_r2/probe_slip' / 'slip_resume'
for p in (JF, JF / 'tools', JF / 'tests', HERE):
    sys.path.insert(0, str(p))
import gen_course2 as gc  # noqa: E402
import car_check as cc  # noqa: E402
from test_fix3_r1_2 import load_mod  # noqa: E402

COLLECT = {'now': JF / 'experiments/20261026_fix/diag_launch/collect.py',
           'r1': JF / 'tests/fixtures/fix7_tools_before/r2/collect.py.orig',
           'before_fix7': JF / 'tests/fixtures/fix7_tools_before/collect.py.orig'}
OUT = HERE.parent / 'out' / 'probe_slip_resume.json'
PRM = {'throttle_gain': 0.9, 'steer_gain': 1.0, 'speed_ceiling': 0, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 0,
       'auto_target_mps': 4.27, 'launch_boost': 0, 'launch_us': 200.0, 'launch_mps': 2.5, 'launch_max_s': 1.2, 'zone_caps': 0,
       'start_kick': 1, 'rec_back_guard': 1, 'rec_hold_keep': 1, 'launch_no_kick': 0}
ON = dict(PRM, launch_boost=1)
DESC = 'ai_no=%d bench=real_bench_20261023 day=2026-11-03 block=%d cand=EXP model=runB memo=x'
SB = 5.0
CASES = [('K0_no_resume', 'none', False), ('K1_abort_spin', 'abort', True), ('K2_abort_nospin', 'abort', False),
         ('K3_stall_spin', 'stall', True), ('K4_abort_gap', 'abort', 'gap'), ('K5_stall_gap', 'stall', 'gap')]


def day(root, kind, spin):
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
    base = gc.launch_keys(1.0, 0.15, 0.3)
    t_st = gc.time_at_distance(base + [(200.0, 2.0)], 23.0 - SB - 1.0, 200.0)
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
        if kind == 'abort':
            kw = dict(stall=[(t_st, t_st + 0.8)], giveup=[(t_st + 0.8, t_flip)],
                      events=[{'kind': 'recovery_guard', 'what': 'hold_long', 'phase': None, 'backed_s': 0.0, 'hold_s': 0.8,
                               'run': 'runB', 'cam_epoch_us': int(gc.T0_US + round((t_st + 0.8) * 1e6)),
                               't_us': int(gc.T0_US + round((t_st + 0.8) * 1e6))}])
        else:
            kw = dict(stall=[(t_st, t_flip + 0.5)])
        if spin == 'gap':
            kw['drop'] = [(2, 'IMU')]
        elif spin:
            kw['imu_scale'] = [(t_res, t_res + 1.2, 0.55)]
    log38 = add(19, ON, 0.15, 0.3, dur=dur, vk=vk, ai=ai, launch=launch, **kw)
    return raw, bench, log38, {'t_st': round(t_st, 3), 't_stop': round(t_stop, 3), 't_flip': round(t_flip, 3), 't_resume': round(t_res, 3)}


def run5(root, raw, bench, name, path, args):
    m = load_mod(name, path)
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
        for case, kind, spin in CASES:
            root = TMP / case
            raw, bench, log38, tt = day(root, kind, spin)
            r = {'times_3_8': tt}
            for tag, path in COLLECT.items():
                j, o = run5(root, raw, bench, '%s_%s' % (case, tag), path, ['--ignore-gain'])
                g = j['summary'][0]
                r['collect_' + tag] = {
                    'on_n': g['by_launch']['on']['n'], 'on_slip_max': g['by_launch']['on']['slip_max'],
                    'compare': g.get('compare'), 'abort_resume_excluded': g.get('abort_resume_excluded'),
                    'first_lap_gap_on': g.get('first_lap_gap_on'),
                    'rows_3_8': [{k: x.get(k) for k in ('t_ai', 'start_no', 'to_r1_s', 'launch_state', 'after_abort', 'first_lap_gap')}
                                 | {'slip_ratio': (x.get('slip') or {}).get('ratio')} for x in j['launches'] if x['log'] == log38],
                    'lines': [x for x in o.splitlines() if 'ON − OFF' in x or 'リカバリー中止のあと' in x or '空転' in x
                              or '記録が欠けている本' in x or '最初の周の記録が欠けた' in x]}
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cc.main([log38, '--raw', str(raw), '--labels', str(root / 'nolab'), '--no-json'])
            r['car_check_3_8'] = {'rc': rc, 'slip_lines': [x for x in buf.getvalue().splitlines() if '空転' in x or '200 µs' in x]}
            res[case] = r
    finally:
        shutil.rmtree(TMP.parent, ignore_errors=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for case, r in res.items():
        print('==', case, r['times_3_8'])
        for tag in COLLECT:
            c = r['collect_' + tag]
            cmp_ = c['compare'] or {}
            print('  ⑤ %-11s on n=%d slip_max %s d_to %s → %s %s' % (tag, c['on_n'], c['on_slip_max'], cmp_.get('d_to_r1_s'),
                                                                 '使ってよい' if cmp_.get('all_ok') else '満たさない', cmp_.get('checks')))
            for x in c['rows_3_8']:
                print('      3-8', x)
            if case.endswith('gap'):
                for x in c['lines']:
                    print('      |', x[:330])
        for x in r['car_check_3_8']['slip_lines']:
            print('  car_check', x[:300])


if __name__ == '__main__':
    main()
