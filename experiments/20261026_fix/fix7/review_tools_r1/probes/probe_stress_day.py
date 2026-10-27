"""反証 7-1 回目・係 tools: 両方 ON の 11/3 を思わせる作り物の 1 日(回 3 の 8 本 + 回 4 の 3 本)に、落ちた本・試し・途中で見張りを
OFF・launch_us 200・人側にした直後の当たり・リカバリー中止のあと置き直し、を混ぜて、③ ④ ⑤ car_check を全部の引数で流し、落ちない
(例外が出ない)か・出力が読めるかを見る。出力 ../out/probe_stress_day.json。作り物は fix7/tmp_review_tools_r1 の下(最後に消す)"""
import contextlib
import io
import json
import shutil
import sys
import traceback
from pathlib import Path

JF = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
for p in (JF, JF / 'tools', JF / 'tests', HERE):
    sys.path.insert(0, str(p))
import gen_course as gc  # noqa: E402
import standing_3lap as s3  # noqa: E402
import quick_eval_by_params as qp  # noqa: E402
import car_check as cc  # noqa: E402
from test_fix3_r1_2 import load_mod  # noqa: E402

TMP = JF / 'experiments/20261026_fix/fix7/tmp_review_tools_r1/stress'
COLLECT = JF / 'experiments/20261026_fix/diag_launch/collect.py'
OUT = HERE.parent / 'out' / 'probe_stress_day.json'
PRM = {'throttle_gain': 0.92, 'steer_gain': 1.0, 'speed_ceiling': 1, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 0,
       'auto_target_mps': 4.27, 'launch_boost': 0, 'launch_us': 250.0, 'launch_mps': 2.5, 'launch_max_s': 1.2, 'zone_caps': 0,
       'start_kick': 1, 'rec_back_guard': 1, 'rec_hold_keep': 1, 'launch_no_kick': 0, 'steer_trim_us': 0.0}
ON = dict(PRM, launch_boost=1)
DESC = 'ai_no=%d bench=real_bench_20261023 day=2026-11-03 block=%d cand=EXP model=runB memo=計画と違うので試し(元 cand=B): 速度 0.92'


def build(root):
    raw = root / 'raw'
    raw.mkdir(parents=True)
    bench = root / 'bench.json'
    blocks = [{'block': b, 'run_id': '3-%d' % (b - 11), 'expect': {'launch_boost': 1 if b % 2 else 0}} for b in range(12, 20)]
    blocks += [{'block': b, 'run_id': '4-%d' % (b - 19), 'expect': {'launch_boost': None}} for b in range(20, 24)]
    bench.write_text(json.dumps({'candidates': [{'id': 'B', 'run': 'runB'}], 'days': [{'date': '2026-11-03', 'blocks': blocks}]}),
                     encoding='utf-8')
    n = [0]
    names = {}

    def add(tag, block, prm, d_move, ramp, dur=30.0, **kw):
        log = '20261103-13%02d00' % n[0] if n[0] < 60 else '20261103-14%02d00' % (n[0] - 60)
        n[0] += 5
        vk = kw.pop('vk', None) or gc.launch_keys(1.0, d_move, ramp, until=dur)
        ai = kw.pop('ai', None) or [(1.0, dur - 2.0)]
        launch = kw.pop('launch', [(1.0, 0.5)] if prm.get('launch_boost') else [])
        desc = kw.pop('desc', None) or DESC % (len(names) + 1, block)
        gc.write_log(raw, log, dur, vk, ai, prm, desc, launch=launch, **kw)
        names[tag] = log
        return log
    add('3-1', 12, PRM, 0.20, 0.6)
    add('3-2', 13, ON, 0.15, 0.3, hits=[(2.2, 3.0)])                    # ON の最初の周に 3G(満たさない)
    add('3-3', 14, PRM, 0.21, 0.6)
    lg = add('3-4', 15, dict(ON, launch_us=200.0), 0.20, 0.3)           # 空転で 200 に下げた
    add('3-5', 16, PRM, 0.22, 0.6, dur=30.0, ai=[(1.0, 12.0)], hits=[(12.3, 9.0)])   # 人側にした直後の 9G
    # 3-6: 試し(走り終わりのメモ)
    lg6 = add('3-6', 17, dict(ON, launch_us=200.0), 0.28, 0.3)
    for sc in sorted(p for p in (raw / lg6).iterdir() if p.is_dir()):
        (sc / 'scene.json').write_text(json.dumps({'description': DESC % (6, 17) + ' / 試し(ゲインを入れ忘れた)'}), encoding='utf-8')
    # 3-7: 走行中に見張りを OFF(残すも OFF)
    lg7 = add('3-7', 18, PRM, 0.23, 0.6, events=[{'kind': 'params', 'changed': {'rec_back_guard': [1, 0], 'rec_hold_keep': [1, 0]},
                                                    'epoch_us': gc.T0_US + 15_000_000}])
    # 3-8: ON。2 回目の発進(置き直し 40 秒)の最初の周で電源断(0003 は CAM_FRONT/ だけ、0002 は 0 バイト)
    vk8 = gc.launch_keys(1.0, 0.25, 0.3, until=15.0) + [(16.0, 0.0), (40.25, 0.0), (40.55, 2.0), (60.0, 2.0)]
    lg8 = add('3-8', 19, dict(ON, launch_us=200.0), 0.25, 0.3, dur=60.0, vk=vk8, ai=[(1.0, 15.0), (40.0, 58.0)],
              launch=[(1.0, 0.5), (40.0, 0.5)])
    d8 = raw / lg8
    for p in (d8 / ('%s-0003' % lg8)).glob('*.npz'):
        p.unlink()
    (d8 / ('%s-0003' % lg8) / 'CAM_FRONT').mkdir()
    for p in list((d8 / ('%s-0002' % lg8)).glob('*.npz')) + [d8 / ('%s-0002' % lg8) / 'scene.json']:
        if p.exists():
            p.write_bytes(b'')
    ev = (d8 / 'events.jsonl').read_text().splitlines()
    (d8 / 'events.jsonl').write_text(ev[0] + '\n', encoding='utf-8')     # 40 秒の start は電源断で消えた
    # 回 4: 止まって 3 周(ON 200)。4-2 はリカバリー中止のあと HS に置き直して 2 回目
    for i, b in enumerate((20, 21, 22)):
        if b == 21:
            vk = gc.launch_keys(1.0, 0.2, 0.3, until=12.0) + [(13.0, 0.0), (20.2, 0.0), (20.5, 2.0), (60.0, 2.0)]
            add('4-%d' % (i + 1), b, dict(ON, launch_us=200.0), 0.2, 0.3, dur=60.0, vk=vk, ai=[(1.0, 17.0), (20.0, 58.0)],
                launch=[(1.0, 0.5), (20.0, 0.5)], giveup=[(13.0, 17.0)], stall=[(12.0, 12.8)],
                events=[{'kind': 'recovery_guard', 'what': 'hold_long', 'phase': None, 'backed_s': 0.0, 'hold_s': 0.8,
                         'cam_epoch_us': gc.T0_US + 12_800_000, 't_us': gc.T0_US + 12_800_000}])
        else:
            add('4-%d' % (i + 1), b, dict(ON, launch_us=200.0), 0.2, 0.3, dur=40.0)
    return raw, bench, names


def capture(fn, *a, **k):
    buf, err = io.StringIO(), None
    try:
        with contextlib.redirect_stdout(buf):
            r = fn(*a, **k)
    except Exception:          # noqa: BLE001
        err = traceback.format_exc()
        r = None
    return r, buf.getvalue(), err


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    out = {'errors': [], 'collect': {}, 'car_check': {}}
    try:
        raw, bench, names = build(TMP)
        out['names'] = names
        for nm, args in (('plain', []), ('ig', ['--ignore-gain']), ('ig_first', ['--ignore-gain', '--first-only']),
                         ('first', ['--first-only']), ('ig_skip', ['--ignore-gain', '--skip', names['3-8']])):
            m = load_mod('col_stress_' + nm, COLLECT)
            m.RAW, m.LAB, m.BENCH, m.OUT = raw, TMP / 'nolab', bench, TMP / 'out'
            o = TMP / ('c_%s.json' % nm)
            _, so, err = capture(m.main, ['--day', '20261103', '--out', str(o)] + args)
            if err:
                out['errors'].append(('collect ' + nm, err))
                continue
            j = json.loads(o.read_text())
            out['collect'][nm] = {'summary_lines': [x for x in so.splitlines() if x.startswith(('11/03', '  ON − OFF', '  注意', '  読めない',
                                                                                              '  試し', '  途中', '  on', '  off', '  →'))],
                                  'all_ok': [(g['session'], (g.get('compare') or {}).get('all_ok'), (g.get('compare') or {}).get('checks'))
                                             for g in j['summary']]}
        logs, skipped = s3.day_logs('20261103', raw)
        R, so, err = capture(s3.run, logs, raw, TMP / 'nolab', str(bench))
        if err:
            out['errors'].append(('standing_3lap', err))
        else:
            md, so2, err2 = capture(s3.to_markdown, R)
            if err2:
                out['errors'].append(('standing_3lap md', err2))
            out['standing_md'] = md.splitlines() if md else None
            for ig in ((), ('gain',), ('gain', 'launch'), ('trim',)):
                r2, so3, err3 = capture(qp.to_markdown, R, '20261103', None, ig)
                if err3:
                    out['errors'].append(('by_params %s' % (ig,), err3))
                else:
                    out.setdefault('by_params', {})[','.join(ig) or 'none'] = r2[0].splitlines()
        for tag, lg in names.items():
            res, so, err = capture(cc.check_log, lg, TMP / 'nolab', raw)
            if err:
                out['errors'].append(('car_check ' + tag, err))
                continue
            txt, so2, err2 = capture(cc.to_text, res)
            if err2:
                out['errors'].append(('car_check text ' + tag, err2))
                continue
            out['car_check'][tag] = [x for x in txt.splitlines() if any(w in x for w in ('判定', '衝突', '試験', '空転', '欠け', '発進'))]
        # --latest(最後の本 = 4-3)と、3-8 を最後にした日
        rc, so, err = capture(cc.main, ['--latest', '--day', '20261103', '--raw', str(raw), '--labels', str(TMP / 'nolab'), '--no-json'])
        out['car_check_latest'] = {'rc': rc, 'err': err}
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print('errors', len(out['errors']))
    for e in out['errors']:
        print(e[0])
        print(e[1][-1500:])
    for nm, v in out['collect'].items():
        print('== ⑤', nm, v['all_ok'])
    for tag, v in out['car_check'].items():
        print('== car_check', tag)
        for x in v:
            print('   ', x[:300])


if __name__ == '__main__':
    main()
