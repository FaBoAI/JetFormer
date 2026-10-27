"""反証 4 回目(記録と夜の集計): 最後のシーンが書けなかった記録(電源断・jetformer-web の落ちの形)を car_check が読むと?

作り物の 11/3 回 4 の本(止まって 3 周、40 秒、シーン 20 秒ごと)。24.0 秒に 9G で当たり、そのあと落ちた = 2 つ目のシーンの npz が
0 バイト(本物の 10/10 20261010-163247 の 0021・0022、10/23 20261023-204825 の 0001 と同じ形)。log.json は読める。
比べ: 同じ本で落ちなかった(2 つ目のシーンも読める)とき。
本物の data/raw・labels は読まない。作り物は tmp/ に作り、終わったら消す(この台本の最後)。
"""
import json
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF / 'tests'))
from test_standing_3lap import write_standing_log  # noqa: E402
import car_check as cc  # noqa: E402

HERE = Path(__file__).resolve().parent
TMP = HERE / 'tmp_partial'
OUT = HERE / 'out'
PRM = {'throttle_gain': 0.92, 'steer_gain': 1.0, 'speed_ceiling': 0, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 0,
       'auto_target_mps': 4.27, 'launch_boost': 0, 'launch_us': 250.0, 'launch_mps': 2.5, 'launch_max_s': 1.2, 'zone_caps': 0}
DESC = ('ai_no=20 bench=real_bench_20261023 day=2026-11-03 block=20 cand=EXP model=runB battery=#__ V_before=__ '
        'V_after=__ check=__ official=0 memo=計画と違うので試し(元 cand=B): throttle_gain 0.92(基準 1.0)')


def make(raw, log, lost, dur=40.0, hit=24.0):
    d, anchors, _ = write_standing_log(raw, log, dur=dur, t_ai=1.0, t_move=1.5, ramp_s=0.5, hits=[(hit, 9.0)],
                                       params=PRM, description=DESC, run='runB', scene_s=20.0)
    sc = sorted(p for p in d.iterdir() if p.is_dir())[-1]
    if lost == 'zero':                           # 2 つ目のシーン(20〜40 秒)の npz を 0 バイトに(電源断)
        for p in sc.glob('*.npz'):
            p.write_bytes(b'')
    elif lost == 'none':                         # 2 つ目のシーンの npz が無い(jetformer-web の落ち: シーンの終わりに書く前に落ちた)
        for p in sc.glob('*.npz'):
            p.unlink()
        (sc / 'CAM_FRONT').mkdir()
    return d


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    OUT.mkdir(exist_ok=True)
    res = {}
    lines = []
    import quick_eval_by_params as qp
    import standing_3lap as s3
    for case in ('full', 'zero', 'none', 'long_full', 'long_none'):
        lost = case.replace('long_', '')
        raw = TMP / case
        raw.mkdir(parents=True)
        log = '20261103-140000'
        if case.startswith('long'):           # 回 1・2 の 4〜6 周の本(60 秒)。45 秒に 9G で当たって落ちた = 3 つ目のシーンが無い
            make(raw, log, lost, dur=60.0, hit=45.0)
        else:                                 # 回 4 の止まって 3 周の本(40 秒)。24 秒に 9G で当たって落ちた
            make(raw, log, lost)
        r = cc.check_log(log, raw / 'nolabels', raw)
        txt = cc.to_text(r)
        latest = cc.latest_log(raw, '20261103')
        newer = cc.newer_unreadable(raw, '20261103', latest)
        r3 = s3.analyze_log(log, raw, raw / 'nolabels', {'runB': 'B'})
        st = [x for x in r3.get('standing') or [] if x.get('to_r1_s') is not None]
        res[case] = {'by_params_flags': qp.flags_of(r3), 'by_params_coll': r3.get('collisions', {}).get('text'),
                     'standing_valid': [x.get('valid') for x in st], 'standing_why': [x.get('why') for x in st],
                     'zero_byte': r3.get('zero_byte'), 'problems': r3.get('problems'),
                     'standing_md_has_zero_line': '0 バイトで飛ばした npz' in s3.to_markdown([r3])}
        res[case].update({'judge': r['judge']['label'], 'code': r['judge']['code'],
                                           'collisions': {k: r['collisions'].get(k) for k in ('n', 'max_g', 'n_ge8')},
                                           'latest': latest, 'newer_unreadable': newer,
                                           'mentions_missing': any(w in txt for w in ('0 バイト', '読めない', '欠け', 'ありません'))})
        lines += ['=== %s' % {'full': '40 秒・落ちていない(2 つ目のシーンも読める)', 'zero': '40 秒・電源断(2 つ目のシーンの npz が 0 バイト)',
                              'none': '40 秒・落ちた(2 つ目のシーンに npz が無い = 書いている途中のシーン)',
                              'long_full': '60 秒・落ちていない', 'long_none': '60 秒・落ちた(3 つ目のシーンに npz が無い)'}[case],
                  txt, '--latest が選ぶ記録: %s・それより新しい読めない記録: %s' % (latest, newer), '']
    (OUT / 'probe_partial_carcheck.txt').write_text('\n'.join(lines), encoding='utf-8')
    (OUT / 'probe_partial_carcheck.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1))
    shutil.rmtree(TMP)


if __name__ == '__main__':
    main()
