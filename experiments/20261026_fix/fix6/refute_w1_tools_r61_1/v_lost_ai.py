"""w1-tools-r6-1-tools-1 の確かめ(反証側)。p1_lost_ai_scene.day を借りて、自分の tmp で ⑤ を流す。
形: A そろっている / B 2 つ目のシーンの npz が無い(CAM_FRONT だけ)/ C 2 つ目のシーンの npz が 0 バイト(電源断の形)
/ D AI 側 15 秒(1 つ目のシーンに AI のフレームあり)で 2 つ目のシーンが無い(比べ用)"""
import json
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
for p in (JF, JF / 'tools', JF / 'tests', JF / 'experiments/20261026_fix/fix6/refute_tools'):
    sys.path.insert(0, str(p))
import test_fix3_r3_2_tools as T  # noqa: E402

TMP = JF / 'experiments/20261026_fix/fix6/tmp_w1tools61'
OUT = Path(__file__).resolve().parent / 'out'


def day(root, form, t_ai=25.0):
    D = T.RealDay(root)
    on = dict(T.G, launch_boost=1)
    D.add_real(12, T.G, T.HAND)
    D.add_real(13, on, T.HAND, **T.on_kw(36))
    D.add_real(14, T.G, T.HAND)
    D.add_real(15, on, T.HAND, **T.on_kw(39))
    D.add_real(16, T.G, T.HAND)
    D.add_real(17, on, T.HAND, **T.on_kw(42))
    D.add_real(18, T.G, T.HAND)
    log, _ = D.add_real(19, on, T.HAND, dur=40.0, t_ai=t_ai, t_move=t_ai + 0.1, launch=(t_ai, 0.5, 1),
                        first_anchor=t_ai + 3.0, hits=[(t_ai + 6.0, 9.0)])
    sc = D.raw / log / ('%s-0002' % log)
    if form in ('B', 'D'):
        for p in sc.glob('*.npz'):
            p.unlink()
        (sc / 'CAM_FRONT').mkdir(exist_ok=True)
    elif form == 'C':
        for p in list(sc.glob('*.npz')) + [sc / 'scene.json']:
            if p.exists():
                p.write_bytes(b'')
    return D, log


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    OUT.mkdir(exist_ok=True)
    res = {}
    for form, t_ai in (('A', 25.0), ('B', 25.0), ('C', 25.0), ('D', 15.0)):
        (TMP / form).mkdir()
        D, log = day(TMP / form, form, t_ai)
        scenes = sorted(p.name for p in (D.raw / log).iterdir() if p.is_dir())
        j, o, _ = T.runc(D, 'v_' + form, args=['--ignore-gain'])
        g = j['summary'][0]
        res[form] = {
            'crashed_log': log, 'scenes': scenes,
            'rows_of_crashed_log': len([r for r in j['launches'] if r['log'] == log]),
            'n_on': (g['by_launch'].get('on') or {}).get('n'), 'n_off': (g['by_launch'].get('off') or {}).get('n'),
            'unreadable': g.get('unreadable'), 'first_lap_gap_on': g.get('first_lap_gap_on'),
            'checks': (g.get('compare') or {}).get('checks'), 'all_ok': (g.get('compare') or {}).get('all_ok'),
            'problems_keys': sorted((j.get('problems') or {}).keys()),
            'name_in_stdout': log in o,
            'stdout_tail': [x for x in o.splitlines() if not x.startswith('20261103-')][:40]}
    (OUT / 'v_lost_ai.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    shutil.rmtree(TMP)


if __name__ == '__main__':
    main()
