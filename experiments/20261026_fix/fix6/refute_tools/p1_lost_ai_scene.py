"""反証 6-1 回目・tools: 回 3 の ON の本で、AI 側にしたシーン(記録の 2 つ目以降)ごと書けずに落ちた本(電源断・jetformer-web の落ち)。
1 つ目のシーン(人側で置いて待った 0〜20 秒)は書けている。⑤ collect.py --ignore-gain の答えは?

作り物の回 3: OFF 4 本(3-1・3-3・3-5・3-7)・ON 3 本(3-2・3-4・3-6。起点まで 2.8・2.7・2.6 秒)・ON 3-8 は 25 秒に AI 側
(10/24 の本物は AI 側が 20 秒より後の本がふつうにある: 27.9・48.2・20.1・28.7・48.9・23.5 秒)、31 秒(最初の周)に 9G で当たる。
  A: 3-8 の記録がそろっている(落ちていない)
  B: 3-8 の 2 つ目のシーン(20〜40 秒)の npz が無い(CAM_FRONT/ だけ = 書いている途中で落ちた)
作り物は fix6/tmp_tools の下に作って最後に消す。本物の data/raw・labels は読まない。
"""
import json
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
for p in (JF, JF / 'tools', JF / 'tests'):
    sys.path.insert(0, str(p))
import test_fix3_r3_2_tools as T  # noqa: E402

HERE = Path(__file__).resolve().parent
TMP = JF / 'experiments/20261026_fix/fix6/tmp_tools/p1'
OUT = HERE / 'out'


def day(root, lost):
    D = T.RealDay(root)
    on = dict(T.G, launch_boost=1)
    D.add_real(12, T.G, T.HAND)
    D.add_real(13, on, T.HAND, **T.on_kw(36))
    D.add_real(14, T.G, T.HAND)
    D.add_real(15, on, T.HAND, **T.on_kw(39))
    D.add_real(16, T.G, T.HAND)
    D.add_real(17, on, T.HAND, **T.on_kw(42))
    D.add_real(18, T.G, T.HAND)
    log, _ = D.add_real(19, on, T.HAND, dur=40.0, t_ai=25.0, t_move=25.1, launch=(25.0, 0.5, 1), first_anchor=28.0,
                        hits=[(31.0, 9.0)])
    if lost:
        sc = D.raw / log / ('%s-0002' % log)
        for p in sc.glob('*.npz'):
            p.unlink()
        (sc / 'CAM_FRONT').mkdir()
    return D, log


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    out = {}
    for case, lost in (('A_intact', False), ('B_lost', True)):
        (TMP / case).mkdir()
        D, log = day(TMP / case, lost)
        j, o, _ = T.runc(D, 'p1_' + case, args=['--ignore-gain'])
        g = j['summary'][0]
        out[case] = {
            'crashed_log': log,
            'rows_of_crashed_log': [{'t_ai': r['t_ai'], 'launch_state': r.get('launch_state'), 'to_r1_s': r.get('to_r1_s'),
                                     'first_lap_hits': r.get('collisions_first_lap')} for r in j['launches'] if r['log'] == log],
            'n_on': (g['by_launch'].get('on') or {}).get('n'), 'n_off': (g['by_launch'].get('off') or {}).get('n'),
            'collisions_first_lap_on': (g['by_launch'].get('on') or {}).get('collisions_first_lap'),
            'unreadable': g.get('unreadable'), 'first_lap_gap_on': g.get('first_lap_gap_on'),
            'compare': g.get('compare'),
            'stdout_lines': [x for x in o.splitlines() if ('ON − OFF' in x or '記録が欠けている本' in x or '読めない' in x
                                                          or x.startswith('11/03'))]}
    (OUT / 'p1_lost_ai_scene.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1))
    shutil.rmtree(TMP)


if __name__ == '__main__':
    main()
