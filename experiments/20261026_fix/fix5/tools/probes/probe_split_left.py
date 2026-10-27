"""反証 4 回目(記録と夜の集計): 回 3 の途中で jetformer-web が起こし直し(RESET_ON_BOOT で左 4.2 → 4.1)、左を入れ直し忘れた日。
⑤ collect.py --ignore-gain の組は左の値で分かれる。分かれたことの注意(ゲインの gain_split にあたるもの)が出るか、答えはどうなるか。

作り物の回 3(本番の形 = 天井 ON・左 4.2): 3-1〜3-4 は左 4.2、起こし直しのあと 3-5〜3-8 は左 4.1(天井 ON のまま)。
起点まで: 3-1 OFF 3.013・3-2 ON 3.133(遅い)・3-3 OFF 3.013・3-4 ON 3.033・3-5 OFF 3.013・3-6 ON 2.713・3-7 OFF 3.013・3-8 ON 2.813。
8 本で 1 組なら ON の中央 2.923 − OFF 3.013 = −0.090(満たさない)。本物の data/raw は読まない。作り物は消す。
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
TMP = HERE / 'tmp_split'
OUT = HERE / 'out'


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    OUT.mkdir(exist_ok=True)
    out = {}
    for case in ('split', 'same'):
        (TMP / case).mkdir(parents=True)
        D = T.RealDay(TMP / case)
        L42 = dict(T.G, speed_ceiling=1, ceil_left_mps=4.2)
        L41 = dict(L42, ceil_left_mps=4.1) if case == 'split' else L42
        on = lambda P: dict(P, launch_boost=1)        # noqa: E731
        D.add_real(12, L42, T.HAND)
        D.add_real(13, on(L42), T.HAND, **T.on_kw(26))     # 起点まで 4.0 − 26/30 = 3.133
        D.add_real(14, L42, T.HAND)
        D.add_real(15, on(L42), T.HAND, **T.on_kw(29))     # 3.033
        D.add_real(16, L41, T.HAND)
        D.add_real(17, on(L41), T.HAND, **T.on_kw(38))     # 2.733
        D.add_real(18, L41, T.HAND)
        D.add_real(19, on(L41), T.HAND, **T.on_kw(35))     # 2.833
        j, o, _ = T.runc(D, 'n_' + case, args=['--ignore-gain'])
        out[case] = {'groups': [{'left': g['ceil_left_mps'], 'n_on': (g['by_launch'].get('on') or {}).get('n'),
                                 'n_off': (g['by_launch'].get('off') or {}).get('n'),
                                 'd_to_r1_s': (g.get('compare') or {}).get('d_to_r1_s'),
                                 'all_ok': (g.get('compare') or {}).get('all_ok')} for g in j['summary']],
                     'notice_lines': [x for x in o.splitlines() if '注意' in x],
                     'compare_lines': [x for x in o.splitlines() if 'ON − OFF' in x or x.startswith('11/03')]}
    (OUT / 'probe_split_left.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1))
    shutil.rmtree(TMP)


if __name__ == '__main__':
    main()
