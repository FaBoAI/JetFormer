"""r2-rec2-5 の確かめ(読むだけ・作り物の記録だけ): 電源断で npz が無い ON の本は collect.py --day のまとめから黙って消えるか
A: 3-4(ON)が 3.0 秒に 9G で当たって 3.1 秒に止まる・npz はある(対照)
B: A と同じで、3-4 のシーンの npz を全部消す(log.json・jpg は残る = 電源断・サーバーの落ち)
C: B で、jpg も scene.json も無い(シーンの dir だけ)
"""
import json
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
sys.path.insert(0, str(JF / 'tools'))
import test_fix3_r1_2 as T  # noqa: E402


def run(tmp, name, cut=None):
    d = tmp / name
    d.mkdir(parents=True)
    D = T.Day(d)
    made = []
    for i in range(8):
        kw = dict(hits=[(3.0, 9.0)], stop_after=3.1) if i == 3 else {}
        made.append(D.add(12 + i, T.ON if i % 2 else T.PRM, **kw))
    bad, ld = made[3]
    if cut:
        cut(ld)
    j, out, txt, _ = D.run(name='collect_' + name)
    g = j['summary'][0]
    return {'bad': bad, 'bad_in_launches': any(r['log'] == bad for r in j['launches']),
            'bad_in_json_text': bad in txt, 'n_launches': len(j['launches']),
            'by_launch_n': {k: v['n'] for k, v in g['by_launch'].items()},
            'not_reached': g.get('not_reached'), 'all_ok': (g.get('compare') or {}).get('all_ok'),
            'checks': (g.get('compare') or {}).get('checks'), 'summary_keys': sorted(g),
            'json_top_keys': sorted(j), 'problems': j.get('problems'),
            'stdout_skip': [x for x in out.splitlines() if x.startswith('skip')],
            'stdout_verdict': [x for x in out.splitlines() if 'ON − OFF' in x],
            'stdout_mentions_bad': [x for x in out.splitlines() if bad in x]}


def cut_npz(ld):
    for p in ld.glob('*/*.npz'):
        p.unlink()


def cut_all(ld):
    for s in ld.iterdir():
        if s.is_dir():
            for p in sorted(s.rglob('*'), reverse=True):
                (p.unlink() if p.is_file() else p.rmdir())


def main():
    tmp = Path(sys.argv[1])
    tmp.mkdir(parents=True, exist_ok=True)
    res = {'A_npz_kept': run(tmp, 'a'), 'B_npz_cut': run(tmp, 'b', cut_npz), 'C_scene_empty': run(tmp, 'c', cut_all)}
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
