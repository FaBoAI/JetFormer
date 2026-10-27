"""反証 6-2 回目・係 tools: 人側にした直後(AI の区間の終わりから 0.5 秒以内)の衝突が、car_check には出るのに
③ standing_3lap・④ quick_eval_by_params の 1 本ごとの「衝突・最大 G・8G 以上・場所」には入るか(本物の 10/23・10/24。読むだけ)。

  python3 -I p2_after_ai_scan.py
出力 ../out/p2_after_ai_scan.json
"""
import json
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
import car_check as cc  # noqa: E402
import quick_eval_by_params as qp  # noqa: E402
import standing_3lap as s3  # noqa: E402

HERE = Path(__file__).resolve().parent
LAB = JF / 'data' / 'labels'


def main():
    hits = []
    for day in ('20261023', '20261024'):
        logs, _ = s3.day_logs(day)
        for lg in logs:
            j, z = LAB / (lg + '.json'), LAB / (lg + '.npz')
            if not (j.exists() and z.exists()):
                continue
            info = json.loads(j.read_text(encoding='utf-8'))
            lab = dict(np.load(z))
            aa = cc.after_ai_collisions(info.get('events') or {}, lab)
            if aa:
                hits.append((lg, [(round(float(c['t_s']), 2), round(float(c['g']), 2), c.get('loc_detail'), c['after_ai_s']) for c in aa]))
    res = {'after_ai': hits, 'rows': {}}
    logs = [lg for lg, _ in hits]
    if logs:
        R = s3.run(logs)
        _, runs, _ = qp.to_markdown(R, None, None)
        cc_rows = {}
        for lg in logs:
            r = cc.check_log(lg)
            cs = r.get('collisions') or {}
            cc_rows[lg] = {'n': cs.get('n'), 'max_g': cs.get('max_g'), 'n_ge8': cs.get('n_ge8'), 'by_place': cs.get('by_place'),
                           'after_ai': cs.get('after_ai')}
        for x, r in zip(runs, R):
            res['rows'][x['log']] = {'by_params_1': {'collisions': x.get('collisions'), 'coll_max_g': x.get('coll_max_g'),
                                                     'coll_ge8': x.get('coll_ge8'), 'coll_places': x.get('coll_places')},
                                     'standing_3lap': (r.get('collisions') or {}).get('text'),
                                     'car_check': cc_rows.get(x['log'])}
    out = HERE.parent / 'out' / 'p2_after_ai_scan.json'
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for lg, h in hits:
        print(lg, h)
        print('   ④', res['rows'].get(lg, {}).get('by_params_1'))
        print('   car_check', {k: v for k, v in (res['rows'].get(lg, {}).get('car_check') or {}).items()})


if __name__ == '__main__':
    main()
