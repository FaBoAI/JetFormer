"""反証 w2-tools-r62-2 の打ち消し(読むだけ): 人側にした直後(0.5 秒以内)の衝突が、
④ quick_eval_by_params 1 章の「衝突・最大 G・8G 以上・場所」(standing_3lap.analyze_log の res['collisions'])に入らず、
car_check の「衝突」には入るか。本物の 10/23・10/24 の記録で、1 章の markdown の行まで出す。
  jf-heavy --lock none --mem 1500M -- python3 -I probe_after_ai_by_params.py
"""
import json
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
import car_check as cc  # noqa: E402
import quick_eval_by_params as qp  # noqa: E402
import standing_3lap as s3  # noqa: E402

LOGS = ['20261023-205309', '20261024-134607', '20261023-213149']
OUT = Path(__file__).resolve().parent / 'out' / 'probe_after_ai_by_params.json'


def main():
    R = [s3.analyze_log(lg) for lg in LOGS]
    md, runs, _ = qp.to_markdown(R, None, None)
    md_lines = md if isinstance(md, list) else str(md).splitlines()
    res = {}
    for lg, r, x in zip(LOGS, R, runs):
        c = cc.check_log(lg)
        cs = c.get('collisions') or {}
        row1 = [ln for ln in md_lines if ('| %s |' % lg) in ln]
        # standing_3lap の衝突の元(score_log の collision_events)に人側の衝突が 1 つでもあるか
        res[lg] = {
            'by_params_1': {k: x.get(k) for k in ('collisions', 'strong', 'coll_max_g', 'coll_ge8', 'coll_places')},
            'by_params_1_md_row': row1[0] if row1 else None,
            's3_collisions_text': (r.get('collisions') or {}).get('text'),
            'car_check': {'n': cs.get('n'), 'max_g': cs.get('max_g'), 'n_ge8': cs.get('n_ge8'),
                          'by_place': {k: v for k, v in (cs.get('by_place') or {}).items() if v},
                          'after_ai': cs.get('after_ai')},
            'car_check_text': [ln for ln in cc.to_text(c).splitlines() if '衝突' in ln],
            'standing_starts_collisions': [(s.get('t_ai'), (s.get('collisions') or {}).get('n'), s.get('end_s'))
                                           for s in r.get('standing') or []],
        }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for lg, v in res.items():
        print('==', lg)
        print('  ④ 1章:', v['by_params_1'])
        print('  ④ 1章 md:', v['by_params_1_md_row'])
        print('  car_check:', v['car_check'])
        print('  standing starts (t_ai, n, end):', v['standing_starts_collisions'])


if __name__ == '__main__':
    main()
