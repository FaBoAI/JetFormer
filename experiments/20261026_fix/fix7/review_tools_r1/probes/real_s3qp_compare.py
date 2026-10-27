"""反証 7-1 回目・係 tools(読むだけ): 本物の 10/10・10/23・10/24 で ③ standing_3lap・④ quick_eval_by_params を、fix6-2 の直す前の写し
(tests/fixtures/fix6_tools_r2_before/*.orig)と今で流し、違う本が「人側にした直後の衝突」のある本だけか・その数が car_check と同じかを見る。
data/raw・labels は読むだけ。出力 ../out/real_s3qp_compare.json"""
import json
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
for p in (JF, JF / 'tools', JF / 'tests'):
    sys.path.insert(0, str(p))
import car_check as cc  # noqa: E402
import standing_3lap as s3  # noqa: E402
import quick_eval_by_params as qp  # noqa: E402
from test_fix3_r1_2 import load_mod  # noqa: E402

FIX = JF / 'tests' / 'fixtures' / 'fix6_tools_r2_before'
OUT = Path(__file__).resolve().parent.parent / 'out' / 'real_s3qp_compare.json'


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in ('after_ai', 'ai_all', 'ai_strong')}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def main():
    os3 = load_mod('s3_old_fix62', FIX / 'standing_3lap.py.orig')
    oqp = load_mod('qp_old_fix62', FIX / 'quick_eval_by_params.py.orig')
    oqp.s3 = os3
    res = {}
    for day in ('20261010', '20261023', '20261024'):
        logs, _ = s3.day_logs(day)
        new = s3.run(logs)
        old = os3.run(logs)
        diff_books = []
        for a, b in zip(new, old):
            ja, jb = json.dumps(a, sort_keys=True, default=str), json.dumps(b, sort_keys=True, default=str)
            if ja == jb:
                continue
            aa = (a.get('collisions') or {}).get('after_ai')
            c = cc.check_log(a['log'])
            cs = c.get('collisions') or {}
            ok = (aa is not None and cs.get('after_ai') == aa and cs.get('n') == a['collisions']['all']
                  and cs.get('n_ge8') == a['collisions']['n_ge8'] and cs.get('max_g') == a['collisions']['max_g'])
            st_diff = [(x.get('t_ai'), x.get('counted'), y.get('counted'), x.get('coll_before'), y.get('coll_before'))
                       for x, y in zip(a.get('standing') or [], b.get('standing') or []) if x != y]
            best_new, best_old = s3.best_standing(a), os3.best_standing(b)
            diff_books.append({'log': a['log'], 'after_ai': aa, 'same_as_car_check': ok, 'standing_diff': st_diff,
                               'best_same': (best_new or {}).get('from_move_s') == (best_old or {}).get('from_move_s'),
                               'only_collisions_differ': json.dumps(strip({k: v for k, v in a.items() if k not in ('collisions', 'standing')}),
                                                                    sort_keys=True, default=str) ==
                               json.dumps(strip({k: v for k, v in b.items() if k not in ('collisions', 'standing')}), sort_keys=True,
                                          default=str)})
        mn, _, gn = qp.to_markdown(new, day, None, ('gain',))
        mo, _, go = oqp.to_markdown(old, day, None, ('gain',))
        gdiff = [(x['text'], {k: (x[k], y[k]) for k in x if k in y and x[k] != y[k]}) for x, y in zip(gn, go)
                 if json.dumps(x, sort_keys=True, default=str) != json.dumps(y, sort_keys=True, default=str)]
        res[day] = {'books': len(new), 'changed_books': diff_books, 'groups_changed': gdiff}
        print(day, 'books', len(new), 'changed', len(diff_books),
              'all match car_check', all(x['same_as_car_check'] for x in diff_books),
              'best same', all(x['best_same'] for x in diff_books), 'only coll', all(x['only_collisions_differ'] for x in diff_books))
        for x in diff_books:
            print('   ', x['log'], x['after_ai'], 'standing_diff', x['standing_diff'])
        for t, dd in gdiff:
            print('   group', t, dd)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
