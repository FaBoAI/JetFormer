"""直す 6-2 回目・係 tools: real_s3qp.py の old / new を比べ、car_check(変えていない)と照らす(読むだけ)。

  jf-heavy --lock none --mem 1500M -- python3 -I compare_s3qp.py → out/compare_s3qp.json・標準出力

見ること:
  1. 人側にした直後の衝突の無い本は、③ の結果(JSON の 1 本)と ④ の 1 章の行・JSON の本がバイト単位で同じ
  2. ある本は、変わった欄が衝突(collisions)・発進の衝突(standing の collisions・coll_before・counted)だけ
  3. 変わった本の ③ の衝突(数・最大 G・8G 以上・場所)が car_check の「衝突」と同じ
  4. md の変わった行は、その本の行と、足した注の行だけ
"""
import json
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
for p in (JF, JF / 'tools'):
    sys.path.insert(0, str(p))
import car_check as cc  # noqa: E402

O = Path(__file__).resolve().parent / 'out'
DAYS = ('20261010', '20261023', '20261024')


def main():
    res, bad = {}, []
    for d in DAYS:
        ro = {r['log']: r for r in json.loads((O / 'real_s3qp_old' / ('s3_%s.json' % d)).read_text())['results']}
        rn = {r['log']: r for r in json.loads((O / 'real_s3qp_new' / ('s3_%s.json' % d)).read_text())['results']}
        assert list(ro) == list(rn), d
        day = {'books': len(rn), 'after_ai_books': {}, 'same_books': 0}
        for lg in rn:
            a, b = ro[lg], rn[lg]
            aa = (b.get('collisions') or {}).get('after_ai')
            if not aa:
                if a != b:
                    bad.append('%s: 人側にした直後の衝突が無いのに ③ の結果が違う' % lg)
                day['same_books'] += 1
                continue
            keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            st_diff = []
            for sa, sb in zip(a.get('standing') or [], b.get('standing') or []):
                ks = sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))
                if ks:
                    st_diff.append({'t_ai': sb['t_ai'], 'keys': ks, 'old': {k: sa.get(k) for k in ks}, 'new': {k: sb.get(k) for k in ks}})
            if not set(keys) <= {'collisions', 'standing'}:
                bad.append('%s: 衝突のほかの欄が変わった %s' % (lg, keys))
            if any(not set(x['keys']) <= {'collisions', 'coll_before', 'counted', 'collisions_by_lap'} for x in st_diff):
                bad.append('%s: 発進の衝突のほかの欄が変わった' % lg)
            c = cc.check_log(lg)
            cs = c.get('collisions') or {}
            C = b['collisions']
            same_cc = (C['all'] == cs.get('n') and C['max_g'] == cs.get('max_g') and C['n_ge8'] == cs.get('n_ge8')
                       and C['by_place'] == cs.get('by_place') and C['after_ai'] == cs.get('after_ai'))
            if not same_cc:
                bad.append('%s: car_check と違う' % lg)
            day['after_ai_books'][lg] = {
                'old': {k: a['collisions'].get(k) for k in ('all', 'strong', 'max_g', 'n_ge8', 'by_place')},
                'new': {k: C.get(k) for k in ('all', 'strong', 'max_g', 'n_ge8', 'by_place', 'after_ai', 'ai_all', 'ai_strong')},
                'car_check': {'n': cs.get('n'), 'max_g': cs.get('max_g'), 'n_ge8': cs.get('n_ge8'),
                              'by_place': {k: v for k, v in (cs.get('by_place') or {}).items() if v}, 'after_ai': cs.get('after_ai')},
                'same_as_car_check': same_cc, 'changed_keys': keys, 'standing_changes': st_diff}
        # ④ の 1 章の行・JSON の本
        for sub in ('qp_%s' % d, 'qp_%s_ig' % d):
            jo = json.loads((O / 'real_s3qp_old' / sub / 'by_params.json').read_text())
            jn = json.loads((O / 'real_s3qp_new' / sub / 'by_params.json').read_text())
            for x, y in zip(jo['runs'], jn['runs']):
                if y['log'] not in day['after_ai_books'] and x != y:
                    bad.append('%s %s: ④ の本が違う' % (sub, y['log']))
            mo = (O / 'real_s3qp_old' / sub / 'by_params.md').read_text().splitlines()
            mn = (O / 'real_s3qp_new' / sub / 'by_params.md').read_text().splitlines()
            extra = [ln for ln in mn if ln not in mo]
            gone = [ln for ln in mo if ln not in mn]
            okx = [ln for ln in extra if any(lg in ln for lg in day['after_ai_books']) or ln.startswith('- 衝突(1 章の衝突(強)')
                   or ln == '' or ln.startswith('| ')]
            day.setdefault('md', {})[sub] = {'added': len(extra), 'removed': len(gone),
                                             'added_not_after_ai_book': [ln[:160] for ln in extra if ln not in okx]}
            # 2 章(組)の行で変わったのは、人側にした直後の衝突の入った組だけか
            gn = {g['text']: g for g in jn['groups']}
            for g in jo['groups']:
                h = gn.get(g['text'])
                if h is None:
                    bad.append('%s: 組が消えた %s' % (sub, g['text'][:60]))
                elif not h.get('collisions_after_ai') and g != h:
                    bad.append('%s: 人側にした直後の衝突の無い組が変わった %s' % (sub, g['text'][:60]))
        res[d] = day
    res['problems'] = bad
    (O / 'compare_s3qp.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for d in DAYS:
        print(d, 'books', res[d]['books'], 'same', res[d]['same_books'], 'after_ai', list(res[d]['after_ai_books']))
        for lg, v in res[d]['after_ai_books'].items():
            print('  ', lg, 'old', v['old'], '\n      new', {k: v['new'][k] for k in ('all', 'strong', 'max_g', 'n_ge8')},
                  'after', v['new']['after_ai'], 'cc same', v['same_as_car_check'], 'standing', v['standing_changes'])
        print('  md', res[d]['md'])
    print('problems', bad)


if __name__ == '__main__':
    main()
