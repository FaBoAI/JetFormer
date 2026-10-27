"""fuzz_onoff.py の出力を比べる(読むだけ)。ON(両方 1)と OFF(両方 0)を同じ場面で。onc = ON の止まりなし。
  python -I an_fuzz.py out/fuzz_onoff_7101.json [...]"""
import json
import sys
from collections import Counter


def main(paths):
    tot = Counter()
    ex = []
    for pth in paths:
        d = json.load(open(pth))
        print(pth, 'identity', {k: (v if isinstance(v, int) else len(v)) for k, v in d['identity'].items()})
        for r in d['runs']:
            on, off, onc = r['on'], r['off'], r['onc']
            tot['runs'] += 1
            tag = '%s#%d' % (d['seed'], r['i'])
            if on['rear_max_v'] > off['rear_max_v'] + 0.05:
                tot['rear_on_worse_than_off'] += 1
                if on['rear_max_v'] > onc['rear_max_v'] + 0.05:
                    tot['rear_on_worse_than_off_and_onc'] += 1
                    ex.append(('rear', tag, on['rear'], off['rear'], onc['rear'], r['params'], r['scene']))
            if off['rear_max_v'] > on['rear_max_v'] + 0.05:
                tot['rear_off_worse_than_on'] += 1
            if on['front_push_s'] > off['front_push_s'] + 0.5:
                tot['front_on_worse'] += 1
                ex.append(('front', tag, on['front_push_s'], off['front_push_s'], onc['front_push_s'], r['params'],
                           {k: v for k, v in r['scene'].items() if k in ('drops', 'veh_skip', 'hit_drop', 'xf', 'xr', 'human')}))
            if off['front_push_s'] > on['front_push_s'] + 0.5:
                tot['front_off_worse'] += 1
            if on['rear_push_s'] > off['rear_push_s'] + 0.1:
                tot['rearpush_on_worse'] += 1
                ex.append(('rearpush', tag, on['rear_push_s'], off['rear_push_s'], onc['rear_push_s'], r['params']))
            if off['rear_push_s'] > on['rear_push_s'] + 0.1:
                tot['rearpush_off_worse'] += 1
            if on['giveup_t'] is not None:
                tot['giveup_on'] += 1
            if off['giveup_t'] is not None:
                tot['giveup_off'] += 1
            if on['rev_dead_s'] > off['rev_dead_s'] + 0.3:
                tot['revdead_on_worse'] += 1
            for h in on['rear']:
                tot['rear_hits_on'] += 1
            for h in off['rear']:
                tot['rear_hits_off'] += 1
            tot['front_push_on'] += on['front_push_s']
            tot['front_push_off'] += off['front_push_s']
            if on['guards'] and any(g[0] == 'back_early' for g in on['guards']):
                tot['runs_with_back_early'] += 1
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in tot.items()})
    for x in ex[:40]:
        print('  ', json.dumps(x, ensure_ascii=False, default=str)[:600])


if __name__ == '__main__':
    main(sys.argv[1:])
