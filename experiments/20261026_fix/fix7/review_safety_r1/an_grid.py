"""grid_onoff.py の出力を ON と OFF で比べる(読むだけ)。
  python -I an_grid.py out/grid_*.json"""
import json
import sys
from collections import Counter


def main(paths):
    for pth in paths:
        d = json.load(open(pth))
        c = Counter()
        ex = []
        ngo, ngf = d['ng_on'], d['ng_off']
        for key, row in d['cells'].items():
            on, off = row['on'], row['off']
            c['cells'] += 1
            bo, bf = on['back'] or 0.0, off['back'] or 0.0
            if on['rear_v'] > off['rear_v'] + 0.05:
                c['rear_on_worse'] += 1
                ex.append(('rear', key, on['rear'], off['rear'], bo, bf))
            if off['rear_v'] > on['rear_v'] + 0.05:
                c['rear_off_worse'] += 1
            if (on['esc_push'] or 0) > (off['esc_push'] or 0) + 0.05:
                c['escpush_on_worse'] += 1
                ex.append(('escpush', key, on['esc_push'], off['esc_push'], bo, bf))
            if on['front_push'] > off['front_push'] + 0.5:
                c['frontpush_on_worse'] += 1
                ex.append(('front', key, on['front_push'], off['front_push'], bo, bf))
            if off['front_push'] > on['front_push'] + 0.5:
                c['frontpush_off_worse'] += 1
            if bo > max(bf, ngo['back'] or 0.0) + 0.05:
                c['back_on_more_than_off_and_ng'] += 1
                ex.append(('back', key, bo, bf, ngo['back']))
            if on['rear_push'] > off['rear_push'] + 0.05:
                c['rearpush_on_worse'] += 1
                ex.append(('rearpush', key, on['rear_push'], off['rear_push']))
            if on['giveup'] is not None and off['giveup'] is None:
                c['giveup_on_only'] += 1
        print(pth, 'ng_on', ngo['back'], ngo['rear_v'], 'ng_off', ngf['back'], ngf['rear_v'], dict(c))
        for x in ex[:12]:
            print('   ', x)


if __name__ == '__main__':
    main(sys.argv[1:])
