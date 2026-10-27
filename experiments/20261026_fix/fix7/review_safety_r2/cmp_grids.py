"""反証 7-2 回目・safety: g8.py の出力(今の版 = fix7 r1 のあと)を、反証 7-1 回目の grid_onoff の出力(そのときの版 = fix6 r2 のあと)と
場面ごとに比べる(読むだけ)。on が変わった場面と、ON と OFF の比べ(反証 7-1 回目の an_grid と同じ数え)。
    python -I cmp_grids.py"""
import glob
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
R1 = HERE.parent / 'review_safety_r1' / 'out'


def onoff(d):
    c = Counter()
    for key, row in d['cells'].items():
        on, off = row['on'], row['off']
        bo, bf = on['back'] or 0.0, off['back'] or 0.0
        c['cells'] += 1
        c['rear_on_worse'] += on['rear_v'] > off['rear_v'] + 0.05
        c['rear_off_worse'] += off['rear_v'] > on['rear_v'] + 0.05
        c['escpush_on_worse'] += (on['esc_push'] or 0) > (off['esc_push'] or 0) + 0.05
        c['fail_on'] += bo < 0.1 and (on['esc_push'] or 0) > 0.05
        c['fail_off'] += bf < 0.1 and (off['esc_push'] or 0) > 0.05
        c['rearpush_on_worse'] += on['rear_push'] > off['rear_push'] + 0.05
        c['giveup_on_only'] += on['giveup'] is not None and off['giveup'] is None
    return dict(c)


def main():
    for f in sorted(glob.glob(str(HERE / 'out' / 'grid_*.json'))):
        name = Path(f).name
        d = json.loads(Path(f).read_text())
        g = R1 / name
        print('==', name, 'now:', onoff(d))
        if not g.exists():
            print('   (反証 7-1 回目の出力なし)')
            continue
        r = json.loads(g.read_text())
        print('   r1 (fix6 r2):', onoff(r))
        ch = Counter()
        ex = []
        for key, row in d['cells'].items():
            if key not in r['cells']:
                ch['missing'] += 1
                continue
            a, b = row['on'], r['cells'][key]['on']
            same = all(a.get(k) == b.get(k) for k in ('back', 'rear', 'esc_push', 'front_push', 'rear_push', 'giveup', 'guards'))
            ch['on_same' if same else 'on_changed'] += 1
            if not same:
                worse = ((a['rear_v'] > b['rear_v'] + 0.05) or ((a['esc_push'] or 0) > (b['esc_push'] or 0) + 0.05)
                         or (a['rear_push'] > b['rear_push'] + 0.05))
                ch['on_worse_than_r1' if worse else 'on_changed_not_worse'] += 1
                if len(ex) < 6:
                    ex.append((key, {k: a.get(k) for k in ('back', 'rear_v', 'esc_push', 'guards')},
                               {k: b.get(k) for k in ('back', 'rear_v', 'esc_push', 'guards')}))
            c_off = all(row['off'].get(k) == r['cells'][key]['off'].get(k) for k in ('back', 'rear', 'esc_push', 'front_push'))
            ch['off_same' if c_off else 'off_changed'] += 1
        print('   now vs r1:', dict(ch))
        for x in ex:
            print('     ', x)


if __name__ == '__main__':
    main()
