"""grid_onoff の出力で、ON の下がる距離が止まりなしの位相の最大(phase_ref.json)+0.05 m を超えた組・OFF の位相の最大も超えた組(読むだけ)
  python -I an_over.py out/grid_*.json"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ref = json.load(open(HERE / 'out' / 'phase_ref.json'))
for pth in sys.argv[1:]:
    d = json.load(open(pth))
    kind = d['grid'].split('_')[1]
    mon, moff = ref[kind + '_on']['max'], ref[kind + '_off']['max']
    over = [(k, r['on']['back'], r['off']['back'], r['on']['rear_v'], r['off']['rear_v'], r['on']['esc_push'], r['on']['guards'][:3])
            for k, r in d['cells'].items() if (r['on']['back'] or 0) > mon + 0.05]
    over2 = [x for x in over if x[1] > max(moff, x[2] or 0) + 0.05]
    push = [(k, r['on']['back'], r['on']['esc_push'], r['off']['esc_push'], r['off']['front_push'], r['on']['front_push'])
            for k, r in d['cells'].items() if (r['on']['esc_push'] or 0) > 0.05]
    zero = [(k, r['on']['back'], r['on']['esc_push'], r['off']['back']) for k, r in d['cells'].items()
            if r['on']['back'] is not None and r['on']['back'] < 0.1]
    print(pth, 'cells', len(d['cells']), 'phase max on/off', mon, moff, '| on>phase_on+0.05:', len(over),
          '| also > off phase max & off:', len(over2), '| esc push on:', len(push), '| on back<0.1:', len(zero),
          '| max on back', max((r['on']['back'] or 0) for r in d['cells'].values()))
    for x in over2[:8]:
        print('   over', x)
    for x in push[:8]:
        print('   push', x)
    for x in zero[:5]:
        print('   zero', x)
