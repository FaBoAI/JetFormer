"""fix8 係 safety-proposal: Recovery だけに でたらめな入力(tests/test_rec_fix7_safety_r2._fuzz_stops の写し)を入れて、
  OFF(back_guard False): 今(base)と案 pA・pB・pC の出す値と状態が全部同じか(ON だけの案 = 同じはず)
  ON(back_guard True): 今と初めて違ったフレームの形(今 → 案 の段の移り)を数える(違ったら両方を新しくして続ける)
を見る。装置・GPU・API なし。出力 out/unit_fuzz10.json
"""
import collections
import hashlib
import importlib.util
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
PSHA = {'base': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f',
        'pA': '82e2f8a584c66fe292d2122800dcacb928d93c24aba78ca53faa5331915528d3',
        'pB': '05d78b8aeaf09a64364e284191ce19259b3b0baee4d449674056f3fe6228e5b6',
        'pC': '7ac97688a93713704762dc29f3ff64d17dc2ec4b758332ec4200ffd2240585b8'}
C, N = 1441.0, 1496.0
FDT = 1.0 / 30
STATE = ('phase', 't_phase', 't_check', 'slow_since', 'impact_t', 'impact_moving', 'cooldown_until', 'armed', 'retries',
         'backed_s', 'est_s', 'seen_rev_s', 'skip_gap', 'noesc_noted', 'esc_ran', 'run_t0', 'back_sure', 't_last', 'turn',
         'back_t')


def load(name):
    fp = HERE / name / 'assist.py'
    assert hashlib.sha256(fp.read_bytes()).hexdigest() == PSHA[name], name
    spec = importlib.util.spec_from_loader('jetformer._p8f_%s' % name, loader=None)
    m = importlib.util.module_from_spec(spec)
    exec(compile(fp.read_text(encoding='utf-8'), str(fp), 'exec'), m.__dict__)
    return m


def fuzz_stops(seed, n=4000):
    """tests/test_rec_fix7_safety_r2._fuzz_stops の写し(同じ並び)"""
    rng = random.Random(seed)
    t, out = 100.0, []
    mode, left = 'run', rng.randint(20, 60)
    esc_rev = rng.choice([None, True, True, False])
    for i in range(n):
        if left <= 0:
            mode = rng.choice(['run', 'stop', 'stop', 'stop', 'back', 'burst'])
            left = {'run': rng.randint(10, 60), 'stop': rng.randint(10, 90), 'back': rng.randint(1, 25),
                    'burst': rng.randint(2, 8)}[mode]
            if rng.random() < 0.2:
                esc_rev = rng.choice([None, True, False])
        left -= 1
        if mode == 'run':
            e = rng.choice([-5000.0, -3000.0, -1500.0])
        elif mode == 'stop':
            e = rng.choice([0.0] * 12 + [301.0, 1322.0, -200.0, 150.0])
        elif mode == 'back':
            e = rng.choice([1400.0, 2200.0, 900.0])
        else:
            e = rng.choice([1322.0, 1628.0, 0.0, -1270.0, 1408.0])
        t += rng.choice([FDT] * 30 + [0.05, 0.1])
        kw = dict(impact_g=rng.choice([0.0] * 60 + [3.0]), rev_us=rng.choice([250.0, 300.0]),
                  rev_s=rng.choice([0.2, 0.3, 0.4, 0.8]), esc_rev=esc_rev)
        if rng.random() < 0.03:
            kw.update(hold_since=t - rng.uniform(0.02, 0.15), hold_back_s=rng.choice([0.0, 0.0, 0.02, 0.06, 0.2]),
                      hold_rev_s=rng.choice([0.0, 0.0, 0.05]), hold_g=rng.choice([0.0] * 10 + [3.0]))
        out.append(((C + rng.uniform(-300, 300), N + rng.choice([-150.0, -60.0, 0.0, 20.0])), e, t, kw))
    return out


def main():
    M = {k: load(k) for k in PSHA}
    res = {'off_identical': {}, 'on_div': {}, 'on_div_kinds': {}, 'frames': 0, 'reverse_frames_base_on': 0}
    for p in ('pA', 'pB', 'pC'):
        same_off, frames = 0, 0
        kinds = collections.Counter()
        ndiv = 0
        for seed in range(12):
            feed = fuzz_stops(seed)
            # OFF
            r0, r1 = M['base'].Recovery((C, N)), M[p].Recovery((C, N))
            ok = True
            for model, e, t, kw in feed:
                a = r0.step(model, e, t, back_guard=False, **kw)
                b = r1.step(model, e, t, back_guard=False, **kw)
                if a != b or any(getattr(r0, k) != getattr(r1, k) for k in STATE):
                    ok = False
                    break
            same_off += ok
            # ON
            r0, r1 = M['base'].Recovery((C, N)), M[p].Recovery((C, N))
            for model, e, t, kw in feed:
                ph0 = r0.phase
                a = r0.step(model, e, t, back_guard=True, **kw)
                b = r1.step(model, e, t, back_guard=True, **kw)
                frames += 1
                if p == 'pA':
                    res['reverse_frames_base_on'] += ph0 == 'reverse'
                ev0, ev1 = r0.guard_event, r1.guard_event
                if a != b:
                    ndiv += 1
                    kinds['%s→ base %s%s / %s %s%s・skip_gap %s・eRPM %s' % (
                        ph0, r0.phase, '(%s)' % ev0[0] if ev0 else '', p, r1.phase, '(%s)' % ev1[0] if ev1 else '',
                        r0.skip_gap if r0.phase == 'reverse' else '-', '> 300' if e > 300 else '<= 300')] += 1
                    r0, r1 = M['base'].Recovery((C, N)), M[p].Recovery((C, N))
                r0.guard_event = r1.guard_event = None
                if r1.phase == 'giveup' or r0.phase == 'giveup':
                    r0, r1 = M['base'].Recovery((C, N)), M[p].Recovery((C, N))
        res['off_identical'][p] = '%d/12' % same_off
        res['on_div'][p] = ndiv
        res['on_div_kinds'][p] = dict(kinds.most_common())
        res['frames'] = frames
    (HERE / 'out' / 'unit_fuzz10.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
