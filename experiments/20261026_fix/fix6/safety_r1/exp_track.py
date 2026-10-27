"""EscTrack(final/assist.py)の見積もりが、閉じたループの偽物の ESC(tests/test_fix3_r2_1.Car)が実際に後退で走らせた区間を
覆うか(多めに数える側か)を、でたらめな書き込みの並びで確かめる(読むだけ)。
    python -I exp_track.py [n_seeds]
"""
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp_common as E    # noqa: E402

N = E.N
_, amod = E.load('final')


def one(seed, dur=6.0):
    rng = random.Random(seed)
    car = E.R2.Car({'xf': 1e9})
    tr = amod.EscTrack(N)
    t, thr, writes = 0.0, N, []
    nxt = 0.0
    while nxt < dur:
        kind = rng.random()
        if kind < 0.3:
            v = N - rng.choice([30.0, 80.0, 120.0, 200.0])
        elif kind < 0.6:
            v = N + rng.choice([45.0, 60.0, 200.0, 300.0, 400.0])
        else:
            v = N + rng.choice([0.0, 10.0, -10.0, 39.0])
        L = rng.choice([1, 1, 2, 3, 3, 4, 5, 9, 15])
        for _ in range(L):
            writes.append((round(nxt, 4), v))
            nxt += 1 / 30.0 + rng.uniform(-0.004, 0.004)
    wi = 0
    rev_steps = []
    while t < dur:
        t = round(t + 0.01, 6)
        while wi < len(writes) and writes[wi][0] <= t - 1e-9:
            thr = writes[wi][1]
            tr.write(int(writes[wi][0] * 1e9), thr)
            wi += 1
        car.step(t, thr)
        car.v = 0.0 if abs(car.v) < 1e-9 else car.v
        if car.rev_on:
            rev_steps.append(t)
    first = {s for i, s in enumerate(rev_steps) if i == 0 or s - rev_steps[i - 1] > 0.0101}
    miss = [s for s in rev_steps if s not in first and tr.rev_s(int((s - 0.01) * 1e9), int(s * 1e9)) < 0.005]
    est = tr.rev_s(0, int(dur * 1e9))
    return len(rev_steps) * 0.01, est, miss


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    tot_car = tot_est = 0.0
    misses = []
    for seed in range(n):
        c, e, miss = one(seed)
        tot_car += c
        tot_est += e
        if miss:
            misses.append((seed, len(miss), miss[:3]))
    print('seeds', n, 'car reverse s %.2f  estimate s %.2f  seeds with missed steps %d' % (tot_car, tot_est, len(misses)))
    for m in misses[:10]:
        print('  miss', m)


if __name__ == '__main__':
    main()
