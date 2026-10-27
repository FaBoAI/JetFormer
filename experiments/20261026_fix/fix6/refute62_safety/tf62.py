"""反証 6-2・安全: 狙った でたらめ(本物の drive._loop・閉じたループ)。前の壁の距離・後ろの物の距離・モデルの形(壁を押し続ける
rev / 壁の手前でブレーキ側 = ESC が後退を待つ brake)・rev_s・rev_us・tof_guard・launch_boost・start_kick を でたらめにし、
まず値の止まりなしで最初のリカバリーの時刻(後退側の始め t_brake・escape t_esc)を見て、その窓に車両の値の止まり(1〜2 回)と
カメラの止まり(0〜1 回)を でたらめに置く。同じ場面を
  now1 = 今(fix6 のあと)keep 1・見張り 1、now0 = 今 keep 0、bef1 = fix6 の直す前(fix5 のあと)keep 1、ng1 = now1 の止まりなし
で回し、最初のリカバリー(t_brake から escape の 1.0 秒あとまで)の後ろの当たり・escape が前の壁を押した秒・下がった距離を比べる。
esc = 'fake'(閉じたループの偽物の ESC。中立 0.1 秒で ready・後退側から 0.08 秒)/ 'slow'(中立 0.12 秒・0.10 秒。本物の
人の中立 約 0.12 秒・下がり始め 約 0.08〜0.1 秒に寄せた)。

    python -I tf62.py [本数] [種] [esc]
"""
import collections
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c62 as K          # noqa: E402
E = K.E
import test_fix3_r2_1 as R2           # noqa: E402
import test_rec_fix5_safety as F5     # noqa: E402
from test_launch_drive import C, N    # noqa: E402

DT = R2.DT
ESC = {'neu': 0.1, 'rdelay': 0.08}


class Car(R2.Car):
    """R2.Car と同じ(中立で ready になる秒・後退側から下がり始める秒だけ ESC から)。revlog も残す"""

    def __init__(self, s):
        super().__init__(s)
        self.revlog = []

    def step(self, t, thr):
        s = self.s
        fwd, rev = thr < N - 20.0, thr > N + 40.0
        v0 = self.v
        if fwd:
            self.ready = self.braked = self.rev_on = False
            self.rev_wait = self.neu_t = 0.0
            if self.v < 0:
                self.v = min(0.0, self.v + 8.0 * DT)
            else:
                vt = min((N - thr) / 80.0 * 1.2, 6.0)
                self.v += float(np.clip((vt - self.v) / 0.3, -6.0, 6.0)) * DT
        elif rev:
            self.neu_t = 0.0
            if self.rev_on:
                a = (thr - N) / 300.0 * 5.0
                self.v = max(self.v - a * DT, -(thr - N) / 300.0 * 3.0)
            elif self.ready and self.v <= 0.075:
                self.rev_wait += DT
                if self.rev_wait >= ESC['rdelay'] - 1e-9:
                    self.rev_on = True
            else:
                self.braked = True
                self.v = max(0.0, self.v - 6.0 * DT) if self.v > 0 else min(0.0, self.v + 6.0 * DT)
        else:
            self.rev_wait = 0.0
            self.neu_t += DT
            if self.braked and self.neu_t >= ESC['neu'] - 1e-9:
                self.ready, self.braked = True, False
            if self.rev_on:
                self.rev_on = False
                self.ready = True
            self.v = max(0.0, self.v - 1.5 * DT) if self.v > 0 else min(0.0, self.v + 1.5 * DT)
        nx = self.x + self.v * DT
        prev_con = self.contact
        self.contact = None
        if nx >= s['xf']:
            if self.v > 0.05 and prev_con != 'front':
                g = min(2.0 + 2.5 * self.v, 9.0)
                self.hits.append((round(t, 3), 'front', round(self.v, 2), round(g, 2)))
                self.spike = [g, g]
            nx, self.v = s['xf'], 0.0
            self.contact = 'front'
        if s.get('xr') is not None:
            if not self.rear_active and nx > s['xr'] + 0.1:
                self.rear_active = True
            if self.rear_active and nx <= s['xr']:
                if self.v < -0.05 and prev_con != 'rear':
                    g = min(2.0 + 2.5 * -self.v, 9.0)
                    self.hits.append((round(t, 3), 'rear', round(self.v, 2), round(g, 2)))
                    self.spike = [g, g]
                nx, self.v = s['xr'], 0.0
                self.contact = 'rear'
        self.x = nx
        acc = (self.v - v0) / DT
        g = self.spike.pop(0) if self.spike else 0.0
        self.revlog.append((t, self.rev_on, self.contact))
        return acc, g


def run(name, params, scene):
    mod, amod = E.load(name)
    E.TMP.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(E.TMP)) as td:
        mp.setattr(R2, 'Sim', F5.SpikeSim)
        mp.setattr(R2, 'Car', Car)
        return R2.run_cl(mod, mp, Path(td), params, scene)


def scene(kind, xf, model_thr, brake_d):
    if kind == 'rev':
        model = (C, N - model_thr)
    else:
        def model(t, car):
            return (C, N - model_thr) if car.x < xf - brake_d else (C, N + 200.0)
    return {'dur': 7.0, 'xf': xf, 'model': model, 'drops': [], 'veh_skip': []}


def win(sim, pca, p, t_from=1.5):
    m = E.first_rec(sim, pca, sim.s['xf'], t_from=t_from, rev_us=p['rev_us'])
    if m is None:
        return None
    te = m['t_escape']
    end = (te + 1.0) if te is not None else m['t_brake'] + 3.5
    m['rear_win'] = [h for h in sim.car.hits if h[1] == 'rear' and m['t_brake'] <= h[0] <= end]
    m['end'] = end
    return m


def main(n=200, seed=6202, esc='fake'):
    if esc == 'slow':
        ESC.update(neu=0.12, rdelay=0.10)
    rng = random.Random(seed)
    out = {'seed': seed, 'esc': esc, 'runs': 0, 'skipped': 0, 'flags': collections.Counter(), 'examples': []}
    try:
        for i in range(n):
            kind = rng.choice(['rev', 'rev', 'brake'])
            xf = round(rng.uniform(1.6, 3.0), 2)
            mthr = rng.choice([100.0, 120.0, 150.0])
            bd = round(rng.uniform(0.35, 0.6), 2)
            rev_s = rng.choice([0.4, 0.4, 0.4, 0.3, 0.6])
            rev_us = rng.choice([300.0, 300.0, 300.0, 200.0, 400.0])
            p = E.P1(rev_s=rev_s, rev_us=rev_us, tof_guard=rng.choice([1, 1, 2, 3]),
                     launch_boost=rng.choice([0, 1]), start_kick=rng.choice([0, 1]))
            p['launch_no_kick'] = rng.choice([0, 1]) if p['launch_boost'] else 0
            sc0 = scene(kind, xf, mthr, bd)
            rear = round(rng.uniform(1.1, 2.0), 2) if rng.random() < 0.7 else None
            if rear is not None:
                sc0['xr'] = xf - rear
            sim, rec, pca = run('now', p, sc0)
            m_ng = win(sim, pca, p)
            if m_ng is None or m_ng['t_escape'] is None:
                out['skipped'] += 1
                continue
            tb, te = m_ng['t_brake'], m_ng['t_escape']
            sc = dict(sc0, veh_skip=[], drops=[])
            for _ in range(rng.choice([1, 1, 1, 2])):
                a = round(rng.uniform(tb - 0.25, te + 0.1), 3)
                sc['veh_skip'].append((a, round(a + rng.uniform(0.16, 0.5), 3)))
            if rng.random() < 0.3:
                a = round(rng.uniform(tb - 0.2, te), 3)
                sc['drops'].append((a, round(a + rng.choice([0.12, 0.15, 0.2, 0.3, 0.45]), 3)))
            r = {}
            for nm, mod, pp in (('now1', 'now', p), ('now0', 'now', dict(p, rec_hold_keep=0)),
                                ('bef1', 'fix6_before', p)):
                s_, rc_, pc_ = run(mod, pp, sc)
                mm = win(s_, pc_, pp)
                if mm is not None:
                    mm['guards'] = E.guards(rc_)[:4]
                r[nm] = mm
            r['ng1'] = m_ng
            # no-gap keep0 for reference
            s_, rc_, pc_ = run('now', dict(p, rec_hold_keep=0), sc0)
            r['ng0'] = win(s_, pc_, p)
            out['runs'] += 1
            flags = []
            n1 = r['now1']
            if n1 is None:
                continue
            refs = [r[k] for k in ('ng1', 'now0', 'ng0') if r[k] is not None]
            v1 = max([-h[2] for h in n1['rear_win']], default=0.0)
            vr = max([max([-h[2] for h in x['rear_win']], default=0.0) for x in refs], default=0.0)
            if v1 > vr + 0.05:
                flags.append(('rear_worse', v1, vr))
            if n1['esc_push_s'] > max(x['esc_push_s'] for x in refs) + 1e-9:
                flags.append(('esc_push', n1['esc_push_s'], max(x['esc_push_s'] for x in refs)))
            if n1['back_m'] > max(x['back_m'] for x in refs) + 0.1:
                flags.append(('more_back', n1['back_m'], max(x['back_m'] for x in refs)))
            for f in flags:
                out['flags'][f[0]] += 1
            if flags:
                out['examples'].append({'run': i, 'flags': flags, 'kind': kind, 'xf': xf, 'rear': rear, 'mthr': mthr,
                                        'bd': bd, 'params': {k: p[k] for k in ('rev_s', 'rev_us', 'tof_guard',
                                                                               'launch_boost', 'start_kick',
                                                                               'launch_no_kick')},
                                        'veh_skip': sc['veh_skip'], 'drops': sc['drops'],
                                        'res': {k: (None if v is None else {kk: v[kk] for kk in
                                                                            ('t_brake', 't_escape', 'back_m',
                                                                             'esc_push_s', 'rear_win', 'guards')
                                                                            if kk in v}) for k, v in r.items()}})
                print(i, flags, {k: (None if v is None else (v['back_m'], v['esc_push_s'], v['rear_win'][:1]))
                                 for k, v in r.items()}, flush=True)
            if (i + 1) % 20 == 0:
                print('..', i + 1, dict(out['flags']), out['skipped'], flush=True)
    finally:
        K.cleanup()
    out['flags'] = dict(out['flags'])
    (HERE / 'out' / ('tf62_%d_%s.json' % (seed, esc))).write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                                                             default=str))
    print(json.dumps({k: v for k, v in out.items() if k != 'examples'}, ensure_ascii=False))


if __name__ == '__main__':
    a = sys.argv[1:]
    main(int(a[0]) if a else 200, int(a[1]) if len(a) > 1 else 6202, a[2] if len(a) > 2 else 'fake')
