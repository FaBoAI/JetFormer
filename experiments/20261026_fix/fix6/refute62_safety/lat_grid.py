"""反証 6-2・安全: fix6 の s3-a(EscTrack・二重に数えない)を、本物の走行ループに近い遅れ(撮像 → 書き込み)で振る。
閉じたループの偽物は、フレームが撮像の 0.008 秒あとに着いてその場で書く(遅れ 8 ms)。本物の記録(10/23・24 の AI 走行
86,551 フレーム)の latency_ms は中央値 31 ms・95% 48 ms。場面は fix6/safety_r1/exp_grid.py の rev・brake(読み込むだけ)。
値の止まりは偽物の ESC が後退に入る時刻 t_on から d0 秒ずらして L 秒。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        experiments/20261026_fix/fix6/refute62_safety/lat_grid.py rev 0.031 [rev_s] [rev_us]
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c62 as K          # noqa: E402
E = K.E
import exp_grid as G     # noqa: E402
import test_fix3_r2_1 as R2           # noqa: E402
import test_rec_fix5_safety as F5     # noqa: E402
from jetformer.sensors.base import Sample   # noqa: E402
from test_launch_drive import T0, NS  # noqa: E402

D0 = [-0.1, -0.06, -0.02, 0.0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3]
LS = [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6]


class LatSim(F5.SpikeSim):
    def wait(self, timeout):
        lat = float(self.s.get('lat', 0.008))
        if self.t >= self.s['dur']:
            self.d._stop.set()
            return False
        while self.dropped(self.next_cam):
            self.next_cam += 1.0 / 30
        arrive = self.next_cam + lat
        if arrive - self.t > timeout + 1e-9:
            self._advance(self.t + timeout)
            return False
        self._advance(arrive)
        cam_t = T0 + int(round(self.next_cam * NS))
        self.cam_slot[0] = Sample(cam_t, {'image': np.zeros((8, 8, 3), np.uint8), 'seq': self.seq})
        self.seq += 1
        self.next_cam += 1.0 / 30
        self.box['ns'] = T0 + int(round(self.t * NS))
        return True


def run(name, params, scene):
    mod, amod = E.load(name)
    E.TMP.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(E.TMP)) as td:
        mp.setattr(R2, 'Sim', LatSim)

        class _Car(E.LogCar):
            def __init__(self, s):
                super().__init__(s)
                self.revlog = []
        mp.setattr(R2, 'Car', _Car)
        return R2.run_cl(mod, mp, Path(td), params, scene)


def one(name, p, kind, lat, skip=None, rear=None):
    sc, xf = G.scene(kind, skip, rear)
    sc['lat'] = lat
    sim, rec, pca = run(name, p, sc)
    m = E.first_rec(sim, pca, xf, rev_us=p['rev_us'])
    m['guards'] = E.guards(rec)[:4]
    return m


def main():
    kind, lat = sys.argv[1], float(sys.argv[2])
    rev_s = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
    rev_us = float(sys.argv[4]) if len(sys.argv) > 4 else 300.0
    names = sys.argv[5].split(',') if len(sys.argv) > 5 else ['now']
    p = E.P1(rev_s=rev_s, rev_us=rev_us)
    out = {'kind': kind, 'lat': lat, 'rev_s': rev_s, 'rev_us': rev_us, 'ref': {}, 'grid': {}}
    try:
        for nm in names:
            out['ref'][nm] = one(nm, p, kind, lat)
            out['ref'][nm + '_keep0'] = one(nm, dict(p, rec_hold_keep=0), kind, lat)
        t_on = out['ref'][names[0]]['t_rev_on']
        out['t_on'] = t_on
        print('ref', {k: (v['back_m'], v['esc_push_s'], v['t_rev_on']) for k, v in out['ref'].items()}, flush=True)
        for d0 in D0:
            for L in LS:
                sk = (round(t_on + d0, 3), round(t_on + d0 + L, 3))
                row = {}
                for nm in names:
                    row[nm] = one(nm, p, kind, lat, sk)
                    row[nm + '_keep0'] = one(nm, dict(p, rec_hold_keep=0), kind, lat, sk)
                k = 'd0=%+.2f L=%.2f' % (d0, L)
                out['grid'][k] = row
                print(k, {nm: (r['back_m'], r['esc_push_s'], len(r['front_hits']), r['guards'][:2])
                          for nm, r in row.items()}, flush=True)
    finally:
        K.cleanup()
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / ('lat_%s_%g_%g_%g.json' % (kind, lat, rev_s, rev_us))).write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
