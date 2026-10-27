"""反証 4 回目・安全 s1 の確かめ: rec_hold_keep = 1・rec_back_guard = 1 で、走っている車(約 3 m/s)の 0.6〜2.0 秒のカメラの止まりの
間に、止まらない 2.5G 以上の衝撃(本物の記録の 10/23・10/24 では走行中に普通に出る: scan_g_moving.py)が来ると、今の assist.py
(r3-r3-safety-1)は明けた最初のフレームで「中止」(中立・ハンドルまっすぐ)にし、車は人が CH3 を戻すまで惰性でまっすぐ進む。
直す前(tests/fixtures/fix3_r3_1_before)と keep = 0 はモデルのまま(ハンドルもモデル)。
ToF ガード 2(減速・ブレーキ)の日は、前の壁の手前のガードのブレーキも中止の中立で上書きされ、壁に当たる。

本物の drive._loop・偽物の ESC と車(fix3/refute2_safety/closed_loop.py)。装置・GPU・API には触れない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        experiments/20261026_fix/fix4/review_safety/probe_giveup_moving.py
出力: out/probe_giveup_moving.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rs_common as R          # noqa: E402
from rs_common import A, C, N, PARAMS, T0   # noqa: E402

STEER = C + 180.0              # モデルのハンドル(曲がっている途中)


DEC = [1.5]
_ORIG_STEP = R.CL.Car.step


def car_step(self, t, thr):
    """中立の惰性の減速を DEC[0] m/s² に(fix3/fix_r3_1/probe_safety.py と同じ考え。本物の 10/10 の長いカメラの止まり 3 回では
    2.6 m/s → 0 が 0.83 秒以内 = 3 m/s² 以上)"""
    if not (thr < N - 20.0 or thr > N + 40.0):
        v_before = self.v
        r = _ORIG_STEP(self, t, thr)
        dv = (DEC[0] - 1.5) * R.CL.DT
        if self.contact is None:
            if v_before > 0:
                self.v = max(0.0, self.v - dv)
            elif v_before < 0:
                self.v = min(0.0, self.v + dv)
        return r
    return _ORIG_STEP(self, t, thr)


def scene(drop_a, L, xf, t_spike, g, thr=N - 200.0):
    return {'dur': drop_a + L + 4.0, 'xf': xf, 'model': (STEER, thr), 'drops': [(drop_a, drop_a + L)], 'veh_skip': [],
            'spike': (t_spike, g)}


class SpikeSim(R.CL.Sim):
    """Sim と同じ。scene['spike'] = (t, g) の時刻に、速さを変えない衝撃(横からこすった・段差)を IMU に入れる"""

    def _advance(self, t_end):
        sp = self.s.get('spike')
        while self.t + R.CL.DT <= t_end + 1e-9:
            if sp is not None and abs((self.t + R.CL.DT) - sp[0]) < R.CL.DT / 2 and not self.car.spike:
                self.car.spike = [sp[1], sp[1]]
            super()._advance(min(self.t + R.CL.DT, t_end))
        super()._advance(t_end)


def one(keep, guard_mode, drop_a, L, xf, t_spike, g, before=False, thr=N - 200.0):
    p = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=keep, tof_guard=guard_mode)
    sc = scene(drop_a, L, xf, t_spike, g, thr)
    orig = R.CL.Sim
    R.CL.Sim = SpikeSim
    try:
        if before:
            B, BA = BEFORE
            sim, rec, pca, d = R.run_closed(p, sc, mod=B, amod=BA)
        else:
            sim, rec, pca, d = R.run_closed(p, sc)
    finally:
        R.CL.Sim = orig
    end = drop_a + L
    rows = [((t - T0) / 1e9, r) for t, r in rec.rows]
    after = [(s, r) for s, r in rows if s >= end and not r['hold']]
    gu = [s for s, r in after if int(r['assist']) == A.ASSIST_GIVEUP]
    v_end = next(v for t, x, v, *_ in sim.log if t >= end)
    x_end = next(x for t, x, v, *_ in sim.log if t >= end)
    res = {'v_at_camera_back': round(v_end, 2), 'giveup_first_s_after_camera': round(gu[0] - end, 3) if gu else None,
           'giveup_rows': len(gu), 'front_hits': [h for h in sim.car.hits if h[1] == 'front'],
           'steer_out_after': sorted({round(float(r['out'][0])) for s, r in after}),
           'thr_out_first5': [round(float(r['out'][1])) for s, r in after[:5]],
           'kinds_first5': [int(r['assist']) for s, r in after[:5]],
           'guard_codes_after': sorted({int(r['guard']) for s, r in after}),
           'events': [(e[1]['what'], e[1].get('phase'), e[1].get('hold_s')) for e in rec.events if e[0] == 'recovery_guard']}
    if gu:
        tg = gu[0]
        xs = [(t, x, v) for t, x, v, *_ in sim.log if t >= tg]
        x0 = next(x for t, x, v in xs)
        stop = next(((t, x) for t, x, v in xs if abs(v) < 0.02), xs[-1][:2])
        res['coast_m_after_giveup_to_stop_or_end'] = round(stop[1] - x0, 2)
        res['coast_s'] = round(stop[0] - tg, 2)
    res['x_at_camera_back'] = round(x_end, 2)
    return res


def main():
    global BEFORE
    BEFORE = R.load_snapshot('fix3_r3_1_before')
    out = {}
    try:
        # A: 壁なし(遠い)。モデルは前進 200 µs・ハンドル C + 180。カメラ 3.0 秒から L 秒止まる。止まって 0.3 秒後に 3.0G
        for L in (0.6, 1.0, 2.0):
            for g in (2.6, 3.0, 5.7):
                for name, keep, bef in (('now_keep1', 1, False), ('before_keep1', 1, True), ('now_keep0', 0, False)):
                    out['A L=%.1f g=%.1f %s' % (L, g, name)] = one(keep, 1, 3.0, L, 60.0, 3.3, g, bef)
        # B: ToF ガード 2(減速・ブレーキ)・3。前の壁はカメラが戻った所(x 9.16 m・2.23 m/s。A と同じ)から d m 先(ガードが見るのは 1.95 m まで)
        for gm in (2, 3):
            for dd in (1.2, 1.5, 1.8, 2.2, 2.6):
                for name, keep, bef in (('now_keep1', 1, False), ('before_keep1', 1, True), ('now_keep0', 0, False)):
                    out['B L=0.6 guard=%d wall=+%.1f %s' % (gm, dd, name)] = one(keep, gm, 3.0, 0.6, 9.16 + dd, 3.3, 3.0, bef)
        # C: 中立の惰性 3.0 m/s²(本物の ESC に近い側)・モデル 前進 250 µs(約 3.75 m/s)
        R.CL.Car.step = car_step
        try:
            for dec in (1.5, 3.0):
                DEC[0] = dec
                for L in (0.55, 0.6, 0.8):
                    for name, keep, bef in (('now_keep1', 1, False), ('before_keep1', 1, True), ('now_keep0', 0, False)):
                        out['C dec=%.1f L=%.2f g=3.0 thr=N-250 %s' % (dec, L, name)] = one(keep, 1, 3.0, L, 60.0, 3.2, 3.0, bef,
                                                                                          thr=N - 250.0)
        finally:
            R.CL.Car.step = _ORIG_STEP
            DEC[0] = 1.5
    finally:
        R.cleanup()
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'probe_giveup_moving.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    for k, v in out.items():
        print(k, v)


if __name__ == '__main__':
    main()
