"""反証 4 回目・安全: fix3/README.md の 31 件のうち安全(走行ループ)の 7 件が、今の drive.py・assist.py で直っているかを
自分の場面で確かめる(本物の drive._loop・偽物の ESC と車 = fix3/refute2_safety/closed_loop.py。装置・GPU・API には触れない)。

  r1-rs1-1  中立止めの最中の後ろの当たり → 明けたあと後ろを押さない(rear_impact_hold)
  r1-rs1-2  走行中の launch_no_kick 1 → 0・start_kick 0 → 1 で区間の途中に StartKick の窓が開かない
  r1-rs1-3  (コードは直さない・手順書)— 確かめない
  r2-r2-rs-1 'reverse' の始めの中立止めで、やり直し(+50 µs)・中立止めなしより多く下がる が無い
  r2-r2-rs-2 2〜5 秒の中立止めのあと、明けてすぐ後退側を出さない
  r3-r3-safety-1 当たりの直後のカメラの止まり(8 つの位相)で、どれも中止
  r3-r3-safety-2 後ろへ速い途中の長い中立止めの中止で、keep 0 より下がらない・当たらない

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -B -I \
        experiments/20261026_fix/fix4/review_safety/verify_1026.py
出力: out/verify_1026.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rs_common as R          # noqa: E402
from rs_common import A, C, N, PARAMS, T0, drive   # noqa: E402

CL = R.CL
DEC = [1.5]
_ORIG_STEP = CL.Car.step


def car_step(self, t, thr):
    """中立の惰性の減速を DEC[0] に(fix3/fix_r3_1/probe_safety.py と同じ考え)"""
    if not (thr < N - 20.0 or thr > N + 40.0):
        v_before = self.v
        r = _ORIG_STEP(self, t, thr)
        dv = (DEC[0] - 1.5) * CL.DT
        if self.contact is None:
            if v_before > 0:
                self.v = max(0.0, self.v - dv)
            elif v_before < 0:
                self.v = min(0.0, self.v + dv)
        return r
    return _ORIG_STEP(self, t, thr)


PH = []


def wrap(orig):
    def step(self, model_us, erpm, now, impact_g=0.0, *a, **kw):
        p0 = self.phase
        r = orig(self, model_us, erpm, now, impact_g, *a, **kw)
        PH.append((round(now - 100.0, 3), p0, self.phase))
        return r
    return step


def run(keep, sc, guard=1, **kw):
    del PH[:]
    p = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=guard, rec_hold_keep=keep)
    p.update(kw)
    sim, rec, pca, d = R.run_closed(p, sc, wrap=wrap)
    return sim, rec, pca, list(PH)


def t_of(ph, phase):
    return next((t for t, p0, p1 in ph if p1 == phase and p0 != p1), None)


def base(xf=2.0, dur=9.0, **kw):
    sc = {'dur': dur, 'xf': xf, 'model': (C, N - 120.0), 'drops': [], 'veh_skip': []}
    sc.update(kw)
    return sc


def rear_push_after(sim, end, win=0.8):
    """明けてから win 秒の間に、後ろの物に接して後退側を出していた秒(反証 1 回目の「最大 約 0.8 秒押す」と同じ窓。
    それより後の 2 回目のリカバリーの当たりは見張りの rear_impact の話で別)"""
    return round(sum(CL.DT for t, x, v, thr, con, rev_on, sl in sim.log
                     if end <= t < end + win and con == 'rear' and thr > N + 40), 3)


def v_r1rs11():
    """後退の途中で後ろの物に当たり、その当たりがカメラの止まり(0.2〜0.45 秒)の中に入る場面。明けてから後ろを押した秒"""
    out = {}
    sim, rec, pca, ph = run(1, base())
    t_rev = t_of(ph, 'reverse')
    reach = 2.0 - min(x for t, x, *_ in sim.log if t > t_rev)
    for back in (0.3, 0.5, 0.7):
        xr = 2.0 - back
        for o in (0.0, 0.1, 0.2, 0.3):
            for L in (0.2, 0.3, 0.45):
                sc = base(xr=xr, drops=[(t_rev + o, t_rev + o + L)])
                sim, rec, pca, ph = run(1, sc)
                rh = [h for h in sim.car.hits if h[1] == 'rear']
                if not rh or not (t_rev + o <= rh[0][0] < t_rev + o + L):
                    continue
                out['back=%.1f o=%.1f L=%.2f' % (back, o, L)] = {
                    'rear_hit': rh[0], 'rear_push_after_camera_s': rear_push_after(sim, t_rev + o + L),
                    'events': [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']}
    ok = all(v['rear_push_after_camera_s'] == 0.0 for v in out.values()) and len(out) > 0
    return {'ok': ok, 'n': len(out), 'reach_nohold_m': round(reach, 3), 'cases': out}


def v_r1rs12():
    """(a) 走行中の start_kick 0 → 1 を set_params が断る (b) launch_no_kick 1 → 0 にしても「出さない」区間は StartKick を出さない"""
    res = {}
    import pytest
    R.TMP.mkdir(parents=True, exist_ok=True)
    mp = pytest.MonkeyPatch()
    mp.setattr(drive, 'SETTINGS_PATH', R.TMP / 'drive_settings_r1rs12.json')     # 本物の data/drive_settings.json は読まない・書かない
    d = drive.Driver(hub=None, recorder=R.FakeRecorder())
    d.params = dict(PARAMS, start_kick=0)
    d.running_override = True
    orig_prop = type(d).__dict__['running']
    try:
        type(d).running = property(lambda self: getattr(self, 'running_override', False))
        try:
            d.set_params({'start_kick': 1})
            res['start_kick_0to1_running'] = 'accepted'
        except ValueError as e:
            res['start_kick_0to1_running'] = 'refused: %s' % str(e)[:40]
        for k in ('launch_no_kick', 'rec_hold_keep', 'rec_back_guard'):
            d.params = dict(PARAMS, start_kick=0, rec_back_guard=1 if k == 'rec_hold_keep' else 0)
            try:
                d.set_params({k: 1})
                res['%s_0to1_running' % k] = 'accepted'
            except ValueError as e:
                res['%s_0to1_running' % k] = 'refused'
    finally:
        type(d).running = orig_prop
        mp.undo()
    # (b) Assist: 人側 → AI 側、発進の前進は ToF が無いので「出さない」→ 1 フレーム目 launch_no_kick = 1、2 フレーム目から 0
    a = A.Assist((C, N))
    t = 100.0
    a.step((C, N - 20.0), 0.0, False, t)
    kinds = []
    for i in range(40):
        t += 1 / 30
        inp = A.LaunchIn(tof_ok=False, veh_ok=True, v=0.0)
        o, k = a.step((C, N - 20.0), 0.0, True, t, kick_on=True, launch=inp, launch_no_kick=(i == 0))
        kinds.append(k)
    res['kick_rows_after_no_kick_1to0'] = sum(1 for k in kinds if k == A.ASSIST_START)
    ok = res['start_kick_0to1_running'].startswith('refused') and res['kick_rows_after_no_kick_1to0'] == 0 and all(
        v == 'refused' for k, v in res.items() if k.endswith('_0to1_running') and k != 'start_kick_0to1_running')
    return {'ok': ok, **res}


def v_r2rs1():
    """'reverse' の始めから 0.05 秒あとに L 秒の中立止め。中立止めなしと比べて下がった距離・最大の後退の強さ"""
    out = {}
    sim, rec, pca, ph = run(1, base())
    t_rev = t_of(ph, 'reverse')
    t_coast = t_of(ph, 'coast')
    nohold_back = round(2.0 - min(x for t, x, *_ in sim.log if t > t_rev and t < t_rev + 3.0), 3)
    nohold_max = max(th for t, s, th in pca.w)
    for o in (0.02, 0.05, 0.1, 0.2):
        for L in (0.1, 0.2, 0.3, 0.45):
            sim, rec, pca, ph = run(1, base(drops=[(t_rev + o, t_rev + o + L)]))
            back = round(2.0 - min(x for t, x, *_ in sim.log if t > t_rev and t < t_rev + 3.0), 3)
            out['o=%.2f L=%.2f' % (o, L)] = {'back_m': back, 'max_thr': max(th for t, s, th in pca.w)}
    worst = max(v['back_m'] for v in out.values())
    ok = worst <= nohold_back + 0.005 and all(v['max_thr'] <= nohold_max for v in out.values())
    return {'ok': ok, 'nohold_back_m': nohold_back, 'nohold_max_thr': nohold_max, 'worst_back_m': worst, 'cases': out,
            't_reverse': t_rev, 't_coast': t_coast}


def v_r2rs2():
    """止まった直後・brake・gap・reverse の途中に 1〜5 秒の中立止め。明けて 0.5 秒以内に後退側を出すか"""
    out = {}
    sim, rec, pca, ph = run(1, base())
    t_stop = next(t for t, x, v, *_ in sim.log if t > 1.2 and abs(v) < 0.01 and x > 1.9)
    starts = {'stuck': t_stop + 0.05, 'brake': t_of(ph, 'brake') + 0.05, 'gap': t_of(ph, 'gap') + 0.05,
              'reverse': t_of(ph, 'reverse') + 0.05}
    for nm, t0 in starts.items():
        for L in (1.0, 2.0, 5.0):
            end = t0 + L
            sim, rec, pca, ph = run(1, base(dur=end + 3.0, drops=[(t0, end)]))
            first_rev = next((round(t - end, 3) for t, s, th in pca.w if t >= end and th > N + 40), None)
            gu = any(int(r['assist']) == A.ASSIST_GIVEUP for _, r in rec.rows)
            out['%s L=%.0f' % (nm, L)] = {'first_reverse_side_after_camera_s': first_rev, 'giveup': gu}
    ok = all(v['first_reverse_side_after_camera_s'] is None or v['first_reverse_side_after_camera_s'] > 0.5
             for v in out.values())
    return {'ok': ok, 'cases': out}


def v_r3s1():
    """前の壁 2.00〜2.07 m(8 つの位相)に当たった直後からカメラが L 秒来ない。明けて最初のフレームで中止か"""
    out = {}
    for L in (0.6, 1.0, 2.0):
        for j in range(8):
            xf = round(2.00 + 0.01 * j, 2)
            sc = {'dur': 8.0, 'xf': xf, 'model': (C, N - 120.0), 'drops': [], 'veh_skip': [], 'hit_drop': (0.0, L)}
            sim, rec, pca, ph = run(1, sc, start_kick=1)
            back = sim.car.hits[0][0] + L
            rows = [((t - T0) / 1e9, r) for t, r in rec.rows if not r['hold']]
            first = next((round(s - back, 3), int(r['assist'])) for s, r in rows if s >= back)
            rev = any(int(r['assist']) == A.ASSIST_RECOVERY and float(r['out'][1]) > N + 40 for s, r in rows if s >= back)
            out['L=%.1f xf=%.2f' % (L, xf)] = {'first_after_camera': first, 'reverse_after': rev}
    ok = all(v['first_after_camera'][1] == A.ASSIST_GIVEUP and not v['reverse_after'] for v in out.values())
    return {'ok': ok, 'cases': out}


def v_r3s2():
    """'reverse' に入って o 秒あとから L 秒カメラが来ない。後ろの物は中立止めなしで下がる所 + add。keep 1 と keep 0 の比べ"""
    out = {}
    CL.Car.step = car_step
    try:
        for dec in (1.5, 3.0):
            DEC[0] = dec
            sim, rec, pca, ph = run(1, base(dur=8.0))
            t_rev = t_of(ph, 'reverse')
            reach = 2.0 - min(x for t, x, *_ in sim.log if t > 1.6)
            for o, L in ((0.40, 0.55), (0.45, 0.60), (0.50, 0.55), (0.30, 1.0)):
                for add in (0.15, 0.3, None):
                    xr = None if add is None else 2.0 - (reach + add)
                    r = {}
                    for keep in (0, 1):
                        sc = base(dur=8.0, drops=[(t_rev + o, t_rev + o + L)])
                        if xr is not None:
                            sc['xr'] = xr
                        sim, rec, pca, ph = run(keep, sc)
                        end = t_rev + o + L
                        rh = [h for h in sim.car.hits if h[1] == 'rear']
                        at = next(x for t, x, *_ in sim.log if t >= end)
                        r[keep] = {'rear_v': max([-h[2] for h in rh], default=0.0),
                                   'back_after_camera_m': round(at - min(x for t, x, *_ in sim.log if end <= t <= end + 2.0), 3)}
                    out['dec=%.1f o=%.2f L=%.2f add=%s' % (dec, o, L, add)] = r
    finally:
        CL.Car.step = _ORIG_STEP
        DEC[0] = 1.5
    ok = all(v[1]['rear_v'] <= v[0]['rear_v'] + 0.05 and v[1]['back_after_camera_m'] <= v[0]['back_after_camera_m'] + 0.005
             for v in out.values())
    return {'ok': ok, 'cases': out}


def main():
    res = {}
    try:
        for name, f in (('r1-rs1-1', v_r1rs11), ('r1-rs1-2', v_r1rs12), ('r2-r2-rs-1', v_r2rs1), ('r2-r2-rs-2', v_r2rs2),
                        ('r3-r3-safety-1', v_r3s1), ('r3-r3-safety-2', v_r3s2)):
            res[name] = f()
            print(name, 'ok' if res[name]['ok'] else 'NG', {k: v for k, v in res[name].items() if k != 'cases'}, flush=True)
    finally:
        R.cleanup()
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'verify_1026.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
