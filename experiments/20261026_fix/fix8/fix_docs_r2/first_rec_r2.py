"""fix8 係 docs 直す 8-2 回目(f8r2-docs-1023-first-recovery-off)。直さない・装置に触れない・API を呼ばない。

main session の決め(10/28 11:xx): その日を OFF にする「おかしな動き」は、ふつうのリカバリーでは起きない形だけ。数は閉じたループ
(票 A の台本 refute_docs_r2_A/sweep_first_rec.py と同じ場面・fix8/refute_safety_r1_A の ESC の模型 18 通り)と本物の記録の
ふつうの後退の幅から、ふつうの ON のリカバリーが当てはまらず、知られている悪い形(0 m の壁押し・止まらない後退)が当てはまるように決める。

本物の drive._loop(今の jetformer/。sha256 を照らす)を cl_r2.py(= refute_docs_r2_A/cl_mine.py の写し。TMP だけ自分の場所)の
閉じたループの偽物の ESC で回す。後ろの物は置かない。形は票 A と同じ 2 つ:
  '1023' : 10/23 の形(前進 m µs → 壁 xf の db 手前からブレーキ側 → 止まって ESC が後退を待つ → リカバリー)。16 場面
  'wall' : ふつうの形(前進 m µs のまま前の壁に当たって止まる → リカバリー)。8 場面
ESC の組(set):
  normal18 : cl_r2.ESC の 18 通り(票 A・fix8 safety r1 と同じ)
  gapslow  : fit・cl・human の中立の受け付け(neu_ready)を 0.2・0.25 秒に = 最初の 'gap' 0.15 秒では下がれず、やり直しの 'gap'
             0.3 秒で下がる ESC(10/10 のログで最初の 'gap' では下がれないのは 11%。approvals 14 の 1 行目)= これもふつう
  k1_noback: 知られている悪い形 1「0 m の壁押し」= fit・cl・human の neu_ready を 0.35 秒(やり直しの 0.3 秒でも下がれない)・99 秒
             (後退を受け付けない)。リカバリーは後退側を 3 回出しても下がらず、escape(前進)が前の壁を押す
  k2_runaway: 知られている悪い形 2「止まらない後退」= fit・cl・human で、車輪の値(eRPM)が後ろへ動いている間 0 のまま(後ろへの
             動きが見えない)。リカバリーは下がったとみないので、後退側を強めながら 3 回出す
  k3_limits: 入れ間違い(rev_s 1.2 秒・rev_us 500 µs = PARAM_LIMITS の上限)の fit・cl・human(参考)
1 本ごとに(後ろの物なし)、最初のリカバリーの始め t0 の位置 x0 から:
  back_m    = 始めからリカバリーが終わる(段が None に戻る)まで + 0.3 秒の間に、いちばん後ろへ行った距離 [m]
  stop_t    = t0 の前で最後に前へ 0.15 m/s(Recovery.STOP_ERPM 300)より速かった時刻(= 止まった時刻)。noback_s = 止まってから、
              t0 の位置より 0.05 m 下がるまでの秒(走りの終わりまで下がらなければ None)
  push_noback_s = 止まってから(下がる前の間)に前進側(< 中立 − 20)を出して前の壁に着いていた秒(モデルの押しも入れる)
  esc_push_noback_s = そのうちリカバリーが始まったあと(escape の前進)の秒
  n_rec / giveup = 走り全体のリカバリーの始めの数 / 中止(giveup)になったか
    jf-heavy … python -I -B first_rec_r2.py [set ...] → out/first_rec_r2_<set>.json(1 つずつ流せる)
"""
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cl_r2 as cl  # noqa: E402

NOW = {'drive.py': '3f2e31a22ecb9abdd9c8f84b3f787d075ef636b6d5679bad5c0fc28aee18388e',
       'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'}
C, N = cl.C, cl.N
CL = cl.CL


def esc_sets():
    base = {k: cl.ESC[k] for k in ('fit', 'cl', 'human')}
    out = {'normal18': {k: v for k, v in cl.ESC.items()}}
    out['gapslow'] = {'%s_neu%s' % (k, n): dict(v, neu_ready=n) for k, v in base.items() for n in (0.2, 0.25)}
    out['k1_noback'] = {'%s_neu%s' % (k, n): dict(v, neu_ready=n) for k, v in base.items() for n in (0.35, 99.0)}
    out['k2_runaway'] = {'%s_erpm0back' % k: dict(v) for k, v in base.items()}
    out['k3_limits'] = {'%s_rev1.2_500' % k: dict(v) for k, v in base.items()}
    return out


SET_OPTS = {'normal18': {}, 'gapslow': {}, 'k1_noback': {'dur_add': 9.0}, 'k2_runaway': {'fault': 'zero_back'},
            'k3_limits': {'params': {'rev_s': 1.2, 'rev_us': 500.0}}}


def scenes():
    for m, (xf, db), bu, vdt in itertools.product((120.0, 200.0), ((2.4, 0.45), (3.0, 0.6)), (200.0, 100.0), (0.01, 0.02)):
        yield {'form': '1023', 'm': m, 'xf': xf, 'db': db, 'brake_us': bu, 'veh_dt': vdt, 'dur': 8.5}
    for m, xf, vdt in itertools.product((120.0, 200.0), (2.4, 3.0), (0.01, 0.02)):
        yield {'form': 'wall', 'm': m, 'xf': xf, 'veh_dt': vdt, 'dur': 9.0}


def fwd_model(m):
    def f(t, car):
        return (C, N - m)
    return f


def _run(params, sc, fault=None):
    """cl_r2._run と同じ(車両の値の間 veh_dt)に、車輪の値の壊れ(fault)を足した版。fault 'zero_back' = 後ろへ動いている間 eRPM 0"""
    vdt = sc.get('veh_dt', 0.02)
    orig = CL.Sim._advance
    np = cl.np

    def adv(self, t_end):
        while self.t + cl.DT <= t_end + 1e-9:
            self.t = round(self.t + cl.DT, 6)
            acc, g = self.car.step(self.t, self.pca.thr)
            t_ns = CL.T0 + int(round(self.t * CL.NS))
            ay = acc + g * 9.80665
            self.imu_buf.append((t_ns, np.array([0.0, ay, 9.8, 0.0, 0.0, 0.0], np.float32)))
            if self.t >= self.next_veh - 1e-9:
                self.next_veh += vdt
                sl = self.sel(self.t)
                erpm = int(round(-self.car.v / CL.KV))
                if fault == 'zero_back' and self.car.v < 0:
                    erpm = 0
                self.veh_buf.append((t_ns, {'pwm': [1441, 1496, 2000 if sl else 1000], 'valid': [True, True, True],
                                            'select': sl, 'moving': abs(self.car.v) > 0.01, 'erpm': erpm}))
            if self.t >= self.next_tof - 1e-9:
                self.next_tof += 1.0 / 30
                dist = self.s['xf'] - self.car.x
                self.tof_buf.append((t_ns, CL.tof_near(int(dist * 1000)) if 0.05 < dist < 1.95 else CL.TOF_FREE))
            self.log.append((self.t, self.car.x, self.car.v, self.pca.thr, self.car.contact, self.car.rev_on,
                             self.sel(self.t)))
        self.box['ns'] = CL.T0 + int(round(self.t * CL.NS))
        self.box['m'] = self.t + 100.0
        self.d.heartbeat_t = self.box['m']
    CL.Sim._advance = adv
    try:
        return CL.run(params, sc)
    finally:
        CL.Sim._advance = orig


def run_one(esc, sw, sc, opts):
    dur = sc['dur'] + opts.get('dur_add', 0.0)
    if sc['form'] == '1023':
        s = {'dur': dur, 'xf': sc['xf'], 'ai_from': 1.0, 'model': cl.brakewall(sc['m'], sc['xf'], sc['db'], sc['brake_us']), 'esc': esc}
    else:
        s = {'dur': dur, 'xf': sc['xf'], 'ai_from': 1.0, 'model': fwd_model(sc['m']), 'esc': esc}
    s['veh_dt'] = sc['veh_dt']
    p = cl.base_params(**cl.SW[sw])
    p.update(opts.get('params') or {})
    sim, rec, pca = _run(p, s, opts.get('fault'))
    ph = list(CL.PH)
    return metrics(sim, rec, ph)


def metrics(sim, rec, ph):
    log = sim.log
    starts = [t for t, p0, p1, *_ in ph if p0 is None and p1 == 'brake']
    giveup = any(p1 == 'giveup' for t, p0, p1, *_ in ph)
    if not starts:
        return {'rec': None, 'n_rec': 0, 'giveup': giveup}
    t0 = starts[0]
    t1 = next((t for t, p0, p1, *_ in ph if t > t0 and p0 is not None and p1 is None), None)
    te = (t1 if t1 is not None else log[-1][0]) + 0.3
    x0 = next(r[1] for r in log if r[0] >= t0 - 1e-9)
    seg = [r for r in log if t0 - 1e-9 <= r[0] <= te + 1e-9]
    back = x0 - min(r[1] for r in seg)
    # 止まった時刻 = t0 の前で最後に前へ 0.15 m/s(Recovery.STOP_ERPM 300 × 0.5 mm/s)より速かった時刻の次(本物の記録と同じ決め)。
    # 下がり始め = t0 のあとで、t0 の位置から 0.05 m 下がった最初の時刻(リカバリーの前の跳ね返りは数えない)
    pre = [r for r in log if r[0] < t0 - 1e-9 and r[2] > 0.15]
    stop_t = (pre[-1][0] + cl.DT) if pre else t0
    t_b = next((r[0] for r in log if r[0] >= t0 - 1e-9 and r[1] <= x0 - 0.05), None)
    push = esc_push = 0.0
    for r in log:
        if r[0] < stop_t - 1e-9 or (t_b is not None and r[0] >= t_b - 1e-9):
            continue
        if r[4] == 'front' and r[3] < N - 20:
            push += cl.DT
            if r[0] >= t0 - 1e-9:
                esc_push += cl.DT
    t_1m = next((round(r[0] - t0, 3) for r in seg if r[1] <= x0 - 1.0), None)
    return {'rec': [round(t0, 3), None if t1 is None else round(t1, 3)], 'back_m': round(back, 3), 'stop_t': round(stop_t, 3),
            'noback_s': None if t_b is None else round(t_b - stop_t, 3), 'push_noback_s': round(push, 3),
            'esc_push_noback_s': round(esc_push, 3), 't_1m': t_1m, 'n_rec': len(starts), 'giveup': giveup,
            'rec_starts': [round(t, 3) for t in starts[:6]], 'dur': round(log[-1][0], 2),
            'events': [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard'][:8]}


def main():
    for fn, h in NOW.items():
        assert hashlib.sha256((cl.ROOT / 'jetformer' / fn).read_bytes()).hexdigest() == h, fn
    sets = esc_sets()
    want = sys.argv[1:] or list(sets)
    try:
        for name in want:
            res, t_s = [], time.time()
            for ek, esc in sorted(sets[name].items()):
                for sc in scenes():
                    row = {'set': name, 'esc': ek, **{k: v for k, v in sc.items() if k != 'dur'}}
                    for sw in ('on', 'off'):
                        row[sw] = run_one(esc, sw, sc, SET_OPTS[name])
                    res.append(row)
                print(name, ek, len(res), '%.0fs' % (time.time() - t_s), flush=True)
            (HERE / 'out' / ('first_rec_r2_%s.json' % name)).write_text(json.dumps(
                {'set': name, 'esc': sets[name], 'opts': SET_OPTS[name], 'sha': NOW, 'rows': res}, ensure_ascii=False))
            print('出力', HERE / 'out' / ('first_rec_r2_%s.json' % name), flush=True)
    finally:
        cl.cleanup()


if __name__ == '__main__':
    main()
