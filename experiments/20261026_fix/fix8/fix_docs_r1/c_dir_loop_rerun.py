"""(fix8 docs r1 が票 A の台本 refute_docs_r1_A/c_dir_loop.py を写して流し直す。変えたのは出力の名前と、短い・同じの数を足したことだけ)
打ち消しの票 A(f8r1-docs-residual2-reversed の (1)): 「'gap' を飛ばした後退で、本当に下がり始めたのに 0.2 秒続く前に一度
車輪の値が 300 以下 → 0.35 秒のあとなら、やり直して下がった時間を捨て、最大 0.2 秒の後退ぶん多く下がる」(approvals 14 の 2 行目)を、
今の jetformer/assist.py の Recovery(見張り ON)と、自分で書いた小さな ESC・車のモデル(指摘・案の h10/closed_loop は使わない)で照らす。
場面: 前の壁に当たって止まる → stuck → brake の途中で 0.12 秒の中立止め(カメラの止まり。drive は中立を書く)→ 明けたフレームに
車輪の値の跳ね 1322(back_early で 'gap' を飛ばした 'reverse')→ ESC は中立を見たので後退側から rd 秒で本当に下がり始める。
その途中の車輪の値の 1 サンプル(w=1)か 2 サンプル(w=2)を 0 にする(dip)。dip なし(clean)と比べる。
ESC のやり直し(early_none の 'gap' 0.15 秒の中立のあとの後退側)の 3 つのモデル:
  v1 = 中立で後退が切れ、もう一度 rd 秒待つ / v2 = 後退のまま、すぐ(0.03 秒)入る / v3 = まだ後ろへ動いていればすぐ、止まっていれば rd 秒。
数える: やり直し・early_none の数が clean と違う組の、下がった距離の差(dip − clean)と、escape が前の壁を押した秒。"""
import importlib.util
import json
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
spec = importlib.util.spec_from_file_location('assist_refA2', JF / 'jetformer' / 'assist.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
C, N = 1441.0, 1496.0
DT = 0.002            # 物理の刻み
FDT = 1.0 / 30        # フレーム
VDT = 0.02            # 車輪の値(50 Hz)
KV = 0.0005           # 1 eRPM = 0.0005 m/s


class Esc(object):
    def __init__(self, rd, variant, neu=0.03):
        self.rd, self.var, self.neu = rd, variant, neu
        self.braked = self.ready = self.rev_on = False
        self.rev_wait = self.neu_t = 0.0
        self.was_rev = False

    def accel(self, thr, v):
        """v: 前へ + m/s。戻り値 dv/dt"""
        u = thr - N
        if u < -20:                                   # 前進
            self.braked = self.ready = self.rev_on = self.was_rev = False
            self.rev_wait = self.neu_t = 0.0
            if v < 0:
                return 8.0
            vt = min(-u / 80.0 * 1.2, 6.0)
            return float(np.clip((vt - v) / 0.3, -6.0, 6.0))
        if u > 40:                                    # 後退側
            self.neu_t = 0.0
            if self.rev_on:
                vt = -(u / 300.0) * 1.8
                return float(np.clip((vt - v) / 0.3, -6.0, 6.0))
            if self.ready and v <= 0.05:
                need = self.rd
                if self.was_rev and self.var == 'v2':
                    need = 0.03
                if self.was_rev and self.var == 'v3' and v < -0.05:
                    need = 0.03
                self.rev_wait += DT
                if self.rev_wait >= need - 1e-9:
                    self.rev_on = True
                return 2.0 if v < 0 else (-2.0 if v > 0 else 0.0)
            self.braked = True
            return -6.0 if v > 0.01 else (6.0 if v < -0.01 else 0.0)
        # 中立
        self.rev_wait = 0.0
        self.neu_t += DT
        if self.braked and self.neu_t >= self.neu - 1e-9:
            self.ready, self.braked = True, False
        if self.rev_on:
            self.rev_on, self.ready, self.was_rev = False, True, True
        return 2.0 if v < -0.01 else (-2.0 if v > 0.01 else 0.0)


def run(rd, variant, dip=None, w=1, hold_len=0.12, t_end=6.0):
    esc = Esc(rd, variant)
    rec = A.Recovery((C, N))
    x, v = -0.3, 1.0                 # 壁(x = 0)の 0.3 m 手前を 1 m/s で前へ
    t = 0.0
    next_f, next_v = 0.0, 0.0
    erpm_s = 0.0
    out_thr = N - 60.0               # モデルの前進(弱い)
    model = (C, N - 60.0)
    hold = None                      # (始め, 終わり)
    t_last_proc = None
    hold_started = None
    glitch_at = None
    entry_t = None
    events = []
    min_x = 0.0
    push_s = 0.0
    dips = []
    early_n = 0
    retries_seen = 0
    while t < t_end:
        # 車輪の値(50 Hz)
        if t >= next_v - 1e-12:
            erpm_s = -v / KV
            if glitch_at is not None and abs(t - glitch_at) < VDT / 2:
                erpm_s = 1322.0
            for d0 in dips:
                if d0 - 1e-9 <= t < d0 + VDT * w - 1e-9:
                    erpm_s = 0.0
            next_v += VDT
        # フレーム(30 Hz)
        if t >= next_f - 1e-12:
            next_f += FDT
            in_hold = hold is not None and hold[0] <= t < hold[1]
            if in_hold:
                if hold_started is None:
                    hold_started = t
                out_thr = N                                  # 中立止め: drive は中立を書く
            else:
                kw = dict(back_guard=True, esc_rev=None)
                e_in = erpm_s
                if hold_started is not None:
                    kw.update(hold_since=hold_started, hold_back_s=0.0)
                    hold_started = None
                    e_in = 1322.0                            # 明けたフレームが見る車輪の値は跳ね(1 サンプル)
                ph0 = rec.phase
                o, kind = rec.step(model, e_in, t, **kw)
                out_thr = model[1] if o is None else o[1]
                if rec.guard_event:
                    events.append((round(t, 3), rec.guard_event[0], ph0))
                    if rec.guard_event[0] == 'early_none':
                        early_n += 1
                    if rec.guard_event[0] == 'back_early' and entry_t is None:
                        entry_t = t
                        if dip is not None:
                            dips = [entry_t + dip]
                    rec.guard_event = None
                retries_seen = max(retries_seen, rec.retries)
                # brake の 2 フレーム目から 0.12 秒の中立止め、明けたフレームの直前のサンプルに跳ね
                if rec.phase == 'brake' and hold is None and ph0 == 'brake':
                    hold = (t + 1e-6, t + hold_len)
                    glitch_at = None
                if rec.phase == 'escape' and x >= -1e-6 and out_thr < N - 20:
                    push_s += FDT
        a = esc.accel(out_thr, v)
        v += a * DT
        x += v * DT
        if x >= 0.0:
            x, v = 0.0, min(v, 0.0) if v > 0 else v
            if x >= 0.0 and v > 0:
                v = 0.0
        min_x = min(min_x, x)
        t += DT
    return {'back_m': round(-min_x, 3), 'push_s': round(push_s, 3), 'early_none': early_n, 'retries': retries_seen,
            'entry_t': entry_t, 'events': events}


def main():
    res = {}
    lines = []
    for var in ('v1', 'v2', 'v3'):
        for rd in (0.15, 0.20, 0.27, 0.35):
            c = run(rd, var)
            assert c['entry_t'] is not None, ('no back_early', rd, var, c['events'])
            changed = []
            for k in range(0, 41):
                dip = 0.30 + k * 0.01
                for w in (1, 2):
                    r = run(rd, var, dip=dip, w=w)
                    if (r['early_none'], r['retries']) != (c['early_none'], c['retries']):
                        changed.append((dip, w, r['back_m'], round(r['back_m'] - c['back_m'], 3), r['push_s'],
                                        r['early_none'], r['retries']))
            key = '%s rd=%.2f' % (var, rd)
            dd = [x[3] for x in changed]
            res[key] = {'clean': {k: c[k] for k in ('back_m', 'push_s', 'early_none', 'retries')},
                        'n_changed': len(changed), 'diff_min': min(dd) if dd else None, 'diff_max': max(dd) if dd else None,
                        'n_longer': sum(1 for x in dd if x > 0.005), 'n_shorter': sum(1 for x in dd if x < -0.005), 'n_same': sum(1 for x in dd if abs(x) <= 0.005), 'n_push': sum(1 for x in changed if x[4] > 0.05),
                        'push_max': max([x[4] for x in changed], default=0.0), 'min_back': min([x[2] for x in changed], default=None),
                        'examples': changed[:3]}
            lines.append('%s: clean back %.3f m push %.2f s | changed %d: diff %s..%s m, longer %d, push>0.05s %d (max %.2f s), '
                         'min back %s m, shorter %d, same %d' % (key, c['back_m'], c['push_s'], len(changed), res[key]['diff_min'],
                                            res[key]['diff_max'], res[key]['n_longer'], res[key]['n_push'],
                                            res[key]['push_max'], res[key]['min_back'], res[key]['n_shorter'], res[key]['n_same']))
            print(lines[-1], flush=True)
    (Path(__file__).resolve().parent / 'out' / 'c_dir_loop_rerun.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
