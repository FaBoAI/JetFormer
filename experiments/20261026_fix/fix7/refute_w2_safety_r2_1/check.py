"""反証の打ち消し(w2-safety-rv7-safety-r2-1)を自分で確かめる(直さない・装置に触れない・GPU なし・API なし)。
review_safety_r2 の h8.py・g8.py(本物の drive._loop を閉じたループの偽物の ESC と車で回す)を読み込むだけ。
一時の場所はこの係の fix7/tmp_refute_w2_safety_r2_1(終わったら消す)。出力は ./out/check.json。
    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        experiments/20261026_fix/fix7/refute_w2_safety_r2_1/check.py
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RV = HERE.parent / 'review_safety_r2'
os.environ['RV8_TAG'] = 'refute_w2_s_r2_1'
sys.path.insert(0, str(RV))
import h8  # noqa: E402
from g8 import first_rec, scene  # noqa: E402

TMPROOT = HERE.parent / 'tmp_refute_w2_safety_r2_1'
h8.TMP = TMPROOT / 'h'
N = h8.N
TRACE = []


def _wrap(orig, ph, force=None):
    def step(self, model_us, erpm, now, impact_g=0.0, *a, **kw):
        if force:
            kw = dict(kw, **force)
        p0 = self.phase
        r = orig(self, model_us, erpm, now, impact_g, *a, **kw)
        TRACE.append((round(now - 100.0, 4), p0, self.phase, kw.get('hold_since') is not None,
                      kw.get('esc_rev', 'n/a'), getattr(self, 'esc_ran', 'n/a'),
                      round(float(getattr(self, 'seen_rev_s', 0.0)), 3), round(float(self.backed_s), 3),
                      round(float(erpm), 1), getattr(self, 'skip_gap', 'n/a')))
        ph.append(TRACE[-1])
        return r
    return step


h8._wrap = _wrap


def stats(res):
    sim, rec, pca, ph = res
    m = first_rec(res)
    tb, te = m['t_brake'], m['t_escape']
    ws = [(t, th) for t, s, th in pca.w if tb <= t <= (te if te is not None else tb + 3.0)]
    rev_side = sum(t1 - t0 for (t0, th), (t1, _) in zip(ws, ws[1:]) if th > N + 40)
    return {'back_m': m['back'], 'escape_push_s': m['esc_push'], 'rev_side_written_s': round(rev_side, 2),
            't_brake': tb, 't_escape': te,
            'guard_events': [(e[1]['what'], e[1]['phase']) for e in rec.events if e[0] == 'recovery_guard'],
            'fake_esc_first_reverse_t': next((t for t, r, c in sim.car.revlog if r), None),
            'front_push_s': m['front_push']}


def trace_rows(res, t0, t1):
    sim, rec, pca, ph = res
    rows = []
    for (t, p0, p1, hs, er, eran, seen, bs, e, sk) in ph:
        if t0 <= t <= t1:
            rows.append('%.3f %-7s->%-7s hold=%d esc_rev=%-5s ran=%-5s seen=%.3f backed=%.3f erpm=%7.1f skip=%s' %
                        (t, p0, p1, hs, er, eran, seen, bs, e, sk))
    w = ['%.3f %+.1f' % (t, th - N) for t, s, th in pca.w if t0 <= t <= t1]
    xs = [(round(t, 2), round(x, 3)) for t, x, *_ in sim.log if t0 <= t <= t1 and round(t * 100) % 10 == 0]
    return {'steps': rows, 'pca_thr_minus_n': w, 'x': xs}


def main():
    p = h8.base_params()
    base = scene('rev', dur=9.0)
    drop = [(2.52, 2.62)]
    gl = [(2.68, 1322), (2.88, 1322)]
    runs = {
        'on': ('now', h8.ON(p), dict(base, drops=drop, glitch=gl), None),
        'off': ('now', h8.OFF(p), dict(base, drops=drop, glitch=gl), None),
        'f7b_on': ('f7b', h8.ON(p), dict(base, drops=drop, glitch=gl), None),
        'on_nohold': ('now', h8.ON(p), dict(base, glitch=gl), None),
        'on_one': ('now', h8.ON(p), dict(base, drops=drop, glitch=gl[:1]), None),
        'on_noglitch': ('now', h8.ON(p), dict(base, drops=drop), None),
        'off_noglitch': ('now', h8.OFF(p), dict(base, drops=drop), None),
        'on_longdrop': ('now', h8.ON(p), dict(base, drops=[(2.52, 2.74)], glitch=gl), None),
        # 真因の確かめ(ファイルは変えない。メモリの中だけ): EscMayReverse の READY_NEUTRAL_S を 0.08 にしたら
        'on_ready008': ('now', h8.ON(p), dict(base, drops=drop, glitch=gl), 0.08),
        # 偽物の ESC が中立 0.02 秒で下がれる(本物に 0.02 秒で入った例がある)とき: 今の ON / 0.08 にした ON / OFF
        'on_esc002': ('now', h8.ON(p), dict(base, drops=drop, glitch=gl, esc=(0.02, 0.08)), None),
        'on_esc002_ready008': ('now', h8.ON(p), dict(base, drops=drop, glitch=gl, esc=(0.02, 0.08)), 0.08),
        'off_esc002': ('now', h8.OFF(p), dict(base, drops=drop, glitch=gl, esc=(0.02, 0.08)), None),
    }
    out = {'sha_now': h8.check_now()}
    traces = {}
    orig_ready = h8.A.EscMayReverse.READY_NEUTRAL_S
    try:
        for k, (name, prm, sc, ready) in runs.items():
            h8.A.EscMayReverse.READY_NEUTRAL_S = orig_ready if ready is None else ready
            try:
                res = h8.run(name, prm, sc)
            finally:
                h8.A.EscMayReverse.READY_NEUTRAL_S = orig_ready
            out[k] = stats(res)
            print(k, json.dumps(out[k], ensure_ascii=False), flush=True)
            if k in ('on', 'off', 'on_one', 'on_ready008', 'on_esc002'):
                traces[k] = trace_rows(res, 2.45, 4.6)
    finally:
        h8.cleanup()
        for q in (TMPROOT,):
            try:
                q.rmdir()
            except OSError:
                pass
    out['traces'] = traces
    (HERE / 'out' / 'check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
