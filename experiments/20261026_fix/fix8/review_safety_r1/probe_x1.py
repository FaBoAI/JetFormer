"""10/23 の形(壁の手前でブレーキ側 → 止まって ESC が後退を待つ)の back_early の見立て: ON・OFF と、直し方の案(メモリの中だけ。
ファイルは変えない)を同じ場面で並べ、後ろの物の距離ごとの当たりの速さを出す。
  案 a: back_early のとき、下がった時間をこの手順の 'brake' を書き始めた時刻から数える(ESC は 'brake' を書いたときから後退で
        走らせていたかもしれない = esc_rev が True の手順)。後退側を出す長さが短くなり、後退の最高の速さが下がる
  案 b: back_early のとき、'gap' を飛ばさずに 'gap'(中立)へ進める(下がった時間は back_early のフレームから数え続ける。
        'gap' の惰性も数える = 見張りの今の決まり)。OFF と同じ「後退側 → 中立 → 後退側」の速さの形で、数え始めだけ早い
    python -I -B probe_x1.py → out/probe_x1.json・標準出力
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RS8_TAG', 'probe_x1')
sys.path.insert(0, str(HERE))
import h11  # noqa: E402
import exp11  # noqa: E402

A = h11.A


def patched_guard(orig_guard, orig_start):
    def _start(self, now):
        orig_start(self, now)
        self._t_start_x1 = now

    def _guard(self, erpm, now, dt, impact_g, rev_s, esc_rev=None):
        before = self.guard_event
        r = orig_guard(self, erpm, now, dt, impact_g, rev_s, esc_rev)
        ev = self.guard_event
        if ev is not before and ev is not None and ev[0] == 'back_early' and getattr(self, '_t_start_x1', None) is not None:
            self.backed_s = max(self.backed_s, now - self._t_start_x1)
        return r
    return _guard, _start


def run_fix(sd):
    mod, amod = h11.load('now')
    g, s = patched_guard(amod.Recovery._guard, amod.Recovery._start)
    return h11.run('now', h11.ON(exp11.params_of(sd)), exp11.build(sd),
                   patch={'Recovery._guard': g, 'Recovery._start': s})


def patched_guard_b(orig_guard):
    def _guard(self, erpm, now, dt, impact_g, rev_s, esc_rev=None):
        before = self.guard_event
        r = orig_guard(self, erpm, now, dt, impact_g, rev_s, esc_rev)
        ev = self.guard_event
        if ev is not before and ev is not None and ev[0] == 'back_early' and self.phase == 'reverse':
            self.backed_s += dt          # このフレームの後ろへの動き('reverse' が数えるはずだった分)
            self._go('gap', now)
            self.skip_gap = False
        return r
    return _guard


def run_fix_b(sd):
    mod, amod = h11.load('now')
    return h11.run('now', h11.ON(exp11.params_of(sd)), exp11.build(sd),
                   patch={'Recovery._guard': patched_guard_b(amod.Recovery._guard)})


def first_hit(m):
    sg = (m['segs'] or [{}])[0]
    hs = [h for h in m['rear'] if sg.get('t0', 0) <= h[0] <= (sg.get('t1') or 99)]
    return (-hs[0][2] if hs else None), sg.get('back'), sg.get('vmin')


def main():
    h11.check_now()
    out = {'rows': []}
    tally = {'on_worse': 0, 'fix_worse': 0, 'fixb_worse': 0, 'n': 0}
    try:
        for coast in (1.5, 0.3, 3.0):
            for esc in ('orig', 'human'):
                for rear in [None] + [round(0.5 + 0.1 * k, 2) for k in range(14)]:
                    sd = {'kind': 'rd', 'form': 'kai1', 'xf': 2.4, 'model': {'kind': 'brakewall', 'm': 120.0, 'db': 0.45},
                          'esc': esc, 'rear': rear, 'veh_dt': 0.01, 'dur': 5.5, 'coast': coast}
                    p = exp11.params_of(sd)
                    ms = {'on': h11.metrics(h11.run('now', h11.ON(p), exp11.build(sd)), 1.0),
                          'off': h11.metrics(h11.run('now', h11.OFF(p), exp11.build(sd)), 1.0),
                          'fix_a': h11.metrics(run_fix(sd), 1.0), 'fix_b': h11.metrics(run_fix_b(sd), 1.0)}
                    row = {'coast': coast, 'esc': esc, 'rear': rear,
                           **{k: first_hit(m) for k, m in ms.items()},
                           'guards_fix': ms['fix_a']['guards'][:4], 'front_push': {k: m['front_push_s'] for k, m in ms.items()},
                           'zero_push': {k: m['zero_push'] for k, m in ms.items()}}
                    out['rows'].append(row)
                    if rear is not None:
                        tally['n'] += 1
                        h_on, h_off, h_fx = row['on'][0], row['off'][0], row['fix_a'][0]
                        if h_on is not None and (h_off is None or h_on > h_off + 0.1):
                            tally['on_worse'] += 1
                        if h_fx is not None and (h_off is None or h_fx > h_off + 0.1):
                            tally['fix_worse'] += 1
                        h_fb = row['fix_b'][0]
                        if h_fb is not None and (h_off is None or h_fb > h_off + 0.1):
                            tally['fixb_worse'] += 1
                    print(coast, esc, rear, 'ON', row['on'], 'OFF', row['off'], 'fix_a', row['fix_a'], 'fix_b', row['fix_b'], flush=True)
    finally:
        h11.cleanup()
    out['tally'] = tally
    print('tally', tally)
    (HERE / 'out' / 'probe_x1.json').write_text(json.dumps(out, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
