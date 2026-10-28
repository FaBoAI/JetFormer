"""classify11 の A・D の組(ON の後ろの当たりが OFF より多い・速い)を ON で流し直し、その当たりの前に、同じリカバリーの中で
見張り・残すが手順を変えたか(recovery_guard の出来事。back_no_esc は出す値を変えないので除く。当たったあとの rear_impact も除く)
を見る。変えていなければ、その当たりは OFF と同じ手順(ふつうのリカバリー)の中の当たり = OFF が乱れでリカバリーを失った・
走りの長さで切れた などの「OFF の得」。
    python -I -B dcheck11.py [kind]  → out/dcheck11_<kind>.json
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RS8_TAG', 'dcheck')
sys.path.insert(0, str(HERE))
import h11  # noqa: E402
import exp11  # noqa: E402


def ev_times(rec):
    t0 = h11.clock.to_epoch_us(h11.T0) / 1e6
    return [(round(float(e[1]['cam_epoch_us']) / 1e6 - t0, 3), e[1]['what'], e[1]['phase'])
            for e in rec.events if e[0] == 'recovery_guard']


def main(kind='fz'):
    d = json.loads((HERE / 'out' / ('classify11_%s.json' % kind)).read_text())
    out = []
    cnt = {}
    try:
        for r in d['rows']:
            if r['cat'] not in ('A', 'D'):
                continue
            sd = r['sd']
            res = h11.run('now', h11.ON(exp11.params_of(sd)), exp11.build(sd), wrap=False)
            m = h11.metrics(res, sd.get('ai_from', 1.0))
            off_hits = r['off']['rear']
            evs = ev_times(res[1])
            # ON の当たりのうち、OFF に同じ時刻(± 0.3 秒)の当たりが無いか、0.1 m/s より速いもの
            bad = []
            for h in m['rear']:
                o = [x for x in off_hits if abs(x[0] - h[0]) <= 0.3]
                if not o or -h[2] > -o[0][2] + 0.1:
                    bad.append(h)
            verdict = 'plain'
            why = []
            for h in bad:
                sg = next((s for s in m['segs'] if s['t0'] <= h[0] <= (s['t1'] if s['t1'] is not None else 99)), None)
                t_lo = sg['t0'] - 0.05 if sg else h[0] - 3.0
                ch = [e for e in evs if t_lo <= e[0] < h[0] - 0.005 and e[1] != 'back_no_esc']
                if ch:
                    verdict = 'guard_changed'
                    why.append((h, ch))
            cnt[verdict] = cnt.get(verdict, 0) + 1
            row = {'i': sd.get('i'), 'cat': r['cat'], 'verdict': verdict, 'bad_hits': bad, 'why': why, 'events': evs[:12],
                   'sd': sd}
            out.append(row)
            print(sd.get('i'), r['cat'], verdict, bad, why[:2], flush=True)
    finally:
        h11.cleanup()
    print('done', cnt)
    (HERE / 'out' / ('dcheck11_%s.json' % kind)).write_text(json.dumps({'counts': cnt, 'rows': out}, ensure_ascii=False,
                                                                       default=str))


if __name__ == '__main__':
    main(*sys.argv[1:2])
