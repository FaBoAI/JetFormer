# r3 の下調べ(経路いろいろ)の集計: 組ごとの有効・衝突・最小すき間・3 cm/5 cm 未満・追従の誤差・開始のそろえのフレーム・シールドのフレーム
import json, sys, glob, collections, statistics as stt
def load(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows
def q(a, p):
    a = sorted(a); 
    if not a: return None
    i = p * (len(a) - 1); lo = int(i); hi = min(lo + 1, len(a) - 1); return a[lo] + (a[hi] - a[lo]) * (i - lo)
for pat in sys.argv[1:]:
    rows = load(pat)
    by = collections.defaultdict(list)
    for r in rows: by[r.get('test') or r.get('set')].append(r)
    for k, rs in sorted(by.items()):
        n = len(rs); val = [r for r in rs if r.get('valid_episode')]; col = [r for r in rs if r.get('collisions')]
        cl = [r['min_clearance'] for r in val]
        trk = []
        for r in val:
            for fr in r.get('fr', []):
                if fr[1] is not None and fr[1] >= 3 and fr[5] is not None: trk.append(abs(fr[3] - fr[5]))
        al = sum(r['labels']['ph'].get('align', 0) for r in rs); sh = sum(r['labels']['ph'].get('shield', 0) for r in rs); fr = sum(r['labels']['frames'] for r in rs)
        lap1 = [r['laps'][0]['elapsed'] for r in val if r.get('laps')]
        print(f"{pat.split('/')[-2]:8s} {k:10s} n={n:3d} valid={len(val):3d} col={len(col)} min={min(cl) if cl else None!s:.6s} <3cm={sum(c<.03 for c in cl)} <5cm={sum(c<.05 for c in cl)} trk med/p95={q(trk,.5) or 0:.4f}/{q(trk,.95) or 0:.4f} align={100*al/max(fr,1):.2f}% shield={100*sh/max(fr,1):.2f}% lap1med={q(lap1,.5) or 0:.3f}")
        for r in sorted(rs, key=lambda r: r['min_clearance'])[:3]:
            print('     ', r['id'], 'g', r['start']['gate'], 'min', round(r['min_clearance'], 4), 't', round(r['min_t'], 2), 'col', r.get('collision'), 'dpsi', round(r['start_dpsi_vs_R'], 2), r.get('family'))
