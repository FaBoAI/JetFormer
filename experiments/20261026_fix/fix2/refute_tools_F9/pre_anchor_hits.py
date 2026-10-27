"""指摘 tools:F9 の確かめ(読むだけ): 走りながらの周で、起点の手前 d m(線 → 起点 = 2 周目の線〜A2 と同じ場所)に
入る衝突がどれだけあるか。周 = collect.laps_of(続いた周)。d = 5・6・7 m。"""
import sys
import numpy as np
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261026_fix/diag_launch')
import collect as C  # noqa: E402
le = C.le
days = tuple(sys.argv[1:]) or ('20261023', '20261024')
blocks, names = C.bench_blocks()
n_laps = 0
cnt = {5: [0, 0], 6: [0, 0], 7: [0, 0]}     # [周の数で当たりあり, 当たりの数]
locs = {}
all_hits_lap = 0
for log in C.logs(days):
    try:
        d = C.load(log, True, blocks, names)
    except Exception as e:  # noqa: BLE001
        print('skip', log, e)
        continue
    if not d['z']['ai'].any():
        continue
    vt, Df = d['vt'], d['Df']
    D = lambda x: float(np.interp(x, vt, Df))  # noqa: E731
    for a, b in C.laps_of(d):
        n_laps += 1
        hl = [h for h in d['hits'] if a <= h[0] < b]
        all_hits_lap += len(hl)
        for dd in cnt:
            m = (vt >= a) & (vt <= b)
            tl = float(np.interp(D(b) - dd, Df[m], vt[m]))
            ex = [h for h in hl if h[0] >= tl]
            if ex:
                cnt[dd][0] += 1
                cnt[dd][1] += len(ex)
                if dd == 7:
                    lc = le.locate([h[0] for h in ex], d['anchors'], d['gm'], d['it'], d['hd'], d['gzs'])
                    for h, k in zip(ex, lc):
                        locs[le.LOC_DETAIL[int(k)]] = locs.get(le.LOC_DETAIL[int(k)], 0) + 1
                        print(log, 'lap %.2f-%.2f' % (a, b), 'hit %.2f (起点の %.2f 秒前) %.1fG %s' % (h[0], b - h[0], h[1], le.LOC_DETAIL[int(k)]))
print('周', n_laps, '周の中の衝突', all_hits_lap, '起点の手前 d m に当たりのある周 [周, 本]', cnt, locs)
