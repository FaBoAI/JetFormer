"""X9: 今日の選択肢を全部混ぜたでたらめな閉じたループ。新しい決まりで見る:
 R4 残した中立止めが 0.5 秒より長かった(見ていない)あと 0.6〜1.6 秒の間に、AI 側で後退側(> 中立 + 40)を書いた
 R5 前進がどの部品の上限(モデル × 掛けたゲインの最大・launch_us・ESCAPE 130・KICK 80)より強い
 R3 launch_boost = 1・launch_no_kick = 1 なのに StartKick の行
 R7 書いた値が数でない・1000〜2000 の外"""
import collections
import json
import math
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'refute2_safety'))
import cl3                                    # noqa: E402
from cl3 import A, C, N, PARAMS, T0           # noqa: E402
import fuzz_cl as F                           # noqa: E402


def params_on(rng):
    p = F.params_of(rng)
    p.update(rec_back_guard=1, rec_hold_keep=1, launch_boost=rng.choice([0, 1]), start_kick=rng.choice([0, 1]),
             speed_ceiling=rng.choice([0, 1]))
    p['launch_no_kick'] = rng.choice([0, 1]) if p['launch_boost'] else 0
    p['speed_auto'] = rng.choice([0, 1]) if p['speed_ceiling'] else 0
    p['launch_us'] = rng.choice([150.0, 250.0, 300.0])
    return p


def check(sim, rec, pca, p, sc_):
    flags = {}
    # 中立止めの区間(書いた値の時刻の空き)を pca.w から: hold の行(中立)が続いた・書かなかった時間
    rows = rec.rows
    # 残した中立止めの長さ: 処理したフレームの行(hold False)の時刻の間
    proc = [(t, r) for t, r in rows if not r['hold']]
    long_ends = []
    for (ta, ra), (tb, rb) in zip(proc, proc[1:]):
        if (tb - ta) / 1e9 > 0.5 and ra['select'] and rb['select']:
            long_ends.append((tb - T0) / 1e9)
    for te in long_ends:
        rev = [t for t, s, th in pca.w if te + 0.6 <= t <= te + 1.6 and th > N + 40 and sim.sel(t)]
        if rev:
            flags.setdefault('R4', []).append((round(te, 3), round(rev[0] - te, 3)))
    gmax = max([r['gains'][1] for t, r in rows if not r['hold']] or [p['throttle_gain']])
    k = float(sc_['model_desc'].split(',')[0].split()[1])
    allowed = max(max(k, 0) * gmax, p['launch_us'] if p['launch_boost'] else 0, A.Recovery.ESCAPE_US, A.StartKick.KICK_US) + 1.0
    strong = [(round(t, 3), th) for t, s, th in pca.w if th < N - allowed and sim.sel(t)]
    if strong:
        flags['R5'] = strong[:3]
    if p['launch_boost'] and p['launch_no_kick']:
        kr = sum(1 for t, r in rows if not r['hold'] and int(r['assist']) == A.ASSIST_START)
        if kr:
            flags['R3'] = kr
    bad = [(t, s, th) for t, s, th in pca.w if not (math.isfinite(s) and math.isfinite(th) and 1000 <= th <= 2000 and 1000 <= s <= 2000)]
    if bad:
        flags['R7'] = bad[:3]
    return flags


def main(n=150, seed=1103):
    rng = random.Random(seed)
    cnt = collections.Counter()
    ex = collections.defaultdict(list)
    for i in range(n):
        sc_ = F.scene_of(rng)
        p = params_on(rng)
        sim, rec, pca, ph = cl3.run(p, sc_)
        fl = check(sim, rec, pca, p, sc_)
        for k, v in fl.items():
            cnt[k] += 1
            if len(ex[k]) < 5:
                ex[k].append({'i': i, 'v': v, 'scene': {kk: vv for kk, vv in sc_.items() if kk != 'model'},
                              'params': {kk: p[kk] for kk in ('start_kick', 'launch_boost', 'launch_no_kick', 'launch_us', 'tof_guard',
                                                              'speed_ceiling', 'speed_auto', 'zone_caps', 'throttle_gain', 'rev_us', 'rev_s')},
                              'events': [(e[0], e[1].get('what') or e[1].get('event')) for e in rec.events][:12]})
    res = {'n': n, 'seed': seed, 'flags': dict(cnt), 'examples': dict(ex)}
    o = Path(__file__).resolve().parent / 'out'
    o.mkdir(exist_ok=True)
    (o / ('x9_fuzz_%d.json' % seed)).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(json.dumps({'n': n, 'flags': dict(cnt)}, ensure_ascii=False))
    shutil.rmtree(cl3.TMP, ignore_errors=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150, int(sys.argv[2]) if len(sys.argv) > 2 else 1103)
