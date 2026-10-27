"""反証 3 回目 refute3_safety の X9(今日の選択肢を全部 ON に混ぜたでたらめな閉じたループ 150 本・種 1103)と X9b(長く見ていない
中立止めのあと 1.6 秒以内に後退側を書いた所の分け方)を、直す 3 回目・1 組目の直しのあとの assist.py で流し直す。台本の場面・決まりは
そのまま読み込み、出力と一時置き場だけこの組の場所にする(反証の out/ は書き換えない)。装置・GPU・API に触れない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        experiments/20261026_fix/fix3/fix_r3_1/x9_rerun.py
出力: out/x9_rerun.json(反証の数: X9 の R4 37 本・X9b の 32 か所。どれも中立止めの前の段は None)
"""
import collections
import json
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'refute3_safety'))
sys.path.insert(0, str(HERE.parent / 'refute2_safety'))
import cl3                                    # noqa: E402
from cl3 import A, N, T0                      # noqa: E402
import fuzz_cl as F                           # noqa: E402
import x9_fuzz as X                           # noqa: E402

cl3.TMP = HERE.parent / 'tmp_fixr31_x9'


def main(n=150, seed=1103):
    rng = random.Random(seed)
    flags, cls, ex = collections.Counter(), collections.Counter(), []
    halt_rows = giveup_runs = long_holds = 0
    r4 = []
    for i in range(n):
        sc_ = F.scene_of(rng)
        p = X.params_on(rng)
        sim, rec, pca, ph = cl3.run(p, sc_)
        fl = X.check(sim, rec, pca, p, sc_)
        for k in fl:
            flags[k] += 1
        if 'R4' in fl:                 # R4 の後退側の行の種類(リカバリーか、モデル・区間のブレーキか)と、明けた所の速さ
            for te, dt_ in fl['R4']:
                kinds = collections.Counter(A.ASSIST_NAMES[int(r['assist'])] or 'なし(モデル)' for t, r in rec.rows
                                            if not r['hold'] and te + 0.6 <= (t - T0) / 1e9 <= te + 1.6 and r['out'][1] > N + 40)
                v_te = next((r[2] for r in sim.log if r[0] >= te), None)
                r4.append({'i': i, 'end': te, 'first_after': dt_, 'kinds': dict(kinds), 'v_at_end': v_te,
                           'hits_after': [h for h in sim.car.hits if te <= h[0] <= te + 1.6]})
        rows = [(t, r) for t, r in rec.rows if not r['hold']]
        giveup_runs += any(int(r['assist']) == A.ASSIST_GIVEUP for t, r in rows)
        for (ta, ra), (tb, rb) in zip(rows, rows[1:]):
            ga, gb = (ta - T0) / 1e9, (tb - T0) / 1e9
            if gb - ga <= 0.5 or not (ra['select'] and rb['select']):
                continue
            long_holds += 1
            # 明けた直後の 'halt'(前進 130 µs・種類はリカバリー)の行
            halt_rows += sum(1 for t, r in rows if gb <= (t - T0) / 1e9 <= gb + 0.35 and int(r['assist']) == A.ASSIST_RECOVERY
                             and r['out'][1] == N - A.Recovery.ESCAPE_US)
            rev = [(t - T0) / 1e9 for t, r in rows if int(r['assist']) == A.ASSIST_RECOVERY and r['out'][1] > N + 40
                   and gb <= (t - T0) / 1e9 <= gb + 1.6]
            if not rev:
                continue
            before = [x for x in ph if x[0] <= ga + 0.02]
            v_a = next((r[2] for r in sim.log if r[0] >= ga), None)
            v_b = next((r[2] for r in sim.log if r[0] >= gb), None)
            hits_in = [h for h in sim.car.hits if ga <= h[0] <= gb]
            key = (before[-1][2] if before else 'no-step', 'moving' if v_a is not None and abs(v_a) > 0.3 else 'slow',
                   'stopped' if v_b is not None and abs(v_b) < 0.05 else 'moving', 'hit-in-gap' if hits_in else 'no-hit')
            cls[' / '.join(map(str, key))] += 1
            if len(ex) < 12:
                ex.append({'i': i, 'gap': (round(ga, 3), round(gb, 3)), 'first_rev_after': round(rev[0] - gb, 3), 'key': key,
                           'v_after': v_b, 'hits_in': hits_in})
    res = {'n': n, 'seed': seed, 'x9_flags': dict(flags), 'x9b_classes': dict(cls), 'x9b_total': sum(cls.values()),
           'long_holds': long_holds, 'halt_rows_after_long_hold': halt_rows, 'giveup_runs': giveup_runs, 'examples': ex, 'r4': r4,
           'refuter': {'x9_flags': {'R4': 37}, 'x9b_total': 32}}
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'x9_rerun.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(json.dumps({k: v for k, v in res.items() if k != 'examples'}, ensure_ascii=False))
    for x in ex:
        print(x)
    for x in r4:
        print('R4', x)


if __name__ == '__main__':
    try:
        main()
    finally:
        shutil.rmtree(cl3.TMP, ignore_errors=True)
