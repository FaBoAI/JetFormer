"""fix5 係 safety: s3 の直しの代わりに起きること(後ろの当たり)の数え。fuzz_fix5.py と同じ場面(fuzz_closed の作り)のうち、
車両の値の止まりがある場面を、今(now1)・直す前の写し(bef1)・同じ場面から車両の値の止まりだけ抜いたもの(今 = now1_nv・
直す前 = bef1_nv)で流し、後ろの当たりの速さ・前の壁を押した秒を比べる。「直したあとの形が、値の止まりが無いときの形に近いか」
を見る。
    ... python -I s3_tradeoff.py 本数 種   出力: out/s3_tradeoff_<種>.json"""
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fuzz_fix5 as F          # noqa: E402
R, FC = F.R, F.FC


def main(n, seed):
    rng = random.Random(seed)
    BEF = R.load_snapshot('fix5_safety_before')
    out = {'seed': seed, 'scenes_with_veh_skip': 0, 'differ': 0, 'cnt': collections.Counter(), 'rows': []}
    orig = R.CL.Sim
    R.CL.Sim = FC.FSim
    try:
        for i in range(n):
            sc = FC.rand_scene(rng)
            p1 = FC.rand_params(rng)
            if not sc['veh_skip']:
                continue
            out['scenes_with_veh_skip'] += 1
            a = R.run_closed(p1, sc)
            b = R.run_closed(p1, sc, mod=BEF[0], amod=BEF[1])
            if a[2].w == b[2].w:
                continue
            out['differ'] += 1
            nv = dict(sc, veh_skip=[])
            an = R.run_closed(p1, nv)
            bn = R.run_closed(p1, nv, mod=BEF[0], amod=BEF[1])
            m = {k: FC.metrics(x[0], x[1]) for k, x in (('now1', a), ('bef1', b), ('now1_nv', an), ('bef1_nv', bn))}
            row = {'i': i, **{k + '_rear': round(v['rear_max_v'], 2) for k, v in m.items()},
                   **{k + '_push': v['front_push_s'] for k, v in m.items()}}
            out['rows'].append(row)
            c = out['cnt']
            ref = max(row['now1_nv_rear'], row['bef1_nv_rear'])
            c['now1_rear_worse_than_bef1'] += row['now1_rear'] > row['bef1_rear'] + 0.05
            c['bef1_rear_worse_than_now1'] += row['bef1_rear'] > row['now1_rear'] + 0.05
            c['now1_rear_worse_than_no_gap'] += row['now1_rear'] > ref + 0.05
            c['bef1_rear_worse_than_no_gap'] += row['bef1_rear'] > ref + 0.05
            c['now1_push_longer_than_bef1'] += row['now1_push'] > row['bef1_push'] + 0.2
            c['bef1_push_longer_than_now1'] += row['bef1_push'] > row['now1_push'] + 0.2
            c['now1_push_longer_than_no_gap'] += row['now1_push'] > max(row['now1_nv_push'], row['bef1_nv_push']) + 0.2
            c['bef1_push_longer_than_no_gap'] += row['bef1_push'] > max(row['now1_nv_push'], row['bef1_nv_push']) + 0.2
            print(i, row, flush=True)
    finally:
        R.CL.Sim = orig
        R.cleanup()
    out['cnt'] = dict(out['cnt'])
    (HERE / 'out' / ('s3_tradeoff_%d.json' % seed)).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, ensure_ascii=False))


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
