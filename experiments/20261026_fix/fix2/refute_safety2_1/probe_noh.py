"""safety2-1 の確かめ: 'brake' の段で下がり始め(10/23 205458-0002 の eRPM の形)、後ろに当たらないとき、今の Recovery が
後退側を出す時間と、出し終わるまでの距離の目安(後退側の間は eRPM が記録の形で増え 4,600 で頭打ち、coast は 4,600 のまま、
という決めた列。閉ループの模型ではない)。比べ: 'reverse' の段で下がり始める普通の形(10/23 211828-0007 の形)。読むだけ"""
import sys
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import assist as A   # noqa: E402
N, C, DT, KV = 1496.0, 1441.0, 1 / 30, 0.5e-3
SHAPE = [1455, 1864, 1909, 2258, 2519, 2876, 3207, 3495, 3600, 3800, 4000, 4100, 4250, 4400, 4550, 4600]


def run(start_back_s):
    a = A.Assist((C, N)); t = 100.0
    for _ in range(10):
        a.step((C, N - 100), -2000, True, t); t += DT
    while a.rec.phase is None:
        a.step((C, N + 20), 0, True, t); t += DT
    s, k, dist, rev_back, back_s = 0.0, 0, 0.0, 0.0, 0.0
    e = 0
    last_out = N + 300
    for i in range(1, 120):
        s = i * DT
        if s >= start_back_s:
            e = SHAPE[min(k, len(SHAPE) - 1)] if last_out > N + 40 else e     # 後退側の間だけ速くなる
            k += 1 if last_out > N + 40 else 0
        o, kind = a.step((C, N + 20), e, True, t); t += DT
        if e > 300:
            dist += e * KV * DT; back_s += DT
            if o[1] > N + 40:
                rev_back += DT
        last_out = o[1]
        if a.rec.phase == 'escape':
            break
    return {'start_back_s': start_back_s, 'reverse_side_while_backing_s': round(rev_back, 2), 'backing_until_escape_s': round(back_s, 2),
            'dist_until_escape_m': round(dist, 2), 'erpm_at_escape': e, 'escape_at_s': round(s, 2)}


print('brake の段で下がる(バグの形):', run(0.1))
print('reverse の段で下がる(普通の形):', run(0.53))
