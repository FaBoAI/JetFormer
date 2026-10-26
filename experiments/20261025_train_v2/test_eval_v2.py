#!/usr/bin/env python3
"""eval_v2.py の決まりの関数の試し(numpy だけ。作った数で、protocol の判定の向きと境目を確かめる)

    /home/jetson/miniconda/envs/jetracer/bin/python -I experiments/20261025_train_v2/test_eval_v2.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_v2 as E  # noqa: E402

T = E.protocol()['thresholds_us']['values']
FIVE_B = [49.146, 49.457, 49.937, 50.389, 49.144]       # 5 本の H1 の L1(best、protocol_numbers.json)
FIVE_L = [49.235, 49.752, 50.646, 50.729, 49.091]
FB, FL = sum(FIVE_B) / 5, sum(FIVE_L) / 5


def arm(vals):
    return {s: v for s, v in enumerate(vals)}


def test_vs5():
    t4b, t4l = T['vs5']['n4']['l1_best'], T['vs5']['n4']['l1_last']
    # 全部の種が 5 本の平均より閾値以上小さい(best・last)→ 良い
    good = E.vs5(arm([FB - t4b - 0.1] * 4), arm([FL - t4l - 0.1] * 4), FIVE_B, FIVE_L, T['vs5'])
    assert good['verdict'] == '良い', good
    # best だけ良い → 同等(良いは両方)
    half = E.vs5(arm([FB - t4b - 0.1] * 4), arm([FL] * 4), FIVE_B, FIVE_L, T['vs5'])
    assert half['verdict'] == '同等', half
    # last だけ悪い → 悪い(悪いはどちらか)
    bad = E.vs5(arm([FB] * 4), arm([FL + t4l + 0.1] * 4), FIVE_B, FIVE_L, T['vs5'])
    assert bad['verdict'] == '悪い', bad
    # 平均は閾値を超えて悪いが、1 本だけ 5 本の平均より小さい → 同等
    mixed = E.vs5(arm([FB + 3, FB + 3, FB + 3, FB - 0.01]), arm([FL] * 4), FIVE_B, FIVE_L, T['vs5'])
    assert mixed['verdict'] == '同等', mixed
    # 種が 3 本なら n3 の閾値
    three = E.vs5(arm([FB] * 3), arm([FL] * 3), FIVE_B, FIVE_L, T['vs5'])
    assert abs(three['thr_best'] - T['vs5']['n3']['l1_best']) < 1e-9


def test_pairs():
    k4 = T['pairs']['k4']
    base = arm([50.0] * 4)
    # 差 0 → 悪くない・同等
    p = E.pairs(base, base, base, base, T['pairs'])
    assert p['not_worse'] is True and p['verdict'] == '同等', p
    # 差 +0.6 → 悪くない でない
    p = E.pairs(arm([50.6] * 4), base, arm([50.6] * 4), base, T['pairs'])
    assert p['not_worse'] is False and p['verdict'] == '同等', p
    # last だけ閾値を超えて全部悪い → 悪い。last_only でも悪い
    lb = arm([50.0 + k4['l1_last'] + 0.1] * 4)
    assert E.pairs(base, base, lb, base, T['pairs'])['verdict'] == '悪い'
    assert E.pairs(base, base, lb, base, T['pairs'], last_only=True)['verdict'] == '悪い'
    # best だけ悪い → 悪い。D2(last だけ)なら悪くない
    bb = arm([50.0 + k4['l1_best'] + 0.1] * 4)
    assert E.pairs(bb, base, base, base, T['pairs'])['verdict'] == '悪い'
    assert E.pairs(bb, base, base, base, T['pairs'], last_only=True)['verdict'] == '同等'
    # 良い = best と last の両方で閾値を超えて小さく、全部の組で小さい
    g = arm([50.0 - k4['l1_last'] - 0.1] * 4)
    assert E.pairs(g, base, g, base, T['pairs'])['verdict'] == '良い'
    # 組が 1 つ → 未
    assert E.pairs({0: 50.0}, {0: 50.0}, {0: 50.0}, {0: 50.0}, T['pairs'])['verdict'] == '未'
    # 種がそろっている組だけ(3 組)→ k3 の閾値
    p = E.pairs(arm([50.0] * 3), base, arm([50.0] * 3), base, T['pairs'])
    assert p['k'] == 3 and abs(p['thr_best'] - T['pairs']['k3']['l1_best']) < 1e-9


def row(iv, sb):
    return {'l1_interv_us': iv, 'steer_bias_us': sb}


def test_vetoes():
    five = [row(128.594, 0.132)] * 5
    t = T['vs5']['n4']['interv_best']
    v = E.vetoes({s: row(128.594 + t + 0.1, 0.132) for s in range(4)}, five, T)
    assert v['interv'] is True and v['bias'] is False and v['any'] is True
    v = E.vetoes({s: row(128.594, 0.132 + T['vs5']['n4']['steer_bias_best'] + 0.1) for s in range(4)}, five, T)
    assert v['interv'] is False and v['bias'] is True
    v = E.vetoes({s: row(128.594, 0.132) for s in range(4)}, five, T)
    assert v['any'] is False
    # v2-M は同じ種の v2-H との差(±k4)
    v2h = {s: row(128.594, 5.0) for s in range(4)}
    v = E.vetoes({s: row(128.594, 5.0 + T['pairs']['k4']['steer_bias_best'] - 0.1) for s in range(4)}, five, T, v2h_rows=v2h)
    assert v['bias'] is False, v
    v = E.vetoes({s: row(128.594, 5.0 + T['pairs']['k4']['steer_bias_best'] + 0.1) for s in range(4)}, five, T, v2h_rows=v2h)
    assert v['bias'] is True, v


def test_genre_skew_and_rise():
    thr = E.protocol()['marks']['genre_skew_threshold_us']
    r30 = {'l1_genre4_last_us': 60.0, 'l1_genre5_last_us': 41.0}
    assert E.genre_skew({'l1_genre4_last_us': 62.0, 'l1_genre5_last_us': 41.0}, r30, thr) is True    # 4 だけ +2.0 > 1.938
    assert E.genre_skew({'l1_genre4_last_us': 62.0, 'l1_genre5_last_us': 47.0}, r30, thr) is False   # 両方悪い → 片寄りでない
    assert E.genre_skew({'l1_genre4_last_us': 61.9, 'l1_genre5_last_us': 46.4}, r30, thr) is True    # 5 だけ +5.4 > 5.308
    assert E.genre_skew({'l1_genre4_last_us': 61.0, 'l1_genre5_last_us': 41.0}, r30, thr) is False
    flat = [{'val_us': 50.0}] * 40
    assert E.rise_mark(flat) is False
    up = [{'val_us': 50.0}] * 10 + [{'val_us': 52.0}] * 20
    assert E.rise_mark(up) is True
    short = [{'val_us': 50.0}] * 10 + [{'val_us': 52.0}] * 8
    assert E.rise_mark(short) is False


def test_gates():
    b = {'l1_best_us': 49.144, 'standstill_go_frac': 0.5}
    r = {'l1_best_us': 50.0, 'engine_ok': True, 'sanity_ok': True, 'trt_p99_ms': 2.6, 'steer_amp_ratio': 1.0,
         'brake_lowspeed_frac': 0.005, 'standstill_go_frac': 0.45, 'thr_bias_us': 12.0, 'straight_thr_vs_b_us': 3.0}
    desk = {'b_trt_p99_ms_same_day': 2.5, 'standstill_frames': 400}
    g = E.gates_v2(r, b, desk, 15.964)
    assert E.gates_pass(g) is True, g
    assert E.gates_pass(E.gates_v2(dict(r, thr_bias_us=16.0), b, desk, 15.964)) is False      # 人の走りのスロットル(上限)
    assert E.gates_pass(E.gates_v2(dict(r, straight_thr_vs_b_us=-10.5), b, desk, 15.964)) is False
    assert E.gates_pass(E.gates_v2(dict(r, sanity_ok=None), b, desk, 15.964)) is None           # C3 の欄が無い → 未
    assert E.gates_v2(r, b, dict(desk, standstill_frames=200), 15.964)['standstill'] == '参考'
    assert E.gates_pass(E.gates_v2(r, b, None, 15.964)) is None                                 # 同じ日の B の p99 が無い → 未
    rb = dict(r, bfast_gb_fwd_vs_b_us=12.0, bfast_gc_fwd_vs_b_us=2.0, straight_fwd_vs_b_us=5.0)
    dk = dict(desk, bfast_gf_limit_us=8.0)
    assert E.gates_pass(E.gates_bfast(rb, b, dk)) is True
    assert E.gates_bfast(dict(rb, bfast_gb_fwd_vs_b_us=9.9), b, dk)['G-b'] is False
    assert E.gates_bfast(dict(rb, l1_best_us=50.2), b, dk)['G-a'] is False                       # 49.144 + 1.0
    assert E.gates_bfast(dict(rb, straight_fwd_vs_b_us=8.0), b, dk)['G-f'] is False              # 「超えて」= 未満が合格


def test_parse_and_slots():
    assert E.parse_name('v2_20261025-v2h-e30-s3') == ('v2', 'v2h-e30', 3)
    assert E.parse_name('v2_20261025-v2m-msf-aug2-e60-s1') == ('v2', 'v2m-msf-aug2-e60', 1)
    assert E.parse_name('bfast_20261025-e15-s2') == ('bfast', 'bfast-e15', 2)
    s = E.slot_names()
    assert len(s) == 22 and s[0] == 'bfast_20261025-e15-s0' and s[3] == 'v2_20261025-v2h-e30-s0'
    assert 'v2_20261025-v2h-msf-aug2-e60-s1' in s
    pats = E.protocol_patterns(E.protocol())
    assert all(any(p.match(n) for p in pats) for n in s)
    assert not any(p.match('v2_20261025-v2x-e30-s0') for p in pats)


if __name__ == '__main__':
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            n += 1
            print('OK', name)
    print('%d 個の試しが通りました' % n)
