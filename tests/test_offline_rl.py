"""オフライン強化学習の価値・アドバンテージ(tools/offline_rl.py)のテスト。GPU不要(numpy の計算だけ)

    cd ~/jetformer && python -m pytest -q tests/test_offline_rl.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import offline_rl as orl  # noqa: E402

G = 0.9


def _brute_advantage(r, seg, term, v, n, gamma):
    """定義どおりに1フレームずつ計算する(比較用)"""
    T = len(r)
    A = np.full(T, np.nan)
    for t in range(T):
        if seg[t] < 0 or not np.isfinite(v[t]):
            continue
        L = t
        while L + 1 < T and seg[L + 1] == seg[t]:
            L += 1
        if t + n > L and term[t]:
            A[t] = sum(gamma ** k * r[t + k] for k in range(L - t + 1)) - v[t]
            continue
        b = min(t + n, L)
        while b > t and not np.isfinite(v[b]):
            b -= 1
        if b <= t:
            continue
        A[t] = sum(gamma ** k * r[t + k] for k in range(b - t)) + gamma ** (b - t) * v[b] - v[t]
    return A


# ---------------------------------------------------------------- N ステップのアドバンテージ
def test_nstep_advantage_inside_segment():
    """区間の中: Σ γ^k r + γ^n V(t+n) − V(t)"""
    r = np.arange(10, dtype=float)
    seg = np.zeros(10, int)
    v = np.linspace(1, 2, 10)
    A, ns, boot = orl.nstep_advantage(r, seg, np.zeros(10, bool), v, n=3, gamma=G)
    assert A[0] == pytest.approx(0 + G * 1 + G ** 2 * 2 + G ** 3 * v[3] - v[0])
    assert A[4] == pytest.approx(4 + G * 5 + G ** 2 * 6 + G ** 3 * v[7] - v[4])
    assert ns[0] == 3 and boot[0]


def test_nstep_advantage_terminal_vs_truncated_end():
    """区間が先に終わる: 失敗(終端)なら最後までの報酬だけ、打ち切りなら最後のフレームの V で補う"""
    r = np.array([1.0, 1.0, 1.0, -3.0])
    seg = np.zeros(4, int)
    v = np.array([5.0, 4.0, 3.0, 2.0])
    A_t, ns_t, b_t = orl.nstep_advantage(r, seg, np.ones(4, bool), v, n=10, gamma=G)
    assert A_t[1] == pytest.approx(1 + G * 1 + G ** 2 * -3 - 4.0)
    assert ns_t[1] == 3 and not b_t[1]
    assert A_t[3] == pytest.approx(-3 - 2.0)                 # 終端の最後のフレームも計算できる
    A_c, ns_c, b_c = orl.nstep_advantage(r, seg, np.zeros(4, bool), v, n=10, gamma=G)
    assert A_c[1] == pytest.approx(1 + G * 1 + G ** 2 * 2.0 - 4.0)      # b = 3(区間の最後)で補う
    assert ns_c[1] == 2 and b_c[1]
    assert np.isnan(A_c[3])                                  # 打ち切りの最後のフレームは補う先が無い


def test_nstep_advantage_does_not_cross_segments_and_skips_missing_values():
    """区間をまたがない。補う先の V が無ければ、その手前で V のある最後のフレームで補う。区間外は NaN"""
    r = np.ones(12)
    seg = np.array([0, 0, 0, 0, 0, -1, 1, 1, 1, 1, 1, 1])
    term = np.array([False] * 5 + [False] + [True] * 6)
    v = np.full(12, 3.0)
    v[3] = np.nan
    A, ns, boot = orl.nstep_advantage(r, seg, term, v, n=3, gamma=G)
    assert A[0] == pytest.approx(1 + G * 1 + G ** 2 * 3.0 - 3.0)       # t+3 に V が無い → t+2 で補う
    assert ns[0] == 2
    assert np.isnan(A[5]) and np.isnan(A[3])
    assert A[2] == pytest.approx(1 + G * 1 + G ** 2 * 3.0 - 3.0)       # 区間 0 の最後(4)で補い、区間 1 の報酬は足さない
    assert A[9] == pytest.approx(1 + G + G ** 2 - 3.0)                 # 区間 1 は失敗で終わる(終端)


def test_nstep_advantage_matches_bruteforce_on_random_logs():
    rng = np.random.default_rng(3)
    for trial in range(5):
        T = 300
        seg = np.cumsum(rng.random(T) < 0.03)
        seg[rng.random(T) < 0.05] = -1
        # 区間外で切れた区間は番号を振り直す(同じ番号は連続)
        new = np.full(T, -1)
        k = -1
        for i in range(T):
            if seg[i] >= 0:
                if i == 0 or seg[i - 1] != seg[i]:
                    k += 1
                new[i] = k
        term_seg = rng.random(k + 2) < 0.5
        term = np.where(new >= 0, term_seg[np.clip(new, 0, None)], False)
        r = rng.normal(0.08, 0.3, T)
        v = rng.normal(5, 2, T)
        v[rng.random(T) < 0.1] = np.nan
        for n in (1, 5, 30):
            A, _, _ = orl.nstep_advantage(r, new, term, v, n=n, gamma=0.99)
            B = _brute_advantage(r, new, term, v, n, 0.99)
            assert np.array_equal(np.isnan(A), np.isnan(B))
            assert np.allclose(A[~np.isnan(A)], B[~np.isnan(B)])


def test_segment_bounds():
    seg = np.array([-1, 0, 0, 0, -1, 1, 1, 2])
    first, last = orl.segment_bounds(seg)
    assert first.tolist() == [-1, 1, 1, 1, -1, 5, 5, 7]
    assert last.tolist() == [-1, 3, 3, 3, -1, 6, 6, 7]


# ---------------------------------------------------------------- ジャンルごとの ε と印
def test_good_thresholds_per_genre_give_target_share():
    rng = np.random.default_rng(0)
    A = np.concatenate([rng.normal(0, 1, 4000), rng.normal(5, 3, 6000)])
    genre = np.array([4] * 4000 + [5] * 6000)
    eligible = np.ones(10000, bool)
    eligible[:100] = False
    A[200:210] = np.nan
    eps = orl.good_thresholds(A, genre, eligible, frac=0.35)
    assert set(eps) == {4, 5}
    I = orl.good_labels(A, genre, eps)
    for g in (4, 5):
        m = (genre == g) & eligible & np.isfinite(A)
        assert np.mean(I[m] == 1) == pytest.approx(0.35, abs=0.005)
    assert (I[200:210] == -1).all()
    assert eps[5] > eps[4]                     # ジャンルごとに別の閾値


def test_good_labels_forced_and_inverted():
    A = np.array([0.0, 1.0, 2.0, np.nan, 3.0])
    genre = np.array([5, 5, 5, 5, 7])          # ジャンル 7 は ε が無い → -1
    forced = np.array([True, False, False, True, False])
    I = orl.good_labels(A, genre, {5: 0.5}, forced=forced)
    assert I.tolist() == [1, 1, 1, 1, -1]
    I_neg = orl.good_labels(A, genre, {5: 1.5}, forced=forced, invert=True)
    assert I_neg.tolist() == [0, 1, 0, 0, -1]


# ---------------------------------------------------------------- ログ単位の分割
def test_fold_ids_by_log():
    folds = [['a', 'b'], ['c'], ['d', 'e']]
    logs = np.array(['a', 'c', 'e', 'a', 'b', 'd'])
    assert orl.fold_ids(logs, folds).tolist() == [0, 1, 2, 0, 0, 2]
    with pytest.raises(ValueError):
        orl.fold_ids(np.array(['a', 'z']), folds)
    with pytest.raises(ValueError):
        orl.fold_ids(np.array(['a']), [['a'], ['a']])


def test_scene_to_log_and_frames_of_a_log_share_a_fold():
    scenes = np.array(['20261010-173445-0001', '20261010-173445-0012', '20261010-172910-0003'])
    logs = np.array([orl.log_of_scene(s) for s in scenes])
    assert logs.tolist() == ['20261010-173445', '20261010-173445', '20261010-172910']
    f = orl.fold_ids(logs, [['20261010-173445'], ['20261010-172910']])
    assert f[0] == f[1] != f[2]


def test_balanced_log_folds_and_inner_splits():
    sizes = {'a': 100, 'b': 90, 'c': 50, 'd': 40, 'e': 10, 'f': 5}
    folds = orl.balanced_log_folds(sizes, 3)
    assert sorted(x for fl in folds for x in fl) == sorted(sizes)
    loads = [sum(sizes[x] for x in fl) for fl in folds]
    assert max(loads) - min(loads) <= 60
    pairs = orl.inner_splits(1, 5)
    assert len(pairs) == 4
    for inner, train in pairs:
        assert inner != 1 and 1 not in train and inner not in train
        assert sorted(train + [inner]) == [0, 2, 3, 4]
    assert len(orl.inner_splits(0, 5, members=2)) == 2


# ---------------------------------------------------------------- 評価の小道具
def test_auroc_and_ties():
    y = np.array([0, 0, 1, 1], bool)
    assert orl.auroc([0.1, 0.2, 0.3, 0.4], y) == pytest.approx(1.0)
    assert orl.auroc([0.4, 0.3, 0.2, 0.1], y) == pytest.approx(0.0)
    assert orl.auroc([1, 1, 1, 1], y) == pytest.approx(0.5)
    assert orl.auroc([0.1, 0.3, 0.2, np.nan], y) == pytest.approx(0.5)        # NaN は除く(陽性 0.2 と陰性 0.1・0.3)
    assert np.isnan(orl.auroc([0.1, 0.2], np.array([0, 0], bool)))
    rng = np.random.default_rng(1)
    s = rng.normal(size=500)
    yy = rng.random(500) < 0.3
    pos, neg = s[yy], s[~yy]
    ref = np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :]))
    assert orl.auroc(s, yy) == pytest.approx(ref)


def test_collision_timing():
    t = np.array([0, 500_000, 1_000_000, 3_000_000, 6_000_000])
    c = np.array([1_200_000, 5_000_000])
    nxt, prv = orl.collision_timing(t, c)
    assert nxt[:4].tolist() == pytest.approx([1.2, 0.7, 0.2, 2.0])
    assert np.isinf(nxt[4])
    assert np.isinf(prv[0]) and prv[3] == pytest.approx(1.8) and prv[4] == pytest.approx(1.0)


def test_rate_by_group_and_spearman():
    g_tr = np.array([0, 0, 0, 0, 1, 1])
    y_tr = np.array([1, 1, 0, 0, 0, 0], float)
    out = orl.rate_by_group(g_tr, y_tr, np.array([0, 1, 2]), prior_weight=0.0)
    assert out.tolist() == pytest.approx([0.5, 0.0, 2 / 6])
    assert orl.spearman([1, 2, 3, 4], [10, 20, 25, 100]) == pytest.approx(1.0)
    assert orl.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_top_share_agreement():
    a = np.arange(100, dtype=float)
    loc = np.array([1] * 50 + [2] * 50)
    same = orl.top_share_agreement(a, a * 2 + 1, loc, frac=0.3)
    assert same['agreement'] == pytest.approx(1.0) and same['jaccard_good'] == pytest.approx(1.0)
    rev = orl.top_share_agreement(a, -a, loc, frac=0.3)
    assert rev['jaccard_good'] == pytest.approx(0.0) and rev['spearman'] == pytest.approx(-1.0)
    assert same['by_loc']['GM']['good_a'] == pytest.approx(0.6)          # 上位 30 は全部 loc 2(GM)
