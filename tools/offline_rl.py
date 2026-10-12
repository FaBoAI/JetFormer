#!/usr/bin/env python3
"""オフライン強化学習の最小形 その1: 価値 V(o) と「良い行動」の印(offline_rl.md 5.2〜5.3、ロードマップ 4.3〜4.4)

    python tools/offline_rl.py frames        # フレームの表(CPU)               → data/rl/<データセット>/frames.npz
    python tools/offline_rl.py features      # 基準モデルの特徴(GPU 1プロセス) → data/rl/<データセット>/features_<シーン>.npz
    python tools/offline_rl.py value         # 価値 V(CPU。ログ単位の5分割で交差推定) → value_<種類>.npz
    python tools/offline_rl.py evaluate      # 保留 fold での V の評価(1秒以内の衝突の AUROC・較正)→ value_eval.json
    python tools/offline_rl.py advantages    # N ステップのアドバンテージ A と印 I                     → advantages.npz
    python tools/offline_rl.py sanity        # 場所ごとの「良い」の割合、R1 の進入速度・DE 出口のヨーとの関係 → sanity.json
    python tools/offline_rl.py figures --fig-dir <dir>   # 図(matplotlib)
    python tools/offline_rl.py all           # 上を順に(図は除く)。できているものは飛ばすので、電源が落ちても続きから

どの段階も、出来たファイルがあれば作り直さない(--force で作り直す)。長い段階(特徴・価値)は、シーン・fold ごとに
途中のファイルを残す(一時ファイルに書いてから置き換えるので、書きかけのファイルは残らない)。

1. 特徴(features): 基準モデル(段階Bのモデル B。今の操作値を入れない)を eval で全フレームに通し、Transformer の
   最後の LayerNorm の後の出力トークン(256次元、trunk の前)と、予測した操作列 [10,2] µs を保存する。ジャンルは
   フレームのジャンル(モデルが学習していないジャンルなら学習したジャンルで代用し、substituted に印)。
   GPU を使うのはこの段階だけ。ほかの GPU の学習・推論と同時に動かさない(8GB・スワップなし)。
2. 価値(value): 入力 [特徴, 速度, ジャンルの one-hot](+ 学習時だけ使う場所: loc_code の one-hot・コース図の進みの位相)
   → 2層の MLP で、割引リターン(γ=0.99)を Huber 損失で回帰する。ログ単位の K 分割(log_folds.json)で交差推定し、
   各フレームの V は、そのフレームのログを学習に使っていないモデルの値にする。
   打ち切り(失敗以外の区間の終わり)の扱い: 1回目は、打ち切りの手前 3.3秒(= 1/(1−γ) フレーム)を回帰の目標から外す。
   2回目は、打ち切りで終わる区間のリターンに γ^(残り+1)・V1(区間の最後のフレーム) を足して(V1 = 1回目の、同じ test
   fold を見ていないモデル)全フレームで回帰する。失敗による交代(end_code=1)で終わる区間だけを終端(その後 0)にする。
   早期終了には、学習 fold のうち1つを内側の検証に使い、内側の fold を替えた members 個のモデルの平均をとる。
   種類: feat(位置なし。ゲートCの判定用)/ feat_loc(位置あり。アドバンテージ用)/ speed・loc(特徴を使わない基準)/
         feat_loc_fwd(報酬を前進距離だけにした比較用)/ feat_loc_x2(罰 2倍の感度)/ feat_risk(罰だけ。危険の価値の診断)
3. 評価(evaluate): AI が前進しているフレーム(assist=0・hold なし・0.5 m/s 超・衝突の後 2秒を除く)で、
   「1秒以内に衝突(水平 2.5G)」を −V で当てる AUROC(全体・fold・ログ・場所ごと、区間単位のブートストラップの
   信頼区間)。比べる基準: 速度だけ・場所だけ(ほかの fold の場所ごとの衝突率)・コース図の 0.5m の区間ごとの衝突率
   (当てはめのある5ログ)・記録された ToF の TTC(009/010/101)。参考として、同じ入力で衝突を直接当てる
   ロジスティック回帰(特徴に危険の情報があるか)。V の較正(予測と実際のリターン)と、−V から作った衝突の確率の較正。
4. アドバンテージ(advantages): A_t = Σ_{k<N} γ^k r_{t+k} + γ^N V(o_{t+N}) − V(o_t)(N=30、区間の中だけ。
   nstep_advantage)。印 I = 1[A > ε_ジャンル](ε はジャンルごとに「良い」が 35% になる値)。人の修正(失敗による交代の
   後に人が前進している区間。intervention_val のフレーム)は I=1。陰性対照として、下位 35% を「良い」にした印も残す。
   比較: 報酬を前進距離だけにした印(V も前進距離のリターンで学習)、コース図の進みを報酬にした印(当てはめのある
   AI のログ5本。V は本命と同じなので、違いは 1秒の報酬の差だけ)。
5. 確認(sanity): 場所ごとの「良い」の割合。R1 の手前 0.4秒の印と進入 eRPM、緑マット出口の印と出口のヨーの関係。

机上の指標は実車での改善を示さない(ロードマップ 4.4: ゲートCは「落とす」ための関門)。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from jetformer import config  # noqa: E402
import label_events as le  # noqa: E402

DATASET = 'ai_mix_20261011'
BASE_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'     # 段階Bの基準(モデル B)
IV_DATASET = 'intervention_val_20261011'
TRACKS_DIR = Path('/home/jetson/jetformer_reports/overnight_20261011/course')   # コース図の当てはめ(AI のログ5本)

GAMMA = 0.99
N_STEP = 30                     # 1秒(チャンク 0.33秒の3倍)
GOOD_FRAC = 0.35                # 「良い」の割合(RECAP の 30〜40%)
FPS = 30.0
TRUNC_S = 1.0 / (1.0 - GAMMA) / FPS     # 3.33秒: 割引の実効の長さ
KV = 0.5e-3                     # m/s / eRPM(eRPM は負 = 前進)
LAP_M = 25.48                   # コース図の基準線の1周
GENRES = (4, 5)                 # 高速走行・低速走行(data/modes.json の番号)
ACTION_OK_US = (950.0, 2050.0)  # 未来の操作がこの範囲を外れるフレームは方策に使わない(人の受信機の読み取りの異常)
FORWARD_MPS = 0.5
HORIZON_S = 1.0                 # 「1秒以内に衝突」
AFTER_COLLISION_S = 2.0         # 衝突の後この秒数は評価から外す
LOC_NAMES = le.LOC_NAMES
FEAT_DIM = 256

VARIANTS = {
    'feat': {'inputs': ('feat', 'speed', 'genre'), 'reward': 'main',
             'desc': '特徴+速度+ジャンル(位置なし。ゲートCの判定に使う)'},
    'feat_loc': {'inputs': ('feat', 'speed', 'genre', 'loc'), 'reward': 'main',
                 'desc': '特徴+速度+ジャンル+場所(学習時だけの特権情報。アドバンテージに使う)'},
    'speed': {'inputs': ('speed', 'genre'), 'reward': 'main', 'desc': '速度+ジャンルだけ(特徴なしの基準)'},
    'loc': {'inputs': ('speed', 'genre', 'loc'), 'reward': 'main', 'desc': '速度+ジャンル+場所(特徴なしの基準)'},
    'feat_loc_fwd': {'inputs': ('feat', 'speed', 'genre', 'loc'), 'reward': 'fwd',
                     'desc': 'feat_loc と同じ入力で、報酬を前進距離だけにしたもの(罰なし)'},
    'feat_loc_x2': {'inputs': ('feat', 'speed', 'genre', 'loc'), 'reward': 'x2',
                    'desc': 'feat_loc と同じ入力で、罰の重みを 2倍にしたもの(ロードマップ 4.3 の感度の確認)'},
    'feat_risk': {'inputs': ('feat', 'speed', 'genre'), 'reward': 'risk',
                  'desc': '特徴+速度+ジャンルで、罰だけの割引和(前進距離なし)を回帰する診断用(危険の価値)'},
}


def out_dir(dataset=DATASET):
    return config.DATA_DIR / 'rl' / dataset


def log_of_scene(scene):
    return str(scene).rsplit('-', 1)[0]


def _save_npz(path, **arrays):
    """一時ファイルに書いてから置き換える(電源断で書きかけのファイルを残さない)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=le._json_default)
    os.replace(tmp, path)


def _r(x, n=3):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, n)


# ---------------------------------------------------------------- 純粋な計算(numpy だけ。tests/test_offline_rl.py)
def fold_ids(logs, folds):
    """ログ名の配列 → fold の番号。folds = [[ログ, ...], ...](同じログのフレームは必ず同じ fold)。
    2つの fold にあるログ・どの fold にも無いログは ValueError"""
    m = {}
    for k, fl in enumerate(folds):
        for lg in fl:
            if lg in m:
                raise ValueError('ログが2つの fold にあります: %s' % lg)
            m[lg] = k
    logs = np.asarray(logs)
    missing = sorted(set(logs.tolist()) - set(m))
    if missing:
        raise ValueError('fold の無いログ: %s' % ', '.join(missing))
    return np.array([m[x] for x in logs.tolist()], np.int8)


def balanced_log_folds(log_sizes, k):
    """fold の割り当てが無いときの分け方: 大きいログから順に、いちばん小さい fold へ入れる。log_sizes = {ログ: フレーム数}"""
    folds = [[] for _ in range(k)]
    load = np.zeros(k)
    for lg in sorted(log_sizes, key=lambda x: (-log_sizes[x], x)):
        j = int(np.argmin(load))
        folds[j].append(lg)
        load[j] += log_sizes[lg]
    return folds


def inner_splits(test, k, members=None):
    """test fold を除いた fold から、(早期終了に使う内側の fold, 学習に使う fold の一覧) の組を members 個(既定は全部)"""
    others = [j for j in range(k) if j != test]
    if len(others) < 2:
        raise ValueError('fold が少なすぎます(k=%d)' % k)
    pairs = [(j, [x for x in others if x != j]) for j in others]
    return pairs[:members] if members else pairs


def segment_bounds(seg):
    """区間の番号の配列(-1 = 区間外。同じ番号は連続している)→ 各フレームの区間の最初・最後のフレーム(区間外は -1)"""
    seg = np.asarray(seg)
    n = len(seg)
    first = np.full(n, -1, np.int64)
    last = np.full(n, -1, np.int64)
    if not n:
        return first, last
    brk = np.ones(n, bool)
    brk[1:] = seg[1:] != seg[:-1]
    starts = np.nonzero(brk)[0]
    ends = np.append(starts[1:] - 1, n - 1)
    run = np.cumsum(brk) - 1
    inseg = seg >= 0
    first[inseg] = starts[run[inseg]]
    last[inseg] = ends[run[inseg]]
    return first, last


def nstep_advantage(reward, seg, terminal, value, n=N_STEP, gamma=GAMMA):
    """区間の中の N ステップのアドバンテージ A_t = Σ_{k<n'} γ^k r_{t+k} + γ^{n'} V(o_{t+n'}) − V(o_t)

    reward [T]: フレームの報酬(1つのログの中で時刻順)/ seg [T]: 区間の番号(-1 = 区間外。同じ番号は連続)
    terminal [T]: そのフレームの区間が失敗で終わる(end_code=1。区間の後の価値は 0)
    value [T]: V(o)(NaN = V の無いフレーム。補う先には使わない)
    n' は普通 n。t+n のフレームに V が無ければ、t+n 以前で V のある最後のフレームで補う。
    区間が t+n より前に終わるとき: 終端なら区間の最後までの報酬だけ(V を足さない)。打ち切り(運用上の交代・
    ログの終わり・途切れ)なら、区間の中で V のある最後のフレーム b で補う(n' = b − t)。
    補う先が t より後に無い(区間の最後のフレームなど)・区間外・V(o_t) が無いフレームは NaN。
    戻り値 (A [T], 報酬を足したフレーム数 [T], V で補ったか [T])"""
    r = np.asarray(reward, np.float64)
    seg = np.asarray(seg)
    term = np.asarray(terminal, bool)
    v = np.asarray(value, np.float64)
    T = len(r)
    A = np.full(T, np.nan)
    n_sum = np.zeros(T, np.int32)
    boot = np.zeros(T, bool)
    if not T:
        return A, n_sum, boot
    first, last = segment_bounds(seg)
    idx = np.arange(T)
    # 区間の中で、そのフレーム以前に V のある最後のフレーム(無ければ -1)
    cand = np.where(np.isfinite(v) & (seg >= 0), idx, -1)
    lfv = np.maximum.accumulate(cand)
    lfv = np.where((seg >= 0) & (lfv >= first), lfv, -1)
    ok = (seg >= 0) & np.isfinite(v)
    within = idx + n <= last                                # 区間が t+n まで続く
    end_term = ok & ~within & term                          # 終端の手前: 最後まで足して補わない
    b = np.where(within, lfv[np.clip(idx + n, 0, T - 1)], lfv[np.clip(last, 0, T - 1)])
    use_boot = ok & ~end_term & (b > idx)
    e = np.where(end_term, last + 1, np.where(use_boot, b, idx))     # 報酬を足す範囲 [t, e)
    good = end_term | use_boot
    k = np.arange(n + 1)
    rows = np.clip(idx[:, None] + k[None, :], 0, T - 1)
    mask = k[None, :] < (e - idx)[:, None]
    disc = gamma ** k
    s = np.sum(np.where(mask, r[rows], 0.0) * disc[None, :], axis=1)
    bv = np.where(use_boot, v[np.clip(b, 0, T - 1)] * gamma ** np.clip(b - idx, 0, None), 0.0)
    A[good] = (s + bv - v)[good]
    n_sum[good] = (e - idx)[good]
    boot[good] = use_boot[good]
    return A, n_sum, boot


def good_thresholds(A, genre, eligible, frac=GOOD_FRAC):
    """ジャンルごとの ε: 対象フレーム(eligible で A が有限)のうち、A > ε のものが frac になる値。{ジャンル: ε}"""
    A = np.asarray(A, np.float64)
    genre = np.asarray(genre)
    m0 = np.asarray(eligible, bool) & np.isfinite(A)
    out = {}
    for g in sorted(set(genre[m0].tolist())):
        out[int(g)] = float(np.quantile(A[m0 & (genre == g)], 1.0 - frac))
    return out


def good_labels(A, genre, eps, forced=None, invert=False):
    """印: I = 1[A > ε_ジャンル]。A が NaN・ε の無いジャンルは -1。forced(人の修正)は 1。
    invert=True は陰性対照: eps に下側の境目(frac の分位)を渡し、I = 1[A < ε](悪いほうを「良い」と呼ぶ)、forced は 0"""
    A = np.asarray(A, np.float64)
    genre = np.asarray(genre)
    I = np.full(len(A), -1, np.int8)
    for g, e in eps.items():
        m = (genre == int(g)) & np.isfinite(A)
        I[m] = ((A[m] < e) if invert else (A[m] > e)).astype(np.int8)
    if forced is not None:
        I[np.asarray(forced, bool)] = 0 if invert else 1
    return I


def auroc(score, y):
    """AUROC(同点は平均の順位)。score が NaN のフレームは除く。陽性か陰性が無ければ NaN"""
    s = np.asarray(score, np.float64)
    y = np.asarray(y, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if not n1 or not n0:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    _, inv, counts = np.unique(s[order], return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg = (cum - counts + 1 + cum) / 2.0
    ranks = np.empty(len(s))
    ranks[order] = avg[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def rankdata(x):
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind='mergesort')
    _, inv, counts = np.unique(x[order], return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    r = np.empty(len(x))
    r[order] = ((cum - counts + 1 + cum) / 2.0)[inv]
    return r


def spearman(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float('nan')
    a, b = rankdata(x[ok]), rankdata(y[ok])
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def collision_timing(t_us, coll_us):
    """各フレームから次の衝突までの秒・前の衝突からの秒(無ければ inf)"""
    t = np.asarray(t_us, np.int64)
    c = np.sort(np.asarray(coll_us, np.int64))
    nxt = np.full(len(t), np.inf)
    prv = np.full(len(t), np.inf)
    if len(c) and len(t):
        k = np.searchsorted(c, t, side='left')        # c[k] >= t
        has_n = k < len(c)
        nxt[has_n] = (c[k[has_n]] - t[has_n]) / 1e6
        has_p = k > 0
        prv[has_p] = (t[has_p] - c[k[has_p] - 1]) / 1e6
    return nxt, prv


def cluster_bootstrap_auroc(score, y, cluster, n_boot=500, seed=0):
    """区間(cluster)単位でリサンプルした AUROC の 2.5〜97.5% 点"""
    score = np.asarray(score, np.float64)
    y = np.asarray(y, bool)
    cl = np.unique(np.asarray(cluster), return_inverse=True)[1]
    groups = [np.nonzero(cl == c)[0] for c in range(cl.max() + 1)] if len(cl) else []
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        ii = np.concatenate([groups[p] for p in pick])
        a = auroc(score[ii], y[ii])
        if np.isfinite(a):
            vals.append(a)
    if len(vals) < 10:
        return [None, None]
    return [_r(np.percentile(vals, 2.5)), _r(np.percentile(vals, 97.5))]


def rate_by_group(group_tr, y_tr, group_te, prior_weight=20.0):
    """ほかの fold の、グループ(場所など)ごとの陽性率(全体の率へ縮める)。学習側に無いグループは全体の率"""
    group_tr = np.asarray(group_tr)
    y_tr = np.asarray(y_tr, np.float64)
    base = float(y_tr.mean()) if len(y_tr) else 0.0
    out = np.full(len(group_te), base)
    for g in np.unique(group_te):
        m = group_tr == g
        if m.sum() + prior_weight > 0:
            out[np.asarray(group_te) == g] = (y_tr[m].sum() + prior_weight * base) / (m.sum() + prior_weight)
    return out


def top_share_agreement(a, b, loc, frac=GOOD_FRAC):
    """2つの値のそれぞれ上位 frac を「良い」にしたときの一致率・Jaccard・順位相関と、場所ごとの「良い」の割合"""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if len(a) < 10:
        return {'frames': int(len(a))}
    ga = a > np.quantile(a, 1 - frac)
    gb = b > np.quantile(b, 1 - frac)
    out = {'frames': int(len(a)), 'agreement': _r(np.mean(ga == gb), 4),
           'jaccard_good': _r((ga & gb).sum() / max((ga | gb).sum(), 1), 4), 'spearman': _r(spearman(a, b), 4),
           'by_loc': {}}
    for i, nm in enumerate(LOC_NAMES):
        mm = np.asarray(loc) == i
        if mm.sum() >= 30:
            out['by_loc'][nm] = {'n': int(mm.sum()), 'good_a': _r(ga[mm].mean(), 4), 'good_b': _r(gb[mm].mean(), 4),
                                 'agreement': _r(np.mean(ga[mm] == gb[mm]), 4)}
    return out


def binned_calibration(pred, actual, n_bins=10):
    """予測の分位で n_bins に分け、各ビンの予測の平均と実際の平均"""
    pred = np.asarray(pred, np.float64)
    actual = np.asarray(actual, np.float64)
    ok = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[ok], actual[ok]
    if len(pred) < n_bins:
        return []
    edges = np.quantile(pred, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(edges, pred, side='right') - 1, 0, n_bins - 1)
    return [{'n': int((b == i).sum()), 'pred': _r(pred[b == i].mean(), 4), 'actual': _r(actual[b == i].mean(), 4)}
            for i in range(n_bins) if (b == i).any()]


# ---------------------------------------------------------------- データの読み込み
def _labels(log):
    with np.load(config.DATA_DIR / 'labels' / (log + '.npz')) as z:
        lab = {k: z[k] for k in z.files}
    info = json.loads((config.DATA_DIR / 'labels' / (log + '.json')).read_text(encoding='utf-8'))
    return info, lab


def track_arrays(log, t_us, tracks_dir=TRACKS_DIR):
    """コース図の当てはめ(あれば)をラベルの行へ: 進みの位相 0〜1(NaN = 無い)と、次の行までのコース上の進み [m]"""
    n = len(t_us)
    phase = np.full(n, np.nan)
    dprog = np.full(n, np.nan)
    p = Path(tracks_dir) / ('tracks_%s.npz' % log)
    if not p.exists():
        return phase, dprog
    with np.load(p) as z:
        tt, val, prog, cum = z['t_us'], z['valid'], z['progress_m'], z['progress_cum_m']
    k = np.clip(np.searchsorted(tt, t_us), 0, len(tt) - 1)
    hit = (tt[k] == t_us) & val[k]
    phase[hit] = (prog[k][hit] % LAP_M) / LAP_M
    c = np.where(hit, cum[k], np.nan)
    dprog[:-1] = c[1:] - c[:-1]           # label の progress_fwd_m と同じ向き(その行から次の行まで)
    return phase, dprog


FRAME_KEYS = ['timestamps_us', 'mode', 'future_valid', 'vehicle_valid', 'source', 'label_valid', 'reward', 'mc_return',
              'segment_id', 'end_code', 'time_to_end_s', 'collision', 'near_collision', 'failure_takeover', 'loc_code',
              'progress_m', 'label_index']


def build_frames(dataset=DATASET, folds_path=None, iv_dataset=IV_DATASET, tracks_dir=TRACKS_DIR):
    """データセットの全フレームの表(シーン名の順・フレームの順)。ラベルの行・fold・コース図の位相・人の修正の印を付ける"""
    root = config.DATASET_DIR / dataset
    splits = json.loads((root / 'splits.json').read_text())
    split_of = {s: k for k in ('train', 'val') for s in splits[k]}
    scenes = sorted(p.stem for p in (root / 'aligned').glob('*.npz'))
    cols = {k: [] for k in FRAME_KEYS + ['scene', 'frame', 'log', 'split', 'erpm', 'gz', 'action_label', 'future_actions']}
    for s in scenes:
        with np.load(root / 'aligned' / (s + '.npz')) as z:
            a = {k: z[k] for k in FRAME_KEYS + ['vehicle', 'imu', 'action_label', 'future_actions']}
        n = len(a['timestamps_us'])
        for k in FRAME_KEYS + ['action_label', 'future_actions']:
            cols[k].append(a[k])
        cols['scene'].append(np.full(n, s))
        cols['frame'].append(np.arange(n, dtype=np.int32))
        cols['log'].append(np.full(n, log_of_scene(s)))
        cols['split'].append(np.full(n, split_of.get(s, '?')))
        cols['erpm'].append(a['vehicle'][:, 3].astype(np.float32))
        cols['gz'].append(np.rad2deg(a['imu'][:, -1, 5]).astype(np.float32))      # 最新のヨーレート [deg/s](左 +)
    f = {k: np.concatenate(v) for k, v in cols.items()}
    n = len(f['scene'])
    # ラベルの行から(assist・hold・TTC・前進距離・区間の残り)
    extra = {'assist': np.full(n, -9, np.int8), 'hold': np.zeros(n, bool), 'ttc': np.full(n, np.nan, np.float32),
             'progress_fwd_m': np.full(n, np.nan, np.float32), 'steps_to_end': np.full(n, -1, np.int32),
             'label_t_us': np.zeros(n, np.int64), 'phase': np.full(n, np.nan, np.float32),
             'takeover_class': np.full(n, -1, np.int8)}
    for log in sorted(set(f['log'].tolist())):
        m = np.nonzero(f['log'] == log)[0]
        info, lab = _labels(log)
        li = f['label_index'][m]
        if (li < 0).any():
            raise ValueError('%s: ラベルの行の無いフレームがあります' % log)
        extra['assist'][m] = lab['assist'][li]
        extra['hold'][m] = lab['hold'][li]
        extra['ttc'][m] = lab['ttc'][li]
        extra['progress_fwd_m'][m] = lab['progress_fwd_m'][li]
        extra['steps_to_end'][m] = lab['steps_to_end'][li]
        extra['label_t_us'][m] = lab['t_us'][li]
        extra['takeover_class'][m] = lab['takeover_class'][li]
        phase, _ = track_arrays(log, lab['t_us'], tracks_dir)
        extra['phase'][m] = phase[li]
    f.update(extra)
    # fold(ログ単位)
    folds_path = Path(folds_path or out_dir(dataset) / 'log_folds.json')
    if folds_path.exists():
        folds = json.loads(folds_path.read_text())['folds']
    else:
        sizes = {lg: int((f['log'] == lg).sum()) for lg in set(f['log'].tolist())}
        folds = balanced_log_folds(sizes, 5)
    f['fold'] = fold_ids(f['log'], folds)
    # 人の修正(失敗による交代の後に人が前進している区間。介入の検証セットの窓のフレーム)
    hc = np.zeros(n, bool)
    hc_scored = np.zeros(n, bool)
    iv = config.DATASET_DIR / iv_dataset / 'aligned'
    if iv.exists():
        key = {(lg, int(t)): i for i, (lg, t) in enumerate(zip(f['log'].tolist(), f['timestamps_us'].tolist()))}
        for p in sorted(iv.glob('*.npz')):
            lg = p.stem.rsplit('-tk', 1)[0]
            with np.load(p) as z:
                corr = z['in_correction'] & (z['correction_kind'] == 0)
                for t, sc in zip(z['timestamps_us'][corr].tolist(), z['future_valid'][corr].tolist()):
                    i = key.get((lg, int(t)))
                    if i is not None and f['source'][i] == 0:
                        hc[i] = True
                        hc_scored[i] |= bool(sc)
    f['human_correction'] = hc
    f['human_correction_scored'] = hc_scored
    f['usable'] = f['future_valid'] & f['vehicle_valid'] & (f['mode'] != 0)
    fa = f['future_actions']
    f['action_ok'] = ((fa >= ACTION_OK_US[0]) & (fa <= ACTION_OK_US[1])).all(axis=(1, 2))
    f['folds_json'] = np.array(json.dumps(folds))
    return f


def load_frames(dataset=DATASET):
    p = out_dir(dataset) / 'frames.npz'
    if not p.exists():
        raise SystemExit('先に python tools/offline_rl.py frames を実行してください')
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def cmd_frames(args):
    p = out_dir(args.dataset) / 'frames.npz'
    if p.exists() and not args.force:
        print('frames: できています(%s)' % p)
        return
    t0 = time.time()
    f = build_frames(args.dataset, args.folds, tracks_dir=args.tracks_dir)
    _save_npz(p, **f)
    print('frames: %d フレーム、%d シーン、人の修正 %d(採点 %d)、位相あり %d → %s(%.0f秒)' % (
        len(f['scene']), len(set(f['scene'].tolist())), int(f['human_correction'].sum()),
        int(f['human_correction_scored'].sum()), int(np.isfinite(f['phase']).sum()), p, time.time() - t0))


# ---------------------------------------------------------------- 1. 特徴(GPU)
def _feature_path(dataset, scene):
    return out_dir(dataset) / ('features_%s.npz' % scene)


def cmd_features(args):
    import cv2
    import torch
    from jetformer.model import JetFormer, PWM_CENTER, PWM_SCALE
    from jetformer.policies import runtime
    from jetformer.train_data import normalize_inputs

    root = config.DATASET_DIR / args.dataset
    scenes = sorted(p.stem for p in (root / 'aligned').glob('*.npz'))
    todo = []
    for s in scenes:
        p = _feature_path(args.dataset, s)
        if p.exists() and not args.force:
            try:
                with np.load(p) as z:
                    if z['feat'].shape[0] == z['n_frames']:
                        continue
            except Exception:       # noqa: BLE001  書きかけ・壊れたファイルは作り直す
                pass
        todo.append(s)
    print('features: %d / %d シーンが残り' % (len(todo), len(scenes)), flush=True)
    if not todo:
        return
    run_dir = config.DATA_DIR / 'runs' / args.run
    cfg = json.loads((run_dir / 'model_config.json').read_text(encoding='utf-8'))
    trained = [int(x) for x in cfg.get('trained_modes') or [5]]
    use_cmd = bool(cfg.get('command_input', True))
    dev = torch.device(args.device)
    model, _, _, _ = runtime.load_run(run_dir, 'best', device=dev)       # どのモデルの種類も(無ければ jetformer_v0)
    grab = {}
    if type(model) is JetFormer:            # v0: 融合の出力トークン(今までと同じ)
        model.norm.register_forward_hook(lambda mod, inp, out: grab.__setitem__('tok', out[:, -1].detach()))
    elif not callable(getattr(model, 'embed', None)):
        raise SystemExit('features: このモデルの種類(%s)は特徴(embed)を出せません' % cfg.get('policy'))

    class SceneFrames(torch.utils.data.Dataset):
        def __init__(self, a):
            self.a = a

        def __len__(self):
            return len(self.a['timestamps_us'])

        def __getitem__(self, i):
            a = self.a
            img = cv2.imread(str(root / str(a['cam_files'][i])))
            if img is None:
                raise RuntimeError('画像が読めません: %s' % a['cam_files'][i])
            im, tof, imu, st = normalize_inputs(img, a['tof'][i], a['imu'][i], a['vehicle'][i], use_cmd)
            mode_id = int(a['mode'][i])
            sub = mode_id not in trained
            if sub:
                mode_id = trained[-1]
            mode = np.zeros(cfg['num_modes'], np.float32)
            mode[mode_id] = 1.0
            return im, tof, imu, st, mode, mode_id, sub

    t0 = time.time()
    done = 0
    for si, s in enumerate(todo):
        with np.load(root / 'aligned' / (s + '.npz')) as z:
            a = {k: z[k] for k in ('timestamps_us', 'cam_files', 'tof', 'imu', 'vehicle', 'mode')}
        n = len(a['timestamps_us'])
        loader = torch.utils.data.DataLoader(SceneFrames(a), batch_size=args.batch, shuffle=False,
                                             num_workers=args.workers)
        feats, preds, modes_used, subs = [], [], [], []
        with torch.inference_mode():
            for im, tof, imu, st, mode, mid, sub in loader:
                x = dict(image=im.to(dev), tof=tof.to(dev), imu=imu.to(dev), state=st.to(dev), mode=mode.to(dev))
                out = runtime.forward(model, x)['actions']
                if type(model) is not JetFormer:
                    grab['tok'] = model.embed(**x).detach()                  # 方策の特徴(embed を持つものだけ)
                feats.append(grab['tok'].float().cpu().numpy())
                preds.append(out.float().cpu().numpy() * PWM_SCALE + PWM_CENTER)
                modes_used.append(mid.numpy())
                subs.append(sub.numpy())
        _save_npz(_feature_path(args.dataset, s), feat=np.concatenate(feats).astype(np.float32),
                  pred=np.concatenate(preds).astype(np.float32), mode_used=np.concatenate(modes_used).astype(np.int8),
                  substituted=np.concatenate(subs).astype(bool), n_frames=np.int64(n), scene=np.array(s),
                  run=np.array(args.run), command_input=np.bool_(use_cmd))
        done += n
        el = time.time() - t0
        print('features: [%d/%d] %s %d フレーム(%.0f フレーム/秒、経過 %.0f秒)' % (si + 1, len(todo), s, n, done / el, el),
              flush=True)
    meta = {'dataset': args.dataset, 'run': args.run, 'command_input': use_cmd, 'trained_modes': trained,
            'feature': 'Transformer の最後の LayerNorm の後の出力トークン(256次元、trunk の前)',
            'pred': '予測した [10,2] µs(フレームのジャンル。学習していないジャンルは代用し substituted)',
            'device': args.device, 'batch': args.batch, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    _save_json(out_dir(args.dataset) / 'features_meta.json', meta)


def load_features(f, dataset=DATASET):
    """frames.npz と同じ順に特徴・予測をつなぐ"""
    feats, preds = [], []
    scenes = []
    for s in f['scene'].tolist():
        if not scenes or scenes[-1] != s:
            scenes.append(s)
    for s in scenes:
        with np.load(_feature_path(dataset, s)) as z:
            feats.append(z['feat'])
            preds.append(z['pred'])
    feat = np.concatenate(feats)
    if len(feat) != len(f['scene']):
        raise ValueError('特徴の数がフレームの表と合いません(%d / %d)' % (len(feat), len(f['scene'])))
    return feat, np.concatenate(preds)


# ---------------------------------------------------------------- 2. 価値(CPU)
def reward_arrays(log, kind, lab):
    """ラベルの行の報酬。main = ラベルの報酬(前進距離 − 罰)、fwd = 前進距離だけ、x2 = 罰を 2倍、
    risk = 罰だけ(前進距離なし。危険の価値の診断用)"""
    main = lab['reward'].astype(np.float64)
    fwd = np.where(lab['segment_id'] >= 0, lab['progress_fwd_m'], 0.0).astype(np.float64)
    if kind == 'main':
        return main
    if kind == 'fwd':
        return fwd
    if kind == 'x2':
        return fwd + 2.0 * (main - fwd)
    if kind == 'risk':
        return main - fwd
    raise ValueError(kind)


def value_targets(f, kind):
    """フレームごとの回帰の目標: MC リターン(区間の終わりで打ち切り)・終端の区間か・区間の残りの行数・区間の最後の
    フレーム(表の行。打ち切りの区間を V で補うときに使う)"""
    n = len(f['scene'])
    ret = np.full(n, np.nan)
    last_row = np.full(n, -1, np.int64)
    for log in sorted(set(f['log'].tolist())):
        m = np.nonzero(f['log'] == log)[0]
        _, lab = _labels(log)
        r = reward_arrays(log, kind, lab)
        g = le.mc_returns(r, lab['segment_id'])
        li = f['label_index'][m]
        ret[m] = g[li]
        # 区間ごとに、表の中で最後のフレーム(時刻順)
        o = m[np.argsort(f['timestamps_us'][m], kind='stable')]
        seg = f['segment_id'][o]
        for sid in np.unique(seg[seg >= 0]):
            rows = o[seg == sid]
            last_row[rows] = rows[-1]
    terminal = f['end_code'] == 1
    return ret, terminal, f['steps_to_end'].astype(np.int64), last_row


def value_inputs(f, feat, inputs):
    """MLP の入力 [N, D](float32)。標準化は学習側でする"""
    parts = []
    if 'feat' in inputs:
        parts.append(feat.astype(np.float32))
    if 'speed' in inputs:
        parts.append((-f['erpm'] * KV / 3.0).clip(-1, 3).astype(np.float32)[:, None])
    if 'genre' in inputs:
        parts.append(np.stack([f['mode'] == g for g in GENRES], axis=1).astype(np.float32))
    if 'loc' in inputs:
        oh = np.eye(len(LOC_NAMES), dtype=np.float32)[np.clip(f['loc_code'], 0, len(LOC_NAMES) - 1)]
        ph = f['phase'].astype(np.float64)
        has = np.isfinite(ph)
        sc = np.stack([np.where(has, np.sin(2 * np.pi * ph), 0), np.where(has, np.cos(2 * np.pi * ph), 0),
                       has.astype(float)], axis=1).astype(np.float32)
        parts.append(np.concatenate([oh, sc], axis=1))
    return np.concatenate(parts, axis=1)


def fit_mlp(X, y, w, tr, va, hidden=128, epochs=40, patience=6, lr=3e-4, wd=1e-2, batch=512, seed=0, delta=1.0,
            threads=4):
    """2層の MLP で y を Huber 回帰(CPU)。tr / va はフレームの番号、w は重み(0 = 使わない)。va の損失で早期終了。
    戻り値 (predict(X) 関数, 学習の記録)
    既定の大きさ・学習率・重みの減衰は、test fold を使わずに内側の fold(fold 1・3 を検証、test は fold 0)で選んだ:
    30fps の続いたフレームは似ていて、特徴の MLP は1〜2エポックでログを覚えて過学習しやすい(検証の Huber が
    hidden 256・lr 1e-3・wd 1e-4 の 1.171 → 1.130、fold 1)"""
    import torch
    import torch.nn as nn
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr = tr[w[tr] > 0]
    va = va[w[va] > 0]
    mu = X[tr].mean(axis=0)
    sd = X[tr].std(axis=0) + 1e-6
    ym = float(np.average(y[tr], weights=w[tr]))
    Xt = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    yt = torch.from_numpy((y - ym).astype(np.float32))
    wt = torch.from_numpy(w.astype(np.float32))
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.HuberLoss(reduction='none', delta=delta)

    def val_loss():
        net.eval()
        with torch.no_grad():
            p = net(Xt[va]).squeeze(1)
            return float((loss_fn(p, yt[va]) * wt[va]).sum() / wt[va].sum())

    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        net.train()
        perm = tr[rng.permutation(len(tr))]
        tot = 0.0
        for i in range(0, len(perm), batch):
            b = torch.from_numpy(perm[i:i + batch])
            p = net(Xt[b]).squeeze(1)
            loss = (loss_fn(p, yt[b]) * wt[b]).sum() / wt[b].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        vl = val_loss()
        hist.append({'epoch': ep + 1, 'train': round(tot / len(perm), 5), 'val': round(vl, 5)})
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(Xn):
        with torch.no_grad():
            Z = torch.from_numpy(((Xn - mu) / sd).astype(np.float32))
            return np.concatenate([net(Z[i:i + 8192]).squeeze(1).numpy() for i in range(0, len(Z), 8192)]) + ym

    return predict, {'best_val_huber': round(best, 5), 'epochs': len(hist), 'history': hist, 'n_train': int(len(tr)),
                     'n_val': int(len(va))}


def _value_part_path(dataset, variant, rnd, k):
    return out_dir(dataset) / 'value' / variant / ('r%d_f%d.npz' % (rnd, k))


def train_value_variant(f, feat, variant, dataset=DATASET, members=None, force=False, epochs=40, log=print):
    """1つの種類の V を、fold ごと・2回(打ち切りの扱い)学習して、out-of-fold の値を返す"""
    spec = VARIANTS[variant]
    X = value_inputs(f, feat, spec['inputs'])
    ret, terminal, steps_to_end, last_row = value_targets(f, spec['reward'])
    fold = f['fold'].astype(int)
    K = int(fold.max()) + 1
    inseg = (f['segment_id'] >= 0) & np.isfinite(ret) & (f['mode'] != 0)
    w1 = (inseg & (terminal | (f['time_to_end_s'] >= TRUNC_S))).astype(np.float64)
    w2 = inseg.astype(np.float64)
    V1 = np.full(len(ret), np.nan)
    V2 = np.full(len(ret), np.nan)
    T2 = np.full(len(ret), np.nan)            # 2回目の目標(test fold のフレームは、その fold を見ていない V1 で補った値)
    info = {'variant': variant, 'desc': spec['desc'], 'inputs': list(spec['inputs']), 'reward': spec['reward'],
            'input_dim': int(X.shape[1]), 'folds': []}
    for k in range(K):
        test = np.nonzero(fold == k)[0]
        fi = {'fold': k, 'test_frames': int(len(test))}
        for rnd in (1, 2):
            p = _value_part_path(dataset, variant, rnd, k)
            if p.exists() and not force:
                with np.load(p) as z:
                    pred_all = z['pred_all']
                    hist = json.loads(str(z['info']))
                    tgt = z['target'] if 'target' in z.files else None
            else:
                if rnd == 1:
                    y, w = np.nan_to_num(ret), w1
                    tgt = None
                else:
                    with np.load(_value_part_path(dataset, variant, 1, k)) as z:
                        v1k = z['pred_all']
                    boot = np.where(terminal | (last_row < 0), 0.0, GAMMA ** (steps_to_end + 1) * v1k[np.clip(last_row, 0, None)])
                    y, w = np.nan_to_num(ret + boot), w2
                    tgt = (ret + boot).astype(np.float32)
                preds, hist = [], []
                for mi, (inner, train_folds) in enumerate(inner_splits(k, K, members)):
                    tr = np.nonzero(np.isin(fold, train_folds))[0]
                    va = np.nonzero(fold == inner)[0]
                    t0 = time.time()
                    predict, h = fit_mlp(X, y, w, tr, va, epochs=epochs, seed=100 * k + 10 * rnd + mi)
                    preds.append(predict(X))
                    h.update({'inner_fold': inner, 'train_folds': train_folds, 'seconds': round(time.time() - t0, 1)})
                    hist.append(h)
                    log('value %s 回%d fold%d member%d: 検証 Huber %.4f(%d エポック、%.0f秒)' % (
                        variant, rnd, k, mi, h['best_val_huber'], h['epochs'], h['seconds']))
                pred_all = np.mean(preds, axis=0).astype(np.float32)
                arrays = {'pred_all': pred_all, 'info': np.array(json.dumps(hist))}
                if tgt is not None:
                    arrays['target'] = tgt
                _save_npz(p, **arrays)
            (V1 if rnd == 1 else V2)[test] = pred_all[test]
            if rnd == 2 and tgt is not None:
                T2[test] = tgt[test]
            fi['round%d' % rnd] = [{kk: h[kk] for kk in ('inner_fold', 'best_val_huber', 'epochs', 'n_train', 'n_val')}
                                   for h in hist]
        info['folds'].append(fi)
    return {'V': V2.astype(np.float32), 'V1': V1.astype(np.float32), 'target_mc': ret.astype(np.float32),
            'target_boot': T2.astype(np.float32), 'terminal': terminal, 'w1': w1 > 0, 'fold': fold.astype(np.int8)}, info


def cmd_value(args):
    f = load_frames(args.dataset)
    feat = None
    variants = args.variants or list(VARIANTS)
    for v in variants:
        p = out_dir(args.dataset) / ('value_%s.npz' % v)
        if p.exists() and not args.force:
            print('value %s: できています' % v)
            continue
        if feat is None and any('feat' in VARIANTS[x]['inputs'] for x in variants):
            feat, _ = load_features(f, args.dataset)
        res, info = train_value_variant(f, feat, v, args.dataset, members=args.members, force=args.force,
                                        epochs=args.epochs, log=lambda s: print(s, flush=True))
        _save_npz(p, **res, info=np.array(json.dumps(info, ensure_ascii=False)))
        print('value %s: → %s' % (v, p), flush=True)


def load_value(variant, dataset=DATASET):
    with np.load(out_dir(dataset) / ('value_%s.npz' % variant)) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- 3. 評価(CPU)
def collision_targets(f):
    """フレームごとの、次の衝突までの秒と前の衝突からの秒(ラベルの衝突。人・AI どちらも)"""
    n = len(f['scene'])
    nxt = np.full(n, np.inf)
    prv = np.full(n, np.inf)
    events = {}
    for log in sorted(set(f['log'].tolist())):
        m = np.nonzero(f['log'] == log)[0]
        info, _ = _labels(log)
        cs = sorted(c['t_us'] for c in info['events']['collisions'])
        events[log] = cs
        nxt[m], prv[m] = collision_timing(f['timestamps_us'][m], cs)
    return nxt, prv, events


def eval_mask(f, prv, source=1):
    """評価のフレーム: AI(source=1)が前進している(assist なし・hold なし・0.5 m/s 超)、直前 2秒に衝突が無い。
    assist は 0 = なし、-1 = 記録の無い古いファーム(リカバリーが無いので「なし」と同じ)、1〜3 = 後退の受け付け・
    リカバリー・中止"""
    speed = -f['erpm'] * KV
    return ((f['source'] == source) & (f['assist'] <= 0) & ~f['hold'] & (speed > FORWARD_MPS)
            & (prv > AFTER_COLLISION_S) & (f['mode'] != 0))


def fit_logistic(X, y, tr, l2=1e-3, iters=300, threads=4, seed=0):
    """ロジスティック回帰(CPU、全バッチ LBFGS)。入力は学習側で標準化。戻り値 predict(X) → 陽性の確率"""
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    mu = X[tr].mean(axis=0)
    sd = X[tr].std(axis=0) + 1e-6
    Xt = torch.from_numpy(((X[tr] - mu) / sd).astype(np.float32))
    yt = torch.from_numpy(y[tr].astype(np.float32))
    wv = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([wv, b], lr=0.5, max_iter=iters, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        z = Xt @ wv + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt) + l2 * (wv ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    W, B = wv.detach().numpy(), float(b.detach())

    def predict(Xn):
        z = ((Xn - mu) / sd).astype(np.float32) @ W + B
        return 1.0 / (1.0 + np.exp(-z))
    return predict


def crossfit_scores(f, X, y, mask):
    """fold ごとに、ほかの fold の評価フレームで学習して test fold のフレームの確率を出す"""
    fold = f['fold'].astype(int)
    out = np.full(len(y), np.nan)
    for k in range(int(fold.max()) + 1):
        tr = np.nonzero(mask & (fold != k))[0]
        te = np.nonzero(mask & (fold == k))[0]
        if not len(te) or y[tr].sum() == 0:
            continue
        predict = fit_logistic(X, y.astype(float), tr)
        out[te] = predict(X[te])
    return out


def events_caught(f, flag, y, nxt, mask):
    """衝突ごと(ログ+衝突の時刻)に、手前 1秒の評価フレームで警報が出たか。戻り値 (捉えた数, 衝突の数)"""
    m = np.nonzero(mask & y)[0]
    keys = {}
    for i in m:
        key = (str(f['log'][i]), int(f['timestamps_us'][i] + nxt[i] * 1e6) // 100000)
        keys.setdefault(key, False)
        keys[key] |= bool(flag[i])
    return sum(keys.values()), len(keys)


def cmd_evaluate(args):
    f = load_frames(args.dataset)
    nxt, prv, _ = collision_targets(f)
    y = nxt <= HORIZON_S
    ev = eval_mask(f, prv, 1)
    ev_h = eval_mask(f, prv, 0)
    speed = -f['erpm'] * KV
    fold = f['fold'].astype(int)
    cluster = np.array(['%s/%d' % (lg, s) for lg, s in zip(f['log'].tolist(), f['segment_id'].tolist())])
    vals = {v: load_value(v, args.dataset) for v in VARIANTS if (out_dir(args.dataset) / ('value_%s.npz' % v)).exists()}
    scores = {}
    for v, d in vals.items():
        scores['V_' + v] = -d['V'].astype(np.float64)
        scores['V1_' + v] = -d['V1'].astype(np.float64)
    if 'feat' in vals and 'speed' in vals:      # 特徴が速度に足した分だけ(V_feat − V_speed)
        scores['V_feat_minus_speed'] = -(vals['feat']['V'].astype(np.float64) - vals['speed']['V'].astype(np.float64))
    scores['speed'] = speed.astype(np.float64)
    # 場所だけ: ほかの fold の評価フレームの、場所ごとの衝突率
    loc_rate = np.full(len(y), np.nan)
    for k in range(int(fold.max()) + 1):
        tr = ev & (fold != k)
        te = ev & (fold == k)
        loc_rate[te] = rate_by_group(f['loc_code'][tr], y[tr], f['loc_code'][te])
    scores['loc_rate'] = loc_rate
    # コース図の 0.5m の区間ごと(当てはめのあるフレームだけ)
    has_ph = np.isfinite(f['phase'])
    pbin = np.where(has_ph, np.floor(np.nan_to_num(f['phase']) * LAP_M / 0.5), -1).astype(int)
    pos_rate = np.full(len(y), np.nan)
    for k in range(int(fold.max()) + 1):
        tr = ev & has_ph & (fold != k)
        te = ev & has_ph & (fold == k)
        if tr.any() and te.any():
            pos_rate[te] = rate_by_group(pbin[tr], y[tr], pbin[te])
    scores['course_pos_rate'] = pos_rate
    ttc = f['ttc'].astype(np.float64)
    scores['ttc'] = np.where(np.isfinite(ttc), -np.minimum(ttc, 30.0), np.nan)
    # 参考: 同じ入力で衝突を直接当てる(ロジスティック回帰、fold で交差)
    feat = None
    if args.direct:
        feat, _ = load_features(f, args.dataset)
        Xf = value_inputs(f, feat, ('feat', 'speed'))
        scores['direct_feat'] = crossfit_scores(f, Xf, y, ev)
        Xl = value_inputs(f, feat, ('speed', 'loc'))
        Xl = np.concatenate([Xl, Xl[:, :1] ** 2], axis=1)
        scores['direct_speed_loc'] = crossfit_scores(f, Xl, y, ev)
    res = {'rule': {'frames': 'AI が前進(source=1・assist なし・hold なし・%.1f m/s 超)、直前 %.0f秒に衝突なし、ジャンルあり'
                              % (FORWARD_MPS, AFTER_COLLISION_S),
                    'positive': '%.0f秒以内に衝突(ラベルの衝突 = 水平 2.5G、0.6秒以内は1件)' % HORIZON_S,
                    'score': 'V はリスクの向きにそろえるため −V。速度は速いほど危険。場所・位置は他の fold の衝突率'},
           'frames': int(ev.sum()), 'positives': int((ev & y).sum()), 'pos_rate': _r((ev & y).sum() / max(ev.sum(), 1), 4),
           'scores': {}}
    for name, s in scores.items():
        m = ev & np.isfinite(s)
        if not m.any():
            continue
        e = {'frames': int(m.sum()), 'positives': int((m & y).sum()), 'auroc': _r(auroc(s[m], y[m]), 4),
             'ci95_cluster': cluster_bootstrap_auroc(s[m], y[m], cluster[m], n_boot=args.boot),
             'per_fold': {}, 'per_log': {}, 'per_loc': {}}
        for k in range(int(fold.max()) + 1):
            mm = m & (fold == k)
            e['per_fold'][k] = {'n': int(mm.sum()), 'pos': int((mm & y).sum()), 'auroc': _r(auroc(s[mm], y[mm]), 4)}
        e['mean_fold_auroc'] = _r(np.nanmean([x['auroc'] if x['auroc'] is not None else np.nan
                                              for x in e['per_fold'].values()]), 4)
        for lg in sorted(set(f['log'][m].tolist())):
            mm = m & (f['log'] == lg)
            e['per_log'][lg] = {'n': int(mm.sum()), 'pos': int((mm & y).sum()), 'auroc': _r(auroc(s[mm], y[mm]), 4)}
        for i, nm in enumerate(LOC_NAMES):
            mm = m & (f['loc_code'] == i)
            if mm.any():
                e['per_loc'][nm] = {'n': int(mm.sum()), 'pos': int((mm & y).sum()), 'auroc': _r(auroc(s[mm], y[mm]), 4)}
        res['scores'][name] = e
    # 同じフレームでの比較(TTC のある3ログ・位置のある5ログ)
    for sub_name, sub in (('ttc_logs', ev & np.isfinite(scores['ttc'])), ('course_fit_logs', ev & has_ph)):
        res[sub_name] = {'frames': int(sub.sum()), 'positives': int((sub & y).sum()),
                         'logs': sorted(set(f['log'][sub].tolist())),
                         'auroc': {name: _r(auroc(s[sub], y[sub]), 4) for name, s in scores.items()
                                   if np.isfinite(s[sub]).all() and sub.any()}}
    # TTC < 0.5秒の規則と、誤警報の数をそろえて比べる
    sub = ev & np.isfinite(scores['ttc'])
    cmp = {}
    for name in ('V_feat', 'V_feat_loc', 'V_feat_risk', 'direct_feat', 'speed'):
        if name not in scores:
            continue
        s = scores[name]
        m = sub & np.isfinite(s)
        rule = m & (ttc < 0.5)
        fp = int((rule & ~y).sum())
        neg = np.sort(s[m & ~y])[::-1]
        thr_s = neg[fp - 1] if 0 < fp <= len(neg) else np.inf
        flag = m & (s >= thr_s)
        c_rule = events_caught(f, rule, y, nxt, m)
        c_s = events_caught(f, flag, y, nxt, m)
        minutes = m.sum() / FPS / 60.0
        cmp[name] = {'frames': int(m.sum()), 'positives': int((m & y).sum()), 'minutes': _r(minutes, 2),
                     'ttc_rule_tp_frames': int((rule & y).sum()), 'ttc_rule_fp_frames': fp,
                     'ttc_rule_fp_frames_per_min': _r(fp / max(minutes, 1e-9), 1),
                     'ttc_rule_collisions_caught': c_rule,
                     'score_tp_frames_at_same_fp': int((flag & y).sum()), 'score_fp_frames': int((flag & ~y).sum()),
                     'score_collisions_caught': c_s}
    res['ttc_rule_vs_score_same_false_alarms'] = cmp
    # 参考: 人が前進しているフレーム
    res['human_frames'] = {'frames': int(ev_h.sum()), 'positives': int((ev_h & y).sum()),
                           'auroc': {name: _r(auroc(scores[name][ev_h], y[ev_h]), 4)
                                     for name in scores if name.startswith('V_') or name == 'speed'}}
    # 較正 (1): V と実際のリターン(打ち切りの影響が小さいフレーム: 終端の区間、または区間の残りが 2×3.3秒以上)
    calib = {}
    for v, d in vals.items():
        V = d['V'].astype(np.float64)
        ok = np.isfinite(V) & np.isfinite(d['target_mc']) & (f['segment_id'] >= 0) & (
            d['terminal'] | (f['time_to_end_s'] >= 2 * TRUNC_S))
        g = d['target_mc'].astype(np.float64)
        c = {}
        for nm, mm in (('all', ok), ('ai', ok & (f['source'] == 1)), ('human', ok & (f['source'] == 0))):
            if mm.sum() < 20:
                continue
            resid = g[mm] - V[mm]
            # 定数(各 test fold について、ほかの fold の平均)だけの予測と比べる
            const = np.zeros(mm.sum())
            fm = fold[mm]
            for k in np.unique(fm):
                tr = ok & (fold != k)
                const[fm == k] = g[tr].mean()
            sst = ((g[mm] - const) ** 2).sum()
            c[nm] = {'frames': int(mm.sum()), 'mae': _r(np.abs(resid).mean(), 3), 'bias': _r(resid.mean(), 3),
                     'r2_vs_fold_const': _r(1 - (resid ** 2).sum() / sst, 3),
                     'corr': _r(np.corrcoef(V[mm], g[mm])[0, 1], 3),
                     'slope': _r(np.polyfit(V[mm], g[mm], 1)[0], 3),
                     'bins': binned_calibration(V[mm], g[mm])}
        calib[v] = c
    res['value_calibration'] = {'rule': 'test fold のフレームの V と MC リターン(失敗で終わる区間か、区間の残りが %.1f秒以上の'
                                         'フレームだけ。打ち切りで小さく出るのを避ける)' % (2 * TRUNC_S), 'variants': calib}
    # 較正 (2): −V から作った「1秒以内に衝突」の確率(ほかの fold の評価フレームで Platt の当てはめ)
    risk = {}
    for name in ('V_feat', 'V_feat_loc', 'V_speed', 'V_loc', 'V_feat_risk'):
        if name not in scores:
            continue
        X = scores[name][:, None]
        p = crossfit_scores(f, np.nan_to_num(X), y, ev & np.isfinite(scores[name]))
        m = ev & np.isfinite(p)
        base = np.zeros(m.sum())
        fm = fold[m]
        for k in np.unique(fm):
            base[fm == k] = y[ev & (fold != k)].mean()
        risk[name] = {'brier': _r(np.mean((p[m] - y[m]) ** 2), 4), 'brier_base_rate': _r(np.mean((base - y[m]) ** 2), 4),
                      'bins': binned_calibration(p[m], y[m].astype(float))}
    res['risk_calibration'] = risk
    # ゲートC 1 の判定(位置なしの V)
    base_names = [n for n in ('speed', 'loc_rate', 'V_speed', 'V_loc', 'direct_speed_loc') if n in res['scores']]
    if 'V_feat' in res['scores']:
        a = res['scores']['V_feat']['auroc']
        best_base = max(res['scores'][n]['auroc'] for n in base_names if res['scores'][n]['auroc'] is not None)
        res['gate_c1'] = {'auroc_V_feat': a, 'ci95': res['scores']['V_feat']['ci95_cluster'],
                          'best_featureless_baseline': best_base,
                          'best_featureless_name': max(base_names, key=lambda n: res['scores'][n]['auroc'] or -1),
                          'pass_0.8': bool(a is not None and a >= 0.8), 'beats_baselines': bool(a is not None and a > best_base),
                          'pass': bool(a is not None and a >= 0.8 and a > best_base),
                          'rule': 'ロードマップ 4.4 ゲートC 1: AUROC ≥ 0.8 かつ特徴を使わない基準を上回る(位置なしの V で判定)'}
    _save_json(out_dir(args.dataset) / 'value_eval.json', res)
    _save_npz(out_dir(args.dataset) / 'value_eval_scores.npz', y=y, eval_ai=ev, eval_human=ev_h, next_collision_s=nxt,
              **{'score_' + k: v.astype(np.float32) for k, v in scores.items()})
    print(json.dumps({'frames': res['frames'], 'positives': res['positives'],
                      'auroc': {n: (e['auroc'], e['ci95_cluster']) for n, e in res['scores'].items()},
                      'gate_c1': res.get('gate_c1')}, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- 4. アドバンテージ
def advantages_for(f, V, kind='main', n=N_STEP, gamma=GAMMA, course=False):
    """ログごとにラベルの行の上で A を計算して、表のフレームへ戻す。V は表のフレームの値(無い行は補う先に使わない)。
    course=True: 報酬の前進距離をコース図の進みに置き換える(当てはめの無い行が窓に入るフレームは NaN)"""
    nf = len(f['scene'])
    A = np.full(nf, np.nan)
    nsum = np.zeros(nf, np.int32)
    boot = np.zeros(nf, bool)
    for log in sorted(set(f['log'].tolist())):
        m = np.nonzero(f['log'] == log)[0]
        _, lab = _labels(log)
        T = len(lab['t_us'])
        r = reward_arrays(log, kind, lab)
        if course:
            _, dprog = track_arrays(log, lab['t_us'])
            if not np.isfinite(dprog).any():
                continue
            r = r - np.where(lab['segment_id'] >= 0, lab['progress_fwd_m'], 0.0) + dprog
        v = np.full(T, np.nan)
        v[f['label_index'][m]] = V[m]
        term = np.zeros(T, bool)
        seg = lab['segment_id']
        term[seg >= 0] = lab['end_code'][seg >= 0] == 1
        a, ns, bt = nstep_advantage(r, seg, term, v, n, gamma)
        li = f['label_index'][m]
        A[m], nsum[m], boot[m] = a[li], ns[li], bt[li]
    return A, nsum, boot


def cmd_advantages(args):
    p = out_dir(args.dataset) / 'advantages.npz'
    if p.exists() and not args.force:
        print('advantages: できています(%s)' % p)
        return
    f = load_frames(args.dataset)
    genre = f['mode'].astype(int)
    eligible = f['usable'] & f['label_valid'] & f['action_ok'] & (f['source'] >= 0)
    hc = f['human_correction']
    out = {'scene': f['scene'], 'frame': f['frame'], 'log': f['log'], 'split': f['split'], 'fold': f['fold'],
           'source': f['source'], 'genre': genre.astype(np.int8), 'loc_code': f['loc_code'], 'eligible': eligible,
           'human_correction': hc}
    meta = {'gamma': GAMMA, 'n_step': N_STEP, 'good_frac': GOOD_FRAC, 'dataset': args.dataset,
            'eligible_rule': 'future_valid & vehicle_valid & ジャンルあり & label_valid & 未来の操作が %g〜%gµs' % ACTION_OK_US,
            'human_correction_rule': '介入の検証セット(%s)の修正区間・前進のフレーム(人。失敗による交代の後 5秒まで)。'
                                     'ai_mix では future_valid=False なので、方策の学習に使うにはデータセットの作り直しが要る'
                                     % IV_DATASET,
            'labels': {}}
    sets = [('main', 'feat_loc', 'main', False), ('noloc', 'feat', 'main', False), ('fwd', 'feat_loc_fwd', 'fwd', False),
            ('x2', 'feat_loc_x2', 'x2', False), ('course', 'feat_loc', 'main', True)]
    for name, variant, kind, course in sets:
        vp = out_dir(args.dataset) / ('value_%s.npz' % variant)
        if not vp.exists():
            print('advantages: value_%s がありません(飛ばします)' % variant)
            continue
        V = load_value(variant, args.dataset)['V'].astype(np.float64)
        A, ns, bt = advantages_for(f, V, kind, course=course)
        if name == 'course':        # 当てはめのあるフレームだけで閾値を決めると比較にならないので、本命の閾値を使う
            eps = meta['labels']['main']['eps']
            eps = {int(k): v for k, v in eps.items()}
        else:
            eps = good_thresholds(A, genre, eligible & ~hc)
        I = good_labels(A, genre, eps, forced=hc if name in ('main', 'noloc', 'fwd', 'x2') else None)
        sfx = '' if name == 'main' else '_' + name
        out['A' + sfx] = A.astype(np.float32)
        out['I' + sfx] = I
        out['V' + sfx] = V.astype(np.float32)
        out['n_reward' + sfx] = ns
        out['bootstrapped' + sfx] = bt
        el = eligible & np.isfinite(A)
        meta['labels'][name] = {'value_variant': variant, 'reward': kind, 'course_progress': course,
                                'eps': {str(k): _r(v, 4) for k, v in eps.items()},
                                'eligible_finite': int(el.sum()),
                                'good_share_eligible': _r(np.mean(I[el] == 1), 4) if el.any() else None,
                                'good_share_by_genre': {str(g): _r(np.mean(I[el & (genre == g)] == 1), 4)
                                                        for g in GENRES if (el & (genre == g)).any()},
                                'A_std_eligible': _r(np.nanstd(A[el]), 4) if el.any() else None,
                                'bootstrapped_share': _r(np.mean(bt[el]), 4) if el.any() else None}
        if name == 'main':
            lo = {g: float(np.quantile(A[el & ~hc & (genre == g)], GOOD_FRAC)) for g in GENRES if (el & (genre == g)).any()}
            out['I_neg'] = good_labels(A, genre, lo, forced=hc, invert=True)
            meta['labels']['neg'] = {'rule': '陰性対照: 下位 %.0f%% を「良い」(人の修正は 0)' % (100 * GOOD_FRAC),
                                     'eps_low': {str(k): _r(v, 4) for k, v in lo.items()}}
    # 本命と比較の一致
    agree = {}
    el = eligible
    for other in ('noloc', 'fwd', 'x2', 'course'):
        if 'I_' + other not in out or 'I' not in out:
            continue
        m = el & (out['I'] >= 0) & (out['I_' + other] >= 0) & ~hc
        a, b = out['I'][m] == 1, out['I_' + other][m] == 1
        e = {'frames': int(m.sum()), 'agreement': _r(np.mean(a == b), 4),
             'jaccard_good': _r((a & b).sum() / max((a | b).sum(), 1), 4),
             'spearman_A': _r(spearman(out['A'][m], out['A_' + other][m]), 4), 'by_loc': {}, 'by_source': {}}
        for i, nm in enumerate(LOC_NAMES):
            mm = f['loc_code'][m] == i
            if mm.sum() >= 30:
                e['by_loc'][nm] = {'n': int(mm.sum()), 'agreement': _r(np.mean(a[mm] == b[mm]), 4),
                                   'good_main': _r(a[mm].mean(), 4), 'good_' + other: _r(b[mm].mean(), 4)}
        for s, sn in ((0, 'human'), (1, 'ai')):
            mm = f['source'][m] == s
            if mm.any():
                e['by_source'][sn] = {'n': int(mm.sum()), 'agreement': _r(np.mean(a[mm] == b[mm]), 4),
                                      'good_main': _r(a[mm].mean(), 4), 'good_' + other: _r(b[mm].mean(), 4)}
        # 食い違いがどこに多いか: 衝突の前後(1秒前〜2秒後)、場所
        nc = f['near_collision'][m]
        e['near_collision'] = {'n': int(nc.sum()), 'agreement': _r(np.mean(a[nc] == b[nc]), 4) if nc.any() else None,
                               'agreement_elsewhere': _r(np.mean(a[~nc] == b[~nc]), 4) if (~nc).any() else None}
        agree[other] = e
    if 'A_course' in out:
        # 当てはめのある AI のフレームの中だけで上位 35% を選び直す(offline_rl.md 1.5 と同じ比べ方)
        m = eligible & np.isfinite(out['A']) & np.isfinite(out['A_course']) & (f['source'] == 1)
        agree['course_subset_top35'] = top_share_agreement(out['A'][m], out['A_course'][m], f['loc_code'][m])
        agree['course_subset_top35']['rule'] = ('当てはめのある AI のログの、方策に使える AI のフレームだけで、本命の A とコース図の'
                                                '進みの A のそれぞれ上位 %.0f%% を「良い」にして比べる' % (100 * GOOD_FRAC))
        # V を使わない 1秒の割引報酬和(罰なし): 前進距離 と コース図の進み(1.5 の 83% の再現)
        zeros = np.zeros(len(f['scene']))
        s_fwd, _, _ = advantages_for(f, zeros, 'fwd')
        s_course, _, _ = advantages_for(f, zeros, 'fwd', course=True)
        m = np.isfinite(s_fwd) & np.isfinite(s_course) & (f['source'] == 1)
        agree['reward_sum_1s_fwd_vs_course'] = top_share_agreement(s_fwd[m], s_course[m], f['loc_code'][m])
        agree['reward_sum_1s_fwd_vs_course']['rule'] = ('V を使わない 1秒の割引報酬和(罰なし)。AI のフレーム(当てはめのある5ログ)'
                                                        'で、前進距離とコース図の進みのそれぞれ上位 %.0f%%' % (100 * GOOD_FRAC))
    meta['agreement_with_main'] = agree
    meta['counts'] = {'frames': int(len(f['scene'])), 'eligible': int(eligible.sum()), 'human_correction': int(hc.sum()),
                      'human_correction_scored': int(f['human_correction_scored'].sum())}
    out['meta'] = np.array(json.dumps(meta, ensure_ascii=False))
    _save_npz(p, **out)
    _save_json(out_dir(args.dataset) / 'advantages.json', meta)
    print(json.dumps(meta, ensure_ascii=False, indent=1)[:4000])


# ---------------------------------------------------------------- 5. 確認(CPU。生ログからジャイロの起点を作り直す)
def course_passes(f, raw_dir=None):
    """R1 の起点ごとの通過: 起点・緑マット出口の時刻 [µs]、進入 eRPM(起点の手前 0.4秒の最大)、DE 出口のヨー
    (緑マット出口の 0.2〜0.8秒後の平均、左 +)、R1・DE の衝突(data_check と同じ規則)"""
    passes = []
    for log in sorted(set(f['log'].tolist())):
        info, lab = _labels(log)
        if not info.get('r1_anchors'):
            continue
        L = le.load_log(log, raw_dir)
        V, I = L['VEHICLE'], L['IMU']
        t0 = info['t0_us']
        sec = lambda x: (np.asarray(x, np.int64) - t0) / 1e6      # noqa: E731
        vt = sec(V['timestamps_us'])
        verpm = V['erpm'].astype(np.float64)
        it = sec(I['timestamps_us'])
        gz = I['gyro'][:, 2].astype(np.float64)
        still = (np.abs(np.interp(it, vt, verpm)) < 50) & (np.abs(gz) < 5)
        hd, gzu = le.heading_deg(it, gz, still)
        anchors, gm, _ = le.course_anchors(it, hd, gzu, vt, verpm)
        cs = info['events']['collisions']
        for j, ta in enumerate(anchors):
            tg = gm[j]
            w = (vt >= ta - 0.4) & (vt <= ta)
            entry = float(np.max(-verpm[w])) if w.any() else np.nan
            c_r1 = [c for c in cs if c['loc'] == 'R1' and ta - 0.5 <= c['t_s'] <= (tg - 0.3 if np.isfinite(tg) else ta + 3.5)]
            c_de = [c for c in cs if c['loc'] == 'DE' and np.isfinite(tg) and tg - 0.3 <= c['t_s'] <= tg + 2.3]
            yaw = np.nan
            if np.isfinite(tg):
                ww = (it >= tg + 0.2) & (it <= tg + 0.8)
                yaw = float(np.mean(gzu[ww])) if ww.any() else np.nan
            k = int(np.clip(np.searchsorted(lab['t_s'], ta), 0, len(lab['t_s']) - 1))
            passes.append({'log': log, 't0_us': int(t0), 'ta_s': _r(ta, 3), 'tg_s': _r(tg, 3) if np.isfinite(tg) else None,
                           'ai': bool(lab['ai'][k]), 'entry_erpm': _r(entry, 0),
                           'r1_collision': bool(c_r1), 'r1_strong': any(c['strong'] for c in c_r1),
                           'de_collision': bool(c_de), 'de_yaw_dps': _r(yaw, 2)})
        del L
    return passes


def _window_stats(rows, A, I, elig):
    a = A[rows]
    i = I[rows]
    e = elig[rows] & np.isfinite(a) & (i >= 0)
    return {'frames': int(len(rows)), 'eligible': int(e.sum()),
            'mean_A': _r(a[e].mean(), 4) if e.any() else None,
            'good_share': _r(np.mean(i[e] == 1), 4) if e.any() else None}


def cmd_sanity(args):
    f = load_frames(args.dataset)
    with np.load(out_dir(args.dataset) / 'advantages.npz') as z:
        adv = {k: z[k] for k in z.files}
    pp = out_dir(args.dataset) / 'passes.json'
    if pp.exists() and not args.force:
        passes = json.loads(pp.read_text())
    else:
        passes = course_passes(f)
        _save_json(pp, passes)
    elig = adv['eligible']
    res = {'note': '印は本命(feat_loc の V、報酬 = 前進距離 − 罰)。eligible = 方策の学習に使えるフレーム', 'by_loc': {}}
    # (a) 場所ごとの「良い」の割合
    for lab_name in ('I', 'I_noloc', 'I_fwd', 'I_course', 'I_neg'):
        if lab_name not in adv:
            continue
        I = adv[lab_name]
        t = {}
        for s, sn in ((0, 'human'), (1, 'ai'), (None, 'all')):
            for i, nm in enumerate(LOC_NAMES):
                m = elig & (I >= 0) & (f['loc_code'] == i) & ((f['source'] == s) if s is not None else True)
                if m.sum() >= 20:
                    t.setdefault(sn, {})[nm] = {'n': int(m.sum()), 'good_share': _r(np.mean(I[m] == 1), 4)}
        res['by_loc'][lab_name] = t
    # (b) R1・DE の通過ごと
    A, I = adv['A'], adv['I']
    rows_by_log = {}
    for lg in set(f['log'].tolist()):
        m = np.nonzero(f['log'] == lg)[0]
        rows_by_log[lg] = m[np.argsort(f['timestamps_us'][m], kind='stable')]
    thr_cmd = []
    for p in passes:
        rows = rows_by_log.get(p['log'])
        if rows is None:
            continue
        t = (f['timestamps_us'][rows] - p['t0_us']) / 1e6
        ta = p['ta_s']
        r1 = rows[(t >= ta - 0.4) & (t <= ta)]
        p['r1_window'] = _window_stats(r1, A, I, elig)
        p['fold'] = int(f['fold'][rows[0]])
        p['split'] = str(f['split'][r1[0]]) if len(r1) else None
        p['src_window'] = int(np.round(np.mean(f['source'][r1]))) if len(r1) else None
        e = r1[elig[r1] & (I[r1] >= 0)]
        if len(e):
            p['r1_throttle_label_mean'] = _r(f['action_label'][e, 1].mean(), 1)
            for i in e:
                thr_cmd.append((int(f['source'][i]), float(f['action_label'][i, 1]), int(I[i]), float(A[i])))
        if p['tg_s'] is not None:
            tg = p['tg_s']
            de = rows[(t >= tg) & (t <= tg + 0.8)]
            p['de_window'] = _window_stats(de, A, I, elig)
            e = de[elig[de] & (I[de] >= 0)]
            if len(e):
                p['de_steer_label_mean'] = _r(f['action_label'][e, 0].mean(), 1)
    res['passes'] = passes

    def r1_summary(ps):
        o = {'passes': len(ps), 'by_entry_band': {}}
        for lo, hi, nm in ((0, 5500, '<5500'), (5500, 6200, '5500-6200'), (6200, 1e9, '>=6200')):
            b = [q for q in ps if q['entry_erpm'] is not None and lo <= q['entry_erpm'] < hi
                 and q['r1_window']['good_share'] is not None]
            if b:
                o['by_entry_band'][nm] = {'passes': len(b), 'strong_hits': sum(q['r1_strong'] for q in b),
                                          'good_share_mean': _r(np.mean([q['r1_window']['good_share'] for q in b]), 3),
                                          'mean_A': _r(np.mean([q['r1_window']['mean_A'] for q in b]), 3)}
        x = [q['entry_erpm'] for q in ps if q['r1_window']['good_share'] is not None and q['entry_erpm'] is not None]
        g = [q['r1_window']['good_share'] for q in ps if q['r1_window']['good_share'] is not None and q['entry_erpm'] is not None]
        a = [q['r1_window']['mean_A'] for q in ps if q['r1_window']['good_share'] is not None and q['entry_erpm'] is not None]
        for nm, sel in (('with_r1_strong_hit', True), ('without_r1_strong_hit', False)):
            b = [q for q in ps if q['r1_strong'] == sel and q['r1_window']['good_share'] is not None]
            if b:
                o[nm] = {'passes': len(b), 'good_share_mean': _r(np.mean([q['r1_window']['good_share'] for q in b]), 3),
                         'entry_erpm_median': _r(np.median([q['entry_erpm'] for q in b if q['entry_erpm'] is not None]), 0)}
        o['spearman_entry_vs_good_share'] = _r(spearman(x, g), 3)
        o['spearman_entry_vs_mean_A'] = _r(spearman(x, a), 3)
        o['n_with_labels'] = len(x)
        return o

    def de_summary(ps):
        ps = [q for q in ps if q.get('de_window') and q['de_window']['good_share'] is not None and q['de_yaw_dps'] is not None]
        o = {'passes': len(ps)}
        if not ps:
            return o
        y = np.array([q['de_yaw_dps'] for q in ps])
        g = np.array([q['de_window']['good_share'] for q in ps])
        a = np.array([q['de_window']['mean_A'] for q in ps])
        o['spearman_yaw_vs_good_share'] = _r(spearman(y, g), 3)
        o['spearman_yaw_vs_mean_A'] = _r(spearman(y, a), 3)
        hit = np.array([q['de_collision'] for q in ps], bool)
        for nm, mm in (('with_de_collision', hit), ('without_de_collision', ~hit)):
            if mm.any():
                o[nm] = {'passes': int(mm.sum()), 'good_share_mean': _r(g[mm].mean(), 3), 'mean_A': _r(a[mm].mean(), 3),
                         'yaw_median': _r(np.median(y[mm]), 1)}
        for nm, mm in (('right_yaw_<-20', y < -20), ('mid_-20..0', (y >= -20) & (y < 0)), ('left_yaw_>=0', y >= 0)):
            if mm.any():
                o[nm] = {'passes': int(mm.sum()), 'good_share_mean': _r(g[mm].mean(), 3), 'mean_A': _r(a[mm].mean(), 3),
                         'de_collisions': int(sum(q['de_collision'] for q, k in zip(ps, mm) if k))}
        return o

    lab_ps = [q for q in passes if 'r1_window' in q]
    res['r1'] = {'ai': r1_summary([q for q in lab_ps if q['src_window'] == 1]),
                 'human': r1_summary([q for q in lab_ps if q['src_window'] == 0]),
                 'rule': 'R1 の起点の手前 0.4秒のフレーム(方策に使えるもの)の「良い」の割合と A の平均。進入 eRPM は'
                         ' 起点の手前 0.4秒の最大'}
    res['de'] = {'ai': de_summary([q for q in lab_ps if q['src_window'] == 1]),
                 'human': de_summary([q for q in lab_ps if q['src_window'] == 0]),
                 'rule': '緑マット出口から 0.8秒のフレームの「良い」の割合と A。ヨーは出口の 0.2〜0.8秒後の平均(左 +)'}
    # R1 の手前のフレームごと: 「良い」のフレームはスロットルの指令が中立に近いか(µs が大きいほど遅い)
    if thr_cmd:
        tc = np.array(thr_cmd)
        o = {}
        for s, sn in ((0, 'human'), (1, 'ai')):
            mm = tc[:, 0] == s
            if mm.sum() > 10:
                good = tc[mm, 2] == 1
                o[sn] = {'frames': int(mm.sum()), 'throttle_us_good': _r(tc[mm][good, 1].mean(), 1) if good.any() else None,
                         'throttle_us_other': _r(tc[mm][~good, 1].mean(), 1) if (~good).any() else None,
                         'spearman_throttle_vs_A': _r(spearman(tc[mm, 1], tc[mm, 3]), 3)}
        res['r1_frames_throttle'] = o
    _save_json(out_dir(args.dataset) / 'sanity.json', res)
    print(json.dumps({k: res[k] for k in ('r1', 'de')}, ensure_ascii=False, indent=1))
    print(json.dumps(res.get('r1_frames_throttle'), ensure_ascii=False))
    print(json.dumps(res['by_loc'].get('I'), ensure_ascii=False))


# ---------------------------------------------------------------- 図
def cmd_figures(args):
    import figures_offline_rl as figs           # noqa: F401(図は別ファイル。tools/figures_offline_rl.py)
    figs.make_all(out_dir(args.dataset), Path(args.fig_dir))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['frames', 'features', 'value', 'evaluate', 'advantages', 'sanity', 'figures', 'all'])
    ap.add_argument('--dataset', default=DATASET)
    ap.add_argument('--run', default=BASE_RUN, help='特徴を取る基準モデル(data/runs/<run>)')
    ap.add_argument('--folds', default=None, help='ログ単位の分割(既定 data/rl/<データセット>/log_folds.json)')
    ap.add_argument('--tracks-dir', default=str(TRACKS_DIR))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--variants', nargs='*', default=None, choices=list(VARIANTS))
    ap.add_argument('--members', type=int, default=None, help='fold ごとのモデル数(内側の検証 fold を替える。既定は全部)')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--boot', type=int, default=500)
    ap.add_argument('--no-direct', dest='direct', action='store_false', help='衝突を直接当てる参考の回帰を省く')
    ap.add_argument('--fig-dir', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if args.batch > 64 or args.workers > 4:
        raise SystemExit('この Jetson では batch ≤ 64・workers ≤ 4 にしてください')
    steps = {'frames': cmd_frames, 'features': cmd_features, 'value': cmd_value, 'evaluate': cmd_evaluate,
             'advantages': cmd_advantages, 'sanity': cmd_sanity, 'figures': cmd_figures}
    if args.command == 'all':
        for name in ('frames', 'features', 'value', 'evaluate', 'advantages', 'sanity'):
            print('==== %s' % name, flush=True)
            steps[name](args)
    else:
        steps[args.command](args)


if __name__ == '__main__':
    main()
