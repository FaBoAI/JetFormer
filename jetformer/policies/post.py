"""推論の後処理(numpy だけ。torch を読まない)。走行(drive._loop)と評価の道具が同じ手順(Stage)で使う

    stage = post.Stage(model_config, ref=(ステアの中心, スロットルの中立))
    res = runner.run_numpy(**feed)                        # または runtime.predict(model, feed)。{名前: [1,...]}
    model_pred = stage.model_pred(res)                    # actions の µs [10,2](後処理・重ね合わせの前)
    pred, info = stage.apply(res, model_pred, v_now, params, t_ns)   # 後処理(あれば)→ 重ね合わせ(ensemble_j > 0)

後処理は model_config.json の 'postprocess'(tcp の制御器など)で決まる。無い方策・知らない kind・読み込めないときは
後処理なし(操作の枝 actions そのまま)。後処理のオブジェクトの約束(spec.md §2.8・§3.2.6):
    obj.kind / obj.reset() / obj.step(outputs, v_now, params, t_ns) → (pred_us [10,2] float32, info dict)
    outputs: {名前: np.ndarray}(バッチの次元を外したもの。actions は正規化のまま [10,2])
    v_now: 今の速さ [m/s] = -eRPM × 0.5e-3(後退は負)、params: 走行のパラメータ(dict)、t_ns: 撮像時刻
    info: {'mode': int, 'branch': str, 'v_hat': float, 'v_cmd': float, 'v_curv': float(nan 可), 'cap': float,
           'thr_cap': float(上限のスロットル µs。ゲインの前の値。上限なしは nan。無くてもよい)}
info に有限の 'thr_cap' があれば、Stage は予測の重ね合わせの **後にも** スロットルを max(·, thr_cap) で掛け直す(重ね合わせで
前のフレームの前進の行と今のブレーキの行が平均され、上限のブレーキが弱まらないように。レビュー 2026-10-12)。
"""

import importlib

import numpy as np

from . import PWM_CENTER, PWM_SCALE
from .ensemble import TemporalEnsembler

KINDS = {'tcp_controller': ('jetformer.policies.controller', 'TCPPost')}   # テストでは monkeypatch で偽物を足す
KIND_TITLES = {'tcp_controller': 'TCP の制御器'}
SPEED_PER_ERPM = 0.5e-3            # 速さ [m/s] ≒ -eRPM × 0.5e-3(65 mm のタイヤ。eRPM が負 = 前進)


class PostError(Exception):
    """後処理の失敗(走行は中立で止める)"""


def speed_of(erpm):
    """eRPM → 速さ [m/s](前進が正)"""
    return -float(erpm) * SPEED_PER_ERPM


def kind_of(cfg):
    pp = (cfg or {}).get('postprocess')
    return pp.get('kind') if isinstance(pp, dict) else (pp or None)


def make(cfg, ref):
    """cfg = model_config(または drive.model_info。'postprocess' を見る)、ref = (ステアの中心, スロットルの中立) µs。
    → (後処理のオブジェクト or None, 説明)。'postprocess' が無い → (None, '')。
    知らない kind・読み込めない → (None, '後処理 <kind> を使えないため、操作の枝(actions)だけで走ります')"""
    pp = (cfg or {}).get('postprocess')
    if not pp:
        return None, ''
    kind = kind_of(cfg)
    note = '後処理 %s を使えないため、操作の枝(actions)だけで走ります' % kind
    if not isinstance(pp, dict) or kind not in KINDS:
        return None, note
    try:
        mod_name, cls_name = KINDS[kind]
        cls = getattr(importlib.import_module(mod_name), cls_name)
        obj = cls(pp, (float(ref[0]), float(ref[1])))
    except Exception as e:          # noqa: BLE001  後処理が無くても操作の枝で走れる
        return None, '%s(%s: %s)' % (note, type(e).__name__, e)
    return obj, '後処理: %s' % KIND_TITLES.get(kind, kind)


class Stage(object):
    """推論の出力 → 後処理(model_config の postprocess があれば)→ 予測の重ね合わせ(params の ensemble_j > 0 のとき)。
    走行と評価で同じ手順にするためのもの。ensemble_j = 0 で後処理が無ければ、actions の µs をそのまま返す"""

    def __init__(self, cfg, ref, fps=30):
        self.post, self.note = make(cfg, ref)
        # model_config に後処理が書いてあるのに読めなかった(操作の枝だけで走る)。走行は速度の上限などを受け付けない
        self.failed = bool((cfg or {}).get('postprocess')) and self.post is None
        self.ens = TemporalEnsembler(fps=fps)

    def reset(self):
        """中立で止めた後・シーンの切れ目"""
        if self.post is not None:
            self.post.reset()
        self.ens.reset()

    @staticmethod
    def model_pred(res):
        """エンジン(run_numpy)・runtime.predict の出力 {名前: [1,...]} → actions の µs [10,2](float32)"""
        return res['actions'][0] * PWM_SCALE + PWM_CENTER

    def apply(self, res, model_pred, v_now, params, t_ns):
        """→ (pred µs [10,2], info or None)。後処理の例外・数値でない結果は PostError"""
        pred, info = model_pred, None
        if self.post is not None:
            try:
                pred, info = self.post.step({k: np.asarray(v)[0] for k, v in res.items()}, float(v_now), params, t_ns)
            except Exception as e:      # noqa: BLE001
                raise PostError('後処理でエラー: %s: %s' % (type(e).__name__, e))
            pred = np.asarray(pred, np.float32)
            if pred.shape != np.shape(model_pred) or not np.all(np.isfinite(pred)):
                raise PostError('後処理の結果が数値ではありません')
        j = int(params.get('ensemble_j', 0) or 0)
        if j > 0:
            pred = self.ens.push(t_ns, pred, j, float(params.get('ensemble_m', 0.5)))
            lim = cap_limit(info)
            if lim is not None:                 # 上限は重ね合わせの後にも(今のフレームの上限で)掛け直す
                pred = pred.copy()
                pred[:, 1] = np.maximum(pred[:, 1], np.float32(lim))
        return pred, info


def cap_limit(info):
    """後処理の info → 上限のスロットル µs(ゲインの前の値。無い・数値でなければ None)"""
    try:
        v = float((info or {}).get('thr_cap', float('nan')))
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def replay(cfg, ref, outputs, erpm, t_ns, params, fps=30, max_gap=2, scene=None):
    """保存した予測を、走行と同じ Stage にフレームの時刻の順に通す(評価の道具用)。

    outputs: {名前: [N,...]}(runtime.predict・エンジンの出力をフレームの順に積んだもの。actions は正規化のまま [N,10,2])
    erpm [N]・t_ns [N](撮像時刻。シーンの中で増える順)。scene [N](あれば、シーンが変わったら reset)。
    抜けたフレームが max_gap 個以上(時刻で数える。続いたフレームの間隔 = 1)なら reset。
    → (pred µs [N,10,2] float32, model_pred [N,10,2], infos)"""
    stage = Stage(cfg, ref, fps=fps)
    n = len(t_ns)
    preds = np.zeros((n, 10, 2), np.float32)
    model_preds = np.zeros((n, 10, 2), np.float32)
    infos = []
    prev_t, prev_s = None, None
    for i in range(n):
        s = None if scene is None else scene[i]
        if prev_t is not None:
            gap = int(round((int(t_ns[i]) - prev_t) * fps / 1e9))
            if s != prev_s or gap < 0 or gap - 1 >= max_gap:
                stage.reset()
        res = {k: np.asarray(v[i:i + 1]) for k, v in outputs.items()}
        mp = stage.model_pred(res)
        pred, info = stage.apply(res, mp, speed_of(erpm[i]), params, int(t_ns[i]))
        preds[i], model_preds[i] = pred, mp
        infos.append(info)
        prev_t, prev_s = int(t_ns[i]), s
    return preds, model_preds, infos
