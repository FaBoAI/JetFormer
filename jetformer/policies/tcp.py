"""tcp — TCP / TransFuser++ 型(通過点 + 目標速度 + 操作の枝)。spec.md §3.2

    blocks.sensor_tokens + pos [1,94,256] → Encoder 4層 → enc_norm = memory
    queries [1,21,256](通過点 10・速度 1・操作 10)→ Decoder 2層(クエリどうしの自己注意 + memory への交差注意)→ dec_norm → hs
      hs[:, 0:10]  → heads_wp: Linear(256, M×2)  → select_mode → Δ(x, y) [B,10,2] → 累積和 → waypoints [B,10,2]
                     (m、自車の座標 x = 前・y = 左、原点 = 撮像時刻の車、0.1 秒刻みで 1 秒先まで)
      hs[:, 10]    → heads_v : Linear(256, M×10) → select_mode → speed_logits [B,10](0, 0.5, …, 4.5 m/s のクラス。期待値 = 目標速度)
      hs[:, 11:21] → heads   : Linear(256, M×2)  → select_mode → actions [B,10,2](操作の枝。v0 と同じ意味・正規化)

TCP の「軌道に導かれた注意」は、同じ Decoder の自己注意で操作のクエリが通過点のクエリを参照できる形に簡単にした(GRU は使わない。
TensorRT でループを作らない)。トークン化と層の名前は v0 と同じ(blocks.add_sensor_tokens)。

損失(policy_loss。重みは先に固定、検証で調整しない):
    l1 = |actions − target|.mean((1,2))、v = y_wp_valid
    wp = |waypoints − y_waypoints|.mean((1,2))·v [m]、ce = −(twohot(y_v_tgt)·log_softmax(speed_logits)).sum(1)·v
    loss = l1 + w_wp·wp + w_speed·ce(既定 0.5・0.1)。検証で報告する誤差は、どの方策とも同じ actions の L1

制御器は ONNX の外(jetformer/policies/controller.py の TCPPost、numpy)。走行の既定(tcp_mode 0)は actions そのまま。
学習の前に prepare が正解(data/derived/<データセット>/tcp_labels_v1.npz)と較正(tcp_calib_v1.json)を作り、
finalize が較正を model_config.json の 'postprocess' に書く(学習結果ごとに当てはめ直さない。検証のシーンは使わない)。
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import FUTURE_STEPS, blocks
from . import controller as ctl
from . import tcp_labels

CUMSUM_KINDS = ('op', 'matmul')
LABEL_VERSIONS = ('tcp_labels_v1',)


def bin_centers(n, vmax):
    """目標速度のクラスの中心 [n](0〜vmax の等間隔)"""
    return [round(float(x), 6) for x in np.linspace(0.0, float(vmax), int(n))]


def twohot(v, bins):
    """速さ [B] → 隣り合う2クラスに線形で分けた分布 [B,n](例 2.3 → 2.0 に 0.4・2.5 に 0.6)。bins は等間隔。
    範囲の外は端に切る(0 未満 → 最初 1.0、最後より大 → 最後 1.0)"""
    n = bins.shape[0]
    v = v.float().clamp(float(bins[0]), float(bins[-1]))
    pos = (v - bins[0]) / (bins[1] - bins[0])
    lo = pos.floor().clamp(0, n - 2)
    frac = (pos - lo).clamp(0.0, 1.0)
    lo = lo.long().unsqueeze(1)
    q = torch.zeros(v.shape[0], n, dtype=torch.float32, device=v.device)
    q.scatter_(1, lo, (1.0 - frac).unsqueeze(1))
    q.scatter_add_(1, lo + 1, frac.unsqueeze(1))
    return q


class TCPPolicy(nn.Module):
    EXPORT_OUTPUTS = ('actions', 'waypoints', 'speed_logits')

    def __init__(self, num_modes, dim=256, enc_layers=4, dec_layers=2, heads=8, ffn=512, dropout=0.1, wp_steps=10,
                 wp_dt=0.1, speed_bins=10, speed_max=4.5, w_wp=0.5, w_speed=0.1, cumsum='op', labels='tcp_labels_v1',
                 pretrained=False):
        super().__init__()
        if cumsum not in CUMSUM_KINDS:
            raise ValueError('cumsum は %s のどれか(指定 %s)' % (CUMSUM_KINDS, cumsum))
        if labels not in LABEL_VERSIONS:
            raise ValueError('labels は %s のどれか(指定 %s)' % (LABEL_VERSIONS, labels))
        if int(wp_steps) < 1 or int(speed_bins) < 2 or not float(speed_max) > 0 or not float(wp_dt) > 0:
            raise ValueError('wp_steps ≥ 1・speed_bins ≥ 2・speed_max > 0・wp_dt > 0 にしてください')
        self.wp_steps, self.wp_dt, self.n_bins = int(wp_steps), float(wp_dt), int(speed_bins)
        self.w_wp, self.w_speed, self.cumsum = float(w_wp), float(w_speed), cumsum
        blocks.add_sensor_tokens(self, dim, num_modes, pretrained)          # v0 と同じ名前・同じ初期化
        self.pos = nn.Parameter(torch.zeros(1, blocks.N_SENSOR_TOKENS, dim))
        self.encoder = blocks.encoder(dim, heads, ffn, enc_layers, dropout)
        self.enc_norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.zeros(1, self.wp_steps + 1 + FUTURE_STEPS, dim))
        self.decoder = blocks.decoder(dim, heads, ffn, dec_layers, dropout)
        self.dec_norm = nn.LayerNorm(dim)
        # ジャンル別の頭(出力の index = ジャンル × 行 + 行の中。genre_keys の heads_wp: 2・heads_v: 10・heads: 2)
        self.heads_wp = nn.Linear(dim, num_modes * 2)
        self.heads_v = nn.Linear(dim, num_modes * self.n_bins)
        self.heads = nn.Linear(dim, num_modes * 2)
        self.register_buffer('tril', torch.tril(torch.ones(self.wp_steps, self.wp_steps)), persistent=False)
        self.register_buffer('bins', torch.tensor(bin_centers(self.n_bins, speed_max), dtype=torch.float32),
                             persistent=False)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.loss_parts = {}

    def forward(self, image, tof, imu, state, mode):
        b, m, n, w = image.shape[0], self.num_modes, FUTURE_STEPS, self.wp_steps
        tok = blocks.sensor_tokens(self, image, tof, imu, state, mode) + self.pos
        memory = self.enc_norm(self.encoder(tok))
        hs = self.dec_norm(self.decoder(self.queries.expand(b, -1, -1), memory))         # [B,21,D]
        delta = blocks.select_mode(self.heads_wp(hs[:, :w]).view(b, w, m, 2).transpose(1, 2), mode)   # [B,W,2]
        if self.cumsum == 'op':
            waypoints = delta.cumsum(dim=1)                                               # ONNX の CumSum
        else:
            waypoints = torch.matmul(self.tril.to(delta.dtype), delta)                    # 下三角の定数行列(同じ結果)
        speed_logits = blocks.select_mode(self.heads_v(hs[:, w]).view(b, m, self.n_bins), mode)      # [B,n]
        actions = blocks.select_mode(self.heads(hs[:, w + 1:]).view(b, n, m, 2).transpose(1, 2), mode)  # [B,10,2]
        return {'actions': actions, 'waypoints': waypoints, 'speed_logits': speed_logits}

    def policy_loss(self, out, target, extras):
        """サンプルごとの損失 [B]。通過点・速度の損失は wp_valid のフレームだけ(無効は 0)"""
        l1 = (out['actions'].float() - target.float()).abs().mean(dim=(1, 2))
        if not extras or 'y_waypoints' not in extras or 'y_v_tgt' not in extras or 'y_wp_valid' not in extras:
            raise RuntimeError('TCP 型の学習には通過点・目標速度の正解(y_waypoints・y_v_tgt・y_wp_valid)が要ります'
                               '(tools/train.py は tcp.extra_targets で渡します)')
        v = extras['y_wp_valid'].float().reshape(-1)
        wp = (out['waypoints'].float() - extras['y_waypoints'].float()).abs().mean(dim=(1, 2)) * v
        q = twohot(extras['y_v_tgt'].reshape(-1), self.bins)
        ce = -(q * F.log_softmax(out['speed_logits'].float(), dim=1)).sum(dim=1) * v
        self.loss_parts = {'wp_m': wp.mean().detach(), 'ce': ce.mean().detach()}
        return l1 + self.w_wp * wp + self.w_speed * ce


def build(num_modes, pretrained=False, **cfg):
    return TCPPolicy(num_modes, pretrained=pretrained, **cfg)


# ==================================================================== tools/train.py が呼ぶモジュールの関数
def _label_shape(cfg):
    cfg = cfg or {}
    if cfg.get('labels', 'tcp_labels_v1') not in LABEL_VERSIONS:
        raise SystemExit('TCP 型の labels は %s のどれか(指定 %s)' % (LABEL_VERSIONS, cfg.get('labels')))
    return int(cfg.get('wp_steps', tcp_labels.WP_STEPS)), float(cfg.get('wp_dt', tcp_labels.WP_DT))


def prepare(data_root, cfg):
    """学習の前に CPU で1回: 正解(通過点・目標速度)と較正が無い・古ければ作る(済んでいれば何もしない)"""
    if not tcp_labels.has_can_bus(data_root):
        raise SystemExit('TCP 型には can_bus の IMU・車両の記録が要ります(%s/can_bus がありません)' % data_root)
    steps, dt = _label_shape(cfg)
    t0 = time.time()
    try:
        path, built = tcp_labels.ensure_labels(data_root, steps, dt)
        cpath, cbuilt = tcp_labels.ensure_calibration(data_root)
    except tcp_labels.LabelError as e:
        raise SystemExit(str(e))
    if built or cbuilt:
        print('TCP 型の正解・較正を作りました(%.1f 秒): %s, %s' % (time.time() - t0, path, cpath), flush=True)
    lab = tcp_labels.load_labels(path)
    print('TCP 型の正解: %s(通過点の有効 %.1f%% / %d フレーム)' % (
        path, 100.0 * float(lab['wp_valid'].mean()), len(lab['wp_valid'])), flush=True)
    return path, cpath


def extra_targets(data_root, cfg):
    """追加の正解 {シーン: {'waypoints' [n,W,2], 'v_tgt' [n], 'wp_valid' [n]}}(n = aligned のフレーム数。
    npz に無いフレームは 0 と無効)。データセットは y_waypoints・y_v_tgt・y_wp_valid として返す"""
    steps, dt = _label_shape(cfg)
    if not tcp_labels.labels_fresh(data_root, steps, dt)[0]:
        prepare(data_root, cfg)
    return tcp_labels.labels_by_scene(data_root, steps, dt)


def postprocess_from(calib, cfg, labels_sha1=None):
    """較正(tcp_calib_v1.json の dict)+ 方策の設定 → model_config.json の 'postprocess'(spec.md §3.2.5)"""
    cfg = cfg or {}
    st, ff = calib['steer_table'], calib['throttle_ff']
    return {
        'kind': ctl.KIND, 'version': ctl.VERSION, 'pwm_center': 1500.0, 'pwm_scale': 500.0,
        'speed_bins': bin_centers(cfg.get('speed_bins', 10), cfg.get('speed_max', 4.5)),
        'wp_dt': float(cfg.get('wp_dt', tcp_labels.WP_DT)), 'fps': ctl.FPS,
        'steer_table': {'lag_ms': st['lag_ms'], 'offset_us': list(st['offset_us']), 'bands_mps': list(st['bands_mps']),
                        'curv': [list(r) for r in st['curv']], 'center_us': st.get('center_us')},
        'throttle_ff': {'c0': ff['c0'], 'c1': ff['c1'], 'c2': ff['c2'], 'tau_s': ff['tau_s'], 'fwd_max_us': ff['fwd_max_us'],
                        'neutral_us': ff.get('neutral_us')},
        'rules': dict(ctl.RULES), 'pursuit': dict(ctl.PURSUIT), 'fusion': dict(ctl.FUSION),
        'calib_source': {'dataset': calib.get('dataset'), 'scenes': calib.get('scenes', 'train'),
                         'created_at': calib.get('created_at'), 'labels_sha1': labels_sha1,
                         'splits_sha1': calib.get('splits_sha1'), 'throttle_train_mae_us': ff.get('train_mae_us'),
                         'steer_r2_by_lag': st.get('r2_by_lag')},
    }


def finalize(run_dir, cfg, data_root):
    """書き出しの後に CPU で: 較正を読み、model_config.json に足す {'postprocess': {...}} を返す(当てはめ直さない)。
    学習の後なので止まらない: 較正が読めなければ警告して {}(後処理なし = 走行は操作の枝 actions だけ)"""
    steps, dt = _label_shape(cfg)
    try:
        tcp_labels.ensure_calibration(data_root)
        lpath = tcp_labels.labels_path(data_root, steps, dt)
        sha = tcp_labels.file_sha1(lpath) if lpath.exists() else None
        pp = postprocess_from(tcp_labels.load_calibration(data_root), cfg, sha)
        ctl.TCPPost(pp, (1500.0, 1500.0))        # 走行で読めるか(形・単調)をここで確かめる
    except (tcp_labels.LabelError, OSError, KeyError, ValueError) as e:
        print('警告: TCP 型の制御器の較正を model_config.json に書けませんでした(走行は操作の枝だけ): %s: %s'
              % (type(e).__name__, e), flush=True)
        return {}
    return {'postprocess': pp}
