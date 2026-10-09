"""VL53L9のゾーン値(u16)の解釈(JetFormer基板 csi_ranging ファームの仕様)

    下位13bit = 距離 [mm](0x1FFF = 測定範囲外/クランプ → 無効)
    上位 3bit = 信頼度 0〜7

収集データ(TOF_FRONT.npz、データセットの16bit PNG、aligned の tof)は生の値のまま保存し、
使う側でここの関数を通して解釈する(後から解釈を変えても情報を失わないように)。
学習環境でも使うので numpy 以外に依存しない。
"""

import numpy as np

DIST_MASK = 0x1FFF
INVALID = 0x1FFF
CONF_SHIFT = 13
CONF_MAX = 7


def distance_mm(raw):
    """生の値 → 距離[mm](float32)。無効ゾーンは 0"""
    d = (np.asarray(raw) & DIST_MASK).astype(np.float32)
    d[d == INVALID] = 0.0
    return d


def confidence(raw):
    """生の値 → 信頼度 0〜7(uint8)"""
    return (np.asarray(raw) >> CONF_SHIFT).astype(np.uint8)


def valid_mask(raw):
    d = np.asarray(raw) & DIST_MASK
    return (d != INVALID) & (d != 0)
