"""時刻の扱い

全センサーの時刻はCLOCK_MONOTONIC(ns)で記録し、保存時にnuScenesと同じ
「UNIXエポックからのマイクロ秒」に変換する。変換オフセットはプロセス起動時に
1回だけ決め、収集中にNTP等で壁時計が動いても時刻列が飛ばないようにする。
"""

import time

_OFFSET_NS = time.time_ns() - time.monotonic_ns()


def now_ns():
    """CLOCK_MONOTONIC [ns]"""
    return time.monotonic_ns()


def to_epoch_us(monotonic_ns):
    """CLOCK_MONOTONIC [ns] → UNIXエポック [µs](nuScenesのtimestamp形式)"""
    return (int(monotonic_ns) + _OFFSET_NS) // 1000


def timeval_to_ns(tv_sec, tv_usec):
    """V4L2バッファのtimestamp(CLOCK_MONOTONIC)→ ns"""
    return int(tv_sec) * 1_000_000_000 + int(tv_usec) * 1000
