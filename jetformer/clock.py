"""時刻の扱い

全センサーの時刻はCLOCK_MONOTONIC(ns)で記録し、保存時にnuScenesと同じ
「UNIXエポックからのマイクロ秒」に変換する。変換オフセットは収集を始めるときに決め、
収集中は変えない(NTP等で壁時計が動いても時刻列が飛ばないように)。
起動直後はNTPの同期前で壁時計がずれていることがある(2026-10-10: 起動2分後に同期し約70秒ずれた)ので、
収集の開始時に resync() で合わせ直す。
"""

import time

_OFFSET_NS = time.time_ns() - time.monotonic_ns()
RESYNC_THRESHOLD_NS = 1_000_000_000      # これ以上ずれていたら(NTPで時計が飛んだ)合わせ直す


def resync():
    """壁時計が飛んでいれば変換オフセットを合わせ直す。合わせ直した量 [ns] を返す(0なら変更なし)"""
    global _OFFSET_NS
    now = time.time_ns() - time.monotonic_ns()
    delta = now - _OFFSET_NS
    if abs(delta) >= RESYNC_THRESHOLD_NS:
        _OFFSET_NS = now
        return delta
    return 0


def now_ns():
    """CLOCK_MONOTONIC [ns]"""
    return time.monotonic_ns()


def to_epoch_us(monotonic_ns):
    """CLOCK_MONOTONIC [ns] → UNIXエポック [µs](nuScenesのtimestamp形式)"""
    return (int(monotonic_ns) + _OFFSET_NS) // 1000


def timeval_to_ns(tv_sec, tv_usec):
    """V4L2バッファのtimestamp(CLOCK_MONOTONIC)→ ns"""
    return int(tv_sec) * 1_000_000_000 + int(tv_usec) * 1000
