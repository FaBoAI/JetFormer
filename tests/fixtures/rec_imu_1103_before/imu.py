"""IMU: ICM-20948(加速度・ジャイロ)+ AK09916(地磁気、バイパス経由 0x0C)"""

import struct
import time

import numpy as np
from smbus2 import SMBus, i2c_msg

from .. import clock
from .base import Sample, SensorThread

REG_BANK_SEL = 0x7F
# bank 0
WHO_AM_I, USER_CTRL, PWR_MGMT_1, PWR_MGMT_2, INT_PIN_CFG = 0x00, 0x03, 0x06, 0x07, 0x0F
ACCEL_XOUT_H = 0x2D           # accel(6) gyro(6) temp(2) の連続14バイト
# bank 2
GYRO_SMPLRT_DIV, GYRO_CONFIG_1 = 0x00, 0x01
ACCEL_SMPLRT_DIV_1, ACCEL_SMPLRT_DIV_2, ACCEL_CONFIG = 0x10, 0x11, 0x14
# AK09916
MAG_ADDR, MAG_WIA2, MAG_ST1, MAG_HXL, MAG_CNTL2, MAG_CNTL3 = 0x0C, 0x01, 0x10, 0x11, 0x31, 0x32

ACCEL_FS = {2: (0, 16384.0), 4: (1, 8192.0), 8: (2, 4096.0), 16: (3, 2048.0)}
GYRO_FS = {250: (0, 131.0), 500: (1, 65.5), 1000: (2, 32.8), 2000: (3, 16.4)}
G = 9.80665


class ImuSensor(SensorThread):
    name = 'IMU'
    MAX_FAIL_STREAK = 20         # 約0.2秒続けて読めなければ開き直す(1回の通信エラーは飛ばすだけ)
    error_backoff = 0.05

    def __init__(self, cfg):
        super(ImuSensor, self).__init__()
        self.cfg = cfg
        self.bus = None
        self.addr = cfg['address']
        self.period = 1.0 / cfg['rate_hz']
        self._next = 0.0
        self.mag_ok = False
        self._mag = [float('nan')] * 3
        self._mag_div = 0
        self.bus_errors = 0
        self.fail_streak = 0
        self._discard = 0

    def _w(self, reg, val):
        self.bus.write_byte_data(self.addr, reg, val)

    def _bank(self, bank):
        self._w(REG_BANK_SEL, bank << 4)

    def _open(self):
        self.bus = SMBus(self.cfg['bus'])
        self._bank(0)
        who = self.bus.read_byte_data(self.addr, WHO_AM_I)
        if who != 0xEA:
            raise RuntimeError('ICM-20948ではありません (WHO_AM_I=0x%02x)' % who)
        self._w(PWR_MGMT_1, 0x80)           # リセット
        time.sleep(0.1)
        self._bank(0)
        self._w(PWR_MGMT_1, 0x01)           # スリープ解除・最適クロック自動選択
        self._w(PWR_MGMT_2, 0x00)           # 加速度・ジャイロ全軸ON
        time.sleep(0.05)
        a_sel, self.a_scale = ACCEL_FS[self.cfg['accel_range_g']]
        g_sel, self.g_scale = GYRO_FS[self.cfg['gyro_range_dps']]
        self._bank(2)
        self._w(GYRO_SMPLRT_DIV, 10)        # 1.1kHz/(1+10) = 100Hz
        self._w(GYRO_CONFIG_1, (g_sel << 1) | (3 << 3) | 1)   # DLPF有効(約51Hz)
        self._w(ACCEL_SMPLRT_DIV_1, 0)
        self._w(ACCEL_SMPLRT_DIV_2, 10)     # 1.125kHz/(1+10) ≈ 102Hz
        self._w(ACCEL_CONFIG, (a_sel << 1) | (3 << 3) | 1)    # DLPF有効(約50Hz)
        self._bank(0)
        self.mag_ok = False
        if self.cfg.get('mag'):
            try:
                self._w(USER_CTRL, 0x00)        # I2Cマスタ無効
                self._w(INT_PIN_CFG, 0x02)      # バイパス有効 → AK09916が0x0Cに見える
                time.sleep(0.01)
                if self.bus.read_byte_data(MAG_ADDR, MAG_WIA2) == 0x09:
                    self.bus.write_byte_data(MAG_ADDR, MAG_CNTL3, 0x01)   # ソフトリセット
                    time.sleep(0.01)
                    self.bus.write_byte_data(MAG_ADDR, MAG_CNTL2, 0x08)   # 連続測定100Hz
                    self.mag_ok = True
            except OSError:
                self.mag_ok = False
        self._next = time.monotonic()
        self._discard = 20       # リセット直後(0.2秒)の値はフィルタが落ち着いておらず乱れるので捨てる

    def _close(self):
        if self.bus is not None:
            self.bus.close()
            self.bus = None

    def _read_mag(self):
        # ST1(DRDY)からST2まで9バイトを1回で読む(ST2の読み出しで次の計測が有効になる)
        msg_w = i2c_msg.write(MAG_ADDR, [MAG_ST1])
        msg_r = i2c_msg.read(MAG_ADDR, 9)
        self.bus.i2c_rdwr(msg_w, msg_r)
        raw = bytes(msg_r)
        if raw[0] & 0x01:
            x, y, z = struct.unpack_from('<hhh', raw, 1)
            self._mag = [x * 0.15, y * 0.15, z * 0.15]      # µT

    def _poll(self):
        now = time.monotonic()
        if now < self._next:
            time.sleep(min(self._next - now, 0.2))
        self._next = max(self._next + self.period, time.monotonic() - self.period)
        try:
            self.bus.write_byte_data(self.addr, REG_BANK_SEL, 0)
            msg_w = i2c_msg.write(self.addr, [ACCEL_XOUT_H])
            msg_r = i2c_msg.read(self.addr, 14)
            self.bus.i2c_rdwr(msg_w, msg_r)
        except OSError as e:                     # 走行中のモーターのノイズ: その周期だけ飛ばす(車両データと同じ)
            self.bus_errors += 1
            self.fail_streak += 1
            if self.fail_streak >= self.MAX_FAIL_STREAK:
                raise RuntimeError('IMUを約%.1f秒続けて読めません: %s' % (self.fail_streak * self.period, e))
            return []
        self.fail_streak = 0
        t_ns = clock.now_ns()
        ax, ay, az, gx, gy, gz, temp = struct.unpack('>hhhhhhh', bytes(msg_r))
        if self._discard > 0:
            self._discard -= 1
            return []
        if self.mag_ok:
            self._mag_div = (self._mag_div + 1) % 2          # 地磁気は50Hzで十分
            if self._mag_div == 0:
                try:
                    self._read_mag()
                except OSError:
                    pass
        data = {
            'accel': np.array([ax, ay, az], dtype=np.float32) / self.a_scale * G,      # m/s^2
            'gyro': np.array([gx, gy, gz], dtype=np.float32) / self.g_scale,           # deg/s
            'mag': np.array(self._mag, dtype=np.float32),                              # µT
            'temp': (temp - 21) / 333.87 + 21.0,                                      # ℃
        }
        return [Sample(t_ns, data)]

    def status(self):
        st = super(ImuSensor, self).status()
        st['mag'] = self.mag_ok
        st['bus_errors'] = self.bus_errors
        return st
