"""PCA9685(I2C PWMドライバ)へのパルス幅[µs]の出力

モデルはプロポの受信機と同じ「パルス幅µs」で操作値を出すので、PCA9685でも同じµsを作って
サーボ/ESCへ送る(JetRacerのように左右・前後の向きを別途設定する必要がない)。
"""

import time

from smbus2 import SMBus

MODE1, MODE2, PRESCALE, LED0_ON_L = 0x00, 0x01, 0xFE, 0x06
MODE1_SLEEP, MODE1_AI, MODE1_RESTART, MODE1_ALLCALL = 0x10, 0x20, 0x80, 0x01
MODE2_OUTDRV = 0x04


class PCA9685(object):
    def __init__(self, cfg, osc_scale=1.0):
        self.cfg = cfg
        self.addr = cfg['address']
        self.bus = SMBus(cfg['bus'])
        self.osc_hz = cfg['osc_hz'] * osc_scale
        try:
            self._set_freq(cfg['freq_hz'])
        except Exception:
            self.bus.close()
            raise
        self.last_us = {}

    def _set_freq(self, freq_hz):
        prescale = int(round(self.osc_hz / (4096.0 * freq_hz))) - 1
        prescale = max(3, min(255, prescale))
        self.bus.write_byte_data(self.addr, MODE2, MODE2_OUTDRV)
        old = self.bus.read_byte_data(self.addr, MODE1) & ~MODE1_RESTART
        if self.bus.read_byte_data(self.addr, PRESCALE) != prescale:
            self.bus.write_byte_data(self.addr, MODE1, old | MODE1_SLEEP)     # 周期はスリープ中しか変えられない
            self.bus.write_byte_data(self.addr, PRESCALE, prescale)
            old |= MODE1_SLEEP
        if old & MODE1_SLEEP:
            self.bus.write_byte_data(self.addr, MODE1, old & ~MODE1_SLEEP)
            time.sleep(0.005)                                                  # 発振器の安定待ち(500µs以上)
        self.bus.write_byte_data(self.addr, MODE1, (old & ~MODE1_SLEEP) | MODE1_AI | MODE1_RESTART | MODE1_ALLCALL)
        self.prescale = prescale
        self.period_us = 4096.0 * (prescale + 1) / self.osc_hz * 1e6

    def set_osc_scale(self, osc_scale):
        """時計補正を変える(周期のレジスタはそのまま、µs→カウントの換算だけ変わる)"""
        self.osc_hz = self.cfg['osc_hz'] * osc_scale
        self.period_us = 4096.0 * (self.prescale + 1) / self.osc_hz * 1e6
        for ch, us in list(self.last_us.items()):
            self.set_us(ch, us)

    def counts(self, us):
        return int(round(us / self.period_us * 4096.0))

    def set_us(self, ch, us):
        c = max(0, min(4095, self.counts(us)))
        self.bus.write_i2c_block_data(self.addr, LED0_ON_L + 4 * ch, [0, 0, c & 0xFF, c >> 8])
        self.last_us[ch] = us

    def set_pair_us(self, steer_ch, steer_us, throttle_ch, throttle_us):
        """ステアリングとスロットルを書く。隣り合うチャネルなら1回のI2C書き込みで同時に変える。
        別々に書くときはスロットルを先に書く(途中で失敗してもスロットルだけは新しい値になる)"""
        s = max(0, min(4095, self.counts(steer_us)))
        t = max(0, min(4095, self.counts(throttle_us)))
        if throttle_ch == steer_ch + 1:
            self.bus.write_i2c_block_data(self.addr, LED0_ON_L + 4 * steer_ch, [0, 0, s & 0xFF, s >> 8, 0, 0, t & 0xFF, t >> 8])
        elif steer_ch == throttle_ch + 1:
            self.bus.write_i2c_block_data(self.addr, LED0_ON_L + 4 * throttle_ch, [0, 0, t & 0xFF, t >> 8, 0, 0, s & 0xFF, s >> 8])
        else:
            self.set_us(throttle_ch, throttle_us)
            self.set_us(steer_ch, steer_us)
            return
        self.last_us[steer_ch], self.last_us[throttle_ch] = steer_us, throttle_us

    def off(self, ch):
        self.bus.write_i2c_block_data(self.addr, LED0_ON_L + 4 * ch, [0, 0, 0, 0x10])   # 完全OFF(パルスなし)
        self.last_us.pop(ch, None)

    def close(self):
        self.bus.close()


def outputs_off(cfg):
    """PCA9685が動いていれば、ステアリング・スロットルのチャネルをパルスなしにする(サーバー起動時・異常終了時の後始末)。
    異常終了したプロセスが残した出力(最後のAIの指令)を止めるため。スリープ中(電源投入直後)なら何もしない"""
    with SMBus(cfg['bus']) as bus:
        if bus.read_byte_data(cfg['address'], MODE1) & MODE1_SLEEP:
            return False
        for ch in (cfg['throttle_ch'], cfg['steering_ch']):
            bus.write_i2c_block_data(cfg['address'], LED0_ON_L + 4 * ch, [0, 0, 0, 0x10])
        return True
