#!/usr/bin/env python3
"""Jetson の電源(INA3221 の電圧・電流)と温度を読むだけの記録係(起動のサービス jetformer-power。走りのプログラムとは別)

2026-10-30 ユーザーの決め(fix_20261026/user_decisions.md 10/30 00:5x):「AS5601(ハンドルの角度)と電源(電力・温度)の記録係を
起動で自動に始める」。10/29 の夜は現地で Jetson の電源が 3 回切れ、カーネルに警告が無く、直前 30〜50 秒の記録が失われた。
電源が切れる直前の入力の電圧・電流が残るように、1 秒に 10 回読み、1 秒ごとに保存(fsync)する。
読むのは sysfs だけ(/sys/class/hwmon の ina3221・/sys/class/thermal)。I2C の装置に直に触れない・何も書かない。
nice は 10(unit の Nice=10 と足さない)。書けないときは 60 秒待つ・空きが 2 GB 未満なら書かずに待つ(録画と同じディスク)。
時計: epoch_us は time.time()(壁の時計)。走りの記録と合わせるときは mono_us(CLOCK_MONOTONIC)と、json の boot_id・
rt_minus_mono_us を使う(tools/as5601_logger.py の説明と同じ)。
出力: <out>/<YYYYMMDD-HHMMSS>.csv と同じ名前の .json(始めの情報)。CSV(行末は \\n)の欄:
  epoch_us(UNIX µs)・mono_us・vin_mv・iin_ma(VDD_IN = モジュールの入力)・vcpu_mv・icpu_ma(VDD_CPU_GPU_CV)・
  vsoc_mv・isoc_ma(VDD_SOC)・temps(1 秒に 1 回。「名前=℃」を ; でつなぐ)・err(読めなかった数)

    cd ~/jetformer
    python3 -I tools/power_logger.py --duration 10        # 10 秒で止める(既定の書く場所は data/power/)
    python3 -I tools/power_logger.py --rotate-s 1800      # 止めるまで(30 分ごとに新しいファイル)。起動のサービスはこの形
"""
import argparse
import csv
import glob
import json
import os
import signal
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from as5601_logger import MIN_FREE, WRITE_ERRNOS, boot_id, free_bytes, lower_priority, sleep_while  # noqa: E402

FIELDS = ['epoch_us', 'mono_us', 'vin_mv', 'iin_ma', 'vcpu_mv', 'icpu_ma', 'vsoc_mv', 'isoc_ma', 'temps', 'err']
CHANNELS = (('VDD_IN', 'vin_mv', 'iin_ma'), ('VDD_CPU_GPU_CV', 'vcpu_mv', 'icpu_ma'), ('VDD_SOC', 'vsoc_mv', 'isoc_ma'))
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'power')


def _read_int(path):
    with open(path) as fh:
        return int(fh.read().strip())


class Sensors(object):
    """sysfs を読むだけ。hwmon_root・thermal_root は試験では偽物のフォルダ"""

    def __init__(self, hwmon_root='/sys/class/hwmon', thermal_root='/sys/class/thermal'):
        self.ina = None
        for d in sorted(glob.glob(os.path.join(hwmon_root, 'hwmon*'))):
            try:
                with open(os.path.join(d, 'name')) as fh:
                    if fh.read().strip() == 'ina3221':
                        self.ina = d
                        break
            except OSError:
                continue
        self.ch = {}                                   # 欄の名前 → 値のファイル
        if self.ina:
            for n in range(1, 8):
                lab = os.path.join(self.ina, 'in%d_label' % n)
                if not os.path.exists(lab):
                    continue
                with open(lab) as fh:
                    name = fh.read().strip()
                for label, vk, ik in CHANNELS:
                    if name == label:
                        self.ch[vk] = os.path.join(self.ina, 'in%d_input' % n)
                        self.ch[ik] = os.path.join(self.ina, 'curr%d_input' % n)
        self.zones = []                                # [(名前, temp のファイル)]
        for z in sorted(glob.glob(os.path.join(thermal_root, 'thermal_zone*'))):
            try:
                with open(os.path.join(z, 'type')) as fh:
                    self.zones.append((fh.read().strip(), os.path.join(z, 'temp')))
            except OSError:
                continue

    def power(self):
        """{欄: 値} と 読めなかった数"""
        out, err = {}, 0
        for k, p in self.ch.items():
            try:
                out[k] = _read_int(p)
            except (OSError, ValueError):
                err += 1
        return out, err

    def temps(self):
        parts = []
        for name, p in self.zones:
            try:
                parts.append('%s=%.1f' % (name, _read_int(p) / 1000.0))
            except (OSError, ValueError):
                continue
        return ';'.join(parts)


def run(sensors, out_dir, hz=10.0, duration=None, fsync_s=1.0, temp_s=1.0, clock=time, stop=None):
    os.makedirs(out_dir, exist_ok=True)
    name = base = datetime.fromtimestamp(clock.time()).strftime('%Y%m%d-%H%M%S')
    k = 2
    while os.path.exists(os.path.join(out_dir, name + '.csv')):
        name = '%s-%d' % (base, k)
        k += 1
    path = os.path.join(out_dir, name + '.csv')
    meta = {'name': name, 'start_epoch_us': int(clock.time() * 1e6), 'hz': hz, 'fsync_s': fsync_s, 'fields': FIELDS,
            'ina3221': sensors.ina, 'channels': sorted(sensors.ch), 'zones': [z for z, _ in sensors.zones],
            'boot_id': boot_id(), 'rt_minus_mono_us': int((clock.time() - clock.monotonic()) * 1e6),
            'note': '読むだけ(tools/power_logger.py)。走りの記録と合わせるときは mono_us(同じ boot_id の中だけ)'}
    with open(os.path.join(out_dir, name + '.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    period = 1.0 / hz
    n = 0
    vmin = None
    t0 = clock.monotonic()
    t_next, t_sync, t_temp = t0, t0 + fsync_s, t0
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(FIELDS)
        while True:
            now = clock.monotonic()
            if (duration is not None and now - t0 >= duration) or (stop is not None and stop()):
                break
            if now < t_next:
                clock.sleep(min(t_next - now, period))
                continue
            t_next += period
            if t_next < now - period:
                t_next = now + period
            vals, err = sensors.power()
            row = dict(vals, epoch_us=int(clock.time() * 1e6), mono_us=int(now * 1e6), err=err, temps='')
            if now >= t_temp:
                row['temps'] = sensors.temps()
                t_temp = now + temp_s
            if 'vin_mv' in vals:
                vmin = vals['vin_mv'] if vmin is None else min(vmin, vals['vin_mv'])
            w.writerow([row.get(f, '') for f in FIELDS])
            n += 1
            if now >= t_sync:
                fh.flush()
                os.fsync(fh.fileno())
                t_sync = now + fsync_s
        fh.flush()
        os.fsync(fh.fileno())
    return {'path': path, 'rows': n, 'seconds': round(clock.monotonic() - t0, 2), 'vin_min_mv': vmin}


def main(argv=None, clock=time, sensors=None, printer=None):
    """sensors・clock・printer は試験のため(既定は本物の sysfs・本物の時計・print)"""
    say = printer or (lambda msg: print(msg, flush=True))
    ap = argparse.ArgumentParser(description='Jetson の電源(INA3221)と温度を読むだけの記録係。jetformer-web とは別に動かす')
    ap.add_argument('--hz', type=float, default=10.0, help='1 秒に読む回数(既定 10。1〜50)')
    ap.add_argument('--out', default=DEFAULT_OUT, help='書く場所(既定 ~/jetformer/data/power)')
    ap.add_argument('--duration', type=float, default=None, help='この秒で止める(無ければ止めるまで)')
    ap.add_argument('--fsync-s', type=float, default=1.0, help='この秒ごとに保存(fsync)')
    ap.add_argument('--rotate-s', type=float, default=None, help='この秒ごとに新しいファイルにする(起動のサービスは 1800)')
    a = ap.parse_args(argv)
    if not 1.0 <= a.hz <= 50.0:
        ap.error('--hz は 1〜50')
    if not 0.2 <= a.fsync_s <= 60.0:
        ap.error('--fsync-s は 0.2〜60')
    if a.rotate_s is not None and a.rotate_s < 10.0:
        ap.error('--rotate-s は 10 以上')
    lower_priority(10)
    flag = {'stop': False}

    def _stop(signum, frame):
        flag['stop'] = True
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    stop = lambda: flag['stop']                                                  # noqa: E731
    se = sensors or Sensors()
    if not se.ch:
        say('INA3221 が見つかりません(/sys/class/hwmon)。温度だけ記録します')
    say('記録を始めます: %s(電源の欄 %s・温度 %d か所)' % (a.out, ','.join(sorted(se.ch)), len(se.zones)))
    t_end = None if a.duration is None else clock.monotonic() + a.duration
    warned_disk = False
    while not flag['stop']:
        dur = a.rotate_s
        if t_end is not None:
            left = t_end - clock.monotonic()
            if left <= 0:
                break
            dur = left if dur is None else min(dur, left)
        if free_bytes(a.out) < MIN_FREE:
            if not warned_disk:
                say('ディスクの空きが %.1f GB より少ないので、書かずに待ちます' % (MIN_FREE / 1024 ** 3))
                warned_disk = True
            sleep_while(clock, 60.0, stop)
            continue
        warned_disk = False
        try:
            s = run(se, a.out, hz=a.hz, duration=dur, fsync_s=a.fsync_s, clock=clock, stop=stop)
        except OSError as e:
            if e.errno in WRITE_ERRNOS:
                say('書けません(%s)。60 秒待ってやり直します' % e)
                sleep_while(clock, 60.0, stop)
                continue
            raise
        done = flag['stop'] or a.rotate_s is None or (t_end is not None and clock.monotonic() >= t_end - 0.05)
        say('%s: %s・%d 行・%.1f 秒・入力の電圧の最小 %s mV' % ('止めました' if done else '区切りました', s['path'], s['rows'],
                                                    s['seconds'], s['vin_min_mv']))
        if a.rotate_s is None:
            break
    return 0


if __name__ == '__main__':
    sys.exit(main())
