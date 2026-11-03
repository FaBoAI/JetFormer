#!/usr/bin/env python3
"""AS5601(サーボの回転角度の磁気センサー・I2C)を読むだけの記録係(走りのプログラム jetformer-web とは別に動かす)

2026-10-29 ユーザーの決め(fix_20261026/user_decisions.md 10/29 23:3x):
  - 明日(10/30)の走行から AS5601 をデータに入れる。取り込み方は「別の記録係」= jetformer-web は変えず、この台本が
    AS5601 だけを読み、別のファイルに書く
  - I2C は「AS5601 の読み取りだけ」許可(アドレス 0x36 を読むだけ。PCA9685 0x40・表示板には何もしない)。
    試すのは走行していないときだけ。つないだ所は 40 ピンの 3・5 番(I2C の 7 番 = PCA9685 と同じ線)
2026-10-30 ユーザーの決め(10/30 00:5x): 起動で自動に始める(systemd の jetformer-as5601。tools/systemd/)。
決まり(この台本で守ること):
  - 読むだけ。AS5601 の設定(ZPOS・CONF・BURN など)にも書かない。0x36 のほかのアドレスは読まない(--addr で変えられない)
  - 走りに響かないように: 1 回の読みは 2 バイト(RAW ANGLE 0x0C・0x0D)だけ。STATUS・AGC・MAGNITUDE は 1 秒に 1 回。
    既定は 1 秒に 50 回。nice は 10(unit の Nice=10 と足さない)。I2C の失敗は数えて続ける(角度の失敗が 5 回続いたら
    0.2 秒休む)。例外で止まらない(書けないときは 60 秒待つ・空きが 2 GB 未満なら書かずに待つ)
  - 電源が急に切れても多くを失わないように: fsync_s ごとに flush と fsync(10/29 の夜は電源断で最後の数十秒の記録が
    0 バイトになった)。fsync の間は読みが止まるので、まれに最大 0.2 秒ほど行が抜ける(分析では補間する)
時計(10/30 反証 S2): epoch_us は time.time()(壁の時計)。走りの記録の AI_CMD の epoch_us は単調な時計 + 収集を始めた
ときのずれなので、ぴったりは同じでない(NTP で時計が飛ぶとずれる)。走りの記録と合わせるときは mono_us(CLOCK_MONOTONIC =
drive の時計と同じ)と events.jsonl の cam_t_us・cam_epoch_us を使う。json に boot_id と rt_minus_mono_us(始めの
壁の時計 − 単調な時計)を残す(同じ boot_id の中だけ mono_us で並べられる)。
出力: <out>/<YYYYMMDD-HHMMSS>.csv と同じ名前の .json(始めの情報)。CSV(行末は \\n)の欄:
  epoch_us(UNIX µs)・mono_us(起動からの単調な時計 µs)・raw(0〜4095。読めなければ -1)・deg(raw × 360 / 4096)・
  status(STATUS の値。読まなかった行は空・読めなかった行は E)・md・ml・mh(磁石あり・弱すぎ・強すぎ)・agc・mag・
  err(角度の読みの続いた失敗の数)

    cd ~/jetformer
    python3 -I tools/as5601_logger.py                  # 止めるまで記録(Ctrl-C で止める)。data/as5601/ に書く
    python3 -I tools/as5601_logger.py --show           # 記録しながら 0.5 秒ごとに角度を出す(左右いっぱい・まっすぐの確かめ)
    python3 -I tools/as5601_logger.py --duration 10    # 10 秒で止める
"""
import argparse
import csv
import errno
import json
import os
import shutil
import signal
import sys
import time
from datetime import datetime

ADDR = 0x36                    # AS5601 の I2C アドレス(決まっている。変えられない)
REG_STATUS, REG_RAW, REG_AGC, REG_MAG = 0x0B, 0x0C, 0x1A, 0x1B
FIELDS = ['epoch_us', 'mono_us', 'raw', 'deg', 'status', 'md', 'ml', 'mh', 'agc', 'mag', 'err']
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'as5601')
MIN_FREE = 2 * 1024 ** 3       # 空きがこれより少なければ書かずに待つ(録画と同じディスク)
WRITE_ERRNOS = (errno.ENOSPC, errno.EROFS, errno.EIO, errno.EDQUOT)


def raw12(b):
    """2 バイト(上位・下位)→ 12 ビットの値"""
    return ((b[0] & 0x0F) << 8) | b[1]


def status_bits(st):
    """STATUS → (md, ml, mh)(磁石あり・弱すぎ・強すぎ)"""
    return (st >> 5) & 1, (st >> 4) & 1, (st >> 3) & 1


def boot_id():
    try:
        with open('/proc/sys/kernel/random/boot_id') as fh:
            return fh.read().strip()
    except OSError:
        return None


def free_bytes(path):
    """path(無ければ、ある親)のディスクの空き"""
    p = os.path.abspath(path)
    while not os.path.exists(p) and os.path.dirname(p) != p:
        p = os.path.dirname(p)
    return shutil.disk_usage(p).free


def lower_priority(target=10):
    """nice を target にする(すでに target 以上なら何もしない。unit の Nice=10 と足して 19 にしない。10/30 反証 S5)"""
    try:
        cur = os.nice(0)
        if cur < target:
            os.nice(target - cur)
    except OSError:
        pass


def sleep_while(clock, seconds, stop):
    """seconds だけ待つ(stop() が True になったらすぐ戻る)"""
    t_end = clock.monotonic() + seconds
    while clock.monotonic() < t_end and not stop():
        clock.sleep(0.1)


class Reader(object):
    """読むだけ。bus は smbus2.SMBus と同じ read_byte_data・read_i2c_block_data を持つもの(試験では偽物)"""

    def __init__(self, bus):
        self.bus = bus

    def angle(self):
        return raw12(self.bus.read_i2c_block_data(ADDR, REG_RAW, 2))

    def health(self):
        st = self.bus.read_byte_data(ADDR, REG_STATUS)
        agc = self.bus.read_byte_data(ADDR, REG_AGC)
        mag = raw12(self.bus.read_i2c_block_data(ADDR, REG_MAG, 2))
        return st, agc, mag


def run(reader, out_dir, hz=50.0, duration=None, fsync_s=2.0, health_s=1.0, show=False, clock=time, stop=None,
        printer=print):
    """記録の本体。止まるのは duration に届いたか stop() が True のとき。終わりに数をまとめて返す"""
    os.makedirs(out_dir, exist_ok=True)
    name = datetime.fromtimestamp(clock.time()).strftime('%Y%m%d-%H%M%S')
    k = 2
    base = name
    while os.path.exists(os.path.join(out_dir, name + '.csv')):          # 同じ秒に区切ったときに上書きしない
        name = '%s-%d' % (base, k)
        k += 1
    path = os.path.join(out_dir, name + '.csv')
    meta = {'name': name, 'start_epoch_us': int(clock.time() * 1e6), 'addr': '0x%02x' % ADDR, 'hz': hz,
            'fsync_s': fsync_s, 'health_s': health_s, 'fields': FIELDS, 'deg_per_count': 360.0 / 4096,
            'boot_id': boot_id(), 'rt_minus_mono_us': int((clock.time() - clock.monotonic()) * 1e6),
            'note': 'AS5601 を読むだけ(tools/as5601_logger.py)。走りの記録と合わせるときは mono_us と events.jsonl の '
                    'cam_t_us・cam_epoch_us(同じ boot_id の中だけ)'}
    with open(os.path.join(out_dir, name + '.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    period = 1.0 / hz
    n = n_err = n_herr = 0
    err_run = 0
    lo = hi = None
    last_ok = None
    t0 = clock.monotonic()
    t_next = t0
    t_sync = t0 + fsync_s
    t_health = t0
    t_show = t0
    t_warn = -1e9
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
            if t_next < now - period:                       # 遅れたら追いつこうとせず、今から数え直す
                t_next = now + period
            row = {'epoch_us': int(clock.time() * 1e6), 'mono_us': int(now * 1e6), 'raw': -1, 'deg': '', 'status': '',
                   'md': '', 'ml': '', 'mh': '', 'agc': '', 'mag': '', 'err': 0}
            try:
                r = reader.angle()
                row.update(raw=r, deg='%.3f' % (r * 360.0 / 4096))
                err_run = 0
                last_ok = r
                lo = r if lo is None else min(lo, r)
                hi = r if hi is None else max(hi, r)
            except OSError as e:
                err_run += 1
                n_err += 1
                row['err'] = err_run
                if now - t_warn >= 5.0:
                    printer('AS5601 の読みが失敗(%s)。続けます(失敗 %d 回)' % (e, n_err))
                    t_warn = now
                if err_run >= 5:
                    clock.sleep(0.2)
            if now >= t_health:                             # 10/30 反証 S7: STATUS などの失敗は角度の失敗と分ける
                t_health = now + health_s
                try:
                    st, agc, mag = reader.health()
                    md, ml, mh = status_bits(st)
                    row.update(status=st, md=md, ml=ml, mh=mh, agc=agc, mag=mag)
                except OSError:
                    n_herr += 1
                    row['status'] = 'E'
            w.writerow([row[k] for k in FIELDS])
            n += 1
            if now >= t_sync:
                fh.flush()
                os.fsync(fh.fileno())
                t_sync = now + fsync_s
            if show and now >= t_show:
                printer('角度 %s 度(raw %s)・失敗 %d' % ('-' if last_ok is None else '%.1f' % (last_ok * 360.0 / 4096),
                                                     '-' if last_ok is None else last_ok, n_err))
                t_show = now + 0.5
        fh.flush()
        os.fsync(fh.fileno())
    summary = {'path': path, 'samples': n, 'errors': n_err, 'health_errors': n_herr,
               'seconds': round(clock.monotonic() - t0, 2), 'raw_min': lo, 'raw_max': hi}
    return summary


def wait_ready(reader, wait, stop, clock=time, printer=print, every_s=10.0):
    """AS5601 が応答するまで待つ(wait = False なら待たない)。応答したら (st, agc, mag)、待たずに失敗・止められたら None。
    2026-10-30: 起動のサービス(jetformer-as5601)は、車の電池をつなぐ前など AS5601 が応答しない間も終わらずに待つ"""
    warned = False
    while True:
        try:
            return reader.health()
        except OSError as e:
            if not wait:
                printer('AS5601 が応答しません(%s)。つなぎ方・電源を見てください' % e)
                return None
            if not warned:
                printer('AS5601 が応答しません(%s)。%d 秒ごとに待ちます' % (e, every_s))
                warned = True
            sleep_while(clock, every_s, stop)
            if stop():
                return None


def main(argv=None, clock=time, bus_factory=None, printer=None):
    """bus_factory・clock・printer は試験のため(既定は smbus2.SMBus・本物の時計・print)"""
    say = printer or (lambda msg: print(msg, flush=True))
    ap = argparse.ArgumentParser(description='AS5601(0x36)を読むだけの記録係。jetformer-web とは別に動かす')
    ap.add_argument('--bus', type=int, default=7, help='I2C の番号(40 ピンの 3・5 番 = 7)')
    ap.add_argument('--hz', type=float, default=50.0, help='1 秒に読む回数(既定 50。1〜200)')
    ap.add_argument('--out', default=DEFAULT_OUT, help='書く場所(既定 ~/jetformer/data/as5601)')
    ap.add_argument('--duration', type=float, default=None, help='この秒で止める(無ければ Ctrl-C まで)')
    ap.add_argument('--show', action='store_true', help='0.5 秒ごとに角度を出す')
    ap.add_argument('--fsync-s', type=float, default=2.0, help='この秒ごとに保存(fsync)。起動のサービスは 1')
    ap.add_argument('--rotate-s', type=float, default=None, help='この秒ごとに新しいファイルにする(起動のサービスは 1800)')
    ap.add_argument('--wait', action='store_true', help='AS5601 が応答しない間も終わらずに待つ(起動のサービス用)')
    a = ap.parse_args(argv)
    if not 1.0 <= a.hz <= 200.0:
        ap.error('--hz は 1〜200')
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
    if bus_factory is None:
        from smbus2 import SMBus
        bus_factory = SMBus
    with bus_factory(a.bus) as bus:
        rd = Reader(bus)
        h = wait_ready(rd, a.wait, stop, clock=clock, printer=say)
        if h is None:
            return 0 if flag['stop'] else 2
        st, agc, mag = h
        md, ml, mh = status_bits(st)
        try:
            say('AS5601: 磁石 %s・弱すぎ %s・強すぎ %s・AGC %d・強さ %d・角度 %.1f 度' % (
                'あり' if md else 'なし', 'はい' if ml else 'いいえ', 'はい' if mh else 'いいえ', agc, mag,
                rd.angle() * 360.0 / 4096))
        except OSError:
            pass
        if not md:
            say('注意: 磁石が見えていません(取り付けを見てください)。記録は続けます')
        elif ml or mh:
            say('注意: 磁石が%sです(読みが揺れやすい)。記録は続けます' % ('弱すぎ' if ml else '強すぎ'))
        say('記録を始めます: %s(Ctrl-C で止める)' % a.out)
        t_end = None if a.duration is None else clock.monotonic() + a.duration
        warned_disk = False
        while not flag['stop']:
            dur = a.rotate_s
            if t_end is not None:
                left = t_end - clock.monotonic()
                if left <= 0:
                    break
                dur = left if dur is None else min(dur, left)
            if free_bytes(a.out) < MIN_FREE:                                    # 10/30 反証 S9: 録画と同じディスクを守る
                if not warned_disk:
                    say('ディスクの空きが %.1f GB より少ないので、書かずに待ちます' % (MIN_FREE / 1024 ** 3))
                    warned_disk = True
                sleep_while(clock, 60.0, stop)
                continue
            warned_disk = False
            try:
                s = run(rd, a.out, hz=a.hz, duration=dur, fsync_s=a.fsync_s, show=a.show, clock=clock, stop=stop,
                        printer=say)
            except OSError as e:
                if e.errno in WRITE_ERRNOS:                                     # 10/30 反証 S4: 再起動を繰り返さない
                    say('書けません(%s)。60 秒待ってやり直します' % e)
                    sleep_while(clock, 60.0, stop)
                    continue
                raise
            done = flag['stop'] or a.rotate_s is None or (t_end is not None and clock.monotonic() >= t_end - 0.05)
            say('%s: %s・%d 行・失敗 %d・STATUS の失敗 %d・%.1f 秒・raw %s〜%s' % (
                '止めました' if done else '区切りました', s['path'], s['samples'], s['errors'], s['health_errors'],
                s['seconds'], s['raw_min'], s['raw_max']))
            if a.rotate_s is None:
                break
    return 0


if __name__ == '__main__':
    sys.exit(main())
