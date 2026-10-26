#!/usr/bin/env python3
"""ファームの電圧の確かめ(2026-10-26、fix_20261026 段 plan): 管理画面の GET を数秒読むだけで、
基板のファームが電池の電圧を送ってきているか・値が 6〜9 V か・止まっているときの揺れ・基板の読みの失敗を出す。

    python3 tools/fw_vbat_check.py                    # 5 秒(4 回/秒)読む
    python3 tools/fw_vbat_check.py --meter 8.31       # LiPo チェッカーの値と比べる(F2。較正の前は ±0.10 V)
    python3 tools/fw_vbat_check.py --seconds 10 --json out.json
    python3 tools/fw_vbat_check.py --url http://127.0.0.1:8080

読むのは GET /api/status(センサー・基板の読みの数)と GET /api/live/values(電圧 vbat・vehicle_now の flags)だけ。
**装置(I2C・カメラ・ToF)は開かない。センサーの開始・走行・較正・保存の API は呼ばない。** 標準ライブラリだけ。
センサーが止まっているときは、管理画面のライブのタブで開始してから(この道具は開始しない)。走っている最中は回さない。

確かめ(fix_20261026/fix_plan.md 課題 5・speed_control_20261025/firmware_vbat_spec.md 6.3 節):
  F1  電圧が来ている     vbat.supported true・読んだ回の 90% 以上で値があり、古さ 1 秒未満
                         (来ていない = 古いファーム: flags bit5 = 0。ファームの前ならこれで正しい)
  範囲 値が 6.0〜9.0 V    broken_v(6〜9 V の外の値)が一度も無い・値が全部 6.0〜9.0 V
  揺れ 止まっているとき   |eRPM| < 50 かつ moving でない回が 8 回以上。中央値から ±0.10 V の中(仕様 6.1 の 5)
  電池 7.70 V 未満は注意(パックを替える)・7.70〜7.90 V は黄
  F2  電圧計と合う       --meter のときだけ。|中央値 − 電圧計| ≤ 0.10 V(外れたら画面の「電池の電圧の較正」。書く操作なのでユーザー)
  F3  基板の読み         読んだ間に read_errors・bus_errors・board_resets が増えない(前のファームと比べるのは人)
  flags                  CH1〜3 の受信(bit0-2)・SELECT(bit3, AI 側)・回転中(bit4)・電圧あり(bit5 の代わりに vbat_ok)の回数
終わりの番号(car_check と同じ): 0 合格・1 注意・2 不合格・3 読めない / 足りない
"""
import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = 'http://127.0.0.1:8080'
VBAT_RANGE_V = (6.0, 9.0)        # jetformer/sensors/vehicle.py の VBAT_RANGE_V と同じ
WARN_V, BAD_V = 7.90, 7.70       # 黄・赤(web/server.py の VBAT_WARN_V・VBAT_BAD_V と同じ)
NOISE_V = 0.10                   # 止まっているときの揺れ(中央値から ±0.10 V。firmware_vbat_spec.md 6.1 の 5)
METER_V = 0.10                   # 電圧計との差(較正の前。6.3 節 J1)
STILL_ERPM = 50.0                # 止まっている(VbatTracker.STILL_ERPM と同じ)
MAX_AGE_S = 1.0                  # 値の古さ
MIN_FRAC = 0.9                   # 値があった回の割合
MIN_STILL = 8                    # 揺れを見るのに要る「止まっている回」
ORDER = {'合格': 0, '注意': 1, '不合格': 2, '足りない': 3}


def http_get(base, path, timeout=2.0):
    """GET して JSON。読めなければ例外(urllib.error.URLError・ValueError)"""
    with urllib.request.urlopen(base.rstrip('/') + path, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def vehicle_status(st):
    return ((st or {}).get('sensors') or {}).get('VEHICLE') or {}


def collect(base, seconds=5.0, hz=4.0, get=http_get, sleep=time.sleep, clock=time.monotonic):
    """GET /api/status(初め・終わり)と GET /api/live/values・/api/status(毎回)を読む → dict"""
    out = {'url': base, 'seconds': seconds, 'hz': hz, 'samples': [], 'errors': []}
    out['status0'] = get(base, '/api/status')
    out['t0'] = clock()
    v0 = vehicle_status(out['status0'])
    if not out['status0'].get('sensors_running') or not v0.get('connected'):
        out['t1'], out['status1'] = out['t0'], out['status0']       # センサーが止まっている: 待たずに返す(開始はしない)
        return out
    n = max(1, int(round(seconds * hz)))
    for i in range(n):
        t = clock()
        try:
            lv = get(base, '/api/live/values')
            st = get(base, '/api/status')
        except (urllib.error.URLError, OSError, ValueError) as e:
            out['errors'].append('%s: %s' % (type(e).__name__, e))
            lv, st = None, None
        if lv is not None:
            v = vehicle_status(st)
            out['samples'].append({'t': round(t - out['t0'], 3), 'vbat': lv.get('vbat'), 'vehicle_now': lv.get('vehicle_now'),
                                   'vbat_ok': v.get('vbat_ok'), 'vbat_v_raw': v.get('vbat_v'), 'vbat_bad_v': v.get('vbat_bad_v'),
                                   'driving': (st or {}).get('driving')})
        if i < n - 1:
            sleep(max(0.0, 1.0 / hz - (clock() - t)))
    out['t1'] = clock()
    out['status1'] = get(base, '/api/status')
    return out


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def judge(data, meter=None):
    """collect の結果 → {'checks': [...], 'code': n, 'summary': {...}}"""
    checks = []

    def add(cid, name, verdict, detail):
        checks.append({'id': cid, 'name': name, 'verdict': verdict, 'detail': detail})

    st0, st1 = data.get('status0') or {}, data.get('status1') or {}
    v0, v1 = vehicle_status(st0), vehicle_status(st1)
    S = data.get('samples') or []
    summ = {'n_samples': len(S), 'read_errors_http': len(data.get('errors') or [])}
    if not st0.get('sensors_running') or not v0.get('connected'):
        add('S1', 'センサーと基板', '足りない',
            'センサーが止まっている(sensors_running %s)か、基板とつながっていない(VEHICLE connected %s)。'
            '管理画面のライブのタブでセンサーを開始してから、もう一度(この道具は開始しない)'
            % (st0.get('sensors_running'), v0.get('connected')))
        return {'checks': checks, 'code': 3, 'summary': summ}
    if len(S) < 3:
        add('S0', '読めた回', '足りない', '読めたのは %d 回(%s)' % (len(S), '・'.join(data.get('errors') or [])[:200]))
        return {'checks': checks, 'code': 3, 'summary': summ}
    if any(s.get('driving') for s in S) or st0.get('driving'):
        add('S2', '走行中', '注意', '走行中に読んだ。止まっているときの揺れは止まっていた回だけで見る(走っている最中は回さない決まり)')

    vb = [s.get('vbat') or {} for s in S]
    vals = [_num(x.get('v')) for x in vb]
    ages = [_num(x.get('age_s')) for x in vb]
    fresh = [v for v, a in zip(vals, ages) if v is not None and a is not None and a < MAX_AGE_S]
    supported = any(x.get('supported') for x in vb) or bool(v0.get('vbat_supported'))
    broken = [_num(x.get('broken_v')) for x in vb if x.get('broken_v') is not None] + \
             [_num(s.get('vbat_bad_v')) for s in S if s.get('vbat_bad_v') is not None]
    summ.update(supported=supported, n_with_value=len(fresh), frac_with_value=round(len(fresh) / len(S), 3),
                scale=(vb[-1].get('scale') if vb else None), broken_v=broken[:5])

    # F1 来ているか
    if not supported and not broken:
        add('F1', '電圧が来ている', '不合格', '来ていない(画面は「電圧: 基板が対応していません」)。古いファーム = flags bit5 が 0。'
            'ファームの前ならこれで正しい。新しいファームを書いたあとなら、書き込み・配線(分圧)を見る')
    elif len(fresh) / len(S) >= MIN_FRAC:
        add('F1', '電圧が来ている', '合格', '%d / %d 回で値あり(古さ %.1f 秒未満)' % (len(fresh), len(S), MAX_AGE_S))
    else:
        add('F1', '電圧が来ている', '不合格', '値があったのは %d / %d 回(%.0f%% 未満)。途切れる・古い(age_s %s)'
            % (len(fresh), len(S), MIN_FRAC * 100, [a for a in ages if a is not None][-3:]))

    # 範囲 6〜9 V
    lo, hi = VBAT_RANGE_V
    out_of = [v for v in fresh if not lo <= v <= hi]
    if broken or out_of:
        add('R', '値が 6.0〜9.0 V', '不合格', '範囲の外の値 %s V(画面は「電圧の値がおかしい」。分圧の外れ・ADC の張り付き・3S の電池)'
            % sorted(set(round(x, 2) for x in (broken + out_of) if x is not None))[:5])
    elif fresh:
        add('R', '値が 6.0〜9.0 V', '合格', '%.2f〜%.2f V' % (min(fresh), max(fresh)))

    # 止まっているときの揺れ
    still = []
    for s, v, a in zip(S, vals, ages):
        vn = s.get('vehicle_now') or {}
        e = _num(vn.get('erpm'))
        if v is not None and a is not None and a < MAX_AGE_S and e is not None and abs(e) < STILL_ERPM and not vn.get('moving'):
            still.append(v)
    if fresh:
        if len(still) >= MIN_STILL:
            med = statistics.median(still)
            dev = max(abs(v - med) for v in still)
            sd = statistics.pstdev(still)
            summ.update(still_n=len(still), still_median_v=round(med, 3), still_min_v=round(min(still), 3),
                        still_max_v=round(max(still), 3), still_p2p_v=round(max(still) - min(still), 3),
                        still_sd_v=round(sd, 4), still_maxdev_v=round(dev, 3))
            add('N', '止まっているときの揺れ', '合格' if dev <= NOISE_V + 1e-9 else '注意',
                '%d 回: 中央 %.2f V・最小 %.2f・最大 %.2f・中央から最大 ±%.3f V(門 ±%.2f)・sd %.3f'
                % (len(still), med, min(still), max(still), dev, NOISE_V, sd))
        else:
            add('N', '止まっているときの揺れ', '足りない', '止まっていた回が %d(%d 回要る)。車を止めて(車輪を回さずに)もう一度'
                % (len(still), MIN_STILL))
    ref = summ.get('still_median_v') or (statistics.median(fresh) if fresh else None)

    # 電池の水準
    if ref is not None:
        if ref < BAD_V:
            add('B', '電池', '注意', '%.2f V < %.2f V: パックを替える' % (ref, BAD_V))
        elif ref < WARN_V:
            add('B', '電池', '注意', '%.2f V: 黄(%.2f〜%.2f V)' % (ref, BAD_V, WARN_V))
        else:
            add('B', '電池', '合格', '%.2f V' % ref)

    # F2 電圧計
    if meter is not None:
        if ref is None:
            add('F2', '電圧計と合う', '足りない', '基板の値が無い')
        else:
            d = ref - float(meter)
            add('F2', '電圧計と合う', '合格' if abs(d) <= METER_V + 1e-9 else '注意',
                '基板 %.2f V − 電圧計 %.2f V = %+.2f V(門 ±%.2f。較正の係数 %s)。外れたら画面の「電池の電圧の較正」(ユーザー)'
                % (ref, float(meter), d, METER_V, summ.get('scale')))

    # F3 基板の読み
    el = max(1e-6, (data.get('t1') or 0) - (data.get('t0') or 0))
    dl = {k: (int(v1.get(k) or 0) - int(v0.get(k) or 0)) for k in ('read_errors', 'bus_errors', 'board_resets', 'count')}
    summ.update(delta=dl, rate_hz=round(dl['count'] / el, 1), read_errors_total=v1.get('read_errors'),
                bus_errors_total=v1.get('bus_errors'), board_resets_total=v1.get('board_resets'))
    bad = {k: dl[k] for k in ('read_errors', 'bus_errors', 'board_resets') if dl[k] > 0}
    add('F3', '基板の読み', '注意' if bad else '合格',
        ('%.1f 秒で増えた %s' % (el, bad) if bad else '%.1f 秒で増えない' % el)
        + '(通算 read_errors %s・bus_errors %s・board_resets %s。読む速さ %.0f 回/秒。前のファームと比べるのは人)'
        % (v1.get('read_errors'), v1.get('bus_errors'), v1.get('board_resets'), summ['rate_hz']))

    # flags(読むだけ。判定に入れない)
    fl = {'ch_recv': [0, 0, 0], 'select': 0, 'moving': 0, 'vbat_ok': 0, 'n': 0}
    for s in S:
        vn = s.get('vehicle_now')
        if not vn:
            continue
        fl['n'] += 1
        for k, x in enumerate((vn.get('valid') or [])[:3]):
            fl['ch_recv'][k] += 1 if x else 0
        fl['select'] += 1 if vn.get('select') else 0
        fl['moving'] += 1 if vn.get('moving') else 0
        fl['vbat_ok'] += 1 if s.get('vbat_ok') else 0
    summ['flags'] = fl
    code = max([ORDER[c['verdict']] for c in checks] or [3])
    if any(c['verdict'] == '不合格' for c in checks):
        code = 2
    return {'checks': checks, 'code': code, 'summary': summ}


VERDICT_TEXT = {0: '合格', 1: '注意あり', 2: '不合格あり', 3: '読めない・足りない'}


def text(res):
    L = ['ファームの電圧の確かめ: %s(終わりの番号 %d)' % (VERDICT_TEXT[res['code']], res['code'])]
    for c in res['checks']:
        L.append('  [%s] %s %s: %s' % (c['verdict'], c['id'], c['name'], c['detail']))
    s = res['summary']
    fl = s.get('flags')
    if fl and fl['n']:
        L.append('  flags(%d 回): CH1〜3 受信 %s・SELECT(AI 側)%d・回転中 %d・電圧あり(vbat_ok)%d'
                 % (fl['n'], fl['ch_recv'], fl['select'], fl['moving'], fl['vbat_ok']))
    return '\n'.join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description='ファームの電圧の確かめ(GET を読むだけ。装置は開かない)')
    ap.add_argument('--url', default=DEFAULT_URL)
    ap.add_argument('--seconds', type=float, default=5.0)
    ap.add_argument('--hz', type=float, default=4.0)
    ap.add_argument('--meter', type=float, default=None, help='LiPo チェッカーの値 [V](F2)')
    ap.add_argument('--json', default=None, help='結果の JSON の書き先(既定は書かない)')
    a = ap.parse_args(argv)
    if not 1.0 <= a.seconds <= 60.0 or not 0.5 <= a.hz <= 10.0:
        ap.error('--seconds は 1〜60、--hz は 0.5〜10')
    try:
        data = collect(a.url, a.seconds, a.hz)
    except (urllib.error.URLError, OSError, ValueError) as e:
        print('管理画面に届きません(%s): %s。jetformer-web が動いているか見る(この道具は起動しない)' % (a.url, e))
        return 3
    res = judge(data, a.meter)
    print(text(res))
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump({'result': res, 'raw': data, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z')}, f, ensure_ascii=False, indent=1)
    return res['code']


if __name__ == '__main__':
    sys.exit(main())
