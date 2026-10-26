#!/usr/bin/env python3
"""tools/fw_vbat_check.py の試験(段 plan、2026-10-26)。偽の管理画面(関数と、127.0.0.1 の偽の HTTP)だけ。装置・本物の画面に触れない。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I test_fw_vbat_check.py -v
"""
import http.server
import importlib.util
import json
import re
import threading
import unittest
from pathlib import Path

TOOL = Path('/home/jetson/jetformer/tools/fw_vbat_check.py')
spec = importlib.util.spec_from_file_location('fw_vbat_check', TOOL)
fw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fw)


class Fake(object):
    """偽の /api/status と /api/live/values。vals = 回ごとの電圧(None = 無し)"""

    def __init__(self, vals, supported=True, erpm=0, moving=False, running=True, connected=True, broken=None,
                 errs_step=0, resets_step=0, driving=False, age=0.05, scale=1.0):
        self.vals, self.i = list(vals), 0
        self.kw = dict(supported=supported, erpm=erpm, moving=moving, running=running, connected=connected, broken=broken,
                       errs_step=errs_step, resets_step=resets_step, driving=driving, age=age, scale=scale)
        self.n_status = 0
        self.paths = []

    def __call__(self, base, path, timeout=2.0):
        self.paths.append(path)
        k = self.kw
        if path == '/api/status':
            self.n_status += 1
            n = self.n_status
            return {'sensors_running': k['running'], 'driving': k['driving'], 'training': False,
                    'sensors': {'VEHICLE': {'connected': k['connected'], 'count': 100 * n, 'read_errors': k['errs_step'] * n,
                                            'bus_errors': 0, 'board_resets': k['resets_step'] * n,
                                            'vbat_supported': k['supported'], 'vbat_ok': k['supported'] and self._v() is not None,
                                            'vbat_v': self._v(), 'vbat_bad_v': k['broken']}}}
        if path == '/api/live/values':
            v = self._v()
            self.i += 1
            return {'imu': {}, 'vehicle': {},
                    'vehicle_now': {'pwm': [1441, 1496, 1980], 'erpm': k['erpm'], 'valid': [True, True, True], 'select': False,
                                    'moving': k['moving'], 'quality': 'snapshot'},
                    'vbat': {'supported': k['supported'], 'v': v, 'age_s': k['age'] if v is not None else None,
                             'level': 'ok', 'broken_v': k['broken'], 'range_v': [6.0, 9.0], 'scale': k['scale']}}
        raise AssertionError('知らない道: %s' % path)

    def _v(self):
        return self.vals[min(self.i, len(self.vals) - 1)] if self.vals else None


def run(fake, meter=None, n=20):
    data = fw.collect('http://fake', seconds=n / 4.0, hz=4.0, get=fake, sleep=lambda s: None)
    return fw.judge(data, meter)


def verdicts(res):
    return {c['id']: c['verdict'] for c in res['checks']}


class T(unittest.TestCase):
    def test_old_firmware(self):
        r = run(Fake([None] * 20, supported=False))
        self.assertEqual(r['code'], 2)
        self.assertEqual(verdicts(r)['F1'], '不合格')
        self.assertIn('古いファーム', fw.text(r))

    def test_good_firmware(self):
        vals = [8.31, 8.30, 8.32, 8.31, 8.30] * 4
        r = run(Fake(vals), meter=8.33)
        v = verdicts(r)
        self.assertEqual(r['code'], 0, fw.text(r))
        self.assertEqual([v[k] for k in ('F1', 'R', 'N', 'B', 'F2', 'F3')], ['合格'] * 6)
        self.assertAlmostEqual(r['summary']['still_median_v'], 8.31, places=2)
        self.assertEqual(r['summary']['flags']['ch_recv'], [20, 20, 20])

    def test_broken_value(self):
        r = run(Fake([None] * 20, supported=True, broken=12.75))
        self.assertEqual(r['code'], 2)
        self.assertEqual(verdicts(r)['R'], '不合格')

    def test_noisy(self):
        vals = [8.30, 8.45, 8.15, 8.30] * 5
        r = run(Fake(vals))
        self.assertEqual(verdicts(r)['N'], '注意')
        self.assertEqual(r['code'], 1)

    def test_sensors_stopped_returns_fast(self):
        f = Fake([8.3] * 20, running=False)
        r = run(f)
        self.assertEqual(r['code'], 3)
        self.assertEqual(f.paths, ['/api/status'])          # 待たずに返す・live/values も読まない

    def test_meter_mismatch(self):
        r = run(Fake([8.31] * 20), meter=8.10)
        self.assertEqual(verdicts(r)['F2'], '注意')
        self.assertEqual(r['code'], 1)

    def test_board_errors_increase(self):
        r = run(Fake([8.31] * 20, errs_step=1, resets_step=1))
        self.assertEqual(verdicts(r)['F3'], '注意')

    def test_moving_car_not_enough_still(self):
        r = run(Fake([8.20] * 20, erpm=3000, moving=True))
        self.assertEqual(verdicts(r)['N'], '足りない')
        self.assertEqual(r['code'], 3)

    def test_gaps_in_value(self):
        vals = [8.3, None] * 10
        r = run(Fake(vals))
        self.assertEqual(verdicts(r)['F1'], '不合格')

    def test_low_battery(self):
        r = run(Fake([7.60] * 20))
        self.assertEqual(verdicts(r)['B'], '注意')

    def test_only_get_and_no_devices(self):
        src = TOOL.read_text(encoding='utf-8')
        code = re.sub(r'""".*?"""', '', src, flags=re.S)        # 説明の文字は外す
        for bad in ('smbus', 'serial', '/dev/', 'POST', 'method=', 'data=', 'import jetformer', 'from jetformer',
                    'cv2', 'subprocess'):
            self.assertNotIn(bad, code, bad)
        self.assertNotIn('/api/sensors/start', code)
        self.assertNotIn('/api/vbat/calib', code)

    def test_real_http_fake_server(self):
        fake = Fake([8.25] * 40)

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(fake(None, self.path)).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = http.server.HTTPServer(('127.0.0.1', 0), H)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            code = fw.main(['--url', 'http://127.0.0.1:%d' % srv.server_address[1], '--seconds', '1', '--hz', '8'])
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(code, 0)
        self.assertTrue(all(p in ('/api/status', '/api/live/values') for p in fake.paths))

    def test_unreachable(self):
        self.assertEqual(fw.main(['--url', 'http://127.0.0.1:9', '--seconds', '1']), 3)


if __name__ == '__main__':
    unittest.main()
