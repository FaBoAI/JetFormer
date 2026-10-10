"""前方カメラ(IMX219, V4L2直接取得)

まれに、開いた直後からセンサーが約21fps(フレーム長が伸びたまま)で動くことがある(2026-10-10、サービス起動直後に
2回発生。開き直すと30fps)。30fpsの周期は走行に必須なので、遅いままなら設定をかけ直し、それでもだめなら開き直す。
"""

import time

from . import v4l2_camera
from .base import Sample, SensorThread
from .v4l2_camera import V4L2Camera

SLOW_RATIO = 0.85          # 期待値の85%未満を「遅い」とみなす
SLOW_SECONDS = 3.0


class CameraSensor(SensorThread):
    name = 'CAM_FRONT'

    def __init__(self, cfg):
        super(CameraSensor, self).__init__()
        self.cfg = cfg
        self.camera = None
        self.dropped = 0
        self.rate_fixes = 0         # 遅いままだったので設定をかけ直した・開き直した回数

    def _open(self):
        self.camera = V4L2Camera(width=self.cfg['width'], height=self.cfg['height'],
                                 capture_fps=self.cfg['fps'], sensor_mode=self.cfg['sensor_mode'],
                                 capture_device=self.cfg['device'])
        self._opened = time.monotonic()
        self._slow_since = None
        self._reapplied = False

    def _poll(self):
        image, t_ns, seq = self.camera.read_stamped(timeout=1.0)
        self.dropped = self.camera.stats['dropped']
        self._check_rate()
        return [Sample(t_ns, {'image': image, 'seq': seq})]

    def _check_rate(self):
        now = time.monotonic()
        if now - self._opened < 2.0:
            return
        if self.rate_hz() >= self.cfg['fps'] * SLOW_RATIO:
            self._slow_since = None
            return
        if self._slow_since is None:
            self._slow_since = now
        elif now - self._slow_since > SLOW_SECONDS:
            self.rate_fixes += 1
            if not self._reapplied:            # まずフレームレートの設定をかけ直す
                self.camera._set_controls([(v4l2_camera._CID_FRAME_RATE, self.cfg['fps'] * 1000000)])
                self._reapplied, self._slow_since = True, None
            else:                              # それでも遅ければ開き直す(SensorThreadが再接続する)
                raise RuntimeError('フレームレートが %.1f fps のままなので開き直します' % self.rate_hz())

    def _close(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def status(self):
        st = super(CameraSensor, self).status()
        st['dropped'] = self.dropped
        st['rate_fixes'] = self.rate_fixes
        if self.camera is not None:
            st['exposure_us'] = self.camera._exposure
            st['gain'] = self.camera._gain
        return st
