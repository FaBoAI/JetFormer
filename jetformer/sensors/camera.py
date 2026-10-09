"""前方カメラ(IMX219, V4L2直接取得)"""

from .base import Sample, SensorThread
from .v4l2_camera import V4L2Camera


class CameraSensor(SensorThread):
    name = 'CAM_FRONT'

    def __init__(self, cfg):
        super(CameraSensor, self).__init__()
        self.cfg = cfg
        self.camera = None
        self.dropped = 0

    def _open(self):
        self.camera = V4L2Camera(width=self.cfg['width'], height=self.cfg['height'],
                                 capture_fps=self.cfg['fps'], sensor_mode=self.cfg['sensor_mode'],
                                 capture_device=self.cfg['device'])

    def _poll(self):
        image, t_ns, seq = self.camera.read_stamped(timeout=1.0)
        self.dropped = self.camera.stats['dropped']
        return [Sample(t_ns, {'image': image, 'seq': seq})]

    def _close(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def status(self):
        st = super(CameraSensor, self).status()
        st['dropped'] = self.dropped
        if self.camera is not None:
            st['exposure_us'] = self.camera._exposure
            st['gain'] = self.camera._gain
        return st
