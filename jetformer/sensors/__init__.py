"""センサー群の起動・停止・状態取得"""

from .. import config
from .audio import AudioSensor
from .camera import CameraSensor
from .imu import ImuSensor
from .tof import TofSensor
from .vehicle import VehicleSensor

SENSOR_CHANNELS = ('CAM_FRONT', 'TOF_FRONT', 'IMU', 'VEHICLE', 'AUDIO')
OPTIONAL_CHANNELS = ('VEHICLE', 'AUDIO')      # つながっていなくても収集はできる


class SensorHub(object):
    def __init__(self):
        self.sensors = {
            'CAM_FRONT': CameraSensor(config.CAMERA),
            'TOF_FRONT': TofSensor(config.TOF),
            'IMU': ImuSensor(config.IMU),
            'VEHICLE': VehicleSensor(config.VEHICLE),
        }
        if config.AUDIO.get('enabled'):
            self.sensors['AUDIO'] = AudioSensor(config.AUDIO)
        self.channels = tuple(self.sensors)

    def __getitem__(self, name):
        return self.sensors[name]

    def start(self):
        for s in self.sensors.values():
            s.start()

    def stop(self):
        for s in self.sensors.values():
            s.stop()

    @property
    def running(self):
        return any(s.running for s in self.sensors.values())

    def status(self):
        return {name: s.status() for name, s in self.sensors.items()}
