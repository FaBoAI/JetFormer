"""センサー共通部品: 別スレッドで計測し、購読者(収集・プレビュー)へ配信する"""

import collections
import threading
import time
import traceback


class Sample(object):
    __slots__ = ('t_ns', 'data')

    def __init__(self, t_ns, data):
        self.t_ns = t_ns      # CLOCK_MONOTONIC [ns]
        self.data = data      # dict


class SensorThread(object):
    """サブクラスは _open / _poll / _close を実装する。

    _poll() は最大0.2秒程度でブロックし、得られたSampleのリストを返す。
    例外が出たら切断扱いにして、数秒後に自動で開き直す。
    """

    name = 'SENSOR'
    reconnect_interval = 2.0

    def __init__(self):
        self._listeners = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.connected = False
        self.latest = None
        self.count = 0
        self.errors = 0
        self.last_error = ''
        self._times = collections.deque(maxlen=120)

    # ---- サブクラスで実装 ----
    def _open(self):
        raise NotImplementedError

    def _poll(self):
        raise NotImplementedError

    def _close(self):
        pass

    # ---- 共通 ----
    def subscribe(self, callback):
        with self._lock:
            self._listeners.append(callback)

    def unsubscribe(self, callback):
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()

    def stop(self, timeout=3.0):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)

    @property
    def running(self):
        return bool(self._thread and self._thread.is_alive())

    def rate_hz(self):
        times = list(self._times)
        if len(times) < 2 or time.monotonic() - times[-1] > 1.0:
            return 0.0
        return (len(times) - 1) / max(times[-1] - times[0], 1e-6)

    def status(self):
        return {
            'name': self.name,
            'running': self.running,
            'connected': self.connected,
            'rate_hz': round(self.rate_hz(), 1),
            'count': self.count,
            'errors': self.errors,
            'last_error': self.last_error,
        }

    def _publish(self, sample):
        self.latest = sample
        self.count += 1
        self._times.append(time.monotonic())
        with self._lock:
            listeners = list(self._listeners)
        for cb in listeners:
            try:
                cb(self.name, sample)
            except Exception:
                traceback.print_exc()

    def _run(self):
        while not self._stop.is_set():
            if not self.connected:
                try:
                    self._open()
                    self.connected = True
                    self.last_error = ''
                except Exception as e:
                    self.errors += 1
                    self.last_error = '%s: %s' % (type(e).__name__, e)
                    self._safe_close()
                    self._stop.wait(self.reconnect_interval)
                    continue
            try:
                for sample in self._poll():
                    self._publish(sample)
            except Exception as e:
                self.errors += 1
                self.last_error = '%s: %s' % (type(e).__name__, e)
                self.connected = False
                self._safe_close()
                self._stop.wait(0.5)
        self._safe_close()
        self.connected = False

    def _safe_close(self):
        try:
            self._close()
        except Exception:
            pass
