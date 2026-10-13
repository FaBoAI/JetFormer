"""シミュレーターのブリッジ(子プロセス)の管理

- ボタンを押したときだけ起動する(JetFormer の起動時には起動しない)。全体で 1 つだけ。
- venv の Python で bridge_runner.py を -I -B で動かす。127.0.0.1 の空きポート(OS が選ぶ)で待ち受け、
  stdout の {"event":"listening","port":...} の行からポートを知る。GET /api/status が 200 を返したら running。
- 止めるときは SIGINT → (5 秒)SIGTERM → (2 秒)SIGKILL。
- 自分で止めていないのに終了したら、10 分間に 3 回まで backoff(2・5・15 秒)を置いて起動し直す。超えたら failed。
- 出力はリングバッファ(300 行)と data/sim/logs/bridge.log(5 MB × 3)に残す。収集中や AI 走行中に毎フレーム出る
  POST /api/predict -> 200 と POST /api/datasets/frame -> 200 の行は書かずに数だけ数える。
"""

import collections
import json
import logging
import logging.handlers
import os
import signal
import subprocess
import threading
import time
import urllib.request

STATES = ('stopped', 'starting', 'running', 'stopping', 'failed')
SUPPRESS = {'[bridge] POST /api/predict -> 200': 'predict', '[bridge] POST /api/datasets/frame -> 200': 'frame'}


class BridgeError(RuntimeError):
    """ブリッジを起動・停止できない(日本語の理由)。status は HTTP の目安"""

    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


def mem_available():
    """MemAvailable(バイト)。読めなければ None"""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def health_check(port, timeout=1.0):
    """GET http://127.0.0.1:<port>/api/status が 200 か"""
    try:
        req = urllib.request.Request('http://127.0.0.1:%d/api/status' % port, headers={'Host': '127.0.0.1:%d' % port})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def rss_mb(pid):
    try:
        with open('/proc/%d/status' % pid) as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError, TypeError):
        pass
    return None


def child_env():
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'LC_ALL', 'HOME') if k in os.environ}
    env.setdefault('PATH', '/usr/bin:/bin')
    env.update(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='2')
    return env


class BridgeProcess(object):
    # nice は 0(優先度を下げない)。シミュレーターは推論の応答を 5 秒で打ち切る(app.js の fetchJSON)ので、
    # 裏で強化学習などが CPU を使い切っていると、下げたブリッジは「The operation timed out.」で評価が止まる(2026-10-13)。
    # 重い学習のほうを nice で下げる。
    def __init__(self, *, python=None, runner=None, sim_dir=None, runtime_root=None, log_path=None, ort_threads=4, nice=0,
                 port=0, restart_limit=3, restart_window_s=600, backoff_s=(2, 5, 15), health_timeout_s=15,
                 listen_timeout_s=20, popen=subprocess.Popen, clock=time.monotonic, sleep=time.sleep,
                 min_mem_bytes=1 << 30, mem_available=mem_available, health_check=health_check, preflight=None,
                 health_interval_s=0.5, oom_score_adj=500):
        self.cfg = {'python': python, 'runner': runner, 'sim_dir': sim_dir, 'runtime_root': runtime_root,
                    'log_path': log_path, 'port': port}
        self.ort_threads, self.nice, self.oom_score_adj = ort_threads, nice, int(oom_score_adj)
        self.restart_limit, self.restart_window_s, self.backoff_s = restart_limit, restart_window_s, tuple(backoff_s)
        self.health_timeout_s, self.listen_timeout_s, self.health_interval_s = health_timeout_s, listen_timeout_s, health_interval_s
        self._popen, self._clock, self._sleep = popen, clock, sleep
        self.min_mem_bytes, self._mem_available, self._health_check = min_mem_bytes, mem_available, health_check
        self.preflight = preflight
        self.lock = threading.RLock()
        self.state = 'stopped'
        self.proc = None
        self.gen = 0
        self._port = None
        self._listen_port = None
        self.started_at = None
        self.restarted_at = None
        self.restart_times = collections.deque()
        self.restarts_total = 0
        self.last_exit = None
        self.last_error = None
        self.user_stopped = False
        self.ring = collections.deque(maxlen=300)
        self.suppressed = {'predict': 0, 'frame': 0}
        self.last_activity = None
        self._logger = None
        self._logger_path = None

    # ------------------------------------------------------------ 外から使うもの
    @property
    def port(self):
        """running のときだけポート番号"""
        with self.lock:
            return self._port if self.state == 'running' else None

    def running(self):
        with self.lock:
            return self.state in ('starting', 'running', 'stopping')

    def status(self):
        """通信しない。{state, pid, port, started_at, restarts, restarted_at, last_exit, last_error, rss_mb}"""
        with self.lock:
            pid = self.proc.pid if self.proc is not None and self.proc.poll() is None else None
            self._prune()
            return {'state': self.state, 'pid': pid, 'port': self._port if self.state == 'running' else None,
                    'started_at': self.started_at, 'restarts': len(self.restart_times), 'restarts_total': self.restarts_total,
                    'restarted_at': self.restarted_at, 'last_exit': self.last_exit, 'last_error': self.last_error,
                    'rss_mb': rss_mb(pid) if pid else None, 'last_activity': self.last_activity}

    def logs(self, n=200):
        with self.lock:
            lines = list(self.ring)[-max(0, int(n)):] if n else []
            return {'lines': lines, 'suppressed': dict(self.suppressed)}

    def note_activity(self):
        self.last_activity = time.time()

    def start(self):
        """すぐ返す(state='starting')。起動できなければ BridgeError(日本語)"""
        with self.lock:
            if self.state in ('starting', 'running'):
                return self.status()
            if self.state == 'stopping':
                raise BridgeError('ブリッジを止めている途中です。少し待ってから起動してください')
            cfg = dict(self.cfg)
            if self.preflight:
                cfg.update(self.preflight() or {})
            for k in ('python', 'runner', 'sim_dir', 'runtime_root'):
                if not cfg.get(k):
                    raise BridgeError('ブリッジの設定が足りません: %s' % k)
            avail = self._mem_available()
            if avail is not None and avail < self.min_mem_bytes:
                raise BridgeError('メモリの空きが足りません(%d MB。%d MB 以上必要です)。ほかの重い処理が終わってから起動してください'
                                  % (avail >> 20, self.min_mem_bytes >> 20))
            self.run_cfg = cfg
            self.user_stopped = False
            self.last_error = None
            self.restart_times.clear()
            self._spawn()
            return self.status()

    def stop(self, timeout=5.0):
        with self.lock:
            self.user_stopped = True
            proc = self.proc
            if proc is None or proc.poll() is not None:
                self.state = 'stopped'
                self._port = None
                return self.status()
            self.state = 'stopping'
        self._terminate(proc, timeout)
        with self.lock:
            if self.proc is proc:
                self.state = 'stopped'
                self._port = None
            return self.status()

    # ------------------------------------------------------------ 中身
    def _prune(self):
        now = self._clock()
        while self.restart_times and now - self.restart_times[0] > self.restart_window_s:
            self.restart_times.popleft()

    def _command(self, cfg):
        cmd = [str(cfg['python']), '-I', '-B', str(cfg['runner']), '--sim-dir', str(cfg['sim_dir']),
               '--runtime-root', str(cfg['runtime_root']), '--port', str(int(cfg.get('port') or 0)),
               '--ort-threads', str(self.ort_threads), '--nice', str(self.nice),
               # JetFormer が強制終了しても孤児のブリッジを残さない・メモリが尽きたらブリッジから止められる(review で追加)
               '--parent-pid', str(os.getpid()), '--oom-score-adj', str(self.oom_score_adj)]
        return cmd

    def _spawn(self):
        cfg = self.run_cfg
        os.makedirs(str(cfg['runtime_root']), exist_ok=True)
        self.gen += 1
        gen = self.gen
        self._port = None
        self._listen_port = None
        self.state = 'starting'
        self._log('[jetformer] ブリッジを起動します(%s)' % ('再起動' if self.restarts_total else '起動'))
        try:
            proc = self._popen(self._command(cfg), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               cwd=str(cfg['runtime_root']), env=child_env(), close_fds=True)
        except OSError as e:
            self.state = 'failed'
            self.last_error = 'ブリッジを起動できません: %s' % e
            raise BridgeError(self.last_error, 500)
        self.proc = proc
        self.started_at = time.time()
        threading.Thread(target=self._reader, args=(proc, gen), name='sim-bridge-log', daemon=True).start()
        threading.Thread(target=self._monitor, args=(proc, gen), name='sim-bridge-mon', daemon=True).start()

    def _get_logger(self):
        path = self.run_cfg.get('log_path') if hasattr(self, 'run_cfg') else None
        if not path:
            return None
        if self._logger is None or self._logger_path != str(path):
            try:
                os.makedirs(os.path.dirname(str(path)), exist_ok=True)
                h = logging.handlers.RotatingFileHandler(str(path), maxBytes=5 << 20, backupCount=3, encoding='utf-8')
                h.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
                lg = logging.Logger('jetformer-sim-bridge')
                lg.propagate = False
                lg.addHandler(h)
                if self._logger is not None:
                    for old in self._logger.handlers:
                        old.close()
                self._logger, self._logger_path = lg, str(path)
            except OSError:
                return None
        return self._logger

    def _log(self, line):
        with self.lock:
            self.ring.append(time.strftime('%H:%M:%S ') + line)
        lg = self._get_logger()
        if lg is not None:
            try:
                lg.info(line)
            except Exception:
                pass

    def _reader(self, proc, gen):
        try:
            for raw in proc.stdout:
                line = raw.decode('utf-8', 'replace') if isinstance(raw, bytes) else raw
                line = line.rstrip('\r\n')
                if not line:
                    continue
                key = SUPPRESS.get(line.strip())
                if key:
                    with self.lock:
                        self.suppressed[key] += 1
                    continue
                if line.startswith('{'):
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        ev = None
                    if isinstance(ev, dict) and ev.get('event') == 'listening' and isinstance(ev.get('port'), int):
                        with self.lock:
                            if gen == self.gen:
                                self._listen_port = ev['port']
                        self._log('[jetformer] 127.0.0.1:%d で待ち受けています(pid %s)' % (ev['port'], ev.get('pid')))
                        continue
                    if isinstance(ev, dict) and ev.get('event') == 'stopped':
                        self._log('[jetformer] ブリッジが止まりました')
                        continue
                self._log(line)
        except (OSError, ValueError):
            pass

    def _kill_quietly(self, proc):
        try:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass

    def _terminate(self, proc, timeout):
        for sig, wait in ((signal.SIGINT, timeout), (signal.SIGTERM, 2.0)):
            try:
                if proc.poll() is not None:
                    return
                proc.send_signal(sig)
                proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue
            except OSError:
                return
        self._kill_quietly(proc)

    def _tail_text(self, n=20):
        with self.lock:
            lines = [x for x in list(self.ring)[-n:] if not x.endswith('ブリッジを起動します(起動)')]
        return ' / '.join(lines)[-1500:]

    def _monitor(self, proc, gen):
        t0 = self._clock()
        # 1) listening の行を待つ
        while True:
            with self.lock:
                port = self._listen_port if gen == self.gen else None
                stale = gen != self.gen or self.user_stopped
            if stale or port is not None or proc.poll() is not None or self._clock() - t0 > self.listen_timeout_s:
                break
            self._sleep(0.05)
        if not stale and port is None:
            if proc.poll() is None:
                self._kill_quietly(proc)
                reason = 'ブリッジが %d 秒以内に待ち受けを始めませんでした' % self.listen_timeout_s
            else:
                reason = 'ブリッジが起動の途中で終了しました(終了コード %s)' % proc.returncode
            return self._fail(proc, gen, reason)
        # 2) /api/status が 200 になるのを待つ
        if not stale:
            t1 = self._clock()
            ok = False
            while proc.poll() is None and self._clock() - t1 <= self.health_timeout_s:
                with self.lock:
                    if gen != self.gen or self.user_stopped:
                        break
                if self._health_check(port):
                    ok = True
                    break
                self._sleep(self.health_interval_s)
            with self.lock:
                stale = gen != self.gen or self.user_stopped
                if ok and not stale:
                    self._port = port
                    self.state = 'running'
            if not ok and not stale:
                if proc.poll() is None:
                    self._kill_quietly(proc)
                    return self._fail(proc, gen, 'ブリッジが %d 秒以内に応答しませんでした' % self.health_timeout_s)
                return self._fail(proc, gen, 'ブリッジが起動の途中で終了しました(終了コード %s)' % proc.returncode)
        # 3) 終了を待つ
        try:
            rc = proc.wait()
        except Exception:
            rc = None
        with self.lock:
            if gen != self.gen:
                return
            self.last_exit = {'code': rc, 'at': time.time()}
            if self.user_stopped or self.state == 'stopping':
                self.state = 'stopped'
                self._port = None
                return
            self._port = None
            self._prune()
            if len(self.restart_times) >= self.restart_limit:
                self.state = 'failed'
                self.last_error = 'ブリッジが繰り返し異常終了したので止めました(%d 分間に %d 回): %s' % (
                    self.restart_window_s // 60, self.restart_limit + 1 if self.restart_limit else 1, self._tail_text())
                self._log('[jetformer] ' + self.last_error[:200])
                return
            delay = self.backoff_s[min(len(self.restart_times), len(self.backoff_s) - 1)] if self.backoff_s else 0
            self.restart_times.append(self._clock())
            self.restarts_total += 1
            self.state = 'starting'
            self._log('[jetformer] ブリッジが終了しました(終了コード %s)。%s 秒後に起動し直します' % (rc, delay))
        self._sleep(delay)
        with self.lock:
            if gen != self.gen or self.user_stopped:
                if self.state == 'starting' and self.user_stopped:
                    self.state = 'stopped'
                return
            self.restarted_at = time.time()
            try:
                self._spawn()
            except BridgeError:
                pass

    def _fail(self, proc, gen, reason):
        with self.lock:
            if gen != self.gen or self.user_stopped:
                if self.user_stopped and gen == self.gen:
                    self.state = 'stopped'
                return
            self.state = 'failed'
            self._port = None
            self.last_exit = {'code': proc.poll(), 'at': time.time()}
            self.last_error = '%s。ログ: %s' % (reason, self._tail_text())
        self._log('[jetformer] ' + reason)
