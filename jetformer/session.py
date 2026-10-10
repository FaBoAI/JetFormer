"""走行ごとに区切るデータセット収集

    「データセット収集を開始します」
    「1回目の走行をスタート、3、2、1、ピ」   ← ピが鳴った瞬間から記録(1回の走行 = 1シーン)
      … 走行(既定20秒。「この走行を終える」で早く終えられる)…
    「ピ、1回目の走行終了」
      (指定秒数停止。車を戻す時間)
    「2回目の走行をスタート、3、2、1、ピ」… を指定回数くり返す(回数1なら1回だけ)
    「データセット収集を終了します。N回の走行を保存しました」

カウントダウンは1つの音として鳴らし、再生が始まった時刻から「ピ」の時刻を計算して記録を始める
(案内がキューで待たされても、ピと記録の開始がずれない)。音を消していても、同じ時間で数えて始める。
"""

import math
import threading
import time
import traceback

from . import modes, sound

LIMITS = {'runs': (1, 200), 'run_seconds': (3, 300), 'pause_s': (0, 600)}
SILENT_COUNTDOWN_S = 3.0      # 音を消しているときのカウントダウンの長さ
END_ADVANCE_S = 0.03          # 終わりの音を作ってから鳴り始めるまでの見込み
END_GRACE_S = 1.5             # 終わりのピーがこれ以上遅れたら、鳴るのを待たずに記録を止める
CLIP_START_TIMEOUT_S = 15.0   # カウントダウンの音がこれだけ待っても鳴り始めなければ、音なしで数える
LABELS = {'runs': '回数', 'run_seconds': '1回の走行の秒数', 'pause_s': '走行の間の停止秒数'}


class CollectSession(object):
    def __init__(self, recorder, player):
        self.recorder = recorder
        self.player = player
        self.thread = None
        self._stop = threading.Event()
        self._skip = threading.Event()        # 停止中の待ちを飛ばして次の走行へ
        self._end_run = threading.Event()     # 今の走行を終える
        self._mode_id = None                  # 準備中に変えられたジャンル(収集が始まったら反映する)
        self.st = {'active': False, 'phase': 'idle'}

    @property
    def active(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, runs=1, run_seconds=20, pause_s=10, **record_kwargs):
        if self.active or self.recorder.recording:
            raise RuntimeError('すでに収集中です')
        vals = {}
        for k, v in (('runs', runs), ('run_seconds', run_seconds), ('pause_s', pause_s)):
            lo, hi = LIMITS[k]
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise ValueError('%sが数値ではありません' % LABELS[k])
            if not lo <= v <= hi:
                raise ValueError('%sは %g〜%g で指定してください' % (LABELS[k], lo, hi))
            if k == 'runs' and v != int(v):
                raise ValueError('回数は整数で指定してください')
            vals[k] = v
        self._stop.clear()
        self._skip.clear()
        self._end_run.clear()
        self._mode_id = None
        self.st = {'active': True, 'phase': 'starting', 'message': 'センサーの準備中…', 'runs': int(vals['runs']),
                   'run_seconds': vals['run_seconds'], 'pause_s': vals['pause_s'], 'run': 0, 'done': 0,
                   'countdown_s': None, 'run_elapsed_s': None, 'pause_left_s': None, 'log': None, 'error': None}
        self.thread = threading.Thread(target=self._run, args=(int(vals['runs']), vals['run_seconds'], vals['pause_s'],
                                                               record_kwargs), name='collect-session', daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()
        self.player.cancel_clips()          # まだ鳴っていないカウントダウンを鳴らさない

    def set_mode(self, mode_id):
        """走行ジャンルを変える(準備中なら、収集が始まった時点で反映する)"""
        self._mode_id = modes.validate(int(mode_id))
        if self.recorder.recording:
            return self.recorder.set_mode(self._mode_id)
        return self._mode_id

    def next_run(self):
        """停止中の待ちを飛ばして、すぐ次の走行のカウントダウンへ"""
        if self.st.get('phase') != 'pause':
            raise RuntimeError('次の走行を待っていません')
        self._skip.set()

    def end_run(self):
        """今の走行を(時間前でも)終える"""
        if self.st.get('phase') != 'running':
            raise RuntimeError('走行中ではありません')
        self._end_run.set()

    def status(self):
        st = dict(self.st)
        st['active'] = self.active
        return st

    # ------------------------------------------------------------ 進行
    def _run(self, runs, run_seconds, pause_s, record_kwargs):
        st = self.st
        log = None
        try:
            if self._mode_id is not None:
                record_kwargs['mode_id'] = self._mode_id
            log = self.recorder.start(runs_mode=True, **record_kwargs)      # 「データセット収集を開始します」
            st['log'] = log
            if self._stop.is_set():                                       # 準備中に止められた
                return
            if self._mode_id is not None and self._mode_id != record_kwargs.get('mode_id'):
                self.recorder.set_mode(self._mode_id)                     # センサーの準備中に変えられた
            for n in range(1, runs + 1):
                if n > 1 and not self._pause(pause_s):
                    break
                st['run'] = n
                if not self._countdown(n):
                    break
                self.recorder.begin_run(run_seconds)
                complete = self._running(n, run_seconds)      # 「ピー、n回目の走行終了」で記録を終える
                if complete:
                    st['done'] += 1
                if not complete:
                    break
        except Exception as e:
            traceback.print_exc()
            st['error'] = '%s: %s' % (type(e).__name__, e)
        finally:
            st['phase'] = 'finishing'
            st['message'] = '保存中…'
            if log:
                try:
                    self.recorder.stop(only_log=log)         # 「データセット収集を終了します」
                except Exception:
                    traceback.print_exc()
            st['phase'] = 'idle'
            st['message'] = '%d回の走行を保存しました' % st['done'] if not st.get('error') else '止まりました(エラー)'
            st['countdown_s'] = st['run_elapsed_s'] = st['pause_left_s'] = None

    def _pause(self, pause_s):
        st = self.st
        st['phase'] = 'pause'
        st['message'] = '次の走行まで停止中'
        self._skip.clear()
        end = time.monotonic() + pause_s
        while time.monotonic() < end:
            st['pause_left_s'] = round(end - time.monotonic(), 1)
            if self._stop.is_set():
                return False
            if self._skip.wait(0.05):
                break
        st['pause_left_s'] = None
        return not self._stop.is_set()

    def _countdown(self, n):
        """「n回目の走行をスタート、3、2、1、ピ」を鳴らし、ピの時刻まで待つ。止められたら False"""
        st = self.st
        st['phase'] = 'countdown'
        st['message'] = '%d回目の走行をスタート' % n
        st['countdown_s'] = None
        if self._stop.is_set():
            return False
        started = threading.Event()
        t0 = [None]

        def on_start(t):
            t0[0] = t
            started.set()
        target = None
        if self.player.wants('run'):
            try:
                mono, beep_at = sound.countdown_clip(self.player.tts, n)
            except Exception as e:                       # 音声が作れなくても収集は続ける(音なしで数える)
                traceback.print_exc()
                st['sound_error'] = '%s: %s' % (type(e).__name__, e)
                mono = None
            if mono is not None and self.player.play_clip('run', mono, on_start):
                deadline = time.monotonic() + CLIP_START_TIMEOUT_S
                while not started.wait(0.05):            # 前の案内が鳴り終わって、この音が始まるまで
                    if self._stop.is_set():
                        return False
                    if time.monotonic() > deadline:      # 鳴らない(キューが詰まった等): 取り消して音なしで数える
                        self.player.cancel_clips()
                        break
                if started.is_set():
                    target = t0[0] + sound.LEAD_S + beep_at
        if target is None:
            target = time.monotonic() + SILENT_COUNTDOWN_S   # 音を消しているとき: 同じ時間だけ数えてから
        while True:
            left = target - time.monotonic()
            if left <= 0:
                break
            st['countdown_s'] = min(3, math.ceil(left)) if left <= 3.0 else None
            if self._stop.wait(min(left, 0.02)):
                return False
        st['countdown_s'] = 0
        return True

    def _running(self, n, run_seconds):
        """走行中。終わりの「ピー」が鳴る瞬間に記録を止める(開始のピーからちょうど run_seconds)。
        時間いっぱい・「この走行を終える」で True、停止で False"""
        st = self.st
        st['phase'] = 'running'
        st['message'] = '%d回目の走行中' % n
        self._end_run.clear()
        t0 = time.monotonic()
        clip = None
        if self.player.wants('run'):
            try:
                clip = sound.end_clip(self.player.tts, n)  # 先に作っておく(鳴らす瞬間に遅れないように)
            except Exception as e:
                traceback.print_exc()
                st['sound_error'] = '%s: %s' % (type(e).__name__, e)
        queued = False
        beep = [None]                                       # 終わりのピーが鳴る時刻

        def on_start(t):
            beep[0] = t + sound.LEAD_S

        def finish(complete):
            self.recorder.end_run(complete=complete)
            st['run_elapsed_s'] = None
            if not queued and clip is not None:             # 途中で終えた: 今からピーを鳴らす
                self.player.play_clip('run', clip)
            return complete

        while True:
            now = time.monotonic()
            el = now - t0
            st['run_elapsed_s'] = round(min(el, run_seconds), 1)
            if self._end_run.is_set():
                return finish(True)
            if beep[0] is None and el >= run_seconds - sound.LEAD_S - END_ADVANCE_S:
                # ピーがちょうど run_seconds で鳴るよう、先頭の無音の分だけ早く再生を始める
                queued = clip is not None and self.player.play_clip('run', clip, on_start)
                if not queued:
                    beep[0] = t0 + run_seconds              # 音を消しているとき
            if beep[0] is not None and now >= beep[0]:
                return finish(True)
            if el >= run_seconds + END_GRACE_S:             # 前の案内が長引いてピーが遅れても、ここで止める
                return finish(True)
            if self._stop.wait(0.01):
                return finish(False)
