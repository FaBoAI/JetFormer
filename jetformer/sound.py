"""RCカー側のスピーカーで動作音・音声案内を鳴らす

JetFormer基板のI2Sスピーカーアンプ(40ピンヘッダのI2S2: pin12=BCLK / pin35=LRCLK / pin40=DOUT)へ、
ALSAの ADMAIF2 → I2S2(48kHz・32bit・2ch、Jetsonがクロックを出す)で再生する。
音声は Open JTalk で作り、data/sounds にキャッシュする(同じ文は2回目からすぐ鳴る)。
鳴らす要求はキューに入れて別スレッドで順に再生する(走行ループなどを待たせない)。
"""

import hashlib
import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import traceback
import wave

import numpy as np

from . import config

RATE = 48000
SETTINGS_PATH = config.DATA_DIR / 'sound_settings.json'
CACHE_DIR = config.DATA_DIR / 'sounds'

EVENTS = {      # キー: (画面の表示名, 既定でON)
    'startup': ('起動したとき', True),
    'record': ('収集の開始・終了', True),
    'mode': ('ジャンルの切り替え', True),
    'scene': ('シーン保存の合図(20秒ごと)', False),
    'drive': ('AI走行の開始・終了', True),
    'handover': ('操縦の切り替え(CH3)', True),
    'warning': ('警告(信号断・異常停止)', True),
    'train': ('学習の終了', True),
}
DEFAULTS = {'enabled': True, 'volume': 60, 'events': {k: v[1] for k, v in EVENTS.items()}}


def _tone(freqs, dur, rate=RATE, amp=0.5):
    """短い音(複数の周波数を重ねる)。はじめと終わりを滑らかにしてプチッと鳴らないようにする"""
    t = np.arange(int(rate * dur)) / rate
    x = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    env = np.minimum(1.0, np.minimum(t / 0.01, (dur - t) / 0.04))
    return (amp * x * np.clip(env, 0, 1)).astype(np.float32)


def _gap(dur, rate=RATE):
    return np.zeros(int(rate * dur), np.float32)


CHIMES = {
    'start': lambda: np.concatenate([_tone([660], 0.12), _gap(0.03), _tone([990], 0.18)]),
    'stop': lambda: np.concatenate([_tone([990], 0.12), _gap(0.03), _tone([660], 0.18)]),
    'info': lambda: _tone([880], 0.15),
    'tick': lambda: _tone([1500], 0.06, amp=0.35),
    'warn': lambda: np.concatenate([_tone([1200], 0.1), _gap(0.06)] * 3),
    'ai': lambda: np.concatenate([_tone([523, 784], 0.12), _gap(0.03), _tone([659, 988], 0.2)]),
    'human': lambda: np.concatenate([_tone([659, 988], 0.12), _gap(0.03), _tone([523, 784], 0.2)]),
}


def speakable(text):
    """読み上げ用: 括弧や記号を読ませない、英字は読みやすく"""
    text = re.sub(r'[()()「」『』\[\]【】]', ' ', str(text))
    for a, b in (('AI', 'エーアイ'), ('JetFormer', 'ジェットフォーマー'), ('CH3', 'チャンネル3'), ('ToF', 'ティーオーエフ')):
        text = text.replace(a, b)
    return re.sub(r'\s+', ' ', text).strip()


class TTS(object):
    """Open JTalk で文を音声(float32、48kHz、モノラル)にする。結果はファイルにキャッシュ"""
    DIC = '/var/lib/mecab/dic/open-jtalk/naist-jdic'
    VOICES = [os.path.expanduser('~/.local/share/hts-voice/tohoku-f01-neutral.htsvoice'),
              '/usr/share/hts-voice/nitech-jp-atr503-m001/nitech_jp_atr503_m001.htsvoice']

    def __init__(self, cache_dir=CACHE_DIR, speed=1.1):
        self.cache_dir = cache_dir
        self.speed = speed
        self.voice = next((v for v in self.VOICES if os.path.exists(v)), None)

    @property
    def available(self):
        return self.voice is not None and os.path.exists(self.DIC)

    def speech(self, text):
        text = speakable(text)
        if not text or not self.available:
            return np.zeros(0, np.float32)
        key = hashlib.sha1(('%s|%s|%s' % (self.voice, self.speed, text)).encode('utf-8')).hexdigest()[:16]
        path = self.cache_dir / ('%s.npy' % key)
        if path.exists():
            return np.load(path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as d:
            txt, wav = os.path.join(d, 'in.txt'), os.path.join(d, 'out.wav')
            with open(txt, 'w', encoding='utf-8') as f:
                f.write(text)
            subprocess.run(['open_jtalk', '-x', self.DIC, '-m', self.voice, '-r', str(self.speed), '-ow', wav, txt],
                           check=True, timeout=30, capture_output=True)
            with wave.open(wav) as w:
                rate = w.getframerate()
                x = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float32) / 32768.0
        if rate != RATE:
            x = np.interp(np.arange(int(len(x) * RATE / rate)) * rate / RATE, np.arange(len(x)), x).astype(np.float32)
        peak = float(np.abs(x).max()) if len(x) else 0.0
        if peak > 0:
            x = x / peak * 0.8
        np.save(path, x)
        return x


def write_wav(path, mono, volume):
    """float32 モノラル → 48kHz・32bit・2ch のWAV(I2S2の設定と同じ形式)。volume: 0〜100"""
    gain = (max(0, min(100, volume)) / 100.0) ** 2           # 耳の感じ方に合わせて2乗
    lead = _gap(0.15)                                         # アンプが起きるまでの間(頭が欠けないように)
    x = np.concatenate([lead, mono.astype(np.float32) * gain, _gap(0.1)])
    pcm = (np.clip(x, -1, 1) * (2 ** 31 - 1)).astype('<i4')
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(4)
        w.setframerate(RATE)
        w.writeframes(np.repeat(pcm[:, None], 2, axis=1).tobytes())


class SoundPlayer(object):
    def __init__(self, cfg=None, tts=None, play=None):
        self.cfg = cfg or config.SPEAKER
        self.tts = tts or TTS()
        self._play = play or self._aplay
        saved = {}
        try:
            with open(SETTINGS_PATH, encoding='utf-8') as f:
                saved = json.load(f)
        except (OSError, ValueError):
            pass
        self.settings = {'enabled': saved.get('enabled', DEFAULTS['enabled']),
                         'volume': saved.get('volume', DEFAULTS['volume']),
                         'events': dict(DEFAULTS['events'], **{k: v for k, v in saved.get('events', {}).items() if k in EVENTS})}
        self.q = queue.Queue(maxsize=20)
        self.last_error = ''
        self.played = 0
        self.last_text = ''
        self.busy = False
        self._mixer_ok = False
        threading.Thread(target=self._worker, name='sound', daemon=True).start()

    # ------------------------------------------------------------ 設定
    def update(self, values):
        s = dict(self.settings)
        if 'enabled' in values:
            s['enabled'] = bool(values['enabled'])
        if values.get('volume') is not None:
            s['volume'] = int(max(0, min(100, float(values['volume']))))
        if isinstance(values.get('events'), dict):
            s['events'] = dict(s['events'], **{k: bool(v) for k, v in values['events'].items() if k in EVENTS})
        self.settings = s
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
                json.dump(s, f, ensure_ascii=False, indent=1)
        except OSError:
            pass
        return s

    def status(self):
        return dict(self.settings, labels={k: v[0] for k, v in EVENTS.items()}, tts=self.tts.available,
                    queue=self.q.qsize(), played=self.played, last_error=self.last_error, last_text=self.last_text,
                    device=self.cfg['device'])

    # ------------------------------------------------------------ 鳴らす
    def notify(self, event, text='', chime='info', urgent=False):
        """event の設定がONなら鳴らす(鳴らす文・音はキューへ。すぐ戻る)"""
        if not self.settings['enabled'] or not self.settings['events'].get(event, False):
            return False
        return self.enqueue(text, chime, urgent)

    def enqueue(self, text='', chime='info', urgent=False):
        if urgent:                                    # 警告は、たまっている案内より先に
            try:
                while True:
                    self.q.get_nowait()
            except queue.Empty:
                pass
        try:
            self.q.put_nowait((text, chime))
            return True
        except queue.Full:
            return False

    def prepare(self, texts):
        """よく使う文を先に音声にしておく(初回の遅れをなくす)"""
        def run():
            for t in texts:
                try:
                    self.tts.speech(t)
                except Exception:
                    pass
        threading.Thread(target=run, name='sound-prepare', daemon=True).start()

    def render(self, text, chime):
        parts = []
        if chime:
            parts += [CHIMES[chime](), _gap(0.08)]
        if text:
            parts.append(self.tts.speech(text))
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def _worker(self):
        while True:
            text, chime = self.q.get()
            self.busy = True
            try:
                mono = self.render(text, chime)
                if len(mono):
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                        path = f.name
                    try:
                        write_wav(path, mono, self.settings['volume'])
                        self._play(path)
                    finally:
                        os.unlink(path)
                    self.played += 1
                    self.last_text = text or '(%s)' % chime
                    self.last_error = ''
            except Exception as e:
                traceback.print_exc()
                self.last_error = '%s: %s' % (type(e).__name__, e)
                self._mixer_ok = False
            finally:
                self.busy = False

    def _setup_mixer(self):
        for name, value in self.cfg['mixer']:
            subprocess.run(['amixer', '-q', '-c', self.cfg['card'], 'cset', 'name=%s' % name, value],
                           check=True, timeout=5, capture_output=True)
        self._mixer_ok = True

    def _aplay(self, path):
        if not self._mixer_ok:
            self._setup_mixer()
        r = subprocess.run(['aplay', '-q', '-D', self.cfg['device'], path], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError('aplay: %s' % (r.stderr.strip() or r.returncode))


# ------------------------------------------------------------ 案内の文
def phrase(event, **kw):
    """(文, 音) を返す。event は収集・走行・学習などの出来事"""
    if event == 'record_start':
        return 'データ収集を開始します。%s' % kw.get('mode_name', ''), 'start'
    if event == 'record_stop':
        n = kw.get('scenes', 0)
        return ('データ収集を終了しました。%d シーンを保存しました' % n) if n else 'データ収集を終了しました', 'stop'
    if event == 'mode':
        return 'ジャンル、%s' % kw.get('mode_name', ''), 'info'
    if event == 'scene_saved':
        return '', 'tick'
    if event == 'drive_start':
        return ('推論の確認を開始します' if kw.get('dry_run') else 'AI走行を開始します'), 'start'
    if event == 'drive_stop':
        reason = kw.get('reason', '')
        if kw.get('error'):
            return 'エラーのため、AI走行を止めました', 'warn'
        if '途絶' in reason:
            return '画面との通信が途絶えたため、AI走行を止めました', 'warn'
        return 'AI走行を終了しました', 'stop'
    if event == 'handover':
        return ('AIが操縦します', 'ai') if kw.get('select') else ('人の操縦に戻りました', 'human')
    if event == 'rc_lost':
        return 'プロポの信号が届きません。停止します', 'warn'
    if event == 'train_done':
        return {'done': '学習が完了しました', 'stopped': '学習を停止しました'}.get(kw.get('phase'), '学習でエラーが出ました'), \
            ('info' if kw.get('phase') in ('done', 'stopped') else 'warn')
    if event == 'startup':
        return 'JetFormer を起動しました', 'info'
    return '', 'info'


EVENT_GROUP = {'record_start': 'record', 'record_stop': 'record', 'mode': 'mode', 'scene_saved': 'scene',
               'drive_start': 'drive', 'drive_stop': 'drive', 'handover': 'handover', 'rc_lost': 'warning',
               'train_done': 'train', 'startup': 'startup'}
URGENT = ('rc_lost', 'handover')

COMMON_TEXTS = ['データ収集を終了しました', 'AIが操縦します', '人の操縦に戻りました', 'AI走行を開始します',
                'AI走行を終了しました', '推論の確認を開始します', 'プロポの信号が届きません。停止します',
                '画面との通信が途絶えたため、AI走行を止めました', 'エラーのため、AI走行を止めました',
                '学習が完了しました', 'JetFormer を起動しました']


def announce(player, event, **kw):
    """出来事 → 案内を鳴らす(設定でOFFなら鳴らさない)"""
    text, chime = phrase(event, **kw)
    return player.notify(EVENT_GROUP.get(event), text, chime, urgent=event in URGENT)


def wait_idle(player, timeout=30.0):
    """テスト用: キューが空になって再生が終わるまで待つ"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if player.q.empty() and not player.busy:
            time.sleep(0.05)
            if player.q.empty() and not player.busy:
                return True
        time.sleep(0.05)
    return False
