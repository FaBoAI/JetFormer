#!/usr/bin/env python3
"""AI走行ダイジェスト動画: 走行カメラ+センサー群(左)と、横でキャラクターが解説(右)→ mp4

    .venv/bin/python make_digest.py work/storyboard.json out.mp4 [--chars chars/final]
    .venv/bin/python make_digest.py work/storyboard.json - --stills 5,20.5,61   # 確認用の静止画だけ(work/stills/)

storyboard.json: {"title": ..., "scenes": [{"id", "kind": title|legend|clip|summary|ending, "caption",
                  "src_start", "src_end", "speed", "highlights": [...], "summary_items": [...],
                  "lines": [{"text", "say", "expression", "pose"}]}]}
"""

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import tof_format  # noqa: E402
from jetformer.laps import LapCounter  # noqa: E402

M_PER_S_PER_ERPM = 0.5e-3    # 65mmタイヤ(TT-02)。ToFの接近速度と旋回の加速度から推定(±6%)

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = '/home/jetson/jetformer/data/raw'
W, H, FPS = 1920, 1080, 30
MAIN_W = 1440
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_B = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
VOICE = os.path.expanduser('~/.local/share/hts-voice/tohoku-f01-neutral.htsvoice')
DIC = '/var/lib/mecab/dic/open-jtalk/naist-jdic'
try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()      # JetPackのffmpegには libx264/aac が無い
except ImportError:
    FFMPEG = 'ffmpeg'

BG = (9, 13, 26)
PANEL = (17, 24, 39)
EDGE = (40, 50, 74)
TEXT = (235, 238, 245)
SUB = (148, 163, 184)
DIM = (90, 102, 128)
PURPLE = (168, 85, 247)
CYAN = (34, 211, 238)
RED = (239, 68, 68)
ORANGE = (245, 158, 11)
GREEN = (34, 197, 94)
WHITE = (255, 255, 255)

LEAD_S, GAP_S, TAIL_S = 0.35, 0.28, 0.5          # ナレーションの前・行間・後の間
MIN_SPEED = 0.5

# レイアウト(左のメイン 1440px)
CAM = (24, 96, 648)                               # x, y, 一辺
TOF = (696, 96, 432, 336)
GMETER = (1148, 96, 268, 336)
STEER = (696, 452, 240, 292)
THROT = (952, 452, 216, 292)
SPEED = (1184, 452, 232, 292)
TL = (24, 764, 1392, 296)
SIDE_X = MAIN_W


_fonts = {}


def font(size, bold=False):
    k = (size, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(FONT_B if bold else FONT, size, index=0)
    return _fonts[k]


import re  # noqa: E402

# 英単語・数字(単位つき)はひとまとまりにして、途中で改行しない
_TOKEN = re.compile(r'[A-Za-z0-9][A-Za-z0-9.,:%\-/_]*(?:秒|回|倍|G|ms|fps|µs|m|%)?|\n|.', re.S)
_NO_HEAD = set('。、)」』,.!?！？ー・%')


def wrap(draw, text, f, width):
    lines, cur = [], ''
    for tok in _TOKEN.findall(text):
        if tok == '\n':
            lines.append(cur)
            cur = ''
            continue
        if cur and draw.textlength(cur + tok, font=f) > width and tok[0] not in _NO_HEAD:
            if draw.textlength(tok, font=f) > width:          # 1語が長すぎるときだけ文字で切る
                for ch in tok:
                    if cur and draw.textlength(cur + ch, font=f) > width:
                        lines.append(cur)
                        cur = ''
                    cur += ch
                continue
            lines.append(cur.rstrip())
            cur = tok.lstrip()
        else:
            cur += tok
    if cur:
        lines.append(cur)
    return lines


def fit(draw, text, width, sizes, max_lines, bold=True):
    """枠の幅に収まる (フォント, 行のリスト) を大きい順に探す。収まらなければ最小の大きさで末尾を…にする"""
    for size in sizes:
        f = font(size, bold)
        lines = wrap(draw, text, f, width)
        if len(lines) <= max_lines:
            return f, lines
    lines = lines[:max_lines]
    while lines[-1] and draw.textlength(lines[-1] + '…', font=f) > width:
        lines[-1] = lines[-1][:-1]
    lines[-1] += '…'
    return f, lines


def rrect(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)


# ------------------------------------------------------------------ 走行データ
class Drive(object):
    def __init__(self, log):
        root = os.path.join(RAW, log)
        self.log = log
        self.meta = json.load(open(os.path.join(root, 'log.json'), encoding='utf-8'))
        scenes = sorted(s for s in os.listdir(root) if s.startswith(log + '-'))

        def load(name):
            return [dict(np.load(os.path.join(root, s, name + '.npz'), allow_pickle=True)) for s in scenes]

        def cat(parts, key):
            return np.concatenate([p[key] for p in parts])
        ai, veh, imu, tof, cam = load('AI_CMD'), load('VEHICLE'), load('IMU'), load('TOF_FRONT'), load('CAM_FRONT')
        self.t0 = int(ai[0]['timestamps_us'][0])
        sec = lambda a: (a - self.t0) / 1e6                                     # noqa: E731
        self.ai_t = sec(cat(ai, 'timestamps_us'))
        self.out, self.raw, self.pred = cat(ai, 'out'), cat(ai, 'raw'), cat(ai, 'pred')
        self.hold, self.sel = cat(ai, 'hold').astype(bool), cat(ai, 'select').astype(bool)
        self.assist = cat(ai, 'assist').astype(int) if all('assist' in p for p in ai) else np.zeros(len(self.sel), int)
        has_guard = all('ttc' in p for p in ai)
        self.ttc = cat(ai, 'ttc').astype(float) if has_guard else np.full(len(self.sel), np.nan)
        self.guard = cat(ai, 'guard').astype(int) if has_guard else np.zeros(len(self.sel), int)
        self.has_guard = has_guard
        self.lat, self.inf = cat(ai, 'latency_ms'), cat(ai, 'infer_ms')
        self.v_t, self.pwm, self.erpm = sec(cat(veh, 'timestamps_us')), cat(veh, 'pwm'), cat(veh, 'erpm')
        self.i_t, self.acc, self.gyr = sec(cat(imu, 'timestamps_us')), cat(imu, 'accel'), cat(imu, 'gyro')
        self.tof_t, self.depth = sec(cat(tof, 'timestamps_us')), cat(tof, 'depth')
        self.cam_t = sec(cat(cam, 'timestamps_us'))
        self.cam_f = [os.path.join(root, s, str(f)) for s, c in zip(scenes, cam) for f in c['files']]
        dr = self.meta.get('drive') or {}
        p = dr.get('params') or {}
        self.center = float(dr.get('steer_center_us', 1500))
        self.neutral = float(dr.get('throttle_neutral_us', 1500))
        self.steer_gain, self.thr_gain = float(p.get('steer_gain', 1)), float(p.get('throttle_gain', 1))
        self.trim, self.lookahead = float(p.get('steer_trim_us', 0)), int(p.get('lookahead', 1))
        self.run = dr.get('run', '')
        self.duration = float(self.ai_t[-1])
        g = np.abs(np.linalg.norm(self.acc, axis=1) - 9.80665) / 9.80665
        self.gmag = g
        self.impacts = []                                   # [(t, g, 誤読の疑い)]
        for i in np.nonzero(g > 2)[0]:
            if not self.impacts or self.i_t[i] - self.impacts[-1][0] > 0.5:
                # 本当の衝撃は数サンプルかけて立ち上がる。前後が普通で1サンプルだけ跳ねたものは誤読の疑い
                near = (np.abs(self.i_t - self.i_t[i]) <= 0.05) & (g > 1.0)
                self.impacts.append([float(self.i_t[i]), float(g[i]), bool(near.sum() <= 1)])
            else:
                self.impacts[-1][1] = max(self.impacts[-1][1], float(g[i]))
        self.ai_on = [float(self.ai_t[i + 1]) for i in np.nonzero(~self.sel[:-1] & self.sel[1:])[0]]
        self.ai_off = [float(self.ai_t[i + 1]) for i in np.nonzero(self.sel[:-1] & ~self.sel[1:])[0]]
        self.segments = self._segs(self.sel)
        self.holds = self._segs(self.hold & self.sel)
        self.g_bias = self.acc[:100].mean(axis=0)            # 走り出す前(静止)の値
        self._img = (None, None)
        self.description = self.meta.get('description', '') or log
        self.recoveries = self._segs(self.assist == 2)
        # コントローラ基板の再起動(基板の時計が巻き戻った時刻)
        bt = cat(veh, 'board_t_us').astype(np.int64)
        self.board_resets = [float(self.v_t[i + 1]) for i in np.nonzero((bt[1:] + 1_000_000 < bt[:-1]) & (bt[:-1] < 4_000_000_000))[0]]
        # 周回(ジャイロで360度ごと)
        lc = LapCounter()
        vsel = cat(veh, 'select').astype(bool)
        hg = np.hypot(self.acc[:, 0], self.acc[:, 1]) / 9.80665
        for ti, gz, h in zip(self.i_t, self.gyr[:, 2], hg):
            k = self.idx(self.v_t, ti)
            lc.add(float(ti), float(gz), float(self.erpm[k]), bool(vsel[k]), float(h))
        self.laps = lc.laps
        self.lap_start0 = lc.lap_start_t if not lc.laps else None

    def _segs(self, m):
        out, s = [], None
        for i, v in enumerate(m):
            if v and s is None:
                s = i
            if s is not None and (not v or i == len(m) - 1):
                out.append((float(self.ai_t[s]), float(self.ai_t[i if v else i - 1])))
                s = None
        return out

    @staticmethod
    def idx(ts, t):
        return int(np.clip(np.searchsorted(ts, t, side='right') - 1, 0, len(ts) - 1))

    def frame(self, t):
        i = self.idx(self.cam_t, t)
        if self._img[0] != i:
            self._img = (i, cv2.imread(self.cam_f[i]))
        return self._img[1]

    def state(self, t):
        a = self.idx(self.ai_t, t)
        v = self.idx(self.v_t, t)
        k = self.idx(self.i_t, t)
        sel, hold = bool(self.sel[a]), bool(self.hold[a] and self.sel[a])
        ok = bool(np.isfinite(self.out[a]).all())
        ai_steer = float(self.out[a, 0]) if ok else self.center              # 中立で止めたフレームは NaN のことがある
        ai_thr = float(self.out[a, 1]) if ok else self.neutral
        pred = self.pred[a] if np.isfinite(self.pred[a]).all() else None
        if sel:
            steer = self.center if hold else ai_steer
            thr = self.neutral if hold else ai_thr
        else:
            steer, thr = float(self.pwm[v, 0]), float(self.pwm[v, 1])
        lo = max(0, k - 30)
        imu_age = t - float(self.i_t[k])
        done = [x for x in self.laps if x['t_end'] <= t]
        clean = [x['time_s'] for x in done if x.get('clean', x['ai_only'] and not x['first'] and not x['recoveries'])]
        veh_age = t - float(self.v_t[v])
        return {
            'imu_ok': imu_age < 0.15, 'veh_ok': veh_age < 0.2,
            't': t, 'sel': sel, 'hold': hold, 'steer': steer, 'thr': thr,
            'ai_steer': ai_steer, 'ai_thr': ai_thr, 'pred': pred, 'erpm': int(self.erpm[v]),
            'acc': (self.acc[k] - self.g_bias) / 9.80665, 'trail': (self.acc[lo:k + 1] - self.g_bias) / 9.80665,
            'gmag': float(self.gmag[max(0, k - 5):k + 1].max()), 'yaw': float(self.gyr[k, 2]),
            'lat': float(self.lat[a]) if np.isfinite(self.lat[a]) else None,
            'inf': float(self.inf[a]) if np.isfinite(self.inf[a]) else None,
            'depth': self.depth[self.idx(self.tof_t, t)],
            'assist': int(self.assist[a]) if sel else 0,
            'ttc': float(self.ttc[a]) if self.has_guard and np.isfinite(self.ttc[a]) else None,
            'guard': int(self.guard[a]) if self.has_guard else 0,
            'board_reset': any(-0.1 <= r - t <= 2.6 or 0 <= t - r <= 0.5 for r in self.board_resets),
            'laps_done': len(done), 'last_lap': done[-1] if done else None,
            'best_lap': min(clean) if clean else None,
            'lap_elapsed': (t - done[-1]['t_end']) if done else None,
        }

    def steer_frac(self, us):
        """+1 = 左いっぱい、-1 = 右いっぱい(ジャイロで確認: 中心より大きい = 左)"""
        return float(np.clip((us - self.center) / 500.0, -1, 1))

    def fwd_frac(self, us):
        """+1 = 前進いっぱい、負 = ブレーキ・後退"""
        return float(np.clip((self.neutral - us) / 400.0, -1, 1))


# ------------------------------------------------------------------ キャラクター
class Character(object):
    def __init__(self, cdir, scale=1.0):
        import cairosvg
        self.dir = cdir
        self.m = json.load(open(os.path.join(cdir, 'manifest.json'), encoding='utf-8'))
        cw, ch = self.m.get('canvas', [480, 720])
        self.size = (int(cw * scale), int(ch * scale))
        self._svg = cairosvg
        self._layers = {}
        self._cache = {}

    def layer(self, name):
        if name not in self._layers:
            png = self._svg.svg2png(url=os.path.join(self.dir, name), output_width=self.size[0] * 2,
                                    output_height=self.size[1] * 2)
            im = Image.open(io.BytesIO(png)).convert('RGBA').resize(self.size, Image.LANCZOS)
            self._layers[name] = im
        return self._layers[name]

    def get(self, pose, eyes, mouth):
        k = (pose, eyes, mouth)
        if k not in self._cache:
            m = self.m
            im = self.layer(m['base'].get(pose) or m['base']['normal']).copy()
            im.alpha_composite(self.layer(m['eyes'].get(eyes) or m['eyes']['open']))
            im.alpha_composite(self.layer(m['mouth'].get(mouth) or m['mouth']['closed']))
            if m.get('front'):
                im.alpha_composite(self.layer(m['front']))
            self._cache[k] = im
        return self._cache[k]


# ------------------------------------------------------------------ 音声
def tts(text, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha1(('v1|' + text).encode('utf-8')).hexdigest()[:16]
    wav = os.path.join(cache_dir, key + '.wav')
    if not os.path.exists(wav):
        txt = wav + '.txt'
        with open(txt, 'w', encoding='utf-8') as fh:
            fh.write(text)
        subprocess.run(['open_jtalk', '-x', DIC, '-m', VOICE, '-r', '1.2', '-fm', '1.0', '-ow', wav + '.tmp', txt],
                       check=True)
        os.replace(wav + '.tmp', wav)
    with wave.open(wav) as w:
        sr, n = w.getframerate(), w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1)
    hop = sr // FPS
    env = np.array([np.sqrt(np.mean(pcm[i:i + hop] ** 2)) for i in range(0, len(pcm), hop)], dtype=np.float32)
    ref = max(float(np.percentile(env, 95)), 1e-4)
    return pcm, sr, len(pcm) / sr, env / ref


def readings(text):
    """Open JTalk がどう読むか(単語(読み) の列)。読み間違いの確認用"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        txt, tr = os.path.join(td, 'a.txt'), os.path.join(td, 'a.trace')
        with open(txt, 'w', encoding='utf-8') as fh:
            fh.write(text)
        subprocess.run(['open_jtalk', '-x', DIC, '-m', VOICE, '-ot', tr, '-ow', os.path.join(td, 'a.wav'), txt],
                       check=True)
        out = []
        for ln in open(tr, encoding='utf-8', errors='replace').read().splitlines()[1:]:
            if not ln.strip() or ln.startswith('['):
                break
            f = ln.split(',')
            if len(f) > 9:
                out.append('%s(%s)' % (f[0], f[9]))
        return ' '.join(out)


# ------------------------------------------------------------------ タイムライン(場面・行・時刻)
def build(story, drives, work):
    scenes, T = [], 0.0
    sr = None
    for sc in story['scenes']:
        drive = drives[sc.get('log') or story.get('log')]
        lines, t = [], LEAD_S
        for ln in sc.get('lines', []):
            pcm, sr, dur, env = tts(ln.get('say') or ln['text'], os.path.join(work, 'tts'))
            lines.append(dict(ln, start=t, dur=dur, pcm=pcm, env=env))
            t += dur + GAP_S
        narr = (t - GAP_S + TAIL_S) if lines else 1.5
        a, b = float(sc['src_start']), float(sc['src_end'])
        a, b = max(0.0, a), min(drive.duration, b)
        clip = max(b - a, 0.0)
        speed = float(sc.get('speed') or 1.0)
        nat = clip / speed if speed > 0 else 0
        if narr > nat:
            speed_eff = max(MIN_SPEED, clip / narr) if clip > 0 else 1.0
            dur = narr
        else:
            speed_eff, dur = speed, nat
        scenes.append(dict(sc, src_start=a, src_end=b, t_out=T, dur=dur, speed_eff=speed_eff, lines=lines))
        T += dur
    return scenes, T, sr


def scene_at(scenes, t):
    for sc in scenes:
        if t < sc['t_out'] + sc['dur']:
            return sc
    return scenes[-1]


# ------------------------------------------------------------------ 描画
class Renderer(object):
    def __init__(self, drives, story, scenes, total, char):
        self.drives = drives
        self.d = drives[scenes[0].get('log') or story.get('log')]
        self._tl = {}
        self.story = story
        self.scenes = scenes
        self.total = total
        self.char = char
        self.base = self._static()
        rng = np.random.default_rng(7)
        self.blinks, t = [], 1.5
        while t < total + 5:
            self.blinks.append(t)
            t += float(rng.uniform(2.6, 4.8))
        self._bubble = (None, None)
        self._cap = (None, None)

    # ---- 静的な背景
    def _static(self):
        im = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(im)
        for box, title in ((TOF, '距離センサー(ToF)'), (GMETER, '加速度・回転'), (STEER, 'ステアリング'),
                           (THROT, 'アクセル'), (SPEED, 'モーター回転数'), (TL, '')):
            x, y, w, h = box
            rrect(d, (x, y, x + w, y + h), 14, fill=PANEL, outline=EDGE, width=2)
            if title:
                d.text((x + 14, y + 8), title, font=font(22, True), fill=SUB)
        x, y, s = CAM
        rrect(d, (x - 2, y - 2, x + s + 2, y + s + 2), 14, fill=PANEL, outline=EDGE, width=2)
        # 右のキャラクター欄
        d.rectangle((SIDE_X, 0, W, H), fill=(14, 19, 34))
        d.line((SIDE_X, 0, SIDE_X, H), fill=EDGE, width=2)
        # ステアリング・アクセル・速度の目盛り
        cx, cy, r = STEER[0] + STEER[2] // 2, STEER[1] + 158, 92
        for k in range(-4, 5):
            a = math.radians(k * 30)
            p1 = (cx + (r + 8) * math.sin(a), cy - (r + 8) * math.cos(a))
            p2 = (cx + (r + 18) * math.sin(a), cy - (r + 18) * math.cos(a))
            d.line((p1, p2), fill=DIM, width=3 if k in (-4, 0, 4) else 1)
        d.text((STEER[0] + 14, STEER[1] + 52), '左', font=font(22), fill=SUB)
        d.text((STEER[0] + STEER[2] - 36, STEER[1] + 52), '右', font=font(22), fill=SUB)
        bx, by0, by1 = THROT[0] + THROT[2] // 2, THROT[1] + 56, THROT[1] + THROT[3] - 56
        rrect(d, (bx - 26, by0, bx + 26, by1), 10, fill=(10, 15, 28), outline=EDGE, width=2)
        mid = (by0 + by1) // 2
        d.line((bx - 40, mid, bx + 40, mid), fill=SUB, width=2)
        d.text((bx + 38, by0 - 4), '前進', font=font(20), fill=SUB)
        d.text((bx + 38, by1 - 24), '後退', font=font(20), fill=SUB)
        d.text((bx - 98, mid - 14), '中立', font=font(20), fill=SUB)
        sx, sy, sr = SPEED[0] + SPEED[2] // 2, SPEED[1] + 176, 90
        d.arc((sx - sr, sy - sr, sx + sr, sy + sr), 150, 390, fill=(30, 40, 62), width=16)
        for k in range(0, 11):
            a = math.radians(150 + 24 * k)
            p1 = (sx + (sr + 12) * math.cos(a), sy + (sr + 12) * math.sin(a))
            p2 = (sx + (sr + 20) * math.cos(a), sy + (sr + 20) * math.sin(a))
            d.line((p1, p2), fill=DIM, width=2)
        gx, gy, gr = GMETER[0] + GMETER[2] // 2, GMETER[1] + 160, 104
        for rr, lab in ((gr, '2G'), (gr // 2, '1G')):
            d.ellipse((gx - rr, gy - rr, gx + rr, gy + rr), outline=(44, 56, 84), width=2)
            d.text((gx + rr * 0.72, gy - rr * 0.72 - 6), lab, font=font(16), fill=DIM)
        d.line((gx - gr, gy, gx + gr, gy), fill=(36, 46, 70), width=1)
        d.line((gx, gy - gr, gx, gy + gr), fill=(36, 46, 70), width=1)
        # ToFの色の凡例
        tx, ty, tw, th = TOF
        for i in range(160):
            c = cv2.applyColorMap(np.uint8([[255 - int(i * 255 / 159)]]), cv2.COLORMAP_TURBO)[0, 0][::-1]
            d.line((tx + tw - 190 + i, ty + 18, tx + tw - 190 + i, ty + 28), fill=tuple(int(v) for v in c))
        d.text((tx + tw - 230, ty + 12), '近', font=font(18), fill=SUB)
        d.text((tx + tw - 24, ty + 12), '遠', font=font(18), fill=SUB)
        return im

    def _tx(self, t):
        x0, x1 = TL[0] + 70, TL[0] + TL[2] - 24
        return x0 + (x1 - x0) * t / self.d.duration

    def _timeline_base(self):
        im = Image.new('RGBA', (TL[2], TL[3]), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        ox = TL[0]
        tx = lambda t: self._tx(t) - ox                                         # noqa: E731
        sy0, sy1, ty0, ty1 = 86, 172, 186, 262
        for a, b in self.d.segments:
            d.rectangle((tx(a), sy0 - 6, tx(b), ty1 + 6), fill=PURPLE + (52,))
        for a, b in self.d.holds:
            d.rectangle((tx(a), sy0 - 6, max(tx(b), tx(a) + 3), ty1 + 6), fill=RED + (150,))
        for a, b in self.d.recoveries:
            d.rectangle((tx(a), ty1 - 4, max(tx(b), tx(a) + 3), ty1 + 6), fill=ORANGE + (230,))
        for lap in self.d.laps:                                  # 1周の区切り
            x = tx(lap['t_end'])
            d.line((x, ty1 + 2, x, ty1 + 8), fill=GREEN + (220,), width=2)
        d.line((tx(0), (sy0 + sy1) / 2, tx(self.d.duration), (sy0 + sy1) / 2), fill=EDGE, width=1)
        d.line((tx(0), (ty0 + ty1) / 2, tx(self.d.duration), (ty0 + ty1) / 2), fill=EDGE, width=1)
        d.text((10, sy0 + 34), 'ステア', font=font(17), fill=SUB)
        d.text((10, ty0 + 30), 'アクセル', font=font(17), fill=SUB)
        ts = np.arange(0, self.d.duration, 1 / 30.0)
        sp, tp = [], []
        a_idx = np.clip(np.searchsorted(self.d.ai_t, ts, side='right') - 1, 0, len(self.d.ai_t) - 1)
        v_idx = np.clip(np.searchsorted(self.d.v_t, ts, side='right') - 1, 0, len(self.d.v_t) - 1)
        sel = self.d.sel[a_idx]
        hold = self.d.hold[a_idx] & sel
        steer = np.where(sel, np.where(hold, self.d.center, self.d.out[a_idx, 0]), self.d.pwm[v_idx, 0])
        thr = np.where(sel, np.where(hold, self.d.neutral, self.d.out[a_idx, 1]), self.d.pwm[v_idx, 1])
        for t, s, th, on in zip(ts, steer, thr, sel):
            sp.append((tx(t), (sy0 + sy1) / 2 - self.d.steer_frac(s) * (sy1 - sy0) / 2, on))
            tp.append((tx(t), (ty0 + ty1) / 2 - self.d.fwd_frac(th) * (ty1 - ty0) / 2, on))
        for pts, col in ((sp, CYAN), (tp, GREEN)):
            for p, q in zip(pts[:-1], pts[1:]):
                d.line((p[0], p[1], q[0], q[1]), fill=col if p[2] else (150, 156, 170), width=2)
        last_lab = -1e9
        for t, g, sus in self.d.impacts:
            x = tx(t)
            col = (148, 163, 184) if sus else ORANGE
            d.polygon([(x - 7, sy0 - 24), (x + 7, sy0 - 24), (x, sy0 - 12)], fill=col)
            if g >= 5 and x - last_lab > 40:            # 大きいものだけ数字を書く(近いものが重ならないように)
                last_lab = x
                lab = '%.0fG%s' % (g, '?' if sus else '')
                d.text((x - d.textlength(lab, font=font(16, True)) / 2, sy0 - 46), lab, font=font(16, True), fill=col)
        for t in self.d.ai_off:
            x = tx(t)
            d.line((x, sy0 - 6, x, ty1 + 6), fill=WHITE + (170,), width=2)
        for k in range(0, int(self.d.duration) + 1, 10):
            x = tx(k)
            d.line((x, ty1 + 6, x, ty1 + 12), fill=SUB, width=1)
            d.text((x - 10, ty1 + 12), '%d秒' % k, font=font(15), fill=SUB)
        # 凡例
        lx = 14
        d.text((lx, 10), '%s 全体(%.0f秒)' % (self.d.description, self.d.duration), font=font(22, True), fill=SUB)
        lx = 300
        for col, lab in ((PURPLE, 'AIが運転'), (RED, '中立で停止(HOLD)'), (ORANGE, '衝撃 2G超'), (WHITE, '人が交代'),
                         (CYAN, 'ステア(左が上)'), (GREEN, 'アクセル(前進が上)')):
            d.rectangle((lx, 18, lx + 16, 32), fill=col)
            d.text((lx + 22, 12), lab, font=font(17), fill=SUB)
            lx += 30 + int(d.textlength(lab, font=font(17))) + 18
        return im

    # ---- 動く部分
    def render(self, t_out):
        sc = scene_at(self.scenes, t_out)
        log = sc.get('log') or self.story.get('log')
        self.d = self.drives[log]
        if log not in self._tl:
            self._tl[log] = self._timeline_base()
        self.tl_base = self._tl[log]
        rel = t_out - sc['t_out']
        src = sc['src_start'] + min(rel * sc['speed_eff'], sc['src_end'] - sc['src_start'])
        frozen = rel * sc['speed_eff'] > sc['src_end'] - sc['src_start'] + 1e-6
        st = self.d.state(src)
        hl = set(sc.get('highlights') or [])
        im = self.base.copy()
        d = ImageDraw.Draw(im, 'RGBA')
        self._camera(im, d, st, hl)
        self._tof(im, d, st, hl)
        self._gmeter(d, st, hl)
        self._steer(d, st, hl)
        self._throttle(d, st, hl)
        self._speed(d, st, hl)
        self._timeline(im, d, sc, src, hl)
        self._header(d, sc, st, frozen, hl)
        if sc['kind'] in ('title', 'summary', 'ending'):
            self._card(im, sc, rel)
        elif sc['kind'] == 'legend':
            self._legend(d, sc, rel)
        self._side(im, d, sc, rel, t_out)
        if rel < 0.25 and sc is not self.scenes[0]:                 # 場面の切り替え: 一瞬暗くしてから戻す
            a = int(200 * (1 - rel / 0.25))
            d.rectangle((0, 0, MAIN_W, H), fill=(0, 0, 0, a))
        return im

    def _glow(self, d, box, col, t, on):
        if not on:
            return
        a = int(140 + 100 * math.sin(t * 6.0))
        x, y, w, h = box
        rrect(d, (x - 4, y - 4, x + w + 4, y + h + 4), 16, outline=col + (a,), width=5)

    def _camera(self, im, d, st, hl):
        x, y, s = CAM
        f = cv2.resize(self.d.frame(st['t']), (s, s), interpolation=cv2.INTER_CUBIC)
        cam = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        ov = Image.new('RGBA', (s * 2, s * 2), (0, 0, 0, 0))
        self._path(ImageDraw.Draw(ov), st, s * 2, 'prediction' in hl)
        cam = cam.convert('RGBA')
        cam.alpha_composite(ov.resize((s, s), Image.LANCZOS))
        mask = Image.new('L', (s, s), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, s - 1, s - 1), 12, fill=255)
        im.paste(cam.convert('RGB'), (x, y), mask)
        rrect(d, (x + 10, y + 10, x + 196, y + 46), 10, fill=(0, 0, 0, 150))
        d.text((x + 22, y + 13), 'カメラ(AIの目)', font=font(22, True), fill=WHITE)
        if st['lat'] is not None:
            msg = '推論 %.1fms ・ 遅れ %.0fms ・ 30fps' % (st['inf'] or 0, st['lat'])
            tw = d.textlength(msg, font=font(19))
            rrect(d, (x + s - tw - 30, y + s - 44, x + s - 10, y + s - 10), 10, fill=(0, 0, 0, 150))
            d.text((x + s - tw - 20, y + s - 41), msg, font=font(19), fill=CYAN if 'latency' in hl else (220, 225, 235))
        # 周回(右上)
        lap_txt = '%d周目' % (st['laps_done'] + 1)
        if st['lap_elapsed'] is not None:
            lap_txt += ' %.1f秒' % st['lap_elapsed']
        if st['last_lap']:
            lap_txt += ' ・ 前 %.1f秒' % st['last_lap']['time_s']
        if st['best_lap']:
            lap_txt += ' ・ ベスト %.2f秒' % st['best_lap']
        fl = font(21, True)
        tw = d.textlength(lap_txt, font=fl)
        rrect(d, (x + s - tw - 34, y + 10, x + s - 10, y + 46), 10, fill=(0, 0, 0, 160))
        d.text((x + s - tw - 22, y + 14), lap_txt, font=fl, fill=GREEN if 'laps' in hl else WHITE)
        # 出来事の帯
        t = st['t']
        lap = st['last_lap']
        if lap and 0 <= t - lap['t_end'] < 1.6:
            best = st['best_lap'] is not None and abs(lap['time_s'] - st['best_lap']) < 1e-6 and lap['ai_only']
            msg = ('ベスト! ' if best else '') + 'ラップ %.2f秒' % lap['time_s'] + ('' if lap['ai_only'] else '(人の運転を含む)')
            f2 = font(28, True)
            tw = d.textlength(msg, font=f2)
            rrect(d, (x + (s - tw) / 2 - 18, y + 116, x + (s + tw) / 2 + 18, y + 162), 12,
                  fill=(GREEN if lap['ai_only'] else (100, 116, 139)) + (225,))
            d.text((x + (s - tw) / 2, y + 120), msg, font=f2, fill=WHITE)
        banner = None
        if st['hold']:
            banner = (RED, 'HOLD  コントローラ基板が再起動 → 車両データなしで中立' if st['board_reset']
                      else 'HOLD  センサーの値が遅れたので中立(安全のため停止)')
        elif st['assist'] == 2:
            banner = (ORANGE, '自動リカバリー中(ぶつかって止まったので下がる)')
        elif st['assist'] == 3:
            banner = (RED, 'リカバリー中止 → 人の交代待ち')
        else:
            for a in self.d.ai_on:
                if 0 <= t - a < 1.3:
                    banner = (PURPLE, 'AIに交代!(プロポのCH3)')
            for a in self.d.ai_off:
                if 0 <= t - a < 1.3:
                    banner = ((100, 116, 139), '人が運転を交代')
        if banner:
            col, msg = banner
            f2, _ = fit(d, msg, s - 60, (26, 23, 20, 18), 1)       # カメラの幅に収める
            tw = d.textlength(msg, font=f2)
            rrect(d, (x + (s - tw) / 2 - 18, y + 60, x + (s + tw) / 2 + 18, y + 104), 12, fill=col + (225,))
            d.text((x + (s - tw) / 2, y + 64), msg, font=f2, fill=WHITE)
        if st['hold']:
            rrect(d, (x - 2, y - 2, x + s + 2, y + s + 2), 14, outline=RED + (230,), width=6)
        for ti, g, suspect in self.d.impacts:
            if suspect and 0 <= t - ti < 1.2:
                msg = '%.1fG? センサーの誤読の疑い' % g
                f3 = font(28, True)
                tw = d.textlength(msg, font=f3)
                rrect(d, (x + (s - tw) / 2 - 16, y + s * 0.62 - 26, x + (s + tw) / 2 + 16, y + s * 0.62 + 22), 12,
                      fill=(71, 85, 105, 220))
                d.text((x + (s - tw) / 2, y + s * 0.62 - 22), msg, font=f3, fill=WHITE)
                continue
            if 0 <= t - ti < 0.7 and g >= 2:
                k = (t - ti) / 0.7
                cx, cy = x + s // 2, y + int(s * 0.62)
                r1, r2 = 70 + 60 * k, 130 + 90 * k
                pts = []
                for j in range(24):
                    a = j * math.pi / 12
                    rr = r2 if j % 2 == 0 else r1
                    pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a) * 0.7))
                d.polygon(pts, fill=ORANGE + (int(170 * (1 - k)),))
                f3 = font(52, True)
                msg = 'ドン! %.1fG' % g
                tw = d.textlength(msg, font=f3)
                d.text((cx - tw / 2, cy - 34), msg, font=f3, fill=WHITE + (int(255 * (1 - k * 0.6)),),
                       stroke_width=4, stroke_fill=(120, 40, 0, int(255 * (1 - k * 0.6))))
        self._glow(d, (x, y, s, s), PURPLE, t, 'prediction' in hl and not st['hold'])

    def _path(self, d, st, s, emph):
        """AIの予測(10フレーム=0.33秒先まで)を地面に投影した帯。左右は車体基準(左=中心より大きいµs)"""
        horizon, f, hcam = 0.36 * s, 0.355 * s, 0.12
        cx = s / 2
        pred = st['pred']
        if pred is None:
            return
        speed = abs(st['erpm'])
        L = float(np.clip(0.45 + speed / 5000.0, 0.45, 1.9))
        kmax = 1 / 0.75
        x, z, hd = 0.0, 0.07, 0.0
        left, right, center, marks = [], [], [], []
        n = 40
        for i in range(n + 1):
            k = min(int(i * 10 / (n + 1)), 9)
            us = self.d.center + (float(pred[k, 0]) - self.d.center) * self.d.steer_gain + self.d.trim
            us = float(np.clip(us, 1000, 2000))
            kap = self.d.steer_frac(us) * kmax
            ds = L / n
            hd += kap * ds
            x += math.sin(hd) * ds
            z += math.cos(hd) * ds
            if z < 0.06:
                break
            hw = 0.055
            for arr, off in ((left, hw), (right, -hw), (center, 0.0)):
                px, pz = x - off * math.cos(hd), z + off * math.sin(hd)        # 左 = +x
                pz = max(pz, 0.05)
                arr.append((cx - f * px / pz, horizon + f * hcam / pz))
            if i % 4 == 0:
                marks.append(center[-1])
        if len(center) < 2:
            return
        if st['hold']:
            col, a = RED, 120
        elif st['sel']:
            col, a = PURPLE, 150 if not emph else 190
        else:
            col, a = (200, 205, 215), 70
        d.polygon(left + right[::-1], fill=col + (a,))
        d.line(center, fill=WHITE + (200 if st['sel'] else 110,), width=6, joint='curve')
        for i, p in enumerate(marks):
            r = 9 - i * 0.6
            d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=WHITE + (220 - i * 15,))
        if emph or st['sel']:
            p = center[-1]
            d.text((p[0] + 16, p[1] - 30), 'AIの予測コース' if st['sel'] else 'AIの予測(人が運転中)', font=font(40, True),
                   fill=WHITE + (235,), stroke_width=4, stroke_fill=(30, 10, 60, 200))

    def _tof(self, im, d, st, hl):
        x, y, w, h = TOF
        dist = tof_format.distance_mm(np.asarray(st['depth'])).astype(np.float32)
        v = np.clip(dist / 2500.0, 0, 1)
        col = cv2.applyColorMap((255 * (1 - v)).astype(np.uint8), cv2.COLORMAP_TURBO)
        col[dist <= 0] = (40, 30, 30)
        pw, ph = w - 28, h - 64
        col = cv2.resize(col, (pw, ph), interpolation=cv2.INTER_NEAREST)
        im.paste(Image.fromarray(cv2.cvtColor(col, cv2.COLOR_BGR2RGB)), (x + 14, y + 48))
        # 「前方の最短」は床・バンパーも含んでしまい誤解を招くので出さない(ToFガードの衝突までの時間を出す)
        if st['ttc'] is not None and st['sel']:                      # ToFガード(記録だけ)の判定
            ttc = min(st['ttc'], 9.9)
            col = RED if ttc < 0.3 else (ORANGE if ttc < 0.5 else (200, 205, 215))
            msg = '衝突まで %.2f秒' % ttc + {1: ' → 減速(記録だけ)', 2: ' → ブレーキ(記録だけ)'}.get(st['guard'], '')
            f3 = font(20, True)
            tw = d.textlength(msg, font=f3)
            rrect(d, (x + w - tw - 34, y + 48, x + w - 14, y + 80), 8, fill=(0, 0, 0, 170))
            d.text((x + w - tw - 24, y + 51), msg, font=f3, fill=col)
        self._glow(d, TOF, CYAN, st['t'], 'tof' in hl)

    def _gmeter(self, d, st, hl):
        x, y, w, h = GMETER
        gx, gy, gr = x + w // 2, y + 160, 104
        sc = gr / 2.0
        tr = st['trail']
        # IMUは90度回して付いている: +x = 右、+y = 前(左右は横、前後は縦に描く)
        pts = [(gx + float(np.clip(p[0], -2.4, 2.4)) * sc, gy - float(np.clip(p[1], -2.4, 2.4)) * sc) for p in tr]
        for i, p in enumerate(pts[:-1]):
            a = int(40 + 160 * i / max(len(pts) - 1, 1))
            d.line((p, pts[i + 1]), fill=CYAN + (a,), width=3)
        if pts:
            p = pts[-1]
            hit = st['gmag'] > 2
            r = 11 if not hit else 16
            d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=(ORANGE if hit else CYAN) + (255,))
        if not st['imu_ok']:                                                 # IMUのデータが途切れている
            d.ellipse((gx - gr, gy - gr, gx + gr, gy + gr), fill=(17, 24, 39, 200))
            msg = 'データなし(途切れ)'
            tw = d.textlength(msg, font=font(24, True))
            d.text((gx - tw / 2, gy - 16), msg, font=font(24, True), fill=RED)
            d.text((x + 14, y + h - 38), '回転 ---', font=font(21), fill=DIM)
            self._glow(d, GMETER, RED, st['t'], True)
            return
        recent = [(g, sus) for ti, g, sus in self.d.impacts if 0 <= st['t'] - ti < 1.0]     # 衝撃の最大値を1秒間表示
        if recent or st['gmag'] > 2:
            g, sus = max(recent) if recent else (st['gmag'], False)
            col = (148, 163, 184) if sus else ORANGE
            d.ellipse((gx - gr - 6, gy - gr - 6, gx + gr + 6, gy + gr + 6), outline=col + (230,), width=6)
            d.text((x + 14, y + h - 70), ('%.1fG?(誤読の疑い)' if sus else '衝撃 %.1fG') % g, font=font(24, True), fill=col)
        yaw = st['yaw']
        side = '左' if yaw > 8 else ('右' if yaw < -8 else '')
        d.text((x + 14, y + h - 38), '回転 %+.0f°/秒 %s' % (yaw, side), font=font(21), fill=TEXT)
        self._glow(d, GMETER, ORANGE, st['t'], 'impact' in hl)

    def _steer(self, d, st, hl):
        x, y, w, h = STEER
        cx, cy, r = x + w // 2, y + 158, 92
        fr = self.d.steer_frac(st['steer'])
        ang = -fr * 120                                                       # 左 = 反時計回り
        col = PURPLE if st['sel'] and not st['hold'] else ((150, 160, 180) if not st['sel'] else RED)
        lim = abs(fr) > 0.995 and st['sel']
        d.arc((cx - r, cy - r, cx + r, cy + r), -90 + min(0, ang), -90 + max(0, ang), fill=col + (255,), width=12)
        a = math.radians(ang)
        tip = (cx + (r - 2) * math.sin(a), cy - (r - 2) * math.cos(a))
        d.ellipse((tip[0] - 10, tip[1] - 10, tip[0] + 10, tip[1] + 10), fill=(RED if lim else WHITE) + (255,))
        # ハンドルの絵
        rw = 54
        d.ellipse((cx - rw, cy - rw, cx + rw, cy + rw), outline=(200, 205, 220, 255), width=9)
        for base in (0, 120, 240):
            b = math.radians(base + ang + 180)
            d.line((cx, cy, cx + rw * math.sin(b), cy - rw * math.cos(b)), fill=(200, 205, 220, 255), width=7)
        d.ellipse((cx - 13, cy - 13, cx + 13, cy + 13), fill=col + (255,))
        if not st['sel']:                                                    # 人が運転中: AIの案をうすく
            fa = self.d.steer_frac(st['ai_steer'])
            aa = math.radians(-fa * 120)
            p = (cx + (r + 14) * math.sin(aa), cy - (r + 14) * math.cos(aa))
            d.ellipse((p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6), fill=PURPLE + (170,))
        if lim:
            msg = ('左' if fr > 0 else '右') + 'いっぱい!'
            c = RED
        elif abs(fr) < 0.05:
            msg, c = 'まっすぐ', TEXT
        else:
            msg, c = '%s %d%%' % ('左' if fr > 0 else '右', round(abs(fr) * 100)), TEXT
        tw = d.textlength(msg, font=font(26, True))
        d.text((cx - tw / 2, y + h - 44), msg, font=font(26, True), fill=c)
        self._glow(d, STEER, CYAN, st['t'], bool({'full_lock', 'weaving', 'gauges'} & hl))

    def _throttle(self, d, st, hl):
        x, y, w, h = THROT
        bx, by0, by1 = x + w // 2, y + 56, y + h - 56
        mid = (by0 + by1) / 2
        fr = self.d.fwd_frac(st['thr'])
        col = GREEN if fr >= 0 else RED
        if st['hold']:
            col = RED
        top = mid - fr * (mid - by0 - 4)
        d.rounded_rectangle((bx - 20, min(mid, top), bx + 20, max(mid, top)), 6, fill=col + (255,))
        if not st['sel']:
            fa = self.d.fwd_frac(st['ai_thr'])
            ya = mid - fa * (mid - by0 - 4)
            d.line((bx - 34, ya, bx - 22, ya), fill=PURPLE + (200,), width=5)
        if fr < -0.08:
            msg = 'ブレーキ/後退'
        elif fr > 0.05:
            msg = '前進 %d%%' % round(fr * 100)
        else:
            msg = '中立'
        tw = d.textlength(msg, font=font(24, True))
        d.text((bx - tw / 2, y + h - 42), msg, font=font(24, True), fill=RED if fr < -0.08 else TEXT)
        self._glow(d, THROT, GREEN, st['t'], bool({'reverse', 'gauges'} & hl))

    def _speed(self, d, st, hl):
        x, y, w, h = SPEED
        sx, sy, sr = x + w // 2, y + 176, 90
        if not st['veh_ok']:
            msg = 'データなし'
            tw = d.textlength(msg, font=font(30, True))
            d.text((sx - tw / 2, sy - 24), msg, font=font(30, True), fill=RED)
            self._glow(d, SPEED, RED, st['t'], True)
            return
        e = abs(st['erpm'])
        fr = min(e / 10000.0, 1.0)
        if fr > 0.002:
            d.arc((sx - sr, sy - sr, sx + sr, sy + sr), 150, 150 + 240 * fr, fill=CYAN + (255,), width=16)
        msg = '%d' % e
        tw = d.textlength(msg, font=font(46, True))
        d.text((sx - tw / 2, sy - 40), msg, font=font(46, True), fill=WHITE)
        d.text((sx - 24, sy + 14), 'eRPM', font=font(18), fill=SUB)
        kmh = '約 %.1f km/h' % (e * M_PER_S_PER_ERPM * 3.6)
        d.text((sx - d.textlength(kmh, font=font(22, True)) / 2, sy + 42), kmh, font=font(22, True), fill=CYAN)
        dirn = '前進' if st['erpm'] < -50 else ('後退' if st['erpm'] > 50 else '停止')
        tw = d.textlength(dirn, font=font(24, True))
        d.text((sx - tw / 2, y + h - 42), dirn, font=font(24, True), fill=RED if dirn == '後退' else TEXT)
        self._glow(d, SPEED, CYAN, st['t'], 'speed' in hl)

    def _timeline(self, im, d, sc, src, hl):
        x, y = TL[0], TL[1]
        im.paste(self.tl_base, (x, y), self.tl_base)
        a, b = self._tx(sc['src_start']), self._tx(sc['src_end'])
        d.rectangle((a, y + 76, b, y + 268), outline=WHITE + (120,), width=2, fill=WHITE + (18,))
        px = self._tx(src)
        d.line((px, y + 76, px, y + 268), fill=WHITE + (255,), width=3)
        lab = '%.1f秒' % src
        f = font(18, True)
        tw = d.textlength(lab, font=f)
        rrect(d, (px - tw / 2 - 8, y + 266, px + tw / 2 + 8, y + 292), 8, fill=(255, 255, 255, 255))
        d.text((px - tw / 2, y + 267), lab, font=f, fill=(15, 23, 42))
        self._glow(d, TL, PURPLE, src, bool({'timeline', 'weaving'} & hl))

    def _header(self, d, sc, st, frozen, hl):
        badges = []
        if st['hold']:
            badges.append((RED, 'HOLD 中立'))
        elif st['assist'] == 2:
            badges.append((ORANGE, 'リカバリー中'))
        elif st['sel']:
            badges.append((PURPLE, 'AIが運転中'))
        else:
            badges.append(((71, 85, 105), '人が運転中'))
        sp = sc['speed_eff']
        if frozen:
            badges.append(((51, 65, 85), '一時停止'))
        elif sp < 0.95:
            badges.append(((14, 116, 144), 'スロー %.1fx' % sp))
        elif sp > 1.05:
            badges.append(((14, 116, 144), '早送り %.1fx' % sp))
        xr = MAIN_W - 24
        f = font(26, True)
        for col, msg in badges:
            tw = d.textlength(msg, font=f)
            rrect(d, (xr - tw - 32, 20, xr, 66), 23, fill=col + (255,))
            d.text((xr - tw - 16, 25), msg, font=f, fill=WHITE)
            xr -= tw + 44
        clock = self.d.t0 / 1e6 + st['t']
        import time as _t
        tm = _t.strftime('%H:%M:%S', _t.localtime(clock))
        msg = '%s %s (%.1f秒)' % (self.d.description, tm, st['t'])
        tw = d.textlength(msg, font=font(22))
        d.text((xr - tw - 6, 32), msg, font=font(22), fill=SUB)
        d.rectangle((24, 18, 30, 74), fill=PURPLE)
        cap = sc.get('caption', '')
        if self._cap[0] != (cap, int(xr)):
            self._cap = ((cap, int(xr)), fit(d, cap, xr - tw - 70, (42, 38, 34, 30), 1) if d.textlength(cap, font=font(30, True)) <= xr - tw - 70
                         else fit(d, cap, xr - tw - 70, (28, 25, 23), 2))
        f, lines = self._cap[1]
        lh = f.size + 6
        yy = 46 - lh * len(lines) / 2 - 4
        for ln in lines:
            d.text((44, yy), ln, font=f, fill=WHITE)
            yy += lh

    def _card(self, im, sc, rel):
        x0, y0, x1, y1 = 24, 96, MAIN_W - 24, 744
        ov = Image.new('RGBA', (x1 - x0, y1 - y0), (8, 10, 22, 214))
        d = ImageDraw.Draw(ov)
        k = min(rel / 0.5, 1.0)
        if sc['kind'] == 'title':
            f, lines = fit(d, self.story.get('title', ''), x1 - x0 - 120, (66, 60, 54, 48), 2)
            yy = 90
            for ln in lines:
                d.text((60, yy), ln, font=f, fill=WHITE)
                yy += f.size + 18
            yy += 10
            d.rectangle((60, yy, 60 + int(560 * k), yy + 8), fill=PURPLE)
            yy += 40
            infos = sc.get('summary_items') or [sc.get('caption', '')]
            for s in infos:
                f2, ls = fit(d, s, x1 - x0 - 130, (34, 30, 27), 2, bold=False)
                for ln in ls:
                    d.text((64, yy), ln, font=f2, fill=(215, 220, 235))
                    yy += f2.size + 16
                yy += 10
        else:
            fc, cl = fit(d, sc.get('caption', ''), x1 - x0 - 120, (52, 46, 40, 34), 1)
            d.text((60, 46), cl[0], font=fc, fill=WHITE)
            items = sc.get('summary_items') or []
            cols = 2 if len(items) > 3 else 1
            cw = (x1 - x0 - 120 - 30 * (cols - 1)) // cols
            yy0 = 140
            for i, s in enumerate(items):
                if rel < 0.25 + i * 0.25:
                    continue
                cx = 60 + (i % cols) * (cw + 30)
                cy = yy0 + (i // cols) * 118
                d.rounded_rectangle((cx, cy, cx + cw, cy + 100), 16, fill=(24, 32, 54, 240), outline=EDGE, width=2)
                d.rectangle((cx, cy + 14, cx + 6, cy + 86), fill=CYAN if sc['kind'] == 'summary' else PURPLE)
                fl, lines = fit(d, s, cw - 50, (30, 27, 24), 2)
                lh = fl.size + 9
                ty = cy + (100 - lh * len(lines)) // 2 - 2
                for ln in lines:
                    d.text((cx + 26, ty), ln, font=fl, fill=WHITE)
                    ty += lh
        im.paste(ov, (x0, y0), ov)

    def _legend(self, d, sc, rel):
        items = [((CAM[0] + 20, CAM[1] + CAM[2] - 110), '① カメラ+AIの予測コース'),
                 ((TOF[0] + 20, TOF[1] + 50), '② 距離センサー(近い=赤)'),
                 ((GMETER[0] + 8, GMETER[1] + 50), '③ 加速度(衝撃)'),
                 ((STEER[0] + 8, STEER[1] + 48), '④ ステア・アクセル'),
                 ((SPEED[0] + 8, SPEED[1] + 48), '⑤ 速さ'),
                 ((TL[0] + 700, TL[1] + 60), '⑥ 走行全体(紫=AI)')]
        n = len(sc.get('lines', [])) or 1
        f = font(26, True)
        for i, (p, msg) in enumerate(items):
            if rel < 0.4 + i * 0.5:
                continue
            tw = d.textlength(msg, font=f)
            rrect(d, (p[0], p[1], p[0] + tw + 24, p[1] + 42), 10, fill=(250, 204, 21, 235))
            d.text((p[0] + 12, p[1] + 4), msg, font=f, fill=(20, 20, 30))
        del n

    def _manpu(self, d, expr, cx, cy, t, active):
        """表情を読みやすくする漫符(驚き=「!」の線、困り=汗、うれしい=きらきら)"""
        if not active or self.char is None:
            return
        x0, y0, x1, y1 = self.char.m.get('face_box') or [140, 140, 360, 320]
        x0, y0, x1, y1 = cx + x0, cy + y0, cx + x1, cy + y1
        k = 0.5 + 0.5 * math.sin(t * 9)
        if expr == 'surprised':
            ox, oy = x0 - 8, y0 - 14
            for a in (-150, -120, -90):
                r = math.radians(a)
                d.line((ox + 18 * math.cos(r), oy + 18 * math.sin(r), ox + (44 + 6 * k) * math.cos(r), oy + (44 + 6 * k) * math.sin(r)),
                       fill=(250, 204, 21, 255), width=6)
        elif expr == 'worried':
            sx, sy = x1 + 4, y0 + 30 + 6 * k
            d.polygon([(sx, sy - 22), (sx - 11, sy), (sx + 11, sy)], fill=(125, 211, 252, 235))
            d.ellipse((sx - 11, sy - 11, sx + 11, sy + 11), fill=(125, 211, 252, 235))
            d.ellipse((sx - 4, sy - 6, sx + 1, sy - 1), fill=(255, 255, 255, 220))
        elif expr == 'happy':
            for i, (ox, oy, rr) in enumerate(((x0 - 30, y0 + 10, 14), (x0 - 6, y0 - 26, 10), (x1 + 18, y0 - 6, 11))):
                rr = rr * (0.7 + 0.5 * (0.5 + 0.5 * math.sin(t * 7 + i * 2)))
                pts = []
                for j in range(8):
                    a = j * math.pi / 4
                    q = rr if j % 2 == 0 else rr * 0.35
                    pts.append((ox + q * math.cos(a), oy + q * math.sin(a)))
                d.polygon(pts, fill=(253, 230, 138, 240))

    # ---- キャラクターと吹き出し
    def _line_at(self, sc, rel):
        cur, prev = None, None
        for ln in sc['lines']:
            if ln['start'] <= rel:
                prev = ln
                if rel < ln['start'] + ln['dur']:
                    cur = ln
        return cur, prev or (sc['lines'][0] if sc['lines'] else None)

    def _side(self, im, d, sc, rel, t_out):
        cur, shown = self._line_at(sc, rel)
        expr = (shown or {}).get('expression', 'normal')
        pose = (shown or {}).get('pose', 'normal')
        level = 0.0
        if cur is not None:
            i = int((rel - cur['start']) * FPS)
            env = cur['env']
            level = float(env[min(max(i, 0), len(env) - 1)])
        if cur is None:
            mouth = 'smile' if expr == 'happy' else 'closed'
        else:
            mouth = 'open' if level > 0.55 else ('half' if level > 0.18 else 'closed')
        eyes = {'happy': 'happy', 'surprised': 'surprised', 'worried': 'worried'}.get(expr, 'open')
        if eyes != 'happy':
            for b in self.blinks:
                if 0 <= t_out - b < 0.12:
                    eyes = 'closed'
        # 吹き出し
        text = (cur or shown or {}).get('text', '') if (cur or (shown and rel < shown['start'] + shown['dur'] + 1.2)) else ''
        bx0, by0, bx1, by1 = SIDE_X + 18, 22, W - 18, 372
        if text:
            if self._bubble[0] != text:
                b = Image.new('RGBA', (bx1 - bx0, by1 - by0 + 40), (0, 0, 0, 0))
                bd = ImageDraw.Draw(b)
                bd.rounded_rectangle((0, 0, bx1 - bx0 - 1, by1 - by0), 26, fill=(248, 250, 252, 250),
                                     outline=PURPLE + (255,), width=4)
                tx = (bx1 - bx0) * 0.38
                bd.polygon([(tx - 26, by1 - by0 - 4), (tx + 26, by1 - by0 - 4), (tx - 6, by1 - by0 + 36)],
                           fill=(248, 250, 252, 250))
                bd.line([(tx - 26, by1 - by0 - 2), (tx - 6, by1 - by0 + 36), (tx + 26, by1 - by0 - 2)],
                        fill=PURPLE + (255,), width=4)
                for size in (34, 31, 28):
                    f = font(size, True)
                    lines = wrap(bd, text, f, bx1 - bx0 - 76)
                    if len(lines) * (size + 16) <= by1 - by0 - 50:
                        break
                yy = (by1 - by0 - len(lines) * (size + 16)) // 2
                for ln in lines:
                    bd.text((28, yy), ln, font=f, fill=(15, 23, 42))
                    yy += size + 16
                self._bubble = (text, b)
            im.paste(self._bubble[1], (bx0, by0), self._bubble[1])
        # キャラクター
        if self.char is not None:
            c = self.char.get(pose, eyes, mouth)
            bob = 3 * math.sin(t_out * 2 * math.pi / 3.2) + (2.0 * level if cur is not None else 0)
            cx = SIDE_X + (W - SIDE_X - c.width) // 2
            cy = H - c.height + 8 + int(round(bob))
            im.paste(c, (cx, cy), c)
            self._manpu(d, expr, cx, cy, t_out, cur is not None or (shown and rel < shown['start'] + shown['dur'] + 1.0))
            name = self.char.m.get('name_reading') or self.char.m.get('name', '')      # 日本語の視聴者向けにカナで
            if name:
                f = font(22, True)
                tw = d.textlength(name, font=f)
                rrect(d, (W - tw - 44, H - 52, W - 14, H - 14), 18, fill=PURPLE + (235,))
                d.text((W - tw - 29, H - 49), name, font=f, fill=WHITE)


# ------------------------------------------------------------------ 出力
def _tone(sr, freqs, dur, vol):
    t = np.arange(int(sr * dur)) / sr
    out = np.zeros_like(t)
    seg = dur / len(freqs)
    for i, f in enumerate(freqs):
        m = (t >= i * seg) & (t < (i + 1) * seg)
        tt = t[m] - i * seg
        out[m] = np.sin(2 * np.pi * f * tt) * np.exp(-tt * 9) * np.minimum(tt * 400, 1)
    return (out * vol).astype(np.float32)


def sfx(kind, sr):
    """効果音(ナレーションより小さく)"""
    if kind == 'impact':
        t = np.arange(int(sr * 0.45)) / sr
        thump = np.sin(2 * np.pi * (55 + 60 * np.exp(-t * 18)) * t) * np.exp(-t * 9)
        noise = np.random.default_rng(1).standard_normal(len(t)) * np.exp(-t * 35) * 0.35
        return ((thump + noise) * 0.32).astype(np.float32)
    if kind == 'ai_on':
        return _tone(sr, [784, 1175], 0.32, 0.16)
    if kind == 'ai_off':
        return _tone(sr, [880, 587], 0.32, 0.13)
    if kind == 'hold':
        return _tone(sr, [330, 330], 0.28, 0.14)
    if kind == 'lap':
        return _tone(sr, [1319, 1568], 0.24, 0.10)
    return np.zeros(1, np.float32)


def sfx_events(scenes, drives, default_log):
    """場面の中で起きた出来事(元の時刻)を、出力の時刻に直す"""
    out = []
    for sc in scenes:
        drive = drives[sc.get('log') or default_log]
        ev = [(t, 'impact') for t, g, sus in drive.impacts if g >= 4 and not sus] + [(t, 'ai_on') for t in drive.ai_on] + \
             [(t, 'ai_off') for t in drive.ai_off] + [(a, 'hold') for a, b in drive.holds] + \
             [(x['t_end'], 'lap') for x in drive.laps if x['ai_only']]
        if sc['kind'] in ('title', 'summary', 'ending'):
            continue
        for te, kind in ev:
            if sc['src_start'] <= te < sc['src_end']:
                to = sc['t_out'] + (te - sc['src_start']) / sc['speed_eff']
                if to < sc['t_out'] + sc['dur']:
                    out.append((to, kind))
    return out


def audio_track(scenes, total, sr, drives, default_log):
    n = int(math.ceil(total * sr)) + sr
    buf = np.zeros(n, dtype=np.float32)
    for sc in scenes:
        for ln in sc['lines']:
            s = int((sc['t_out'] + ln['start']) * sr)
            p = ln['pcm']
            buf[s:s + len(p)] += p[:max(0, n - s)]
    for to, kind in sfx_events(scenes, drives, default_log):
        p = sfx(kind, sr)
        s = int(to * sr)
        buf[s:s + len(p)] += p[:max(0, n - s)]
    return np.clip(buf, -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('storyboard')
    ap.add_argument('out')
    ap.add_argument('--chars', default=os.path.join(HERE, 'chars', 'final'))
    ap.add_argument('--log', default='20261010-160306')
    ap.add_argument('--stills', default='', help='確認用: 出力時刻(秒)のカンマ区切り。静止画だけ作る')
    ap.add_argument('--crf', default='20')
    ap.add_argument('--readings', action='store_true', help='ナレーションの読みを表示して終わる')
    args = ap.parse_args()
    work = os.path.join(HERE, 'work')
    story = json.load(open(args.storyboard, encoding='utf-8'))
    if args.readings:
        for sc in story['scenes']:
            for ln in sc.get('lines', []):
                print('[%s] %s' % (sc['id'], readings(ln.get('say') or ln['text'])))
        return
    default_log = story.get('log') or args.log
    story['log'] = default_log
    logs = {default_log} | {sc['log'] for sc in story['scenes'] if sc.get('log')}
    drives = {lg: Drive(lg) for lg in sorted(logs)}
    scenes, total, sr = build(story, drives, work)
    char = Character(args.chars) if os.path.exists(os.path.join(args.chars, 'manifest.json')) else None
    r = Renderer(drives, story, scenes, total, char)
    print('全体 %.1f秒 / %d場面' % (total, len(scenes)))
    for sc in scenes:
        print('  %-5s %-7s %5.1f-%5.1fs 出力 %5.1f+%4.1fs 速度 %.2f  %s' % (
            sc['id'], sc['kind'], sc['src_start'], sc['src_end'], sc['t_out'], sc['dur'], sc['speed_eff'], sc.get('caption', '')))
    if args.stills:
        od = os.path.join(work, 'stills')
        os.makedirs(od, exist_ok=True)
        for s in args.stills.split(','):
            t = float(s)
            p = os.path.join(od, 'still_%06.2f.png' % t)
            r.render(t).save(p)
            print('saved', p)
        return
    wav = os.path.join(work, 'narration.wav')
    pcm = (audio_track(scenes, total, sr, drives, default_log) * 32767).astype(np.int16)
    with wave.open(wav, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    # 場面ごとに部品の動画を作る(並列・途中で止まっても続きから): work/parts/<鍵>.mp4
    parts_dir = os.path.join(work, 'parts')
    os.makedirs(parts_dir, exist_ok=True)
    code = hashlib.sha1(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:8]
    chars_key = hashlib.sha1(open(os.path.join(args.chars, 'manifest.json'), 'rb').read()).hexdigest()[:8] if char else 'none'
    n_all = int(round(total * FPS))
    jobs = []
    for k, sc in enumerate(scenes):
        f0 = int(round(sc['t_out'] * FPS))
        f1 = n_all if k == len(scenes) - 1 else int(round((sc['t_out'] + sc['dur']) * FPS))
        desc = {key: v for key, v in sc.items() if key != 'lines'}
        desc['lines'] = [{key: v for key, v in ln.items() if key not in ('pcm', 'env')} for ln in sc['lines']]
        key = hashlib.sha1(json.dumps([desc, f0, f1, code, chars_key, args.crf, story.get('title')], ensure_ascii=False,
                                      sort_keys=True, default=str).encode()).hexdigest()[:12]
        jobs.append((k, f0, f1, os.path.join(parts_dir, '%02d_%s_%s.mp4' % (k, sc['id'][:12], key))))
    todo = [j for j in jobs if not os.path.exists(j[3])]
    print('部品 %d個(作り直し %d個)' % (len(jobs), len(todo)))
    global _R, _CRF
    _R, _CRF = r, args.crf
    if todo:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(int(os.environ.get('DIGEST_JOBS', '3'))) as pool:
            for msg in pool.imap_unordered(_render_part, todo):
                print('  ' + msg, flush=True)
    lst = os.path.join(work, 'parts.txt')
    with open(lst, 'w') as fh:
        fh.writelines("file '%s'\n" % j[3] for j in jobs)
    tmp = args.out + '.tmp.mp4'
    subprocess.run([FFMPEG, '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-i', wav,
                    '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '160k', '-ar', '48000',
                    '-shortest', '-movflags', '+faststart', tmp], check=True)
    os.replace(tmp, args.out)
    print('完成: %s (%.1f秒)' % (args.out, total))


_R, _CRF = None, '20'


def _render_part(job):
    """1場面ぶんのフレームを描いて mp4 にする(fork した子プロセスで動く)"""
    import time as _t
    k, f0, f1, path = job
    tmp = path + '.tmp.mp4'
    cmd = [FFMPEG, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (W, H),
           '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', _CRF, '-threads', '2',
           '-pix_fmt', 'yuv420p', '-g', '60', tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = _t.time()
    for i in range(f0, f1):
        p.stdin.write(_R.render(i / FPS).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError('ffmpeg が失敗しました: %s' % path)
    os.replace(tmp, path)
    return '場面%02d %dフレーム %.0f秒' % (k + 1, f1 - f0, _t.time() - t0)


if __name__ == '__main__':
    main()
