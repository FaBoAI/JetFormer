#!/usr/bin/env python3
"""カメラのコマを選ぶための一覧(下見用。出力は作業用のフォルダ)。
    python3 -I contact.py OUTDIR"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
OUT = Path(sys.argv[1])
HERE = Path(__file__).resolve().parent
d = json.loads((HERE / 'col_data.json').read_text(encoding='utf-8'))
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 14)
RAW = Path('/home/jetson/jetformer/data/raw')
cams = {}


def cam_list(log):
    if log not in cams:
        lj = json.loads((Path('/home/jetson/jetformer/data/labels') / (log + '.json')).read_text(encoding='utf-8'))
        t0 = int(lj['t0_us'])
        ts, ps = [], []
        for s in sorted(p for p in (RAW / log).iterdir() if p.is_dir()):
            try:
                with np.load(s / 'CAM_FRONT.npz') as z:
                    ts.append((z['timestamps_us'] - t0) / 1e6); ps += [str(s / f) for f in z['files']]
            except Exception:
                pass
        cams[log] = (np.concatenate(ts), ps)
    return cams[log]


def frame(log, t):
    ts, ps = cam_list(log)
    j = int(np.argmin(np.abs(ts - t)))
    return ps[j], float(ts[j])


def sheet(name, rows, offsets):
    W = 224
    im = Image.new('RGB', (W * len(offsets) + 260, (W + 4) * len(rows)), 'white')
    dr = ImageDraw.Draw(im)
    for r, (label, log, t) in enumerate(rows):
        dr.text((4, r * (W + 4) + 4), label, fill='black', font=FONT)
        for c, off in enumerate(offsets):
            p, tt = frame(log, t + off)
            try:
                im.paste(Image.open(p).convert('RGB').resize((W, W)), (260 + c * W, r * (W + 4)))
            except Exception:
                continue
            dr.text((262 + c * W, r * (W + 4) + 2), '%+.1f' % off, fill='yellow', font=FONT)
    im.save(OUT / name)
    print(name, len(rows))


ev = [e for e in d['events'] if e['class'] == 'main']
sw = [e for e in ev if e['loc'] in ('SW', 'HS')]
sheet('sw_hs_events.jpg', [('%s No.%s %s\nt=%.1f %.1fm/s %.1fG' % (e['loc'], e['ai_no'], e['setting'][:10], e['t_s'], e['speed_before_mps'] or 0, e['g']), e['log'], e['t_s']) for e in sw], [-1.0, -0.6, -0.3, -0.1, 0.2])
de = [e for e in ev if e['loc'] == 'DE']
sheet('de_events_a.jpg', [('DE No.%s %s\nt=%.1f %.1fm/s %.1fG %s' % (e['ai_no'], e['setting'][:10], e['t_s'], e['speed_before_mps'] or 0, e['g'], e['side']), e['log'], e['t_s']) for e in de[:18]], [-0.6, -0.4, -0.2, -0.1, 0.1])
sheet('de_events_b.jpg', [('DE No.%s %s\nt=%.1f %.1fm/s %.1fG %s' % (e['ai_no'], e['setting'][:10], e['t_s'], e['speed_before_mps'] or 0, e['g'], e['side']), e['log'], e['t_s']) for e in de[18:]], [-0.6, -0.4, -0.2, -0.1, 0.1])
