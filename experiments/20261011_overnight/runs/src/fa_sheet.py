"""False-alarm episodes of the logged guard (AI side): camera + ToF obstacle mask at the onset, and context numbers."""
import json

import cv2
import numpy as np

from load import L, KV
from tof import evaluate, collisions, frame_speed
from sheets import label
from jetformer.tof_guard import ToFGuard

LOGS = ['20261010-203924', '20261010-204208', '20261010-204605']
REF = np.load('/home/jetson/jetformer/data/tof_floor_ref.npy')
OUT = '/home/jetson/jetformer_reports/overnight_20261011/runs/'


def tof_img(g, raw, erpm, gz, steer):
    """obstacle mask (rows 4..27) colored by distance + in-path points"""
    r = np.asarray(raw)[g.ROWS]
    dd = (r & 0x1FFF).astype(np.float32); dd[dd == 0x1FFF] = 0; dd *= 0.001
    conf = r >> 13
    m = g.mask(dd, conf)
    full = (np.asarray(raw) & 0x1FFF).astype(np.float32) * 0.001
    base = np.clip(full / 2.5 * 255, 0, 255).astype(np.uint8)
    im = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    sub = im[4:28]
    sub[m] = (0, 0, 255)
    im[4:28] = sub
    return cv2.resize(im, (216, 168), interpolation=cv2.INTER_NEAREST)


def main():
    g = ToFGuard(REF, center=1443.0, neutral=1498.0)
    rows, ctx = [], []
    for log in LOGS:
        d = L(log, tof=True)
        coll = collisions(d)
        ev = evaluate(d, d.guard >= 1, coll=coll, all_ev=d.events(2.0))
        v = frame_speed(d)
        ti = np.clip(np.searchsorted(d.tt, d.at, side='right') - 1, 0, len(d.tt) - 1)
        for a, b in ev['fa']:
            k = a + int(np.argmin(d.ttc[a:b + 1]))
            cam = cv2.imread(d.cf[min(k, len(d.cf) - 1)])
            gz = float(np.interp(d.at[k], d.it, d.gyr[:, 2]))
            f = g.features(d.depth[ti[k]], d.erpm_at(d.at[k]), gz, float(d.mout[k, 0]))
            info = {'log': log, 'name': d.name, 't_on': round(float(d.at[a]), 2), 'dur_s': round(float(d.at[b] - d.at[a] + 1 / 30), 2),
                    't_min': round(float(d.at[k]), 2), 'ttc_min': round(float(d.ttc[k]), 2), 'x_free': round(f['x_free'], 2),
                    'v': round(float(v[k]), 2), 'steer_us': round(float(d.mout[k, 0] - 1443)), 'gz': round(gz),
                    'n_obs': f['n_obs'], 'nL': f['nL'], 'nR': f['nR'], 'brake': bool((d.guard[a:b + 1] == 2).any())}
            ctx.append(info)
            timg = tof_img(g, d.depth[ti[k]], d.erpm_at(d.at[k]), gz, float(d.mout[k, 0]))
            cam = label(cam, ['%s %.2fs FA' % (d.name, d.at[k]), 'ttc %.2f x %.2fm v %.1f' % (d.ttc[k], f['x_free'], v[k]),
                              'st %+d gz %+d' % (info['steer_us'], info['gz'])])
            pad = np.zeros((224, 216, 3), np.uint8); pad[28:196] = timg
            rows.append(np.hstack([cam, pad]))
    # 2 per row
    while len(rows) % 3:
        rows.append(np.zeros_like(rows[0]))
    sheet = np.vstack([np.hstack(rows[i:i + 3]) for i in range(0, len(rows), 3)])
    cv2.imwrite(OUT + 'fig/tof_false_alarms.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    json.dump(ctx, open(OUT + 'src/false_alarms.json', 'w'), indent=1)
    for c in ctx:
        print(c)


if __name__ == '__main__':
    main()
