"""Mean camera image (300 frames spread over each log) + vertical brightness profile, to compare camera pitch
between the afternoon runs (006/007) and the evening runs (clean data 20:09-20:26, ai_run_009/010/101)."""
import os, sys, json
import numpy as np, cv2
LOGS = ['20261010-173445', '20261010-192141', '20261010-201626', '20261010-204208', '20261010-204605']
OUT = '/home/jetson/jetformer_reports/overnight_20261011/runs/fig/'
tiles, prof = [], {}
for log in LOGS:
    root = '/home/jetson/jetformer/data/raw/' + log
    sc = sorted(s for s in os.listdir(root) if s.startswith(log + '-'))
    files = []
    for s in sc:
        p = os.path.join(root, s, 'CAM_FRONT.npz')
        if os.path.exists(p):
            files += [os.path.join(root, s, str(x)) for x in np.load(p, allow_pickle=True)['files']]
    idx = np.linspace(0, len(files) - 1, 300).astype(int)
    acc = np.zeros((224, 224, 3), np.float64); n = 0
    for i in idx:
        im = cv2.imread(files[i])
        if im is not None:
            acc += im; n += 1
    m = (acc / n).astype(np.uint8)
    g = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY).astype(float)
    prof[log] = g[:, 60:164].mean(axis=1).round(1).tolist()
    t = cv2.resize(m, (336, 336), interpolation=cv2.INTER_NEAREST)
    cv2.putText(t, log[9:], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for y in (56, 112, 168):
        cv2.line(t, (0, int(y * 1.5)), (335, int(y * 1.5)), (0, 0, 255), 1)
    tiles.append(t)
cv2.imwrite(OUT + 'camera_mean_images.jpg', np.hstack(tiles))
json.dump(prof, open(os.path.dirname(os.path.abspath(__file__)) + '/meanimg_profile.json', 'w'))
for log in LOGS:
    p = np.array(prof[log]); d = np.diff(p)
    print(log, 'rows of strongest brightness drop (top->down) 30-150:', (np.argsort(d[30:150])[:3] + 30).tolist())
