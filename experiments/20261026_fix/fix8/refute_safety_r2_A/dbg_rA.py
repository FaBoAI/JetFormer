import glob
import numpy as np
for day in ('20261010', '20261023', '20261024'):
    for f in sorted(glob.glob('/home/jetson/jetformer/data/raw/%s-*/*/AI_CMD.npz' % day)):
        try:
            z = np.load(f, allow_pickle=False)
        except Exception:
            continue
        if 'assist' not in z.files:
            continue
        a = z['assist'].astype(int)
        m = (a == 2) & (~z['hold'].astype(bool))
        if m.sum():
            v, c = np.unique(np.round(z['out'][m, 1]), return_counts=True)
            print(f.split('/')[-2], m.sum(), dict(zip(v.tolist(), c.tolist())))
