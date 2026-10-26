import importlib.util, json, math, sys
from pathlib import Path
import numpy as np
C3 = Path('/home/jetson/jetformer/experiments/20261025_train_v2/c3/desk_v2.py')
EV = Path('/home/jetson/jetformer/experiments/20261025_train_v2/eval_v2.py')
DESK = Path('/home/jetson/jetformer_reports/train_v2_20261025/desk')
P3 = json.loads(Path('/home/jetson/jetformer_reports/fix_20261026/protocol_v3.json').read_text())
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent)); spec.loader.exec_module(m); return m
D = load(C3, 'd'); E = load(EV, 'e')
S = np.load(DESK / 'sets_v2.npz')
st, ss, sd = S['straight'].astype(bool), S['sess'], S['stand'].astype(bool)
pb = np.load(DESK / 'pred' / ('%s_tab.npz' % D.B_RUN))['pred'].astype(np.float64)
refb = (1441.0, 1496.0)
five = [run for rid, run, *_ in E.FIVE]
rows = {}
for f in sorted((DESK / 'pred').glob('*_tab.npz')):
    run = f.name[:-8]
    pc = np.load(f)['pred'].astype(np.float64)
    a = D.thr_vs_b(pc, pb, refb, refb, st & (ss == 0), 0.9, 0.9)
    b = D.thr_vs_b(pc, pb, refb, refb, st & (ss == 1), 0.9, 0.9)
    t0 = D.car_out(pc[sd][:, 0], refb, 0.9)[:, 1]
    back = float(np.mean(t0 > refb[1] + 30.0))
    fails = [n for n, bad in (('st1010', abs(a) > 10.0), ('st1023', abs(b) > 12.138), ('back', back > 0.0640)) if bad]
    rows[run] = {'st1010': round(a, 2), 'st1023': round(b, 2), 'back': round(back, 4), 'fails_new_desk_gates': fails, 'five': run in five}
print(json.dumps(rows, ensure_ascii=False, indent=0))
Path(sys.argv[1]).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
