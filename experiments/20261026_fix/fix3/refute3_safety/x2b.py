import json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cl3
import x2_crash_stall as X
res = {}
for dl in (0.0, 0.03, 0.06, 0.1, 0.2, 0.35, 0.5):
    for L in (0.6, 1.0, 2.0):
        r = X.one(1, 1, hit_drop=(dl, L))
        res['delay=%.2f L=%.1f' % (dl, L)] = r
        print('delay=%.2f L=%.1f' % (dl, L), {k: r[k] for k in ('first_rev_after_s', 'front_push_after_s', 'giveup_rows', 'guard_events')}, r['phases_from_ref'][:3])
(Path(__file__).resolve().parent / 'out' / 'x2b.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
shutil.rmtree(cl3.TMP, ignore_errors=True)
