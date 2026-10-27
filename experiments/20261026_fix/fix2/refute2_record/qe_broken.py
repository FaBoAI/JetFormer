"""反証 2 回目・record: 電源断で 0 バイトの npz だけになった記録が 1 本あると、quick_eval.py の表が全部出ない(KeyError)ことを確かめる。
本物の data/raw・data/labels・~/jetformer_reports には書かない(tmp_record の下だけ)。quick_eval.py は読むだけ(写さない。
モジュールの RAW・Path を差し替えて動かす)。"""
import importlib.util
import io
import json
import shutil
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
sys.path.insert(0, str(JF / 'tools'))
import section_scorecard as ss  # noqa: E402
from test_standing_3lap import write_standing_log  # noqa: E402

TMP = JF / 'experiments/20261026_fix/fix2/tmp_record/qe'
if TMP.exists():
    shutil.rmtree(TMP)
raw = TMP / 'raw'
raw.mkdir(parents=True)
DAY = '20001103'
good = DAY + '-100000'
bad = DAY + '-101500'
write_standing_log(raw, good, description='ai_no=2 bench=real_bench_20261023 day=2000-11-03 block=2 cand=EXP tune=1-2', params={'throttle_gain': 0.9})
# 電源断: log.json は走行開始で書けている。シーンの npz は保存の途中で 0 バイト(20261024-162305 の 0007・0008 と同じ形)
d = raw / bad
(d / (bad + '-0001')).mkdir(parents=True)
(d / 'log.json').write_text(json.dumps({'description': 'ai_no=3 bench=real_bench_20261023 day=2000-11-03 block=3 cand=EXP tune=1-3',
                                        'mode_id': 5, 'drive': {'run': 'runX', 'dry_run': False,
                                                                'params': {'throttle_gain': 0.9}}}), encoding='utf-8')
for nm in ('AI_CMD', 'VEHICLE', 'IMU', 'CAM_FRONT', 'TOF_FRONT'):
    (d / (bad + '-0001') / (nm + '.npz')).write_bytes(b'')

# bench_nightly の段 2 と同じ: section_scorecard --day
out = TMP / 'reports' / ('bench_%s_tune' % DAY) / 'since_0000'
ss.main(['--day', DAY, '--raw', str(raw), '--labels', str(TMP / 'labels'), '--out', str(out / 'nightly' / 'scorecard'), '--boot', '50'])
sc = json.loads((out / 'nightly' / 'scorecard' / 'scorecard.json').read_text(encoding='utf-8'))
print('per_log:', {k: sorted(v)[:6] if 'error' not in v else {'error': v['error']} for k, v in sc['per_log'].items()})

spec = importlib.util.spec_from_file_location('quick_eval', str(JF / 'experiments/20261024_tune/quick_eval.py'))
qe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qe)
qe.RAW = raw
real_path = qe.Path


def fake_path(p, *a):
    s = str(p)
    if s.startswith('/home/jetson/jetformer_reports'):
        s = str(TMP / 'reports') + s[len('/home/jetson/jetformer_reports'):]
    return real_path(s, *a)


qe.Path = fake_path
sys.argv = ['quick_eval.py', '0000', '--day=' + DAY, '--no-run']
buf = io.StringIO()
try:
    with redirect_stdout(buf):
        qe.main()
    print('quick_eval: ok')
except Exception:
    print('quick_eval: 落ちた')
    traceback.print_exc(limit=2)
print('出た表(%d 行):' % len(buf.getvalue().splitlines()))
print(buf.getvalue())
