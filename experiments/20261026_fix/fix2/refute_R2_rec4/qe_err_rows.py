"""R2-rec-4 の反証: per_log に 'error' の行があると quick_eval.py が KeyError で止まるかを、合成の記録で確かめる。
本物の data/raw・data/labels・~/jetformer_reports には書かない(tmp_rec4 の下だけ)。quick_eval.py は読むだけ。
場合: A = よい 1 本 + 全部 0 バイト 1 本 + よい 1 本(壊れた本は真ん中)。B = よい 1 本 + VEHICLE だけ 0 バイトの 1 本。"""
import importlib.util, io, json, shutil, sys, traceback
from contextlib import redirect_stdout
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
sys.path.insert(0, str(JF / 'tools'))
import section_scorecard as ss  # noqa: E402
from test_standing_3lap import write_standing_log  # noqa: E402

BASE = JF / 'experiments/20261026_fix/fix2/tmp_rec4'


def load_qe(raw, tmp):
    spec = importlib.util.spec_from_file_location('quick_eval_t', str(JF / 'experiments/20261024_tune/quick_eval.py'))
    qe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qe)
    qe.RAW = raw
    real = qe.Path

    def fp(p, *a):
        s = str(p)
        if s.startswith('/home/jetson/jetformer_reports'):
            s = str(tmp / 'reports') + s[len('/home/jetson/jetformer_reports'):]
        return real(s, *a)
    qe.Path = fp
    return qe


def case(name, day, build_bad):
    tmp = BASE / name
    if tmp.exists():
        shutil.rmtree(tmp)
    raw = tmp / 'raw'
    raw.mkdir(parents=True)
    g1, bad, g2 = day + '-100000', day + '-101500', day + '-103000'
    write_standing_log(raw, g1, description='ai_no=1 cand=EXP tune=1-1', params={'throttle_gain': 0.9})
    build_bad(raw, bad)
    write_standing_log(raw, g2, description='ai_no=3 cand=EXP tune=1-3', params={'throttle_gain': 0.9})
    out = tmp / 'reports' / ('bench_%s_tune' % day) / 'since_0000'
    with redirect_stdout(io.StringIO()):
        ss.main(['--day', day, '--raw', str(raw), '--labels', str(tmp / 'labels'), '--out', str(out / 'nightly' / 'scorecard'), '--boot', '20'])
    sc = json.loads((out / 'nightly' / 'scorecard' / 'scorecard.json').read_text(encoding='utf-8'))
    print('=== 場合', name)
    print('per_log の error:', {k: v.get('error') for k, v in sc['per_log'].items()})
    qe = load_qe(raw, tmp)
    sys.argv = ['quick_eval.py', '0000', '--day=' + day, '--no-run']
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            qe.main()
        print('quick_eval: 落ちない')
    except Exception as e:
        print('quick_eval: 落ちた ->', type(e).__name__, e)
        traceback.print_exc(limit=-1, file=sys.stdout)
    print('出た行 %d:' % len(buf.getvalue().splitlines()))
    print(buf.getvalue())


def bad_all_zero(raw, bad):
    d = raw / bad
    (d / (bad + '-0001')).mkdir(parents=True)
    (d / 'log.json').write_text(json.dumps({'description': 'ai_no=2 cand=EXP tune=1-2', 'mode_id': 5,
                                            'drive': {'run': 'runX', 'dry_run': False, 'params': {'throttle_gain': 0.9}}}), encoding='utf-8')
    for nm in ('AI_CMD', 'VEHICLE', 'IMU', 'CAM_FRONT', 'TOF_FRONT'):
        (d / (bad + '-0001') / (nm + '.npz')).write_bytes(b'')


def bad_vehicle_zero(raw, bad):
    write_standing_log(raw, bad, description='ai_no=2 cand=EXP tune=1-2', params={'throttle_gain': 0.9})
    for p in (raw / bad).glob('*/VEHICLE.npz'):
        p.write_bytes(b'')


case('A_all_zero', '20001103', bad_all_zero)
case('B_vehicle_zero', '20001104', bad_vehicle_zero)
