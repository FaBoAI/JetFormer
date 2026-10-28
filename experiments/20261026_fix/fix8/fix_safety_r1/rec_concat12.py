"""fix8 係 safety 直す 8-1 回目(rs8-1-real-record-backing-missed): 本物の記録(data/raw の 10/10・10/23・10/24。読むだけ)を、
1 本の場面をつないで流す形と、場面ごとに作り直す形(fix7 の再生の形)で流し、今と fix7 の前(tests/fixtures/fix7_safety_before)の
出す値の違いを数える。流し方の部品は tests/test_rec_fix8_safety_r1.py(試験と同じ関数)。
  (A) Assist だけ(fix7 の replay_all9 と同じ形): --assist
  (D) 本物の drive._loop(両方 ON / 両方 OFF × 今 / fix7 の前): --drive [--logs a,b]
  (A) で、今と 7-1 回目のあと(tests/fixtures/fix7_safety_r2_before = 7-2 回目 s4-c の前)を比べる: --r1
出力 out/rec_concat12_assist.json・out/rec_concat12_drive.json・out/rec_concat12_r1.json。装置・GPU・API には触れない
    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -B experiments/20261026_fix/fix8/fix_safety_r1/rec_concat12.py --assist
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))
import test_rec_fix8_safety_r1 as T  # noqa: E402

TMP = ROOT / 'experiments' / '20261026_fix' / 'fix8' / 'tmp_safety_r1' / 'rec'
NOW = {'drive.py': '3f2e31a22ecb9abdd9c8f84b3f787d075ef636b6d5679bad5c0fc28aee18388e',
       'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'}


def shas():
    got = {f: hashlib.sha256((ROOT / 'jetformer' / f).read_bytes()).hexdigest() for f in NOW}
    assert got == NOW, got
    return got


def run_assist():
    out = {'sha256': shas(), 'started': time.strftime('%F %T')}
    t_s = time.time()
    for how in ('scene', 'concat'):
        tot = T.replay_all_assist(how)
        tot['elapsed_s'] = round(time.time() - t_s, 1)
        out[how] = tot
        print(how, {k: v for k, v in tot.items() if k not in ('diff_on', 'backing_bwg', 'on_vs_off_at_diff_on')}, flush=True)
        for x in tot['diff_on']:
            print('   diff', x, flush=True)
        print('   backing', tot['backing_bwg'], flush=True)
    for name, fn, how in (('check_concat', T.check_concat, 'concat'), ('check_scene', T.check_scene, 'scene')):
        try:
            fn(out[how])
            out[name] = 'ok'
        except AssertionError as e:
            out[name] = 'NG %r' % (e,)
        print(name, out[name], flush=True)
    (HERE / 'out' / 'rec_concat12_assist.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


def run_drive(logs):
    out = {'sha256': shas(), 'started': time.strftime('%F %T'), 'logs': {}}
    TMP.mkdir(parents=True, exist_ok=True)
    floor = T.floor_ref()
    t_s = time.time()
    try:
        for log in logs:
            r = T.drive_compare(log, TMP, floor)
            r['elapsed_s'] = round(time.time() - t_s, 1)
            out['logs'][log] = r
            print(log, {k: v for k, v in r.items() if k not in ('diff_on', 'problems')}, flush=True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    diff = {k: v['diff_on_s'] for k, v in out['logs'].items() if v.get('diff_on_s')}
    out['summary'] = {'logs': len(out['logs']), 'skipped': [k for k, v in out['logs'].items() if 'skipped' in v],
                      'diff_on': diff,
                      'off_all_same': all(v.get('off_same_rows') and v.get('off_same_pca') and v.get('off_same_events')
                                          for v in out['logs'].values() if 'skipped' not in v),
                      'on_eq_off_at_diff': {k: v['on_eq_off_at_diff'] for k, v in out['logs'].items() if v.get('diff_on_s')},
                      'elapsed_s': round(time.time() - t_s, 1)}
    print('summary', out['summary'], flush=True)
    name = 'rec_concat12_drive.json' if len(logs) > 1 else 'rec_concat12_drive_%s.json' % logs[0]
    (HERE / 'out' / name).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


R1_SHA = '6a8950d9963bb4a663dd540cb4b819f13f22f9da956a56a657c98a66ea717e3f'      # fix7 の 7-1 回目のあと(7-2 回目の前)の assist.py


def load_r1():
    """tests/fixtures/fix7_safety_r2_before/assist.py(7-2 回目 s4-c の前 = 7-1 回目 s4-a・s4-b のあと)"""
    import importlib.util
    fp = ROOT / 'tests' / 'fixtures' / 'fix7_safety_r2_before' / 'assist.py'
    assert hashlib.sha256(fp.read_bytes()).hexdigest() == R1_SHA
    spec = importlib.util.spec_from_loader('jetformer._rc12_assist_r1', loader=None)
    m = importlib.util.module_from_spec(spec)
    exec(compile(fp.read_text(encoding='utf-8'), str(fp), 'exec'), m.__dict__)
    return m


def run_r1():
    """(A) のつないだ流し方で、今の見張り ON と 7-1 回目のあと(r1。s4-c の前)の見張り ON を比べる(fix7/fix_safety_r2.md の
    「直す前の写しと今に流した: 1 フレームも違わない」の照らし)。場面ごとも"""
    r1 = load_r1()
    spec = {'now_on': ('now', True, True), 'r1_on': ('r1', True, True), 'now_off': ('now', False, False), 'r1_off': ('r1', False, False)}
    out = {'sha256': shas(), 'r1_sha256': R1_SHA}
    for how in ('scene', 'concat'):
        tot = {'units': 0, 'frames': 0, 'diff_on': [], 'diff_off': 0}
        for log in T.raw_logs():
            for u in T.units(log, how):
                cat, meta, sk = T.load_unit(log, u)
                if cat is None:
                    continue
                r = T.assist_replay(cat, meta, mods={'now': T.A, 'r1': r1}, spec=spec)
                t0 = T.log_t0_us(log) or int(r['t_us'][0])
                tot['units'] += 1
                tot['frames'] += len(r['t_us'])
                tot['diff_on'] += [{'unit': u[0].name if how == 'scene' else log, 's': round((int(r['t_us'][i]) - t0) / 1e6, 3)}
                                   for i in T.diff_idx(r, 'now_on', 'r1_on')]
                tot['diff_off'] += len(T.diff_idx(r, 'now_off', 'r1_off'))
        out[how] = tot
        print('r1', how, tot, flush=True)
    (HERE / 'out' / 'rec_concat12_r1.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assist', action='store_true')
    ap.add_argument('--drive', action='store_true')
    ap.add_argument('--r1', action='store_true')
    ap.add_argument('--logs', default=None)
    a = ap.parse_args()
    if a.assist:
        run_assist()
    if a.r1:
        run_r1()
    if a.drive:
        run_drive(a.logs.split(',') if a.logs else T.raw_logs())


if __name__ == '__main__':
    main()
