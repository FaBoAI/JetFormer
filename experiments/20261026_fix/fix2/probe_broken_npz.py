#!/usr/bin/env python3
"""0 バイト・壊れた npz の記録で tools/ の台本が落ちるかを確かめる(2026-10-26 fix2、係 code)。

本物の記録 20261024-185926(3 シーン・約 5 MB)の npz・json だけを tmp に写し(画像は写さない)、名前を 2000 年に替えて、
壊した形を作る(下の VARIANTS)。台本は全部 --raw・--labels・--out を tmp にして回す。本物の data/raw・data/labels・
~/jetformer_reports には書かない。GPU・torch は使わない。終わったら tmp を消す(--keep で残す)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -I experiments/20261026_fix/fix2/probe_broken_npz.py --tag before

結果: experiments/20261026_fix/fix2/out/probe_broken_npz_<tag>.json(台本 × 形 → ok / 落ちた(例外の種類)・警告の有無)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

JF = Path('/home/jetson/jetformer')
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
SRC = JF / 'data' / 'raw' / '20261024-185926'
HERE = Path(__file__).resolve().parent
DAY = '20001024'

# 形の名前: (ログの名前, 説明)
VARIANTS = {
    'good': ('20001024-100000', 'そのままの写し'),
    'zero_scene': ('20001024-100001', '写し + 4 番目のシーン(npz 5 つ・scene.json・thumb.jpg が 0 バイト。20261024-162305-0007 と同じ形)'),
    'trunc_ai': ('20001024-100002', '2 番目のシーンの AI_CMD.npz を半分で切った(BadZipFile)'),
    'crc_veh': ('20001024-100003', '2 番目のシーンの VEHICLE.npz の中身を 0 で潰した(Bad CRC-32。開けるが読むと落ちる)'),
    'zero_logjson': ('20001024-100004', 'log.json が 0 バイト(20261023-204825 と同じ形)'),
    'zero_imu_tof': ('20001024-100005', '2 番目のシーンの IMU.npz・TOF_FRONT.npz・CAM_FRONT.npz が 0 バイト'),
    'zero_scenejson': ('20001024-100006', '2 番目のシーンの scene.json が 0 バイト(npz は読める)'),
}


def build_raw(raw):
    raw.mkdir(parents=True)
    src_log = SRC.name
    for key, (log, _) in VARIANTS.items():
        dst = raw / log
        dst.mkdir()
        meta = json.loads((SRC / 'log.json').read_text(encoding='utf-8'))
        meta['logfile'] = log
        meta['description'] = meta.get('description', '').replace('day=2026-10-24', 'day=2000-10-24')
        (dst / 'log.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
        for s in sorted(d for d in SRC.iterdir() if d.is_dir()):
            ds = dst / s.name.replace(src_log, log)
            ds.mkdir()
            for f in s.iterdir():
                if f.is_file():
                    shutil.copy2(f, ds / f.name)
        s2 = dst / (log + '-0002')
        if key == 'zero_scene':
            s4 = dst / (log + '-0004')
            s4.mkdir()
            for n in ('AI_CMD', 'CAM_FRONT', 'IMU', 'TOF_FRONT', 'VEHICLE'):
                (s4 / (n + '.npz')).write_bytes(b'')
            (s4 / 'scene.json').write_bytes(b'')
            (s4 / 'thumb.jpg').write_bytes(b'')
            (s4 / 'CAM_FRONT').mkdir()
        elif key == 'trunc_ai':
            b = (s2 / 'AI_CMD.npz').read_bytes()
            (s2 / 'AI_CMD.npz').write_bytes(b[:len(b) // 2])
        elif key == 'crc_veh':
            b = bytearray((s2 / 'VEHICLE.npz').read_bytes())
            n = len(b)
            b[n // 4: n // 4 + 256] = b'\x00' * 256
            (s2 / 'VEHICLE.npz').write_bytes(bytes(b))
        elif key == 'zero_logjson':
            (dst / 'log.json').write_bytes(b'')
        elif key == 'zero_imu_tof':
            for n in ('IMU', 'TOF_FRONT', 'CAM_FRONT'):
                (s2 / (n + '.npz')).write_bytes(b'')
        elif key == 'zero_scenejson':
            (s2 / 'scene.json').write_bytes(b'')


def short(e):
    return '%s: %s' % (type(e).__name__, str(e).splitlines()[0][:160] if str(e) else '')


def cli(name, args, tmp, timeout=900):
    """台本を別のプロセスで回す。戻り値 {rc, tail, traceback(落ちたとき)}"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    r = subprocess.run(['nice', '-n', '10', PY, str(JF / 'tools' / name)] + [str(a) for a in args], cwd=str(JF),
                       capture_output=True, text=True, timeout=timeout, env=env)
    out = r.stdout + r.stderr
    tb = None
    if 'Traceback (most recent call last)' in out:
        tb = out[out.rindex('Traceback (most recent call last)'):][-1500:]
    return {'rc': r.returncode, 'tail': out[-1200:], 'traceback': tb}


def per_variant(fn):
    res = {}
    for key, (log, _) in VARIANTS.items():
        try:
            v = fn(log)
            res[key] = {'ok': True, 'note': v}
        except Exception as e:
            res[key] = {'ok': False, 'error': short(e), 'where': traceback.format_exc().strip().splitlines()[-3][:200]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='run')
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('--only', default=None, help='カンマ区切りの台本の名前だけ')
    a = ap.parse_args()
    tmp = HERE / 'tmp_code' / ('probe_' + a.tag)
    if tmp.exists():
        shutil.rmtree(tmp)
    raw, lbl, out = tmp / 'raw', tmp / 'labels', tmp / 'out'
    build_raw(raw)
    lbl.mkdir()
    out.mkdir()
    logs = [v[0] for v in VARIANTS.values()]
    only = set(a.only.split(',')) if a.only else None
    want = lambda n: only is None or n in only      # noqa: E731
    R = {'tag': a.tag, 'source': str(SRC), 'variants': {k: v[1] for k, v in VARIANTS.items()}, 'logs': dict(
        (k, v[0]) for k, v in VARIANTS.items()), 'results': {}}

    # ---- 別のプロセス(夜の 1 コマンドの段と同じ形)
    if want('label_events'):
        R['results']['label_events.py'] = cli('label_events.py', logs + ['--raw', raw, '--out', lbl], tmp)
        R['results']['label_events.py']['labels_made'] = sorted(p.stem for p in lbl.glob('2*.json'))
    if want('section_scorecard'):
        R['results']['section_scorecard.py'] = cli('section_scorecard.py', ['--day', DAY, '--raw', raw, '--labels', lbl,
                                                                            '--out', out / 'sc', '--boot', 50], tmp)
    if want('cap_replay'):
        R['results']['cap_replay.py'] = cli('cap_replay.py', ['--day', DAY, '--raw', raw, '--labels', lbl, '--out', out / 'cr'], tmp)
    if want('brake_table'):
        R['results']['brake_table.py'] = cli('brake_table.py', ['--day', DAY, '--raw', raw, '--labels', lbl, '--out', out / 'bt'], tmp)
    if want('de_swing'):
        R['results']['de_swing.py'] = cli('de_swing.py', ['--day', DAY, '--raw', raw, '--labels', lbl, '--out', out / 'ds'], tmp)
    if want('car_check'):
        cc = {}
        for key, (log, _) in VARIANTS.items():
            cc[key] = cli('car_check.py', [log, '--raw', raw, '--labels', lbl, '--json', out / 'cc' / (log + '.json')], tmp)
        cc['--latest'] = cli('car_check.py', ['--latest', '--day', DAY, '--raw', raw, '--labels', lbl, '--no-json'], tmp)
        R['results']['car_check.py'] = cc
    if want('bench_report'):
        R['results']['bench_report.py'] = cli('bench_report.py', ['--logs'] + logs + [
            '--raw', raw, '--labels', lbl, '--out', out / 'br', '--boot', 50, '--min-laps', 1, '--k-first', 0, '--k-later', 0], tmp)

    # ---- 同じプロセスで関数を呼ぶ(記録を直接読むもの)
    sys.path.insert(0, str(JF / 'tools'))
    sys.path.insert(0, str(JF))
    from jetformer import config
    if want('load_log'):
        import label_events as le
        R['results']['label_events.load_log'] = per_variant(lambda lg: le.load_log(lg, raw)['problems'])
    if want('tof_floor_ref'):
        import tof_floor_ref as tfr
        R['results']['tof_floor_ref.load'] = per_variant(lambda lg: tfr.load(lg, raw)[-1])
    if want('drive_report'):
        import drive_report as dr
        old = config.RAW_DIR
        config.RAW_DIR = raw
        try:
            R['results']['drive_report.analyze'] = per_variant(lambda lg: dr.analyze(lg)[0].get('problems', 'ok'))
        finally:
            config.RAW_DIR = old
    if want('course_fit'):
        import course_fit as cf
        R['results']['course_fit.LogData'] = per_variant(lambda lg: {'scenes': cf.LogData(lg, str(raw)).scenes})
    if want('course_landmarks'):
        import course_landmarks as cl
        R['results']['course_landmarks.list_frames'] = per_variant(lambda lg: len(cl.list_frames(lg, str(raw))[1]))
    if want('build_ai_dataset'):
        import build_ai_dataset as bad

        def f(lg):
            info, arr = bad.load_labels(lg, lbl, raw)
            use, sk = bad.choose_ai_scenes(lg, bad.LogLabels(lg, info, arr), raw_dir=raw)
            return {'use': use, 'skipped': sk}
        R['results']['build_ai_dataset.choose_ai_scenes'] = per_variant(f)
    if want('make_intervention_set'):
        import make_intervention_set as mis
        R['results']['make_intervention_set.load_raw'] = per_variant(lambda lg: mis.load_raw(lg, raw)[1])

    od = HERE / 'out'
    od.mkdir(exist_ok=True)
    p = od / ('probe_broken_npz_%s.json' % a.tag)
    p.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    # 要約
    for name, r in R['results'].items():
        if 'rc' in r:
            print('%-36s rc=%s %s' % (name, r['rc'], 'TRACEBACK ' + r['traceback'].strip().splitlines()[-1] if r['traceback'] else ''))
        else:
            for k, v in r.items():
                if 'rc' in v:
                    print('%-36s %-15s rc=%s %s' % (name, k, v['rc'], 'TRACEBACK ' + v['traceback'].strip().splitlines()[-1]
                                                     if v['traceback'] else ''))
                else:
                    print('%-36s %-15s %s' % (name, k, 'ok' if v['ok'] else 'FAIL ' + v['error']))
    print('結果:', p)
    if not a.keep:
        shutil.rmtree(tmp)


if __name__ == '__main__':
    main()
