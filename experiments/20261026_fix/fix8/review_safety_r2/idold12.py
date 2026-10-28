"""全部 OFF(rec_back_guard 0・rec_hold_keep 0・launch_no_kick 0・発進の前進 0・天井 0・自動合わせ 0・区間の上限 0)の今の drive._loop が、
前の版とビット単位で同じかを、本物の drive._loop の閉じたループ(h12)で照らす。前の版 = git の HEAD(9da6d0d。push 済み)・
tests/fixtures の写し(fix7_safety_r2_before = r1・fix7_safety_before = f7b・fix6_safety_before・fix5_safety_before・fix3_before・
round2_before = 後ろの見張りを入れる前・launch_before = 発進の前進を入れる前)。比べるもの: PCA9685 に書いた値(時刻・舵・スロットル)の
列・記録の行(両方にある欄。infer_ms のほか)・出来事。場面: 場面 rev / react × ESC の模型 18 × 乱れ 12(なし・カメラの止まり 5 / 16
フレーム・人側 0.5 秒・車両の値の止まり 0.16 秒・eRPM の跳ね・段差の衝撃を最初のリカバリーの brake / reverse / coast に)。
    python -I -B idold12.py <i/n> → out/idold12_<i>of<n>.jsonl(1 組 1 行。流し直すと済んだ組は飛ばす)
"""
import hashlib
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RS82_TAG', 'idold_' + '_'.join(sys.argv[1:2]).replace('/', 'of'))
sys.path.insert(0, str(HERE))
import h12  # noqa: E402
import exp12 as E  # noqa: E402

h11 = h12.h11
ROOT = h12.ROOT
FIX = ROOT / 'tests' / 'fixtures'
VERS = ('HEAD', 'r1', 'f7b', 'f6b', 'f5b', 'fix3_before', 'round2_before', 'launch_before')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def load_dir(tag, files):
    """files = {'drive.py': bytes, 'assist.py': bytes} → (drive のモジュール, assist のモジュール)(h11.load と同じ作り方)"""
    if tag in h11._MODS:
        return h11._MODS[tag]
    aspec = importlib.util.spec_from_loader('jetformer._h12_assist_%s' % tag, loader=None)
    amod = importlib.util.module_from_spec(aspec)
    exec(compile(files['assist.py'].decode('utf-8'), '<%s assist.py>' % tag, 'exec'), amod.__dict__)
    spec = importlib.util.spec_from_loader('jetformer._h12_drive_%s' % tag, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile(files['drive.py'].decode('utf-8'), '<%s drive.py>' % tag, 'exec'), mod.__dict__)
    mod.assist_mod = amod
    h11._MODS[tag] = (mod, amod)
    return mod, amod


def version_files(v):
    if v == 'HEAD':
        return {fn: subprocess.run(['git', '-C', str(ROOT), 'show', 'HEAD:jetformer/' + fn], check=True,
                                   capture_output=True).stdout for fn in ('drive.py', 'assist.py')}
    dn = {'r1': 'fix7_safety_r2_before', 'f7b': 'fix7_safety_before', 'f6b': 'fix6_safety_before',
          'f5b': 'fix5_safety_before'}.get(v, v)
    return {fn: (FIX / dn / fn).read_bytes() for fn in ('drive.py', 'assist.py')}


def compare(ra, rb):
    """PCA9685 の書き込み・記録の行(両方にある欄)・出来事 → (同じか, 最初の違い, 欄の違い)"""
    sa, reca, pa = ra[:3]
    sb, recb, pb = rb[:3]
    if pa.w != pb.w:
        i = next((k for k, (x, y) in enumerate(zip(pa.w, pb.w)) if x != y), min(len(pa.w), len(pb.w)))
        return False, ('pca', i, pa.w[i:i + 1], pb.w[i:i + 1]), None
    if len(reca.rows) != len(recb.rows):
        return False, ('rows_len', len(reca.rows), len(recb.rows)), None
    keys_a = set(reca.rows[0][1]) if reca.rows else set()
    keys_b = set(recb.rows[0][1]) if recb.rows else set()
    common = keys_a & keys_b
    for (t0, r0), (t1, r1) in zip(reca.rows, recb.rows):
        if t0 != t1:
            return False, ('row_t', (t0 - h12.T0) / 1e9), None
        for k in common:
            if k != 'infer_ms' and not h11._same(r0[k], r1[k]):
                return False, ('row', (t0 - h12.T0) / 1e9, k), None
    ea = [(e[0], dict(e[1])) for e in reca.events]
    eb = [(e[0], dict(e[1])) for e in recb.events]
    same_ev = len(ea) == len(eb) and all(x[0] == y[0] and h11._same(x[1], y[1]) for x, y in zip(ea, eb))
    return same_ev, None if same_ev else ('ev', len(ea), len(eb), [x[0] for x in ea][:5], [y[0] for y in eb][:5]), \
        {'only_now': sorted(keys_a - keys_b), 'only_old': sorted(keys_b - keys_a)}


def perts(tl):
    out = [('none', [])]
    for ev in ('brake', 'reverse', 'coast'):
        if ev not in tl:
            continue
        c = E.cam_of(tl[ev])
        out += [(ev, [('drop', c, 5)]), (ev, [('drop', c, 16)])]
    if 'brake' in tl:
        c = E.cam_of(tl['brake'])
        out += [('brake', [('human', c, 0.5, 'neutral')]), ('brake', [('spike', c, 3.0)])]
    if 'reverse' in tl:
        c = E.cam_of(tl['reverse'])
        out += [('reverse', [('human', c, 0.5, 'neutral')]), ('reverse', [('veh', c, 0.16)]), ('reverse', [('glitch', c, 1322)])]
    return out


def main(part='0/1'):
    h11.check_now()
    pi, pn = (int(x) for x in part.split('/'))
    vers = {}
    for v in VERS:
        f = version_files(v)
        vers[v] = {fn: sha(b) for fn, b in f.items()}
        load_dir(v, f)
    jl = E.JL(HERE / 'out' / ('idold12_%s.jsonl' % part.replace('/', 'of')))
    if not jl.bases:
        jl.add({'t': 'base', 'sd': {'versions': True}, 'versions': vers, 'now': h11.check_now()})
    t_s = time.time()
    n = 0
    try:
        jobs = list(itertools.product(('rev', 'react'), E.ESCS))
        only = os.environ.get('RS82_ESC')     # 10/28 10:0x: 短めに(ESC の模型を選ぶ)
        if only:
            jobs = [j for j in jobs if j[1] in only.split(',')]
        for ji, (base, esc) in enumerate(jobs):
            if ji % pn != pi:
                continue
            bsd = {'base': base, 'esc': esc, 'form': 'kai1'}
            p = h12.OFF(E.params_of(bsd))
            tl = None
            for ev, pt in perts(E.timeline(h12.run(h12.ON(E.params_of(bsd)), E.build(bsd))[3])):
                sd = dict(bsd, pert=pt, dur=10.0)
                if E.skey(sd) in jl.done:
                    continue
                rn = h12.run(p, E.build(sd))
                row = {'sd': sd, 'ev': ev, 'cmp': {}}
                for v in VERS:
                    ro = h12.run(p, E.build(sd), ver=v)
                    ok, why, kd = compare(rn, ro)
                    row['cmp'][v] = {'same': ok, 'why': str(why)[:240] if why else None, 'keys': kd}
                jl.add({'t': 'row', 'sd': sd, 'row': row})
                n += 1
                bad = [v for v in VERS if not row['cmp'][v]['same']]
                print(n, round(time.time() - t_s), base, esc, ev, pt, 'differs:', bad,
                      [row['cmp'][v]['why'] for v in bad][:2], flush=True)
    finally:
        h12.cleanup()
    print('done idold', part, 'new', n, round(time.time() - t_s), 's')


if __name__ == '__main__':
    main(*sys.argv[1:2])
