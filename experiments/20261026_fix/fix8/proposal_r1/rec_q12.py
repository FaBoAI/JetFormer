"""fix8 係 safety 直す 8-1 回目: 案 qA・qB・qC を本物の記録(data/raw の 10/10・10/23・10/24。読むだけ)に開いたループで流し、
今の見張り ON と出す値が違うフレームを数える(車は記録のとおり動いたまま = 違いが出たあとの車の動きは比べられない。どこで・何が
違うかを見る)。流し方は tests/test_rec_fix8_safety_r1.py と同じ(1 本の場面をつなぐ)。
  (A) Assist だけ(見張り ON・中立止めで reset・書いた値を形ごとの EscMayReverse に入れて esc_rev)。全部の記録
  (D) 本物の drive._loop(両方 ON)で、記録の中の 'brake' で後ろへ動いた 2 本(10/23 205458 = No.6・10/24 160847)
出力 out/rec_q12.json
    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -B experiments/20261026_fix/fix8/proposal_r1/rec_q12.py
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
os.environ.setdefault('R1_TAG', 'rec_%d' % os.getpid())
import cl_r1  # noqa: E402
import test_rec_fix8_safety_r1 as T  # noqa: E402

PROPS = ('qA', 'qB', 'qC')
TMP = ROOT / 'experiments' / '20261026_fix' / 'fix8' / 'tmp_safety_r1' / 'recq'


def assist_all():
    mods = {'now': T.A}
    spec = {'now_on': ('now', True, True), 'now_off': ('now', False, False)}
    for p in PROPS:
        mods[p] = cl_r1.load_version(p)[1]
        spec['%s_on' % p] = (p, True, True)
        spec['%s_off' % p] = (p, False, False)
    tot = {'logs': 0, 'frames': 0, 'diff': {}, 'off_diff': {}}
    for log in T.raw_logs():
        cat, meta, sk = T.load_unit(log, T.scene_dirs(log))
        if cat is None:
            continue
        r = T.assist_replay(cat, meta, mods=mods, spec=spec)
        t0 = T.log_t0_us(log) or int(r['t_us'][0])
        tot['logs'] += 1
        tot['frames'] += len(r['t_us'])
        for p in PROPS:
            d = T.diff_idx(r, '%s_on' % p, 'now_on')
            if d:
                tot['diff'].setdefault(p, []).append({
                    'log': log, 'n': len(d), 'first_s': round((int(r['t_us'][d[0]]) - t0) / 1e6, 3),
                    'last_s': round((int(r['t_us'][d[-1]]) - t0) / 1e6, 3),
                    'ev_now': [(round((int(r['t_us'][k]) - t0) / 1e6, 3), e) for k, e in r['ev']['now_on']],
                    'ev_prop': [(round((int(r['t_us'][k]) - t0) / 1e6, 3), e) for k, e in r['ev']['%s_on' % p]],
                    'frames': [{'s': round((int(r['t_us'][k]) - t0) / 1e6, 3), 'erpm': r['erpm'][k],
                                'now': r['out']['now_on'][k].tolist(), 'prop': r['out']['%s_on' % p][k].tolist(),
                                'off': r['out']['now_off'][k].tolist()} for k in d[:40]]})
            od = T.diff_idx(r, '%s_off' % p, 'now_off')
            if od:
                tot['off_diff'].setdefault(p, []).append({'log': log, 'n': len(od)})
    return tot


def drive_two():
    out = {}
    TMP.mkdir(parents=True, exist_ok=True)
    floor = T.floor_ref()
    try:
        for log in (T.LOG_1023, T.LOG):
            meta, rec, _ = T.load_rec_drive(log)
            t0 = T.log_t0_us(log)
            rows = {}
            for ver in ('now',) + PROPS:
                mod = T.drive if ver == 'now' else cl_r1.load_version(ver)[0]
                for g in (1, 0):
                    rr = T.drive_replay(mod, rec, T.drive_params(meta, g, g, mod), TMP, floor)
                    rows[(ver, g)] = (T.drive_rows(rr), [(e[1].get('what'), e[1].get('phase'))
                                                         for e in rr[1].events if e[0] == 'recovery_guard'], rr[2].w)
            ent = {}
            for p in PROPS:
                a, b = rows[(p, 1)][0], rows[('now', 1)][0]
                d = [tn for tn in sorted(set(a) | set(b)) if a.get(tn) != b.get(tn)]
                ent[p] = {'n': len(d), 'guards': rows[(p, 1)][1],
                          'off_same': rows[(p, 0)][0] == rows[('now', 0)][0] and rows[(p, 0)][2] == rows[('now', 0)][2],
                          'frames': [{'s': round((tn / 1000 - t0) / 1e6, 3), 'now': b.get(tn), 'prop': a.get(tn),
                                      'off': rows[('now', 0)][0].get(tn)} for tn in d]}
            ent['guards_now'] = rows[('now', 1)][1]
            out[log] = ent
            print(log, {p: (ent[p]['n'], ent[p]['off_same'], ent[p]['guards']) for p in PROPS}, flush=True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
        cl_r1.cleanup()
    return out


def main():
    cl_r1.check_now()
    t_s = time.time()
    res = {'assist': assist_all()}
    print('assist', {p: [(x['log'], x['n'], x['first_s'], x['last_s']) for x in v] for p, v in res['assist']['diff'].items()},
          'off_diff', res['assist']['off_diff'], 'frames', res['assist']['frames'], flush=True)
    res['drive'] = drive_two()
    res['elapsed_s'] = round(time.time() - t_s, 1)
    (HERE / 'out' / 'rec_q12.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print('done %.0fs' % (time.time() - t_s))


if __name__ == '__main__':
    main()
