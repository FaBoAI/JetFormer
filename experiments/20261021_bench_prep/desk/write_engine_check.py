#!/usr/bin/env python3
"""段 Desk: engine_load_check.py の結果(logs/engine_check_<id>.json)→ data/runs/<学習結果>/engine_check.json
(10/13 cotrain_train.write_engine_check・10/21 gpu_queue.py enginecheck と同じ形)。

    python -I write_engine_check.py <id>

- 書くのは、学習結果に engine_check.json が無い(か今のエンジンのものでない)ときだけ。今のエンジンのものがあれば書かない
- 確認の後にエンジンが変わっていたら書かない。確認が全部の門に合格したときだけ書く(不合格は logs に残して知らせる。
  走行タブの表示を「不合格」に変えるのは、ユーザーに知らせてから)
- torch を読まない
"""
import datetime
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import desk_lib as D  # noqa: E402

RUNS = Path('/home/jetson/jetformer/data/runs')
ENGINE = 'jetformer_fp16.engine'


def main(cid):
    run = RUNS / D.RUN_OF[cid]
    src = HERE / 'logs' / ('engine_check_%s.json' % cid)
    eng = run / ENGINE
    dst = run / 'engine_check.json'
    st = eng.stat()
    if dst.exists():
        old = json.load(open(dst, encoding='utf-8'))
        if D.engine_current(old, st.st_size, st.st_mtime_ns):
            print('今のエンジンの engine_check.json があります(書きません): %s' % dst)
            return 0
    c = json.load(open(src, encoding='utf-8'))
    checked = datetime.datetime.strptime(c['checked_at'], '%Y-%m-%d %H:%M:%S').timestamp()
    if st.st_mtime > checked:
        print('確認の後にエンジンが変わっています(書きません)')
        return 1
    acc, lat = c.get('accuracy') or {}, c.get('latency_ms') or {}
    out = {'ok': bool(c.get('all_gates')), 'accuracy_ok': bool(acc.get('gate_actions_2p5us')) and bool(acc.get('finite')),
           'speed_ok': bool(lat.get('gate_run_numpy')), 'actions_max_us': acc.get('actions_max_us'),
           'actions_p99_us': acc.get('actions_p99_us'), 'gate_us': D.GATE_ACC_US,
           'run_numpy_median_ms': lat.get('run_numpy_median'), 'run_numpy_p99_ms': lat.get('run_numpy_p99'),
           'checked_at': c['checked_at'], 'note': '実車ベンチ(10/23)の机上の確認(段 Desk)で、無かったので作った',
           'engine': {'file': eng.name, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns},
           'source': str(src),
           'method': '実フレーム 64 個で TensorRT(run_numpy)と PyTorch FP32(CPU・best.pth)の actions の差。速さは 100 回の'
                     '空回しの後 600 回(experiments/20261013_sim_cotrain/train/engine_load_check.py、10/21 段 Desk)'}
    if not out['ok']:
        print('確認が門に合格していません(書きません): %s' % json.dumps(out, ensure_ascii=False))
        return 1
    tmp = str(dst) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, dst)
    print('書きました: %s %s' % (dst, json.dumps({k: out[k] for k in ('ok', 'actions_max_us', 'run_numpy_median_ms',
                                                                      'run_numpy_p99_ms')})))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
