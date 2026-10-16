#!/usr/bin/env python3
"""10/16 の電源断(10/15 23:46 ごろ〜10/16 10:03)の後の片付け: すぐ済む段だけを済ませて、残りは削る。

    cd ~/jetformer && /home/jetson/miniconda/envs/jetracer/bin/python -u \\
        experiments/20261015_real_accuracy/train/resume_20261016.py \\
        >> /home/jetson/jetformer_reports/real_accuracy_20261015/logs/driver.log 2>&1

やること(新しい学習はしない):
  1. sizeHeq-s0(固定の外の参考の腕 SIZE-EQ): 学習は 23:11 に最後(59 エポック)まで済んでいた(posthoc の道具が途中で止まり、
     state.json に終わりが書かれていなかった)→ realacc_driver.train_one で確かめだけ(学習のサンプル数 3,046・検証 1,559)
  2. gainl-s0・sizeHeq-s0 の best・last の予測(GPU・realacc_eval.py predict)→ 読み直し → last.pth を消す → evaluate(CPU)
     → progress.md(realacc_driver.do_predict のまま)
  3. 削る: gainl-s1(23:44 に始まり 8 エポック目で電源断)・sizeHeq-s1・sizeQeq-s0/s1。state.json の cuts と deviations.md に書く。
     gainl-s1 の途中の出力(自分で作った未完の学習結果のフォルダ)は消す
車は動かさない・PCA9685・I2C に触らない。jetformer-web を再起動しない。GPU は 1 つずつ(/tmp/realacc_driver.lock)。
"""
import fcntl
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import realacc_driver as D  # noqa: E402

WHEN = '2026-10-16'
CUT_TEXT = '電源断(10/15 23:46〜10/16 10:03)の後に削った'


def main():
    lock = open('/tmp/realacc_driver.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit('realacc_driver.py がまだ動いています')
    proto, sha = D.load_protocol()
    state = D.load(D.STATE_PATH, {}) or {}
    state['protocol_sha256'] = sha
    free = D.disk_free_gb()
    D.log('==== 10/16 の片付け(resume_20261016.py)・空き %.2f GB ====' % free)
    if free < D.DISK_STOP_GB:
        raise SystemExit('空き容量 %.2f GB < %.1f GB なので止めます' % (free, D.DISK_STOP_GB))
    runs = state.setdefault('runs', {})
    cuts = state.setdefault('cuts', [])
    # ---- 3. 削る(先に書く: この後で止まっても、削ったことは残る)
    for name, why in ((D.PREFIX + 'gainl-s1', '23:44 に始まり 8 エポック目で電源断。続きからの学習(約 17 分)はしない'),
                      (D.PREFIX + 'sizeHeq-s1', '始まっていない'), (D.PREFIX + 'sizeQeq-s0', '始まっていない'),
                      (D.PREFIX + 'sizeQeq-s1', '始まっていない')):
        rs = runs.get(name)
        if rs is not None and not rs.get('check_ok'):
            rs['cut'] = '電源断で未完'
        c = 'train:%s: %s(%s・%s)' % (name, CUT_TEXT, why, D.now())
        if not any(x.startswith('train:%s:' % name) for x in cuts):
            cuts.append(c)
    for g in ('predict:GAINL', 'predict:SIZEEQ_H', 'predict:SIZEEQ_Q'):
        if not any(x.startswith(g + ':') for x in cuts):
            cuts.append('%s: %s(その段の学習結果がそろわない。済んだ gainl-s0・sizeHeq-s0 は predict:RESUME_20261016 で予測・%s)'
                        % (g, CUT_TEXT, D.now()))
    D.save_state(state)
    part = D.RUNS / (D.PREFIX + 'gainl-s1')
    if part.exists() and not D.run_done(part, 30):
        size = sum(p.stat().st_size for p in part.rglob('*') if p.is_file())
        shutil.rmtree(part)
        runs[D.PREFIX + 'gainl-s1']['partial_removed'] = {'at': D.now(), 'bytes': size}
        D.log('gainl-s1 の未完の学習結果(%.0f MB)を消しました' % (size / 1e6))
        D.save_state(state)
    # ---- 1. sizeHeq-s0 の確かめ(学習は済んでいる。train_one は済みなら学習しない)
    step = next(s for s in D.build_posthoc_steps(proto) if s.get('name') == D.PREFIX + 'sizeHeq-s0')
    run = D.RUNS / step['name']
    if not D.run_done(run, 59):
        raise SystemExit('sizeHeq-s0 が済んでいません(新しい学習はしない)')
    rs = runs[step['name']]
    res = D.load(run / 'result.json') or {}
    for a in rs.get('attempts') or []:
        if 'exit' not in a:
            a.update({'exit': 0, 'wall_s': res.get('duration_s'), 'ended': '2026-10-15 23:11',
                      'note': 'posthoc の道具が途中で止まり終わりが書かれていなかった。学習は最後まで(result.json の state done・'
                              '59 エポック・train.log)'})
    D.save_state(state)
    if not D.train_one(state, step):
        raise SystemExit('sizeHeq-s0 の確かめが通りません')
    done = set(state.get('done_steps') or [])
    done.add(step['id'])
    order = [x['id'] for x in D.build_steps(proto) + D.build_posthoc_steps(proto)]
    state['done_steps'] = sorted(done, key=lambda x: order.index(x) if x in order else len(order))
    D.save_state(state)
    # ---- 2. 予測 → last.pth を消す → evaluate → progress.md
    names = [D.PREFIX + 'gainl-s0', D.PREFIX + 'sizeHeq-s0']
    ok = D.do_predict(state, {'id': 'predict:RESUME_20261016', 'names': names})
    D.log('10/16 の片付け: 予測 %s・削った %d 件・空き %.2f GB' % ('OK' if ok else 'NG', len(cuts), D.disk_free_gb()))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
