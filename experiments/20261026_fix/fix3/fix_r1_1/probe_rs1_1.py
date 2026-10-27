"""r1-rs1-1 の直しの確かめ(本物の drive._loop。偽物のセンサー。反証の台本 refute1_check/r1_rs1_1.py の場面をそのまま使う)。
当たったあとに PCA9685 へ後退側(中立 + 40 µs より上)を書いたフレームの数と秒を、今のコードで数える。出力は out/probe_rs1_1.json。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python experiments/20261026_fix/fix3/fix_r1_1/probe_rs1_1.py
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'refute1_check'))
import r1_rs1_1 as R      # noqa: E402

R.TMP = HERE.parent / 'tmp_fix_r1_1_probe'
N = R.N


def frames_after(pca, hit):
    return sum(1 for t, s, th in pca.w if t >= hit and th > N + 40.0)


def main():
    res = {}
    hit = 3.25
    combos = ((0, 0), (0, 1), (1, 0), (1, 1))
    for drop_len in (0.0, 0.08, 0.1, 0.134, 0.167, 0.25, 0.5):
        drop = (hit - 0.02, hit - 0.02 + drop_len) if drop_len else None
        for keep, guard in combos:
            frames, imu = R.scene(hit, drop)
            pca, rec = R.run(dict(R.PARAMS, recovery=1, rec_back_guard=guard, rec_hold_keep=keep), frames, imu)
            r = R.rear_push(pca, rec, hit)
            r['rev_frames_after_hit'] = frames_after(pca, hit)
            res['camdrop=%.3f keep=%d guard=%d' % (drop_len, keep, guard)] = r
    for start in (-0.10, -0.05, 0.0, 0.02, 0.05):
        drop = (hit + start, hit + start + 0.2)
        frames, imu = R.scene(hit, drop)
        pca, rec = R.run(dict(R.PARAMS, recovery=1, rec_back_guard=1, rec_hold_keep=1), frames, imu)
        r = R.rear_push(pca, rec, hit)
        r['rev_frames_after_hit'] = frames_after(pca, hit)
        res['shift start=%+.2f len=0.2 keep=1 guard=1' % start] = r
    for keep, guard in combos:
        frames, imu = R.scene(3.30, None, veh_hold=(3.10, 3.50))
        pca, rec = R.run(dict(R.PARAMS, recovery=1, rec_back_guard=guard, rec_hold_keep=keep), frames, imu)
        r = R.rear_push(pca, rec, 3.30)
        r['rev_frames_after_hit'] = frames_after(pca, 3.30)
        res['vehstale 3.10-3.50 hit=3.30 keep=%d guard=%d' % (keep, guard)] = r
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'probe_rs1_1.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    for k, v in res.items():
        print(k, '| push_s', v['push_s'], '| frames', v['rev_frames_after_hit'], '| us', v['us'], '| ev', v['guard_events'])
    shutil.rmtree(R.TMP, ignore_errors=True)


if __name__ == '__main__':
    main()
