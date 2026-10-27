"""fix5 係 safety: 反証 4 回目の台本(fix4/review_safety の probe_*.py)を、出力の場所だけ変えて流し直す(台本は変えない・
反証の out/ は上書きしない)。今の jetformer/(直す前 = 直したあと、どちらも)で流す。装置・GPU・API には触れない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        experiments/20261026_fix/fix5/safety/rerun_probes.py TAG [s1 s2 s3]
出力: fix5/safety/out/<台本>_<TAG>.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RS = HERE.parents[1] / 'fix4' / 'review_safety'
sys.path.insert(0, str(RS))
import rs_common as R          # noqa: E402

R.TMP = HERE.parents[1] / 'fix5' / 'tmp_safety' / 'rs'
R.CL.TMP = R.TMP


def redirect(mod, tag):
    """台本の HERE を fix5/safety/<仮の場所> にして、out/<名前>.json → fix5/safety/out/<名前>_<tag>.json"""
    box = HERE / ('_redir_' + tag)
    (box / 'out').mkdir(parents=True, exist_ok=True)
    mod.HERE = box
    return box


def collect(box, tag):
    for f in sorted((box / 'out').glob('*.json')):
        dst = HERE / 'out' / ('%s_%s.json' % (f.stem, tag))
        f.replace(dst)
        print('->', dst)
    (box / 'out').rmdir()
    box.rmdir()


def main(argv):
    tag = argv[0]
    which = argv[1:] or ['s1', 's2', 's3']
    if 's1' in which:
        import probe_giveup_moving as P1
        box = redirect(P1, tag)
        P1.main()
        collect(box, tag)
    if 's2' in which:
        import probe_launch_stall as P2
        box = redirect(P2, tag)
        P2.main()
        # probe_launch_stall_keep0.py と同じ(モジュールの上で流す台本なので、ここで同じことをする)
        out = {}
        try:
            for keep in (1, 0):
                for xf in (1.3, 1.6):
                    for L in (1.0, 2.0):
                        p = dict(P2.PARAMS, recovery=1, rec_back_guard=1, rec_hold_keep=keep, tof_guard=1, launch_no_kick=0,
                                 launch_boost=1, start_kick=0)
                        sc = {'dur': 1.3 + L + 5.0, 'xf': xf, 'model': P2.model, 'drops': [(1.3, 1.3 + L)], 'veh_skip': []}
                        out['launch_boost=1 keep=%d wall=%.1f stall=1.30+%.1f' % (keep, xf, L)] = P2.one(p, sc)
        finally:
            R.cleanup()
        (box / 'out' / 'probe_launch_stall_keep0.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
        collect(box, tag)
    if 's3' in which:
        import probe_veh_gap_brake as P3
        box = redirect(P3, tag)
        P3.main()
        collect(box, tag)
    R.cleanup()


if __name__ == '__main__':
    main(sys.argv[1:])
