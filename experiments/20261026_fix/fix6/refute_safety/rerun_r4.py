"""反証 6-1・安全: 反証 4 回目の台本(fix4/review_safety の probe_giveup_moving・probe_launch_stall・probe_veh_gap_brake)を
今の jetformer/ で流し直す(台本は変えない。HERE だけこの場所の仮の箱に向け、出力は out/r4_<名前>.json)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        experiments/20261026_fix/fix6/refute_safety/rerun_r4.py [s1 s2 s3]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r6_common as K          # noqa: E402,F401
from r6_common import R        # noqa: E402


def box_for(mod, name):
    b = HERE / ('_box_' + name)
    (b / 'out').mkdir(parents=True, exist_ok=True)
    mod.HERE = b
    return b


def collect(b):
    for f in sorted((b / 'out').glob('*.json')):
        dst = HERE / 'out' / ('r4_' + f.name)
        f.replace(dst)
        print('->', dst, flush=True)
    (b / 'out').rmdir()
    b.rmdir()


def main(which):
    try:
        if 's1' in which:
            import probe_giveup_moving as P1
            b = box_for(P1, 's1')
            P1.main()
            collect(b)
        if 's2' in which:
            import probe_launch_stall as P2
            b = box_for(P2, 's2')
            P2.main()
            collect(b)
        if 's3' in which:
            import probe_veh_gap_brake as P3
            b = box_for(P3, 's3')
            P3.main()
            collect(b)
    finally:
        R.cleanup()


if __name__ == '__main__':
    main(sys.argv[1:] or ['s1', 's2', 's3'])
