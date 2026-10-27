"""反証 6-2・安全: 反証 4 回目の台本(fix4/review_safety の probe_giveup_moving・probe_launch_stall・probe_veh_gap_brake)を
今の jetformer/(fix6 のあと)で流し直す(台本は変えない。HERE だけこの場所の仮の箱に向け、出力は out/r4_<名前>.json)。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c62 as K          # noqa: E402


def box_for(mod, name):
    b = K.TMPROOT / ('box_' + name)
    (b / 'out').mkdir(parents=True, exist_ok=True)
    mod.HERE = b
    return b


def collect(b):
    for f in sorted((b / 'out').glob('*.json')):
        dst = HERE / 'out' / ('r4_' + f.name)
        dst.write_bytes(f.read_bytes())
        print('->', dst, flush=True)


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
        K.cleanup()


if __name__ == '__main__':
    main(sys.argv[1:] or ['s1', 's2', 's3'])
