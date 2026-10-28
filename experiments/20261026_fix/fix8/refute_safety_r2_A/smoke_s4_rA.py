"""票 A: s4 のモデル(後ろへ動いている間は中立)で 1 場面だけ。 python -I -B smoke_s4_rA.py MODEL ESC K L"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rA  # noqa: E402
import sweep_rA as S  # noqa: E402

mdl, esc, k, ln = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
try:
    b = rA.run(esc, 'on', S.model(mdl, 2.0), xf=2.0, dur=7.0)
    mb = S.metrics(*b)
    fe = rA.frame_at(b[0], mb['rev_end'])
    print('base', mb['back'], mb['vpk'], mb['hits'], 'rev_end', mb['rev_end'], 'frame', fe)
    for sw in ('on', 'off'):
        r = rA.run(esc, sw, S.model(mdl, 2.0), xf=2.0, dur=7.0, drop_frames=[(fe + k, ln)])
        m = S.metrics(*r)
        print(sw, m['back'], m['vpk'], m['hits'], 'giveup', m['giveup'], 'ev', m['ev'][:4])
        print('   writes', rA.writes(r[2], mb['rev_end'] - 0.1, mb['rev_end'] + 1.2)[:10])
finally:
    rA.cleanup()
