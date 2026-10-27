"""直す 7-1 回目・係 safety: 場面 brake(10/23 の形)の跳ねで、今と直す前の写しが分かれる所を見る(読むだけ・出力は標準出力)"""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, '/home/jetson/jetformer/tests')
sys.path.insert(0, '/home/jetson/jetformer')
import test_rec_fix7_safety as T

A = T.A


def main(tg=2.78, e=1322, esc=None):
    amod, before = T.load_before()
    tmp = Path('/home/jetson/jetformer/experiments/20261026_fix/fix7/tmp_safety/dbg')
    tmp.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        for mod in (before, T.drive):
            log = []
            am = mod.assist_mod
            orig = am.Recovery.step

            def spy(self, model_us, erpm, now, *a, **kw):
                p0 = self.phase
                r = orig(self, model_us, erpm, now, *a, **kw)
                log.append((round(now - 100.0, 3), p0, self.phase, erpm, kw.get('esc_rev', '-'), round(self.backed_s, 3),
                            round(getattr(self, 'seen_rev_s', -1), 3), r[0] and r[0][1]))
                return r
            mp.setattr(am.Recovery, 'step', spy)
            sc = T.scene_brake(glitch=[(tg, e)] if tg else [], esc=esc)
            sim, rec, pca = T.run(mod, mp, tmp, T.P(), sc)
            mp.setattr(am.Recovery, 'step', orig)
            print(mod.__name__, T.R2.guards(rec))
            for x in log:
                if 2.5 < x[0] < 3.6:
                    print('  ', x)
            print('  car', [(t, round(xx, 3), round(v, 2), th) for t, xx, v, th in sim.log if 2.6 < t < 3.4 and abs(round(t * 100) % 4) == 0][:30])


if __name__ == '__main__':
    main(*(float(x) for x in sys.argv[1:3]))
