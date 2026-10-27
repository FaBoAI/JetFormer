"""直す 7-1 回目・係 safety: 試験の GCar の spring と反証の rv7 の spring の違いを見る(読むだけ・出力は標準出力)"""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, '/home/jetson/jetformer/tests')
sys.path.insert(0, '/home/jetson/jetformer')
import test_rec_fix7_safety as T


def main():
    amod, before = T.load_before()
    tmp = Path('/home/jetson/jetformer/experiments/20261026_fix/fix7/tmp_safety/dbg')
    tmp.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        for mod in (before, T.drive):
            sim, rec, pca = T.run(mod, mp, tmp, T.P(), T.scene_rev(cam0=0.15, spring=(0.16, 0.05, 0.2)))
            print(mod.__name__, T.first(sim, pca), T.R2.guards(rec))
            for t, x, v, th in sim.log:
                if 2.45 < t < 2.75:
                    print('  %.2f x=%.4f v=%.3f thr=%.0f' % (t, x, v, th))


if __name__ == '__main__':
    main()
