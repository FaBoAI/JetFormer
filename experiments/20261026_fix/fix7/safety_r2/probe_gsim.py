"""直す 7-2 回目・係 safety: 試験の閉じたループ(tests/test_rec_fix7_safety の GSim)で指摘の入力が再現するか・今の版で直るかを見る
(試験を書く前の確かめ。装置に触れない)。tmp は fix7/tmp_safety_r2/probe(終わったら消す)
    cd ~/jetformer && jf-heavy --lock none --mem 1500M -- python -I experiments/20261026_fix/fix7/safety_r2/probe_gsim.py
"""
import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path('/home/jetson/jetformer')
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT)]
import pytest  # noqa: E402
import test_rec_fix7_safety as F7  # noqa: E402
from jetformer import drive  # noqa: E402

TMP = ROOT / 'experiments/20261026_fix/fix7/tmp_safety_r2/probe'


def load_r1():
    d = ROOT / 'tests/fixtures/fix7_safety_r2_before'
    aspec = importlib.util.spec_from_loader('jetformer._assist_probe_r1', loader=None)
    amod = importlib.util.module_from_spec(aspec)
    exec(compile((d / 'assist.py').read_text(), str(d / 'assist.py'), 'exec'), amod.__dict__)
    spec = importlib.util.spec_from_loader('jetformer._drive_probe_r1', loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile((d / 'drive.py').read_text(), str(d / 'drive.py'), 'exec'), mod.__dict__)
    mod.assist_mod = amod
    return mod


def main():
    r1 = load_r1()
    TMP.mkdir(parents=True, exist_ok=True)
    try:
        for name, sc in (('finding', F7.scene_rev(dur=9.0, drops=[(2.52, 2.62)], glitch=[(2.68, 1322), (2.88, 1322)])),
                         ('one', F7.scene_rev(dur=9.0, drops=[(2.52, 2.62)], glitch=[(2.68, 1322)])),
                         ('nohold', F7.scene_rev(dur=9.0, glitch=[(2.68, 1322), (2.88, 1322)]))):
            for vname, mod, p in (('now_on', drive, F7.P()), ('r1_on', r1, F7.P()), ('now_off', drive, F7.P(0, 0))):
                with pytest.MonkeyPatch.context() as mp:
                    td = TMP / ('%s_%s' % (name, vname))
                    td.mkdir()
                    s, rec, pca = F7.run(mod, mp, td, p, sc)
                    m = F7.first(s, pca)
                    print(name, vname, m, F7.R2.guards(rec))
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == '__main__':
    main()
