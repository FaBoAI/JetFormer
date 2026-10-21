"""Fix-drive の変異の確認: 直す前の drive.py(引数のファイル)で、新しい試験の確かめ方が差を見つけるか。
    cd ~/jetformer/tests && jf-heavy --lock none --mem 1500M -- python ../experiments/20261021_bench_prep/fix_drive/mutation_check.py <直す前の drive.py>
"""
import importlib.util
import sys

import pytest

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from jetformer import drive                      # noqa: E402
import test_drive_caps as tdc                     # noqa: E402
import test_drive_caps_fix as fix                 # noqa: E402


def load(path):
    spec = importlib.util.spec_from_loader('jetformer._drive_before_fix', loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile(open(path, encoding='utf-8').read(), path, 'exec'), mod.__dict__)
    return mod


def counters(mod, tmp):
    mp = pytest.MonkeyPatch()
    try:
        frames, imu = tdc.scenario()
        pca = fix.FlakyPCA()
        mp.setattr(tdc, 'FakePCA', lambda: pca)
        d, _, rec = tdc.run_det(mod, mp, tmp, dict(tdc.PARAMS, zone_caps=1), frames, imu)
        rows = [r for _, r in rec.rows]
        return (d.st['stats']['zone_applied'], sum(bool(r['cap_applied']) for r in rows),
                d.st['stats']['zone_brakes'], sum(bool(r['cap_brake_start']) for r in rows))
    finally:
        mp.undo()


if __name__ == '__main__':
    import pathlib
    import tempfile
    old = load(sys.argv[1])
    with tempfile.TemporaryDirectory() as t:
        for name, mod in (('before_fix', old), ('after_fix', drive)):
            a, ra, b, rb = counters(mod, pathlib.Path(t))
            print('%-10s screen applied %d / log %d, screen brakes %d / log %d -> %s'
                  % (name, a, ra, b, rb, 'same' if (a, b) == (ra, rb) else 'DIFFERENT'))
        print('before_fix has unknown_note:', 'unknown_note' in old.ZONE_CAPS_SPEC)
