"""fix8 係 safety 直す 8-1 回目: 案の写し(proposal_r1/<案>/assist.py)を、今の jetformer/drive.py と組にしてモジュールにする(読み込む
だけ。ほかのモジュールを書き換えない)。本物の jetformer/ には書かない。"""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
QSHA = {'qA': '9a5b06b41fd03f588919ebc11a998f4940fdf3b55585f6f611df7de291103c95',
        'qB': '795ed9083255536429a48080686525f8661801e1117a6648ff904fb34c91cd4e',
        'qC': '7984903beba02e898b6ab95af403df594f3cc355cad4c4d3571b4230cd3ca851'}
NOW_SHA = {'drive.py': '3f2e31a22ecb9abdd9c8f84b3f787d075ef636b6d5679bad5c0fc28aee18388e',
           'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'}
_VERS = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_now():
    got = {f: sha(ROOT / 'jetformer' / f) for f in NOW_SHA}
    assert got == NOW_SHA, got
    return got


def load_version(name):
    """(drive のモジュール, assist のモジュール)。'now' は本物の jetformer の 2 つ"""
    if name == 'now':
        from jetformer import assist, drive
        return drive, assist
    if name in _VERS:
        return _VERS[name]
    fp = HERE / name / 'assist.py'
    assert sha(fp) == QSHA[name], name
    aspec = importlib.util.spec_from_loader('jetformer._r1_assist_%s' % name, loader=None)
    amod = importlib.util.module_from_spec(aspec)
    exec(compile(fp.read_text(encoding='utf-8'), str(fp), 'exec'), amod.__dict__)
    dp = ROOT / 'jetformer' / 'drive.py'
    spec = importlib.util.spec_from_loader('jetformer._r1_drive_%s' % name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile(dp.read_text(encoding='utf-8'), str(dp), 'exec'), mod.__dict__)
    mod.assist_mod = amod
    _VERS[name] = (mod, amod)
    return mod, amod
