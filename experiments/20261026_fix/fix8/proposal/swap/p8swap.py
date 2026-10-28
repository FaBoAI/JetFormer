"""fix8 係 safety-proposal: pytest の plugin(-p p8swap)。環境変数 P8_SWAP(pA / pB / pC / base)の案の写し
fix8/proposal/<案>/assist.py を、試験が import する前に sys.modules['jetformer.assist'] に入れる(本物の jetformer/assist.py には
書かない・読み込みもこの写しに差し替える)。drive.py は `from . import assist` でこの写しを使う。
試験のうち、ファイルの sha256・AST を読むものは本物のファイルを読む(そこは案と関係なく通る)。動きの試験だけが案を見る。
"""
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

PSHA = {'base': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f',
        'pA': '82e2f8a584c66fe292d2122800dcacb928d93c24aba78ca53faa5331915528d3',
        'pB': '05d78b8aeaf09a64364e284191ce19259b3b0baee4d449674056f3fe6228e5b6',
        'pC': '7ac97688a93713704762dc29f3ff64d17dc2ec4b758332ec4200ffd2240585b8'}
_name = os.environ['P8_SWAP']
_fp = Path(__file__).resolve().parents[1] / _name / 'assist.py'
assert hashlib.sha256(_fp.read_bytes()).hexdigest() == PSHA[_name], _name
sys.path.insert(0, '/home/jetson/jetformer')
import jetformer  # noqa: E402
assert 'jetformer.assist' not in sys.modules, '本物の assist がもう読まれている'
_spec = importlib.util.spec_from_file_location('jetformer.assist', str(_fp))
_m = importlib.util.module_from_spec(_spec)
sys.modules['jetformer.assist'] = _m
_spec.loader.exec_module(_m)
jetformer.assist = _m
# 写しには台本用の定数 P8_OFF_TOO(False)がある。試験が「足したモジュールの定数」を数えると、これで落ちることがある(本当に入れる
# ときは無い定数なので、その落ちは案の動きと関係ない)
