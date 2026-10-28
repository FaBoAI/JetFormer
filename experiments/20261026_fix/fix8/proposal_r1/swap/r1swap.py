"""fix8 係 safety 直す 8-1 回目: pytest の plugin(-p r1swap)。環境変数 R1_SWAP(qA / qC など)の案の写し
fix8/proposal_r1/<案>/assist.py を、試験が import する前に sys.modules['jetformer.assist'] に入れる(本物の jetformer/assist.py には
書かない)。fix8/proposal/swap/p8swap.py(前の案の係)と同じやり方。drive.py は `from . import assist` でこの写しを使う。
ファイルの sha256・AST を読む試験は本物のファイルを読む(案と関係なく通る)。動きの試験だけが案を見る。
"""
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

PSHA = {'qA': '9a5b06b41fd03f588919ebc11a998f4940fdf3b55585f6f611df7de291103c95',
        'qB': '795ed9083255536429a48080686525f8661801e1117a6648ff904fb34c91cd4e',
        'qC': '7984903beba02e898b6ab95af403df594f3cc355cad4c4d3571b4230cd3ca851'}
_name = os.environ['R1_SWAP']
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
