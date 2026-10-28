"""fix8 係 safety-decided 仕事 2: jetformer/drive.py のモジュールの説明の文(docstring)の見張りの段に 1 文を足し(コードは変えない)、
drive.py の sha256 を固定している試験 8 本に、その直す前の写し(tests/fixtures/fix8_safety_doc_before/)へのたどり
(LATER_FIX8_SAFETY_DOC)を足す。

    python -I apply_doc_fix.py check   → 変える前の確かめ(写し・sha256・置き換える文がちょうど 1 回)だけ
    python -I apply_doc_fix.py apply   → 試験ファイルの写しを before_tests/ に(SHA256SUMS)とってから書く。drive.py は AST
                                          (docstring を除く)が同じことを確かめてから書く
"""
import ast
import hashlib
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
DRIVE = JF / 'jetformer' / 'drive.py'
FIXD = JF / 'tests' / 'fixtures' / 'fix8_safety_doc_before'
OLD_SHA = '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f'
NEW_SHA = '3f2e31a22ecb9abdd9c8f84b3f787d075ef636b6d5679bad5c0fc28aee18388e'
DOC_OLD = "動いていたときの衝撃ですぐ前進に移る。0 のとき PCA9685 に書く値と記録は手を入れる前と同じ。見張りが手順を変えたら\n"
DOC_NEW = ("動いていたときの衝撃ですぐ前進に移る。ただし、その手順でまだ ESC が後退で走らせていない間(前進のあとのブレーキと、\n"
           "そのあとの中立)の後ろへの動きは数えない(このESCはそこでは後退しない。10/27 fix7)。\n"
           "0 のとき PCA9685 に書く値と記録は手を入れる前と同じ。見張りが手順を変えたら\n")
CMT = ("# そのあと 10/28 fix8 の安全の係(safety-decided)が drive.py のモジュールの説明の文(docstring)の見張りの段に 1 文を足した\n"
       "# (コメントだけ。docstring を除いた AST は同じ = 走行の動きは同じ)。直す前の写しは tests/fixtures/fix8_safety_doc_before/(= 上の\n"
       "# LATER_FIX7_SAFETY の drive.py の版。SHA256SUMS)から同じようにたどる。fix8 のあとの版の sha256 はここ\n")

EDITS = {
    'tests/test_fix3_r3_2_tools.py': [
        ("LATER_FIX7_SAFETY_R2 = {\n    'jetformer/assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f',\n}\n",
         "LATER_FIX7_SAFETY_R2 = {\n    'jetformer/assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f',\n}\n"
         + CMT +
         "FIX8_SAFETY_DOC_BEFORE = JF / 'tests' / 'fixtures' / 'fix8_safety_doc_before'\n"
         "LATER_FIX8_SAFETY_DOC = {\n    'jetformer/drive.py': '%s',\n}\n" % NEW_SHA),
        ("    assist.py は 10/27 fix7 安全の係の直す 7-2 回目の版でもよい(7-2 回目の直す前の写し = 7-1 回目の版へたどる)\"\"\"\n"
         "    for rel, h in UNTOUCHED.items():\n        cur = _sha(JF / rel)\n",
         "    assist.py は 10/27 fix7 安全の係の直す 7-2 回目の版でもよい(7-2 回目の直す前の写し = 7-1 回目の版へたどる)。\n"
         "    drive.py は 10/28 fix8 安全の係の説明の文だけの版でもよい(その直す前の写し = 7-1 回目の版へたどる)\"\"\"\n"
         "    for rel, h in UNTOUCHED.items():\n        cur = _sha(JF / rel)\n"
         "        if rel in LATER_FIX8_SAFETY_DOC and cur == LATER_FIX8_SAFETY_DOC[rel]:\n"
         "            cur = _sha(FIX8_SAFETY_DOC_BEFORE / Path(rel).name)  # fix8 安全の係の説明の文の直す前の写し(LATER_FIX7_SAFETY の版)\n"),
    ],
    'tests/test_fix3_r3_1.py': [
        ("               'fix7_safety': '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f'}     # 10/27 fix7 safety 7-1 回目(LATER_FIX7_SAFETY)\n",
         "               'fix7_safety': '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f',     # 10/27 fix7 safety 7-1 回目(LATER_FIX7_SAFETY)\n"
         "               'fix8_safety_doc': '%s'}   # 10/28 fix8 safety 説明の文だけ(LATER_FIX8_SAFETY_DOC)\n" % NEW_SHA),
        ("    chain = {LATER_DRIVE['fix7_safety']: ROOT / 'tests' / 'fixtures' / 'fix7_safety_before' / 'drive.py',\n",
         "    chain = {LATER_DRIVE['fix8_safety_doc']: ROOT / 'tests' / 'fixtures' / 'fix8_safety_doc_before' / 'drive.py',\n"
         "             LATER_DRIVE['fix7_safety']: ROOT / 'tests' / 'fixtures' / 'fix7_safety_before' / 'drive.py',\n"),
    ],
    'tests/test_fix4_small.py': [
        ("    cur = _sha(DRIVE_PY)\n    if cur == FIX7_DRIVE:",
         "    cur = _sha(DRIVE_PY)\n"
         "    if cur == FIX8_DOC_DRIVE:               # 10/28 fix8 safety の説明の文だけの直す前の写し(= fix7 7-1 回目の版。AST は同じ)\n"
         "        p8 = ROOT / 'tests' / 'fixtures' / 'fix8_safety_doc_before' / 'drive.py'\n"
         "        assert _sha(p8) == FIX7_DRIVE\n"
         "        cur = FIX7_DRIVE\n"
         "    if cur == FIX7_DRIVE:"),
        ("FIX7_DRIVE = '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f'                                                         # 同じ LATER_FIX7_SAFETY\n",
         "FIX7_DRIVE = '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f'                                                         # 同じ LATER_FIX7_SAFETY\n"
         "FIX8_DOC_DRIVE = '%s'                                                     # 同じ LATER_FIX8_SAFETY_DOC\n" % NEW_SHA),
    ],
    'tests/test_rec_fix5_safety.py': [
        ("LATER_FIX7_SAFETY_R2 = {'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'}\nFDT = 1 / 30.0\n",
         "LATER_FIX7_SAFETY_R2 = {'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'}\n"
         "# さらに 10/28 fix8 safety が drive.py の説明の文だけを変えた(AST は同じ)。その直す前の写し tests/fixtures/fix8_safety_doc_before/\n"
         "# (= LATER_FIX7_SAFETY の drive.py の版)から同じようにたどる(tests/test_fix3_r3_2_tools.py の LATER_FIX8_SAFETY_DOC と同じ値)\n"
         "LATER_FIX8_SAFETY_DOC = {'drive.py': '%s'}\nFDT = 1 / 30.0\n" % NEW_SHA),
        ("    p = ROOT / 'jetformer' / name\n    if name in LATER_FIX7_SAFETY_R2 and sha(p) == LATER_FIX7_SAFETY_R2[name]:\n"
         "        p = ROOT / 'tests' / 'fixtures' / 'fix7_safety_r2_before' / name\n    if name in LATER_FIX7_SAFETY and",
         "    p = ROOT / 'jetformer' / name\n    if name in LATER_FIX8_SAFETY_DOC and sha(p) == LATER_FIX8_SAFETY_DOC[name]:\n"
         "        p = ROOT / 'tests' / 'fixtures' / 'fix8_safety_doc_before' / name\n"
         "    if name in LATER_FIX7_SAFETY_R2 and sha(p) == LATER_FIX7_SAFETY_R2[name]:\n"
         "        p = ROOT / 'tests' / 'fixtures' / 'fix7_safety_r2_before' / name\n    if name in LATER_FIX7_SAFETY and"),
    ],
    'tests/test_rec_fix6_safety.py': [
        ("LATER_FIX7_SAFETY_R2 = {'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'}\nDT = R2.DT\n",
         "LATER_FIX7_SAFETY_R2 = {'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'}\n"
         "# さらに 10/28 fix8 safety が drive.py の説明の文だけを変えた(AST は同じ)。その直す前の写し tests/fixtures/fix8_safety_doc_before/\n"
         "# (= 7-1 回目のあとの drive.py)から同じようにたどる(tests/test_fix3_r3_2_tools.py の LATER_FIX8_SAFETY_DOC)\n"
         "LATER_FIX8_SAFETY_DOC = {'drive.py': '%s'}\nDT = R2.DT\n" % NEW_SHA),
        ("    p = ROOT / 'jetformer' / name\n    if name in LATER_FIX7_SAFETY_R2 and sha(p) == LATER_FIX7_SAFETY_R2[name]:\n",
         "    p = ROOT / 'jetformer' / name\n    if name in LATER_FIX8_SAFETY_DOC and sha(p) == LATER_FIX8_SAFETY_DOC[name]:\n"
         "        p = ROOT / 'tests' / 'fixtures' / 'fix8_safety_doc_before' / name\n"
         "    if name in LATER_FIX7_SAFETY_R2 and sha(p) == LATER_FIX7_SAFETY_R2[name]:\n"),
    ],
    'tests/test_rec_fix6_safety_r2.py': [
        ("LATER_FIX7_SAFETY_R2 = {'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'}\n\n\ndef _fix6r2_version",
         "LATER_FIX7_SAFETY_R2 = {'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'}\n"
         "# さらに 10/28 fix8 safety が drive.py の説明の文だけを変えた(AST は同じ)。その直す前の写し tests/fixtures/fix8_safety_doc_before/\n"
         "# (= LATER_FIX7_SAFETY の drive.py の版)から同じようにたどる(tests/test_fix3_r3_2_tools.py の LATER_FIX8_SAFETY_DOC と同じ値)\n"
         "LATER_FIX8_SAFETY_DOC = {'drive.py': '%s'}\n\n\ndef _fix6r2_version" % NEW_SHA),
        ("    p = ROOT / 'jetformer' / name\n    if name in LATER_FIX7_SAFETY_R2 and sha(p) == LATER_FIX7_SAFETY_R2[name]:\n",
         "    p = ROOT / 'jetformer' / name\n    if name in LATER_FIX8_SAFETY_DOC and sha(p) == LATER_FIX8_SAFETY_DOC[name]:\n"
         "        p = ROOT / 'tests' / 'fixtures' / 'fix8_safety_doc_before' / name\n"
         "    if name in LATER_FIX7_SAFETY_R2 and sha(p) == LATER_FIX7_SAFETY_R2[name]:\n"),
    ],
    'tests/test_rec_fix7_safety.py': [
        ("LATER_FIX7_SAFETY_R2 = {'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'}\n\n\ndef _fix7r1_version",
         "LATER_FIX7_SAFETY_R2 = {'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'}\n"
         "# さらに 10/28 fix8 safety が drive.py の説明の文だけを変えた(AST は同じ)。その直す前の写し tests/fixtures/fix8_safety_doc_before/\n"
         "# (= この組のあとの drive.py)から同じようにたどる(tests/test_fix3_r3_2_tools.py の LATER_FIX8_SAFETY_DOC と同じ値)\n"
         "LATER_FIX8_SAFETY_DOC = {'drive.py': '%s'}\n\n\ndef _fix7r1_version" % NEW_SHA),
        ("    p = ROOT / 'jetformer' / name\n    if name in LATER_FIX7_SAFETY_R2 and sha(p) == LATER_FIX7_SAFETY_R2[name]:\n",
         "    p = ROOT / 'jetformer' / name\n    if name in LATER_FIX8_SAFETY_DOC and sha(p) == LATER_FIX8_SAFETY_DOC[name]:\n"
         "        p = ROOT / 'tests' / 'fixtures' / 'fix8_safety_doc_before' / name\n"
         "    if name in LATER_FIX7_SAFETY_R2 and sha(p) == LATER_FIX7_SAFETY_R2[name]:\n"),
    ],
    'tests/test_rec_fix7_safety_r2.py': [
        ("          'drive.py': '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f'}\nFDT = F7.FDT\n",
         "          'drive.py': '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f'}\n"
         "# この組のあと 10/28 fix8 safety が drive.py の説明の文だけを変えた(AST は同じ)。そのときは、その直す前の写し\n"
         "# tests/fixtures/fix8_safety_doc_before/(= この組の drive.py。sha256 を照らす)で照らす(tests/test_fix3_r3_2_tools.py の\n"
         "# LATER_FIX8_SAFETY_DOC と同じ値)\n"
         "LATER_FIX8_SAFETY_DOC = {'drive.py': '%s'}\nFDT = F7.FDT\n" % NEW_SHA),
        ("    assert sha(ROOT / 'jetformer' / 'drive.py') == SHA_R1['drive.py']\n",
         "    pd = ROOT / 'jetformer' / 'drive.py'\n"
         "    if sha(pd) == LATER_FIX8_SAFETY_DOC['drive.py']:         # 10/28 fix8 safety の説明の文だけの版 → 直す前の写し\n"
         "        pd = ROOT / 'tests' / 'fixtures' / 'fix8_safety_doc_before' / 'drive.py'\n"
         "    assert sha(pd) == SHA_R1['drive.py']\n"),
    ],
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def nodoc(src):
    t = ast.parse(src)
    for n in ast.walk(t):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body \
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) \
                and isinstance(n.body[0].value.value, str):
            n.body = n.body[1:]
    return ast.dump(t)


def check():
    sums = dict(reversed(x.split()) for x in (FIXD / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert sums == {'drive.py': OLD_SHA} and sha(FIXD / 'drive.py') == OLD_SHA
    cur = sha(DRIVE)
    assert cur in (OLD_SHA, NEW_SHA), cur
    src = DRIVE.read_text(encoding='utf-8')
    if cur == OLD_SHA:
        assert src.count(DOC_OLD) == 1
        new = src.replace(DOC_OLD, DOC_NEW)
        assert hashlib.sha256(new.encode('utf-8')).hexdigest() == NEW_SHA
        assert nodoc(src) == nodoc(new) and ast.dump(ast.parse(src)) != ast.dump(ast.parse(new))
    for rel, eds in EDITS.items():
        t = (JF / rel).read_text(encoding='utf-8')
        if 'LATER_FIX8_SAFETY_DOC' in t or 'FIX8_DOC_DRIVE' in t:
            print('済み', rel)
            continue
        for old, new in eds:
            assert t.count(old) == 1, (rel, old[:70], t.count(old))
        print('ok', rel, len(eds))
    print('drive.py', cur[:12])


def apply():
    check()
    bt = HERE / 'before_tests'
    bt.mkdir(exist_ok=True)
    lines = []
    for rel in EDITS:
        p = JF / rel
        if 'LATER_FIX8_SAFETY_DOC' in p.read_text(encoding='utf-8') or 'FIX8_DOC_DRIVE' in p.read_text(encoding='utf-8'):
            continue
        shutil.copy2(p, bt / p.name)
        lines.append('%s  %s' % (sha(bt / p.name), p.name))
    if lines:
        with open(bt / 'SHA256SUMS', 'a', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
    for rel, eds in EDITS.items():
        p = JF / rel
        t = p.read_text(encoding='utf-8')
        if 'LATER_FIX8_SAFETY_DOC' in t or 'FIX8_DOC_DRIVE' in t:
            continue
        for old, new in eds:
            assert t.count(old) == 1, (rel, old[:70])
            t = t.replace(old, new)
        compile(t, rel, 'exec')
        p.write_text(t, encoding='utf-8')
        print('書いた', rel, sha(p))
    if sha(DRIVE) == OLD_SHA:
        src = DRIVE.read_text(encoding='utf-8')
        new = src.replace(DOC_OLD, DOC_NEW)
        assert nodoc(src) == nodoc(new)
        DRIVE.write_text(new, encoding='utf-8')
    assert sha(DRIVE) == NEW_SHA
    assert nodoc(DRIVE.read_text(encoding='utf-8')) == nodoc((FIXD / 'drive.py').read_text(encoding='utf-8'))
    print('drive.py', sha(DRIVE))


if __name__ == '__main__':
    {'check': check, 'apply': apply}[sys.argv[1]]()
