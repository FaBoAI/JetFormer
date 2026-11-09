"""文書の版の台本 experiments/20261026_fix/fix6/ui/docs_ban9.py の 10/31 の直し(tether_1031_docsver)の試験(2026-10-31)。

2026-10-31 USB を外して走る(tether_1031。~/jetformer_reports/bench_20261030/tether_fix/)が画面の版を 9 → 10 にした(app.js の
DRIVE_CLIENT・走行開始の body の client・server.py の DRIVE_CLIENT_VERSION)。docs_ban9.py の versions() は本物の app.js・server.py
から「今の画面の版」を読むので (10, 10, 10) になり、~/jetformer_reports/fix_20261026/ の 4 つの文書(10/26〜10/28 の記録と、封をした
11/3 の手順書。書き換えない)に残る「画面の版 9」「版 8 以下」25 か所を「古い」と数えて、文書の試験 10 件が落ちた
(tests/test_ui_fix6.py の 2 件と、test_fix7_docs_final・r1・r2、test_fix8_docs_decided・final・r1・r2・sync1 の script_edits_all_done)。
直し(docs_ban9.py だけ): 本物の app.js と server.py が両方ともこの直しの版(sha256 が LATER_TETHER_1031)のときだけ、直す前の写し
tests/fixtures/tether_1031_before/(SHA256SUMS と TETHER_1031_BEFORE_SHA で中身を確かめる。git の 29aa281)から版を読む(= 9)。
どちらか 1 つでも違えば今まで通り本物から読む。直す前の docs_ban9.py の写しは tests/fixtures/tether_1031_docsver_before/(SHA256SUMS)。

2026-11-01 電圧の前送り(speed_ff。~/jetformer_reports/bench_20261030/speed_ff/)が app.js・server.py・docs_ban9.py を変えて画面の版を
10 → 11 にした(docs_ban9.py は「その版なら直す前の写し tests/fixtures/speed_ff_1101_before/ を今の 2 つにして 10/31 の段をたどる」段を
足した)。この試験の見張りには、今の本物・台本がその直しの版(tests/speed_ff_1101_back.py の LATER_SPEED_FF_1101)なら、直す前の写し
(= 10/31 の版。中身を確かめてから)で照らす 1 段を足した(FB.back。is_tether・fake_tree・test_live_this_fix_reads_the_before_copy・
test_tool_changed_only_the_version_reading)。照らす中身は前と同じ。直しの中身は tests/test_speed_ff.py。

本物の ~/jetformer のファイルは書き換えない。偽の ~/jetformer(tmp_path に、今の app.js・server.py と直す前の写しを写したもの)を作り、
docs_ban9 の APP・SERVER・TETHER_1031_BEFORE をそこへ差しかえて確かめる。文書(~/jetformer_reports/fix_20261026/)は読むだけ
(main は --check だけで呼ぶ。書くのは --apply のときだけ)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock big --mem 3000M --no-drive-wait -- nice -n 10 \\
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider tests/test_docs_version_after_tether.py
"""
import ast
import hashlib
import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import speed_ff_1101_back as FB                     # 2026-11-01 電圧の前送りの段(この直しの版なら直す前の写し)

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'experiments' / '20261026_fix' / 'fix6' / 'ui' / 'docs_ban9.py'
FIX = ROOT / 'tests' / 'fixtures' / 'tether_1031_before'
TOOL_BEFORE = ROOT / 'tests' / 'fixtures' / 'tether_1031_docsver_before'
TOOL_BEFORE_SHA = '077973352c4c7bbfda1c5117b99f9e5425edcc05bdd8dc8283db24f57ff8f005'      # git の 29aa281 の docs_ban9.py
NAMES = ('app.js', 'server.py')
OLD_WORDS = {'画面の版 9', '版 8 以下', '版 9 と app.js', '・版 9・'}       # 版 10 で照らすと「古い」に数えられる、版 9 の時代の言葉
ADDED = ['def:_sha', 'def:version_files', 'import:hashlib', 'set:LATER_TETHER_1031', 'set:TETHER_1031_BEFORE',
         'set:TETHER_1031_BEFORE_SHA']
NOTE = '(今の app.js・server.py は 10/31 USB を外して走るの版(画面の版 10)。'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sums(d):
    return dict(reversed(x.split(None, 1)) for x in (Path(d) / 'SHA256SUMS').read_text().splitlines() if x.strip())


_SEQ = [0]


def load(path=TOOL):
    """docs_ban9 を新しいモジュールとして読む(読むたびに別の名前。ほかの試験が読んだものと混ぜない)"""
    _SEQ[0] += 1
    spec = importlib.util.spec_from_file_location('docs_ban9_after_tether_%d' % _SEQ[0], path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_before():
    """直す前の docs_ban9.py(写し。sha256 を照らしてから)"""
    assert sums(TOOL_BEFORE) == {'docs_ban9.py': TOOL_BEFORE_SHA} and sha(TOOL_BEFORE / 'docs_ban9.py') == TOOL_BEFORE_SHA
    return load(TOOL_BEFORE / 'docs_ban9.py')


def parse(app, server):
    """3 つの版の数を、試験の側で別に読む(docs_ban9 と同じ読み方)"""
    js, py = Path(app).read_text(encoding='utf-8'), Path(server).read_text(encoding='utf-8')
    return (int(re.search(r'^const DRIVE_CLIENT = (\d+);', js, re.M).group(1)),
            int(re.search(r'const body = \{ client: (\d+),', js).group(1)),
            int(re.search(r'^DRIVE_CLIENT_VERSION = (\d+)', py, re.M).group(1)))


def is_tether(m):
    # 2026-11-01 電圧の前送りの版なら、その直す前の写し(= 10/31 の版)で照らす(FB.back)
    return (sha(FB.back(m.APP)), sha(FB.back(m.SERVER))) == (m.LATER_TETHER_1031['app.js'], m.LATER_TETHER_1031['server.py'])


def need_tether(m):
    assert is_tether(m), ('今の本物の app.js・server.py が 10/31 USB を外して走るの版ではありません。docs_ban9.py の版の読み替え'
                          '(LATER_TETHER_1031)と、この試験を見直す')


def need_docs(m):
    if not all((m.REPORTS / f).exists() for f in m.FILES):
        pytest.skip('~/jetformer_reports の文書がありません')
    return m.read_all()


def literal(path, name):
    """path の一番外の name = <値> の値(import しないで読む)"""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    vals = [ast.literal_eval(n.value) for n in tree.body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(vals) == 1, (path, name)
    return vals[0]


class Tree(object):
    """偽の ~/jetformer(tmp_path の下)"""

    def __init__(self, base):
        self.jf = base / 'jf'
        self.app = self.jf / 'jetformer' / 'web' / 'static' / 'app.js'
        self.server = self.jf / 'jetformer' / 'web' / 'server.py'
        self.fix = self.jf / 'tests' / 'fixtures' / 'tether_1031_before'


def fake_tree(tmp_path):
    """偽の ~/jetformer を作る: app.js・server.py は今の本物(= この直しの版。でなければ落ちる)の写し、直す前の写しは
    tests/fixtures/tether_1031_before/ をまるごと(SHA256SUMS ごと)写したもの"""
    live = load()
    need_tether(live)
    t = Tree(tmp_path)
    for d in (t.app.parent, t.server.parent):
        d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FB.back(live.APP), t.app)                   # 2026-11-01 電圧の前送りの版なら直す前の写し(= この直しの版)
    shutil.copyfile(FB.back(live.SERVER), t.server)
    shutil.copytree(FIX, t.fix)
    assert sums(t.fix) == sums(FIX) and all(sha(t.fix / f) == h for f, h in sums(FIX).items())
    return t


def point(m, t):
    """docs_ban9 の読む所を偽の ~/jetformer へ差しかえる(直す前の写しの台本には TETHER_1031_BEFORE が無い)"""
    m.JF, m.APP, m.SERVER = t.jf, t.app, t.server
    if hasattr(m, 'TETHER_1031_BEFORE'):
        m.TETHER_1031_BEFORE = t.fix
    return m


def flip(p):
    """p の 1 バイトを変える(最初の ASCII の英小文字を別の英小文字に。UTF-8 は崩さない・版の行には当たらない)"""
    b = bytearray(Path(p).read_bytes())
    i = next(i for i, c in enumerate(b) if 0x61 <= c <= 0x7a)
    b[i] = 0x61 if b[i] != 0x61 else 0x62
    Path(p).write_bytes(bytes(b))


def bump(t, to):
    """偽の本物の版を to にする(app.js の DRIVE_CLIENT と走行開始の client、server.py の DRIVE_CLIENT_VERSION)"""
    js = t.app.read_text(encoding='utf-8')
    js, n1 = re.subn(r'^const DRIVE_CLIENT = 10;', 'const DRIVE_CLIENT = %d;' % to, js, count=1, flags=re.M)
    js, n2 = re.subn(r'const body = \{ client: 10,', 'const body = { client: %d,' % to, js, count=1)
    py, n3 = re.subn(r'^DRIVE_CLIENT_VERSION = 10\b', 'DRIVE_CLIENT_VERSION = %d' % to, t.server.read_text(encoding='utf-8'),
                     count=1, flags=re.M)
    assert (n1, n2, n3) == (1, 1, 1)
    t.app.write_text(js, encoding='utf-8')
    t.server.write_text(py, encoding='utf-8')


def resum(d):
    """d の SHA256SUMS を今の中身に書き直す(写しと SHA256SUMS を両方変えた形)"""
    names = list(sums(d))
    (d / 'SHA256SUMS').write_text(''.join('%s  %s\n' % (sha(d / f), f) for f in names))


def drop_sum(d, name):
    lines = [x for x in (d / 'SHA256SUMS').read_text().splitlines(True) if x.split()[-1:] != [name]]
    (d / 'SHA256SUMS').write_text(''.join(lines))


# ------------------------------------------------------------------ 決めた値・写し
def test_pins_are_the_same_as_the_earlier_tests():
    """決めた sha256 は前の係の試験と同じ: LATER_TETHER_1031 = tests/test_ui_fix6.py の LATER_TETHER_1031(app.js・server.py)、
    TETHER_1031_BEFORE_SHA = tests/test_tether_free.py の BEFORE_SHA(app.js・server.py)= 直す前の写しの SHA256SUMS = 写しの中身
    (写しは版 9)。読む所は前と同じ ~/jetformer の app.js・server.py と、その tests/fixtures/tether_1031_before/"""
    m = load()
    ui6 = literal(ROOT / 'tests' / 'test_ui_fix6.py', 'LATER_TETHER_1031')
    tf = literal(ROOT / 'tests' / 'test_tether_free.py', 'BEFORE_SHA')
    assert m.LATER_TETHER_1031 == {'app.js': ui6['jetformer/web/static/app.js'], 'server.py': ui6['jetformer/web/server.py']}
    assert m.TETHER_1031_BEFORE_SHA == {f: tf[f] for f in NAMES}
    s = sums(FIX)
    assert {f: s[f] for f in NAMES} == m.TETHER_1031_BEFORE_SHA == {f: sha(FIX / f) for f in NAMES}
    assert parse(FIX / 'app.js', FIX / 'server.py') == (9, 9, 9)
    jf = Path('/home/jetson/jetformer')
    assert (m.JF, m.APP, m.SERVER) == (jf, jf / 'jetformer' / 'web' / 'static' / 'app.js', jf / 'jetformer' / 'web' / 'server.py')
    assert m.TETHER_1031_BEFORE == jf / 'tests' / 'fixtures' / 'tether_1031_before'
    load_before()


@pytest.mark.parametrize('rel, copy', [('experiments/20261026_fix/fix6/ui/docs_ban9.py', TOOL_BEFORE / 'docs_ban9.py'),
                                       ('jetformer/web/static/app.js', FIX / 'app.js'),
                                       ('jetformer/web/server.py', FIX / 'server.py')],
                         ids=['docs_ban9.py', 'app.js', 'server.py'])
def test_before_copies_are_git_29aa281(rel, copy):
    """直す前の写し(docs_ban9.py・app.js・server.py)は git の 29aa281(10/29 fix9 のあと = 入れる前の ~/jetformer)と同じ"""
    r = subprocess.run(['git', '-C', str(ROOT), 'show', '29aa281:' + rel], capture_output=True, timeout=60)
    if r.returncode != 0:
        pytest.skip('git で 29aa281:%s を読めません' % rel)
    assert hashlib.sha256(r.stdout).hexdigest() == sha(copy)


# ------------------------------------------------------------------ 本物(読むだけ)
def test_live_this_fix_reads_the_before_copy():
    """本物の ~/jetformer(書き換えない): app.js・server.py がこの直しの版なら、本物は版 10 で、versions() は直す前の写しから
    (9, 9, 9)。この直しの版でなければ(のちの直しなど)、今まで通り本物から読む"""
    m = load()
    if is_tether(m):
        if FB.later_rel(m.APP) and FB.later_rel(m.SERVER):        # 2026-11-01 電圧の前送りの版(版 11): 写し(版 10)からたどる
            assert parse(m.APP, m.SERVER) == (11, 11, 11)
        assert parse(FB.back(m.APP), FB.back(m.SERVER)) == (10, 10, 10)
        assert m.version_files() == (m.TETHER_1031_BEFORE / 'app.js', m.TETHER_1031_BEFORE / 'server.py')
        assert m.versions() == (9, 9, 9)
    else:
        assert m.version_files() == (m.APP, m.SERVER) and m.versions() == parse(m.APP, m.SERVER)


def test_docs_checks_same_as_before_the_fix():
    """照らす中身は前と同じ: この直しの版のとき照らす版は 9 で、今の文書(読むだけ)の古い版の数は 0・直しは全部「済」。直す前の
    docs_ban9.py に直す前の写し(= 10/29 の本物)を読ませたときと同じ答え。版 10(直す前の versions() が今の本物から読んだ版)で照らすと
    残るのは、版 9 の時代の言葉(「画面の版 9」「版 8 以下」など)だけ = 落ちた 10 件の理由"""
    m = load()
    need_tether(m)
    docs = need_docs(m)
    m0 = load_before()
    m0.APP, m0.SERVER = FIX / 'app.js', FIX / 'server.py'
    assert m.versions() == m0.versions() == (9, 9, 9)
    assert m.stale(docs, m.versions()[0]) == m0.stale(docs, m0.versions()[0]) == []
    _, rep = m.apply_text(docs)
    assert m.EDITS == m0.EDITS and [s for _, _, s in rep] == ['done'] * len(m.EDITS)
    left = m.stale(docs, 10)
    assert left == m0.stale(docs, 10) and left and {s for _, _, s in left} <= OLD_WORDS, left


def test_main_check_says_which_version_it_reads(tmp_path, capsys):
    """--check(書かない): この直しの版のときは写しの版 9 を出し、写しから読んだ知らせの 1 行を足して 0 で終わる。本物の 1 バイトでも
    違えば、直す前の台本と同じ出力(版 10・古い版の数が残って 1 で終わる)"""
    m = load()
    need_tether(m)
    need_docs(m)
    assert m.main(['--check']) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == '画面の版: app.js DRIVE_CLIENT 9・走行開始の client 9・server.py DRIVE_CLIENT_VERSION 9'
    assert out[1].startswith(NOTE) and str(m.TETHER_1031_BEFORE) in out[1]
    assert out[-1] == '残り 0 か所' and not any(NOTE in x for x in out[2:])
    t = fake_tree(tmp_path)
    flip(t.app)
    m, m0 = point(load(), t), point(load_before(), t)
    assert m.main(['--check']) == 1
    new = capsys.readouterr().out
    assert m0.main(['--check']) == 1
    assert new == capsys.readouterr().out and NOTE not in new
    assert new.startswith('画面の版: app.js DRIVE_CLIENT 10・走行開始の client 10・server.py DRIVE_CLIENT_VERSION 10\n')


# ------------------------------------------------------------------ 偽の ~/jetformer
def test_fake_this_fix_reads_the_before_copy(tmp_path):
    """偽の ~/jetformer の app.js・server.py がこの直しの版(版 10)なら、偽の直す前の写しから (9, 9, 9)"""
    t = fake_tree(tmp_path)
    m = point(load(), t)
    assert parse(t.app, t.server) == (10, 10, 10)
    assert m.version_files() == (t.fix / 'app.js', t.fix / 'server.py') and m.versions() == (9, 9, 9)


@pytest.mark.parametrize('name', NAMES)
def test_fake_one_byte_changed_reads_the_live(tmp_path, name):
    """本物の app.js か server.py の 1 バイトでも違えば(もう 1 つはこの直しの版のまま)、今まで通り本物から読む(版 10 = 文書の試験は
    前と同じに落ちて知らせる)。直す前の docs_ban9.py と同じ答え"""
    t = fake_tree(tmp_path)
    flip(t.app if name == 'app.js' else t.server)
    m, m0 = point(load(), t), point(load_before(), t)
    assert m.version_files() == (t.app, t.server)
    assert m.versions() == m0.versions() == (10, 10, 10)


def test_fake_later_version_reads_the_live(tmp_path):
    """本物の版が 11 になったつもり(app.js の 2 か所と server.py): 本物から (11, 11, 11)。直す前の docs_ban9.py と同じ。今の文書で
    照らすと古い版の数が残る(= 文書の試験は落ちて知らせる)"""
    t = fake_tree(tmp_path)
    bump(t, 11)
    m, m0 = point(load(), t), point(load_before(), t)
    assert m.version_files() == (t.app, t.server)
    assert m.versions() == m0.versions() == (11, 11, 11)
    docs = need_docs(m)
    assert m.stale(docs, 11) and m.stale(docs, 11) == m0.stale(docs, 11)


BREAKS = {
    'before_app_1byte': lambda t: flip(t.fix / 'app.js'),
    'before_server_1byte': lambda t: flip(t.fix / 'server.py'),
    'before_app_and_sums': lambda t: (flip(t.fix / 'app.js'), resum(t.fix)),       # 写しと SHA256SUMS を両方変えた → 決めた値で落ちる
    'before_server_and_sums': lambda t: (flip(t.fix / 'server.py'), resum(t.fix)),
    'sums_without_app': lambda t: drop_sum(t.fix, 'app.js'),
    'sums_bad_line': lambda t: (t.fix / 'SHA256SUMS').write_text((t.fix / 'SHA256SUMS').read_text() + 'x\n'),
    'no_before_server': lambda t: (t.fix / 'server.py').unlink(),
    'no_sums': lambda t: (t.fix / 'SHA256SUMS').unlink(),
    'no_before_dir': lambda t: shutil.rmtree(t.fix),
}


@pytest.mark.parametrize('name', sorted(BREAKS))
def test_fake_before_copy_must_match(tmp_path, name):
    """本物がこの直しの版のとき、直す前の写しの中身が SHA256SUMS・決めた値(TETHER_1031_BEFORE_SHA)とそろわなければ(無いときも)、
    本物の版にも写しの版にも黙って切りかえず RuntimeError で落ちる(versions() も main も)"""
    t = fake_tree(tmp_path)
    BREAKS[name](t)
    m = point(load(), t)
    with pytest.raises(RuntimeError, match='直す前の写し'):
        m.versions()
    with pytest.raises(RuntimeError, match='直す前の写し'):
        m.main(['--check'])


# ------------------------------------------------------------------ 変えた所
def _top(tree):
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            k = 'def:' + n.name
        elif isinstance(n, ast.Assign):
            k = 'set:' + ','.join(t.id for t in n.targets if isinstance(t, ast.Name))
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            k = 'import:' + ','.join(a.name for a in n.names)
        elif isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            k = 'doc'
        else:
            k = 'other:' + ast.dump(n)
        assert k not in out, k
        out[k] = n
    return out


def _body(fn):
    """説明の文を除いた本体"""
    b = fn.body
    return b[1:] if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) else b


def test_tool_changed_only_the_version_reading():
    """docs_ban9.py を直す前の写しと比べる: 足したのは import hashlib・定数 3 つ(TETHER_1031_BEFORE・LATER_TETHER_1031・
    TETHER_1031_BEFORE_SHA)・_sha・version_files だけ。versions は読む 2 つを version_files() から取るだけ(ほかの行は同じ)、main は
    写しから読んだときの知らせ(print 1 つ)の if を 1 つ足しただけ、説明の文は後ろに 2026-10-31 の段を足しただけ。EDITS・HISTORY・
    STALE・apply_text・stale・read_all など、ほかは AST で同じ(照らす中身は前と同じ)"""
    o = _top(ast.parse((TOOL_BEFORE / 'docs_ban9.py').read_text(encoding='utf-8')))
    n = _top(ast.parse(FB.back(TOOL).read_text(encoding='utf-8')))     # 2026-11-01 電圧の前送りの版なら直す前の写し(= この直しの版)
    assert sorted(set(n) - set(o)) == ADDED and set(o) <= set(n)
    assert sorted(k for k in o if ast.dump(o[k]) != ast.dump(n[k])) == ['def:main', 'def:versions', 'doc']
    od, nd = o['doc'].value.value, n['doc'].value.value
    assert nd.startswith(od) and nd[len(od):].startswith('\n2026-10-31 USB を外して走る')
    ov, nv = _body(o['def:versions']), _body(n['def:versions'])
    assert ast.unparse(nv[0]) == 'app, server = version_files()'
    swap = {'app.read_text': 'APP.read_text', 'server.read_text': 'SERVER.read_text'}
    got = [ast.unparse(s) for s in nv[1:]]
    for a, b in swap.items():
        got = [x.replace(a, b) for x in got]
    assert got == [ast.unparse(s) for s in ov]
    assert ast.dump(n['def:versions'].args) == ast.dump(o['def:versions'].args)
    om, nm = o['def:main'], n['def:main']
    extra = [i for i, s in enumerate(nm.body) if isinstance(s, ast.If) and ast.unparse(s.test) == 'version_files() != (APP, SERVER)']
    assert len(extra) == 1
    s = nm.body[extra[0]]
    assert len(s.body) == 1 and not s.orelse and ast.unparse(s.body[0]).startswith("print('(今の app.js・server.py は 10/31")
    assert [ast.dump(x) for x in nm.body[:extra[0]] + nm.body[extra[0] + 1:]] == [ast.dump(x) for x in om.body]
    assert ast.dump(nm.args) == ast.dump(om.args)
