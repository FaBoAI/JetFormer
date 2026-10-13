"""ブリッジ用の Python 環境(data/sim/venv)の準備と状態

- JetFormer の Python(sys.executable = conda jetracer の 3.13)から venv を作る。jetracer env には何も入れない。
- 入れるものは jetformer/sim/requirements-bridge.txt で版とハッシュを固定する(--require-hashes --only-binary=:all: --no-deps)。
  data/sim/wheels/ に wheel があれば先にオフラインで試し、だめなら PyPI から取る。
- シミュレーターの requirements-model.txt の版(numpy・opencv-python-headless・onnxruntime)が固定版と違えば、
  勝手に別の版を入れずに止める。
- 最後に版を確かめ、ブリッジの self-test(bridge.py --self-test 相当)を 1 回動かす。
"""

import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

from . import install, paths

CHECKED = ('numpy', 'opencv-python-headless', 'onnxruntime')
PYPI = 'https://pypi.org/simple'
_REQ_RE = re.compile(r'^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9.+!_-]+)((?:\s+--hash=sha256:[0-9a-f]{64})*)\s*$')
CHECK_CODE = ("import importlib.metadata as m, json, sys\n"
              "import numpy, cv2, onnxruntime\n"
              "names = ['numpy', 'opencv-python-headless', 'onnxruntime', 'protobuf', 'flatbuffers', 'packaging']\n"
              "print(json.dumps({'python': sys.version.split()[0], 'packages': {n: m.version(n) for n in names},\n"
              "                  'providers': onnxruntime.get_available_providers()}))\n")


class EnvError(RuntimeError):
    """準備できない(日本語の理由)"""

    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


def parse_requirements(text):
    """requirements-bridge.txt を読む: [(名前, 版, [sha256 ...])]。書式が違えば ValueError"""
    out = []
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].strip() if not raw.lstrip().startswith('#') else ''
        if not line:
            continue
        m = _REQ_RE.match(line)
        if not m:
            raise ValueError('requirements の書式が違います: %s' % raw)
        hashes = re.findall(r'sha256:([0-9a-f]{64})', m.group(3))
        out.append((m.group(1).lower(), m.group(2), hashes))
    return out


def pinned():
    """{名前: 版}(JetFormer 側で固定した版)"""
    return {n: v for n, v, _ in parse_requirements(paths.REQUIREMENTS.read_text(encoding='utf-8'))}


def requirements_sha256():
    return hashlib.sha256(paths.REQUIREMENTS.read_bytes()).hexdigest()


def release_requirements(root):
    """シミュレーターの requirements-model.txt の {名前: 版}(== の行だけ)"""
    out = {}
    try:
        text = (root / 'requirements-model.txt').read_text(encoding='utf-8')
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].strip()
        m = re.match(r'^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([A-Za-z0-9.+!_-]+)', line)
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


def version_mismatch(release):
    """シミュレーターが求める版と固定版の違い(なければ空)"""
    want = release_requirements(release.root)
    have = pinned()
    return ['%s: シミュレーター %s / JetFormer の固定版 %s' % (n, want[n], have.get(n, 'なし'))
            for n in CHECKED if n in want and want[n] != have.get(n)]


def _read_record():
    try:
        with open(paths.venv_record(), encoding='utf-8') as f:
            r = json.load(f)
        return r if isinstance(r, dict) else None
    except (OSError, ValueError):
        return None


def _venv_problem():
    """venv が使えそうか(実行はしない。pyvenv.cfg の home と python があるか)"""
    vd = paths.venv_dir()
    py = paths.venv_python()
    cfg = vd / 'pyvenv.cfg'
    if not cfg.is_file():
        return 'venv がありません'
    try:
        home = None
        for line in cfg.read_text(encoding='utf-8').splitlines():
            k, _, v = line.partition('=')
            if k.strip() == 'home':
                home = v.strip()
        if home and not os.path.isdir(home):
            return 'venv の元の Python(%s)が見つかりません。準備し直してください' % home
    except OSError:
        return 'venv の設定を読めません'
    if not os.path.exists(py):                    # シンボリックリンクの先まで確かめる
        return 'venv の python が動きません(元の Python が更新された可能性)。準備し直してください'
    return None


def status():
    """{ready, python_version, packages, problem, created_at}。通信も実行もしない(2 秒ごとに呼ばれる)"""
    rec = _read_record()
    problem = _venv_problem()
    if rec is None and problem == 'venv がありません':
        return {'ready': False, 'problem': None, 'python_version': None, 'packages': {}, 'created_at': None}
    if not problem:
        if not rec or rec.get('self_test') != 'ok':
            problem = '準備が終わっていません(途中で止まった可能性)。準備し直してください'
        elif rec.get('requirements_sha256') != requirements_sha256():
            problem = 'JetFormer 側の固定版が変わりました。準備し直してください'
    rec = rec or {}
    return {'ready': problem is None, 'problem': problem, 'python_version': rec.get('python_version'),
            'packages': rec.get('packages') or {}, 'created_at': rec.get('created_at'), 'python': rec.get('python'),
            'source': rec.get('source')}


def _env():
    keep = ('PATH', 'HOME', 'LANG', 'LC_ALL', 'http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'no_proxy', 'NO_PROXY')
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.setdefault('PATH', '/usr/bin:/bin')
    env.update(PYTHONDONTWRITEBYTECODE='1', PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_INPUT='1', CUDA_VISIBLE_DEVICES='')
    return env


def run_command(cmd, on_line=None, timeout=1800, cwd=None):
    """コマンドを動かして (終了コード, 出力の行) を返す(stdout と stderr をまとめる)"""
    lines = []
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         env=_env(), cwd=str(cwd) if cwd else None, text=True, errors='replace')
    try:
        for line in p.stdout:
            line = line.rstrip('\n')
            lines.append(line)
            if len(lines) > 2000:
                del lines[:1000]
            if on_line:
                on_line(line)
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        return -9, lines + ['時間切れで止めました']
    finally:
        p.stdout.close()
    return rc, lines


def pip_commands(python, wheels_dir=None):
    """pip の呼び方の候補(順に試す)。wheels があればオフラインを先に"""
    base = [str(python), '-m', 'pip', 'install', '--require-hashes', '--only-binary=:all:', '--no-deps',
            '--disable-pip-version-check', '--no-input', '-r', str(paths.REQUIREMENTS)]
    cmds = []
    try:
        has_wheels = bool(wheels_dir) and any(n.endswith('.whl') for n in os.listdir(wheels_dir))
    except OSError:
        has_wheels = False
    if has_wheels:
        cmds.append(('offline', base + ['--no-index', '--find-links', str(wheels_dir)]))
    cmds.append(('pypi', base + ['--index-url', PYPI]))
    return cmds


def _tail(lines, n=12):
    return ' / '.join(x.strip() for x in lines[-n:] if x.strip())


def setup(force=False, progress=None, run=run_command):
    """venv を作って部品を入れ、版と self-test を確かめる(ジョブで動かす)"""
    progress = progress or (lambda p, m='': None)
    if not paths.OP_LOCK.acquire(blocking=False):
        raise EnvError('ほかの導入・準備が動いています。終わってからもう一度押してください')
    try:
        return _setup(force, progress, run)
    finally:
        paths.OP_LOCK.release()


def _setup(force, progress, run):
    release = install.active_release()
    if release is None:
        raise EnvError('先にシミュレーター本体を導入してください')
    diff = version_mismatch(release)
    if diff:
        raise EnvError('シミュレーターが求める版が変わりました。JetFormer 側の固定版(jetformer/sim/requirements-bridge.txt)の更新が必要です: '
                       + ' / '.join(diff))
    vd = paths.venv_dir()
    if force or (vd.exists() and not status()['ready']):
        if vd.resolve().parent != paths.sim_dir().resolve():
            raise EnvError('venv の場所がおかしいので消せません: %s' % vd)
        progress(0.02, '前の venv を消しています')
        shutil.rmtree(vd, ignore_errors=True)
        paths.venv_record().unlink(missing_ok=True)
    elif status()['ready']:
        progress(1.0, '準備できています')
        return dict(status(), already_ready=True)
    paths.venv_record().unlink(missing_ok=True)
    paths.sim_dir().mkdir(parents=True, exist_ok=True)
    progress(0.05, 'venv を作っています')
    rc, lines = run([sys.executable, '-m', 'venv', str(vd)], timeout=600)
    if rc != 0:
        raise EnvError('venv を作れませんでした: %s' % _tail(lines))

    count = [0]

    def on_line(_line):
        count[0] += 1
        progress(min(0.1 + 0.7 * count[0] / 40.0, 0.8), '部品を入れています(pip)')
    progress(0.1, '部品を入れています(pip)')
    errors = []
    source = None
    for name, cmd in pip_commands(paths.venv_python(), paths.wheels_dir()):
        rc, lines = run(cmd, on_line=on_line, timeout=1800)
        if rc == 0:
            source = name
            break
        errors.append('%s: %s' % (name, _tail(lines, 6)))
    if source is None:
        raise EnvError('部品を入れられませんでした: ' + ' | '.join(errors))

    progress(0.85, '版を確かめています')
    rc, lines = run([paths.venv_python(), '-I', '-c', CHECK_CODE], timeout=300)
    info = None
    for line in reversed(lines):
        try:
            info = json.loads(line)
            break
        except ValueError:
            continue
    if rc != 0 or not isinstance(info, dict):
        raise EnvError('入れた部品を読み込めません: %s' % _tail(lines))
    want = pinned()
    wrong = ['%s %s(期待 %s)' % (n, info['packages'].get(n), v) for n, v in want.items() if info['packages'].get(n) != v]
    if wrong:
        raise EnvError('入った版が固定版と違います: ' + ', '.join(wrong))
    if 'CPUExecutionProvider' not in info.get('providers', []):
        raise EnvError('onnxruntime の CPU 実行が使えません')

    progress(0.9, 'ブリッジの自己診断')
    with tempfile.TemporaryDirectory(prefix='selftest-', dir=str(paths.sim_dir())) as tmp:
        rc, lines = run([paths.venv_python(), '-I', '-B', paths.RUNNER, '--sim-dir', release.root, '--runtime-root', tmp,
                         '--self-test'], timeout=600, cwd=tmp)
    m = None
    for line in lines:
        m = re.search(r'Ran (\d+) tests?', line) or m
    tests = int(m.group(1)) if m else None
    if rc != 0:
        raise EnvError('ブリッジの自己診断が通りませんでした: %s' % _tail(lines))
    rec = {'python': sys.executable, 'python_version': info['python'], 'packages': info['packages'],
           'requirements_sha256': requirements_sha256(), 'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
           'self_test': 'ok', 'self_test_count': tests, 'release': release.id, 'source': source}
    # pip が書いた数千のファイルを先にディスクへ書き切ってから「準備済み」の記録を書く(電源断で、記録だけあって
    # 中身が 0 バイトの venv を残さない。review で追加)
    os.sync()
    paths.write_durable(paths.venv_record(), json.dumps(rec, ensure_ascii=False, indent=1))
    progress(1.0, '準備できました(自己診断 %s 件)' % (tests if tests is not None else '?'))
    return dict(status(), self_test_count=tests, source=source)
