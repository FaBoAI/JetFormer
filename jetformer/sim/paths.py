"""シミュレーターのデータの置き場所(すべて config.DATA_DIR の下。テストは config.DATA_DIR を差し替えるだけで切り替わる)"""

import os
import threading
from pathlib import Path

from .. import config

HERE = Path(__file__).resolve().parent
STATIC = HERE / 'static'                       # JetFormer が持つ shim・汎用描画など
PKG_DATA = HERE / 'coursedata'                 # 既知のコースの情報・Default の physics-world(data/ は .gitignore に当たるので別名)
RUNNER = HERE / 'bridge_runner.py'             # ブリッジの起動スクリプト(venv の Python で動かす)
REQUIREMENTS = HERE / 'requirements-bridge.txt'

# 導入(install)と準備(setup)は同時に 1 本だけ
OP_LOCK = threading.Lock()


def sim_dir():
    return Path(config.DATA_DIR) / 'sim'


def releases_dir():
    return sim_dir() / 'releases'


def active_file():
    return sim_dir() / 'active.json'


def runtime_dir():
    return sim_dir() / 'runtime'


def venv_dir():
    return sim_dir() / 'venv'


def venv_python():
    return venv_dir() / 'bin' / 'python'


def venv_record():
    return sim_dir() / 'venv.json'


def wheels_dir():
    return sim_dir() / 'wheels'


def logs_dir():
    return sim_dir() / 'logs'


def courses_dir():
    return Path(config.DATA_DIR) / 'sim_courses'


def inbox_dir():
    """シミュレーター本体の ZIP を置くフォルダー(既定 ~/sim。JETFORMER_SIM_INBOX で変えられる)"""
    return Path(os.environ.get('JETFORMER_SIM_INBOX') or (Path.home() / 'sim'))


# ---------------------------------------------------------------- 電源断に強い書き方(review で追加)
def fsync_dir(d):
    """フォルダーの中の名前の変更(作成・置き換え)をディスクに書き切る。できなくても続ける"""
    try:
        fd = os.open(str(d), os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def fsync_file(p):
    """書き終えたファイルの中身をディスクに書き切る"""
    with open(p, 'rb') as f:
        os.fsync(f.fileno())


def write_durable(path, data):
    """<path>.tmp に書いて fsync してから置き換え、フォルダーも fsync する。
    電源断や強制終了のあとに 0 バイトや途中までのファイル(記録だけあって中身がない状態)を残さないため"""
    path = Path(path)
    if isinstance(data, str):
        data = data.encode('utf-8')
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)
