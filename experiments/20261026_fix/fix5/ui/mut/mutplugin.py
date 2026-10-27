"""fix5 画面の係: わざと壊した写しで tests/test_ui_fix5.py を流すための pytest の差し込み(本物のファイルは変えない)。
MUT_STATIC = 壊した app.js と index.html の場所・MUT_RUNS = 壊した runs_1103.py・MUT_SERVER_NOCHECK = 1 なら server._boot_refusal を無効に"""
import os
import pathlib


def pytest_collection_finish(session):
    import test_ui_fix5 as T
    if os.environ.get('MUT_STATIC'):
        T.STATIC = pathlib.Path(os.environ['MUT_STATIC'])
    if os.environ.get('MUT_RUNS'):
        T.RUNS = pathlib.Path(os.environ['MUT_RUNS'])
    if os.environ.get('MUT_SERVER_NOCHECK'):
        from jetformer.web import server as srv
        srv._boot_refusal = lambda body, boot: None
