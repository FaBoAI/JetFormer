"""反証用(w1-ui-r7-ui-2): 試験の台本を「長いタイマーを捨てない」写しに差し替えて、同じ試験を流す。本物のファイルは変えない。"""
import sys
from pathlib import Path
D = Path(__file__).resolve().parent
MAP = {'test_ui_fix6_r2': 'keep_app_sim_fix6_r2.js', 'test_ui_fix6': 'keep_app_sim_fix6.js', 'test_ui_fix5': 'keep_app_sim_fix5.js'}


def pytest_collection_finish(session):
    for mod, f in MAP.items():
        m = sys.modules.get(mod)
        if m is not None and hasattr(m, 'SIM'):
            m.SIM = D / f
            print('SIM ->', mod, m.SIM)
