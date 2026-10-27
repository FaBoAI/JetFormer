"""反証用(w1-ui-r7-ui-2): 前の回の試験の台本も「長いタイマーを捨てない」写しに差し替える。本物のファイルは変えない。"""
import sys
from pathlib import Path
D = Path(__file__).resolve().parent
MAP = {'test_ui_fix3': 'keep_app_sim_fix3.js', 'test_ui_fix3_r1': 'keep_app_sim_fix3_r1.js', 'test_ui_fix3_r1_2': 'keep_app_sim_fix3_r1_2.js',
       'test_ui_fix3_r2_1': 'keep_app_sim_fix3_r2_1.js', 'test_ui_fix3_r3_1': 'keep_app_sim_fix3_r3_1.js', 'test_bench_ui_round2': 'keep_app_sim_round2.js'}


def pytest_collection_finish(session):
    for mod, f in MAP.items():
        m = sys.modules.get(mod)
        if m is not None and hasattr(m, 'SIM'):
            m.SIM = D / f
            print('SIM ->', mod, m.SIM)
