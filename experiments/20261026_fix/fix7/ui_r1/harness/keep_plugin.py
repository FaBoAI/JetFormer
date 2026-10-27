"""fix7 画面の係(w1-ui-r7-ui-2 の確かめ): 前の試験の台本を、短いタイマーを鳴らすときに長いタイマーを捨てない写し(この場所の keep/)に
差し替えて、前の試験をそのまま流す。本物の台本・試験・app.js は変えない。JF_KEEP_VARIANT=keep(runShortTimers だけ直す)か
keepdrv(さらに app_sim_fix6_r2 の ['ntimers'] から上の状態の輪 pollStatus・ライブの値の輪 pollValues を除く)"""
import os
import sys
from pathlib import Path

D = Path(__file__).resolve().parent / 'keep'
MAP = {'test_bench_ui_round2': 'app_sim_round2.js', 'test_ui_fix3': 'app_sim_fix3.js', 'test_ui_fix3_r1': 'app_sim_fix3_r1.js',
       'test_ui_fix3_r1_2': 'app_sim_fix3_r1_2.js', 'test_ui_fix3_r2_1': 'app_sim_fix3_r2_1.js', 'test_ui_fix3_r3_1': 'app_sim_fix3_r3_1.js',
       'test_ui_fix5': 'app_sim_fix5.js', 'test_ui_fix6': 'app_sim_fix6.js', 'test_ui_fix6_r2': 'app_sim_fix6_r2.js'}


def pytest_collection_finish(session):
    v = os.environ.get('JF_KEEP_VARIANT', 'keep')
    for mod, f in MAP.items():
        m = sys.modules.get(mod)
        if m is not None and hasattr(m, 'SIM'):
            p = D / ('%s_%s' % (v, f))
            if not p.exists():
                p = D / ('keep_%s' % f)
            m.SIM = p
            print('SIM ->', mod, p.name)
