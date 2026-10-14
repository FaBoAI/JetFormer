"""昇格の段(2026-10-15、r3/promote): 昇格した 3 モード(no_wall・recover・line)の見本を、記録の段と同じ道(record/record_plan.py)で記録する。

記録の段の record/record_plan.py をそのまま使い、置き場所だけを r3 に替える:
  結果・ログ → jetformer_reports/sim_rl_modes_20261013/r3/promote/record/、画面 → r3/screens/record_*.png、
  Firefox のプロファイル → r3/promote/record/ff_profile(終わったら消す)、geckodriver のポート 4448。
操作は同じ(http://192.168.55.1:8080 → シミュレーター タブ → カード「強化学習モード」→ 分 → 「取り込んだら、元のシーンを消す」を外す →
「プランを作って開く」→ 新しいタブで「開始」)。

  jf-heavy --mem 2500M --lock big --need-mb 1800 -- python -u r3/promote/record_plan_r3.py --minutes no_wall=5,recover=5,line=5 --keep-sources
  (途中で止まったら)  ... --resume <プランの ID>
"""
import importlib.util
import sys
from pathlib import Path

EXP = Path('/home/jetson/jetformer/experiments/20261013_sim_rl_modes')
spec = importlib.util.spec_from_file_location('record_plan', EXP / 'record' / 'record_plan.py')
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

R3 = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3')
rp.OUT = R3 / 'promote' / 'record'
rp.SCREENS = R3 / 'screens'
rp.PROFILE = rp.OUT / 'ff_profile'
rp.bc.GECKO_PORT = 4448

if __name__ == '__main__':
    rp.main()
    sys.exit(0)
