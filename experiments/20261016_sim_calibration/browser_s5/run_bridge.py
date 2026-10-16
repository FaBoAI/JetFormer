#!/usr/bin/env python3
# (sim_calibration_20261016 の写し、2026-10-16: 写しの場所を sim_calibration_20261016/sim、ポートを 8831〜8839 に変えた)
"""(sim_diverse_train_20261015 の写し: コピーの場所を sim_diverse_train_20261015/sim、ポートを 8821〜8829 に変えた。元は sim_dart_20261014 の写し)コピーしたシミュレーター(CourseLab)のブリッジを、指定したポートで起動する(2026-10-13、強化学習のベストラップ)

    source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer
    python -I ~/jetformer/experiments/20261013_sim_rl_bestlap/run_bridge.py --port 8791 --no-browser

  - ユーザーのシミュレーター(/home/jetson/sim/extracted/course-simulator)は変更しない。読むのは
    /home/jetson/jetformer_reports/sim_rl_bestlap_20261013/sim/ にコピーしたもの(dist はバイト単位で同じ)。
  - python -I(隔離モード)で起動し、コピーのフォルダーだけを明示的に sys.path に足して bridge.py を実行する
    (カレントディレクトリや PYTHONPATH のモジュールは読まない)。
  - ポートは 8811〜8819 だけ(別の作業が 8766 などを使うため)。収集したデータは sim/runtime/datasets/ に入る。
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

SIM_DIR = Path('/home/jetson/jetformer_reports/sim_calibration_20261016/sim')
PORTS = range(8831, 8840)


def main():
    args = sys.argv[1:]
    port = None
    for i, a in enumerate(args):
        if a == '--port' and i + 1 < len(args):
            port = int(args[i + 1])
        elif a.startswith('--port='):
            port = int(a.split('=', 1)[1])
    if port not in PORTS:
        sys.exit(f'--port は 8831〜8839 を指定してください(指定: {port})')
    if '--policy' in args:
        # 外部ポリシーは PWM 換算を渡すだけの calibration_only_policy.py に限る
        # (RL の方策はブラウザ内の教師モジュールとして動かす)
        i = args.index('--policy')
        pol = Path(args[i + 1]).expanduser()
        pol = (Path(__file__).resolve().parent / pol).resolve() if not pol.is_absolute() else pol.resolve()
        if pol != Path(__file__).resolve().parent / 'calibration_only_policy.py':
            sys.exit('--policy は calibration_only_policy.py だけを指定できます')
        args[i + 1] = str(pol)
    sys.path.insert(0, str(SIM_DIR))
    sys.argv = [str(SIM_DIR / 'bridge.py'), *args]
    runpy.run_path(str(SIM_DIR / 'bridge.py'), run_name='__main__')


if __name__ == '__main__':
    main()
