"""昇格の段(2026-10-15): 記録の段の record/record_figs.py の図に「経路いろいろ(line)」を足して描く(元のファイルは変えない)。
    python -I r3/promote/record_figs_r3.py --figdata ... --variation ... --outdir ... [--replay k=...]
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

EXP = Path('/home/jetson/jetformer/experiments/20261013_sim_rl_modes')
spec = importlib.util.spec_from_file_location('record_figs', EXP / 'record' / 'record_figs.py')
rf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf)

KEYS = ('teacher', 'fastest', 'no_wall', 'recover', 'speed', 'line')
rf.COLORS['line'] = '#7d4fc4'
rf.NAMES['line'] = '経路いろいろ'
rf.NAMES['no_wall'] = '絶対壁にぶつからない(r4)'
rf.NAMES['recover'] = '離脱復活(r4)'
rf.order = lambda fig: [k for k in KEYS if k in fig['groups']]


def summary(var, out):
    plt = rf.plt
    j = var.get('judged') or {}
    ref = j.get('reference') or {}
    keys = [k for k in KEYS[1:] if k in (var.get('modes') or {})]
    labels = [rf.NAMES[k] for k in keys] + ['全モード合わせ']
    t3, t7 = ref.get('teacher_VM3_m'), ref.get('teacher_VM7_cells_15')
    vm3 = [var['modes'][k]['VM3_m']['mean_abs_dev'] / t3 for k in keys] + [var['pooled']['VM3_m']['mean_abs_dev'] / t3]
    vm7 = [var['modes'][k]['VM7']['cells'] / t7 for k in keys] + [var['pooled']['VM7_equal_frames']['cells'] / t7]
    cols = [rf.COLORS[k] for k in keys] + ['#4a3aa7']
    f, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, vals, ttl, goal in ((axs[0], vm3, 'VM3 線のばらつき(教師 = 1)', 10), (axs[1], vm7, 'VM7 状態の網羅(教師 = 1)', 3)):
        y = np.arange(len(labels))[::-1]
        ax.barh(y, vals, color=cols, height=.62, edgecolor='white', linewidth=2)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f' {v:.1f} 倍', va='center', fontsize=9, color=rf.INK, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
        ax.axvline(1, color=rf.INK2, lw=1)
        ax.axvline(goal, color=rf.INK, lw=1, ls=(0, (4, 3)))
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.set_title(f'{ttl}\n点線 = 全モード合わせの目標 {goal} 倍', loc='left', fontsize=10.5)
        ax.grid(axis='x', color=rf.GRID, lw=.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(max(vals), goal) * 1.18)
    f.suptitle('前の教師のデータ(RC加速教師 60 シーン)と比べたばらつき — 第 4 回の 5 モード', x=.01, ha='left', fontsize=11.5, color=rf.INK)
    f.tight_layout(rect=(0, 0, 1, .93))
    f.savefig(out, dpi=130)
    plt.close(f)


rf.summary = summary

if __name__ == '__main__':
    rf.main()
    sys.exit(0)
