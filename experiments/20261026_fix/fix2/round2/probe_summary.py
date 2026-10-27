"""直す 2 回目 safety2-1: 簡単な ESC の模型(tests/test_recovery_back_guard.esc_sim)で OFF・ON を並べる(報告の表)"""
import json
import sys
sys.path.insert(0, '/home/jetson/jetformer')
sys.path.insert(0, '/home/jetson/jetformer/tests')
import test_recovery_back_guard as T  # noqa: E402
out = {}
for name, (ready, wall) in {'R_wall(直前ブレーキ・後ろ 5 cm に壁)': (True, 0.05), 'R_free(直前ブレーキ・壁なし)': (True, None),
                            'F_free(直前前進・壁なし)': (False, None), 'F_wall(直前前進・後ろ 5 cm)': (False, 0.05)}.items():
    for g in (False, True):
        r = T.esc_sim(g, ready, wall)
        out['%s %s' % (name, 'ON' if g else 'OFF')] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != 'rows'}
print(json.dumps(out, ensure_ascii=False, indent=1))
