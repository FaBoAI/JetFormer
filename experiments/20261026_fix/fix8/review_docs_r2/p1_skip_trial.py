"""反証 8-2 docs: 手順書の「試し」の 2 つの道(走り終わりのメモ・シーンの説明に「試し」/ 紙に「試し」→ main session の `--skip 記録名`)で、
⑤ collect.py の回 3 の答えが変わるかを、本物の ⑤(10/28 06:23 の d2b210cc の写し snap/collect.py)と作り物の日で流す(読むだけ)。

作り物の日は tests/test_fix8_tools_decided.py の mtday(回 3: OFF 4 本・ON 3 本(launch_us 200)・3-8 = ON の本。3-8 が無ければ
ON の中央値 = OFF − 0.190 秒 → 使ってよい)をそのまま使う。3-8 の形:
  hit   最初の周(発進から 2 秒)に水平 6G の当たり(test_c5_trial_collisions_still_counted と同じ)
  spin  空転(比 ≈ 1.8)
  gap   最初の周の途中のシーンの IMU.npz が無い(最初の周の記録の欠け)
  none  不具合なし(比べ)
道: memo = 走り終わりのメモ(かシーンの「走行の説明」)に「試し」(手順書 4 章 9・7 章「起こし直した」の行の終わりの道)
    paper = メモに書けず紙に「試し」→ main session が `--skip 記録名`(手順書 7 章「走った値が計画と違った」「起こし直した」の行の終わり・
            3 章 5・4 章 9・todo 3 章)
    plain = どちらもしない(ふつうの本として数える)

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -I -B experiments/20261026_fix/fix8/review_docs_r2/p1_skip_trial.py
  → out/p1_skip_trial.txt
"""
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAP = HERE / 'snap'
os.environ['JF_FIX8_COLLECT'] = str(SNAP / 'collect.py')
os.environ['JF_FIX8R1_COLLECT'] = str(SNAP / 'collect.py')
os.environ['JF_FIX8_S3'] = str(SNAP / 'standing_3lap.py')
os.environ['JF_FIX8_QP'] = str(SNAP / 'quick_eval_by_params.py')
JF = Path('/home/jetson/jetformer')
for p in (JF, JF / 'tools', JF / 'tests'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import test_fix8_tools_decided as t8  # noqa: E402

TMP = JF / 'experiments/20261026_fix/fix8/tmp_review_docs_r2'
IG = ['--ignore-gain']


def verdict(g):
    c = g.get('compare') or {}
    if not c:
        return '比べなし'
    return '使ってよい' if c.get('all_ok') else '満たさない'


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    lines = ['# ⑤ の版: snap/collect.py %s(本物の experiments/20261026_fix/diag_launch/collect.py の 10/28 06:23 の写し)'
             % (SNAP / 'SHA256SUMS').read_text().split()[0][:12],
             '# 回 3: OFF 4 本・ON 3 本(launch_us 200)+ 3-8(ON・launch_us 200。計画と違って走った本 = 手順書 7 章「走った値が計画と違った」)']
    shapes = (('none', 'nospin', {}), ('hit', 'nospin', {'hits': [(3.0, 6.0)]}), ('spin', 'spin', {}), ('gap', 'gap', {}))
    try:
        for name, shape, kw in shapes:
            for way in ('memo', 'paper', 'plain'):
                tmp = TMP / ('%s_%s' % (name, way))
                D, log = t8.mtday(tmp, 'trial' if way == 'memo' else 'none', shape, **kw)
                args = IG + (['--skip', log] if way == 'paper' else [])
                j, out, _ = t8.runc(D, 'new', args, '%s%s' % (name, way))
                g = t8.g0(j)
                c = g.get('compare') or {}
                bad = sorted(k for k, v in (c.get('checks') or {}).items() if v is not True)
                cmp_line = [x for x in out.splitlines() if x.startswith('  ON − OFF')]
                skip_line = [x for x in out.splitlines() if 'skip' in x.lower() and log in x][:1]
                lines.append('3-8 %-4s 道 %-5s → %s(満たさない条件 %s。trial_on_collisions %s・trial_slip %s・first_lap_gap_on %s・'
                             'summary_skipped %s)' % (name, way, verdict(g), bad or '—', g.get('trial_on_collisions'),
                                                     bool(g.get('trial_slip')), bool(g.get('first_lap_gap_on')),
                                                     j.get('summary_skipped') if isinstance(j, dict) else None))
                for x in cmp_line[:1]:
                    lines.append('      行: ' + x.strip()[:260])
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    txt = '\n'.join(lines) + '\n'
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'p1_skip_trial.txt').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
