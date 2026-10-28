"""打ち消しの票 A(指摘 f8r2-docs-paper-trial-skip-drops-faults を確かめる)。直さない。

本物の ⑤ collect.py(今の版を snap/ に写した。sha256 は snap/SHA256SUMS)と ③④(tools/standing_3lap.py・quick_eval_by_params.py の写し)を、
作り物の回 3(tests/test_fix8_tools_decided.mtday と同じ作り方: OFF 4 本・ON 3 本(launch_us 200)+ 3-8 = ON)で流す。
3-8 の不具合の形(shape): hit = 最初の周の 3.0 秒に 6G / spin = 空転(IMU の前の加速 0.55 倍) / gap = 最初の周の IMU.npz が無い /
clean = 不具合なし。読み方(path):
  memo       = 走り終わりのメモに「試し」(説明欄の memo=試し(…))= 手順書のふつうの道
  plain      = 何も書かない(試しにしないで数えた)
  paper_skip = 紙に「試し」だけ → main session が手順書 7 章どおり `--skip 3-8 の記録名`
形の違い(form): same = 3-8 の値は計画と同じ / gain = 3-8 は throttle_gain 0.92(ほかは 0.90。「ゲインを取り違えた」)
読みは手順書 8.1 ⑤ と同じ --ignore-gain。作り物の日は fix8/tmp_refute_docs_r2_A/skip に作って消す。
    jf-heavy … python -B p_skip_trial.py → 標準出力・out/p_skip_trial.json
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAP = HERE / 'snap'
for h, name in (x.split() for x in (SNAP / 'SHA256SUMS').read_text().splitlines() if x.strip()):
    assert hashlib.sha256((SNAP / name).read_bytes()).hexdigest() == h, name
os.environ['JF_FIX8_COLLECT'] = str(SNAP / 'collect.py')
os.environ['JF_FIX8_S3'] = str(SNAP / 'standing_3lap.py')
os.environ['JF_FIX8_QP'] = str(SNAP / 'quick_eval_by_params.py')
JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
import test_fix8_tools_decided as T  # noqa: E402

TMP = JF / 'experiments' / '20261026_fix' / 'fix8' / 'tmp_refute_docs_r2_A' / 'skip'
SHAPES = {'hit': ('nospin', {'hits': [(3.0, 6.0)]}), 'spin': ('spin', {}), 'gap': ('gap', {}), 'clean': ('nospin', {})}


def verdict(out):
    v = ['使ってよい' if '→ 使ってよい' in x else '満たさない' for x in out.splitlines()
         if x.strip().startswith('ON − OFF') and ('→ 使ってよい' in x or '→ 満たさない' in x)]
    return '・'.join(v) if v else '?'


def main():
    res = []
    if TMP.exists():
        shutil.rmtree(TMP)
    try:
        k = 0
        for shape, (sh, kw) in SHAPES.items():
            for form in ('same', 'gain'):
                for path in ('memo', 'plain', 'paper_skip'):
                    k += 1
                    tmp = TMP / ('c%02d' % k)
                    D, log = T.mtday(tmp, 'trial' if path == 'memo' else 'none', sh, **kw)
                    if form == 'gain':
                        T.edit_log(D.raw, log, lambda j: j['drive']['params'].update(throttle_gain=0.92))
                    args = list(T.IG) + (['--skip', log] if path == 'paper_skip' else [])
                    j, out, _ = T.runc(D, 'new', args, 'k%d' % k)
                    gs = j['summary']
                    groups = [{'session': g.get('session'), 'model': g.get('model'), 'all_ok': (g.get('compare') or {}).get('all_ok'),
                               'false_checks': sorted(c for c, v in ((g.get('compare') or {}).get('checks') or {}).items() if v is False),
                               'd_to_r1_s': (g.get('compare') or {}).get('d_to_r1_s'),
                               'trial_on_collisions': g.get('trial_on_collisions'),
                               'trial_slip': g.get('trial_slip'), 'first_lap_gap_on': g.get('first_lap_gap_on')} for g in gs]
                    row = {'shape': shape, 'form': form, 'path': path, 'log': log, 'verdict': verdict(out), 'groups': groups,
                           'skipped': j.get('summary_skipped'),
                           'trial_line': [x.strip()[:80] for x in out.splitlines() if '試しの' in x][:2]}
                    res.append(row)
                    print('%-5s %-4s %-10s %s | %s' % (shape, form, path, row['verdict'],
                                                      ' / '.join('ok=%s false=%s d=%s trial_coll=%s' % (
                                                          g['all_ok'], g['false_checks'], g['d_to_r1_s'], g['trial_on_collisions'])
                                                          for g in groups)), flush=True)
                    shutil.rmtree(tmp, ignore_errors=True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
        try:
            TMP.parent.rmdir()
        except OSError:
            pass
    (HERE / 'out' / 'p_skip_trial.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
