#!/usr/bin/env python3
"""fix9 tonight-plan 係(2026-10-29): 今夜(10/29)の日付で走ったとき、夜の道具が動くかを作り物の記録で確かめる。

    cd /home/jetson/jetformer/experiments/20261026_fix/fix9/tonight/tonight-plan
    PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1200M --need-mb 2000 -- \
        /home/jetson/miniconda/envs/jetracer/bin/python sim_dates.py

- 作り物の記録は tests/test_fix7_tools.py の write_log(試験と同じ作り方のコース。距離で曲がりを決める)で、この場所の sim/ に作る。
  本物の data/raw・data/labels・data/bench_candidates.json は読むだけ(bench_candidates.json は写しを sim/<形>/data/ に置く)。
- 形(どれも同じ 9 本: 回 3 の OFF・ON 交互の 8 本 + 止まって 3 周の 1 本。発進・速さ・衝突は同じ):
    a1103  11/3 の本番の形: 記録の名前 20261103-13xxxx・説明欄 day=2026-11-03 block=12〜20(手順書どおり。比べのもと)
    b1029  今夜、カードのボタンで 11/03 のブロックを押した形: 記録の名前 20261029-19xxxx・説明欄は a と同じ day=2026-11-03
    c1029  今夜、「基準の設定」で走った形: 記録の名前 20261029-19xxxx・説明欄に bench・day・block が無い(battery・V_before だけ)
- 道具(どれも読むだけ。出力はこの場所の out/<形>/):
    ② runs_1103.py(RAW・JF をこの場所の写しに差し替えて main)・③ standing_3lap.py・④ quick_eval_by_params.py・
    ⑤ diag_launch/collect.py(RAW・LAB・BENCH・OUT を差し替え。--ignore-gain)・car_check.py --latest
- 本物のコードは変えない(モジュールを読み込んで、その中の場所の変数だけを差し替える。試験 tests/test_fix7_tools.py の runc と同じ)。
"""
import contextlib
import io
import json
import shutil
import sys
import traceback
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF / 'tests'))
import test_fix7_tools as T  # noqa: E402
from test_fix3_r1_2 import load_mod  # noqa: E402

HERE = Path(__file__).resolve().parent
SIM = HERE / 'sim'
OUT = HERE / 'out'
B_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'
COLLECT = JF / 'experiments' / '20261026_fix' / 'diag_launch' / 'collect.py'
RUNS = JF / 'experiments' / '20261026_fix' / 'plan' / 'runs_1103.py'
PRM = dict(T.PRM)
ON = dict(T.ON)
# 回 3 の 8 本(tests/test_fix7_tools.py の round3 の 'N' と同じ動き)と止まって 3 周の 1 本(4-1。発進の前進 OFF・40 秒)
BOOKS = [(12, PRM, 0.20, 0.6, 30.0), (13, ON, 0.15, 0.3, 30.0), (14, PRM, 0.21, 0.6, 30.0), (15, ON, 0.20, 0.3, 30.0),
         (16, PRM, 0.22, 0.6, 30.0), (17, ON, 0.28, 0.3, 30.0), (18, PRM, 0.23, 0.6, 30.0), (19, ON, 0.31, 0.3, 30.0),
         (20, PRM, 0.22, 0.6, 40.0)]
FORMS = {
    'a1103': ('20261103', '13', 'ai_no=%d bench=real_bench_20261023 day=2026-11-03 block=%d cand=EXP model=%s '
              'battery=#A V_before=8.30 V_after=__ check=__ memo='),
    'b1029': ('20261029', '19', 'ai_no=%d bench=real_bench_20261023 day=2026-11-03 block=%d cand=EXP model=%s '
              'battery=#A V_before=8.30 V_after=__ check=__ memo='),
    'c1029': ('20261029', '19', 'ai_no=%d battery=#A V_before=8.30'),
}


def build(form):
    day, hh, tpl = FORMS[form]
    root = SIM / form
    if root.exists():
        shutil.rmtree(root)
    raw = root / 'data' / 'raw'
    raw.mkdir(parents=True)
    (root / 'data' / 'labels').mkdir()
    shutil.copy2(JF / 'data' / 'bench_candidates.json', root / 'data' / 'bench_candidates.json')    # 写し(読むだけの元)
    logs = []
    for n, (block, prm, d_move, ramp, dur) in enumerate(BOOKS, 1):
        log = '%s-%s%02d00' % (day, hh, n * 3)
        vk = T.launch_keys(1.0, d_move, ramp, until=dur)
        ai = [(1.0, dur - 2.0)]
        launch = [(1.0, 0.5)] if prm.get('launch_boost') else []
        desc = tpl % ((n, block, B_RUN) if '%s' in tpl else (n,))
        T.write_log(raw, log, dur, vk, ai, prm, desc, launch=launch, run=B_RUN)
        logs.append(log)
    return root, day, logs


def capture(fn):
    buf, err = io.StringIO(), io.StringIO()
    rc = None
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        try:
            rc = fn()
        except SystemExit as e:
            rc = 'SystemExit(%s)' % e.code
        except Exception:                      # noqa: BLE001  落ちたら落ちた形を残す
            rc = 'EXC'
            traceback.print_exc()
    return rc, buf.getvalue(), err.getvalue()


def run_tools(form):
    root, day, logs = build(form)
    raw, lab, bench = root / 'data' / 'raw', root / 'data' / 'labels', root / 'data' / 'bench_candidates.json'
    out = OUT / form
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    res = {'form': form, 'day': day, 'logs': logs}
    # ② runs_1103(--day は今夜の日)
    m2 = load_mod('runs_1103_%s' % form, RUNS)
    m2.JF, m2.RAW = root, raw
    saved = sys.argv
    sys.argv = ['runs_1103.py', '--day', day, '--json']
    try:
        res['r2'] = capture(m2.main)
    finally:
        sys.argv = saved
    # ③ standing_3lap
    import standing_3lap as s3
    res['r3'] = capture(lambda: s3.main(['--day', day, '--raw', str(raw), '--labels', str(lab), '--bench', str(bench),
                                         '--json', str(out / 'standing_3lap.json'), '--md', str(out / 'standing_3lap.md')]))
    # ④ quick_eval_by_params
    import quick_eval_by_params as qp
    res['r4'] = capture(lambda: qp.main(['0000', '--day=%s' % day, '--raw', str(raw), '--labels', str(lab), '--bench', str(bench),
                                         '--out', str(out / 'by_params')]))
    # ⑤ collect --ignore-gain
    m5 = load_mod('collect_%s' % form, COLLECT)
    m5.RAW, m5.LAB, m5.BENCH, m5.OUT = raw, lab, bench, out
    res['r5'] = capture(lambda: m5.main(['--day', day, '--ignore-gain', '--out', str(out / 'launches_ig.json')]))
    # car_check --latest(その日のいちばん新しい本)
    import car_check as cc
    res['cc'] = capture(lambda: cc.main(['--latest', '--day', day, '--raw', str(raw), '--labels', str(lab),
                                         '--json', str(out / 'car_check_latest.json')]))
    for k in ('r2', 'r3', 'r4', 'r5', 'cc'):
        rc, so, se = res[k]
        (out / ('%s.stdout.txt' % k)).write_text(so, encoding='utf-8')
        (out / ('%s.stderr.txt' % k)).write_text(se, encoding='utf-8')
        res[k] = {'rc': rc, 'stdout_lines': len(so.splitlines()), 'stderr_tail': se.splitlines()[-5:]}
    return res


def main():
    OUT.mkdir(exist_ok=True)
    allres = {}
    for form in FORMS:
        allres[form] = run_tools(form)
        print(form, json.dumps({k: (v['rc'] if isinstance(v, dict) else v) for k, v in allres[form].items()
                                if k.startswith(('r', 'cc'))}, ensure_ascii=False))
    (OUT / 'summary.json').write_text(json.dumps(allres, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
