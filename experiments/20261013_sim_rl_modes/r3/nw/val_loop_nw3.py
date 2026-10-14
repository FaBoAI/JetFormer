"""no_wall r3: チェックポイントの検証の組 NW-V-R3(シールド v4)を走らせて判定する(2026-10-14)。止めても同じコマンドで続きから。

選びの決まり(protocol_no_wall_r3.json selection.rule = 全部を満たす点のうち一番後)に合わせて、いつも「まだ調べていない一番後の点」から調べる:
  - 学習の途中: できた点のうち一番後のもの(検証は 1 本が学習より遅いので、途中の点は飛ぶことがある)
  - 学習のあと: 後ろから順に。「全部を満たす点」が見つかり、それより後の点が全部調べてあれば、それが選ぶ点なので止める
    (それより前の点は決まりに関係しない)。見つからなければ全部の点を調べる(満たす数で選ぶため)
    python3 val_loop_nw3.py [--jobs 2]
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

E = Path("/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/nw")
O = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/no_wall_r3")
KEYS = ["N1_r3", "N2_r3", "N3_r3", "N4", "N5_r3", "N6", "N7_r3", "stall_zero"]
ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=2)
a = ap.parse_args()


def evaluated(st):
    return (O / "val" / st / "eval_validation.json").exists()


def passes(st):
    J = json.loads((O / "val" / st / "eval_validation.json").read_text())["judged"]
    return all(J[k]["pass"] for k in KEYS)


while True:
    pols = sorted((O / "train/policies").glob("step_*.json"))
    done = (O / "train/done.json").exists()
    names = [p.stem for p in pols]
    todo = [p for p in pols if not evaluated(p.stem)]
    if done:
        # 後ろから: 調べてある一番後の「全部を満たす点」より後が全部調べてあれば終わり
        stop = False
        for n in reversed(names):
            if not evaluated(n):
                break
            if passes(n):
                stop = True
                break
        if stop or not todo:
            print(time.strftime("[%H:%M:%S]"), "選びに要る点を調べ終えました", flush=True)
            break
    if not todo:
        time.sleep(60)
        continue
    p = todo[-1]
    V = O / "val" / p.stem
    V.mkdir(parents=True, exist_ok=True)
    print(time.strftime("[%H:%M:%S]"), "検証", p.stem, flush=True)
    r = subprocess.run(["/home/jetson/bin/jf-heavy", "--mem", "1500M", "--lock", "none", "--need-mb", "700", "--", "nice", "-n", "15", "node",
                        str(E / "run_sets_nw3.mjs"), "--which", "validation", "--policy", str(p), "--out", str(V), "--jobs", str(a.jobs)], cwd=str(E))
    if r.returncode != 0:
        print("検証が失敗しました", p.stem, r.returncode, flush=True)
        time.sleep(30)
        continue
    j = subprocess.run(["python3", str(E / "judge_nw3.py"), "--rows", str(V / "rows_validation.jsonl"), "--which", "validation", "--out", str(V / "eval_validation.json")],
                       capture_output=True, text=True)
    (V / "judge.txt").write_text(j.stdout + j.stderr)
    print(j.stdout.strip()[:600], flush=True)
