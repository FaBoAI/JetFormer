"""no_wall r4b: 選びの候補(protocol_no_wall_r4b.json selection)を検証の組 NW-V-R4(シールド v5)で走らせて判定する(2026-10-14 20 時、電源断のあとの再開で足した)。
val_loop_nw4.py の写しで、候補に r4 の学習(種 24、100 万で止めた train_a/)の点を足し、決まりの順(r4b の点(ステップの順)> r4 の点 > step_000000000)で
後ろから調べる(21 時: protocol_no_wall_r4c.json で r4c の点(train_c/、val/c_step_*)を一番後の順で足した)。「全部を満たす点」が見つかり、それより順が後の点が全部調べてあれば、それが選ぶ点なので止める(それより前の点は決まりに関係しない)。
見つからなければ全部の点を調べる(満たす数で選ぶため)。学習(r4b)が終わってから(train/done.json)走らせる。止めても同じコマンドで続きから。

    python3 val_loop_nw4b.py [--jobs 5]
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

E = Path("/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/nw4")
O = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/no_wall_r4")
KEYS = ["N1_r4", "N2_r4", "N3_r4", "N4", "N5_r4", "N6", "N7_r4", "stall_zero"]
ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=3)
a = ap.parse_args()


def candidates():
    """(名前, 方策のパス, 順の鍵)。名前は val/ の下のフォルダー名。順の鍵が大きいほど「後」"""
    c = []
    for p in sorted((O / "train/policies").glob("step_*.json")):
        s = int(p.stem.split("_")[1])
        c.append((p.stem, p, (2, s) if s > 0 else (0, 0)))
    for p in sorted((O / "train_a/policies").glob("step_*.json")):
        s = int(p.stem.split("_")[1])
        if s > 0:
            c.append(("a_" + p.stem, p, (1, s)))
    # r4c(protocol_no_wall_r4c.json、21 時): train_c/ の点を順 3 で(step_000000000 は r4b の終わりの点と同じ方策なので足さない)
    for p in sorted((O / "train_c/policies").glob("step_*.json")):
        s = int(p.stem.split("_")[1])
        if s > 0:
            c.append(("c_" + p.stem, p, (3, s)))
    return sorted(c, key=lambda x: x[2])


def evaluated(n):
    return (O / "val" / n / "eval_validation.json").exists()


def passes(n):
    J = json.loads((O / "val" / n / "eval_validation.json").read_text())["judged"]
    return all(J[k]["pass"] for k in KEYS)


if not (O / "train/done.json").exists():
    raise SystemExit("r4b の学習が終わっていません(train/done.json が無い)")
if (O / "train_c").exists() and not (O / "train_c/done.json").exists():
    raise SystemExit("r4c の学習が終わっていません(train_c/done.json が無い)")
while True:
    C = candidates()
    todo = None
    for n, p, k in reversed(C):
        if not evaluated(n):
            todo = (n, p)
            break
        if passes(n):
            break
    if todo is None:
        print(time.strftime("[%H:%M:%S]"), "選びに要る点を調べ終えました", flush=True)
        break
    n, p = todo
    V = O / "val" / n
    V.mkdir(parents=True, exist_ok=True)
    print(time.strftime("[%H:%M:%S]"), "検証", n, p, flush=True)
    r = subprocess.run(["/home/jetson/bin/jf-heavy", "--mem", "1800M", "--lock", "none", "--need-mb", "700", "--", "nice", "-n", "15", "node",
                        str(E / "run_sets_nw4.mjs"), "--which", "validation", "--policy", str(p), "--out", str(V), "--jobs", str(a.jobs)], cwd=str(E))
    if r.returncode != 0:
        print("検証が失敗しました", n, r.returncode, flush=True)
        time.sleep(30)
        continue
    j = subprocess.run(["python3", str(E / "judge_nw4.py"), "--rows", str(V / "rows_validation.jsonl"), "--which", "validation", "--out", str(V / "eval_validation.json")],
                       capture_output=True, text=True)
    (V / "judge.txt").write_text(j.stdout + j.stderr)
    print(j.stdout.strip()[:600], flush=True)
