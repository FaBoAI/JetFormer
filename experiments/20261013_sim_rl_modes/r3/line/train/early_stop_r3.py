"""経路いろいろ r3 の早めの終わりの判定(protocol_line_r3.json training.early_stop)。

300 万ステップ以降、続けて 3 つのチェックポイント(100 万ごと、全部が評価済み)が検証の組 LN-V60-R3 + LN-VHEAD40-R3 + LN-FAM6 で
L1_r3〜L8 を全部満たしたら 0 を返す(run_line_r3.sh が学習に止める合図を送る)。満たさなければ 1。

    python3 r3/line/train/early_stop_r3.py <学習の出力の場所>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MIN_STEP = 3_000_000
NEED = 3


def passing_streak(out: Path) -> tuple[bool, list]:
    ev = out / "evals" / "line"
    rows = []
    for p in sorted((out / "policies").glob("step_*.json")):
        step = int(p.stem.split("_")[1])
        e = ev / p.name
        try:
            d = json.loads(e.read_text()) if e.exists() else None
        except Exception:  # noqa: BLE001 (書きかけ)
            d = None
        rows.append((step, None if d is None else bool(d.get("all_judged_pass"))))
    done = []
    for step, ok in rows:   # 評価済みの続いた並び(途中に未評価があればそこで切る)
        if ok is None:
            break
        done.append((step, ok))
    tail = done[-NEED:]
    ok = len(tail) == NEED and all(x for _, x in tail) and tail[0][0] >= MIN_STEP
    return ok, done


if __name__ == "__main__":
    ok, done = passing_streak(Path(sys.argv[1]))
    print(json.dumps({"stop": ok, "evaluated": [[s, x] for s, x in done[-6:]]}))
    sys.exit(0 if ok else 1)
