"""経路いろいろ r4 の早めの終わりの判定(protocol_line_r4.json training.early_stop)。r3/line/train/early_stop_r3.py の写し。

200 万ステップ以降、続けて 2 つのチェックポイント(100 万ごと、全部が評価済み)が検証の組 LN-V60-R4 + LN-VHEAD40-R4 + LN-VCORNER-R4 +
LN-VBOX40-R4 + LN-FAM6 で L1_r4〜L10 を全部満たしたら 0 を返す(run_line_r4.sh が学習に止める合図を送る)。満たさなければ 1。
初期化元(step_000000000)は数えない。

    python3 r3/line4/train/early_stop_r4.py <学習の出力の場所>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MIN_STEP = 2_000_000
NEED = 2


def passing_streak(out: Path) -> tuple[bool, list]:
    ev = out / "evals" / "line"
    rows = []
    for p in sorted((out / "policies").glob("step_*.json")):
        step = int(p.stem.split("_")[1])
        if step == 0:
            continue
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
