"""経路いろいろ r2 の早めの終わりの判定(protocol_line_r2.json training.early_stop)。

800 万ステップ以降、続けて 3 つのチェックポイント(100 万ごと、全部が評価済み)が検証の組 LN-V60-R2 + LN-FAM6 で
L1〜L6 を全部満たしたら 0 を返す(run_line_r2.sh が学習に止める合図を送る)。満たさなければ 1。

    python3 rl/line_r2/early_stop.py <学習の出力の場所>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MIN_STEP = 8_000_000
NEED = 3


def passing_streak(out: Path) -> tuple[bool, list]:
    ev = out / "evals" / "line"
    pol = sorted((out / "policies").glob("step_*.json"))
    rows = []
    for p in pol:
        step = int(p.stem.split("_")[1])
        e = ev / p.name
        if not e.exists():
            rows.append((step, None))
            continue
        try:
            d = json.loads(e.read_text())
        except Exception:  # noqa: BLE001 (書きかけ)
            rows.append((step, None))
            continue
        rows.append((step, bool(d.get("all_judged_pass"))))
    # 評価済みの続いた並び(途中に未評価があればそこで切る)
    done = []
    for step, ok in rows:
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
