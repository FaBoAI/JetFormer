"""障害物よけ r2 の早めの終わりの判定(protocol_obstacle_r2.json training.early_stop)。

800 万ステップ以降、続けて 3 つのチェックポイント(100 万ごと、全部が評価済み)が OB-V100-R2 で O1〜O4 を全部満たし、
かつ有効 ≥ 98/100 なら 0 を返す(train/obstacle/r2/chain.sh が学習に止める合図を送る)。満たさなければ 1。

    python3 rl/ob2/early_stop_r2.py <学習の出力の場所>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MIN_STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 8_000_000   # r2b(後半)は 100 万 = 通しで 800 万
NEED = 3
MIN_VALID = 98


def streak(out: Path) -> tuple[bool, list]:
    rows = []
    for p in sorted((out / "policies").glob("step_*.json")):
        step = int(p.stem.split("_")[1])
        e = out / "evals_r2" / p.name
        try:
            d = json.loads(e.read_text())
        except Exception:  # noqa: BLE001 (まだ無い・書きかけ)
            rows.append((step, None))
            continue
        rows.append((step, bool(d.get("all_judged_pass")) and int(d["judged"]["O1_valid"]["n"]) >= MIN_VALID))
    done = []
    for step, ok in rows:
        if ok is None:
            break
        done.append((step, ok))
    tail = done[-NEED:]
    return len(tail) == NEED and all(x for _, x in tail) and tail[0][0] >= MIN_STEP, done


if __name__ == "__main__":
    ok, done = streak(Path(sys.argv[1]))
    print(json.dumps({"stop": ok, "evaluated": [[s, x] for s, x in done[-6:]]}))
    sys.exit(0 if ok else 1)
