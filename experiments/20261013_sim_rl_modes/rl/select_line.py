"""経路いろいろ(line)のチェックポイントの選び方(protocol.json selection のとおり。2026-10-13、学習の段 line)。

    python rl/select_line.py <run_dir> [<run_dir> ...] [--mode line]

各 run の evals/<mode>/step_N.json(検証の組・高速版。train_modes.py --eval が書く)を読み、
「検証の組で judged を全部満たすチェックポイントのうち一番後のもの。無ければ満たす項目が一番多いもの
(同じなら主な数字: line = VM3、speed = S3 の差)」を選ぶ。複数の run は渡した順に続けたものとして並べる(ステップ数は run ごと)。
結果は JSON で標準出力へ(選んだ方策の policies/step_N.json のパスを含む)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main_number(mode: str, J: dict) -> float:
    if mode == "line":
        v = J.get("L5_vm3", {}).get("value")
        return float(v) if v is not None else -1.0
    if mode == "speed":
        v = (J.get("S3_lap_order", {}) or {}).get("value") or {}
        t1, t5 = v.get("t1"), v.get("t5")
        return float(t5 - t1) if t1 is not None and t5 is not None else -1.0
    return 0.0


def main() -> None:
    args = sys.argv[1:]
    mode = "line"
    if "--mode" in args:
        i = args.index("--mode")
        mode = args[i + 1]
        del args[i:i + 2]
    rows = []
    for order, run in enumerate(Path(a) for a in args):
        for p in sorted((run / "evals" / mode).glob("step_*.json")):
            d = json.loads(p.read_text())
            J = d["judged"]
            js = [v for v in J.values() if v.get("pass") is not None]
            rows.append({"run": str(run), "order": order, "step": int(p.stem.split("_")[1]), "eval": str(p),
                         "policy": str(run / "policies" / (p.stem + ".json")), "all_pass": bool(d["all_judged_pass"]),
                         "n_pass": sum(1 for v in js if v["pass"] is True), "n_judged": len(js), "main": main_number(mode, J),
                         "failed": [k for k, v in J.items() if v.get("pass") is False]})
    if not rows:
        raise SystemExit("検証の結果がありません")
    passing = [r for r in rows if r["all_pass"]]
    if passing:
        best = max(passing, key=lambda r: (r["order"], r["step"]))
        rule = "検証の組で judged を全部満たすチェックポイントのうち一番後のもの"
    else:
        best = max(rows, key=lambda r: (r["n_pass"], r["main"], r["order"], r["step"]))
        rule = "全部を満たすものが無いので、満たす項目が一番多いもの(同じなら主な数字 " + ("VM3" if mode == "line" else "S3 の差") + " が大きいもの)"
    print(json.dumps({"mode": mode, "rule": rule, "selected": best, "candidates": rows}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
