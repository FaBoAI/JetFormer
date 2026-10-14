"""障害物よけ r2 のチェックポイントの選び(protocol_obstacle_r2.json selection.rule)。2026-10-14 学習の段 obstacle r2。

    python3 rl/ob2/select_r2.py <書き出し selection.json> <学習の出力>[@通しのステップのずれ] ...
    例: python3 rl/ob2/select_r2.py r2/selection.json r2/main r2/r2b/main@7012352

前半(r2/main)と後半(r2/r2b/main、通しのステップ = 後半のステップ + 7,012,352)の全部のチェックポイントのうち、選びの組 OB-V100-R2(evals_r2/)で O1〜O4 を全部満たすものの中で (1) EMPTY-VAR6 が全部 9.0 秒以内 (2) 有効な回が多い (3) O2 が大きい (4) 後のもの。
全部を満たすものが無ければ、満たす項目が一番多いもの(同じなら有効な回の数・O2・後のもの)。protocol の検証の組(evals/obstacle/)の値は報告に並べる。
試験の組を使う条件(protocol_obstacle_r2b.json selection_addition.strict_valid): 選んだものが、有効だが 1 周も回っていない回を無効に数えても O1 を満たすこと。
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    out = Path(sys.argv[1])
    rows = []
    for arg in sys.argv[2:]:
        path, _, off = arg.partition("@")
        rows += collect(Path(path), int(off or 0))
    finish(rows, out)


def collect(run: Path, offset: int) -> list:
    rows = []
    for p in sorted((run / "policies").glob("step_*.json")):
        e = run / "evals_r2" / p.name
        if not e.exists():
            continue
        d = json.loads(e.read_text())
        j = d["judged"]
        v = run / "evals" / "obstacle" / p.name
        pv = json.loads(v.read_text()) if v.exists() else None
        ok = [r for r in d["rows"] if r["test"] == "OB-V100-R2" and r["valid_episode"]]
        nolap = sum(1 for r in ok if len(r["laps"]) < 1)
        rows.append({"checkpoint": f"{run.parent.name}/{p.stem}" if offset else p.stem, "step": int(p.stem.split("_")[1]) + offset, "run": str(run), "policy": str(p), "policy_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                     "eval_r2": str(e), "all_pass": bool(d["all_judged_pass"]), "n_pass": int(d["n_pass"]), "valid": int(j["O1_valid"]["n"]), "valid_no_lap": nolap, "valid_strict": int(j["O1_valid"]["n"]) - nolap,
                     "strict_O1_pass": int(j["O1_valid"]["n"]) - nolap >= 95,
                     "O2": j["O2_clearance"]["value"], "O3": j["O3_lap1_median"]["value"], "O4": j["O4_empty"]["value"],
                     "var6_max": d["reported"]["var6"]["max"], "var6_ok": bool(d["reported"]["var6"]["all_le_9"]),
                     "early_collisions": d["reported"]["failures"]["early_le_2_1s"], "VM3": (d["reported"].get("VM3_valid_runs") or {}).get("mean_abs_dev"),
                     "protocol_validation": None if pv is None else {"O1": pv["judged"]["O1_valid"]["value"], "O2": pv["judged"]["O2_clearance"]["value"],
                                                                     "O3": pv["judged"]["O3_lap1_median"]["value"], "O4": pv["judged"]["O4_empty"]["value"],
                                                                     "all_pass": pv["all_judged_pass"]}})
    return rows


def finish(rows: list, out: Path) -> None:
    if not rows:
        raise SystemExit("評価済みのチェックポイントがありません")
    o2 = lambda r: r["O2"] if r["O2"] is not None else -1.0  # noqa: E731
    passing = [r for r in rows if r["all_pass"]]
    if passing:
        best = max(passing, key=lambda r: (r["var6_ok"], r["valid"], o2(r), r["step"]))
        how = "OB-V100-R2 で O1〜O4 を全部満たすもののうち、EMPTY-VAR6 が全部 9.0 秒以内 → 有効な回が多い → O2 が大きい → 後のもの"
    else:
        best = max(rows, key=lambda r: (r["n_pass"], r["valid"], o2(r), r["step"]))
        how = "全部を満たすものが無いので、満たす項目が一番多いもの(同じなら有効な回の数 → O2 → 後のもの)"
    res = {"schema": "jf-sim-rl-mode-selection/v1", "mode": "obstacle", "round": "r2", "selected_at": datetime.now().isoformat(timespec="seconds"),
           "rule": "protocol_obstacle_r2.json selection.rule", "how": how, "selected": best["checkpoint"], "policy": best["policy"],
           "policy_sha256": best["policy_sha256"], "selected_passes_selection_set": best["all_pass"],
           "selected_strict_O1_pass": best["strict_O1_pass"], "test_allowed": best["all_pass"] and best["strict_O1_pass"], "candidates": rows}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("selected", "policy_sha256", "selected_passes_selection_set", "selected_strict_O1_pass", "test_allowed", "how")}, ensure_ascii=False))
    for r in rows:
        print(f"{r['checkpoint']}({r['step'] / 1e6:.2f}M): {'全部合格' if r['all_pass'] else str(r['n_pass']) + '/4'} 有効 {r['valid']}/100(周 0 {r['valid_no_lap']})O2 {o2(r) * 100:.2f} cm O3 {r['O3']} O4 {r['O4']:.3f} "
              f"VAR6 {r['var6_max']:.3f} 早い衝突 {r['early_collisions']} VM3 {r['VM3']} 検証27 {r['protocol_validation'] and r['protocol_validation']['O1']}")


if __name__ == "__main__":
    main()
