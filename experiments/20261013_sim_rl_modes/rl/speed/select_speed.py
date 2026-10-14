"""速さいろいろ(speed)のチェックポイントの選び方(protocol.json selection のとおり)。2026-10-13、sim_rl_modes の学習の段 speed。

    python rl/speed/select_speed.py --run <cond の run の出力> [--out selection.json]

検証の組(evals/speed/step_N.json、高速版、種 validation_base・Halton 10,001〜、試験の 1/4)で judged を全部満たすチェックポイントのうち一番後のもの。
無ければ満たす項目が一番多いもの(同じなら主な数字 = S3 の差 t(0.5) − t(1.0) が大きいもの、それも同じなら後のもの)。
試験の組は選んだチェックポイントだけに使う(試験の結果を見て選び直さない)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rows = []
    for ev in sorted((a.run / "evals" / "speed").glob("step_*.json")):
        d = json.loads(ev.read_text())
        pol = a.run / "policies" / ev.name
        if not pol.exists() or d["policy"]["sha256"] != sha256(pol):
            raise SystemExit(f"{ev} の方策が policies/{ev.name} と違います")
        J = d["judged"]
        n_pass = sum(1 for v in J.values() if v.get("pass") is True)
        s3 = J["S3_lap_order"]["value"]
        diff = (s3["t5"] - s3["t1"]) if s3.get("t5") is not None and s3.get("t1") is not None else None
        rows.append({"step": int(ev.stem.split("_")[1]), "eval": str(ev), "policy": str(pol), "policy_sha256": d["policy"]["sha256"],
                     "all_judged_pass": d["all_judged_pass"], "n_pass": n_pass, "n_judged": len(J), "s3_diff_s": diff,
                     "judged": {k: {"pass": v.get("pass"), "value": v.get("value")} for k, v in J.items()}})
    if not rows:
        raise SystemExit("検証の結果がありません")
    ok = [r for r in rows if r["all_judged_pass"]]
    if ok:
        best, rule = max(ok, key=lambda r: r["step"]), "検証の組で judged を全部満たすもののうち一番後"
    else:
        best = max(rows, key=lambda r: (r["n_pass"], r["s3_diff_s"] if r["s3_diff_s"] is not None else -1e9, r["step"]))
        rule = "全部を満たすものが無いので、満たす項目が一番多いもの(同じなら S3 の差が大きいもの)"
    out = {"schema": "jf-sim-rl-mode-selection/v1", "mode": "speed", "run": str(a.run), "created_at": datetime.now().isoformat(timespec="seconds"),
           "rule": rule, "selected": best, "candidates": [{k: r[k] for k in ("step", "all_judged_pass", "n_pass", "s3_diff_s")} for r in rows]}
    text = json.dumps(out, ensure_ascii=False, indent=1) + "\n"
    if a.out:
        a.out.write_text(text)
    print(json.dumps({"selected_step": best["step"], "all_judged_pass": best["all_judged_pass"], "rule": rule,
                      "candidates": [(r["step"], r["all_judged_pass"], r["n_pass"]) for r in rows]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
