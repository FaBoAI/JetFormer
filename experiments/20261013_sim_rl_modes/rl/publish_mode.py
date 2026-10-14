"""モードの登録の一式(mode.json・policy.json・eval.json・status.json)を作る(2026-10-13、sim_rl_modes)。

統合の段の登録簿 data/sim_rl/modes/<key>/ の形(design.md 9.3 節)。この段では既定で experiments/…/bundle/<key>/ に書く(--registry で登録簿へ)。
検査: 方策の schema・obs_dim がモードの観測の包みと合う、eval.json が同じ方策(sha256)・試験の組(test)・元のコア(original)で作られた、
protocol.json の publication の決まり(合格 → accepted、優先 1 で不合格 → not_accepted「不合格(参考)」、優先 2・3 で不合格 → 出さない)。

    python rl/publish_mode.py --mode fastest --policy <policy.json> --eval <eval_test.json> [--source "..."] [--out DIR | --registry]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
REGISTRY = Path("/home/jetson/jetformer/data/sim_rl/modes")
WRAPPER_DIM = {"base79": 79, "base79v": 79, "cond97": 97, "obst95": 95}


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(mode: str, policy: Path, ev_path: Path | None, source: str, out: Path, allow_pending: bool = True, overrides: dict | None = None) -> dict:
    modes = json.loads((REPORTS / "modes.json").read_text())
    proto = json.loads((REPORTS / "protocol.json").read_text())
    m = next(x for x in modes["modes"] if x["key"] == mode)
    w = json.loads(policy.read_text())
    wrapper = m["policy"]["obs_wrapper"]
    if w.get("obs_dim") != WRAPPER_DIM[wrapper]:
        raise SystemExit(f"方策の obs_dim {w.get('obs_dim')} が {mode} の観測の包み {wrapper}({WRAPPER_DIM[wrapper]})と違います")
    if w.get("schema") not in ("rl-bestlap-policy/v1", "jf-sim-rl-mode-policy/v1"):
        raise SystemExit("方策の schema が違います: " + str(w.get("schema")))
    psha = sha256(policy)
    ev = json.loads(ev_path.read_text()) if ev_path else None
    if ev:
        if ev.get("mode") != mode or ev.get("policy", {}).get("sha256") != psha:
            raise SystemExit("eval.json が違うモードか違う方策のものです")
        if ev.get("set") != "test" or ev.get("core") != "original":
            raise SystemExit("eval.json は試験の組(test)・元のコア(original)の結果でなければなりません")
    judged = (ev or {}).get("judged", {})
    pending = [k for k, v in judged.items() if v.get("pass") is None]
    failed = [k for k, v in judged.items() if v.get("pass") is False]
    prio = m["priority"]
    if ev is None:
        state = "untested"
    elif failed:
        state = "not_accepted" if prio == 1 else "not_published"
    elif pending:
        state = "pending" if allow_pending else "not_accepted"
    else:
        state = "accepted"
    if state == "not_published":
        print(f"{mode}: 優先 {prio} で不合格({failed})なので出しません(protocol publication)")
        return {"state": state, "failed": failed}
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(policy, out / "policy.json")
    mode_json = {"schema": "jf-sim-rl-mode/v1", "key": mode, "name_ja": m["name_ja"], "short_ja": m["short_ja"], "order": m["order"], "priority": prio,
                 "course": m["course"], "obs_wrapper": wrapper, "runtime": m["runtime"], "recording": m["recording"], "tags": m["tags"],
                 "ui_phrase": m.get("ui_phrase"), "policy_sha256": psha, "policy_source": source,
                 "modes_json_sha256": sha256(REPORTS / "modes.json"), "protocol_sha256": sha256(REPORTS / "protocol.json")}
    # modes.json は変えずに、登録の一式だけで上書きする値(no_wall r2: short_ja・記録の前の予想のすき間・選べないシナリオ。deviations.md に書く)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(mode_json.get(k), dict):
            mode_json[k] = {**mode_json[k], **v}
        else:
            mode_json[k] = v
    if overrides:
        mode_json["overrides_from_modes_json"] = sorted(overrides)
    (out / "mode.json").write_text(json.dumps(mode_json, ensure_ascii=False, indent=1) + "\n")
    if ev:
        slim = {k: ev[k] for k in ("schema", "created_at", "mode", "set", "core", "policy", "judged", "all_judged_pass", "reported", "runs", "protocol_sha256") if k in ev}
        # 画面の文言に合否を足す(不合格なら「・不合格(参考)」。protocol publication.priority_1_not_accepted)
        rp = slim.get("reported") or {}
        if rp.get("ui_text") and mode == "no_wall":
            rp["ui_text_base"] = rp["ui_text"]
            if failed:
                rp["ui_text"] = rp["ui_text"] + "・不合格(参考)"
        if ev.get("shield"):
            slim["shield"] = ev["shield"]
        (out / "eval.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1) + "\n")
    status = {"schema": "jf-sim-rl-mode-status/v1", "key": mode, "state": state, "label_ja": {"accepted": "合格", "not_accepted": "不合格(参考)",
              "pending": "確認待ち", "untested": "未試験"}[state], "failed": failed, "pending": pending, "updated_at": datetime.now().isoformat(timespec="seconds")}
    (out / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1) + "\n")
    print(f"{mode}: {state} → {out}")
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--eval", type=Path, default=None)
    ap.add_argument("--source", default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--registry", action="store_true", help="登録簿 data/sim_rl/modes/<key>/ に書く(統合の段)")
    ap.add_argument("--overrides", default=None, help="mode.json に上書きする値(JSON の文字列かファイル)。deviations.md に書くこと")
    a = ap.parse_args()
    out = REGISTRY / a.mode if a.registry else (a.out or HERE.parent / "bundle" / a.mode)
    ov = None
    if a.overrides:
        ov = json.loads(Path(a.overrides).read_text()) if Path(a.overrides).exists() else json.loads(a.overrides)
    build(a.mode, a.policy, a.eval, a.source, out, overrides=ov)


if __name__ == "__main__":
    sys.exit(main())
