"""合格しなかったモードの方策を、ブラウザで読める形の「候補」として書き出す(2026-10-14、学習の段 line)。

protocol.json publication.priority_2_3_not_accepted は「出さない(報告だけ)」なので、登録簿(data/sim_rl/modes/)にも統合の段が置く bundle/ にも書かない。
ここは candidates/<key>/ に、bundle と同じ形(mode.json・policy.json・eval.json・status.json)で置く。status の state は 'not_published'。
ユーザーや本体の段が「参考として使う」と決めたときにそのまま使えるようにするためのもので、自動では使われない。

    python rl/export_candidate.py --mode line --policy <policy.json> --eval <eval_test.json> [--source "..."] [--note "..."]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from publish_mode import REPORTS, WRAPPER_DIM, sha256  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--source", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    modes = json.loads((REPORTS / "modes.json").read_text())
    m = next(x for x in modes["modes"] if x["key"] == a.mode)
    w = json.loads(a.policy.read_text())
    wrapper = m["policy"]["obs_wrapper"]
    if w.get("obs_dim") != WRAPPER_DIM[wrapper] or w.get("schema") != "jf-sim-rl-mode-policy/v1":
        raise SystemExit("方策の形がモードと合いません")
    psha = sha256(a.policy)
    ev = json.loads(a.eval.read_text())
    if ev.get("mode") != a.mode or ev.get("policy", {}).get("sha256") != psha or ev.get("set") != "test" or ev.get("core") != "original":
        raise SystemExit("eval.json は同じ方策・試験の組・元のコアの結果でなければなりません")
    judged = ev["judged"]
    failed = [k for k, v in judged.items() if v.get("pass") is False]
    out = a.out or HERE.parent / "candidates" / a.mode
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.policy, out / "policy.json")
    mode_json = {"schema": "jf-sim-rl-mode/v1", "key": a.mode, "name_ja": m["name_ja"], "short_ja": m["short_ja"], "order": m["order"], "priority": m["priority"],
                 "course": m["course"], "obs_wrapper": wrapper, "runtime": m["runtime"], "recording": m["recording"], "tags": m["tags"],
                 "ui_phrase": m.get("ui_phrase"), "policy_sha256": psha, "policy_source": a.source,
                 "modes_json_sha256": sha256(REPORTS / "modes.json"), "protocol_sha256": sha256(REPORTS / "protocol.json")}
    (out / "mode.json").write_text(json.dumps(mode_json, ensure_ascii=False, indent=1) + "\n")
    slim = {k: ev[k] for k in ("schema", "created_at", "mode", "set", "core", "policy", "judged", "all_judged_pass", "reported", "runs", "protocol_sha256") if k in ev}
    (out / "eval.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1) + "\n")
    status = {"schema": "jf-sim-rl-mode-status/v1", "key": a.mode, "state": "not_published" if failed else "candidate",
              "label_ja": "不合格(出さない・参考の候補)" if failed else "候補", "failed": failed, "pending": [],
              "rule": "protocol.json publication.priority_2_3_not_accepted: 優先 2・3 で不合格は出さない(報告だけ)。登録簿・bundle には置いていない",
              "note_ja": a.note, "updated_at": datetime.now().isoformat(timespec="seconds")}
    (out / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"out": str(out), "state": status["state"], "failed": failed, "policy_sha256": psha}, ensure_ascii=False))


if __name__ == "__main__":
    main()
