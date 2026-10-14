"""障害物よけ r2 の書き出し(候補・出さない)。2026-10-14 学習の段 obstacle r2。

protocol publication.priority_2_3_not_accepted のとおり、合格していない優先 3 のモードは登録簿にも bundle/ にも置かない。
r1 と同じ形で candidates/obstacle/ に置く(policy.json・mode.json・eval.json・status.json(state not_published)・F-ID)。r1 の候補は candidates/obstacle_r1/ に写して残す。

    python3 rl/ob2/export_candidate_r2.py --policy <policy.json> --selection <selection.json> --eval <選びの組の元のコアの結果.json> --fid <fid.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
REP = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--fid", type=Path, required=True)
    a = ap.parse_args()
    cand = EXP / "candidates" / "obstacle"
    r1 = EXP / "candidates" / "obstacle_r1"
    if not r1.exists():
        shutil.copytree(cand, r1)
    sel = json.loads(a.selection.read_text())
    ev = json.loads(a.eval.read_text())
    psha = sha(a.policy)
    assert psha == sel["policy_sha256"] == ev["policy"]["sha256"], "方策の sha256 が選び・評価と合いません"
    modes = json.loads((REP / "modes.json").read_text())
    m = next(x for x in modes["modes"] if x["key"] == "obstacle")
    shutil.copyfile(a.policy, cand / "policy.json")
    shutil.copyfile(a.fid, cand / "fid_std_validation000.json")
    best = next(c for c in sel["candidates"] if c["checkpoint"] == sel["selected"])
    mode_json = {"schema": "jf-sim-rl-mode/v1", "key": "obstacle", "name_ja": m["name_ja"], "short_ja": m["short_ja"], "order": m["order"], "priority": m["priority"],
                 "course": m["course"], "obs_wrapper": m["policy"]["obs_wrapper"], "runtime": m["runtime"],
                 "recording": {**m["recording"], "start_rule_r2": "記録の組の配置ではゲート 0(アプリの既定の STD)から始めない。開始はいつも Halton + 前後 1.5 m の検査(protocol_obstacle_r2.json recording.start_rule)"},
                 "tags": m["tags"], "ui_phrase": m.get("ui_phrase"), "policy_sha256": psha,
                 "policy_source": f"sim_rl_modes_20261013 train/obstacle/r2/{sel['selected'].replace('r2b/', 'r2b/main/policies/') if sel['selected'].startswith('r2b/') else 'main/policies/' + sel['selected']}.json"
                                  f"(obstacle r2、種 62、main_s1_10M から通しで {best['step']:,} ステップ = 前半 obstacle_r2 701 万 + 後半 obstacle_r2b。protocol_obstacle_r2.json・protocol_obstacle_r2b.json)",
                 "modes_json_sha256": sha(REP / "modes.json"), "protocol_sha256": sha(REP / "protocol.json"),
                 "protocol_obstacle_r2_sha256": sha(REP / "protocol_obstacle_r2.json"), "protocol_obstacle_r2b_sha256": sha(REP / "protocol_obstacle_r2b.json")}
    (cand / "mode.json").write_text(json.dumps(mode_json, ensure_ascii=False, indent=1) + "\n")
    slim = {"schema": ev["schema"], "created_at": ev["created_at"], "mode": "obstacle", "set": "selection(OB-V100-R2、試験の組ではない)", "core": ev["core"],
            "policy": {"path": str(a.policy), "sha256": psha, "schema": "jf-sim-rl-mode-policy/v1", "obs_dim": 95},
            "judged": ev["judged"], "all_judged_pass": ev["all_judged_pass"], "reported": {k: ev["reported"][k] for k in ("var6", "failures", "valid_by_start", "valid_by_count",
            "valid_min_clearances_lowest", "valid_below_3cm", "passages", "lap1_valid")},
            "protocol_validation_27": best.get("protocol_validation"), "runs": ev["runs"], "protocol_sha256": sha(REP / "protocol.json"),
            "note_ja": "選びの組(試験と同じ大きさの新しい配置 50 × 2 開始)の結果。試験の組 OB-T100 は、選んだものが選びの組で合格しなかったので使っていない(2 回の枠が残っている)。画面の数字には使わない"}
    (cand / "eval.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1) + "\n")
    failed = [k for k, v in ev["judged"].items() if not v["pass"]]
    status = {"schema": "jf-sim-rl-mode-status/v1", "key": "obstacle", "state": "not_published", "label_ja": "不合格・出さない(参考の候補)", "failed": failed, "pending": [],
              "set": "selection", "round": "r2",
              "rule": "protocol.json publication.priority_2_3_not_accepted: 優先 3 で合格していないので出さない(報告だけ)。登録簿・bundle には置いていない",
              "note_ja": (f"r2(通しで {best['step']:,} ステップ)の選んだ方策。選びの組 OB-V100-R2(元のコア)で 有効 {ev['judged']['O1_valid']['value']}(O1 は 95 が要る)・"
                          f"有効な回の最小すき間 {ev['judged']['O2_clearance']['value'] * 100:.2f} cm(O2 は 3 cm)・1 周目の中央値 {ev['judged']['O3_lap1_median']['value']:.2f} 秒・"
                          f"障害物なしの 1 周 {ev['judged']['O4_empty']['value']:.3f} 秒。/home/jetson/jetformer_reports/sim_rl_modes_20261013/train_obstacle.md"),
              "updated_at": datetime.now().isoformat(timespec="seconds"), "previous_round": "candidates/obstacle_r1/(r1、4ba77f44…06bb、not_published)"}
    (cand / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"out": str(cand), "policy_sha256": psha, "failed": failed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
