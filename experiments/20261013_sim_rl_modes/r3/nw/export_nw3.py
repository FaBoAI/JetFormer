"""no_wall r3 の候補の一式を書き出す(2026-10-14、r3 の学習の段)。bundle/ と登録簿には書かない(昇格の段が使う)。

    python3 export_nw3.py --policy <選んだ policy.json> --eval <試験の eval_test.json> --selection <selection.json> --fid-dir <F-ID の置き場所>
        [--out experiments/20261013_sim_rl_modes/r3/candidates/no_wall_r3]

書くもの(bundle と同じ形 + r3 の走行中の仕組み):
  policy.json・mode.json(modes.json の no_wall に r3 の上書き: runtime.shield_params = jf-shield/v4 の値・記録の前の予想の決まり・選べないシナリオ・short_ja)・
  eval.json(試験の組の判定。不合格なら ui_text に「・不合格(参考)」)・status.json(合格 → accepted「合格」、不合格 → not_accepted「不合格(参考)」。
  protocol publication: no_wall は優先 1 なので不合格でも数字を見せて出す)・shield_config.json(シールド v4 の値と shield.mjs の sha256)・
  runtime/(r3/nw/runtime_v4 の写し = 走らせた仕組みそのもの。ブラウザの組 web/rl を作り直すのは昇格の段)・fid_*.json(F-ID)・
  bundle_status_fix.json(元の一式 bundle/no_wall/status.json を正す中身。r2 の学習の段が accepted を書いたが確認 r2 は不合格。昇格の段が置く)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

REP = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
EXP = Path("/home/jetson/jetformer/experiments/20261013_sim_rl_modes")
RT = EXP / "r3/nw/runtime_v4"


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--fid-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=EXP / "r3/candidates/no_wall_r3")
    a = ap.parse_args()
    modes = json.loads((REP / "modes.json").read_text())
    prot3 = json.loads((REP / "r3/protocol_no_wall_r3.json").read_text())
    m = next(x for x in modes["modes"] if x["key"] == "no_wall")
    w = json.loads(a.policy.read_text())
    if w.get("obs_dim") != 79 or w.get("schema") != "jf-sim-rl-mode-policy/v1":
        raise SystemExit("方策の形が no_wall(base79)と合いません")
    psha = sha256(a.policy)
    ev = json.loads(a.eval.read_text())
    if ev.get("mode") != "no_wall" or ev["policy"]["sha256"] != psha or ev.get("set") != "test" or ev.get("core") != "original":
        raise SystemExit("eval.json は同じ方策・試験の組・元のコアの結果でなければなりません")
    if ev.get("protocol_no_wall_r3_sha256") != sha256(REP / "r3/protocol_no_wall_r3.json"):
        raise SystemExit("eval.json の追補の決まりの sha256 が今のファイルと違います")
    sel = json.loads(a.selection.read_text())
    if sel["selected"]["policy_sha256"] != psha:
        raise SystemExit("選んだ点と方策が違います")
    judged = ev["judged"]
    keys = ev["judged_keys"]
    failed = [k for k in keys if judged[k]["pass"] is False]
    state = "not_accepted" if failed else "accepted"
    label = "不合格(参考)" if failed else "合格"
    out = a.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copyfile(a.policy, out / "policy.json")
    params = prot3["runtime"]["shield"]["params"]
    if ev["shield"]["version"] != "jf-shield/v4.1" or ev["shield"]["shield_mjs_sha256"] != sha256(RT / "shield.mjs"):
        raise SystemExit("試験の shield.mjs が今の runtime_v4 と違います")
    shield_params = {k: v for k, v in ev["shield"]["params_used"].items() if k != "backups"}   # backups は説明の文字列(v3 のまま)なので値に入れない
    rt_files = {p.name: sha256(p) for p in sorted(RT.iterdir()) if p.is_file()}
    shutil.copytree(RT, out / "runtime")
    ui = ev["reported"]["ui_numbers"]
    overrides = {
        "short_ja": "壁・柱からはなれて遅めに走る(記録するシーンは前もって走らせ、すき間が 8 cm 未満・止まる・1 周できない・シールドの作動が 25% を超えるものは替える)。"
                    "物理のずれ(舵の速さ・μ・遅れ・RC の応答・摩擦円)でもぶつからないよう学習し、走行中は安全シールド(v4.1)が舵の遅れとブレーキを見積もり、"
                    "摩擦円・低い μ・弱いブレーキの悪い端でも止まれるかを見張る。遅くてよい",
        "runtime": {"shield": True, "shield_version": "jf-shield/v4.1", "shield_params": shield_params,
                    "shield_note_ja": "jf-shield/v4.1(r3): v3(見積もり・速さの見張り・悲観の模型)に、見積もれない μ・摩擦円・前進のゲインを悪い端にした模型 Q(45 フレーム・7 cm)を"
                                      "「替えない」の条件にも入れ、弱いブレーキのとき速さの上限に 0.92 を掛け、c_nominal を 0.08 m に(protocol_no_wall_r3.json runtime.shield = v4)。"
                                      "v4.1 で、ブレーキの上限を舵の見積もりを待たずにかけ、走り出しで 1 フレームだけ速さを落としてブレーキを見積もる(deviations.md)",
                    "perturbation": None, "conditioning": None, "gate_index": "base79"},
        "recording": {**m["recording"], "prefilter_min_clearance_m": 0.08, "prefilter_min_valid_laps": 1, "prefilter_no_stall": True,
                      "prefilter_max_shield_fraction": 0.25,
                      "prefilter_note_ja": "記録の前の予想(r3/nw/predict_scenes_nw3.mjs と同じ決まり): valid_episode_r3(衝突・途中の終了・止まり 1 秒以上・有効な周 0 は不可)・"
                                           "最小すき間 ≥ 0.08 m・シールドの作動 ≤ 25% でないシーンは次の候補に替える"},
        "scenario_blocklist": ["low-friction"],
        "scenario_note_ja": "アプリの「低摩擦」シナリオ(μ × 0.4)は試験した物理の範囲の外で、ぶつかります。強化学習モードでは選べないようにするか、選んだら警告してください",
        "ui_numbers": ui,
    }
    mode_json = {"schema": "jf-sim-rl-mode/v1", "key": "no_wall", "name_ja": m["name_ja"], "short_ja": m["short_ja"], "order": m["order"], "priority": m["priority"],
                 "course": m["course"], "obs_wrapper": m["policy"]["obs_wrapper"], "runtime": m["runtime"], "recording": m["recording"], "tags": m["tags"],
                 "ui_phrase": m.get("ui_phrase"), "policy_sha256": psha,
                 "policy_source": f"no_wall r3: r3/no_wall_r3/train {sel['selected']['checkpoint']}(r2 の選んだ点 step_004014080 から種 23 で続けた学習。"
                                  f"物理の隅・むずかしい開始・止まりの見張り。protocol_no_wall_r3.json training)。走行中のシールド jf-shield/v4",
                 "modes_json_sha256": sha256(REP / "modes.json"), "protocol_sha256": sha256(REP / "protocol.json"),
                 "protocol_no_wall_r3_sha256": sha256(REP / "r3/protocol_no_wall_r3.json")}
    for k, v in overrides.items():
        mode_json[k] = v
    mode_json["overrides_from_modes_json"] = sorted(overrides)
    (out / "mode.json").write_text(json.dumps(mode_json, ensure_ascii=False, indent=1) + "\n")
    slim = {k: ev[k] for k in ("schema", "created_at", "mode", "set", "core", "core_note", "policy", "judged", "judged_keys", "all_judged_pass", "runs",
                               "protocol_sha256", "protocol_no_wall_r3_sha256", "shield") if k in ev}
    rp = dict(ev["reported"])
    rp.pop("sets", None)
    rp["ui_text_base"] = rp["ui_text"]
    if failed:
        rp["ui_text"] = rp["ui_text"] + "・不合格(参考)"
    slim["reported"] = rp
    slim["sets_summary"] = {k: {x: y for x, y in v.items() if x in ("n", "collisions", "min_clearance_m", "below_5cm", "below_8cm", "valid_episode_r3", "stall",
                                                                      "shield_fraction", "lap1_median_s")} for k, v in ev["reported"]["sets"].items()}
    (out / "eval.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1) + "\n")
    status = {"schema": "jf-sim-rl-mode-status/v1", "key": "no_wall", "state": state, "label_ja": label, "failed": failed, "pending": [],
              "round": "r3", "candidate": True,
              "note_ja": ("r3 の試験(protocol_no_wall_r3.json、1 回だけ)で全部合格。昇格の段で bundle/no_wall/ と登録簿をこの一式で置き換える" if not failed else
                          "r3 の試験(protocol_no_wall_r3.json、1 回だけ)で不合格。優先 1 なので「不合格(参考)」として数字を見せる。昇格の段で、"
                          "元の一式(r2)と r3 の候補のどちらを「不合格(参考)」で出すか決める"),
              "updated_at": datetime.now().isoformat(timespec="seconds")}
    (out / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1) + "\n")
    (out / "shield_config.json").write_text(json.dumps({"version": "jf-shield/v4.1", "params": shield_params, "protocol_params_v4": params,
                                                        "v41_governor_extra": ev["shield"]["params_used"]["governor"],
                                                        "shield_mjs_sha256": rt_files["shield.mjs"], "runtime_files_sha256": rt_files,
                                                        "runtime_note_ja": "runtime/ は web/rl(manifest aa74e23f…)の写しに shield.mjs(v4)を替え、mode_runtime.mjs のシールドの記録に"
                                                                           "模型 Q の予想(wq)と速さの上限(cap)を足したもの(記録だけ。操作は同じ)。manifest.json は写したときのまま(shield.mjs・"
                                                                           "mode_runtime.mjs の sha256 とは合わない。ブラウザの組は昇格の段が rl/make_web_bundle.mjs で作り直す)",
                                                        "v3_equivalent_params": {"c_nominal": 0.07, "worst": {"enabled": False},
                                                                                 "governor": {"weak_extra": 1, "brake_without_known": False, "brake_probe": {"enabled": False}}},
                                                        "v4_equivalent_params": {"governor": {"brake_without_known": False, "brake_probe": {"enabled": False}}}},
                                                       ensure_ascii=False, indent=1) + "\n")
    for f in sorted(a.fid_dir.glob("fid_*.json")):
        shutil.copyfile(f, out / f.name)
    # 元の一式(bundle/no_wall/status.json、r2 の学習の段が accepted を書いた)を正す中身。昇格の段が置く(この段は bundle を変えない)
    fix = {"schema": "jf-sim-rl-mode-status/v1", "key": "no_wall", "state": "not_accepted", "label_ja": "不合格(参考)",
           "failed": ["check_r2_corner_collisions", "check_r2_N2_fresh", "check_r2_N5_stall"], "pending": [],
           "note_ja": "r2 の学習の段の試験(340 回)は合格だったが、確認 r2(check_no_wall_r2.md)で不合格: DR-T の範囲の中の隅(速い舵・弱いブレーキ・μ 0.85・摩擦円・"
                      "前進のゲイン 1.04)で衝突(頂点 480 回で 10)、標準の物理の新しい開始で 7.0 cm(N2 は試験の組の引き当て)、止まったまま(有効な周 0)のシーンが記録の前の確かめを通る",
           "applies_to": "experiments/20261013_sim_rl_modes/bundle/no_wall/status.json(r2 の一式、方策 153c709c…)",
           "written_by": "no_wall r3 の学習の段(2026-10-14)。昇格の段で置く", "updated_at": datetime.now().isoformat(timespec="seconds")}
    if not failed:
        fix["superseded_by_r3"] = "r3 の一式(この候補、accepted)で bundle/no_wall/ と登録簿を置き換えるなら、この中身は使わない"
    (out / "bundle_status_fix.json").write_text(json.dumps(fix, ensure_ascii=False, indent=1) + "\n")
    files = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / "sha256.json").write_text(json.dumps(files, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"out": str(out), "state": state, "failed": failed, "policy_sha256": psha, "ui_text": rp["ui_text"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
