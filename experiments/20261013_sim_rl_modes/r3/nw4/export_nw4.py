"""no_wall r4 の候補の一式を書き出す(2026-10-14、r4 の段。r3/nw/export_nw3.py の写しを r4 に合わせたもの)。bundle/ と登録簿には書かない(昇格の段が使う)。

    python3 export_nw4.py --policy <選んだ policy.json> --eval <試験の eval_test.json> --selection <selection.json> --fid-dir <F-ID の置き場所>
        [--out experiments/20261013_sim_rl_modes/r3/candidates/no_wall_r4]

書くもの: policy.json・mode.json(modes.json の no_wall に r4 の上書き: runtime.shield_version jf-shield/v5・shield_params = 決まりの値・記録の前の予想の決まり
  (後ろへ動いたシーンも替える)・選べないシナリオ・short_ja・ui_numbers・dataset_card_notes_ja)・eval.json・status.json・shield_config.json・
  runtime/(r3/nw4/runtime_v5 の写し)・fid_*.json・bundle_status_fix.json(元の一式と r3 の候補の status を正す中身。昇格の段が使う)・sha256.json
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
RT = EXP / "r3/nw4/runtime_v5"
PROT4 = REP / "r3/protocol_no_wall_r4.json"


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--fid-dir", type=Path, required=True)
    ap.add_argument("--regression", type=Path, default=None, help="確認 r3 の組の走らせ直しのまとめ(報告だけ)")
    ap.add_argument("--out", type=Path, default=EXP / "r3/candidates/no_wall_r4")
    a = ap.parse_args()
    modes = json.loads((REP / "modes.json").read_text())
    prot4 = json.loads(PROT4.read_text())
    m = next(x for x in modes["modes"] if x["key"] == "no_wall")
    w = json.loads(a.policy.read_text())
    if w.get("obs_dim") != 79 or w.get("schema") != "jf-sim-rl-mode-policy/v1":
        raise SystemExit("方策の形が no_wall(base79)と合いません")
    psha = sha256(a.policy)
    ev = json.loads(a.eval.read_text())
    if ev.get("mode") != "no_wall" or ev["policy"]["sha256"] != psha or ev.get("set") != "test" or ev.get("core") != "original":
        raise SystemExit("eval.json は同じ方策・試験の組・元のコアの結果でなければなりません")
    if ev.get("protocol_no_wall_r4_sha256") != sha256(PROT4):
        raise SystemExit("eval.json の追補の決まりの sha256 が今のファイルと違います")
    if ev["shield"]["version"] != "jf-shield/v5" or ev["shield"]["shield_mjs_sha256"] != sha256(RT / "shield.mjs") or ev["shield"].get("params_override"):
        raise SystemExit("試験の shield.mjs・値が今の runtime_v5(決まりの既定の値)と違います")
    sel = json.loads(a.selection.read_text())
    if sel["selected"]["policy_sha256"] != psha:
        raise SystemExit("選んだ点と方策が違います")
    judged, keys = ev["judged"], ev["judged_keys"]
    failed = [k for k in keys if judged[k]["pass"] is False]
    state, label = ("not_accepted", "不合格(参考)") if failed else ("accepted", "合格")
    out = a.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copyfile(a.policy, out / "policy.json")
    params = prot4["runtime"]["shield"]["params"]
    rt_files = {p.name: sha256(p) for p in sorted(RT.iterdir()) if p.is_file()}
    shutil.copytree(RT, out / "runtime")
    rp = dict(ev["reported"])
    ui = rp["ui_numbers"]
    sets = rp["sets"]
    rev_runs = {k: v.get("runs_with_reverse", 0) for k, v in sets.items() if v.get("runs_with_reverse")}
    notes = [
        f"r4 の試験(protocol_no_wall_r4.json、{ui['runs']} 回、1 回だけ): 衝突 {ui['collisions']}・最小すき間 {100 * ui['min_clearance_nominal_m']:.2f} cm(標準の物理 {ui['min_clearance_nominal_runs']} 回: "
        f"記録の開始から一様 {ui['min_clearance_nominal_uniform_runs']} 回では {100 * ui['min_clearance_nominal_uniform_m']:.2f} cm、開始のすき間 12〜15 cm の層 400 回・分布の端の格子を含む)・"
        f"{100 * ui['min_clearance_randomised_m']:.2f} cm(物理のずれ)",
        "シールド v5 は標準の物理では予想が sim と同じなので、止まれる操縦を残しながら 8.1 cm を守る(しきい値ぎりぎりまで寄ることがある = 最小すき間はほぼ 8.0〜8.1 cm になる)",
        "確認 r3(シールド v4.1)の数字: 記録の開始から一様 1,000 回で 8.01 cm、まれな開始(ゲート 20・左 −0.13 m・−13°)で衝突 1・隣の開始で 5.63 cm、ゲート 23 の柱の足へ向いた開始で止まり(約 0.05%)。v5 で直した開始の結果は train_no_wall_r4.md",
        "記録の前の予想で、衝突・8 cm 未満・止まり・有効な周 0・シールドの作動 25% 超・後ろへ動いた(抜け出し)シーンは替える(データセットに後ろへ下がる正解ラベルは入らない)",
        "最速と比べて遅く、壁から離れた 1 本の線に集まる(モードの中の線のばらつきは小さい)",
    ]
    overrides = {
        "short_ja": "壁・柱からはなれて遅めに走る(記録するシーンは前もって走らせ、すき間が 8 cm 未満・止まる・1 周できない・シールドの作動が 25% を超える・後ろへ下がったものは替える)。"
                    "物理のずれ(舵の速さ・μ・遅れ・RC の応答・摩擦円)でもぶつからないよう学習し、走行中は安全シールド(v5)が舵の遅れとブレーキを見積もり、"
                    "止まれる距離まで先読みして、摩擦円・低い μ・弱いブレーキの悪い端でも止まれるかを見張る。柱の前で動けなくなったら少し下がってから抜ける。遅くてよい",
        "runtime": {"shield": True, "shield_version": "jf-shield/v5", "shield_params": params,
                    "shield_note_ja": "jf-shield/v5(r4): v4.1(見積もり・模型 Q・速さの見張り・ブレーキの探り)に、E と Q の先読みを止まるまで続ける条件(止まれる操縦がいつも残る)・"
                                      "どの候補も Q に合格しないときは E に合格する候補からブレーキを先に選ぶ・c_nominal 0.081 m・止まって動けないときの抜け出し(前へ / 後ろへ下がってから前へ、"
                                      "全部のしきい値のまま)を足した(protocol_no_wall_r4.json runtime.shield)",
                    "perturbation": None, "conditioning": None, "gate_index": "base79"},
        "recording": {**m["recording"], "prefilter_min_clearance_m": 0.08, "prefilter_min_valid_laps": 1, "prefilter_no_stall": True,
                      "prefilter_max_shield_fraction": 0.25, "prefilter_no_reverse": True,
                      "prefilter_note_ja": "記録の前の予想(r3/nw4/predict_scenes_nw4.mjs と同じ決まり): valid_episode_r3(衝突・途中の終了・止まり 1 秒以上・有効な周 0 は不可)・"
                                           "最小すき間 ≥ 0.08 m・シールドの作動 ≤ 25%・後ろへ動いたフレームが無い、でないシーンは次の候補に替える"},
        "scenario_blocklist": ["low-friction"],
        "scenario_note_ja": "アプリの「低摩擦」シナリオ(μ × 0.4)は試験した物理の範囲の外で、ぶつかります。強化学習モードでは選べないようにするか、選んだら警告してください",
        "ui_numbers": ui,
        "dataset_card_notes_ja": notes,
    }
    mode_json = {"schema": "jf-sim-rl-mode/v1", "key": "no_wall", "name_ja": m["name_ja"], "short_ja": m["short_ja"], "order": m["order"], "priority": m["priority"],
                 "course": m["course"], "obs_wrapper": m["policy"]["obs_wrapper"], "runtime": m["runtime"], "recording": m["recording"], "tags": m["tags"],
                 "ui_phrase": m.get("ui_phrase"), "policy_sha256": psha,
                 "policy_source": f"no_wall r4: {sel['selected']['checkpoint']}(" + ("r3 の選んだ方策そのまま" if sel["selected"]["steps"] == 0 else
                                  "r3 の選んだ点 step_006012928 から種 24 で続けた学習。開始のすき間 12〜15 cm・分布の端の開始を多く") + "。protocol_no_wall_r4.json)。走行中のシールド jf-shield/v5",
                 "modes_json_sha256": sha256(REP / "modes.json"), "protocol_sha256": sha256(REP / "protocol.json"),
                 "protocol_no_wall_r4_sha256": sha256(PROT4)}
    for k, v in overrides.items():
        mode_json[k] = v
    mode_json["overrides_from_modes_json"] = sorted(overrides)
    (out / "mode.json").write_text(json.dumps(mode_json, ensure_ascii=False, indent=1) + "\n")
    slim = {k: ev[k] for k in ("schema", "created_at", "mode", "set", "core", "core_note", "policy", "judged", "judged_keys", "all_judged_pass", "runs",
                               "protocol_sha256", "protocol_no_wall_r4_sha256", "shield") if k in ev}
    rp.pop("sets", None)
    rp["ui_text_base"] = rp["ui_text"]
    if failed:
        rp["ui_text"] = rp["ui_text"] + "・不合格(参考)"
    rp["reverse_runs_by_set"] = rev_runs
    slim["reported"] = rp
    slim["sets_summary"] = {k: {x: y for x, y in v.items() if x in ("n", "collisions", "min_clearance_m", "below_5cm", "below_8cm", "valid_episode_r3", "stall",
                                                                      "shield_fraction", "lap1_median_s", "runs_with_reverse", "runs_with_escape")} for k, v in sets.items()}
    if a.regression and a.regression.exists():
        slim["regression_check_r3_sets"] = json.loads(a.regression.read_text())
    (out / "eval.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1) + "\n")
    now = datetime.now().isoformat(timespec="seconds")
    status = {"schema": "jf-sim-rl-mode-status/v1", "key": "no_wall", "state": state, "label_ja": label, "failed": failed, "pending": [],
              "round": "r4", "candidate": True,
              "note_ja": ("r4 の試験(protocol_no_wall_r4.json、1 回だけ)で全部合格。確認の段のあと、昇格の段で bundle/no_wall/ と登録簿をこの一式で置き換える" if not failed else
                          "r4 の試験(protocol_no_wall_r4.json、1 回だけ)で不合格。優先 1 なので「不合格(参考)」として数字を見せる。昇格の段で、どの一式を出すか決める"),
              "updated_at": now}
    (out / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1) + "\n")
    (out / "shield_config.json").write_text(json.dumps({
        "version": "jf-shield/v5", "params": params, "shield_mjs_sha256": rt_files["shield.mjs"], "runtime_files_sha256": rt_files,
        "v41_equivalent_params": {"c_nominal": 0.08, "stop": {"enabled": False}, "select": "v4", "cont_candidates": False, "escape": {"enabled": False}},
        "runtime_note_ja": "runtime/ は r3/nw/runtime_v4(web/rl の写し + v4.1)の shield.mjs を v5 に替えたもの(ほかのファイルは r3 と同じ。mode_runtime.mjs はシールドの記録に wq・cap)。"
                           "manifest.json は写したときのまま(shield.mjs・mode_runtime.mjs の sha256 とは合わない。ブラウザの組は昇格の段が rl/make_web_bundle.mjs で作り直す)",
        "browser_note_ja": "teacher.js は shieldParams を渡さないので shield.mjs の既定の値で走る(既定 = 上の params。test_shield_v5.mjs で照合)。抜け出しの計画のフレームは Node で 0.1〜0.2 秒かかる(まれ)。"
                           "ブラウザでの 1 フレームの時間と F-ID(fid_std・fid_slow・fid_corner・fid_escape)は昇格の段で確かめる"},
        ensure_ascii=False, indent=1) + "\n")
    for f in sorted(a.fid_dir.glob("fid_*.json")):
        shutil.copyfile(f, out / f.name)
    fix = {
        "schema": "jf-sim-rl-status-fixes/v1", "written_by": "no_wall r4 の段(2026-10-14)。昇格の段が使う(この段は bundle・登録簿を変えない)", "updated_at": now,
        "bundle_no_wall_r2": {"applies_to": "experiments/20261013_sim_rl_modes/bundle/no_wall/status.json(r2 の一式、方策 153c709c…)",
                              "state_now": "not_accepted「不合格(参考)」(審査の段が 2026-10-14 13:55 に直した。failed: check_r2_corner_collisions・check_r2_N2_fresh・check_r2_N5_stall)",
                              "action": "r2 の一式のまま出すなら変えなくてよい(もう正しい)。r4 が合格して置き換えるなら r4 の status.json(accepted)を置く"},
        "r3_candidate": {"applies_to": "experiments/20261013_sim_rl_modes/r3/candidates/no_wall_r3/status.json(方策 3b099f9b…・シールド v4.1)",
                         "status_if_published": {"schema": "jf-sim-rl-mode-status/v1", "key": "no_wall", "state": "not_accepted", "label_ja": "不合格(参考)",
                                                 "failed": ["N5_r3", "check_r3_nominal_collision", "check_r3_N2_5p6cm"], "pending": [], "round": "r3",
                                                 "note_ja": "r3 の試験で N5_r3(ゲート 23 の柱の足へ向いた開始で止まり)が不合格。確認 r3(check_no_wall_r3.md)で、標準の物理でもシールド v4.1 が衝突を起こす開始"
                                                            "(ゲート 20・左 −0.13 m・−13°、6.12 秒)と最小すき間 5.63 cm の開始が見つかった。画面の「8.5 cm(標準の物理)」は試験の組の最小で、"
                                                            "記録の開始から一様 1,000 回では 8.01 cm",
                                                 "updated_at": now}},
        "r4_candidate": {"applies_to": "experiments/20261013_sim_rl_modes/r3/candidates/no_wall_r4/status.json", "status": status},
    }
    (out / "bundle_status_fix.json").write_text(json.dumps(fix, ensure_ascii=False, indent=1) + "\n")
    files = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / "sha256.json").write_text(json.dumps(files, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"out": str(out), "state": state, "failed": failed, "policy_sha256": psha, "ui_text": rp["ui_text"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
