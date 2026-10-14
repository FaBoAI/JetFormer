"""離脱復活 r4 の候補の一式を書き出す(2026-10-14、学習の段 recover r4)。bundle/ は変えない(昇格の段が使う)。r3/rc/export_candidate_rc3.py が元。

    python3 -I r3/rc4/export_candidate_rc4.py --policy <選んだ policy.json> --eval <eval_test.json> [--fresh …] [--v1 …] [--fastest …] [--r3 …] [--selection …]

出力 experiments/20261013_sim_rl_modes/r3/candidates/recover_r4/:
  policy.json        選んだ方策(jf-sim-rl-mode-policy/v1・base79v)
  mode.json          r3 の候補の mode.json の形に r4 の走行中の仕組み(乱し v2 + 表 v2・シールド r4)・記録の決まり・データセットの説明
  shield.json        走行中のシールドの値(protocol_recover_r4.json runtime.shield.shield_r4。乱しのフレームは通さない。すくみをほどくあり)
  perturbation.json  乱し v2 の決まり(表 v2 の sha256・使えるしきい値・替え方・散らし)
  eval.json          試験の組の結果(判定・報告。行ごとの値は jetformer_reports/…/r3/rc4/eval_test.json)
  status.json        accepted「合格」/ not_accepted「不合格(参考)」
  runtime/           走行中の仕組みのファイル(r3/rc4/rt の写し)
  recoverable_table_v2.json  立て直せる表 v2
  promote/           昇格の段の手当て(r3/rc4/promote の写し: JetFormer 本体の差分 2 つ・試験・記録したシーンの確かめ)
  manifest.json      全ファイルの sha256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

EXP = Path("/home/jetson/jetformer/experiments/20261013_sim_rl_modes")
REPORTS = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
R3 = REPORTS / "r3"
OUT = EXP / "r3/candidates/recover_r4"
RT = EXP / "r3/rc4/rt"
PROMOTE = EXP / "r3/rc4/promote"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def judged_summary(p: Path | None) -> dict | None:
    if not p or not p.exists():
        return None
    e = json.loads(p.read_text())
    return {"set": e["set"], "label": e.get("label"), "policy_sha256": e["policy"]["sha256"], "all_judged_pass": e["all_judged_pass"],
            "judged": {k: {"pass": v.get("pass"), "value": v.get("value")} for k, v in e["judged"].items()},
            "reported": {k: e["reported"].get(k) for k in ("kick_band", "gate_miss", "shield", "invalid_p", "replaced_fraction", "start_zones") if k in e["reported"]},
            "eval_sha256": sha(p)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--fresh", type=Path)
    ap.add_argument("--v1", type=Path)
    ap.add_argument("--fastest", type=Path)
    ap.add_argument("--r3", type=Path)
    ap.add_argument("--selection", type=Path)
    a = ap.parse_args()
    ev = json.loads(a.eval.read_text())
    psha = sha(a.policy)
    if ev["policy"]["sha256"] != psha or ev["set"] != "test" or ev["core"] != "original" or ev.get("smoke"):
        raise SystemExit("eval は選んだ方策の試験の組(test・元の sim-core.js・固定の決まり)の結果でなければなりません")
    w = json.loads(a.policy.read_text())
    if w.get("obs_dim") != 79:
        raise SystemExit("方策の観測が base79v(79 次元)ではありません")
    p4 = json.loads((R3 / "protocol_recover_r4.json").read_text())
    p4sha = (R3 / "protocol_recover_r4.sha256").read_text().split()[0]
    if p4sha != ev["protocol_recover_r4_sha256"]:
        raise SystemExit("試験の結果の protocol_recover_r4.json の sha256 が固めた値と違います")
    table_sha = (R3 / "protocol_recover_r4_table.sha256").read_text().split()[0]
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "runtime").mkdir(parents=True)
    (OUT / "promote").mkdir()
    shutil.copy2(a.policy, OUT / "policy.json")
    for f in sorted(RT.iterdir()):
        if f.is_file():
            shutil.copy2(f, OUT / "runtime" / f.name)
    for f in sorted(PROMOTE.iterdir()):
        if f.is_file():
            shutil.copy2(f, OUT / "promote" / f.name)
    shutil.copy2(EXP / "r3/rc4/recoverable_table_v2.json", OUT / "recoverable_table_v2.json")
    J = ev["judged"]
    failed = [k for k, v in J.items() if v.get("pass") is False]
    state = "accepted" if not failed else "not_accepted"
    rep = ev["reported"]
    status = {"schema": "jf-sim-rl-mode-status/v1", "key": "recover", "round": "r4", "state": state,
              "label_ja": "合格" if state == "accepted" else "不合格(参考)", "failed": failed,
              "pending": ["確認の段 r4(学習・評価のコードを使わない反証)の前。この合否は学習の段の試験の組の判定"],
              "judged_by": "protocol_recover_r4.json acceptance(C1_r3・C2・C3・C4・C5_r4・C6_r4・C7・C8)", "protocol_recover_r4_sha256": p4sha,
              "updated_at": datetime.now().isoformat(timespec="seconds")}
    notes = []
    for k in failed:
        notes.append(f"{k} 不合格: {json.dumps(J[k].get('value'), ensure_ascii=False)}")
    kb = rep.get("kick_band") or {}
    if kb:
        notes.append(f"報告: スタートのゲートの区間(s 0.30〜3.70 m)のキック 右 {kb.get('band_right')}・左 {kb.get('band_left')}・区間の外 {kb.get('outside')}")
    gm = J.get("C8", {}).get("value") or {}
    if gm:
        notes.append(f"ゲートの外し: 乱しありの有効な回 {gm.get('perturbed_valid_with_miss')}、乱しなし {gm.get('unperturbed_with_miss')}。"
                     "記録したブラウザのシーンは取り込む前に promote/check_recorded_scenes_rc4.mjs で止まり・進み・ゲートの外しを確かめ直す")
    inv = [r for r in rep.get("invalid_p", []) if r.get("stall")]
    if inv:
        notes.append(f"試験の組の止まり {len(inv)} 回({', '.join(r['id'] for r in inv)}。例 P27 = ゲート 7・右 0.197 m の開始で壁へ寄ってシールドが止める。r3 の方策では走る)。"
                     "valid_episode_r3 と記録の前の確かめ(predict_scenes_rc4.mjs)で外れる")
    notes.append(p4["runtime"]["shield"]["ui_note"])
    status["notes_ja"] = notes
    (OUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1))
    shield = {"schema": "jf-sim-rl-recover-shield/v2", "round": "r4", "applies": "乱しのフレーム(info.jf.ph = 'perturb')以外",
              "shield_mjs": "runtime/shield.mjs(jf-shield/v4-proto。下の値。すくみをほどくを使う)",
              "params": p4["runtime"]["shield"]["shield_r4"], "labels": p4["runtime"]["shield"]["labels"], "ui_note_ja": p4["runtime"]["shield"]["ui_note"]}
    (OUT / "shield.json").write_text(json.dumps(shield, ensure_ascii=False, indent=1))
    pert = {"schema": "jf-sim-rl-recover-perturbation/v2", "round": "r4", "id": "jf-perturb-record/v2", "table": "recoverable_table_v2.json", "table_sha256": table_sha,
            "min_rate_record": .95, "min_rate_train": .80, "min_n": 3, "s_bin_m": .25, "v_bin_mps": .5,
            "table_rule": p4["perturbation_r4"]["recoverable_table"], "gate_rule": p4["definitions_added"]["gate_rule_table"],
            "scheduler": "r3 と同じ(runtime/perturb_v2.mjs。頼んだ組 → 逆の向き → 同じ強さのわざと外す → 同じ強さのブレーキ → 1 つ弱く。散らし 25%)"}
    (OUT / "perturbation.json").write_text(json.dumps(pert, ensure_ascii=False, indent=1))
    extra = {name: judged_summary(p) for name, p in (("fresh150", a.fresh), ("v1_same_starts", a.v1), ("fastest_policy", a.fastest), ("r3_candidate_same_sets", a.r3))
             if p and p.exists()}
    evo = {"schema": "jf-sim-rl-recover-r4-eval-summary/v1", "policy_sha256": psha, "eval_test_sha256": sha(a.eval), "set": "test", "core": "original",
           "protocol_recover_r4_sha256": p4sha, "recoverable_table_sha256": ev["recoverable_table_sha256"],
           "all_judged_pass": ev["all_judged_pass"], "judged": J, "reported": {k: v for k, v in rep.items() if k not in ("failed_events",)}, "extra": extra}
    if a.selection and a.selection.exists():
        evo["selection"] = json.loads(a.selection.read_text())["selected"]
    (OUT / "eval.json").write_text(json.dumps(evo, ensure_ascii=False, indent=1))
    base = json.loads((EXP / "r3/candidates/recover_r3/mode.json").read_text())
    c1, c2 = J["C1_r3"]["value"], J["C2"]["value"]
    mode = {**base,
            "policy_sha256": psha,
            "policy_source": f"離脱復活 r4(2026-10-14): PPO・初期化 r3 の 5,013,504 ステップ(8d686076…)・種 33・乱し v2(立て直せる表 v2: ゲートを外さない条件つき)・"
                             f"スタートのゲートの区間のキックと右の余地の壁・ゲートの外しで終わる。検証の組で選んだ {w.get('meta', {}).get('steps')} ステップ。"
                             f"試験の組 RC-P100-R4 で有効 {c1}・出来事の成功 {c2:.3f}",
            "runtime": {"shield": True, "shield_params": p4["runtime"]["shield"]["shield_r4"], "shield_skip_perturb": True,
                        "perturbation": "jf-perturb-record/v2", "perturbation_table": "recoverable_table_v2", "perturbation_table_sha256": table_sha,
                        "perturbation_min_rate": .95, "conditioning": None, "gate_index": "base79v"},
            "protocol_recover_r4_sha256": p4sha,
            "recording": {**base.get("recording", {}),
                          "prefilter_required_ja": "記録の前に、プランの全シーンを Node(元の sim-core.js・r4 の mode_runtime.mjs)で走らせ(promote/../predict_scenes_rc4.mjs と同じ決まり)、"
                                                   "valid_episode_r3 でない開始・本当のゲートの外しのある開始は次の候補に替える",
                          "post_record_check_ja": "記録したブラウザのシーンは、取り込む前に promote/check_recorded_scenes_rc4.mjs で止まり・進み・ゲートの外しを確かめ直し、外れたものは取り込まない"
                                                  "(ブラウザと Node は閉ループで分かれるため)",
                          "candidates_note_ja": "r4 の値は train_recover_r4.md(試験の組・新しい開始 150 回)"},
            "dataset_card": {"schema": "jf-sim-rl-dataset-card-notes/v1", "notes_ja": [
                "乱しは「立て直せる組」だけ(乱し v2・表 v2): 4 つの参照の走り × 8 開始で作った表で、その位置・速さで方策を使わない操縦か方策で 95% 以上、"
                "ゲートを外さずに立て直せた組だけを使う。頼んだ組が使えなければ 逆の向き → 同じ強さのわざと外す → 同じ強さのブレーキ → 1 つ弱く の順で替える",
                f"替えた出来事の割合 {rep.get('replaced_fraction')}(試験の組)。強さ・種類の割合は eval.json の C7",
                "乱しのフレーム以外にすき間シールド(3.5 cm、すくみをほどくあり)。シールドが替えた操作は学習に使う(sh = 1)",
                "乱した操作(l = 0)は取り込みで学習から外す(外す範囲 [ps − 10, pe − 1])"]},
            "ui_note_ja": p4["runtime"]["shield"]["ui_note"],
            "overrides_from_modes_json": sorted(set(base.get("overrides_from_modes_json", [])) | {"runtime", "dataset_card", "recording"}),
            "round": "r4", "status_label_ja": status["label_ja"]}
    mode.pop("protocol_recover_r3_sha256", None)
    (OUT / "mode.json").write_text(json.dumps(mode, ensure_ascii=False, indent=1))
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"schema": "jf-sim-rl-candidate-manifest/v1", "created_at": datetime.now().isoformat(timespec="seconds"),
                                                   "note_ja": "runtime/manifest.json は web/rl の写したときのまま。web/rl に入れるときは rl/make_web_bundle.mjs で作り直す",
                                                   "files": {str(p.relative_to(OUT)): sha(p) for p in files}}, ensure_ascii=False, indent=1))
    print(json.dumps({"out": str(OUT), "state": state, "failed": failed, "policy_sha256": psha}, ensure_ascii=False))


if __name__ == "__main__":
    main()
