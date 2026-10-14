"""離脱復活 r3 の候補の一式を書き出す(2026-10-14、学習の段 recover r3)。bundle/ は変えない(昇格の段が使う)。

    python3 -I r3/rc/export_candidate_rc3.py --policy <選んだ policy.json> --eval <eval_test.json> [--fallback-eval <eval_fallback.json>]
           [--fresh <eval_fresh.json>] [--v1 <eval_v1.json>] [--fastest <eval_fastest.json>]

出力 experiments/20261013_sim_rl_modes/r3/candidates/recover_r3/:
  policy.json      選んだ方策(jf-sim-rl-mode-policy/v1・base79v)
  mode.json        bundle/recover/mode.json の形に r3 の走行中の仕組み(乱し v2・シールド)・記録の決まり・データセットの説明を足したもの
  shield.json      走行中のシールドの値(protocol_recover_r3.json runtime.shield。乱しのフレームは通さない)
  perturbation.json 乱し v2 の決まり(表の sha256・使えるしきい値・替え方・散らし)
  eval.json        試験の組の結果(判定・報告。行ごとの値は eval_test.json に)
  status.json      accepted「合格」/ not_accepted「不合格(参考)」(protocol publication: 優先 1 なので不合格でも参考で出す)
  runtime/         走行中の仕組みのファイル(r3/rc/rt の写し: mode_runtime.mjs・perturb_v2.mjs・recoverable_table.mjs・shield.mjs ほか)
  recoverable_table_v1.json  立て直せる表
  manifest.json    全ファイルの sha256
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
OUT = EXP / "r3/candidates/recover_r3"
RT = EXP / "r3/rc/rt"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--fallback-eval", type=Path)
    ap.add_argument("--fresh", type=Path)
    ap.add_argument("--v1", type=Path)
    ap.add_argument("--fastest", type=Path)
    ap.add_argument("--selection", type=Path)
    a = ap.parse_args()
    ev = json.loads(a.eval.read_text())
    psha = sha(a.policy)
    if ev["policy"]["sha256"] != psha or ev["set"] != "test" or ev["core"] != "original":
        raise SystemExit("eval は選んだ方策の試験の組(test・元の sim-core.js)の結果でなければなりません")
    w = json.loads(a.policy.read_text())
    if w.get("obs_dim") != 79 or w.get("obs_wrapper", "base79v") != "base79v":
        raise SystemExit("方策の観測が base79v(79 次元)ではありません")
    p3 = json.loads((R3 / "protocol_recover_r3.json").read_text())
    table_sha = (R3 / "protocol_recover_r3_table.sha256").read_text().split()[0]
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "runtime").mkdir(parents=True)
    shutil.copy2(a.policy, OUT / "policy.json")
    for f in sorted(RT.iterdir()):
        if f.is_file():
            shutil.copy2(f, OUT / "runtime" / f.name)
    shutil.copy2(EXP / "r3/rc/recoverable_table_v1.json", OUT / "recoverable_table_v1.json")
    J = ev["judged"]
    failed = [k for k, v in J.items() if v.get("pass") is False]
    state = "accepted" if not failed else "not_accepted"
    status = {"schema": "jf-sim-rl-mode-status/v1", "key": "recover", "round": "r3", "state": state,
              "label_ja": "合格" if state == "accepted" else "不合格(参考)", "failed": failed, "pending": [],
              "judged_by": "protocol_recover_r3.json acceptance(C1_r3・C2・C3・C4・C5_r3・C6_r3・C7)", "updated_at": datetime.now().isoformat(timespec="seconds")}
    # 落ちた項目の中身(試験の組の値から書く。画面・昇格の段がそのまま読めるように)
    rep0 = ev["reported"]
    notes = []
    if "C5_r3" in failed:
        c5 = J["C5_r3"]["value"]; kn = rep0.get("kick_fail_no_candidate") or {}
        notes.append(f"C5_r3 キックの生き残り 向き {c5['yaw']}・横 {c5['lateral']}(要 0.90)。落ちた {kn.get('judged_failed')} 回のうち、キックと同じステップでぶつかった "
                     f"{kn.get('same_step_collision')}・キックの直後にシールドのどの候補も 3.5 cm を守れないと見た {kn.get('no_candidate')}(スタートのゲートのそば、4 m/s 前後)")
    if "C6_r3" in failed:
        c6 = J["C6_r3"]["value"]; w = rep0.get("c6_without_spawn_limited") or {}; lim = rep0.get("c6_spawn_limited") or []
        notes.append(f"C6_r3 は開始のすき間が 3 cm 未満の開始({', '.join(f'{r['id']} {r['initial_clearance_m'] * 100:.2f} cm' for r in lim)})のためだけに落ちる"
                     f"(決まりの穴。方策によらない)。それを除くと REC60 {c6['rec60']}・乱しなし {c6['unperturbed']}・最小すき間 {(w.get('min_clearance') or 0) * 100:.2f} cm・停止から 1 周 {c6['lap_from_stop']:.2f} 秒")
    gm = rep0.get("gate_miss") or {}
    if gm.get("p_valid"):
        notes.append(f"報告: 有効な乱しありの回のうちゲートを外した回 {gm['p_valid']}(乱しなしの回は 0)。検証の組で調べると、全部が乱しの終わりから 2 秒以内で、"
                     "多くは奥の直線の出口(s 19.8〜20.8 m)の中・強の舵のパルスのあと。記録の前の確かめでゲートの外しのあるシーンは外す")
    status["notes_ja"] = notes
    (OUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1))
    shield = {"schema": "jf-sim-rl-recover-shield/v1", "applies": "乱しのフレーム(info.jf.ph = 'perturb')以外", "shield_mjs": "runtime/shield.mjs(jf-shield/v4-proto。下の値で v3 と同じ動き)",
              "params": p3["runtime"]["shield"]["params"], "labels": p3["runtime"]["shield"]["labels"]}
    (OUT / "shield.json").write_text(json.dumps(shield, ensure_ascii=False, indent=1))
    pert = {"schema": "jf-sim-rl-recover-perturbation/v2", "id": "jf-perturb-record/v2", "table": "recoverable_table_v1.json", "table_sha256": table_sha,
            "min_rate_record": .95, "min_rate_train": .80, "min_n": 3, "s_bin_m": .25, "v_bin_mps": .5,
            "scheduler": p3["perturbation_r3"]["scheduler"], "spread_hash": "mix32(mix32(シーンの種, f), 0x5a7e) / 2³² < 0.25(runtime/perturb_v2.mjs)",
            "fallback_note_ja": "ブレーキ/踏み込みの「向きを逆に」は 踏み込み ⇄ ブレーキ。弱い強さへの替えは固定の強さ(代わりの試験 弱・中)のときもその中だけ"}
    (OUT / "perturbation.json").write_text(json.dumps(pert, ensure_ascii=False, indent=1))
    rep = ev["reported"]
    extra = {}
    for name, p in (("fallback_weak_medium", a.fallback_eval), ("fresh150", a.fresh), ("v1_same_starts", a.v1), ("fastest_policy", a.fastest)):
        if p and p.exists():
            e = json.loads(p.read_text())
            extra[name] = {"set": e["set"], "policy_sha256": e["policy"]["sha256"], "all_judged_pass": e["all_judged_pass"],
                           "judged": {k: {"pass": v.get("pass"), "value": v.get("value")} for k, v in e["judged"].items()}, "eval_sha256": sha(p)}
    evo = {"schema": "jf-sim-rl-recover-r3-eval-summary/v1", "policy_sha256": psha, "eval_test_sha256": sha(a.eval), "set": "test", "core": "original",
           "protocol_recover_r3_sha256": ev["protocol_recover_r3_sha256"], "recoverable_table_sha256": ev["recoverable_table_sha256"],
           "all_judged_pass": ev["all_judged_pass"], "judged": J, "reported": {k: v for k, v in rep.items() if k not in ("failed_events",)}, "extra": extra}
    if a.selection and a.selection.exists():
        evo["selection"] = json.loads(a.selection.read_text())["selected"]
    # r1 との比べ(publication: 不合格でも r1 の一式より数字が良いときだけ r3 を「不合格(参考)」で出す)
    r1v = R3 / "rc3" / "val_r1_init.json"
    if r1v.exists():
        e1 = json.loads(r1v.read_text())
        sel_eval = Path(evo["selection"]["eval"]) if "selection" in evo else None
        e3 = json.loads(sel_eval.read_text()) if sel_eval and sel_eval.exists() else None
        pick = lambda e: {k: v.get("value") for k, v in e["judged"].items()}  # noqa: E731
        evo["compare_r1"] = {
            "same_runtime_validation": {"note_ja": "同じ走行中の仕組み(乱し v2・シールド)・同じ検証の組で、r1 の方策(7720f275…)と選んだ r3 の方策",
                                        "r1": pick(e1), "r3_selected": pick(e3) if e3 else None},
            "r1_published_test_v1": {"note_ja": "r1 の一式(乱し v1・シールドなし)の試験の組の値(check_recover_r2.md 4 章の表)",
                                     "C1": "38/100", "C2": 0.760, "C4": 2.92, "C5": "35/40・36/40", "C6_min_clearance_rec60_m": 0.02124},
            "r3_test_v2": {"C1_r3": J["C1_r3"]["value"], "C2": J["C2"]["value"], "C4": J["C4"]["value"], "C5_r3": J["C5_r3"]["value"],
                           "C6_min_clearance_rec60_m": J["C6_r3"]["value"].get("min_clearance_rec60")},
            "r3_policy_with_v1": extra.get("v1_same_starts", {}).get("judged"),
            "verdict_ja": "r3 の一式は r1 の一式より C1・C2・C4・C6(REC60 のすき間)で良く、C5 は同じくらい。乱しの作り直しとシールドの効果が大きく、"
                          "方策そのものは同じ仕組みで r1 より少し良い(検証の組)。元の乱し v1 では r3 の方策も r1 と同じくらい(立て直せない乱しは学習でも救えない)"}
    (OUT / "eval.json").write_text(json.dumps(evo, ensure_ascii=False, indent=1))
    base = json.loads((EXP / "bundle/recover/mode.json").read_text())
    c1, c2 = J["C1_r3"]["value"], J["C2"]["value"]
    mode = {**base,
            "policy_sha256": psha,
            "policy_source": f"離脱復活 r3(2026-10-14): PPO・初期化 r1 の 2,015,232 ステップ(7720f275…)・種 32・乱し v2(立て直せる表)・立て直しの報酬・すき間の壁・止まりの見張り。"
                             f"検証の組で選んだ {w.get('meta', {}).get('steps')} ステップ。試験の組 RC-P100-R3 で有効 {c1}・出来事の成功 {c2:.3f}",
            "runtime": {"shield": True, "shield_params": p3["runtime"]["shield"]["params"], "shield_skip_perturb": True,
                        "perturbation": "jf-perturb-record/v2", "perturbation_table_sha256": table_sha, "perturbation_min_rate": .95,
                        "conditioning": None, "gate_index": "base79v"},
            "protocol_recover_r3_sha256": ev["protocol_recover_r3_sha256"],
            "recording": {**base["recording"],
                          "prefilter_required_ja": "記録の前に、プランの全シーンを Node(元の sim-core.js・r3 の mode_runtime.mjs)で走らせ、valid_episode_r3(衝突 0・"
                                                   "止まり(1 秒以上 0.2 m/s 未満、ブレーキの乱しのあと 1 秒は除く)なし・20 秒で 1 周以上進む)でない開始は次の候補に替える",
                          "expected_completion_rate": None, "candidates_for_15_scenes": None,
                          "candidates_note_ja": "r3 の値は train_recover_r3.md(試験の組・新しい開始 150 回)"},
            "dataset_card": {"schema": "jf-sim-rl-dataset-card-notes/v1", "notes_ja": [
                "乱しは「立て直せる組」だけ(乱し v2): 4 つの参照の走り × 8 開始で作った表で、その位置・速さで方策を使わない操縦か方策で 95% 以上立て直せた組だけを使う。"
                "頼んだ組が使えなければ 逆の向き → 同じ強さのわざと外す → 同じ強さのブレーキ → 1 つ弱く の順で替える",
                f"替えた出来事の割合 {rep.get('replaced_fraction')}(試験の組)。強さ・種類の割合は eval.json の C7",
                "乱しのフレーム以外にすき間シールド(3.5 cm)。シールドが替えた操作は学習に使う(sh = 1)",
                "乱した操作(l = 0)は取り込みで学習から外す(外す範囲 [ps − 10, pe − 1])"]},
            "overrides_from_modes_json": sorted(set(base.get("overrides_from_modes_json", [])) | {"runtime", "dataset_card", "recording"}),
            "round": "r3", "status_label_ja": status["label_ja"]}
    (OUT / "mode.json").write_text(json.dumps(mode, ensure_ascii=False, indent=1))
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"schema": "jf-sim-rl-candidate-manifest/v1", "created_at": datetime.now().isoformat(timespec="seconds"),
                                                   "files": {str(p.relative_to(OUT)): sha(p) for p in files}}, ensure_ascii=False, indent=1))
    print(json.dumps({"out": str(OUT), "state": state, "failed": failed, "policy_sha256": psha}, ensure_ascii=False))


if __name__ == "__main__":
    main()
