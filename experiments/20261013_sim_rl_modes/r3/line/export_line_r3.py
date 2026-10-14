"""経路いろいろ r3 の候補の書き出し(2026-10-14、r3 の学習の段 line)。

選んだ方策と、走行中のすき間シールドの設定(protocol_line_r3.json runtime.shield.params)を、
experiments/20261013_sim_rl_modes/r3/candidates/line_r3/ に bundle と同じ形(mode.json・policy.json・eval.json・status.json)+ shield.json・F-ID で置く。
bundle/・web/rl/・rl/runtime/ は変えない(昇格の段が、modes の作業が終わってから行う)。

    python3 r3/line/export_line_r3.py --policy <policy.json> --eval <eval_test_r3.json> --selection <selection.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

EXP = Path("/home/jetson/jetformer/experiments/20261013_sim_rl_modes")
REP = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
L = EXP / "r3/line"
OUT = EXP / "r3/candidates/line_r3"
P3 = REP / "r3/protocol_line_r3.json"


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    a = ap.parse_args()
    proto = json.loads(P3.read_text())
    want = {ln.split()[1]: ln.split()[0] for ln in (REP / "r3/protocol_r3.sha256").read_text().splitlines() if ln.strip()}["protocol_line_r3.json"]
    if sha256(P3) != want:
        raise SystemExit("protocol_line_r3.json の sha256 が固めた値と違います")
    w = json.loads(a.policy.read_text())
    psha = sha256(a.policy)
    if w.get("schema") != "jf-sim-rl-mode-policy/v1" or w.get("obs_dim") != 97 or (w.get("meta") or {}).get("line_profile") != "v2":
        raise SystemExit("方策の形が経路いろいろ(cond97・目標の線 v2)と合いません")
    ev = json.loads(a.eval.read_text())
    if ev.get("set") != "test" or ev.get("core") != "original" or (ev.get("policy") or {}).get("sha256") != psha:
        raise SystemExit("eval は同じ方策・試験の組・元のコアの結果でなければなりません")
    if (ev.get("shield") or {}).get("params") != proto["runtime"]["shield"]["params"]:
        raise SystemExit("eval のシールドの値が protocol_line_r3.json runtime.shield.params と違います")
    sel = json.loads(a.selection.read_text())
    passed = bool(ev["all_judged_pass"])
    failed = [k for k, v in ev["judged"].items() if v.get("pass") is False]
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.policy, OUT / "policy.json")
    # 試験の結果(回ごとの軌跡は除く。全部は jetformer_reports/…/r3/line_r3/eval_test_r3.json)
    slim = {k: v for k, v in ev.items() if k != "tests"}
    slim["tests"] = {t: [{k: v for k, v in r.items() if k not in ("path", "ey_s", "prof", "sh_frames")} for r in rows] for t, rows in ev["tests"].items()}
    slim["full_result"] = str(a.eval)
    (OUT / "eval.json").write_text(json.dumps(slim, ensure_ascii=False) + "\n")
    shield = {"schema": "jf-sim-rl-line-shield/v1", "mode": "line", "params": proto["runtime"]["shield"]["params"],
              "what_ja": "経路いろいろの走行中のすき間シールド(標準の物理の模型で、すき間 4 cm を守る安全網。悲観の模型・速さの見張り・模型 Q・すくみをほどく・見積もりは使わない)",
              "labels_ja": "シールドが替えた指令は正解ラベル(l = 1、sh = 1)。no_wall と同じ",
              "implementation": {"shield_mjs": "r3/line/rt/shield.mjs(jf-shield/v4-proto の写し)", "shield_mjs_sha256": sha256(L / "rt/shield.mjs"),
                                 "mode_runtime": "r3/line/rt/mode_runtime.mjs(r3 の試作の切り替え globalThis.JF_R3.modeShield.line にこの params を入れて試験した)",
                                 "mode_runtime_sha256": sha256(L / "rt/mode_runtime.mjs")},
              "promote_note_ja": "昇格の段で、共有の mode_runtime.mjs に「line のシールド」を mode.json の runtime.shield / shield_params で切り替える形で入れ、"
                                 "shield.mjs を v4 にしたあと、F-ID(fid_std_left08.json = シールドが働かない回、fid_<回>_shield.json = シールドが働く回)が差 0 であることを確かめる",
              "protocol_line_r3_sha256": want}
    (OUT / "shield.json").write_text(json.dumps(shield, ensure_ascii=False, indent=1) + "\n")
    r2mode = json.loads((EXP / "candidates/line_r2/mode.json").read_text())
    mode = dict(r2mode)
    mode["short_ja"] = ("同じコースを、左寄り・右寄り・中央・イン・アウト・レイトエイペックス・ランダムな線など、シーンごとに違う経路で走る(第 3 回: "
                        "大きく横を向いた開始からも線へ安全に寄せるよう学び直し、走行中は壁・柱から 4 cm を守るシールドつき)。記録のシーンは開始のすき間 12 cm 以上で、"
                        "前もって走らせて、すき間 3 cm 未満・止まり・有効な周なし・シールドの作動 10% 超のものは替える")
    rt = dict(r2mode["runtime"])
    rt.update({"shield": True, "shield_params": proto["runtime"]["shield"]["params"], "shield_note": "jf-shield/v4(r3 の試作)を標準の物理の模型だけで。shield.json"})
    mode["runtime"] = rt
    rec = dict(r2mode["recording"])
    rec.update({"prefilter_min_clearance_m": 0.03, "prefilter_min_valid_laps": 1, "prefilter_no_stall": True, "prefilter_max_shield_fraction": 0.10,
                "prefilter_rule": protocol_text(proto)})
    mode["recording"] = rec
    s = sel["selected"]
    mode.update({"policy_sha256": psha,
                 "policy_source": f"line r3(r3/line_r3/train/main、種 45、初期化 = line r2 の選んだ 801 万)の {s['step']:,} ステップ(protocol_line_r3.json selection)",
                 "protocol_line_r3_sha256": want, "round": "line r3(r3/line_r3/、protocol_line_r3.json)",
                 "overrides_from_modes_json": sorted(set(r2mode.get("overrides_from_modes_json", [])) | {"protocol_line_r3_sha256"})})
    (OUT / "mode.json").write_text(json.dumps(mode, ensure_ascii=False, indent=1) + "\n")
    J = ev["judged"]
    status = {"schema": "jf-sim-rl-mode-status/v1", "key": "line",
              "state": "accepted" if passed else "not_published",
              "label_ja": "合格(昇格の段で bundle/line/ に置く)" if passed else "不合格(出さない・参考の候補)",
              "failed": failed, "pending": ["promote"] if passed else [],
              "rule": "protocol_line_r3.json publication: 合格なら bundle/line/(status accepted)、不合格なら candidates/line_r3/(not_published)。"
                      "bundle/ は modes の作業が終わってから昇格の段が変える",
              "summary": {k: v.get("value") for k, v in J.items()},
              "note_ja": "第 3 回の試験の組 LN-P60-R3 + LN-HEAD60-R3 + LN-EXT60-R3 + LN-FAM6(元の sim-core.js、シールドあり)。"
                         "/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/train_line_r3.md",
              "updated_at": datetime.now().isoformat(timespec="seconds")}
    (OUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1) + "\n")
    # F-ID(STD の開始・左寄り 0.8・c_v 1.0 の 600 フレーム。シールドあり)
    subprocess.run(["node", str(L / "make_fid_ln3.mjs"), "--policy", str(OUT / "policy.json"), "--out", str(OUT / "fid_std_left08.json")], check=True)
    # STD の回ではシールドが働かないので、シールドが一番多く働いた試験の回(LN-P60-R3)の F-ID も置く(ブラウザでシールドの道も同じことを確かめる)
    rows = [r for r in ev["tests"].get("LN-P60-R3", []) if r.get("shield_frames")]
    if rows:
        r = max(rows, key=lambda q: q["shield_frames"])
        st = {k: r["start"][k] for k in ("gate", "left_offset_m", "left_yaw_deg", "seed")}
        subprocess.run(["node", str(L / "make_fid_ln3.mjs"), "--policy", str(OUT / "policy.json"), "--out", str(OUT / f"fid_{r['id'].lower()}_shield.json"),
                        "--start", json.dumps(st), "--params", json.dumps({"line": r["line"], "c_v": r["c_v"]})], check=True)
        shield["fid_with_shield"] = {"file": f"fid_{r['id'].lower()}_shield.json", "test_run": r["id"], "shield_frames": r["shield_frames"]}
        (OUT / "shield.json").write_text(json.dumps(shield, ensure_ascii=False, indent=1) + "\n")
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "sha256.txt")
    (OUT / "sha256.txt").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files))
    print(json.dumps({"out": str(OUT), "state": status["state"], "failed": failed, "policy_sha256": psha}, ensure_ascii=False))


def protocol_text(proto: dict) -> str:
    return proto["recording"]["prefilter"]


if __name__ == "__main__":
    main()
