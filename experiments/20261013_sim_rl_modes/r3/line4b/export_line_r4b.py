"""経路いろいろ r4b の候補の書き出し(2026-10-14 夜、line r4 の段のやり直し。r3/line4/export_line_r4.py の写しを元に)。

選んだ方策と、走行中のシールドの設定(protocol_line_r4b.json runtime.shield.mode_runtime = mode.json の runtime.shield・shield_params)を、
experiments/20261013_sim_rl_modes/r3/candidates/line_r4/ に bundle と同じ形(mode.json・policy.json・eval.json・status.json)+ shield.json・F-ID・
走行中の仕組みの写し(runtime/ = r3/line4b/rt4b の全部。共有の web/rl と違うのは shield.mjs・mode_runtime.mjs だけ)・昇格の段への差分の案
(runtime/*.diff・promote/*.diff)で置く。bundle/・web/rl/・rl/runtime/・JetFormer 本体は変えない。

    python3 r3/line4b/export_line_r4b.py --policy <policy.json> --eval <eval_test_r4b.json> --selection <selection.json>
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

EXP = Path("/home/jetson/jetformer/experiments/20261013_sim_rl_modes")
REP = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
L = EXP / "r3/line4b"
OUT = EXP / "r3/candidates/line_r4"
P4B = REP / "r3/protocol_line_r4b.json"
JF = Path("/home/jetson/jetformer")


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fid(policy: Path, out: Path, start: dict | None = None, params: dict | None = None) -> dict:
    cmd = ["node", str(L / "make_fid_ln4b.mjs"), "--policy", str(policy), "--out", str(out)]
    if start:
        cmd += ["--start", json.dumps(start), "--params", json.dumps(params)]
    r = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def udiff(old: Path, new: Path, a: str, b: str) -> str:
    return "".join(difflib.unified_diff(old.read_text().splitlines(keepends=True), new.read_text().splitlines(keepends=True), a, b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None, help="試しの書き出しだけ(既定 r3/candidates/line_r4)")
    a = ap.parse_args()
    global OUT
    if a.out_dir:
        OUT = a.out_dir
    proto = json.loads(P4B.read_text())
    want = (REP / "r3/protocol_line_r4b.sha256").read_text().split()[0]
    if sha256(P4B) != want:
        raise SystemExit("protocol_line_r4b.json の sha256 が固めた値と違います")
    w = json.loads(a.policy.read_text())
    psha = sha256(a.policy)
    if w.get("schema") != "jf-sim-rl-mode-policy/v1" or w.get("obs_dim") != 97 or (w.get("meta") or {}).get("line_profile") != "v2":
        raise SystemExit("方策の形が経路いろいろ(cond97・目標の線 v2)と合いません")
    ev = json.loads(a.eval.read_text())
    if ev.get("set") != "test" or ev.get("core") != "original" or (ev.get("policy") or {}).get("sha256") != psha or ev.get("compare"):
        raise SystemExit("eval は同じ方策・試験の組・元のコアの結果でなければなりません")
    if ev.get("protocol_line_r4b_sha256") != want:
        raise SystemExit("eval が protocol_line_r4b.json で走らせたものではありません")
    mr = proto["runtime"]["shield"]["mode_runtime"]
    if (ev.get("shield") or {}).get("mode_runtime") != mr:
        raise SystemExit("eval のシールドの値が protocol_line_r4b.json runtime.shield.mode_runtime と違います")
    if (ev["shield"]["shield_mjs_sha256"] != proto["runtime"]["shield"]["file_sha256_at_freeze"]
            or ev["shield"]["mode_runtime_sha256"] != proto["runtime"]["shield"]["mode_runtime_sha256_at_freeze"]):
        raise SystemExit("eval の shield.mjs・mode_runtime.mjs が固めたときと違います")
    if sha256(L / "rt4b/shield.mjs") != proto["runtime"]["shield"]["file_sha256_at_freeze"]:
        raise SystemExit("rt4b/shield.mjs が固めたときと違います")
    sel = json.loads(a.selection.read_text())
    if sel["selected"]["policy_sha256"] != psha:
        raise SystemExit("選んだ方策と --policy が違います")
    passed = bool(ev["all_judged_pass"])
    failed = [k for k, v in ev["judged"].items() if v.get("pass") is False]
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.policy, OUT / "policy.json")
    slim = {k: v for k, v in ev.items() if k != "tests"}
    slim["tests"] = {t: [{k: v for k, v in r.items() if k not in ("path", "ey_s", "prof", "sh_frames")} for r in rows] for t, rows in ev["tests"].items()}
    slim["full_result"] = str(a.eval)
    (OUT / "eval.json").write_text(json.dumps(slim, ensure_ascii=False) + "\n")
    # 走行中の仕組みの写し(全部。ブラウザの確かめでこのフォルダーをそのまま配る)と、共有のファイルとの差分(昇格の段が共有の rl/runtime・web/rl に入れる)
    rt = OUT / "runtime"
    if rt.exists():
        shutil.rmtree(rt)
    rt.mkdir()
    for p in sorted((L / "rt4b").iterdir()):
        if p.is_file():
            shutil.copyfile(p, rt / p.name)
    diffs = {}
    for f in ("shield.mjs", "mode_runtime.mjs"):
        for base, tag in ((EXP / "web/rl" / f, "web_rl"), (EXP / "rl/runtime" / f, "rl_runtime")):
            d = udiff(base, L / "rt4b" / f, f"{tag}/{f}", f"r3/line4b/rt4b/{f}")
            (rt / f"{f}.{tag}.diff").write_text(d)
            diffs[f"{f}.{tag}.diff"] = {"base_sha256": sha256(base), "new_sha256": sha256(L / "rt4b" / f), "lines": d.count("\n")}
    same_as_web = sorted(p.name for p in (L / "rt4b").iterdir() if p.is_file() and (EXP / "web/rl" / p.name).exists() and sha256(p) == sha256(EXP / "web/rl" / p.name))
    # ブラウザ側の案(JetFormer 本体の sim/static/sim-rl.js・rl/teacher.js。この段では当てない。昇格の段が当てる)
    pr = OUT / "promote"
    pr.mkdir(exist_ok=True)
    for f in ("sim-rl.js.diff", "teacher.js.diff", "sha256_original.txt"):
        shutil.copyfile(L / "promote_patch" / f, pr / f)
    cur = {"sim-rl.js": sha256(JF / "jetformer/sim/static/sim-rl.js"), "teacher.js": sha256(JF / "jetformer/sim/static/rl/teacher.js")}
    orig = {ln.split()[1].split("/")[-1]: ln.split()[0] for ln in (L / "promote_patch/sha256_original.txt").read_text().splitlines() if ln.strip()}
    tst = json.loads((REP / "r3/line_r4b/dev/test_shield_v5b.json").read_text())
    shield = {"schema": "jf-sim-rl-line-shield/v3", "mode": "line", "version": "jf-shield/v5", "select": "v5b", "mode_runtime": mr,
              "what_ja": "経路いろいろの走行中のすき間シールド v5b(標準の物理の模型で、すき間 4 cm を守る安全網。止まれることを合格の条件にし、"
                         "合格する候補が無いときはブレーキ(速さ 0)の候補だけから選び、方策の指令をそのまま通さない。止まったまま進めないときは決まった操縦の表で抜け出す。"
                         "悲観の模型は衝突の予想だけに使い、速さの見張り・模型 Q・見積もりは使わない)",
              "labels_ja": "シールドが替えた指令は正解ラベル(l = 1、sh = 1)。後ろへ動いたフレームのあるシーンは記録の前の確かめで替える",
              "implementation": {"files": "runtime/(r3/line4b/rt4b の全部の写し)。共有の web/rl と同じもの: " + ", ".join(same_as_web),
                                 "shield_mjs_sha256": sha256(L / "rt4b/shield.mjs"), "mode_runtime_sha256": sha256(L / "rt4b/mode_runtime.mjs"),
                                 "diffs": diffs,
                                 "v5_compat_test": {"file": str(REP / "r3/line_r4b/dev/test_shield_v5b.json"), "summary": tst["summary"],
                                                    "note_ja": "select 'v5'(と既定)のときの操作の列が no_wall r4 の v5(8ece9be4…)と同じ(26 回)。昇格の段は no_wall にもこの shield.mjs を使える"}},
              "promote": {"teacher_js_diff": "promote/teacher.js.diff(createModeTeacher に modeRuntime: entry.runtime を渡す)",
                          "sim_rl_js_diff": "promote/sim-rl.js.diff(登録簿の runtime を RL.modes[key].runtime に残す)",
                          "base_sha256_now": cur, "base_sha256_when_made": orig, "applies_to_current": cur == orig},
              "promote_note_ja": "昇格の段で: (1) 共有の rl/runtime/shield.mjs・web/rl/shield.mjs・jetformer/sim/static/rl/shield.mjs をこの runtime/shield.mjs に"
                                 "(select を渡さなければ no_wall r4 の v5 と同じ動き)、(2) mode_runtime.mjs を runtime/mode_runtime.mjs に(差分 runtime/mode_runtime.mjs.web_rl.diff。"
                                 "mode.json の runtime.shield・shield_params でシールドを切り替え、metadata.shield に値の全部を入れる)、(3) promote/teacher.js.diff・sim-rl.js.diff を当てる、"
                                 "(4) bundle/line/ を mode.json(runtime.shield = true・shield_params 全部・記録の前の確かめの鍵)で置き換え、web/rl の manifest を作り直す、"
                                 "(5) 取り込んだあとのページの teacher.js を通して、この F-ID(fid_*.json)が Node・ブラウザで差 0 であることを確かめる(この段の確かめは browser_fid.json)、"
                                 "(6) 記録の前の確かめで後ろへ動いたシーンを替える",
              "protocol_line_r4b_sha256": want, "fids": {}}
    mode = json.loads((EXP / "r3/candidates/line_r3/mode.json").read_text())
    mode["short_ja"] = ("同じコースを、左寄り・右寄り・中央・イン・アウト・レイトエイペックス・ランダムな線など、シーンごとに違う経路で走る(第 4 回: "
                        "開始の範囲の角(大きく横を向いて柵・柱の近くから始まる所、右寄りの線へ寄せる所)からも安全に線へ寄せるよう学び直し、"
                        "走行中は壁・柱から 4 cm を守るシールド v5b つき)。記録のシーンは開始のすき間 12 cm 以上で、前もって走らせて、"
                        "すき間 3 cm 未満・止まり・有効な周なし・シールドの作動 10% 超・後ろへ動いたものは替える")
    rtj = dict(mode["runtime"])
    rtj.update({"shield": True, "shield_params": mr["shield_params"], "shield_note": "jf-shield/v5(select 'v5b')を標準の物理の模型だけで(すき間 4 cm)。shield.json"})
    mode["runtime"] = rtj
    rec = dict(mode["recording"])
    rec.update({"prefilter_min_clearance_m": 0.03, "prefilter_min_valid_laps": 1, "prefilter_no_stall": True, "prefilter_max_shield_fraction": 0.10,
                "prefilter_no_reverse": True, "prefilter_rule": proto["recording"]["prefilter"]})
    mode["recording"] = rec
    s = sel["selected"]
    src = (f"line r4b(r3/line_r4b/train/main、種 47、初期化 = line r4 の 4,014,080)の {s['step']:,} ステップ(protocol_line_r4b.json selection)" if s["origin"] == "r4b"
           else f"line r4(r3/line_r4/train/main、種 46)の {s['step']:,} ステップ(protocol_line_r4b.json selection の候補)")
    for k in ("protocol_line_r3_sha256", "protocol_line_r4_sha256"):
        mode.pop(k, None)
    mode.update({"policy_sha256": psha, "policy_source": src, "protocol_line_r4b_sha256": want, "round": "line r4b(r3/line_r4b/、protocol_line_r4b.json)",
                 "overrides_from_modes_json": sorted((set(mode.get("overrides_from_modes_json", [])) - {"protocol_line_r3_sha256", "protocol_line_r4_sha256"})
                                                     | {"protocol_line_r4b_sha256"})})
    (OUT / "mode.json").write_text(json.dumps(mode, ensure_ascii=False, indent=1) + "\n")
    J = ev["judged"]
    status = {"schema": "jf-sim-rl-mode-status/v1", "key": "line",
              "state": "accepted" if passed else "not_published",
              "label_ja": "合格(確認の段のあと、昇格の段で bundle/line/ に置く)" if passed else "不合格(出さない・参考の候補)",
              "failed": failed, "pending": ["check", "promote"] if passed else [],
              "rule": "protocol_line_r4b.json publication: 合格なら candidates/line_r4/(status accepted。昇格の段が bundle/line/ に置く)、不合格なら not_published。"
                      "bundle/ はこの段では変えない",
              "summary": {k: v.get("value") for k, v in J.items()},
              "note_ja": "第 4 回(r4b)の試験の組 LN-P60-R4 + LN-HEAD60-R4 + LN-EXT60-R4 + LN-CORNER-R4 + LN-BOX120-R4 + LN-G30R120-R4B + LN-FAM6"
                         "(542 回、元の sim-core.js、シールド v5b あり)。/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/train_line_r4.md",
              "updated_at": datetime.now().isoformat(timespec="seconds")}
    (OUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1) + "\n")
    # F-ID: (1) STD・左寄り 0.8・c_v 1.0、(2) シールドが働く試験の回(LN-G30R120-R4B で一番多い回。無ければ P60、ほかの組の有効な回)、
    #        (3) brake_first か抜け出しが働いた回があればその 1 つ
    for old in OUT.glob("fid_*.json"):
        old.unlink()
    shield["fids"]["fid_std_left08.json"] = fid(OUT / "policy.json", OUT / "fid_std_left08.json")
    allrows = [r for t in ev["tests"].values() for r in t if not r.get("invalid_spawn")]
    picks = []
    for pool in (ev["tests"].get("LN-G30R120-R4B", []), ev["tests"].get("LN-P60-R4", []), allrows):
        cand = [r for r in pool if r.get("shield_frames") and r.get("valid_r3") and not r.get("reverse_frames")]
        if cand:
            picks.append(("shield", max(cand, key=lambda q: q["shield_frames"])))
            break
    special = [r for r in allrows if r.get("escape_frames") or (r.get("shield_fallbacks") or {}).get("brake_first")]
    if special:
        picks.append(("special", max(special, key=lambda q: q["shield_frames"])))
    for kind, r in picks:
        st = {k: r["start"][k] for k in ("gate", "left_offset_m", "left_yaw_deg", "seed")}
        name = f"fid_{r['test'].lower().replace('-', '_')}_{r['id'].lower()}_{kind}.json"
        shield["fids"][name] = {**fid(OUT / "policy.json", OUT / name, st, {"line": r["line"], "c_v": r["c_v"]}), "test": r["test"], "id": r["id"]}
    # F-ID を Node で 1 フレームずつ新しい教師に与えて、操作・info.jf・metadata が同じことを確かめる(ブラウザは browser_fid_ln4b.py)
    r = subprocess.run(["node", str(L / "fid_check_ln4b.mjs"), str(OUT / "policy.json"), *[str(OUT / n) for n in shield["fids"]]], check=True, capture_output=True, text=True)
    shield["fid_check_node"] = json.loads(r.stdout.strip().splitlines()[-1])
    (OUT / "shield.json").write_text(json.dumps(shield, ensure_ascii=False, indent=1) + "\n")
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "sha256.txt")
    (OUT / "sha256.txt").write_text("".join(f"{sha256(p)}  {p.relative_to(OUT)}\n" for p in files))
    print(json.dumps({"out": str(OUT), "state": status["state"], "failed": failed, "policy_sha256": psha, "fids": list(shield["fids"]),
                      "fid_check_node": shield["fid_check_node"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
