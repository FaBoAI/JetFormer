"""no_wall r4: 確認 r3 の組(NBR・LOWC・HCALL・EDGE2・と確認 r3 の止まり・衝突の回)を、確認 r3 の走らせ方 r3/ck_nw3/run_ck3.mjs(変えない)で
r4 の候補に走らせるための回の一覧を作る(2026-10-14)。回の中身(開始・物理・種)は確認 r3 の jobs_*.jsonl のまま、走らせる仕組み(rtDir)・方策(policy)・
シールドの値(shieldParams = 候補の mode.json runtime.shield_params)だけを足す。確認 r3 の組は下調べで見たので、判定には使わない(退行の確かめ・報告だけ)。

    python3 make_reg_jobs_nw4.py <候補のフォルダー> <出力のフォルダー>
"""
import json
import sys
from pathlib import Path

CK = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_no_wall_r3")
cand, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
mode = json.loads((cand / "mode.json").read_text())
sp = mode["runtime"]["shield_params"]
extra = {"rtDir": str(cand / "runtime"), "policy": str(cand / "policy.json"), "shieldParams": sp}
main = [json.loads(l) for l in open(CK / "jobs_ck3.jsonl")]
sets = {"LOWC": [j for j in main if j["set"] == "LOWC"], "HCALL": [j for j in main if j["set"] == "HCALL"],
        "NBR": [json.loads(l) for l in open(CK / "jobs_nbr_ck3.jsonl")], "EDGE2": [json.loads(l) for l in open(CK / "jobs_edge2_ck3.jsonl")]}
x = [json.loads(l) for l in open(CK / "jobs_x_ck3.jsonl")]
sets["REG"] = [j for j in x if j["set"].startswith("REG")]
for k, js in sets.items():
    with open(out / f"jobs_{k}.jsonl", "w") as f:
        for j in js:
            f.write(json.dumps({**j, **extra}) + "\n")
    print(k, len(js))
