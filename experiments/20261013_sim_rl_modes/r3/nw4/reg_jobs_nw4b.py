"""no_wall r4b: 確認 r3 の組(NBR・LOWC・HCALL・EDGE2・REG)を、確認 r3 の走らせ方 r3/ck_nw3/run_ck3.mjs(変えない)で選んだ方策に走らせる回の一覧を作る
(2026-10-14 20 時、電源断のあとの再開で足した。make_reg_jobs_nw4.py の写しで、候補の一式を書き出す前に試験と並べて走らせられるように、
方策・走らせる仕組み(runtime_v5 = 候補の runtime/ に写すもの)・シールドの値(protocol_no_wall_r4.json runtime.shield.params = 候補の mode.json の値)を直接渡す)。
シールドあり / なしの対(救った・起こした)を数えるため、NBR・LOWC はシールドなしの回(NBR-OFF・LOWC-OFF)も、HCALL は確認 r3 の HCOFF も足す。
回の中身(開始・物理・種)は確認 r3 の jobs_*.jsonl のまま。確認 r3 の組は下調べで一部を見たので、判定には使わない(退行の確かめ・報告だけ)。

    python3 reg_jobs_nw4b.py --policy <選んだ policy.json> --out <出力のフォルダー>
"""
import argparse
import hashlib
import json
from pathlib import Path

CK = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_no_wall_r3")
RT = Path("/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/nw4/runtime_v5")
PROT4 = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/protocol_no_wall_r4.json")
ap = argparse.ArgumentParser()
ap.add_argument("--policy", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
sp = json.loads(PROT4.read_text())["runtime"]["shield"]["params"]
extra = {"rtDir": str(RT), "policy": str(a.policy.resolve()), "shieldParams": sp}
main = [json.loads(l) for l in open(CK / "jobs_ck3.jsonl")]
x = [json.loads(l) for l in open(CK / "jobs_x_ck3.jsonl")]
sets = {"LOWC": [j for j in main if j["set"] == "LOWC"], "HCALL": [j for j in main if j["set"] == "HCALL"],
        "HCOFF": [j for j in main if j["set"] == "HCOFF"],
        "NBR": [json.loads(l) for l in open(CK / "jobs_nbr_ck3.jsonl")], "EDGE2": [json.loads(l) for l in open(CK / "jobs_edge2_ck3.jsonl")],
        "REG": [j for j in x if j["set"].startswith("REG")]}
sets["NBR-OFF"] = [{**j, "set": "NBR-OFF", "id": j["id"].replace("NBR:", "NBR-OFF:", 1), "shield": False} for j in sets["NBR"]]
sets["LOWC-OFF"] = [{**j, "set": "LOWC-OFF", "id": j["id"].replace("LOWC:", "LOWC-OFF:", 1), "shield": False} for j in sets["LOWC"]]
assert all(j.get("shield") is False for j in sets["HCOFF"]), "HCOFF はシールドなしの回のはず"
ids = set()
n = 0
with open(a.out / "jobs_all.jsonl", "w") as f:
    for k, js in sets.items():
        for j in js:
            assert j["id"] not in ids, j["id"]
            ids.add(j["id"])
            f.write(json.dumps({**j, **extra}) + "\n")
            n += 1
        print(k, len(js))
meta = {"policy": str(a.policy.resolve()), "policy_sha256": hashlib.sha256(a.policy.read_bytes()).hexdigest(),
        "runtime_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RT.iterdir()) if p.is_file()},
        "protocol_no_wall_r4_sha256": hashlib.sha256(PROT4.read_bytes()).hexdigest(), "counts": {k: len(v) for k, v in sets.items()}, "total": n,
        "source_jobs": {p: hashlib.sha256((CK / p).read_bytes()).hexdigest() for p in ["jobs_ck3.jsonl", "jobs_x_ck3.jsonl", "jobs_nbr_ck3.jsonl", "jobs_edge2_ck3.jsonl"]}}
(a.out / "jobs_all.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
print("合計", n)
