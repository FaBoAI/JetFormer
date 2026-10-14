"""図と報告のための組の集計(no_wall r2、2026-10-14)。compare_sets.py の結果を版ごとにまとめる。

    python build_sets_json.py <出力.json>
"""
import json
import subprocess
import sys
from pathlib import Path

R2 = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013/train/no_wall/r2")
CHK = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_no_wall_r1/runs")
HERE = Path(__file__).resolve().parent


def cmp(rows, off=None, ref=None):
    tmp = R2 / "diag" / "_cmp.json"
    cmd = [sys.executable, str(HERE / "compare_sets.py"), str(rows), "--json", str(tmp)]
    if off:
        cmd += ["--off", str(off)]
    if ref:
        cmd += ["--ref", str(ref)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return json.loads(tmp.read_text())


out = {}
# r1 + v2(確認 r1 の行)。シールドなしは確認 r1 の CORNEROFF と、この段の G16OFF・ABLOFF
off_dir = R2 / "r1policy_off"
r1v2 = cmp(CHK)
r1off = cmp(off_dir)
out["r1_v2"] = r1v2
out["r1_v3"] = cmp(R2 / "shield_v3c_r1policy", off=off_dir)
cv3 = cmp(R2 / "shield_v3c_r1policy", off=CHK)
out["r1_v3"]["CORNER"] = cv3["CORNER"]
cv2 = cmp(CHK, off=CHK)
out["r1_v2"]["CORNER"] = cv2.get("CORNER", out["r1_v2"].get("CORNER"))
g = cmp(CHK, off=off_dir)
for k in ("ABL", "G16"):
    if k in g:
        out["r1_v2"][k] = g[k]
out["r1_off"] = {"G16": r1off.get("G16OFF", {}).get("on") if "G16OFF" in r1off else None}
# compare_sets は *OFF を飛ばすので、シールドなしの数は on/off の off から取る
out["r1_off"] = {k: (out["r1_v3"].get(k, {}).get("off") or {}) for k in ("G16", "ABL", "CORNER", "FDR")}
p2, pc = R2 / "probes_r2policy", R2 / "probes_checker_tools"
if p2.exists():
    # 確認 r1 の道具の行(ABL・CORNER・CORNEROFF)と nw2 の行(VAL・G16・FNOM・FDR とシールドなし)を 1 つのフォルダーに並べて比べる
    mg = R2 / "probes_merged"
    mg.mkdir(exist_ok=True)
    for d in (pc, p2):
        for f in d.glob("rows_*.jsonl"):
            if f.stem.rsplit("_", 1)[-1].isdigit():
                continue
            t = mg / f.name
            if t.exists() or t.is_symlink():
                t.unlink()
            t.symlink_to(f)
    out["r2_v3"] = cmp(mg)
    out["r2_off"] = {k: (out["r2_v3"].get(k, {}).get("off") or {}) for k in ("G16", "ABL", "CORNER", "FDR", "VAL", "FNOM")}
Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=1))
for k, v in out.items():
    print(k, {s: (x.get("on") or x).get("collisions") if isinstance(x, dict) else None for s, x in v.items()})
