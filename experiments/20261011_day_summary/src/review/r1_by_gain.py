"""レビュー用: 006/007 の R1 の進入速度と強い衝突を、走行中に変わっていたゲインの区間で分ける(読み取りのみ)。

    cd ~/jetformer_reports/summary_20261010/src/review && nice -n 15 python r1_by_gain.py > r1_by_gain.json

入力: explore_learning_src/r1_cap_replay.json(しきい値 1150µs の通過表。強い衝突 = 4G 以上)。
ゲインの区間は collision_map.md §1.2(AI_CMD の gains / lookahead を1フレームずつ確認した表)。
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SEG = {  # (終わりの時刻, steer/throttle/lookahead)
    "ai_run_006": [(48.6, "1.80/1.20/5"), (52.3, "移行中"), (67.5, "1.45/1.40/5"), (79.8, "1.45/1.60/5"),
                   (95.0, "1.45/1.80/5"), (102.9, "1.25/1.80/5"), (112.8, "1.25/2.00/5"), (1e9, "1.45/2.00/5")],
    "ai_run_007": [(78.4, "1.40/2.00/3"), (229.2, "1.60/2.00/3"), (236.6, "移行中"), (1e9, "1.00/1.00/3")],
}
d = json.load(open(os.path.join(HERE, "..", "..", "explore_learning_src", "r1_cap_replay.json")))["runs"]["TH1150"]
rows, by = [], {}
for run, v in d.items():
    for p in v["passes"]:
        g = next(name for end, name in SEG[run] if p["t_s"] < end)
        thr = None if g == "移行中" else float(g.split("/")[1])
        band = "移行中" if thr is None else ("thr<=1.0" if thr <= 1.0 else ("1.0<thr<=1.6" if thr <= 1.6 else "thr>=1.8"))
        rows.append({"run": run, "t_s": p["t_s"], "gains": g, "entry_erpm": p["entry_erpm"], "hit_strong": p["hit_strong"]})
        b = by.setdefault(band, {"passes": 0, "ge5500": 0, "ge6200": 0, "strong": 0, "strong_entries_erpm": []})
        b["passes"] += 1; b["ge5500"] += p["entry_erpm"] >= 5500; b["ge6200"] += p["entry_erpm"] >= 6200
        if p["hit_strong"]:
            b["strong"] += 1; b["strong_entries_erpm"].append(p["entry_erpm"])
# 予測ステアの合図だけで、上限まで落とすのに要る減速(r1_cap_replay と同じ式: (進入 − 上限)×KV / 先行時間)
feas = {}
for cap in (5500.0, 5000.0):
    for run, v in d.items():
        need, nosig, n = [], 0, 0
        for p in v["passes"]:
            if p["entry_erpm"] >= cap:
                n += 1
                if p["signal_lead_s"] is None:
                    nosig += 1; continue
                need.append(round(max(0.0, (p["entry_erpm"] - cap) * 0.5e-3) / p["signal_lead_s"], 2))
        feas[f"{int(cap)}|{run}"] = {"passes_ge_cap": n, "no_signal": nosig, "decel_needed_sorted": sorted(need),
                                     "le_2.8": sum(x <= 2.8 for x in need)}
print(json.dumps({"note": "R1 の検出は r1_cap_replay の条件(collision_map より通過が多い: 006 20 対 16、007 25 対 20)。件数は少ない。",
                  "by_throttle_gain": by, "brake_feasibility_pred_signal_only": feas, "passes": rows}, ensure_ascii=False, indent=1))
