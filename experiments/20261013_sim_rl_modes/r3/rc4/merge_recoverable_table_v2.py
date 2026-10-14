"""離脱復活 r4(2026-10-14、学習の段 recover r4): build_recoverable_table_v2.mjs の行(JSONL)から「立て直せる表 v2」を作る。
r3/rc/merge_recoverable_table.py の写し。違い: 行の {pair_done} を読み飛ばす・表の id/説明/作り手を v2 に・続きから作った shard の done 行を数える。

    python3 -I r3/rc4/merge_recoverable_table_v2.py <rows_0.jsonl> ... --out-json <recoverable_table_v2.json> --out-mjs <rt/recoverable_table.mjs>
           --sha-file <protocol_recover_r4_table.sha256> --summary <table_summary.json>

マス = (R 上の s の 0.25 m のマス floor(s/0.25), 速さの 0.5 m/s のマス floor(v/0.5), 18 通りの組)。値 [サンプルの数, 立て直せた数]。
使える = サンプル 3 つ以上かつ割合 ≥ 0.95(記録・試験)/ 0.80(学習)。サンプルの無いマスは書かない(= 使えない)。
わざと外すの ok = null の行(その側の幅が足りない)はサンプルに入れない(protocol: 表に入れない)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

LOOP = 26.119648724030924
S_BIN, V_BIN = 0.25, 0.5
EXP = Path("/home/jetson/jetformer/experiments/20261013_sim_rl_modes")


def write_atomic(p, text: str) -> None:
    """途中の形を読まれないよう、一時ファイルに書いてから置き換える(同じ中身を 2 回まとめても安全)"""
    p = Path(p)
    tmp = p.with_name(p.name + ".tmp-%d" % os.getpid())
    tmp.write_text(text)
    os.replace(tmp, p)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-mjs", required=True)
    ap.add_argument("--sha-file", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--allow-partial", action="store_true", help="試しだけ: 終わっていない shard の行でも作る(本番の表には使わない)")
    a = ap.parse_args()
    # 組の並び(build_recoverable_table.mjs の COMBOS と同じ)
    combos = []
    for c in ["弱", "中", "強"]:
        combos += [f"steer_pulse:{c}:L", f"steer_pulse:{c}:R", f"excursion:{c}:L", f"excursion:{c}:R", f"brake:{c}", f"burst:{c}"]
    cells: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    runs, selftests, done, n_rows, n_na = [], [], [], 0, 0
    pair_done: set = set()
    by_combo = defaultdict(lambda: [0, 0])
    by_combo_rec_zone = defaultdict(lambda: [0, 0])
    plans = defaultdict(int)
    for p in a.rows:
        for line in open(p):
            r = json.loads(line)
            if "selftest" in r:
                selftests.append(r["selftest"]); continue
            if "run" in r:
                runs.append(r["run"]); continue
            if r.get("done"):
                done.append(r); continue
            if "pair_done" in r:
                pair_done.add((r["pair_done"]["d"], r["pair_done"]["s"])); continue
            n_rows += 1
            if r["ok"] is None:
                n_na += 1; continue
            key = combos[r["c"]]
            sb = int(math.floor((r["s"] % LOOP) / S_BIN)); vb = int(math.floor(max(0.0, r["v"]) / V_BIN))
            c = cells[f"{sb}|{vb}|{key}"]; c[0] += 1; c[1] += int(r["ok"])
            bc = by_combo[key]; bc[0] += 1; bc[1] += int(r["ok"])
            if r["ar"]:
                bz = by_combo_rec_zone[key]; bz[0] += 1; bz[1] += int(r["ok"])
            plans[r["p"]] += 1
    if len(pair_done) != 32 and not a.allow_partial:
        raise SystemExit(f"終わっていない (driver, 開始) の組があります({len(pair_done)} / 32)")
    usable95 = sum(1 for v in cells.values() if v[0] >= 3 and v[1] / v[0] >= 0.95)
    usable80 = sum(1 for v in cells.values() if v[0] >= 3 and v[1] / v[0] >= 0.80)
    table = {
        "schema": "jf-recoverable-table/v1",
        "id": "recoverable_table_v2",
        "protocol": "protocol_recover_r4.json perturbation_r4.recoverable_table(立て直せたの条件に「ゲートを外さない・pe のあとに次のゲートを通る」を足した版)",
        "loop_length_m": LOOP, "s_bin_m": S_BIN, "v_bin_mps": V_BIN, "min_n": 3, "min_rate_record": 0.95, "min_rate_train": 0.80,
        "key": "cells['<floor(s/0.25)>|<floor(v/0.5)>|<組>'] = [サンプルの数, 立て直せた数](s は R 上の s mod L、v は sim の速さ)",
        "combos": combos,
        "builder": {"file": "r3/rc4/build_recoverable_table_v2.mjs", "sha256": sha(EXP / "r3/rc4/build_recoverable_table_v2.mjs"),
                    "core": "r3/rc/core_oracle.mjs", "core_sha256": sha(EXP / "r3/rc/core_oracle.mjs"),
                    "selftest": selftests[0] if selftests else None,
                    "gate_rule": "枝の [ps, pe + 3 秒] で R 上の s が sim の次のゲートの s + 0.30 m を越えない、かつ pe のあと次のゲートを通る(20 秒の終わりで切れたら外していなければよい)"},
        "drivers": ["離脱復活 r3 の方策 8d686076…(乱しなし・シールドなし)", "最速の方策 7c43cb53…", "r3 + 速さの上限 3.0 m/s", "r3 + 速さの上限 2.2 m/s"],
        "runs": runs,
        "cells": {k: v for k, v in sorted(cells.items())},
    }
    body = json.dumps(table, ensure_ascii=False, separators=(",", ":"))
    write_atomic(a.out_json, body)
    h = hashlib.sha256(body.encode()).hexdigest()
    mjs = (f"// 自動生成(r3/rc4/merge_recoverable_table_v2.py)。手で変えない。sha256(recoverable_table_v2.json の本文) = {h}\n"
           f"export const RECOVERABLE_TABLE_SHA256 = '{h}';\n"
           f"export const RECOVERABLE_TABLE = Object.freeze({body});\n")
    write_atomic(a.out_mjs, mjs)
    write_atomic(a.sha_file, f"{h}  recoverable_table_v2.json(experiments/20261013_sim_rl_modes/r3/rc4/recoverable_table_v2.json。学習の前に固定)\n"
                                f"{sha(Path(a.out_mjs))}  rt/recoverable_table.mjs(experiments/20261013_sim_rl_modes/r3/rc4/rt/recoverable_table.mjs)\n")
    summ = {"merged_at": datetime.now().isoformat(timespec="seconds"), "pairs_done": len(pair_done), "rows": n_rows, "excursion_not_applicable_rows": n_na, "cells": len(cells), "usable_cells_0.95": usable95, "usable_cells_0.80": usable80,
            "table_sha256": h, "runs": runs, "selftests": selftests,
            "by_combo": {k: {"n": v[0], "rec": v[1], "rate": round(v[1] / v[0], 4) if v[0] else None} for k, v in by_combo.items()},
            "by_combo_record_zone": {k: {"n": v[0], "rec": v[1], "rate": round(v[1] / v[0], 4) if v[0] else None} for k, v in by_combo_rec_zone.items()},
            "first_plan_index": dict(sorted(plans.items())),
            "wall_s": [d.get("wall_s") for d in done]}
    write_atomic(a.summary, json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps({k: summ[k] for k in ["rows", "cells", "usable_cells_0.95", "usable_cells_0.80", "table_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
