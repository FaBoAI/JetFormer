"""離脱復活 r3 の学習を早く止める見張り(2026-10-14、学習の段 recover r3。deviations.md に書いた読み方)。

protocol_recover_r3.json training.early_stop は「400 万ステップ以降、続けて 3 つのチェックポイントが検証の組で全部合格したら止めてよい」。
ただし検証の組の RC-VU30-R3 には開始のすき間が 3 cm 未満の開始(VU8 0.32 cm・VU16 1.89 cm)があり、C6_r3(最小すき間 ≥ 3 cm、開始のフレームも数える)は
方策によらず満たせない。そこで「C6_r3 だけが、開始のすき間が 3 cm 未満の開始のためだけに落ちている(それを除くと C6_r3 の全部の条件を満たす)」ものも
早く止める条件の「全部合格」に数える。選び(select_rc3.py)と試験の判定は protocol のまま(C6_r3 は不合格に数える)。
条件を満たしたら early_stop.json を書き、学習(train_rc3.py)に SIGTERM を送る(区切りで last.pt を保存して終わる)。

    python3 -I r3/rc/early_stop_watch.py --run <学習の出力>
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

FROM = 4_000_000


def ok_except_spawn(d: dict) -> tuple[bool, str]:
    J, rep = d["judged"], d["reported"]
    others = [k for k, v in J.items() if k != "C6_r3" and not v.get("pass")]
    if others:
        return False, "落ちた項目 " + ",".join(others)
    if J["C6_r3"].get("pass"):
        return True, "全部合格"
    c6 = J["C6_r3"]["value"]
    w = rep.get("c6_without_spawn_limited") or {}
    lim = rep.get("c6_spawn_limited") or []
    rec_ok = c6["rec60"].split("/")[0] == c6["rec60"].split("/")[1]
    good = (c6.get("lap_from_stop") or 99) <= 9.0 and rec_ok and w.get("runs") and w.get("valid") == w.get("runs") \
        and (w.get("min_clearance") or 0) >= 0.03 and all(r.get("dmin_after_1s") is not None for r in lim)
    # 開始のすき間で決まる回も、有効(valid_episode_r3)でなければだめ
    vu_ok = c6["unperturbed"].split("/")[0] == c6["unperturbed"].split("/")[1]
    return bool(good and vu_ok), ("C6_r3 は開始のすき間だけで落ちている" if good and vu_ok else "C6_r3 がほかの理由でも落ちている")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=60)
    a = ap.parse_args()
    out = a.run / "early_stop.json"
    while True:
        if (a.run / "done.json").exists():
            print("学習が終わっています"); return
        rows = []
        for p in sorted((a.run / "evals" / "recover").glob("step_*.json")):
            try:
                d = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            ok, why = ok_except_spawn(d)
            rows.append((int(p.stem.split("_")[1]), ok, why))
        rows = [r for r in rows if r[0] >= FROM]
        for i in range(len(rows) - 2):
            if all(r[1] for r in rows[i:i + 3]):
                info = {"at": datetime.now().isoformat(timespec="seconds"), "steps": [r[0] for r in rows[i:i + 3]], "why": [r[2] for r in rows[i:i + 3]],
                        "rule": "400 万以降、続けて 3 つのチェックポイントが検証の組で全部合格(C6_r3 は開始のすき間 3 cm 未満の開始だけで落ちるものも合格に数える。"
                                "deviations.md「離脱復活 r3 の学習の段」)", "by": "early_stop_watch.py(外から SIGTERM)"}
                out.write_text(json.dumps(info, ensure_ascii=False, indent=1))
                pids = subprocess.run(["pgrep", "-f", "^python -u r3/rc/train/train_rc3.py --run recover_r3"], capture_output=True, text=True).stdout.split()
                for pid in pids:
                    os.kill(int(pid), signal.SIGTERM)
                print(json.dumps({**info, "sigterm": pids}, ensure_ascii=False))
                return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
