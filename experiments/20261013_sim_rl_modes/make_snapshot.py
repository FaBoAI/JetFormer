"""ベストラップの作業のファイルを snapshot/ に写す(2026-10-13、sim_rl_modes の実装の段)。

protocol.json の snapshot_sources(元のパスと sha256)と比べてから写し、SNAPSHOT.json に元のパス・sha256・写した時刻を残す。
違っていれば protocol の snapshot_rule のとおり: features.mjs・reference_loop.mjs/.json・policy_forward.mjs は git の c0a334c の版、
ほかは今の版を写して deviations.md に書く(この段では違いがあれば止まる。手で判断する)。
ベストラップの作業のファイルは読むだけ(変えない)。

    python make_snapshot.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAP = HERE / "snapshot"
BESTLAP = Path("/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap")
PROTOCOL = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013/protocol.json")
# protocol に無いが、教師の基準(RC 教師)と env_core の import に要るもの(sim/dist の残り)
EXTRA = ["sim/dist/rc-speed-controller.js", "sim/dist/speed-controller.js", "sim/dist/optimized-reference.js",
         "sim/dist/control-delay.js", "sim/dist/world.js", "sim/dist/scene.js", "sim/dist/sensors.js", "sim/dist/ego-sensors.js",
         "sim/SHA256SUMS.txt", "lib/app_emulation.mjs"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    proto = json.loads(PROTOCOL.read_text())
    rows, bad = {}, []
    for rel, spec in proto["snapshot_sources"].items():
        src = Path(spec["path"])
        got = sha(src)
        if got != spec["sha256"]:
            bad.append((rel, spec["sha256"], got))
        rows[rel] = {"source": str(src), "sha256": got, "protocol_sha256": spec["sha256"], "match": got == spec["sha256"]}
    if bad:
        for b in bad:
            print("sha256 が protocol と違います:", b, file=sys.stderr)
        return 1
    for rel in EXTRA:
        src = BESTLAP / rel
        rows[rel] = {"source": str(src), "sha256": sha(src), "protocol_sha256": None, "match": None}
    for rel, r in rows.items():
        dst = SNAP / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(r["source"], dst)
        if sha(dst) != r["sha256"]:
            print("写しの sha256 が違います:", rel, file=sys.stderr)
            return 1
    out = {"schema": "jf-sim-rl-modes-snapshot/v1", "created_at": datetime.now().isoformat(timespec="seconds"),
           "note": "ベストラップの作業のファイルの写し。env_core.mjs の EXP_ROOT は snapshot/ を指す(sim/・rl/ の相対の位置は元と同じ)。"
                   "features.mjs・reference_loop.mjs/.json・policy_forward.mjs はブラウザと Node で同じファイル(バイト単位で同じ写し)",
           "protocol_sha256": sha(PROTOCOL), "files": rows}
    (SNAP / "SNAPSHOT.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"写しました: {len(rows)} ファイル → {SNAP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
