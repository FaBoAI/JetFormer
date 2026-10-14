"""no_wall r4c の追記の決まり protocol_no_wall_r4c.json を書いて固める(2026-10-14 21 時、電源断のあとの再開で足した)。
protocol_no_wall_r4.json(ad90c926…)・protocol_no_wall_r4b.json(e6a77b7f…)は変えない。training と selection.candidates だけを足す。
r4b の終わりの点の検証の結果を読む前に書いた(決めたのは下調べの組 dev4f・dev4e の点だけを見て)。sha256 は r3/protocol_r4c.sha256 に書く。
    python3 write_protocol_nw4c.py
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

REP = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3")
E = Path("/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/nw4")
O = REP / "no_wall_r4"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


out = REP / "protocol_no_wall_r4c.json"
if out.exists():
    raise SystemExit("protocol_no_wall_r4c.json はもう固めてあります(書き直さない)")
init = O / "train/checkpoints/step_003014656.pt"
code = {p.relative_to(E).as_posix(): sha(p) for p in sorted(E.rglob("*")) if p.is_file() and p.suffix in (".py", ".mjs", ".json", ".sh")
        and "__pycache__" not in p.parts and not p.name.startswith("manifest")}
P = {
    "title": "no_wall r4 の学習の続き(r4c)の追記 — protocol_no_wall_r4.json(ad90c926…)と protocol_no_wall_r4b.json(e6a77b7f…)を変えずに、training と selection.candidates だけを足す。"
             "シールド v5・検証の組・試験の組・判定は r4 のまま",
    "version": "no_wall-r4c-1",
    "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "written_by": "no_wall r4 の段(電源断 16:28 のあとの 20 時の再開)。r4b の学習(種 25、300 万)が終わったあと、r4b の点の検証の結果を読む前・試験の前に固定",
    "protocol_no_wall_r4_sha256": sha(REP / "protocol_no_wall_r4.json"),
    "protocol_no_wall_r4b_sha256": sha(REP / "protocol_no_wall_r4b.json"),
    "why": [
        "下調べの組だけで見た(検証・試験の点を含まない): ゲート 16〜18 の右 0.15・0.2 m × 左向き 12・15°(dev4e の点 12)で、止まったまま(周 0)の点は r3 の方策 3・"
        "r4b の 50 万 1・100 万 1・150 万 1・200 万 1・250 万 1・300 万 1(どれもゲート 17 の右 0.2 m・左向き 15°)。",
        "ゲート 17 の細かい格子 dev4f(左右 {−0.2, −0.185, −0.175, −0.165, −0.155} m × 向き {15, 14, 13.5, 12.5, 12}°。検証の ±0.18 m・±13°、試験の ±0.16・±0.19 m × ±11.5・±14.5° を含まない)で、"
        "止まる点は r4b の 200 万で 3/25 → 250 万で 1/25 → 300 万で 1/25(右 0.2 m・左向き 15°)。学習が進むほど狭くなるが、記録の開始の分布の角(左右 ±0.2 m・向き ±15° は分布の中)に残る",
        "止まり(N5)は確認 r3 の必ず直すことの 1 つ(止まりを無くす。開始の分布や 8 cm のしきい値は変えない)。r4b の箱(ゲート 17 の右 0.1〜0.2 m・左向き 8〜15°)は広く、角(0.17〜0.2 m・12〜15°)は箱の約 13%",
        "lock rl は空いていて、r4b の学習は 1 本 30 分ほどで終わったので、r4b の終わりの点から角を多くして 150 万ステップ続ける(r4c)。r4c の点が検証の組で全部を満たさなければ、"
        "決まりのとおり r4b・r4 の点・r3 の方策から選ぶ(r4c は候補を足すだけ)"
    ],
    "training": {
        "run": "no_wall_r4c(r3/nw4/train_c/train_nw4.py・env_nw4.mjs NW4C_DEFAULTS。train/ の写しに run を足しただけ。報酬・物理の内訳・速さの見張り・止まりの見張りは r3・r4・r4b のまま)",
        "init": "r4b の終わりの点 no_wall_r4/train/checkpoints/step_003014656.pt(方策 22042f96…)",
        "init_sha256": sha(init),
        "init_policy_sha256": sha(O / "train/policies/step_003014656.json"),
        "seed": 26,
        "steps_max": 1500000,
        "lr": 3e-05,
        "ckpt_every": 500000,
        "save_every": 200000,
        "start_mix": [
            {"p": 0.30, "what": "記録の開始の分布(ゲート U{0..32}・左右 ±0.2 m・向き ±15°・開始のすき間 ≥ 0.12 m)"},
            {"p": 0.15, "what": "開始のすき間 12〜15 cm の一覧 starts_lowc_nw4.json(r4・r4b と同じ)"},
            {"p": 0.15, "what": "分布の角: 全ゲート、|左右| 0.17〜0.2 m・|向き| 12〜15°(符号はそれぞれ半々、開始のすき間 ≥ 0.12 m)"},
            {"p": 0.12, "what": "ゲート 17、左右 −0.2〜−0.17 m・向き +12〜+15°"},
            {"p": 0.03, "what": "ゲート 16、同じ箱"},
            {"p": 0.03, "what": "ゲート 18、同じ箱"},
            {"p": 0.07, "what": "r3 のむずかしい開始(ゲート 7〜10・16・17・20〜23、|左右| 0.1〜0.2 m・|向き| 7.5〜15°)"},
            {"p": 0.15, "what": "広め ±0.25 m・±20°(開始のすき間 ≥ 0.10 m)"}
        ],
        "lowc_list_sha256": sha(E / "train/starts_lowc_nw4.json"),
        "lock": "jf-heavy --mem 2G --lock rl -- nice -n 15(r3/nw4/run_train_nw4c.sh。last.pt は 20 万ごと、止めたあとは同じコマンドで続きから)",
        "note": "角の箱は試験の EDGE の点(ゲート 16〜18 の −0.19 m・+14.5°)と検証の点(−0.18 m・+13°)を含む範囲。学習の開始は一様乱数なので同じ点を覚えるわけではないが、"
                "試験の組の作り方(固定)を知っていて箱を決めたことはここに書いておく(r4b と同じ)"
    },
    "selection": {
        "candidates": "r4c の学習の 50 万ステップごとの点(train_c/、step_000000000 は r4b の終わりの点と同じ方策なので候補に足さない)、r4b の点(train/)、r4 の点(train_a/)、step_000000000(r3 の方策)",
        "order": "「一番後」の順は r4c の点(ステップの順)> r4b の点 > r4 の点 > step_000000000",
        "rule": "protocol_no_wall_r4.json selection.rule のまま(検証の組 NW-V-R4 で全部を満たし止まり 0 の点のうち一番後。無ければ満たす数・標準の物理の最小すき間)",
        "code": "r3/nw4/val_loop_nw4b.py・select_nw4b.py(r4c の点を順 3 で足した版)",
        "test_runs": "試験の組は選んだ 1 つにだけ 1 回(r4 の試験の組はまだ使っていない)"
    },
    "unchanged": "シールド jf-shield/v5 の値・ファイル(runtime_v5/shield.mjs 8ece9be4…)、検証の組・試験の組・判定(N1_r4〜N7_r4)・記録の前の予想・画面の数字の決まりは protocol_no_wall_r4.json のまま",
    "files": {"code_sha256_at_freeze": code}
}
out.write_text(json.dumps(P, ensure_ascii=False, indent=1) + "\n")
(REP / "protocol_r4c.sha256").write_text(f"{sha(out)}  protocol_no_wall_r4c.json\n")
print(sha(out))
