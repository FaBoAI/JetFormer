"""RL の学習の集計: 学習曲線の図・チェックポイントの選定・選んだ方策の書き出し(2026-10-13、RL の学習の段)

    cd ~/jetformer/experiments/20261013_sim_rl_bestlap
    python rl/analyze_rl.py curves            # 学習曲線(fig/rl_learning_curves.png・fig/rl_laptime_eval.png)
    python rl/analyze_rl.py select            # protocol selection のとおりに選ぶ → rl/selection.json、rl/best/<variant>/ に書き出す
    python rl/analyze_rl.py realcar           # 実車(clean_driving_20261011)の現実味の指標 → rl/realism_realcar.json
    python rl/analyze_rl.py trajectory        # 上から見た軌跡 RL と教師(fig/rl_trajectory_vs_teacher.png)

選び方(protocol.json selection): 選べる(eligible)= REC60 60/60 有効・最小すき間 ≥ 0.03 m・STD の単周が完走・ROB24 22/24 以上。
点数 = REC60 の (1 周目 + 2 周目) の平均(小さいほど良い。同点は後のチェックポイント)。種ごとに一番良いもの → 種の中で一番良いもの。
"""
from __future__ import annotations

import glob
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPORTS = Path("/home/jetson/jetformer_reports/sim_rl_bestlap_20261013")
RL = REPORTS / "rl"
FIG = REPORTS / "fig"
TEACHER = json.loads((REPORTS / "env" / "teacher_baseline.json").read_text())
T_LAP1 = TEACHER["teacher_rec60_means"]["lap1_mean_s"]
T_LAP2 = TEACHER["teacher_rec60_means"]["lap2_mean_s"]
T_SINGLE = TEACHER["STD_original"]["single_lap_from_stop_s"]
LEARNED = 8.964117142565293
H1_SINGLE = 8.831

# 図の色(dataviz の参照パレット。種ごとに固定の順番)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"main-s0": "#2a78d6", "main-s1": "#eb6834", "main-s2": "#1baf7a", "open-s0": "#eda100", "pilot-main-s100": "#8a8984"}
LABELS = {"main-s0": "main 種0", "main-s1": "main 種1", "main-s2": "main 種2", "open-s0": "open 種0(参考)", "pilot-main-s100": "パイロット 種100"}


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def runs() -> list[str]:
    names = [n for n in ["pilot-main-s100", "main-s0", "main-s1", "main-s2", "open-s0"] if (RL / n / "progress.jsonl").exists()]
    # r2-* は手順の外の参考の追加実験(物理のずれのランダム化)。選定・本番の曲線には入れない(rl/r2/ の図と表で別に扱う)
    names += sorted(p.parent.name for p in RL.glob("*/progress.jsonl") if p.parent.name not in names and not p.parent.name.startswith("r2-"))
    return names


def load_progress(name: str) -> list[dict]:
    out = []
    for line in (RL / name / "progress.jsonl").read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass   # 電源断で途中まで書かれた最後の行
    # 再開で同じステップが 2 回あれば後のものを使う
    by = {}
    for r in out:
        by[r["steps"]] = r
    return [by[k] for k in sorted(by)]


def load_evals(name: str) -> list[dict]:
    rows = []
    for p in sorted((RL / name / "evals").glob("step_*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        step = int(p.stem.split("_")[1])
        r = d.get("REC60", {}).get("summary", {})
        rows.append({"step": step, "path": str(p), "policy": str(RL / name / "policies" / p.name), "eligible": bool(d.get("eligible")),
                     "score": d.get("score"), "rec60_valid": r.get("valid_two_laps"), "rec60_lap1_mean": r.get("lap1", {}).get("mean"),
                     "rec60_lap2_mean": r.get("lap2", {}).get("mean"), "rec60_lap1_n": r.get("lap1", {}).get("n", 0),
                     "rec60_lap2_n": r.get("lap2", {}).get("n", 0), "rec60_min_clearance": r.get("min_clearance"),
                     "rec60_collisions": r.get("collisions"), "rec60_reasons": r.get("reasons"),
                     "std_single": d.get("STD", {}).get("single_lap_from_stop"), "std_flying": d.get("STD", {}).get("flying_lap"),
                     "rob24_valid": d.get("ROB24", {}).get("summary", {}).get("valid_two_laps")})
    return rows


def smooth(y, k=9):
    y = np.asarray(y, float)
    if len(y) < 3:
        return y
    k = min(k, len(y))
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        lo = max(0, i - k + 1)
        w = y[lo:i + 1]
        w = w[np.isfinite(w)]
        out[i] = w.mean() if len(w) else np.nan
    return out


def _style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_xlabel("環境のステップ数 [百万]", color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def _jp_font():
    import matplotlib
    from matplotlib import font_manager
    for name in ("Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAGothic", "TakaoGothic", "VL Gothic", "Droid Sans Fallback"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = name
            return name
    return None


def curves() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _jp_font()
    FIG.mkdir(parents=True, exist_ok=True)
    names = runs()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), facecolor=SURFACE)
    (a_ret, a_comp), (a_eval, a_lap) = axes
    xmax = 0.0
    for n in names:
        P = load_progress(n)
        if not P:
            continue
        c = SERIES.get(n, "#4a3aa7")
        lab = LABELS.get(n, n)
        pilot = n.startswith("pilot")
        x = np.array([r["steps"] for r in P]) / 1e6
        xmax = max(xmax, float(x[-1]))
        ret = [r["episodes"].get("return_mean", np.nan) for r in P]
        a_ret.plot(x, smooth(ret), color=c, lw=1.6, ls=":" if pilot else "-", label=lab + ("(調整前の報酬。比べられない)" if pilot else ""))
        one = [r["episodes"].get("lap1", {}).get("n", 0) / r["episodes"]["n"] if r["episodes"].get("n") else np.nan for r in P]
        two = [r["episodes"].get("two_valid_laps_frac", np.nan) if r["episodes"].get("n") else np.nan for r in P]
        a_comp.plot(x, smooth(one), color=c, lw=1.6, label=lab)
        a_comp.plot(x, smooth(two), color=c, lw=1.0, ls="--")
        lap1 = [r["episodes"].get("lap1", {}).get("median", np.nan) for r in P]
        fly = [r["episodes"].get("flying", {}).get("median", np.nan) for r in P]
        a_lap.plot(x, smooth(lap1), color=c, lw=0.9, alpha=0.35)
        a_lap.plot(x, smooth(fly), color=c, lw=0.9, alpha=0.35, ls="--")
        E = load_evals(n)
        if E:
            xe = np.array([e["step"] for e in E]) / 1e6
            a_eval.plot(xe, [e["rec60_valid"] or 0 for e in E], color=c, lw=1.6, marker="o", ms=4, label=lab)
            l1 = [e["rec60_lap1_mean"] if e["rec60_lap1_n"] == 60 else np.nan for e in E]
            l2 = [e["rec60_lap2_mean"] if e["rec60_lap2_n"] == 60 else np.nan for e in E]
            if not pilot and any(np.isfinite(v) for v in l1):
                a_lap.plot(xe, l1, color=c, lw=1.8, marker="o", ms=5, label=lab)
                a_lap.plot(xe, l2, color=c, lw=1.4, ls="--", marker="s", ms=4, mfc=SURFACE)
    _style(a_ret, "学習中の 1 回(20 秒)あたりの報酬(確率的な方策、9 更新の移動平均)", "報酬の平均")
    _style(a_comp, "有効な周の割合(確率的。実線 1 周目が有効・破線 2 周とも有効)", "割合")
    a_comp.set_ylim(-0.02, 1.02)
    _style(a_eval, "決定的な方策で REC60(記録の 60 条件)を 2 周とも有効に走れた数", "シーン数 / 60")
    a_eval.set_ylim(-1, 61)
    _style(a_lap, "周回時間(丸 1 周目・四角 2 周目。太線 決定的 REC60 平均・薄線 学習中)", "秒")
    for ref, txt in ((T_LAP1, f"教師 1 周目 {T_LAP1:.3f}"), (T_LAP2, f"教師 2 周目 {T_LAP2:.3f}")):
        a_lap.axhline(ref, color=INK2, lw=1.1, ls="-.")
        a_lap.text(0.01, ref, " " + txt, color=INK2, fontsize=8, va="bottom", transform=a_lap.get_yaxis_transform())
    lo = [v for n in names for e in load_evals(n) for v in (e["rec60_lap1_mean"], e["rec60_lap2_mean"]) if v is not None and e["rec60_lap1_n"] == 60]
    a_lap.set_ylim(min(lo + [T_LAP2 - 0.2]) - 0.08, max(lo + [T_LAP1 + 0.9]) + 0.05)
    for ax in axes.ravel():
        ax.set_xlim(0, max(xmax, 1) * 1.02)
    for ax in (a_ret, a_comp, a_eval):
        ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2)
    a_lap.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="upper right")
    fig.suptitle("RL(PPO)の学習曲線 — main は速度上限 4.5 m/s、open は 7.24 m/s(参考)", color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "rl_learning_curves.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)

    # 周回時間(決定的な方策): REC60 1 周目・2 周目の平均と STD の停止から 1 周
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    for ax, key, title, ref, ref_label in [
        (axes[0], "rec60_lap1_mean", "REC60 1 周目(停止から)の平均", T_LAP1, "教師"),
        (axes[1], "rec60_lap2_mean", "REC60 2 周目(フライング)の平均", T_LAP2, "教師"),
        (axes[2], "std_single", "STD 単周(停止から 1 周、標準の開始)", T_SINGLE, "教師")]:
        vals = []
        for n in names:
            E = load_evals(n)
            if not E:
                continue
            need = "rec60_lap1_n" if key == "rec60_lap1_mean" else "rec60_lap2_n"
            xs = np.array([e["step"] for e in E]) / 1e6
            ys = [e[key] if (key == "std_single" or e[need] == 60) else np.nan for e in E]
            ys = [np.nan if v is None else v for v in ys]
            vals += [v for v in ys if np.isfinite(v)]
            if not any(np.isfinite(v) for v in ys):
                continue
            ax.plot(xs, ys, color=SERIES.get(n, "#4a3aa7"), lw=1.8, marker="o", ms=4, label=LABELS.get(n, n))
            el = [(x, y) for x, y, e in zip(xs, ys, E) if e["eligible"] and np.isfinite(y)]
            if el:
                ax.scatter(*zip(*el), s=70, facecolors="none", edgecolors=SERIES.get(n, "#4a3aa7"), linewidths=1.4)
        _style(ax, title, "秒")
        ax.axhline(ref, color=INK2, lw=1.2, ls="--")
        ax.text(0.01, ref, f" {ref_label} {ref:.3f}", color=INK2, fontsize=8, va="bottom", transform=ax.get_yaxis_transform())
        thr = ref - 0.1 if key != "std_single" else H1_SINGLE
        ax.axhline(thr, color=INK2, lw=1.0, ls=":")
        ax.text(0.01, thr, f" H1 の条件 {thr:.3f}", color=INK2, fontsize=8, va="top", transform=ax.get_yaxis_transform())
        if key == "std_single":
            ax.axhline(LEARNED, color="#8a8984", lw=1.0, ls="-.")
            ax.text(0.99, LEARNED, f"学習済み ONNX {LEARNED:.3f} ", color=INK2, fontsize=8, va="bottom", ha="right", transform=ax.get_yaxis_transform())
        if vals:
            top = max(ref, LEARNED if key == "std_single" else ref) + 0.15
            ax.set_ylim(min(vals + [thr]) - 0.08, top)
    fig.suptitle("決定的な方策の周回時間(200 万ステップごとの評価、高速版のコア。白抜きの丸 = 選べるチェックポイント。上の範囲の外の点は省略)", color=INK, fontsize=12, x=0.01, ha="left")
    h, l = axes[2].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=len(l), fontsize=8, frameon=False, labelcolor=INK2, bbox_to_anchor=(0.995, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(FIG / "rl_laptime_eval.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print("図:", FIG / "rl_learning_curves.png", FIG / "rl_laptime_eval.png")


def select() -> dict:
    import torch
    sys.path.insert(0, str(HERE))
    out = {"schema": "rl-bestlap-selection/v1", "rule": "protocol.json selection", "teacher_rec60_lap1_mean_s": T_LAP1, "teacher_rec60_lap2_mean_s": T_LAP2,
           "teacher_single_lap_from_stop_s": T_SINGLE, "runs": {}}
    for n in runs():
        if n.startswith("pilot"):
            continue
        E = load_evals(n)
        el = [e for e in E if e["eligible"] and e["score"] is not None]
        best = min(el, key=lambda e: (e["score"], -e["step"])) if el else None
        fastest_any = min([e for e in E if e["score"] is not None], key=lambda e: e["score"], default=None)
        run_json = json.loads((RL / n / "run.json").read_text())
        out["runs"][n] = {"variant": run_json["variant"], "seed": run_json["seed"], "total_steps": run_json["total_steps"],
                          "done": (RL / n / "done.json").exists(), "evals": len(E), "eligible_evals": len(el),
                          "eligible_steps": [e["step"] for e in el], "best": best, "best_score_any_ineligible": fastest_any}
    for variant in ("main", "open"):
        cands = [(n, r["best"]) for n, r in out["runs"].items() if r["variant"] == variant and r["best"]]
        if not cands:
            out[f"selected_{variant}"] = None
            continue
        n, b = min(cands, key=lambda c: (c[1]["score"], -c[1]["step"]))
        dst = RL / "best" / variant
        dst.mkdir(parents=True, exist_ok=True)
        pol = Path(b["policy"])
        shutil.copy2(pol, dst / "policy.json")
        w = json.loads(pol.read_text())
        (dst / "policy_weights.mjs").write_text("// RL の方策の重み(rl/ppo.py export_policy_json)。ブラウザの差し込みモジュール用の ESM\nexport default " + json.dumps(w) + ";\n")
        shutil.copy2(Path(b["path"]), dst / "eval_fast.json")
        ck = RL / n / "checkpoints" / (pol.stem + ".pt")
        shutil.copy2(ck, dst / "checkpoint.pt")
        export_torch(ck, w, dst)
        rj = json.loads((RL / n / "run.json").read_text())
        (dst / "provenance.json").write_text(json.dumps({
            "run": n, "variant": variant, "seed": rj["seed"], "step": b["step"], "score_rec60_lap1_plus_lap2": b["score"],
            "policy_json_sha256": sha256(dst / "policy.json"), "source_policy": str(pol), "source_checkpoint": str(ck),
            "reward_version": rj.get("env", {}).get("reward_version"), "features_version": rj.get("env", {}).get("features_version"),
            "reference_loop_sha256": rj.get("env", {}).get("reference_loop_sha256"), "speed_cap_mps": rj.get("env", {}).get("speed_cap_mps"),
            "protocol_sha256": rj.get("protocol_sha256"), "ppo": rj.get("ppo"), "control_delay_frames_trained": 0,
            "note": "rl/best の方策。JS は rl/policy_forward.mjs + policy.json(または policy_weights.mjs)、観測は rl/features.mjs"},
            ensure_ascii=False, indent=1) + "\n")
        out[f"selected_{variant}"] = {"run": n, "step": b["step"], "score": b["score"], "policy_json": str(dst / "policy.json"),
                                      "policy_json_sha256": sha256(dst / "policy.json"), "policy_weights_mjs": str(dst / "policy_weights.mjs"),
                                      "policy_pt": str(dst / "policy.pt"), "policy_torchscript": str(dst / "policy_ts.pt"),
                                      "checkpoint": str(dst / "checkpoint.pt"), "eval_fast": str(dst / "eval_fast.json"), "source_policy": str(pol)}
    (RL / "selection.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k.startswith("selected")}, ensure_ascii=False, indent=1))
    for n, r in out["runs"].items():
        print(n, "選べる", r["eligible_evals"], "/", r["evals"], "最良", (r["best"] or {}).get("step"), (r["best"] or {}).get("score"))
    return out


def export_torch(ckpt: Path, weights: dict, dst: Path) -> None:
    """PyTorch の書き出し: policy.pt(方策の重み + 固定の正規化 + meta)と TorchScript(正規化・切り出し込みの決定的な方策)。"""
    import torch
    from torch import nn
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    actor_sd = {k[len("actor."):]: v for k, v in d["model"].items() if k.startswith("actor.")}
    mean = torch.tensor(weights["obs_mean"], dtype=torch.float32)
    std = torch.tensor(weights["obs_std"], dtype=torch.float32)
    torch.save({"schema": "rl-bestlap-policy-torch/v1", "actor_state_dict": actor_sd, "obs_mean": mean, "obs_std": std,
                "obs_clip": weights["obs_clip"], "log_std": d["model"]["log_std"], "meta": weights["meta"],
                "arch": "Linear(79,256)-Tanh-Linear(256,256)-Tanh-Linear(256,2)、出力を [-1,1] に切る(展開時は平均)"}, dst / "policy.pt")

    class Deterministic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(79, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 2))
            self.net.load_state_dict(actor_sd)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
            self.clip = float(weights["obs_clip"])

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            x = torch.clamp((obs - self.mean) / self.std, -self.clip, self.clip)
            return torch.clamp(self.net(x), -1.0, 1.0)

    m = Deterministic().eval()
    torch.jit.script(m).save(str(dst / "policy_ts.pt"))


def realcar() -> dict:
    """実車 clean_driving_20261011 の aligned から、protocol realism_metrics と同じ指標(走行中 = 0.3 m/s 超)。

    速さ = −eRPM × 0.5e-3、ヨーレート = IMU のジャイロ z(aligned は rad/s、正が左)の直前 10 サンプルの平均、横加速度 = |v·ω|。
    舵の指令の角度は、シミュレーターの記録と同じ換算(中心 1442 µs、左 558・右 442 µs で ±0.52 rad)を当てた目安。
    """
    files = sorted(glob.glob("/home/jetson/jetformer/data/datasets/clean_driving_20261011/aligned/*.npz"))
    sp, lat, steer_rate, edge, thr, steer_all = [], [], [], [], [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        veh, ok = z["vehicle"].astype(np.float64), z["vehicle_valid"] & z["erpm_valid"]
        v = -veh[:, 3] * 0.5e-3
        w = z["imu"][:, :, 5].mean(1).astype(np.float64)
        su = veh[:, 0]
        a0 = np.where(su >= 1442, (su - 1442) / 558, (su - 1442) / 442)
        d = np.clip(a0, -1, 1) * 0.52
        t = z["timestamps_us"].astype(np.float64) * 1e-6
        dd = np.abs(np.diff(d, prepend=d[0])) / np.maximum(np.diff(t, prepend=t[0] - 1 / 30), 1e-3)
        mv = ok & (v > 0.3)
        sp.append(v[mv]); lat.append(np.abs(v * w)[mv]); steer_rate.append(dd[mv]); thr.append(veh[mv, 1])
        edge.append(((su <= 1063) | (su >= 1972))[mv]); steer_all.append(su[mv])
    sp, lat, steer_rate, edge, thr = map(np.concatenate, (sp, lat, steer_rate, edge, thr))
    out = {"source": "clean_driving_20261011 aligned(" + str(len(files)) + " ファイル)", "moving_frames": int(sp.size),
           "max_speed": float(sp.max()), "speed_median": float(np.median(sp)), "speed_p99": float(np.quantile(sp, .99)),
           "frac_over_4mps": float((sp > 4).mean()), "lat_acc_max": float(lat.max()), "lat_acc_p95": float(np.quantile(lat, .95)),
           "lat_acc_median": float(np.median(lat)), "steer_cmd_rate_p95": float(np.quantile(steer_rate, .95)),
           "steer_edge_frac": float(edge.mean()), "throttle_pwm_p01": float(np.quantile(thr, .01)),
           "notes": ["eRPM は車輪の回転なので滑ると速く出る", "横加速度はジャイロ × eRPM の速さの目安(滑りで大きく出る)",
                     "舵の角度はシミュレーターの換算を当てた目安。実車の舵の速さはサーボの応答を含まない指令の変化",
                     "舵が端 = PWM ≤ 1063 µs または ≥ 1972 µs(design.md 3.4 節)"]}
    (RL / "realism_realcar.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def trajectory(final_eval_path: Path | None = None) -> None:
    """上から見た軌跡: RL(選んだ main の STD 単周)と教師(STD 単周)と学習済み ONNX の 8.964 秒の周。障害物・ゲート・黒シートつき。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    _jp_font()
    fe = json.loads(Path(final_eval_path or RL / "final_eval_main.json").read_text())
    world = json.loads((EXP / "sim" / "world-v3.json").read_text())
    rec = {"rl": fe["rl"]["STD"]["single"]["record"], "teacher": fe["teacher"]["STD"]["single"]["record"], "learned": fe["learned_onnx"]["record"]}
    tr = {k: json.loads(Path(v).read_text())["trace"] for k, v in rec.items()}
    times = {"rl": fe["rl"]["STD"]["single"]["single_lap_s"], "teacher": fe["teacher"]["STD"]["single"]["single_lap_s"], "learned": fe["learned_onnx"]["single_lap_s"]}
    colors = {"rl": "#2a78d6", "teacher": "#eb6834", "learned": "#1baf7a"}
    names = {"rl": "RL(PPO main)", "teacher": "作者の RC 教師", "learned": "学習済み ONNX(遅延 1)"}
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.6, 1]})
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for s in world["surfaces"]:
            if s["id"] == "carpet":
                continue
            col = {"shortcut": "#2b2b2b", "blue-mat": "#b7d3f6", "turf": "#cfe8cf", "ptfe": "#ece6f5"}.get(s["id"], "#e4e3df")
            ax.add_patch(Rectangle((s["x"] - s["w"] / 2, s["z"] - s["d"] / 2), s["w"], s["d"], facecolor=col, edgecolor="none", alpha=0.55, zorder=0))
            ax.text(s["x"], s["z"], {"shortcut": "黒シート", "blue-mat": "青マット", "turf": "芝", "ptfe": "PTFE"}.get(s["id"], s["id"]),
                    fontsize=7, color=INK2 if s["id"] != "shortcut" else "#fcfcfb", ha="center", va="center", zorder=1)
        for g in world["segments"]:
            ax.plot([g["a"][0], g["b"][0]], [g["a"][1], g["b"][1]], color="#6b6a66", lw=max(0.8, g["radius"] * 120), solid_capstyle="round", zorder=2)
        for d in world["disks"]:
            ax.add_patch(Circle((d["x"], d["z"]), d["radius"], facecolor="#9b9a95" if d["radius"] > 0.05 else "#52514e", edgecolor="none", alpha=0.7, zorder=2))
        r = world["route"]
        for i, p in enumerate(r):
            ax.plot(p[0], p[1], marker="o", ms=3, color=INK2, zorder=3)
            if ax is axes[0]:
                ax.text(p[0], p[1] + 0.08, str(i), fontsize=6, color=INK2, ha="center", zorder=3)
        for k in ("teacher", "learned", "rl"):
            x = [t["x"] for t in tr[k]]
            zz = [t["z"] for t in tr[k]]
            ax.plot(x, zz, color=colors[k], lw=1.8 if k != "learned" else 1.2, ls="-" if k != "learned" else "--", zorder=4 if k == "rl" else 3,
                    label=f"{names[k]} {times[k]:.3f} 秒" if times[k] else names[k])
        ax.set_aspect("equal")
        ax.tick_params(colors=INK2, labelsize=7)
        for s in ax.spines.values():
            s.set_color(GRID)
    b = world["bounds"]
    axes[0].set_xlim(b["minX"], b["maxX"]); axes[0].set_ylim(b["maxZ"], b["minZ"])
    axes[0].set_title("停止から 1 周(標準の開始、元の sim-core.js)。数字はゲートの番号", color=INK, fontsize=11, loc="left")
    axes[0].legend(fontsize=9, frameon=False, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    axes[0].set_xlabel("x [m]", color=INK2); axes[0].set_ylabel("z [m](上が −z)", color=INK2)
    # 速さの色(RL)
    x = np.array([t["x"] for t in tr["rl"]]); zz = np.array([t["z"] for t in tr["rl"]]); v = np.array([t["speed"] for t in tr["rl"]])
    sc = axes[1].scatter(x, zz, c=v, cmap="Blues", vmin=0, vmax=4.6, s=9, zorder=5)
    cb = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.02)
    cb.set_label("RL の速さ [m/s]", color=INK2); cb.ax.tick_params(colors=INK2, labelsize=7)
    axes[1].set_xlim(-1.2, 4.6); axes[1].set_ylim(1.4, -3.2)
    axes[1].set_title("拡大: 黒シートの近道とその先(点 = RL の各フレーム)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "rl_trajectory_vs_teacher.png", dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("図:", FIG / "rl_trajectory_vs_teacher.png")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "curves"
    {"curves": curves, "select": select, "realcar": realcar, "trajectory": lambda: trajectory(Path(sys.argv[2]) if len(sys.argv) > 2 else None)}[cmd]()
