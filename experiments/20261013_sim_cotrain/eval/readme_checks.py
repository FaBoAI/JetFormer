"""README 用の参考の数字(CPU だけ・判定には使わない)。

1. カメラ画像の見た目: 実機(clean_driving_20261011 の 52 シーン)とシミュレーター(60 シーン)から
   300 枚ずつ無作為に選び、明るさ・コントラスト(灰色の標準偏差)・彩度・輪郭の強さ・下側(床)の明るさの中央値を出す。
2. シミュレーターの 60 シーンがどれだけ違うか: 各シーンの操作値(VEHICLE.npz の pwm、600 フレーム)を
   フレーム番号でそろえ、60 シーンの平均の軌跡からのずれ [µs] を出す。
   比べる目安として、1 シーンの中で操作値が時間でどれだけ動くか(標準偏差)も出す。

出力: readme_checks.json(このファイルの隣)。
"""
import glob
import json
import os
import random

import numpy as np
from PIL import Image, ImageFilter

JF = "/home/jetson/jetformer"
SIM_DS = "jetformer-simulator-id8-20min_20261012-150956_b55616cf"
REAL_DS = "clean_driving_20261011"
HERE = os.path.dirname(os.path.abspath(__file__))


def real_frames():
    """実機のデータセットの元のシーンのカメラ画像の一覧。"""
    with open(f"{JF}/data/datasets/{REAL_DS}/dataset_card.json") as f:
        card = json.load(f)
    out = []
    for s in card["source_scenes"]:
        log = s.rsplit("-", 1)[0]
        out += sorted(glob.glob(f"{JF}/data/raw/{log}/{s}/CAM_FRONT/*.jpg"))
    return out


def sim_scenes():
    return sorted(glob.glob(f"{JF}/data/raw/ds-{SIM_DS}/ds-{SIM_DS}-00??"))


def image_stats(files, n=300, seed=0):
    """無作為の n 枚の見た目の中央値。"""
    rng = random.Random(seed)
    rows = []
    for f in rng.sample(files, n):
        im = Image.open(f).convert("RGB")
        g = np.asarray(im.convert("L"), dtype=float)
        hsv = np.asarray(im.convert("HSV"), dtype=float)
        edge = np.asarray(im.convert("L").filter(ImageFilter.FIND_EDGES), dtype=float)
        h = g.shape[0]
        rows.append((g.mean(), g.std(), hsv[..., 1].mean(), edge.mean(),
                     g[: h // 3].mean(), g[h * 2 // 3:].mean()))
    med = np.median(np.array(rows), axis=0)
    keys = ["brightness", "contrast_gray_std", "saturation", "edge_strength", "top_third_brightness",
            "bottom_third_brightness"]
    return {k: round(float(v), 1) for k, v in zip(keys, med)}


def sim_trajectory_spread():
    """60 シーンの操作値を平均の軌跡と比べる。"""
    P = []
    for s in sim_scenes():
        P.append(np.load(s + "/VEHICLE.npz")["pwm"].astype(float))
    n = min(len(p) for p in P)
    A = np.stack([p[:n] for p in P])  # (シーン, フレーム, [ステア, スロットル, …])
    out = {"scenes": int(A.shape[0]), "frames": int(n)}
    for ch, name in ((0, "steer"), (1, "throttle")):
        x = A[:, :, ch]
        dev = np.abs(x - x.mean(0)).mean(1)
        out[name] = {
            "mean_abs_dev_from_mean_traj_us": round(float(dev.mean()), 1),
            "max_scene_dev_us": round(float(dev.max()), 1),
            "within_scene_std_over_time_us": round(float(x.std(1).mean()), 1),
        }
    return out


def main():
    sims = sorted(glob.glob(f"{JF}/data/raw/ds-{SIM_DS}/*/CAM_FRONT/*.jpg"))
    reals = real_frames()
    res = {
        "note": "README の参考の数字。判定には使わない。画像は 224x224、300 枚ずつ(random.Random(0))の中央値。",
        "images": {"sim": image_stats(sims), "real": image_stats(reals),
                   "frames_available": {"sim": len(sims), "real": len(reals)}},
        "sim_trajectory_spread": sim_trajectory_spread(),
    }
    path = os.path.join(HERE, "readme_checks.json")
    with open(path, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
