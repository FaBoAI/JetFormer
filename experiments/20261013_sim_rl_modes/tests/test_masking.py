"""学習から外す決まり(Python の参照実装と、ブラウザ・Node の perturb.mjs の maskFromLearn が同じ)。"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "rl"))
from masking import mask_from_learn, apply_label_mask, learn_from_frames_jsonl  # noqa: E402


def test_interval_rule():
    learn = np.ones(600, int)
    learn[100:110] = 0                       # ps = 100, pe = 110
    m = mask_from_learn(learn)
    assert np.array_equal(np.nonzero(m)[0], np.arange(90, 109))   # k ∈ [ps − 10, pe − 2]


def test_apply_keeps_tail_and_counts():
    fv = np.ones(600, bool)
    fv[-10:] = False
    mode = np.full(600, 8)
    learn = np.ones(600, int)
    learn[595:600] = 0
    fv2, mode2, n, newly = apply_label_mask(fv, mode, learn)
    assert not fv2[-10:].any() and (mode2[585:599] == 0).all() and mode2[599] == 8
    assert n == 14 and newly == 5


def test_same_as_js():
    rng = np.random.default_rng(3)
    learn = (rng.random(600) > 0.08).astype(int)
    js = ("const m = await import('" + str(EXP / "rl/runtime/perturb.mjs") + "'); const l = JSON.parse(process.argv[1]);"
          " process.stdout.write(JSON.stringify(Array.from(m.maskFromLearn(l))));")
    out = subprocess.run(["node", "--input-type=module", "-e", js, json.dumps(learn.tolist())], capture_output=True, text=True, check=True)
    assert np.array_equal(np.asarray(json.loads(out.stdout), bool), mask_from_learn(learn))


def test_learn_from_frames():
    # 安全側(fail closed、2026-10-14 recover r2): jf・l の無い行、0/1 でない l は学習しない(0)
    rows = [{"action_info": {"jf": {"l": 1}}}, {"action_info": {"jf": {"l": 0}}}, {"action_info": None}, {},
            {"action_info": {"jf": {}}}, {"action_info": {"jf": {"l": "1"}}}, {"action_info": {"jf": {"l": True}}}]
    assert learn_from_frames_jsonl(rows).tolist() == [1, 0, 0, 0, 0, 0, 0]


def test_same_as_jetformer_import():
    """JetFormer の取り込み(jetformer/sim/label_mask.py)は、この参照実装より 1 フレーム広く外す(安全側)。
    統合の段で [ps − 10, pe − 2] を [ps − 10, pe − 1] に広げた(deviations.md)。増えるのは各区間の pe − 1 だけ"""
    sys.path.insert(0, "/home/jetson/jetformer")
    from jetformer.sim import label_mask
    rng = np.random.default_rng(5)
    learn = (rng.random(600) > 0.1).astype(int)
    jf, ref = label_mask.mask_from_learn(learn), mask_from_learn(learn)
    assert not (ref & ~jf).any()                                   # 参照で外すものは JetFormer でも外す
    extra = np.nonzero(jf & ~ref)[0]
    ends = [j for j in range(600) if learn[j] == 0 and (j + 1 == 600 or learn[j + 1] == 1)]   # 各区間の pe − 1
    assert set(extra.tolist()) <= set(ends)
    rows = [{"frame": f, "action_info": {"jf": {"l": int(learn[f])}}} for f in range(600)]
    assert np.array_equal(label_mask.learn_from_rows(rows, 600)[0], learn_from_frames_jsonl(rows))
