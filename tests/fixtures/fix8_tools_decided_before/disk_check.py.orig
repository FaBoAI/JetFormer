"""係 disk(10/26 夜): 空きディスクの見積もりと、消してよい候補の大きさを測る。**読むだけ。何も消さない・書かない。**

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -I experiments/20261026_fix/fix3/disk/disk_check.py

出すもの: 今の空き・学習 v3 の段 1 の残り・11/3 の朝と夜の見込み・候補ごとの大きさ・v2/B-FAST の last.pth の予測が
保存してあるか(gpu_queue_v2.py prune-last と同じ条件)。報告: jetformer_reports/fix_20261026/fix3/disk.md
"""
import json
import os
import shutil
import sys
import time

RUNS = '/home/jetson/jetformer/data/runs'
RAW = '/home/jetson/jetformer/data/raw'
REP = '/home/jetson/jetformer_reports'
PRED_V2 = REP + '/train_v2_20261025/eval/pred'
KEY_V2 = '3d339512828b17195d568f1b93ee4a28208f2e85'     # gpu_queue_v2.pred_ok と同じ
STAGE1 = ['v3_20261026-%s-aug2-e60-s%d' % (a, s) for s in range(4) for a in ('v3d', 'v3n', 'v3b')]

# 見積もりの数(disk.md 2 章。測った値)
RUN_GB = 0.312        # 1 本の学習の場所(best 56.5 + last 169.4 + ONNX 56.5 + エンジン 29.2 MB。v2・v3d-s0 で測った)
PRED_GB = 0.010       # 1 本の机上の予測(v2 の eval/pred・desk/pred・desk/trt から)
PYTEST_GB = 3.9       # pytest 一式 1 回の作業場所(verify.md)
OTHER_GB = (0.7, 1.0, 2.0)    # 10/27〜11/2 のほかの増え(会話の記録・報告・動画など。少なめ・中・多め)
E5_GB = (0.1, 0.15, 0.2)      # 11/1 の封を開ける(集合と予測)
DAY_GB = (0.5, 1.0, 1.25)     # 11/3 の記録(10/24: 20 本 0.836 GB → 1 本 41.8 MB × 23 本。時間で数えると 0.5)
NIGHT_GB = (0.1, 0.2, 0.4)    # 11/3 の休み・夜の道具・報告・動画


def du(path):
    """ディスクの上の大きさ(st_blocks。ハードリンクは 1 回だけ数える)"""
    seen, tot = set(), 0
    if os.path.isfile(path):
        st = os.lstat(path)
        return st.st_blocks * 512
    for root, _, files in os.walk(path):
        for f in files:
            try:
                st = os.lstat(os.path.join(root, f))
            except OSError:
                continue
            if (st.st_dev, st.st_ino) in seen:
                continue
            seen.add((st.st_dev, st.st_ino))
            tot += st.st_blocks * 512
    return tot


def pred_ok(p, ck):
    try:
        import numpy as np
        with np.load(p, allow_pickle=False) as z:
            return (str(z['key']) == KEY_V2 and str(z['ckpt']) == ck and z['pred'].shape == (10985, 10, 2)
                    and bool(np.isfinite(z['pred']).all()))
    except Exception:          # noqa: BLE001
        return False


def done_run(name):
    d = os.path.join(RUNS, name)
    try:
        res = json.load(open(os.path.join(d, 'result.json')))
    except Exception:          # noqa: BLE001
        return False
    return res.get('state') == 'done' and all(os.path.exists(os.path.join(d, f)) for f in
                                              ('best.pth', 'last.pth', 'jetformer.onnx', 'jetformer_fp16.engine'))


def main():
    free = shutil.disk_usage('/').free / 1e9
    print('# 測った時刻 %s' % time.strftime('%Y-%m-%d %H:%M'))
    print('今の空き: %.2f GB' % free)

    # ---- 学習 v3 の段 1 の残り
    done = [n for n in STAGE1 if done_run(n)]
    part = [n for n in STAGE1 if n not in done and os.path.isdir(os.path.join(RUNS, n))]
    part_gb = sum(du(os.path.join(RUNS, n)) for n in part) / 1e9
    left = (12 - len(done)) * RUN_GB - part_gb + 12 * PRED_GB
    print('段 1: 済み %d 本・途中 %d 本(%.2f GB)・残りで使う見込み %.2f GB(予測込み)' % (len(done), len(part), part_gb, left))
    after1 = free - left
    print('段 1 のあと: %.2f GB(途中でやめる決まりで 1 つの腕の種 2・3 を飛ばすと +%.2f GB)' % (after1, 2 * RUN_GB))
    print('pytest 一式の最中: %.2f GB(作業場所 %.1f GB)。終わって消せば %.2f GB に戻る' % (after1 - PYTEST_GB, PYTEST_GB, after1))
    for label, k in (('少なめ', 0), ('中', 1), ('多め', 2)):
        m = after1 - OTHER_GB[k] - E5_GB[k]
        n = m - DAY_GB[k] - NIGHT_GB[k]
        print('11/3(増え %s): 朝 %.1f GB・夜 %.1f GB / pytest の作業場所を残すと 朝 %.1f・夜 %.1f' % (
            label, m, n, m - PYTEST_GB, n - PYTEST_GB))

    # ---- 候補の大きさ
    v2 = sorted(n for n in os.listdir(RUNS) if n.startswith('v2_20261025-'))
    bf = sorted(n for n in os.listdir(RUNS) if n.startswith('bfast_20261025-'))
    rows = []
    for group, names in (('A1 v2', v2), ('A2 B-FAST', bf)):
        tot, ok = 0, 0
        for n in names:
            lp = os.path.join(RUNS, n, 'last.pth')
            if not os.path.exists(lp):
                continue
            tot += du(lp)
            good = (done_run(n) and pred_ok(os.path.join(PRED_V2, n + '.npz'), 'best')
                    and pred_ok(os.path.join(PRED_V2, n + '_last.npz'), 'last'))
            ok += good
            if not good:
                print('  注意: %s は予測(best・last)が読めないか学習が済んでいない → 消さない' % n)
        rows.append((group + ' の last.pth(%d 本中 予測あり %d 本)' % (len(names), ok), tot))
    rows.append(('C1 ゴミ箱 ~/.local/share/Trash', du('/home/jetson/.local/share/Trash')))
    jl = 0
    for root, _, files in os.walk(REP + '/sim_rl_modes_20261013'):
        for f in files:
            if f.endswith('.jsonl') and f != 'progress.jsonl':
                jl += os.lstat(os.path.join(root, f)).st_blocks * 512
    rows.append(('C2 sim_rl_modes_20261013 の *.jsonl(progress.jsonl 以外)', jl))
    dup = 0
    for d in ('sim_rl_bestlap_20261013', 'sim_dart_20261014', 'sim_diverse_train_20261015'):
        b = os.path.join(REP, d, 'bundles')
        for n in sorted(os.listdir(b)) if os.path.isdir(b) else []:
            f = os.path.join(b, n, 'jetformer.onnx')
            r = os.path.join(RUNS, n, 'jetformer.onnx')
            if os.path.isfile(f) and os.path.isfile(r) and os.stat(f).st_ino != os.stat(r).st_ino \
                    and os.path.getsize(f) == os.path.getsize(r):
                dup += os.lstat(f).st_blocks * 512
    z = REP + '/sim_dart_20261014/eval/bundles/shortcut-rc-finetuned.zip'
    z2 = REP + '/sim_rl_bestlap_20261013/eval/bundles/shortcut-rc-finetuned.zip'
    if os.path.isfile(z) and os.path.isfile(z2) and os.stat(z).st_ino != os.stat(z2).st_ino:
        dup += os.lstat(z).st_blocks * 512
    rows.append(('B1 bundle の ONNX・ZIP の重なり(ハードリンクにまとめると空く。大きさが同じものだけ数えた)', dup))
    for what, b in sorted(rows, key=lambda r: -r[1]):
        print('%6.2f GB  %s' % (b / 1e9, what))
    raw24 = sum(du(os.path.join(RAW, n)) for n in os.listdir(RAW) if n.startswith('20261024-'))
    print('参考: 10/24 の記録 %.3f GB(%d 本)' % (raw24 / 1e9, sum(1 for n in os.listdir(RAW) if n.startswith('20261024-'))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
