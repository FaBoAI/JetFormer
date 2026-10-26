#!/usr/bin/env python3
"""学習の途中で「直線で緩めていないか」「低速・止まった場面でブレーキ側を出していないか」を見る小さな物差し(2026-10-26 原因 A)。
見るだけ。最良のエポックの選び方・損失・データには使わない(protocol の選び方は H1 の検証の L1 のまま)。

フレーム(out/probe_set.npz を make で作る。C3 の集合 desk/sets_v2.npz と 10/10 の AI 走行の低速から、決まった間引きで選ぶ):
  st10  直線(C3 の straight)の 10/10 のフレーム  ・ st23  同じく 10/23
  stand 止まった場面(C3 の stand。前が 1.0 m 以上空いている AI 側・|eRPM| < 400。ほとんど 10/23)
  low10 10/10 の AI 走行の v < 1.0 m/s(門「低速のブレーキ」の 10/10 の部分と同じ母集団)
B の値は G1 の予測(PyTorch FP32・ジャンル 5)を写して入れておく。

欄(どれも生の予測。ゲインの前):
  st10_thr_vs_b_us・st23_thr_vs_b_us  行 2 のスロットルの(このモデル − B)の平均。+ = B より緩める
  stand_brake_side_frac  止まった場面で 行 0 のスロットル > 1528(中立 1498 + 30)の割合(B 0.013)
  stand_go_frac          止まった場面で 行 0 のスロットル < 1466(中立 1496 − 30)の割合(ゲインの前で数える)
  low10_brake_frac       10/10 の低速で 行 0 のスロットル > 1598 の割合(門と同じ式・TensorRT でなく FP32)

使い方
  作る(CPU・1 回だけ):  python -I probe_gates.py make
  学習済みのモデルで(GPU): jf-heavy --lock big ... python -u probe_gates.py run --runs <学習の名前> ...
  学習の中から:          import probe_gates; P = probe_gates.Probe(); r = P(model, device, num_modes)
                         (エポックの終わりに 1 回。約 2,000 フレームで 10 秒ほど。r を history[-1]['probe'] に入れる)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
PROBE = OUT / 'probe_set.npz'
REPO = Path('/home/jetson/jetformer')
DS = REPO / 'data' / 'datasets'
RUNS = REPO / 'data' / 'runs'
SETS = Path('/home/jetson/jetformer_reports/train_v2_20261025/desk/sets_v2.npz')
B_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'
GENRE = 5
NEUTRAL = 1498.0           # 門の決まり(desk_lib.NEUTRAL_US)
STEPS = {'st10': 2, 'st23': 2, 'stand': 3, 'low10': 4}     # 間引き(決まった並びの何個に 1 個)


def cmd_make():
    g = np.load(OUT / ('g1_%s.npz' % B_RUN))
    part, pred = g['part'], g['pred'].astype(np.float64)
    S = np.load(SETS)
    st_rows, sd_rows = g['st_rows'], g['sd_rows']
    import m2_brake_frames as M2
    rows = M2.ai_frames()
    a_sel = g['a1010_sel']
    gi = {1: np.nonzero(part == 1)[0], 2: np.nonzero(part == 2)[0], 3: np.nonzero(part == 3)[0]}
    sess = S['sess'][st_rows]
    picks = {'st10': ('dataset_v2_20261025', S['scene'][st_rows][sess == 0], S['frame'][st_rows][sess == 0], gi[1][sess == 0]),
             'st23': ('dataset_v2_20261025', S['scene'][st_rows][sess == 1], S['frame'][st_rows][sess == 1], gi[1][sess == 1]),
             'stand': ('dataset_v2_20261025', S['scene'][sd_rows], S['frame'][sd_rows], gi[2]),
             'low10': ('ai_mix_20261011', np.array([rows[i][0] for i in a_sel]), np.array([rows[i][1] for i in a_sel]), gi[3])}
    out = {}
    for k, (ds, sc, fr, gidx) in picks.items():
        sl = slice(0, None, STEPS[k])
        out[k + '_ds'] = np.array(ds)
        out[k + '_scene'] = np.asarray(sc)[sl].astype(str)
        out[k + '_frame'] = np.asarray(fr)[sl].astype(np.int64)
        out[k + '_B'] = pred[gidx][sl][:, :3, 1].astype(np.float32)      # B の 行 0〜2 のスロットル
    np.savez_compressed(PROBE, **out)
    print({k: len(out[k + '_frame']) for k in picks})


class Probe(object):
    def __init__(self, path=PROBE, batch=64, workers=2):
        z = np.load(path)
        self.parts = {}
        for k in ('st10', 'st23', 'stand', 'low10'):
            self.parts[k] = (str(z[k + '_ds']), z[k + '_scene'], z[k + '_frame'], z[k + '_B'].astype(np.float64))
        self.batch, self.workers = batch, workers
        self.items = {k: self._items(*v[:3]) for k, v in self.parts.items()}

    @staticmethod
    def _items(ds, scenes, frames):
        items, cur, a = [], None, None
        for s, i in zip(scenes, frames):
            s = str(s)
            if s != cur:
                with np.load(DS / ds / 'aligned' / (s + '.npz'), allow_pickle=False) as z:
                    a = {k: z[k] for k in ('cam_files', 'tof', 'imu', 'vehicle')}
                cur = s
            i = int(i)
            items.append((str(DS / ds / str(a['cam_files'][i])), a['tof'][i].copy(), a['imu'][i].copy(), a['vehicle'][i].copy()))
        return items

    def _predict(self, model, device, num_modes, items, genre):
        import torch
        from torch.utils.data import DataLoader
        from jetformer.policies import runtime
        sys.path.insert(0, str(HERE))
        from g1_embed import Frames
        dl = DataLoader(Frames(items), batch_size=self.batch, shuffle=False, num_workers=self.workers)
        outs = []
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for image, tof, imu, state in dl:
                oh = torch.nn.functional.one_hot(torch.full((len(image),), genre, dtype=torch.int64), num_modes).float()
                x = dict(image=image.to(device), tof=tof.to(device), imu=imu.to(device), state=state.to(device), mode=oh.to(device))
                outs.append(runtime.forward(model, x)['actions'][:, :3, 1].float().cpu().numpy())
        if was_training:
            model.train()
        return np.concatenate(outs).astype(np.float64) * 500.0 + 1500.0

    def __call__(self, model, device, num_modes, genre=GENRE):
        r = {}
        P = {k: self._predict(model, device, num_modes, self.items[k], genre) for k in self.parts}
        for k in ('st10', 'st23'):
            r['%s_thr_vs_b_us' % k] = round(float(np.mean(P[k][:, 2] - self.parts[k][3][:, 2])), 2)
        r['stand_brake_side_frac'] = round(float(np.mean(P['stand'][:, 0] > NEUTRAL + 30)), 4)
        r['stand_go_frac'] = round(float(np.mean(P['stand'][:, 0] < 1496.0 - 30)), 4)
        r['low10_brake_frac'] = round(float(np.mean(P['low10'][:, 0] > NEUTRAL + 100)), 4)
        r['n'] = {k: int(len(v)) for k, v in P.items()}
        return r


def cmd_run(args):
    import torch
    sys.path.insert(0, str(REPO))
    from jetformer.policies import runtime
    P = Probe()
    res = {}
    for name in args.runs:
        rd = Path(name) if '/' in name else RUNS / name          # 学習の名前か、学習のディレクトリ
        model, cfg, spec, pcfg = runtime.load_run(rd, args.ckpt, device='cuda')
        res[name] = P(model, 'cuda', cfg['num_modes'])
        print(name, json.dumps(res[name], ensure_ascii=False), flush=True)
        del model
        torch.cuda.empty_cache()
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    sp.add_parser('make')
    r = sp.add_parser('run')
    r.add_argument('--runs', nargs='+', required=True)
    r.add_argument('--ckpt', default='best')
    r.add_argument('--out', default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    if a.cmd == 'make':
        cmd_make()
    else:
        cmd_run(a)


if __name__ == '__main__':
    sys.exit(main())
