"""反証 8-1 回目・打ち消しの票 B・指摘 w1-tools-r8-1-stall-inflates-off(読むだけ。data/raw・labels は読むだけ。コード・文書は変えない)。

本物の 10/10・10/23・10/24 の発進(⑤ collect.py --day の launches。今の版)ごとに、AI 側にしてから R1 の起点まで(届かなかった発進は
AI 側の区間の終わりか 8 秒まで)の、AI 側の処理したフレーム(select・中立止めの行でない)どうしのいちばん長い間を数える
(review_tools_r1 の指摘の「発進 → R1 の間の長いカメラの止まり」が本物で起きた形か。数え方は fix8/decided_safety/scan_gaps10.py と同じ)。
OFF の比べに入る発進(R1 に届いた・発進の前進 OFF・走り直し / 試し / 途中で変えたでない)を分けて出す。
⑤ の出力は ../tmp_refute_tools_B/real に書き、終わって消す。出力 out/b2_launch_gaps_real.txt・.json

  PYTHONDONTWRITEBYTECODE=1 jf-heavy --lock none --mem 1500M -- python -B b2_launch_gaps_real.py
"""
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JF = Path('/home/jetson/jetformer')
for p in (JF, JF / 'tools', JF / 'tests'):
    sys.path.insert(0, str(p))
from test_fix3_r1_2 import load_mod  # noqa: E402

RAW = JF / 'data/raw'
TMP = HERE.parent / 'tmp_refute_tools_B' / 'real'
OUT = HERE / 'out'
COLLECT = JF / 'experiments/20261026_fix/diag_launch/collect.py'
DAYS = ('20261010', '20261023', '20261024')
KV = 0.0005


def ai_frames(log):
    ts, sel, hold = [], [], []
    for s in sorted(d for d in (RAW / log).iterdir() if d.is_dir()):
        try:
            z = np.load(s / 'AI_CMD.npz')
        except Exception:      # noqa: BLE001  無い・壊れた npz は飛ばす(⑤ と同じく)
            continue
        t_ = z['timestamps_us'].astype(np.int64)
        sl = z['select'].astype(bool) & ~(z['dry_run'].astype(bool) if 'dry_run' in z else np.zeros(len(t_), bool))
        ts.append(t_)
        sel.append(sl)
        hold.append(z['hold'].astype(bool))
    ts, sel, hold = np.concatenate(ts), np.concatenate(sel), np.concatenate(hold)
    o = np.argsort(ts, kind='stable')
    return ts[o], sel[o], hold[o]


def vehicle(log):
    vt, ve = [], []
    for s in sorted(d for d in (RAW / log).iterdir() if d.is_dir()):
        try:
            v = np.load(s / 'VEHICLE.npz')
        except Exception:      # noqa: BLE001
            continue
        vt.append(v['timestamps_us'].astype(np.int64))
        ve.append(v['erpm'].astype(float))
    vt, ve = np.concatenate(vt), np.concatenate(ve)
    o = np.argsort(vt, kind='stable')
    return vt[o], ve[o]


def max_gap(ts, sel, hold, t0, a, b):
    """[a, b] 秒(t0 から)の中の、処理したフレームどうしの間のいちばん長いもの(間に人側の行があれば数えない)"""
    tt = (ts - t0) / 1e6
    proc = np.nonzero(sel & ~hold & (tt >= a - 1e-9) & (tt <= b + 1e-9))[0]
    best = (0.0, None, None)
    for i, j in zip(proc, proc[1:]):
        if not sel[i:j + 1].all():
            continue
        g = (ts[j] - ts[i]) / 1e6
        if g > best[0]:
            best = (g, round(float(tt[i]), 3), int(hold[i:j + 1].sum()))
    return best


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    res = {'days': {}, 'launches': []}
    lines = []
    for d in DAYS:
        m = load_mod('c_b2_%s' % d, COLLECT)
        out = TMP / ('c5_%s.json' % d)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m.main(['--day', d, '--out', str(out), '--ignore-gain'])
        j = json.loads(out.read_text(encoding='utf-8'))
        cache = {}
        st = {'launches': 0, 'gt0.15': 0, 'gt0.3': 0, 'gt0.5': 0, 'off_compared': 0, 'off_compared_gt0.3': 0,
              'max_gap_s': 0.0, 'max_gap_off_compared_s': 0.0}
        for r in j['launches']:
            log = r['log']
            if log not in cache:
                try:
                    ts, sel, hold = ai_frames(log)
                    vt, ve = vehicle(log)
                    cache[log] = (ts, sel, hold, vt, ve)
                except Exception as e:      # noqa: BLE001
                    cache[log] = None
                    lines.append('  skip %s: %s' % (log, e))
            if cache[log] is None:
                continue
            ts, sel, hold, vt, ve = cache[log]
            t0 = int(ts[0])
            a = float(r['t_ai'])
            if r.get('to_r1_s') is not None:
                b = a + float(r['to_r1_s'])
            else:
                b = a + min(float(r.get('ai_seg_s') or 8.0), 8.0)
            g, at, nh = max_gap(ts, sel, hold, t0, a, b)
            v_at = None
            if at is not None:
                k = np.searchsorted(vt, t0 + int(at * 1e6), side='right') - 1
                v_at = round(-float(ve[k]) * KV, 2) if k >= 0 else None
            off_cmp = (r.get('launch_state', 'off') == 'off' and r.get('to_r1_s') is not None and not r.get('after_abort')
                       and not r.get('trial') and not r.get('midrun'))
            row = {'log': log, 't_ai': a, 'to_r1_s': r.get('to_r1_s'), 'launch_state': r.get('launch_state', 'off'),
                   'session': r.get('session'), 'off_compared': off_cmp, 'max_gap_s': round(g, 3), 'gap_at_s': at,
                   'hold_rows': nh, 'v_at_gap_mps': v_at}
            res['launches'].append(row)
            st['launches'] += 1
            st['max_gap_s'] = max(st['max_gap_s'], round(g, 3))
            for th in (0.15, 0.3, 0.5):
                if g > th:
                    st['gt%s' % th] += 1
            if off_cmp:
                st['off_compared'] += 1
                st['max_gap_off_compared_s'] = max(st['max_gap_off_compared_s'], round(g, 3))
                if g > 0.3:
                    st['off_compared_gt0.3'] += 1
        res['days'][d] = st
        lines.append('%s: %s' % (d, json.dumps(st, ensure_ascii=False)))
        for row in res['launches']:
            if row['log'][:8] == d and row['max_gap_s'] > 0.15:
                lines.append('   0.15 秒より長い: %s' % json.dumps(row, ensure_ascii=False))
    (OUT / 'b2_launch_gaps_real.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    txt = '\n'.join(lines) + '\n'
    (OUT / 'b2_launch_gaps_real.txt').write_text(txt, encoding='utf-8')
    print(txt)
    shutil.rmtree(TMP)


if __name__ == '__main__':
    main()
