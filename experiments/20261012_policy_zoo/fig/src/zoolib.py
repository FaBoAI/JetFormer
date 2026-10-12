"""policy zoo の報告の図の共通部品(図の大きさ・色は common.py = 2026-10-11 の日報の図と同じ)。

- 数字は手で写さず、評価の結果(eval/zoo_eval.json)と学習の記録(train/summary.json)から読む
- 描く部品(箱・文字・矢印)と、文字どうしの重なり・図の外へのはみ出しを調べる check_overlaps()
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa: F401,F403  (W, H, DPI, 色, tint, new_fig)
from common import INK, INK2, MUTED, GRID, BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED, tint, new_fig, DPI
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.abspath(os.path.join(HERE, '..', '..'))
FIG = os.path.join(REPORT, 'fig')

POLICIES = ('jetformer_v0', 'act', 'tcp', 'msfusion', 'dinov2')
# 方策の色(eval_*.png・train_*.png と同じ割り当て)
PCOL = {'jetformer_v0': BLUE, 'act': ORANGE, 'tcp': AQUA, 'msfusion': YELLOW, 'dinov2': MAGENTA, 'B': '#52514e'}
PNAME = {'jetformer_v0': 'TransFuser軽量型', 'act': 'ACT型', 'tcp': 'TCP型', 'msfusion': 'TransFuser多段型',
         'dinov2': 'DINOv2型', 'B': 'モデル B'}


def load():
    """→ dict: 図で使う数字(方策ごと)。出どころ: eval/zoo_eval.json・train/summary.json"""
    ev = json.load(open(os.path.join(REPORT, 'eval', 'zoo_eval.json')))
    tr = json.load(open(os.path.join(REPORT, 'train', 'summary.json')))
    prim = ev['primary']
    out = {'delta': prim['delta_us']['best'], 'L_best': prim['L_us']['best'], 'L_last': prim['L_us']['last'],
           'policy': {}, 'ev': ev}
    runs = {r['run']: r for r in tr['runs']}
    for p in POLICIES:
        pp = prim['per_policy'][p]
        lat = ev['latency']['per_run']
        s0, s1 = p + '_s0', p + '_s1'
        tmin = []
        for s in (s0, s1):
            r = runs.get(ev['runs'][s])
            if r:
                tmin.append((r['train_wall_s'] + r.get('engine_wall_s', 0)) / 60.0)
        out['policy'][p] = {
            'L_mean': pp['best']['mean'], 'diff': pp['best']['diff'], 'diff_pct': pp['diff_pct'],
            'L_seeds': [prim['L_us']['best'][s0], prim['L_us']['best'][s1]],
            'L_last_seeds': [prim['L_us']['last'][s0], prim['L_us']['last'][s1]],
            'seed_range': prim['seed_range_us'][p]['best'],
            'ms': ev['latency']['per_policy'][p]['run_numpy_median_ms_mean'],
            'ms_seeds': [lat[s0]['run_numpy_median_ms'], lat[s1]['run_numpy_median_ms']],
            'params': lat[s0]['params_total_m'], 'trainable': lat[s0]['params_trainable_m'],
            'train_min': sum(tmin) / max(len(tmin), 1),
            'peak_mb': runs[ev['runs'][s0]]['peak_mem_mb'],
        }
    out['B'] = {'L': prim['L_us']['best']['B'], 'ms': ev['latency']['per_run']['B']['run_numpy_median_ms']}
    return out


class Canvas(object):
    """0〜135 × 0〜84.375 の座標(1 単位 = 0.1 インチ = 7.2 pt)で描く"""

    def __init__(self):
        self.fig = new_fig()
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 135)
        self.ax.set_ylim(0, 84.375)
        self.ax.axis('off')
        self.boxes = []

    def box(self, x, y, w, h, fc='white', ec=MUTED, lw=2.0, r=0.8, z=2, ls='-'):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=%s' % r,
                                         fc=fc, ec=ec, lw=lw, zorder=z, ls=ls))
        self.boxes.append((x, y, w, h))

    def txt(self, x, y, s, size=16, weight='normal', color=INK, ha='center', va='center', z=5, **kw):
        assert size >= 16, '文字は 16pt 以上(%s: %s)' % (size, s)
        return self.ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z, **kw)

    def lines(self, x, yc, rows, ha='center', lh=2.9):
        """rows = [(文字, 大きさ, 太さ, 色)] を yc を中心に縦に並べる"""
        n = len(rows)
        y0 = yc + (n - 1) * lh / 2.0
        for i, (s, size, weight, color) in enumerate(rows):
            if s:
                self.txt(x, y0 - i * lh, s, size=size, weight=weight, color=color, ha=ha)

    def arrow(self, x0, y0, x1, y1, color=MUTED, lw=2.2, ms=16, style='-|>', z=3):
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms,
                                          color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0))

    def title(self, s, sub=None):
        self.txt(1.2, 81.2, s, size=28, weight='bold', ha='left')
        if sub:
            self.txt(1.2, 77.0, sub, size=17, color=INK2, ha='left')

    def tile(self, x, y, w, h, big, l1, l2=None, color=INK, big_size=24):
        """数字のタイル(日報の図と同じ形)"""
        self.box(x, y, w, h, 'white', GRID, lw=1.6)
        self.txt(x + 1.3, y + h - 2.9, big, size=big_size, weight='bold', ha='left', color=color)
        if l2:
            self.txt(x + 1.3, y + 4.3, l1, size=16, ha='left')
            self.txt(x + 1.3, y + 1.65, l2, size=16, color=INK2, ha='left')
        else:
            self.txt(x + 1.3, y + 2.6, l1, size=16, ha='left')

    def save(self, name):
        path = os.path.join(FIG, name)
        self.fig.savefig(path, dpi=DPI)
        problems = check_overlaps(self.fig) + self.check_in_boxes()
        print('saved', path, '重なり・はみ出し:', len(problems))
        for p in problems:
            print('   ', p)
        return path, problems

    def check_in_boxes(self, margin_px=3.0):
        """中心が箱の中にある文字が、その箱(中心を含む一番小さい箱)からはみ出していないか"""
        r = self.fig.canvas.get_renderer()
        tr = self.ax.transData
        bxs = []
        for (x, y, w, h) in self.boxes:
            (px0, py0), (px1, py1) = tr.transform([(x, y), (x + w, y + h)])
            bxs.append((px0, py0, px1, py1))
        out = []
        for t in self.ax.texts:
            s = t.get_text()
            if not s.strip():
                continue
            bb = t.get_window_extent(renderer=r)
            cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
            cont = [b for b in bxs if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
            if not cont:
                continue
            b = min(cont, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            if not _inside(bb, b[0] + margin_px, b[1] + margin_px, b[2] - margin_px, b[3] - margin_px):
                out.append('箱からはみ出し: %r' % s[:30])
        return out


def _inside(bb, x0, y0, x1, y1, tol=0.5):
    return bb.x0 >= x0 - tol and bb.x1 <= x1 + tol and bb.y0 >= y0 - tol and bb.y1 <= y1 + tol


def check_overlaps(fig, pad_px=1.0):
    """図の中の文字(空でないもの)の外枠どうしの重なりと、図の外へのはみ出しの一覧(画素)"""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W_px, H_px = fig.get_size_inches() * fig.dpi
    items = []
    for t in fig.findobj(lambda a: hasattr(a, 'get_text') and hasattr(a, 'get_window_extent')):
        try:
            s = t.get_text()
        except Exception:
            continue
        if not s or not s.strip() or not t.get_visible():
            continue
        bb = t.get_window_extent(renderer=r)
        if bb.width <= 0 or bb.height <= 0:
            continue
        items.append((s, bb))
    out = []
    for s, bb in items:
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W_px + 0.5 or bb.y1 > H_px + 0.5:
            out.append('はみ出し: %r' % s[:30])
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            if (min(a.x1, b.x1) - max(a.x0, b.x0) > pad_px) and (min(a.y1, b.y1) - max(a.y0, b.y0) > pad_px):
                out.append('重なり: %r × %r' % (items[i][0][:24], items[j][0][:24]))
    return out
