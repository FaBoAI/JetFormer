#!/usr/bin/env python3
"""動画の作り直し(10/16、ユーザーの依頼): 始まりは「10月15日のレポート」(2 本目とは言わない)。
今の課題 → 対策 → 結果 をはっきり見せる図 2 枚を足す(CPU だけ)。

  k1_issue  今の課題(r1_problem と同じ中身。題・副題・帯の文だけ「今の課題」に)
  k2_table  課題 → 対策 → 結果 の一覧(数は review/recount.json = 保存した予測から数え直した値)
出力: /home/jetson/jetformer_reports/real_accuracy_20261015/video/fig/k*.png(2160 × 1350)
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('slides_review', HERE / 'make_slides_review.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
plt = M.plt


def k1_issue():
    """r1_problem の文だけを差し替えて、名前を k1_issue にする"""
    orig_slide, orig_banner = M.slide, M.banner
    M.slide = lambda title, sub: orig_slide('今の課題: 実機のデータでの精度が 49 µs から下がらない',
                                            'いろいろな走りのデータで、シミュレーターでは 12 回中 5 回完走。でも実機の保留データの誤差は、どの作り方でも約 49 µs')
    M.banner = lambda fig, text, y=0.035: orig_banner(fig, '課題: 実機のデータでの精度を上げる方法を見つける(判定の決まりは学習の前に固定)', y)
    saved = []
    orig_savefig = plt.Figure.savefig

    def savefig(self, path, *a, **k):
        saved.append(path)
        return orig_savefig(self, M.OUT / 'k1_issue.png', *a, **k)
    plt.Figure.savefig = savefig
    try:
        M.r1_problem()
    finally:
        M.slide, M.banner, plt.Figure.savefig = orig_slide, orig_banner, orig_savefig


def mean(*xs):
    return sum(xs) / len(xs)


def k2_table():
    b = M.RC['per_variant_primary']['best']
    a = mean(b['A_s0'], b['A_s1'])
    ens = mean(b['P2'], b['P3']) - a
    gain = mean(b['GAIN(A_s0)'], b['GAIN(A_s1)']) - a
    same = mean(b['SAME_A'], b['SAME_new']) - a          # 判定の腕 SAME = SAME_A と SAME_new(evaluation.json と同じ)
    te = mean(b['TE(A_s0)'], b['TE(A_s1)']) - a
    rot = mean(b['rot-s0'], b['rot-s1']) - a
    half = mean(b['sizeH-s0'], b['sizeH-s1']) - a
    quarter = mean(b['sizeQ-s0'], b['sizeQ-s1']) - a
    fig = M.slide('課題 → 対策 → 結果', '実機の保留データ(人のきれいな走行 11 シーン・4,670 フレーム)の誤差で判定。今の作り方 A = %.2f µs。マイナス = 良くなった' % a)
    rows = [
        ('① 形の違う 2 つのモデルの平均', '軽量型 + 多段型の予測を平均', '%+.1f µs  良い(際どい)' % ens, M.BLUE),
        ('② 電池の補正', '電池の減りに合わせてスロットルを強める', '%+.1f µs  良い(ほぼ 1 シーン分)' % gain, M.GREEN),
        ('③ 学習の工夫', '同じ形の平均・時間の平均・全フレーム', '%+.1f / %+.1f / %+.1f µs  効かず' % (same, te, rot), M.GRAY),
        ('④ データの量を調べる', '実機データを 1/2・1/4 に減らして学習', '1/2 で %+.1f µs → 増やせば下がる' % half, M.ORANGE),
    ]
    y0, h = 0.70, 0.125
    fig.text(0.04, y0 + 0.075, '対策', fontsize=15, color=M.INK2, fontweight='bold')
    fig.text(0.31, y0 + 0.075, 'やったこと', fontsize=15, color=M.INK2, fontweight='bold')
    fig.text(0.635, y0 + 0.075, '結果(A との差)', fontsize=15, color=M.INK2, fontweight='bold')
    for i, (name, what, res, col) in enumerate(rows):
        y = y0 - i * (h + 0.02)
        M.box(fig, 0.03, y - h / 2, 0.94, h, col, '#ffffff', [(' ', 6, False, M.INK)])
        fig.text(0.05, y, name, fontsize=17, fontweight='bold', color=col, va='center')
        fig.text(0.31, y, what, fontsize=14, color=M.INK, va='center')
        fig.text(0.635, y, res, fontsize=15, fontweight='bold', color=M.INK, va='center')
    fig.text(0.04, 0.155, 'データを 1/4 にすると %+.1f µs。精度の床(人の運転の周ごとの揺れから見た、下げられない誤差)は約 31 µs → まだ余地がある' % quarter, fontsize=15, color=M.INK2)
    M.banner(fig, '結論: 少し下がった。でも実用上の差(10 µs)には届かず、走行のおすすめはモデル B のまま')
    fig.savefig(M.OUT / 'k2_table.png', facecolor=M.SURFACE)
    plt.close(fig)


def main():
    M.jp_font()
    k1_issue()
    k2_table()
    print('ok', M.OUT)


if __name__ == '__main__':
    main()
