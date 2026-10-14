"""10/14 の動画(作り直し)の図 3 枚: 前回の課題 → 解決(5 つのモード・シミュレーターの画面)→ いろいろな線(2026-10-15)

    python3 -I make_slides_r5.py

ユーザーの依頼: 日付は 10 月 14 日。前回の強化学習の走りは同じ経路に集中したので、シミュレーターのデータ収集のために
いろいろなパターンの強化学習でいろいろな走行を実現した、という前回の課題と解決を最初に説明。シミュレーターの画像も見せる。
画面は r3/screens/ の実際の画面(http://192.168.55.1:8080 の Firefox)、数字は r3/variation.json から読む。
出力: /home/jetson/jetformer_reports/sim_rl_modes_20261013/video/fig/r5_v{0,1,2}_*.png(2160 × 1350)
"""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('slides_r4', HERE / 'make_slides_r4.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
plt = S.plt
from PIL import Image  # noqa: E402

R3, OUT = S.R3, S.OUT
SCR = R3 / 'screens'
TRAJ = R3 / 'fig/promote_record_trajectories.png'
KEYS = ['fastest', 'no_wall', 'recover', 'speed', 'line']
WHAT = {'fastest': 'いちばん速い線', 'no_wall': '壁から 8 cm 以上はなれる', 'recover': 'わざと乱して、戻る',
        'speed': '速さ 50〜100% の 5 段', 'line': '左寄り・右寄り・イン・アウト'}


def view3d(path):
    """シミュレーターの画面から 3D の表示の部分だけを切り出す(画面 1500 × 878 の左上)"""
    im = Image.open(path).convert('RGB')
    w, h = im.size
    return im.crop((int(0.02 * w), int(0.17 * h), int(0.76 * w), int(0.77 * h)))


def show(fig, im, rect, label=None, color=None):
    ax = fig.add_axes(rect)
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(color or S.GRID)
        sp.set_linewidth(3 if color else 2)
    if label:
        ax.set_title(label, fontsize=15, color=color or S.INK2, fontweight='bold', pad=6)
    return ax


def traj_panel(col, row):
    """軌跡の図(3 列 × 2 段)の 1 枚を切り出す"""
    im = Image.open(TRAJ).convert('RGB')
    w, h = im.size
    top = int(0.035 * h)
    ph = (h - top - int(0.04 * h)) // 2
    return im.crop((int(col * w / 3), top + row * ph, int((col + 1) * w / 3), top + (row + 1) * ph))


def v0_problem():
    fig = S.slide('前回の課題: 強化学習の走りが 1 本の線に集まった',
                  '前回のベストラップの強化学習は、停止から 1 周 8.09 秒と速い。でも、どこから走り出しても同じ線に戻って走る')
    show(fig, traj_panel(1, 0), [0.03, 0.2, 0.44, 0.62], '前回の強化学習(上から見た図・15 シーン)', S.BLUE)
    show(fig, view3d(SCR / 'drive_fastest_window.png'), [0.5, 0.45, 0.32, 0.37], 'シミュレーターの画面')
    show(fig, Image.open(SCR / 'drive_fastest_camera.png').convert('RGB'), [0.835, 0.5, 0.14, 0.28], '車のカメラ')
    S.box(fig, 0.5, 0.17, 0.47, 0.24, S.RED, '#fbefed', [
        ('走る線がほぼ 1 本 → データの場面が増えない', 16, True, S.INK),
        ('その走りをまねした JetFormer は、少しずれると', 14, False, S.INK),
        ('戻れずに衝突(12 回とも約 1 秒)', 14, False, S.INK)])
    S.banner(fig, '課題: シミュレーターでデータを集めるには、いろいろな走りが要る')
    fig.savefig(OUT / 'r5_v0_problem.png', facecolor=S.SURFACE)
    plt.close(fig)


def v1_solution():
    fig = S.slide('解決: 走り方の違う強化学習モードを 5 つ作った',
                  'シミュレーターのデータ収集のために、強化学習でいろいろな走りを作った(JetFormer の「シミュレーター」タブで選ぶ)')
    for i, k in enumerate(KEYS):
        x = 0.035 + i * 0.19
        show(fig, Image.open(SCR / ('drive_%s_camera.png' % k)).convert('RGB'), [x, 0.525, 0.165, 0.27], S.NAMES[k], S.COLORS[k])
        fig.text(x + 0.0825, 0.497, WHAT[k], fontsize=13, color=S.INK, ha='center', va='center')
    show(fig, view3d(SCR / 'drive_line_window.png'), [0.035, 0.125, 0.42, 0.265], 'シミュレーター(経路いろいろで走行中)')
    S.box(fig, 0.5, 0.14, 0.465, 0.24, S.PURPLE, '#f3f1fa', [
        ('5 つとも、結果を見る前に決めた試験に合格', 15, True, S.INK),
        ('収集プラン(モード × 分)で、まとめて記録', 14, False, S.INK),
        ('JetFormer にそのまま取り込み(乱した操作は学習から外す)', 14, False, S.INK)])
    S.banner(fig, '走り方を選んで、いろいろな場面のデータを集められるようになった')
    fig.savefig(OUT / 'r5_v1_solution.png', facecolor=S.SURFACE)
    plt.close(fig)


def v2_routes(var):
    t = var['teacher']['all60']
    p = var['pooled_5modes']
    fig = S.slide('結果: いろいろな線を走るようになった', '記録した見本(各モード 15 シーン)を上から見た図。灰色の線と点はコースの柵と柱')
    show(fig, Image.open(TRAJ).convert('RGB'), [0.03, 0.16, 0.94, 0.68])
    S.banner(fig, '線のばらつき: 5 モード合わせて %.1f cm(前の教師 %.2f cm の %.1f 倍)・訪れた状態 %.1f 倍' % (
        100 * p['VM3_lateral_spread_m'], 100 * t['VM3_lateral_spread_m'], p['ratio_vs_teacher']['VM3'], p['ratio_vs_teacher']['VM7']))
    fig.savefig(OUT / 'r5_v2_routes.png', facecolor=S.SURFACE)
    plt.close(fig)


def main():
    S.jp_font()
    OUT.mkdir(parents=True, exist_ok=True)
    var = json.loads((R3 / 'variation.json').read_text(encoding='utf-8'))
    v0_problem()
    v1_solution()
    v2_routes(var)
    print('ok', OUT)


if __name__ == '__main__':
    main()
