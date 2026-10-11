"""fig/sensor_geometry_timeline.png from geom_timeline.json (two panels, one y-scale each)."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(fp)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': '#b5b4ad', 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK, 'axes.titlesize': 11, 'font.size': 9,
                     'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False})
d = json.load(open('geom_timeline.json'))
TRAIN = {'20261010-135449', '20261010-144028', '20261010-150424', '20261010-152405', '20261010-152707'}
NEW = {'20261010-203924', '20261010-204208', '20261010-204605'}
OLDAI = {'20261010-172910', '20261010-173445', '20261010-181504'}
groups = [('学習データ(使ったモデル first_trial)', TRAIN, '#2a78d6', 'o'),
          ('以前のAI走行 006/007/008', OLDAI, '#1baf7a', 's'),
          ('今回のAI走行 009/010/101', NEW, '#eb6834', 'D'),
          ('その他(収集・試走)', None, '#9a9993', '.')]
def hour(log):
    h, m, s = int(log[9:11]), int(log[11:13]), int(log[13:15])
    return h + m / 60 + s / 3600
fig, ax = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
for name, S, col, mk in groups:
    xs, b, bl, bh, t9, t18 = [], [], [], [], [], []
    for log, r in d.items():
        inS = (log in S) if S is not None else (log not in TRAIN | NEW | OLDAI)
        if not inS:
            continue
        x = hour(log)
        ax[0].errorbar(x, r['bumper_rows_median'], yerr=[[r['bumper_rows_median'] - r['bumper_rows_iqr'][0]],
                                                         [r['bumper_rows_iqr'][1] - r['bumper_rows_median']]],
                       fmt=mk, color=col, ms=7 if mk != '.' else 9, lw=1.2, capsize=0)
        if 'tof_row9_pct' in r:
            ax[1].plot(x, r['tof_row9_pct'], mk, color=col, ms=7 if mk != '.' else 9)
            ax[1].plot(x, r['tof_row18_pct'], mk, color=col, ms=5 if mk != '.' else 7, mfc='none')
    ax[0].plot([], [], mk, color=col, label=name, ms=7)
for log in NEW | OLDAI:
    r = d[log]
    ax[0].annotate({'20261010-172910': '006', '20261010-173445': '007', '20261010-181504': '008', '20261010-203924': '009',
                    '20261010-204208': '010', '20261010-204605': '101'}[log], (hour(log), r['bumper_rows_median']),
                   xytext={'20261010-172910': (-9, 9), '20261010-173445': (9, 9), '20261010-181504': (0, 9),
                           '20261010-203924': (-14, -4), '20261010-204208': (-4, 12), '20261010-204605': (14, -4)}[log],
                   textcoords='offset points', ha='center', fontsize=8, color=INK2)
ax[0].set_ylabel('青いバンパーの高さ [行 / 224]')
ax[0].set_title('カメラ: 画像の下に写る自車バンパーの高さ(大きいほどカメラが下向き、または前が下がっている)', loc='left')
ax[0].legend(loc='upper left', ncol=2, fontsize=8)
ax[0].set_ylim(-1, 20)
ax[1].axhline(0, color='#b5b4ad', lw=1)
ax[1].axhline(-20, color='#e34948', lw=1, ls='--')
ax[1].text(13.7, -19.3, '床を「物」とみなす境目(床の基準より20%近い)', color=INK2, fontsize=8)
ax[1].set_ylabel('床までの距離 − 基準 [%]')
ax[1].set_title('ToF: まっすぐ走行中の床の見え方(塗り = 遠い床 行9 約2 m / 白抜き = 中くらい 行18 約0.85 m)', loc='left')
ax[1].set_xlabel('2026-10-10 の時刻(ログの開始)')
ax[1].set_xticks(range(14, 22))
ax[1].set_xticklabels(['%d:00' % h for h in range(14, 22)])
for a in ax:
    a.grid(axis='y', color=GRID, lw=0.8)
    a.set_axisbelow(True)
fig.tight_layout()
fig.savefig('../fig/sensor_geometry_timeline.png', dpi=110)
print('ok')
