"""1 分動画の絵コンテ(video/storyboard.json)を第 4 回の最終の状態で書き直す(第 3〜4 回の審査の段、2026-10-15)
    python3 -I make_storyboard_r4.py
決まり(feedback-video-report): 約 1 分・時刻や予定のセリフなし・最後は「今日のまとめ」と「課題」。図は make_slides_r4.py の r4_v{1,2,3}。
場面の長さ = 読み上げの字数 / 5.5 字/秒 + 0.9 秒(Achernar は約 5.5 字/秒)。背景の log は前の版と同じ実車の走行(15 シーン × 20 秒)から続けて切り出す。
"""
import json
from pathlib import Path

REP = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013')
FIG = REP / 'video' / 'fig'
LOG = '20261010-173445'

scenes = [
    {'id': 's1_title', 'kind': 'title', 'caption': '10月15日のレポート',
     'summary_items': ['強化学習モードは6つ', '5つが合格して選べる', '見本を記録して比べた', '今日のまとめと課題'],
     'lines': [('ラピカです。学び直した強化学習モードが、3つとも合格しました!',
                'ラピカです。学び直した強化学習モードが、3つとも合格しました。', 'happy', 'normal')]},
    {'id': 's2_modes', 'kind': 'summary', 'caption': '選べるモードは5つ', 'image': str(FIG / 'r4_v1_modes.png'),
     'lines': [('最速、壁にぶつからない、離脱復活、速さいろいろ、経路いろいろの5つが選べます。',
                '最速、壁にぶつからない、離脱復活、速さいろいろ、経路いろいろの、5つが選べます。', 'happy', 'point'),
               ('障害物よけは、まだ出していません。', '障害物よけは、まだ出していません。', 'normal', 'normal')]},
    {'id': 's3_results', 'kind': 'summary', 'caption': '学び直した3つの結果', 'image': str(FIG / 'r4_v2_results.png'),
     'lines': [('壁にぶつからないは、3,038回の試験で衝突0。離脱復活は、99%立て直せました。',
                '壁にぶつからないは、3千回の試験で衝突ゼロ。離脱復活は、99パーセント立て直せました。', 'happy', 'point'),
               ('経路いろいろも、柱から4cm以上はなれて走ります。', '経路いろいろも、柱から4センチ以上はなれて走ります。', 'normal', 'point')]},
    {'id': 's4_variation', 'kind': 'summary', 'caption': '線のばらつきは約23倍', 'image': str(FIG / 'r4_v3_variation.png'),
     'lines': [('5つの見本を合わせると、線のばらつきは前の教師の約23倍。いちばん効くのは経路いろいろです。',
                '5つの見本を合わせると、線のばらつきは、前の教師の約23倍。いちばん効くのは経路いろいろです。', 'happy', 'point')]},
    {'id': 's5_check', 'kind': 'summary', 'caption': '決まりを先に固めて確認',
     'summary_items': ['合否の決まりは結果の前に固定', '試験の組は1回だけ使う', '学習と別のコードで確かめ直し', '乱した操作は学習に使わない', 'テスト一式679件が合格'],
     'lines': [('結果を見る前に決まりを固め、学習と別のコードでも確かめ直しました。',
                '結果を見る前に決まりを固め、学習と別のコードでも、確かめ直しました。', 'normal', 'point')]},
    {'id': 's6_summary', 'kind': 'summary', 'caption': '今日のまとめ',
     'summary_items': ['強化学習モード5つが合格', '見本45シーンを追加で記録', '線のばらつきは約23倍', 'すき間はシールドが守る', '数字はシミュレーターの中'],
     'lines': [('いろいろな走りのデータを、安全に集められるようになりました。',
                'いろいろな走りのデータを、安全に集められるようになりました。', 'happy', 'normal')]},
    {'id': 's7_issues', 'kind': 'ending', 'caption': '課題',
     'summary_items': ['壁モードは線が1本だけ', '離脱復活は余裕が小さい', '経路は角でシールド頼み', 'このデータで学習して効くか', '実車ではまだ確かめていない'],
     'lines': [('課題は、余裕を広げることと、このデータで学習して効くかの確認です。',
                '課題は、余裕を広げることと、このデータで学習して、効くかの確認です。', 'worried', 'point'),
               ('おつかれさまでした!', 'おつかれさまでした。', 'happy', 'normal')]},
]

t = 160.6
out = []
for sc in scenes:
    n = sum(len(s) for _, s, _, _ in sc['lines'])
    dur = round(n / 5.5 + 0.9, 1)
    d = {'id': sc['id'], 'kind': sc['kind'], 'log': LOG, 'caption': sc['caption'], 'src_start': round(t, 2), 'src_end': round(t + dur, 2),
         'speed': 1, 'highlights': []}
    if sc.get('summary_items'):
        d['summary_items'] = sc['summary_items']
    if sc.get('image'):
        d['image'] = sc['image']
    d['lines'] = [{'text': a, 'say': b, 'expression': e, 'pose': p} for a, b, e, p in sc['lines']]
    out.append(d)
    t += dur + 0.2
sb = {'title': 'JetFormer 10月15日\n強化学習モード 5つが合格', 'log': LOG,
      '_note': '約 1 分の報告動画の絵コンテ(第 3〜4 回の審査の段で書き直し。前の版は r3/review/before/storyboard.json)。'
               '図は video/fig/r4_v1〜v3(experiments/20261013_sim_rl_modes/r3/review/video/make_slides_r4.py が登録簿の card/eval/status・'
               '昇格の前の登録簿の写し・確認の段の summary・r3/variation.json・r3/review/indep_variation_r4.json と r3/screens/ の実際の画面から作った)。'
               '背景の log は実車の走行で、タイトルと終わりの札の後ろにだけ薄く見える。'
               '確認: python3 ~/jetformer_reports/sim_cotrain_20261013/video/src/check_storyboard.py <この JSON>',
      'scenes': out}
(REP / 'video' / 'storyboard.json').write_text(json.dumps(sb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('合計 %.1f 秒(%.1f〜%.1f)' % (out[-1]['src_end'] - out[0]['src_start'], out[0]['src_start'], out[-1]['src_end']))
