"""10/15 の約 1 分の報告動画の絵コンテ(storyboard.json)を書く(最終の確かめの段、2026-10-15)

    python3 -I make_storyboard.py

形と決まりは sim_rl_modes_20261013/video/storyboard.json と同じ。確かめ: python3 -I ~/jetformer_reports/sim_cotrain_20261013/video/src/check_storyboard.py <出力>
src_start/src_end は読み上げの長さの見込み(1 秒 5.3 字 + 間)。描くときに読み上げに合わせて詰めてよい。図は make_slides.py が作る video/fig/d1〜d5。
"""
import json
from pathlib import Path

REP = Path('/home/jetson/jetformer_reports/sim_diverse_train_20261015')
FIG = REP / 'video' / 'fig'
OUT = REP / 'video' / 'storyboard.json'
LOG = '20261010-173445'


def L(text, say, expression, pose):
    return {'text': text, 'say': say, 'expression': expression, 'pose': pose}


SCENES = [
    ('s1_title', 'title', '10月15日のレポート', None,
     ['前回の課題', 'いろいろな走りで学習', 'シミュレーターの結果', '実機の保留データの結果', '今日のまとめと課題'],
     [L('ラピカです。今日は、いろいろな走りのデータで学習しました!', 'ラピカです。今日は、いろいろな走りのデータで、学習しました。', 'happy', 'normal')]),
    ('s2_problem', 'summary', '前回の課題', 'd1_problem.png', None,
     [L('前回は、いろいろな走りのデータを作れましたが、まねしたモデルは完走できませんでした。',
        '前回は、いろいろな走りを作れましたが、まねしたモデルは、完走できませんでした。', 'worried', 'point'),
      L('実機のデータに混ぜても、精度は変わりませんでした。', '実機のデータに混ぜても、精度は変わりませんでした。', 'worried', 'point')]),
    ('s3_plan', 'summary', '今日: このデータで学習', 'd2_plan.png', None,
     [L('今日は、このデータで学習して、効くか確かめました。', '今日は、このデータで学習して、効くか確かめました。', 'normal', 'point')]),
    ('s4_sim', 'summary', 'シミュレーター: 完走5回', 'd3_sim_result.png', None,
     [L('シミュレーターでは、12回中5回、1周を完走しました。前回は0回です。', 'シミュレーターでは、12回中5回、1周を完走しました。前回は、0回です。', 'happy', 'point')]),
    ('s5_run', 'summary', '最初のカーブを越えた', 'd4_sim_run.png', None,
     [L('最初のカーブは越えましたが、S字の出口の柱で、まだぶつかります。', '最初のカーブは越えましたが、エス字の出口の柱で、まだぶつかります。', 'surprised', 'point')]),
    ('s6_real', 'summary', '実機の精度は上がらず', 'd5_real.png', None,
     [L('実機の保留データの精度は上がらず、差は1.5%で、ほぼ同じでした。', '実機の保留データの精度は上がらず、差は1.5パーセントで、ほぼ同じでした。', 'worried', 'point')]),
    ('s7_summary', 'summary', '今日のまとめ', None,
     ['シミュレーターの完走 0→5/12', '最初のカーブのゲート 0→8/9', '5モードだけでもゲートを通過', 'DAggerは完走3/12で伸びず',
      '実機の保留データは上がらず', '数は生の記録で数え直し済み'],
     [L('走りが上がったのは、シミュレーターの中だけでした。', '走りが上がったのは、シミュレーターの中だけでした。', 'normal', 'normal')]),
    ('s8_issues', 'ending', '課題', None,
     ['S字の出口の柱で衝突', '種を増やして確かめる', '量・ばらつき・先生を分ける', '実機の精度に効く使い方', '実車ではまだ確かめていない'],
     [L('課題は、S字の出口の衝突と、実機の精度に効く使い方を探すことです。', '課題は、エス字の出口の衝突と、実機の精度に効く使い方を探すことです。', 'worried', 'point'),
      L('おつかれさまでした!', 'おつかれさまでした。', 'happy', 'normal')]),
]


def main():
    t = 160.6
    scenes = []
    for sid, kind, cap, img, items, lines in SCENES:
        n = sum(len(x['say']) for x in lines)
        dur = round(n / 5.3 + 0.9, 2)
        sc = {'id': sid, 'kind': kind, 'log': LOG, 'caption': cap, 'src_start': round(t, 2), 'src_end': round(t + dur, 2), 'speed': 1, 'highlights': []}
        if img:
            sc['image'] = str(FIG / img)
        if items:
            sc['summary_items'] = items
        sc['lines'] = lines
        scenes.append(sc)
        t += dur
    sb = {
        'title': 'JetFormer 10月15日のレポート\nいろいろな走りのデータで学習',
        'log': LOG,
        '_note': ('約 1 分の報告動画の絵コンテ(sim_diverse_train_20261015 の最終の確かめの段で作成。動画はメインのセッションが Gemini TTS(Achernar)で描く)。'
                  '最初に前回の課題(いろいろな走りのデータは作れた・まねしたモデルは完走できない・前の混ぜ方は実機の精度が変わらない)と今日の問い'
                  '(このデータで学習して確かめる)、次にシミュレーターの実際の画面と結果、実機の保留データの結果、今日のまとめ、課題の順。'
                  '図 video/fig/d1〜d5 は experiments/20261015_sim_diverse_train/final_review/make_slides.py が走行の生の JSON・review/final の数え直し・'
                  'eval_real.json・screens/ の実際の画面から作った。src_start/src_end は読み上げの長さの見込み(1 秒 5.3 字)で、描くときに詰めてよい。'
                  '確認: python3 ~/jetformer_reports/sim_cotrain_20261013/video/src/check_storyboard.py <この JSON>'),
        'scenes': scenes,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(sb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    tmp.replace(OUT)
    print('ok', OUT, '見込み %.1f 秒' % (t - 160.6))


if __name__ == '__main__':
    main()
