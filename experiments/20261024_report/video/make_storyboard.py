#!/usr/bin/env python3
"""10月24日のレポート(実車ベンチマーク 2 日目と予選への計画・約 3 分)の絵コンテ ../storyboard.json を作る(CPU だけ・ネットワークなし)

    /usr/bin/python3 -I make_storyboard.py          # → bench_20261024_report/video/storyboard.json(と見積もりの表)

- 10/23 の real_bench_20261023/video/src/make_storyboard.py と同じ形(元のファイルは変えていない)。
- セリフ・見出し・図はこのファイルの SCENES と TITLE。直すときはここを直して、作り直す。
- 図は make_slides_1024.py で作った ../fig/*.png(2160 × 1350)。図が無いとここで止める。
  タイトル・今日のまとめ・課題の場面は図なし(描画側 make_digest.py が title と summary_items を描く)。
- 構成(ユーザーの決まり): 「10月24日のレポート」→ 前回(10/23)の課題と答え → 今日の課題 → 対策(1 回目 比べる走り /
  2・3 回目 速さのチューニング)→ 結果(1 回目)→ 結果(満充電で速すぎ・最終コーナー)→ 結果(ベスト 9 番 8.25 秒)
  → 結果(R1 の入口の板。9 番の続き)→ 結果(3 回目 DE)→ 課題・対策・結果の表 → 条件の違い → 3 周のタイムトライアルと目標
  → うまくいきそうな 3 つのパターン → 次の実車までにやること → 今日のまとめ → 課題(最後は「おつかれさまでした」)。
  時刻と予定の日付・曜日のセリフは入れない(「予選突破を目指す」「次の実車までに」はよい)。「2本目」と言わない。
- 背景の走行は 10/24 の 9 番(No.9 = 20261024-162033、記録は最初の AI_CMD から 0〜137.5 秒)。
  カメラのコマに人が写らないところだけを使う: 126.0〜137.5 秒(走り終わって止まったところ。0.5 秒ごとに目で確かめた)。
  走っている区間は、どの周も R1 の手前・SW のまわり・バックストレートの奥に人が写るコマがあるので使わない
  (図の場面は描画側がほぼ不透明に覆うが、念のため同じ区間にした)。kind=clip(走行の映像そのもの)の場面も同じ理由で置かない。
- src_start / src_end は仮の値: src_start はすべて 126.0、src_end は day8 の estimate_narration.py(Open JTalk のモーラ数)の
  見積もりで置く。書き出しの前に、メインのセッションが ai_drive_digest/narr_len.py ../storyboard.json --fit で
  src_end を本当の読み上げに合わせる(声の書き出しはメインのセッションがする)。読み上げが 11.5 秒より長い場面は、
  記録の終わり(137.5 秒)で描画側が止める(make_digest.py の build は src_end を記録の長さで切り、速さを落とす)。
- 決まりの確かめ: check_storyboard_3min.py ../storyboard.json --final
"""
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True                  # 読み込む day8 などのフォルダに __pycache__ を作らない

VIDEO = '/home/jetson/jetformer_reports/bench_20261024_report/video'   # src は experiments へのリンクなので絶対パス
FIG = os.path.join(VIDEO, 'fig')
EST = '/home/jetson/jetformer_reports/day8_explainer_20261017/video/src/estimate_narration.py'
LOG = '20261024-162033'                        # 10/24 の 9 番(B・速さ 0.90・ステア 1.00。ベスト 8.25 秒)
START = 126.0                                   # 人の写らない、走り終わりの区間(126.0〜137.5 秒)
TITLE = 'JetFormer 10月24日のレポート\n実車ベンチマーク 2 日目と、予選への作戦'
RATE, LEAD, GAP, TAIL = 6.4, 0.35, 0.28, 0.5


def L(text, say, expression='normal', pose='point'):
    return {'text': text, 'say': say, 'expression': expression, 'pose': pose}


SCENES = [
    dict(id='s01_title', kind='title', caption='10月24日のレポート',      # 図なし: 描画側が title と summary_items を描く
         summary_items=['前回の課題と今日の課題', '1回目: 比べる走り', '2・3回目: 速さのチューニング', '3周のタイムと予選の目標',
                        '3つのパターンとやること'],
         lines=[L('ラピカです。10月24日のレポートです!', 'ラピカです。10月24日のレポートです。', 'happy', 'normal'),
                L('今日は実車で20本。ベストは1周8.25秒!', 'きょうは実車で20本。ベストは、1周8.25秒です。', 'happy', 'normal')]),
    dict(id='s02_prev', kind='summary', caption='前回の課題と答え', image='p1_prev.png',
         lines=[L('前回の課題は、DEの揺れ、走り切ること、発進の補助、電圧の記録。',
                  '前回の課題は、ディーイーの揺れ、走り切ること、発進の補助、電圧の記録。'),
                L('発進の補助は効きましたが、電圧の記録は、まだ空です。',
                  '発進の補助は効きましたが、電圧の記録は、まだ空です。', 'worried')]),
    dict(id='s03_issue', kind='summary', caption='今日の課題', image='k1_issue.png',
         lines=[L('今日の課題は、Bより良いモデル探しと、Bを速く走らせること。目標は8秒台です。',
                  'きょうの課題は、ビーより良いモデルさがしと、ビーを速く走らせること。目標は、8秒台です。')]),
    dict(id='s04_plan', kind='summary', caption='対策: 比べる走りと速さ', image='k2_plan.png',
         lines=[L('対策は、1回目がBとほかのモデルの比べる走り。',
                  '対策は、1回目が、ビーとほかのモデルの、くらべる走り。'),
                L('2・3回目は、走行タブで速さの段を選び、チューニングしました。',
                  '2回目と3回目は、走行タブで速さの段をえらび、チューニングしました。')]),
    dict(id='s05_res1', kind='summary', caption='結果: 1回目 比べる走り', image='k3_res1.png',
         lines=[L('遅く入る型はR1で衝突ゼロ。でもラップはBとほぼ同じです。',
                  '遅く入る型は、アールワンで衝突ゼロ。でもラップは、ビーとほぼ同じです。'),
                L('2本の平均は途中で止まり、判定はBのままです。',
                  '2本の平均は途中で止まり、判定は、ビーのままです。', 'worried')]),
    dict(id='s06_full', kind='summary', caption='結果: 満充電で速すぎ', image='k4_full.png',
         lines=[L('2回目は満充電。速さ1.0以上では速すぎて、最終コーナーを曲がりきれません。',
                  '2回目は満充電。速さ1.0以上では速すぎて、最終コーナーを曲がりきれません。', 'surprised'),
                L('出っ張りに9回当たり、10番では後輪も壊れました。',
                  '出っぱりに9回あたり、10番では、後輪もこわれました。', 'worried')]),
    dict(id='s07_best', kind='summary', caption='結果: ベストは9番', image='k5_best.png',
         lines=[L('速さを0.90に下げた9番がベスト。11周中9周がきれい!',
                  '速さを0.90に下げた9番がベスト。11周中9周が、きれいです。', 'happy'),
                L('中央値8.53秒で、前回のBより1周約0.9秒速いです。',
                  '中央値は8.53秒で、前回のビーより、1周約0.9秒速いです。', 'happy')]),
    dict(id='s08_r1', kind='summary', caption='結果: R1の入口の板', image='k6_r1.png',
         lines=[L('ただ9番は、R1の入口の板で、ほぼ毎周3から6Gの上下の衝撃。',
                  'ただ9番は、アールワンの入口の板で、ほぼ毎周、3から6ジーの、上下の衝撃。', 'surprised'),
                L('速く入るほど大きく、止める条件の案に入れました。',
                  '速く入るほど大きく、止める条件の案に入れました。')]),
    dict(id='s09_de', kind='summary', caption='結果: 3回目はDE', image='k7_de.png',
         lines=[L('3回目は同じ設定で1周9.3秒台。衝突34件中25件がDEです。',
                  '3回目は同じ設定で、1周9.3秒台。衝突34件中、25件がディーイーです。', 'worried'),
                L('ステア1.1では悪化し、最後の3本はきれいな周ゼロです。',
                  'ステア1.1では悪化し、最後の3本は、きれいな周ゼロです。', 'worried')]),
    dict(id='s10_table', kind='summary', caption='課題・対策・結果', image='k8_table.png',
         lines=[L('8秒台は出せましたが、同じ設定が再現せず、7秒台はまだです。',
                  '8秒台は出せましたが、同じ設定が再現せず、7秒台はまだです。')]),
    dict(id='s11_cond', kind='summary', caption='条件の違い', image='k9_cond.png',
         lines=[L('一番の疑いは後輪の修理。左右の曲がりの比が0.96から0.76に下がりました。',
                  '一番の疑いは、後輪の修理。左右の曲がりの比が、0.96から0.76に下がりました。'),
                L('電池の差は小さく、明るさはほぼ効いていません。',
                  '電池の差は小さく、明るさは、ほぼ効いていません。')]),
    dict(id='s12_tt', kind='summary', caption='3周のタイムと目標', image='k10_tt.png',
         lines=[L('予選は3周のタイムトライアル。40チーム中12チームが決勝へ。',
                  '予選は、3周のタイムトライアル。40チーム中、12チームが決勝へ。'),
                L('トップは約23秒、目安は27秒。目標は25秒を安定して出すことです!',
                  'トップは約23秒、目安は27秒。目標は、25秒を安定して出すことです。', 'happy'),
                L('9番の速い周を毎周そろえれば、届きます。',
                  '9番の速い周を、毎周そろえれば、届きます。')]),
    dict(id='s13_patterns', kind='summary', caption='うまくいきそうな3つ', image='k11_patterns.png',
         lines=[L('パターン1は9番の再現。車の左右と速さをそろえます。',
                  'パターン1は、9番の再現。車の左右と速さを、そろえます。'),
                L('パターン2は速さの天井と自動合わせ。25秒の候補で、まだ仮説です。',
                  'パターン2は、速さの天井と自動合わせ。25秒の候補で、まだ仮説です。'),
                L('パターン3は、速い周で学び直すB-FAST。上積みの候補です。',
                  'パターン3は、速い周で学び直す、ビーファスト。上積みの候補です。')]),
    dict(id='s14_todo', kind='summary', caption='次の実車までにやること', image='k12_todo.png',
         lines=[L('次の実車までに、車の点検、電圧の記録、確認の道具をそろえます。',
                  '次の実車までに、車の点検、電圧の記録、確認の道具をそろえます。'),
                L('了承をもらえたら、天井と自動合わせ、B-FAST、上積みへ進みます!',
                  '了承をもらえたら、天井と自動合わせ、ビーファスト、上積みへと進みます。', 'happy')]),
    dict(id='s15_summary', kind='summary', caption='今日のまとめ',
         summary_items=['ベスト: 9番 1周8.25秒', '3周: 最速25.29秒(9番)', '満充電で速すぎ→最終コーナー',
                        '3回目はDE: 修理後の左右差', 'R1の入口の板で上下の衝撃', 'モデルはBのまま'],
         lines=[L('8秒台は出せました。予選突破へ、25秒の安定が次の目標です!',
                  '8秒台は出せました。予選突破へ、25秒の安定が、次の目標です。', 'happy', 'normal')]),
    dict(id='s16_issue', kind='ending', caption='課題',
         summary_items=['9番の再現(車の左右をそろえる)', '電圧を毎回測って書く', '3周25秒を安定して出す',
                        '速さの天井・自動合わせ(了承)', 'B-FASTの学習(了承)', 'R1の板の衝撃を3軸で数える'],
         lines=[L('課題は、9番の走りを再現し、3周25秒を安定して出すことです。',
                  '課題は、9番の走りを再現し、3周25秒を、安定して出すことです。', 'worried'),
                L('おつかれさまでした!', 'おつかれさまでした。', 'happy', 'normal')]),
]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


try:
    E = load('estimate_narration', EST)
    line_sec, how = E.line_sec, 'モーラ数'
except Exception as ex:                                    # Open JTalk が読めないときは字数で見積もる
    print('estimate_narration を読めない(%s)。字数 %.1f 字/秒で見積もる' % (ex, RATE))
    line_sec, how = (lambda s: len(s) / RATE), '字数'


def main():
    scenes, total = [], 0.0
    for sc in SCENES:
        sc = dict(sc)
        img = sc.pop('image', None)
        secs = [line_sec(ln.get('say') or ln['text']) for ln in sc['lines']]
        est = LEAD + sum(secs) + GAP * (len(secs) - 1) + TAIL
        total += est
        out = {'id': sc['id'], 'kind': sc['kind'], 'log': LOG, 'caption': sc['caption'],
               'src_start': START, 'src_end': round(START + est, 2), 'speed': 1, 'highlights': []}
        if img:
            path = os.path.join(FIG, img)
            if not os.path.exists(path):
                sys.exit('図が無い: %s(先に make_slides_1024.py)' % path)
            out['image'] = path
        if sc.get('summary_items'):
            out['summary_items'] = sc['summary_items']
        out['lines'] = sc['lines']
        scenes.append(out)
        print('%-13s %6.2f → %6.2f  見積もり %4.1f 秒  %s' % (sc['id'], out['src_start'], out['src_end'], est, sc['caption']))
    story = {'title': TITLE, 'log': LOG,
             '_note': '10/24 の実車ベンチマーク 2 日目(AI 走行 No.1〜20)のレポートの動画・約 3 分(読み上げ 170〜190 秒)。'
                      '通常の報告の回なので、時刻と予定の日付・曜日のセリフは入れない(「予選突破」「次の実車までに」はよい)。'
                      '構成 = タイトル(図なし)→ 前回(10/23)の課題と答え → 今日の課題 → 対策(1 回目 比べる走り / 2・3 回目 速さの'
                      'チューニング、走行タブの本物の画面)→ 結果: 1 回目 → 結果: 満充電で速すぎ(最終コーナー)→ 結果: ベストは 9 番'
                      '(8.25 秒)→ 結果: R1 の入口の板 → 結果: 3 回目は DE → 課題・対策・結果の表 → 条件の違い → 3 周のタイムと'
                      '目標(25 秒を安定)→ うまくいきそうな 3 つのパターン(課題と直し方の表)→ 次の実車までにやること(25 秒の安定と'
                      '上積み)→ 今日のまとめ(項目)→ 課題(項目)。図は video/src/make_slides_1024.py(前回の動画の最後の場面・'
                      '走行タブの写真・人の写っていない実車のカメラ 6 コマ・tt_data.json などの数)。数の出典は '
                      'bench_20261024_report/README.md・qualify_plan.md・fixes.md(反証のあとに直した版)。'
                      'ToF の表記は STEVAL-VL53L9。背景の走行は 10/24 の 9 番(20261024-162033)の 126.0〜137.5 秒'
                      '(走り終わりで止まったところ。カメラに人が写らない区間だけ)。'
                      'src_start/src_end は見積もりで置いた仮の値。書き出しの前に narr_len.py --fit で src_end を合わせる。'
                      'チェッカーは video/src/check_storyboard_3min.py',
             'scenes': scenes}
    path = os.path.join(VIDEO, 'storyboard.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(story, f, ensure_ascii=False, indent=1)
    print('wrote %s  %d 場面・見積もり %.1f 秒(%s)' % (path, len(scenes), total, how))


if __name__ == '__main__':
    main()
