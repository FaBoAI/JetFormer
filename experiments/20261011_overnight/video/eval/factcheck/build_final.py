#!/usr/bin/env python3
"""ファクトチェック後の確定版ストーリーボード(eval_storyboard.json)を作る。

元: video/eval_storyboard_draft.json(build_eval_storyboard.py の出力)
直した点は、このファイルの FIX の各行と、出力の "log" に書いた。
根拠の計算: factcheck/v1.py〜v6.py(生ログから再計算)、カメラの確認: factcheck/*.jpg
"""
import copy
import json
import sys

HERE = '/home/jetson/jetformer_reports/overnight_20261011/video/'
L101 = '20261010-204605'
draft = json.load(open(HERE + 'eval_storyboard_draft.json', encoding='utf-8'))
story = {'title': draft['title'], 'log': draft['log'], 'scenes': copy.deepcopy(draft['scenes'])}  # log = 既定の走行ログ(make_digest.py が使う)
S = {s['id']: s for s in story['scenes']}


def ln(text, say, expression='normal', pose='normal'):
    return dict(text=text, say=say, expression=expression, pose=pose)


# e04: 基板の再起動の 0.2 秒(72.81〜72.98)は「出力が中立」で、車は止まっていない(前後とも約 -3650 eRPM = 1.8 m/s)
S['e04_lap']['lines'][0] = ln(
    '010の3周目は、AIだけで走った1周。途中、基板の再起動で0.2秒だけ出力が中立になります。',
    'ゼロイチゼロの3周目は、エーアイだけで走った1周。とちゅう、基板の再起動で、0.2秒だけ出力が中立になります。',
    'happy', 'point')

# e05: 文の形だけ直す(数値は確認済み: 衝突 16.89、停止 17.21〜、交代 17.69、停止 0.47 秒、必要 0.5 秒)
S['e05_recov_near']['lines'][0] = ln(
    'GM出口で当たって停止。止まって0.47秒、検出まであと0.03秒のところで人が交代しました。',
    'ジーエム出口で当たって停止。止まって0.47秒、検出まであと0.03秒のところで、人が交代しました。',
    'surprised', 'point')

# e06: 「12回」と「中央値0.49秒(衝突で止まった8回)」の母数を分ける
S['e06_recov_why']['lines'][0] = ln(
    '止まった12回すべてで、検出より先に人が交代。衝突からの交代は中央値0.49秒です。',
    '止まった12回すべてで、検出より先に人が交代。衝突からの交代は、ちゅうおうち0.49秒です。',
    'normal', 'point')
S['e06_recov_why']['summary_items'][1] = '衝突8回の交代は0.2〜1.0秒後(中央値0.49)'

# e07: 誤警報 42.55 / 43.28 はバックストレート〜DE の角(43.3 は角の手前)。「直線」より場所で言う
S['e07_tof_false']['lines'][0] = ln(
    '距離センサーのガードは記録だけ。バックストレートで2回警告が出ても、当たっていません。誤警報です。',
    '距離センサーのガードは、記録だけ。バックストレートで2回、警告が出ても、当たっていません。ごけいほうです。',
    'normal', 'point')

# e08: モデル自身は衝突の直前にアクセルを戻している(45.6 で 1444µs)。言えるのは「ガードのブレーキがかからなかった」
S['e08_tof_hit']['lines'][0] = ln(
    'この衝突は0.39秒前に警告。でも記録だけなので、秒速約3.5mのまま当たりました。',
    'この衝突は、0.39秒前に警告。でも記録だけなので、秒速やく3.5メートルのまま当たりました。',
    'surprised', 'point')

# e09: 誤警報は 22回 / 1.26分 = 17.4回/分(再計算)。ブレーキ判定 0.3 秒以上前は 3 回 + 境目 1 回(28.24 は 0.295〜0.30 秒)。
#      止まるまでの時間は ESC のブレーキ減速度 2.8 m/s²(未確認の推定値)からの計算
S['e09_tof_card']['summary_items'] = [
    '衝突21回中20回を0.3秒以上前に警告',
    'ブレーキ判定が0.3秒以上前は3〜4回だけ',
    '誤警報 1分に約17回(当たらない警告)',
    '秒速3.5mから止まるには約1.25秒(推定)',
    '効かせても当たる速さを下げるだけ(推定)',
    '1周あたり約1秒遅くなる(推定)']
S['e09_tof_card']['lines'] = [
    ln('21回中20回は0.3秒以上前に警告。でも止まるには約1.25秒かかる計算です。',
       '21回中20回は、0.3秒以上前に警告。でも止まるには、約1.25秒かかる計算です。', 'happy', 'point'),
    ln('効かせても当たる速さが下がるだけの見込みで、誤警報も1分に約17回です。',
       '効かせても、当たる速さが下がるだけの見込みで、ごけいほうも1分に約17回です。', 'worried')]

# e10: DE は「バックストレート終端の右外壁の角(支柱)」(collision_map.md・course_map.json)。「仕切り」ではない
S['e10_de']['lines'][0] = ln(
    '衝突がいちばん多いのはDE。バックストレートの終わり、右の板の角です。',
    '衝突がいちばん多いのはディーイー。バックストレートの終わり、右の板のかどです。',
    'surprised', 'point')

# e11: R1 は AI の通過が 4 回しかない(010: 34.6/62.3/71.6、101: 31.7。factcheck/v7.py)。「1回だけ」は改善とは言えない
S['e11_hotspots']['summary_items'][4] = 'R1 AI通過4回中1回(007は20回中8回)'
S['e11_hotspots']['lines'][0] = ln(
    '止まる衝突はLH、駐車枠、GMに集中。R1はAIの通過が4回だけで、当たったのは1回です。',
    '止まる衝突は、エルエイチ、駐車枠、ジーエムに集中。アールワンは、エーアイの通過が4回だけで、当たったのは1回です。',
    'normal', 'point')

# e12: 19.23 の 7.8G はカーテンだけでは出ない値。言えるのは「LHで当たって止まった」(19.47 で eRPM -156)
S['e12_lookahead']['lines'][0] = ln(
    '009の前半は先読み10。ハンドルが左右いっぱいに振れて、LHで当たって止まりました。',
    'ゼロゼロキューの前半は、先読み10。ハンドルが左右いっぱいに振れて、エルエイチで当たって止まりました。',
    'surprised', 'point')

# e13: 比較は 009 の中(同じゲイン)で、指標は距離あたり。ゲインごとの AI 時間は 20〜30 秒
S['e13_settings']['lines'][0] = ln(
    '009の中で比べると、先読み10は距離あたりの衝突が約3倍、切り返しも1.6倍でした。',
    'ゼロゼロキューの中で比べると、先読み10は、距離あたりの衝突が約3倍、切り返しも1.6倍でした。',
    'normal', 'point')
S['e13_settings']['summary_items'][4] = 'ゲインの差は判定できない(各20〜30秒)'

# e14: 傾きは roadmap.md の幾何計算どおり ToF で約0.6〜1度(0.3〜0.6度は根拠なし)。図を出す場面なので項目は予備
S['e14_sensor']['summary_items'][2] = 'ToFは下向き約0.6〜1度の傾きに相当(推定)'

# e15: roadmap.md(S0〜S2・S10・M3・10/14の計画)に合わせる。「4分以上」。根拠のない「1m/sで当てる」は外す
S['e15_next']['summary_items'] = [
    '① センサーの向きを点検(床のずれ±5%以内)',
    '② 基板の電源(配線・電源の分離・コンデンサ)',
    '③ 1走行=1設定・4分以上・先読み3で固定',
    '④ 止まっても2秒待ち、リカバリーv2を試す',
    '⑤ 床の基準を作り直し→モード2はTTC0.4から',
    '⑥ 走り終えたら、もう一度向きを点検']
S['e15_next']['lines'][1] = ln(
    '先読み3に固定し、1つの設定で4分以上。基板の電源の確認もお願いします。',
    '先読み3に固定し、1つの設定で4分以上。基板の電源の確認も、お願いします。', 'normal')


# 場面の長さ: ナレーションより短いクリップは最後で止まり(一時停止)、カードは「スロー」表示になるので、範囲を少し広げる。
# 広げた部分に新しい出来事(交代・衝突)が入らないことを確認済み(e08 は 48.78 の AI 交代の手前で止める)
for sid, a, b in (('e04_lap', 66.8, 82.6), ('e05_recov_near', 16.6, 24.3), ('e07_tof_false', 39.8, 44.4),
                  ('e08_tof_hit', 44.4, 48.7), ('e09_tof_card', 40.0, 56.5), ('e10_de', 26.9, 33.4),
                  ('e13_settings', 6.5, 21.5), ('e11_hotspots', 27.0, 37.5)):
    S[sid]['src_start'], S[sid]['src_end'] = a, b
assert story['log'] == '20261010-204208'

# 上限の確認
bad = []
if len(story['title']) > 16:
    bad.append(('title', len(story['title'])))
for s in story['scenes']:
    if len(s['caption']) > 18:
        bad.append((s['id'], 'caption', len(s['caption']), s['caption']))
    for it in s.get('summary_items', []):
        if len(it) > 30:
            bad.append((s['id'], 'item', len(it), it))
    if len(s.get('summary_items', [])) > 6:
        bad.append((s['id'], 'n_items'))
    for x in s['lines']:
        if len(x['text']) > 60:
            bad.append((s['id'], 'line', len(x['text']), x['text']))
    if not (0.5 <= s['speed'] <= 3):
        bad.append((s['id'], 'speed'))
    if not s.get('log'):
        bad.append((s['id'], 'no log'))
    assert s['log'] != L101, 'ai_run_101 は make_digest.py で読めない(0005 に VEHICLE.npz が無い)'
print('problems:', bad)
out = sys.argv[1] if len(sys.argv) > 1 else HERE + 'eval_storyboard.json'
json.dump(story, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote', out, len(story['scenes']), 'scenes')
