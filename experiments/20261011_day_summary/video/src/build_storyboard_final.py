#!/usr/bin/env python3
"""10/10 のまとめ動画の絵コンテ(確定版)を作り、決まりごとを確かめる。

    python build_storyboard_final.py      # → ../summary_storyboard.json

下書き(build_storyboard.py → summary_storyboard_draft.json)を事実確認して直した版。
確かめること: 見出し ≤18字、項目 ≤30字・≤6個、字幕 ≤60字、読みに「値」「柵」「ふり返り」「%」や
読み替え前の略語が残っていないこと、図のファイルがあって開けること。
長さの目安は Open JTalk 5.70字/秒・Gemini 6.42字/秒(過去のキャッシュの実測)。本当の長さは
make_digest.py の --stills 1 の時間表で確かめる。
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'summary_storyboard.json')
FIG = '/home/jetson/jetformer_reports/summary_20261010/fig/'
COURSE = '/home/jetson/jetformer_reports/overnight_20261011/course/course_map.png'

L006, L007, L008 = '20261010-172910', '20261010-173445', '20261010-181504'
LTRY, L010 = '20261010-155820', '20261010-204208'


def ln(text, say, expression='normal', pose='normal'):
    return {'text': text, 'say': say, 'expression': expression, 'pose': pose}


def sc(id_, kind, log, caption, a, b, speed=1.0, highlights=(), items=None, image=None, lines=()):
    d = {'id': id_, 'kind': kind, 'log': log, 'caption': caption, 'src_start': a, 'src_end': b,
         'speed': speed, 'highlights': list(highlights)}
    if items:
        d['summary_items'] = items
    if image:
        d['image'] = image
    d['lines'] = list(lines)
    return d


SCENES = [
    # ---------------------------------------------------------------- (1) オープニング
    sc('t01_title', 'title', L006, '10月10日の一日', 60.0, 66.0, items=[
        'TT-02 SRX+Jetson Orin Nano',
        '10/10 01:58 着手 〜 10/11 朝',
        '作る・集める・学ぶ・走らせる',
    ], lines=[
        ln('ラピカです。10月10日の一日を、約5分でまとめます。',
           'ラピカです。10月10日の一日を、約5分でまとめます。', 'happy'),
        ln('JetFormerを作り、データを集めて学習し、AIで走らせた一日でした。',
           'ジェットフォーマーを作り、データを集めて学習し、エーアイで走らせた一日でした。', 'normal', 'point'),
    ]),
    # ---------------------------------------------------------------- (2) 設計
    sc('d01_design', 'summary', L006, 'TransFuserを手本に', 66.0, 72.0, image=FIG + 'architecture.png', lines=[
        ln('手本は、カメラとLiDARをTransformerで融合する自動運転モデル、TransFuserです。',
           '手本は、カメラとライダーをトランスフォーマーで融合する、トランスフューザーです。',
           'normal', 'point'),
        ln('JetFormerは5種類の入力を、出力用と合わせて95個のトークンにします。',
           'ジェットフォーマーは、5種類の入力を、出力用と合わせて、95個のトークンにします。',
           'normal', 'point'),
        ln('融合は最終段の1回だけ。推論1.9ミリ秒で、0.33秒先までの操作を出します。',
           '融合は最終段の1回だけ。推論は1.9ミリ秒で、0.33秒先までの操作を出します。', 'happy'),
    ]),
    # ---------------------------------------------------------------- (3) データ収集
    sc('c01_data', 'summary', L006, 'データを集める', 72.0, 80.0, highlights=['timeline'], items=[
        '20秒ずつのシーン(nuScenes風)',
        'カメラ・ToF・IMU・操作を同時に記録',
        '声のカウントダウンとピーで開始',
        '午後: 人の低速走行 22シーン 約7分',
        '夜: 「綺麗な走行」30シーンを追加',
        '合計52シーン → 夜のモデルA・B',
    ], lines=[
        ln('データは20秒ずつのシーンで集め、車のスピーカーの「ピー」で記録を始めます。',
           'データは20秒ずつのシーンで集め、車のスピーカーのピーで、記録を始めます。',
           'normal', 'point'),
        ln('午後は人の低速走行を22シーン、約7分。この日のAIが学んだのは、これだけです。',
           '午後は、人の低速走行を22シーン、約7分。この日のエーアイが学んだのは、これだけです。', 'surprised'),
        ln('夜には「綺麗な走行」を30シーン足して、52シーンにしました。',
           '夜には、綺麗な走行を30シーン足して、52シーンにしました。'),
    ]),
    # ---------------------------------------------------------------- (4) 学習とオフライン評価
    sc('m01_copycat', 'summary', L006, '学習とコピーキャット', 80.0, 86.0, image=FIG + 'training_results.png', lines=[
        ln('学習は、人の操作をまねる模倣学習です。',
           '学習は、人の操作をまねる、模倣学習です。', 'normal', 'point'),
        ln('でも14時56分、モデルが入力の「今の操作」をまねていると分かりました。',
           'でも14時56分、モデルが、入力の今の操作をまねていると分かりました。', 'surprised', 'point'),
        ln('ステアの誤差は、人の操作を入れると36.1、自分の予測では64.7マイクロ秒です。',
           'ステアの誤差は、人の操作を入れると36.1、自分の予測を入れると64.7マイクロ秒です。', 'worried'),
    ]),
    sc('m02_ab', 'summary', L006, '夜のモデルAとB', 86.0, 92.0, image=FIG + 'training_results.png', lines=[
        ln('夜は52シーンで、今の操作を入れるAと、入れないBを学習しました。',
           '夜は52シーンで、今の操作を入れるエーと、入れないビーを学習しました。', 'normal', 'point'),
        ln('走行と同じ条件ではBが最良。当日のモデルより誤差がステア36%、スロットル57%減。',
           '走行と同じ条件ではビーが最良で、当日のモデルより誤差が、ステアで36、スロットルで57パーセント減りました。',
           'happy', 'point'),
        ln('ただ、この比べ方はBに有利で、AもBもまだ実車では走っていません。',
           'ただ、この比べ方はビーに有利で、エーもビーも、まだ実車では走っていません。', 'worried'),
    ]),
    # ---------------------------------------------------------------- (5) AI走行の評価
    sc('r01_timeline', 'summary', L006, '一日の流れ', 92.0, 98.0, image=FIG + 'day_timeline.png', lines=[
        ln('AI走行は15時50分から20時47分まで。初期、ゲイン調整、リカバリー、夜の4段階です。',
           'エーアイ走行は、15時50分から20時47分まで。初期、ゲイン調整、リカバリー、夜の、4つの段階です。',
           'normal', 'point'),
        ln('Jetsonは電池で動いていて、電池切れで5回以上止まりました。',
           'ジェットソンは電池で動いていて、電池切れで、5回以上止まりました。', 'worried'),
    ]),
    sc('r02_course', 'summary', L006, 'コースと3つの難所', 98.0, 104.0, image=COURSE, lines=[
        ln('コースは左回りで、1周約25.5メートル。',
           'コースは左回りで、1周、約25.5メートル。', 'normal', 'point'),
        ln('難所は、右ヘアピンのR1、バックストレートの角のDE、緑マットのGMです。',
           '難所は、右ヘアピンのアールワン、バックストレートのかどのディーイー、緑のマットのジーエムです。',
           'normal', 'point'),
    ]),
    sc('r03_legend', 'legend', L006, '画面の見かた', 20.0, 27.0,
       highlights=['prediction', 'timeline', 'tof', 'gauges'], lines=[
           ln('左はAIの目のカメラとセンサー。紫の帯は、AIが予想した進路です。',
              '左は、エーアイの目のカメラとセンサー。紫の帯は、エーアイが予想した進路です。', 'normal', 'point'),
           ln('下は走行の全体。紫がAIの運転、白い線が人の交代、オレンジの印が衝撃です。',
              '下は、走行の全体。紫がエーアイの運転、白い線が人の交代、オレンジの印が衝撃です。', 'normal', 'point'),
       ]),
    sc('r04_hold', 'clip', LTRY, '初期: HOLDで止まる', 28.5, 41.0, speed=0.75,
       highlights=['timeline', 'gauges'], lines=[
           ln('初期の走行では、車両データが途切れて、AIが中立で止まるHOLDが何度も起きました。',
              '初期の走行では、車両データが途切れて、エーアイが中立で止まる、ホールドが何度も起きました。',
              'surprised', 'point'),
           ln('16時10分に通信エラーとして直し、006と007では0回に。',
              '16時10分に通信エラーとして直し、ゼロゼロロクとゼロゼロナナでは0回に。'),
           ln('ただ記録は基板の再起動と同じ形で、原因はまだ確かめていません。',
              'ただ、記録は基板の再起動と同じ形で、原因はまだ確かめていません。', 'worried'),
       ]),
    sc('r05_006', 'clip', L006, '006 約16周・交代1回', 37.9, 57.4, speed=1.0,
       highlights=['laps', 'prediction', 'speed'], lines=[
           ln('17時29分の006は、AIが150秒以上走り続けて約16周。交代は最後の1回だけ。',
              '17時29分のゼロゼロロクは、エーアイが150秒以上走り続けて、約16周。交代は最後の1回だけ。',
              'happy', 'point'),
           ln('この2周は9.75秒と9.82秒。2.5G以上の衝突がない周で、人と同じくらいです。',
              'この2周は、9.75秒と9.82秒。2.5ジー以上の衝突がない周で、人と同じくらいです。',
              'happy'),
           ln('ただ衝突は22回。半分はDEで、当たっても走り続けていました。',
              'ただ、衝突は22回。半分はディーイーで、当たっても走り続けていました。', 'worried'),
       ]),
    sc('r06_007_r1', 'clip', L007, '007 R1で止まる', 137.5, 147.5, speed=0.5,
       highlights=['full_lock', 'impact', 'reverse'], lines=[
           ln('007では、R1で白い板に正面から当たって止まる場面が続きました。',
              'ゼロゼロナナでは、アールワンで白い板に正面から当たって、止まる場面が続きました。',
              'surprised', 'point'),
           ln('AIは後退を出し続けますが、ESCは中立をはさまないと下がりません。',
              'エーアイはバックを出し続けますが、スピードコントローラーは、中立をはさまないと下がりません。',
              'worried', 'point'),
           ln('交代11回のうち10回がR1から。人が中立をはさむと、すぐ下がりました。',
              '交代11回のうち10回が、アールワンから。人が中立をはさむと、すぐに下がりました。'),
       ]),
    sc('r07_008', 'clip', L008, '008 リカバリーv1', 25.0, 35.8, speed=0.5,
       highlights=['reverse', 'speed', 'timeline'], lines=[
           ln('18時15分の008では、自動リカバリーの初版を試しました。',
              '18時15分のゼロゼロハチでは、自動リカバリーの初版を試しました。', 'normal', 'point'),
           ln('ぶつかって止まると働きますが、後退が弱く、4回とも下がれませんでした。',
              'ぶつかって止まると働きますが、バックが弱く、4回とも下がれませんでした。', 'worried', 'point'),
           ln('下の帯の赤い所は、基板の再起動でAIが止まった所。再起動はこの走行で12回です。',
              '下の帯の赤い所は、基板の再起動で、エーアイが止まった所。再起動は、この走行で12回です。', 'surprised', 'point'),
       ]),
    sc('r08_night', 'clip', L010, '夜: 衝突が2倍以上', 26.9, 41.2, speed=0.75,
       highlights=['impact', 'timeline', 'tof'], lines=[
           ln('夜の3本は、1周あたりの衝突が2.86回。006と007の2倍以上でした。',
              '夜の3本は、1周あたりの衝突が、2.86回。ゼロゼロロクとゼロゼロナナの、2倍以上でした。',
              'surprised', 'point'),
           ln('カメラとToFが下向きにずれた疑いがありますが、衝突が増えた原因かは未確定です。',
              'カメラと距離センサーが下向きにずれた疑いがありますが、衝突が増えた原因かは、まだ分かりません。',
              'worried'),
           ln('リカバリーv2は、人の交代のほうが早く、一度も作動しませんでした。',
              'リカバリーブイツーは、人の交代のほうが早く、一度も作動しませんでした。', 'worried'),
       ]),
    sc('r09_runs', 'summary', L006, 'AI走行11本の結果', 104.0, 110.0, image=FIG + 'runs_results.png', lines=[
        ln('記録が残る11本で、AIの時間は748秒、66.5周。衝突は1周あたり1.56回でした。',
           '記録が残る11本で、エーアイの時間は748秒、66.5周。衝突は1周あたり1.56回でした。', 'normal', 'point'),
        ln('衝突なしの周のベストは007の9.68秒。タイムは人並みで、衝突は人より多めです。',
           '衝突なしの周のベストは、ゼロゼロナナの9.68秒。タイムは人並みで、衝突は、人より多めです。',
           'happy', 'point'),
    ]),
    # ---------------------------------------------------------------- (6) 日中に作ったもの
    sc('b01_built', 'summary', L007, '走りながら作ったもの', 180.0, 188.0, highlights=['laps', 'tof'], items=[
        '17:54 リカバリーv1・後退前に中立',
        '17:58 周回とラップの表示・読み上げ',
        '18:33 リカバリーv2(人とESCの手順)',
        '18:39 基板の再起動からすぐ復帰',
        '18:43 ToFガード(当日は記録だけ)',
        '20:36 衝突の前後を学習から外す',
    ], lines=[
        ln('走りながら、この6つの道具を作りました。',
           '走りながら、この6つの道具を作りました。', 'happy', 'point'),
        ln('記録だけのガードは、夜の衝突21回中20回を0.3秒以上前に警告。誤警報は1分に17.5回。',
           '記録だけのガードは、夜の衝突21回中20回を、0.3秒以上前に警告。ごけいほうは、1分に17.5回。',
           'normal', 'point'),
        ln('リカバリーv2の後退と、ガードの減速は、まだ実車で試せていません。',
           'リカバリーブイツーのバックと、ガードの減速は、まだ実車で試せていません。', 'worried'),
    ]),
    # ---------------------------------------------------------------- (7) 探索の結果: 速くて安全な計画
    sc('p01_plan', 'summary', L006, 'ぶつからずに速く', 110.0, 116.0, image=FIG + 'plan.png', lines=[
        ln('006と007の衝突の9割は、R1・DE・GMの3か所。AIが人より遅いのは周の後半です。',
           'ゼロゼロロクとゼロゼロナナの衝突の9割は、3つの難所。エーアイが人より遅いのは、周の後半です。',
           'normal', 'point'),
        ln('だからヘアピンは遅く入り、直線で速く走ります。',
           'だから、ヘアピンは遅く入り、直線で速く走ります。', 'happy', 'point'),
        ln('目標は、1周1.17回の衝突を、10月25日に0.9回以下、12月ごろに0.7回以下へ。',
           '目標は、1周1.17回の衝突を、10月25日に0.9回以下、12月ごろに0.7回以下へ。',
           'normal', 'point'),
        ln('数字は目標で、効果は見込みです。実車ではまだ試していません。',
           '数字は目標で、効果は見込みです。実車では、まだ試していません。', 'worried'),
    ]),
    # ---------------------------------------------------------------- (8) 次にやること・エンディング
    sc('n01_next', 'summary', L006, '次にやる3つ', 116.0, 124.0, items=[
        '① 測れる状態に(取付・電源) 〜10/14',
        '10/14 基準を測り直す(候補はモデルB)',
        '② 安全層v1: R1・GM・DEの速度上限',
        '再生で確認 → 10/20 実車でA/B/A',
        '③ 遅く入って速く出る手本 10/16',
        '学習し直し → 10/22 実車でA/B/A',
    ], lines=[
        ln('まず14日の朝までに、取り付けと電源を直して、測れる状態にします。',
           'まず14日の朝までに、取り付けと電源を直して、測れる状態にします。',
           'normal', 'point'),
        ln('次に、R1・GM・DEの速度上限を作り、記録の再生で確かめて、20日に実車で比べます。',
           '次に、3つの難所の速度上限を作り、記録の再生で確かめて、20日に実車で比べます。',
           'normal', 'point'),
        ln('そして16日に「遅く入って速く出る」手本を集めて学習し直し、22日に実車で比べます。',
           'そして16日に、遅く入って速く出る手本を集めて学習し直し、22日に実車で比べます。'),
    ]),
    sc('e01_end', 'ending', L006, 'おつかれさまでした', 124.0, 130.0, items=[
        'タイムは人並み・衝突は人より多め',
        '次は測れる状態から(10/14)',
        '安全層を作るかはご判断ください',
    ], lines=[
        ln('次は衝突を減らす番です。安全層を作るかどうか、ご判断をお願いします。',
           '次は、衝突を減らす番です。安全層を作るかどうか、ご判断をお願いします。', 'normal', 'point'),
        ln('おつかれさまでした!',
           'おつかれさまでした。', 'happy'),
    ]),
]

STORY = {'title': 'JetFormer 10月10日のまとめ', 'log': L006, 'scenes': SCENES}

# ---------------------------------------------------------------- 確認
BAD_SAY = ['値', '柵', 'ふり返', '振り返', '%', 'AI', 'ToF', 'eRPM', 'ESC', 'TransFuser', 'JetFormer', 'ms', 'µs',
           'HOLD', 'IMU', 'R1', 'DE', 'GM', 'v1', 'v2', 'G ', 'fps', 'Jetson', 'LiDAR', 'TensorRT']
RATES = {'openjtalk': 5.70, 'gemini': 6.42}
LEAD, GAP, TAIL = 0.35, 0.28, 0.5


def check():
    errs, total = [], dict.fromkeys(RATES, 0.0)
    ids = set()
    for s in SCENES:
        if s['id'] in ids:
            errs.append('id 重複 %s' % s['id'])
        ids.add(s['id'])
        if len(s['caption']) > 18:
            errs.append('%s 見出し %d字: %s' % (s['id'], len(s['caption']), s['caption']))
        items = s.get('summary_items') or []
        if len(items) > 6:
            errs.append('%s 項目が %d個' % (s['id'], len(items)))
        for it in items:
            if len(it) > 30:
                errs.append('%s 項目 %d字: %s' % (s['id'], len(it), it))
        if s.get('image'):
            try:
                from PIL import Image
                Image.open(s['image']).verify()
            except Exception as e:                      # noqa: BLE001
                errs.append('%s 図が開けない: %s (%s)' % (s['id'], s['image'], e))
        chars = 0
        for x in s['lines']:
            if len(x['text']) > 60:
                errs.append('%s 字幕 %d字: %s' % (s['id'], len(x['text']), x['text']))
            for b in BAD_SAY:
                if b in x['say']:
                    errs.append('%s 読みに「%s」: %s' % (s['id'], b, x['say']))
            if re.search(r'[A-Za-z]', x['say']):
                errs.append('%s 読みに英字: %s' % (s['id'], x['say']))
            if x['expression'] not in ('normal', 'happy', 'surprised', 'worried') or x['pose'] not in ('normal', 'point'):
                errs.append('%s 表情/ポーズ' % s['id'])
            chars += len(x['say'])
        clip = (s['src_end'] - s['src_start']) / s['speed']
        row = []
        for k, cps in RATES.items():
            narr = LEAD + chars / cps + GAP * (len(s['lines']) - 1) + TAIL
            total[k] += max(clip, narr)
            row.append('%s %5.1f' % (k[:4], max(clip, narr)))
        print('%-12s %-7s 映像 %5.1fs  %s  %s' % (s['id'], s['kind'], clip, '  '.join(row), s['caption']))
    print('見積もり合計: ' + ' / '.join('%s %.1f秒' % kv for kv in total.items()) + '(%d場面)' % len(SCENES))
    return errs


if __name__ == '__main__':
    errs = check()
    for e in errs:
        print('NG', e)
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(STORY, fh, ensure_ascii=False, indent=1)
    print('wrote', os.path.abspath(OUT))
    sys.exit(1 if errs else 0)
