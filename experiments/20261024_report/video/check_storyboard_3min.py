#!/usr/bin/env python3
"""10月24日のレポート(実車ベンチマーク 2 日目と予選への作戦・約 3 分)の絵コンテの決まりごとを確かめる(CPU だけ・ネットワークなし)

    /usr/bin/python3 -I check_storyboard_3min.py [../storyboard.json] [--final]

元: day8_explainer_20261017/video/src/check_storyboard_2min.py(2 分用)と
real_bench_20261023/video/src/check_storyboard_bench.py(10/23 の 3 分版)。元のファイルは変えていない。日付を 10月24日 にした。
作業(まとめ)は 10/25 の未明だが、10/24 の実車の日のレポートにする。だから 10/24 より後の日付は「予定の日付」として誤りにする。
  - 時刻(12:00・N時・何時・午前・午後)を入れない(「N時間」は時刻ではないので可)
  - 予定の日付・曜日・予定の言葉を入れない: 10/24 より後の日付(10月25日・11/3・3日 など)、曜日、予定・明日 など。
    過去の日付(10/23)と今日(10月24日)、「予選突破」「次の実車までに」は可
  - 「2本目」と言わない(「2本の平均」は方策の呼び名なので可)
  - 最初の場面の見出しが「10月24日のレポート」。冒頭(最初の 2 場面)で「前回」の課題を言う
  - 流れ(ユーザーの決まり)の順: レポート → 前回 → 今日の課題 → 対策 → 結果(1 回目 → 満充電 → ベスト → 3 回目の DE)
    → 課題・対策・結果 → 条件の違い → 3 周のタイムと目標 → 3 つのパターン → 次の実車までにやること → 今日のまとめ → 課題
  - 「課題・対策・結果」と「3 つのパターン」の場面には表の図がある。3 周の場面で 25 秒・27 秒・23 秒と「安定」を言う
  - 最後の 2 場面が「今日のまとめ」と「課題」。最後のセリフに「おつかれさまでした」
  - 本物の画像を見せる: 前回(10/23)の動画の最後の場面・走行タブの本物の画面の写真・実車のカメラ(人の写っていないコマ 4 枚と、
    9 番・12 番の記録)があり、make_slides_1024.py がそれを使っている
  - 背景の走行は 10/24 の 9 番(20261024-162033)。src_start が人の写らない区間(126.0〜137.5 秒)にある
  - スライドの元(make_slides_1024.py)の文字にも、時刻・予定の日付・曜日・「VL53L9A」・「TransFuser型」だけの呼び方が無い
  - 読み上げの長さ: 見積もり 170〜190 秒(モーラ数。外れたら誤り)。字数の見積もり(150〜190 秒)・合計の字数(900〜1,150 字)は
    目安(外れたら注意)。数字の多いセリフは字数の割に長く読むので、字数の見積もりは短めに出る
  - 場面は 14〜18(外れたら誤り)
  - make_digest.py で書き出せる形(鍵・kind・src_end > src_start・背景の走行の記録がある・セリフがある)
そのまま引き継いだもの: 見出し 18 字まで・まとめの項目 6 個まで 1 個 20 字まで(半角 0.5)・図の無いタイトルの場面は項目 5 個まで・
  吹き出し 60 字まで・読み上げ(say)に英字・µ・%・×・→・/ を残さない・ToF の表記は「STEVAL-VL53L9」・
  「TransFuser型」だけの呼び方は使わない・図が無いときは注意(--final で誤り)・図は 2160×1350 か(外れたら注意)
"""
import importlib.util
import json
import os
import re
import sys
import unicodedata

sys.dont_write_bytecode = True                  # 読み込む day8 などのフォルダに __pycache__ を作らない

VIDEO = '/home/jetson/jetformer_reports/bench_20261024_report/video'   # src は experiments へのリンクなので絶対パス
ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
FINAL = '--final' in sys.argv
SB = ARGS[0] if ARGS else os.path.join(VIDEO, 'storyboard.json')
SLIDES = '/home/jetson/jetformer/experiments/20261024_report/video/make_slides_1024.py'

SAY_BAD_RE = re.compile(r'[A-Za-zＡ-Ｚａ-ｚµμ%％×→/]')
CLOCK_RE = re.compile(r'\d{1,2}[:：]\d{2}|\d+\s*時(?!間)|何時|午前|午後')
PLAN_RE = re.compile(r'予定|明日|あした|明後日|来週|再来週|今週末|週末|昨日|きのう|あさって|今夜|今晩|夕方|今朝')
WEEKDAY_RE = re.compile(r'[月火水木金土日]曜|[(（]\s*[月火水木金土日]\s*[)）]|[(（]\s*[月火水木金土日]・祝\s*[)）]')
DATE_RE = re.compile(r'(\d{1,2})月(\d{1,2})日|(?<![\d.])(\d{1,2})/(\d{1,2})(?![\d/])|(?<![\d月/.])(\d{1,2})日(?!目|間|分)')
NTH_RE = re.compile(r'2\s*本目|二本目|２本目')
TOF_BAD_RE = re.compile(r'VL53L9A|(?<!STEVAL-)VL53L9')
TF_BAD_RE = re.compile(r'TransFuser型|トランスフューザー型')
TODAY = (10, 24)
FIRST = '10月24日のレポート'
LAST_WORD = 'おつかれさまでした'
SUMMARY = '今日のまとめ'
# 流れの順(見出しの先頭か中身で探す)
FLOW = [('レポート', lambda c: c.startswith(FIRST)), ('前回', lambda c: c.startswith('前回')),
        ('今日の課題', lambda c: c.startswith('今日の課題')), ('対策', lambda c: c.startswith('対策')),
        ('結果: 1回目', lambda c: c.startswith('結果') and '1回目' in c), ('結果: 満充電', lambda c: c.startswith('結果') and '満充電' in c),
        ('結果: ベスト', lambda c: c.startswith('結果') and 'ベスト' in c), ('結果: 3回目のDE', lambda c: c.startswith('結果') and 'DE' in c),
        ('課題・対策・結果', lambda c: c == '課題・対策・結果'), ('条件の違い', lambda c: c.startswith('条件の違い')),
        ('3周と目標', lambda c: c.startswith('3周')), ('3つのパターン', lambda c: '3つ' in c),
        ('次の実車までにやること', lambda c: c.startswith('次の実車まで')), ('今日のまとめ', lambda c: c == SUMMARY),
        ('課題', lambda c: c == '課題')]
TABLES = ('課題・対策・結果', '3つのパターン')
SCENES_LO, SCENES_HI = 14, 18
CHARS_LO, CHARS_HI = 900, 1150
CSEC_LO = 150.0                                 # 字数の見積もりの下限(数字の多いセリフは字数より長く読む。目安だけ)
SEC_LO, SEC_HI = 170.0, 190.0
RATE = 6.4                                      # Achernar の実測 6.2〜6.6 字/秒
LEAD, GAP, TAIL = 0.35, 0.28, 0.5               # make_digest.py の間: 前 0.35・行間 0.28・後 0.5 秒
EST_MORA = '/home/jetson/jetformer_reports/day8_explainer_20261017/video/src/estimate_narration.py'
REP = '/home/jetson/jetformer_reports'
RAW = '/home/jetson/jetformer/data/raw'
LOG9 = '20261024-162033'
PEOPLE_FREE = [(126.0, 137.5)]                  # 9 番のカメラに人が写らない区間(0.5 秒ごとに目で確かめた)
NEED_KEYS = ('id', 'kind', 'log', 'caption', 'src_start', 'src_end', 'speed', 'highlights', 'lines')
KINDS = ('title', 'legend', 'clip', 'summary', 'ending')
FIG = REP + '/bench_20261024_report/fig/'
# 本物の画像(図の元)。(名前, パス, make_slides_1024.py の中で使っている印)
REAL = [('前回(10/23)の動画の最後の場面', VIDEO + '/frames/prev_issue_20261023.png', 'prev_issue_20261023.png'),
        ('前回(10/23)の動画', REP + '/real_bench_20261023/JetFormer_10月23日のレポート_実車ベンチマーク1日目_3分版.mp4', 'PREV_VIDEO'),
        ('走行タブの本物の画面の写真(10/24)', REP + '/bench_20261024_s1/ui_check/ui_tune_block5.png', 'ui_tune_block5.png'),
        ('実車のカメラ: 最終コーナーの出口', FIG + 'cam_sw_exit.png', "cam('sw_exit')"),
        ('実車のカメラ: 9 番の直線', FIG + 'cam_no9_straight.png', "cam('no9_straight')"),
        ('実車のカメラ: DE で当たる直前(12 番)', FIG + 'cam_de_hit.png', "cam('de_hit')"),
        ('実車のカメラ: DE の同じ場所(9 番)', FIG + 'cam_de_clean.png', "cam('de_clean')"),
        ('実車の記録: 9 番(R1 の板)', RAW + '/' + LOG9, 'R1_FRAMES'),
        ('実車の記録: 12 番(R1 の板)', RAW + '/20261024-183437', 'R1_FRAMES')]


def mora_line_sec():
    """day8 の estimate_narration.line_sec(Open JTalk のモーラ数)。読めなければ None"""
    try:
        spec = importlib.util.spec_from_file_location('estimate_narration', EST_MORA)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m.line_sec
    except Exception as ex:
        print('注意 モーラの見積もりを読めない(%s)。字数の見積もりだけ' % ex)
        return None


def width(s):
    """全角 = 1、半角 = 0.5 として数える"""
    return sum(1.0 if unicodedata.east_asian_width(c) in 'WFA' else 0.5 for c in s)


def future_dates(s):
    """今日(10/24)より後の日付。月の無い「N日」は 10 月とみなす"""
    out = []
    for m in DATE_RE.finditer(s):
        if m.group(1):
            md = (int(m.group(1)), int(m.group(2)))
        elif m.group(3):
            md = (int(m.group(3)), int(m.group(4)))
        else:
            md = (TODAY[0], int(m.group(5)))
        if md > TODAY:
            out.append(m.group(0))
    return out


def bad_words(x):
    """時刻・予定・曜日・予定の日付・2本目・ToF・TransFuser の誤り(見つかった種類の一覧)"""
    out = []
    if CLOCK_RE.search(x):
        out.append('時刻の言葉')
    if PLAN_RE.search(x):
        out.append('予定・相対の日の言葉')
    if WEEKDAY_RE.search(x):
        out.append('曜日')
    fd = future_dates(x)
    if fd:
        out.append('予定の日付 %s' % '・'.join(fd))
    if NTH_RE.search(x):
        out.append('「2本目」')
    if TOF_BAD_RE.search(x):
        out.append('ToF の表記は STEVAL-VL53L9')
    if TF_BAD_RE.search(x):
        out.append('「TransFuser型」だけの呼び方')
    return out


def slide_strings(path):
    """make_slides_1024.py の文字列(図に描く文字)だけを取り出す。説明の docstring と注釈は除く"""
    import ast
    tree = ast.parse(open(path, encoding='utf-8').read())
    doc = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef)) and ast.get_docstring(node):
            doc.add(ast.get_docstring(node, clean=False))
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value not in doc]


def main():
    sb = json.load(open(SB, encoding='utf-8'))
    errs, warns, n_say, est, est_m = [], [], 0, 0.0, 0.0
    line_sec = mora_line_sec()
    try:
        from PIL import Image
    except Exception:
        Image = None
    n_img = 0
    for sc in sb['scenes']:
        sid = sc['id']
        miss = [k for k in NEED_KEYS if k not in sc]
        if miss:
            errs.append('%s: 鍵が無い %s' % (sid, miss))
        if sc.get('kind') not in KINDS:
            errs.append('%s: kind が不明: %s' % (sid, sc.get('kind')))
        if not float(sc.get('src_end', 0)) > float(sc.get('src_start', 0)):
            errs.append('%s: src_end が src_start 以下' % sid)
        lg = sc.get('log') or sb.get('log', '')
        if not os.path.isdir(os.path.join(RAW, lg)):
            errs.append('%s: 背景の走行の記録が無い: %s' % (sid, lg))
        if lg != LOG9:
            errs.append('%s: 背景の走行が 9 番(%s)ではない: %s' % (sid, LOG9, lg))
        if not any(a <= float(sc.get('src_start', -1)) < b for a, b in PEOPLE_FREE):
            errs.append('%s: src_start %.2f が人の写らない区間 %s の外' % (sid, float(sc.get('src_start', -1)), PEOPLE_FREE))
        if sc.get('kind') == 'clip':
            errs.append('%s: kind=clip(走行の映像)は人が写るので置かない' % sid)
        if not sc.get('lines'):
            errs.append('%s: セリフが無い' % sid)
        cap = sc.get('caption', '')
        if len(cap) > 18:
            errs.append('%s: caption %d 字 > 18: %s' % (sid, len(cap), cap))
        items = sc.get('summary_items') or []
        if len(items) > 6:
            errs.append('%s: summary_items %d 個 > 6' % (sid, len(items)))
        if sc.get('kind') == 'title' and not sc.get('image') and len(items) > 5:
            errs.append('%s: 図の無いタイトルの場面の summary_items %d 個 > 5(はみ出す)' % (sid, len(items)))
        for it in items:
            if width(it) > 20:
                errs.append('%s: 項目 %.1f 字 > 20: %s' % (sid, width(it), it))
        if sc.get('image'):
            if not os.path.exists(sc['image']):
                (errs if FINAL else warns).append('%s: 図が未作成: %s' % (sid, sc['image']))
            else:
                n_img += 1
                if Image is not None:
                    wh = Image.open(sc['image']).size
                    if wh != (2160, 1350):
                        warns.append('%s: 図の大きさ %s(目安 2160×1350)' % (sid, wh))
        shown = [cap] + items
        sc_say = 0
        for ln in sc.get('lines', []):
            t, s = ln['text'], ln.get('say') or ln['text']
            n_say += len(s)
            sc_say += len(s)
            shown.append(t)
            if len(t) > 60:
                errs.append('%s: text %d 字 > 60: %s' % (sid, len(t), t))
            bad = sorted(set(SAY_BAD_RE.findall(s)))
            if bad:
                errs.append('%s: say に英字・記号 %s: %s' % (sid, ''.join(bad), s))
            if ln.get('expression', 'normal') not in ('normal', 'happy', 'surprised', 'worried'):
                errs.append('%s: expression が不明: %s' % (sid, ln.get('expression')))
            if ln.get('pose', 'normal') not in ('normal', 'point'):
                errs.append('%s: pose が不明: %s' % (sid, ln.get('pose')))
            print('  %-13s text %2d字 say %2d字  %s' % (sid, len(t), len(s), s))
        for x in shown + [ln.get('say') or '' for ln in sc.get('lines', [])]:
            for b in bad_words(x):
                errs.append('%s: %s: %s' % (sid, b, x))
        nl = len(sc.get('lines', []))
        sec = (LEAD + sc_say / RATE + GAP * (nl - 1) + TAIL) if nl else 1.5
        est += sec
        if line_sec:
            secm = (LEAD + sum(line_sec(ln.get('say') or ln['text']) for ln in sc['lines']) + GAP * (nl - 1) + TAIL) if nl else 1.5
            est_m += secm
            print('  %-13s 見積もり 字数 %.1f 秒・モーラ %.1f 秒  %s' % ('', sec, secm, cap))
        else:
            print('  %-13s 見積もり %.1f 秒  %s' % ('', sec, cap))
    for b in bad_words(sb.get('title', '')):
        errs.append('title: %s: %s' % (b, sb.get('title')))
    caps = [sc.get('caption') or '' for sc in sb['scenes']]
    if not caps[0].startswith(FIRST):
        errs.append('最初の場面の見出しが「%s」ではない: %s' % (FIRST, caps[0]))
    head = ' '.join(ln['text'] + (ln.get('say') or '') for sc in sb['scenes'][:2] for ln in sc['lines']) + ' '.join(caps[:2])
    if '前回' not in head:
        errs.append('冒頭(最初の 2 場面)で前回の課題を言っていない')
    # 流れの順
    pos = []
    for name, f in FLOW:
        idx = [i for i, c in enumerate(caps) if f(c)]
        if not idx:
            errs.append('流れの場面が無い: %s' % name)
            pos.append(None)
        else:
            pos.append(idx[0])
            if name in TABLES and not sb['scenes'][idx[0]].get('image'):
                errs.append('「%s」の場面に表の図が無い' % name)
    got = [p for p in pos if p is not None]
    if got != sorted(got):
        errs.append('流れの順が違う: %s' % [(n, p) for (n, _), p in zip(FLOW, pos)])
    print('  流れ: ' + ' → '.join('%s(%s)' % (n, p + 1 if p is not None else '-') for (n, _), p in zip(FLOW, pos)))
    if len(caps) < 2 or caps[-2] != SUMMARY or caps[-1] != '課題':
        errs.append('最後の 2 場面が「今日のまとめ」と「課題」ではない: %s' % caps[-2:])
    last = sb['scenes'][-1].get('lines') or [{}]
    if LAST_WORD not in (last[-1].get('text', '') + (last[-1].get('say') or '')):
        errs.append('最後のセリフに「%s」が無い' % LAST_WORD)
    tt = [sc for sc in sb['scenes'] if (sc.get('caption') or '').startswith('3周')]
    if tt:
        said = ' '.join(ln['text'] + (ln.get('say') or '') for ln in tt[0]['lines'])
        for w in ('25秒', '27秒', '23秒', '安定'):
            if w not in said:
                errs.append('3 周の場面で「%s」を言っていない' % w)
    # 本物の画像
    src = open(SLIDES, encoding='utf-8').read() if os.path.exists(SLIDES) else ''
    for name, path, mark in REAL:
        ok_path = os.path.exists(path)
        ok_use = mark in src
        print('  本物の画像 %-30s %s %s' % (name, 'あり' if ok_path else '無い', '使用' if ok_use else '未使用'))
        if not ok_path:
            errs.append('本物の画像が無い: %s(%s)' % (name, path))
        if not ok_use:
            errs.append('make_slides_1024.py が本物の画像を使っていない: %s' % name)
    # スライドに描く文字(make_slides_1024.py の文字列)
    n_str = 0
    for x in slide_strings(SLIDES) if src else []:
        n_str += 1
        for b in bad_words(x):
            errs.append('スライドの文字: %s: %s' % (b, x.replace('\n', ' ')[:60]))
    print('  スライドの文字 %d 個を確かめた(時刻・予定の日付・曜日・ToF・TransFuser の呼び名)' % n_str)
    for k in ('title', 'log', 'scenes'):
        if k not in sb:
            errs.append('storyboard に %s が無い' % k)
    if not SCENES_LO <= len(caps) <= SCENES_HI:
        errs.append('場面が %d(%d〜%d)' % (len(caps), SCENES_LO, SCENES_HI))
    print('場面 %d(図あり %d)/ 読み上げ 合計 %d 字(目安 %d〜%d)/ 見積もり 字数 %.1f 秒(%.1f 字/秒で計算)'
          % (len(caps), n_img, n_say, CHARS_LO, CHARS_HI, est, RATE))
    if not CHARS_LO <= n_say <= CHARS_HI:
        warns.append('読み上げ %d 字が目安から外れている' % n_say)
    if not CSEC_LO <= est <= SEC_HI:
        warns.append('字数の見積もり %.1f 秒が %.0f〜%.0f 秒から外れている(モーラの見積もりを正とする)' % (est, CSEC_LO, SEC_HI))
    if line_sec:
        print('モーラの見積もり %.1f 秒(目標 %.0f〜%.0f 秒。過去の動画で実長との差 −2.1〜+1.4 秒、10/23 の 3 分版は 177.4 → 実長 177.8 秒)'
              % (est_m, SEC_LO, SEC_HI))
        if not SEC_LO <= est_m <= SEC_HI:
            errs.append('モーラの見積もり %.1f 秒が %.0f〜%.0f 秒から外れている' % (est_m, SEC_LO, SEC_HI))
    elif not SEC_LO <= est <= SEC_HI:
        errs.append('字数の見積もり %.1f 秒が %.0f〜%.0f 秒から外れている' % (est, SEC_LO, SEC_HI))
    for w in warns:
        print('注意', w)
    for e in errs:
        print('NG', e)
    print('OK' if not errs else 'NG %d 件' % len(errs))
    return 1 if errs else 0


if __name__ == '__main__':
    sys.exit(main())
