#!/usr/bin/env python3
"""1 分動画の絵コンテ(storyboard)の決まりごとを確かめる(CPU だけ・ネットワークなし)

    python3 check_storyboard.py [../simreal_1min_storyboard.json]

確かめること:
  - 見出し(caption)は 18 字まで
  - まとめの項目(summary_items)は 6 個まで・1 個 20 字まで(半角は 0.5 字と数える)
  - 吹き出し(text)は 60 字まで
  - 読み上げ(say)に英字の略語・µs・% が残っていない(AI・ToF・R1・DE・finetuning・TransFuser など)
  - 読み上げ・吹き出しに時刻(12:00 など)や予定の言葉が無い
  - 最後の 2 場面が「今日のまとめ」と「課題」
  - 図(image)のファイルがある
"""
import json
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
SB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'simreal_1min_storyboard.json')

# 読み上げに残ってはいけないもの(読みにくい記号・英字の略語)
SAY_BAD = ['AI', 'ToF', 'R1', 'DE', 'finetuning', 'fine-tuning', 'TransFuser', 'JetFormer', 'µs', 'us', '%', '％',
           'L1', 'A/B', 'C+FT']
TIME_RE = re.compile(r'\d{1,2}[:：]\d{2}|\d+時|午前|午後|明日|来週|予定')


def width(s):
    """全角 = 1、半角 = 0.5 として数える"""
    return sum(1.0 if unicodedata.east_asian_width(c) in 'WFA' else 0.5 for c in s)


def main():
    sb = json.load(open(SB, encoding='utf-8'))
    errs, n_say = [], 0
    for sc in sb['scenes']:
        sid = sc['id']
        cap = sc.get('caption', '')
        if len(cap) > 18:
            errs.append('%s: caption %d 字 > 18: %s' % (sid, len(cap), cap))
        items = sc.get('summary_items') or []
        if len(items) > 6:
            errs.append('%s: summary_items %d 個 > 6' % (sid, len(items)))
        for it in items:
            if width(it) > 20:
                errs.append('%s: 項目 %.1f 字 > 20: %s' % (sid, width(it), it))
        if sc.get('image') and not os.path.exists(sc['image']):
            errs.append('%s: 図が無い: %s' % (sid, sc['image']))
        for ln in sc.get('lines', []):
            t, s = ln['text'], ln.get('say') or ln['text']
            n_say += len(s)
            if len(t) > 60:
                errs.append('%s: text %d 字 > 60: %s' % (sid, len(t), t))
            for b in SAY_BAD:
                if re.search(r'(?<![A-Za-z])' + re.escape(b) + r'(?![A-Za-z])', s):
                    errs.append('%s: say に「%s」: %s' % (sid, b, s))
            for x in (t, s):
                if TIME_RE.search(x):
                    errs.append('%s: 時刻・予定の言葉: %s' % (sid, x))
            print('  %-11s text %2d字 say %2d字  %s' % (sid, len(t), len(s), t))
    caps = [sc.get('caption') for sc in sb['scenes']]
    if caps[-2:] != ['今日のまとめ', '課題']:
        errs.append('最後の 2 場面が「今日のまとめ」「課題」ではない: %s' % caps[-2:])
    print('読み上げ 合計 %d 字 / %d 場面' % (n_say, len(sb['scenes'])))
    for e in errs:
        print('NG', e)
    print('OK' if not errs else 'NG %d 件' % len(errs))
    return 1 if errs else 0


if __name__ == '__main__':
    sys.exit(main())
