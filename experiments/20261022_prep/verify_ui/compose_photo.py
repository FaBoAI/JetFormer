"""動画用の写真 ui_candidates_20261023.png を、v_ui_live2.py が保存した部分の写真(photo_part_*.png)から並べ直す。
ブラウザは使わない(読むのは画像と v_ui_live.json だけ)。

    python3 -I compose_photo.py

- 左: 10/23 の並び(ブロック 2 を押したところ)とモデルの欄。モデルの欄の選択肢は幅で切れて run の名前が見えないので、
  画面から読んだ値(v_ui_live.json の photo.model と block2.desc)を、下に「画面から読んだ値」として書き足す(写真ではないと明記)。
- 中: 候補の一覧(6 本とも「走れる」)。右: 区間の速度上限(OFF)。
"""
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPORT = Path('/home/jetson/jetformer_reports/prep_20261022')
OUT = REPORT / 'verify_ui'
RES = OUT / 'V-ui_out' / 'v_ui_live.json'
PHOTO = REPORT / 'ui_candidates_20261023.png'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
REG = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def main():
    res = json.loads(RES.read_text(encoding='utf-8'))
    ph, b2 = res['photo'], res['block2']
    assert res['ok'] and ph['model'] == b2['model'] == 'bench_20261022-slow-sec-s1' and ph['zone'] is False
    parts = {k: Image.open(OUT / ('photo_part_%s.png' % k)).convert('RGB') for k in ('blocks', 'cands', 'model', 'zone')}
    f_title, f_sub = ImageFont.truetype(BOLD, 44, index=0), ImageFont.truetype(REG, 30, index=0)
    f_lab, f_note, f_noteb = ImageFont.truetype(BOLD, 32, index=0), ImageFont.truetype(REG, 26, index=0), ImageFont.truetype(BOLD, 28, index=0)
    pad, gap, head, lab_h = 40, 30, 150, 52

    # 画面から読んだ値の箱(写真ではない)
    desc = b2['desc']
    i = desc.index('block=')
    j = desc.index(' battery=')
    w1 = parts['blocks'].width
    note_lines = [('画面から読んだ値(写真ではない)', f_note, (100, 110, 125)),
                  ('モデルの欄: %s' % ph['model'], f_noteb, (17, 24, 39)),
                  ('説明欄: …%s' % desc[i:desc.index(' model=')], f_note, (17, 24, 39)),
                  ('　　　 %s…' % desc[desc.index(' model=') + 1:j], f_note, (17, 24, 39))]
    nh = 20 + sum(42 for _ in note_lines) + 14
    note = Image.new('RGB', (w1, nh), (255, 255, 255))
    dn = ImageDraw.Draw(note)
    dn.rectangle([0, 0, w1 - 1, nh - 1], outline=(203, 213, 225), width=2)
    y = 16
    for t, f, c in note_lines:
        dn.text((22, y), t, font=f, fill=c)
        y += 42

    cols = [[parts['blocks'], parts['model'], note], [parts['cands']], [parts['zone']]]
    labels = ['10/23 の並び・ブロック 2 を押したところ', '候補の一覧(%d 本とも「走れる」)' % ph['drivable_rows'],
              '区間の速度上限(%s)' % ('OFF' if ph['zone'] is False else 'ON')]
    widths = [max(im.width for im in c) for c in cols]
    col_h = [sum(im.height for im in c) + gap * (len(c) - 1) for c in cols]
    W = pad * 2 + sum(widths) + gap * 2
    H = head + lab_h + max(col_h) + pad
    canvas = Image.new('RGB', (W, H), (236, 239, 243))
    d = ImageDraw.Draw(canvas)
    shot_at = time.strftime('%Y-%m-%d %H:%M', time.strptime(res['finished'], '%Y-%m-%d %H:%M:%S'))
    d.text((pad, 26), '走行タブ「ベンチマークの候補」(本物の画面)', font=f_title, fill=(20, 24, 31))
    d.text((pad, 88), '%s ごろ撮影・%s・走行開始と保存は押していない' % (shot_at, res['base']), font=f_sub, fill=(75, 85, 99))
    x = pad
    for c, w, lab in zip(cols, widths, labels):
        d.text((x, head), lab, font=f_lab, fill=(17, 94, 89))
        y = head + lab_h
        for im in c:
            canvas.paste(im, (x, y))
            y += im.height + gap
        x += w + gap
    canvas.save(PHOTO, optimize=True)
    print(PHOTO, W, H)


if __name__ == '__main__':
    main()
