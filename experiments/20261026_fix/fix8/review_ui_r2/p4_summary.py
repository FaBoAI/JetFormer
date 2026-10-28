"""p4_toast_phone.txt の走行中の行から、幅・高さごとの浮いている ■ の箱とトーストの重なり(縦の px)を並べる(読むだけ)。出力 out/p4_float_stop.txt"""
import re
from pathlib import Path
HERE = Path(__file__).resolve().parent
rows = [l for l in (HERE / 'out' / 'p4_toast_phone.txt').read_text(encoding='utf-8').splitlines()
        if l.startswith(('now', 'before')) and '|' in l and l.split()[1].isdigit() and ' 走行中 ' in l]
seen, out = set(), ['css 幅 高さ 文 | トースト[左,上,幅,高さ] | 浮いている ■[左,上,幅,高さ] | 重なり 面積 px²・縦 px(■ の高さのうち)']
for l in rows:
    parts = [p.strip() for p in l.split('|')]
    head = parts[0].split()
    css, w, h, name = head[0], int(head[1]), int(head[2]), head[-1]
    k = (css, w, h, name)
    if k in seen:
        continue
    seen.add(k)
    t, fs = eval(parts[1]), eval(parts[4])
    m = re.search(r'浮いている ■ (\d+)', parts[3])
    ov = int(m.group(1)) if m else 0
    dv = max(0, min(t[1] + t[3], fs[1] + fs[3]) - max(t[1], fs[1]))
    out.append('%-6s %d %d %s | %s | %s | %d・縦 %d / %d' % (css, w, h, name, t, fs, ov, dv, fs[3]))
(HERE / 'out' / 'p4_float_stop.txt').write_text('\n'.join(out) + '\n', encoding='utf-8')
print('\n'.join(out))
