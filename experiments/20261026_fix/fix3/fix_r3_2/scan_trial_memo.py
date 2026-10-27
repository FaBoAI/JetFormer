"""本物の記録の説明欄(log.json と scene.json をまとめた effective_description)で「試し」を探す(読むだけ)。
2026-10-27 fix3 直す 3 回目・2 組目 r3-r3-rec-3。自動の「計画と違うので試し(元 cand=…)」と人の書いた「試し」を分けて数える"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, '/home/jetson/jetformer/tools')
sys.path.insert(0, '/home/jetson/jetformer')
import section_scorecard as ss  # noqa: E402

RAW = Path('/home/jetson/jetformer/data/raw')
AUTO = re.compile(r'計画と違うので試し\(元 cand=[^)]*\)')
n = auto = human = 0
for p in sorted(RAW.iterdir()):
    if not p.is_dir() or len(p.name) != 15:
        continue
    try:
        meta = json.loads((p / 'log.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        continue
    if not (meta.get('drive') or {}):
        continue
    n += 1
    desc, _ = ss.effective_description(p.name, meta.get('description', ''), RAW)
    if '試し' not in desc:
        continue
    rest = AUTO.sub('', desc)
    a = bool(AUTO.search(desc))
    h = '試し' in rest
    auto += a
    human += h
    print(p.name, 'auto' if a else '-', 'HUMAN' if h else '-', desc[:200])
print('AI の記録', n, '自動の試し', auto, '人の試し', human)
