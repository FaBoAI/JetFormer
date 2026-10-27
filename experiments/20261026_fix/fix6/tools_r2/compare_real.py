"""直す 6-2 回目・係 tools: run_real.sh の old / new の出力を比べる(JSON はバイト単位・標準出力は「出力 」の行のほか)。
  python3 -I compare_real.py → out/compare_real.txt"""
import sys
from pathlib import Path
O = Path(__file__).resolve().parent / 'out'
old, new = O / 'real_old', O / 'real_new'
lines = []
for p in sorted(old.glob('*.json')):
    q = new / p.name
    lines.append('%s json %s' % ('SAME' if q.exists() and p.read_bytes() == q.read_bytes() else 'DIFF', p.name))
for p in sorted(old.glob('*.stdout')):
    q = new / p.name
    f = lambda x: [ln for ln in x.read_text(encoding='utf-8').splitlines() if not ln.startswith('出力 ')]   # noqa: E731
    lines.append('%s stdout(出力の行のほか) %s' % ('SAME' if q.exists() and f(p) == f(q) else 'DIFF', p.name))
for p in sorted(old.glob('*.stderr')):
    q = new / p.name
    f = lambda x: [ln.replace(str(x.parent), '') for ln in x.read_text(encoding='utf-8').splitlines()]   # noqa: E731
    lines.append('%s stderr %s' % ('SAME' if q.exists() and f(p) == f(q) else 'DIFF(下に)', p.name))
(O / 'compare_real.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
sys.exit(0)
