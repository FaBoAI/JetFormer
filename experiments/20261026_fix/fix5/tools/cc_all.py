"""10/23・10/24 の本物の記録の全部に car_check を流す(読むだけ。JSON・文を作業場所にだけ書く)。
    python3 -I cc_all.py <tools の根(tools/car_check.py がある所)> <出力の場所>
created_at・seconds は比べから外す(毎回変わる)"""
import json
import sys
from pathlib import Path

root, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(root / 'tools'))
sys.path.insert(0, str(root))
import car_check as cc  # noqa: E402

RAW = Path('/home/jetson/jetformer/data/raw')
LAB = Path('/home/jetson/jetformer/data/labels')
out.mkdir(parents=True, exist_ok=True)
logs = sorted(p.name for p in RAW.iterdir() if p.is_dir() and p.name[:8] in ('20261023', '20261024') and len(p.name) == 15)
for lg in logs:
    try:
        meta = json.loads((RAW / lg / 'log.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        continue
    if not (meta.get('drive') or {}) or (meta.get('drive') or {}).get('dry_run'):
        continue
    res = cc.check_log(lg, LAB, RAW)
    txt = cc.to_text(res)
    res.pop('created_at', None)
    res.pop('seconds', None)
    (out / ('cc_%s.json' % lg)).write_text(json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True,
                                                     default=cc._json_default), encoding='utf-8')
    (out / ('cc_%s.txt' % lg)).write_text('\n'.join(x for x in txt.splitlines() if not x.startswith('  (')) + '\n',
                                          encoding='utf-8')
    print(lg, res['judge']['code'])
