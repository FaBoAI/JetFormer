#!/usr/bin/env python3
"""記録の段: 記録した 60 シーンを、正しい名前で 1 つの ZIP にまとめ直す(2026-10-14)

    python -I ~/jetformer/experiments/20261013_sim_rl_bestlap/record/rebuild_collection.py <session_ids.json> [--name jetformer-simulator-rl-bestlap-20min]

  - アプリの「1つのZIPにまとめる」と同じ要求(app.js buildCollection の本文そのまま)を、自分のブリッジ(127.0.0.1:8791)の
    /api/datasets/collection に送る。まとめるのはブリッジ(ユーザーの qa/combine_scene_datasets.py)で、データは変えない。
  - なぜ: 最初の記録では、アプリが連続試験の最後に自動でまとめたときの名前が既定の jetformer-simulator-id8-20min(教師の 20 分データと
    同じ名前)になった(browser_collect.py が名前を入れたのは連続試験のあとだった)。教師データと取り違えないよう、名前だけを変えて作り直す。
  結果: /home/jetson/jetformer_reports/sim_rl_bestlap_20261013/recording/collection.json
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

REPORT = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sessions_file')
    ap.add_argument('--name', default='jetformer-simulator-rl-bestlap-20min')
    ap.add_argument('--bridge', default='http://127.0.0.1:8791')
    a = ap.parse_args()
    ids = json.loads(Path(a.sessions_file).read_text())
    assert isinstance(ids, list) and len(ids) == 60 and len(set(ids)) == 60
    body = {'name': a.name, 'session_ids': ids, 'expected_frames': 600, 'expected_duration_s': 20, 'require_collision_free': True}
    req = urllib.request.Request(a.bridge + '/api/datasets/collection', data=json.dumps(body).encode(), method='POST',
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        result = json.loads(r.read())
    out = {'request': body, 'response': result, 'wall_s': round(time.time() - t0, 1),
           'zip': str(REPORT / 'sim/runtime/datasets' / result['session_id'] / result['filename'])}
    (REPORT / 'recording' / 'collection.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
