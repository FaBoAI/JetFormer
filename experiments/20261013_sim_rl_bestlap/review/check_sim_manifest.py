"""ユーザーのシミュレーター(展開物)が COMPLETE-MANIFEST.json の sha256 と同じかを確かめる(読むだけ)

    python3 -I check_sim_manifest.py <展開したフォルダー> <出力 JSON>

マニフェストに無いファイル(展開のあとに増えたもの)も数える。ファイルは開いて読むだけで、何も書かない。
"""
import hashlib
import json
import os
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    root, out = sys.argv[1], sys.argv[2]
    man = json.load(open(os.path.join(root, 'COMPLETE-MANIFEST.json'), encoding='utf-8'))
    listed = {}
    bad, missing = [], []
    for e in man['files']:
        p = os.path.join(root, e['path'])
        listed[os.path.normpath(e['path'])] = e
        if not os.path.isfile(p):
            missing.append(e['path'])
            continue
        if os.path.getsize(p) != e['bytes'] or sha256(p) != e['sha256']:
            bad.append(e['path'])
    extra = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in ('__pycache__', 'node_modules')]
        for f in files:
            rel = os.path.normpath(os.path.relpath(os.path.join(d, f), root))
            if rel not in listed and rel != 'COMPLETE-MANIFEST.json':
                extra.append(rel)
    # データセットの ZIP(マニフェストの dataset 欄)
    ds = man.get('dataset') or {}
    ds_ok = None
    if ds.get('path') and ds.get('sha256'):
        p = os.path.join(root, ds['path'])
        ds_ok = os.path.isfile(p) and os.path.getsize(p) == ds.get('bytes') and sha256(p) == ds['sha256']
    pyc = []
    for d, dirs, files in os.walk(root):
        if os.path.basename(d) == '__pycache__':
            pyc += [os.path.relpath(os.path.join(d, f), root) for f in files]
    res = {'root': root, 'manifest_files': len(man['files']), 'checked_ok': len(man['files']) - len(bad) - len(missing),
           'mismatch': bad, 'missing': missing, 'extra_not_in_manifest': sorted(extra), 'pycache_files': sorted(pyc),
           'dataset_zip': {'path': ds.get('path'), 'sha256_in_manifest': ds.get('sha256'), 'ok': ds_ok}}
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in res.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
