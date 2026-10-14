"""第 3〜4 回の審査: ユーザーのシミュレーターのファイルが変わっていないことを COMPLETE-MANIFEST.json の sha256 で確かめる(読むだけ)。
ベストラップの作業のファイル(experiments/20261013_sim_rl_bestlap・jetformer_reports/sim_rl_bestlap_20261013)が、その作業の完了のコミット 9363a19 のあと
書き換わっていないことも見る(git の差分と、9363a19 の時刻より後に変わったファイル)。

    python3 -I check_sim_files.py > ~/jetformer_reports/sim_rl_modes_20261013/r3/review/sim_files_check.json
"""
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOTS = [Path('/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5'), Path('/home/jetson/sim/extracted/course-simulator')]
REL_JSON = Path('/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5.json')
JF = Path('/home/jetson/jetformer')
BESTLAP = [JF / 'experiments/20261013_sim_rl_bestlap', Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013')]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def check_root(root, skipped):
    man = json.loads((root / 'COMPLETE-MANIFEST.json').read_text(encoding='utf-8'))
    files = man['files']
    ok, bad, miss, skip = 0, [], [], 0
    listed = set()
    for f in files:
        listed.add(f['path'])
        p = root / f['path']
        if not p.exists():
            if f['path'] in skipped:
                skip += 1
            else:
                miss.append(f['path'])
            continue
        if sha(p) == f['sha256']:
            ok += 1
        else:
            bad.append(f['path'])
    extra = sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and str(p.relative_to(root)) not in listed)
    newest = max(os.stat(p).st_mtime for p in root.rglob('*') if p.is_file())
    return {'manifest_files': len(files), 'ok': ok, 'mismatched': len(bad), 'mismatched_paths': bad[:20], 'missing': len(miss),
            'missing_paths': miss[:20], 'skipped_by_installer': skip, 'extra_files': len(extra), 'extra_sample': extra[:10],
            'newest_mtime': dt.datetime.fromtimestamp(newest).isoformat(timespec='seconds'),
            'manifest_sha256': sha(root / 'COMPLETE-MANIFEST.json')}


def main():
    rel = json.loads(REL_JSON.read_text(encoding='utf-8'))
    skipped = set(json.loads(json.dumps(rel.get('skipped') or [])))
    out = {'schema': 'jf-sim-rl-r3-review-sim-files/v1', 'release_manifest_sha256_expected': rel.get('manifest_sha256')}
    for r in ROOTS:
        out[str(r)] = check_root(r, skipped)
    # ベストラップ
    t = subprocess.run(['git', '-C', str(JF), 'log', '-1', '--format=%cI', '9363a19'], capture_output=True, text=True).stdout.strip()
    t0 = dt.datetime.fromisoformat(t).timestamp()
    diff = subprocess.run(['git', '-C', str(JF), 'diff', '--stat', '9363a19', '--', 'experiments/20261013_sim_rl_bestlap'], capture_output=True, text=True).stdout
    status = subprocess.run(['git', '-C', str(JF), 'status', '--porcelain', '--', 'experiments/20261013_sim_rl_bestlap'], capture_output=True, text=True).stdout
    newer = []
    for b in BESTLAP:
        for p in b.rglob('*'):
            if p.is_file() and os.stat(p).st_mtime > t0:
                newer.append(str(p))
    out['bestlap'] = {'commit_9363a19_time': t, 'git_diff_since_9363a19': diff.strip(), 'git_status': status.strip(),
                      'files_newer_than_commit': len(newer), 'newer_sample': newer[:10]}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
