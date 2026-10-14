# 確認 line r3(再実行): シールドが働いた回(新しい組 + 試験の組)を、シールドなしで走らせ直す jobs。
# シールドが 1 フレームも替えなかった回は、シールドなしでも同じ走りになる(決定的)ので走らせない
import json, glob
OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3b'
ids = set()
for p in glob.glob(f'{OUT}/runs/cand_*.jsonl') + glob.glob(f'{OUT}/runs/test_*.jsonl'):
    for l in open(p):
        if l.strip():
            r = json.loads(l)
            if r.get('sh_frames', 0) > 0:
                ids.add(r['id'])
jobs = [j for f in ('jobs.json', 'testjobs.json', 'probejobs.json') for j in json.load(open(f'{OUT}/{f}'))['jobs'] if j['id'] in ids]
json.dump({'meta': {'what': 'シールドが働いた回をシールドなしで'}, 'jobs': jobs}, open(f'{OUT}/noshjobs.json', 'w'))
print(len(jobs))
