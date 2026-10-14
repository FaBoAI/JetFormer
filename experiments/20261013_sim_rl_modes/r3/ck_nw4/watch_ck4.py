#!/usr/bin/env python3
# 確認 r4: 走らせている組を見張り、崩れ(衝突・すき間がしきい値未満・止まり・有効でない・エラー・同時に動かした sim との違い)の行と、10 分ごとの進みを 1 行ずつ出す
import json, glob, sys, time, os
d = sys.argv[1]; total = int(sys.argv[2]); seen = {}; t0 = time.time(); last = 0
DRK = ('phys', 'fc', 'delay')
while True:
    n = 0
    for f in glob.glob(f'{d}/rows_*.jsonl'):
        lines = open(f).read().split('\n')
        n += sum(1 for l in lines if l.strip())
        start = seen.get(f, 0)
        for l in lines[start:]:
            if not l.strip(): continue
            try: r = json.loads(l)
            except Exception: continue
            seen[f] = seen.get(f, 0) + 1
            if r.get('set') in ('UOFF', 'FAST', 'LOWMU', 'DELAY2', 'APP') and not r.get('error'): continue
            if r.get('error'): print('ERROR', r['id'], r['error'][:160], flush=True); continue
            if r.get('invalid_spawn'): continue
            dr = bool(r.get('phys')) or r.get('fc') or r.get('delay')
            fl = []
            if r['collisions']: fl.append('COLLISION')
            if r['min_clearance'] < (.05 if dr else .08): fl.append(f"MIN {100*r['min_clearance']:.2f}cm")
            if r.get('stall'): fl.append('STALL')
            if r.get('episode') == 'continuous' and not r.get('valid_episode_r3'): fl.append('NOT_VALID')
            if r.get('lock_mismatch'): fl.append('LOCK')
            if r.get('label_mismatch_sh0'): fl.append('LABEL')
            if fl: print('FIND', r['id'], ' '.join(fl), json.dumps(r['start']), flush=True)
    if time.time() - last > 600:
        print(f'PROGRESS {n}/{total}', flush=True); last = time.time()
    if n >= total: print('DONE', n, flush=True); break
    time.sleep(20)
