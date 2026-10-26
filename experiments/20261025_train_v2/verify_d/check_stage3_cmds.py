#!/usr/bin/env python3
"""別の目の確かめ(2026-10-25 夜): キューが埋める段 3・4 のコマンドと名前が protocol_v2.json の決まりどおりか。
gpu_queue_v2.py の resolve・status(読むだけ。学習を始めない)の出力を、protocol の型から自分で作った期待と比べる。
標準ライブラリだけ(torch を読まない)。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python -I \
        /home/jetson/jetformer/experiments/20261025_train_v2/verify_d/check_stage3_cmds.py
"""
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

REPO = Path('/home/jetson/jetformer')
REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
Q = REPO / 'experiments' / '20261025_train_v2' / 'gpu_queue_v2.py'
OUT = Path(__file__).resolve().parent / 'check_stage3_cmds.json'
PREFIX = 'cd ~/jetformer && '

proto = json.load(open(REP / 'protocol_v2.json'))
dec = json.load(open(REP / 'decisions.json'))
ph = proto['choice_placeholders']
x, E, aug = dec['D1']['x'], dec['D2']['E'], dec['D3']['aug']
res = {'decisions': {'x': x, 'E': E, 'aug': aug}, 'checks': []}


def check(name, ok, info=None):
    res['checks'].append({'name': name, 'ok': bool(ok), 'info': info})
    print('%s  %s%s' % ('OK ' if ok else 'NG ', name, '' if info is None else '  ' + json.dumps(info, ensure_ascii=False)[:400]))


def flags(cmd):
    a = shlex.split(cmd)
    a = a[a.index('tools/train.py') + 1:]
    out, i = {}, 0
    while i < len(a):
        if a[i].startswith('--'):
            if i + 1 < len(a) and not a[i + 1].startswith('--'):
                out[a[i]] = a[i + 1]
                i += 2
            else:
                out[a[i]] = True
                i += 1
        else:
            i += 1
    return out


def resolve(t):
    env = dict(os.environ, GPU_QUEUE_WORKERS='2')
    r = subprocess.run(['python', '-I', str(Q), 'resolve', t], cwd=str(REPO), env=env, capture_output=True, text=True,
                       check=True)
    return r.stdout.strip().split('\t')


aug2 = flags('tools/train.py ' + ph['{AUG_ARGS}']['AUG2'])
check('AUG2 の定義(protocol choice_placeholders)は 5 つの引数', len(aug2) == 5, aug2)
detail_want = {'--aug-brightness': '0.15', '--aug-contrast': '0.2', '--aug-shift-px': '4', '--aug-tof-noise': '0.02',
               '--aug-top-brightness': '0.15'}
check('AUG2 の定義の値が AUG2_detail の表(±0.15・±0.2・±4 画素・σ 0.02・上 1/3 ±0.15)と同じ', aug2 == detail_want,
      {'AUG2_detail': ph['AUG2_detail']})

st3 = [r for r in proto['runs'] if r['stage'] == 3]
st4 = [r for r in proto['runs'] if r['stage'] == 4]
st2 = {r['seed']: r for r in proto['runs'] if r['stage'] == 2 and r['arm'] == '{x}-e60'}
for r in st3:
    k = r['seed']
    got = resolve(r['name'])
    want_name = r['name'].replace('{x}', x).replace('{E}', str(E))
    want_cmd = r['command'][len(PREFIX):].replace('{X_DATA}', ph['{X_DATA}'][x]).replace('{x}', x).replace('{E}', str(E))
    want_cmd = re.sub(r'--workers \d+', '--workers 2', ' '.join(want_cmd.split()))
    ok_run = got[0] == 'RUN' and got[1] == want_name and got[2] == want_cmd
    f = flags(got[2]) if got[0] == 'RUN' else {}
    f_aug = {a: v for a, v in f.items() if a.startswith('--aug-')}
    # 段 2 の同じ種(拡張なし・60)と比べて、違うのは拡張と --out だけか
    c2 = st2[k]['command'][len(PREFIX):].replace('{X_DATA}', ph['{X_DATA}'][x]).replace('{x}', x).replace('{E}', str(E))
    f2 = flags(re.sub(r'--workers \d+', '--workers 2', c2))
    only = sorted(set(a for a in set(f) | set(f2) if f.get(a) != f2.get(a)))
    check('段 3 の種 %d: 名前 %s・コマンドが protocol の型(x=%s・E=%s・workers 2)と同じ' % (k, want_name, x, E), ok_run,
          {'got': got[:2]})
    check('段 3 の種 %d: 拡張の引数が AUG2 と同じ・反転なし・エポック %s・種 %d' % (k, E, k),
          f_aug == aug2 and not any('flip' in a for a in f) and f.get('--epochs') == str(E) and f.get('--seed') == str(k)
          and f.get('--policy') == 'jetformer_v0', f_aug)
    check('段 3 の種 %d: 段 2 の同じ種(%s-e%s-s%d)との違いは拡張 5 つと --out だけ' % (k, x, E, k),
          only == sorted(list(aug2) + ['--out']), only)
for r in st4:
    got = resolve(r['name'])
    check('段 4 の種 %d: D3 が null なので STOP(名前は未定のまま)' % r['seed'],
          aug is None and got[0] == 'STOP' and 'D3' in got[1], got)

# status の名前
st = subprocess.run(['python', '-I', str(Q), 'status'], cwd=str(REPO), env=dict(os.environ, GPU_QUEUE_WORKERS='2'),
                    capture_output=True, text=True, check=True).stdout
names = [ln.split()[1] for ln in st.splitlines() if ln.split() and ln.split()[0] in ('3', '4')]
want3 = ['v2_20261025-%s-aug2-e%s-s%d' % (x, E, k) for k in range(4)]
check('status の段 3 の名前が決まりどおり(%s-aug2-e%s-s0〜s3)' % (x, E), names[:4] == want3, names[:4])
check('status の段 4 は型のまま(未定)', names[4:] == ['v2_20261025-{x}-msf{A}-e{E}-s0', 'v2_20261025-{x}-msf{A}-e{E}-s1'],
      names[4:])
res['status_stdout'] = st
res['all_ok'] = all(c['ok'] for c in res['checks'])
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
print('all_ok', res['all_ok'])
