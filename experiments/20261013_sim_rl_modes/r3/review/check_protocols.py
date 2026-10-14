"""第 3〜4 回の審査: 固めた決まり(protocol と追補)の sha256 と、固めた時刻 → 学習 → 選び → 試験 の順を確かめる(CPU だけ・読むだけ)

    python3 -I check_protocols.py > ~/jetformer_reports/sim_rl_modes_20261013/r3/review/protocols.json

確かめること:
  - 各 .sha256 に書かれた sha256 = 今のファイルの sha256(protocol.json・r2 の追補・r3/r4 の追補・立て直せる表)
  - protocol.json の files の sha256(ユーザーのアプリ・元の sim-core.js・ベストラップの方策など)= 今
  - 固めた時刻(.sha256 のファイルの時刻)< 学習の始まり(run.json の started_at)< 選び(selection.json)< 試験(eval_test の時刻)
  - 試験の結果の JSON が、どの追補の sha256 で走ったかを書いていて、それが今の sha256 と同じ
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

REP = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013')
R3 = REP / 'r3'
EXP = Path('/home/jetson/jetformer/experiments/20261013_sim_rl_modes')
HOME = Path('/home/jetson')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def mtime(p):
    return dt.datetime.fromtimestamp(os.stat(p).st_mtime).isoformat(timespec='seconds')


def iso_local(s):
    """'2026-10-14T14:28:03+00:00' や 'Z' つきは日本時間に、時差の無いものはそのまま"""
    if s is None:
        return None
    s = s.replace('Z', '+00:00')
    t = dt.datetime.fromisoformat(s)
    if t.tzinfo is not None:
        t = t.astimezone(dt.timezone(dt.timedelta(hours=9))).replace(tzinfo=None)
    return t.isoformat(timespec='seconds')


def main():
    out = {'schema': 'jf-sim-rl-r3-review-protocols/v1', 'created_at': dt.datetime.now().isoformat(timespec='seconds'), 'sha_files': [], 'protocol_files': []}
    # 1. .sha256 のファイル
    shafiles = [REP / 'protocol.sha256'] + sorted(REP.glob('protocol_*.sha256')) + sorted(R3.glob('protocol_*.sha256'))
    for sf in shafiles:
        for line in sf.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            want, rest = line.split(None, 1)
            name = rest.split('(')[0].strip()
            # 表の行は「名前(実際の場所)」の形
            if '(' in rest and 'experiments/' in rest:
                loc = rest.split('(', 1)[1].split('。')[0].rstrip(')')
                p = HOME / 'jetformer' / loc
            else:
                p = sf.parent / name
            now = sha(p) if p.exists() else None
            out['sha_files'].append({'sha_file': str(sf.relative_to(REP)), 'file': str(p), 'frozen_sha256': want, 'now_sha256': now,
                                     'same': now == want, 'sha_file_mtime': mtime(sf), 'file_mtime': mtime(p) if p.exists() else None})
    # 2. protocol.json の files
    proto = json.loads((REP / 'protocol.json').read_text(encoding='utf-8'))
    files = proto.get('files') or {}
    items = files.items() if isinstance(files, dict) else [(f.get('path'), f.get('sha256')) for f in files]
    for k, v in items:
        if isinstance(v, dict):
            path, want = v.get('path', k), v.get('sha256')
        else:
            path, want = k, v
        p = Path(path)
        if not p.is_absolute():
            for base in (REP, HOME / 'jetformer', EXP, HOME):
                if (base / path).exists():
                    p = base / path
                    break
        now = sha(p) if p.exists() else None
        out['protocol_files'].append({'path': str(p), 'frozen_sha256': want, 'now_sha256': now, 'same': now == want})
    # 3. 順番(固めた → 学習 → 選び → 試験)
    def runj(p):
        return json.loads(Path(p).read_text(encoding='utf-8'))
    tl = []
    nw_test = runj(R3 / 'no_wall_r4/test/eval_test.json')
    tl.append({'mode': 'no_wall', 'addenda': {'r4': mtime(R3 / 'protocol_r4.sha256'), 'r4b': mtime(R3 / 'protocol_r4.sha256'), 'r4c': mtime(R3 / 'protocol_r4c.sha256')},
               'note': 'protocol_r4.sha256 は r4b を足したときに書き直された(r4 本体の sha256 ad90c926… は試験の JSON と同じ)。r4 本体の時刻は protocol_no_wall_r4.json の mtime',
               'protocol_no_wall_r4_mtime': mtime(R3 / 'protocol_no_wall_r4.json'),
               'protocol_no_wall_r4b_mtime': mtime(R3 / 'protocol_no_wall_r4b.json'),
               'protocol_no_wall_r4c_mtime': mtime(R3 / 'protocol_no_wall_r4c.json'),
               'train_starts': {d: runj(R3 / 'no_wall_r4' / d / 'run.json').get('started_at') for d in ('train_a', 'train', 'train_c')},
               'selection_mtime': mtime(R3 / 'no_wall_r4/selection.json'),
               'test_policy_file_mtime': mtime(R3 / 'no_wall_r4/test/TEST_POLICY'),
               'test_created_at': iso_local(nw_test.get('created_at')),
               'test_protocol_sha256': nw_test.get('protocol_no_wall_r4_sha256'),
               'test_policy_sha256': nw_test['policy']['sha256']})
    rc_test = runj(R3 / 'rc4/eval_test.json')
    tl.append({'mode': 'recover', 'protocol_recover_r4_sha_mtime': mtime(R3 / 'protocol_recover_r4.sha256'),
               'table_sha_mtime': mtime(R3 / 'protocol_recover_r4_table.sha256'),
               'train_start': runj(R3 / 'rc4/train/main/run.json').get('started_at'),
               'selection_mtime': mtime(R3 / 'rc4/train/main/selection.json'),
               'test_created_at': iso_local(rc_test.get('created_at')),
               'test_protocol_sha256': rc_test.get('protocol_recover_r4_sha256'),
               'test_table_sha256': rc_test.get('recoverable_table_sha256'),
               'test_policy_sha256': rc_test['policy']['sha256']})
    ln_test = runj(R3 / 'line_r4b/eval_test_r4b.json')
    ln_sel = runj(R3 / 'line_r4b/selection.json')
    tl.append({'mode': 'line', 'protocol_line_r4_sha_mtime': mtime(R3 / 'protocol_line_r4.sha256'),
               'protocol_line_r4b_sha_mtime': mtime(R3 / 'protocol_line_r4b.sha256'),
               'train_start': runj(R3 / 'line_r4b/train/main/run.json').get('started_at'),
               'selection_created_at': ln_sel.get('created_at'),
               'test_created_at': iso_local(ln_test.get('created_at')),
               'test_protocol_sha256': {'r4': ln_test.get('protocol_line_r4_sha256'), 'r4b': ln_test.get('protocol_line_r4b_sha256'),
                                        'r2': ln_test.get('protocol_line_r2_sha256')},
               'test_policy_sha256': ln_test['policy']['sha256']})
    out['timeline'] = tl
    out['all_sha_same'] = all(x['same'] for x in out['sha_files']) and all(x['same'] for x in out['protocol_files'])
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
