#!/usr/bin/env python3
"""B2 の学習の段の道具(GPU の段で使う。準備の段では plan・status・--dry-run だけ動かした)

    cd ~/jetformer
    python experiments/20261016_b2_curation/train_b2.py plan          # コマンドを出すだけ
    python experiments/20261016_b2_curation/train_b2.py status        # 各学習の状態(fresh / resume / done)
    nohup setsid /home/jetson/miniconda/envs/jetracer/bin/python -u experiments/20261016_b2_curation/train_b2.py run 0 1 \\
        >> /home/jetson/jetformer_reports/b2_curation_20261016/logs/train_driver.log 2>&1 < /dev/null &
    python experiments/20261016_b2_curation/train_b2.py run 0 --dry-run   # 何をするかを出すだけ(GPU を使わない)

protocol.json の model.train_cmd・resume_cmd の文字列をそのまま bash で動かす(GPU は jf-heavy --lock big で 1 つずつ)。
  - 始める前: protocol.json の sha256 が protocol.sha256 と同じ・空きディスク ≥ 8 GB・管理画面の学習/走行/収集/取り込みと
    ほかの GPU の処理が動いていない(動いていれば待つ)・run.json を書く(protocol の model.run_json の欄)
  - 学習の最初の数行(衝突の除外・drop・サンプル数・ジャンル別)が protocol の expected_train_log と違えば、止めて deviations.md
  - 電源断のあと: 同じコマンド(run)をもう一度。last.pth があれば resume_cmd(--resume)で続きから。済んだものは飛ばす
  - メモリの上限(3G)で止まったら(SIGKILL)、protocol の memory.on_cgroup_kill のとおり上限を 3872M にして --resume(deviations.md)
  - 終わったら result.json を確かめる(サンプル数・ジャンル別・衝突の除外・drop・表の sha1)
車は動かさない・PCA9685・I2C に触らない・drive.py と走行の既定は変えない・jetformer-web は再起動しない。
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_b2 as P  # noqa: E402

REPO = P.REPO
RUNS = REPO / 'data' / 'runs'
DEVIATIONS = P.REPORT / 'deviations.md'
STATE = P.REPORT / 'logs' / 'train_state.json'
MEM_DEFAULT, MEM_RAISED = '3G', '3872M'
KEEP_FREE_GB = 8.0
GPU_PATTERNS = ('tools/train.py', 'build_engine.py', 'policy_zoo.py', 'engine_check.py', 'engine_load_check.py',
                'rl_evaluate.py', 'compare_models.py', 'tools/evaluate.py', 'sim_cotrain_eval', 'trtexec', 'realacc_eval.py',
                'b2_eval.py predict', 'b2_eval.py crosscheck', 'b2_eval.py engine')
TABLES = {'drop_mask': 'data/derived/b2_curation_20261016/drop_b2.npz',
          'mode_remap_npz': 'data/derived/b2_curation_20261016/remap_b2.npz'}


def log(msg):
    print('[%s] %s' % (P.now_jst(), msg), flush=True)


# ---------------------------------------------------------------- 学習の一覧(protocol から)

def targets(protocol):
    """{名前: {'run', 'cmd', 'resume', 'seed', 'check'}}。B2 の種 0・1 は protocol の model、任意の腕は optional_arms"""
    m = protocol['model']
    out = {}
    for s in ('0', '1'):
        out[s] = {'run': m['runs'][s], 'cmd': m['train_cmd'][s], 'resume': m['resume_cmd'][s], 'seed': int(s), 'check': True,
                  'arm': 'B2'}
    for k, arm in protocol.get('optional_arms', {}).items():
        if isinstance(arm, dict) and 'cmd' in arm:
            out[k] = {'run': arm['run'], 'cmd': arm['cmd'], 'resume': arm['cmd'] + ' --resume', 'seed': 0, 'check': False,
                      'arm': k}
    return out


def with_mem(cmd, mem):
    """jf-heavy の --mem だけ変える"""
    new, n = re.subn(r'(jf-heavy --mem )\S+', r'\g<1>%s' % mem, cmd, count=1)
    if n != 1:
        raise ValueError('コマンドに jf-heavy --mem がありません')
    return new


def stage(run_dir, epochs=30):
    """'done'(result.json が done で epochs_done = epochs)・'resume'(last.pth がある)・'fresh'"""
    run_dir = Path(run_dir)
    r = run_dir / 'result.json'
    if r.exists():
        try:
            res = json.loads(r.read_text(encoding='utf-8'))
            if res.get('state') == 'done' and res.get('epochs_done') == epochs:
                return 'done'
        except ValueError:
            pass
    if (run_dir / 'last.pth').exists():
        return 'resume'
    return 'fresh'


def run_json(protocol, t, created_at):
    """protocol の model.run_json の欄"""
    sha = {k: P.sha256(REPO / v) for k, v in TABLES.items()}
    return {'dataset': 'clean_driving_20261011', 'stride': 3, 'epochs': 30, 'batch': 64, 'lr': 0.0003, 'pretrained': True,
            'build_engine': False, 'command_input': False, 'mask_impacts': True, 'policy': 'jetformer_v0',
            'policy_cfg': {'dim': 256, 'layers': 4, 'heads': 8}, 'impact_g': 2.5, 'seed': t['seed'], 'name': t['run'],
            'created_at': created_at,
            'note': ('B2: 選び直したデータ(R1 の速い進入の手前・R1 で当たった通過を外し、152405・152707 をジャンル 4 に)。'
                     '机上の G3 に合格するまで走らせない') if t['arm'] == 'B2' else
                    ('B2 の参考の腕 %s(判定しない)。走らせない' % t['arm']),
            'experiment': 'b2_curation_20261016', 'arm': t['arm'],
            'drop_mask': TABLES['drop_mask'] if '--drop-mask' in t['cmd'] else None,
            'mode_remap_npz': TABLES['mode_remap_npz'] if '--mode-remap-npz' in t['cmd'] else None,
            'tables_sha256': {k: v for k, v in sha.items() if ('--drop-mask' if k == 'drop_mask' else '--mode-remap-npz') in t['cmd']},
            'protocol_sha256': P.sha256(P.PROTOCOL),
            'train_env': {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'MALLOC_ARENA_MAX': '2'}}


def check_result(res, protocol, t):
    """終わった学習の result.json を確かめる → 違い [文字列](空なら合格)"""
    exp = protocol['model']['expected_train_log']
    diffs = []
    if res.get('state') != 'done' or res.get('epochs_done') != 30:
        diffs.append('state %s・epochs_done %s' % (res.get('state'), res.get('epochs_done')))
    if not t['check']:
        return diffs
    want = {'train_samples': exp['train_samples'], 'val_samples': exp['val_samples'],
            'impact_masked_frames': exp['impact_masked_printed'], 'train_mode_counts': exp['genre_counts_train'],
            'seed': t['seed'], 'impact_g': 2.5, 'command_input': False, 'policy': 'jetformer_v0', 'stride': 3}
    for k, w in want.items():
        got = res.get(k)
        if k == 'train_mode_counts':
            got = {str(a): b for a, b in (got or {}).items()}
            w = {str(a): b for a, b in w.items()}
        if got != w:
            diffs.append('%s: %s(protocol %s)' % (k, got, w))
    rl = res.get('rl_options') or {}
    if rl.get('dropped_frames') != exp['dropped_printed']:
        diffs.append('dropped_frames: %s' % rl.get('dropped_frames'))
    want_sha1 = {k: hashlib.sha1((REPO / v).read_bytes()).hexdigest()[:12] for k, v in TABLES.items()}
    if rl.get('sha1') != want_sha1:
        diffs.append('表の sha1: %s(今のファイル %s)' % (rl.get('sha1'), want_sha1))
    return diffs


# ---------------------------------------------------------------- 待つ・止める

def busy():
    """GPU を使いそうなほかのプロセスと、管理画面の学習・走行・収集・取り込み"""
    me = {os.getpid(), os.getppid()}
    out = subprocess.run(['ps', '-eo', 'pid,ppid,args'], capture_output=True, text=True).stdout.splitlines()[1:]
    why = []
    for line in out:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid, args = int(parts[0]), int(parts[1]), parts[2]
        if pid in me or ppid in me or 'train_b2.py' in args or 'grep' in args:
            continue
        if ('python' in args or 'trtexec' in args) and any(p in args for p in GPU_PATTERNS):
            why.append(args[:160])
    try:
        with urllib.request.urlopen('http://127.0.0.1:8080/api/status', timeout=3) as r:
            st = json.load(r)
        for k, v in (('training', st.get('training')), ('driving', st.get('driving')),
                     ('recording', (st.get('recorder') or {}).get('recording')),
                     ('session', (st.get('session') or {}).get('active'))):
            if v:
                why.append('管理画面: %s' % k)
        with urllib.request.urlopen('http://127.0.0.1:8080/api/jobs', timeout=3) as r:
            jobs = json.load(r)
        if any(j.get('state') in ('running', 'queued') for j in jobs):
            why.append('管理画面: 処理(jobs)が動いている')
    except Exception:          # noqa: BLE001  管理画面が止まっていれば ps の確認だけ
        pass
    return why


def disk_free_gb():
    st = os.statvfs(str(REPO))
    return st.f_bavail * st.f_frsize / 1e9


def add_deviation(text):
    """deviations.md に 1 項目足す(時刻つき)"""
    s = DEVIATIONS.read_text(encoding='utf-8')
    s = s.replace('(まだありません)\n', '')
    if not s.endswith('\n'):
        s += '\n'
    s += '\n- %s(train_b2.py): %s\n' % (P.now_jst(), text)
    P._write_text(DEVIATIONS, s)


def save_state(state):
    P.save_json(STATE, state)


def _watch(cmd, log_path, check, protocol):
    """bash -c cmd(出力は train.log に足す)。最初の数行を確かめ、違えば止める → (rc, 違い, 自分で止めたか)"""
    names = P.mode_names_from_card(P.config_dataset_root())
    exp = protocol['model']['expected_train_log']
    start = log_path.stat().st_size if log_path.exists() else 0
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('==== %s 開始(%s)====\n' % (P.now_jst(), cmd))
        f.flush()
        p = subprocess.Popen(['bash', '-c', cmd], cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    checked, diffs, stopped = not check, [], False
    while p.poll() is None:
        time.sleep(5)
        if checked:
            continue
        with open(log_path, encoding='utf-8', errors='replace') as f:
            f.seek(start)
            text = f.read()
        parsed = P.parse_train_log(text)
        if 'genres' in parsed or re.search(r'^epoch +\d+ ', text, re.M):
            ok, diffs = P.compare_train_log(parsed, exp, names, full=True)
            checked = True
            if not ok:
                os.killpg(p.pid, signal.SIGTERM)          # train.py は SIGTERM で最良を書いて止まる(ここでは最初なので何も無い)
                time.sleep(20)
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGKILL)
                stopped = True
    rc = p.wait()
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('==== %s 終了 rc=%s ====\n' % (P.now_jst(), rc))
    if not checked and check:
        with open(log_path, encoding='utf-8', errors='replace') as f:
            f.seek(start)
            parsed = P.parse_train_log(f.read())
        if parsed:
            ok, diffs = P.compare_train_log(parsed, exp, names, full=True)
    return rc, diffs, stopped


def train(name, protocol, dry_run=False):
    t = targets(protocol)[name]
    d = RUNS / t['run']
    st = stage(d)
    if st == 'done':
        log('%s(%s): 済み' % (name, t['run']))
        return True
    mem = MEM_DEFAULT
    state = {}
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding='utf-8'))
    mem = state.get(name, {}).get('mem', MEM_DEFAULT)
    for attempt in range(4):
        st = stage(d)
        cmd = with_mem(t['resume'] if st == 'resume' else t['cmd'], mem)
        if dry_run:
            log('%s: %s → %s' % (name, st, cmd))
            return True
        if disk_free_gb() < KEEP_FREE_GB:
            log('空きディスク %.1f GB < %.0f GB のため止めます' % (disk_free_gb(), KEEP_FREE_GB))
            return False
        said = None
        while True:
            why = busy()
            if not why:
                break
            if why != said:
                log('待ちます: %s' % why)
                said = why
            time.sleep(30)
        d.mkdir(parents=True, exist_ok=True)
        if not (d / 'run.json').exists():
            P.save_json(d / 'run.json', run_json(protocol, t, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        log('%s: %s(上限 %s)を始めます' % (name, st, mem))
        state.setdefault(name, {}).update({'mem': mem, 'last_start': P.now_jst(), 'stage': st})
        save_state(state)
        rc, diffs, stopped = _watch(cmd, d / 'train.log', t['check'], protocol)
        if stopped or diffs:
            add_deviation('%s の学習の最初の数行が protocol の expected_train_log と違ったので止めました: %s。判定には進みません'
                          % (t['run'], '; '.join(diffs)))
            log('%s: 最初の数行が違うので止めました %s' % (name, diffs))
            return False
        if rc == 0 and stage(d) == 'done':
            res = json.loads((d / 'result.json').read_text(encoding='utf-8'))
            bad = check_result(res, protocol, t)
            state[name].update({'done': P.now_jst(), 'result_check': bad or 'OK', 'best_epoch': res.get('best_epoch'),
                                'best_val_us': res.get('best_val_us'), 'duration_s': res.get('duration_s')})
            save_state(state)
            if bad:
                add_deviation('%s の result.json が protocol と違います: %s' % (t['run'], '; '.join(bad)))
                log('%s: result.json が違います %s' % (name, bad))
                return False
            log('%s: 済み(最良 %s エポック目・%s µs)' % (name, res.get('best_epoch'), res.get('best_val_us')))
            return True
        if rc in (137, -9) and mem == MEM_DEFAULT and (d / 'last.pth').exists():
            mem = MEM_RAISED
            add_deviation('%s: メモリの上限 %s で止まったので、protocol の memory.on_cgroup_kill のとおり上限を %s にして続きから'
                          '(--resume。計算は変わらない)' % (t['run'], MEM_DEFAULT, MEM_RAISED))
            state[name]['mem'] = mem
            save_state(state)
            continue
        if rc in (137, -9) and mem == MEM_DEFAULT:
            mem = MEM_RAISED                       # 1 エポック目の前に止まった(10/15 の v0-s2 と同じ)→ 上げて最初から
            add_deviation('%s: 1 エポック目の前にメモリの上限 %s で止まったので、上限を %s にして最初から(計算は変わらない)'
                          % (t['run'], MEM_DEFAULT, MEM_RAISED))
            state[name]['mem'] = mem
            save_state(state)
            continue
        log('%s: rc=%s で止まりました(train.log を見てください)' % (name, rc))
        return False
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('plan')
    sub.add_parser('status')
    r = sub.add_parser('run')
    r.add_argument('names', nargs='+', help='0・1(B2 の種)・drop_only・relabel_only(任意の腕)')
    r.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    protocol = P.load_protocol()
    ts = targets(protocol)
    if args.cmd == 'plan':
        for k, t in ts.items():
            print('%s(%s)\n  学習: %s\n  再開: %s' % (k, t['run'], t['cmd'], t['resume']))
        return
    if args.cmd == 'status':
        for k, t in ts.items():
            print('%s(%s): %s' % (k, t['run'], stage(RUNS / t['run'])))
        print('ほかの処理: %s' % (busy() or 'なし'))
        print('空きディスク: %.1f GB' % disk_free_gb())
        return
    line = [ln for ln in P.PROTOCOL_SHA.read_text(encoding='utf-8').splitlines() if ln.endswith('  protocol.json')]
    if not line or line[0].split()[0] != P.sha256(P.PROTOCOL):
        raise SystemExit('protocol.json の sha256 が protocol.sha256 と違います(固定の後に変わった)。止めます')
    for n in args.names:
        if n not in ts:
            raise SystemExit('知らない名前: %s(%s)' % (n, ', '.join(ts)))
        if not train(n, protocol, dry_run=args.dry_run):
            sys.exit(1)


if __name__ == '__main__':
    main()
