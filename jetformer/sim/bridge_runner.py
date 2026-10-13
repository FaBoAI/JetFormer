"""シミュレーターのブリッジを JetFormer の子プロセスとして動かす起動スクリプト

ブリッジ用の venv(data/sim/venv)の Python で、-I -B を付けて動かす。JetFormer のコードは import しない。

    data/sim/venv/bin/python -I -B jetformer/sim/bridge_runner.py \\
        --sim-dir data/sim/releases/<id> --runtime-root data/sim/runtime [--port 0] [--ort-threads 4] [--nice 10]
        [--parent-pid <JetFormer の pid>] [--oom-score-adj 500] [--self-test]

- bridge.BridgeServer を直接作る(main() は使わない: ブラウザを開かず、--policy も使わず、runtime をユーザーのフォルダーに置かない)。
  ブリッジは 127.0.0.1 にだけ bind する(ブリッジのコードで固定)。
- onnxruntime のスレッド数を絞る(ort.InferenceSession を差し替え。シミュレーターのファイルは変えない)。
- 起動すると stdout に {"event":"listening","port":<実ポート>,"pid":<pid>} を 1 行出す。
- SIGTERM・SIGINT で止まる。
- --parent-pid: その親(JetFormer)がいなくなったら(強制終了・手で動かした JetFormer の kill -9 など)自分も止まる。
  systemd の再起動では cgroup ごと止まるが、それ以外で孤児のブリッジがメモリとポートを持ち続けないように(review で追加)。
- --oom-score-adj: メモリが尽きたときに、JetFormer やデスクトップより先にブリッジが止められるようにする(8 GB・スワップなしのため)。
"""

import argparse
import json
import os
import shutil
import signal
import sys
import threading
import time
from pathlib import Path


def limit_ort_threads(threads):
    """ort.InferenceSession を、sess_options がなければスレッド数を絞った SessionOptions を入れるものに差し替える。
    jetformer_policy は関数の中で import onnxruntime as ort → ort.InferenceSession(...) を呼ぶので、ファイルを変えずに効く"""
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    base = ort.InferenceSession
    if getattr(base, '_jetformer_limited', False):
        return True

    class LimitedInferenceSession(base):
        _jetformer_limited = True

        def __init__(self, path_or_bytes, sess_options=None, *args, **kwargs):
            if sess_options is None:
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = int(threads)
                sess_options.inter_op_num_threads = 1
            super().__init__(path_or_bytes, sess_options, *args, **kwargs)

    ort.InferenceSession = LimitedInferenceSession
    return True


def clear_uploads(runtime_root):
    """runtime/uploads/ の中身を消す(アップロードのトークンはメモリにしかなく、再起動後は使えないため。2 GiB の上限で詰まらないように)"""
    d = Path(runtime_root) / 'uploads'
    if d.is_symlink():
        d.unlink()
        return
    if not d.is_dir():
        return
    for p in d.iterdir():
        if p.is_symlink() or p.is_file():
            p.unlink(missing_ok=True)
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


def raise_oom_score(value):
    """/proc/self/oom_score_adj を上げる(上げるのは一般ユーザーでもできる)。できなければ False"""
    try:
        with open('/proc/self/oom_score_adj', 'w') as f:
            f.write(str(int(value)))
        return True
    except (OSError, ValueError):
        return False


def watch_parent(parent_pid, on_gone, interval=1.0):
    """親の pid が変わったら(親が終了して init などに引き取られたら)on_gone() を 1 回呼ぶ見張りのスレッドを始める"""
    def loop():
        while True:
            if os.getppid() != parent_pid:
                try:
                    sys.stderr.write('[runner] JetFormer(pid %d)が終了したのでブリッジを止めます\n' % parent_pid)
                    sys.stderr.flush()
                except (BrokenPipeError, OSError, ValueError):   # 出力先のパイプは親と一緒に閉じている
                    pass
                on_gone()
                return
            time.sleep(interval)
    t = threading.Thread(target=loop, name='parent-watch', daemon=True)
    t.start()
    return t


def emit(event, **info):
    try:
        sys.stdout.write(json.dumps(dict(event=event, **info)) + '\n')
        sys.stdout.flush()
    except (BrokenPipeError, OSError, ValueError):     # 親(JetFormer)がもういないとき
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description='JetFormer: シミュレーターのブリッジを起動する')
    ap.add_argument('--sim-dir', required=True)
    ap.add_argument('--runtime-root', required=True)
    ap.add_argument('--port', type=int, default=0)
    ap.add_argument('--ort-threads', type=int, default=4)
    ap.add_argument('--nice', type=int, default=10)
    ap.add_argument('--parent-pid', type=int, default=0)
    ap.add_argument('--oom-score-adj', type=int, default=0)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args(argv)
    if args.nice:
        try:
            os.nice(args.nice)
        except OSError:
            pass
    sim_dir = Path(args.sim_dir).resolve()
    if not (sim_dir / 'bridge.py').is_file():
        sys.stderr.write('[runner] bridge.py がありません: %s\n' % sim_dir)
        return 2
    sys.path.insert(0, str(sim_dir))           # bridge・dataset_recorder・jetformer_policy・qa・training
    import bridge
    if args.self_test:
        return bridge.main(['--self-test'])
    if not 0 <= args.port <= 65535:
        sys.stderr.write('[runner] --port は 0〜65535\n')
        return 2
    limit_ort_threads(max(1, args.ort_threads))
    runtime = Path(args.runtime_root).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    clear_uploads(runtime)
    server = bridge.BridgeServer(args.port, sim_dir / 'dist', bridge.PolicyRunner(), runtime_root=runtime)
    port = server.server_address[1]

    def on_term(_sig, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, on_term)
    if args.oom_score_adj:
        raise_oom_score(max(-1000, min(1000, args.oom_score_adj)))
    if args.parent_pid > 1:
        watch_parent(args.parent_pid, server.shutdown)
    emit('listening', port=port, pid=os.getpid(), host=server.server_address[0])
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        try:
            clear_uploads(runtime)         # 受け取ったモデルの写し(数十 MB)を止めるときにも消す(トークンはもう使えない。review で追加)
        except OSError:
            pass
        emit('stopped', pid=os.getpid())
    return 0


if __name__ == '__main__':
    sys.exit(main())
