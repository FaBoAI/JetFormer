#!/usr/bin/env python3
"""別の環境(シミュレーター・エミュレーターの Mac など)で作ったデータセットを、Jetson の JetFormer にまとめて送る

JetFormer のリポジトリも追加のパッケージも要らない(Python 3.8 以上の標準ライブラリだけ)。

    python3 upload_datasets.py --host http://192.168.55.1:8080 sim_run_01/ sim_run_02/ more.tar
    python3 upload_datasets.py --host http://jetson.local:8080 --real exported.tar      # 実機のデータとして
    python3 upload_datasets.py --host http://192.168.55.1:8080 --check sim_run_01/      # 送らずに中身だけ確かめる

送れるもの:
  - データセットのフォルダ(中に dataset_card.json・splits.json・aligned/*.npz・samples/ がある)。その場で tar にして送る
  - JetFormer の「エクスポート」と同じ形の .tar / .tar.gz / .tgz、または .zip(1ファイル = 1データセット)
既定ではシミュレーターのデータとして取り込む(simulator のタグが付く)。--real で実機のデータとして取り込む。
同じ名前のデータセットが Jetson にあるものは取り込まれない(Jetson 側で先に削除する)。
"""

import argparse
import http.client
import io
import json
import os
import sys
import tarfile
import time
import urllib.parse
import urllib.request

JUNK = ('.DS_Store', 'Thumbs.db')


class _Pipe(io.RawIOBase):
    """tarfile が書いたバイトを溜めて、送る側が少しずつ取り出す"""

    def __init__(self):
        self.buf = bytearray()

    def writable(self):
        return True

    def write(self, b):
        self.buf += b
        return len(b)


def folder_chunks(folder, chunk=1 << 20):
    """データセットのフォルダを tar(先頭のフォルダ = フォルダ名)にしながら少しずつ返す"""
    name = os.path.basename(os.path.normpath(folder))
    pipe = _Pipe()
    with tarfile.open(fileobj=pipe, mode='w|', format=tarfile.PAX_FORMAT) as tar:
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames[:] = sorted(d for d in dirnames if d != '__MACOSX' and not d.startswith('._'))
            for fn in sorted(filenames):
                if fn in JUNK or fn.startswith('._'):
                    continue
                full = os.path.join(dirpath, fn)
                if os.path.islink(full):
                    continue
                tar.add(full, arcname=os.path.join(name, os.path.relpath(full, folder)), recursive=False)
                if len(pipe.buf) >= chunk:
                    yield bytes(pipe.buf)
                    pipe.buf.clear()
    if pipe.buf:
        yield bytes(pipe.buf)


def file_chunks(path, chunk=1 << 20):
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            yield block


def check_folder(folder):
    """送る前の簡単な確認(Jetson でも取り込むときにもっと詳しく確かめる)"""
    probs = []
    for need in ('dataset_card.json', 'splits.json'):
        if not os.path.exists(os.path.join(folder, need)):
            probs.append('%s がありません' % need)
    try:
        with open(os.path.join(folder, 'splits.json'), encoding='utf-8') as f:
            sp = json.load(f)
        for s in list(sp.get('train', [])) + list(sp.get('val', [])):
            if not os.path.exists(os.path.join(folder, 'aligned', s + '.npz')):
                probs.append('aligned/%s.npz がありません' % s)
    except (OSError, ValueError):
        pass
    return probs


def upload(host, item, simulator, timeout=3600):
    """1つのデータセットを送る。戻り値 Jetson の応答(処理状況のジョブ)"""
    u = urllib.parse.urlsplit(host)
    is_dir = os.path.isdir(item)
    name = os.path.basename(os.path.normpath(item)) + ('.tar' if is_dir else '')
    q = urllib.parse.urlencode({'filename': name, 'simulator': '1' if simulator else '0'})
    conn = (http.client.HTTPSConnection if u.scheme == 'https' else http.client.HTTPConnection)(
        u.hostname, u.port or (443 if u.scheme == 'https' else 80), timeout=timeout)
    headers = {'Content-Type': 'application/octet-stream'}
    total = None if is_dir else os.path.getsize(item)
    sent = [0]
    t0 = time.time()

    def counted(chunks):
        for c in chunks:
            sent[0] += len(c)
            mb = sent[0] / 1e6
            pct = '' if total is None else ' (%d%%)' % (100 * sent[0] / max(total, 1))
            sys.stdout.write('\r  送信 %.1f MB%s  %.1f MB/s   ' % (mb, pct, mb / max(time.time() - t0, 1e-3)))
            sys.stdout.flush()
            yield c
    body = counted(folder_chunks(item) if is_dir else file_chunks(item))
    if total is not None:
        headers['Content-Length'] = str(total)
        conn.request('PUT', '/api/datasets/import?' + q, body=body, headers=headers)
    else:
        conn.request('PUT', '/api/datasets/import?' + q, body=body, headers=headers, encode_chunked=True)
    r = conn.getresponse()
    data = r.read().decode('utf-8', 'replace')
    print()
    if r.status >= 300:
        raise RuntimeError('Jetson がエラーを返しました(HTTP %d): %s' % (r.status, data[:400]))
    return json.loads(data)


def wait_job(host, job_id, timeout=3600):
    """取り込み(展開・確認)の終わりを待つ"""
    t_end = time.time() + timeout
    last = ''
    while time.time() < t_end:
        with urllib.request.urlopen(host.rstrip('/') + '/api/jobs', timeout=30) as r:
            jobs = json.load(r)
        j = next((x for x in jobs if x.get('id') == job_id), None)
        if j is None:
            return None
        msg = '  取り込み: %s' % j.get('message', '')
        if msg != last:
            print(msg)
            last = msg
        if j.get('state') != 'running':
            return j
        time.sleep(1.0)
    return None


def main():
    ap = argparse.ArgumentParser(description='データセットを Jetson の JetFormer にまとめて送る')
    ap.add_argument('items', nargs='+', help='データセットのフォルダ、または .tar / .tar.gz / .tgz / .zip')
    ap.add_argument('--host', default='http://192.168.55.1:8080', help='JetFormer の管理画面の URL(既定は USB 接続の Jetson)')
    ap.add_argument('--real', action='store_true', help='実機のデータとして取り込む(既定はシミュレーター)')
    ap.add_argument('--check', action='store_true', help='送らずに、フォルダの中身だけ確かめる')
    args = ap.parse_args()
    failed = 0
    for item in args.items:
        print('== %s' % item)
        if not os.path.exists(item):
            print('  ありません'); failed += 1
            continue
        if os.path.isdir(item):
            probs = check_folder(item)
            if probs:
                print('  足りないもの: ' + ' / '.join(probs[:6])); failed += 1
                continue
        if args.check:
            print('  確認だけ: 問題なし')
            continue
        try:
            job = upload(args.host, item, simulator=not args.real)
            j = wait_job(args.host, job['id'])
            if not j or j.get('state') != 'done':
                print('  失敗: %s' % ((j or {}).get('error') or '状況が分かりません')); failed += 1
            else:
                res = j.get('result') or {}
                print('  取り込みました: %s(タグ %s)' % (res.get('name'), ', '.join(res.get('tags') or [])))
                for w in res.get('warnings') or []:
                    print('  注意: %s' % w)
        except Exception as e:                                          # noqa: BLE001
            print('  失敗: %s' % e); failed += 1
    print('完了: %d 件中 %d 件が失敗' % (len(args.items), failed))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
