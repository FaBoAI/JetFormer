#!/usr/bin/env python3
"""第 2 の固定 protocol_v2_lock2.json を書く(2026-10-25 夜)。numpy も torch も使わない。data/ は変えない(読むだけ)。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I \
        /home/jetson/jetformer/experiments/20261025_train_v2/write_lock2_v2.py

protocol_v2.json の lock.second_lock・stage0_checks のとおり、sha256 と段 0 の 8 つの答えを書く。決まりは変えない(数を埋めるだけ)。
書いたあと chmod a-w し、protocol_v2_lock2.sha256 を書く(これも chmod a-w)。もう lock2 があれば止まる(1 回だけ)。
"""
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/jetson/jetformer')
HERE = ROOT / 'experiments' / '20261025_train_v2'
REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
LOCK2 = REP / 'protocol_v2_lock2.json'
LOCK2_SHA = REP / 'protocol_v2_lock2.sha256'
DS = ROOT / 'data' / 'datasets' / 'dataset_v2_20261025'
TB = ROOT / 'data' / 'derived' / 'dataset_v2_20261025'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def entry(p):
    p = Path(p)
    if not p.exists():
        return {'path': str(p), 'missing': True}
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def main():
    if LOCK2.exists():
        raise SystemExit('lock2 はもうあります(1 回だけ): %s' % LOCK2)
    r = subprocess.run(['sha256sum', '-c', 'protocol_v2.sha256'], cwd=str(REP), capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit('protocol の固定が合いません:\n' + r.stdout + r.stderr)
    r2 = subprocess.run([sys.executable, '-I', str(HERE / 'check_seal_v2.py')], capture_output=True, text=True)
    if r2.returncode != 0 or 'すべて同じ' not in r2.stdout:
        raise SystemExit('封の照らし合わせが合いません:\n' + r2.stdout + r2.stderr)
    dc = load(REP / 'data_checks.json')
    tc = load(TB / 'tables_card.json')
    xb = load(REP / 'eval' / 'crosscheck_B.json')
    rc = load(REP / 'recount_stage0_check3.json')

    tables = {n: entry(TB / n) for n in ('drop_v2h.npz', 'drop_v2m.npz', 'weights_v2m.npz', 'remap_v2m.npz',
                                         'drop_bfast.npz', 'weights_bfast.npz', 'tables_card.json')}
    # 表の 6 つは data_checks.json(学習の前)・tables_card.json と同じか
    for n, e in tables.items():
        if n == 'tables_card.json':
            continue
        if e.get('sha256') != dc['checks']['tables_sha256_match_card']['sha256'][n] or e.get('sha256') != tc['files'][n]['sha256']:
            raise SystemExit('表 %s の sha256 が学習の前と違います' % n)
    card = entry(DS / 'dataset_card.json')
    modes = entry(ROOT / 'data' / 'modes.json')
    if card['sha256'] != dc['dataset_card_sha256'] or modes['sha256'] != dc['modes_json_sha256']:
        raise SystemExit('dataset_card.json か data/modes.json が学習の前と違います')

    out = {
        'schema': 'jf-train-v2-lock2/v1',
        'title': '学習 v2 と B-FAST の第 2 の固定(lock2)。表・C3・C4・T1・数え直しの台本・data/modes.json の sha256 と、段 0 の 8 つの答え',
        'written_at': time.strftime('%Y-%m-%dT%H:%M:%S+09:00'),
        'written_by': 'workflow のサブエージェント(予測と判定の係)。車・装置に触れていない。git commit していない',
        'rule': 'protocol_v2.json lock.second_lock: 中身は stage0_checks の決まりで出す。lock2 で決まりを変えない(数を埋めるだけ)',
        'protocol_v2_sha256': sha(REP / 'protocol_v2.json'),
        'protocol_check': {'sha256sum -c protocol_v2.sha256': 'OK(6 つ)', 'check_seal_v2.py': 'すべて同じ(固定のファイル・封の 14 本)'},
        'when': {
            'protocol': '段 0 のあと・最初の v2 と B-FAST の予測の前(10/25 夜)',
            'actual': '10/25 夜。学習 16 本(B-FAST 3・段 1 の 8・段 2 の 5)のあと、v2・B-FAST の机上の予測の前',
            'note': '学習のログ(H1 の検証の L1・best のエポック)は、キューの一覧で先に見えていた。机上の予測(H1・介入の 10 行・best と last)はまだ 1 本もない。deviations.md の 6',
        },
        'files': {
            'dataset_card': card,
            'dataset_splits(足した)': entry(DS / 'splits.json'),
            'tables_7': tables,
            'data_modes_json': modes,
            'C3(机上のベンチ v2)': {'missing': True, 'why': 'C3 の台本と desk/desk_v2.json がまだ無い(experiments/20261025_train_v2 にも、ほかの場所にも無い)。'
                                   'C3 が出す欄(sanity・低速のブレーキ・ずれの試し・AI の場面・直線・止まった場面・G-b・G-c・G-f・同じ日の推論)を使う門と D3 は「未」のまま'},
            'C4(キュー v2)': {n: entry(HERE / n) for n in ('gpu_queue_v2.sh', 'gpu_queue_v2.py', 'gpu_queue_v2.tsv', 'eval_v2.py',
                                                           'test_eval_v2.py')},
            'C4 の動いている写し': entry(HERE / 'logs' / '.gpu_queue_v2.run.py'),
            'T1(表)': entry(HERE / 'make_v2_tables.py'),
            '数え直しの台本': {n: entry(HERE / n) for n in ('protocol_numbers.py', 'check_v2_data.py', 'check_seal_v2.py',
                                                         'torch_dataset_check.py', 'recount_stage12_v2.py')},
            '予測の手順(eval_v2.py predict が使う 10/21 の台本)': entry(ROOT / 'experiments' / '20261021_bench_prep' / 'desk' / 'desk_gpu.py'),
            'データを作った台本': {n: entry(HERE / n) for n in ('build_v2_dataset.py', 'add_genre9.py')},
            'この台本': entry(Path(__file__)),
        },
        'stage0': {
            '1': {'what': 'B を予測し直し、10/12 の保存と同じか(差の最大 ≤ 0.01 µs)', 'status': '済・合格',
                  'answer': {'best_max_abs_us': xb['best']['max_abs_us'], 'last_max_abs_us': xb['last']['max_abs_us'],
                             'same_genre': xb['best']['same_genre'] and xb['last']['same_genre'], 'checked_at': xb['checked_at']},
                  'src': entry(REP / 'eval' / 'crosscheck_B.json')},
            '2': {'what': '10/24 2 回目の AI のフレームで、B の予測(PyTorch)と記録の model_out の差の中央値(> 10 µs なら止めて調べる)',
                  'status': '未', 'why': 'C3 がまだ無い。10/24 2 回目の AI のフレームの予測の表を作る台本が無い。H1・介入の判定(D1・D2・拒否権)は 10/10・10/11 のフレームなので、この確かめに依らない。AI の場面・G-b の集合を使う門は C3 のあと'},
            '3': {'what': '印「ジャンルの片寄り」の閾値を同じに出すか(差 ≤ 0.01 µs)', 'status': '済・合格(C3 の代わりに、判定に使う 2 つの台本で)',
                  'answer': rc},
            '4': {'what': '止まった場面のフレーム数(300 以上なら門)。B と 2 本の平均の standstill_go_frac', 'status': '未',
                  'why': 'C3 がまだ無い(sets.standstill を数える台本が無い)。門「止まった場面」・G-d は「未」'},
            '5': {'what': 'データセットと表の確かめ(tables.expected_after_build)', 'status': '済(外れ 2 つは main session が了承)',
                  'answer': {'fails': dc['fails'], 'accepted': ['train_samples_v2m_within_1pct', 'bfast_teacher_within_1pct'],
                             'accepted_how': 'キューの起動の GPU_QUEUE_ACCEPT_DATA_FAILS(logs/gpu_queue_v2.log の 10 行目)。deviations.md の 3',
                             'samples': {'v2h': 6506, 'v2m': 7638, 'bfast': 6697, 'val': 1559}},
                  'src': entry(REP / 'data_checks.json')},
            '6': {'what': '下見 20 ステップ(v2-M 軽量型・多段型・AUG2・B-FAST の --init)。メモリの最大が 3.5 GB を超えたら workers 2',
                  'status': '一部済', 'answer': {
                      'workers': 2, 'why_workers2': 'B-FAST の下見(workers 2)で最大 3,720 MB > 3.5 GB(protocol common_recipe.workers)。キューは GPU_QUEUE_WORKERS=2 で動いた',
                      'bfast_init': '通った(ジャンル 6 → 10 を train.py が広げた・trained_modes [4, 5])',
                      'msf_v2m': '通った(4608M で。3872M は 137。最大 4,165 MB)',
                      'v2m_light': '下見はしていない。段 1 の本番の 4 本が 137 なしで通った',
                      'aug2': '未(train.py に --aug-top-brightness(C1)がまだ無い)'},
                  'src': entry(REP / 'queue.md')},
            '7': {'what': 'dataset_v2 の steer_center_us・throttle_neutral_us(B は 1441.0・1496.0、±2 µs)', 'status': '済・外れ(ユーザーに聞く)',
                  'answer': {'dataset_v2': dc['checks']['action_reference_within_2us_of_B']['v2_all'], 'B': [1441.0, 1496.0],
                             'diff': [-8.0, 4.0], 'rule': '11/3 に出す前にユーザーに聞く。机上はそれぞれの model_config の値で計算'},
                  'src': 'deviations.md の 4'},
            '8': {'what': '同じ日の B の机上の推論の p99(門「推論」= その 1.3 倍)', 'status': '未',
                  'why': 'C3 がまだ無い。engine_check の run_numpy p99(64 フレーム)は参考で、門に使わない(eval_v2.py の決め方)。門「推論」は「未」'},
        },
        'genre_skew_threshold_us': {'genre4_last': 1.938, 'genre5_last': 5.308, 'from': 'protocol marks(変えない)'},
        'unfinished_and_effect': [
            'C3 が無い: 段 0 の 2・4・8 が未。C3 の欄を使う門(v2 の sanity・推論・低速のブレーキ・止まった場面・直線、B-FAST の G-b・G-c・G-d・G-f と安全の門)は「未」で合格にしない',
            'D3 は robust_dl1_us(C3 のずれの試し)と段 3 の学習が要る。C3 と段 3 のあと',
            'D1・D2・拒否権・対 5 本・同じ種・印(長すぎ・ジャンルの片寄り)は H1・介入の予測だけで決まる(C3 に依らない)',
        ],
    }
    tmp = str(LOCK2) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, LOCK2)
    os.chmod(LOCK2, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    h = sha(LOCK2)
    with open(LOCK2_SHA, 'w', encoding='utf-8') as f:
        f.write('# protocol_v2_lock2 の固定(%s)。確かめ方: cd %s && sha256sum -c protocol_v2_lock2.sha256(この行は無視される)\n'
                % (out['written_at'], REP))
        f.write('%s  protocol_v2_lock2.json\n' % h)
    os.chmod(LOCK2_SHA, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print('書いた:', LOCK2, h)
    return 0


if __name__ == '__main__':
    sys.exit(main())
