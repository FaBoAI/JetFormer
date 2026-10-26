#!/usr/bin/env python3
"""段 plan(2026-10-26): 11/3 の日を足した bench_candidates.json を、画面と同じ読み方で確かめる(読むだけ)。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        check_bench_day.py [新しいファイル] [控え]

既定: 新しいファイル = data/bench_candidates.json、控え = data/bench_candidates.json.bak-20261026-plan。
確かめること:
  1. web/bench.load_candidates(画面の GET /api/bench/candidates と同じ)で読める。11/3 の 23 ブロックが全部「走れる」・警告 0
  2. 説明欄の雛形: day=2026-11-03 block=N cand=B tune=… throttle_gain=0.9。天井・発進の前進の値は入らない
  3. 今の日・候補・基準: 控えと新しいファイルで、画面に出す形がまったく同じ
  4. ブロックの上書きで使える項目(_block_overrides と PARAM_LIMITS)と、画面が欄に入れて走行開始で送る項目(app.js)
  5. plan_check: 11/3 の本は、速度ゲイン 0.9 か天井 ON で cand=EXP・official=0 になる(10/24 からの決まり)
  6. bench_nightly.sh の「1 日目」の自動の決め方: 20261103 → 20261023(days[0])。bench_report.planned_block
結果は logs/check_bench_day.json。車・基板・jetformer-web・ファイルの中身は変えない(一時ファイルは scratch に作って消す)。
"""
import copy
import json
import re
import sys
import tempfile
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
from jetformer import drive  # noqa: E402
from jetformer.web import bench  # noqa: E402

HERE = Path(__file__).resolve().parent
DATE = '2026-11-03'
PROBE = {'speed_ceiling': 1, 'speed_auto': 1, 'zone_caps': 1, 'ceil_all_mps': 4.4, 'ceil_left_mps': 4.2,
         'auto_target_mps': 4.27, 'auto_vref_v': 8.35, 'launch_boost': 1, 'launch_us': 250, 'launch_mps': 2.5,
         'launch_max_s': 1.2, 'throttle_gain': 0.95, 'steer_gain': 1.0, 'steer_trim_us': 15, 'lookahead': 3,
         'start_kick': 1, 'ensemble_j': 2, 'tof_guard': 1}


def ui_keys():
    """app.js: ブロックを押したときに画面の欄に入る項目(benchSetFields)と、走行開始で送る項目(readDriveParams)"""
    js = (JF / 'jetformer' / 'web' / 'static' / 'app.js').read_text(encoding='utf-8')
    m = re.search(r'const DR_PARAMS = \[(.*?)\n\];', js, re.S)
    dr = re.findall(r"\['#[\w-]+', '(\w+)'", m.group(1))
    m = re.search(r'const DR_FLAGS = \[(.*?)\];', js, re.S)
    fl = re.findall(r"\['#[\w-]+', '(\w+)'\]", m.group(1))
    m = re.search(r'function readDriveParams\(\) \{(.*?)\n\}', js, re.S)
    extra = re.findall(r'\bp\.(\w+) =', m.group(1))
    m = re.search(r'function benchSetFields\(p\) \{(.*?)\n\}', js, re.S)
    body = m.group(1)
    forced_off = [k for k in ('zone_caps', 'speed_ceiling', 'speed_auto') if k in body]
    sends = sorted(set(dr + fl + extra))
    return {'DR_PARAMS': dr, 'DR_FLAGS': fl, 'readDriveParams_extra': extra, 'sends': sends,
            'benchSetFields_forces_off': forced_off}


def main():
    new_path = Path(sys.argv[1]) if len(sys.argv) > 1 else JF / 'data' / 'bench_candidates.json'
    old_path = Path(sys.argv[2]) if len(sys.argv) > 2 else JF / 'data' / 'bench_candidates.json.bak-20261026-plan'
    res, fails = {'new': str(new_path), 'old': str(old_path)}, []

    # 1・2 画面と同じ読み方
    new = bench.load_candidates(new_path)
    old = bench.load_candidates(old_path)
    res['ok'] = new['ok']
    if not new['ok']:
        fails.append('読めない: %s' % new['error'])
    day = next((d for d in new['days'] if d['date'] == DATE), None)
    if day is None:
        fails.append('%s の日が無い' % DATE)
        day = {'blocks': []}
    w1103 = [w for w in new['warnings'] if DATE in w]
    res['warnings_all'] = new['warnings']
    res['warnings_1103'] = w1103
    if w1103:
        fails.append('11/3 の警告 %d' % len(w1103))
    rows = []
    for b in day['blocks']:
        d = b.get('description', '')
        ok_desc = ('day=%s block=%d cand=B' % (DATE, b['block']) in d and 'throttle_gain=0.9' in d
                   and 'tune=' in d and not re.search(r'ceil_|speed_ceiling|speed_auto|launch_', d))
        rows.append({'block': b['block'], 'id': b['id'], 'drivable': b['drivable'], 'why_not': b['why_not'],
                     'params': b.get('params'), 'tune': b.get('tune'), 'desc_ok': ok_desc, 'description': d})
        if not b['drivable']:
            fails.append('ブロック %d が走れない: %s' % (b['block'], b['why_not']))
        if not ok_desc:
            fails.append('ブロック %d の説明欄: %s' % (b['block'], d))
    res['blocks_1103'] = rows
    if len(rows) != 23 or [r['block'] for r in rows] != list(range(1, 24)):
        fails.append('11/3 のブロックが 1〜23 でない')

    # 3 今の日・候補・基準が同じ(画面に出す形)
    for k in ('baseline', 'candidates', 'description_template', 'bench', 'protocol_sha256', 'created_at', 'version'):
        if new.get(k) != old.get(k):
            fails.append('%s が控えと違う' % k)
    if new['days'][:len(old['days'])] != old['days']:
        fails.append('今の日の並びが控えと違う')
    raw_new = json.loads(new_path.read_text(encoding='utf-8'))
    raw_old = json.loads(old_path.read_text(encoding='utf-8'))
    if raw_new['days'][:len(raw_old['days'])] != raw_old['days'] or raw_new['candidates'] != raw_old['candidates'] \
            or raw_new['baseline'] != raw_old['baseline']:
        fails.append('ファイルの今の日・候補・基準が控えと違う')
    res['same_as_backup'] = not any('控え' in f for f in fails)
    res['old_warnings_same'] = [w for w in new['warnings'] if DATE not in w] == old['warnings']

    # 4 ブロックの上書きで使える項目
    probe_doc = copy.deepcopy(raw_new)
    probe_doc['days'] = [{'date': '2099-01-01', 'blocks': [{'block': 1, 'id': 'B', 'params': PROBE}]}]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'probe.json'
        p.write_text(json.dumps(probe_doc, ensure_ascii=False), encoding='utf-8')
        pr = bench.load_candidates(p)
    kept = (pr['days'][0]['blocks'][0].get('params') or {}) if pr['ok'] and pr['days'] else {}
    ui = ui_keys()
    table = []
    for k in PROBE:
        in_limits = k in drive.PARAM_LIMITS
        server = 'kept' if k in kept else ('dropped(switch)' if k in bench.SWITCH_KEYS else 'dropped(not in PARAM_LIMITS)')
        sent = k in ui['sends']
        if k in ui['benchSetFields_forces_off']:
            verdict = '使えない: 押すと画面が OFF にする(スイッチで入れる)'
        elif server != 'kept':
            verdict = '使えない: サーバーが捨てる(警告)'
        elif not sent:
            verdict = '使えない: サーバーは受けて説明欄に書くが、画面に欄が無く走行開始で送らない(説明欄だけ違う値になる)'
        else:
            verdict = '使える'
        table.append({'key': k, 'in_PARAM_LIMITS': in_limits, 'server': server, 'ui_sends': sent, 'verdict': verdict})
    res['override_probe'] = {'kept_by_server': kept, 'warnings': pr['warnings'], 'ui': ui, 'table': table}

    # 5 plan_check(走行開始の照らし合わせ。走らせない)
    b4 = next((b for b in day['blocks'] if b['block'] == 4), None)
    pc = []
    if b4:
        desc = b4['description'].replace('memo=', 'official=1 memo=')
        for name, params in (('ゲイン 0.95・天井 OFF', {'throttle_gain': 0.95, 'speed_ceiling': 0, 'speed_auto': 0}),
                             ('ゲイン 1.00・天井 ON', {'throttle_gain': 1.0, 'speed_ceiling': 1, 'speed_auto': 0}),
                             ('ゲイン 1.00・天井 OFF(基準と同じ)', {'throttle_gain': 1.0, 'speed_ceiling': 0, 'speed_auto': 0})):
            p = dict(new['baseline']['params'], **params)
            out, diffs = bench.plan_check(desc, b4['run'], 5, p, data=new)
            pc.append({'case': name, 'cand': bench.get_field(out, 'cand'), 'official': bench.get_field(out, 'official'),
                       'diffs': diffs})
    res['plan_check'] = pc
    if not pc or pc[0]['cand'] != 'EXP' or pc[1]['cand'] != 'EXP' or pc[2]['cand'] != 'B':
        fails.append('plan_check が思った形でない: %s' % pc)

    # 6 夜の道具の読み方
    dates = [d.get('date') for d in raw_new.get('days', []) if d.get('date')]
    i = dates.index(DATE) if DATE in dates else -1
    res['nightly_auto_day1'] = '' if i <= 0 else dates[0].replace('-', '')
    import bench_report as br  # noqa: E402
    res['planned_block_4'] = br.planned_block(raw_new, DATE, 4)
    if res['planned_block_4'] != (True, True, 'B'):
        fails.append('bench_report.planned_block が違う')

    res['fails'] = fails
    res['result'] = 'OK' if not fails else 'NG'
    out = HERE / 'logs' / 'check_bench_day.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('結果: %s(%s)' % (res['result'], out))
    print('11/3: %d ブロック・走れる %d・説明欄 OK %d・11/3 の警告 %d' % (
        len(rows), sum(r['drivable'] for r in rows), sum(r['desc_ok'] for r in rows), len(w1103)))
    print('今の日・候補・基準が控えと同じ: %s・今の警告も同じ: %s' % (res['same_as_backup'], res['old_warnings_same']))
    print('上書きの項目:')
    for t in table:
        print('  %-16s PARAM_LIMITS %-5s サーバー %-28s 画面が送る %-5s → %s' % (
            t['key'], t['in_PARAM_LIMITS'], t['server'], t['ui_sends'], t['verdict']))
    for c in pc:
        print('plan_check %s: cand=%s official=%s %s' % (c['case'], c['cand'], c['official'], '・'.join(c['diffs'])))
    print('bench_nightly の 1 日目の自動: %s・planned_block(11/3, 4) = %s' % (res['nightly_auto_day1'] or 'なし',
                                                                       res['planned_block_4']))
    for f in fails:
        print('NG:', f)
    sys.exit(0 if not fails else 1)


if __name__ == '__main__':
    main()
