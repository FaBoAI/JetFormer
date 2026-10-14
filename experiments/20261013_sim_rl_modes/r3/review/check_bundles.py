"""第 3〜4 回の審査: 昇格した一式(bundle・登録簿・配るファイル・画面の API)が、試験の JSON・確認の段の数字と合うかを確かめる(読むだけ)

    python3 -I check_bundles.py > ~/jetformer_reports/sim_rl_modes_20261013/r3/review/bundles.json

確かめること(モード no_wall・recover・line):
  1. 方策の sha256: 試験の JSON の policy = 候補 = bundle の policy.json = mode.json の policy_sha256 = 登録簿 = 画面の API
  2. bundle/<mode>/sha256.json に書かれた sha256 = 今のファイル
  3. status.json の test の値 = 試験の JSON の judged の値(1 つずつ)、check の数字 = 確認の段の数字
  4. 登録簿 data/sim_rl/modes/<mode>/ の状態・eval・mode = bundle(line の eval は回ごとの行を除いた版)
  5. 配るファイル rl/runtime = web/rl = jetformer/sim/static/rl、manifest の sha256 = ファイル
  6. 第 3 回の候補の状態が不合格になっていること
"""
import hashlib
import json
import urllib.request
from pathlib import Path

EXP = Path('/home/jetson/jetformer/experiments/20261013_sim_rl_modes')
R3 = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3')
REG = Path('/home/jetson/jetformer/data/sim_rl/modes')
STATIC = Path('/home/jetson/jetformer/jetformer/sim/static/rl')
API = 'http://192.168.55.1:8080'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rj(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode('utf-8'))


TEST = {'no_wall': R3 / 'no_wall_r4/test/eval_test.json', 'recover': R3 / 'rc4/eval_test.json', 'line': R3 / 'line_r4b/eval_test_r4b.json'}
CAND = {'no_wall': 'no_wall_r4', 'recover': 'recover_r4', 'line': 'line_r4'}


def jv(v):
    """試験の judged の値(dict なら value)"""
    if isinstance(v, dict) and 'value' in v:
        return v['value']
    return v


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))
    return a == b


def main():
    out = {'schema': 'jf-sim-rl-r3-review-bundles/v1', 'modes': {}, 'problems': []}
    api = get(API + '/api/sim/rl/modes')
    api_modes = {m['key']: m for m in api['modes']}
    for m in ('no_wall', 'recover', 'line'):
        r = {}
        test = rj(TEST[m])
        b = EXP / 'bundle' / m
        cand = EXP / 'r3/candidates' / CAND[m]
        mode = rj(b / 'mode.json')
        st = rj(b / 'status.json')
        reg_st = rj(REG / m / 'status.json')
        reg_mode = rj(REG / m / 'mode.json')
        api_m = api_modes.get(m, {})
        shas = {'test_json': test['policy']['sha256'], 'candidate_policy': sha(cand / 'policy.json'), 'bundle_policy': sha(b / 'policy.json'),
                'bundle_mode_json': mode.get('policy_sha256'), 'registry_policy': sha(REG / m / 'policy.json'),
                'registry_mode_json': reg_mode.get('policy_sha256'), 'api': api_m.get('policy_sha256')}
        r['policy_sha256'] = shas
        if len(set(shas.values())) != 1:
            out['problems'].append('%s: 方策の sha256 がそろわない %s' % (m, shas))
        # 2. sha256.json
        sj = rj(b / 'sha256.json')
        bad = {f: (v, sha(b / f)) for f, v in sj['files'].items() if sha(b / f) != v}
        r['bundle_sha256_json_mismatch'] = bad
        if bad:
            out['problems'].append('%s: bundle の sha256.json と違うファイル %s' % (m, list(bad)))
        # 候補と bundle で中身が同じファイル(mode・status 以外)
        diff_files = []
        for f in sorted(p.name for p in b.iterdir() if p.is_file()):
            if f in ('mode.json', 'status.json', 'sha256.json'):
                continue
            if (cand / f).is_file() and sha(cand / f) != sha(b / f):
                diff_files.append(f)
        r['bundle_vs_candidate_differs'] = diff_files
        # 3. status の test = 試験の judged
        j = test['judged']
        cmp = {}
        for k, v in (st.get('test') or {}).items():
            if k in j:
                tv = jv(j[k])
                cmp[k] = {'status': v, 'test': tv,
                          'same': same(v, tv) if not isinstance(v, (dict, list)) else json.dumps(v, sort_keys=True) == json.dumps(tv, sort_keys=True)}
        r['status_test_vs_judged'] = cmp
        for k, c in cmp.items():
            if c['same'] is False:
                out['problems'].append('%s: status.test.%s %s != 試験 %s' % (m, k, c['status'], c['test']))
        r['test_all_judged_pass'] = test.get('all_judged_pass')
        r['status_state'] = {'bundle': st.get('state'), 'registry': reg_st.get('state'), 'api': api_m.get('state'),
                             'candidate': rj(cand / 'status.json').get('state')}
        if len(set(r['status_state'].values())) != 1 or st.get('state') != 'accepted':
            out['problems'].append('%s: 状態がそろわない %s' % (m, r['status_state']))
        if not test.get('all_judged_pass'):
            out['problems'].append('%s: 試験の all_judged_pass が真でない' % m)
        # 4. 登録簿 = bundle
        r['registry_vs_bundle'] = {f: sha(REG / m / f) == sha(b / f) for f in ('mode.json', 'eval.json', 'policy.json')}
        r['registry_card_state'] = rj(REG / m / 'card.json').get('state')
        r['api_card_headline'] = (api_m.get('card') or {}).get('headline_ja')
        r['mode_runtime'] = mode.get('runtime', {}).get('shield_params', {}).get('select') if isinstance(mode.get('runtime'), dict) else None
        out['modes'][m] = r
    # eval.json の差を詳しく(line は除いた版・recover は 1 バイト違い)
    for m in ('recover', 'line', 'no_wall'):
        a, bb = rj(EXP / 'bundle' / m / 'eval.json'), rj(REG / m / 'eval.json')
        keys_a, keys_b = set(a) if isinstance(a, dict) else set(), set(bb) if isinstance(bb, dict) else set()
        dk = sorted(k for k in keys_a | keys_b if json.dumps(a.get(k), sort_keys=True) != json.dumps(bb.get(k), sort_keys=True))
        out['modes'][m]['eval_keys_differ_registry_vs_bundle'] = dk
    # 5. 配るファイル
    rt = EXP / 'rl/runtime'
    web = EXP / 'web/rl'
    names = sorted(p.name for p in rt.iterdir() if p.is_file())
    files = {}
    for n in names:
        files[n] = {'runtime': sha(rt / n), 'web': sha(web / n) if (web / n).exists() else None, 'static': sha(STATIC / n) if (STATIC / n).exists() else None}
    man_web = rj(web / 'manifest.json')
    man_static = rj(STATIC / 'manifest.json')
    rows_bad = [n for n, v in files.items() if len(set(v.values())) != 1]
    man_bad = [n for n, v in man_web['files'].items() if v['sha256'] != files.get(n, {}).get('runtime')]
    extra_static = sorted(p.name for p in STATIC.iterdir() if p.is_file() and p.name not in names and p.name not in ('manifest.json', 'teacher.js'))
    out['runtime'] = {'files': len(names), 'differ': rows_bad, 'manifest_mismatch': man_bad, 'manifest_web_sha256': sha(web / 'manifest.json'),
                      'manifest_static_same_as_web': sha(web / 'manifest.json') == sha(STATIC / 'manifest.json'),
                      'manifest_files': len(man_web['files']), 'extra_in_static': extra_static,
                      'shield': files.get('shield.mjs', {}).get('runtime'), 'shield_v4p': files.get('shield_v4p.mjs', {}).get('runtime'),
                      'mode_runtime': files.get('mode_runtime.mjs', {}).get('runtime')}
    if rows_bad or man_bad or not out['runtime']['manifest_static_same_as_web'] or extra_static:
        out['problems'].append('配るファイルがそろわない: %s' % {k: out['runtime'][k] for k in ('differ', 'manifest_mismatch', 'extra_in_static')})
    # 6. 第 3 回の候補
    out['r3_candidates'] = {c: rj(EXP / 'r3/candidates' / c / 'status.json').get('state') for c in ('no_wall_r3', 'recover_r3', 'line_r3')}
    if any(v != 'not_accepted' for v in out['r3_candidates'].values()):
        out['problems'].append('第 3 回の候補の状態 %s' % out['r3_candidates'])
    # 画面の API 全体
    out['api_states'] = {k: v.get('state') for k, v in api_modes.items()}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
