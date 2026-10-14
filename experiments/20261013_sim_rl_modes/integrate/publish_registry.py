"""強化学習モードを JetFormer に登録する(2026-10-14、sim_rl_modes の統合の段)

  1. ブラウザに配る組 web/rl/(Node の試験と同じ rl/runtime/ の写し)を jetformer/sim/static/rl/ に置く。
     manifest.json の sha256・rl/runtime/ との一致を確かめる(違えば止まる)。teacher.js は JetFormer のファイルなので触らない。
  2. 登録簿 data/sim_rl/modes/<key>/ に mode.json・policy.json・eval.json・status.json・card.json・scenes.json を置く
     (作業用のフォルダーに書いてから置き換える)。中身は学習の段の書き出し(bundle/<key>/・train/speed/r2/export/・candidates/)と、
     確認の段の判定(下の VERDICTS。check_*_r2.md)と、記録のシーンの組(integrate/pools/pool_<key>.json)。
     優先 2・3 で不合格のモード(line・obstacle)は方策を置かず、状態 not_published(画面では「出していません」)で説明だけを出す。

    python integrate/publish_registry.py [--registry /home/jetson/jetformer/data/sim_rl/modes] [--static /home/jetson/jetformer/jetformer/sim/static/rl]
           [--browser-fid integrate/browser_fid.json] [--dry-run]

--browser-fid: 8080 のブラウザで F-ID(Node の操作とブラウザの操作の一致)を確かめた結果。最速の F4(ブラウザの同一性)をこれで判定する。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPORTS = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013')
JF = Path('/home/jetson/jetformer')
REGISTRY = JF / 'data/sim_rl/modes'
STATIC = JF / 'jetformer/sim/static/rl'
POOLS = REPORTS / 'integrate/pools'


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rj(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


# ---------------------------------------------------------------- 1. ブラウザの組
def sync_static(static: Path, dry: bool) -> dict:
    web, runtime = EXP / 'web/rl', EXP / 'rl/runtime'
    man = rj(web / 'manifest.json')
    out = {}
    for name, meta in sorted(man['files'].items()):
        src = web / name
        s = sha256(src)
        if s != meta['sha256']:
            raise SystemExit(f'web/rl/{name} の sha256 が manifest と違います')
        if (runtime / name).exists() and sha256(runtime / name) != s:
            raise SystemExit(f'web/rl/{name} が rl/runtime/{name} と違います(make_web_bundle.mjs で作り直してください)')
        out[name] = s
    extra = sorted(p.name for p in runtime.iterdir() if p.suffix in ('.mjs', '.json') and p.name not in man['files'])
    if extra:
        raise SystemExit('rl/runtime にあって web/rl に無いファイルがあります: ' + ', '.join(extra))
    if not dry:
        static.mkdir(parents=True, exist_ok=True)
        for name in out:
            tmp = static / ('.' + name + '.tmp')
            shutil.copyfile(web / name, tmp)
            os.replace(tmp, static / name)
        shutil.copyfile(web / 'manifest.json', static / '.manifest.json.tmp')
        os.replace(static / '.manifest.json.tmp', static / 'manifest.json')
        for p in static.iterdir():                 # 前の版にだけあったファイル(teacher.js は残す)
            if p.is_file() and p.name not in out and p.name not in ('manifest.json', 'teacher.js'):
                p.unlink()
    return {'files': out, 'manifest_sha256': sha256(web / 'manifest.json'), 'runtime_version': man.get('runtime_version')}


# ---------------------------------------------------------------- 2. 判定と画面の要約
# 確認の段(独立の再現・新しい開始)の判定。学習の段の status.json より優先する(確認で反証されたものは「不合格(参考)」)
VERDICTS = {
    'fastest': {'source': '学習しない(ベストラップの作業の確認 r1・r3 で本物と判定された main 種 1・1,000 万ステップ 7c43cb53…)'},
    'no_wall': {
        'state': 'not_accepted',
        'why': 'check_no_wall_r2.md: 凍結した試験 340 回は全部合格だが、試験した物理の範囲の中の角(摩擦円・μ×0.85・ブレーキ 0.7・舵の速さ 1.8・前進ゲイン 1.04)'
               'では 33 回中 30 回ぶつかる。標準の物理の新しい開始 600 回では最小すき間 7.0 cm(N2 の 8 cm は試験の組の引きがよかった)',
        'failed_add': ['check_r2_corner_collisions', 'check_r2_N2_fresh'],
    },
    'recover': {'state': 'not_accepted', 'why': 'check_recover_r2.md: 新しい乱し 150 回で有効 51/150・成功 75%。合格の基準(90/100・95%)に届かない'},
    'speed': {'state': 'accepted', 'why': 'check_speed_r2.md: 凍結した試験を独立に再現して全部一致、新しい開始 1,500 回で衝突 0'},
    'line': {'state': 'not_published', 'why': 'check_line_r2.md: L2(最小すき間 ≥ 3 cm)が 1.33 cm で不合格(優先 2 なので出さない)'},
    'obstacle': {'state': 'not_published', 'why': 'check_obstacle_r2.md: O1 66/100・O2 0.03 cm で不合格(優先 3 なので出さない)'},
}


def card_fastest(ev, browser):
    j = ev['judged']
    f4 = browser.get('fastest') if browser else None
    return {
        'headline_ja': '停止から 1 周 %.2f 秒・走りながら %.2f 秒(RC加速教師 8.93 / 8.37 秒)。REC60 の 60 回すべて完走・最小すき間 %.1f cm'
                       % (j['F2_lap']['value'], ev['reported']['flying_lap'], 100 * ev['reported']['rec60_min_clearance']),
        'lap_from_stop_s': j['F2_lap']['value'], 'flying_lap_s': ev['reported']['flying_lap'],
        'tests': {'runs': 62, 'collisions': 0, 'text_ja': '62 回の試験(停止から・走りながら + REC60)で衝突 0'},
        'min_clearance_cm': {'value': round(100 * ev['reported']['rec60_min_clearance'], 1), 'text_ja': 'REC60 の最小すき間'},
        'main': [{'label_ja': '線のばらつき(VM3)', 'value_ja': '%.1f cm(1 本の速い線。開始を広げてばらつきを出す)' % (100 * ev['reported']['variation']['VM3_m']['mean_abs_dev'])}],
        'checks': [{'id': 'F2_lap', 'label_ja': '停止から 1 周がベストラップの値と同じ', 'pass': j['F2_lap']['pass']},
                   {'id': 'F3_rec60', 'label_ja': 'REC60 60/60', 'pass': j['F3_rec60']['pass']},
                   {'id': 'F4_identity', 'label_ja': 'ブラウザ(8080)と Node の操作が同じ', 'pass': (f4 or {}).get('pass')}],
        'notes_ja': ['速さが目的のモードなので、線は 1 本(ばらつきは開始の位置・向きを広げて出す)',
                     'ベストラップの確認: 遅い舵 + 弱いブレーキ + 制御遅延 1 フレームの組み合わせには弱い(シミュレーターの標準の物理では問題なし)'],
    }


def card_no_wall(ev, browser):
    j = ev['judged']
    return {
        'headline_ja': '試験 %s 回で衝突 0。新しい開始 600 回(標準の物理)の最小すき間 7.0 cm。停止から 1 周 %.2f 秒' % (j['N1_no_collision']['runs'], j['N4_lap_from_stop']['value']),
        'lap_from_stop_s': j['N4_lap_from_stop']['value'], 'flying_lap_s': None,
        'tests': {'runs': j['N1_no_collision']['runs'], 'collisions': 0, 'text_ja': '%d 回の試験(標準の物理 + 物理のずれ)で衝突 0' % j['N1_no_collision']['runs']},
        'min_clearance_cm': {'value': 7.0, 'text_ja': '新しい開始 600 回(標準の物理)の最小。試験の組では 11.7 cm(標準)・7.0 cm(物理のずれ)'},
        'main': [{'label_ja': '安全シールドの作動', 'value_ja': '標準の物理 0%(物理をずらした R200 で 18% のフレーム。ほとんど速さの見張り)'},
                 {'label_ja': '走る速さ', 'value_ja': '平均 2.7 m/s(最速 3.3 m/s)'}],
        'checks': [{'id': k, 'label_ja': lab, 'pass': j[k]['pass']} for k, lab in (
            ('N1_no_collision', '試験 340 回で衝突 0'), ('N2_clearance_nominal', '標準の物理ですき間 ≥ 8 cm(試験の組)'),
            ('N3_clearance_randomised', '物理のずれですき間 ≥ 5 cm'), ('N4_lap_from_stop', '停止から 1 周 ≤ 12 秒'),
            ('N5_completion', '完走'), ('N6_friction', '無理のない加減速'), ('N7_shield', 'シールドの作動 ≤ 5%'))]
            + [{'id': 'check_r2', 'label_ja': '確認(新しい開始・物理の範囲の端)', 'pass': False}],
        'notes_ja': ['不合格(参考): 確認で、物理の範囲の端(摩擦円・μ×0.85・弱いブレーキ・遅い舵などの組み合わせ)ではぶつかることが分かった。'
                     'シミュレーターの標準の物理(記録で使う条件)では新しい開始 600 回でも衝突 0',
                     '「絶対」はシミュレーターの中の試験の結果で、実車の保証ではない。安全シールドは外せない(方策だけでは標準の物理でも 600 回に 1 回ぶつかる)',
                     '記録のシーンは前もって Node で走らせ、すき間 8 cm 未満・止まったまま・ゲートの外しのものを替えてある'],
    }


def card_recover(ev, browser):
    j = ev['judged']
    c3 = j['C3_recovery_time']['value']
    return {
        'headline_ja': '乱した 100 回のうち %s 回が 20 秒を走り切った(合格は 90 回)。立て直しの成功 %d%%・立て直しの時間 中央値 %.2f 秒'
                       % (j['C1_valid']['value'].split('/')[0], round(100 * j['C2_success']['value']), c3['median']),
        'lap_from_stop_s': j['C6_unperturbed']['value']['lap_from_stop'], 'flying_lap_s': None,
        'tests': {'runs': 100, 'collisions': 62, 'text_ja': '乱しつきの 100 回のうち 62 回で衝突(乱しのあと)'},
        'min_clearance_cm': {'value': round(100 * j['C6_unperturbed']['value']['rec60_min_clearance'], 1), 'text_ja': '乱しなし REC60 の最小すき間'},
        'main': [{'label_ja': '立て直しの成功', 'value_ja': '%d%%(%d 回の乱し)' % (round(100 * j['C2_success']['value']), j['C2_success']['events'])},
                 {'label_ja': '立て直しの時間', 'value_ja': '中央値 %.2f 秒・90%% 点 %.2f 秒' % (c3['median'], c3['p90'])},
                 {'label_ja': '乱し', 'value_ja': '1 シーン %.1f 回・学習から外すフレーム %d%%' % (j['C4_events_mask']['value']['events_per_episode'], round(100 * j['C4_events_mask']['value']['masked_fraction']))}],
        'checks': [{'id': k, 'label_ja': lab, 'pass': j[k]['pass']} for k, lab in (
            ('C1_valid', '乱しつき 100 回で 90 回以上走り切る'), ('C2_success', '立て直しの成功 95% 以上'), ('C3_recovery_time', '立て直しの時間'),
            ('C4_events_mask', '乱しの数・外す割合'), ('C5_kick', '押されたあとの生き残り'), ('C6_unperturbed', '乱しなしの走り'))],
        'notes_ja': ['不合格(参考): 乱しのあとにぶつかる開始が多い。記録のシーンは前もって Node で走らせ、20 秒を走り切るもの(候補の約 3 割)だけを使う',
                     '黒シートの出口での右への舵のパルスはほとんど戻れない。走り切ったシーンの約 4 割にシミュレーターのゲートの外しがある',
                     '乱した操作は記録に印(info.jf.l = 0)が付き、JetFormer の取り込みで学習から外す(シーンタブでジャンルを付け直しても外したまま)'],
    }


def card_speed(ev, browser):
    j = ev['judged']
    s3 = j['S3_lap_order']['value']
    return {
        'headline_ja': '速さの倍率 100%%・75%%・50%% で停止から 1 周 %.2f・%.2f・%.2f 秒。60 回すべて走り切り、最小すき間 %.1f cm'
                       % (s3['t1'], s3['t75'], s3['t5'], 100 * j['S2_clearance']['value']),
        'lap_from_stop_s': s3['t1'], 'flying_lap_s': None,
        'tests': {'runs': 63, 'collisions': 0, 'text_ja': '63 回の試験で衝突 0(新しい開始 1,500 回でも衝突 0)'},
        'min_clearance_cm': {'value': round(100 * j['S2_clearance']['value'], 1), 'text_ja': '試験の組の最小すき間'},
        'main': [{'label_ja': '速さの段', 'value_ja': ' / '.join('%d%% %.2f m/s' % (round(100 * r['c_v']), r['mean_speed']) for r in j['S4_speed_order']['value'])},
                 {'label_ja': '速さのばらつき(VM4)', 'value_ja': '%.2f m/s' % j['S5_vm4']['value']}],
        'checks': [{'id': k, 'label_ja': lab, 'pass': j[k]['pass']} for k, lab in (
            ('S1_valid', '60 回のうち 57 回以上走り切る'), ('S2_clearance', 'すき間 ≥ 3 cm'), ('S3_lap_order', '1 周の時間が速さの順で 3 秒以上違う'),
            ('S4_speed_order', '平均の速さが倍率の順'), ('S5_vm4', '速さのばらつき ≥ 0.4 m/s'))],
        'notes_ja': ['速さのばらつきは主に上限(4.5 m/s × 倍率)から来る。同じ倍率の中の線はほぼ 1 本',
                     '開始の範囲の端(ゲート 5 の左寄り・100% など)ではまれにぶつかる(確認の格子 117 回中 6 回)。記録のシーンは前もって走らせ、すき間 3 cm 未満・止まったまま・ゲートの外しのものを替えてある'],
    }


# 出していないモードのカードの合否の項目の名前(画面に出る。審査の段 2026-10-14: それまでは L1_valid などの ID のまま出ていた)
CHECK_LABELS_JA = {
    'L1_valid': '60 回のうち 57 回以上走り切る', 'L2_clearance': 'すき間 ≥ 3 cm', 'L3_tracking': '線の追従(中央値 ≤ 5 cm・95% 点 ≤ 12 cm)',
    'L4_distinct': '線の形を描き分けられる(左右・イン/アウト)', 'L5_vm3': '線のばらつき ≥ 8 cm', 'L6_lap': '真ん中の線で 1 周目 ≤ 10 秒',
    'O1_valid': '障害物のある 100 回のうち 95 回以上走り切る', 'O2_clearance': 'すき間 ≥ 3 cm(足した障害物も)',
    'O3_lap1_median': '1 周目の中央値 ≤ 10.5 秒', 'O4_empty': '障害物なしで停止から 1 周 ≤ 9 秒'}


def card_line(ev, browser):
    j = ev['judged']
    return {
        'headline_ja': '試験で 1 回だけ柱から %.1f cm まで寄った(合格は 3 cm 以上)ので出していません' % (100 * j['L2_clearance']['value']),
        'lap_from_stop_s': j['L6_lap']['value'], 'flying_lap_s': None,
        'tests': {'runs': 66, 'collisions': 1, 'text_ja': '66 回の試験で 1 回衝突'},
        'min_clearance_cm': {'value': round(100 * j['L2_clearance']['value'], 2), 'text_ja': '試験の組の最小すき間'},
        'main': [{'label_ja': '線の追従', 'value_ja': '中央値 %.1f cm・95%% 点 %.1f cm' % (100 * j['L3_tracking']['value']['median'], 100 * j['L3_tracking']['value']['p95'])},
                 {'label_ja': '線の違い', 'value_ja': '左右 %.0f cm・イン/アウト %.0f cm' % (100 * j['L4_distinct']['value']['left_right_mean'], 100 * j['L4_distinct']['value']['inside_outside_mean'])}],
        'checks': [{'id': k, 'label_ja': CHECK_LABELS_JA.get(k, k), 'pass': v['pass']} for k, v in j.items()],
        'notes_ja': ['経路(左寄り・右寄り・イン・アウトなど)は描き分けられているが、参照の線から大きく向きがずれた開始(全体の約 7%)で危ないことがある。'
                     '優先 2 のモードで不合格なので、選べるようにしていない(次の学習で直す)'],
    }


def card_obstacle(ev, browser):
    j = ev['judged']
    return {
        'headline_ja': '障害物のある 100 回(選びの組。試験の組はまだ使っていない)のうち %s 回だけ走り切った(合格は 95 回)ので出していません' % j['O1_valid']['value'].split('/')[0],
        'lap_from_stop_s': j['O4_empty']['value'], 'flying_lap_s': None,
        'tests': {'runs': 100, 'collisions': 34, 'text_ja': '障害物のある 100 回(選びの組)のうち 34 回で衝突'},
        'min_clearance_cm': {'value': round(100 * j['O2_clearance']['value'], 2), 'text_ja': '走り切った回の最小すき間'},
        'main': [],
        'checks': [{'id': k, 'label_ja': CHECK_LABELS_JA.get(k, k), 'pass': v['pass']} for k, v in j.items()],
        'notes_ja': ['すぐ前(3.5 m 以内)に障害物がある開始でぶつかりやすく、右へよける癖がある。優先 3 のモードで不合格なので、選べるようにしていない'],
    }


CARDS = {'fastest': card_fastest, 'no_wall': card_no_wall, 'recover': card_recover, 'speed': card_speed, 'line': card_line, 'obstacle': card_obstacle}
SOURCES = {
    'fastest': EXP / 'bundle/fastest', 'no_wall': EXP / 'bundle/no_wall', 'recover': EXP / 'bundle/recover',
    'speed': REPORTS / 'train/speed/r2/export', 'line': EXP / 'candidates/line_r2', 'obstacle': EXP / 'candidates/obstacle',
}


def build_mode(key: str, browser: dict | None) -> dict:
    src = SOURCES[key]
    mode, ev, st0 = rj(src / 'mode.json'), rj(src / 'eval.json'), rj(src / 'status.json')
    v = VERDICTS[key]
    state = v.get('state') or st0['state']
    failed = list(st0.get('failed') or []) + list(v.get('failed_add') or [])
    pending = list(st0.get('pending') or [])
    if key == 'fastest':
        f4 = (browser or {}).get('fastest')
        if f4 and f4.get('pass') is True:
            pending = [p for p in pending if p != 'F4_identity']
            state = 'accepted' if not failed and not pending else st0['state']
        elif f4 and f4.get('pass') is False:
            failed.append('F4_identity'); state = 'not_accepted'
    label = {'accepted': '合格', 'not_accepted': '不合格(参考)', 'pending': '確認待ち', 'not_published': '出していません(試験で不合格)'}[state]
    card = {'schema': 'jf-sim-rl-mode-card/v1', 'key': key, 'state': state, **CARDS[key](ev, browser),
            'verdict_ja': v.get('why') or v.get('source'), 'updated_at': datetime.now().isoformat(timespec='seconds')}
    if browser and browser.get(key):
        card['browser_identity'] = browser[key]
    status = {'schema': 'jf-sim-rl-mode-status/v1', 'key': key, 'state': state, 'label_ja': label, 'failed': failed, 'pending': pending,
              'from_training_status': {k: st0.get(k) for k in ('state', 'label_ja', 'updated_at')}, 'verdict_ja': card['verdict_ja'],
              'updated_at': datetime.now().isoformat(timespec='seconds')}
    files = {'mode.json': mode, 'eval.json': ev, 'status.json': status, 'card.json': card}
    out = {'key': key, 'state': state, 'files': files, 'policy': None, 'pool': None}
    if state != 'not_published':
        pol = src / 'policy.json'
        psha = sha256(pol)
        if psha != mode['policy_sha256']:
            raise SystemExit(f'{key}: policy.json の sha256 {psha[:12]} が mode.json の {mode["policy_sha256"][:12]} と違います')
        out['policy'] = pol
        pool_p = POOLS / f'pool_{key}.json'
        if pool_p.exists():
            pool = rj(pool_p)
            if pool.get('policy_sha256') != psha:
                raise SystemExit(f'{key}: シーンの組が別の方策で作られています(作り直してください)')
            out['pool'] = pool
    return out


def write_mode(reg: Path, m: dict) -> None:
    reg.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.%s-' % m['key'], dir=reg))
    os.chmod(tmp, 0o775)
    try:
        for name, data in m['files'].items():
            (tmp / name).write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        if m['policy'] is not None:
            shutil.copyfile(m['policy'], tmp / 'policy.json')
        if m['pool'] is not None:
            (tmp / 'scenes.json').write_text(json.dumps(m['pool'], ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        for p in tmp.iterdir():
            with open(p, 'rb') as f:
                os.fsync(f.fileno())
        dest = reg / m['key']
        old = reg / ('.%s-old' % m['key'])
        if old.exists():
            shutil.rmtree(old)
        if dest.exists():
            os.replace(dest, old)
        os.replace(tmp, dest)
        if old.exists():
            shutil.rmtree(old)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', type=Path, default=REGISTRY)
    ap.add_argument('--static', type=Path, default=STATIC)
    ap.add_argument('--browser-fid', type=Path, default=None)
    ap.add_argument('--modes', default='fastest,no_wall,recover,speed,line,obstacle')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    browser = rj(a.browser_fid) if a.browser_fid and a.browser_fid.exists() else None
    browser = (browser or {}).get('modes') if browser else None
    st = sync_static(a.static, a.dry_run)
    print(f'ブラウザの組: {len(st["files"])} ファイル(manifest {st["manifest_sha256"][:12]})→ {a.static}' + ('(dry-run)' if a.dry_run else ''))
    summary = {'schema': 'jf-sim-rl-publish/v1', 'at': datetime.now().isoformat(timespec='seconds'), 'static': st, 'modes': {}}
    for key in a.modes.split(','):
        m = build_mode(key, browser)
        summary['modes'][key] = {'state': m['state'], 'policy_sha256': sha256(m['policy']) if m['policy'] else None,
                                 'pool_scenes': len(m['pool']['scenes']) if m['pool'] else 0}
        if not a.dry_run:
            write_mode(a.registry, m)
        print(f'{key}: {m["state"]}・方策 {"あり" if m["policy"] else "なし"}・シーンの組 {summary["modes"][key]["pool_scenes"]}')
    if not a.dry_run:
        (REPORTS / 'integrate').mkdir(exist_ok=True)
        (REPORTS / 'integrate/publish_last.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
