"""強化学習モードを JetFormer に登録する(2026-10-14、sim_rl_modes の統合の段)

  1. ブラウザに配る組 web/rl/(Node の試験と同じ rl/runtime/ の写し)を jetformer/sim/static/rl/ に置く。
     manifest.json の sha256・rl/runtime/ との一致を確かめる(違えば止まる)。teacher.js は JetFormer のファイルなので触らない。
  2. 登録簿 data/sim_rl/modes/<key>/ に mode.json・policy.json・eval.json・status.json・card.json・scenes.json を置く
     (作業用のフォルダーに書いてから置き換える)。中身は学習の段の書き出し(bundle/<key>/・train/speed/r2/export/・candidates/)と、
     確認の段の判定(下の VERDICTS。check_*_r2.md)と、記録のシーンの組(integrate/pools/pool_<key>.json)。
     優先 2・3 で不合格のモード(line・obstacle)は方策を置かず、状態 not_published(画面では「出していません」)で説明だけを出す。

    python integrate/publish_registry.py [--registry /home/jetson/jetformer/data/sim_rl/modes] [--static /home/jetson/jetformer/jetformer/sim/static/rl]
           [--browser-fid integrate/browser_fid.json] [--dry-run]

2026-10-15 昇格の段(r3/promote): no_wall・recover・line を第 4 回(r4・r4b)の一式(bundle/<key>/、status accepted・確認 r4 合格)に替えた。
  VERDICTS・カードは r4 の試験と確認 r4 の数字。line は bundle/line/ から出す(選べる)。bundle の status.json の round・judged_by・test・check・caveats_ja・
  runtime を登録簿の status.json に写す。登録簿の eval.json は bundle の eval.json から回ごとの行(tests・full_result)を除いたもの(line は 0.85 MB あり、
  API が毎回読むため)。除く前の sha256 を eval_full_sha256 に残す。前の版の写しは r3/promote/before/publish_registry.py。

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
    # 2026-10-15 昇格の段: no_wall・recover・line は第 4 回(r4・r4b)。学習の段の試験(1 回だけ)で全部合格し、確認 r4(学習・評価のコードを使わない反証)でも合格
    'no_wall': {
        'state': 'accepted',
        'why': 'check_no_wall_r4.md: 合格。r4 の試験 3,038 回(衝突 0・標準の物理の最小 8.10 cm・物理のずれ 8.16 cm)を作り直して一致し、新しい組 6,429 回'
               '(範囲の中・シールドあり 5,749 回)でも衝突 0・標準の物理の最小 8.10 cm・物理のずれの最小 7.03 cm',
    },
    'recover': {'state': 'accepted', 'why': 'check_recover_r4.md: 合格。r4 の試験(乱しつき 100 回で 99 回走り切り・立て直しの成功 0.993)を作り直して一致し、'
                                            '新しい開始 200 回でも有効 200/200・成功 0.977・ゲートの外し 0、Firefox の閉ループ 100 回で有効 100/100'},
    'speed': {'state': 'accepted', 'why': 'check_speed_r2.md: 凍結した試験を独立に再現して全部一致、新しい開始 1,500 回で衝突 0'},
    'line': {'state': 'accepted', 'why': 'check_line_r4.md: 合格(反証できなかった)。r4b の試験 542 回(衝突 0・止まり 0・最小 4.13 cm)を作り直して同じ値、'
                                         '新しい組 1,717 回 + 攻めの組 360 回(シールドあり)でも衝突 0・最小 4.0 cm'},
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
    # 2026-10-15 昇格の段: r4(protocol_no_wall_r4.json の試験 + 確認 r4)
    j, rep = ev['judged'], ev['reported']
    ui = rep['ui_numbers']
    return {
        'headline_ja': '試験 %d 回で衝突 0・最小すき間 %.1f cm(標準の物理。シールドのしきい値)・%.1f cm(物理のずれ)。停止から 1 周 %.2f 秒'
                       % (ui['runs'], 100 * ui['min_clearance_nominal_m'], 100 * ui['min_clearance_randomised_m'], j['N4']['value']),
        'lap_from_stop_s': j['N4']['value'], 'flying_lap_s': None,
        'tests': {'runs': ui['runs'], 'collisions': 0, 'text_ja': '%d 回の試験(標準の物理 %d + 物理のずれ)で衝突 0。確認 r4 の新しい組 5,749 回でも衝突 0'
                  % (ui['runs'], ui['min_clearance_nominal_runs'])},
        'min_clearance_cm': {'value': round(100 * ui['min_clearance_nominal_m'], 1),
                             'text_ja': '標準の物理(記録の開始から一様 1,000 回も同じ)。8.1 cm はシールドのしきい値で、方策だけでは 4.5 cm まで寄る'},
        'main': [{'label_ja': '物理のずれの最小すき間', 'value_ja': '試験の組 %.1f cm・確認 r4 の隅のまわり 7.0 cm' % (100 * ui['min_clearance_randomised_m'])},
                 {'label_ja': '安全シールドの作動', 'value_ja': '%.2f%% のフレーム(回ごとの最大 %.1f%%)。柱の前で動けないときは少し下がってから抜ける'
                  % (100 * j['N7_r4']['value']['fraction'], 100 * j['N7_r4']['value']['run_max'])},
                 {'label_ja': '最速との違い', 'value_ja': '線の違い 平均 %.0f cm・速さ %.2f m/s 遅い・最速は 8 cm 未満に寄る回が 150 回中 %d 回(このモードは 0)'
                  % (rep['fastest_compare']['line_diff_mean_cm'], -rep['fastest_compare']['speed_diff_mean_mps'], rep['fastest_compare']['fastest_below_8cm'])}],
        'checks': [{'id': k, 'label_ja': lab, 'pass': j[k]['pass']} for k, lab in (
            ('N1_r4', '試験 3,038 回で衝突 0'), ('N2_r4', '標準の物理ですき間 ≥ 8 cm(記録の開始から一様 1,000 回を含む)'),
            ('N3_r4', '物理のずれですき間 ≥ 5 cm'), ('N4', '停止から 1 周 ≤ 12 秒'),
            ('N5_r4', '完走(止まり・有効な周なしは不合格)'), ('N6', '無理のない加減速'), ('N7_r4', 'シールドの作動 ≤ 5%'))]
            + [{'id': 'check_r4', 'label_ja': '確認 r4(学習・評価のコードを使わない作り直し 6,429 回)', 'pass': True}],
        'notes_ja': ['「絶対」はシミュレーターの中の試験の結果で、実車の保証ではない。安全シールド v5 は外せない(8.1 cm は方策の余裕ではなく、シールドの仕組みで守っている)',
                     '記録の開始の範囲(左右 ±0.2 m・向き ±15°・開始のすき間 12 cm 以上)の外から走らせると、止まったまま動かないことがある(確認 r4: 200 回中 6 回、衝突 0)',
                     '物理のずれの隅(舵の速さ 0.8 で柱へ向いた開始など)では 7.0 cm まで寄り、まれに止まる(記録は標準の物理だけ)',
                     '記録のシーンは前もって Node で走らせ、すき間 8 cm 未満・止まり・有効な周なし・シールドの作動 25% 超・後ろへ動いたものを替えてある',
                     '最速と比べて遅く、壁から離れた 1 本の線に集まる(モードの中の線のばらつきは小さい)'],
    }


def card_recover(ev, browser):
    # 2026-10-15 昇格の段: r4(protocol_recover_r4.json の試験 + 確認 r4)
    j = ev['judged']
    c3, c4, c5, c6 = j['C3']['value'], j['C4']['value'], j['C5_r4']['value'], j['C6_r4']['value']
    return {
        'headline_ja': '乱した 100 回のうち %s 回が 20 秒を走り切った。立て直しの成功 %d%%(%d 回の乱し)・立て直しの時間 90%% 点 %.2f 秒(中央値 %.1f 秒 = 半分以上の乱しは線の帯から出ない)'
                       % (j['C1_r3']['value'].split('/')[0], round(100 * j['C2']['value']), j['C2']['events'], c3['p90'], c3['median']),
        'lap_from_stop_s': c6['lap_from_stop'], 'flying_lap_s': None,
        'tests': {'runs': 100, 'collisions': 0, 'text_ja': '乱しつきの 100 回で衝突 0(走り切り 99・止まり 1)。確認 r4 の新しい開始 200 回で有効 200/200・Firefox の閉ループ 100/100'},
        'min_clearance_cm': {'value': round(100 * c6['min_clearance'], 1), 'text_ja': '乱しなし(開始のすき間 3 cm 以上の 60 回)の最小。乱しのフレーム以外のシールド 3.5 cm でそろう'},
        'main': [{'label_ja': '立て直しの成功', 'value_ja': '%d%%(%d 回の乱し)。確認 r4 の新しい開始では 98%%' % (round(100 * j['C2']['value']), j['C2']['events'])},
                 {'label_ja': '押されたあとの生き残り', 'value_ja': '時刻の格子 %s・%s、新しい開始 %s・%s' % (c5['grid_yaw'], c5['grid_lateral'], c5['random_yaw'], c5['random_lateral'])},
                 {'label_ja': '乱し', 'value_ja': '1 シーン %.1f 回・学習から外すフレーム %d%%・ゲートの外し %s' % (c4['events_per_episode'], round(100 * c4['masked_fraction']),
                                                                                         j['C8']['value']['perturbed_valid_with_miss'])}],
        'checks': [{'id': k, 'label_ja': lab, 'pass': j[k]['pass']} for k, lab in (
            ('C1_r3', '乱しつき 100 回で 90 回以上走り切る(止まり・進みの不足も不合格)'), ('C2', '立て直しの成功 95% 以上'), ('C3', '立て直しの時間'),
            ('C4', '乱しの数・外す割合'), ('C5_r4', '押されたあとの生き残り(4 つとも 90% 以上)'), ('C6_r4', '乱しなしの走り(停止から 1 周 ≤ 9 秒・すき間 ≥ 3 cm)'),
            ('C7', '乱しの強さ・種類の割合'), ('C8', 'ゲートの外し'))]
            + [{'id': 'check_r4', 'label_ja': '確認 r4(学習・評価のコードを使わない作り直し 4,060 回 + Firefox 100 回)', 'pass': True}],
        'notes_ja': ['乱しは「立て直せる組」だけ(乱し v2・表 v2)。乱した操作は記録に印(info.jf.l = 0)が付き、JetFormer の取り込みで学習から外す'
                     '(シーンタブでジャンルを付け直しても外したまま)',
                     '開始のすき間が 5 cm より小さい開始と、ゲート 7 の右寄りの開始では、車が止まったまま動かないことがある(記録の前の確かめで替わるので記録には入らない)',
                     '余裕は小さい: 停止から 1 周 %.2f 秒(基準 9.0 秒)・乱しなしの最小すき間 %.1f cm(基準 3 cm)' % (c6['lap_from_stop'], 100 * c6['min_clearance']),
                     '乱しの強さを「強」に固定すると立て直しの成功は 96%(線の帯から出た出来事だけでは 94%)。既定の自動では 98%',
                     '前の版(r1)で記録したデータセット sim-rl-recover_20261014-1025・-1125 は古い方策で、ゲートを外したシーンがある(2/3・6/15)'],
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
    'L1_r4': '組ごとに 95% 以上走り切る(大きく横を向いた開始・範囲の端・角を含む)', 'L2_r4': 'すき間 ≥ 3 cm(全部の組)',
    'L7_shield': 'シールドの作動 ≤ 2%', 'L8_no_collision': '衝突 0', 'L9_no_stall': '止まり 0', 'L10_no_reverse_P_FAM6': '後ろへ動かない',
    'L11_escape_r4b': 'シールドの抜け出し ≤ 1%',
    'O1_valid': '障害物のある 100 回のうち 95 回以上走り切る', 'O2_clearance': 'すき間 ≥ 3 cm(足した障害物も)',
    'O3_lap1_median': '1 周目の中央値 ≤ 10.5 秒', 'O4_empty': '障害物なしで停止から 1 周 ≤ 9 秒'}


def card_line(ev, browser):
    # 2026-10-15 昇格の段: r4b(protocol_line_r4b.json の試験 + 確認 r4)。選べる
    j, rep = ev['judged'], ev['reported']
    l3, l4 = j['L3_tracking']['value'], j['L4_distinct']['value']
    return {
        'headline_ja': '試験 542 回で衝突 0・止まり 0・最小すき間 %.1f cm。左右の線の差 %.0f cm・イン/アウト %.0f cm・線のばらつき %.1f cm'
                       % (100 * j['L2_r4']['value'], 100 * l4['left_right_mean'], 100 * l4['inside_outside_mean'], 100 * j['L5_vm3']['value']),
        'lap_from_stop_s': j['L6_lap']['value'], 'flying_lap_s': None,
        'tests': {'runs': 542, 'collisions': 0, 'text_ja': '542 回の試験(シールドあり)で衝突 0。確認 r4 の新しい組 1,717 回 + 攻めの組 360 回でも衝突 0'},
        'min_clearance_cm': {'value': round(100 * j['L2_r4']['value'], 2), 'text_ja': '試験の組の最小(走行中のシールドが 4 cm を守る)'},
        'main': [{'label_ja': '線の追従', 'value_ja': '中央値 %.1f cm・95%% 点 %.1f cm' % (100 * l3['median'], 100 * l3['p95'])},
                 {'label_ja': '線の違い', 'value_ja': '左右 %.0f cm・イン/アウト %.0f cm(最速との線の差 平均 16 cm)' % (100 * l4['left_right_mean'], 100 * l4['inside_outside_mean'])},
                 {'label_ja': 'シールドの作動', 'value_ja': '%.2f%% のフレーム' % (100 * j['L7_shield']['value']['P_fraction'])}],
        'checks': [{'id': k, 'label_ja': CHECK_LABELS_JA.get(k, k), 'pass': v['pass']} for k, v in j.items()]
            + [{'id': 'check_r4', 'label_ja': '確認 r4(学習・評価のコードを使わない作り直し 542 回 + 新しい組 2,077 回)', 'pass': True}],
        'notes_ja': ['線の形(左寄り・右寄り・中央・イン・アウト・レイト/アーリーエイペックス・ランダム)はシーンの種から決まる',
                     '走行中は壁・柱から 4 cm を守るシールド v5b つき。方策だけでは角の一部(ゲート 32 の左寄りの端など)でまだ崩れる(確認 r4: シールドなしの攻めの組で 23/360 衝突、シールドありでは 0)',
                     '右寄り・アウトの線は追従の 95% 点が 15〜18 cm(全体は 9 cm)。ゲート 11 の開始では黒シートの横断が数えられないことがある(記録の前の確かめで替える)',
                     '記録のシーンは前もって Node で走らせ、すき間 3 cm 未満・止まり・有効な周なし・シールドの作動 10% 超・後ろへ動いたものを替えてある'],
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
    'speed': REPORTS / 'train/speed/r2/export', 'line': EXP / 'bundle/line', 'obstacle': EXP / 'candidates/obstacle',
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
    # 2026-10-15 昇格の段: bundle の status.json の r4 の中身(試験・確認・注意・走行中の仕組み)を写す
    for k in ('round', 'judged_by', 'test', 'check', 'caveats_ja', 'runtime', 'replaces'):
        if k in st0:
            status[k] = st0[k]
    if st0.get('caveats_ja'):
        card['caveats_ja'] = st0['caveats_ja']
    # 登録簿の eval.json は回ごとの行を除く(API が毎回読む)。除く前の sha256 を残す
    ev_reg = ev
    if any(k in ev for k in ('tests', 'full_result')):
        ev_reg = {k: v for k, v in ev.items() if k not in ('tests', 'full_result')}
        ev_reg['eval_full'] = {'path': str(src / 'eval.json'), 'sha256': sha256(src / 'eval.json'), 'removed_keys': [k for k in ('tests', 'full_result') if k in ev]}
    files = {'mode.json': mode, 'eval.json': ev_reg, 'status.json': status, 'card.json': card}
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
