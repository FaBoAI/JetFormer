"""昇格の段(2026-10-15、r3/promote): r4 で合格した 3 つのモード(no_wall・recover・line)の一式を bundle/<mode>/ に置く。

  - 中身は候補(r3/candidates/<mode>_r4/)の書き出しそのもの(policy.json・eval.json・F-ID・シールドの値など)。runtime/ と promote/ は置かない
    (走行中の仕組みは共有の rl/runtime・web/rl にまとめた。r3/promote/equiv_runtime.mjs で同じ動きを確かめた)。
  - status.json: accepted「合格」(r4 の試験の数字・確認 r4 の結果・注意)。
  - mode.json: 候補のまま + 確認の段の申し送り(画面の説明・データセットの注意)。変えた鍵は overrides_from_promote に書く。
    no_wall・line には、取り込んだデータセットのカードに入る dataset_card(notes_ja)を足す(JetFormer は mode.json の dataset_card だけを rl_notes に写す)。
  - 前の一式は r3/promote/before/bundle/ にある。作業用のフォルダーに書いてから置き換える。

    python3 -I r3/promote/write_bundles.py [--dry-run]
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

EXP = Path('/home/jetson/jetformer/experiments/20261013_sim_rl_modes')
CAND = EXP / 'r3/candidates'
BUNDLE = EXP / 'bundle'
R3 = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3')
NOW = datetime.now().isoformat(timespec='seconds')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rj(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def runtime_info():
    man = rj(EXP / 'web/rl/manifest.json')
    eq = rj(R3 / 'promote/equiv/summary.json')
    return {'shared_runtime': 'experiments/20261013_sim_rl_modes/rl/runtime(= web/rl = jetformer/sim/static/rl)',
            'web_manifest_sha256': sha(EXP / 'web/rl/manifest.json'), 'runtime_version': man['runtime_version'],
            'files': {k: v['sha256'] for k, v in man['files'].items() if k in ('mode_runtime.mjs', 'shield.mjs', 'shield_v4p.mjs', 'perturb_v2.mjs', 'recoverable_table.mjs')},
            'equivalence': {'how_ja': '試験に使った候補の runtime と、まとめた runtime を同じ開始・種・物理で閉ループに走らせ、全フレームの操作・info.jf・metadata・乱しの出来事を比べた'
                                      '(r3/promote/equiv_runtime.mjs)。保存した F-ID も与えた', **eq}}


# ---------------------------------------------------------------- モードごとの中身
def no_wall():
    src = CAND / 'no_wall_r4'
    files = ['policy.json', 'eval.json', 'shield_config.json', 'fid_std.json', 'fid_slow.json', 'fid_corner.json', 'fid_escape.json', 'fid_escape_rev.json',
             'fid_summary.json']
    mode = rj(src / 'mode.json')
    notes = list(mode.get('dataset_card_notes_ja') or [])
    notes += ['確認 r4(check_no_wall_r4.md、学習・評価のコードを使わない作り直し 6,429 回): 範囲の中・シールドありの 5,749 回で衝突 0。標準の物理の最小すき間 8.10 cm'
              '(記録の開始から一様 1,500 回も 8.10 cm)、物理のずれの最小 7.03 cm(隅のまわり。試験の組では 8.16 cm)',
              '方策だけ(シールドなし)では 4.54 cm まで寄る(確認 r4)。8 cm は方策の余裕ではなく、シールドの仕組みで守っている',
              'アプリが許す開始(左右 ±0.3 m・向き ±25°)のうち記録の分布(±0.2 m・±15°・開始のすき間 12 cm 以上)の外では、止まったまま動かないことがある'
              '(確認 r4: 200 回中 6 回、衝突 0。開始のすき間が 8.1 cm 未満だと最初から動かない)。記録のシーンは分布の中から選び、前もって確かめる']
    mode['dataset_card_notes_ja'] = notes
    mode['dataset_card'] = {'schema': 'jf-sim-rl-dataset-card-notes/v1', 'notes_ja': notes}
    mode['ui_note_ja'] = ('記録の開始の範囲(左右 ±0.2 m・向き ±15°・開始のすき間 12 cm 以上)の外から走らせると、シールドが 8.1 cm を守れず止まったまま動かないことがあります'
                          '(衝突はしません)。柱の前で動けなくなったときは、少し下がってから抜けます')
    mode['overrides_from_promote'] = ['dataset_card', 'dataset_card_notes_ja', 'ui_note_ja']
    ev = rj(src / 'eval.json')
    J = ev['judged']
    status = {
        'schema': 'jf-sim-rl-mode-status/v1', 'key': 'no_wall', 'state': 'accepted', 'label_ja': '合格', 'failed': [], 'pending': [], 'round': 'r4',
        'judged_by': 'protocol_no_wall_r4.json(sha256 ad90c926…)の N1_r4・N2_r4・N3_r4・N4・N5_r4・N6・N7_r4。試験の組は選んだ点で 1 回だけ(4,010 回・判定 3,038 回)',
        'test': {k: v.get('value') for k, v in J.items()},
        'check': {'pass': True, 'report': 'jetformer_reports/sim_rl_modes_20261013/r3/check_no_wall_r4.md', 'runs_new': 6429,
                  'summary_ja': '学習の段の試験 53 行を作り直して 53 回とも完全に一致。新しい組 6,429 回(範囲の中・シールドあり 5,749 回)で衝突 0、標準の物理の最小 8.10 cm・'
                                '物理のずれの最小 7.03 cm、止まり: 標準の物理 0/3,303・物理のずれ 4/2,411(許しの中)。sh = 0 のフレームの指令は方策そのもの(3,892,200 フレーム)',
                  'numbers': {'in_range_collisions': '0/5749', 'nominal_min_cm': 8.10, 'uniform_1500_min_cm': 8.10, 'dr_min_cm': 7.03, 'nominal_stalls': '0/3303',
                              'dr_stalls': '4/2411', 'N7_shield_pct': 0.146, 'policy_only_min_cm': 4.54, 'app_range_stalls': '6/200', 'frame_ms_max_node': 803.7}},
        'caveats_ja': ['標準の物理の最小すき間はシールドのしきい値 8.1 cm に張り付く(方策だけでは 4.54 cm まで寄る)',
                       '物理のずれの隅(舵の速さ 0.8・柱へ向いた開始)では止まりの小さな島がある(VV9 のまわり 35 回中 13 回。記録は標準の物理だけなので入らない)',
                       '記録の分布の外の開始(アプリの範囲)では止まることがある(200 回中 6 回、衝突 0)',
                       '抜け出しを計画するフレームは Node で 70〜100 ms(混んだ CPU で最大 0.8 秒)。ブラウザの 1 フレームの時間は昇格の段で測る',
                       'モードの中の線のばらつきは小さい(0.64 cm。最速 2.54 cm)'],
        'replaces': {'previous': 'r2(方策 153c709c…・シールド v3、不合格(参考))', 'backup': 'experiments/20261013_sim_rl_modes/r3/promote/before/bundle/no_wall/'},
        'candidate': 'experiments/20261013_sim_rl_modes/r3/candidates/no_wall_r4/', 'runtime': runtime_info(),
        'note_ja': 'r4 の試験(protocol_no_wall_r4.json、1 回だけ)で全部合格し、確認 r4 でも反証できなかった。昇格の段(2026-10-15)が bundle/no_wall/ と登録簿をこの一式で置き換えた',
        'updated_at': NOW}
    return src, files, mode, status


def recover():
    src = CAND / 'recover_r4'
    files = ['policy.json', 'eval.json', 'perturbation.json', 'recoverable_table_v2.json', 'shield.json', 'manifest.json']
    extra = {('fid/%s' % p.name): p for p in sorted((R3 / 'check_recover_r4/fid').glob('fid_*.json'))}
    mode = rj(src / 'mode.json')
    ui = ('開始のすき間が 5 cm より小さい開始と、ゲート 7 の右寄りの開始では、車が止まったまま動かないことがある(記録の前の確かめで替わるので記録には入らない)。'
          '手で走らせると 20 秒止まったままに見える')
    mode['ui_note_ja'] = ui
    dc = mode.get('dataset_card') or {'schema': 'jf-sim-rl-dataset-card-notes/v1', 'notes_ja': []}
    dc['notes_ja'] = list(dc.get('notes_ja') or []) + [
        '確認 r4(check_recover_r4.md): 試験の組を作り直して全部一致。新しい開始 200 回(乱しあり)で有効 200/200・立て直しの成功 0.977・ゲートの外し 0/200、Firefox の閉ループ 100 回でも有効 100/100',
        '余裕は小さい: 停止から 1 周 8.85 秒(基準 9.0 秒)、乱しなしの最小すき間 3.50 cm(基準 3 cm。シールドの 3.5 cm でそろう)',
        '乱しの強さを「強」に固定すると立て直しの成功は 0.957(線の帯から出た出来事だけでは 0.941)。既定の自動(弱・中・強)では 0.977',
        ui]
    mode['dataset_card'] = dc
    mode['status_label_ja'] = '合格'
    mode['overrides_from_promote'] = ['dataset_card', 'ui_note_ja', 'status_label_ja']
    ev = rj(src / 'eval.json')
    J = ev['judged']
    status = {
        'schema': 'jf-sim-rl-mode-status/v1', 'key': 'recover', 'state': 'accepted', 'label_ja': '合格', 'failed': [], 'pending': [], 'round': 'r4',
        'judged_by': 'protocol_recover_r4.json(sha256 1b6bd8ad…)の C1_r3・C2・C3・C4・C5_r4・C6_r4・C7・C8。試験の組は 1 回だけ',
        'test': {k: v.get('value') for k, v in J.items()},
        'check': {'pass': True, 'report': 'jetformer_reports/sim_rl_modes_20261013/r3/check_recover_r4.md', 'runs_new': 4060,
                  'summary_ja': '確認 r4 合格。試験の組を自前の作り直しで 1 回ずつ全部同じに再現(8 項目すべて合格)。新しい種・開始でも 8 項目すべて合格'
                                '(乱しあり 200/200・成功 0.977・外し 0/200、キック 83/84・84/84・65/65・49/49、乱しなし 120/120・最小 3.505 cm)。'
                                'Firefox の閉ループ 100 回で有効 100/100・外し 0/100。ラベルの漏れ 0(91,823 フレーム)',
                  'numbers': {'fresh_FP200_valid': '200/200', 'fresh_C2': 0.977, 'browser_closed_loop_100': '100/100', 'band_kicks_STD': '244/244',
                              'low_clearance_3to5cm_stalls': '9/60', 'strong_fixed_C2': 0.957, 'strong_fixed_left_band_C2': 0.941}},
        'caveats_ja': ['(a) 止まりが残る: 開始のすき間が 3〜5 cm の開始(一様な開始の約 0.9%)を 60 個集めると 9/60 が止まる(3〜3.5 cm 5/25、3.5〜5 cm 4/35)',
                       '(b) ゲート 7 の右の開始で方策が壁へ寄って止まる(r4 で新しく出た。格子のゲート 7・右 0.2 m の 5 つの向きのうち 3 つと試験の P27)。'
                       '一様な開始で約 0.2〜0.3%。valid_episode_r3 と記録の前の確かめで外れる',
                       '(c) 余裕が小さい: 停止から 1 周 8.847 秒(基準 9.0)、乱しなしの最小すき間 3.50 cm(基準 3 cm)',
                       '(d) 乱しの強さを「強」に固定すると C2 は 0.957(375/392)、線の帯から出た出来事だけでは 0.941。既定の auto では 0.977'],
        'replaces': {'previous': 'r1(方策 7720f275…・乱し v1・シールドなし、不合格(参考))', 'backup': 'experiments/20261013_sim_rl_modes/r3/promote/before/bundle/recover/'},
        'candidate': 'experiments/20261013_sim_rl_modes/r3/candidates/recover_r4/', 'runtime': runtime_info(),
        'note_ja': 'r4 の試験(protocol_recover_r4.json、1 回だけ)で 8 項目すべて合格し、確認 r4 でも合格。昇格の段(2026-10-15)が bundle/recover/ と登録簿をこの一式で置き換えた',
        'updated_at': NOW}
    return src, files + list(extra), mode, status, extra


def line():
    src = CAND / 'line_r4'
    files = ['policy.json', 'eval.json', 'shield.json', 'fid_std_left08.json', 'fid_ln_g30r120_r4b_g8_shield.json', 'browser_fid.json']
    mode = rj(src / 'mode.json')
    notes = ['線の形(左寄り・右寄り・中央・イン・アウト・レイトエイペックス・アーリーエイペックス・ランダム)はシーンの種から決まる(フレーム 0 の info.jf.p.line に記録)',
             'r4b の試験(protocol_line_r4b.json、542 回、1 回だけ): 衝突 0・止まり 0・抜け出し 0・最小すき間 4.13 cm、線の追従 中央値 1.5 cm・95% 点 9.1 cm、'
             '左右の線の差 34 cm・イン/アウト 16 cm、線のばらつき VM3 9.9 cm',
             '走行中は壁・柱から 4 cm を守るシールド v5b(標準の物理の模型だけ)。作動は 0.11% のフレーム。方策だけ(シールドなし)では角の一部でまだ崩れる'
             '(確認 r4: ゲート 32 の左寄りの端など、攻めの組 360 回で 23 回衝突。シールドありでは 0・最小 4.0 cm)',
             '右寄り・アウトの線は追従の 95% 点が 15〜18 cm(全体は 9 cm)',
             'ゲート 11 の開始では 1 周目の黒シートの横断が数えられないことがある(記録の前の確かめで替える)',
             '記録の前の予想で、valid_episode_r3 でない・最小すき間 3 cm 未満・シールドの作動 10% 超・後ろへ動いたシーンは替える']
    mode['dataset_card'] = {'schema': 'jf-sim-rl-dataset-card-notes/v1', 'notes_ja': notes}
    mode['ui_note_ja'] = '経路の形はシーンの種から決まります。走行中は壁・柱から 4 cm を守るシールドが見張ります(作動したフレームの数を下に出します)'
    mode['overrides_from_promote'] = ['dataset_card', 'ui_note_ja']
    ev = rj(src / 'eval.json')
    J = ev['judged']
    status = {
        'schema': 'jf-sim-rl-mode-status/v1', 'key': 'line', 'state': 'accepted', 'label_ja': '合格', 'failed': [], 'pending': [], 'round': 'r4b',
        'judged_by': 'protocol_line_r4b.json(sha256 c7a1f677…)の L1_r4〜L11_escape_r4b。試験の組は 1 回だけ(542 回)',
        'test': {k: v.get('value') for k, v in J.items()},
        'check': {'pass': True, 'report': 'jetformer_reports/sim_rl_modes_20261013/r3/check_line_r4.md', 'runs_new': 2077,
                  'summary_ja': '確認 r4 合格(反証できなかった)。試験の組 542 回を自前に作り直して同じ値。新しい組 1,717 回 + 攻めの組 360 回(シールドあり)で衝突・止まり・'
                                '抜け出し・後ろへの動き 0、最小 3.995 cm(4 cm 未満は 2 回)。ラベルの問題 0(1,571,004 フレーム)',
                  'numbers': {'fresh_runs': 1717, 'adv_runs': 360, 'collisions': 0, 'min_cm': 3.995, 'noshield_adv_collisions': '23/360',
                              'vs_fastest_line_gap_cm': 15.8, 'lap1_line_vs_fastest_s': '8.99 vs 8.13'}},
        'caveats_ja': ['方策だけ(シールドなし)では、ゲート 32 (−,−) × 左寄りの端など角の一部でまだ崩れる(攻めの組で 23 回衝突)。シールドありでは全部有効・4.0 cm 以上',
                       'シールドの proposal_nominal(0.8 秒の先読みでは合格・止まるまでの先読みでは不合格の提案を通す)は試験・新・攻めで 3,896・12,970・3,978 フレーム。崩れは 0',
                       '右寄り・アウトの線の追従の 95% 点 17.9・15.3 cm(12 cm を越える。全体の L3 は合格)',
                       'ゲート 11 の開始で黒シートの横断が数えられず無効になることがある(1/1,717・1/542)'],
        'replaces': {'previous': '出していなかった(r2 は L2 1.33 cm で不合格・not_published)'},
        'candidate': 'experiments/20261013_sim_rl_modes/r3/candidates/line_r4/', 'runtime': runtime_info(),
        'note_ja': 'r4b の試験(protocol_line_r4b.json、1 回だけ)で L1〜L11 すべて合格し、確認 r4 でも反証できなかった。昇格の段(2026-10-15)が bundle/line/ に置き、登録簿で選べるようにした',
        'updated_at': NOW}
    return src, files, mode, status


def write(key, src, files, mode, status, extra=None, dry=False):
    dest = BUNDLE / key
    tmp = Path(tempfile.mkdtemp(prefix='.%s-' % key, dir=BUNDLE))
    os.chmod(tmp, 0o775)
    out = {}
    try:
        for f in files:
            s = (extra or {}).get(f) or (src / f)
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(s, tmp / f)
            out[f] = sha(tmp / f)
        if sha(tmp / 'policy.json') != mode['policy_sha256']:
            raise SystemExit('%s: policy.json の sha256 が mode.json と違います' % key)
        (tmp / 'mode.json').write_text(json.dumps(mode, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        (tmp / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        out['mode.json'] = sha(tmp / 'mode.json'); out['status.json'] = sha(tmp / 'status.json')
        cand = {f: sha(src / f) for f in ['mode.json', 'status.json'] if (src / f).exists()}
        (tmp / 'sha256.json').write_text(json.dumps({'files': out, 'candidate_originals': cand, 'written_by': 'r3/promote/write_bundles.py', 'at': NOW},
                                                    ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        if dry:
            print(key, 'dry-run', sorted(out))
            return out
        old = BUNDLE / ('.%s-old' % key)
        if old.exists():
            shutil.rmtree(old)
        if dest.exists():
            os.replace(dest, old)
        os.replace(tmp, dest)
        if old.exists():
            shutil.rmtree(old)
        print(key, '→', dest, len(out), 'ファイル')
        return out
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    res = {}
    s, f, m, st = no_wall(); res['no_wall'] = write('no_wall', s, f, m, st, dry=a.dry_run)
    s, f, m, st, ex = recover(); res['recover'] = write('recover', s, f, m, st, ex, dry=a.dry_run)
    s, f, m, st = line(); res['line'] = write('line', s, f, m, st, dry=a.dry_run)
    if not a.dry_run:
        (R3 / 'promote/bundles_written.json').write_text(json.dumps({'at': NOW, 'files': res}, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
