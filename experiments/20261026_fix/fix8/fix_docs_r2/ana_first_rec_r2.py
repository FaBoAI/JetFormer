"""fix8 係 docs 直す 8-2 回目: out/first_rec_r2_<set>.json(first_rec_r2.py の閉じたループ)と out/real_rec_r2.json(本物の記録)を数え、
その日を OFF にする「おかしな動き」の数を決める(f8r2-docs-1023-first-recovery-off。main session の決め 10/28 11:xx)。
  ① 下がらない: 止まってから NOBACK_S 秒たっても、リカバリーの始めの位置から 0.05 m 下がらない(noback_s > NOBACK_S か None)
  ② 長く下がる: 1 回のリカバリーで、リカバリーの始めの位置から BACK_M m より長く下がる(back_m > BACK_M)
ふつう(normal18・gapslow の ON)がどちらにも当たらず、知られている悪い形(k1_noback = 0 m の壁押し・k2_runaway = 止まらない後退)
が当たるか、と、1 m 以上下がるふつうの ON の数(前の「1 m 以上下がる → 止めて OFF」が当たる数)を出す。k3_limits は参考。
    python -I -B ana_first_rec_r2.py → 標準出力・out/ana_first_rec_r2.json
"""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOBACK_S = 3.0          # ① の秒(手順書の数)
BACK_M = 2.5            # ② の m(手順書の数)
SETS = ('normal18', 'gapslow', 'k1_noback', 'k2_runaway', 'k3_limits')


def st(x):
    x = sorted(v for v in x if v is not None)
    if not x:
        return None
    return {'n': len(x), 'min': round(x[0], 3), 'median': round(x[len(x) // 2], 3), 'max': round(x[-1], 3)}


def shape1(o):
    return o['rec'] is not None and (o['noback_s'] is None or o['noback_s'] > NOBACK_S)


def shape2(o):
    return o['rec'] is not None and o['back_m'] > BACK_M


def main():
    out = {'NOBACK_S': NOBACK_S, 'BACK_M': BACK_M, 'sets': {}}
    for name in SETS:
        p = HERE / 'out' / ('first_rec_r2_%s.json' % name)
        if not p.exists():
            continue
        J = json.loads(p.read_text())
        rows = J['rows']
        d = {'n_esc': len({r['esc'] for r in rows}), 'sha': J['sha']}
        for form in ('1023', 'wall', 'all'):
            R = [r for r in rows if form == 'all' or r['form'] == form]
            for sw in ('on', 'off'):
                ok = [r[sw] for r in R if r[sw]['rec'] is not None]
                d['%s_%s' % (form, sw)] = {
                    'runs': len(R), 'no_rec': len(R) - len(ok),
                    'back_m': st([o['back_m'] for o in ok]), 'ge1m': sum(o['back_m'] >= 1.0 for o in ok),
                    'noback_s': st([o['noback_s'] for o in ok]), 'noback_none': sum(o['noback_s'] is None for o in ok),
                    'esc_push_noback_s': st([o['esc_push_noback_s'] for o in ok]),
                    'esc_push_gt_0p05': sum(o['esc_push_noback_s'] > 0.05 for o in ok),
                    'shape1': sum(shape1(o) for o in ok), 'shape2': sum(shape2(o) for o in ok),
                    'either': sum(shape1(o) or shape2(o) for o in ok),
                    'giveup': sum(bool(o['giveup']) for o in ok), 'n_rec': dict(collections.Counter(o['n_rec'] for o in ok)),
                    't_1m': st([o['t_1m'] for o in ok]),
                }
        # 10/23 No.6 の本物の記録に合わせた模型(fit)だけの下がる距離(approvals 3「なぜ」の No.6 の幅)
        d['fit_only'] = {'%s_%s' % (form, sw): st([r[sw]['back_m'] for r in rows if r['esc'] == 'fit' and r['form'] == form
                                                    and r[sw]['rec'] is not None])
                         for form in ('1023', 'wall') for sw in ('on', 'off')}
        out['sets'][name] = d
    # 本物の記録(rev_s 0.4・rev_us 300 のリカバリー)
    rr = json.loads((HERE / 'out' / 'real_rec_r2.json').read_text())
    s300 = [r for r in rr if r['rev_us'] == 300.0 and r['rev_s'] == 0.4]
    out['real'] = {'n': len(s300), 'ai_through': [(r['log'], r['t_rec'], r['back_m'], r['noback_s']) for r in s300 if not r['human_after']],
                   'back_m': st([r['back_m'] for r in s300]), 'noback_s': st([r['noback_s'] for r in s300]),
                   'shape1': sum(1 for r in s300 if r['noback_s'] is None or r['noback_s'] > NOBACK_S),
                   'shape2': sum(1 for r in s300 if r['back_m'] > BACK_M),
                   'old_weak': [(r['log'], r['t_rec'], r['back_m'], r['noback_s']) for r in rr if r['rev_us'] is None]}
    for name, d in out['sets'].items():
        print('== %s(ESC %d 通り)' % (name, d['n_esc']))
        for form in ('1023', 'wall', 'all'):
            for sw in ('on', 'off'):
                x = d['%s_%s' % (form, sw)]
                print('  %-4s %-3s 走り %d・リカバリーなし %d | 下がる m %s・1 m 以上 %d | 止まって下がるまで 秒 %s・下がらない %d | '
                      '下がる前の escape の押し 秒 %s(0.05 秒より %d) | ①(> %.1f 秒) %d・②(> %.1f m) %d・どちらか %d | 中止 %d・始めの数 %s' % (
                          form, sw, x['runs'], x['no_rec'], x['back_m'], x['ge1m'], x['noback_s'], x['noback_none'],
                          x['esc_push_noback_s'], x['esc_push_gt_0p05'], NOBACK_S, x['shape1'], BACK_M, x['shape2'], x['either'],
                          x['giveup'], x['n_rec']))
    for name, d in out['sets'].items():
        if any(d['fit_only'].values()):
            print('  %s の fit(10/23 No.6 に合わせた模型)の下がる m: %s' % (name, d['fit_only']))
    r = out['real']
    print('== 本物の記録 rev_s 0.4・rev_us 300 のリカバリー %d: 下がる m %s・止まって下がるまで 秒 %s・① %d・② %d。AI 側のまま終わった %s' % (
        r['n'], r['back_m'], r['noback_s'], r['shape1'], r['shape2'], r['ai_through']))
    print('   10/10 の前の弱い後退(rev_us が log.json に無い本):', r['old_weak'])
    (HERE / 'out' / 'ana_first_rec_r2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
