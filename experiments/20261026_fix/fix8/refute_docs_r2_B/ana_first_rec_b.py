"""打ち消しの票 B(f8r2-docs-1023-first-recovery-off): 「その日の最初のリカバリーが 10/23 の形で ON が 1 m 以上下がる → 手順書どおり
止めて OFF → その日の残りは OFF」が、残りのリカバリーの後ろの当たりにどれだけ効くかを、
  (1) 案の係の閉じたループ(fix8/proposal_r1/out/sweep_r1_*of2.json。10/23 の形 288 場面・距離 0.1〜2.0 m。読むだけ)の今の ON・今の OFF の
      後ろに物が無いときの下がる距離(指摘の 288/288 の照らし)
  (2) 本物の 10/10〜10/24 のリカバリー(out/real_rec_b.json = real_rec_b.py)で当たった後ろの物の距離(止まった所から)と、
      当たらなかったリカバリーの「少なくともここまでは物が無い」距離
  (3) (2) の本物の距離ごとに、(1) の模型で ON と OFF のどちらが強く当たるか(0.1 m/s より速い・片方だけ当たる)
で照らす。 python -I -B ana_first_rec_b.py → out/ana_first_rec_b.txt
"""
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path('/home/jetson/jetformer/experiments/20261026_fix/fix8/proposal_r1/out')


def grid(ds, x):
    return min(ds, key=lambda d: abs(d - x))


def cmp_at(sel, d):
    on_w = off_w = same = on_hit = off_hit = 0
    von, voff = [], []
    for r in sel:
        a = r['now_on']['hit'][str(d)]
        b = r['now_off']['hit'][str(d)]
        on_hit += a is not None
        off_hit += b is not None
        if a is not None:
            von.append(a)
        if b is not None:
            voff.append(b)
        if a is not None and (b is None or a > b + 0.1):
            on_w += 1
        elif b is not None and (a is None or b > a + 0.1):
            off_w += 1
        else:
            same += 1
    med = lambda v: ('%.2f' % statistics.median(v)) if v else '—'   # noqa: E731
    return on_w, off_w, same, on_hit, off_hit, med(von), med(voff)


def main():
    rows, ds = [], None
    for f in sorted(SRC.glob('sweep_r1_*of2.json')):
        J = json.loads(f.read_text())
        ds = J['ds']
        rows += J['rows']
    R = json.loads((HERE / 'out' / 'real_rec_b.json').read_text())
    out = []
    out.append('# (1) 閉じたループ 10/23 の形 %d 場面(fix8/proposal_r1/out/sweep_r1_*of2.json)' % len(rows))
    for name, sel in (('全部', rows), ('fit 16', [r for r in rows if r['esc'] == 'fit'])):
        for f in ('now_on', 'now_off'):
            b = [r[f]['back'] for r in sel]
            out.append('  %s %s: 後ろに物が無いときの下がる距離 %.3f〜%.3f m(中央 %.3f)・1.0 m 以上 %d/%d'
                       % (name, f, min(b), max(b), statistics.median(b), sum(x >= 1.0 for x in b), len(b)))
    out.append('')
    out.append('# (2) 本物のリカバリー(data/raw 10/10〜10/24。assist == 2。out/real_rec_b.json)')
    out.append('  日ごと: ' + json.dumps(R['days'], ensure_ascii=False))
    out.append('  その日の最初のリカバリー: ' + json.dumps(R['first_recovery_of_day'], ensure_ascii=False))
    obs, cens = [], []
    for r in R['rows']:
        h = r['rear_hit']
        form = '10/23 の形(始めて %.2f 秒で下がり、1 m 以上)' % r['first_back_s'] if (
            r['first_back_s'] is not None and r['first_back_s'] < 0.15 and r['back_m_rec'] >= 1.0) else 'ほか'
        if h:
            obs.append(h['back_m'])
            out.append('  %s@%.1f 秒: 後ろの物 %.3f m で %.2fG・%.2f m/s(%s)・%s'
                       % (r['log'], r['start_s'], h['back_m'], h['g'], h['v_mps'], 'AI 側' if h['ai_side'] else '人側', form))
        else:
            cens.append(r['back_m_incl_after'])
            out.append('  %s@%.1f 秒: 後ろの当たりなし・少なくとも %.3f m まで物なし・%s' % (r['log'], r['start_s'], r['back_m_incl_after'], form))
    out.append('  当たった後ろの物 %d 回: %s m(1.0 m より近い %d 回)。当たらなかった %d 回: 下がった距離 %s m'
               % (len(obs), '・'.join('%.2f' % x for x in sorted(obs)), sum(x < 1.0 for x in obs), len(cens),
                  '・'.join('%.2f' % x for x in sorted(cens))))
    out.append('  (No.6 = 20261023-205458 の後ろの物は、止まった所からは約 1.16〜1.22 m = fix8/fix_safety_r1 facts12)')
    out.append('')
    out.append('# (3) 本物の後ろの物の距離ごとに、10/23 の形の模型で ON と OFF のどちらが強く当たるか(ON 悪 / OFF 悪 / 同じ・当たる数 ON/OFF・速さの中央 ON/OFF)')
    pts = sorted(set([grid(ds, x) for x in obs] + [1.2]))
    for name, sel in (('全部 288', rows), ('fit 16', [r for r in rows if r['esc'] == 'fit'])):
        tot_on = tot_off = 0
        for d in pts:
            on_w, off_w, same, on_hit, off_hit, mon, moff = cmp_at(sel, d)
            if d in [grid(ds, x) for x in obs]:
                tot_on += on_w
                tot_off += off_w
            out.append('  %s d %.2f m: ON 悪 %d・OFF 悪 %d・同じ %d・当たる ON %d / OFF %d・速さ ON %s / OFF %s m/s'
                       % (name, d, on_w, off_w, same, on_hit, off_hit, mon, moff))
        out.append('  %s 本物の当たった 4 つの距離の合計: ON 悪 %d・OFF 悪 %d' % (name, tot_on, tot_off))
    txt = '\n'.join(out) + '\n'
    (HERE / 'out' / 'ana_first_rec_b.txt').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
