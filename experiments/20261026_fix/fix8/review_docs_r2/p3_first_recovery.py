"""反証 8-2 docs: 手順書 3 章「その日の最初のリカバリーを人が見る」の行の「1 m 以上下がる・後ろの物に近づく → すぐ人側 → 止めて OFF」と、
新しく入った「10/23 の形の後ろの近い物」の行(約 1 m より遠い物には ON のほうが遅く当たるか当たらない・先に人側にしただけなら OFF に
しない = OFF にするかは「最初のリカバリー」の行で決める)を、案の係の閉じたループの出力(fix8/proposal_r1/out/sweep_r1_*of2.json。
10/23 の形の 288 場面 = 18 の ESC の模型 × 16 場面。読むだけ)で照らす。

  ① 10/23 の形で、後ろに物が無いときの下がる距離(back)。今の ON・今の OFF。1 m 以上の場面の数(3 章の「1 m 以上下がる」に当たる)
  ② 止まった所から 1.0〜2.0 m の後ろの物(人が「先に人側」にしない距離)で、今の ON・今の OFF の当たり(速さ)
  ③ fit(10/23 No.6 の記録に合わせた模型)の 16 場面だけの ① ②

    python3 -I -B p3_first_recovery.py → out/p3_first_recovery.txt
"""
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path('/home/jetson/jetformer/experiments/20261026_fix/fix8/proposal_r1/out')


def main():
    rows, ds = [], None
    for f in sorted(SRC.glob('sweep_r1_*of2.json')):
        J = json.loads(f.read_text())
        ds = J['ds']
        rows += J['rows']
    out = ['# 元: fix8/proposal_r1/out/sweep_r1_0of2.json・sweep_r1_1of2.json(10/23 の形 %d 場面。後ろの物は置かずに走り、'
           '止まった所から d m で最初に当たる速さ = 票 A の hit_profile)' % len(rows)]
    for name, sel in (('全部の模型(288 場面)', rows), ('fit の 16 場面', [r for r in rows if r['esc'] == 'fit'])):
        out.append('== ' + name)
        for f in ('now_on', 'now_off'):
            b = [r[f]['back'] for r in sel]
            out.append('  ① %s: 後ろに物が無いときの下がる距離 最短 %.3f・中央 %.3f・最長 %.3f m。1.0 m 以上 %d / %d 場面'
                       % (f, min(b), statistics.median(b), max(b), sum(1 for x in b if x >= 1.0), len(b)))
        far = [d for d in ds if d >= 1.0 - 1e-9]
        on_hit = sum(1 for r in sel for d in far if r['now_on']['hit'][str(d)] is not None)
        off_hit = sum(1 for r in sel for d in far if r['now_off']['hit'][str(d)] is not None)
        on_worse = sum(1 for r in sel for d in far
                       if r['now_on']['hit'][str(d)] is not None and (r['now_off']['hit'][str(d)] is None
                                                                     or r['now_on']['hit'][str(d)] > r['now_off']['hit'][str(d)] + 0.1))
        off_worse = sum(1 for r in sel for d in far
                        if r['now_off']['hit'][str(d)] is not None and (r['now_on']['hit'][str(d)] is None
                                                                       or r['now_off']['hit'][str(d)] > r['now_on']['hit'][str(d)] + 0.1))
        n = len(sel) * len(far)
        out.append('  ② 後ろの物 1.0〜2.0 m(%d 組): 当たる ON %d・OFF %d。ON のほうが悪い(ON だけ当たる・0.1 m/s より速い)%d・'
                   'OFF のほうが悪い %d' % (n, on_hit, off_hit, on_worse, off_worse))
        for d in (1.2, 1.5, 1.8):
            v_on = [r['now_on']['hit'][str(d)] for r in sel]
            v_off = [r['now_off']['hit'][str(d)] for r in sel]
            out.append('     d %.1f m: 当たる ON %d / OFF %d・速さの中央(当たった組)ON %s / OFF %s m/s' % (
                d, sum(v is not None for v in v_on), sum(v is not None for v in v_off),
                ('%.2f' % statistics.median([v for v in v_on if v is not None])) if any(v is not None for v in v_on) else '—',
                ('%.2f' % statistics.median([v for v in v_off if v is not None])) if any(v is not None for v in v_off) else '—'))
    txt = '\n'.join(out) + '\n'
    (HERE / 'out' / 'p3_first_recovery.txt').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
