"""反証 2 回目・safety: リカバリーの「ブレーキ」の段(中立 + rev_us を 0.25 秒。前進の直後の後退側は必ずブレーキ = 捨てる 1 回目、
が前提)で、ESC がもう下がり始めた(直前の非中立がブレーキ。10/23 205458-0002 と同じ形)ときの assist.Recovery の出力。
車の動き(eRPM・衝撃)は決めた入力の列(閉ループの模型ではない)。装置は開かない。

  A: ブレーキの段の 0.1 秒で下がり始め、そのまま下がる(10/23 の記録の eRPM の形)
  B: A と同じに下がり始め、0.2 秒(ブレーキの段の中)で後ろの物に当たって(5G)止まる(eRPM 0 のまま)
結果は out/probe_recovery_brake_reverse.json"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from jetformer import assist as A   # noqa: E402

N, C = 1496.0, 1441.0
DT = 1 / 30
# 10/23 205458-0002 行 103〜125 の eRPM(リカバリーの始まりから)
REC_ERPM = [0, 0, 0, 1455, 1864, 1909, 2258, 2519, 2876, 3207, 3495, 2637, 3156, 3166, 3104, 3998, 3605, 3699, 3980, 4076,
            4246, 4372, 4558, 4607]


def run(case):
    a = A.Assist((C, N))
    t = 100.0
    # 走って armed → 止まる(0.8 秒で「止まった」)。モデルは中立の帯
    for k in range(10):
        a.step((C, N - 100), -2000, True, t, rec_on=True); t += DT
    k0 = 0
    while a.rec.phase is None and k0 < 60:
        a.step((C, N + 20), 0, True, t, rec_on=True); t += DT; k0 += 1
    # この呼び出しでリカバリーが始まった(phase = brake)。ここから入力の列
    rows = [{'s': 0.0, 'phase': a.rec.phase, 'out': N + 300.0, 'erpm': 0, 'impact_g': 0.0}]
    s = 0.0
    for i in range(1, 80):
        s = round(i * DT, 3)
        if case == 'A':
            e = REC_ERPM[i] if i < len(REC_ERPM) else 4600
            ig = 0.0
        else:
            e = REC_ERPM[i] if i < 6 else 0           # 0.2 秒で当たって止まる
            ig = 5.0 if i == 6 else 0.0
        o, kind = a.step((C, N + 20), e, True, t, rec_on=True, impact_g=ig, rev_us=300.0, rev_s=0.4)
        t += DT
        rows.append({'s': s, 'phase': a.rec.phase, 'out': round(o[1], 1), 'erpm': e, 'impact_g': ig})
        if a.rec.phase in (None, 'escape') and kind != A.ASSIST_RECOVERY or a.rec.phase == 'escape':
            break
    rev = [r for r in rows if r['out'] > N + 40]
    out = {'case': case, 'reverse_side_frames': len(rev), 'reverse_side_s': round(len(rev) * DT, 2),
           'reverse_side_while_backing_s': round(sum(DT for r in rev if r['erpm'] > 300), 2),
           'reverse_side_after_hit_s': round(sum(DT for r in rev if case == 'B' and r['s'] > 0.2), 2),
           'rev_us_seen': sorted(set(round(r['out'] - N) for r in rev)), 'rows': rows}
    return out


res = {c: run(c) for c in ('A', 'B')}
(Path(__file__).resolve().parent / 'out' / 'probe_recovery_brake_reverse.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
for c, r in res.items():
    print(c, {k: v for k, v in r.items() if k != 'rows'})
    print('   ', [(x['s'], x['phase'], x['out'], x['erpm']) for x in r['rows']])
