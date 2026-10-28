"""読むだけ(本物の data/raw は読まない・書かない): 手順書 3 章 5・7 章「起こし直した」に書く「シーンのタブの「走行の説明」に試し」の
2 つの書き方(memo= が無い説明の終わりに ` memo=試し(理由)`・memo= がある説明の memo= のすぐ後ろに `試し(理由)。`)を、夜の道具が
使う本物の関数(tools/section_scorecard.effective_description → tools/standing_3lap.trial_memo)が「試し」と読むかを、作り物の記録
(fix8/tmp_docs_r1/raw。終わって消す)で見る。シーンの説明 = scene.json の description(PATCH /api/scenes/<名前> が書く所)。"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TMP = HERE.parent / 'tmp_docs_r1' / 'raw_scene_memo'
sys.path.insert(0, '/home/jetson/jetformer/tools')
sys.path.insert(0, '/home/jetson/jetformer')
import section_scorecard as ss  # noqa: E402
import standing_3lap as s3  # noqa: E402

BENCH = ('bench=real_bench_20261023 day=2026-11-03 block=8 cand=B model=m battery=#A V_before=8.31 V_after=__ check=合格 '
         'official=1 memo=計画と違うので試し(元 cand=B): throttle_gain 0.92')
CASES = [
    # (名前, 記録の説明 = log.json, 書き換えたシーンの説明)
    ('朝の確かめ(説明欄が空・memo= なし)の終わりに memo=試し(…)', '', ' memo=試し(朝の発進の前進の確かめ)'),
    ('朝の確かめ(block の無い文)の終わりに memo=試し(…)', 'AI走行 No.1', 'AI走行 No.1 memo=試し(朝の発進の前進の確かめ)'),
    ('ベンチマークの本(自動の memo= あり)の memo= のすぐ後ろに 試し(…)。', BENCH,
     BENCH.replace('memo=', 'memo=試し(起こし直しで左 4.1 のまま)。', 1)),
    ('ベンチマークの本(memo= なし)の終わりに memo=試し(…)', BENCH.split(' memo=')[0],
     BENCH.split(' memo=')[0] + ' memo=試し(起こし直しで左 4.1 のまま)'),
]
ok = True
for k, (name, base, edited) in enumerate(CASES):
    log = '20261103-%02d0000' % (8 + k)
    root = TMP / log
    if root.exists():
        shutil.rmtree(root)
    for i in range(3):                       # シーン 3 つ。どれも記録の説明のまま
        p = root / ('%s-%04d' % (log, i))
        p.mkdir(parents=True)
        (p / 'scene.json').write_text(json.dumps({'description': base}, ensure_ascii=False), encoding='utf-8')
    before = s3.trial_memo(ss.effective_description(log, base, TMP)[0])
    # シーンのタブで 1 つ(2 つ目)だけ「走行の説明」を書き換えて保存した形
    (root / ('%s-%04d' % (log, 1)) / 'scene.json').write_text(json.dumps({'description': edited}, ensure_ascii=False),
                                                             encoding='utf-8')
    eff, info = ss.effective_description(log, base, TMP)
    after = s3.trial_memo(eff)
    good = before is None and bool(after) and '試し' in after
    ok &= good
    print('%s: 前 %r → 書いたあと %r(直しのシーン %s)%s' % (name, before, after, info['edited_scenes'], '' if good else '  ← 読めない'))
shutil.rmtree(TMP)
print('全部「試し」と読む' if ok else '読めない形がある')
sys.exit(0 if ok else 1)
