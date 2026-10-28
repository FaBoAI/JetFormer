"""反証 8-1 回目・係 rehearsal: 手順書が「」で引く画面・道具の文が、今のコード(画面・サーバー・夜の道具)に本当にあるかを照らす(読むだけ)。
引いた文を「…」・N・X・Y・HH:MM などの置き場所で切り、4 字以上の切れ端が 1 つでもコードの文字に無いものを出す(人が読んで判断する)。

    python -I -B quote_check.py
"""
import re
from pathlib import Path

JF = Path('/home/jetson/jetformer')
RB = Path('/home/jetson/jetformer_reports/fix_20261026/runbook_20261103.md')
SRC = [JF / 'jetformer/web/static/app.js', JF / 'jetformer/web/static/index.html', JF / 'jetformer/web/server.py',
       JF / 'jetformer/web/bench.py', JF / 'jetformer/drive.py', JF / 'jetformer/assist.py',
       JF / 'tools/car_check.py', JF / 'tools/standing_3lap.py', JF / 'tools/quick_eval_by_params.py', JF / 'tools/bench_nightly.sh',
       JF / 'tools/fw_vbat_check.py', JF / 'tools/section_scorecard.py',
       JF / 'experiments/20261026_fix/diag_launch/collect.py', JF / 'experiments/20261026_fix/plan/runs_1103.py',
       JF / 'experiments/20261024_tune/quick_eval.py']
CODE = '\n'.join(p.read_text(encoding='utf-8') for p in SRC)
# コードの文は '%s'・'%d'・${…}・{…} の置き場所があるので、比べるときは置き場所の前後で切る
CODE_N = re.sub(r'\s+', ' ', CODE)
HOLES = r'…|\.\.\.|〜|\bN\b|\bM\b|\bX\b|\bY\b|\bZ\b|HH:MM(:SS)?|MM/DD|\d+(\.\d+)?|<[^>]*>|\(…\)|(…)'


def pieces(q):
    return [x.strip(' 、。()()「」') for x in re.split(HOLES, q) if x and len(x.strip(' 、。()()「」')) >= 4]


def main():
    s = RB.read_text(encoding='utf-8')
    out, seen = [], set()
    for ln_no, ln in enumerate(s.splitlines(), 1):
        for q in re.findall(r'「([^「」]{4,200})」', ln):
            if q in seen:
                continue
            seen.add(q)
            miss = [p for p in pieces(q) if p not in CODE_N and p not in CODE]
            if miss:
                out.append('%d: 「%s」 → 無い切れ端 %s' % (ln_no, q, miss))
    txt = '\n'.join(out) + '\n'
    (Path(__file__).resolve().parent / 'out' / 'quote_check.txt').write_text(txt, encoding='utf-8')
    print(txt)
    print(len(out), '件(人が読んで判断する)')


if __name__ == '__main__':
    main()
