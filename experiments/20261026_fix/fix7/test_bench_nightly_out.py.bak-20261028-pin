"""tools/bench_nightly.sh の出力先の試験(2026-10-26 fix2)。
前は既定の --report が ~/jetformer_reports/<bench> そのもので、11/3 の夜に既定のまま回すと 10/23 の README.md・report.json を
上書きした。今は既定を日付つきの新しい場所(<bench>/report_<日>)にし、既定の場所に前の結果があれば何も書かずに終了コード 3 で止まる。
--out・--report・-- のあとの bench_report の選択肢(--logs など)の使い方は今まで通り(渡した場所は上書きする: quick_eval が回し直す)。

台本の写しを tmp に置き、tmp の data/raw(空のログの dir 1 本)と偽物の jf-heavy(渡されたコマンドを書くだけ)で回す。
本物の data/raw・data/labels・~/jetformer_reports には触れない。GPU・torch は使わない。

2026-10-26 fix3(係 tools、反証 1 回目 code:F2〜F5)で足した試験:
  F2 写しが JF_REPORTS を読むかを確かめ、写しの中の本物の場所も tmp に置き換える(直す前の HEAD の台本でも本物に書かない)
  F3 既定の report_<日> を新しく作るとき、前の報告の deviations.md の「## 人が書いたこと」を写す(--report を渡したときは写さない)
  F4 記録が無い日は既定の出力先に何も作らない(次の既定の回が終了コード 3 で止まらない)・空の dir は前の結果にしない
  F5 段 3・4 が 0 以外で終わったら、残っている前の回の cap_replay.json・brake_table.json をレポートに渡さない

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q tests/test_bench_nightly_out.py
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOG1 = '20001024-120000'
REAL_REPORTS = '/home/jetson/jetformer_reports'


@pytest.fixture(autouse=True)
def real_reports_untouched():
    """本物の ~/jetformer_reports/x(試験の bench 名)を作らない(2026-10-26 fix3、反証 1 回目 code:F2)"""
    x = Path(REAL_REPORTS) / 'x'
    before = x.exists()
    yield
    assert before or not x.exists(), '試験が本物の %s を作った' % x


def sandbox(tmp_path, src=None, check=True):
    """tmp/jf/tools/bench_nightly.sh(写し)・tmp/jf/data/raw/20001024-120000/・偽物の jf-heavy・候補の表 bench x(2 日)。
    2026-10-26 fix3(反証 1 回目 code:F2): 前は出力先を tmp にするのが環境変数 JF_REPORTS だけで、JF_REPORTS を読まない台本
    (直す前の HEAD の台本)で回すと本物の ~/jetformer_reports/x/nightly_20001024/nightly.log を作った(10/26 14:06・14:42)。
    今は (1) 写しが JF_REPORTS を読むことを確かめ(check)、(2) 写しの中の本物の場所を tmp に置き換え、(3) HOME も tmp にする。
    偽物の jf-heavy は、FAKE_FAIL(正規表現)に合うコマンドだけ終了コード 1 にする"""
    jf = tmp_path / 'jf'
    (jf / 'tools').mkdir(parents=True)
    text = Path(src or ROOT / 'tools' / 'bench_nightly.sh').read_text(encoding='utf-8')
    if check:                  # 説明の行ではなく、変数として読んでいること
        assert re.search(r'\$\{?JF_REPORTS\b', text), '台本が JF_REPORTS を読まない(試験が本物の ~/jetformer_reports に書いてしまう)'
    text = text.replace(REAL_REPORTS, str(tmp_path / 'hardcoded_reports'))
    (jf / 'tools' / 'bench_nightly.sh').write_text(text, encoding='utf-8')
    (jf / 'tools' / 'bench_nightly.sh').chmod(0o755)
    (jf / 'data' / 'raw' / LOG1).mkdir(parents=True)
    heavy = tmp_path / 'fake_heavy.sh'
    heavy.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$FAKE_HEAVY_LOG"\n'
                     'if [ -n "${FAKE_FAIL:-}" ] && printf "%s" "$*" | grep -Eq "$FAKE_FAIL"; then exit 1; fi\nexit 0\n')
    heavy.chmod(0o755)
    bench = tmp_path / 'bench.json'
    bench.write_text(json.dumps({'bench': 'x', 'days': [{'date': '2000-10-23', 'blocks': []}, {'date': '2000-10-24', 'blocks': []}]}))
    (tmp_path / 'home').mkdir()
    env = dict(os.environ, HEAVY=str(heavy), PY=sys.executable, JF_REPORTS=str(tmp_path / 'reports'),
               FAKE_HEAVY_LOG=str(tmp_path / 'heavy.log'), HOME=str(tmp_path / 'home'))
    env.pop('FAKE_FAIL', None)
    return jf, bench, env


def nightly(jf, env, *args):
    return subprocess.run(['bash', str(jf / 'tools' / 'bench_nightly.sh')] + [str(a) for a in args],
                          capture_output=True, text=True, timeout=120, env=env)


def heavy_cmds(tmp_path):
    p = tmp_path / 'heavy.log'
    return p.read_text(encoding='utf-8').splitlines() if p.exists() else []


def report_cmd(tmp_path):
    cs = [c for c in heavy_cmds(tmp_path) if 'tools/bench_report.py' in c]
    assert len(cs) == 1, heavy_cmds(tmp_path)
    return cs[0]


def test_default_report_is_dated_and_old_root_report_untouched(tmp_path):
    jf, bench, env = sandbox(tmp_path)
    root = tmp_path / 'reports' / 'x'
    root.mkdir(parents=True)
    (root / 'README.md').write_text('10/23 の報告')                       # 前の既定の場所(10/23 の報告)
    (root / 'report.json').write_text('{"old": 1}')
    r = nightly(jf, env, '20001024', '--bench', bench)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (root / 'README.md').read_text() == '10/23 の報告' and (root / 'report.json').read_text() == '{"old": 1}'
    cmd = report_cmd(tmp_path)
    assert cmd.endswith('--out %s' % (root / 'report_20001024'))           # 日付つきの新しい場所
    assert '--day 20001023 --day 20001024' in cmd                           # 1 日目は days から(今まで通り)
    log = (root / 'nightly_20001024' / 'nightly.log').read_text(encoding='utf-8').splitlines()
    assert '1 日目 20001023、bench_candidates.json の days から' in log[0]
    assert log[1] == '出力 %s(既定)・レポート %s(既定)' % (root / 'nightly_20001024', root / 'report_20001024')


def test_stops_before_overwriting_default_report(tmp_path):
    jf, bench, env = sandbox(tmp_path)
    rep = tmp_path / 'reports' / 'x' / 'report_20001024'
    rep.mkdir(parents=True)
    (rep / 'README.md').write_text('前の結果')
    r = nightly(jf, env, '20001024', '--bench', bench)
    assert r.returncode == 3 and '止めます' in r.stderr and str(rep) in r.stderr
    assert (rep / 'README.md').read_text() == '前の結果'
    assert not (tmp_path / 'reports' / 'x' / 'nightly_20001024').exists()     # 何も書かない
    assert heavy_cmds(tmp_path) == []                                          # 何も回さない


def test_stops_before_overwriting_default_nightly(tmp_path):
    jf, bench, env = sandbox(tmp_path)
    out = tmp_path / 'reports' / 'x' / 'nightly_20001024'
    out.mkdir(parents=True)
    (out / 'nightly.log').write_text('前の夜')
    r = nightly(jf, env, '20001024', '--bench', bench)
    assert r.returncode == 3 and str(out) in r.stderr
    assert (out / 'nightly.log').read_text() == '前の夜' and heavy_cmds(tmp_path) == []
    # --report だけ渡しても、既定の --out に前の結果があれば止まる
    r = nightly(jf, env, '20001024', '--bench', bench, '--report', tmp_path / 'r')
    assert r.returncode == 3 and (out / 'nightly.log').read_text() == '前の夜' and not (tmp_path / 'r').exists()


def test_overwrite_flag_allows_default(tmp_path):
    jf, bench, env = sandbox(tmp_path)
    root = tmp_path / 'reports' / 'x'
    for d, f in (('nightly_20001024', 'nightly.log'), ('report_20001024', 'README.md')):
        (root / d).mkdir(parents=True)
        (root / d / f).write_text('前')
    r = nightly(jf, env, '20001024', '--bench', bench, '--overwrite')
    assert r.returncode == 0, r.stdout + r.stderr
    assert (root / 'nightly_20001024' / 'nightly.log').read_text(encoding='utf-8').startswith('# bench_nightly 20001024')
    assert report_cmd(tmp_path).endswith('--out %s' % (root / 'report_20001024'))


def test_explicit_out_report_and_extra_args_unchanged(tmp_path):
    """--out・--report を渡したら、前の結果があっても今まで通り上書きする(quick_eval が同じ場所で回し直す)。
    -- のあとの選択肢は bench_report の最後にそのまま付く"""
    jf, bench, env = sandbox(tmp_path)
    out, rep = tmp_path / 'n', tmp_path / 'r'
    out.mkdir()
    rep.mkdir()
    (out / 'nightly.log').write_text('前の夜')
    (rep / 'README.md').write_text('前の結果')
    r = nightly(jf, env, '20001024', '--bench', bench, '--out', out, '--report', rep, '--',
                '--desk', 'D.json', '--logs', LOG1, '--k-first', '0', '--min-laps', '1')
    assert r.returncode == 0, r.stdout + r.stderr
    log = (out / 'nightly.log').read_text(encoding='utf-8').splitlines()
    assert log[0].startswith('# bench_nightly 20001024') and log[1] == '出力 %s・レポート %s' % (out, rep)
    cmd = report_cmd(tmp_path)
    assert cmd.endswith('--out %s --desk D.json --logs %s --k-first 0 --min-laps 1' % (rep, LOG1))
    cs = heavy_cmds(tmp_path)
    assert any('tools/section_scorecard.py --day 20001024' in c and c.endswith('--out %s/scorecard' % out) for c in cs)
    assert not (tmp_path / 'reports').exists()                               # 既定の場所には何も作らない


def test_no_records_and_usage_unchanged(tmp_path):
    jf, bench, env = sandbox(tmp_path)
    r = nightly(jf, env, '20001023', '--bench', bench, '--out', tmp_path / 'n', '--report', tmp_path / 'r')
    assert r.returncode == 1 and 'data/raw/20001023-* がありません' in r.stdout
    r = nightly(jf, env)
    assert r.returncode == 2 and '--overwrite' in r.stdout and '環境変数' in r.stdout     # 使い方(頭の説明)を全部出す
    r = nightly(jf, env, '20001024', '--nope')
    assert r.returncode == 2 and '知らない選択肢' in r.stderr


# ------------------------------------------------------------------ 2026-10-26 fix3(係 tools)
def _head_script(tmp_path):
    """直す前(commit 81a4d34 = HEAD)の台本。JF_REPORTS を読まず、出力先が本物の ~/jetformer_reports に決め打ち"""
    r = subprocess.run(['git', '-C', str(ROOT), 'show', 'HEAD:tools/bench_nightly.sh'], capture_output=True, text=True)
    if r.returncode != 0 or REAL_REPORTS not in r.stdout:
        pytest.skip('HEAD の台本が読めない')
    p = tmp_path / 'head_bench_nightly.sh'
    p.write_text(r.stdout, encoding='utf-8')
    return p


def test_f2_sandbox_refuses_script_without_jf_reports(tmp_path):
    """反証 1 回目 code:F2: JF_REPORTS を読まない台本は、回す前に試験が止まる"""
    head = _head_script(tmp_path)
    assert 'JF_REPORTS' not in head.read_text(encoding='utf-8')
    with pytest.raises(AssertionError, match='JF_REPORTS'):
        sandbox(tmp_path / 'a', src=head)
    # 説明の行に名前だけ残して、変数を読まなくなった台本も止まる
    cur = (ROOT / 'tools' / 'bench_nightly.sh').read_text(encoding='utf-8')
    bad = tmp_path / 'no_var.sh'
    bad.write_text(cur.replace('REPORTS_ROOT=${JF_REPORTS:-%s}' % REAL_REPORTS, 'REPORTS_ROOT=%s' % REAL_REPORTS), encoding='utf-8')
    assert 'JF_REPORTS' in bad.read_text(encoding='utf-8') and bad.read_text(encoding='utf-8') != cur
    with pytest.raises(AssertionError, match='JF_REPORTS'):
        sandbox(tmp_path / 'b', src=bad)


def test_f2_old_script_writes_only_into_tmp(tmp_path):
    """確かめを外して HEAD の台本を回しても、本物の ~/jetformer_reports ではなく tmp に書く(本物の x は autouse の fixture が見る)"""
    head = _head_script(tmp_path)
    jf, bench, env = sandbox(tmp_path / 'b', src=head, check=False)
    r = nightly(jf, env, '20001024', '--bench', bench)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / 'b' / 'hardcoded_reports' / 'x' / 'nightly_20001024' / 'nightly.log').exists()
    assert REAL_REPORTS not in (jf / 'tools' / 'bench_nightly.sh').read_text(encoding='utf-8')


def test_f4_no_records_default_creates_nothing_and_next_run_goes(tmp_path):
    """反証 1 回目 code:F4: 記録の無い日に既定のまま回す → 終了コード 1・既定の出力先を作らない。記録が来たあとの次の既定の回は進む
    (前は nightly.log だけの dir が残り、次の回が「前の結果があります」で終了コード 3)"""
    jf, bench, env = sandbox(tmp_path)
    r = nightly(jf, env, '20001025', '--bench', bench)
    assert r.returncode == 1 and 'data/raw/20001025-* がありません' in r.stdout
    assert not (tmp_path / 'reports' / 'x' / 'nightly_20001025').exists()
    assert not (tmp_path / 'reports').exists() and heavy_cmds(tmp_path) == []
    (jf / 'data' / 'raw' / '20001025-090000').mkdir(parents=True)
    r = nightly(jf, env, '20001025', '--bench', bench)
    assert r.returncode == 0, r.stdout + r.stderr
    assert report_cmd(tmp_path).endswith('--out %s' % (tmp_path / 'reports' / 'x' / 'report_20001025'))


def test_f4_empty_default_dirs_are_not_previous_results(tmp_path):
    """空の既定の dir は「前の結果」にしない(`ls -A` の確かめ。dir があるだけで止めると、この試験が落ちる)"""
    jf, bench, env = sandbox(tmp_path)
    for d in ('nightly_20001024', 'report_20001024'):
        (tmp_path / 'reports' / 'x' / d).mkdir(parents=True)
    r = nightly(jf, env, '20001024', '--bench', bench)
    assert r.returncode == 0, r.stdout + r.stderr
    assert report_cmd(tmp_path).endswith('--out %s' % (tmp_path / 'reports' / 'x' / 'report_20001024'))


def _old_stage_json(out):
    for d, f in (('cap_replay', 'cap_replay.json'), ('brake_table', 'brake_table.json')):
        (out / d).mkdir(parents=True)
        (out / d / f).write_text('{"run": "前の回"}', encoding='utf-8')


def test_f5_failed_stage3_4_do_not_pass_old_json(tmp_path):
    """反証 1 回目 code:F5: 同じ --out で回し直し、段 3・4 が落ちた(終了コード 1)→ 前の回の JSON をレポートに渡さない・注意を書く
    (前は [ -f ] だけを見て、前の回の G1・ブレーキの表をこの回のレポートに使った)"""
    jf, bench, env = sandbox(tmp_path)
    out, rep = tmp_path / 'n', tmp_path / 'r'
    _old_stage_json(out)
    env['FAKE_FAIL'] = 'tools/(cap_replay|brake_table)\\.py'
    r = nightly(jf, env, '20001024', '--bench', bench, '--out', out, '--report', rep)
    assert r.returncode == 0, r.stdout + r.stderr
    cmd = report_cmd(tmp_path)
    assert '--cap-replay' not in cmd and '--brake' not in cmd
    log = (out / 'nightly.log').read_text(encoding='utf-8')
    assert '注意: 段 3(G1)が終了コード 1' in log and '注意: 段 4(ブレーキの表)が終了コード 1' in log
    # 片方だけ落ちたとき: 落ちなかった方は今まで通り渡す
    (tmp_path / 'heavy.log').unlink()
    env['FAKE_FAIL'] = 'tools/brake_table\\.py'
    r = nightly(jf, env, '20001024', '--bench', bench, '--out', out, '--report', rep)
    cmd = report_cmd(tmp_path)
    assert '--cap-replay %s' % (out / 'cap_replay' / 'cap_replay.json') in cmd and '--brake' not in cmd


def test_f5_ok_stages_pass_json_as_before(tmp_path):
    """段 3・4 が 0 で終わったら、今まで通り --cap-replay・--brake を渡す(既定の場所 + --overwrite でも同じ)"""
    jf, bench, env = sandbox(tmp_path)
    out, rep = tmp_path / 'n', tmp_path / 'r'
    _old_stage_json(out)
    r = nightly(jf, env, '20001024', '--bench', bench, '--out', out, '--report', rep)
    assert r.returncode == 0, r.stdout + r.stderr
    cmd = report_cmd(tmp_path)
    assert ('--cap-replay %s --brake %s' % (out / 'cap_replay' / 'cap_replay.json', out / 'brake_table' / 'brake_table.json')) in cmd
    assert '注意: 段' not in (out / 'nightly.log').read_text(encoding='utf-8')
    (tmp_path / 'heavy.log').unlink()
    dflt = tmp_path / 'reports' / 'x' / 'nightly_20001024'
    _old_stage_json(dflt)
    env['FAKE_FAIL'] = 'tools/cap_replay\\.py'
    r = nightly(jf, env, '20001024', '--bench', bench, '--overwrite')
    cmd = report_cmd(tmp_path)
    assert '--cap-replay' not in cmd and '--brake %s' % (dflt / 'brake_table' / 'brake_table.json') in cmd


NOTES_1023 = ('# 外れたこと(protocol.json から)\n\n- 自動で見つけたもの(tools/bench_report.py、2000-10-23T23:00:00)。\n\n'
              '## 自動\n\n- なし\n\n## 人が書いたこと\n\n\n- 注 A: 説明欄の直し\n- 注 B: 電池の欄が未記入\n')


def _write_report_like_bench_report(out_dir, monkeypatch):
    """本物の bench_report.write_report(図・README・report.json だけ偽物)で deviations.md を書き直す"""
    sys.path.insert(0, str(ROOT / 'tools'))
    import bench_report as br
    monkeypatch.setattr(br, 'make_figs', lambda rep, d: [])
    monkeypatch.setattr(br, 'to_markdown', lambda rep: '# 偽物の README\n')
    monkeypatch.setattr(br, 'report_json', lambda rep: {'fake': 1})
    br.write_report({'created_at': '2000-10-24T23:00:00', 'deviations': ['自動の外れ 1']}, out_dir)
    return (Path(out_dir) / 'deviations.md').read_text(encoding='utf-8')


def test_f3_default_report_carries_human_notes(tmp_path, monkeypatch):
    """反証 1 回目 code:F3: 既定の report_<日> を新しく作るとき、<bench>/deviations.md の「## 人が書いたこと」を写す。
    bench_report が書き直したあとも人の注が残る。次の日は前の日の report_<日> から写す(写したことの行は 1 行だけ)"""
    jf, bench, env = sandbox(tmp_path)
    root = tmp_path / 'reports' / 'x'
    root.mkdir(parents=True)
    (root / 'deviations.md').write_text(NOTES_1023, encoding='utf-8')
    r = nightly(jf, env, '20001024', '--bench', bench)
    assert r.returncode == 0, r.stdout + r.stderr
    dv = root / 'report_20001024' / 'deviations.md'
    s = dv.read_text(encoding='utf-8')
    assert s.splitlines()[0] == '# 外れたこと(bench_nightly.sh が用意した。上は bench_report が書き直す)'
    assert '- 注 A: 説明欄の直し\n- 注 B: 電池の欄が未記入' in s and '(bench_nightly.sh が写した) %s から' % (root / 'deviations.md') in s
    assert '人が書いたこと: %s から' % (root / 'deviations.md') in (root / 'nightly_20001024' / 'nightly.log').read_text(encoding='utf-8')
    assert (root / 'deviations.md').read_text(encoding='utf-8') == NOTES_1023                # 元は変えない
    s2 = _write_report_like_bench_report(root / 'report_20001024', monkeypatch)
    assert '## 自動\n\n- 自動の外れ 1\n' in s2 and '- 注 A: 説明欄の直し\n- 注 B: 電池の欄が未記入' in s2
    assert s2.index('## 自動') < s2.index('## 人が書いたこと') < s2.index('注 A')
    # 次の日: report_20001024 に人が注 C を足した → report_20001025 は A・B・C(写したことの行は 1 行)
    dv.write_text(s2 + '- 注 C: 2 日目の注\n', encoding='utf-8')
    (jf / 'data' / 'raw' / '20001025-090000').mkdir(parents=True)
    r = nightly(jf, env, '20001025', '--bench', bench)
    assert r.returncode == 0, r.stdout + r.stderr
    s3 = (root / 'report_20001025' / 'deviations.md').read_text(encoding='utf-8')
    assert all(x in s3 for x in ('注 A', '注 B', '注 C')) and s3.count('(bench_nightly.sh が写した)') == 1
    assert '%s から' % dv in s3


def test_f3_explicit_report_and_no_notes_unchanged(tmp_path):
    """--report を渡したとき・前の報告に人の注が無いときは、今まで通り何も写さない"""
    jf, bench, env = sandbox(tmp_path)
    root = tmp_path / 'reports' / 'x'
    root.mkdir(parents=True)
    (root / 'deviations.md').write_text(NOTES_1023, encoding='utf-8')
    r = nightly(jf, env, '20001024', '--bench', bench, '--out', tmp_path / 'n', '--report', tmp_path / 'r')
    assert r.returncode == 0 and not (tmp_path / 'r').exists()
    (root / 'deviations.md').write_text(NOTES_1023.split('## 人が書いたこと')[0] + '## 人が書いたこと\n\n', encoding='utf-8')
    r = nightly(jf, env, '20001024', '--bench', bench)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (root / 'report_20001024' / 'deviations.md').exists()
    assert '人が書いたこと: 前の報告に無い' in (root / 'nightly_20001024' / 'nightly.log').read_text(encoding='utf-8')


def test_f3_seed_only_report_dir_is_not_previous_result(tmp_path):
    """前の回が段 5 の前に落ちて、写しただけの deviations.md が残った既定の report_<日> は「前の結果」にしない(写し直す)。
    bench_report が書いた deviations.md なら今まで通り止まる"""
    jf, bench, env = sandbox(tmp_path)
    root = tmp_path / 'reports' / 'x'
    rep = root / 'report_20001024'
    rep.mkdir(parents=True)
    (root / 'deviations.md').write_text(NOTES_1023, encoding='utf-8')
    (rep / 'deviations.md').write_text('# 外れたこと(bench_nightly.sh が用意した。上は bench_report が書き直す)\n\n## 人が書いたこと\n\n- 古い\n',
                                       encoding='utf-8')
    r = nightly(jf, env, '20001024', '--bench', bench)
    assert r.returncode == 0, r.stdout + r.stderr
    s = (rep / 'deviations.md').read_text(encoding='utf-8')
    assert '注 A' in s and '- 古い' not in s
    (rep / 'deviations.md').write_text(NOTES_1023, encoding='utf-8')                      # bench_report が書いた形
    import shutil as _sh
    _sh.rmtree(root / 'nightly_20001024')
    r = nightly(jf, env, '20001024', '--bench', bench)
    assert r.returncode == 3 and str(rep) in r.stderr
