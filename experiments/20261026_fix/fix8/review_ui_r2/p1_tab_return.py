"""反証 8-2 回目・係 ui: 8-1 の app.js の直し(走行タブに戻った直後の断り・drTabGen)が、断りが解けないまま ▶ が押せない形・
走行中の画面・■・タブの素早い行き来・一覧の GET の遅れ/失敗・status の 5 秒の打ち切り・起き直しと重なる形で、どう動くか。
本物の app.js・index.html を node の vm で(sim_r2.js = tests/fixtures/app_sim_fix8.js の写しに操作を足したもの)。読むだけ。
▶・■ は人が押す形(ボタンが disabled なら押さない = hstart・hstop)。出力 out/p1_tab_return.txt"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import run, blk, DI, HERE, steps_of, confirm_lines, RESET  # noqa: E402

lines = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    lines.append(s)
    print(s)


MORNING = [['@', 'A', ['tab', 'drive']], ['tickall'], ['tickall'], ['@', 'A', ['baseline']],
           ['@', 'A', ['check', '#dr-rec-back-guard', True]], ['@', 'A', ['check', '#dr-rec-hold-keep', True]], ['tickall']]


def show(name, out):
    log('## ' + name)
    for s in out['steps']:
        op = s.get('op')
        if op == 'dstate':
            v = s['value']
            log('  [%s] 状態: loaded=%s gen=%s lost=%s ▶disabled=%s tab=%s running=%s msg=%s' % (
                s.get('label', ''), v['loaded'], v['gen'], v['lost'], v['startDisabled'], v['tab'], v['running'], v['msg'][:40]))
        elif op == 'start':
            if s.get('disabled'):
                log('  ▶: 押せない(disabled)')
            else:
                c = confirm_lines(s.get('confirm'))
                ss = s.get('server_at_start') or {}
                log('  ▶: 送った=%s 返事=%s 確認の文=%s 新しいトースト=%s' % (
                    s['sent'] is not None, s.get('status'), ' / '.join(c) if s.get('confirm') else '(出ない)',
                    [t[:38] for t in s.get('new_toasts', [])]))
                if s['sent'] is not None:
                    log('      サーバーが走った値: 見張り %s・残す %s・左 %s' % (ss.get('rec_back_guard'), ss.get('rec_hold_keep'), ss.get('ceil_left_mps')))
        elif op == 'stop':
            log('  ■: %s' % ('押せない(disabled)' if s.get('disabled') else '押した'))
        elif op == 'posted':
            log('  届いた POST: %s' % [(p[0], p[1], p[3]) for p in s['value']])
        elif op == 'tlist':
            log('  タイマー: %s' % s['value'])
        elif op == 'toasts':
            log('  トースト(最後の 3): %s' % [t[:50] for t in s['value'][-3:]])


def D(label):
    return ['@', 'A', ['dstate']], label


def mk(steps):
    """['D', 'ラベル'] を dstate に替え、結果にラベルを付ける"""
    real, labels = [], []
    for s in steps:
        if s and s[0] == 'D':
            real.append(['@', s[2] if len(s) > 2 else 'A', ['dstate']])
            labels.append(s[1])
        else:
            real.append(s)
    return real, labels


def go(name, steps, pages=('A',), **kw):
    real, labels = mk(steps)
    out = run(real, pages=list(pages), name='p1_%d.json' % len(lines), **kw)
    it = iter(labels)
    for s in out['steps']:
        if s.get('op') == 'dstate':
            s['label'] = next(it)
    show(name, out)
    return out


# C1 一覧の GET が返らない(遅い Wi-Fi)まま: 戻ってすぐ・1 秒ごと 3 回・5 秒(期限)のあと
go('C1 一覧の GET が返らないまま(戻ってすぐ → 1 秒 × 3 → 5 秒の期限)', MORNING + [
    ['@', 'A', ['tab', 'live']], ['@', 'A', ['tickle', 1000]], ['hold', '/api/drive/models', True], ['@', 'A', ['tab', 'drive']],
    ['D', '戻ってすぐ'], ['@', 'A', ['hstart']],
    ['@', 'A', ['tickle', 1000]], ['@', 'A', ['tickle', 1000]], ['@', 'A', ['tickle', 1000]], ['D', '3 秒'], ['@', 'A', ['hstart']],
    ['@', 'A', ['tick']], ['D', '5 秒の期限のあと(tick)'], ['@', 'A', ['hstart']], ['@', 'A', ['hstop']],
    ['@', 'A', ['tickle', 1000]], ['D', '+1 秒'], ['release', '/api/drive/models'], ['D', '一覧が返った'], ['posted']])

# C2 一覧の GET がすぐ失敗
go('C2 一覧の GET がすぐ失敗(models_fail)', MORNING + [
    ['@', 'A', ['tab', 'live']], ['models_fail', True], ['@', 'A', ['tab', 'drive']], ['D', '戻ってすぐ'], ['@', 'A', ['hstart']],
    ['@', 'A', ['tickle', 1000]], ['D', '+1 秒'], ['@', 'A', ['hstart']], ['posted']])

# C3 戻ったあとの status が 5 秒で打ち切られる → 次の問い合わせで通る
go('C3 戻ったあとの表示の問い合わせが返らず 5 秒で打ち切り → 次で通る', MORNING + [
    ['@', 'A', ['tab', 'live']], ['hold', '/api/drive/status', True], ['@', 'A', ['tab', 'drive']], ['D', '戻ってすぐ(status 待ち)'],
    ['@', 'A', ['hstart']], ['@', 'A', ['tick']], ['D', '5 秒の打ち切りのあと'], ['@', 'A', ['hstart']],
    ['hold', '/api/drive/status', False], ['@', 'A', ['tickle', 1000]], ['D', '+1 秒(通った)'], ['@', 'A', ['hstart']], ['posted']])

# C4 タブを素早く行き来(一覧の GET を 3 本止めたまま)→ まとめて返す / 新しい順に返す
for order in ('release', 'newest'):
    rel = [['release', '/api/drive/models']] if order == 'release' else [['releaseN', '/api/drive/models']] * 3
    go('C4%s 素早い行き来(走行→ライブ→走行 × 3。一覧の GET 3 本を%s)' % ('a' if order == 'release' else 'b', '古い順に返す' if order == 'release' else '新しい順に返す'),
       MORNING + [['hold', '/api/drive/models', True]] + [['@', 'A', ['tab', 'live']], ['@', 'A', ['tab', 'drive']]] * 3 + [
           ['D', '3 回戻った直後'], ['@', 'A', ['hstart']]] + rel + [['D', '一覧が返った'], ['@', 'A', ['hstart']], ['@', 'A', ['hstop']],
           ['@', 'A', ['tickle', 1000]], ['@', 'A', ['tickle', 1000]], ['@', 'A', ['tlist']], ['posted']])

# C5 離れる前に出した問い合わせが、戻ったあとの問い合わせより後に返る(古い値。drTabGen)
go('C5 離れる前の問い合わせ(古い値)が、戻ったあとの問い合わせより後に返る', MORNING + [
    ['hold', '/api/drive/status', True], ['@', 'A', ['tickle', 1000]], ['@', 'A', ['tab', 'live']],
    ['api', {'rec_back_guard': 0}], ['@', 'A', ['tab', 'drive']], ['hold', '/api/drive/status', False],
    ['releaseN', '/api/drive/status'], ['D', '新しい方が返った'], ['release1', '/api/drive/status'], ['D', '古い方が後から返った'],
    ['@', 'A', ['hstart']], ['@', 'A', ['tickle', 1000]], ['D', '+1 秒'], ['@', 'A', ['hstart']], ['posted']])

# C6 この画面が走らせている(token)→ ライブ → 戻ってすぐ ■(一覧の GET は止めたまま)→ 止まったあとの ▶
go('C6 この画面が走行中 → ライブ → 戻ってすぐ ■(一覧の GET を止めたまま)→ ▶', MORNING + [
    ['@', 'A', ['pick', DI, blk(1)]], ['@', 'A', ['hstart']], ['@', 'A', ['tickle', 250]], ['@', 'A', ['tab', 'live']],
    ['@', 'A', ['tickle', 250]], ['hold', '/api/drive/models', True], ['@', 'A', ['tab', 'drive']], ['D', '戻ってすぐ(走行中)'],
    ['@', 'A', ['hstop']], ['D', '■ のあと'], ['@', 'A', ['pick', DI, blk(2)]], ['@', 'A', ['hstart']], ['release', '/api/drive/models'],
    ['@', 'A', ['tickle', 1000]], ['D', '一覧が返った'], ['@', 'A', ['hstart']], ['posted']])

# C7 ほかの画面が走っている間に離れ、離れている間に止まった → 戻ってすぐ ▶
go('C7 B が走っている間に A がライブへ・B が止めた → A が戻ってすぐ ▶(一覧を止めたまま)→ 一覧が返って ▶', [
    ['@', 'A', ['tab', 'drive']], ['@', 'B', ['tab', 'drive']], ['tickall'], ['tickall'], ['@', 'B', ['baseline']],
    ['@', 'B', ['check', '#dr-rec-back-guard', True]], ['@', 'B', ['check', '#dr-rec-hold-keep', True]], ['tickall'],
    ['@', 'B', ['pick', DI, blk(1)]], ['@', 'B', ['hstart']], ['tickall'], ['@', 'A', ['tab', 'live']], ['@', 'A', ['tickle', 250]],
    ['@', 'B', ['hstop']], ['@', 'A', ['tickle', 250]], ['@', 'A', ['tickle', 250]], ['@', 'B', ['check', '#dr-rec-back-guard', False]],
    ['hold', '/api/drive/models', True], ['@', 'A', ['tab', 'drive']], ['D', 'A 戻ってすぐ'], ['@', 'A', ['hstart']],
    ['release', '/api/drive/models'], ['D', 'A 一覧が返った'], ['@', 'A', ['hstart']], ['posted']], pages=('A', 'B'))

# C8 起き直しと重なる: ライブの間にサーバーが起き直し(見張り・残すは OFF に戻る)→ 戻ってすぐ ▶ → 一覧が返る → 入れ直す → ▶
go('C8 ライブの間に起き直し → 戻ってすぐ ▶ → 一覧が返る → 入れ直す → ▶', MORNING + [
    ['@', 'A', ['tab', 'live']], ['down_all', True], ['@', 'A', ['tickle', 1000]], ['reboot', 2000.0, RESET], ['down_all', False],
    ['hold', '/api/drive/models', True], ['@', 'A', ['tab', 'drive']], ['D', '戻ってすぐ'], ['@', 'A', ['hstart']],
    ['release', '/api/drive/models'], ['D', '一覧が返った'], ['@', 'A', ['toasts']],
    ['@', 'A', ['check', '#dr-rec-back-guard', True]], ['@', 'A', ['check', '#dr-rec-hold-keep', True]], ['@', 'A', ['tickle', 1000]],
    ['@', 'A', ['pick', DI, blk(1)]], ['@', 'A', ['hstart']], ['posted']])

# C9 起き直しの途中に戻る(一覧も status も通らない)→ 起き上がる
go('C9 起き直しの途中に戻る(全部通らない)→ 起き上がる → ▶', MORNING + [
    ['@', 'A', ['tab', 'live']], ['down_all', True], ['@', 'A', ['tab', 'drive']], ['D', '戻ってすぐ(落ちている)'], ['@', 'A', ['hstart']],
    ['@', 'A', ['tickle', 1000]], ['reboot', 2000.0, RESET], ['down_all', False], ['@', 'A', ['tickle', 1000]], ['D', '起き上がった +1 秒'],
    ['@', 'A', ['toasts']], ['@', 'A', ['hstart']], ['@', 'A', ['tickle', 1000]], ['@', 'A', ['tickle', 1000]], ['D', '+2 秒'],
    ['@', 'A', ['check', '#dr-rec-back-guard', True]], ['@', 'A', ['check', '#dr-rec-hold-keep', True]], ['@', 'A', ['tickle', 1000]],
    ['@', 'A', ['pick', DI, blk(1)]], ['@', 'A', ['hstart']], ['posted']])

# C10 status だけ途切れる(1 本)と重なる
go('C10 戻った直後の表示の問い合わせだけ途切れる(1 本)→ 次で通る', MORNING + [
    ['@', 'A', ['tab', 'live']], ['status_fail_next', 1], ['@', 'A', ['tab', 'drive']], ['D', '戻ってすぐ'], ['@', 'A', ['hstart']],
    ['@', 'A', ['tickle', 1000]], ['D', '+1 秒'], ['@', 'A', ['hstart']], ['posted']])

(HERE / 'out' / 'p1_tab_return.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
