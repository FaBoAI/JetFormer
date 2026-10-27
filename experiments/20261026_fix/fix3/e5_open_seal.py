#!/usr/bin/env python3
"""係 e5(2026-10-26): 11/1 の朝に封を 1 回だけ開けて、11/3 の新しいモデルの 1 枠を決める台本(protocol_v3 10 章 E5・12 章・13 章)。

決まり(protocol_v3.json の slot_20261103_v3・vetoes_v3・seal。変えない):
  1. 11/1 朝に封を 1 回だけ開ける(v2 と一緒の 1 回)。seal_opened.json を 1 つ書き、v2・v3 の両方の予測の一覧を入れる
  2. v2 の D5b を v2 の決まりで書く(候補 0 → 無し。v2 の拒否権は封の T だけ・順は BF1 → BF2 → C1 → C2)
  3. v3 の候補(BF2-1・V1・V2)に封の T・V の拒否権を掛ける
       T: seal_t_l1_us − 5 本の平均(封の上)> 2 × 5 本の sd(封の上)
       V: seal_v_thr_vs_b_us − 5 本の平均(封の上)> max(10 µs, 2 × 5 本の sd(封の上))
  4. 残りを BF2-1 → V1 → V2 の順に見て、最初の 1 つを 11/3 の 1 枠にする
  5. 全部落ちたら枠は無し(B のまま)。開けたあとに作り直さない
  予測するもの: B・5 本・V1・V2・BF2-1(あれば)。bench_candidates.json に入れるのも走らせるのもユーザーの了承のあと。

この台本で足した決め(封を開ける前の 10/26 に書いた。fix3/e5.md 2 章):
  a. 候補が 0(v2・v3 とも)なら封を開けない。「B のまま」を出す。開けるかはユーザーが決める
     (開けると決めたら --user-approved-open-without-candidates "<ユーザーの言葉>" で開ける。数は参考だけ)
  b. decisions_v3.json の E3 が「決まった」(final)でないと開けない(BF2-0 が「学習する」なら E4 も)
  c. 拒否権が計算できない候補(集合が 0 フレーム・予測が無い・数え直しが違う)は通さない。その候補で止めて「決まらない」
     (後ろの候補を見ない。main session に聞く)
  d. 集合の作り(読むのは開けたあとだけ):
     T = make_intervention_set.build の既定(失敗の交代 after_collision・stuck・preemptive・前進だけ・前置き 3 秒・
         交代から 5 秒か戻しまで・採点 1 フレーム以上)を封の 14 本に。データセットは書かず、同じ関数でその場で作る。
         L1 = 10 行 × ステア・スロットルの |予測 − 人| の平均(eval_v3 の l1_interv_us と同じ式)。ジャンルはフレームのまま
     V = 3 回目の AI のきれいな周の直線(inventory の straight: eRPM < −1000・|ヨー| < 40)の AI の側。
         封の周は学習に使っていないので、desk の「B の 9.5 秒以上」は掛けない(desk の決まりの数も参考に出す)。
         行 2・車に出る値(ゲイン 0.90 どうし)の 候補 − B の平均(desk の straight_thr_vs_b_us と同じ式。+ = 緩める)
     AI の側のフレームは build_ai_dataset.py(dataset_v2_20261025 と同じ引数: --label-gains eval・ゲイン 1.0・
         --keep-ai-log-human)と同じ関数でその場で作る。集合のマスクは desk_v2.scene_masks(C3 と同じ)
     参考(拒否権でない): H(人の走り・交代の直後。plan_checks の式)・A(B のきれいな周の、車に出た値との差)・
         D(3 回目の DE の衝突の手前 1 秒・DE の通過)・S(1 回目の止まった場面)。G-b の封は B-FAST の候補が無いので出さない
  e. 封を開けたあとに足した腕(--reference で予測したもの)は参考。1 枠に入れない
  f. 予測するモデルの best.pth の sha256 は開けたときに seal_opened.json に書き、予測のときに照らす(作り直さない)

車・PCA9685・I2C・カメラ・ToF・基板に触れない。走行の API は呼ばない(GPU の仕事の前に GET の状態だけ見る)。
data/・protocol・lock2・decisions*.json・bench_candidates.json は読むだけ(decisions に写すのは main session)。
11/1 に書くもの: seal_opened.json(v2 と同じ場所 train_v2_20261025/。1 回だけ。あれば止まる)・
  fix_20261026/e5_seal_20261101/(seal_rows.npz・seal_rows.json・pred/・crosscheck_B.json・e5.json・e5.md・e5_proposed.json)

    cd ~/jetformer
    P=/home/jetson/miniconda/envs/jetracer/bin/python
    E=experiments/20261026_fix/fix3/e5_open_seal.py
    L="/home/jetson/bin/jf-heavy --lock none --mem 3G --need-mb 1500 --"
    G="/home/jetson/bin/jf-heavy --lock big --mem 3872M --need-mb 3200 -- nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 $P -u"
    $P -I $E plan                       # 読むだけ(既定)。候補・開けるか・コマンド
    $P -I $E commands                   # 11/1 のコマンドの一覧
    $L $P -I $E selfcheck               # 封でない記録で、作りが make_intervention_set・dataset_v2・sets_v2・plan_checks と同じか(CPU)
    $L $P -I $E open                    # 11/1: 固定の確かめ → seal_opened.json → 集合(CPU)
    $G $E predict --runs <名前> ...      # GPU。1 回に 1〜2 本(初回は B の予測し直しを v2 の保存と照らす)
    $L $P -I $E judge                   # CPU。T・V の拒否権 → 1 枠(案。decisions に写すのは main session)
  作り物の封(本物の封を読まない・開けない。fix3/e5_dryrun/<場面>-<時刻>/ に書く):
    $P -I $E --dry-run all --scenario veto_v          # 全部が作り物(数秒)
    $L $P -I $E --dry-run all --dry-logs1 <記録> --dry-logs3 <記録>   # 封でない記録から同じ作りで集合を作る(予測は作り物)
"""

import argparse
import datetime
import hashlib
import importlib.util
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                       # experiments/20261026_fix/fix3
FIX = HERE.parent                                            # experiments/20261026_fix
REPO = Path('/home/jetson/jetformer')
for _p in (str(REPO), str(REPO / 'tools')):
    if _p not in sys.path:
        sys.path.append(_p)


def _load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


D = _load_module('e5_desk_v3', FIX / 'c3' / 'desk_v3.py')    # 机上の台本 v3(式・5 本・場所。読むだけ)
D2_PATH = REPO / 'experiments' / '20261025_train_v2' / 'c3' / 'desk_v2.py'
_D2 = None

REP2, REP3 = D.REP2, D.REP3
PROTOCOL2, PROTOCOL3 = D.PROTOCOL2, D.PROTOCOL
LOCK2_V2, LOCK2_V3 = REP2 / 'protocol_v2_lock2.json', D.LOCK2
DECISIONS2, DECISIONS3 = REP2 / 'decisions.json', REP3 / 'decisions_v3.json'
CHECK_SEAL = REPO / 'experiments' / '20261025_train_v2' / 'check_seal_v2.py'
SEAL_OPENED = D.SEAL_OPENED                 # train_v2_20261025/seal_opened.json(desk_v2・desk_v3 も見る場所。v2 と同じ 1 つ)
REAL_SEAL_OPENED = Path('/home/jetson/jetformer_reports/train_v2_20261025/seal_opened.json')   # 試験の止めに使う(差し替えない)
if Path(SEAL_OPENED) != REAL_SEAL_OPENED:
    raise SystemExit('desk_v3.SEAL_OPENED が v2 の場所と違います: %s' % SEAL_OPENED)
OUT = REP3 / 'e5_seal_20261101'
DRY_ROOT = REP3 / 'fix3' / 'e5_dryrun'
SELFCHECK_JSON = REP3 / 'fix3' / 'e5_selfcheck.json'
OPEN_DATE = datetime.date(2026, 11, 1)
RUN_DAY = datetime.date(2026, 11, 3)
SCHEMA = 'jf-e5-seal/v1'

GAIN = D.GAIN                               # 0.90(B・候補のジャンル 5 の走るゲイン)
ROW0, ROW2 = D.ROW0, D.ROW2
G5 = D.SANITY_GENRE                         # AI の側の集合はジャンル 5(desk と同じ)
FIVE_RUNS = list(D.FIVE_RUNS)               # A_s0・A_s1・v0-s2・v0-s3・B
RID_OF = dict(D.RID_OF)
B_RUN = D.B_RUN
V_MIN_US = 10.0                             # V の拒否権の下限(protocol vetoes_v3.seal_V)
SE_K = 2.0                                  # 2 × 5 本の sd
T_LEAD_S, T_MAX_S, T_SKIP_S, T_KINDS, T_MIN_SCORED = 3.0, 5.0, 0.0, ('forward',), 1   # make_intervention_set.build の既定
AI_GAINS = {'steer_gain': 1.0, 'throttle_gain': 1.0, 'steer_trim_us': 0.0}           # dataset_v2 の --eval-*-gain 1.0
AI_MIN_DURATION = 5.0
H_RUN_K, H_RUN_ERPM = 61, 3000.0            # plan_checks.py の RUN_K・RUN_ERPM(走りの区間)
STAND_ERPM = 400.0
TOF_FREE_M = 1.0
SEAL_LAP_MODEL = '封の周(学習に使っていない)'
SESS_NAME = {1: '10/24 1 回目', 3: '10/24 3 回目'}
V3_ORDER = ('BF2-1', 'V1', 'V2')            # protocol slot_20261103_v3 の 4
V2_ORDER = ('BF1', 'BF2', 'C1', 'C2')       # protocol_v2 11 章の 3(予備は 1 枠に入れない)
# 設計のときの数(dataset_v2_plan plan.md 3.2・3.3、plan_checks.json)。合わなければ注を書く(止めない)
DESIGN = {'takeovers': 28, 'h_samples': 197, 'de_collisions_sess3': 25, 'stand_label_frames_sess1': 57,
          'ai_clean_lap_samples': 2251, 'b_clean_lap_samples': 1684}
MASKS = ('m_T', 'm_V', 'm_Dpre', 'm_Dpass', 'm_S', 'm_Sall', 'm_A', 'm_H', 'm_V_desk', 'm_Dpass_desk')
SCENARIOS = ('pass', 'veto_v', 'veto_t', 'veto_all', 'none')
MIN_FREE_GB = 1.0

log = D.log
num = D.num
sha256 = D.sha256
load_json = D.load_json


def write_json(path, obj):
    D.write_json(path, D.clean(obj))


def load_d2():
    """desk_v2.py(C3 の集合の作り: scene_masks・lap_bounds・EXTRA・stage0_2_diffs。読むだけ)"""
    global _D2
    if _D2 is None:
        _D2 = _load_module('e5_desk_v2', D2_PATH)
    return _D2


def under_pytest():
    return 'PYTEST_CURRENT_TEST' in os.environ


def today():
    return datetime.date.today()


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def mean_sd(xs):
    """平均と標本の sd(n − 1)。値が無い(None・NaN)ものがあれば (None, None)"""
    if any(num(x) is None for x in xs):
        return None, None
    xs = [float(x) for x in xs]
    if len(xs) < 2:
        return None, None
    return float(np.mean(xs)), float(np.std(xs, ddof=1))


# ================================================================ 場所

class Paths(object):
    """本物(11/1)と作り物(--dry-run)の書く場所。作り物は本物の seal_opened.json に触れない"""

    def __init__(self, root, seal_opened, dry):
        self.root, self.seal_opened, self.dry = Path(root), Path(seal_opened), bool(dry)
        self.rows = self.root / 'seal_rows.npz'
        self.rows_json = self.root / 'seal_rows.json'
        self.pred = self.root / 'pred'
        self.cross = self.root / 'crosscheck_B.json'
        self.e5_json = self.root / 'e5.json'
        self.e5_md = self.root / 'e5.md'
        self.proposed = self.root / 'e5_proposed.json'
        if self.dry and self.seal_opened.resolve() == REAL_SEAL_OPENED.resolve():
            raise SystemExit('作り物の封が本物の seal_opened.json を使おうとしました')


def real_paths():
    return Paths(OUT, SEAL_OPENED, False)


def dry_paths(root):
    return Paths(root, Path(root) / 'seal_opened.json', True)


def seal_logs():
    return list(D.protocol2()['seal']['logs'])


def seal_sessions():
    """封の記録 → 回(1・3)。protocol_v2 の seal.per_log.session の頭で決める"""
    p2 = D.protocol2()
    out = {}
    for lg in p2['seal']['logs']:
        s = p2['seal']['per_log'][lg]['session']
        code = [k for k, v in SESS_NAME.items() if s.startswith(v)]
        if len(code) != 1:
            raise SystemExit('封の記録 %s の回が読めません: %s' % (lg, s))
        out[lg] = code[0]
    return out


# ================================================================ 候補

def _name(x):
    return x if isinstance(x, str) and x.strip() else None


def read_candidates(dec3, dec2):
    """decisions_v3.json(E3・E4・BF2-0)と v2 の decisions.json(D5a・BF・D4)から候補を読む。
    → {'v3': {BF2-1・V1・V2: 名前 or None}, 'v2': {BF1・BF2・C1・C2・予備}, 'final': bool, 'problems': [...], 'status': {...}}"""
    dec3, dec2 = dec3 or {}, dec2 or {}
    probs, status = [], {}
    e3 = dec3.get('E3')
    if not isinstance(e3, dict):
        probs.append('decisions_v3.json に E3(V1・V2)がまだ無い(10/27〜28 に main session が写す)')
        e3 = {}
    elif e3.get('final') is not True:
        probs.append('decisions_v3.json の E3 が「決まった」になっていない(final = %r)' % e3.get('final'))
    status['E3'] = {'V1': _name(e3.get('V1')), 'V2': _name(e3.get('V2')), 'final': e3.get('final')}
    bf20 = dec3.get('BF2-0') or {}
    if bf20.get('train') is False:
        bf21, status['E4'] = None, 'B-FAST2 は学習しない(BF2-0)→ BF2-1 は無し'
    else:
        e4 = dec3.get('E4') if isinstance(dec3.get('E4'), dict) else {}
        bf21 = _name(e4.get('BF2-1'))
        status['E4'] = {'BF2-1': bf21, 'final': e4.get('final')}
        if e4.get('final') is not True:
            probs.append('BF2-0 が「学習しない」でないのに、E4(BF2-1)が「決まった」になっていない')
    v3 = {'BF2-1': bf21, 'V1': _name(e3.get('V1')), 'V2': _name(e3.get('V2'))}
    for k, n in v3.items():
        if n is None:
            continue
        ok = D.RE_BF2.match(n) if k == 'BF2-1' else D.RE_V3.match(n)
        if not ok or (k == 'BF2-1' and not n.startswith('bfast2_')):
            probs.append('%s の名前が protocol_v3 の学習の名前ではありません: %s' % (k, n))
    if v3['V1'] and v3['V1'] == v3['V2']:
        probs.append('V1 と V2 が同じです: %s' % v3['V1'])
    d5a, bf, d4 = dec2.get('D5a') or {}, dec2.get('BF') or {}, dec2.get('D4') or {}
    v2 = {'BF1': _name(bf.get('BF1')), 'BF2': _name(bf.get('BF2')), 'C1': _name(d5a.get('C1')), 'C2': _name(d5a.get('C2')),
          '予備': None}
    if d4.get('reserve') is True:
        v2['予備'] = '2 本の平均(D4)'
    status['v2'] = {'D5a': d5a.get('status'), 'BF': bf.get('status'), 'D4': d4.get('status')}
    return {'v3': v3, 'v2': v2, 'final': not probs, 'problems': probs, 'status': status}


def plan_of(cands):
    """候補 → 予測するもの・開けるか"""
    v3 = [n for k in V3_ORDER for n in [cands['v3'].get(k)] if n]
    v2 = [n for k in V2_ORDER for n in [cands['v2'].get(k)] if n]
    n_cand = len(v3) + len(v2)
    pred_v3 = FIVE_RUNS + v3
    pred_v2 = FIVE_RUNS + v2
    allp = list(dict.fromkeys(pred_v2 + pred_v3))
    if not cands['final']:
        decision = '開けない(候補が決まっていない: %s)' % ' / '.join(cands['problems'])
    elif n_cand == 0:
        decision = ('B のまま(候補 0: v2 は C1・C2・BF1・BF2 が無し、v3 は V1・V2・BF2-1 が無し)。'
                    '封は開けない。開けるかはユーザーが決める(開けると次に「まだ見ていない」に使えない。数は参考だけ)')
    else:
        decision = '開ける(11/1 朝・1 回だけ)。予測 %d 本 → T・V の拒否権 → %s の順の最初' % (len(allp), ' → '.join(V3_ORDER))
    return {'n_candidates': n_cand, 'v3_candidates': v3, 'v2_candidates': v2,
            'predict': {'v2': {'candidates': dict(cands['v2']), 'models': pred_v2},
                        'v3': {'candidates': dict(cands['v3']), 'models': pred_v3}, 'all': allp},
            'open': bool(n_cand and cands['final']), 'decision': decision}


def run_dir(name):
    if name in FIVE_RUNS:
        return D.RUNS / name
    return D.run_dir_of(name, D.protocol(check=False))


# ================================================================ 固定の確かめ

def fix_checks():
    """protocol_v2・lock2(v2)・protocol_v3・lock2_v3 の sha256sum -c と check_seal_v2.py(中身は開かない)"""
    out = []
    for d, f in ((REP3, 'protocol_v3.sha256'), (REP3, 'protocol_v3_lock2.sha256'), (REP2, 'protocol_v2.sha256'),
                 (REP2, 'protocol_v2_lock2.sha256')):
        r = subprocess.run(['sha256sum', '-c', f], cwd=str(d), capture_output=True, text=True)
        out.append({'what': str(Path(d) / f), 'rc': r.returncode, 'out': (r.stdout + r.stderr).strip()[-1500:]})
    r = subprocess.run(['python3', '-I', str(CHECK_SEAL)], capture_output=True, text=True)
    out.append({'what': str(CHECK_SEAL), 'rc': r.returncode, 'out': (r.stdout + r.stderr).strip()[-1500:]})
    return out


def weights_of(names):
    """予測するモデルの best.pth・model_config.json の sha256(開けたときに書く。予測のときに照らす)"""
    out = {}
    for n in names:
        rd = run_dir(n)
        w, c = rd / 'best.pth', rd / 'model_config.json'
        if not w.exists() or not c.exists():
            out[n] = {'run_dir': str(rd), 'missing': True}
            continue
        out[n] = {'run_dir': str(rd), 'best_sha256': sha256(w), 'model_config_sha256': sha256(c),
                  **D.weights_stamp(w)}
    return out


def write_once(path, obj):
    """1 回だけ書く(O_EXCL。あれば FileExistsError)。書いたあと読み取りだけにする"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(D.clean(obj), f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())


def opened_record(P):
    rec = load_json(P.seal_opened)
    if not isinstance(rec, dict):
        raise SystemExit('封はまだ開けていません(%s が無い)。先に open' % P.seal_opened)
    if rec.get('schema') != SCHEMA or bool(rec.get('dry_run')) != P.dry:
        raise SystemExit('%s はこの台本の%sの記録ではありません(main session に聞く)' % (P.seal_opened, '作り物' if P.dry else '本物'))
    return rec


# ================================================================ 集合(CPU。開けたあとだけ。作り物では封でない記録)

H_EXTRA = '''
    # ---- e5_open_seal で足した取り出し(H: plan_checks.py の EXTRA と同じ式。ここから)
    _e5e = -erpm
    if n >= E5_RUN_K:
        from numpy.lib.stride_tricks import sliding_window_view as _e5swv
        _e5h = E5_RUN_K // 2
        _e5pad = np.concatenate([np.full(_e5h, np.nan), _e5e, np.full(_e5h, np.nan)])
        _e5med = np.nanmedian(_e5swv(_e5pad, E5_RUN_K), axis=1)
    else:
        _e5med = np.full(n, np.nanmedian(_e5e) if n else np.nan)
    _e5run = _e5med >= E5_RUN_ERPM
    _e5ivw = np.zeros(n, bool)
    for _e5w in E5_IV.get(log, []):
        _e5ivw |= (t >= _e5w) & (t < _e5w + 5.0)
    _e5new = a_mask & ~near & ~base_tr & ~base_va & np.isin(mode, (4, 5)) & ~_e5ivw
    PC_OUT[log].update({'h_new': _e5new.copy(), 'h_run': _e5run.copy(), 'h_tk': b_mask.copy(),
                        'h_sample': (idx_in % STRIDE == 0).copy(),
                        'stop_ai': (ai & ~near & (np.abs(erpm) < E5_STAND_ERPM)).copy(),
                        'c_mask': c_mask.copy()})
    # ---- e5_open_seal で足した取り出し(ここまで)
'''


def iv_windows_1010():
    """plan_checks.iv_windows と同じ: intervention_val_20261011 の窓(10/10 の記録だけ。封の記録には無い)"""
    card = load_json(D.DATASETS / 'intervention_val_20261011' / 'dataset_card.json') or {}
    out = {}
    for w in (card.get('intervention') or {}).get('windows') or []:
        if w.get('included'):
            out.setdefault(w['log'], []).append(float(w['t_s']))
    return out


def extract_inventory(logs, iv=None):
    """inventory.py の process_log の写しに desk_v2 の EXTRA と H の EXTRA を足して、記録ごとのラベルの行の値を取り出す
    (desk_v2.extract_logs・plan_checks.py と同じ方法。封の止めは呼ぶ側: 開けたあとか、封でない記録だけ)"""
    D2 = load_d2()
    spec = importlib.util.spec_from_file_location('e5_inv_copy', str(D2.INV_PY))
    inv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inv)
    inv.scene_brightness = D2._fast_scene_span
    src = inspect.getsource(inv.process_log)
    anchor = "    rec['candidates'] = {"
    if src.count(anchor) != 1:
        raise SystemExit('inventory.process_log の形が変わった')
    ns = inv.__dict__
    ns.update(PC_OUT={}, E5_RUN_K=H_RUN_K, E5_RUN_ERPM=H_RUN_ERPM, E5_IV=iv or {}, E5_STAND_ERPM=STAND_ERPM)
    exec(compile(src.replace(anchor, D2.EXTRA + H_EXTRA + anchor), str(D2.INV_PY) + ' (e5 の写し)', 'exec'), ns)
    modes = inv.mode_table()
    membership = inv.dataset_membership()
    all_logs = sorted(d for d in os.listdir(inv.RAW) if d[:8] in inv.DAYS and (inv.RAW / d).is_dir())
    sess = inv.sessions(all_logs)              # 記録の名前(時刻)だけで回を決める
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for lg in logs:
            ns['process_log'](lg, membership, modes, sess.get(lg))
            if lg not in ns['PC_OUT']:
                raise SystemExit('%s の取り出しができません(ラベルか記録が読めない)' % lg)
            log('  取り出し %s %s' % (lg, sess.get(lg)))
    return ns['PC_OUT'], sess


def scene_aligned(sname, L):
    """build_dataset + build_ai_dataset.augment_aligned と同じ aligned を、データセットを書かずに作る(1 シーン)。
    → (aligned の dict, 画像のパス [n], 注)"""
    from jetformer import dataset
    import build_ai_dataset as bad
    p = dataset.scene_path(sname)
    meta = dataset.load_scene_meta(sname)
    cam = dataset._load(p, 'CAM_FRONT')
    tof, imu, veh = dataset._load(p, 'TOF_FRONT'), dataset._load(p, 'IMU'), dataset._load(p, 'VEHICLE')
    cam_t = cam['timestamps_us'].astype(np.int64)
    a = dataset._aligned_arrays(cam_t, tof, imu, veh)
    a['mode'] = dataset._frame_modes(meta, cam_t)
    files = np.array([str(p / str(f)) for f in cam['files']])
    note = None
    try:
        ai_cmd = dataset._load(p, 'AI_CMD')
    except Exception as e:      # noqa: BLE001  0 バイト・壊れた AI_CMD.npz(電源断)。ジャンルはシーンのまま
        ai_cmd, note = None, '%s/AI_CMD.npz が読めない(%s)' % (sname, type(e).__name__)
    bad.augment_aligned(a, L, ai_cmd, veh, keep_ai_log_human=True, human_label_mask=True)
    return a, files, note


def ai_log_labels(lg):
    """dataset_v2 と同じラベルの作り(--label-gains eval・ゲイン 1.0・トリム 0)と、使うシーン(choose_ai_scenes)"""
    import build_ai_dataset as bad
    info, arr = bad.load_labels(lg)
    L = bad.LogLabels(lg, info, arr, 'eval', AI_GAINS)
    scenes, skipped = bad.choose_ai_scenes(lg, L, keep_ai_log_human=True, min_duration=AI_MIN_DURATION)
    return info, arr, L, scenes, skipped


def log_context(d, lap_rule):
    """1 本の記録の周・DE の入り(desk_v2.cmd_sets と同じ)。lap_rule='seal' は周の model を封の印にして、
    lap_unused(DE の通過)を全部の周・lap_straight(直線)をきれいな周に(封の周は学習に使っていない)"""
    D2 = load_d2()
    bounds = D2.lap_bounds(d)
    if lap_rule == 'seal':
        bounds_m = [(a, b, dict(x, model=SEAL_LAP_MODEL)) for a, b, x in bounds]
    elif lap_rule == 'desk':
        bounds_m = bounds
    else:
        raise ValueError(lap_rule)
    te_idx = D2.run_starts(d['loc_detail'], D2.LOC_DETAIL['DE'])
    te = d['t_s'][te_idx]
    lap_te = D2.lap_of(te, [(a, b) for a, b, _ in bounds])
    a_ok = np.array([bool(x.get('clean')) and x.get('model') == 'B' for _, _, x in bounds] + [False], bool)
    return {'bounds': bounds, 'bounds_m': bounds_m, 'de_entries': list(zip(te.tolist(), lap_te.tolist())), 'a_ok': a_ok}


def scene_masks_e5(d, ctx, a, guard):
    """1 シーンのマスク: desk_v2.scene_masks(C3 と同じ)+ 止まった場面の ToF + A・H。→ (M, k, tl, xf)"""
    D2 = load_d2()
    ts = a['timestamps_us'].astype(np.int64)
    li = a['label_index'].astype(np.int64)
    src = a['source'].astype(np.int64)
    vv = a['vehicle_valid'].astype(bool)
    n = len(ts)
    has = (li >= 0) & vv
    k = np.clip(li, 0, None)
    if has.any() and (k.max() >= len(d['t_s']) or not np.array_equal(ts[has], d['t_us'][k[has]])):
        raise SystemExit('aligned の時刻と label_index の行の時刻が違います')
    tl = np.where(li >= 0, d['t_s'][k], np.nan)
    M = D2.scene_masks(d, li, has, src, tl, ctx['bounds_m'], ctx['de_entries'], d['anchors'], False, False, np.zeros(n, bool))
    xf = np.full(n, np.nan, np.float32)
    for i in np.nonzero(M['stand_pre'])[0]:
        xf[i] = guard.features(a['tof'][i], 0.0, 0.0, guard.center)['x_free']
    M['stand'] = M['stand_pre'] & (xf > TOF_FREE_M)
    ai = has & (src == 1) & d['ai'][k]
    lapf = D2.lap_of(tl, [(x0, x1) for x0, x1, _ in ctx['bounds']])
    mo = d['model_out'][k]
    M['A'] = ai & ctx['a_ok'][lapf] & (d['run'][k] == B_RUN) & np.isfinite(mo).all(axis=1) & ~d['hold'][k]
    hm = d['h_new'] & (d['h_run'] | d['h_tk'])
    M['H'] = (li >= 0) & (src == 0) & hm[k] & a['future_valid'].astype(bool) & vv
    return M, k, tl, xf


def t_windows(lg, info, arr):
    """T の窓: make_intervention_set.build と同じ(LogLabels は D3・交代の直後のマスクを無視・load_raw・takeover_windows・
    window_arrays・失敗の交代で採点 1 以上)。→ (窓の一覧, 外したシーン, 交代の数)"""
    import build_ai_dataset as bad
    import label_events as le
    import make_intervention_set as mis
    Lt = bad.LogLabels(lg, info, arr, 'eval', bad.D3, human_ignore_bits=le.MASK_BITS['handover'])
    raw, skipped = mis.load_raw(lg)
    tks = (info.get('events') or {}).get('takeovers', [])
    out = []
    if raw is None:
        return out, skipped, len(tks)
    for w in mis.takeover_windows(info, T_LEAD_S, T_MAX_S, T_SKIP_S):
        tk = w['takeover']
        a, srcf, scenes = mis.window_arrays(raw, Lt, w, T_KINDS, Lt.ref[1])
        scored = int(a['future_valid'].sum())
        inc = tk['class'] in le.FAILURE_CLASSES and scored >= T_MIN_SCORED
        out.append({'name': '%s-tk%02d' % (lg, w['n']), 'class': tk['class'], 'scored': scored, 'included': bool(inc),
                    'a': a if inc else None, 'src': srcf if inc else None, 'scenes': scenes})
    return out, skipped, len(tks)


class _Rows(object):
    """集合の行(1 行 = 予測する 1 フレーム)をためる"""
    KEYS = ('part', 'log', 'scene', 'frame', 'window', 't_us', 'sess', 'img', 'tof', 'imu', 'vehicle', 'mode', 'genre_kind',
            'y', 'y_valid', 'model_out', 'gains', 'trim', 'lookahead', 'loc_detail', 'x_free', 'h_sample') + MASKS

    def __init__(self):
        self.parts = {k: [] for k in self.KEYS}

    def add(self, **cols):
        n = len(cols['frame'])
        for k in self.KEYS:
            v = cols.get(k)
            if v is None:
                v = np.zeros(n, bool) if k.startswith('m_') else None
            if v is None:
                raise KeyError(k)
            self.parts[k].append(np.asarray(v))

    def arrays(self):
        out = {}
        for k, v in self.parts.items():
            out[k] = np.concatenate(v) if v else np.zeros(0)
        return out


def build_rows(logs, sess_of, lap_rule='seal', iv=None, progress=log):
    """封(か作り物の封)の記録から、予測する行の表を作る(CPU)。sess_of[記録] = 1(1 回目)・3(3 回目)。
    → (arrays, meta)"""
    from jetformer.tof_guard import ToFGuard
    t0 = time.time()
    po, sess_names = extract_inventory(logs, iv)
    guard = ToFGuard(np.load(D.REPO / 'data' / 'tof_floor_ref.npy'))
    R = _Rows()
    meta = {'logs': list(logs), 'sess_of': dict(sess_of), 'lap_rule': lap_rule, 'per_log': {}, 'notes': []}
    lab = {'takeovers': 0, 'h_samples': 0, 'de_collisions_sess3': 0, 'stand_label_frames_sess1': 0,
           'ai_clean_lap_samples': 0, 'b_clean_lap_samples': 0}
    for lg in logs:
        d = po[lg]
        sc = int(sess_of[lg])
        info, arr, L, scenes, skipped = ai_log_labels(lg)
        ctx = log_context(d, lap_rule)
        pl = {'session_inventory': sess_names.get(lg), 'sess': sc, 'scenes': len(scenes),
              'skipped_scenes': [{'scene': s, 'why': w} for s, w in skipped], 'rows_ai': 0, 'notes': []}
        hm = d['h_new'] & (d['h_run'] | d['h_tk'])
        lab['h_samples'] += int((hm & d['h_sample']).sum())
        if sc == 3:
            lab['de_collisions_sess3'] += sum(1 for c in d['colls'] if c.get('ai') and c.get('loc') == 'DE')
        if sc == 1:
            lab['stand_label_frames_sess1'] += int(d['stop_ai'].sum())
        ctx_desk = log_context(d, 'desk') if lap_rule == 'seal' else None
        for a_, b_, x in ctx['bounds']:
            if x.get('clean'):
                m = (d['t_s'] >= a_) & (d['t_s'] < b_) & d['h_sample'] & d['c_mask']
                lab['ai_clean_lap_samples'] += int(m.sum())
                if x.get('model') == 'B':
                    lab['b_clean_lap_samples'] += int(m.sum())
        for sname in scenes:
            a, files, note = scene_aligned(sname, L)
            if note:
                pl['notes'].append(note)
            M, k, tl, xf = scene_masks_e5(d, ctx, a, guard)
            n = len(files)
            s3, s1 = sc == 3, sc == 1
            m = {'m_V': M['straight'] & s3, 'm_Dpre': M['pre_de'] & s3, 'm_Dpass': M['de_pass'] & s3,
                 'm_S': M['stand'] & s1, 'm_Sall': M['stand'], 'm_A': M['A'], 'm_H': M['H']}
            if ctx_desk is not None:        # 参考: desk の周の決まり(B の 9.5 秒以上・学習に使っていない周)の直線・DE の通過
                Md, _, _, _ = scene_masks_e5(d, ctx_desk, a, guard)
                m['m_V_desk'] = Md['straight'] & s3
                m['m_Dpass_desk'] = Md['de_pass'] & s3
            sel = np.zeros(n, bool)
            for v in m.values():
                sel |= v
            idx = np.nonzero(sel)[0]
            pl['rows_ai'] += len(idx)
            if not len(idx):
                continue
            kk = k[idx]
            human = m['m_H'][idx]
            R.add(part=np.zeros(len(idx), np.int8), log=np.array([lg] * len(idx)), scene=np.array([sname] * len(idx)),
                  frame=idx.astype(np.int32), window=np.array([''] * len(idx)), t_us=a['timestamps_us'][idx].astype(np.int64),
                  sess=np.full(len(idx), sc, np.int8), img=files[idx], tof=a['tof'][idx], imu=a['imu'][idx].astype(np.float32),
                  vehicle=a['vehicle'][idx].astype(np.float32), mode=a['mode'][idx].astype(np.int16),
                  genre_kind=human.astype(np.int8),           # 人のフレーム(H)はそのフレームのジャンル、AI の側はジャンル 5
                  y=a['future_actions'][idx].astype(np.float32),
                  y_valid=(a['future_valid'] & a['vehicle_valid'])[idx].astype(bool),
                  model_out=d['model_out'][kk].astype(np.float32), gains=d['gains'][kk].astype(np.float32),
                  trim=d['trim'][kk].astype(np.float32), lookahead=d['lookahead'][kk].astype(np.int16),
                  loc_detail=d['loc_detail'][kk].astype(np.int8), x_free=xf[idx], h_sample=d['h_sample'][kk].astype(bool),
                  **{kk_: v[idx] for kk_, v in m.items()})
            del a
        wins, wsk, ntk = t_windows(lg, info, arr)
        lab['takeovers'] += ntk
        pl['takeovers'] = ntk
        pl['t_windows'] = [{'name': w['name'], 'class': w['class'], 'scored': w['scored'], 'included': w['included']}
                           for w in wins]
        pl['t_skipped_scenes'] = [{'scene': s, 'why': w} for s, w in wsk]
        for w in wins:
            if not w['included']:
                continue
            a = w['a']
            idx = np.nonzero(a['future_valid'])[0]       # window_arrays の採点(vehicle_valid も入っている)= pred_table の part 1
            R.add(part=np.ones(len(idx), np.int8), log=np.array([lg] * len(idx)),
                  scene=np.array([''] * len(idx)), frame=idx.astype(np.int32), window=np.array([w['name']] * len(idx)),
                  t_us=a['timestamps_us'][idx].astype(np.int64), sess=np.full(len(idx), sc, np.int8), img=w['src'][idx],
                  tof=a['tof'][idx], imu=a['imu'][idx].astype(np.float32), vehicle=a['vehicle'][idx].astype(np.float32),
                  mode=a['mode'][idx].astype(np.int16), genre_kind=np.ones(len(idx), np.int8),
                  y=a['future_actions'][idx].astype(np.float32), y_valid=np.ones(len(idx), bool),
                  model_out=np.full((len(idx), 2), np.nan, np.float32), gains=np.full((len(idx), 2), np.nan, np.float32),
                  trim=np.full(len(idx), np.nan, np.float32), lookahead=np.zeros(len(idx), np.int16),
                  loc_detail=np.zeros(len(idx), np.int8), x_free=np.full(len(idx), np.nan, np.float32),
                  h_sample=np.zeros(len(idx), bool), m_T=np.ones(len(idx), bool))
        meta['per_log'][lg] = pl
        progress('  集合 %s(%s): AI の表 %d 行・T の窓 %d(交代 %d)' % (lg, SESS_NAME.get(sc, sc), pl['rows_ai'],
                                                               sum(1 for w in wins if w['included']), ntk))
    arrs = R.arrays()
    arrs['key'] = np.array(rows_key(arrs))
    meta['label_counts'] = lab
    meta['design'] = DESIGN
    meta['seconds'] = round(time.time() - t0, 1)
    return arrs, meta


def rows_key(R):
    ident = '|'.join('%d:%s:%s:%s:%d' % (int(p), w, s, lg, int(f)) for p, w, s, lg, f in
                     zip(R['part'], R['window'], R['scene'], R['log'], R['frame']))
    return hashlib.sha1(ident.encode()).hexdigest()


def counts_of(R):
    out = {'rows': int(len(R['frame']))}
    for m in MASKS:
        out[m] = int(np.asarray(R[m], bool).sum())
    out['T_windows'] = int(len(set(np.asarray(R['window'])[np.asarray(R['m_T'], bool)].tolist())))
    out['H_samples_rows'] = int((np.asarray(R['m_H'], bool) & np.asarray(R['h_sample'], bool)).sum())
    return out


def design_notes(meta, counts):
    """設計のときの数と照らす(止めない。注だけ)"""
    lab = meta.get('label_counts') or {}
    notes = []
    for key, got in (('takeovers', lab.get('takeovers')), ('h_samples', lab.get('h_samples')),
                     ('de_collisions_sess3', lab.get('de_collisions_sess3')),
                     ('stand_label_frames_sess1', lab.get('stand_label_frames_sess1')),
                     ('ai_clean_lap_samples', lab.get('ai_clean_lap_samples')),
                     ('b_clean_lap_samples', lab.get('b_clean_lap_samples'))):
        want = DESIGN[key]
        notes.append({'what': key, 'design': want, 'got': got, 'same': got == want})
    return notes


def save_rows(P, arrs, meta):
    D.save_npz(P.rows, **arrs)
    meta = dict(meta)
    meta['counts'] = counts_of(arrs)
    meta['key'] = str(arrs['key'])
    meta['rows_sha256'] = sha256(P.rows)
    meta['script_sha256'] = sha256(Path(__file__).resolve())
    meta['made_at'] = now_iso()
    meta['dry_run'] = P.dry
    if not P.dry or meta.get('from_logs'):
        meta['design_check'] = design_notes(meta, meta['counts'])
    write_json(P.rows_json, meta)
    return meta


def load_rows(P):
    if not D.npz_ok(P.rows):
        raise SystemExit('集合がありません: %s(先に open か sets)' % P.rows)
    meta = load_json(P.rows_json) or {}
    if meta.get('rows_sha256') != sha256(P.rows):
        raise SystemExit('%s の sha256 が seal_rows.json と違います' % P.rows)
    with np.load(P.rows, allow_pickle=False) as z:
        R = {k: z[k] for k in z.files}
    R['key'] = str(R['key'])
    for m in MASKS + ('y_valid', 'h_sample'):
        R[m] = R[m].astype(bool)
    return R, meta


# ================================================================ 作り物の封(--dry-run。データを読まない)

def synthetic_rows(seed=0):
    """全部が作り物の行の表(画像・ToF は無い。予測は fake_pred)。T 240 行・AI の表 720 行"""
    rng = np.random.default_rng(seed)
    nt, na = 240, 720
    n = nt + na
    part = np.r_[np.ones(nt, np.int8), np.zeros(na, np.int8)]
    y = np.zeros((n, 10, 2), np.float32)
    y[:nt, :, 0] = 1443 + rng.normal(0, 120, (nt, 10))
    y[:nt, :, 1] = 1420 + rng.normal(0, 60, (nt, 10))
    y[nt:, :, 0] = 1443 + rng.normal(0, 80, (na, 10))
    y[nt:, :, 1] = 1350 + rng.normal(0, 40, (na, 10))
    R = {'part': part, 'log': np.array(['dry-log'] * n), 'scene': np.array([''] * nt + ['dry-scene'] * na),
         'frame': np.r_[np.arange(nt), np.arange(na)].astype(np.int32),
         'window': np.array(['dry-tk%02d' % (i // 40 + 1) for i in range(nt)] + [''] * na),
         't_us': np.arange(n, dtype=np.int64) * 33333, 'sess': np.r_[np.full(nt, 3), np.full(na, 3)].astype(np.int8),
         'img': np.array([''] * n), 'tof': np.zeros((n, 42, 54), np.uint16), 'imu': np.zeros((n, 10, 6), np.float32),
         'vehicle': np.zeros((n, 4), np.float32), 'mode': np.full(n, 5, np.int16),
         'genre_kind': np.r_[np.ones(nt), np.zeros(na)].astype(np.int8), 'y': y, 'y_valid': np.ones(n, bool),
         'model_out': np.zeros((n, 2), np.float32), 'gains': np.tile(np.float32([1.0, GAIN]), (n, 1)),
         'trim': np.zeros(n, np.float32), 'lookahead': np.full(n, 3, np.int16), 'loc_detail': np.zeros(n, np.int8),
         'x_free': np.full(n, np.nan, np.float32), 'h_sample': np.zeros(n, bool)}
    for m in MASKS:
        R[m] = np.zeros(n, bool)
    R['m_T'][:nt] = True
    o = nt
    for m, k in (('m_V', 200), ('m_Dpre', 80), ('m_Dpass', 40), ('m_S', 60), ('m_A', 220), ('m_H', 120)):
        R[m][o:o + k] = True
        o += k
    R['m_V_desk'][nt:nt + 50] = True
    R['m_Dpass_desk'][nt + 280:nt + 300] = True
    R['m_Sall'] = R['m_S'].copy()
    R['sess'][nt + 320:nt + 380] = 1
    R['genre_kind'][nt + 600:] = 1                          # H は人のフレーム
    R['h_sample'][nt + 600::3] = True
    ref = (1433.0, 1500.0)
    R['model_out'] = D.car_out(y[:, 2], ref, GAIN).astype(np.float32)
    R['key'] = np.array(rows_key(R))
    return R


# 作り物の予測の中身: (ステアのずれ, スロットルのずれ(全部の行), 雑音の sd, スロットルのずれ(V の行だけ))
FAKE_FIVE = {'A_s0': (1.0, 2.0, 21.0, 2.0), 'A_s1': (-1.0, -3.0, 19.0, -3.0), 'v0-s2': (0.0, 4.0, 22.0, 4.0),
             'v0-s3': (2.0, -1.0, 20.0, -1.0), 'B': (0.0, 0.0, 20.0, 0.0)}
FAKE_CAND = {'pass': {'V1': (0.0, 1.0, 20.5, 1.0), 'V2': (0.0, 1.0, 20.0, 0.0)},
             'veto_v': {'V1': (0.0, 0.0, 20.0, 25.0), 'V2': (0.0, 1.0, 20.0, 1.0)},
             'veto_t': {'V1': (0.0, 0.0, 45.0, 0.0), 'V2': (0.0, 1.0, 20.0, 1.0)},
             'veto_all': {'V1': (0.0, 0.0, 20.0, 25.0), 'V2': (0.0, 0.0, 45.0, 0.0)},
             'none': {}}
FAKE_REF = {'B': (1433.0, 1500.0), 'v3': (1441.0, 1496.0)}


def dry_decisions(scenario):
    """作り物の封の候補(decisions_v3 の形)。none は候補 0"""
    c = FAKE_CAND[scenario]
    e3 = {'V1': 'v3_20261026-v3d-aug2-e60-s0' if 'V1' in c else None, 'V2': 'v3_20261026-v3d-aug2-e60-s1' if 'V2' in c else None,
          'final': True}
    return {'E3': e3, 'BF2-0': {'train': False}}


def fake_params(name, scenario, cands):
    if name in RID_OF:
        return FAKE_FIVE[RID_OF[name]]
    for lab, n in (cands or {}).items():
        if n == name and lab in FAKE_CAND[scenario]:
            return FAKE_CAND[scenario][lab]
    return (0.0, 0.0, 20.0, 0.0)


def fake_predict_fn(scenario, cands, R):
    """作り物の予測(GPU を使わない)。土台 = 行の正解(無ければ 1443・1400)+ ずれ + 名前で決まる雑音"""
    def fn(name, rd, items_n, genres_raw):
        st, th, sd, thv = fake_params(name, scenario, cands)
        base = np.asarray(R['y'], np.float64).copy()
        bad = ~np.asarray(R['y_valid'], bool)
        base[bad] = np.array([1443.0, 1400.0])
        seed = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        pred = base + np.array([st, th]) + rng.normal(0.0, sd, base.shape)
        pred[np.asarray(R['m_V'], bool), :, 1] += thv
        ref = FAKE_REF['B'] if name in RID_OF else FAKE_REF['v3']
        return pred.astype(np.float32), np.asarray(genres_raw), np.zeros(len(pred), bool), ref
    return fn


# ================================================================ 予測(GPU。作り物は fake_predict_fn)

def genres_raw_of(R):
    """行のジャンル(代用の前): AI の側はジャンル 5、人のフレーム(T・H)はそのフレームのジャンル"""
    return np.where(np.asarray(R['genre_kind']) == 0, G5, np.asarray(R['mode'])).astype(np.int64)


def gpu_predict_fn(batch=32, workers=3):
    """PyTorch FP32 の best(desk_v3.predict_tab・desk_gpu.predict_one と同じ: runtime.load_run・policy_zoo._predict_frames・
    rl_evaluate._genres の代用)。この関数を呼ぶときだけ torch を読む"""
    def fn(name, rd, items, genres_raw):
        import gc
        import torch
        import policy_zoo as pz
        import rl_evaluate as ev
        from jetformer.policies import runtime
        cfg = D.model_cfg(rd)
        trained = np.array([int(x) for x in cfg['trained_modes']])
        g, sub = ev._genres(list(genres_raw), None, trained)
        model = runtime.load_run(rd, 'best', device='cuda')[0]
        try:
            res = pz._predict_frames(model, D._Frames(items, bool(cfg.get('command_input', True))), g, cfg['num_modes'],
                                     batch, workers)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        return res['actions'].astype(np.float32), g, sub, D.ref_of(cfg)
    return fn


def items_of(R):
    return [(str(R['img'][i]), R['tof'][i], R['imu'][i], R['vehicle'][i], int(R['mode'][i])) for i in range(len(R['img']))]


def pred_path(P, name):
    return P.pred / ('%s.npz' % name)


def load_pred(P, name, key):
    p = pred_path(P, name)
    if not D.npz_ok(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        if str(z['key']) != key:
            raise SystemExit('%s: 予測の表の key が集合と違います' % p)
        pred = z['pred'].astype(np.float64)
        ref = tuple(float(x) for x in z['ref'])
        meta = {k: str(z[k]) for k in ('best_sha256', 'reference') if k in z.files}
    if pred.ndim != 3 or pred.shape[1:] != (10, 2) or not np.isfinite(pred).all():
        raise SystemExit('%s: 予測の形か値がおかしい' % p)
    return {'pred': pred, 'ref': ref, 'path': str(p), 'sha256': sha256(p), **meta}


def crosscheck_b(P, fn, n_rows=256):
    """B を予測し直して、v2 の C3 の保存(B の tab)の最初の n_rows 行と照らす(≤ 0.01 µs)。予測の作りが今も同じか"""
    proto = D.protocol()
    S = D.load_sets(proto)
    rows = np.arange(min(n_rows, len(S['frame'])))
    items = D.frame_items(D.DS_V2, S['scene'][rows], S['frame'][rows])
    pred, _, _, _ = fn(B_RUN, D.RUNS / B_RUN, items, np.full(len(rows), G5))
    with np.load(D.PRED2 / ('%s_tab.npz' % B_RUN), allow_pickle=False) as z:
        saved = z['pred'][rows].astype(np.float64)
        same_key = str(z['key']) == S['key']
    mx = float(np.abs(np.asarray(pred, np.float64) - saved).max())
    out = {'checked_at': now_iso(), 'run': B_RUN, 'rows': int(len(rows)), 'max_abs_us': round(mx, 6),
           'limit_us': D.CROSS_LIMIT_US, 'same_key': same_key, 'pass': bool(same_key and mx <= D.CROSS_LIMIT_US),
           'train_data_sha256': sha256(REPO / 'jetformer' / 'train_data.py')}
    write_json(P.cross, out)
    log('B の予測し直し(v2 の保存の %d 行): %s' % (len(rows), json.dumps(out, ensure_ascii=False)))
    if not out['pass']:
        raise SystemExit('B の予測し直しが v2 の保存と合いません(≤ %.2f µs)' % D.CROSS_LIMIT_US)


def predict(P, names, fn, reference=False, force=False, check_b=True):
    rec = opened_record(P)
    R, meta = load_rows(P)
    allowed = rec['predict']['all']
    extra = [n for n in names if n not in allowed]
    if extra and not reference:
        raise SystemExit('封を開けたときの予測の一覧に無い名前です(開けたあとに足す腕は --reference で参考に): %s' % ' '.join(extra))
    if check_b and not P.dry:
        cj = load_json(P.cross) or {}
        if not cj.get('pass') or cj.get('train_data_sha256') != sha256(REPO / 'jetformer' / 'train_data.py'):
            crosscheck_b(P, fn)
    items = None if P.dry else items_of(R)
    graw = genres_raw_of(R)
    P.pred.mkdir(parents=True, exist_ok=True)
    for name in names:
        is_ref = name in extra
        if not force and load_pred(P, name, R['key']) is not None:
            log('%s: 予測はあります(飛ばす)' % name)
            continue
        rd = run_dir(name) if not P.dry else Path('/nonexistent-dry') / name
        wsha = None
        if not P.dry:
            w = rd / 'best.pth'
            if not w.exists():
                raise SystemExit('%s: best.pth がありません' % name)
            wsha = sha256(w)
            want = (rec.get('weights') or {}).get(name, {}).get('best_sha256')
            if not is_ref and wsha != want:
                raise SystemExit('%s: best.pth が封を開けたときと違います(開けたあとに作り直さない。main session に聞く)' % name)
        t0 = time.time()
        pred, g, sub, ref = fn(name, rd, items if items is not None else len(graw), graw)
        if pred.shape != (len(graw), 10, 2) or not np.isfinite(pred).all():
            raise SystemExit('%s: 予測の形か値がおかしい' % name)
        D.save_npz(pred_path(P, name), key=np.array(R['key']), run=np.array(name), ckpt=np.array('best'),
                   pred=np.asarray(pred, np.float32), genre=np.asarray(g).astype(np.int8), substituted=np.asarray(sub, bool),
                   ref=np.asarray(ref, np.float64), best_sha256=np.array(wsha or ''), reference=np.array('1' if is_ref else ''),
                   dry_run=np.array(bool(P.dry)), script_sha256=np.array(sha256(Path(__file__).resolve())))
        log('%s: 予測 %d 行 %.0f 秒%s' % (name, len(pred), time.time() - t0, '(参考)' if is_ref else ''))
    return 0


# ================================================================ 欄と判定(CPU。numpy だけ)

def l1_rows(pc, R, mask):
    m = np.asarray(mask, bool) & np.asarray(R['y_valid'], bool)
    if not m.any():
        return None
    return float(np.mean(np.abs(np.asarray(pc)[m] - np.asarray(R['y'], np.float64)[m])))


def a_diff(pc, refc, R, mask):
    """A: 記録の先読み・ゲイン・トリムで車に出す値にした候補の予測と、記録の model_out(B が出した値)の |差| の平均(ステア・スロットル)"""
    m = np.asarray(mask, bool)
    if not m.any():
        return None
    D2 = load_d2()
    d = D2.stage0_2_diffs(np.asarray(pc)[m], R['model_out'][m], refc, R['gains'][m], R['trim'][m], R['lookahead'][m])
    return float(np.mean(np.abs(d)))


def fields_of(pc, pb, refc, refb, R):
    """1 本のモデルの封の欄(protocol seal.fields_at_open)。pc・pb は全部の行の予測 [N,10,2]"""
    f = {'seal_t_l1_us': l1_rows(pc, R, R['m_T']),
         'seal_v_thr_vs_b_us': D.thr_vs_b(pc, pb, refc, refb, R['m_V'], GAIN, GAIN, ROW2),
         'seal_h_l1_us': l1_rows(pc, R, R['m_H']),
         'seal_a_us': a_diff(pc, refc, R, R['m_A']),
         'seal_de_pre_thr_vs_b_us': D.thr_vs_b(pc, pb, refc, refb, R['m_Dpre'], GAIN, GAIN, ROW2),
         'seal_de_overshoot_vs_b_us': D.steer_vs_b(pc, pb, R['m_Dpass'], ROW0),
         'seal_standstill_go_frac': D.go_frac(pc, refc, R['m_S'], GAIN),
         'seal_standstill_back_frac': D.back_frac(pc, refc, R['m_S'], GAIN),
         'seal_bfast_gb_fwd_vs_b_us': None,
         # 参考の参考: desk の周の決まりの直線・DE の通過、両方の回の止まった場面
         'ref_seal_v_thr_vs_b_us_desk_laps': D.thr_vs_b(pc, pb, refc, refb, R['m_V_desk'], GAIN, GAIN, ROW2),
         'ref_seal_de_overshoot_vs_b_us_desk_laps': D.steer_vs_b(pc, pb, R['m_Dpass_desk'], ROW0),
         'ref_seal_standstill_go_frac_both': D.go_frac(pc, refc, R['m_Sall'], GAIN)}
    m = np.asarray(R['m_T'], bool)
    if m.any():
        dd = np.abs(np.asarray(pc)[m] - np.asarray(R['y'], np.float64)[m])
        f['seal_t_l1_steer_us'] = float(dd[:, :, 0].mean())
        f['seal_t_l1_thr_us'] = float(dd[:, :, 1].mean())
    return f


def veto_one(t, v, st):
    """1 つの候補の封の T・V の拒否権。st = {'T': (平均, sd), 'V': (平均, sd)}。→ dict(veto: True・False・None)"""
    out = {'t': t, 'v': v}
    tm, tsd = st['T']
    vm, vsd = st['V']
    out['t_thr'] = None if tsd is None else SE_K * tsd
    out['v_thr'] = None if vsd is None else max(V_MIN_US, SE_K * vsd)
    out['t_diff'] = None if (t is None or tm is None) else t - tm
    out['v_diff'] = None if (v is None or vm is None) else v - vm
    out['veto_T'] = None if (out['t_diff'] is None or out['t_thr'] is None) else bool(out['t_diff'] > out['t_thr'])
    out['veto_V'] = None if (out['v_diff'] is None or out['v_thr'] is None) else bool(out['v_diff'] > out['v_thr'])
    if out['veto_T'] or out['veto_V']:
        out['veto'] = True
    elif out['veto_T'] is None or out['veto_V'] is None:
        out['veto'] = None
    else:
        out['veto'] = False
    return out


def choose(order, cands, vetoes):
    """順に見て、拒否権の無い最初 = 1 枠。拒否権が決まらない候補で止まる(後ろを見ない)。→ (枠の名前, 枠の印, 状態, 理由)"""
    seen = []
    for lab in order:
        name = cands.get(lab)
        if not name:
            continue
        v = vetoes.get(lab) or {}
        if v.get('veto') is None:
            return None, None, '決まらない', '%s(%s)の拒否権が計算できない(%s)。後ろの候補は見ない。main session に聞く' % (
                lab, name, v.get('why') or '値が無い')
        if v['veto'] is False:
            return name, lab, '決まった', '%s = %s(拒否権なし%s)' % (lab, name, '。前の %s は拒否権' % '・'.join(seen) if seen else '')
        seen.append(lab)
    if not seen:
        return None, None, '無し(B のまま)', '候補が無い'
    return None, None, '無し(B のまま)', '候補 %s が全部拒否権に落ちた' % '・'.join(seen)


def judge_core(fields, cands, v2_cands):
    """5 本の封の上の平均・sd → v3 の候補に T・V、v2 の候補に T。→ dict"""
    missing = [n for n in FIVE_RUNS if n not in fields]
    st = {'T': mean_sd([fields[n].get('seal_t_l1_us') for n in FIVE_RUNS]) if not missing else (None, None),
          'V': mean_sd([fields[n].get('seal_v_thr_vs_b_us') for n in FIVE_RUNS]) if not missing else (None, None)}
    if any(fields.get(n, {}).get(k) is None for n in FIVE_RUNS for k in ('seal_t_l1_us', 'seal_v_thr_vs_b_us')):
        st = {'T': (None, None), 'V': (None, None)}
    out = {'five': {'runs': FIVE_RUNS, 'missing': missing,
                    'T': {'values': {RID_OF[n]: fields.get(n, {}).get('seal_t_l1_us') for n in FIVE_RUNS},
                          'mean': st['T'][0], 'sd': st['T'][1]},
                    'V': {'values': {RID_OF[n]: fields.get(n, {}).get('seal_v_thr_vs_b_us') for n in FIVE_RUNS},
                          'mean': st['V'][0], 'sd': st['V'][1]}}}
    vet3 = {}
    for lab in V3_ORDER:
        name = cands.get(lab)
        if not name:
            continue
        f = fields.get(name)
        if f is None:
            vet3[lab] = {'run': name, 'veto': None, 'why': '予測が無い'}
            continue
        vet3[lab] = dict(veto_one(f.get('seal_t_l1_us'), f.get('seal_v_thr_vs_b_us'), st), run=name)
        if vet3[lab]['veto'] is None:
            vet3[lab]['why'] = '集合が 0 フレームか 5 本がそろわない'
    slot, lab, status, why = choose(V3_ORDER, cands, vet3)
    out['v3'] = {'order': list(V3_ORDER), 'candidates': dict(cands), 'vetoes': vet3, 'slot': slot, 'slot_label': lab,
                 'status': status, 'why': why}
    vet2 = {}
    for lab2 in V2_ORDER:
        name = v2_cands.get(lab2)
        if not name:
            continue
        f = fields.get(name)
        if f is None:
            vet2[lab2] = {'run': name, 'veto': None, 'why': '予測が無い'}
            continue
        o = veto_one(f.get('seal_t_l1_us'), None, st)
        o['veto'] = o['veto_T']                    # v2 の封の拒否権は T だけ(protocol_v2 9 章)
        vet2[lab2] = dict(o, run=name, note='v2 は T だけ')
    s2, l2, st2, why2 = choose(V2_ORDER, v2_cands, vet2)
    out['v2_D5b'] = {'order': list(V2_ORDER), 'candidates': dict(v2_cands), 'vetoes': vet2, 'slot': s2, 'slot_label': l2,
                     'status': st2 if any(v2_cands.get(k) for k in V2_ORDER) else '無し(候補 0)',
                     'why': why2 if any(v2_cands.get(k) for k in V2_ORDER) else 'v2 の決まりで候補 0(D5a の C1・C2 と BF の BF1・BF2 が無し)'}
    return out


def recount(preds, R, cands, v2_cands):
    """決まりの計算を別の書き方で数え直す(protocol_v3 16 章): T は 1 行ずつ、V は車に出す式をスカラーで、
    平均・sd は statistics、拒否権と順は別に書いた if。→ {'T': {名前: 値}, 'V': {...}, 'slot_v3': 名前, 'slot_v2': 名前}"""
    def l1_loop(pc, mask):
        s, c = 0.0, 0
        for i in np.nonzero(mask)[0]:
            if not R['y_valid'][i]:
                continue
            for r in range(10):
                for ch in range(2):
                    s += abs(float(pc[i, r, ch]) - float(R['y'][i, r, ch]))
                    c += 1
        return s / c if c else None

    def out_thr(raw, ref):
        dt = raw - ref[1]
        t = ref[1] + (GAIN * dt if dt < 0 else dt)
        return min(max(t, 1000.0), 2000.0)

    def v_loop(pc, refc, pb, refb):
        idx = np.nonzero(R['m_V'])[0]
        if not len(idx):
            return None
        return sum(out_thr(float(pc[i, ROW2, 1]), refc) - out_thr(float(pb[i, ROW2, 1]), refb) for i in idx) / len(idx)

    if B_RUN not in preds:
        return None
    pb, refb = preds[B_RUN]['pred'], preds[B_RUN]['ref']
    T, V = {}, {}
    for n, p in preds.items():
        T[n] = l1_loop(p['pred'], R['m_T'])
        V[n] = v_loop(p['pred'], p['ref'], pb, refb)
    if any(n not in T or T[n] is None or V[n] is None for n in FIVE_RUNS):
        return {'T': T, 'V': V, 'slot_v3': None, 'slot_v2': None, 'complete': False}
    tm, tsd = statistics.mean([T[n] for n in FIVE_RUNS]), statistics.stdev([T[n] for n in FIVE_RUNS])
    vm, vsd = statistics.mean([V[n] for n in FIVE_RUNS]), statistics.stdev([V[n] for n in FIVE_RUNS])
    slot3 = None
    for lab in V3_ORDER:
        n = cands.get(lab)
        if not n:
            continue
        if n not in T or T[n] is None or V[n] is None:
            slot3 = '決まらない'
            break
        if not (T[n] - tm > 2 * tsd) and not (V[n] - vm > max(10.0, 2 * vsd)):
            slot3 = n
            break
    slot2 = None
    for lab in V2_ORDER:
        n = v2_cands.get(lab)
        if not n:
            continue
        if n not in T or T[n] is None:
            slot2 = '決まらない'
            break
        if not (T[n] - tm > 2 * tsd):
            slot2 = n
            break
    return {'T': T, 'V': V, 'T_mean': tm, 'T_sd': tsd, 'V_mean': vm, 'V_sd': vsd, 'slot_v3': slot3, 'slot_v2': slot2,
            'complete': True}


def judge(P):
    rec = opened_record(P)
    R, meta = load_rows(P)
    names = list(rec['predict']['all'])
    preds, missing, refs_extra = {}, [], []
    for n in names:
        p = load_pred(P, n, R['key'])
        if p is None:
            missing.append(n)
        else:
            preds[n] = p
    for f in sorted(P.pred.glob('*.npz')) if P.pred.exists() else []:
        n = f.stem
        if n.endswith('.tmp'):
            continue
        if n not in preds and n not in missing:
            p = load_pred(P, n, R['key'])
            if p is not None and p.get('reference') == '1':
                refs_extra.append(n)
                preds[n] = p
    cands = rec['predict']['v3']['candidates']
    v2c = rec['predict']['v2']['candidates']
    fields = {}
    if B_RUN in preds:
        pb, refb = preds[B_RUN]['pred'], preds[B_RUN]['ref']
        for n, p in preds.items():
            fields[n] = fields_of(p['pred'], pb, p['ref'], refb, R)
    J = judge_core({n: f for n, f in fields.items() if n not in refs_extra}, cands, v2c)
    b_rep = None
    if B_RUN in preds and np.asarray(R['m_A'], bool).any():   # 参考: B の予測で、封の回に B が出した値を作り直せるか(段 0 の 2 と同じ式)
        D2 = load_d2()
        m = np.asarray(R['m_A'], bool)
        dd = D2.stage0_2_diffs(preds[B_RUN]['pred'][m], R['model_out'][m], preds[B_RUN]['ref'], R['gains'][m], R['trim'][m],
                               R['lookahead'][m])
        b_rep = D2.stage0_2_summary(dd)
        b_rep['note'] = ('参考(決まりではない)。B が走った封の周で、B の予測を記録のゲイン・トリム・先読みで車に出す値にして、'
                         '記録の model_out と比べた(段 0 の 2 と同じ: 中央値 ≤ 10 µs)。外れたら集合か予測の作りを main session が見る。'
                         'T・V は同じ作りの 5 本との比べなので、答えはそのまま')
    rc = recount(preds, R, cands, v2c)
    same = None
    if rc and rc.get('complete'):
        diffs = []
        for n in fields:
            for key, rk in (('seal_t_l1_us', 'T'), ('seal_v_thr_vs_b_us', 'V')):
                a, b = fields[n].get(key), rc[rk].get(n)
                if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-6):
                    diffs.append('%s %s: %r 対 %r' % (n, key, a, b))
        want3 = J['v3']['slot'] if J['v3']['status'] != '決まらない' else '決まらない'
        if (rc['slot_v3'] or None) != (want3 or None):
            diffs.append('v3 の 1 枠: %r 対 %r' % (J['v3']['slot'], rc['slot_v3']))
        want2 = J['v2_D5b']['slot'] if J['v2_D5b']['status'] != '決まらない' else '決まらない'
        if (rc['slot_v2'] or None) != (want2 or None):
            diffs.append('v2 の 1 枠: %r 対 %r' % (J['v2_D5b']['slot'], rc['slot_v2']))
        same = not diffs
        J['recount'] = {'same': same, 'diffs': diffs}
        if not same:
            J['v3'].update(slot=None, slot_label=None, status='決まらない', why='数え直しが違う: %s' % ' / '.join(diffs))
    else:
        J['recount'] = {'same': None, 'diffs': ['予測がそろっていない']}
    if missing:
        J['v3'].update(slot=None, slot_label=None, status='まだ(予測が足りない)', why='予測が無い: %s' % ' '.join(missing))
        J['v2_D5b'].update(status='まだ(予測が足りない)' if any(v2c.values()) else J['v2_D5b']['status'])
    out = {'schema': SCHEMA, 'made_at': now_iso(), 'dry_run': P.dry, 'seal_opened': str(P.seal_opened),
           'seal_opened_sha256': sha256(P.seal_opened), 'rows': str(P.rows), 'rows_sha256': meta.get('rows_sha256'),
           'rows_key': R['key'], 'counts': counts_of(R), 'design_check': meta.get('design_check'),
           'predicted': sorted(preds), 'missing': missing, 'reference_after_open': refs_extra,
           'fields': fields, 'judge': J, 'b_reproduces_recorded': b_rep, 'script': str(Path(__file__).resolve()), 'script_sha256': sha256(Path(__file__).resolve()),
           'rules': rules_text()}
    write_json(P.e5_json, out)
    D.write_text(P.e5_md, e5_md(out, rec))
    prop = {'made_at': out['made_at'], 'dry_run': P.dry,
            'note': 'e5_open_seal.py judge の案。decisions_v3.json の E5・v2 の decisions.json の D5b に写すのは main session。'
                    'bench_candidates.json に入れるのも走らせるのもユーザーの了承のあと(add_v1_block.py)',
            'E5': {'slot': J['v3']['slot'], 'slot_label': J['v3']['slot_label'], 'status': J['v3']['status'],
                   'why': J['v3']['why'], 'order': list(V3_ORDER), 'candidates': dict(cands),
                   'vetoes': {k: {kk: v.get(kk) for kk in ('run', 'veto', 'veto_T', 'veto_V', 't', 'v', 't_diff', 'v_diff',
                                                           't_thr', 'v_thr', 'why')} for k, v in J['v3']['vetoes'].items()},
                   'five': J['five'], 'recount': J['recount'], 'e5_json': str(P.e5_json)},
            'v2_D5b': {'slot': J['v2_D5b']['slot'], 'status': J['v2_D5b']['status'], 'why': J['v2_D5b']['why']}}
    write_json(P.proposed, prop)
    log('1 枠: %s(%s)。v2 の D5b: %s' % (J['v3']['slot'] or '—', J['v3']['status'], J['v2_D5b']['status']))
    return out


def rules_text():
    return {'T': 'seal_t_l1_us − 5 本の平均(封の上)> 2 × 5 本の sd(封の上)→ 外す',
            'V': 'seal_v_thr_vs_b_us − 5 本の平均(封の上)> max(10 µs, 2 × 5 本の sd(封の上))→ 外す',
            'order_v3': ' → '.join(V3_ORDER), 'order_v2': ' → '.join(V2_ORDER) + '(v2 は T だけ)',
            'none': '全部落ちたら枠は無し(B のまま)。開けたあとに作り直さない',
            'sd': '標本の標準偏差(n − 1 で割る。protocol_numbers_v3 の 5 本の sd と同じ)'}


def fmt(v, nd=2):
    if v is None:
        return '—'
    if isinstance(v, bool):
        return 'あり' if v else 'なし'
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        return ('%.' + str(nd) + 'f') % float(v)
    except (TypeError, ValueError):
        return str(v)


def e5_md(o, rec):
    J = o['judge']
    L = ['# E5: 封を開けて決めた 11/3 の 1 枠%s' % ('(作り物の封)' if o['dry_run'] else ''), '']
    if o['dry_run']:
        L += ['> **作り物の封**(--dry-run)。本物の封は読んでいない・開けていない。予測も作り物で、学習の名前は形だけ'
              '(その学習の結果ではない)。台本の流れと書く物の形を見るためのもの。', '']
    L += [
         '- 作った時刻: %s。封を開けた記録: `%s`(%s)' % (o['made_at'], o['seal_opened'], rec.get('opened_at')),
         '- 決まり: protocol_v3 12 章(T・V の拒否権 → %s の順の最初。全部落ちたら B のまま)' % ' → '.join(V3_ORDER), '',
         '## 1. 答え', '',
         '| 項目 | 答え | 理由 |', '|---|---|---|',
         '| **11/3 の 1 枠(v3・E5)** | **%s** | %s |' % (J['v3']['slot'] or 'B のまま(枠は無し)' if J['v3']['status'].startswith('無し')
                                                   else (J['v3']['slot'] or '—'), J['v3']['status'] + '。' + J['v3']['why']),
         '| v2 の D5b | %s | %s |' % (J['v2_D5b']['slot'] or '無し', J['v2_D5b']['status'] + '。' + J['v2_D5b']['why']),
         '| 数え直し | %s | %s |' % ({True: '同じ', False: '違う', None: '—'}[J['recount']['same']],
                                   ' / '.join(J['recount']['diffs']) or '別の書き方の T・V・順が同じ'), '',
         '## 2. 5 本(封の上)', '', '| 物差し | A_s0 | A_s1 | v0-s2 | v0-s3 | B | 平均 | sd | 拒否権の線 |', '|---|---|---|---|---|---|---|---|---|']
    for k, nm, line in (('T', 'T(交代の窓の L1)[µs]', lambda sd: None if sd is None else SE_K * sd),
                        ('V', 'V(3 回目の直線・候補 − B)[µs]', lambda sd: None if sd is None else max(V_MIN_US, SE_K * sd))):
        f = J['five'][k]
        L.append('| %s | %s | %s | %s | %s |' % (nm, ' | '.join(fmt(f['values'].get(r)) for r in ('A_s0', 'A_s1', 'v0-s2', 'v0-s3', 'B')),
                                               fmt(f['mean']), fmt(f['sd'], 3), ('平均 + %s' % fmt(line(f['sd']), 3))))
    L += ['', '## 3. 候補', '', '| 枠 | 学習 | T | T − 平均 | 線 | V | V − 平均 | 線 | 拒否権 |', '|---|---|---|---|---|---|---|---|---|']
    for lab in V3_ORDER:
        v = J['v3']['vetoes'].get(lab)
        if not v:
            L.append('| %s | 無し | — | — | — | — | — | — | — |' % lab)
            continue
        L.append('| %s | `%s` | %s | %s | %s | %s | %s | %s | %s |' % (
            lab, v.get('run'), fmt(v.get('t')), fmt(v.get('t_diff')), fmt(v.get('t_thr'), 3), fmt(v.get('v')), fmt(v.get('v_diff')),
            fmt(v.get('v_thr'), 3), {True: '**あり**', False: 'なし', None: '決まらない'}[v.get('veto')] +
            ('(%s)' % '・'.join(x for x, y in (('T', v.get('veto_T')), ('V', v.get('veto_V'))) if y) if v.get('veto') else '')))
    L += ['', '## 4. 欄(参考も)', '',
          '| 学習 | T | H | A | V | V(desk の周) | DE 手前 | DE 行き過ぎ | 止まり go | 止まり 後ろ |', '|---|---|---|---|---|---|---|---|---|---|']
    for n, f in o['fields'].items():
        L.append('| `%s`%s | %s |' % (n, '(参考)' if n in o['reference_after_open'] else '', ' | '.join(fmt(f.get(k), 4 if 'frac' in k else 2) for k in (
            'seal_t_l1_us', 'seal_h_l1_us', 'seal_a_us', 'seal_v_thr_vs_b_us', 'ref_seal_v_thr_vs_b_us_desk_laps',
            'seal_de_pre_thr_vs_b_us', 'seal_de_overshoot_vs_b_us', 'seal_standstill_go_frac', 'seal_standstill_back_frac'))))
    c = o['counts']
    L += ['', '## 5. 集合の大きさ', '',
          '| 集合 | 行 | 中身 |', '|---|---|---|',
          '| T | %d(窓 %d) | 失敗の交代の窓(make_intervention_set と同じ作り)の採点するフレーム |' % (c['m_T'], c['T_windows']),
          '| V | %d | 3 回目の AI のきれいな周の直線(desk の周の決まりなら %d) |' % (c['m_V'], c['m_V_desk']),
          '| D 手前 | %d | 3 回目の DE の衝突の手前 1 秒 |' % c['m_Dpre'],
          '| D 通過 | %d | 3 回目の DE の入りから 0.5 秒(desk の周の決まりなら %d) |' % (c['m_Dpass'], c['m_Dpass_desk']),
          '| S | %d | 1 回目の止まった場面(前 1 m 空き。両方の回なら %d) |' % (c['m_S'], c['m_Sall']),
          '| A | %d | B のきれいな周の AI の側(記録の model_out と比べる) |' % c['m_A'],
          '| H | %d(間引き 3 で %d) | 人の走り・交代の直後(plan_checks の式) |' % (c['m_H'], c['H_samples_rows'])]
    br = o.get('b_reproduces_recorded')
    if br:
        L += ['', 'B の予測で封の回の B の出力を作り直せるか(参考。段 0 の 2 と同じ式): %s フレーム・|差| の中央値 %s µs・'
              '答え %s' % (br.get('frames'), fmt(br.get('median_abs_us_pooled')),
                         {True: '合格(≤ 10 µs)', False: '**外れ**(main session が作りを見る。答えはそのまま)', None: '決まらない'}[br.get('pass')])]
    if o.get('design_check'):
        L += ['', '設計のときの数と(止めない。注だけ%s):' % ('。作り物の封なので合わなくてよい' if o['dry_run'] else ''), '',
              '| 数 | 設計 | 今 | 同じ |', '|---|---|---|---|']
        for x in o['design_check']:
            L.append('| %s | %s | %s | %s |' % (x['what'], x['design'], x['got'], 'はい' if x['same'] else '**いいえ**'))
    if o['missing']:
        L += ['', '**予測が足りない**: %s(`e5_open_seal.py commands`)' % ' '.join(o['missing'])]
    L += ['', '- bench_candidates.json に入れるのも走らせるのもユーザーの了承のあと(`add_v1_block.py` は差分を出すだけ)。',
          '- 1 枠を使う決まり(protocol_v3 12 章): 2 本続けて 3 周の窓の中央値が 0.3 秒以上速く、DE・R1 の衝突/通過が増えないとき。'
          '見込みは「B より速くならない」(走らせるのは主に次の学習の材料のため)。', '']
    return '\n'.join(L)


# ================================================================ 照らし(封でない記録。CPU)

def cmp_arrays(a, b, keys):
    """同じ名前の配列が同じか(NaN は同じとみなす)。両方に無い項目は同じ(_aligned_arrays は人のサンプルが無いと vehicle_age_ms を作らない)"""
    out = {}
    for k in keys:
        if k not in a and k not in b:
            out[k] = {'same': True, 'why': '両方に無い'}
            continue
        if k not in a or k not in b:
            out[k] = {'same': False, 'why': '片方に無い(この台本 %s・保存 %s)' % (k in a, k in b)}
            continue
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            out[k] = {'same': False, 'why': '形 %s 対 %s' % (x.shape, y.shape)}
            continue
        if x.dtype.kind in 'fc':
            same = bool(np.array_equal(np.nan_to_num(x, nan=-12345.0), np.nan_to_num(y, nan=-12345.0)))
            mx = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0
            out[k] = {'same': same, 'max_abs': mx}
        else:
            out[k] = {'same': bool(np.array_equal(x, y))}
    return out


SELF_T_LOGS = ('20261010-171639', '20261010-173445')
SELF_AI_LOGS = ('20261024-162033', '20261024-160654', '20261010-173445', '20261023-210109', '20261024-162305')
SELF_H_LOGS = ('20261024-160133', '20261024-160536', '20261024-160654', '20261024-160847', '20261024-162033', '20261024-162305')
AL_KEYS = ('timestamps_us', 'tof', 'tof_age_ms', 'imu', 'vehicle', 'vehicle_valid', 'vehicle_age_ms', 'future_actions',
           'future_valid', 'erpm_valid', 'mode', 'source', 'label_index', 'label_valid')
T_KEYS = ('timestamps_us', 'tof', 'imu', 'vehicle', 'vehicle_valid', 'future_actions', 'future_valid', 'mode', 'correction_kind')
SET_MASKS = ('pre_r1', 'pre_de', 'pre_sw', 'de_pass', 'straight', 'stand')


def selfcheck(t_logs=SELF_T_LOGS, ai_logs=SELF_AI_LOGS, h_logs=SELF_H_LOGS):
    """封でない記録で、この台本の作りが今の物と同じかを照らす(読むだけ。封の記録は読まない)
      1. T: make_intervention_set の窓をその場で作る → intervention_val_20261011 の aligned・窓の一覧と同じか
      2. AI の側の aligned: その場で作る → dataset_v2_20261025 の aligned と同じか(シーンの一覧も)
      3. マスク(desk の周の決まり): sets_v2.npz の同じ記録の行と同じか(pre_r1・pre_de・pre_sw・de_pass・straight・stand)
      4. H: plan_checks.json の「10/24 2 回目」の run_or_tk(間引き 3)と同じか(10/24 2 回目の 6 本)
      5. A: この台本の A の行(B のきれいな周)の記録の値(model_out・ゲイン・トリム・先読み)が sets_v2 の s0_2 の行と同じで、
         B の保存した予測(v2 の C3 の B の tab)から作り直すと段 0 の 2 と同じに合うか(中央値 ≤ 10 µs。10/24 2 回目の B の記録)
      参考: 封の周の決まり(lap_rule='seal')の直線・DE の通過と A・H の数"""
    sealed = set(seal_logs())
    if (set(t_logs) | set(ai_logs) | set(h_logs)) & sealed:
        raise SystemExit('照らしに封の記録は使いません')
    res = {'made_at': now_iso(), 'script_sha256': sha256(Path(__file__).resolve()), 'parts': {}}
    import build_ai_dataset as bad
    # 1. T
    card = load_json(D.DATASETS / 'intervention_val_20261011' / 'dataset_card.json')
    t_part = {'logs': list(t_logs), 'windows': [], 'same': True}
    for t_log in t_logs:
        info, arr = bad.load_labels(t_log)
        wins, _, ntk = t_windows(t_log, info, arr)
        want = [w for w in card['intervention']['windows'] if w['log'] == t_log]
        t_part['same'] &= len(want) == len(wins)
        for w, cw in zip(wins, want):
            e = {'name': w['name'], 'scored': w['scored'], 'card_scored': cw['scored'], 'included': w['included'],
                 'card_included': cw['included'], 'class': w['class'], 'card_class': cw['class']}
            ok = w['name'] == cw['name'] and w['scored'] == cw['scored'] and w['included'] == cw['included'] \
                and w['class'] == cw['class']
            if w['included']:
                with np.load(D.DATASETS / 'intervention_val_20261011' / 'aligned' / ('%s.npz' % w['name']),
                             allow_pickle=False) as z:
                    ref = {k: z[k] for k in z.files}
                c = cmp_arrays(w['a'], ref, T_KEYS)
                e['diff'] = {k: v for k, v in c.items() if not v['same']}
                ok &= not e['diff']
            e['same'] = bool(ok)
            t_part['same'] &= bool(ok)
            t_part['windows'].append(e)
    t_part['same'] = bool(t_part['same'])
    res['parts']['T_vs_intervention_val'] = t_part
    log('照らし 1(T): %s(窓 %d)' % ('同じ' if t_part['same'] else '違う', len(t_part['windows'])))
    # 2・3. AI の側の aligned とマスク
    po, sess = extract_inventory(list(dict.fromkeys(list(ai_logs) + list(h_logs))), iv_windows_1010())
    from jetformer.tof_guard import ToFGuard
    guard = ToFGuard(np.load(D.REPO / 'data' / 'tof_floor_ref.npy'))
    with np.load(D.SETS_NPZ, allow_pickle=False) as z:
        S = {k: z[k] for k in ('scene', 'frame', 'log', 's0_2', 'model_out', 'gains', 'trim', 'lookahead') + SET_MASKS}
    with np.load(D.PRED2 / ('%s_tab.npz' % B_RUN), allow_pickle=False) as z:
        b_tab = z['pred'].astype(np.float64)
    ref_b = D.ref_of(D.model_cfg(D.RUNS / B_RUN))
    arow = {'rows': [], 'model_out': [], 'gains': [], 'trim': [], 'lookahead': [], 'not_in_s0_2': 0}
    splits = load_json(D.DATASETS / D.DS_V2 / 'splits.json') or {}
    a_part = {'logs': list(ai_logs), 'scenes': [], 'same': True}
    m_part = {'logs': list(ai_logs), 'masks': {}, 'same': True}
    seal_rule = {'straight': 0, 'de_pass': 0, 'A': 0, 'H': 0}
    tot = {}
    for ai_log in ai_logs:
        info, arr, L, scenes, skipped = ai_log_labels(ai_log)
        d = po[ai_log]
        ctx = log_context(d, 'desk')
        ctx_s = log_context(d, 'seal')
        want_sc = sorted(s_ for s_ in splits.get('train', []) if s_.rsplit('-', 1)[0] == ai_log)
        if sorted(scenes) != want_sc:
            a_part['same'] = False
            a_part.setdefault('scene_list_diff', {})[ai_log] = {'mine': sorted(scenes), 'dataset_v2': want_sc}
        for sname in scenes:
            a, files, note = scene_aligned(sname, L)
            fp = D.DATASETS / D.DS_V2 / 'aligned' / ('%s.npz' % sname)
            if not fp.exists():
                a_part['scenes'].append({'scene': sname, 'same': False, 'diff': 'dataset_v2 に無いシーン'})
                a_part['same'] = False
                continue
            with np.load(fp, allow_pickle=False) as z:
                ref = {k: z[k] for k in z.files}
            c = cmp_arrays(a, ref, AL_KEYS)
            diff = {k: v for k, v in c.items() if not v['same']}
            a_part['scenes'].append({'scene': sname, 'same': not diff, 'diff': diff, 'frames': int(len(files))})
            a_part['same'] &= not diff
            M, k, tl, xf = scene_masks_e5(d, ctx, a, guard)
            Ms, _, _, _ = scene_masks_e5(d, ctx_s, a, guard)
            seal_rule['straight'] += int(Ms['straight'].sum())
            seal_rule['de_pass'] += int(Ms['de_pass'].sum())
            seal_rule['A'] += int(Ms['A'].sum())
            seal_rule['H'] += int(Ms['H'].sum())
            sel = S['scene'] == sname
            jrow = {int(f): int(j) for j, f in zip(np.nonzero(sel)[0], S['frame'][sel])}
            for f in np.nonzero(M['A'])[0]:
                j = jrow.get(int(f))
                if j is None or not S['s0_2'][j]:
                    arow['not_in_s0_2'] += 1
                    continue
                kk = k[f]
                arow['rows'].append(j)
                arow['model_out'].append(d['model_out'][kk])
                arow['gains'].append(d['gains'][kk])
                arow['trim'].append(d['trim'][kk])
                arow['lookahead'].append(d['lookahead'][kk])
            for mname in SET_MASKS:
                mine = set(np.nonzero(M[mname])[0].tolist())
                theirs = set(S['frame'][sel & S[mname].astype(bool)].tolist())
                t = tot.setdefault(mname, {'mine': 0, 'sets_v2': 0, 'only_mine': 0, 'only_sets_v2': 0})
                t['mine'] += len(mine)
                t['sets_v2'] += len(theirs)
                t['only_mine'] += len(mine - theirs)
                t['only_sets_v2'] += len(theirs - mine)
            del a
    for mname, t in tot.items():
        t['same'] = t['only_mine'] == 0 and t['only_sets_v2'] == 0
        m_part['masks'][mname] = t
        m_part['same'] &= t['same']
    a_part['same'] = bool(a_part['same'])
    m_part['same'] = bool(m_part['same'])
    res['parts']['aligned_vs_dataset_v2'] = a_part
    res['parts']['masks_vs_sets_v2'] = m_part
    res['reference_seal_rule_counts'] = dict(seal_rule, note='同じ記録で、封の周の決まり(9.5 秒を掛けない)の直線・DE の通過と A・H の数(参考)')
    log('照らし 2(aligned): %s(シーン %d)・3(マスク): %s %s' % (
        '同じ' if a_part['same'] else '違う', len(a_part['scenes']), '同じ' if m_part['same'] else '違う',
        json.dumps({m: (t['mine'], t['sets_v2']) for m, t in tot.items()}, ensure_ascii=False)))
    # 4. H
    pc = load_json(Path('/home/jetson/jetformer_reports/dataset_v2_plan_20261025/plan_checks.json')) or {}
    want_h = ((pc.get('new_human') or {}).get('10/24 2 回目') or {}).get('run_or_tk')
    got_h = 0
    for lg in h_logs:
        dd = po[lg]
        got_h += int((dd['h_new'] & (dd['h_run'] | dd['h_tk']) & dd['h_sample']).sum())
    res['parts']['H_vs_plan_checks'] = {'logs': list(h_logs), 'plan_checks_run_or_tk': want_h, 'got': got_h,
                                        'same': got_h == want_h}
    log('照らし 4(H): plan_checks %s・今 %s' % (want_h, got_h))
    # 5. A
    j = np.array(arow['rows'], np.int64)
    a5 = {'rows': int(len(j)), 'not_in_s0_2': arow['not_in_s0_2']}
    if len(j):
        mo, g = np.array(arow['model_out'], np.float64), np.array(arow['gains'], np.float64)
        tr, la = np.array(arow['trim'], np.float64), np.array(arow['lookahead'], np.int64)
        a5['meta_same_as_sets_v2'] = bool(np.array_equal(mo.astype(np.float32), S['model_out'][j]) and
                                          np.array_equal(g.astype(np.float32), S['gains'][j]) and
                                          np.array_equal(tr.astype(np.float32), S['trim'][j]) and
                                          np.array_equal(la.astype(np.int16), S['lookahead'][j]))
        D2 = load_d2()
        a5['b_reproduces_recorded'] = D2.stage0_2_summary(D2.stage0_2_diffs(b_tab[j], mo, ref_b, g, tr, la))
        a5['seal_a_us_B'] = a_diff(b_tab[j], ref_b, {'model_out': mo, 'gains': g, 'trim': tr, 'lookahead': la},
                                   np.ones(len(j), bool))
        a5['same'] = bool(a5['meta_same_as_sets_v2'] and a5['b_reproduces_recorded'].get('pass') is True
                          and a5['not_in_s0_2'] == 0)
    else:
        a5['same'] = False
    res['parts']['A_vs_stage0_2'] = a5
    log('照らし 5(A): 行 %d・記録の値 %s・B の作り直し %s(中央値 %s µs)' % (
        a5['rows'], a5.get('meta_same_as_sets_v2'), (a5.get('b_reproduces_recorded') or {}).get('pass'),
        (a5.get('b_reproduces_recorded') or {}).get('median_abs_us_pooled')))
    res['same'] = all(p_['same'] for p_ in res['parts'].values())
    return res


# ================================================================ コマンド

def cmd_plan(args, P, cands):
    pl = plan_of(cands)
    print(json.dumps({'candidates': cands, 'plan': pl, 'seal_opened': str(P.seal_opened), 'seal_opened_exists': P.seal_opened.exists()},
                     ensure_ascii=False, indent=1))
    print('\n答え: %s' % pl['decision'])
    if not P.dry:
        d = today()
        if d < OPEN_DATE:
            print('今日は %s。封を開けるのは %s の朝(open は 11/1 より前は止まる)' % (d, OPEN_DATE))
    return 0


def cmd_open(args, P, cands, checks=fix_checks, date_fn=today):
    """封を開ける(1 回だけ): 確かめ → seal_opened.json → 集合"""
    pl = plan_of(cands)
    probs = []
    if P.seal_opened.exists():
        raise SystemExit('封はもう開けました(%s)。1 回だけ。作り直さない(main session に聞く)' % P.seal_opened)
    if not P.dry:
        if under_pytest() and P.seal_opened.resolve() == REAL_SEAL_OPENED.resolve():
            raise SystemExit('試験の中で本物の封を開けようとしました')
        d = date_fn()
        if d < OPEN_DATE:
            probs.append('今日は %s。開けるのは %s の朝(11/1 より前は開けない)' % (d, OPEN_DATE))
        elif d > RUN_DAY:
            probs.append('今日は %s。11/3 のあとは開けない(main session に聞く)' % d)
    if not cands['final']:
        probs += cands['problems']
    elif pl['n_candidates'] == 0 and not args.user_approved_open_without_candidates:
        print('答え: %s' % pl['decision'])
        print('封は開けていません(候補 0)。B のまま。開けるかはユーザーが決める'
              '(決めたら --user-approved-open-without-candidates "<ユーザーの言葉>")')
        return 0
    if probs:
        raise SystemExit('封を開けません: ' + ' / '.join(probs))
    ck = None
    if not P.dry:
        import shutil
        free_gb = shutil.disk_usage(str(REP3)).free / 1e9
        if free_gb < MIN_FREE_GB:
            raise SystemExit('空きディスクが %.1f GB(%.1f GB 以上で開ける。集合と予測で約 0.1 GB)' % (free_gb, MIN_FREE_GB))
        ck = checks()
        bad = [c for c in ck if c['rc'] != 0]
        if bad:
            raise SystemExit('固定の確かめが合いません(封を開けない): %s' % json.dumps(bad, ensure_ascii=False))
        w = weights_of(pl['predict']['all'])
        miss = [n for n, x in w.items() if x.get('missing')]
        if miss:
            raise SystemExit('予測するモデルの best.pth か model_config.json が無い(封を開けない): %s' % ' '.join(miss))
    else:
        w = {n: {'dry_run': True} for n in pl['predict']['all']}
    rec = {'schema': SCHEMA, 'dry_run': P.dry, 'opened_at': now_iso(),
           'opened_by': 'experiments/20261026_fix/fix3/e5_open_seal.py open(main session)',
           'date_note': None if P.dry or date_fn() == OPEN_DATE else '11/1 ではない日に開けた(deviations_v3.md に書く)',
           'rule': '封(10/24 1 回目・3 回目の 14 本)を開けるのは 1 回だけ(v2 と一緒)。この記録があれば desk_v2・desk_v3・この台本は止まる',
           'seal_logs': seal_logs() if not P.dry else (args.dry_logs or ['作り物']),
           'seal_manifest_sha256': D.protocol2()['seal']['manifest_sha256'],
           'protocol_v2_sha256': sha256(PROTOCOL2), 'protocol_v2_lock2_sha256': sha256(LOCK2_V2) if LOCK2_V2.exists() else None,
           'protocol_v3_sha256': sha256(PROTOCOL3), 'protocol_v3_lock2_sha256': sha256(LOCK2_V3) if LOCK2_V3.exists() else None,
           'decisions_v3_sha256': sha256(DECISIONS3) if DECISIONS3.exists() and not P.dry else None,
           'decisions_v2_sha256': sha256(DECISIONS2) if DECISIONS2.exists() else None,
           'candidates': cands, 'predict': pl['predict'], 'weights': w, 'fix_checks': ck,
           'user_approved_open_without_candidates': args.user_approved_open_without_candidates,
           'script': str(Path(__file__).resolve()), 'script_sha256': sha256(Path(__file__).resolve())}
    write_once(P.seal_opened, rec)
    log('封を開けた記録を書きました: %s' % P.seal_opened)
    return cmd_sets(args, P)


def cmd_sets(args, P):
    rec = opened_record(P)
    if D.npz_ok(P.rows) and (load_json(P.rows_json) or {}).get('rows_sha256') == sha256(P.rows):
        log('集合はあります: %s' % P.rows)
        return 0
    if P.dry and not (args.dry_logs1 or args.dry_logs3):
        R = synthetic_rows()
        meta = {'logs': ['作り物'], 'synthetic': True, 'notes': ['全部が作り物(データを読まない)']}
        save_rows(P, R, meta)
        log('作り物の集合: %s' % json.dumps(counts_of(R), ensure_ascii=False))
        return 0
    if P.dry:
        sealed = set(seal_logs())
        logs = list(args.dry_logs1 or []) + list(args.dry_logs3 or [])
        if set(logs) & sealed:
            raise SystemExit('作り物の封に本物の封の記録は使いません: %s' % ' '.join(sorted(set(logs) & sealed)))
        sess_of = dict([(lg, 1) for lg in args.dry_logs1 or []] + [(lg, 3) for lg in args.dry_logs3 or []])
    else:
        if under_pytest():
            raise SystemExit('試験の中で本物の封の記録を読もうとしました')
        logs = list(rec['seal_logs'])
        if logs != seal_logs():
            raise SystemExit('seal_opened.json の記録が protocol の封と違います')
        sess_of = seal_sessions()
    arrs, meta = build_rows(logs, sess_of, 'seal')
    meta['from_logs'] = True
    inv_sess = {lg: (meta['per_log'][lg] or {}).get('session_inventory') for lg in logs}
    if not P.dry:
        bad = [lg for lg in logs if not str(inv_sess[lg] or '').startswith(SESS_NAME[sess_of[lg]])]
        if bad:
            raise SystemExit('inventory の回と protocol の回が違う記録: %s' % ' '.join(bad))
    meta = save_rows(P, arrs, meta)
    log('集合を書きました: %s(%s)' % (P.rows, json.dumps(meta['counts'], ensure_ascii=False)))
    return 0


def gpu_line(names):
    G = ('/home/jetson/bin/jf-heavy --lock big --mem 3872M --need-mb 3200 -- nice -n 10 env '
         'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 /home/jetson/miniconda/envs/jetracer/bin/python -u')
    return '%s experiments/20261026_fix/fix3/e5_open_seal.py predict --runs %s' % (G, ' '.join(names))


def commands_text(pl):
    P_ = '/home/jetson/miniconda/envs/jetracer/bin/python'
    L_ = '/home/jetson/bin/jf-heavy --lock none --mem 3G --need-mb 1500 --'
    E = 'experiments/20261026_fix/fix3/e5_open_seal.py'
    out = ['cd ~/jetformer',
           '# 0. 固定の確かめ(open も自分で流す。外れたら開けない)',
           '(cd /home/jetson/jetformer_reports/fix_20261026 && sha256sum -c protocol_v3.sha256 && sha256sum -c protocol_v3_lock2.sha256)',
           '(cd /home/jetson/jetformer_reports/train_v2_20261025 && sha256sum -c protocol_v2.sha256 && sha256sum -c protocol_v2_lock2.sha256)',
           'python3 -I experiments/20261025_train_v2/check_seal_v2.py',
           '# 1. 読むだけ: 候補・開けるか',
           '%s -I %s plan' % (P_, E)]
    out += ['# 今の答え: %s' % pl['decision']]
    if not pl['n_candidates']:
        out += ['# 候補 0 なら B のまま。封は開けない(開けるかはユーザーが決める。開けるなら open に --user-approved-open-without-candidates "<ユーザーの言葉>")']
    out += ['# 2. 封を開ける(11/1 朝・1 回だけ。CPU。seal_opened.json → 集合 約 1〜3 分の見込み)',
            '%s %s -I %s open' % (L_, P_, E),
            '# 3. 予測(GPU。学習・エンジンのあと。1 回に 1〜2 本・あいだ 60 秒。初回は B の予測し直しを v2 の保存と照らす)']
    names = pl['predict']['all']
    for i in range(0, len(names), 2):
        out.append(gpu_line(names[i:i + 2]))
    out += ['# 4. 判定(CPU)', '%s %s -I %s judge' % (L_, P_, E),
            '# 5. 1 枠があれば: 11/3 の bench_candidates.json の差分(書かない。書くのはユーザーの了承のあと main session)',
            '%s -I experiments/20261026_fix/fix3/add_v1_block.py --dry-run' % P_]
    return '\n'.join(out)


def run_all_dry(args, P):
    dec3 = dry_decisions(args.scenario)
    dec2 = load_json(DECISIONS2) or {}
    cands = read_candidates(dec3, dec2)
    log('作り物の封(%s): %s' % (args.scenario, P.root))
    cmd_plan(args, P, cands)
    pl = plan_of(cands)
    if not pl['n_candidates']:
        log('候補 0 → B のまま。封は開けない(作り物でも開けない)')
        write_json(P.root / 'plan_only.json', {'scenario': args.scenario, 'plan': pl, 'opened': P.seal_opened.exists()})
        return 0
    cmd_open(args, P, cands)
    R, _ = load_rows(P)
    predict(P, pl['predict']['all'], fake_predict_fn(args.scenario, cands['v3'], R), check_b=False)
    out = judge(P)
    print(D.load_json(P.proposed) and json.dumps(D.load_json(P.proposed)['E5'], ensure_ascii=False, indent=1)[:3000])
    return 0 if out else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', nargs='?', default='plan', choices=('plan', 'commands', 'selfcheck', 'open', 'sets', 'predict',
                                                               'judge', 'all'))
    ap.add_argument('--dry-run', action='store_true', help='作り物の封(本物の封を読まない・開けない)')
    ap.add_argument('--scenario', default='veto_v', choices=SCENARIOS, help='作り物の封の場面')
    ap.add_argument('--dry-dir', help='作り物の封の書く場所(既定 fix3/e5_dryrun/<場面>-<時刻>)')
    ap.add_argument('--dry-logs1', nargs='*', default=[], help='作り物の封の「1 回目」にする封でない記録')
    ap.add_argument('--dry-logs3', nargs='*', default=[], help='作り物の封の「3 回目」にする封でない記録')
    ap.add_argument('--runs', nargs='*', default=[])
    ap.add_argument('--reference', action='store_true', help='開けたときの一覧に無いモデルを参考に予測する(1 枠に入れない)')
    ap.add_argument('--force', action='store_true', help='予測し直す(同じ best.pth のときだけ)')
    ap.add_argument('--user-approved-open-without-candidates', default=None, metavar='TEXT',
                    help='候補 0 でも開けるとユーザーが決めたときだけ(ユーザーの言葉をそのまま)')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args(argv)
    args.dry_logs = list(args.dry_logs1) + list(args.dry_logs3)
    if args.cmd == 'selfcheck':
        res = selfcheck()
        write_json(SELFCHECK_JSON, res)
        print(json.dumps({k: v.get('same') for k, v in res['parts'].items()}, ensure_ascii=False))
        log('書きました: %s(全部同じ: %s)' % (SELFCHECK_JSON, res['same']))
        return 0 if res['same'] else 1
    if args.dry_run:
        root = Path(args.dry_dir) if args.dry_dir else DRY_ROOT / ('%s-%s' % (args.scenario, time.strftime('%Y%m%d-%H%M%S')))
        P = dry_paths(root)
        if args.cmd == 'all':
            return run_all_dry(args, P)
        cands = read_candidates(dry_decisions(args.scenario), load_json(DECISIONS2) or {})
    else:
        if args.cmd == 'all':
            raise SystemExit('all は --dry-run だけ(本物は open → predict → judge を順に)')
        P = real_paths()
        cands = read_candidates(load_json(DECISIONS3) or {}, load_json(DECISIONS2) or {})
    if args.cmd == 'plan':
        return cmd_plan(args, P, cands)
    if args.cmd == 'commands':
        print(commands_text(plan_of(cands)))
        return 0
    if args.cmd == 'open':
        return cmd_open(args, P, cands)
    if args.cmd == 'sets':
        return cmd_sets(args, P)
    if args.cmd == 'predict':
        if not args.runs:
            raise SystemExit('--runs を付けてください')
        if P.dry:
            R, _ = load_rows(P)
            fn = fake_predict_fn(args.scenario, cands['v3'], R)
        else:
            import rl_policies as rp
            import desk_gpu
            busy = desk_gpu.web_busy()        # GET の状態だけ
            if busy:
                raise SystemExit('始めません: %s' % busy)
            rp._check_gpu_free(min_gb=2.0)
            fn = gpu_predict_fn(args.batch, args.workers)
        return predict(P, list(dict.fromkeys(args.runs)), fn, args.reference, args.force)
    if args.cmd == 'judge':
        judge(P)
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
