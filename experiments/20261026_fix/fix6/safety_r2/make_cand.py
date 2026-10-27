"""fix6 係 safety 直す 6-2 回目: 比べに使う候補 cand/assist.py を、今の版(before/assist.py.orig = fix6 r1 のあと)から作る。
どの直しも EXP のスイッチで入れたり外したりできる(既定は全部 False = 今の版と同じ動き)。drive.py は変えない(写しだけ)。
    python -I make_cand.py
"""
import hashlib
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'before' / 'assist.py.orig'
DST = HERE / 'cand' / 'assist.py'

REPL = [
    # EXP のスイッチ
    ("""class Recovery(object):
    STOP_ERPM = 300""",
     """EXP = {'seen_only': False, 'obs_check': False, 'drop_retry': False, 'drop_gap': False, 'drop_rev': False, 'drop_rev_w': False, 'drop_rev_z': False,
       'z_erpm': 0.0}


class Recovery(object):
    STOP_ERPM = 300"""),
    # 見積もりで足した秒を別に持つ
    ("""        self.backed_s = 0.0
        self.t_last = None
        self.turn = 0.0""",
     """        self.backed_s = 0.0
        self.est_s = 0.0
        self.est_t = None
        self.t_last = None
        self.turn = 0.0"""),
    ("""        self.retries = 0
        self.backed_s = 0.0
        self.back_t = None
        self._go('brake', now)""",
     """        self.retries = 0
        self.backed_s = 0.0
        self.est_s = 0.0
        self.est_t = None
        self.back_t = None
        self._go('brake', now)"""),
    ("""            if est > 0.0:       # fix6 s3-a: 見ていない間の後退の見積もり。後退を HOLD_EST_MIN_LEFT_S 残すところまで足す
                self.backed_s += min(est, max(0.0, rev_s - HOLD_EST_MIN_LEFT_S - self.backed_s))""",
     """            if est > 0.0:       # fix6 s3-a: 見ていない間の後退の見積もり。後退を HOLD_EST_MIN_LEFT_S 残すところまで足す
                add = min(est, max(0.0, rev_s - HOLD_EST_MIN_LEFT_S - self.backed_s))
                self.backed_s += add
                self.est_s += add
                self.est_t = now
                self.est_moved = False"""),
    ("""        if seen > 0.0 or est > 0.0:                     # 後ろへ動いていたサンプルか、見ていない間の後退の見積もりがあった""",
     """        if seen > 0.0 or (est > 0.0 and not EXP['seen_only']):"""),
    ("""            if self.backed_s == 0.0 and now - self.t_check > CHECK_S:
                if self.retries < self.MAX_RETRIES:
                    self.retries += 1
                    self._go('gap', now)""",
     """            obs = (self.backed_s - self.est_s) if EXP['obs_check'] else self.backed_s
            if obs == 0.0 and now - self.t_check > CHECK_S:
                if self.retries < self.MAX_RETRIES:
                    self.retries += 1
                    if EXP['drop_retry']:
                        self.backed_s = self.est_s = 0.0
                    self._go('gap', now)"""),
    ("""        if self.phase is None:
            if not self._detect(model_us, erpm, now, impact_g):""",
     """        if self.est_s and erpm > EXP['z_erpm']:
            self.est_moved = True
        if self.phase is None:
            if not self._detect(model_us, erpm, now, impact_g):"""),
    ("""            if el < gap_s:
                return (rev_steer, self.n + GAP_OFFSET_US), ASSIST_RECOVERY
            self._go('reverse', now)""",
     """            if el < gap_s:
                return (rev_steer, self.n + GAP_OFFSET_US), ASSIST_RECOVERY
            if EXP['drop_rev'] and self.est_s and self.backed_s - self.est_s == 0.0:
                self.backed_s = self.est_s = 0.0
            if (EXP['drop_rev_w'] and self.est_s and self.backed_s - self.est_s == 0.0
                    and (self.back_t is None or self.est_t is None or self.back_t < self.est_t)):
                self.backed_s = self.est_s = 0.0
            if EXP['drop_rev_z'] and self.est_s and not self.est_moved:
                self.backed_s = max(0.0, self.backed_s - self.est_s)
                self.est_s = 0.0
            self._go('reverse', now)"""),
    ("""                self._go('brake', now)
                return (self.center, self.n + rev), ASSIST_RECOVERY
            self._go('gap', now)
            el = 0.0""",
     """                self._go('brake', now)
                return (self.center, self.n + rev), ASSIST_RECOVERY
            if EXP['drop_gap'] and self.est_s and self.backed_s - self.est_s == 0.0:
                self.backed_s = self.est_s = 0.0
            self._go('gap', now)
            el = 0.0"""),
]


def main():
    s = SRC.read_text(encoding='utf-8')
    for a, b in REPL:
        assert s.count(a) == 1, a[:80]
        s = s.replace(a, b)
    DST.write_text(s, encoding='utf-8')
    shutil.copyfile(HERE / 'before' / 'drive.py.orig', HERE / 'cand' / 'drive.py')
    for p in (DST, HERE / 'cand' / 'drive.py'):
        print(hashlib.sha256(p.read_bytes()).hexdigest(), p.name)


if __name__ == '__main__':
    main()
