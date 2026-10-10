"""周回の計測: ジャイロのZ(左回り+、deg/s)を積分して、360度回るごとに1周と数える

単純な周回コース(8の字でない)を前提にする。止まっている間(|eRPM| < STILL_ERPM)のジャイロの平均を
ずれ(バイアス)として引き、動いている間(|eRPM| > MOVING_ERPM)だけ積分する。回る向きは最初に
DIRECTION_DEG 回ったときの向きで決める(2026-10-10 のコースは左回り)。

周の区切りは、最初の1周で「いちばん鋭く曲がるところ」(向き10度ごとの平均ヨーレートが最大)に置く。
直線の途中で区切ると、走る向きが数度違うだけで区切りの時刻が大きくずれる(2026-10-10 ai_run_006 で
8.15秒と11.43秒の組 → この方法では 8.65〜10.24秒)。最初の区切りが決まるまで(約1周)はタイムが出ない。

1周ごとに: タイム、その周がずっとAIだったか、人の交代・リカバリー・中立で止めた・ぶつかった回数を残す。
ベストと中央値は「AIだけ・ぶつからず・リカバリーなし」の周(clean)で出す(ぶつかって向きが変わった周は区切りもずれる)。
走行ループ(30fps)からも、記録したログの評価(tools/drive_report.py)からも同じものを使う。
"""


class LapCounter(object):
    STILL_ERPM = 50
    MOVING_ERPM = 200
    DIRECTION_DEG = 270.0
    MAX_DT = 0.2             # これより間が空いたサンプルは積分しない(途切れ)
    IMPACT_G = 2.5           # 水平の加速度(重力を除く)がこれ以上 = その周は「ぶつかった周」

    def __init__(self):
        self.reset()

    def reset(self):
        self.heading = 0.0
        self.direction = 0               # +1 = 左回り, -1 = 右回り, 0 = まだ分からない
        self.bias = 0.0
        self._bias_n = 0
        self.t_last = None
        self.lap_start_t = None          # 今の周の始まり(区切りが決まるまでは計測の始まり)
        self.h0 = None                   # 計測を始めたときの向き
        self.phase = None                # 区切りの位置(計測の始まりからの角度 0〜360)
        self.next_rel = None
        self._early = []                 # 向きが決まるまでの (向き, |ヨーレート|, 時刻)
        self._bins = [0.0] * 36
        self._cnt = [0] * 36
        self._bin_t = [None] * 36
        self.laps = []
        self._last_impact = -1e9
        self._cur = self._new_lap()

    @staticmethod
    def _new_lap():
        return {'ai_only': True, 'human_s': 0.0, 'takeovers': 0, 'recoveries': 0, 'holds': 0, 'impacts': 0,
                'last_ai': None}

    def clean(self, lap):
        """タイムの比較に使う周: AIだけ・最初の周でない・リカバリーなし・ぶつかっていない"""
        return lap['ai_only'] and not lap['first'] and not lap['recoveries'] and not lap.get('impacts')

    def _bin(self, rel, agz, t):
        if 0 <= rel < 360.0:
            b = int(rel // 10)
            self._bins[b] += agz
            self._cnt[b] += 1
            if self._bin_t[b] is None:
                self._bin_t[b] = t

    def add(self, t, gz_dps, erpm, ai=None, impact_g=0.0):
        """IMUの1サンプル。t [s]、gz [deg/s]、erpm(そのときの車両の値)、ai(AI側か。分からなければ None)、
        impact_g(水平の加速度 [G]、重力を除く)。1周が終わったらその周の dict を返す"""
        lap = None
        if impact_g >= self.IMPACT_G and t - self._last_impact > 0.5:
            self._cur['impacts'] += 1
            self._last_impact = t
        if abs(erpm) < self.STILL_ERPM and abs(gz_dps) < 5.0:   # 止まっている(ぶつかって回っている最中は除く): バイアスを更新
            self._bias_n = min(self._bias_n + 1, 500)
            self.bias += (gz_dps - self.bias) / self._bias_n
        if self.t_last is not None:
            dt = t - self.t_last
            if 0 < dt <= self.MAX_DT and abs(erpm) > self.MOVING_ERPM:
                self.heading += (gz_dps - self.bias) * dt
                if ai is not None:
                    self._track(ai, dt)
                if self.h0 is None:
                    self.h0, self.lap_start_t = self.heading, self.t_last
                agz = abs(gz_dps - self.bias)
                if self.direction == 0:
                    self._early.append((self.heading - self.h0, agz, t))
                    if abs(self.heading - self.h0) >= self.DIRECTION_DEG:
                        self.direction = 1 if self.heading > self.h0 else -1
                        for h, g, tt in self._early:
                            self._bin(h * self.direction, g, tt)
                        self._early = []
                elif self.phase is None:
                    rel = (self.heading - self.h0) * self.direction
                    self._bin(rel, agz, t)
                    if rel >= 360.0:                     # 1周したので、いちばん鋭いところを区切りにする
                        means = [b / c if c else -1.0 for b, c in zip(self._bins, self._cnt)]
                        k = max(range(36), key=lambda i: means[i])
                        self.phase = k * 10 + 5.0
                        self.lap_start_t = self._bin_t[k] if self._bin_t[k] is not None else t
                        self.next_rel = self.phase + 360.0
                        if rel >= self.next_rel:
                            self.next_rel += 360.0
                else:
                    rel = (self.heading - self.h0) * self.direction
                    if rel >= self.next_rel:
                        lap = self._finish(t)
                        self.next_rel += 360.0
        self.t_last = t
        return lap

    def _track(self, ai, dt):
        c = self._cur
        if not ai:
            c['ai_only'] = False
            c['human_s'] += dt
            if c['last_ai']:
                c['takeovers'] += 1
        c['last_ai'] = ai

    def note(self, kind):
        """'recovery' / 'hold' をその周に数える"""
        if kind == 'recovery':
            self._cur['recoveries'] += 1
        elif kind == 'hold':
            self._cur['holds'] += 1

    def _finish(self, t):
        c = self._cur
        lap = {'n': len(self.laps) + 1, 't_end': round(t, 3), 'time_s': round(t - self.lap_start_t, 2),
               'ai_only': c['ai_only'], 'human_s': round(c['human_s'], 2), 'takeovers': c['takeovers'],
               'recoveries': c['recoveries'], 'holds': c['holds'], 'impacts': c['impacts'],
               'first': not self.laps}       # 最初の周は発進や計測の始まりを含むことがある
        clean = [x['time_s'] for x in self.laps if self.clean(x)]
        lap['clean'] = self.clean(lap)
        lap['best'] = bool(lap['clean'] and (not clean or lap['time_s'] < min(clean)))
        self.laps.append(lap)
        self.lap_start_t = t
        last_ai = c['last_ai']
        self._cur = self._new_lap()
        self._cur['last_ai'] = last_ai
        return lap

    def summary(self):
        clean = [x['time_s'] for x in self.laps if self.clean(x)]
        ai = [x['time_s'] for x in self.laps if x['ai_only'] and not x['first']]
        return {
            'laps': len(self.laps), 'direction': {1: '左回り', -1: '右回り'}.get(self.direction, ''),
            'last_s': self.laps[-1]['time_s'] if self.laps else None,
            'best_ai_s': min(clean) if clean else None,
            'median_ai_s': round(sorted(clean)[len(clean) // 2], 2) if clean else None,
            'clean_laps': len(clean),
            'ai_laps': len(ai), 'median_ai_all_s': round(sorted(ai)[len(ai) // 2], 2) if ai else None,
            'recent': [{k: x[k] for k in ('n', 'time_s', 'ai_only', 'recoveries', 'takeovers', 'impacts', 'clean', 'best',
                                          'first')}
                       for x in self.laps[-12:]],
            'progress_deg': (round(((self.heading - self.h0) * self.direction - (self.next_rel - 360.0)) % 360.0, 0)
                             if self.next_rel is not None else 0),
            'timing': self.next_rel is not None,      # 区切りが決まって計測中か
        }
