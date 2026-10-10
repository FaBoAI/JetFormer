"""周回の計測: ジャイロのZ(左回り+、deg/s)を積分して、360度回るごとに1周と数える

単純な周回コース(8の字でない)を前提にする。止まっている間(|eRPM| < STILL_ERPM)のジャイロの平均を
ずれ(バイアス)として引き、動いている間(|eRPM| > MOVING_ERPM)だけ積分する。回る向きは最初に
DIRECTION_DEG 回ったときの向きで決める(2026-10-10 のコースは左回り)。

1周ごとに: タイム、その周がずっとAIだったか、人の交代・リカバリー・中立で止めた回数を残す。
走行ループ(30fps)からも、記録したログの評価(tools/drive_report.py)からも同じものを使う。
"""


class LapCounter(object):
    STILL_ERPM = 50
    MOVING_ERPM = 200
    DIRECTION_DEG = 270.0
    MAX_DT = 0.2             # これより間が空いたサンプルは積分しない(途切れ)

    def __init__(self):
        self.reset()

    def reset(self):
        self.heading = 0.0
        self.direction = 0               # +1 = 左回り, -1 = 右回り, 0 = まだ分からない
        self.bias = 0.0
        self._bias_n = 0
        self.t_last = None
        self.lap_start_t = None
        self.lap_start_heading = 0.0
        self.laps = []
        self._cur = self._new_lap()

    @staticmethod
    def _new_lap():
        return {'ai_only': True, 'human_s': 0.0, 'takeovers': 0, 'recoveries': 0, 'holds': 0, 'last_ai': None}

    def add(self, t, gz_dps, erpm, ai=None):
        """IMUの1サンプル。t [s]、gz [deg/s]、erpm(そのときの車両の値)、ai(AI側か。分からなければ None)。
        1周が終わったらその周の dict を返す"""
        lap = None
        if abs(erpm) < self.STILL_ERPM and abs(gz_dps) < 5.0:   # 止まっている(ぶつかって回っている最中は除く): バイアスを更新
            self._bias_n = min(self._bias_n + 1, 500)
            self.bias += (gz_dps - self.bias) / self._bias_n
        if self.t_last is not None:
            dt = t - self.t_last
            if 0 < dt <= self.MAX_DT and abs(erpm) > self.MOVING_ERPM:
                self.heading += (gz_dps - self.bias) * dt
                if ai is not None:
                    self._track(ai, dt)
                if self.lap_start_t is None:
                    self.lap_start_t, self.lap_start_heading = self.t_last, self.heading
                if self.direction == 0 and abs(self.heading - self.lap_start_heading) >= self.DIRECTION_DEG:
                    self.direction = 1 if self.heading > self.lap_start_heading else -1
                if self.direction and (self.heading - self.lap_start_heading) * self.direction >= 360.0:
                    lap = self._finish(t)
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
               'recoveries': c['recoveries'], 'holds': c['holds'],
               'first': not self.laps}       # 最初の周は止まった状態からの発進を含む
        clean = [x['time_s'] for x in self.laps if x['ai_only'] and not x['first'] and not x['recoveries']]
        lap['best'] = bool(lap['ai_only'] and not lap['first'] and not lap['recoveries']
                           and (not clean or lap['time_s'] < min(clean)))
        self.laps.append(lap)
        self.lap_start_t = t
        self.lap_start_heading += 360.0 * self.direction
        last_ai = c['last_ai']
        self._cur = self._new_lap()
        self._cur['last_ai'] = last_ai
        return lap

    def summary(self):
        clean = [x['time_s'] for x in self.laps if x['ai_only'] and not x['first'] and not x['recoveries']]
        return {
            'laps': len(self.laps), 'direction': {1: '左回り', -1: '右回り'}.get(self.direction, ''),
            'last_s': self.laps[-1]['time_s'] if self.laps else None,
            'best_ai_s': min(clean) if clean else None,
            'median_ai_s': round(sorted(clean)[len(clean) // 2], 2) if clean else None,
            'clean_laps': len(clean),
            'recent': [{k: x[k] for k in ('n', 'time_s', 'ai_only', 'recoveries', 'takeovers', 'best', 'first')}
                       for x in self.laps[-12:]],
            'progress_deg': round(abs(self.heading - self.lap_start_heading), 0) if self.lap_start_t is not None else 0,
        }
