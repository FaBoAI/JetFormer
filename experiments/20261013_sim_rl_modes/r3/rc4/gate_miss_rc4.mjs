// 離脱復活 r4(2026-10-14): r3/rc/gate_miss_rc3.mjs の写し(読み込む先だけ r3/rc4/rt)
// 本当のゲートの外し(sim がゲートを数えなくなったこと)を、走りの記録から数える(2026-10-14、学習の段 recover r2)。
//
// 決まり(確認 recover r1 の数え方と同じ。rl/check_rc/analyze_rc.py の gate_stuck):
//   sim の passed_gates(と completed_laps)が変わらないまま、R 上の s が「ゲートの間の最大 1.519 m + 0.5 m = 2.019 m」より先へ進んだら、
//   そのゲートを外した(sim の route_index がそのゲートで止まり、1 周後に同じゲートを通るまで周を数えない)。
//   ふつうに走っていれば、次のゲートまでの s の増え方はゲートの間の最大(1.519 m)を越えないので、0.5 m の余白で誤りは出ない。
// 使わないもの: 学習の段 r1 の「valid_laps < laps.length」(laps は sim が数えた周だけなので、いつも 0 になる)と、
//   ランタイムの仮想のゲートの外し(wrapper.misses。外側の線で面を越えるのが遅れただけでも数えるので、本当の外しの数ではない)。
import {GATE_S, LOOP_LENGTH} from './rt/features.mjs';

const gs = [...GATE_S].sort((a, b) => a - b);
export const MAX_GATE_GAP_M = Math.max(...gs.map((s, i) => ((gs[(i + 1) % gs.length] - s) % LOOP_LENGTH + LOOP_LENGTH) % LOOP_LENGTH));
export const GATE_MISS_MARGIN_M = 0.5;
export const GATE_MISS_THRESHOLD_M = MAX_GATE_GAP_M + GATE_MISS_MARGIN_M;

// sArr: フレームごとの R 上の s(巻き戻さない連続の値。mode_runtime の log の s)。
// gArr: 同じフレームの sim の数え(passed_gates と completed_laps をまとめた値。変わったかどうかだけを見る)。
// 戻り値: {stuck_max_m, misses, first_frame, frames}。misses = しきい値を越えた「止まり」の数(1 つの止まりは 1 回)
export function realGateMisses(sArr, gArr, threshold = GATE_MISS_THRESHOLD_M) {
  const n = Math.min(sArr.length, gArr.length);
  let lastS = n ? sArr[0] : 0, stuckMax = 0, misses = 0, counted = false, firstFrame = null;
  const frames = [];
  for (let f = 1; f < n; f++) {
    if (gArr[f] !== gArr[f - 1]) { lastS = sArr[f]; counted = false; }
    const st = sArr[f] - lastS;
    if (st > stuckMax) stuckMax = st;
    if (st > threshold && !counted) { counted = true; misses++; frames.push(f); if (firstFrame == null) firstFrame = f; }
  }
  return {stuck_max_m: stuckMax, misses, first_frame: firstFrame, frames};
}
